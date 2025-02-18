#!/usr/bin/env python
import os
import sys
import glob
import time
import datetime
import random
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils  # for gradient clipping
import torch.utils.checkpoint  # for gradient checkpointing
import torch.profiler  # for profiling
from torch.profiler import ProfilerActivity
import tiktoken
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm  # For progress bar
from tabulate import tabulate  # For printing a nice table
import concurrent.futures  # For asynchronous prefetching
import argparse

# Attempt to import FlashAttention.
try:
    from flash_attn.flash_attn_interface import flash_attn
    print("[Info] FlashAttention imported successfully.")
except ImportError:
    flash_attn = None
    print("[Info] FlashAttention not found; using default scaled dot-product attention.")

# #############################################################################
# MUON OPTIMIZER IMPLEMENTATION
# #############################################################################

def zeropower_via_svd(G, steps=None):
    # SVD-based orthogonalization: G --> U V^T
    U, S, V = torch.linalg.svd(G, full_matrices=False)
    return U @ V

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Orthogonalize G using the Newton-Schulz iteration. 
    The iteration projects G onto the set of orthonormal matrices.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.half() / (G.norm() + eps)
    transposed = (X.shape[0] > X.shape[1])
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    if transposed:
        X = X.T
    return X.to(G.dtype)

ZEROPOWER_BACKENDS = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5
}

class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Newton-Schulz Orthogonalization.
    Each 2D parameter is treated as a matrix and its momentum update is orthogonalized.
    """
    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        nesterov=True,
        backend='newtonschulz5',
        backend_steps=5
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            orth_fn = ZEROPOWER_BACKENDS[group['backend']]
            steps = group['backend_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                if grad.ndim == 2 and grad.size(0) > 1 and grad.size(1) > 1:
                    grad_ortho = orth_fn(grad, steps=steps)
                    scale = max(grad.size(0), grad.size(1)) ** 0.5
                    p.add_(grad_ortho, alpha=-lr * scale)
                else:
                    p.add_(grad, alpha=-lr)

# #############################################################################
# DEFAULT HYPERPARAMS & GLOBALS
# #############################################################################
batch_size = 32
block_size = 512
max_iters = 100
eval_interval = 100
learning_rate = 3e-4
warmup_steps = 10
warmdown_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
num_experts = 8
top_k = 2
capacity_factor = 1.0

# #############################################################################
# ROTARY, RMSNORM, DATALOADER, ETC.
# #############################################################################
class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[2]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()[None, None, :, :]
            self.sin_cached = freqs.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    new_x1 = x1 * cos + x2 * sin
    new_x2 = -x1 * sin + x2 * cos
    return torch.cat([new_x1, new_x2], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

class ShardedDataLoader:
    def __init__(self, data_dir, pattern, B, T, split="train", vocab_size=50304):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size
        self.split = split.lower()
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))[:5]
        if not self.files:
            raise ValueError(f"No files found with pattern {pattern} in {data_dir}")
        self.original_files_order = list(self.files)
        random.shuffle(self.files)
        self.current_shard_index = 0
        self.memmap_obj = None
        self.full_data = None
        self.load_shard(self.files[self.current_shard_index])

    def close_current_shard(self):
        if self.memmap_obj is not None:
            self.memmap_obj._mmap.close()
            self.memmap_obj = None
        self.full_data = None

    def load_shard(self, filepath):
        self.close_current_shard()
        print(f"[DataLoader-{self.split}] Loading shard: {filepath}")
        self.memmap_obj = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.full_data = self.memmap_obj
        self.shard_length = len(self.full_data)
        if self.split == "train":
            self.split_start = 0
            self.split_end = int(0.7 * self.shard_length)
        else:
            self.split_start = int(0.7 * self.shard_length)
            self.split_end = self.shard_length
        self.data = self.full_data[self.split_start:self.split_end]
        self.shard_name = filepath
        self.pos = 0
        print(f"[DataLoader-{self.split}] Shard tokens: {len(self.data)} (from {self.split_start} to {self.split_end})")

    def reset_epoch(self):
        random.shuffle(self.files)
        self.current_shard_index = 0
        self.load_shard(self.files[self.current_shard_index])

    def next_batch(self):
        required_tokens = self.B * self.T + 1
        if self.pos + required_tokens > len(self.data):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.files)
            if self.current_shard_index == 0:
                self.reset_epoch()
            else:
                self.load_shard(self.files[self.current_shard_index])
        batch_tokens = self.data[self.pos:self.pos + required_tokens]
        self.pos += self.B * self.T
        batch_tokens = torch.from_numpy(batch_tokens.astype(np.int64))
        if batch_tokens.max() >= self.vocab_size or batch_tokens.min() < 0:
            print(f"[Warning] Out-of-range tokens in shard {self.shard_name}. Clamping.")
            batch_tokens = torch.clamp(batch_tokens, 0, self.vocab_size - 1)
        x = batch_tokens[:-1].view(self.B, self.T)
        y = batch_tokens[1:].view(self.B, self.T)
        print(f"[DataLoader-{self.split}] Batch range: min={x.min().item()}, max={x.max().item()}")
        return x, y

# ----------------------------------------------------------------------------- 
# Token decoding helper (using tiktoken)
def decode(tokens):
    enc = tiktoken.get_encoding("gpt2")
    MAX_ID = 50256
    decoded_pieces = []
    for token in tokens:
        if 0 <= token <= MAX_ID:
            decoded_pieces.append(enc.decode([token]))
        else:
            decoded_pieces.append("<unk>")
    return "".join(decoded_pieces)

# ----------------------------------------------------------------------------- 
# Model Configuration and Components
@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = n_layer
    n_head: int = n_head
    n_embed: int = n_embed
    num_experts: int = num_experts
    top_k: int = top_k
    capacity_factor: float = capacity_factor
    use_gradient_checkpointing: bool = True
    moe_load_balancing_weight: float = 0.01

# ----------------- MODEL COMPONENTS -----------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head

        self.c_attention = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

        self.rotary = Rotary(self.head_dim)

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        qkv = self.c_attention(x)
        query, key, value = qkv.split(self.n_embed, dim=2)
        query = query.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        key   = key.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        value = value.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        cos, sin = self.rotary(query)
        query = apply_rotary_emb(query, cos, sin)
        key   = apply_rotary_emb(key, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            key   = torch.cat([pk, key], dim=2)
            value = torch.cat([pv, value], dim=2)
        attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        y = attn_out.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        new_kv = (key, value)
        return y, new_kv

class Expert(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.n_embed, config.num_experts)
        self.noise_linear = nn.Linear(config.n_embed, config.num_experts)
        nn.init.xavier_normal_(self.topkroute_linear.weight)
        nn.init.zeros_(self.topkroute_linear.bias)
        nn.init.xavier_normal_(self.noise_linear.weight)
        nn.init.zeros_(self.noise_linear.bias)
    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise_stddev = F.softplus(noise_logits)
        noise = torch.randn_like(logits) * noise_stddev
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

def moe_load_balancing_loss(gates: torch.Tensor):
    expert_fraction = gates.mean(dim=(0,1))
    num_experts = gates.size(-1)
    target = 1.0 / num_experts
    loss = torch.mean((expert_fraction - target) ** 2)
    return loss

class SparseMoE(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.num_experts = config.num_experts
        self.load_balance_weight = config.moe_load_balancing_weight

    def forward(self, x):
        B, T, C = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_gating_output = gating_output.view(-1, self.num_experts)
        tokens_per_batch = B * T * self.top_k
        expert_capacity = max(int((tokens_per_batch / self.num_experts) * self.capacity_factor), 1)
        updates = torch.zeros_like(flat_x)
        lb_loss = moe_load_balancing_loss(gating_output)
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity]
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)
        final_output += updates.view(B, T, C)
        return final_output, self.load_balance_weight * lb_loss

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attention = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.moe = SparseMoE(config)
        self.config = config

    def forward_fn(self, x):
        a_out, _ = self.attention(self.layer_norm_1(x))
        x = x + a_out
        moe_in = self.layer_norm_2(x)
        moe_out, moe_loss = self.moe(moe_in)
        x = x + moe_out
        return x, moe_loss

    def forward(self, x):
        if self.config.use_gradient_checkpointing:
            x, moe_loss = torch.utils.checkpoint.checkpoint(self.forward_fn, x)
        else:
            x, moe_loss = self.forward_fn(x)
        return x, moe_loss

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layernorm_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None, past_kv=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        if past_kv is None:
            past_kv = [None] * self.config.n_layer
        total_moe_loss = 0.0
        new_past_kv = []
        for block, pkv in zip(self.transformer.h, past_kv):
            a_in = block.layer_norm_1(x)
            attn_out, updated_kv = block.attention(a_in, pkv)
            x = x + attn_out
            moe_in = block.layer_norm_2(x)
            moe_out, moe_loss = block.moe(moe_in)
            x = x + moe_out
            total_moe_loss += moe_loss
            new_past_kv.append(updated_kv)
        x = self.transformer.layernorm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + total_moe_loss
        return logits, loss, new_past_kv

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def print_data_stats(files, B, T):
    total_tokens = 0
    for f in files:
        data = np.memmap(f, dtype=np.uint16, mode='r')
        total_tokens += len(data)
        data._mmap.close()
    tokens_per_batch = B * T
    table_data = [
        ["Total Tokens in Shards", total_tokens],
        ["Batch Size", B],
        ["Block Size", T],
        ["Tokens per Batch", tokens_per_batch]
    ]
    print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid") + "\n")

def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    elif step > max_iters - warmdown_steps:
        decay_ratio = (max_iters - step) / warmdown_steps
        return learning_rate * decay_ratio
    else:
        return learning_rate

# ----------------------------------------------------------------------------- 
# Main Training Loop with an Optional Profiler
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT MoE Training with Muon, LR Warmup, Gradient Checkpointing, and Profiling")
    parser.add_argument("--experiment_name", type=str, default="GPT_MoE_Training", help="Name of the experiment")
    parser.add_argument("--enable_profiling", action="store_true", help="Enable PyTorch Profiler")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    data_dir = "finewebedu10B"
    train_pattern = "finewebedu_train_*.bin"

    # Create train and validation loaders
    train_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size, split="train")
    valid_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size, split="test")
    print_data_stats(train_loader.files, batch_size, block_size)

    cfg = Config(
        block_size=block_size,
        vocab_size=50304,
        n_layer=n_layer,
        n_head=n_head,
        n_embed=n_embed,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
    )
    model = GPT(cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Mixed precision scaler
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type=="cuda"))

    # Compile model for optimization (PyTorch 2.x)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    # ------------------- SPLIT PARAMETERS: MUON vs. ADAMW -----------------------
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and ("wte." not in name) and ("lm_head." not in name):
            muon_params.append(param)
        else:
            adam_params.append(param)
    print(f"[Info] Muon params: {len(muon_params)} -> {sum(p.numel() for p in muon_params)} elements")
    print(f"[Info] AdamW params: {len(adam_params)} -> {sum(p.numel() for p in adam_params)} elements")

    optimizer_muon = Muon(muon_params, lr=learning_rate * 0.1, momentum=0.95)
    optimizer_adam = torch.optim.AdamW(adam_params, lr=learning_rate, betas=(0.9, 0.95))

    scheduler = None
    use_lr_scheduler = True
    if use_lr_scheduler:
        def lr_lambda(step):
            return get_lr(step) / learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("block_size", block_size)
        mlflow.log_param("max_iters", max_iters)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_experts", num_experts)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("capacity_factor", capacity_factor)
        mlflow.log_param("data_shards", train_loader.files)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        mlflow.log_param("use_gradient_checkpointing", cfg.use_gradient_checkpointing)
        mlflow.log_param("enable_profiling", args.enable_profiling)

        print("[Training] Starting training loop...")
        start_time = time.time()

        profiler_logs_dir = "profiler_logs"
        os.makedirs(profiler_logs_dir, exist_ok=True)
        if args.enable_profiling:
            schedule = torch.profiler.schedule(wait=2, warmup=2, active=5)
            profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logs_dir),
                record_shapes=True,
                profile_memory=True
            )
            profiler.__enter__()

        for i in tqdm(range(max_iters), desc="Training"):
            iter_start = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adam.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=="cuda")):
                    logits, loss, _ = model(x, targets=y)
                scaler.scale(loss).backward()
                clip_grad_norm = True
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer_adam)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer_muon)
                scaler.step(optimizer_adam)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[Error] CUDA out of memory. Saving emergency checkpoint...")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    os.makedirs("emergency_ckpt", exist_ok=True)
                    ckpt_path = os.path.join("emergency_ckpt", f"model_oom_{timestamp}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[Checkpoint] Emergency model saved at {ckpt_path}")
                    raise e
                else:
                    raise e

            if scheduler is not None:
                scheduler.step()

            iter_time = time.time() - iter_start

            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = valid_loader.next_batch()
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=="cuda")):
                        v_logits, v_loss, _ = model(vx, targets=vy)
                model.train()
                tokens_processed = batch_size * block_size
                mfu = (6.0 * total_params * tokens_processed) / (250e12 * iter_time) * 100
                msg = (f"[Training] Step {i}: Loss = {loss.item():.4f}, Val Loss = {v_loss.item():.4f}, "
                       f"Iter Time = {iter_time*1000:.2f} ms, Tokens = {tokens_processed}, MFU ~ {mfu:.2f}%")
                print(msg)
                mlflow.log_metric("train_loss", loss.item(), step=i)
                mlflow.log_metric("val_loss", v_loss.item(), step=i)
                mlflow.log_metric("iteration_time_ms", iter_time * 1000, step=i)
                mlflow.log_metric("mfu", mfu, step=i)
                mlflow.log_metric("tokens_processed", tokens_processed, step=i)

            if args.enable_profiling:
                profiler.step()

        if args.enable_profiling:
            profiler.__exit__(None, None, None)

        total_time = time.time() - start_time
        print(f"[Training] Training complete in {total_time:.2f} seconds")
        mlflow.log_metric("total_training_time_s", total_time)
        mlflow.log_metric("total_run_time_s", total_time)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Checkpoint] Model saved at {model_save_path}")
        mlflow.log_artifact(model_save_path)
        if args.enable_profiling:
            mlflow.log_artifact(profiler_logs_dir)

    print("[Done] Training script finished.")
