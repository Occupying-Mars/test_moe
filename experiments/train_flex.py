#!/usr/bin/env python
import os
import sys
import glob
import time
import datetime
import random
import argparse
from dataclasses import dataclass
# tar -czvf output.tar.gz final_trainers

import mlflow
import numpy as np
from tqdm import tqdm  # For progress bar
from tabulate import tabulate  # For printing a nice table
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import concurrent.futures  # For asynchronous prefetching

# -----------------------------------------------------------------------------
# Attempt to import FlashAttention.
try:
    from flash_attn.flash_attn_interface import flash_attn
    print("[Info] FlashAttention imported successfully.")
except ImportError:
    flash_attn = None
    print("[Info] FlashAttention not found; using default scaled dot-product attention.")

# -----------------------------------------------------------------------------
# Helper: RMS normalization (replacement for F.rms_norm)
def rms_norm(x, dim=-1, eps=1e-6):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)

# -----------------------------------------------------------------------------
# MUON OPTIMIZER IMPLEMENTATION
def zeropower_via_svd(G, steps=None):
    # SVD-based orthogonalization: G --> U V^T
    U, S, V = torch.linalg.svd(G, full_matrices=False)
    return U @ V

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Orthogonalize G using the Newton-Schulz iteration. 
    Projects G onto the set of orthonormal matrices (similar to U V^T).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.half() / (G.norm() + eps)  # use half precision for speed
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
    This optimizer treats each 2D parameter as a matrix and orthogonalizes
    the momentum update each step.
    """
    def __init__(self, params, lr=6e-4 * 0.1, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
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

                # Standard momentum update
                buf.mul_(momentum).add_(grad)
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

                # Orthogonalize only if 2D and with valid dimensions
                if grad.ndim == 2 and grad.size(0) > 1 and grad.size(1) > 1:
                    grad_ortho = orth_fn(grad, steps=steps)
                    scale = max(grad.size(0), grad.size(1)) ** 0.5
                    p.add_(grad_ortho, alpha=-lr * scale)
                else:
                    p.add_(grad, alpha=-lr)

# -----------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS & DEFAULTS
batch_size = 32
block_size = 512
max_iters = 250
eval_interval = 100
learning_rate = 6e-4  # doubled from 3e-4
warmup_iters = 10
warmdown_iters = 10
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
num_experts = 8
top_k = 2
capacity_factor = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# ROTARY EMBEDDING AND ASSOCIATED FUNCTIONS
class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        # x shape: (B, n_head, T, head_dim)
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

# -----------------------------------------------------------------------------
# DATA LOADER: ShardedDataLoader loads tokens from bin shards
class ShardedDataLoader:
    def __init__(self, data_dir, pattern, B, T, split="train"):
        """
        Loads tokens from the first 5 bin files found in data_dir matching pattern.
        Splits each shard into train/test parts (70/30 split).
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))[:5]
        if not self.files:
            raise ValueError(f"No files found with pattern {pattern} in {data_dir}")
        random.shuffle(self.files)
        self.B = B
        self.T = T
        self.split = split.lower()
        self.current_shard_index = 0
        self.load_shard(self.files[self.current_shard_index])

    def load_shard(self, filepath):
        print(f"[DataLoader-{self.split}] Loading shard: {filepath}")
        self.full_data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.shard_length = len(self.full_data)
        if self.split == "train":
            self.split_start = 0
            self.split_end = int(0.7 * self.shard_length)
        else:
            self.split_start = int(0.7 * self.shard_length)
            self.split_end = self.shard_length
        self.data = self.full_data[self.split_start:self.split_end]
        self.pos = 0
        print(f"[DataLoader-{self.split}] Shard tokens: {len(self.data)} (from {self.split_start} to {self.split_end})")

    def next_batch(self):
        required_tokens = self.B * self.T + 1
        if self.pos + required_tokens > len(self.data):
            self.current_shard_index = (self.current_shard_index + 1) % len(self.files)
            self.load_shard(self.files[self.current_shard_index])
        batch_tokens = self.data[self.pos:self.pos + required_tokens]
        batch_tokens = torch.from_numpy(batch_tokens.astype(np.int64))
        x = batch_tokens[:-1].view(self.B, self.T)
        y = batch_tokens[1:].view(self.B, self.T)
        self.pos += self.B * self.T

        VOCAB_SIZE = 50304  # Padded vocabulary size
        x = torch.clamp(x, min=0, max=VOCAB_SIZE - 1)
        y = torch.clamp(y, min=0, max=VOCAB_SIZE - 1)
        print(f"[DataLoader-{self.split}] Batch token range: min={x.min().item()}, max={x.max().item()}")
        return x, y

def decode(tokens):
    enc = tiktoken.get_encoding("gpt2")
    # GPT-2 vocabulary is typically 0..50256
    MAX_ID = 50256
    
    # Build the decoded string piece by piece
    decoded_pieces = []
    for token in tokens:
        if 0 <= token <= MAX_ID:
            # Decode this single token normally
            decoded_pieces.append(enc.decode([token]))
        else:
            # Out-of-range token becomes an <unk> marker
            decoded_pieces.append("<unk>")
    
    # Join everything into one string
    return "".join(decoded_pieces)

# -----------------------------------------------------------------------------
# MODEL CONFIGURATION DATACLASS
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # Padded vocabulary size
    n_layer: int = n_layer
    n_head: int = n_head
    n_embed: int = n_embed
    num_experts: int = num_experts  # Unused in new block design
    top_k: int = top_k            # Unused in new block design
    capacity_factor: float = capacity_factor  # Unused in new block design

# -----------------------------------------------------------------------------
# MODEL COMPONENTS

# Causal Self-Attention with FlashAttention integration
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head
        # Single linear layer for q, k, v
        self.c_attention = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.weight.data.zero_()  # zero-init projection weights
        # Create causal mask buffer (if needed for fallback attention)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.rotary = Rotary(self.head_dim)
        # Learnable lambda for value residual mixing (initialized to 0.5)
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, v1=None):
        B, T, C = x.size()
        qkv = self.c_attention(x)
        # Split into q, k, v and reshape to (B, n_head, T, head_dim)
        q, k, v = qkv.split(self.n_embed, dim=2)
        query = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        key   = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        value = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Value residual mixing
        if v1 is None:
            v1 = value
        else:
            v1 = v1.view_as(value)
        value = (1 - self.lamb) * value + self.lamb * v1

        # QK normalization using rms_norm helper
        query = rms_norm(query, dim=-1)
        key   = rms_norm(key, dim=-1)

        # Apply rotary embedding
        cos, sin = self.rotary(query)
        query = apply_rotary_emb(query, cos, sin)
        key   = apply_rotary_emb(key, cos, sin)

        # Use flash attention if available
        if flash_attn is not None:
            attn_output = flash_attn(query, key, value, dropout_p=0.0, causal=True)
        else:
            attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        return attn_output, value

# MLP block
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Transformer block with embed shortcut (U-Net style)
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        # Learnable scalars for the embed shortcut (initialized to [1., 0.])
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
    def forward(self, x, x0, v1=None):
        # Embed shortcut: mix current state with the original embedding
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        attn_in = rms_norm(x, dim=-1)
        attn_out, v_new = self.attn(attn_in, v1)
        x = x + attn_out
        x = x + self.mlp(rms_norm(x, dim=-1))
        return x, v_new

# A simple linear layer wrapper (for consistency)
class CastedLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x)

# -----------------------------------------------------------------------------
# GPT Model with U-Net Style Skip Connections
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        # U-Net design: split layers into encoder and decoder
        self.encoder_layers = config.n_layer // 2  # first half for encoder
        self.decoder_layers = config.n_layer - self.encoder_layers  # remaining for decoder
        # Learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))
        # Untied lm_head using CastedLinear
        self.lm_head = CastedLinear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()  # zero-init lm_head
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets):
        # idx: (B, T)
        x = self.transformer.wte(idx)  # token embeddings (B, T, n_embed)
        x = rms_norm(x, dim=-1)
        x0 = x  # store initial embedding for shortcut
        v1 = None
        skip_connections = []
        # Encoder pass: store outputs for skip connections
        for i in range(self.encoder_layers):
            x, v1 = self.transformer.h[i](x, x0, v1)
            skip_connections.append(x)
        # Decoder pass: use skip connections from encoder
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()
            weighted_skip = self.skip_weights[i] * skip_connection
            x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, x0, v1)
        x = rms_norm(x, dim=-1)
        logits = self.lm_head(x)
        # Soft cap logits
        logits = 30 * torch.tanh(logits / 30)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Greedy generation with multinomial sampling.
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            x = self.transformer.wte(idx_cond)
            x = rms_norm(x, dim=-1)
            x0 = x
            v1 = None
            skip_connections = []
            for i in range(self.encoder_layers):
                x, v1 = self.transformer.h[i](x, x0, v1)
                skip_connections.append(x)
            for i in range(self.decoder_layers):
                skip_connection = skip_connections.pop()
                weighted_skip = self.skip_weights[i] * skip_connection
                x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, x0, v1)
            x = rms_norm(x, dim=-1)
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train()
        return idx

# -----------------------------------------------------------------------------
# Helpers for printing data stats and learning rate scheduling
def print_data_stats(files, B, T):
    total_tokens = 0
    for f in files:
        data = np.memmap(f, dtype=np.uint16, mode='r')
        total_tokens += len(data)
    tokens_per_batch = B * T
    table_data = [
        ["Total Tokens in Shards", total_tokens],
        ["Batch Size", B],
        ["Block Size", T],
        ["Tokens per Batch", tokens_per_batch]
    ]
    print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid") + "\n")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    elif it > max_iters - warmdown_iters:
        decay_ratio = (max_iters - it) / warmdown_iters
        return learning_rate * decay_ratio
    else:
        return learning_rate

# -----------------------------------------------------------------------------
# MAIN TRAINING SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT MoE Training with Rotary, RMSNorm, LR Warmup, and Validation + Muon (with U-Net style GPT block and FlashAttention)"
    )
    parser.add_argument("--experiment_name", type=str, default="GPT_MoE_Training",
                        help="Name of the experiment")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    data_dir = "finewebedu10B"
    train_pattern = "finewebedu_train_*.bin"

    # Create train and validation loaders
    train_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size, split="train")
    valid_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size, split="test")

    print_data_stats(train_loader.files, batch_size, block_size)

    # Instantiate GPT model with U-Net skip connections
    config_obj = GPTConfig()
    model = GPT(config_obj)
    model.to(device)

    # Use GradScaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # Compile model with torch.compile (PyTorch 2.0+)
    model = torch.compile(model)

    # -----------------------------------------------------------------------------
    # Split parameters into Muon vs. AdamW groups
    muon_params = []
    adam_normal = []
    adam_special = []  # For scalar parameters (e.g., lamb in attention, lambdas in block)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Use Muon for 2D weight parameters (excluding embeddings and final lm_head)
        if param.ndim == 2 and ("wte." not in name) and ("lm_head." not in name):
            muon_params.append(param)
        elif param.ndim <= 1:
            adam_special.append(param)
        else:
            adam_normal.append(param)

    total_muon = sum(p.numel() for p in muon_params)
    total_adam = sum(p.numel() for p in (adam_normal + adam_special))
    print(f"[Info] Muon params: {len(muon_params)} -> {total_muon} elements")
    print(f"[Info] AdamW params: {len(adam_normal) + len(adam_special)} -> {total_adam} elements")

    optimizer_muon = Muon(muon_params, lr=learning_rate * 0.1, momentum=0.95)
    optimizer_adam = torch.optim.AdamW([
        {'params': adam_normal, 'lr': learning_rate, 'betas': (0.9, 0.95)},
        {'params': adam_special, 'lr': 0.04, 'betas': (0.9, 0.95)}
    ])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # -----------------------------------------------------------------------------
    # MLflow experiment logging
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

        print("[Training] Starting training loop...")
        start_time = time.time()

        if args.profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.__enter__()

        for i in tqdm(range(max_iters), desc="Training"):
            iter_start = time.time()

            # 1) Get batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # 2) Zero gradients for both optimizers
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adam.zero_grad(set_to_none=True)

            # 3) Forward and backward pass with autocast
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()

            # Momentum warmup for Muon optimizer:
            frac = min(i / 500, 1)
            optimizer_muon.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

            # Learning rate scheduling
            lr_now = get_lr(i)
            for pg in optimizer_muon.param_groups:
                pg['lr'] = lr_now * 0.1
            optimizer_adam.param_groups[0]['lr'] = lr_now

            # 4) Step both optimizers and update scaler
            scaler.step(optimizer_muon)
            scaler.step(optimizer_adam)
            scaler.update()

            iter_time = time.time() - iter_start

            # 5) Validation every 10 steps
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = valid_loader.next_batch()
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                        v_logits, v_loss = model(vx, vy)
                model.train()

                tokens_processed = batch_size * block_size
                mfu = (6.0 * total_params * tokens_processed) / (250e12 * iter_time) * 100

                msg = (f"[Training] Step {i}: Loss={loss.item():.4f}, ValLoss={v_loss.item():.4f}, "
                       f"IterTime={iter_time*1000:.2f} ms, Tokens={tokens_processed}, MFU={mfu:.2f}%")
                print(msg)

                mlflow.log_metric("train_loss", loss.item(), step=i)
                mlflow.log_metric("val_loss", v_loss.item(), step=i)
                mlflow.log_metric("iteration_time_ms", iter_time * 1000, step=i)
                mlflow.log_metric("mfu", mfu, step=i)
                mlflow.log_metric("tokens_processed", tokens_processed, step=i)

            # 6) Text generation occasionally
            if i % 100 == 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=50)
                gen_text = decode(generated[0].tolist())
                print(f"[Generation] Step {i}: {gen_text}")
                mlflow.log_param(f"gen_text_{i}", gen_text)

            if args.profile:
                profiler.step()

        total_time = time.time() - start_time
        print(f"[Training] Training complete in {total_time:.2f} seconds")
        mlflow.log_metric("total_training_time_s", total_time)
        mlflow.log_metric("total_run_time_s", total_time)

        if args.profile:
            profiler.__exit__(None, None, None)
            print("[Profiler] Profiling results saved to ./profiler_logs")

        # Save checkpoint
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Checkpoint] Model saved at {model_save_path}")
        mlflow.log_artifact(model_save_path)

    # Final generation after training
    print("[Generation] Generating final text sample...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=500)
    generated_text = decode(output[0].tolist())
    print("[Generation] Generated Text:")
    print(generated_text)

    with open("output.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)
