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
import tiktoken
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm  # For progress bar
import argparse

# -----------------------------
# Custom FP8 Operators
# -----------------------------
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    @torch.compile
    def impl(x: torch.Tensor, w: torch.Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.mul(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.mul(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.t(),
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(1 / x_s, dtype=torch.float32),
            scale_b=x.new_tensor(1 / w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8
    return impl(x, w)

@mm_op.register_fake
def _(x: torch.Tensor, w: torch.Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.t(), x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    @torch.compile
    def impl(grad: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(1 / x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(1 / w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(1 / grad_s, dtype=torch.float32)
        grad_f8 = grad.mul(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.t().contiguous().t(),
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            x_f8.t().contiguous(),
            grad_f8.t().contiguous().t(),
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).t()
        return grad_x, grad_w
    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward(ctx, grad_out: torch.Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

def lm_head_fp8(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _x = x.flatten(0, -2)
    out: torch.Tensor = torch.ops.nanogpt.mm(_x, w, x_s=2.0, w_s=32.0, grad_s=2.0**29)[0]
    return out.reshape(*x.shape[:-1], -1)

# -----------------------------
# Helper functions
# -----------------------------
def rms_norm(x, dim=-1, eps=1e-6):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)

# -----------------------------
# MUON OPTIMIZER IMPLEMENTATION
# -----------------------------
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
    This optimizer treats each parameter as a matrix (or batch of matrices)
    and orthogonalizes the momentum update each step.
    """
    def __init__(
        self,
        params,
        lr=6e-4 * 0.1,  # Muon uses 1/10 of the base LR.
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
                # Orthogonalize for 2D or batched (3D) parameters.
                if grad.ndim == 3 and grad.size(1) > 1 and grad.size(2) > 1:
                    # Apply batched orthogonalization along the first dimension.
                    grad_ortho = torch.stack([orth_fn(g, steps=steps) for g in grad], dim=0)
                    scale = max(grad.size(1), grad.size(2)) ** 0.5
                    p.add_(grad_ortho, alpha=-lr * scale)
                elif grad.ndim == 2 and grad.size(0) > 1 and grad.size(1) > 1:
                    grad_ortho = orth_fn(grad, steps=steps)
                    scale = max(grad.size(0), grad.size(1)) ** 0.5
                    p.add_(grad_ortho, alpha=-lr * scale)
                else:
                    p.add_(grad, alpha=-lr)

# -----------------------------
# IMPROVED ROTARY EMBEDDINGS
# -----------------------------
class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536, base=1024):
        """
        Improved rotary embedding module.
        Computes inverse frequencies using a linspace over dim//4 and pads with zeros.
        Precomputes cosine and sine matrices for up to max_seq_len tokens.
        """
        super().__init__()
        steps = dim // 4  # number of frequencies
        inv_freq = (1 / base) ** torch.linspace(0.0, 1.0, steps=steps, dtype=torch.float32)
        inv_freq = torch.cat([inv_freq, torch.zeros(steps, dtype=torch.float32)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i, j -> ij", t, inv_freq)  # shape: (max_seq_len, 2*steps)
        self.register_buffer("cos", theta.cos())
        self.register_buffer("sin", theta.sin())
    def forward(self, x):
        # x shape: (B, num_heads, T, head_dim)
        B, H, T, head_dim = x.shape
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    new_x1 = x1 * cos + x2 * sin
    new_x2 = -x1 * sin + x2 * cos
    return torch.cat([new_x1, new_x2], dim=-1)

# -----------------------------
# Additional Modules: RMSNorm, DataLoader, etc.
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

class ShardedDataLoader:
    def __init__(self, data_dir, pattern, B, T, split="train"):
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
        VOCAB_SIZE = 50304
        x = torch.clamp(x, min=0, max=VOCAB_SIZE - 1)
        y = torch.clamp(y, min=0, max=VOCAB_SIZE - 1)
        print(f"[DataLoader-{self.split}] Batch token range: min={x.min().item()}, max={x.max().item()}")
        return x, y

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

@dataclass
class config:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embed: int = 384
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.0

# -----------------------------
# MODEL COMPONENTS
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.n_embed // config.n_head
        # Instead of using a single linear layer for QKV, we use a stacked parameter.
        self.qkv_weight = nn.Parameter(torch.empty(3, config.n_embed, config.n_embed))
        nn.init.normal_(self.qkv_weight, mean=0.0, std=0.02)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.weight.data.zero_()
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.rotary = Rotary(self.head_dim, max_seq_len=config.block_size, base=1024)
        self.lamb = nn.Parameter(torch.tensor(0.5))
    def forward(self, x, v1=None, attn_window=None):
        B, T, C = x.size()
        # Batched computation of Q, K, V using the stacked weight.
        # x: [B, T, C] -> unsqueeze to [1, B, T, C] to broadcast with qkv_weight.
        x_unsq = x.unsqueeze(0)  # shape: [1, B, T, C]
        # Transpose each weight slice to match matmul: [3, C, C] becomes used for x @ (weight^T)
        qkv = torch.matmul(x_unsq, self.qkv_weight.transpose(-1, -2).unsqueeze(1))  # shape: [3, B, T, C]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Reshape for multi-head attention.
        query = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        key   = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        value = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        if v1 is not None:
            v1 = v1.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        else:
            v1 = value
        value = (1 - self.lamb) * value + self.lamb * v1
        query = rms_norm(query, dim=-1)
        key   = rms_norm(key, dim=-1)
        cos, sin = self.rotary(query)
        query = apply_rotary_emb(query, cos, sin)
        key   = apply_rotary_emb(key, cos, sin)
        if attn_window is not None:
            i = torch.arange(T, device=x.device)
            j = torch.arange(T, device=x.device)
            mask = (i.unsqueeze(1) - j.unsqueeze(0)) < attn_window
            mask = mask & (j.unsqueeze(0) <= i.unsqueeze(1))
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None
        
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0,
            is_causal=(attn_window is None)
        )
        attn_output = attn_output.transpose(1,2).contiguous().view(B, T, C)
        attn_output = self.c_proj(attn_output)
        return attn_output, value

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).square()

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            ReLUSquared(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(0.2),
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.n_embed, config.num_experts)
        self.noise_linear = nn.Linear(config.n_embed, config.num_experts)
    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output

class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.num_experts = config.num_experts
    def forward(self, x):
        B, T, C = x.shape
        gating_output = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_gating_output = gating_output.view(-1, self.num_experts)
        tokens_per_batch = B * T * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        updates = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            expert_mask = (gating_output.argmax(dim=-1) == i)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = (selected_indices[:expert_capacity]
                               if selected_indices.numel() > expert_capacity
                               else selected_indices)
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)
        final_output += updates.view(B, T, C)
        return final_output

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.norm1 = RMSNorm(config.n_embed)
        if layer_idx != 7:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = None
        self.norm2 = RMSNorm(config.n_embed)
        self.moe = SparseMoE(config)
    def forward(self, x, x0, v1=None, attn_window=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            attn_in = rms_norm(x, dim=-1)
            attn_out, _ = self.attn(attn_in, v1=v1, attn_window=attn_window)
            x = x + attn_out
        x = x + self.moe(rms_norm(x, dim=-1))
        return x, None

class ValueEmbedding(nn.Module):
    """
    Implements U-net style sparse value embeddings.
    Only the first three and last three positions receive a learned embedding,
    while the middle positions are set to None.
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.n_embed)
            for _ in range(3)
        ])
    def forward(self, inputs) -> list[torch.Tensor | None]:
        v0 = self.embeddings[0](inputs)
        v1 = self.embeddings[1](inputs)
        v2 = self.embeddings[2](inputs)
        return [v0, v1, v2, None, None, None, None, None, None, v0, v1, v2]

class CastedLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
        ))
        self.value_embeds = ValueEmbedding(config)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embed)
        self.encoder_layers = config.n_layer // 2
        self.decoder_layers = config.n_layer - self.encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))
        self.lm_head = CastedLinear(config.n_embed, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        self.embed_norm = RMSNorm(config.n_embed)
        self.apply(self.__init__weights)
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    def forward(self, idx, targets, attn_window=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        x = self.transformer.wte(idx)
        x = self.embed_norm(x)
        x0 = x
        v_chunks = self.value_embeds(idx)  # List of length 12
        skip_connections = []
        for i in range(self.encoder_layers):
            x, _ = self.blocks[i](x, x0, v1=v_chunks[i], attn_window=attn_window)
            skip_connections.append(x)
        for i in range(self.decoder_layers):
            skip = skip_connections.pop()
            weighted_skip = self.skip_weights[i] * skip
            x, _ = self.blocks[self.encoder_layers + i](x + weighted_skip, x0, v1=v_chunks[self.encoder_layers + i], attn_window=attn_window)
        x = self.norm_f(x)
        # Use FP8 head when training; otherwise use standard linear projection.
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training else self.lm_head(x)
        # Lowered softcapping: using 15 instead of 30.
        logits = 15 * torch.tanh(logits / 15)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens, attn_window=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            x = self.transformer.wte(idx_cond)
            x = self.embed_norm(x)
            x0 = x
            v_chunks = self.value_embeds(idx_cond)
            skip_connections = []
            for i in range(self.encoder_layers):
                x, _ = self.blocks[i](x, x0, v1=v_chunks[i], attn_window=attn_window)
                skip_connections.append(x)
            for i in range(self.decoder_layers):
                skip = skip_connections.pop()
                weighted_skip = self.skip_weights[i] * skip
                x, _ = self.blocks[self.encoder_layers + i](x + weighted_skip, x0, v1=v_chunks[self.encoder_layers + i], attn_window=attn_window)
            x = self.norm_f(x)
            logits = self.lm_head(x)
            logits = 15 * torch.tanh(logits / 15)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train()
        return idx

def print_data_stats(files, B, T):
    total_tokens = 0
    for f in files:
        data = np.memmap(f, dtype=np.uint16, mode='r')
        total_tokens += len(data)
    tokens_per_batch = B * T
    from tabulate import tabulate
    table_data = [
        ["Total Tokens in Shards", total_tokens],
        ["Batch Size", B],
        ["Block Size", T],
        ["Tokens per Batch", tokens_per_batch]
    ]
    print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="fancy_grid") + "\n")

def get_lr(it, max_iters, learning_rate, warmup_iters, warmdown_iters):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    elif it > max_iters - warmdown_iters:
        decay_ratio = (max_iters - it) / warmdown_iters
        return learning_rate * decay_ratio
    else:
        return learning_rate

# -----------------------------
# DEFAULT HYPERPARAMS & GLOBALS
# -----------------------------
batch_size = 32
block_size_train = 512
max_iters = 250
eval_interval = 100
learning_rate = 6e-4  # doubled from 3e-4
warmup_iters = 10
warmdown_iters = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 12
dropout = 0.2
num_experts = 8
top_k = 2
capacity_factor = 1.0

# -----------------------------
# MAIN TRAINING SCRIPT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT MoE Training with U-Net Skip Connections, Rotary, RMSNorm, LR Warmup, Muon, FP8 Head, and Sparse U-Net Value Embeddings"
    )
    parser.add_argument("--experiment_name", type=str, default="GPT_MoE_Training", help="Name of the experiment")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    args = parser.parse_args()
    experiment_name = args.experiment_name

    data_dir = "finewebedu10B" 
    train_pattern = "finewebedu_train_*.bin"
    train_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size_train, split="train")
    valid_loader = ShardedDataLoader(data_dir, train_pattern, batch_size, block_size_train, split="test")
    print_data_stats(train_loader.files, batch_size, block_size_train)

    model = GPT(config())
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    model = torch.compile(model)

    # Split parameters into Muon and AdamW groups.
    muon_params = []
    adam_normal = []
    adam_special = []  # For scalar parameters.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Include 2D and 3D parameters (e.g. the stacked QKV weight) for Muon if they are not part of embeddings or LM head.
        if (param.ndim == 2 or param.ndim == 3) and ("wte." not in name) and ("lm_head." not in name) and ("value_embeds" not in name):
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
    # Set AdamW epsilon to 1e-10.
    optimizer_adam = torch.optim.AdamW([
        {'params': adam_normal, 'lr': learning_rate, 'betas': (0.9, 0.95)},
        {'params': adam_special, 'lr': 0.04, 'betas': (0.9, 0.95)}
    ], eps=1e-10)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("block_size", block_size_train)
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
            attn_window = int(64 + (block_size_train - 64) * (i / max_iters))
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                logits, loss = model(x, y, attn_window=attn_window)
            scaler.scale(loss).backward()
            frac = min(i / 500, 1)
            optimizer_muon.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
            lr_now = get_lr(i, max_iters, learning_rate, warmup_iters, warmdown_iters)
            for pg in optimizer_muon.param_groups:
                pg['lr'] = lr_now * 0.1  # LR decay to 0.1 factor.
            optimizer_adam.param_groups[0]['lr'] = lr_now
            scaler.step(optimizer_muon)
            scaler.step(optimizer_adam)
            scaler.update()
            iter_time = time.time() - iter_start
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = valid_loader.next_batch()
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                        v_logits, v_loss = model(vx, vy, attn_window=attn_window)
                model.train()
                tokens_processed = batch_size * block_size_train
                mfu = (6.0 * total_params * tokens_processed) / (250e12 * iter_time) * 100
                msg = (f"[Training] Step {i}: Loss={loss.item():.4f}, ValLoss={v_loss.item():.4f}, "
                       f"IterTime={iter_time*1000:.2f} ms, Tokens={tokens_processed}, MFU={mfu:.2f}%")
                print(msg)
                mlflow.log_metric("train_loss", loss.item(), step=i)
                mlflow.log_metric("val_loss", v_loss.item(), step=i)
                mlflow.log_metric("iteration_time_ms", iter_time * 1000, step=i)
                mlflow.log_metric("mfu", mfu, step=i)
                mlflow.log_metric("tokens_processed", tokens_processed, step=i)
            if i % 100 == 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=50, attn_window=attn_window)
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, f"model_{timestamp}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Checkpoint] Model saved at {model_save_path}")
        mlflow.log_artifact(model_save_path)

    print("[Generation] Generating final text sample...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output, _ = model(context, context, attn_window=attn_window)
    generated_text = decode(output[0].tolist())
    print("[Generation] Generated Text:")
    print(generated_text)
    with open("output.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)
