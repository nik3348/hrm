from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # root mean square normalization (no bias)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return self.scale * x_norm


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj_in = nn.Linear(d_in, 2 * d_out, bias=False)
        self.proj_out = nn.Linear(d_out, d_in, bias=False)

    def forward(self, x):
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(F.silu(x) * gate)


# Rotary positional embedding (RoPE) for self-attention
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "Rotary dim must be even"
        self.dim = dim
        self.base = base

    def _build_freqs(self, seqlen: int, device: torch.device):
        half_dim = self.dim // 2
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, half_dim, device=device).float() / half_dim)
        )
        t = torch.arange(seqlen, device=device).float()  # (L,)
        freqs = torch.einsum("i,j->ij", t, theta)  # (L, half_dim)
        return freqs

    @staticmethod
    def _apply_rotary(x, cos, sin):
        # x: (B, H, L, Dh), cos/sin: (L, Dh/2)
        x1, x2 = x[..., ::2], x[..., 1::2]   # (B,H,L,Dh/2)
        cos = cos[None, None, :, :]          # (1,1,L,Dh/2)
        sin = sin[None, None, :, :]
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return xr

    def forward(self, q, k):
        L = q.shape[-2]
        device = q.device
        freqs = self._build_freqs(L, device)  # (L, Dh/2)
        cos, sin = freqs.cos(), freqs.sin()
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)


# -------------------------
# Transformer building blocks
# -------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: bool = True):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.use_rope = rope
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, L, Dh)
        def reshape(t):
            return (
                t.view(B, L, self.n_heads, self.head_dim).transpose(
                    1, 2).contiguous()
            )

        q, k, v = map(reshape, (q, k, v))

        if self.use_rope:
            q, k = self.rope(q, k)  # type: ignore

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B,H,L,L)
        if attn_mask is not None:
            scores = scores + attn_mask  # add -inf on masked positions if provided
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)  # (B,H,L,Dh)

        # Merge heads
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: int = 4, rope: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.swiglu = SwiGLU(dim, dim * mlp_mult)

    def forward(self, x):
        # Pre-Norm residual
        x = x + self.attn(self.norm1(x))
        x = x + self.swiglu(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, n_heads: int, depth: int, rope: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, rope=rope) for _ in range(depth)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
