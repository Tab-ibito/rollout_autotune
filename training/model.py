"""PyTorch causal Transformer shared by the offline critic and exported actor.

Weight names, RMSNorm epsilon (1e-6), RoPE frequency base (10000), and SwiGLU
projections must stay strictly aligned with ``rollout_autotune.dit.CPUModel``
so ``.eval()`` forward passes match NumPy CPU inference within ``1e-4``.
"""

import torch
from torch import nn
from torch.nn import functional as F

from rollout_autotune.dit import CONTEXT, HEAD_DIM, HEADS, INNER, INPUT, LAYERS, WIDTH


class Block(nn.Module):
    """Pre-RMSNorm causal self-attention (RoPE) + SwiGLU FFN block."""

    def __init__(self):
        super().__init__()
        self.n1 = nn.RMSNorm(WIDTH, eps=1e-6)
        self.n2 = nn.RMSNorm(WIDTH, eps=1e-6)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.up = nn.Linear(WIDTH, 2 * INNER, bias=False)
        self.down = nn.Linear(INNER, WIDTH, bias=False)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)

        def rotate(a):
            p, q = a[..., ::2], a[..., 1::2]
            return torch.stack((p * cos - q * sin, p * sin + q * cos), -1).flatten(-2)

        a = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, WIDTH)))
        gate, up = self.up(self.n2(x)).chunk(2, -1)
        return x + self.drop(self.down(F.silu(gate) * up))


class Transformer(nn.Module):
    """Causal step-token Transformer mapping ``(B, T, input_dim) -> (B, T, outputs)``."""

    def __init__(self, outputs, input_dim=INPUT):
        super().__init__()
        self.embed = nn.Linear(input_dim, WIDTH)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.final = nn.RMSNorm(WIDTH, eps=1e-6)
        self.head = nn.Linear(WIDTH, outputs)
        freq = 10000 ** (-torch.arange(0, HEAD_DIM, 2) / HEAD_DIM)
        pos = torch.arange(CONTEXT)[:, None] * freq
        self.register_buffer("cos", pos.cos(), persistent=False)
        self.register_buffer("sin", pos.sin(), persistent=False)

    def forward(self, x):
        t = x.shape[1]
        x = self.embed(x)
        for b in self.blocks:
            x = b(x, self.cos[:t], self.sin[:t])
        return self.head(self.final(x))
