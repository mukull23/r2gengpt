"""Visual mappers: image features -> LLM word-embedding space (paper Sec. 2.2.1 and research gap 3).

Each mapper returns (tokens, attn). attn is (B, T_out, T_in) when output tokens mix input positions (the
Perceiver), else None. explain.py uses it to trace a visual token back to image regions.
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

MAPPERS = ("linear", "mlp", "perceiver")


class LinearMapper(nn.Module):
    """The paper's mapper: one linear projection (followed by LayerNorm, as in the official code)."""

    def __init__(self, enc_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(enc_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        return self.norm(self.proj(x)), None


class MLPMapper(nn.Module):
    """Two-layer GELU MLP, as in LLaVA-1.5."""

    def __init__(self, enc_dim, llm_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(enc_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        return self.norm(self.proj(x)), None


class _PerceiverLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, q, x):
        qn = self.norm_q(q)
        kv = torch.cat([self.norm_kv(x), qn], dim=1)  # Flamingo: latents also attend to themselves
        out, attn = self.attn(qn, kv, kv, need_weights=True, average_attn_weights=True)
        q = q + out
        return q + self.ff(q), attn[:, :, : x.shape[1]]


class PerceiverResampler(nn.Module):
    """A fixed number of learned queries cross-attend to the image features (Flamingo / BLIP-2 style)."""

    def __init__(self, enc_dim, llm_dim, num_queries=32, depth=2):
        super().__init__()
        heads = next(h for h in (8, 4, 2, 1) if enc_dim % h == 0)
        self.queries = nn.Parameter(torch.randn(num_queries, enc_dim) * 0.02)
        self.layers = nn.ModuleList(_PerceiverLayer(enc_dim, heads) for _ in range(depth))
        self.proj = nn.Linear(enc_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        attns = []
        for layer in self.layers:
            q, attn = layer(q, x)
            attns.append(attn)
        attn = torch.stack(attns).mean(0)
        return self.norm(self.proj(q)), attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)


def build_mapper(kind, enc_dim, llm_dim, num_queries=32):
    if kind == "linear":
        return LinearMapper(enc_dim, llm_dim)
    if kind == "mlp":
        return MLPMapper(enc_dim, llm_dim)
    if kind == "perceiver":
        return PerceiverResampler(enc_dim, llm_dim, num_queries)
    raise ValueError(f"mapper must be one of {MAPPERS}")


def grid_tokens(feats):
    """Drop leading non-spatial tokens (CLS, registers) so the rest form a square grid.

    Returns (feats, side); side is None when the tokens do not form a grid.
    """
    n = feats.shape[1]
    for extra in (0, 1, 5):
        side = math.isqrt(max(n - extra, 0))
        if n > extra and side * side == n - extra:
            return feats[:, extra:], side
    return feats, None


@lru_cache(maxsize=32)
def _pooling_matrix(n, side, k):
    eye = torch.eye(n)
    s = math.isqrt(k)
    if side is not None and s * s == k:
        pooled = F.adaptive_avg_pool2d(eye.view(n, 1, side, side), (s, s)).view(n, k)
    else:
        pooled = F.adaptive_avg_pool1d(eye.view(n, 1, n), k).view(n, k)
    return pooled.t().contiguous()  # (k, n): each output token averages a block of input tokens


def pooling_matrix(n, side, k, device):
    """(k, n) average-pooling matrix that reduces n grid tokens to k tokens (2-D blocks when possible)."""
    return _pooling_matrix(n, side, k).to(device)
