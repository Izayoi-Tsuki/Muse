"""Decoder-only Transformer + RoPE，用于 REMI token 的自回归建模。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        seq_len = x.shape[-2]
        cos, sin = self.cos[..., :seq_len, :], self.sin[..., :seq_len, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class Attention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d, h = cfg["d_model"], cfg["n_heads"]
        assert d % h == 0
        self.n_heads = h
        self.head_dim = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.dropout = cfg["dropout"]
        self.rope = RotaryEmbedding(self.head_dim, cfg["max_seq_len"]) if cfg["rope"] else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg["ff_mult"], bias=False),
            nn.GELU(),
            nn.Linear(d * cfg["ff_mult"], d, bias=False),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MuseTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.head.weight = self.embed.weight  # 权重绑定
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """贪心窗口续写：超出上下文时截断保留最近 max_seq_len 个 token。"""
        self.eval()
        idx = prompt
        max_len = self.cfg["max_seq_len"]
        for _ in range(max_new_tokens):
            ctx = idx[:, -max_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            # nucleus (top-p) 采样
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            nxt = torch.multinomial(sorted_probs, 1)
            nxt = sorted_idx.gather(-1, nxt)
            idx = torch.cat([idx, nxt], dim=1)
        return idx