"""The transformer and the arm switch.

A decoder-only, pre-norm transformer with learned absolute positions and tied
input and output embeddings. Only the attention scores differ between arms.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import stable_seed
from .scores import allowed_mask, parse_arm, raw_scores


@dataclass(frozen=True)
class ModelConfig:
    arm: str
    head_dim: int = 16          # D, the query-key dimension of each head
    vocab_size: int = 65
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    context: int = 64
    mlp_mult: int = 4

    @property
    def value_dim(self):        # width of the values of each head (16)
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spec = parse_arm(cfg.arm)
        H, D, dv = cfg.n_heads, cfg.head_dim, cfg.value_dim
        if self.spec.has_scores:
            self.w_q = nn.Linear(cfg.d_model, H * D, bias=False)     # also the key matrix when tied
            if not self.spec.tied:
                self.w_k = nn.Linear(cfg.d_model, H * D, bias=False)
            self.log_beta = nn.Parameter(torch.zeros(H))
        self.w_v = nn.Linear(cfg.d_model, H * dv, bias=False)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model)
        # frozen calibration constants: the scale c and the epsilon of the distance
        self.register_buffer("c", torch.ones(()))
        self.register_buffer("eps", torch.zeros(()))
        self.register_buffer("allowed", allowed_mask(cfg.context, self.spec.diag_allowed),
                             persistent=False)
        self.capture = False
        self.captured = None

    def forward(self, x):
        B, T, _ = x.shape
        H, D, dv = self.cfg.n_heads, self.cfg.head_dim, self.cfg.value_dim
        v = self.w_v(x).view(B, T, H, dv).transpose(1, 2)
        allowed = self.allowed[:T, :T]
        if self.spec.kind == "uniform":
            attn = allowed.to(x.dtype) / allowed.sum(-1, keepdim=True)
            y = attn @ v
            if self.capture:
                self.captured = {"attn": attn.expand(B, H, T, T).detach()}
        else:
            q = self.w_q(x).view(B, T, H, D).transpose(1, 2)
            k = q if self.spec.tied else self.w_k(x).view(B, T, H, D).transpose(1, 2)
            raw = raw_scores(self.spec, q, k, float(self.eps))
            scores = self.log_beta.exp().view(1, H, 1, 1) * self.c * raw
            scores = scores.masked_fill(~allowed, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            y = attn @ v
            if self.capture:
                self.captured = {"raw": raw.detach(), "scores": scores.detach(),
                                 "attn": attn.detach(), "q": q.detach(), "k": k.detach()}
        return self.w_o(y.transpose(1, 2).reshape(B, T, H * dv))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model)
        self.fc2 = nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def hidden(self, idx, upto=None):
        """Hidden states after the first `upto` blocks (all blocks by default)."""
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks[:upto]:
            x = block(x)
        return x

    def forward(self, idx, targets=None):
        x = self.ln_f(self.hidden(idx))
        logits = x @ self.tok_emb.weight.t()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def set_calibration(self, layer_constants):
        """layer_constants: one dict per layer with the keys "c" and "eps"."""
        for block, consts in zip(self.blocks, layer_constants):
            block.attn.c.fill_(consts["c"])
            block.attn.eps.fill_(consts["eps"])


# ---------------------------------------------------------------- initialisation

def init_parameters(model, seed):
    """Initialise every parameter from a generator seeded by the seed and the parameter name.

    Arms with the same seed share the initial value of every parameter that they have in
    common. In the untied arm w_q has the name of the tied matrix, so it starts equal to it.
    """
    cfg = model.cfg
    residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("log_beta"):
                p.zero_()
            elif ".ln" in name or name.startswith("ln_f"):
                p.fill_(1.0 if name.endswith("weight") else 0.0)
            elif p.ndim == 2:
                gen = torch.Generator().manual_seed(stable_seed("init", seed, name))
                std = 0.02
                if name.endswith("w_o.weight") or name.endswith("fc2.weight"):
                    std *= residual_scale
                p.copy_(torch.randn(p.shape, generator=gen) * std)
            else:
                p.zero_()      # biases


def build_model(arm, head_dim, seed, vocab_size=65, calibration=None):
    cfg = ModelConfig(arm=arm, head_dim=head_dim, vocab_size=vocab_size)
    model = Transformer(cfg)
    init_parameters(model, seed)
    if calibration is not None:
        model.set_calibration(calibration)
    return model
