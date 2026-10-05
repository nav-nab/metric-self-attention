"""Character-level language model and the score module (shared by every task, including the shelved MQAR task).

Score module (`Scorer`):
  distance arms : logit = -beta_h * c * dist_p(z_q, z_k)       (dist from lp.py, eps relative)
  dot arms      : logit =  beta_h * c * (z_q . z_k)
beta_h = exp(log_beta_h) is a learnable per-head scale initialised at 1 for every arm; c is a frozen
calibrated constant (a buffer) chosen so that the logit spread at initialisation equals the common target.
eps (buffer) is the relative epsilon inside the root: eps = r * (typical sum inside the root).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lp
from .arms import parse_arm

NEG = float("-inf")


def lm_allowed_mask(T, device=None):
    """Causal mask with the diagonal masked; position 0 may attend to itself."""
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    m[0, 0] = True
    return m


class Scorer(nn.Module):
    def __init__(self, arm, n_heads, impl="auto", chunk=16):
        super().__init__()
        self.arm = arm
        self.kind, self.p, self.tied = parse_arm(arm)
        self.log_beta = nn.Parameter(torch.zeros(n_heads))
        self.register_buffer("c", torch.ones(()))
        self.register_buffer("eps", torch.zeros(()))
        self.impl, self.chunk = impl, chunk
        self.calib = False      # when True: record raw statistics (see calibration.py)
        self.calib_out = None
        self.collect = False    # when True: record diagnostics of the last forward
        self.collect_grad = False
        self.stats = None
        self.grad_stats = None

    def raw(self, zq, zk, causal=False):
        if self.kind == "null":
            return zq.new_zeros(zq.shape[:-1] + (zk.shape[-2],))
        if self.kind == "dot":
            return zq @ zk.transpose(-1, -2)
        return lp.lp_dist(zq, zk, self.p, float(self.eps), self.impl, self.chunk, causal)

    def forward(self, zq, zk, causal=False):
        s = self.raw(zq, zk, causal)
        beta = self.log_beta.exp().view(1, -1, 1, 1)
        sgn = 1.0 if self.kind == "dot" else -1.0
        return sgn * beta * self.c * s

    # ---- statistics for calibration (raw scores, c = 1, no epsilon) ----
    @torch.no_grad()
    def record_raw(self, zq, zk, allowed, min_keys):
        if self.kind == "null":      # constant score: nothing to calibrate
            H = zq.shape[1]
            self.calib_out = {"std_sum": 1.0, "std_count": 1, "std_per_head_sum": [1.0] * H, "std_per_head_count": 1}
            return
        if self.kind == "dot":
            raw = zq @ zk.transpose(-1, -2)
            S = None
        elif self.p == lp.INF:
            raw = lp.naive(zq, zk, lp.INF, 0.0)
            S = None
        else:
            S = lp.power_sum(zq, zk, self.p)
            raw = S.pow(1.0 / self.p)
        am = allowed.to(raw.dtype)                       # Tq,Tk
        n = am.sum(-1)                                   # allowed keys per query
        mean = (raw * am).sum(-1) / n.clamp_min(1)
        var = (((raw - mean.unsqueeze(-1)) ** 2) * am).sum(-1) / n.clamp_min(1)
        std = var.sqrt()                                 # B,H,Tq
        rows = n >= min_keys
        out = {"std_sum": float(std[..., rows].sum()), "std_count": int(std[..., rows].numel()),
               "std_per_head_sum": std[..., rows].sum((0, 2)).tolist(), "std_per_head_count": int(std[..., rows].shape[0] * std[..., rows].shape[2])}
        if S is not None:
            vals = S[..., allowed]                        # B,H,Npairs
            out["S_sum"] = float(vals.sum())
            out["S_count"] = int(vals.numel())
            sub = vals.flatten()
            if sub.numel() > 2_000_000:
                sub = sub[torch.randperm(sub.numel())[:2_000_000]]
            out["S_median"] = float(sub.median())
            out["S_q01"], out["S_q99"] = [float(x) for x in torch.quantile(sub[:1_000_000].float(), torch.tensor([0.01, 0.99]))]
        self.calib_out = out

    # ---- diagnostics ----
    @torch.no_grad()
    def record_stats(self, logits, att, zq, zk, allowed):
        am = allowed
        lg = logits.masked_fill(~am, 0.0)
        ent = -(att * torch.log(att.clamp_min(1e-30))).sum(-1)          # B,H,Tq
        n = am.sum(-1)
        rows = n >= 2
        mean = (lg * am).sum(-1) / n.clamp_min(1)
        var = (((lg - mean.unsqueeze(-1)) ** 2) * am).sum(-1) / n.clamp_min(1)
        zn = zq.norm(dim=-1)                                             # B,H,T
        zkn = zk.norm(dim=-1)
        self.stats = {
            "beta": self.log_beta.exp().tolist(),
            "max_abs_logit": lg.abs().amax((0, 2, 3)).tolist(),           # per head
            "logit_std": var.sqrt()[..., rows].mean((0, 2)).tolist(),    # per head
            "entropy": ent[..., rows].mean((0, 2)).tolist(),             # per head
            "entropy_uniform": float(torch.log(n[rows].float()).mean()),
            "q_norm_mean": zn.mean((0, 2)).tolist(), "q_norm_max": zn.amax((0, 2)).tolist(),
            "k_norm_mean": zkn.mean((0, 2)).tolist(), "k_norm_max": zkn.amax((0, 2)).tolist(),
        }

    def hook_grad(self, z):
        """Gradient diagnostics on the projected vectors (B,H,T,D); useful for Chebyshev sparsity."""
        def fn(g):
            with torch.no_grad():
                a = g.abs()
                mx = a.amax(-1, keepdim=True)
                nz = mx.squeeze(-1) > 0
                frac_active = ((a > 0.01 * mx) & (mx > 0)).float().sum(-1)[nz] / g.shape[-1]
                self.grad_stats = {
                    "z_grad_norm_per_head": g.flatten(2).norm(dim=-1).mean(0).tolist() if g.dim() == 4 else None,
                    "frac_dims_active_1pct": float(frac_active.mean()) if frac_active.numel() else 0.0,
                    "frac_exact_zero": float((g == 0).float().mean()),
                }
        if z.requires_grad:
            z.register_hook(fn)


def scores_to_att(logits, allowed):
    logits = logits.masked_fill(~allowed, NEG)
    return logits, torch.softmax(logits, dim=-1)


# ============================================================================ char LM
class CausalSelfAttention(nn.Module):
    def __init__(self, arm, d_model, n_heads, T, impl="auto", chunk=16, qk_dim=None):
        super().__init__()
        # Dv: value width per head (always d_model / H). D: query-key width per head (qk_dim, default = Dv).
        self.H, self.Dv = n_heads, d_model // n_heads
        self.D = qk_dim or self.Dv
        kind, p, tied = parse_arm(arm)
        self.tied = tied
        self.W_q = nn.Linear(d_model, n_heads * self.D, bias=False)
        if not tied:
            self.W_k = nn.Linear(d_model, n_heads * self.D, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.scorer = Scorer(arm, n_heads, impl, chunk)
        self.register_buffer("allowed", lm_allowed_mask(T), persistent=False)
        self.min_keys = 8

    def forward(self, x):
        B, T, C = x.shape
        H, D, Dv = self.H, self.D, self.Dv
        q = self.W_q(x).view(B, T, H, D).transpose(1, 2)
        k = q if self.tied else self.W_k(x).view(B, T, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, Dv).transpose(1, 2)
        allowed = self.allowed[:T, :T]
        sc = self.scorer
        if sc.calib:
            sc.record_raw(q, k, allowed, self.min_keys)
        if sc.collect_grad:
            sc.hook_grad(q)
        logits = sc(q, k, causal=True)
        logits, att = scores_to_att(logits, allowed)
        if sc.collect:
            sc.record_stats(logits.masked_fill(~allowed, 0.0), att, q, k, allowed)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.W_o(y)


class Block(nn.Module):
    def __init__(self, arm, d_model, n_heads, T, mlp_mult, impl, chunk, qk_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(arm, d_model, n_heads, T, impl, chunk, qk_dim)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class CharLM(nn.Module):
    task = "lm"

    def __init__(self, arm, vocab, d_model=64, n_layers=2, n_heads=4, T=64, mlp_mult=4, impl="auto", chunk=16, qk_dim=None):
        super().__init__()
        self.arm, self.T, self.n_layers = arm, T, n_layers
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList([Block(arm, d_model, n_heads, T, mlp_mult, impl, chunk, qk_dim) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def scorers(self):
        return [b.attn.scorer for b in self.blocks]

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ self.tok_emb.weight.t()


# ============================================================================ init
# Embedding scales: the language model uses N(0, 0.02^2); other tasks pass their own mapping (parameter-name substring -> std).
LM_EMB_STD = {"tok_emb": 0.02, "pos_emb": 0.02}


def init_model(model, seed, emb_std=None):
    """Per-parameter-name seeded initialisation: a parameter that exists in two arms gets the same
    initial values in both (paired design). Untied W_k is an extra independent draw."""
    import zlib
    emb_std = LM_EMB_STD if emb_std is None else emb_std
    n_layers = getattr(model, "n_layers", 1)
    for name, p in model.named_parameters():
        g = torch.Generator().manual_seed((seed * 1_000_003 + zlib.crc32(name.encode())) % (2 ** 63))
        with torch.no_grad():
            if name.endswith("log_beta"):
                p.zero_()
            elif p.dim() == 1:
                if "ln" in name and name.endswith("weight"):
                    p.fill_(1.0)
                else:
                    p.zero_()
            elif any(k in name for k in emb_std):
                std_e = next(v for k, v in emb_std.items() if k in name)
                p.copy_(torch.randn(p.shape, generator=g) * std_e)
            else:
                fan_in = p.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                if "W_o" in name or "fc2" in name:
                    std /= math.sqrt(2 * n_layers)
                p.copy_(torch.randn(p.shape, generator=g) * std)
    return model
