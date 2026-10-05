"""L^p distance kernel for attention scores.

Computes dist[b,h,i,j] = (sum_d |q[b,h,i,d] - k[b,h,j,d]|^p + eps)^(1/p) for finite p,
and dist = max_d |q - k| for p = inf (no epsilon, ties split evenly in the gradient).

Implementations (all fp32 in the experiments, fp64 in the reference tests):
  naive   : plain autograd over the full (B,H,Tq,Tk,D) difference tensor (reference).
  ckpt    : query-chunked, torch.utils.checkpoint around each chunk.
  custom  : query-chunked custom autograd Function. Saves only q, k and the output
            distances; the backward recomputes differences chunk by chunk.
  matmul  : p = 2 only. S = |a|^2 + |b|^2 - 2 a.b (clamped at 0), no (Tq,Tk,D) tensor.

`causal=True` (custom only): chunk i only touches keys [0, end of chunk), so about half
of the pairs are skipped when the caller masks the future anyway. Entries outside the
computed block are returned as 0 and MUST be masked by the caller.

Analytic gradient (finite p): d dist/d q = dist^(1-p) |Delta|^(p-1) sign(Delta), and the
negative of this for k, with Delta = q - k. This holds with the epsilon inside the root.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint

INF = float("inf")


def _abs_pow(x, p):
    """|x|^p with cheap special cases (avoids the generic pow)."""
    if p == 1.0:
        return x.abs()
    if p == 1.5:
        a = x.abs()
        return a * a.sqrt()
    if p == 2.0:
        return x * x
    if p == 3.0:
        return x.abs() * x * x
    if p == 4.0:
        x2 = x * x
        return x2 * x2
    if p == 8.0:
        x2 = x * x
        x4 = x2 * x2
        return x4 * x4
    return x.abs().pow(p)


def _abs_pow_m1_sign(x, p):
    """|x|^(p-1) * sign(x)."""
    if p == 1.0:
        return torch.sign(x)
    if p == 1.5:
        return torch.sign(x) * x.abs().sqrt()
    if p == 2.0:
        return x
    if p == 3.0:
        return x * x.abs()
    if p == 4.0:
        return x * x * x
    if p == 8.0:
        x2 = x * x
        return x * (x2 * (x2 * x2))
    return x.abs().pow(p - 1.0) * torch.sign(x)


def _root(s, p):
    """s^(1/p) with cheap special cases."""
    if p == 1.0:
        return s
    if p == 2.0:
        return s.sqrt()
    if p == 4.0:
        return s.sqrt().sqrt()
    if p == 8.0:
        return s.sqrt().sqrt().sqrt()
    return s.pow(1.0 / p)


def _inv_pow(d, p):
    """d^(1-p)."""
    if p == 1.0:
        return torch.ones_like(d)
    if p == 2.0:
        return d.reciprocal()
    return d.pow(1.0 - p)


def power_sum(q, k, p):
    """S[b,h,i,j] = sum_d |q_i - k_j|^p (finite p), naive, full tensor."""
    diff = q.unsqueeze(-2) - k.unsqueeze(-3)
    return _abs_pow(diff, p).sum(-1)


# --------------------------------------------------------------------------- naive
def naive(q, k, p, eps):
    diff = q.unsqueeze(-2) - k.unsqueeze(-3)  # B,H,Tq,Tk,D
    if p == INF:
        return diff.abs().amax(-1)
    S = _abs_pow(diff, p).sum(-1)
    return _root(S + eps, p)


def ckpt(q, k, p, eps, chunk):
    outs = []
    for i in range(0, q.shape[-2], chunk):
        qc = q[..., i:i + chunk, :]
        outs.append(checkpoint(naive, qc, k, p, eps, use_reentrant=False))
    return torch.cat(outs, dim=-2)


def matmul_l2(q, k, eps):
    """p = 2 via the matmul expansion; sum clamped at zero before eps and the root."""
    a2 = (q * q).sum(-1, keepdim=True)
    b2 = (k * k).sum(-1).unsqueeze(-2)
    S = (a2 + b2 - 2.0 * (q @ k.transpose(-1, -2))).clamp_min(0.0)
    return (S + eps).sqrt()


# --------------------------------------------------------------------------- custom
class LpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, p, eps, chunk, causal):
        Tq, Tk = q.shape[-2], k.shape[-2]
        out = q.new_zeros(*q.shape[:-2], Tq, Tk)
        for i in range(0, Tq, chunk):
            j1 = min(Tk, i + chunk) if causal else Tk
            qc = q[..., i:i + chunk, :]
            diff = qc.unsqueeze(-2) - k[..., :j1, :].unsqueeze(-3)
            if p == INF:
                out[..., i:i + chunk, :j1] = diff.abs().amax(-1)
            else:
                S = _abs_pow(diff, p).sum(-1)
                out[..., i:i + chunk, :j1] = _root(S + eps, p)
        ctx.save_for_backward(q, k, out)
        ctx.p, ctx.chunk, ctx.causal = p, chunk, causal
        return out

    @staticmethod
    def backward(ctx, g):
        q, k, d = ctx.saved_tensors
        p, chunk, causal = ctx.p, ctx.chunk, ctx.causal
        Tq, Tk = q.shape[-2], k.shape[-2]
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        for i in range(0, Tq, chunk):
            j1 = min(Tk, i + chunk) if causal else Tk
            qc = q[..., i:i + chunk, :]
            diff = qc.unsqueeze(-2) - k[..., :j1, :].unsqueeze(-3)  # B,H,c,j1,D
            gc = g[..., i:i + chunk, :j1]
            dc = d[..., i:i + chunk, :j1]
            if p == INF:
                a = diff.abs()
                m = (a == dc.unsqueeze(-1)).to(diff.dtype)
                m = m / m.sum(-1, keepdim=True)
                t = gc.unsqueeze(-1) * m * torch.sign(diff)
            else:
                coef = gc * _inv_pow(dc, p)
                t = coef.unsqueeze(-1) * _abs_pow_m1_sign(diff, p)
            dq[..., i:i + chunk, :] = t.sum(-2)
            dk[..., :j1, :] -= t.sum(-3)
        return dq, dk, None, None, None, None


def lp_dist(q, k, p, eps=0.0, impl="auto", chunk=16, causal=False):
    """Distance matrix (B,H,Tq,Tk). impl in {auto, naive, ckpt, custom, matmul}."""
    p = float(p)
    if impl == "auto":      # direct differences for every p; the p = 2 matmul expansion is opt-in (it loses accuracy when distances are tiny)
        impl = "custom"
    if impl == "matmul":
        assert p == 2.0
        return matmul_l2(q, k, eps)
    if impl == "naive":
        return naive(q, k, p, eps)
    if impl == "ckpt":
        return ckpt(q, k, p, eps, chunk)
    if impl == "custom":
        return LpFn.apply(q, k, p, float(eps), int(chunk), bool(causal))
    raise ValueError(impl)
