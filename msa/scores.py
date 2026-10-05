"""Arms, masks and score functions, including the L^p distance kernel.

The kernel is a custom autograd function. The forward pass works in chunks of
query rows and saves only q, k and the output. The backward pass recomputes the
coordinate differences chunk by chunk. A plain float64 version is kept as the
reference for the tests.
"""
import math
from dataclasses import dataclass

import torch

INF = math.inf
P_VALUES = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0, INF)
HEAD_SIZES = (2, 4, 8, 12, 16)
EPS_RATIO = 1e-12          # r of design section 7
DEFAULT_CHUNK = 16         # query rows per chunk of the kernel


# ---------------------------------------------------------------- arms

@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str              # "dot", "distance" or "uniform"
    p: float = None        # the exponent of a distance arm
    tied: bool = True      # query and key share one matrix
    diag_allowed: bool = False   # True only for the unmasked untied dot product

    @property
    def has_scores(self):
        return self.kind != "uniform"


def _p_label(p):
    return "inf" if p == INF else f"{p:g}"


def arm_name_for_p(p):
    return f"lp_{_p_label(p)}"


def parse_arm(name):
    if name == "dot_tied":
        return ArmSpec(name, "dot")
    if name == "dot_untied":
        return ArmSpec(name, "dot", tied=False)
    if name == "dot_untied_unmasked":
        return ArmSpec(name, "dot", tied=False, diag_allowed=True)
    if name == "uniform":
        return ArmSpec(name, "uniform")
    for p in P_VALUES:
        if name == arm_name_for_p(p):
            return ArmSpec(name, "distance", p=p)
    raise ValueError(f"unknown arm {name!r}")


# the nine arms crossed with the head sizes
CROSSED_ARMS = ("dot_tied",) + tuple(arm_name_for_p(p) for p in P_VALUES) + ("dot_untied",)
EXTRA_ARMS = ("uniform", "dot_untied_unmasked")


# ---------------------------------------------------------------- masks

def allowed_mask(T, diag_allowed=False):
    """Boolean (T, T) matrix, True where query i may use key j.

    Causal, with the diagonal masked, except that position 0 may use itself.
    With diag_allowed the diagonal is allowed everywhere.
    """
    if diag_allowed:
        return torch.tril(torch.ones(T, T, dtype=torch.bool))
    m = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)
    m[0, 0] = True
    return m


# ---------------------------------------------------------------- the kernel

def _abs_pow(a, e):
    """a**e for a >= 0, with cheap paths for the exponents that occur."""
    if e == 1:
        return a
    if e == 2:
        return a * a
    if e == 3:
        return a * a * a
    if e == 4:
        s = a * a
        return s * s
    if e == 7:
        s = a * a
        s4 = s * s
        return s4 * s * a
    if e == 8:
        s = a * a
        s = s * s
        return s * s
    return a.pow(e)


def _root(s, p):
    if p == 2:
        return s.sqrt()
    if p == 4:
        return s.sqrt().sqrt()
    if p == 8:
        return s.sqrt().sqrt().sqrt()
    return s.pow(1.0 / p)


def _forward_chunk(qc, k, p, eps):
    """Distances from the query rows qc (B,H,c,D) to the keys k (B,H,n,D): (B,H,c,n)."""
    a = (qc.unsqueeze(3) - k.unsqueeze(2)).abs()
    if p == INF:
        return a.amax(-1)
    if p == 1:
        return a.sum(-1)
    return _root(_abs_pow(a, p).sum(-1) + eps, p)


def _backward_chunk(qc, k, out_c, go_c, p):
    """Gradient pieces for one chunk: g = d(loss)/d(difference), shape (B,H,c,n,D)."""
    diff = qc.unsqueeze(3) - k.unsqueeze(2)
    sign = torch.sign(diff)
    if p == INF:
        a = diff.abs()
        ties = (a == out_c.unsqueeze(-1)).to(diff.dtype)
        weight = ties / ties.sum(-1, keepdim=True)
        return go_c.unsqueeze(-1) * sign * weight
    if p == 1:
        return go_c.unsqueeze(-1) * sign
    # entries with zero upstream gradient (masked entries) must give zero, even if out_c is 0
    coef = torch.where(go_c == 0, torch.zeros_like(go_c), go_c * out_c.pow(1.0 - p))
    return coef.unsqueeze(-1) * sign * _abs_pow(diff.abs(), p - 1.0)


class _LpDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, p, eps, chunk, causal):
        q = q.contiguous()
        k = k.contiguous()
        T = q.shape[2]
        out = q.new_zeros(q.shape[0], q.shape[1], T, k.shape[2])
        for i0 in range(0, T, chunk):
            i1 = min(i0 + chunk, T)
            n = min(i1, k.shape[2]) if causal else k.shape[2]
            out[:, :, i0:i1, :n] = _forward_chunk(q[:, :, i0:i1], k[:, :, :n], p, eps)
        ctx.save_for_backward(q, k, out)
        ctx.p, ctx.chunk, ctx.causal = p, chunk, causal
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, out = ctx.saved_tensors
        p, chunk, causal = ctx.p, ctx.chunk, ctx.causal
        T = q.shape[2]
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        for i0 in range(0, T, chunk):
            i1 = min(i0 + chunk, T)
            n = min(i1, k.shape[2]) if causal else k.shape[2]
            g = _backward_chunk(q[:, :, i0:i1], k[:, :, :n], out[:, :, i0:i1, :n],
                                grad_out[:, :, i0:i1, :n], p)
            grad_q[:, :, i0:i1] = g.sum(3)
            grad_k[:, :, :n] -= g.sum(2)
        return grad_q, grad_k, None, None, None, None


def lp_distance(q, k, p, eps=0.0, chunk=DEFAULT_CHUNK, causal=False):
    """Pairwise L^p distances between the rows of q and k, both (B, H, T, D).

    The result has shape (B, H, T, T). For p = 1 and p = infinity no epsilon is used.
    With causal=True only the entries with j below the end of the chunk of row i are
    computed, and the rest are zero. The caller must mask the entries with j > i.
    """
    return _LpDistance.apply(q, k, float(p), float(eps), int(chunk), bool(causal))


def lp_distance_reference(q, k, p, eps=0.0):
    """The plain float64 version, built from ordinary autograd operations."""
    a = (q.double().unsqueeze(-2) - k.double().unsqueeze(-3)).abs()
    if p == INF:
        return a.amax(-1)
    if p == 1:
        return a.sum(-1)
    return (a.pow(p).sum(-1) + eps).pow(1.0 / p)


# ---------------------------------------------------------------- scores

def raw_scores(spec, q, k, eps=0.0, chunk=DEFAULT_CHUNK):
    """The raw score before scaling: the dot product, or minus the distance."""
    if spec.kind == "dot":
        return q @ k.transpose(-1, -2)
    if spec.kind == "distance":
        return -lp_distance(q, k, spec.p, eps, chunk, causal=True)
    raise ValueError("the uniform arm has no scores")
