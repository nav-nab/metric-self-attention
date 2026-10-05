import math

import pytest
import torch

from msa.scores import (INF, P_VALUES, allowed_mask, lp_distance, lp_distance_reference,
                        parse_arm)

PS = list(P_VALUES)


def _inputs(shape=(2, 3, 20, 5), seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(shape, generator=g, dtype=dtype)
    k = torch.randn(shape, generator=g, dtype=dtype)
    return q, k


def _eps_for(q, k, p):
    """The relative epsilon of design section 7: r times the typical sum."""
    if p in (1.0, INF):
        return 0.0
    s = (q.double().unsqueeze(-2) - k.double().unsqueeze(-3)).abs().pow(p).sum(-1)
    return 1e-12 * s.median().item()


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("chunk", [7, 16, 64])
def test_value_and_gradient_match_float64_reference(p, chunk):
    q, k = _inputs()
    eps = _eps_for(q, k, p)
    w = torch.randn(2, 3, 20, 20, generator=torch.Generator().manual_seed(1))
    q1, k1 = q.clone().requires_grad_(), k.clone().requires_grad_()
    q2, k2 = q.double().requires_grad_(), k.double().requires_grad_()
    out = lp_distance(q1, k1, p, eps, chunk=chunk)
    ref = lp_distance_reference(q2, k2, p, eps)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-6)
    (out * w).sum().backward()
    (ref * w.double()).sum().backward()
    torch.testing.assert_close(q1.grad.double(), q2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k1.grad.double(), k2.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("p", PS)
def test_causal_mode_agrees_on_allowed_entries(p):
    q, k = _inputs(shape=(2, 2, 24, 4))
    eps = _eps_for(q, k, p)
    low = torch.tril(torch.ones(24, 24, dtype=torch.bool))
    w = torch.randn(2, 2, 24, 24, generator=torch.Generator().manual_seed(2)) * low
    q1, k1 = q.clone().requires_grad_(), k.clone().requires_grad_()
    q2, k2 = q.double().requires_grad_(), k.double().requires_grad_()
    out = lp_distance(q1, k1, p, eps, chunk=5, causal=True)
    ref = lp_distance_reference(q2, k2, p, eps)
    torch.testing.assert_close(out.double()[..., low], ref[..., low], rtol=1e-5, atol=1e-6)
    (out * w).sum().backward()
    (ref * w.double()).sum().backward()
    torch.testing.assert_close(q1.grad.double(), q2.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k1.grad.double(), k2.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("p", PS)
def test_gradient_is_finite_at_exact_matches(p):
    q, _ = _inputs(shape=(1, 2, 12, 4))
    k = q.clone()                       # every query equals its own key
    eps = _eps_for(q, q + torch.randn_like(q), p)
    q1, k1 = q.clone().requires_grad_(), k.clone().requires_grad_()
    out = lp_distance(q1, k1, p, eps)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(q1.grad).all() and torch.isfinite(k1.grad).all()
    if p not in (1.0, INF):
        floor = eps ** (1.0 / p)        # the smallest possible distance
        assert out.diagonal(dim1=-2, dim2=-1).min().item() == pytest.approx(floor, rel=1e-4)


def test_infinity_splits_the_gradient_evenly_among_ties():
    q = torch.zeros(1, 1, 1, 3, requires_grad=True)
    k = torch.tensor([[[[1.0, -1.0, 0.5]]]], requires_grad=True)
    out = lp_distance(q, k, INF)
    out.sum().backward()
    # coordinates 0 and 1 tie for the maximum, so each gets half the gradient
    torch.testing.assert_close(q.grad, torch.tensor([[[[-0.5, 0.5, 0.0]]]]))
    ref_q = torch.zeros(1, 1, 1, 3, requires_grad=True)
    lp_distance_reference(ref_q, k.detach(), INF).sum().backward()
    torch.testing.assert_close(q.grad.double(), ref_q.grad.double())


def test_p2_uses_direct_differences_and_is_accurate_for_close_points():
    # Nearby points with large coordinates: the matrix-product expansion loses all accuracy here.
    g = torch.Generator().manual_seed(3)
    k = torch.randn(1, 1, 16, 8, generator=g) + 100.0
    q = k + 1e-3 * torch.randn(1, 1, 16, 8, generator=g)
    out = lp_distance(q, k, 2.0, 0.0)
    ref = lp_distance_reference(q, k, 2.0, 0.0)
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=0.0)


def test_only_q_k_and_output_are_saved_for_backward():
    q, k = _inputs(shape=(1, 2, 8, 3))
    q.requires_grad_(), k.requires_grad_()
    saved = []

    def pack(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lp_distance(q, k, 3.0, 1e-12)
    assert sorted(saved) == sorted([(1, 2, 8, 3), (1, 2, 8, 3), (1, 2, 8, 8)])


def test_infinity_has_no_epsilon():
    q, k = _inputs(shape=(1, 1, 6, 3))
    torch.testing.assert_close(lp_distance(q, k, INF, 5.0), lp_distance(q, k, INF, 0.0))


def test_masks():
    m = allowed_mask(6)
    assert m[0, 0] and not m[1, 1] and not m[5, 5]       # only position 0 may use itself
    assert m[3, 2] and m[3, 0] and not m[3, 4]           # causal
    assert m.sum().item() == 15 + 1                      # strictly lower triangle plus (0, 0)
    u = allowed_mask(6, diag_allowed=True)
    assert u.diagonal().all() and not u[2, 3] and u.sum().item() == 21


def test_arm_parsing():
    assert parse_arm("lp_inf").p == INF and parse_arm("lp_1.5").p == 1.5 and parse_arm("lp_8").p == 8.0
    assert parse_arm("dot_tied").tied and not parse_arm("dot_untied").tied
    assert parse_arm("dot_untied_unmasked").diag_allowed and not parse_arm("dot_untied").diag_allowed
    with pytest.raises(ValueError):
        parse_arm("lp_5")
