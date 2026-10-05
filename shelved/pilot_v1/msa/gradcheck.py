"""Float64 gradient checks of the L^p kernel against the naive autograd reference."""
import json
import torch
from . import lp

P_FINITE = [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]


def _run(fn, q, k, g, mask=None):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    out = fn(q, k)
    gg = g if mask is None else g * mask
    (out * gg).sum().backward()
    o = out.detach()
    if mask is not None:
        o = o * mask
    return o, q.grad, k.grad


def _err(a, b):
    return [float((x - y).abs().max()) for x, y in zip(a, b)]


def run_gradcheck(seed=0, verbose=True):
    torch.manual_seed(seed)
    B, H, T, D = 2, 2, 24, 8
    results = {"float64_vs_naive": [], "autograd_gradcheck": [], "inf_ties": [], "fp32_vs_fp64": [], "ok": True}
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    g = torch.randn(B, H, T, T, dtype=torch.float64)
    causal = torch.tril(torch.ones(T, T, dtype=torch.float64), diagonal=-1)
    causal[0, 0] = 1.0

    def rec(table, **kw):
        table.append(kw)
        if verbose:
            print("  " + json.dumps(kw))

    tol = 1e-9
    for p in P_FINITE + [lp.INF]:
        for eps in ([0.0] if p == lp.INF else [1e-9, 1e-3]):
            ref = _run(lambda a, b: lp.lp_dist(a, b, p, eps, "naive"), q, k, g)
            impls = ["ckpt", "custom"] + (["matmul"] if p == 2.0 else [])
            for impl in impls:
                if p == lp.INF and impl == "ckpt":
                    pass
                res = _run(lambda a, b: lp.lp_dist(a, b, p, eps, impl, chunk=8), q, k, g)
                e = _err(res, ref)
                rec(results["float64_vs_naive"], p=p, eps=eps, impl=impl, causal=False,
                    max_abs_err_out_dq_dk=e, ok=max(e) < tol)
                results["ok"] &= max(e) < tol
            if p != 2.0 or True:
                refm = _run(lambda a, b: lp.lp_dist(a, b, p, eps, "naive"), q, k, g, mask=causal)
                res = _run(lambda a, b: lp.lp_dist(a, b, p, eps, "custom", chunk=8, causal=True), q, k, g, mask=causal)
                e = _err(res, refm)
                rec(results["float64_vs_naive"], p=p, eps=eps, impl="custom", causal=True,
                    max_abs_err_out_dq_dk=e, ok=max(e) < tol)
                results["ok"] &= max(e) < tol

    # torch.autograd.gradcheck on small shapes (finite differences, float64)
    sq = torch.randn(1, 1, 5, 3, dtype=torch.float64)
    sk = torch.randn(1, 1, 6, 3, dtype=torch.float64)
    for p in P_FINITE + [lp.INF]:
        eps = 0.0 if p == lp.INF else 1e-6
        impls = ["custom"] + (["matmul"] if p == 2.0 else [])
        for impl in impls:
            fn = lambda a, b: lp.lp_dist(a, b, p, eps, impl, chunk=2)
            try:
                ok = torch.autograd.gradcheck(fn, (sq.clone().requires_grad_(True), sk.clone().requires_grad_(True)),
                                              eps=1e-6, atol=1e-5, rtol=1e-4)
            except Exception as ex:  # noqa
                ok = False
            rec(results["autograd_gradcheck"], p=p, impl=impl, ok=bool(ok))
            results["ok"] &= bool(ok)

    # Chebyshev ties: integer data makes several coordinates attain the maximum.
    tq = torch.randint(-2, 3, (1, 1, 6, 4)).double()
    tk = torch.randint(-2, 3, (1, 1, 7, 4)).double()
    tg = torch.randn(1, 1, 6, 7, dtype=torch.float64)
    ref = _run(lambda a, b: lp.lp_dist(a, b, lp.INF, 0.0, "naive"), tq, tk, tg)
    res = _run(lambda a, b: lp.lp_dist(a, b, lp.INF, 0.0, "custom", chunk=4), tq, tk, tg)
    e = _err(res, ref)
    rec(results["inf_ties"], what="custom vs naive amax on tied data", max_abs_err_out_dq_dk=e, ok=max(e) < tol)
    results["ok"] &= max(e) < tol
    # even split: for a single pair with all-equal |diff|, each coordinate gets 1/D of the gradient
    a = torch.zeros(1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[[1.0, -1.0, 1.0, 1.0]]]], dtype=torch.float64)
    d = lp.lp_dist(a, b, lp.INF, 0.0, "custom", chunk=1)
    d.sum().backward()
    exp = torch.tensor([-0.25, 0.25, -0.25, -0.25], dtype=torch.float64)
    ok = bool(torch.allclose(a.grad.flatten(), exp))
    rec(results["inf_ties"], what="even split among 4 tied coordinates", grad=a.grad.flatten().tolist(), ok=ok)
    results["ok"] &= ok

    # fp32 custom vs fp64 naive: expected size of float error (not a pass/fail on exactness)
    q32, k32, g32 = q.float(), k.float(), g.float()
    for p in P_FINITE + [lp.INF]:
        eps = 0.0 if p == lp.INF else 1e-6
        ref = _run(lambda a, b: lp.lp_dist(a, b, p, eps, "naive"), q, k, g)
        res = _run(lambda a, b: lp.lp_dist(a, b, p, eps, "auto", chunk=8), q32, k32, g32)
        rel = [float((x.double() - y).abs().max() / y.abs().max()) for x, y in zip(res, ref)]
        rec(results["fp32_vs_fp64"], p=p, rel_max_err_out_dq_dk=rel, ok=max(rel) < 1e-4)
        results["ok"] &= max(rel) < 1e-4
    return results


if __name__ == "__main__":
    r = run_gradcheck()
    print("ALL OK" if r["ok"] else "FAILURES")
