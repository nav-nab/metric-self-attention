import torch
from msa import gradcheck


def test_kernel_gradients_float64():
    r = gradcheck.run_gradcheck(verbose=False)
    assert r["ok"], [x for k in ("float64_vs_naive", "autograd_gradcheck", "inf_ties", "fp32_vs_fp64") for x in r[k] if not x["ok"]]
