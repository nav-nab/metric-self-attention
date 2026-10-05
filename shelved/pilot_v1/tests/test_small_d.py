"""Small query-key dimension in the LM and the uniform (null) control arm."""
import numpy as np
import torch

from msa import train
from msa.arms import ARMS, NULL_ARM
from msa.models import CharLM, init_model


def _lm(arm, qk_dim=None, seed=0):
    return init_model(CharLM(arm, 65, 64, 2, 4, 16, 4, "auto", 4, qk_dim), seed)


def test_default_qk_dim_unchanged():
    m = _lm("dot_tied")
    assert m.blocks[0].attn.W_q.weight.shape == (64, 64)
    assert m.blocks[0].attn.D == 16 and m.blocks[0].attn.Dv == 16


def test_small_qk_dim_shapes_and_forward():
    x = torch.randint(0, 65, (3, 16))
    for arm in ARMS + [NULL_ARM]:
        m = _lm(arm, qk_dim=2)
        a = m.blocks[0].attn
        assert a.W_q.weight.shape == (8, 64) and a.W_v.weight.shape == (64, 64)
        y = m(x)
        assert y.shape == (3, 16, 65) and torch.isfinite(y).all()


def test_shared_parameters_identical_across_qk_dims():
    a, b = _lm("lp_2", 2), _lm("lp_2", 4)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        if "W_q" in n1:
            continue
        assert torch.equal(p1, p2), n1


def test_uniform_arm_gives_uniform_attention():
    m = _lm(NULL_ARM, 2)
    m.eval()
    x = torch.randint(0, 65, (2, 16))
    sc = m.scorers()[0]
    sc.collect = True
    m(x)
    st = sc.stats
    assert max(st["logit_std"]) < 1e-6
    assert abs(st["entropy"][0] - st["entropy_uniform"]) < 1e-4


def test_calibration_and_run_id_for_small_d():
    cfg = train.make_cfg("lm", "lp_3", qk_dim=2, steps=2)
    assert train.run_id(cfg).startswith("lm_lm_D2_lp_3")
    consts = train.get_constants(cfg)
    assert len(consts) == 2 and all(np.isfinite(c["c"]) and c["c"] > 0 for c in consts)
    c0 = train.get_constants(train.make_cfg("uniform" and "lm", NULL_ARM, qk_dim=2, steps=2))
    assert all(np.isfinite(c["c"]) for c in c0)
