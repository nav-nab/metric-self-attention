import math

import pytest
import torch

from msa.data import load_corpus, train_batch
from msa.model import build_model
from msa.scores import CROSSED_ARMS, EXTRA_ARMS, HEAD_SIZES

ALL_ARMS = CROSSED_ARMS + EXTRA_ARMS


def _attention(arm, D=4, seed=1):
    model = build_model(arm, D, seed)
    for b in model.blocks:
        b.attn.capture = True
    x, y = train_batch(load_corpus(), seed, 0, batch=4)
    model(x, y)
    return model, [b.attn.captured["attn"] for b in model.blocks]


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_forward_shape_loss_and_mask(arm):
    model, attns = _attention(arm)
    x, y = train_batch(load_corpus(), 1, 0, batch=4)
    logits, loss = model(x, y)
    assert logits.shape == (4, 64, 65)
    assert loss.item() == pytest.approx(math.log(65), abs=0.1)
    for A in attns:
        assert torch.allclose(A.sum(-1), torch.ones(1))
        assert (A.triu(1) == 0).all()                          # causal
        if arm == "dot_untied_unmasked":
            assert (A.diagonal(dim1=-2, dim2=-1) > 0).all()    # the position itself is allowed
        else:
            assert (A.diagonal(dim1=-2, dim2=-1)[..., 1:] == 0).all()   # diagonal masked
            assert (A[..., 0, 0] == 1).all()                   # position 0 attends to itself


def test_uniform_arm_weights_and_parameters():
    model, _ = _attention("uniform")
    attn = model.blocks[0].attn
    names = [n for n, _ in model.named_parameters()]
    assert not any("w_q" in n or "w_k" in n or "log_beta" in n for n in names)
    x = torch.zeros(1, 5, 64)
    y = attn.allowed[:5, :5].float()
    weights = y / y.sum(-1, keepdim=True)
    assert weights[3].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0, 0])
    assert weights[0, 0] == 1.0


@pytest.mark.parametrize("D", HEAD_SIZES)
def test_only_query_key_width_depends_on_D(D):
    model = build_model("lp_2", D, 1)
    attn = model.blocks[0].attn
    assert attn.w_q.weight.shape == (4 * D, 64)
    assert attn.w_v.weight.shape == (4 * 16, 64)
    assert not hasattr(attn, "w_k")
    assert build_model("dot_untied", D, 1).blocks[0].attn.w_k.weight.shape == (4 * D, 64)


def test_arms_with_the_same_seed_share_initial_parameters():
    ref = dict(build_model("dot_tied", 8, 3).named_parameters())
    for arm in ["lp_1", "lp_1.5", "lp_inf", "dot_untied", "dot_untied_unmasked", "uniform"]:
        other = dict(build_model(arm, 8, 3).named_parameters())
        for name, p in other.items():
            if name in ref:
                assert torch.equal(p, ref[name]), (arm, name)
    untied = dict(build_model("dot_untied", 8, 3).named_parameters())
    assert torch.equal(untied["blocks.0.attn.w_q.weight"], ref["blocks.0.attn.w_q.weight"])
    assert not torch.equal(untied["blocks.0.attn.w_k.weight"], untied["blocks.0.attn.w_q.weight"])
    other_seed = dict(build_model("dot_tied", 8, 4).named_parameters())
    assert not torch.equal(other_seed["tok_emb.weight"], ref["tok_emb.weight"])


def test_initial_values_and_log_beta():
    model = build_model("lp_2", 4, 1)
    p = dict(model.named_parameters())
    assert (p["blocks.0.attn.log_beta"] == 0).all()
    assert p["ln_f.weight"].eq(1).all() and p["blocks.0.fc1.bias"].eq(0).all()
    assert p["tok_emb.weight"].std().item() == pytest.approx(0.02, rel=0.1)
    assert p["blocks.0.attn.w_o.weight"].std().item() == pytest.approx(0.01, rel=0.1)   # 0.02/sqrt(4)
    assert "blocks.0.attn.w_q.bias" not in p


@pytest.mark.parametrize("arm", ["dot_tied", "lp_1", "lp_1.5", "lp_2", "lp_4", "lp_8", "lp_inf", "dot_untied"])
@pytest.mark.parametrize("D", [2, 16])
def test_gradients_are_finite_at_initialisation(arm, D):
    model = build_model(arm, D, 1)
    x, y = train_batch(load_corpus(), 1, 0, batch=4)
    _, loss = model(x, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name
