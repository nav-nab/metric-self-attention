import tempfile
import torch
from msa import train


def _cfg(**kw):
    return train.make_cfg("lm", "lp_3", steps=24, eval_every=12, diag_every=12, ckpt_every=8, log_every=4, eval_windows_mid=16,
                          batch=8, threads=1, **kw)


def test_resume_is_bit_exact():
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg()
        assert train.run(cfg, d, tag="a", verbose=False, stop_at_step=10) is None
        train.run(cfg, d, tag="a", verbose=False)
        train.run(cfg, d, tag="b", verbose=False)
        sa = torch.load(f"{d}/{train.run_id(cfg, 'a')}/final.pt", weights_only=False)["state_dict"]
        sb = torch.load(f"{d}/{train.run_id(cfg, 'b')}/final.pt", weights_only=False)["state_dict"]
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_shared_init_across_arms():
    a = train.build_model(train.make_cfg("lm", "dot_tied"), seed=3)
    b = train.build_model(train.make_cfg("lm", "lp_4"), seed=3)
    c = train.build_model(train.make_cfg("lm", "dot_untied"), seed=3)
    for n, p in a.named_parameters():
        assert torch.equal(p, dict(b.named_parameters())[n]) and torch.equal(p, dict(c.named_parameters())[n])
    assert "blocks.0.attn.W_k.weight" in dict(c.named_parameters())



def test_dot_constant_near_inverse_sqrt_D():
    cfg = train.make_cfg("lm", "dot_tied")      # head size 16 (d_model 64 / 4 heads)
    c = train.get_constants(cfg)[0]["c"]
    assert 0.15 < c < 0.35   # 1/sqrt(16) = 0.25
