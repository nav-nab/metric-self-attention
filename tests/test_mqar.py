import numpy as np
from msa import mqar_gen


def test_generator_properties():
    r = mqar_gen.check_generator(P=16, Q=8, n=5000, check_n=500)
    assert r["ok"]


def test_vectorised_matches_spec():
    rng = np.random.default_rng(0)
    ctx, qry, tgt = mqar_gen.make_batch(rng, 300, 12, 6, 64, 16)
    for b in range(300):
        d = dict(zip(ctx[b, :, 0], ctx[b, :, 1]))
        assert len(d) == 12 and len(set(qry[b])) == 6
        assert all(d[k] == t for k, t in zip(qry[b], tgt[b]))


def test_fixed_sets_reproducible():
    a, b = mqar_gen.fixed_set(5, 50, 8, 4, 64, 16), mqar_gen.fixed_set(5, 50, 8, 4, 64, 16)
    assert all((x == y).all() for x, y in zip(a, b))
