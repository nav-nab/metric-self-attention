import torch

from msa.data import (CONTEXT, Corpus, calibration_batches, load_corpus, stable_seed,
                      train_batch, validation_windows)


def test_corpus_split_and_vocabulary():
    c = load_corpus()
    assert c.vocab_size == 65
    assert len(c.train) + len(c.val) == 1115394
    assert len(c.train) == int(0.9 * 1115394)
    assert c.chars == sorted(c.chars)


def test_same_seed_and_step_give_the_same_batch():
    c = load_corpus()
    x1, y1 = train_batch(c, 3, 17)
    x2, y2 = train_batch(c, 3, 17)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    x3, _ = train_batch(c, 4, 17)
    x4, _ = train_batch(c, 3, 18)
    assert not torch.equal(x1, x3) and not torch.equal(x1, x4)


def test_batch_shape_and_targets():
    c = load_corpus()
    x, y = train_batch(c, 1, 0)
    assert x.shape == (64, CONTEXT) and y.shape == (64, CONTEXT)
    assert torch.equal(x[:, 1:], y[:, :-1])           # targets are inputs shifted by one
    assert x.max() < c.vocab_size


def test_batches_come_from_the_training_part_only():
    # a synthetic corpus: the training part is all "a", the validation part is all "b"
    c = Corpus("a" * 900 + "b" * 100)
    assert len(c.train) == 900 and len(c.val) == 100
    for step in range(300):
        x, y = train_batch(c, 5, step)
        assert x.max() == 0 and y.max() == 0


def test_validation_windows_do_not_overlap_and_drop_the_partial_one():
    c = load_corpus()
    x, y = validation_windows(c)
    assert len(x) == (len(c.val) - 1) // CONTEXT
    assert torch.equal(x.reshape(-1), c.val[: len(x) * CONTEXT])
    assert torch.equal(y.reshape(-1), c.val[1: len(x) * CONTEXT + 1])


def test_calibration_batches_are_fixed():
    c = load_corpus()
    a, b = calibration_batches(c, 3), calibration_batches(c, 3)
    assert all(torch.equal(u[0], v[0]) for u, v in zip(a, b))


def test_stable_seed_is_deterministic():
    assert stable_seed("init", 1, "tok_emb.weight") == stable_seed("init", 1, "tok_emb.weight")
    assert stable_seed("init", 1, "a") != stable_seed("init", 2, "a")
