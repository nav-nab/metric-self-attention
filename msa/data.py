"""Corpus, split, seeds and batches.

Batches are a pure function of (seed, step). A run that resumes at step t
therefore sees exactly the batches an uninterrupted run would see.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "tinyshakespeare.txt"
CORPUS_MD5 = "6fb458f1232090904fb40fe944165e91"

CONTEXT = 64
BATCH = 64
TRAIN_FRACTION = 0.9

FINAL_SEEDS = (1, 2, 3, 4, 5)
SCAN_SEED = 100
CALIBRATION_SEED = 999


def stable_seed(*parts):
    """A 63-bit integer derived from the parts. It does not depend on the Python hash seed."""
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


class Corpus:
    """The encoded text, split into a training part and a validation part."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n_train = int(TRAIN_FRACTION * len(ids))
        self.train = ids[:n_train]
        self.val = ids[n_train:]


@lru_cache(maxsize=None)
def load_corpus(path=None, check_md5=True):
    path = Path(path) if path is not None else CORPUS_PATH
    raw = path.read_bytes()
    if check_md5 and path == CORPUS_PATH:
        found = hashlib.md5(raw).hexdigest()
        if found != CORPUS_MD5:
            raise ValueError(f"corpus md5 is {found}, expected {CORPUS_MD5}")
    return Corpus(raw.decode("utf-8"))


def train_batch(corpus, seed, step, batch=BATCH, context=CONTEXT):
    """The training batch of a given seed and step: inputs and targets, both (batch, context)."""
    gen = torch.Generator().manual_seed(stable_seed("data", seed, step))
    starts = torch.randint(0, len(corpus.train) - context, (batch,), generator=gen)
    window = corpus.train[starts.unsqueeze(1) + torch.arange(context + 1)]
    return window[:, :-1].contiguous(), window[:, 1:].contiguous()


def validation_windows(corpus, context=CONTEXT):
    """All non-overlapping validation windows. A final partial window is dropped."""
    n = (len(corpus.val) - 1) // context
    idx = torch.arange(n).unsqueeze(1) * context + torch.arange(context + 1)
    window = corpus.val[idx]
    return window[:, :-1].contiguous(), window[:, 1:].contiguous()


def calibration_batches(corpus, n_batches):
    """A fixed list of batches, drawn with the calibration seed and shared by all runs."""
    return [train_batch(corpus, CALIBRATION_SEED, i) for i in range(n_batches)]
