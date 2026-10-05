"""Tiny Shakespeare, character level. 90/10 contiguous train/validation split."""
import os
import urllib.request
import numpy as np
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MD5 = "6fb458f1232090904fb40fe944165e91"
def _find_data_dir():
    """The corpus lives in the data/ folder of the repository. This code now sits two levels below it (shelved/pilot_v1/),
    so look upwards for a data/ folder that holds the corpus; if there is none, use data/ next to this folder (the download target)."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(5):
        d = os.path.dirname(d)
        if os.path.exists(os.path.join(d, "data", "tinyshakespeare.txt")):
            return os.path.join(d, "data")
    return os.path.join(os.path.dirname(here), "data")


DATA_DIR = _find_data_dir()


def load_text(path=None):
    path = path or os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return open(path, encoding="utf-8").read()


class LMData:
    def __init__(self, T=64, val_frac=0.1, path=None):
        text = load_text(path)
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab = len(chars)
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = int(len(ids) * val_frac)
        self.train, self.val = ids[:-n_val], ids[-n_val:]
        self.T = T
        # fixed validation windows (non-overlapping, identical for every arm)
        nw = (len(self.val) - 1) // T
        self.val_x = torch.stack([self.val[i * T:i * T + T] for i in range(nw)])
        self.val_y = torch.stack([self.val[i * T + 1:i * T + T + 1] for i in range(nw)])

    def batch(self, rng, B):
        T = self.T
        starts = rng.integers(0, len(self.train) - T - 1, size=B)
        x = torch.stack([self.train[s:s + T] for s in starts])
        y = torch.stack([self.train[s + 1:s + T + 1] for s in starts])
        return x, y
