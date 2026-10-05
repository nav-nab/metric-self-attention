"""MQAR variant A model: one attention layer, one head of dimension D, no MLP, no positions.

A context token is a key embedding plus a value embedding. A query token is the same key embedding plus a blank
value embedding. Queries attend only to the P context tokens (query-to-context mask). The output is a
classification over the values, with the loss at the query positions only.

The score module, the arms and the shared-initialisation rule come from the main package (msa).
"""
import torch
import torch.nn as nn

from msa.arms import parse_arm
from msa.models import Scorer, init_model

# Embedding scales used by the recall model (N(0, 1)); the language model uses N(0, 0.02^2).
EMB_STD = {"key_emb": 1.0, "val_emb": 1.0}


class MQARModel(nn.Module):
    task = "mqar"

    def __init__(self, arm, n_keys=256, n_vals=64, d_model=64, D=16, impl="auto", chunk=16):
        super().__init__()
        kind, p, tied = parse_arm(arm)
        self.arm, self.tied, self.n_vals, self.D = arm, tied, n_vals, D
        self.key_emb = nn.Embedding(n_keys, d_model)
        self.val_emb = nn.Embedding(n_vals + 1, d_model)     # last row = blank value
        self.ln_in = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, D, bias=False)
        if not tied:
            self.W_k = nn.Linear(d_model, D, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vals)
        self.scorer = Scorer(arm, 1, impl, chunk)
        self.min_keys = 1

    def scorers(self):
        return [self.scorer]

    def forward(self, ctx, qry, ablate_context=False, return_att=False):
        xc = self.key_emb(ctx[..., 0]) + self.val_emb(ctx[..., 1])               # B,P,d
        xq = self.key_emb(qry) + self.val_emb.weight[self.n_vals]                 # B,Q,d
        uc, uq = self.ln_in(xc), self.ln_in(xq)
        zq = self.W_q(uq).unsqueeze(1)                                            # B,1,Q,D
        zk = (self.W_q(uc) if self.tied else self.W_k(uc)).unsqueeze(1)           # B,1,P,D
        sc = self.scorer
        allowed = torch.ones(zq.shape[-2], zk.shape[-2], dtype=torch.bool)
        if sc.calib:
            sc.record_raw(zq, zk, allowed, self.min_keys)
        if sc.collect_grad:
            sc.hook_grad(zq)
        logits = sc(zq, zk, causal=False)
        att = torch.softmax(logits, dim=-1)
        if sc.collect:
            sc.record_stats(logits, att, zq, zk, allowed)
        if ablate_context:
            a = torch.zeros_like(xq)
        else:
            a = att[:, 0] @ self.W_v(uc)                                          # B,Q,d
        out = self.head(self.ln_out(xq + self.W_o(a)))
        if return_att:
            return out, att[:, 0], logits[:, 0], zk[:, 0]
        return out


def build(cfg, seed):
    m = MQARModel(cfg["arm"], cfg["n_keys"], cfg["n_vals"], cfg["d_model"], cfg["D"], cfg["impl"], cfg["chunk"])
    return init_model(m, seed, emb_std=EMB_STD)
