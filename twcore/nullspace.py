"""Null-space LoRA baseline: the new adapter may not act along the input directions that the reference prompts
(and stored prompts of earlier tasks) use at each adapted projection, in the spirit of InfLoRA's
interference-free subspaces, here built from the same unlabeled reference data the tripwire uses.

For every wrapped projection m we estimate the top principal subspace of its inputs x on reference sequences of
the previous model (prompt + one sampled response) with a two-pass randomized eigendecomposition (Halko et al.):
  pass 1: Y = sum x x^T Omega (Omega Gaussian, d x s) and the total energy sum ||x||^2;  Q = orth(Y)
  pass 2: Bq = sum (Q^T x)(Q^T x)^T;  eig(Bq) = V diag(lam) V^T;  U = Q V
and keep the smallest k (<= max_rank) whose eigenvalues reach `energy` of the total energy. Memory is O(d s) per
projection, never O(d^2). The adapter then computes A (x - U U^T x): no first-order change along span(U). Unlike a
tripwire this gives no certificate: drift still passes through the remaining directions and through nonlinearities.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .gating import GateState, wrappers_of


def _right_pad(seqs, pad_id, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), T), dtype=torch.long, device=device)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(list(s), dtype=torch.long, device=device)
        attn[j, :len(s)] = 1
    return ids, attn


@torch.no_grad()
def _passes(model, state: GateState, seqs: List[List[int]], pad_id: int, batch_size: int, fn) -> None:
    """Run the model over seqs; for every wrapped projection call fn(name, X) with its inputs X [n, d] (float32) at
    real token positions after the shared template prefix."""
    device = next(model.parameters()).device
    wrappers = wrappers_of(model)
    holder = {"keep": None}
    handles = []

    def make_hook(name):
        def hook(module, args):
            x = args[0]
            fn(name, x[holder["keep"]].float())
        return hook

    for name, w in wrappers:
        handles.append(w.register_forward_pre_hook(make_hook(name)))
    try:
        order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), i))
        for b0 in range(0, len(order), batch_size):
            batch = [seqs[i] for i in order[b0:b0 + batch_size]]
            ids, attn = _right_pad(batch, pad_id, device)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            keep = attn.bool() & (pos >= state.gate_from)
            holder["keep"] = keep
            state.begin("full")
            try:
                state.valid = keep
                model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            finally:
                state.end()
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def reference_subspaces(model, state: GateState, seqs: Sequence[Sequence[int]], pad_id: int, batch_size: int,
                        energy: float, max_rank: int, oversample: int = 32, seed: int = 0
                        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict]]:
    """name -> U [d_in, k] (float32, on the model device) and per-projection statistics."""
    device = next(model.parameters()).device
    seqs = [list(s) for s in seqs if len(s) > 1]
    s = max_rank + oversample
    dims = {name: w.in_features for name, w in wrappers_of(model)}
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    omega = {n: torch.randn(d, min(s, d), generator=gen).to(device) for n, d in dims.items()}
    Y = {n: torch.zeros(d, min(s, d), device=device) for n, d in dims.items()}
    tot = {n: torch.zeros((), dtype=torch.float64, device=device) for n in dims}

    def pass1(name, X):
        Y[name] += X.t() @ (X @ omega[name])
        tot[name] += X.pow(2).sum(dtype=torch.float64)

    _passes(model, state, seqs, pad_id, batch_size, pass1)
    Q = {n: torch.linalg.qr(Y[n])[0] for n in dims}
    del Y, omega
    Bq = {n: torch.zeros(Q[n].shape[1], Q[n].shape[1], dtype=torch.float64, device=device) for n in dims}

    def pass2(name, X):
        Z = (X @ Q[name]).double()
        Bq[name] += Z.t() @ Z

    _passes(model, state, seqs, pad_id, batch_size, pass2)
    out, stats = {}, {}
    for n in dims:
        lam, V = torch.linalg.eigh(Bq[n])                      # ascending
        lam, V = lam.flip(0).clamp(min=0), V.flip(1)
        frac = (lam.cumsum(0) / tot[n].clamp(min=1e-30)).tolist()
        k = next((i + 1 for i, f in enumerate(frac) if f >= energy), len(frac))
        k = min(k, max_rank) if float(tot[n]) > 0 else 0   # no reference inputs: nothing to block
        out[n] = (Q[n] @ V[:, :k].to(Q[n].dtype)).contiguous()
        stats[n] = {"k": k, "energy_kept": float(frac[k - 1]) if k else 0.0, "d": dims[n]}
    return out, stats
