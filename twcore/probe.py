"""Tripwire fitting: trunk features, class-balanced logistic regression, post-hoc trajectory scores.

Trunk features are read with a temporary pre-hook on block `gate_layer` that captures the residual stream
and stops the forward pass, so only the trunk is computed. Gate positions of a sequence s are
0..len(s)-2 (the last token's state influences nothing), matching tripwire evaluation during decoding.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .gating import GateState, Tripwire, decoder_layers


class _Captured(Exception):
    pass


def _right_pad(seqs: Sequence[Sequence[int]], pad_id: int, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), T), dtype=torch.long, device=device)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(list(s), dtype=torch.long, device=device)
        attn[j, :len(s)] = 1
    return ids, attn


@torch.no_grad()
def trunk_states(model, state: GateState, gate_layer: int, seqs: List[List[int]], pad_id: int, batch_size: int):
    """Yields (index, h) with h = trunk features at gate positions gate_from..len(s)-2 (bf16, on device)."""
    device = next(model.parameters()).device
    holder = {}

    def cap(module, args, kwargs):
        holder["h"] = args[0] if len(args) > 0 else kwargs["hidden_states"]
        raise _Captured()

    handle = decoder_layers(model)[gate_layer].register_forward_pre_hook(cap, with_kwargs=True)
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]), i))
    try:
        for b0 in range(0, len(order), batch_size):
            idx = order[b0:b0 + batch_size]
            batch = [seqs[i] for i in idx]
            ids, attn = _right_pad(batch, pad_id, device)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            state.begin("full")
            state.valid = attn.bool() & (pos >= state.gate_from)
            try:
                model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
            except _Captured:
                pass
            finally:
                state.end()
            h = holder.pop("h")
            P = state.gate_from
            for j, i in enumerate(idx):
                L = len(seqs[i])
                yield i, h[j, P:max(L - 1, P)]
    finally:
        handle.remove()


def sample_features(model, state, gate_layer, seqs, pad_id, batch_size, per_seq: int, seed: int) -> torch.Tensor:
    """Up to `per_seq` random gate positions per sequence -> float32 features [N, d] on the model device."""
    feats = []
    for i, h in trunk_states(model, state, gate_layer, seqs, pad_id, batch_size):
        n = h.shape[0]
        if n == 0:
            continue
        r = random.Random(seed * 1000003 + i)  # independent of batch order
        sel = sorted(r.sample(range(n), min(per_seq, n)))
        feats.append(Tripwire.normalize(h[sel]))
    return torch.cat(feats, dim=0)


@torch.no_grad()
def max_scores(model, state, gate_layer, tw: Tripwire, seqs, pad_id, batch_size) -> List[float]:
    """Trajectory score S(s) = max over gate positions, from one full forward pass (post-hoc analysis)."""
    out = [float("-inf")] * len(seqs)
    for i, h in trunk_states(model, state, gate_layer, seqs, pad_id, batch_size):
        if h.shape[0]:
            out[i] = float(tw.score(h).max())
    return out


def last_prompt_features(model, state, gate_layer, prompts: Sequence[Sequence[int]], pad_id, batch_size) -> torch.Tensor:
    """RMS-normalized float32 trunk features at the last prompt token, one row per prompt (prompt-mode tripwires).
    A dummy token is appended so that the last prompt position is a gate position of the padded sequence; causal
    attention means it does not affect the feature."""
    seqs = [list(p) + [pad_id] for p in prompts]
    feats: List[torch.Tensor] = [None] * len(seqs)  # type: ignore[list-item]
    for i, h in trunk_states(model, state, gate_layer, seqs, pad_id, batch_size):
        if h.shape[0] == 0:
            raise RuntimeError("prompt shorter than the shared chat-template prefix")
        feats[i] = Tripwire.normalize(h[-1:])
    return torch.cat(feats, dim=0)


@torch.no_grad()
def prompt_scores(model, state, gate_layer, tw: Tripwire, prompts, pad_id, batch_size) -> List[float]:
    """Score at the last prompt token (post-hoc analysis of prompt-mode tripwires)."""
    X = last_prompt_features(model, state, gate_layer, prompts, pad_id, batch_size)
    return (((X - tw.mu) / tw.sigma) @ tw.a + tw.b).tolist()


def fit_logreg(pos: torch.Tensor, neg: torch.Tensor, l2: float, iters: int) -> Tuple[Tripwire, Dict[str, float]]:
    """Class-balanced logistic regression on standardized (already RMS-normalized) features."""
    device = pos.device
    X = torch.cat([pos, neg], dim=0)
    mu = X.mean(0)
    sigma = X.std(0) + 1e-6
    Z = (X - mu) / sigma
    y = torch.cat([torch.ones(len(pos), device=device), torch.zeros(len(neg), device=device)])
    w = torch.cat([torch.full((len(pos),), 0.5 / len(pos), device=device),
                   torch.full((len(neg),), 0.5 / len(neg), device=device)])
    a = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Z @ a + b
        loss = (w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum() + l2 * (a * a).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        loss0 = float(opt.step(closure))
        loss = float(closure())
    tw = Tripwire(X.shape[1]).to(device)
    with torch.no_grad():
        tw.mu.copy_(mu)
        tw.sigma.copy_(sigma)
        tw.a.copy_(a.detach())
        tw.b.copy_(b.detach())
    return tw, {"loss_start": loss0, "loss": loss}


def auroc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via ranks."""
    s = torch.cat([pos_scores, neg_scores]).double().cpu()
    ranks = torch.empty_like(s)
    order = torch.argsort(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.double)
    # average ranks for ties
    vals, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    sums = torch.zeros_like(vals).index_add_(0, inv, ranks)
    ranks = (sums / counts.double())[inv]
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
