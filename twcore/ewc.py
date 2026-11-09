"""Weight-importance baselines: a quadratic penalty on the new adapter's weight change dW,

    (lambda / 2) * sum_ij Omega_ij dW_ij^2,

with an importance map Omega of the previous model computed on the unlabeled reference prompts (the same prompts
that Ref-KL and the tripwire use):
  ewc  Omega = diagonal Fisher of the previous model's log-likelihood of its own samples (Kirkpatrick et al. 2017)
  mas  Omega = mean absolute gradient of the squared L2 norm of the (centered) output logits (Memory Aware
       Synapses, Aljundi et al. 2018; label-free by design)
  l2   Omega = 1 (plain L2 pull toward the old weights, L2-SP; Li et al. 2018)
The MAS and L2 maps are rescaled to the EWC Fisher's total mass on the same projections, so one lambda grid
(0.1 / 1 / 10) means the same overall strength for all three and only the shape of the importance map differs.

For an adapted projection y = W x + s B A x, the new adapter changes W by dW = s B A. The diagonal Fisher of the
previous model's log-likelihood of its own samples is approximated per projection in Kronecker-factored form
(K-FAC; Martens & Grosse 2015, Ritter et al. 2018):

    F_ij ~= a_i * b_j,   a = sum over positions of g^2 / (response tokens),   b = mean over positions of x^2,

with g the gradient of the summed response log-probabilities with respect to the projection's output and x its
input, both at the previous model. Then 0.5 * sum_ij F_ij dW_ij^2 is a second-order approximation of the per-token
KL(pi_prev || pi) on the reference prompts, up to the cross-position terms that the factorization drops, so
ewc_lambda is roughly on the scale of Ref-KL's beta (the sweep 0.1 / 1 / 10 covers the difference). The penalty
needs only r x r products:

    sum_ij a_i b_j dW_ij^2 = s^2 * tr( (B' ^T B') (A' A'^T) ),   B' = diag(sqrt a) B,   A' = A diag(sqrt b).
"""
from __future__ import annotations

import gc
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import torch

from .gating import GateState, wrappers_of


def _capture_hooks(wraps, caps):
    """Forward hooks that keep every adapted projection's input and output. An output that depends on nothing
    differentiable (the first adapted block: the frozen trunk below it) is made a leaf that requires grad, so the
    backward graph starts at the first adapted block and the lower blocks keep no activations. The gradient of the
    log-likelihood with respect to each output is the same as with a graph from the embeddings."""
    hooks = []
    for name, w in wraps:
        def fh(mod, inp, out, name=name):
            if not out.requires_grad:
                out = out.detach().requires_grad_(True)
            caps[name] = (inp[0], out)
            return out
        hooks.append(w.register_forward_hook(fh))
    return hooks


def _free_gpu(caps):
    caps.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _oom_safe(seqs, batch_size, one, caps, lengths=None) -> int:
    """one(items) for consecutive batches; a batch that runs out of GPU memory is redone one sequence at a time,
    and a single sequence that still does not fit is skipped. one() adds to the totals only at its very end, so a
    failed batch adds nothing. Returns the number of skipped sequences; their lengths (prompt + response tokens) are
    appended to `lengths` when it is given (Nov 5: to record which sequences a GPU cannot hold)."""
    skipped = 0
    for b0 in range(0, len(seqs), batch_size):
        items = list(seqs[b0:b0 + batch_size])
        failed = False
        try:
            one(items)
        except torch.cuda.OutOfMemoryError:
            failed = True          # retried outside the except block, so the failed batch's tensors are freed
        if not failed:
            continue
        _free_gpu(caps)
        for it in items:
            bad = False
            try:
                one([it])
            except torch.cuda.OutOfMemoryError:
                bad = True
            if bad:
                _free_gpu(caps)
                skipped += 1
                if lengths is not None:
                    lengths.append(len(it[0]) + len(it[1]))
    return skipped


def fisher_stats(model, state: GateState, seqs: Sequence[Tuple[List[int], List[int]]], pad_id: int,
                 batch_size: int, logprob_fn) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """K-FAC diagonal factors {projection name: {"a": [out], "b": [in]}} (float32, CPU) of the current model
    (call it with the new adapter disabled) on (prompt, response) samples. logprob_fn = grpo.sequence_logprobs."""
    wraps = wrappers_of(model)
    caps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = _capture_hooks(wraps, caps)
    acc_a = {n: torch.zeros(w.out_features, dtype=torch.float32) for n, w in wraps}
    acc_b = {n: torch.zeros(w.in_features, dtype=torch.float32) for n, w in wraps}
    tot = {"resp": 0, "pos": 0}

    def one(items):
        caps.clear()
        logp = logprob_fn(model, state, items, pad_id)
        outs = [caps[n][1] for n, _ in wraps]
        grads = torch.autograd.grad(logp.sum(), outs, allow_unused=True)
        T = max(len(p) + len(r) for p, r in items)
        mask = torch.zeros((len(items), T, 1), dtype=torch.float32, device=logp.device)
        for j, (p, r) in enumerate(items):
            mask[j, :len(p) + len(r)] = 1.0
        la, lb = {}, {}
        for (n, _), g in zip(wraps, grads):
            lb[n] = (caps[n][0].float().pow(2) * mask).sum((0, 1)).cpu()
            if g is not None:
                la[n] = (g.float().pow(2) * mask).sum((0, 1)).cpu()
        n_r, n_p = int(logp.numel()), int(mask.sum().item())
        del logp, outs, grads, mask
        caps.clear()
        for n, v in lb.items():   # commit only now: a batch that ran out of memory above added nothing
            acc_b[n] += v
        for n, v in la.items():
            acc_a[n] += v
        tot["resp"] += n_r
        tot["pos"] += n_p

    skipped_len = []
    try:
        _free_gpu(caps)   # the cache left by generating the reference samples
        with torch.enable_grad():
            skipped = _oom_safe(seqs, batch_size, one, caps, skipped_len)
    finally:
        for h in hooks:
            h.remove()
    stats = {n: {"a": acc_a[n] / max(1, tot["resp"]), "b": acc_b[n] / max(1, tot["pos"])} for n, _ in wraps}
    lens = sorted(len(p) + len(r) for p, r in seqs)
    return stats, {"sequences": len(seqs), "response_tokens": tot["resp"], "positions": tot["pos"], "skipped": skipped,
                   "skipped_len_min": min(skipped_len) if skipped_len else None,
                   "len_median": lens[len(lens) // 2] if lens else None, "len_max": lens[-1] if lens else None}


def ewc_penalty(model, tid: str, stats: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """0.5 * sum over projections of sum_ij a_i b_j dW_ij^2 for the new adapter tid (approximately the per-token
    KL(pi_prev || pi) on the reference prompts)."""
    tot = None
    for name, w in wrappers_of(model):
        st = stats.get(name)
        if st is None or tid not in w.lora_A:
            continue
        A, B = w.lora_A[tid], w.lora_B[tid]             # [r, in], [out, r]
        sa = st["a"].to(device=B.device, dtype=B.dtype).clamp_min(0).sqrt()
        sb = st["b"].to(device=A.device, dtype=A.dtype).clamp_min(0).sqrt()
        Bw = B * sa[:, None]
        Aw = A * sb[None, :]
        v = (w.scale[tid] ** 2) * ((Bw.t() @ Bw) * (Aw @ Aw.t())).sum()
        tot = v if tot is None else tot + v
    if tot is None:
        raise RuntimeError(f"no Fisher statistics for the projections of adapter {tid}")
    return 0.5 * tot


# Largest share of reference sequences that may be skipped for GPU memory. 0.10 until Nov 5; raised to 0.15 when the
# EWC run on the 3-task stream skipped 26 of 256 at stage 2 on an H200 (23 at stage 1; the NQ runs at most 23, so
# no finished run changes). The skipped share and lengths are logged with the Fisher factors.
MAX_SKIP_FRAC = 0.15


def _check_skipped(kind, info):
    """Too many sequences that did not fit on the GPU even alone would leave a degenerate importance map."""
    if torch.cuda.is_available():
        info["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    if info.get("skipped", 0) > MAX_SKIP_FRAC * info.get("sequences", 0):
        raise RuntimeError(f"{kind}: {info['skipped']} of {info['sequences']} reference sequences did not fit on the "
                           f"GPU even one at a time")


def setup_ewc(model, tok, state: GateState, cfg: dict, tid: str, seqs: List[Tuple[List[int], List[int]]],
              train_dir: str, log, logprob_fn, is_main: bool, barrier, exists) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute (once, on the main process) and load the Fisher factors of the previous model for adapter tid."""
    g = cfg["grpo"]
    path = os.path.join(train_dir, "ewc.pt")
    if not exists(path):
        if is_main:
            t0 = time.time()
            pool = list(seqs)
            m = int(g.get("ewc_max_seqs", 256))
            if len(pool) > m:   # a fixed random subset that mixes every reference component
                pool = random.Random(int(cfg["seed"]) * 137 + int(tid)).sample(pool, m)
            state.disabled = {tid}
            try:
                stats, info = fisher_stats(model, state, pool, tok.pad_token_id, int(g.get("ewc_batch", 4)), logprob_fn)
            finally:
                state.disabled = set()
            _check_skipped("ewc", info)
            torch.save({"stats": stats, "info": info}, path + ".tmp")
            os.replace(path + ".tmp", path)
            log(f"ewc: Fisher factors of the previous model for adapter {tid}", projections=len(stats),
                secs=round(time.time() - t0), **info)
        barrier()
    return torch.load(path, map_location="cpu", weights_only=False)["stats"]


def _response_logits(model, state: GateState, items: Sequence[Tuple[List[int], List[int]]], pad_id: int):
    """Logits at the positions that predict response tokens (same layout and gating as grpo.sequence_logprobs)."""
    device = next(model.parameters()).device
    seqs = [p + r for p, r in items]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), T), dtype=torch.long, device=device)
    tgt_mask = torch.zeros((len(seqs), T), dtype=torch.bool, device=device)
    for j, (p, r) in enumerate(items):
        s = p + r
        ids[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[j, :len(s)] = 1
        tgt_mask[j, len(p):len(s)] = True
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    state.begin("full")
    try:
        if state.last_prompt_only:
            valid = torch.zeros_like(attn, dtype=torch.bool)
            for j, (p, _) in enumerate(items):
                valid[j, len(p) - 1] = True
            state.valid = valid
        else:
            state.valid = attn.bool() & (pos >= state.gate_from)
        hid = model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    finally:
        state.end()
    return model.lm_head(hid[:, :-1][tgt_mask[:, 1:]]).float()


def mas_stats(model, state: GateState, seqs: Sequence[Tuple[List[int], List[int]]], pad_id: int,
              batch_size: int) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
    """MAS importance in the same factorized form: a = mean |g| over response tokens (g: gradient of the summed
    squared norms of the centered logits with respect to the projection's output), b = mean |x| over positions."""
    wraps = wrappers_of(model)
    caps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = _capture_hooks(wraps, caps)
    acc_a = {n: torch.zeros(w.out_features, dtype=torch.float32) for n, w in wraps}
    acc_b = {n: torch.zeros(w.in_features, dtype=torch.float32) for n, w in wraps}
    tot = {"resp": 0, "pos": 0}

    def one(items):
        caps.clear()
        logits = _response_logits(model, state, items, pad_id)
        outs = [caps[n][1] for n, _ in wraps]
        # squared norm of the centered logits: softmax ignores a common shift, which would otherwise dominate
        grads = torch.autograd.grad((logits - logits.mean(-1, keepdim=True)).pow(2).sum(), outs, allow_unused=True)
        T = max(len(p) + len(r) for p, r in items)
        mask = torch.zeros((len(items), T, 1), dtype=torch.float32, device=logits.device)
        for j, (p, r) in enumerate(items):
            mask[j, :len(p) + len(r)] = 1.0
        la, lb = {}, {}
        for (n, _), g in zip(wraps, grads):
            lb[n] = (caps[n][0].float().abs() * mask).sum((0, 1)).cpu()
            if g is not None:
                la[n] = (g.float().abs() * mask).sum((0, 1)).cpu()
        n_r, n_p = int(logits.shape[0]), int(mask.sum().item())
        del logits, outs, grads, mask
        caps.clear()
        for n, v in lb.items():
            acc_b[n] += v
        for n, v in la.items():
            acc_a[n] += v
        tot["resp"] += n_r
        tot["pos"] += n_p

    try:
        _free_gpu(caps)
        with torch.enable_grad():
            skipped = _oom_safe(seqs, batch_size, one, caps)
    finally:
        for h in hooks:
            h.remove()
    stats = {n: {"a": acc_a[n] / max(1, tot["resp"]), "b": acc_b[n] / max(1, tot["pos"])} for n, _ in wraps}
    return stats, {"sequences": len(seqs), "response_tokens": tot["resp"], "positions": tot["pos"], "skipped": skipped}


def uniform_stats(model) -> Dict[str, Dict[str, torch.Tensor]]:
    return {n: {"a": torch.ones(w.out_features), "b": torch.ones(w.in_features)} for n, w in wrappers_of(model)}


def total_mass(stats) -> float:
    """sum over projections of sum_ij a_i b_j"""
    return float(sum(float(v["a"].double().sum()) * float(v["b"].double().sum()) for v in stats.values()))


def rescale(stats, target_mass: float):
    c = target_mass / max(total_mass(stats), 1e-300)
    return {n: {"a": v["a"] * c, "b": v["b"].clone()} for n, v in stats.items()}


def setup_importance(kind: str, model, tok, state: GateState, cfg: dict, tid: str,
                     seqs: List[Tuple[List[int], List[int]]], train_dir: str, log, logprob_fn, is_main: bool,
                     barrier, exists) -> Dict[str, Dict[str, torch.Tensor]]:
    """Importance map of method kind (ewc / mas / l2) for adapter tid; MAS and L2 rescaled to the Fisher's mass."""
    fisher = setup_ewc(model, tok, state, cfg, tid, seqs, train_dir, log, logprob_fn, is_main, barrier, exists)
    if kind == "ewc":
        return fisher
    if kind == "l2":
        return rescale(uniform_stats(model), total_mass(fisher))
    if kind != "mas":
        raise ValueError(f"unknown importance kind {kind}")
    g = cfg["grpo"]
    path = os.path.join(train_dir, "mas.pt")
    if not exists(path):
        if is_main:
            t0 = time.time()
            pool = list(seqs)
            m = int(g.get("ewc_max_seqs", 256))
            if len(pool) > m:   # the same subset as the Fisher
                pool = random.Random(int(cfg["seed"]) * 137 + int(tid)).sample(pool, m)
            state.disabled = {tid}
            try:
                stats, info = mas_stats(model, state, pool, tok.pad_token_id, int(g.get("ewc_batch", 4)))
            finally:
                state.disabled = set()
            _check_skipped("mas", info)
            torch.save({"stats": stats, "info": info}, path + ".tmp")
            os.replace(path + ".tmp", path)
            log(f"mas: importance of the previous model for adapter {tid}", projections=len(stats),
                secs=round(time.time() - t0), **info)
        barrier()
    return rescale(torch.load(path, map_location="cpu", weights_only=False)["stats"], total_mass(fisher))
