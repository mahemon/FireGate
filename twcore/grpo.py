"""GRPO for one stage of the stream (one gradient step per rollout batch, so the importance ratio is 1).

Methods (all train only the new adapter `tid`; earlier adapters and the base model are frozen):
  tripwire, tripwire_prompt, router
            gated adapter (anytime tripwire / calibrated prompt-time tripwire / uncalibrated prompt-time gate);
            rollouts on which the gate never fires carry no gradient and are skipped
  seq       ungated adapter
  kl        ungated adapter + beta * k3 estimate of KL(pi || pi_prev) on the current task's samples
  replay    ungated adapter; a fraction of each batch comes from earlier tasks (with their rewards)
  safemix   (paper v3) ungated adapter; a fraction grpo.safemix_frac (default replay_frac) of each batch are harmful
            reference requests rewarded for a refusal (pool advsafe): behaviour-preserving data with a behaviour
            label, in the spirit of behaviour-aware safety sampling (Pham et al. 2025) for RL
  refkl     ungated adapter + refkl_beta * KL(pi_prev || pi) on pi_prev's samples for the unlabeled reference
            prompts that the tripwire uses (LwF-style regularization on the same data); refkl_samples samples of
            pi_prev per reference prompt (default 1)
  olora     ungated adapter + olora_lambda * sum ||A_new A_prev^T||_F^2 (O-LoRA orthogonal subspaces)
  nullspace ungated adapter that ignores, at every adapted projection, the input directions used by the
            unlabeled reference prompts (twcore/nullspace.py; InfLoRA-style, same data as the tripwire);
            nullspace_refresh > 0: the subspaces are recomputed every that many steps on the current model (the
            adapter of earlier blocks changes the inputs of later projections), saved with the checkpoint
  ewc, mas, l2
            ungated adapter + <method>_lambda * 0.5 * sum Omega_ij dW_ij^2, with the importance map Omega of pi_prev on
            the same unlabeled reference prompts: K-FAC diagonal Fisher (ewc), mean |grad| of the squared logit norm
            (MAS), or uniform (L2 to the old weights); MAS and L2 rescaled to the Fisher's total mass (twcore/ewc.py)
  agem      ungated adapter; A-GEM gradient projection: when the RL gradient would increase the KL(pi_prev || pi) of
            the same reference prompts (sampled as in refkl), it is projected onto the half-space that does not
            increase it to first order (Chaudhry et al. 2019; approximate under AdamW's preconditioning)

Loss = -(1/N) sum_i sum_t A_i log pi(y_it | ...) [+ beta/N sum k3], N = all response tokens of the step.
Objective grpo.objective = "sft" (v7; methods seq, tripwire, tripwire_prompt, router): supervised fine-tuning instead
of GRPO, on rejection-sampled data as in RFT / STaR: grpo.sft_samples samples per training prompt of the model before
the stage (new adapter off), the first one the task's reward accepts is kept; behind a gate only prompts on which the
gate fires are kept (the others would carry no gradient). Each step: grpo.sft_batch sequences, loss = -(1/N) sum of
the response tokens' log-probabilities. grpo.sft_source (PLAN_SFT.md) picks the data: self (the above, default),
gold (the dataset's reference response, tasks.GOLD) or teacher (samples of another model, grpo.sft_teacher, accepted by
the reward); gold and teacher data work with the prompt-time gates only (the gate is read on the prompt).
Full-rank updates (lora.dense) work with every method except olora, ewc, mas
and l2 (their penalties are written for low-rank factors).
Resumable: a checkpoint (adapter, optimizer, step) is written every ckpt_every steps and on a stop request.
Multi-GPU: rollouts are generated in parallel and gathered; each rank back-propagates its share of the samples,
gradients are summed across ranks, and every rank takes the same optimizer step.
"""
from __future__ import annotations

import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import dist
from .gating import (GateState, adapter_params, adapter_state_dict, is_dense, load_adapter_state_dict,
                     set_projections, wrappers_of)
from .generate import generate
from .io_utils import Logger, StopRequested, ensure_dir, read_jsonl, write_jsonl
from .tasks import GOLD, POOLS, batch_rewards

GATED = ("tripwire", "tripwire_prompt", "router")


def _rollout_batch(n: int, cap: int) -> int:
    """Batch size for n sampled sequences: the configured size, split so that every GPU gets work. Sampling uses
    per-prompt uniforms (generate.py), so the samples do not depend on the batch size."""
    W = dist.world()
    return cap if W == 1 else max(1, min(cap, -(-n // W)))


def sequence_logprobs(model, state: GateState, items: Sequence[Tuple[List[int], List[int]]], pad_id: int):
    """Log-probabilities of the response tokens (concatenated in item order) under the current policy.
    items = [(prompt_ids, response_ids)]; sequences are right-padded."""
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
        if state.last_prompt_only:  # prompt-time gates: the decision is read at the last prompt token only
            valid = torch.zeros_like(attn, dtype=torch.bool)
            for j, (p, _) in enumerate(items):
                valid[j, len(p) - 1] = True
            state.valid = valid
        else:
            state.valid = attn.bool() & (pos >= state.gate_from)
        hid = model.model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state
    finally:
        state.end()
    m = tgt_mask[:, 1:]
    h = hid[:, :-1][m]
    targets = ids[:, 1:][m]
    logits = model.lm_head(h).float()
    return -F.cross_entropy(logits, targets, reduction="none")


def _select_prompts(step: int, P: int, n_rep: int, main: List[dict], replay: Dict[str, List[dict]]):
    n_main = P - n_rep
    items = [main[(step * n_main + i) % len(main)] for i in range(n_main)]
    out = [(None, ex) for ex in items]
    names = sorted(replay)
    for i in range(n_rep):
        c = step * n_rep + i
        t = names[c % len(names)]
        pool = replay[t]
        out.append((t, pool[(c // len(names)) % len(pool)]))
    return out


def olora_penalty(model, tid: str, prev: List[str]) -> torch.Tensor:
    """sum over wrapped projections and earlier adapters of ||A_new A_prev^T||_F^2 (O-LoRA)."""
    tot = None
    for _, w in wrappers_of(model):
        A = w.lora_A[tid]
        for t in prev:
            if t in w.lora_A:
                v = (A @ w.lora_A[t].t()).pow(2).sum()
                tot = v if tot is None else tot + v
    return tot


def _reference_samples(model, tok, state, cfg, tid, refs, encode, stop, path, log,
                       n: int = 1) -> List[Tuple[List[int], List[int]]]:
    """n samples of pi_prev (the new adapter switched off) per reference prompt, generated once and stored; the
    prompts are repeated n times in order (sample-major), so a batch of consecutive samples mixes prompts and n = 1
    gives exactly the earlier samples."""
    if not dist.exists(path):
        g = cfg["grpo"]
        prompts = [encode(POOLS[pool].prompt(ex)) for pool, ex in refs] * max(1, int(n))
        state.disabled = {tid}
        try:
            res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop,
                           max_new_tokens=cfg["gen"]["max_new_tokens"],
                           batch_size=_rollout_batch(len(prompts), cfg["gen"]["rollout_batch_size"]),
                           greedy=False, temperature=g["refkl_temperature"], seed=int(cfg["seed"]) * 7919 + int(tid))
        finally:
            state.disabled = set()
        if dist.is_main():
            write_jsonl(path, [{"p": p, "o": r["out"]} for p, r in zip(prompts, res) if r["out"]])
        dist.barrier()
        log(f"refkl: {len(prompts)} reference samples of the previous model stored")
    return [(r["p"], r["o"]) for r in read_jsonl(path)]


def _sft_samples(model, tok, state, cfg, tid, task, main, encode, stop, path, log, gated) -> List[Tuple[List[int], List[int]]]:
    """SFT data of the stage, generated once and stored: grpo.sft_samples samples per training prompt from the model
    before the stage (adapter tid off), the first accepted by the task's reward kept; behind a gate, fired prompts only.
    grpo.sft_source = gold / teacher: the data come from the dataset or another model (_sft_external)"""
    src = str(cfg["grpo"].get("sft_source", "self"))
    if src != "self":
        if not dist.exists(path):
            _sft_external(model, tok, state, cfg, tid, task, main, encode, stop, path, log, gated, src)
        return [(r["p"], r["o"]) for r in read_jsonl(path)]
    if not dist.exists(path):
        g = cfg["grpo"]
        n = max(1, int(g.get("sft_samples", 4)))
        exs = list(main[:int(g.get("sft_max_prompts", len(main)))])
        prompts = [encode(POOLS[task].prompt(ex)) for ex in exs] * n          # sample-major, as in _reference_samples
        state.disabled = {tid}
        try:
            res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop,
                           max_new_tokens=cfg["gen"]["max_new_tokens"],
                           batch_size=_rollout_batch(len(prompts), cfg["gen"]["rollout_batch_size"]),
                           greedy=False, temperature=float(g.get("sft_temperature", 1.0)),
                           seed=int(cfg["seed"]) * 7907 + int(tid), record=gated)
        finally:
            state.disabled = set()
        if dist.is_main():
            keep, done, fired, accepted = [], set(), set(), 0
            for i, r in enumerate(res):
                j = i % len(exs)
                if gated and r["fire"].get(tid) is None:
                    continue
                fired.add(j)
                if j in done or not r["out"]:
                    continue
                if POOLS[task].reward(exs[j], tok.decode(r["out"], skip_special_tokens=True)) > 0:
                    keep.append({"p": prompts[i], "o": r["out"], "j": j})
                    done.add(j)
                    accepted += 1
            write_jsonl(path, keep)
            log(f"sft: {accepted} of {len(exs)} training prompts with an accepted sample ({n} samples each)"
                + (f"; the gate fired on {len(fired)}" if gated else ""))
        dist.barrier()
    return [(r["p"], r["o"]) for r in read_jsonl(path)]


def _turn_end(tok) -> List[int]:
    """Token ids that close an assistant turn in the chat template (<|im_end|> for Qwen, <|eot_id|> for Llama 3,
    <end_of_turn> for Gemma, </s> for Mistral), appended to SFT responses that are not the model's own samples."""
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Zq7"}],
                                    tokenize=False)
        tail = s.split("Zq7", 1)[1].rstrip()
        ids = tok(tail, add_special_tokens=False)["input_ids"] if tail else []
    except Exception:  # no chat template
        ids = []
    return list(ids) or ([int(tok.eos_token_id)] if tok.eos_token_id is not None else [])


def _teacher_texts(cfg, task, exs, n, seed) -> List[List[str]]:
    """n sampled responses per example from the model grpo.sft_teacher, with its own tokenizer and chat template; only
    complete responses (ending with one of its stop tokens) are returned, as text"""
    from .model import load_model, make_encoder, stop_ids
    g = cfg["grpo"]
    path = str(g.get("sft_teacher") or "")
    if not path:
        raise ValueError("grpo.sft_source=teacher needs grpo.sft_teacher (a model folder)")
    t_model, t_tok = load_model(path, cfg["dtype"], device=dist.device())
    try:
        t_enc = make_encoder(t_tok, cfg["gen"]["max_prompt_tokens"])
        t_stop = stop_ids(t_model, t_tok)
        prompts = [t_enc(POOLS[task].prompt(ex)) for ex in exs] * n      # sample-major
        res = generate(t_model, GateState(), prompts, pad_id=t_tok.pad_token_id, stop=t_stop,
                       max_new_tokens=cfg["gen"]["max_new_tokens"],
                       batch_size=_rollout_batch(len(prompts), cfg["gen"]["rollout_batch_size"]),
                       greedy=False, temperature=float(g.get("sft_temperature", 1.0)), seed=seed)
        texts = [t_tok.decode(r["out"], skip_special_tokens=True) if r["out"] and r["out"][-1] in t_stop else ""
                 for r in res]
    finally:
        del t_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [[texts[i * len(exs) + j] for i in range(n)] for j in range(len(exs))]


def _sft_external(model, tok, state, cfg, tid, task, main, encode, stop, path, log, gated, src) -> None:
    """SFT data from the dataset (src = gold: tasks.GOLD[task]) or from another model (src = teacher: grpo.sft_samples
    samples per prompt of grpo.sft_teacher), the first response the task's reward accepts kept, encoded with this
    model's tokenizer and closed with its end-of-turn token. Behind a (prompt-time) gate only the prompts on which it
    fires are kept; the gate is read on the prompt with the new adapter off, as during training."""
    g = cfg["grpo"]
    if src not in ("gold", "teacher"):
        raise ValueError(f"grpo.sft_source must be self, gold or teacher, not {src}")
    if src == "gold" and task not in GOLD:
        raise ValueError(f"grpo.sft_source=gold: no reference responses for task {task} (tasks.GOLD)")
    exs = list(main[:int(g.get("sft_max_prompts", len(main)))])
    prompts = [encode(POOLS[task].prompt(ex)) for ex in exs]
    fired = list(range(len(exs)))
    if gated:
        state.disabled = {tid}
        try:
            res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop, max_new_tokens=1,
                           batch_size=_rollout_batch(len(prompts), cfg["gen"]["rollout_batch_size"]),
                           greedy=True, record=True)
        finally:
            state.disabled = set()
        fired = [j for j, r in enumerate(res) if r["fire"].get(tid) is not None]
    if src == "gold":
        cands = [[GOLD[task](exs[j])] for j in fired]
    else:   # every prompt is sampled, then the fired ones are taken: gated runs see the same teacher responses
        n = max(1, int(g.get("sft_samples", 4)))
        every = _teacher_texts(cfg, task, exs, n, int(cfg["seed"]) * 7907 + int(tid) + 17) if fired else []
        cands = [every[j] for j in fired]
    if dist.is_main():
        end = _turn_end(tok)
        cap = int(cfg["gen"]["max_new_tokens"])
        keep = []
        for j, texts in zip(fired, cands):
            for text in texts:
                if not text or POOLS[task].reward(exs[j], text) <= 0:
                    continue
                o = tok(text, add_special_tokens=False)["input_ids"] + end
                if len(o) <= cap:
                    keep.append({"p": prompts[j], "o": o, "j": j})
                    break
        write_jsonl(path, keep)
        log(f"sft ({src}): {len(keep)} of {len(exs)} training prompts with an accepted response"
            + (f"; the gate fired on {len(fired)}" if gated else ""))
    dist.barrier()


def _setup_nullspace(model, tok, state, cfg, tid, refs, encode, stop, train_dir, log) -> None:
    """Compute (once, rank 0) and install the reference subspaces of every adapted projection for adapter tid."""
    from .nullspace import reference_subspaces
    g = cfg["grpo"]
    path = os.path.join(train_dir, "nullspace.pt")
    if not dist.exists(path):
        seqs = _reference_samples(model, tok, state, cfg, tid, refs, encode, stop,
                                  os.path.join(train_dir, "ref_samples.jsonl"), log)
        if dist.is_main():
            t0 = time.time()
            pool = [p + o for p, o in seqs]
            m = int(g["nullspace_max_seqs"])
            if len(pool) > m:  # a fixed random subset mixing every component (refs are ordered by component)
                pool = random.Random(int(cfg["seed"]) * 131 + int(tid)).sample(pool, m)
            state.disabled = {tid}
            try:
                subs, stats = reference_subspaces(
                    model, state, pool, tok.pad_token_id,
                    cfg["gen"]["feature_batch_size"], float(g["nullspace_energy"]), int(g["nullspace_max_rank"]),
                    seed=int(cfg["seed"]) * 131 + int(tid))
            finally:
                state.disabled = set()
            torch.save({"U": {k: v.cpu() for k, v in subs.items()}, "stats": stats}, path + ".tmp")
            os.replace(path + ".tmp", path)
            ks = [v["k"] for v in stats.values()]
            log(f"nullspace: reference subspaces for adapter {tid}", projections=len(ks),
                mean_rank=round(sum(ks) / len(ks), 1), max_rank=max(ks), secs=round(time.time() - t0))
        dist.barrier()
    set_projections(model, tid, torch.load(path, map_location="cpu", weights_only=False)["U"])


def _refresh_nullspace(model, tok, state, cfg, tid, train_dir, step, log) -> None:
    """Null-space baseline with refreshed subspaces: the reference subspaces recomputed on the current model (adapter
    tid on, with its current projections), on the same reference sequences as at the start, and installed."""
    from .nullspace import reference_subspaces
    g = cfg["grpo"]
    path = os.path.join(train_dir, f"nullspace_s{step}.pt")
    if not dist.exists(path):
        if dist.is_main():
            t0 = time.time()
            pool = [r["p"] + r["o"] for r in read_jsonl(os.path.join(train_dir, "ref_samples.jsonl"))]
            m = int(g["nullspace_max_seqs"])
            if len(pool) > m:  # the same subset as at the start
                pool = random.Random(int(cfg["seed"]) * 131 + int(tid)).sample(pool, m)
            subs, stats = reference_subspaces(
                model, state, pool, tok.pad_token_id, cfg["gen"]["feature_batch_size"], float(g["nullspace_energy"]),
                int(g["nullspace_max_rank"]), seed=int(cfg["seed"]) * 131 + int(tid) + int(step))
            torch.save({"U": {k: v.cpu() for k, v in subs.items()}, "stats": stats, "step": int(step)}, path + ".tmp")
            os.replace(path + ".tmp", path)
            ks = [v["k"] for v in stats.values()]
            log(f"nullspace: subspaces of adapter {tid} recomputed after step {step}", projections=len(ks),
                mean_rank=round(sum(ks) / len(ks), 1), max_rank=max(ks), secs=round(time.time() - t0))
        dist.barrier()
    for _, w in wrappers_of(model):
        w.proj.pop(tid, None)
    set_projections(model, tid, torch.load(path, map_location="cpu", weights_only=False)["U"])


def train_stage(model, tok, state: GateState, cfg: dict, method: str, tid: str, task: str, main: List[dict],
                replay: Optional[Dict[str, List[dict]]], encode: Callable[[str], List[int]], stop: set,
                train_dir: str, log: Logger, stop_file: Optional[str], refs: Optional[List[tuple]] = None,
                prev_tids: Optional[List[str]] = None) -> None:
    g = cfg["grpo"]
    gen_cfg = cfg["gen"]
    if dist.is_main():
        ensure_dir(train_dir)
    dist.barrier()
    W, me = dist.world(), dist.rank()
    gated = method in GATED
    ckpt_path = os.path.join(train_dir, "ckpt.pt")
    steplog = Logger(os.path.join(train_dir, "log.jsonl"), prefix=f"{method}/s{tid} ")
    params = adapter_params(model, tid)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=g["lr"], betas=(0.9, 0.999), weight_decay=g["weight_decay"])
    start = 0
    if dist.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_adapter_state_dict(model, tid, ck["adapter"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        log(f"resumed training of adapter {tid} at step {start}")

    def save(step_done: int):
        if dist.is_main():
            tmp = ckpt_path + ".tmp"
            torch.save({"step": step_done, "adapter": adapter_state_dict(model, tid), "opt": opt.state_dict()}, tmp)
            os.replace(tmp, ckpt_path)
        dist.barrier()

    objective = str(g.get("objective", "grpo"))
    if objective not in ("grpo", "sft"):
        raise ValueError(f"grpo.objective must be grpo or sft, not {objective}")
    if objective == "sft" and method not in ("seq", "tripwire", "tripwire_prompt", "router"):
        raise ValueError(f"grpo.objective=sft is implemented for seq / tripwire / tripwire_prompt / router, not {method}")
    if objective == "sft" and method == "tripwire" and str(g.get("sft_source", "self")) != "self":
        raise ValueError("grpo.sft_source=gold / teacher reads the gate on the prompt: use tripwire_prompt, not tripwire")
    if is_dense(model, tid) and method in ("olora", "ewc", "mas", "l2"):
        raise ValueError(f"full-rank updates (lora.dense) are not implemented for {method}")
    sft_seqs = (_sft_samples(model, tok, state, cfg, tid, task, main, encode, stop,
                             os.path.join(train_dir, "sft_samples.jsonl"), log, gated) if objective == "sft" else [])
    ref_seqs = []
    if method in ("refkl", "agem") and refs:
        ref_seqs = _reference_samples(model, tok, state, cfg, tid, refs, encode, stop,
                                      os.path.join(train_dir, "ref_samples.jsonl"), log,
                                      n=int(g.get("refkl_samples", 1)) if method == "refkl" else 1)
    refresh = int(g.get("nullspace_refresh", 0)) if method == "nullspace" else 0
    if method == "nullspace":
        _setup_nullspace(model, tok, state, cfg, tid, refs or [], encode, stop, train_dir, log)
        if start > 0 and refresh > 0 and any(k.endswith(".U") for k in ck["adapter"]):
            load_adapter_state_dict(model, tid, ck["adapter"])   # the projections in use when the checkpoint was saved
    ewc_stats = None
    if method in ("ewc", "mas", "l2"):
        from .ewc import setup_importance
        ewc_seqs = _reference_samples(model, tok, state, cfg, tid, refs or [], encode, stop,
                                      os.path.join(train_dir, "ref_samples.jsonl"), log)
        ewc_stats = setup_importance(method, model, tok, state, cfg, tid, ewc_seqs, train_dir, log, sequence_logprobs,
                                     dist.is_main(), dist.barrier, dist.exists)
    prev_tids = [t for t in (prev_tids or []) if t != tid]

    P, G = g["prompts_per_step"], g["group_size"]
    n_rep = int(round(g["replay_frac"] * P)) if (method == "replay" and replay) else 0
    if method == "safemix" and replay:
        n_rep = int(round(float(g.get("safemix_frac", g["replay_frac"])) * P))
    for step in range(start, g["steps"]):
        try:
            dist.check_stop(stop_file)
        except StopRequested:
            save(step)
            raise
        t0 = time.time()
        lr = g["lr"] * min(1.0, (step + 1) / max(1, g["warmup_steps"]))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if objective == "sft":   # supervised step on the stored accepted samples (no rollouts)
            nb = int(g.get("sft_batch", 64))
            batch = [sft_seqs[(step * nb + i) % len(sft_seqs)] for i in range(nb)] if sft_seqs else []
            n_tok = max(1, sum(len(o) for _, o in batch))
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            mine = batch[me::W]
            with torch.enable_grad():
                for b0 in range(0, len(mine), g["micro_batch"]):
                    logp = sequence_logprobs(model, state, mine[b0:b0 + g["micro_batch"]], tok.pad_token_id)
                    loss = -logp.sum() / n_tok
                    loss.backward()
                    loss_sum += float(loss.detach())
            dist.all_reduce_grads(params)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, g["grad_clip"])) if batch else 0.0
            if batch:
                opt.step()
            loss_sum = dist.sum_scalars([loss_sum])[0]
            steplog("step", step=step + 1, lr=lr, sft_loss=loss_sum, grad_norm=gnorm, samples=len(batch),
                    sft_data=len(sft_seqs), t_step=round(time.time() - t0, 1))
            if (step + 1) % g["ckpt_every"] == 0 or step + 1 == g["steps"]:
                save(step + 1)
            continue
        chosen = _select_prompts(step, P, n_rep, main, replay or {})
        prompts, owners = [], []
        for owner, ex in chosen:
            pool = POOLS[owner or task]
            ids = encode(pool.prompt(ex))
            for _ in range(G):
                prompts.append(ids)
                owners.append((owner or task, ex))
        seed = (int(cfg["seed"]) * 1_000_003 + int(tid) * 10_007 + step) % (2 ** 62)
        model.eval()
        res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop,
                       max_new_tokens=gen_cfg["max_new_tokens"],
                       batch_size=_rollout_batch(len(prompts), gen_cfg["rollout_batch_size"]),
                       greedy=False, temperature=g["temperature"], seed=seed, record=gated)
        t_gen = time.time() - t0
        rewards = batch_rewards([o for o, _ in owners], [ex for _, ex in owners],
                                [tok.decode(r["out"], skip_special_tokens=True) for r in res])
        rw = torch.tensor(rewards).view(-1, G)
        adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-4)
        adv = adv.view(-1).tolist()
        n_tok = max(1, sum(len(r["out"]) for r in res))

        samples = []
        for k, r in enumerate(res):
            if len(r["out"]) == 0:
                continue
            if gated and r["fire"].get(tid) is None:
                continue  # the adapter was never active: zero gradient
            if method != "kl" and adv[k] == 0.0:
                continue
            samples.append((prompts[k], r["out"], adv[k]))
        ref_batch = []
        if ref_seqs:
            nb = g["refkl_batch"]
            ref_batch = [ref_seqs[(step * nb + i) % len(ref_seqs)] for i in range(nb)]
        n_ref_tok = max(1, sum(len(o) for _, o in ref_batch))
        use_olora = method == "olora" and bool(prev_tids)
        use_ewc = ewc_stats is not None
        use_agem = method == "agem" and bool(ref_batch)
        agem_proj = 0.0

        opt.zero_grad(set_to_none=True)
        loss_sum, kl_sum, refkl_sum, olora_val, ewc_val = 0.0, 0.0, 0.0, 0.0, 0.0
        mb = g["micro_batch"]
        mine = samples[me::W]  # this rank's share; the loss is normalized by the global token count
        with torch.enable_grad():
            for b0 in range(0, len(mine), mb):
                chunk = mine[b0:b0 + mb]
                items = [(p, o) for p, o, _ in chunk]
                logp = sequence_logprobs(model, state, items, tok.pad_token_id)
                a_tok = torch.cat([torch.full((len(o),), a, device=logp.device) for _, o, a in chunk])
                loss = -(a_tok * logp).sum() / n_tok
                if method == "kl":
                    with torch.no_grad():
                        state.disabled = {tid}
                        try:
                            ref = sequence_logprobs(model, state, items, tok.pad_token_id)
                        finally:
                            state.disabled = set()
                    d = ref - logp
                    k3 = torch.exp(d) - d - 1.0
                    loss = loss + g["kl_beta"] * k3.sum() / n_tok
                    kl_sum += float(k3.detach().sum())
                loss.backward()
                loss_sum += float(loss.detach())
            my_ref = ref_batch[me::W]
            g_rl = None
            if use_agem:  # keep the RL gradient apart; the reference gradient is computed next
                dist.all_reduce_grads(params)
                g_rl = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                for p in params:
                    p.grad = None
            for b0 in range(0, len(my_ref), mb):
                items = my_ref[b0:b0 + mb]
                logp = sequence_logprobs(model, state, items, tok.pad_token_id)
                with torch.no_grad():
                    state.disabled = {tid}
                    try:
                        old = sequence_logprobs(model, state, items, tok.pad_token_id)
                    finally:
                        state.disabled = set()
                # sample estimate of KL(pi_prev || pi) on pi_prev's own samples; its gradient is -grad log pi
                kl_ref = (old - logp).sum() / n_ref_tok
                ((1.0 if use_agem else g["refkl_beta"]) * kl_ref).backward()
                refkl_sum += float(kl_ref.detach())
            if use_olora:
                pen = olora_penalty(model, tid, prev_tids)
                (g["olora_lambda"] * pen / W).backward()  # summed over ranks below
                olora_val = float(pen.detach())
            if use_ewc:
                from .ewc import ewc_penalty
                pen = ewc_penalty(model, tid, ewc_stats)
                (g[f"{method}_lambda"] * pen / W).backward()  # summed over ranks below
                ewc_val = float(pen.detach())
        dist.all_reduce_grads(params)
        if use_agem:
            # A-GEM projection of the gradient handed to AdamW: <g, g_ref> >= 0 after projection (first-order
            # non-increase of the reference KL for an SGD step; Adam's preconditioning makes it approximate)
            g_ref = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            dot = sum(float((a * b).sum()) for a, b in zip(g_rl, g_ref))
            nref = sum(float((b * b).sum()) for b in g_ref)
            proj = dot < 0 and nref > 0
            for p, a, b in zip(params, g_rl, g_ref):
                p.grad = a - (dot / nref) * b if proj else a
            agem_proj = float(proj)
        has_update = (bool(samples) if use_agem else
                      (bool(samples) or bool(ref_batch) or use_olora or use_ewc))
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, g["grad_clip"])) if has_update else 0.0
        if has_update:
            opt.step()
        loss_sum, kl_sum, refkl_sum = dist.sum_scalars([loss_sum, kl_sum, refkl_sum])

        own = [rewards[k] for k in range(len(rewards)) if owners[k][0] == task]
        rep = [rewards[k] for k in range(len(rewards)) if owners[k][0] != task]
        info = {"step": step + 1, "lr": lr, "reward": sum(own) / max(1, len(own)), "loss": loss_sum,
                "grad_norm": gnorm, "samples": len(samples), "resp_len": n_tok / len(res),
                "t_gen": round(t_gen, 1), "t_step": round(time.time() - t0, 1)}
        if rep:
            info["reward_replay"] = sum(rep) / len(rep)
        if method == "kl":
            info["kl"] = kl_sum / n_tok
        if ref_batch:
            info["refkl"] = refkl_sum
        if use_olora:
            info["olora"] = olora_val
        if use_ewc:
            info[method] = ewc_val
        if use_agem:
            info["agem_projected"] = agem_proj
        if gated:
            info["active"] = sum(r["fire"].get(tid) is not None for r in res) / len(res)
        steplog("step", **info)
        if refresh > 0 and (step + 1) % refresh == 0 and step + 1 < g["steps"]:
            _refresh_nullspace(model, tok, state, cfg, tid, train_dir, step + 1, log)
        if (step + 1) % g["ckpt_every"] == 0 or step + 1 == g["steps"]:
            save(step + 1)
    for p in params:
        p.requires_grad_(False)
