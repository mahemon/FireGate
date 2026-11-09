"""Building the tripwire of stage k (Algorithm 1, lines 1-5), resumable step by step.

  1. greedy trajectories of pi_{k-1} on fitting prompts (new task; reference components, stored earlier-task
     prompts); prompt mode needs only the prompts
  2. logistic-regression score on trunk features (10% of trajectories held out for AUROC); anytime mode samples
     positions from prompt and response, prompt mode uses the last prompt token
  3. calibration scores of pi_{k-1} on calibration prompts with the new tripwire attached in shadow mode
     (scored by the deployment code path, no effect on the computation)
  4. PAC calibration of the threshold per reference component (Theorem 3.6), or a fixed threshold when
     tripwire.calibrate is false (router baseline; the implied firing rates are still recorded)
  5. post-hoc scores on pi_{k-1}'s stored evaluation outputs (ROC / trade-off analysis)
Outputs in <stage_dir>/tripwire/: fit_<group>.jsonl, probe.pt, cal_<component>.json, tripwire.pt, tripwire.json.
Multi-GPU: generation is split across ranks; probe fitting and post-hoc scoring run on rank 0; rank 0 writes.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List

import torch

from . import dist
from .certify import calibrate
from .data import eval_pool, ref_split, task_split
from .evaluate import stored_sequences
from .gating import GateState, Tripwire, tripwire_from_state
from .generate import generate
from .io_utils import ensure_dir, load_json, read_jsonl, save_json, write_jsonl
from .probe import auroc, fit_logreg, last_prompt_features, max_scores, prompt_scores, sample_features
from .tasks import POOLS


def _greedy(model, tok, state, cfg, pool: str, exs: List[dict], encode, stop, record: bool, max_new_tokens=None):
    prompts = [encode(POOLS[pool].prompt(ex)) for ex in exs]
    res = generate(model, state, prompts, pad_id=tok.pad_token_id, stop=stop,
                   max_new_tokens=max_new_tokens or cfg["gen"]["max_new_tokens"],
                   batch_size=cfg["gen"]["eval_batch_size"], greedy=True, record=record)
    return prompts, res


def ref_components(cfg: dict, which: str) -> List[str]:
    """Reference components a tripwire of this config fits on (which = fit_ref) or is calibrated on (cal_ref).
    Paper v3 (PLAN_PAPER3.md): tripwire.protect, when set, keeps only those components, the ones the user protects;
    the others stay in the config, so every data split keeps its rows."""
    comps = list(cfg["tripwire"][which])
    prot = cfg["tripwire"].get("protect") or []
    if isinstance(prot, str):   # --set tripwire.protect=advbench,general
        prot = [c.strip() for c in prot.strip("[]").split(",") if c.strip()]
    prot = list(prot)
    unknown = [c for c in prot if c not in comps]
    if unknown:
        raise ValueError(f"tripwire.protect names {unknown}, which are not in tripwire.{which} ({comps})")
    return [c for c in comps if not prot or c in prot]


def fitting_sets(cfg: dict, task: str, prev: List[str]) -> Dict[str, tuple]:
    """group -> (pool used for the prompt format, examples)."""
    sets = {"new": (task, task_split(cfg, task, "fit_new"))}
    for name in ref_components(cfg, "fit_ref"):
        sets[name] = (name, ref_split(cfg, name, "fit"))
    for t in prev:
        sets[f"old_{t}"] = (t, task_split(cfg, t, "fit_old"))
    return sets


def calibration_sets(cfg: dict, prev: List[str]) -> Dict[str, tuple]:
    sets = {}
    for name in ref_components(cfg, "cal_ref"):
        sets[name] = (name, ref_split(cfg, name, "cal"))
    for t in prev:
        sets[f"old_{t}"] = (t, task_split(cfg, t, "cal_old"))
    return sets


def reference_prompts(cfg: dict, prev: List[str]) -> List[tuple]:
    """(pool, example) for every unlabeled reference prompt a tripwire of this stage may use (fitting and
    calibration components): the data the refkl baseline regularizes on, for a like-for-like comparison."""
    out = []
    for name in ref_components(cfg, "fit_ref"):   # tripwire.protect restricts these too (Ref-KL on what is protected)
        out += [(name, ex) for ex in ref_split(cfg, name, "fit")]
    for name in ref_components(cfg, "cal_ref"):
        out += [(name, ex) for ex in ref_split(cfg, name, "cal")]
    for t in prev:
        out += [(t, ex) for ex in task_split(cfg, t, "fit_old")] + [(t, ex) for ex in task_split(cfg, t, "cal_old")]
    n = int(cfg["grpo"].get("refkl_max_prompts", 0) or 0)
    if n and len(out) > n:  # a fixed random subset, mixing all components
        import random
        idx = sorted(random.Random(int(cfg["seed"]) * 31 + len(prev)).sample(range(len(out)), n))
        out = [out[i] for i in idx]
    return out


def build_tripwire(model, tok, state: GateState, cfg: dict, k: int, task: str, prev: List[str], stage_dir: str,
                   prev_stage_dir: str, encode: Callable, stop: set, log, stop_file) -> Tripwire:
    tid = str(k)
    tdir = os.path.join(stage_dir, "tripwire")
    if dist.is_main():
        ensure_dir(tdir)
    dist.barrier()
    tcfg = cfg["tripwire"]
    prompt_mode = tcfg.get("mode", "anytime") == "prompt"
    gl = cfg["gate_layer"]
    fb = cfg["gen"]["feature_batch_size"]
    device = next(model.parameters()).device
    final_json = os.path.join(tdir, "tripwire.json")
    final_pt = os.path.join(tdir, "tripwire.pt")
    if dist.exists(final_json) and dist.exists(final_pt):
        return tripwire_from_state(torch.load(final_pt, map_location="cpu", weights_only=False), device)

    # 1. fitting trajectories of pi_{k-1} (one file per group, so a stop keeps finished groups); prompt mode
    #    needs only the prompts, so it decodes a single token
    rows = []
    for group, (pool, exs) in fitting_sets(cfg, task, prev).items():
        gpath = os.path.join(tdir, f"fit_{group}.jsonl")
        if not dist.exists(gpath):
            dist.check_stop(stop_file)
            t0 = time.time()
            prompts, res = _greedy(model, tok, state, cfg, pool, exs, encode, stop, record=False,
                                   max_new_tokens=1 if prompt_mode else None)
            if dist.is_main():
                write_jsonl(gpath, [{"group": group, "id": ex["id"], "plen": len(p), "seq": p + r["out"]}
                                    for ex, p, r in zip(exs, prompts, res)])
            dist.barrier()
            log(f"stage{k} tripwire: fit trajectories", group=group, n=len(exs), secs=round(time.time() - t0))
        if dist.is_main():
            rows += read_jsonl(gpath)

    # 2. probe (rank 0; the others wait for probe.pt)
    probe_path = os.path.join(tdir, "probe.pt")
    if not dist.exists(probe_path):
        dist.check_stop(stop_file)
        if dist.is_main():
            _fit_probe(model, tok, state, cfg, k, rows, probe_path, prompt_mode, log)
        dist.barrier()
    del rows
    ck = torch.load(probe_path, map_location="cpu", weights_only=False)
    tw = tripwire_from_state(ck["tw"], device)
    pinfo = ck["info"]

    # 3. calibration scores, computed online in shadow mode (one file per component); in prompt mode the score
    #    is read at the last prompt token, so one decoded token suffices
    scores = {}
    tw.kappa.fill_(float("inf"))
    for comp, (pool, exs) in calibration_sets(cfg, prev).items():
        cpath = os.path.join(tdir, f"cal_{comp}.json")
        if not dist.exists(cpath):
            dist.check_stop(stop_file)
            t0 = time.time()
            state.shadow[tid] = tw
            try:
                _, res = _greedy(model, tok, state, cfg, pool, exs, encode, stop, record=True,
                                 max_new_tokens=1 if prompt_mode else None)
            finally:
                state.shadow.pop(tid, None)
            if dist.is_main():
                save_json(cpath, [r["maxs"][tid] for r in res])
            dist.barrier()
            log(f"stage{k} tripwire: calibration scores", comp=comp, n=len(exs), secs=round(time.time() - t0))
        scores[comp] = load_json(cpath)

    # 4. PAC calibration (or a fixed threshold)
    cal = calibrate(scores, tcfg["eps"], tcfg["delta"])
    if not tcfg.get("calibrate", True):
        kappa = float(tcfg.get("threshold", 0.0))
        cal["calibrated_kappa"] = cal["kappa"]
        cal["kappa"] = kappa
        cal["uncalibrated"] = True
        cal["implied_firing"] = {c: sum(v > kappa for v in vals) / len(vals) for c, vals in scores.items()}
    tw.kappa.fill_(float(cal["kappa"]))
    # every trajectory shares the chat-template prefix, so the lowest trajectory score is close to the prefix
    # score; a threshold within rounding of it would make firing depend on batch padding
    low = min(min(v) for v in scores.values())
    cal["margin_above_lowest_score"] = cal["kappa"] - low
    log(f"stage{k} tripwire: threshold set", kappa=round(cal["kappa"], 4), calibrated=not cal.get("uncalibrated", False),
        margin=round(cal["kappa"] - low, 4), **{f"eps_{c}": round(v["eps_cert"], 4) for c, v in cal["components"].items()})

    # 5. post-hoc scores on pi_{k-1}'s evaluation outputs (rank 0)
    if dist.is_main():
        posthoc = {}
        for pool in [task, "dolly", "advbench", "xstest", "mmlu"] + list(prev):
            if pool not in cfg["pools"]:
                continue
            exs = eval_pool(cfg, pool)
            seqs = stored_sequences(prev_stage_dir, pool, encode, exs)
            if prompt_mode:
                prompts = [encode(POOLS[pool].prompt(ex)) for ex in exs]
                posthoc["new" if pool == task else pool] = prompt_scores(model, state, gl, tw, prompts, tok.pad_token_id, fb)
            else:
                posthoc["new" if pool == task else pool] = max_scores(model, state, gl, tw, seqs, tok.pad_token_id, fb)
        torch.save({k_: v.cpu() for k_, v in tw.state_dict().items()}, final_pt + ".tmp")
        os.replace(final_pt + ".tmp", final_pt)
        save_json(final_json, {"stage": k, "task": task, "mode": tcfg.get("mode", "anytime"), "kappa": cal["kappa"],
                               "eps": tcfg["eps"], "delta": tcfg["delta"], "components": cal["components"],
                               "calibration": {k_: v for k_, v in cal.items() if k_ != "components"},
                               "margin_above_lowest_score": cal["margin_above_lowest_score"],
                               "auroc": pinfo.get("auroc_trajectories"), "probe": pinfo, "posthoc": posthoc})
    dist.barrier()
    return tripwire_from_state(torch.load(final_pt, map_location="cpu", weights_only=False), device)


def _fit_probe(model, tok, state, cfg, k, rows, probe_path, prompt_mode: bool, log) -> None:
    tcfg = cfg["tripwire"]
    gl = cfg["gate_layer"]
    fb = cfg["gen"]["feature_batch_size"]
    t0 = time.time()
    held = lambda i: i % 10 == 0  # noqa: E731  every 10th trajectory of each group is held out
    by_group: Dict[str, List[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    pos_fit = [r for i, r in enumerate(by_group["new"]) if not held(i)]
    pos_ho = [r for i, r in enumerate(by_group["new"]) if held(i)]
    neg_fit, neg_ho = [], []
    for g, rs in by_group.items():
        if g == "new":
            continue
        neg_fit += [r for i, r in enumerate(rs) if not held(i)]
        neg_ho += [r for i, r in enumerate(rs) if held(i)]
    pad = tok.pad_token_id
    seed = int(cfg["seed"]) * 101 + k
    if prompt_mode:
        feats = lambda rs: last_prompt_features(model, state, gl, [r["seq"][:r["plen"]] for r in rs], pad, fb)  # noqa: E731
        Xp, Xn = feats(pos_fit), feats(neg_fit)
    else:
        P = tcfg["positions_per_traj"]
        Xp = sample_features(model, state, gl, [r["seq"] for r in pos_fit], pad, fb, P, seed)
        Xn = sample_features(model, state, gl, [r["seq"] for r in neg_fit], pad, fb, P, seed + 1)
    tw, info = fit_logreg(Xp, Xn, tcfg["l2"], tcfg["lbfgs_iters"])
    with torch.no_grad():
        if prompt_mode:
            hp, hn = feats(pos_ho), feats(neg_ho)
        else:
            hp = sample_features(model, state, gl, [r["seq"] for r in pos_ho], pad, fb, P, seed + 2)
            hn = sample_features(model, state, gl, [r["seq"] for r in neg_ho], pad, fb, P, seed + 3)
        # features are already RMS-normalized: standardize and score directly
        sp = ((hp - tw.mu) / tw.sigma) @ tw.a + tw.b
        sn = ((hn - tw.mu) / tw.sigma) @ tw.a + tw.b
        info["auroc_positions"] = auroc(sp, sn)
        if prompt_mode:
            info["auroc_trajectories"] = info["auroc_positions"]
        else:
            tp = max_scores(model, state, gl, tw, [r["seq"] for r in pos_ho], pad, fb)
            tn = max_scores(model, state, gl, tw, [r["seq"] for r in neg_ho], pad, fb)
            info["auroc_trajectories"] = auroc(torch.tensor(tp), torch.tensor(tn))
    info.update(n_pos=len(Xp), n_neg=len(Xn), secs=round(time.time() - t0), mode="prompt" if prompt_mode else "anytime")
    torch.save({"tw": {k_: v.cpu() for k_, v in tw.state_dict().items()}, "info": info}, probe_path + ".tmp")
    os.replace(probe_path + ".tmp", probe_path)
    log(f"stage{k} tripwire: probe", **{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in info.items()})
