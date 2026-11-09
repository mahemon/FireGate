#!/usr/bin/env python
"""Trunk features of every prompt a gate is built from or evaluated on (no training; one GPU job per experiment).

    python scripts/gate_features.py --config configs/v2/overlap_q15.yaml [--rl 1000]
    python scripts/gate_features.py --config configs/v2/overlap_q15.yaml --add    # only prompt sets not stored yet

A prompt-time tripwire reads one vector per prompt: the RMS-normalized input of block gate_layer at the last prompt
token, computed by the frozen trunk. It depends on the prompt only (not on any adapter, method or seed), so every
gate variant of scripts/switch_report.py can be built and evaluated on the CPU from these vectors.

Writes <out_dir>/<exp>/analysis/gate_features.npz (float16 arrays [n, d]) and gate_features.json (for every key:
pool, n, example ids; gate layer, model, GPU). Keys:
  fit__new__<task>      the tripwire's fitting prompts of the task (new-task side)
  rl__<task>            the first --rl RL prompts of the task (unlabeled new-task prompts no probe was fitted on)
  fit__ref__<comp>, cal__ref__<comp>      reference fitting / calibration prompts
  fit__old__<task>, cal__old__<task>      stored prompts of a task once it is an earlier task (all but the last)
  eval__<pool>          held-out evaluation pools, also advwrap_<task>, dolly_sys, dolly_turn, advbench_sys and (v5)
                        advbench_turn, jb_prefix, jb_roleplay, jb_hypo (AdvBench requests after an earlier exchange /
                        in jailbreak templates) and (Nov 1) wildchat: the first WILDCHAT_N real-user requests of
                        data/wildchat_eval.jsonl when the file exists (the config's pool when it lists one); paper v3:
                        the stress families of the config's tasks (st_*, twcore/stress.py: harmful requests written
                        to look like the task in ways the remedy never saw); paper v3 (p3.7): dolly_fresh, rows
                        2000-11999 of the shuffled Dolly file, in experiments without a ladder task (twcore/data.py);
                        Nov 2 (p3.10b): the second-round forms st2_* of twcore/stress.py, in every experiment;
                        Nov 4 (p3.19): the prospective sets pr_* of twcore/prospective.py; Nov 5 (p3.23): the sets
                        wd_aya, wd_dna of twcore/widen.py when data/wd_*.jsonl exist
  wrap__<task>__fit, wrap__<task>__cal    AdvBench fitting / calibration requests written in <task>'s answer format
                        (a candidate extra reference component, disjoint from the evaluated advwrap requests)
  probe__<k>__{mu,sigma,a,b,kappa}        the deployed prompt-time tripwire of stage k, copied from
                        <out_dir>/<exp>/s0/tripwire_prompt/stage<k>/tripwire/tripwire.pt when that run exists
--add keeps every stored array and computes only the prompt sets that are missing (new pools); run it on the GPU
model the stored features were computed on (their "device"): the job refuses another GPU model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from twcore import dist  # noqa: E402
from twcore.config import load_config  # noqa: E402
from twcore.data import dolly_fresh_rows, eval_pool, ref_split, task_split, wildchat_path, wildchat_rows  # noqa: E402
from twcore.evaluate import device_name  # noqa: E402
from twcore.gating import GateState  # noqa: E402
from twcore.model import load_model, make_encoder, template_prefix_len  # noqa: E402
from twcore.probe import last_prompt_features  # noqa: E402
from twcore.stress import set_data_dir, stress2_pools, stress_pools  # noqa: E402
from twcore.external import external_pools, external_rows  # noqa: E402
from twcore.prospective import prospective_pools, prospective_rows, sysprompts_present  # noqa: E402
from twcore.prospective import set_data_dir as set_prospective_dir  # noqa: E402
from twcore.widen import widen_pools, widen_rows  # noqa: E402
from twcore.tasks import POOLS  # noqa: E402

EXTRA_POOLS = ("dolly_sys", "dolly_turn", "advbench_sys", "advbench_turn", "jb_prefix", "jb_roleplay", "jb_hypo")


def prompt_sets(cfg: dict, n_rl: int) -> dict:
    """key -> (pool whose prompt format is used, examples), in a fixed order."""
    tasks = list(cfg["stream"])
    sets = {}
    for t in tasks:
        sets[f"fit__new__{t}"] = (t, task_split(cfg, t, "fit_new"))
        sets[f"rl__{t}"] = (t, task_split(cfg, t, "rl")[:n_rl])
    for t in tasks[:-1]:   # the last task never becomes an earlier task
        sets[f"fit__old__{t}"] = (t, task_split(cfg, t, "fit_old"))
        sets[f"cal__old__{t}"] = (t, task_split(cfg, t, "cal_old"))
    for comp in cfg["tripwire"]["fit_ref"]:
        sets[f"fit__ref__{comp}"] = (comp, ref_split(cfg, comp, "fit"))
    for comp in cfg["tripwire"]["cal_ref"]:
        sets[f"cal__ref__{comp}"] = (comp, ref_split(cfg, comp, "cal"))
    for pool in cfg["pools"]:
        sets[f"eval__{pool}"] = (pool, eval_pool(cfg, pool))
    extra = [f"advwrap_{t}" for t in tasks if f"advwrap_{t}" in POOLS] + list(EXTRA_POOLS)
    for name in extra:
        if POOLS[name].source in cfg["pools"]:
            sets[f"eval__{name}"] = (name, eval_pool(cfg, POOLS[name].source))
    if "eval__wildchat" not in sets and os.path.exists(wildchat_path(cfg)):   # Nov 1: real-user requests
        sets["eval__wildchat"] = ("wildchat", wildchat_rows(cfg))
    fresh = dolly_fresh_rows(cfg)   # paper v3 (p3.7): 10,000 Dolly requests no pool of this experiment uses
    if fresh:
        sets["eval__dolly_fresh"] = ("dolly", fresh)
    set_data_dir(cfg["data_dir"])   # paper v3: the stress families of the config's tasks (passages from Dolly)
    for name in stress_pools(tasks):
        if POOLS[name].source in cfg["pools"]:
            sets[f"eval__{name}"] = (name, eval_pool(cfg, POOLS[name].source))
    for name in stress2_pools():   # Nov 2 (step p3.10b): the second round of forms, the same in every experiment
        if POOLS[name].source in cfg["pools"]:
            sets[f"eval__{name}"] = (name, eval_pool(cfg, POOLS[name].source))
    for name in external_pools(cfg):   # Nov 2 (step p3.12): harmful requests from benchmarks we did not write
        sets[f"eval__{name}"] = (name, external_rows(cfg, name))
    for name in prospective_pools(cfg):   # Nov 4 (step p3.19): prompt sets written by others, firing only
        sets[f"eval__{name}"] = (name, prospective_rows(cfg, name))
    if sysprompts_present(cfg["data_dir"]) and "dolly" in cfg["pools"]:   # p3.19: Dolly behind others' system prompts
        set_prospective_dir(cfg["data_dir"])
        sets["eval__pr_sys_dolly"] = ("pr_sys_dolly", eval_pool(cfg, "dolly"))
    for name in widen_pools(cfg):   # Nov 5 (step p3.23): harmful sets no gate is fitted or calibrated on
        sets[f"eval__{name}"] = (name, widen_rows(cfg, name))
    if "advbench" in cfg["tripwire"]["fit_ref"] and "advbench" in cfg["tripwire"]["cal_ref"]:
        for t in tasks:
            if f"advwrap_{t}" in POOLS:
                sets[f"wrap__{t}__fit"] = (f"advwrap_{t}", ref_split(cfg, "advbench", "fit"))
                sets[f"wrap__{t}__cal"] = (f"advwrap_{t}", ref_split(cfg, "advbench", "cal"))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--rl", type=int, default=1000, help="RL prompts per task to keep (unlabeled new-task prompts)")
    ap.add_argument("--out", default="")
    ap.add_argument("--add", action="store_true", help="keep the stored features, compute only missing prompt sets")
    args = ap.parse_args()
    dist.init()
    if not dist.is_main():   # one process does everything (features of the trunk only)
        dist.finish()
        return
    cfg = load_config(args.config, args.set)
    out_dir = args.out or os.path.join(cfg["out_dir"], cfg["exp"], "analysis")
    os.makedirs(out_dir, exist_ok=True)
    gl = int(cfg["gate_layer"])
    fb = int(cfg["gen"]["feature_batch_size"])
    t0 = time.time()

    model, tok = load_model(cfg["model_path"], cfg["dtype"], device=dist.device())
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    state = GateState()
    state.gate_from = template_prefix_len(encode)
    pad = tok.pad_token_id

    arrays, meta = {}, {"exp": cfg["exp"], "model": cfg["model_path"], "gate_layer": gl, "device": device_name(),
                        "stream": list(cfg["stream"]), "keys": {}, "probes": []}
    old_npz, old_json = os.path.join(out_dir, "gate_features.npz"), os.path.join(out_dir, "gate_features.json")
    if args.add and os.path.exists(old_npz) and os.path.exists(old_json):
        meta = json.load(open(old_json))
        if meta.get("device") != device_name() or int(meta.get("gate_layer", gl)) != gl:
            raise SystemExit(f"--add: stored features are from {meta.get('device')} (gate layer {meta.get('gate_layer')}),"
                             f" this job runs on {device_name()} (gate layer {gl}); submit it on the same GPU model")
        with np.load(old_npz) as z:
            arrays = {k: z[k] for k in z.files if not k.startswith("probe__")}
        meta["probes"] = []
        print(f"--add: {len(arrays)} stored prompt sets kept", flush=True)
    with torch.no_grad():
        for key, (pool, exs) in prompt_sets(cfg, args.rl).items():
            if key in arrays:
                continue
            prompts = [encode(POOLS[pool].prompt(ex)) for ex in exs]
            X = last_prompt_features(model, state, gl, prompts, pad, fb)
            arrays[key] = X.float().cpu().numpy().astype(np.float16)
            meta["keys"][key] = {"pool": pool, "n": len(exs), "ids": [ex.get("id") for ex in exs]}
            print(f"  {key}: {len(exs)} prompts ({time.time() - t0:.0f}s)", flush=True)

    # the deployed prompt-time tripwires (same for every seed: they read the frozen trunk on prompts only)
    for k in range(1, len(cfg["stream"]) + 1):
        p = os.path.join(cfg["out_dir"], cfg["exp"], "s0", "tripwire_prompt", f"stage{k}", "tripwire", "tripwire.pt")
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)
        for name in ("mu", "sigma", "a", "b", "kappa"):
            arrays[f"probe__{k}__{name}"] = sd[name].float().numpy().astype(np.float32).reshape(-1)
        meta["probes"].append(k)
    tmp = os.path.join(out_dir, "gate_features.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(out_dir, "gate_features.npz"))
    with open(os.path.join(out_dir, "gate_features.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out_dir}/gate_features.npz: {len(meta['keys'])} prompt sets, deployed probes of stages "
          f"{meta['probes'] or 'none'} ({time.time() - t0:.0f}s, {device_name()})", flush=True)
    dist.finish()


if __name__ == "__main__":
    main()
