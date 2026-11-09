#!/usr/bin/env python
"""How separable is each new task from the reference, before any training? (CPU: numpy + scipy; seconds.)

    python scripts/separability.py --config configs/v2/ladder_concisectx_q15.yaml [--range 0.70,0.95]

Reads <runs>/analysis/gate_features.npz (scripts/gate_features.py), fits the prompt-time probe of every stage on its
fitting prompts (the deployed probe when one is stored) and reports the AUROC of held-out new-task prompts against
held-out Dolly prompts and against every held-out reference pool together, and the firing on new-task RL prompts at
the threshold that fires on eps of the held-out Dolly prompts (the empirical Neyman-Pearson power). Writes
<runs>/analysis/separability.json with "in_range": the stage-1 AUROC against Dolly lies inside --range (slurm/plan.sh
trains the middle-rung task only then). Run by slurm/features.sbatch after the features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from twcore.config import load_config  # noqa: E402
from switch_report import REF_POOLS, Experiment  # noqa: E402


def auroc(pos, neg):
    """P(score of a positive > score of a negative) + 0.5 P(tie), by ranks"""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)   # average ranks of ties
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--features", default="")
    ap.add_argument("--range", default="0.70,0.95", help="AUROC range of a middle-rung task")
    ap.add_argument("--eps", default="0.03,0.1")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    runs = os.path.join(cfg["out_dir"], cfg["exp"])
    feat = args.features or os.path.join(runs, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        raise SystemExit(f"{feat} is missing (scripts/gate_features.py first)")
    ex = Experiment(cfg, feat, feat[:-4] + ".json")
    lo, hi = (float(x) for x in args.range.split(","))
    out = {"exp": cfg["exp"], "range": [lo, hi], "stages": []}
    for k in range(1, ex.K + 1):
        task, pos, neg, _, _ = ex.stage_data(k, "ratio", None)
        probe, dep = ex.probe(k, "ratio", None, pos, neg)
        if not ex.has(f"eval__{task}"):
            continue
        s_new = probe.score(ex.X(f"eval__{task}"))
        s_rl = probe.score(ex.X(f"rl__{task}"))
        refs = {p: probe.score(ex.X(f"eval__{p}")) for p in REF_POOLS if ex.has(f"eval__{p}") and p in ("dolly", "advbench")}
        st = {"stage": k, "task": task, "deployed_probe": bool(dep),
              "auroc_dolly": auroc(s_new, refs["dolly"]) if "dolly" in refs else None,
              "auroc_ref": auroc(s_new, np.concatenate(list(refs.values()))) if refs else None, "np_power": {}}
        if "dolly" in refs:
            for e in (float(x) for x in args.eps.split(",")):
                kap = float(np.quantile(refs["dolly"], 1.0 - e))
                st["np_power"][f"{e:g}"] = float((s_rl > kap).mean())
        out["stages"].append(st)
        print(f"{cfg['exp']} stage {k} ({task}): AUROC vs Dolly {st['auroc_dolly']:.3f}, vs reference "
              f"{st['auroc_ref']:.3f}; RL firing at Dolly firing eps: {st['np_power']}")
    a = out["stages"][0]["auroc_dolly"] if out["stages"] else None
    out["in_range"] = a is not None and lo <= a <= hi
    path = os.path.join(runs, "analysis", "separability.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(f"wrote {path}: in_range {out['in_range']}")


if __name__ == "__main__":
    main()
