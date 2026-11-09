#!/usr/bin/env python
"""Drop stored prompt sets from an experiment's gate features, so that gate_features.py --add computes them again
(Nov 5: step p3.23, after data/wd_aya.jsonl was written again). CPU, seconds; keeps a backup of both files.

    python scripts/drop_feature_sets.py --config configs/v2/ladder_concise_q15.yaml --keys eval__wd_aya

Writes <out_dir>/<exp>/analysis/gate_features.npz and .json without the named keys, after copying the old files to
gate_features.before_drop.npz / .json (an existing backup is never overwritten). Only keys named on the command line
are removed; deployed probes (probe__*) cannot be dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from twcore.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--keys", required=True, help="comma-separated stored keys, e.g. eval__wd_aya")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d = os.path.join(cfg["out_dir"], cfg["exp"], "analysis")
    npz, js = os.path.join(d, "gate_features.npz"), os.path.join(d, "gate_features.json")
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if any(k.startswith("probe__") for k in keys):
        raise SystemExit("deployed probes (probe__*) cannot be dropped")
    meta = json.load(open(js))
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    gone = [k for k in keys if k in arrays or k in meta["keys"]]
    if not gone:
        print(f"{npz}: none of {keys} stored; nothing changed")
        return
    for src, dst in ((npz, os.path.join(d, "gate_features.before_drop.npz")),
                     (js, os.path.join(d, "gate_features.before_drop.json"))):
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    for k in gone:
        arrays.pop(k, None)
        meta["keys"].pop(k, None)
    tmp = os.path.join(d, "gate_features.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz)
    json.dump(meta, open(js, "w"), indent=1)
    print(f"{npz}: dropped {gone}; {len(meta['keys'])} prompt sets left (backup: gate_features.before_drop.*)")


if __name__ == "__main__":
    main()
