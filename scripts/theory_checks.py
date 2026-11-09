#!/usr/bin/env python
"""Numerical checks of the theory items planned for the next paper pass (Nov 3; claude/tripwire_weakness_strategy_nov3.md
in the project). CPU, seconds, stored results only.

    python scripts/theory_checks.py [--np-gap paper/v3_results/p3_14/np_gap.json] [--out paper/v3_results/theory]

1. Finite-sample price of certification. With n calibration prompts and the order statistic k of Theorem 3.6, the
   true size of a gate with continuous scores is Beta(n + 1 - k, k) distributed, so its mean is (n + 1 - k) / (n + 1)
   (exact) and about eps - sqrt(eps * log(1 / delta_c) / n) (the approximation in the plan). The coverage lost to the
   slack is beta(eps) - beta(size). Checked against the synthetic study of p3.14: where the optimum is known in closed
   form (shift: beta(e) = Phi(2.5 - z_{1-e})), the predicted PAC coverage is compared with the measured one.
2. Per-component limit: the dual weights of the optimal test in the overlap cells of p3.14 (which component binds).
3. Gain behind the gate: kept - beta * G against the worst case 2 beta (1 - beta) (rewards in [0, 1]) and the observed
   errors, from every stored switch summary.
Outputs: theory_checks.json and theory_checks.txt in --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from scipy.stats import beta as beta_dist, norm  # noqa: E402

from twcore.certify import calib_index  # noqa: E402


def size_prediction(n, eps, delta):
    k = calib_index(n, eps, delta)
    if k is None:
        return None
    a, b = n + 1 - k, k
    return {"n": n, "k": k, "mean": a / (a + b), "q05": float(beta_dist.ppf(0.05, a, b)),
            "q95": float(beta_dist.ppf(0.95, a, b)), "p_exceed": float(beta_dist.sf(eps, a, b)),
            "approx": eps - math.sqrt(eps * math.log(1 / delta) / n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np-gap", default=os.path.join("paper", "v3_results", "p3_14", "np_gap.json"))
    ap.add_argument("--switch-glob", default="paper/*/*/switch/switch_summary.json")
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "theory"))
    args = ap.parse_args()
    res, L = {}, []
    eps, delta = 0.03, 0.05

    # 1. finite-sample price
    L.append("1. finite-sample price of certification (eps 0.03, delta 0.05)")
    sizes = {}
    for n, K in ((1570, 1), (700, 3), (170, 3), (135, 1), (400, 1), (5000, 1)):
        p = size_prediction(n, eps, delta / K)
        sizes[f"{n}/{K}"] = p
        L.append(f"  n {n:5d}, {K} component(s): k {p['k']}, true size mean {100 * p['mean']:.2f}% "
                 f"[{100 * p['q05']:.2f}, {100 * p['q95']:.2f}], P(size > eps) {p['p_exceed']:.3f}, "
                 f"approximation {100 * p['approx']:.2f}%")
    res["sizes"] = sizes
    if os.path.exists(args.np_gap):
        cells = json.load(open(args.np_gap))["cells"]
        s = sizes["1570/1"]["mean"]
        L.append(f"  shift (closed form beta(e) = Phi(2.5 - z_(1-e))): beta(3%) = "
                 f"{100 * norm.cdf(2.5 - norm.ppf(1 - eps)):.1f}%, beta(mean size {100 * s:.2f}%) = "
                 f"{100 * norm.cdf(2.5 - norm.ppf(1 - s)):.1f}%: predicted slack "
                 f"{100 * (norm.cdf(2.5 - norm.ppf(1 - eps)) - norm.cdf(2.5 - norm.ppf(1 - s))):.1f} points")
        rows = []
        for c in cells:
            mean = lambda v: sum(v) / len(v)  # noqa: E731
            rows.append({"scenario": c["scenario"], "d": c["d"], "n": c["n_new"],
                         "slack_measured": mean(c["probe_oracle"]) - mean(c["probe"]),
                         "fire_measured": mean(c["probe_fire"])})
            if c["scenario"] == "shift":
                pred = norm.cdf(2.5 - norm.ppf(1 - s))
                L.append(f"  shift d={c['d']:3d} n={c['n_new']:5d}: measured slack {100 * rows[-1]['slack_measured']:.1f} "
                         f"points, measured size {100 * rows[-1]['fire_measured']:.2f}%; PAC coverage measured "
                         f"{100 * mean(c['probe']):.1f}%, NP at the predicted size {100 * pred:.1f}%")
        res["np_gap"] = rows
        # 2. per-component limit
        L.append("2. per-component limit: dual weights of the optimal test (overlap cells of p3.14)")
        for c in cells:
            if c["scenario"] == "overlap":
                L.append(f"  d={c['d']:3d} n={c['n_new']:5d}: weights {[round(w, 3) for w in c['np_weights']]}, "
                         f"NP per component {100 * c['np']:.1f}% vs mixture constraint {100 * c['np_mix']:.1f}%")
    # 3. gain behind the gate
    L.append("3. gain behind the deployed gate: kept - beta * G against the worst case 2 beta (1 - beta)")
    errs = []
    seen = set()
    for path in sorted(glob.glob(args.switch_glob)):
        d = json.load(open(path))
        if d["exp"] in seen or len(d["tasks"]) != 1 or "seq" not in d.get("families", {}):
            continue
        seen.add(d["exp"])
        t = d["tasks"][0]
        g = d["gates"].get("ratio|0.03")
        sw = d["families"]["seq"].get("switch", {}).get("ratio|0.03", {})
        own = d["families"]["seq"]["own"]
        key = f"{t}:correct"
        if g is None or key not in sw or key not in own:
            continue
        b, G, kept = g["power_pre"][t], own[key][0], sw[key][0]
        err = kept - b * G
        errs.append({"exp": d["exp"], "beta": b, "gain": G, "kept": kept, "error": err, "worst": 2 * b * (1 - b)})
        L.append(f"  {d['exp']:24s} beta {100 * b:5.1f}%  G {100 * G:+5.1f}  kept {100 * kept:+5.1f}  error "
                 f"{100 * err:+5.2f}  worst case +-{100 * 2 * b * (1 - b):.1f} points")
    if errs:
        L.append(f"  largest |error| {100 * max(abs(e['error']) for e in errs):.2f} points over {len(errs)} experiments; "
                 f"largest worst case {100 * max(e['worst'] for e in errs):.1f} points")
    res["gain"] = errs
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "theory_checks.json"), "w"), indent=1)
    open(os.path.join(args.out, "theory_checks.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
