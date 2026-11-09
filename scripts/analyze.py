#!/usr/bin/env python
"""Offline analyses for the paper (no torch, no GPU; reads run folders).

  limit     Theorem 4.1 on the overlap experiment. (a) The probe's ROC (firing on the new task vs on the
            reference) traced from post-hoc scores; (b) every method's operating point (share of reference and
            new-task responses that changed); (c) a Donsker-Varadhan estimate of a LOWER bound on
            KL(D_new || D_ref) from the tripwire scores, which shows whether the KL form of the bound,
            (KL + log 2) / log(1/eps), is informative (it is vacuous once it exceeds 1).
              python scripts/analyze.py limit --runs runs/v2_overlap_q15/s0 --out runs/v2_overlap_q15/s0/analysis
  fired     what the certificate is silent about: score changes on the reference prompts where a gate fired.
              python scripts/analyze.py fired --runs runs/v2_stream_q15/s0
  coverage  resampling check of the PAC calibration (Theorem 3.6): pool the calibration scores with the post-hoc
            scores of the same component, recalibrate on random subsets, count how often the firing rate of the
            pool exceeds eps (should be at most delta).
              python scripts/analyze.py coverage --runs runs/v2_stream_q15/s0 --method tripwire
  monitor   deployment monitoring and repair (Proposition 5.x): traffic mixes reference prompts with an unseen
            distribution; a mixture e-process tests "firing rate <= eps" at every step (anytime-valid, Ville);
            after an alarm, fresh traffic samples join the calibration and the threshold is raised.
              python scripts/analyze.py monitor --run runs/stream_qwen15/tripwire --stage 3 --shift mmlu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from twcore.certify import calib_index, clopper_pearson_upper  # noqa: E402

REF = ("dolly", "advbench", "xstest", "mmlu")
METRIC = {"dolly": "leak", "advbench": "refused", "xstest": "refused", "mmlu": "correct"}


# ----------------------------------------------------------------------------- loading
def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    out.sort(key=lambda r: r["i"])
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stage_eval(mdir, k):
    edir = os.path.join(mdir, f"stage{k}", "eval")
    if not os.path.exists(os.path.join(edir, "DONE")):
        return None
    return {fn[:-6]: rows(os.path.join(edir, fn)) for fn in os.listdir(edir) if fn.endswith(".jsonl")}


def final_stage(mdir):
    ks = [int(d[5:]) for d in os.listdir(mdir) if d.startswith("stage") and d[5:].isdigit()
          and os.path.exists(os.path.join(mdir, d, "eval", "DONE"))]
    return max(ks) if ks else None


def fired(r):
    return any(v is not None for v in r.get("fire", {}).values())


def method_dirs(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "stage0")))


def write_dat(path, header, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- limit (Theorem 4.1)
def roc_points(new, ref):
    thr = sorted(set(new + ref), reverse=True)
    pts = [(0.0, 0.0)]
    for t in thr:
        pts.append((mean(float(s >= t) for s in ref), mean(float(s >= t) for s in new)))
    pts.append((1.0, 1.0))
    return pts


def dv_lower_bound(new, ref):
    """max over a >= 0 of mean_new(a s) - log mean_ref(exp(a s)) (Donsker-Varadhan with f = a s; the offset b
    cancels). A lower bound on KL(D_new || D_ref) up to sampling error."""
    if not new or not ref:
        return float("nan"), float("nan")
    c = max(ref)  # stabilize the log-mean-exp

    def dv(a):
        lme = a * c + math.log(mean(math.exp(a * (s - c)) for s in ref))
        return a * mean(new) - lme

    best_a, best = 0.0, 0.0
    for a in [0.0] + [10 ** (x / 10) for x in range(-30, 21)]:
        v = dv(a)
        if v > best:
            best_a, best = a, v
    lo, hi = best_a / 1.6, best_a * 1.6 + 1e-9
    for _ in range(60):  # golden-section refinement
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if dv(m1) > dv(m2):
            hi = m2
        else:
            lo = m1
    a = 0.5 * (lo + hi)
    return (dv(a), a) if dv(a) > best else (best, best_a)


def cmd_limit(args):
    root = args.runs
    out = args.out or os.path.join(root, "analysis")
    task = None
    res = {"methods": {}}
    for m in method_dirs(root):
        md = os.path.join(root, m)
        K = final_stage(md)
        s0, sK = stage_eval(md, 0), stage_eval(md, K) if K else None
        if not K or s0 is None or sK is None:
            continue
        cfg = json.load(open(os.path.join(md, "config.json")))["config"] if os.path.exists(os.path.join(md, "config.json")) else {}
        task = task or (cfg.get("stream") or [None])[-1]
        if task not in sK or "dolly" not in sK:
            continue
        chg = lambda p: mean(float(a["out"] != b["out"]) for a, b in zip(sK[p], s0[p]))  # noqa: E731
        acc = lambda st: mean(r["m"]["correct"] for r in st[task])  # noqa: E731
        res["methods"][m] = {"ref_changed": chg("dolly"), "new_changed": chg(task), "gain": acc(sK) - acc(s0),
                             "fire_ref": mean(float(fired(r)) for r in sK["dolly"]),
                             "fire_new": mean(float(fired(r)) for r in sK[task])}
    # the probe of the default tripwire run gives the ROC and the KL estimate; the prompt-time probe first (by
    # Proposition "responses carry no evidence" it is the better test), the anytime probe's ROC is written too
    probe_run = next((m for m in ("tripwire_prompt", "tripwire") if m in res["methods"]), None)
    if probe_run is None:
        raise SystemExit("no finished tripwire run in " + root)
    tj = json.load(open(os.path.join(root, probe_run, "stage1", "tripwire", "tripwire.json")))
    new, ref = tj["posthoc"]["new"], tj["posthoc"]["dolly"]
    os.makedirs(out, exist_ok=True)
    for other, fname in (("tripwire", "limit_roc_anytime.dat"), ("tripwire_prompt", "limit_roc_prompt.dat")):
        oj = os.path.join(root, other, "stage1", "tripwire", "tripwire.json")
        if other in res["methods"] and os.path.exists(oj):
            ph = json.load(open(oj))["posthoc"]
            write_dat(os.path.join(out, fname), "ref new",
                      [f"{100 * x:.3f} {100 * y:.3f}" for x, y in roc_points(ph["new"], ph["dolly"])])
    # data processing on the channel x -> 1[response changed]: every run gives KL(D_new||D_ref) >= kl(a||b)
    def bkl(a, b):
        a, b = min(max(a, 1e-9), 1 - 1e-9), min(max(b, 1e-9), 1 - 1e-9)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    pts_kl = {m: bkl(v["new_changed"], v["ref_changed"]) for m, v in res["methods"].items()
              if v["new_changed"] > v["ref_changed"]}
    if pts_kl:
        best = max(pts_kl, key=pts_kl.get)
        res["kl_lower_bound_from_runs"] = {"value": pts_kl[best], "run": best}
    kl_lb, a = dv_lower_bound(new, ref)
    rng = random.Random(0)
    boots = sorted(dv_lower_bound([rng.choice(new) for _ in new], [rng.choice(ref) for _ in ref])[0] for _ in range(200))
    res["probe_run"] = probe_run
    res["kl_dv_lower_bound"] = {"estimate": kl_lb, "a": a, "boot_5pct": boots[10], "boot_95pct": boots[189]}
    res["kl_form_bound_at_eps"] = {str(e): (kl_lb + math.log(2)) / math.log(1 / e) for e in (0.01, 0.03, 0.1, 0.3)}
    pts = roc_points(new, ref)
    write_dat(os.path.join(out, "limit_roc.dat"), "ref new", [f"{100 * x:.3f} {100 * y:.3f}" for x, y in pts])
    write_dat(os.path.join(out, "limit_methods.dat"), "ref new gain label",
              [f"{100 * v['ref_changed']:.2f} {100 * v['new_changed']:.2f} {100 * v['gain']:.2f} {m.replace('_', chr(92) + '_')}"
               for m, v in res["methods"].items()])
    print(f"probe ({probe_run}) ROC: {len(pts)} points; AUROC from the posthoc scores is in tripwire.json")
    print(f"KL(D_new || D_ref) >= {kl_lb:.2f} nats (Donsker-Varadhan estimate, bootstrap 90% "
          f"[{boots[10]:.2f}, {boots[189]:.2f}])")
    if "kl_lower_bound_from_runs" in res:
        kb = res["kl_lower_bound_from_runs"]
        print(f"KL(D_new || D_ref) >= {kb['value']:.2f} nats from the binary KL of run {kb['run']}'s changed shares "
              "(data processing; point estimate)")
        kl_lb = kb["value"]   # the Donsker-Varadhan sample estimate is unstable when the score separates well
    for e in (0.01, 0.03, 0.1, 0.3):
        v = (kl_lb + math.log(2)) / math.log(1 / e)
        res["kl_form_bound_at_eps"][str(e)] = v
        print(f"  KL form of Theorem 4.1 at eps={e}: right side >= {v:.2f}{'  (vacuous)' if v >= 1 else ''} "
              "(KL is only bounded from below, so this is a lower bound on the bound)")
    with open(os.path.join(out, "limit.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("| method | reference changed % | new task changed % | new-task gain (pts) | fires ref % | fires new % |")
    print("|---|---|---|---|---|---|")
    for m, v in res["methods"].items():
        print(f"| {m} | {100 * v['ref_changed']:.1f} | {100 * v['new_changed']:.1f} | {100 * v['gain']:+.1f} | "
              f"{100 * v['fire_ref']:.1f} | {100 * v['fire_new']:.1f} |")
    print("wrote", out)


# ----------------------------------------------------------------------------- fired subset
def cmd_fired(args):
    root = args.runs
    print("| method | pool | fired / n | score on fired: base -> final | changed among fired % | changed overall % |")
    print("|---|---|---|---|---|---|")
    res = {}
    for m in method_dirs(root):
        md = os.path.join(root, m)
        K = final_stage(md)
        if not K:
            continue
        s0, sK = stage_eval(md, 0), stage_eval(md, K)
        for p in REF:
            if p not in sK:
                continue
            pairs = list(zip(sK[p], s0[p]))
            fi = [(a, b) for a, b in pairs if fired(a)]
            met = METRIC[p]
            row = {"n": len(pairs), "fired": len(fi), "metric": met,
                   "fired_base": mean(b["m"][met] for _, b in fi), "fired_final": mean(a["m"][met] for a, _ in fi),
                   "fired_changed": mean(float(a["out"] != b["out"]) for a, b in fi),
                   "changed": mean(float(a["out"] != b["out"]) for a, b in pairs)}
            res.setdefault(m, {})[p] = row
            if any(fired(a) for a, _ in pairs) or m.startswith(("tripwire", "router")):
                f = lambda v: "--" if v != v else f"{100 * v:.1f}"  # noqa: E731
                print(f"| {m} | {p} ({met}) | {row['fired']}/{row['n']} | {f(row['fired_base'])} -> {f(row['fired_final'])} | "
                      f"{f(row['fired_changed'])} | {f(row['changed'])} |")
    out = args.out or os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fired.json"), "w") as f:
        json.dump(res, f, indent=2)


# ----------------------------------------------------------------------------- calibration coverage
def cmd_coverage(args):
    md = os.path.join(args.runs, args.method)
    rng = random.Random(args.seed)
    print("| stage | component | pooled n | calibration n | violation rate | delta per component |")
    print("|---|---|---|---|---|---|")
    for sd in sorted(d for d in os.listdir(md) if d.startswith("stage") and d[5:].isdigit() and d != "stage0"):
        tdir = os.path.join(md, sd, "tripwire")
        if not os.path.exists(os.path.join(tdir, "tripwire.json")):
            continue
        tj = json.load(open(os.path.join(tdir, "tripwire.json")))
        eps = tj["eps"]
        d_comp = tj.get("calibration", {}).get("delta_per_component", tj["delta"] / max(1, len(tj["components"])))
        for comp in ("dolly", "advbench"):
            cpath = os.path.join(tdir, f"cal_{comp}.json")
            if not os.path.exists(cpath) or comp not in tj["posthoc"]:
                continue
            cal = json.load(open(cpath))
            pool = cal + list(tj["posthoc"][comp])
            n = len(cal)
            k = calib_index(n, eps, d_comp)
            if k is None:
                continue
            viol = 0
            for _ in range(args.reps):
                sub = sorted(rng.sample(pool, n))
                kap = sub[k - 1]
                viol += mean(float(s > kap) for s in pool) > eps
            print(f"| {sd} | {comp} | {len(pool)} | {n} | {viol / args.reps:.3f} | {d_comp:.4f} |")


# ----------------------------------------------------------------------------- monitoring and repair
class MixtureEProcess:
    """Anytime-valid test of H0: P(fire) <= eps. Average of betting martingales prod(1 + lam (X - eps)) over a grid
    of lam in (0, 1/eps); under H0 each is a non-negative supermartingale, so is the average, and by Ville's
    inequality P(sup_t M_t >= 1/delta) <= delta."""

    def __init__(self, eps, fracs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        self.eps = eps
        self.lams = [f / eps for f in fracs]
        self.logw = [0.0] * len(self.lams)

    def update(self, x):
        for i, lam in enumerate(self.lams):
            self.logw[i] += math.log(1 + lam * (x - self.eps))
        mx = max(self.logw)
        return mx + math.log(mean(math.exp(v - mx) for v in self.logw))  # log M_t


def cmd_monitor(args):
    tdir = os.path.join(args.run, f"stage{args.stage}", "tripwire")
    tj = json.load(open(os.path.join(tdir, "tripwire.json")))
    ph, kappa, eps, delta = tj["posthoc"], tj["kappa"], tj["eps"], tj["delta"]
    if args.shift not in ph:
        raise SystemExit(f"no post-hoc scores for {args.shift}; available: {sorted(ph)}")
    ref, shift, new = list(ph["dolly"]), list(ph[args.shift]), list(ph["new"])
    print(f"stage {args.stage}: kappa {kappa:.3f}, eps {eps}; firing on dolly {mean(float(s > kappa) for s in ref):.3f}, "
          f"on {args.shift} {mean(float(s > kappa) for s in shift):.3f}, on the new task {mean(float(s > kappa) for s in new):.3f}")
    rng = random.Random(args.seed)
    print("| shift share | alarm rate | median steps to alarm | firing on traffic: before -> after repair (alarmed runs) | "
          "new-task activation: before -> after (alarmed runs) |")
    print("|---|---|---|---|---|")
    out = {}
    for share in args.shares:
        alarms, steps, before, after, act = 0, [], [], [], []
        act0 = mean(float(s > kappa) for s in new)
        for _ in range(args.reps):
            traffic = [(rng.choice(shift) if rng.random() < share else rng.choice(ref)) for _ in range(args.horizon)]
            ep = MixtureEProcess(eps)
            t_alarm = None
            for t, s in enumerate(traffic):
                if ep.update(float(s > kappa)) >= math.log(1 / delta):
                    t_alarm = t + 1
                    break
            pop = [(rng.choice(shift) if rng.random() < share else rng.choice(ref)) for _ in range(4000)]
            before.append(mean(float(s > kappa) for s in pop))
            if t_alarm is None:
                continue
            alarms += 1
            steps.append(t_alarm)
            # repair: fresh traffic samples (independent of the alarm data) become a calibration component
            fresh = sorted((rng.choice(shift) if rng.random() < share else rng.choice(ref)) for _ in range(args.repair_n))
            k = calib_index(len(fresh), eps, delta)
            kap2 = max(kappa, fresh[k - 1]) if k is not None else float("inf")
            after.append(mean(float(s > kap2) for s in pop))
            act.append(mean(float(s > kap2) for s in new))
        med = sorted(steps)[len(steps) // 2] if steps else None
        out[str(share)] = {"alarm_rate": alarms / args.reps, "median_steps": med, "fire_before": mean(before),
                           "fire_after_alarmed": mean(after), "activation_before": act0, "activation_after_alarmed": mean(act)}
        f3 = lambda v: "--" if v != v else f"{v:.3f}"  # noqa: E731
        print(f"| {share:.2f} | {alarms / args.reps:.3f} | {med if med is not None else '--'} | "
              f"{mean(before):.3f} -> {f3(mean(after))} | {act0:.3f} -> {f3(mean(act))} |")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"monitor_stage{args.stage}_{args.shift}.json"), "w") as f:
            json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("limit"); a.add_argument("--runs", required=True); a.add_argument("--out", default=None)
    a = sub.add_parser("fired"); a.add_argument("--runs", required=True); a.add_argument("--out", default=None)
    a = sub.add_parser("coverage"); a.add_argument("--runs", required=True); a.add_argument("--method", default="tripwire")
    a.add_argument("--reps", type=int, default=2000); a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("monitor"); a.add_argument("--run", required=True); a.add_argument("--stage", type=int, required=True)
    a.add_argument("--shift", default="mmlu"); a.add_argument("--shares", type=float, nargs="*", default=[0.0, 0.05, 0.2, 0.5])
    a.add_argument("--horizon", type=int, default=2000); a.add_argument("--repair-n", type=int, default=300)
    a.add_argument("--reps", type=int, default=200); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default=None)
    args = ap.parse_args()
    {"limit": cmd_limit, "fired": cmd_fired, "coverage": cmd_coverage, "monitor": cmd_monitor}[args.cmd](args)


if __name__ == "__main__":
    main()
