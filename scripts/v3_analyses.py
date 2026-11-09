#!/usr/bin/env python
"""Paper v3 analyses from stored gate features and stored responses (CPU only: numpy + scipy). PLAN_PAPER3.md.

    python scripts/v3_analyses.py --exp configs/v2/ladder_concise_q15.yaml:Concise:advbench,general \\
        --exp configs/v2/overlap_q15.yaml:Nq --out paper/v3_results

--exp <config>:<Name>[:<protected components>]; Name is letters only (macro prefix). Needs
runs/<exp>/analysis/gate_features.npz (scripts/gate_features.py) and finished runs. For every experiment:
  ledger    the deployed and the two-sided tripwire at the config's eps: the target, the in-sample firing on each
            reference component's calibration prompts, the firing on held-out prompts of every reference and
            evaluation pool, its 95% Clopper-Pearson bound, and the one-sided binomial p-value of the held-out count
            under a firing rate equal to the target (a small p-value would be evidence that the certificate failed)
  lattice   every non-empty set of reference components (earlier tasks of a stream always kept): a probe fitted and
            calibrated on those components only (the deployed probe when all are kept), without and with the region
            check; its coverage beta, its firing on every evaluation pool, and every ungated run behind it
  size      calibration prompts per component, 25 to all, 20 random subsets each: the level the order statistic
            certifies, beta, and the held-out firing on each reference pool
  budget    (single tasks) one level per component: harmful requests and task-like prompts at 0.01, Dolly at 0.01, 0.03,
            0.1, 0.3 or not protected (the deployed probe); beta and every ungated run behind the gate
  stress    the deployed, two-sided, remedy and protect gates (scripts/switch_report.py variants) on the stress families
            (twcore/stress.py), the jailbreak and format-shift pools and WildChat: firing, its bound, and refusal of every
            run on them, as trained and (ungated runs) behind each gate
  methods   every run family as trained: gain, changed Dolly responses, refusal, over-refusal, MMLU, Dolly abstention
Outputs in --out: <exp>/v3.json; v3_tables/*.tex and *.dat; v3_numbers.tex (macros \\vthree<Name>...).
"""
from __future__ import annotations

import argparse
import itertools
import gc
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

import switch_report as sr  # noqa: E402
from twcore.certify import binom_sf_ge, calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402
from twcore.squad2 import abstains  # noqa: E402

GATED = sr.GATED
REF_EVAL = ("dolly", "advbench")                 # reference components that have a held-out evaluation pool
OFF_EVAL = ("xstest", "mmlu", "wildchat")        # never used to fit or calibrate
ROBUST_PREFIX = ("st_", "advwrap_", "jb_", "advbench_", "dolly_")
SIZES = (25, 50, 100, 200, 400, 0)               # 0: every calibration prompt
SIZE_REPS = 20
SAFETY_EPS = 0.01
DOLLY_EPS = (0.01, 0.03, 0.1, 0.3, 1.0)          # 1.0: Dolly not protected
KEYS = ("task:gain", "dolly:chg", "advbench:refused", "xstest:refused_safe", "mmlu:correct", "dolly:abstain",
        "dolly:subst")


# ----------------------------------------------------------------------------- gates
def make_gate(ex, comps, eps_of, region=False, alpha=0.02, deployed=None, n_cal=0, rng=None):
    """A gate fitted against `comps` (earlier tasks always added) and calibrated per component at eps_of[c] (a
    component at 1.0 or above is not protected; earlier tasks at the config's eps), delta split over the protected
    components. deployed=None: the deployed probe exactly when every reference component is kept. n_cal > 0: that
    many random calibration prompts per component (rng). Returns fire {pool: bool[]} (any stage), rl {task: bool[]},
    cal_fire and cert {"k:c": value}, kappa {k: value}."""
    delta, eps0 = float(ex.tc["delta"]), float(ex.tc["eps"])
    comps = [c for c in ex.cal_ref if c in comps]
    if deployed is None:
        deployed = len(comps) == len(ex.cal_ref)
    out = {"fire": {}, "rl": {}, "cal_fire": {}, "cert": {}, "kappa": {}, "comps": comps, "deployed": deployed}
    stages = []
    for k in range(1, ex.K + 1):
        task, prev = ex.tasks[k - 1], ex.tasks[:k - 1]
        pos = ex.X(f"fit__new__{task}")
        probe = None
        if deployed:
            dep = ex.deployed(k)
            probe = dep[0] if dep is not None else None
        if probe is None:
            key = ("v3probe", k, tuple(comps))
            if key not in ex._cache:
                neg = np.concatenate([ex.X(f"fit__ref__{c}") for c in ex.fit_ref if c in comps]
                                     + [ex.X(f"fit__old__{t}") for t in prev])
                ex._cache[key] = sr.fit_probe(pos, neg, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"]))
            probe = ex._cache[key]
        sup = None
        if region:
            key = ("v3sup", k, tuple(comps), bool(deployed), alpha)
            if key not in ex._cache:
                ex._cache[key] = sr.Support(probe, pos, ex.X(f"rl__{task}"), alpha)
            sup = ex._cache[key]
        gate = sr.StageGate(probe, sup, 0.0)
        cal = {c: ex.X(f"cal__ref__{c}") for c in comps}
        cal.update({f"old_{t}": ex.X(f"cal__old__{t}") for t in prev})
        if n_cal:
            cal = {c: (X[rng.choice(len(X), size=n_cal, replace=False)] if len(X) > n_cal else X) for c, X in cal.items()}
        cs = {c: gate.score(X) for c, X in cal.items()}
        lev = {c: (eps0 if c.startswith("old_") else float(eps_of.get(c, eps0))) for c in cs}
        prot = [c for c in cs if lev[c] < 1.0]
        kappa = -math.inf
        for c in prot:
            r = calibrate({c: cs[c].tolist()}, lev[c], delta / len(prot))
            kappa = max(kappa, float(r["kappa"]))
            out["cert"][f"{k}:{c}"] = float(r["components"][c]["eps_cert"])
        gate.kappa = kappa
        for c, v in cs.items():
            out["cal_fire"][f"{k}:{c}"] = float(np.mean(v > kappa)) if len(v) else None
        out["kappa"][k] = kappa
        stages.append((task, gate))
    for p in ex.eval_keys():
        X = ex.X(f"eval__{p}")
        out["fire"][p] = np.any([g.fires(X) for _, g in stages], axis=0)
    for t in ex.tasks:
        out["rl"][t] = np.any([g.fires(ex.X(f"rl__{t}")) for _, g in stages], axis=0)
    return out


def fire_stats(g, pools=None):
    """firing, count, n and 95% Clopper-Pearson bound per pool, and beta (mean firing on unlabeled RL prompts)"""
    st = {"pools": {}, "beta": float(np.mean([f.mean() for f in g["rl"].values()]))}
    for p, f in g["fire"].items():
        if pools is not None and p not in pools:
            continue
        X, n = int(f.sum()), len(f)
        st["pools"][p] = {"fire": X / n if n else None, "X": X, "n": n,
                          "cp": clopper_pearson_upper(X, n) if n else None}
    return st


# ----------------------------------------------------------------------------- runs
FAMILIES = ("seq", "kl", "refkl", "refkl_protect", "nullspace", "safemix", "router", "tripwire_prompt",
            "tripwire_prompt_protect")


def slim(rows):
    """keep what the metrics read: the response as a hash (only equality is used), its first 400 characters (the
    abstention check), scores, label and index"""
    if rows is None:
        return None
    return [{"i": r["i"], "out": hash(tuple(r["out"])), "text": (r.get("text") or "")[:400], "m": r["m"],
             **({"label": r["label"]} if "label" in r else {})} for r in rows]


class Runs:
    """Finished runs of an experiment (the families in FAMILIES): family -> list of (seed, method, B, R, labels)."""

    def __init__(self, root, K, families=FAMILIES):
        self.fams = {}
        if not os.path.isdir(root):
            return
        seeds = sorted((d for d in os.listdir(root) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
        for sd in seeds:
            for fam in sorted(os.listdir(os.path.join(root, sd))):
                if families and fam not in families:
                    continue
                md = os.path.join(root, sd, fam)
                cj = os.path.join(md, "config.json")
                if not (os.path.isdir(md) and os.path.exists(os.path.join(md, "DONE")) and os.path.exists(cj)):
                    continue
                info = json.load(open(cj))
                cfg = info.get("config", {})
                if cfg.get("lora", {}).get("dense") or cfg.get("grpo", {}).get("objective", "grpo") != "grpo":
                    continue   # the v7 pilot (full-rank, SFT) is reported elsewhere
                loaded = sr.load_run(md, K)
                if loaded is None:
                    continue
                B, R = ({p: slim(v) for p, v in d.items()} for d in loaded)
                jp = os.path.join(root, "analysis", "judge", f"{sd}__{fam}.json")
                labels = json.load(open(jp)).get("labels") if os.path.exists(jp) else None
                self.fams.setdefault(fam, []).append((sd, info.get("method"), B, R, labels))

    def ungated(self):
        return {f: v for f, v in self.fams.items() if v and v[0][1] not in GATED}


def abstain_delta(R, B, fire=None):
    """change in the share of Dolly responses that abstain (twcore/squad2.py), from the stored texts"""
    r, b = R.get("dolly"), B.get("dolly")
    if not r or not b or len(r) != len(b) or "text" not in r[0]:
        return None
    if fire is not None:
        f = fire.get("dolly")
        if f is None or len(f) != len(r):
            return None
        r = [ri if fi else bi for ri, bi, fi in zip(r, b, f)]
    return float(np.mean([abstains(x["text"]) for x in r]) - np.mean([abstains(x["text"]) for x in b]))


def run_metrics(R, B, tasks, labels, fire=None):
    m = sr.metrics(R, B, tasks, labels, fire)
    m["dolly:abstain"] = abstain_delta(R, B, fire)
    return m


def agg_family(lst, tasks, fire=None, keys=None):
    per = [run_metrics(R, B, tasks, labels, fire) for _, _, B, R, labels in lst]
    keys = keys or sorted({k for m in per for k in m})
    return {k: sr.agg([m.get(k) for m in per]) for k in keys}


def behind(runs, tasks, g, keys=None):
    """every ungated family behind the gate g"""
    return {fam: agg_family(lst, tasks, g["fire"], keys) for fam, lst in runs.ungated().items()}


# ----------------------------------------------------------------------------- analyses
def ledger(ex):
    """target: the certified level of the union of the K stage tripwires, K * eps by a union bound over stages
    (each stage calibrated at eps, Theorem 3.7); eps for a single task"""
    eps = float(ex.tc["eps"])
    target = min(1.0, eps * ex.K)
    res = {}
    for name, region in (("deployed", False), ("twosided", True)):
        g = make_gate(ex, ex.cal_ref, {}, region=region)
        st = fire_stats(g)
        for p, v in st["pools"].items():
            v["p_target"] = binom_sf_ge(v["X"], v["n"], target) if v["n"] else None   # P[Bin(n, target) >= X]
        res[name] = {"eps": eps, "target": target, "K": ex.K, "beta": st["beta"], "cal_fire": g["cal_fire"],
                     "cert": g["cert"], "pools": st["pools"]}
    return res


def lattice(ex, runs, alpha):
    res = []
    comps = list(ex.cal_ref)
    for r in range(len(comps), 0, -1):
        for sub in itertools.combinations(comps, r):
            for region in (False, True):
                g = make_gate(ex, list(sub), {}, region=region, alpha=alpha)
                st = fire_stats(g)
                res.append({"comps": list(sub), "region": region, "deployed": g["deployed"], "beta": st["beta"],
                            "pools": st["pools"], "behind": behind(runs, ex.tasks, g, KEYS)})
    return res


def size(ex, seed=0):
    rng = np.random.default_rng(seed)
    res = []
    full = min(len(ex.X(f"cal__ref__{c}")) for c in ex.cal_ref)
    for n in [n for n in SIZES if n == 0 or n < full]:   # sizes every component has, and every prompt
        reps = []
        for _ in range(SIZE_REPS if n else 1):
            g = make_gate(ex, ex.cal_ref, {}, n_cal=n, rng=rng)
            st = fire_stats(g, REF_EVAL)
            reps.append({"cert": max(g["cert"].values()), "beta": st["beta"],
                         **{f"fire_{p}": st["pools"][p]["fire"] for p in REF_EVAL if p in st["pools"]}})
        keys = sorted({k for r in reps for k in r})
        res.append({"n": n, "reps": len(reps), **{k: sr.agg([r.get(k) for r in reps]) for k in keys},
                    **{f"p95_{k}": float(np.percentile([r[k] for r in reps if r.get(k) is not None], 95))
                       for k in keys if k.startswith("fire_")}})
    return res


def budget(ex, runs):
    if ex.K != 1 or "dolly" not in ex.cal_ref:
        return []
    res = []
    for e in DOLLY_EPS:
        lev = {c: SAFETY_EPS for c in ex.cal_ref}
        lev["dolly"] = e
        g = make_gate(ex, ex.cal_ref, lev, deployed=True)
        st = fire_stats(g)
        res.append({"dolly_eps": e, "safety_eps": SAFETY_EPS, "beta": st["beta"], "pools": st["pools"],
                    "behind": behind(runs, ex.tasks, g, KEYS)})
    return res


def stress(ex, runs, alpha, protect):
    eps = float(ex.tc["eps"])
    variants = ["ratio", "twosided"]
    if any(ex.has(f"wrap__{t}__fit") for t in ex.tasks):
        variants += ["remedy", "twosided_remedy"]
    if protect:
        variants.append("protect")
        if any(ex.has(f"wrap__{t}__fit") for t in ex.tasks):   # p3.8 (paper v3.4): the protect gate with its remedy
            variants.append("protect_remedy")
    pools = [p for p in ex.eval_keys() if p.startswith(ROBUST_PREFIX) or p == "wildchat"]
    keys = [f"{p}:refused" for p in pools] + [k for k in KEYS]
    res = {"pools": pools, "gates": {}, "own": {}, "behind": {}}
    # firing is also kept on the plain reference and off-reference pools (paper v3.4: the protect remedy's cost)
    fire_pools = pools + [p for p in REF_EVAL + OFF_EVAL if p in ex.eval_keys() and p not in pools]
    for v in variants:
        g = sr.build_variant(ex, v, eps, alpha, protect)
        res["gates"][v] = {"beta": float(np.mean([f.mean() for f in g["rl_any"].values()])),
                           "pools": {p: {"fire": float(g["any"][p].mean()), "X": int(g["any"][p].sum()),
                                         "n": len(g["any"][p]),
                                         "cp": clopper_pearson_upper(int(g["any"][p].sum()), len(g["any"][p]))}
                                     for p in fire_pools}}
        res["behind"][v] = behind(runs, ex.tasks, {"fire": g["any"]}, keys)
    for fam, lst in runs.fams.items():
        res["own"][fam] = agg_family(lst, ex.tasks, None, keys)
    return res


def methods(ex, runs):
    g = make_gate(ex, ex.cal_ref, {})
    return {"own": {fam: {"method": lst[0][1], "n": len(lst), **agg_family(lst, ex.tasks, None, KEYS)}
                    for fam, lst in runs.fams.items()},
            "behind_deployed": behind(runs, ex.tasks, g, KEYS), "beta_deployed": fire_stats(g)["beta"]}


# ----------------------------------------------------------------------------- output
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    return o


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def spts(v):
    return "--" if v is None else (f"{100 * v:+.1f}".replace("-", "$-$", 1))


def m0(a):
    return None if a is None else a[0]


def label_of(exp_res):
    """e.g. 'Concise-ctx, Qwen2.5-1.5B'"""
    model = exp_res["label"].rsplit(", ", 1)[-1].replace("-Instruct", "").replace("-Chat", "")
    return (exp_res["short"] + ", " + model).replace("_", r"\_")


def write_outputs(results, out):
    tdir = os.path.join(out, "v3_tables")
    os.makedirs(tdir, exist_ok=True)
    nums = {}

    # ledger: target, in-sample firing, held-out firing, bound and p-value, deployed tripwire
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{The certificate ledger of the deployed prompt-time tripwire. Target: the level $\varepsilon$ of "
         r"\cref{thm:calib}, a PAC guarantee before training for prompts drawn like the calibration prompts; for a "
         r"stream of $K$ stages, each calibrated at $\varepsilon$, the union is certified at $K\varepsilon$. Cal.: "
         r"in-sample firing on the calibration prompts. Held-out: firing on held-out prompts of the same pool, $X/n$, "
         r"its 95\% Clopper--Pearson bound and the one-sided binomial $p$-value of $X$ under a firing rate equal to the "
         r"target; a small $p$-value would show the certificate failing on that pool. Fresh Dolly: Dolly requests that no "
         r"pool of the experiment uses (rows 2000 to 11999 of the shuffled file), drawn like the calibration rows, in "
         r"experiments whose tasks are not built from them. MMLU is never calibrated on.}",
         r"\label{tab:ledger}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tabular}{l r rrrr rrr rrrr rr}", r"\toprule",
         r" & & \multicolumn{4}{c}{Dolly (\%)} & \multicolumn{3}{c}{fresh Dolly (\%)} & \multicolumn{4}{c}{AdvBench (\%)} & "
         r"\multicolumn{2}{c}{MMLU (\%)} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){10-13}\cmidrule(lr){14-15}",
         r"Experiment & Target & Cal. & Held-out & Bound & $p$ & Held-out & Bound & $p$ & Cal. & Held-out & Bound & $p$ & "
         r"Held-out & Bound \\",
         r"\midrule"]
    for name, r in results.items():
        led = r.get("ledger", {}).get("deployed")
        if not led:
            continue
        def cal(c):
            v = [x for k, x in led["cal_fire"].items() if k.endswith(":" + c)]
            return pct(max(v)) if v else "--"
        row = [label_of(r), f"{100 * led.get('target', led['eps']):g}"]
        for p in ("dolly", "dolly_fresh", "advbench"):
            s = led["pools"].get(p)
            held = ["--"] * 3 if not s else [f"{s['X']}/{s['n']}", pct(s["cp"]), f"{s['p_target']:.2f}"]
            row += held if p == "dolly_fresh" else [cal(p)] + held
        s = led["pools"].get("mmlu")
        row += ["--", "--"] if not s else [pct(s["fire"]), pct(s["cp"])]
        L.append(" & ".join(row) + r" \\")
        if led["pools"].get("dolly"):
            nums[f"vthree{name}LedgerDollyP"] = f"{led['pools']['dolly']['p_target']:.2f}"
        fr = led["pools"].get("dolly_fresh")
        if fr and fr.get("n"):
            nums[f"vthree{name}FreshDollyN"] = f"{fr['n']:,}".replace(",", "{,}")
            nums[f"vthree{name}FreshDollyFire"] = pct(fr["fire"])
            nums[f"vthree{name}FreshDollyCp"] = pct(fr["cp"])
            nums[f"vthree{name}FreshDollyP"] = f"{fr['p_target']:.2f}"
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(tdir, "ledger_table.tex"), "w").write("\n".join(L) + "\n")
    # the fresh Dolly check over every experiment that has it (main text, Section 6.2)
    fr = [(r["ledger"]["deployed"]["pools"]["dolly_fresh"], r["ledger"]["deployed"].get("K", 1)) for r in results.values()
          if r.get("ledger", {}).get("deployed", {}).get("pools", {}).get("dolly_fresh", {}).get("n")]
    for lab, sel in (("Single", [f for f, k in fr if k == 1]), ("Stream", [f for f, k in fr if k > 1])):
        if sel:
            nums[f"vthreeFresh{lab}Count"] = str(len(sel))
            nums[f"vthreeFresh{lab}N"] = f"{min(f['n'] for f in sel):,}".replace(",", "{,}")
            nums[f"vthreeFresh{lab}FireMin"] = pct(min(f["fire"] for f in sel))
            nums[f"vthreeFresh{lab}FireMax"] = pct(max(f["fire"] for f in sel))
            nums[f"vthreeFresh{lab}CpMin"] = pct(min(f["cp"] for f in sel))
            nums[f"vthreeFresh{lab}CpMax"] = pct(max(f["cp"] for f in sel))
            nums[f"vthreeFresh{lab}PMin"] = f"{min(f['p_target'] for f in sel):.2f}"

    # protect: Seq-GRPO as trained, behind the deployed tripwire and behind the protect tripwire (main text)
    P = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
         r"\caption{Protecting what must not change. Seq-GRPO as trained, behind the deployed tripwire (every "
         r"reference component, Dolly included) and behind a tripwire calibrated only on harmful requests and "
         r"task-like prompts, at $\varepsilon=0.03$; mean over seeds. $\beta$: firing on unlabeled new-task prompts, "
         r"estimated before training; changes in points; Dolly: share of Dolly responses changed (\%).}",
         r"\label{tab:protect}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tabular}{l l r r r r r r}", r"\toprule",
         r"Task & Deployed & $\beta\uparrow$ & Gain$\uparrow$ & Refusal & Over-ref. & MMLU & Dolly \\", r"\midrule"]
    for name, r in results.items():
        lat = r.get("lattice")
        if not lat or not r.get("protect"):
            continue
        seq_own = r.get("methods", {}).get("own", {}).get("seq")
        if not seq_own:
            continue
        full = next((x for x in lat if x["deployed"] and not x["region"]), None)
        prot = next((x for x in lat if sorted(x["comps"]) == sorted(r["protect"]) and not x["region"]), None)
        rows = [("as trained", None, seq_own)]
        if full:
            rows.append(("every component", full["beta"], full["behind"].get("seq", {})))
        if prot:
            rows.append(("protected only", prot["beta"], prot["behind"].get("seq", {})))
        for i, (lab, beta, m) in enumerate(rows):
            P.append(" & ".join([r["short"] if i == 0 else "", lab, "--" if beta is None else pct(beta),
                                 spts(m0(m.get("task:gain"))), spts(m0(m.get("advbench:refused"))),
                                 spts(m0(m.get("xstest:refused_safe"))), spts(m0(m.get("mmlu:correct"))),
                                 pct(m0(m.get("dolly:chg")))]) + r" \\")
        P.append(r"\midrule")
        if prot:
            nums[f"vthree{name}ProtectBeta"] = pct(prot["beta"])
            nums[f"vthree{name}ProtectGain"] = spts(m0(prot["behind"].get("seq", {}).get("task:gain")))
            nums[f"vthree{name}ProtectAdv"] = spts(m0(prot["behind"].get("seq", {}).get("advbench:refused")))
        nums[f"vthree{name}SeqGain"] = spts(m0(seq_own.get("task:gain")))
        nums[f"vthree{name}SeqAdv"] = spts(m0(seq_own.get("advbench:refused")))
    if P[-1] == r"\midrule":
        P.pop()
    P += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(tdir, "protect_table.tex"), "w").write("\n".join(P) + "\n")

    # lattice: every set of reference components (appendix, one table)
    names = {"dolly": "D", "advbench": "H", "general": "T"}
    T = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
         r"\caption{The cost of the reference. A tripwire fitted and calibrated at $\varepsilon=0.03$ on each set of "
         r"reference components (D: Dolly, H: harmful requests, T: task-like prompts; earlier tasks of a stream always "
         r"kept), without and with the region check. $\beta$: firing on unlabeled new-task prompts; firing (\%) on "
         r"held-out prompts of each pool, computed before training; Seq-GRPO behind the gate: gain and refusal change "
         r"(points). Pools that a set does not calibrate on carry no guarantee.}",
         r"\label{tab:lattice}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l l c r r r r r r r r}", r"\toprule",
         r"Experiment & Calibrated on & Region & $\beta\uparrow$ & Dolly & AdvBench & XSTest & MMLU & WildChat & "
         r"Gain$^{\mathrm{sw}}$ & Refusal$^{\mathrm{sw}}$ \\", r"\midrule"]
    for name, r in results.items():
        for i, x in enumerate(r.get("lattice", [])):
            s = x["behind"].get("seq", {})
            T.append(" & ".join([label_of(r) if i == 0 else "", "+".join(names.get(c, c) for c in x["comps"]),
                                 "yes" if x["region"] else "no", pct(x["beta"])]
                                + [pct(x["pools"].get(p, {}).get("fire")) for p in ("dolly", "advbench", "xstest", "mmlu", "wildchat")]
                                + [spts(m0(s.get("task:gain"))), spts(m0(s.get("advbench:refused")))]) + r" \\")
        if r.get("lattice"):
            T.append(r"\midrule")
    if T[-1] == r"\midrule":
        T.pop()
    T += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(tdir, "lattice_table.tex"), "w").write("\n".join(T) + "\n")

    # size: the certified level and the held-out firing against the number of calibration prompts per component
    for name, r in results.items():
        rows = ["n cert cert_sd beta beta_sd fire_dolly fire_dolly_sd p95_dolly"]
        for x in r.get("size", []):
            n = x["n"] or r.get("n_cal_full", 0)
            fd = x.get("fire_dolly")
            rows.append(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in
                                 [n, m0(x["cert"]), x["cert"][1], m0(x["beta"]), x["beta"][1],
                                  m0(fd) if fd else float("nan"), fd[1] if fd else float("nan"),
                                  x.get("p95_fire_dolly", float("nan"))]))
        if len(rows) > 1:
            open(os.path.join(tdir, f"size_{name.lower()}.dat"), "w").write("\n".join(rows) + "\n")

    # budget: one level per component (single tasks)
    B = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
         r"\caption{A level per reference component: harmful requests and task-like prompts at 0.01, Dolly at the level "
         r"shown or not protected, all with the deployed probe. $\beta$ before training; Seq-GRPO behind the gate, "
         r"changes in points; Dolly: share of Dolly responses changed (\%).}",
         r"\label{tab:budget}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
         r"\begin{tabular}{l r r r r r r}", r"\toprule",
         r"Task & Dolly level & $\beta\uparrow$ & Gain$\uparrow$ & Refusal & Over-ref. & Dolly \\", r"\midrule"]
    for name, r in results.items():
        for i, x in enumerate(r.get("budget", [])):
            s = x["behind"].get("seq", {})
            B.append(" & ".join([r["short"] if i == 0 else "", "none" if x["dolly_eps"] >= 1 else f"{x['dolly_eps']:g}",
                                 pct(x["beta"]), spts(m0(s.get("task:gain"))), spts(m0(s.get("advbench:refused"))),
                                 spts(m0(s.get("xstest:refused_safe"))), pct(m0(s.get("dolly:chg")))]) + r" \\")
        if r.get("budget"):
            B.append(r"\midrule")
    if B[-1] == r"\midrule":
        B.pop()
    B += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(tdir, "budget_table.tex"), "w").write("\n".join(B) + "\n")

    # stress: firing per family and gate, and refusal behind the deployed gate and as trained
    vnames = {"ratio": "dep.", "twosided": "two-s.", "remedy": "rem.", "twosided_remedy": "two-s.+rem.", "protect": "prot.",
              "protect_remedy": "prot.+rem."}
    S = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.6pt}",
         r"\caption{Harmful requests written to look like each task (\cref{app:stress}). Firing (\%) of each gate on the "
         r"250 held-out AdvBench requests in every form, computed before training: deployed tripwire (dep.), "
         r"two-sided (two-s.), with harmful requests in the task's own format added to the reference (rem.), both, and "
         r"calibrated only on the protected components (prot.), with harmful requests followed by a passage added "
         r"(prot.+rem.). Refusal change (points) of Seq-GRPO as trained and "
         r"behind the deployed gate, and of \method{}-prompt and safety replay as trained, where evaluated.}",
         r"\label{tab:stress}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l l r r r r r r r r r r}", r"\toprule",
         r"Experiment & Form & dep. & two-s. & rem. & two-s.+rem. & prot. & prot.+rem. & Seq & Seq$^{\mathrm{sw}}$ & "
         r"\method{}-prompt & Safety replay \\", r"\midrule"]
    for name, r in results.items():
        st = r.get("stress")
        if not st:
            continue
        first = True
        for p in st["pools"]:
            if not (p.startswith(("st_", "advwrap_")) or p.startswith("jb_")):
                continue
            g = st["gates"]
            own = st["own"]
            row = [label_of(r) if first else "", p.replace("_", r"\_")]
            row += [pct(g[v]["pools"][p]["fire"]) if v in g else "--" for v in vnames]
            row += [spts(m0(own.get("seq", {}).get(f"{p}:refused"))),
                    spts(m0(st["behind"].get("ratio", {}).get("seq", {}).get(f"{p}:refused"))),
                    spts(m0(own.get("tripwire_prompt", {}).get(f"{p}:refused"))),
                    spts(m0(own.get("safemix", {}).get(f"{p}:refused")))]
            S.append(" & ".join(row) + r" \\")
            first = False
        if not first:
            S.append(r"\midrule")
    if S[-1] == r"\midrule":
        S.pop()
    S += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(tdir, "stress_table.tex"), "w").write("\n".join(S) + "\n")

    # methods: every family as trained (new task and new baselines)
    for name, r in results.items():
        me = r.get("methods")
        if not me or not r.get("method_table"):
            continue
        M = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{" + label_of(r) + r": every method as trained, mean $\pm$ standard deviation over $n$ seeds. Gain "
             r"and changes in points; Dolly: share of Dolly responses changed (\%); Abst.: change in the share of Dolly "
             r"responses that say the request cannot be answered from the text. $^{\mathrm{sw}}$: an ungated run behind "
             r"the deployed tripwire ($\beta=" + pct(me["beta_deployed"]) + r"\%$).}",
             r"\label{tab:methods_" + name.lower() + "}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{l c r r r r r r r}", r"\toprule",
             r"Method & $n$ & Gain$\uparrow$ & Gain$^{\mathrm{sw}}\uparrow$ & Refusal & Over-ref. & MMLU & Dolly & Abst. \\",
             r"\midrule"]
        for fam in sorted(me["own"], key=sr.fam_key):
            o = me["own"][fam]
            sw = me["behind_deployed"].get(fam, {})
            M.append(" & ".join([sr.fam_label(fam, o["method"]), str(o["n"]), sr.fmt(o.get("task:gain"), True),
                                 sr.fmt(sw.get("task:gain"), True) if sw else "--",
                                 sr.fmt(o.get("advbench:refused"), True), sr.fmt(o.get("xstest:refused_safe"), True),
                                 sr.fmt(o.get("mmlu:correct"), True), sr.fmt(o.get("dolly:chg")),
                                 sr.fmt(o.get("dolly:abstain"), True)]) + r" \\")
            mn = "".join(w.capitalize() for w in fam.split("_"))
            nums[f"vthree{name}{mn}Gain"] = spts(m0(o.get("task:gain")))
            nums[f"vthree{name}{mn}Adv"] = spts(m0(o.get("advbench:refused")))
            nums[f"vthree{name}{mn}Xs"] = spts(m0(o.get("xstest:refused_safe")))
            nums[f"vthree{name}{mn}Mmlu"] = spts(m0(o.get("mmlu:correct")))
            nums[f"vthree{name}{mn}Chg"] = pct(m0(o.get("dolly:chg")))
            if sw:
                nums[f"vthree{name}{mn}SwGain"] = spts(m0(sw.get("task:gain")))
        M += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        open(os.path.join(tdir, f"methods_{name.lower()}.tex"), "w").write("\n".join(M) + "\n")

    with open(os.path.join(out, "v3_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/v3_analyses.py\n")
        for k in sorted(nums):
            fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")


SHORT = {"nq": "NQ", "ifc": "IF", "concise": "Concise", "concisectx": "Concise-ctx", "squad2": "SQuAD 2.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True, help="<config>:<Name>[:<protected components>]")
    ap.add_argument("--out", default="paper/v3_results")
    ap.add_argument("--runs-root", default="", help="folder holding <exp>/ (default: the config's out_dir)")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--only", default="ledger,lattice,size,budget,stress,methods")
    ap.add_argument("--method-table", default="", help="names whose per-method table is written (e.g. Squad)")
    ap.add_argument("--merge", action="store_true",
                    help="compute nothing: read <out>/<exp>/v3.json of every --exp and write the tables (run each "
                         "experiment in its own process first, so that only one is in memory at a time)")
    args = ap.parse_args()
    want = set(args.only.split(","))
    mt = {n for n in args.method_table.split(",") if n}
    results = {}
    for spec in args.exp:
        parts = spec.split(":")
        cfg_path, name = parts[0], parts[1]
        protect = [c for c in (parts[2].split(",") if len(parts) > 2 else []) if c]
        if not name.isalpha():
            raise SystemExit("Name must be letters only")
        cfg = load_config(cfg_path)
        if args.merge:
            jp = os.path.join(args.out, cfg["exp"], "v3.json")
            if os.path.exists(jp):
                results[name] = json.load(open(jp))
                results[name]["method_table"] = name in mt
            else:
                print(f"{cfg['exp']}: no {jp}; left out")
            continue
        root = os.path.join(args.runs_root or cfg["out_dir"], cfg["exp"])
        feat = os.path.join(root, "analysis", "gate_features.npz")
        if not os.path.exists(feat):
            print(f"{cfg['exp']}: no gate features ({feat}); left out")
            continue
        ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
        runs = Runs(root, ex.K)
        tasks = ex.tasks
        res = {"exp": cfg["exp"], "label": sr.readable_label(cfg),
               "short": SHORT.get(tasks[0], tasks[0]) if len(tasks) == 1 else f"{len(tasks)} tasks",
               "protect": protect, "families": sorted(runs.fams), "method_table": name in mt,
               "n_cal_full": int(min(len(ex.X(f"cal__ref__{c}")) for c in ex.cal_ref))}
        print(f"== {cfg['exp']}: {len(runs.fams)} run families, protect {protect or 'none'}", flush=True)
        if "ledger" in want:
            res["ledger"] = ledger(ex)
        if "lattice" in want:
            res["lattice"] = lattice(ex, runs, args.alpha)
        if "size" in want:
            res["size"] = size(ex)
        if "budget" in want:
            res["budget"] = budget(ex, runs)
        if "stress" in want:
            res["stress"] = stress(ex, runs, args.alpha, protect)
        if "methods" in want:
            res["methods"] = methods(ex, runs)
        os.makedirs(os.path.join(args.out, cfg["exp"]), exist_ok=True)
        with open(os.path.join(args.out, cfg["exp"], "v3.json"), "w") as fh:
            json.dump(jsonable(res), fh, indent=1)
        results[name] = jsonable(res)
        del ex, runs, res
        gc.collect()
    write_outputs(results, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
