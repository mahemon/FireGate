#!/usr/bin/env python
"""Certified prompt-level switches built from stored features and outputs (CPU only: numpy + scipy).

    python scripts/switch_report.py --config configs/v2/overlap_q15.yaml --name Nq [--eps 0.03,0.1,0.3]
    python scripts/switch_report.py --config configs/v2/ladder_concise_q15.yaml --name Concise --protect advbench,general

A prompt-time gate decides from the prompt alone, so any trained model can be put behind it: a prompt on which the
gate fires is answered by the trained model, every other prompt by the base model, and the drift on any prompt
distribution is at most the gate's firing rate there, whatever the trained model does. With greedy decoding both
answers are stored (stage K and stage 0 of the run, same GPU model), so the switched evaluation is exact and needs no
GPU. Needs <runs>/analysis/gate_features.npz (scripts/gate_features.py).

Gate variants (per stage; the stream's gate fires if any stage fires):
  ratio      the deployed tripwire's probe (refitted when no tripwire_prompt run exists), PAC-calibrated at each eps
  twosided   ratio AND the prompt lies in the new task's region: shrunk Mahalanobis distance to the task's fitting
             prompts below the (1 - alpha) quantile of its RL prompts (unlabeled); calibrated on the conjunction
  router     the probe's natural threshold (logit > 0), not calibrated
  *_union    (streams) one budget for the whole stream: stage k is calibrated so that "any stage <= k fires" stays
             within eps * k / K on every reference component; earlier-task components keep eps per stage
  remedy     (tasks with a format of their own: the stream tasks, ifc) probe refitted with AdvBench requests in the
             tasks' formats as extra reference prompts, and those requests (calibration split) as an extra
             calibrated component; its cost is the drop in firing on the new task's prompts
  twosided_remedy  both: the remedy's probe AND the new task's region (in the remedy probe's space); calibrated on
             the conjunction with the remedy's components
  protect    (--protect c1,c2) probe and calibration use only these reference components (what the user protects)
Runs: every finished run under <runs>/s<seed>/. Ungated runs (seq, kl, refkl, nullspace, scale, ...) are evaluated
as they are and behind every gate variant; gated runs (tripwire_prompt, router, tripwire) as they are, and for
streams also with the union thresholds (tightening only: prompts that lose some but not all firings are counted).
Judge labels (scripts/judge_changes.py) are used when present.
Outputs in --out: switch_summary.json, switch_table.tex, gates_table.tex, switch_numbers.tex (macros \\sw<name>...),
switch_frontier_<family>.dat, switch_cost.dat.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402

GATED = ("tripwire", "tripwire_prompt", "router")
LETTERS = "ABCDEFGH"
REF_POOLS = ("dolly", "advbench", "xstest", "mmlu")
SUBSTANTIVE = ("DIFFERENT",)
MATERIAL = ("STYLE", "DIFFERENT")


# ----------------------------------------------------------------------------- small helpers
def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def agg(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return [m, sd, len(v)]


def read_rows(path):
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    out.sort(key=lambda r: r["i"])
    return out


def pct(v):
    return "--" if v is None else f"{100 * v:.1f}"


def spts(v):
    return "--" if v is None else (f"{100 * v:+.1f}".replace("-", "$-$", 1))


def fmt(a, signed=False):
    if a is None:
        return "--"
    core = spts(a[0]) if signed else pct(a[0])
    return core if a[2] < 2 else f"{core}{{\\scriptsize$\\pm${100 * a[1]:.1f}}}"


# ----------------------------------------------------------------------------- probes and supports
class Probe:
    def __init__(self, mu, sigma, a, b):
        self.mu, self.sigma, self.a, self.b = (np.asarray(mu, np.float64), np.asarray(sigma, np.float64),
                                               np.asarray(a, np.float64), float(np.asarray(b).reshape(-1)[0]))

    def z(self, X):
        return (np.asarray(X, np.float64) - self.mu) / self.sigma

    def score(self, X):
        return self.z(X) @ self.a + self.b


def fit_probe(pos, neg, l2, iters):
    """Class-balanced logistic regression on standardized features (twcore.probe.fit_logreg in numpy)."""
    X = np.concatenate([pos, neg]).astype(np.float64)
    mu = X.mean(0)
    sigma = X.std(0, ddof=1) + 1e-6
    Z = (X - mu) / sigma
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([np.full(len(pos), 0.5 / len(pos)), np.full(len(neg), 0.5 / len(neg))])

    def f(theta):
        a, b = theta[:-1], theta[-1]
        z = Z @ a + b
        loss = float((w * (np.logaddexp(0.0, z) - y * z)).sum() + l2 * (a @ a))
        g = w * (1.0 / (1.0 + np.exp(-z)) - y)
        return loss, np.concatenate([Z.T @ g + 2 * l2 * a, [g.sum()]])

    res = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": int(iters)})
    return Probe(mu, sigma, res.x[:-1], res.x[-1])


class Support:
    """Region of the new task: shrunk Mahalanobis distance (in the probe's standardized space) to its fitting
    prompts; tau = (1 - alpha) quantile of the distances of unlabeled RL prompts of the task."""

    def __init__(self, probe, Xfit, Xrl, alpha, gamma=0.1):
        Z = probe.z(Xfit)
        self.probe, self.m = probe, Z.mean(0)
        C = np.cov((Z - self.m).T)
        d = C.shape[0]
        C = (1 - gamma) * C + gamma * (np.trace(C) / d) * np.eye(d)
        self.P = np.linalg.inv(C)
        self.tau = float(np.quantile(self.dist(Xrl), 1.0 - alpha))

    def dist(self, X):
        D = self.probe.z(X) - self.m
        return ((D @ self.P) * D).sum(-1)

    def inside(self, X):
        return self.dist(X) <= self.tau


TASK_LABELS = {"nq": "NQ", "ifc": "IF", "concise": "Concise", "concisectx": "Concise on context requests", "gsm8k": "GSM8K",
               "countdown": "Countdown", "kk": "K\\&K",
               "math": "MATH", "sudoku": "Sudoku", "code": "Code", "trivia": "TriviaQA"}


def readable_label(cfg):
    """caption label of an experiment, e.g. 'NQ, Qwen2.5-1.5B-Instruct' or '5-task stream, Qwen2.5-1.5B-Instruct'"""
    tasks = list(cfg["stream"])
    what = TASK_LABELS.get(tasks[0], tasks[0]) if len(tasks) == 1 else f"{len(tasks)}-task stream"
    return f"{what}, {os.path.basename(str(cfg['model_path']).rstrip('/'))}"


# ----------------------------------------------------------------------------- gates
class Experiment:
    def __init__(self, cfg, npz_path, json_path):
        self.cfg = cfg
        self.z = np.load(npz_path)
        self.meta = json.load(open(json_path))
        self.tasks = list(cfg["stream"])
        self.K = len(self.tasks)
        self.tc = cfg["tripwire"]
        self.fit_ref = list(self.tc["fit_ref"])
        self.cal_ref = list(self.tc["cal_ref"])
        self._cache = {}

    def has(self, key):
        return key in self.z.files

    def X(self, key):
        if ("X", key) not in self._cache:
            self._cache[("X", key)] = self.z[key].astype(np.float32)
        return self._cache[("X", key)]

    def deployed(self, k):
        if not self.has(f"probe__{k}__a"):
            return None
        g = lambda n: self.z[f"probe__{k}__{n}"]  # noqa: E731
        return Probe(g("mu"), g("sigma"), g("a"), g("b")), float(g("kappa")[0])

    def eval_keys(self):
        return sorted(k[len("eval__"):] for k in self.z.files if k.startswith("eval__"))

    def stage_data(self, k, base, protect):
        """new-task fitting prompts, reference / earlier-task fitting prompts, calibration components, and the
        names of the reference components (the ones a stream budget is shared over)."""
        task, prev = self.tasks[k - 1], self.tasks[:k - 1]
        only = protect if base in ("protect", "protect_remedy") else None
        refs_fit = [c for c in self.fit_ref if only is None or c in only]
        refs_cal = [c for c in self.cal_ref if only is None or c in only]
        neg = [self.X(f"fit__ref__{c}") for c in refs_fit] + [self.X(f"fit__old__{t}") for t in prev]
        cal = {c: self.X(f"cal__ref__{c}") for c in refs_cal}
        if probe_kind(base) in ("remedy", "protect_remedy"):
            neg += [self.X(f"wrap__{t}__fit") for t in self.tasks if self.has(f"wrap__{t}__fit")]
            wraps = [self.X(f"wrap__{t}__cal") for t in self.tasks if self.has(f"wrap__{t}__cal")]
            if wraps:
                cal["advwrap"] = np.concatenate(wraps)
                refs_cal = refs_cal + ["advwrap"]
        cal.update({f"old_{t}": self.X(f"cal__old__{t}") for t in prev})
        return task, self.X(f"fit__new__{task}"), np.concatenate(neg), cal, refs_cal

    def probe(self, k, base, protect, pos, neg):
        """The deployed probe for the deployed tripwire's variants (refitted if no run exists), a refit otherwise."""
        key = ("probe", k, probe_kind(base))
        if key not in self._cache:
            dep = self.deployed(k) if key[2] == "deployed" else None
            self._cache[key] = (dep[0], True) if dep is not None else \
                (fit_probe(pos, neg, float(self.tc["l2"]), int(self.tc["lbfgs_iters"])), False)
        return self._cache[key]

    def support(self, k, probe_key, probe, pos, task, alpha):
        key = ("support", k, probe_key, alpha)
        if key not in self._cache:
            self._cache[key] = Support(probe, pos, self.X(f"rl__{task}"), alpha)
        return self._cache[key]


def probe_kind(base):
    """which probe a gate variant uses: the deployed one, the remedy's refit or the protect refit"""
    if base in ("remedy", "twosided_remedy"):
        return "remedy"
    return base if base in ("protect", "protect_remedy") else "deployed"   # protect_remedy: p3.8, paper v3.4


class StageGate:
    def __init__(self, probe, support, kappa):
        self.probe, self.support, self.kappa = probe, support, kappa

    def score(self, X):
        s = self.probe.score(X)
        if self.support is not None:
            s = np.where(self.support.inside(X), s, -np.inf)
        return s

    def fires(self, X):
        return self.score(X) > self.kappa


def build_variant(ex: Experiment, variant: str, eps: float, alpha: float, protect=None):
    """-> {"kappa", "tau", "eps_cert", "deployed", "fire": {k: {pool: bool[]}}, "any": {pool: bool[]},
    "rl_any": {task: bool[]}} (rl_any: the stream's gate on unlabeled RL prompts of each task)"""
    delta = float(ex.tc["delta"])
    base = variant.replace("_union", "")
    union = variant.endswith("_union")
    out = {"kappa": {}, "tau": {}, "eps_cert": {}, "deployed": {}, "fire": {}}
    stages, fired_cal = [], {}
    for k in range(1, ex.K + 1):
        task, pos, neg, cal, refs_cal = ex.stage_data(k, base, protect)
        probe, is_dep = ex.probe(k, base, protect, pos, neg)
        sup = ex.support(k, probe_kind(base), probe, pos, task, alpha) \
            if base in ("twosided", "twosided_remedy") else None
        gate = StageGate(probe, sup, 0.0)
        cs = {c: gate.score(X) for c, X in cal.items()}
        if base == "router":
            kappa, certs = 0.0, {c: None for c in cs}
        else:
            n_all = len(cs)
            refs = {c: v for c, v in cs.items() if c in refs_cal}
            olds = {c: v for c, v in cs.items() if c not in refs_cal}
            e_ref = eps * k / ex.K if union else eps
            if union:
                refs = {c: np.where(fired_cal.get(c, np.zeros(len(v), bool)), np.inf, v) for c, v in refs.items()}
            kappa, certs = -np.inf, {}
            for part, e in ((refs, e_ref), (olds, eps)):
                if not part:
                    continue
                cal_res = calibrate({c: v.tolist() for c, v in part.items()}, e, delta * len(part) / n_all)
                kappa = max(kappa, float(cal_res["kappa"]))
                certs.update({c: v["eps_cert"] for c, v in cal_res["components"].items()})
        gate.kappa = float(kappa)
        if union:
            for c in refs_cal:
                fired_cal[c] = fired_cal.get(c, np.zeros(len(cs[c]), bool)) | (cs[c] > gate.kappa)
        stages.append((task, gate))
        out["kappa"][k] = gate.kappa
        out["tau"][k] = None if sup is None else sup.tau
        out["eps_cert"][k] = certs
        out["deployed"][k] = is_dep
        out["fire"][k] = {p: gate.fires(ex.X(f"eval__{p}")) for p in ex.eval_keys()}
    out["any"] = {p: np.any([out["fire"][k][p] for k in out["fire"]], axis=0) for p in ex.eval_keys()}
    out["rl_any"] = {t: np.any([g.fires(ex.X(f"rl__{t}")) for _, g in stages], axis=0) for t in ex.tasks}
    return out


def gate_stats(ex: Experiment, g):
    """Firing rate and 95% Clopper-Pearson bound on every evaluation pool, and the firing on unlabeled RL prompts
    of each task (the activation, known before training)."""
    st = {"fire": {}, "cp": {}, "power_pre": {}, "eps_cert": g["eps_cert"], "kappa": g["kappa"], "tau": g["tau"],
          "deployed": g["deployed"]}
    for p, f in g["any"].items():
        X = int(f.sum())
        st["fire"][p] = X / len(f) if len(f) else None
        st["cp"][p] = clopper_pearson_upper(X, len(f)) if len(f) else None
    for t, f in g["rl_any"].items():
        st["power_pre"][t] = float(f.mean()) if len(f) else None
    return st


# ----------------------------------------------------------------------------- runs and metrics
def load_run(md, K):
    s0, sK = {}, {}
    for st, dest in ((0, s0), (K, sK)):
        edir = os.path.join(md, f"stage{st}", "eval")
        if not os.path.exists(os.path.join(edir, "DONE")):
            return None
        for fn in os.listdir(edir):
            if fn.endswith(".jsonl"):
                dest[fn[:-6]] = read_rows(os.path.join(edir, fn))
    for what in ("advwrap", "shift", "wildchat", "stress"):   # evaluations of scripts/eval_extra.py, when present
        d = os.path.join(md, "extra", what)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.startswith("final_") and fn.endswith(".jsonl"):
                    name = fn[len("final_"):-6]
                    b = read_rows(os.path.join(d, f"base_{name}.jsonl"))
                    if b is not None:
                        sK[name], s0[name] = read_rows(os.path.join(d, fn)), b
    return s0, sK


def numeric_metrics(rows):
    keys = [k for k, v in rows[0]["m"].items() if isinstance(v, (int, float, bool))] if rows else []
    return keys


def metrics(R, B, tasks, labels=None, fire=None):
    """R, B: {pool: rows}; fire: {pool: bool[]} for the switched version (None: R as it is)."""
    out = {}
    for p in sorted(R):
        if p not in B or R[p] is None or B[p] is None or len(R[p]) != len(B[p]):
            continue
        r, b = R[p], B[p]
        if fire is not None:
            f = fire.get(p)
            if f is None or len(f) != len(r):
                continue
            r = [ri if fi else bi for ri, bi, fi in zip(r, b, f)]
        chg = [ri["out"] != bi["out"] for ri, bi in zip(r, b)]
        out[f"{p}:chg"] = mean([float(c) for c in chg])
        for key in numeric_metrics(r):
            out[f"{p}:{key}"] = mean([float(ri["m"][key]) for ri in r]) - mean([float(bi["m"][key]) for bi in b])
            out[f"{p}:{key}_abs"] = mean([float(ri["m"][key]) for ri in r])
        if p == "xstest":
            for lab in ("safe", "unsafe"):
                sel = [(ri, bi) for ri, bi in zip(r, b) if ri.get("label") == lab]
                if sel:
                    out[f"{p}:refused_{lab}"] = (mean([float(ri["m"]["refused"]) for ri, _ in sel])
                                                 - mean([float(bi["m"]["refused"]) for _, bi in sel]))
        lab = (labels or {}).get(p)
        if lab is not None:
            ls = [lab.get(str(ri["i"])) if c else "SAME" for ri, c in zip(r, chg)]
            known = [x for x, c in zip(ls, chg) if c and x is not None]
            out[f"{p}:judged"] = (len(known) / sum(chg)) if sum(chg) else 1.0
            out[f"{p}:subst"] = mean([float(x in SUBSTANTIVE) for x in ls if x is not None])
            out[f"{p}:material"] = mean([float(x in MATERIAL) for x in ls if x is not None])
    accs = [out.get(f"{t}:correct_abs") for t in tasks]
    gains = [out.get(f"{t}:correct") for t in tasks]
    if all(a is not None for a in accs):
        out["task:acc"] = mean(accs)
        out["task:gain"] = mean(gains)
    return out


def union_tightened(R, B, ex, g):
    """Gated stream run with its per-stage thresholds raised to the union thresholds: rows and counts."""
    out, unresolved, spurious = {}, 0, 0
    for p in R:
        if p not in B or not R[p] or len(R[p]) != len(B[p]):
            continue
        rows = []
        for ri, bi in zip(R[p], B[p]):
            old = {t for t, v in ri.get("fire", {}).items() if v is not None}
            new = {t for t, s in ri.get("maxs", {}).items() if int(t) in g["kappa"] and s > g["kappa"][int(t)]}
            spurious += len(new - old)
            new &= old
            if new == old:
                rows.append(ri)
            elif not new:
                rows.append(bi)
            else:
                unresolved += 1
                rows.append(ri)
        out[p] = rows
    return out, unresolved, spurious


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="", help="letters only: macro prefix \\sw<name>...")
    ap.add_argument("--eps", default="0.03,0.1,0.3")
    ap.add_argument("--main-eps", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.02, help="share of new-task prompts outside the support check")
    ap.add_argument("--protect", default="", help="reference components to protect (variant 'protect')")
    args = ap.parse_args()
    if args.name and not args.name.isalpha():
        raise SystemExit("--name must be letters only")
    cfg = load_config(args.config, args.set)
    runs = args.runs or os.path.join(cfg["out_dir"], cfg["exp"])
    feat = args.features or os.path.join(runs, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        raise SystemExit(f"{feat} is missing: run scripts/gate_features.py (bash slurm/plan.sh v3.1) first")
    out_dir = args.out or os.path.join("paper", "v2_results", cfg["exp"], "switch")
    os.makedirs(out_dir, exist_ok=True)
    ex = Experiment(cfg, feat, feat[:-4] + ".json")
    eps_list = [float(e) for e in args.eps.split(",") if e.strip()]
    main_eps = args.main_eps if args.main_eps is not None else float(cfg["tripwire"]["eps"])
    if main_eps not in eps_list:
        eps_list.append(main_eps)
    protect = [c.strip() for c in args.protect.split(",") if c.strip()]

    variants = ["ratio", "twosided"]
    if ex.K > 1:
        variants += ["ratio_union", "twosided_union"]
    if any(ex.has(f"wrap__{t}__fit") for t in ex.tasks):   # tasks with a format of their own: stream tasks, ifc
        variants += ["remedy", "twosided_remedy"]
    if protect:
        variants.append("protect")
        if any(ex.has(f"wrap__{t}__fit") for t in ex.tasks):   # p3.8: the protect gate with its remedy
            variants.append("protect_remedy")
    gates = {}
    for v in variants:
        for e in eps_list:
            gates[(v, e)] = build_variant(ex, v, e, args.alpha, protect)
    gates[("router", None)] = build_variant(ex, "router", 0.0, args.alpha)
    stats = {f"{v}|{'' if e is None else e}": gate_stats(ex, g) for (v, e), g in gates.items()}
    print(f"{cfg['exp']}: {len(gates)} gates; probes {'deployed' if ex.meta.get('probes') else 'refitted'}")

    # every finished run, every seed
    seeds = sorted((d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:]))
    per = {}   # family -> list of per-seed results
    for sd in seeds:
        for fam in sorted(os.listdir(os.path.join(runs, sd))):
            md = os.path.join(runs, sd, fam)
            cj = os.path.join(md, "config.json")
            if not (os.path.isdir(md) and os.path.exists(os.path.join(md, "DONE")) and os.path.exists(cj)):
                continue
            info = json.load(open(cj))
            loaded = load_run(md, ex.K)
            if loaded is None:
                continue
            B, R = loaded
            jp = os.path.join(runs, "analysis", "judge", f"{sd}__{fam}.json")
            labels = json.load(open(jp)).get("labels") if os.path.exists(jp) else None
            method = info.get("method")
            res = {"seed": sd, "method": method, "own": metrics(R, B, ex.tasks, labels), "switch": {}}
            if method not in GATED:
                for (v, e), g in gates.items():
                    res["switch"][f"{v}|{'' if e is None else e}"] = metrics(R, B, ex.tasks, labels, g["any"])
            elif ex.K > 1 and method == "tripwire_prompt":
                for e in eps_list:
                    if ("ratio_union", e) in gates and abs(e - float(info["config"]["tripwire"]["eps"])) < 1e-9:
                        rows, unres, spur = union_tightened(R, B, ex, gates[("ratio_union", e)])
                        m = metrics(rows, B, ex.tasks, labels)
                        m["unresolved"], m["spurious"] = unres, spur
                        res["switch"][f"ratio_union|{e}"] = m
            per.setdefault(fam, []).append(res)
    if not per:
        print("no finished runs under", runs)

    # aggregate over seeds
    summ = {"exp": cfg["exp"], "label": readable_label(cfg), "tasks": ex.tasks, "eps": eps_list, "main_eps": main_eps, "alpha": args.alpha,
            "gates": stats, "families": {}}
    for fam, lst in per.items():
        keys_own = sorted({k for r in lst for k in r["own"]})
        f = {"method": lst[0]["method"], "n": len(lst), "own": {k: agg([r["own"].get(k) for r in lst]) for k in keys_own},
             "switch": {}}
        for sw in sorted({s for r in lst for s in r["switch"]}):
            ks = sorted({k for r in lst for k in r["switch"].get(sw, {})})
            f["switch"][sw] = {k: agg([r["switch"].get(sw, {}).get(k) for r in lst]) for k in ks}
        summ["families"][fam] = f
    with open(os.path.join(out_dir, "switch_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    write_tables(summ, out_dir, args.name, ex)
    print("wrote", out_dir)


# ----------------------------------------------------------------------------- tables
ORDER = ["seq", "scale", "kl", "refkl", "nullspace", "ewc", "mas", "l2", "agem", "olora", "replay", "safemix", "router",
         "tripwire_prompt", "tripwire"]
NAMES = {"seq": "Seq-GRPO", "scale": "Seq, adapter $\\times\\alpha$", "kl": "+ KL", "refkl": "+ Ref-KL",
         "nullspace": "+ Null-space", "ewc": "+ EWC", "mas": "+ MAS", "l2": "+ L2", "agem": "+ A-GEM",
         "olora": "+ O-LoRA", "replay": "+ Replay", "safemix": "+ Safety replay", "router": "Router (uncal.)",
         "tripwire_prompt": "\\method{}-prompt",
         "tripwire": "\\method{}-anytime"}
VNAMES = {"ratio": "probe (deployed tripwire)", "twosided": "probe + new-task region", "router": "probe, logit $>0$",
          "ratio_union": "probe, one stream budget", "twosided_union": "probe + region, one stream budget",
          "remedy": "probe, + harmful requests in task formats", "protect": "probe, protected components only",
          "protect_remedy": "probe, protected components + harmful requests in task formats",
          "twosided_remedy": "probe + region, + harmful requests in task formats"}


def fam_key(fam):
    for i, base in enumerate(ORDER):
        if fam == base or fam.startswith(base + "_"):
            return (i, fam)
    return (len(ORDER), fam)


def fam_label(fam, method):
    base = next((b for b in ORDER if fam == b or fam.startswith(b + "_")), fam)
    rest = fam[len(base) + 1:] if fam != base else ""
    if base in ("tripwire_prompt", "router") and (rest == "seq" or rest.startswith("seq_")):
        rest = "whole sequence" + rest[3:].replace("_", ", ")
    name = NAMES.get(base, fam.replace("_", r"\_"))
    return name + (f" ({rest.replace('_', ' ')})" if rest else "")


def write_tables(summ, out_dir, name, ex):
    tasks = summ["tasks"]
    stream = len(tasks) > 1
    gain_key = "task:gain" if stream else f"{tasks[0]}:correct"
    e = summ["main_eps"]
    sw_main = f"ratio|{e}"
    fams = sorted(summ["families"], key=fam_key)
    judged = any(k.endswith(":subst") for f in summ["families"].values() for k in f["own"])
    # main table: every method as trained and behind the deployed tripwire's switch at the main eps
    cols = ["Method", "$n$", "Gain", "Gain$^{\\mathrm{sw}}$", "Chg.", "Chg.$^{\\mathrm{sw}}$",
            "Refusal$^{\\mathrm{sw}}$", "Over-ref.$^{\\mathrm{sw}}$", "MMLU$^{\\mathrm{sw}}$"]
    if judged:
        cols += ["Subst.", "Subst.$^{\\mathrm{sw}}$"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Every trained model behind the certified switch (" + summ.get("label", summ["exp"]).replace("_", r"\_") +
             r"). A prompt on which the deployed prompt-time tripwire ($\varepsilon=" + f"{e:g}" + r"$) fires is answered by "
             r"the trained model, every other prompt by the base model; $^{\mathrm{sw}}$: with the switch. Gain: "
             + ("average accuracy over the stream's tasks" if stream else "new-task accuracy") + r" (points); Chg.: share "
             r"of Dolly responses that differ from the base model's; Refusal, Over-ref., MMLU: change in AdvBench refusal, "
             r"XSTest over-refusal and MMLU accuracy (points)" + (r"; Subst.: share of Dolly responses that the judge "
             r"rates as substantively different" if judged else "") + r". Mean $\pm$ standard deviation over seeds. "
             r"Gated rows are shown as trained.}",
             r"\label{tab:switch_" + (name.lower() or "x") + "}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l c " + "r" * (len(cols) - 2) + "}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for fam in fams:
        f = summ["families"][fam]
        o, s = f["own"], f["switch"].get(sw_main, {})
        gated = f["method"] in GATED
        row = [fam_label(fam, f["method"]), str(f["n"]), fmt(o.get(gain_key), True), "--" if gated else fmt(s.get(gain_key), True),
               fmt(o.get("dolly:chg")), "--" if gated else fmt(s.get("dolly:chg")),
               fmt((o if gated else s).get("advbench:refused"), True), fmt((o if gated else s).get("xstest:refused_safe"), True),
               fmt((o if gated else s).get("mmlu:correct"), True)]
        if judged:
            row += [fmt(o.get("dolly:subst")), "--" if gated else fmt(s.get("dolly:subst"))]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    with open(os.path.join(out_dir, "switch_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # gates: firing (known before training from unlabeled prompts) on every evaluation pool
    pools = [p for p in (tasks + ["dolly", "advbench", "xstest", "mmlu"]) if p in next(iter(summ["gates"].values()))["fire"]]
    # WildChat (Nov 1, PLAN_NOV.md) is reported apart (scripts/wildchat_report.py), so these tables stay as they were
    # paper v3: the stress families (st_*) and the fresh Dolly check (dolly_fresh) are reported apart too
    # (scripts/v3_analyses.py)
    extra = [p for p in next(iter(summ["gates"].values()))["fire"]
             if p not in pools and p not in ("wildchat", "dolly_fresh") and not p.startswith("st_")]
    gl = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
          r"\caption{Gate variants (" + summ.get("label", summ["exp"]).replace("_", r"\_") + r"): firing rate (\%) on held-out "
          r"prompts of every pool, computed from unlabeled prompts before any training; it bounds the drift of any "
          r"trained model behind the gate. Power: firing on unlabeled RL prompts of the new task(s). Region check: "
          r"$\alpha=" + f"{summ['alpha']:g}" + r"$.}",
          r"\label{tab:gates_" + (name.lower() or "x") + "}", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{l r r " + "r" * (len(pools) + len(extra)) + "}", r"\toprule",
          " & ".join(["Gate", r"$\varepsilon$", "Power"] + [p.replace("_", r"\_") for p in pools + extra]) + r" \\",
          r"\midrule"]
    for key, st in summ["gates"].items():
        v, ev = key.split("|")
        power = mean(list(st["power_pre"].values()))
        gl.append(" & ".join([VNAMES.get(v, v), ev or "--", pct(power)] + [pct(st["fire"].get(p)) for p in pools + extra])
                  + r" \\")
    gl += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    with open(os.path.join(out_dir, "gates_table.tex"), "w") as fh:
        fh.write("\n".join(gl) + "\n")

    # frontier of each ungated family behind the switch (eps sweep) and the predicted-vs-actual cost
    cost = ["family eps predicted actual own"]
    for fam in fams:
        f = summ["families"][fam]
        if f["method"] in GATED:
            continue
        pts = ["chg gain chg_sd gain_sd eps subst subst_sd"]   # subst: judged substantive change (nan: not judged)
        og = f["own"].get(gain_key)
        for e2 in summ["eps"]:
            s = f["switch"].get(f"ratio|{e2}", {})
            if s.get(gain_key) is None or s.get("dolly:chg") is None:
                continue
            sb = s.get("dolly:subst")
            pts.append(f"{100 * s['dolly:chg'][0]:.2f} {100 * s[gain_key][0]:.2f} {100 * s['dolly:chg'][1]:.2f} "
                       f"{100 * s[gain_key][1]:.2f} {e2:g} " + (f"{100 * sb[0]:.2f} {100 * sb[1]:.2f}" if sb else "nan nan"))
            pw = summ["gates"][f"ratio|{e2}"]["power_pre"]
            pred = None
            if og is not None:
                if stream:
                    per_t = [(pw.get(t), f["own"].get(f"{t}:correct")) for t in tasks]
                    if all(a is not None and b is not None for a, b in per_t):
                        pred = mean([a * b[0] for a, b in per_t])
                elif pw.get(tasks[0]) is not None:
                    pred = pw[tasks[0]] * og[0]
            if pred is not None:
                cost.append(f"{fam} {e2:g} {100 * pred:.2f} {100 * s[gain_key][0]:.2f} {100 * og[0]:.2f}")
        with open(os.path.join(out_dir, f"switch_frontier_{fam}.dat"), "w") as fh:
            fh.write("\n".join(pts) + "\n")
    with open(os.path.join(out_dir, "switch_cost.dat"), "w") as fh:
        fh.write("\n".join(cost) + "\n")

    # macros \sw<name><Family><Field>[<eps letter>]: seq and refkl as trained and behind the switch; gate firing
    nums = {}
    pre = "sw" + name
    letters = {e2: LETTERS[i] for i, e2 in enumerate(sorted(summ["eps"]))}
    for fam, mn in (("seq", "Seq"), ("refkl", "Refkl"), ("kl", "Kl"), ("nullspace", "Null")):
        f = summ["families"].get(fam)
        if f is None:
            continue
        og, oc = f["own"].get(gain_key), f["own"].get("dolly:chg")
        nums[f"{pre}{mn}Gain"] = spts(og[0]) if og else "--"
        nums[f"{pre}{mn}Chg"] = pct(oc[0]) if oc else "--"
        for e2, L in letters.items():
            s = f["switch"].get(f"ratio|{e2}", {})
            nums[f"{pre}{mn}SwGain{L}"] = spts(s[gain_key][0]) if s.get(gain_key) else "--"
            nums[f"{pre}{mn}SwChg{L}"] = pct(s["dolly:chg"][0]) if s.get("dolly:chg") else "--"
            nums[f"{pre}{mn}SwAdv{L}"] = spts(s["advbench:refused"][0]) if s.get("advbench:refused") else "--"
    for key, st in summ["gates"].items():
        v, ev = key.split("|")
        vn = "".join(w.capitalize() for w in v.split("_"))
        L = letters.get(float(ev), "") if ev else ""
        nums[f"{pre}Power{vn}{L}"] = pct(mean(list(st["power_pre"].values())))
        for p in ("dolly", "advbench", "xstest", "mmlu"):
            if p in st["fire"]:
                nums[f"{pre}Fire{vn}{p.capitalize()}{L}"] = pct(st["fire"][p])
    for e2, L in letters.items():
        nums[f"{pre}Eps{L}"] = f"{e2:g}"
    with open(os.path.join(out_dir, "switch_numbers.tex"), "w") as fh:
        fh.write("% generated by scripts/switch_report.py\n")
        for k in sorted(nums):
            fh.write(f"\\newcommand{{\\{k}}}{{{nums[k]}}}\n")


if __name__ == "__main__":
    main()
