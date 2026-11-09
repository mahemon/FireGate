#!/usr/bin/env python
"""Step p3.14 (Nov 2, PLAN_PAPER3.md): the linear probe against the optimal Neyman-Pearson test, on synthetic
features whose likelihood ratio is known (CPU, a few minutes; no model, no stored features).

    python scripts/np_gap.py --out paper/v3_results/p3_14 [--repeats 20] [--dims 32,256] [--n-new 100,300,1000,3000]

Theorem 4.1 prices protection by the power beta*(eps) of the best size-eps test of the reference against the new
task, and the paper estimates that price from below with the gate it trains. Here the price is known exactly. In
whitened coordinates z every distribution is Gaussian with a diagonal covariance (features x = L z, Sigma = L L^T with
eigenvalues proportional to 1/k and a random rotation, so the probe sees correlated features as in a trunk):
  shift    reference N(0, I), new task N(2.5 u, I)                  the likelihood ratio is linear: a probe can reach it
  spread   reference N(0, I), new task N(1.5 u, A), A = 3 on a quarter of the coordinates, 1 elsewhere: quadratic
  overlap  reference = three components, weights 600 : 100 : 900 (as Dolly, AdvBench, general): N(0, I),
           N(4 u1, 0.5 I), N(3 u2, I); new task N(3 u2 + 1.5 u3, I), close to the third component (task-like prompts)
Per cell (scenario, dimension d, number n of new-task fitting prompts; reference fitting prompts 1.6 n split by the
weights; calibration prompts 700, 170, 700 per component, or 1570 for one component), repeated --repeats times:
  probe    the gate of the paper: class-balanced logistic regression on standardized features (switch_report.fit_probe,
           L2 1e-4, 200 L-BFGS steps), PAC-calibrated per component with Theorem 3.6 (twcore.certify.calibrate,
           eps 0.03, delta 0.05 split over components)
  probe*   the same score at the threshold whose true firing on the worst component is exactly eps (the calibration
           slack is probe* minus probe)
  plug-in  the Gaussian likelihood ratio with estimated means and shrunk covariances per component (the right model
           class, estimated), calibrated as the probe
  NP       the optimal test under the same constraint (every reference component fires at most eps): the likelihood
           ratio of the new task to a weighted mixture of components, weights by Lagrangian iteration (one component:
           the likelihood ratio itself); NP-mix: the same with the constraint on the mixture only (Theorem 4.1's
           beta*, an upper bound on NP)
True firing and power are Monte Carlo estimates on --mc fresh draws per distribution. Outputs in --out: np_gap.json,
np_gap_table.tex, np_gap_summary.txt (with the predictions of PLAN_PAPER3.md, written before the run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import norm as gauss  # noqa: E402

from switch_report import fit_probe  # noqa: E402
from twcore.certify import calibrate  # noqa: E402

EPS, DELTA, L2, ITERS = 0.03, 0.05, 1e-4, 200
WEIGHTS3, CAL3 = (600, 100, 900), (700, 170, 700)


# ----------------------------------------------------------------------------- the synthetic world
class World:
    """Gaussians with diagonal covariance in whitened coordinates z; features x = z @ L^T."""

    def __init__(self, scenario, d, seed):
        r = np.random.default_rng(seed)
        lam = 1.0 / np.arange(1, d + 1)
        lam *= d / lam.sum()
        Q, _ = np.linalg.qr(r.standard_normal((d, d)))
        self.L = Q * np.sqrt(lam)        # x = L z
        U, _ = np.linalg.qr(r.standard_normal((d, 3)))
        u1, u2, u3 = U.T
        one = np.ones(d)
        if scenario == "shift":
            self.ref = [(np.zeros(d), one)]
            self.new = (2.5 * u1, one)
            self.w = np.array([1.0])
            self.cal = (1570,)
        elif scenario == "spread":
            a = one.copy()
            a[r.permutation(d)[: d // 4]] = 3.0
            self.ref = [(np.zeros(d), one)]
            self.new = (1.5 * u1, a)
            self.w = np.array([1.0])
            self.cal = (1570,)
        elif scenario == "overlap":
            self.ref = [(np.zeros(d), one), (4.0 * u1, 0.5 * one), (3.0 * u2, one)]
            self.new = (3.0 * u2 + 1.5 * u3, one)
            self.w = np.array(WEIGHTS3, float) / sum(WEIGHTS3)
            self.cal = CAL3
        else:
            raise ValueError(scenario)

    @staticmethod
    def _logpdf(z, m, v):
        return -0.5 * (((z - m) ** 2 / v).sum(1) + np.log(v).sum())

    def draw(self, r, comp, n):
        m, v = self.new if comp == "new" else self.ref[comp]
        z = m + np.sqrt(v) * r.standard_normal((n, len(m)))
        return z, z @ self.L.T

    def log_new(self, z):
        return self._logpdf(z, *self.new)

    def log_comps(self, z):
        return np.stack([self._logpdf(z, m, v) for m, v in self.ref], 1)


def np_power(world, Zref, Znew, eps, iters=400):
    """Optimal power with every component's firing at most eps (Lagrangian weights), and with the mixture's only."""
    ln = [world.log_new(z) for z in Zref]
    lc = [world.log_comps(z) for z in Zref]
    ln_new, lc_new = world.log_new(Znew), world.log_comps(Znew)
    lw = np.log(world.w)
    # mixture constraint: threshold the ratio to the mixture at its (1 - eps) quantile under the mixture
    s_ref = np.concatenate([ln[c] - logsumexp(lc[c] + lw, 1) for c in range(len(Zref))])
    wts = np.concatenate([np.full(len(Zref[c]), world.w[c] / len(Zref[c])) for c in range(len(Zref))])
    o = np.argsort(-s_ref)
    t_mix = s_ref[o][np.searchsorted(np.cumsum(wts[o]), eps)]
    beta_mix = float(np.mean(ln_new - logsumexp(lc_new + lw, 1) > t_mix))
    if len(Zref) == 1:
        t = np.quantile(ln[0] - lc[0][:, 0], 1 - eps)
        return float(np.mean(ln_new - lc_new[:, 0] > t)), beta_mix, [1.0]
    # per-component constraints: psi = 1[log p_new - log sum_c lam_c p_c > 0]; multiplicative updates of lam
    loglam = np.log(world.w) + float(t_mix)
    for it in range(iters):
        fire = np.array([np.mean(ln[c] - logsumexp(lc[c] + loglam, 1) > 0) for c in range(len(Zref))])
        step = 2.0 / (1 + it / 20)
        loglam = np.clip(loglam + step * (fire - eps) / eps, -40, 40)
    fire = np.array([np.mean(ln[c] - logsumexp(lc[c] + loglam, 1) > 0) for c in range(len(Zref))])
    # make the result feasible: raise the threshold until no component exceeds eps
    s = [ln[c] - logsumexp(lc[c] + loglam, 1) for c in range(len(Zref))]
    t = max(0.0, max(float(np.quantile(sc, 1 - eps)) for sc in s))
    beta = float(np.mean(ln_new - logsumexp(lc_new + loglam, 1) > t))
    return beta, beta_mix, np.exp(loglam - logsumexp(loglam)).tolist()


# ----------------------------------------------------------------------------- the estimated tests
class PlugIn:
    """Gaussian likelihood ratio with estimated means and covariances (standardized space, shrunk to the diagonal)."""

    def __init__(self, Xnew, Xcomps, w, gamma=0.1):
        X = np.concatenate([Xnew] + list(Xcomps))
        self.mu, self.sd = X.mean(0), X.std(0, ddof=1) + 1e-6
        self.parts = [self._fit(self._z(Xnew), gamma)] + [self._fit(self._z(Xc), gamma) for Xc in Xcomps]
        self.lw = np.log(np.asarray(w, float))

    def _z(self, X):
        return (X - self.mu) / self.sd

    @staticmethod
    def _fit(Z, gamma):
        m = Z.mean(0)
        C = np.cov((Z - m).T)
        C = (1 - gamma) * C + gamma * np.diag(np.diag(C)) + 1e-6 * np.eye(len(m))
        sign, logdet = np.linalg.slogdet(C)
        return m, np.linalg.inv(C), logdet

    @staticmethod
    def _lp(Z, part):
        m, P, logdet = part
        D = Z - m
        return -0.5 * (((D @ P) * D).sum(1) + logdet)

    def score(self, X):
        Z = self._z(X)
        lc = np.stack([self._lp(Z, p) for p in self.parts[1:]], 1)
        return self._lp(Z, self.parts[0]) - logsumexp(lc + self.lw, 1)


def pac_fire(score_fn, Xcal, Xmc_ref, Xmc_new):
    cal = calibrate({str(c): score_fn(X).tolist() for c, X in enumerate(Xcal)}, EPS, DELTA)
    k = cal["kappa"]
    fires = [float(np.mean(score_fn(X) > k)) for X in Xmc_ref]
    return float(np.mean(score_fn(Xmc_new) > k)), max(fires)


def oracle_fire(score_fn, Xmc_ref, Xmc_new):
    k = max(float(np.quantile(score_fn(X), 1 - EPS)) for X in Xmc_ref)
    return float(np.mean(score_fn(Xmc_new) > k))


def cell(scenario, d, n_new, repeats, mc, seed):
    world = World(scenario, d, seed)
    r = np.random.default_rng(seed + 1)
    Zmc = [world.draw(r, c, mc) for c in range(len(world.ref))]
    Znew = world.draw(r, "new", mc)
    beta_np, beta_mix, lam = np_power(world, [z for z, _ in Zmc], Znew[0], EPS)
    Xmc_ref, Xmc_new = [x for _, x in Zmc], Znew[1]
    out = {"scenario": scenario, "d": d, "n_new": n_new, "np": beta_np, "np_mix": beta_mix, "np_weights": lam,
           "probe": [], "probe_fire": [], "probe_oracle": [], "plugin": [], "plugin_fire": []}
    n_ref = [max(1, int(round(1.6 * n_new * w))) for w in world.w]
    for rep in range(repeats):
        rr = np.random.default_rng(10_000 * seed + rep)
        Xn = world.draw(rr, "new", n_new)[1]
        Xr = [world.draw(rr, c, n_ref[c])[1] for c in range(len(world.ref))]
        Xcal = [world.draw(rr, c, world.cal[c])[1] for c in range(len(world.ref))]
        pr = fit_probe(Xn, np.concatenate(Xr), L2, ITERS)
        b, f = pac_fire(pr.score, Xcal, Xmc_ref, Xmc_new)
        out["probe"].append(b)
        out["probe_fire"].append(f)
        out["probe_oracle"].append(oracle_fire(pr.score, Xmc_ref, Xmc_new))
        pl = PlugIn(Xn, Xr, world.w)
        b, f = pac_fire(pl.score, Xcal, Xmc_ref, Xmc_new)
        out["plugin"].append(b)
        out["plugin_fire"].append(f)
    if scenario == "shift":
        out["np_analytic"] = float(gauss.cdf(2.5 - gauss.ppf(1 - EPS)))
    return out


def stats(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def check(res, repeats):
    """the predictions of PLAN_PAPER3.md, step p3.14 (written before the first run)"""
    L = ["pre-registered predictions (PLAN_PAPER3.md, step p3.14)"]
    bad = [(c["scenario"], c["d"], c["n_new"], m) for c in res for m in ("probe", "plugin")
           if sum(f > EPS for f in c[m + "_fire"]) > 2]
    L.append(f"P1 the PAC-calibrated probe and plug-in fire on more than eps of some component in at most 2 of "
             f"{repeats} repeats, in every cell: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    g = lambda c, m: c["np"] - stats(c[m])[0]  # noqa: E731
    sh = {(c["d"], c["n_new"]): g(c, "probe") for c in res if c["scenario"] == "shift"}
    p2 = [(k, v) for k, v in sh.items() if (k[0] <= 32 and k[1] >= 1000) or (k[0] > 32 and k[1] >= 3000)]
    L.append(f"P2 shift: the probe is within 5 points of NP for d = 32 at n >= 1000 and within 10 points for d = 256 "
             f"at n = 3000: {'PASS' if p2 and all(v <= (0.05 if k[0] <= 32 else 0.10) for k, v in p2) else 'FAIL' if p2 else 'n/a'}  "
             + ", ".join(f"d={k[0]} n={k[1]}: {100 * v:.1f}" for k, v in sorted(p2)))
    sp = [c for c in res if c["scenario"] == "spread"]
    a = [(c["d"], c["n_new"], g(c, "probe")) for c in sp]
    b = [(c["d"], c["n_new"], g(c, "plugin")) for c in sp if c["d"] <= 32 and c["n_new"] >= 3000]
    L.append(f"P3a spread: the linear probe stays more than 5 points below NP at every n: "
             f"{'PASS' if a and all(v > 0.05 for _, _, v in a) else 'FAIL' if a else 'n/a'}  "
             + ", ".join(f"d={x} n={y}: {100 * v:.1f}" for x, y, v in a))
    L.append(f"P3b spread: the plug-in comes within 5 points of NP at d = 32, n = 3000: "
             f"{'PASS' if b and all(v <= 0.05 for _, _, v in b) else 'FAIL' if b else 'n/a'}  "
             + ", ".join(f"d={x} n={y}: {100 * v:.1f}" for x, y, v in b))
    ov = [(c["d"], c["n_new"], g(c, "probe")) for c in res if c["scenario"] == "overlap" and c["d"] <= 32
          and c["n_new"] >= 1000]
    L.append(f"P4 overlap: the probe is within 5 points of NP at d = 32, n >= 1000: "
             f"{'PASS' if ov and all(v <= 0.05 for _, _, v in ov) else 'FAIL' if ov else 'n/a'}  "
             + ", ".join(f"d={x} n={y}: {100 * v:.1f}" for x, y, v in ov))
    return "\n".join(L) + "\n"


def write_table(res, out):
    pc = lambda v: f"{100 * v:.1f}"  # noqa: E731
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{The trained probe against the optimal test where the likelihood ratio is known (synthetic "
         r"Gaussian features, step p3.14). Power (\%) on the new task at $\varepsilon=0.03$, $\delta=0.05$; NP: the "
         r"optimal test with every reference component firing at most $\varepsilon$ ($\beta^\ast$ of "
         r"\cref{thm:limit} for one component); probe: the gate of the paper, PAC-calibrated; probe$^\ast$: the same "
         r"score at the threshold with true firing $\varepsilon$; plug-in: the estimated Gaussian likelihood ratio, "
         r"PAC-calibrated; Fire: the probe's largest true firing on a reference component (\%), mean over repeats. "
         r"$d$: feature dimension; $n$: new-task fitting prompts.}",
         r"\label{tab:npgap}", r"\begin{tabular}{l r r r r r r r}", r"\toprule",
         r"Scenario & $d$ & $n$ & NP & Probe & Probe$^\ast$ & Plug-in & Fire \\", r"\midrule"]
    prev = None
    for c in res:
        if prev is not None and c["scenario"] != prev:
            L.append(r"\midrule")
        prev = c["scenario"]
        L.append(f"{c['scenario']} & {c['d']} & {c['n_new']} & {pc(c['np'])} & {pc(stats(c['probe'])[0])} & "
                 f"{pc(stats(c['probe_oracle'])[0])} & {pc(stats(c['plugin'])[0])} & {pc(stats(c['probe_fire'])[0])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "np_gap_table.tex"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_14"))
    ap.add_argument("--scenarios", default="shift,spread,overlap")
    ap.add_argument("--dims", default="32,256")
    ap.add_argument("--n-new", default="100,300,1000,3000")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--mc", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0, res = time.time(), []
    for si, sc in enumerate(args.scenarios.split(",")):
        for d in [int(x) for x in args.dims.split(",")]:
            for n in [int(x) for x in args.n_new.split(",")]:
                c = cell(sc, d, n, args.repeats, args.mc, args.seed + 100 * si + d)
                res.append(c)
                print(f"{sc:8s} d={d:4d} n={n:5d}  NP {100 * c['np']:5.1f} (mixture {100 * c['np_mix']:5.1f}"
                      + (f", analytic {100 * c['np_analytic']:5.1f}" if "np_analytic" in c else "") + ")  "
                      f"probe {100 * stats(c['probe'])[0]:5.1f} +- {100 * stats(c['probe'])[1]:.1f}  "
                      f"probe* {100 * stats(c['probe_oracle'])[0]:5.1f}  plug-in {100 * stats(c['plugin'])[0]:5.1f}  "
                      f"fire {100 * stats(c['probe_fire'])[0]:.2f} (max {100 * max(c['probe_fire']):.2f})  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    json.dump({"eps": EPS, "delta": DELTA, "repeats": args.repeats, "mc": args.mc, "cells": res},
              open(os.path.join(args.out, "np_gap.json"), "w"), indent=1)
    write_table(res, args.out)
    txt = check(res, args.repeats)
    open(os.path.join(args.out, "np_gap_summary.txt"), "w").write(txt)
    print(txt + f"wrote {args.out}/np_gap.json, np_gap_table.tex, np_gap_summary.txt")


if __name__ == "__main__":
    main()
