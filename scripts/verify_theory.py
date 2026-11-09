#!/usr/bin/env python
"""Numerical checks of every theorem and proposition of the paper on small synthetic problems (numpy + scipy, CPU,
about a minute). Exit code 0 means every check passed.

    python scripts/verify_theory.py [--quick]

Exact checks enumerate every response of small autoregressive models (vocabulary 4, 5 response tokens): the
coupling certificate (Thm 3.1) with its equality P_new(F) = phi and the shared-randomness identity, tightness
(Prop 3.2), the stream certificate and sequential composition (Thm 3.3, Cor 3.4). Monte Carlo checks: Clopper-Pearson
coverage (Prop 3.5), PAC calibration with a Bonferroni split (Thm 3.6), the audit estimator (Prop 5.1), the
anytime-valid monitor's false-alarm rate and detection time (Prop 5.2), repair (Prop 5.3). Linear programs: the
Neyman-Pearson limit for every policy and its KL form (Thm 4.1), prompt-only optimality (Prop 4.3), and the loss of
an estimated tripwire (Prop 4.4).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys

import numpy as np
from scipy import optimize, stats

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


# --------------------------------------------------------------------------------- small autoregressive models
class Table:
    """A random function of the token prefix (memoized): next-token distributions or firing decisions."""

    def __init__(self, rng, kind, V=4, temp=1.0, q=0.2):
        self.rng, self.kind, self.V, self.temp, self.q, self.memo = rng, kind, V, temp, q, {}

    def __call__(self, prefix):
        if prefix not in self.memo:
            if self.kind == "dist":
                z = self.rng.normal(size=self.V) / self.temp
                p = np.exp(z - z.max())
                self.memo[prefix] = p / p.sum()
            else:
                self.memo[prefix] = bool(self.rng.random() < self.q)
        return self.memo[prefix]


class Gated:
    """pi_new = old base plus adapters; adapter i acts from the first position where its tripwire fired (latch)."""

    def __init__(self, base, stages):
        self.base, self.stages = base, stages   # stages: list of (tripwire, adapter distribution table)

    def step(self, prefix, latched):
        """next-token distribution and the updated latches (the decision at prefix acts on the next token)"""
        latched = tuple(l or g(prefix) for l, (g, _) in zip(latched, self.stages))
        p = self.base(prefix)
        for l, (_, a) in zip(latched, self.stages):
            if l:
                p = a(prefix)   # the latest active adapter determines the head output
        return p, latched


def enumerate_law(model, x, N, V):
    """{response: (probability, latches at the end)} by exhaustive enumeration."""
    out = {}

    def rec(prefix, prob, latched):
        if len(prefix) - 1 == N:
            out[prefix[1:]] = (prob, latched)
            return
        p, latched2 = model.step(prefix, latched)
        for v in range(V):
            if p[v] > 0:
                rec(prefix + (v,), prob * p[v], latched2)

    rec((x,), 1.0, tuple(False for _ in model.stages))
    return out


def fired_on(stages, seq, upto):
    """whether each tripwire fires on a prefix of seq at a position that influences the response (Def. 3.1)"""
    x, y = seq
    s = (x,) + tuple(y)
    return [any(g(s[:t]) for t in range(1, len(s))) for g, _ in stages[:upto]]


def decode(model, x, U, V):
    """inverse-CDF decoding with shared uniforms U (one per position)"""
    prefix, latched = (x,), tuple(False for _ in model.stages)
    for u in U:
        p, latched = model.step(prefix, latched)
        prefix = prefix + (int(min(np.searchsorted(np.cumsum(p), u, side="right"), V - 1)),)
    return prefix[1:], latched


def coupling_checks(rng, M=6, N=5, V=4, trials=6):
    worst = 0.0
    for trial in range(trials):
        D = rng.dirichlet(np.ones(M))
        base = Table(rng, "dist", V)
        g1, a1 = Table(rng, "fire", q=rng.uniform(0.02, 0.3)), Table(rng, "dist", V, temp=0.3)
        old, new = Gated(base, []), Gated(base, [(g1, a1)])
        tv = phi = pnewF = 0.0
        for x in range(M):
            lo, ln = enumerate_law(old, x, N, V), enumerate_law(new, x, N, V)
            tv += D[x] * 0.5 * sum(abs(ln.get(y, (0, ()))[0] - lo.get(y, (0, ()))[0]) for y in set(lo) | set(ln))
            phi += D[x] * sum(p for y, (p, _) in lo.items() if fired_on([(g1, a1)], (x, y), 1)[0])
            pnewF += D[x] * sum(p for y, (p, _) in ln.items() if fired_on([(g1, a1)], (x, y), 1)[0])
        worst = max(worst, tv - phi)
        if trial == 0:
            check(abs(pnewF - phi) < 1e-12, f"Thm 3.1: P_new(F) = phi exactly ({pnewF:.6f} vs {phi:.6f})")
    check(worst <= 1e-12, f"Thm 3.1: TV(new, old) <= phi on {trials} random models (max TV - phi = {worst:.2e})")
    # shared randomness: every non-fired response is identical
    viol = nonfired = 0
    for _ in range(2000):
        x = int(rng.integers(M))
        U = rng.random(N)
        yo, _ = decode(old, x, U, V)
        yn, latched = decode(new, x, U, V)
        f = fired_on([(g1, a1)], (x, yo), 1)[0]
        f2 = fired_on([(g1, a1)], (x, yn), 1)[0]
        if f != f2:
            viol += 1
        if not f:
            nonfired += 1
            viol += int(yo != yn)
    check(viol == 0, f"Thm 3.1: shared uniforms keep all {nonfired} non-fired responses and the firing record (violations {viol})")
    # Prop 3.2 tightness: an adapter that puts all mass on token 0
    pt = Table(rng, "dist", V)
    point = lambda prefix: np.eye(V)[0]   # noqa: E731
    g = Table(rng, "fire", q=0.25)
    old, new = Gated(pt, []), Gated(pt, [(g, point)])
    tv = phi = 0.0
    eta = 0.0
    for x in range(M):
        lo, ln = enumerate_law(old, x, N, V), enumerate_law(new, x, N, V)
        tv += (1 / M) * 0.5 * sum(abs(ln.get(y, (0, ()))[0] - lo.get(y, (0, ()))[0]) for y in set(lo) | set(ln))
        phi += (1 / M) * sum(p for y, (p, _) in lo.items() if fired_on([(g, point)], (x, y), 1)[0])
    eta = max(pt(k)[0] for k in pt.memo)
    check(tv >= (1 - eta) * phi - 1e-12, f"Prop 3.2: TV >= (1 - eta) phi ({tv:.4f} >= {(1 - eta) * phi:.4f}, eta {eta:.3f})")


def stream_checks(rng, M=5, N=4, V=4, trials=6):
    worst_union = worst_comp = 0.0
    for _ in range(trials):
        D = rng.dirichlet(np.ones(M))
        base = Table(rng, "dist", V)
        st = [(Table(rng, "fire", q=rng.uniform(0.02, 0.25)), Table(rng, "dist", V, temp=0.3)) for _ in range(2)]
        p0, p1, p2 = Gated(base, []), Gated(base, st[:1]), Gated(base, st)
        tv = union = phi1 = phi2_on_p1 = 0.0
        for x in range(M):
            l0, l1, l2 = (enumerate_law(m, x, N, V) for m in (p0, p1, p2))
            tv += D[x] * 0.5 * sum(abs(l2.get(y, (0, ()))[0] - l0.get(y, (0, ()))[0]) for y in set(l0) | set(l2))
            for y, (p, _) in l0.items():
                f = fired_on(st, (x, y), 2)
                union += D[x] * p * float(f[0] or f[1])
                phi1 += D[x] * p * float(f[0])
            for y, (p, _) in l1.items():
                phi2_on_p1 += D[x] * p * float(fired_on(st, (x, y), 2)[1])
        worst_union = max(worst_union, tv - union)
        worst_comp = max(worst_comp, tv - (phi1 + phi2_on_p1))
    check(worst_union <= 1e-12, f"Thm 3.3: TV(pi_2, pi_0) <= P_pi0(F1 u F2) (max excess {worst_union:.2e})")
    check(worst_comp <= 1e-12, f"Cor 3.4: TV(pi_2, pi_0) <= P_pi0(F1) + P_pi1(F2), stage-wise calibration "
          f"(max excess {worst_comp:.2e})")


# --------------------------------------------------------------------------------- calibration
def cp_upper(x, n, delta):
    return 1.0 if x >= n else float(stats.beta.ppf(1 - delta, x + 1, n - x))


def calib_index(n, eps, delta):
    """smallest k with P[Bin(n, 1 - eps) >= k] <= delta (None if none <= n)"""
    for k in range(1, n + 1):
        if stats.binom.sf(k - 1, n, 1 - eps) <= delta:
            return k
    return None


def calibration_checks(rng, reps):
    for p in (0.01, 0.03, 0.1):
        n, delta = 400, 0.05
        X = rng.binomial(n, p, size=reps)
        U = np.array([cp_upper(int(x), n, delta) for x in X])
        cov = float(np.mean(p <= U))
        check(cov >= 1 - delta - 3 * math.sqrt(delta * (1 - delta) / reps),
              f"Prop 3.5: Clopper-Pearson coverage at p={p}: {cov:.3f} >= {1 - delta}")
    # PAC calibration, C components with different score distributions, Bonferroni split
    eps, delta, C = 0.03, 0.05, 3
    ns = (700, 170, 700)
    dists = (stats.norm(0, 1), stats.t(3), stats.expon())
    ks = [calib_index(n, eps, delta / C) for n in ns]
    viol_joint = 0
    viol = [0] * C
    for _ in range(reps):
        kap = max(np.sort(d.rvs(n, random_state=rng))[k - 1] for d, n, k in zip(dists, ns, ks))
        bad = [d.sf(kap) > eps for d in dists]
        viol_joint += any(bad)
        for c in range(C):
            viol[c] += bad[c]
    rate = viol_joint / reps
    check(rate <= delta + 3 * math.sqrt(delta * (1 - delta) / reps),
          f"Thm 3.6: P(some component fires above eps) = {rate:.4f} <= delta {delta} (per component "
          + ", ".join(f"{v / reps:.4f}" for v in viol) + f"; delta/C = {delta / C:.4f})")
    # component calibrated alone hits delta/C closely (the order statistic is not conservative by much)
    n, k = 700, calib_index(700, eps, delta / C)
    single = np.mean([stats.norm.sf(np.sort(rng.normal(size=n))[k - 1]) > eps for _ in range(reps)])
    check(single <= delta / C + 3 * math.sqrt(delta / C / reps), f"Thm 3.6: one component alone: {single:.4f} <= {delta / C:.4f}")
    # Prop 5.3 repair: raising the threshold never increases firing; recalibrating on a shifted distribution
    base_k = calib_index(700, eps, delta / C)
    old_kap = np.sort(rng.normal(size=700))[base_k - 1]
    shifted = stats.norm(1.0, 1.0)
    n2, k2 = 300, calib_index(300, eps, 0.05)
    ok, mono = 0, True
    for _ in range(reps):
        new_kap = max(old_kap, np.sort(shifted.rvs(n2, random_state=rng))[k2 - 1])
        mono &= stats.norm.sf(new_kap) <= stats.norm.sf(old_kap) + 1e-15
        ok += shifted.sf(new_kap) <= eps
    check(mono and ok / reps >= 0.95 - 3 * math.sqrt(0.05 * 0.95 / reps),
          f"Prop 5.3: repair keeps old certificates (monotone {mono}) and certifies the new traffic in {ok / reps:.3f} >= 0.95")


# --------------------------------------------------------------------------------- the limit
def np_power(pref, pnew, eps):
    """power of the size-eps Neyman-Pearson test (randomized) of pref against pnew on a finite space"""
    lam = np.where(pref > 0, pnew / np.maximum(pref, 1e-300), np.inf)
    order = np.argsort(-lam, kind="stable")
    size = power = 0.0
    for i in order:
        if size + pref[i] <= eps + 1e-15:
            size += pref[i]
            power += pnew[i]
        else:
            frac = (eps - size) / pref[i]
            power += frac * pnew[i]
            break
    return power


def lp_max_change(pref, pnew, eps):
    """max E_new delta over delta(x) in [0, 1] with E_ref delta <= eps (any policy's per-prompt TV is such a delta)"""
    res = optimize.linprog(-pnew, A_ub=pref[None], b_ub=[eps], bounds=[(0, 1)] * len(pref), method="highs")
    return -res.fun


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def limit_checks(rng, trials=200):
    worst_np = worst_kl = 0.0
    for _ in range(trials):
        M = int(rng.integers(3, 12))
        pref, pnew = rng.dirichlet(np.ones(M) * 0.7), rng.dirichlet(np.ones(M) * 0.7)
        eps = float(rng.choice([0.01, 0.03, 0.1, 0.3]))
        b = np_power(pref, pnew, eps)
        worst_np = max(worst_np, abs(lp_max_change(pref, pnew, eps) - b))
        worst_kl = max(worst_kl, b - (kl(pnew, pref) + math.log(2)) / math.log(1 / eps))
    check(worst_np < 1e-7, f"Thm 4.1: the largest new-task change at reference drift eps is the Neyman-Pearson power "
          f"(LP vs NP, max difference {worst_np:.1e})")
    check(worst_kl <= 1e-12, f"Thm 4.1: beta*(eps) <= (KL + log 2) / log(1/eps) (max excess {worst_kl:.2e})")
    # a concrete policy: the per-prompt TV of a random policy pair never beats the limit
    viol = 0
    for _ in range(trials):
        M, V = 8, 3
        pref, pnew = rng.dirichlet(np.ones(M)), rng.dirichlet(np.ones(M))
        po, pn = rng.dirichlet(np.ones(V), size=M), rng.dirichlet(np.ones(V), size=M)
        mix = rng.random(M)[:, None]
        pn = mix * pn + (1 - mix) * po
        delta = 0.5 * np.abs(pn - po).sum(1)
        viol += float(pnew @ delta) > np_power(pref, pnew, float(pref @ delta)) + 1e-12
    check(viol == 0, f"Thm 4.1: {trials} random policies: E_new delta <= beta*(E_ref delta) (violations {viol})")
    # Remark: the KL form is below one only if KL < log(1 / (2 eps))
    e = 0.03
    check(abs(math.log(1 / (2 * e)) - 2.8134) < 1e-3, f"Remark 4.2: threshold log(1/(2 eps)) at eps=0.03 is {math.log(1 / (2 * e)):.3f} nats")


def noevidence_checks(rng, trials=50):
    worst = 0.0
    for _ in range(trials):
        M, V = 6, 4
        pref, pnew = rng.dirichlet(np.ones(M)), rng.dirichlet(np.ones(M))
        pi = rng.dirichlet(np.ones(V), size=M)
        jr, jn = (pref[:, None] * pi).ravel(), (pnew[:, None] * pi).ravel()
        for eps in (0.03, 0.1, 0.3):
            worst = max(worst, np_power(jr, jn, eps) - np_power(pref, pnew, eps))
    check(worst <= 1e-12, f"Prop 4.3: tests on prompt-response pairs have no more power than on prompts (max excess {worst:.2e})")


def estimated_checks(rng, trials=300):
    worst = -1.0
    for _ in range(trials):
        M = int(rng.integers(4, 15))
        pref, pnew = rng.dirichlet(np.ones(M)), rng.dirichlet(np.ones(M))
        lam_true = pnew / pref
        rhat = np.abs(lam_true * np.exp(rng.normal(0, rng.uniform(0, 1.5), size=M)))
        eps = float(rng.choice([0.03, 0.1, 0.3]))
        # randomized threshold test on rhat with size exactly eps
        order = np.argsort(-rhat, kind="stable")
        phi = np.zeros(M)
        size = 0.0
        for i in order:
            if size + pref[i] <= eps:
                phi[i] = 1.0
                size += pref[i]
            else:
                phi[i] = (eps - size) / pref[i]
                break
        gap = np_power(pref, pnew, eps) - float(pnew @ phi)
        bound = optimize.minimize_scalar(lambda l: float(pref @ np.abs(lam_true - l * rhat)),
                                         bounds=(1e-6, 1e3 * max(1.0, lam_true.max() / max(rhat.min(), 1e-9))),
                                         method="bounded").fun
        worst = max(worst, gap - bound)
    check(worst <= 1e-9, f"Prop 4.4: NP power - power of a threshold test on an estimate <= inf_l ||Lambda - l r||_1 "
          f"(max excess {worst:.2e})")


# --------------------------------------------------------------------------------- audit and monitoring
def audit_checks(rng, reps):
    n, phi = 200, 0.07
    a, b = 0.0, 1.0
    ests = []
    for _ in range(reps):
        F = rng.random(n) < phi
        fo = rng.random(n)
        fn = np.where(F, np.clip(fo + rng.normal(0.2, 0.3, n), a, b), fo)   # non-fired: identical (Thm 3.1)
        ests.append(float(np.sum((fn - fo)[F])) / n)
    true = phi * float(np.mean(np.clip(rng.random(400000) + rng.normal(0.2, 0.3, 400000), a, b) - rng.random(400000)))
    m, v = float(np.mean(ests)), float(np.var(ests))
    check(abs(m - true) < 4 * math.sqrt(v / reps) + 2e-3, f"Prop 5.1: audit estimator unbiased ({m:.4f} vs {true:.4f})")
    check(v <= phi * (b - a) ** 2 / n, f"Prop 5.1: variance {v:.2e} <= phi (b-a)^2 / n = {phi * (b - a) ** 2 / n:.2e}")


def monitor(xs, eps, lams, ws, delta):
    """first t with M_t >= 1/delta (or None): mixture of betting products"""
    logp = np.zeros(len(lams))
    for t, x in enumerate(xs, start=1):
        logp += np.log1p(lams * (x - eps))
        if np.log(np.sum(ws * np.exp(logp - logp.max()))) + logp.max() >= math.log(1 / delta):
            return t
    return None


def monitor_checks(rng, reps):
    eps, delta, T = 0.03, 0.05, 2000
    lams = np.array([0.5, 2.0, 8.0, 16.0, 30.0])
    ws = np.ones(len(lams)) / len(lams)
    for p in (0.03, 0.02):
        alarms = sum(monitor(rng.random(T) < p, eps, lams, ws, delta) is not None for _ in range(reps))
        check(alarms / reps <= delta + 3 * math.sqrt(delta * (1 - delta) / reps),
              f"Prop 5.2(a): false alarms within {T} prompts at p={p} <= eps: {alarms / reps:.3f} <= {delta}")
    # (b) expected detection time against the bound for the best single component
    for p in (0.08, 0.2):
        times = [monitor(rng.random(20000) < p, eps, lams, ws, delta) for _ in range(max(50, reps // 4))]
        times = [t for t in times if t is not None]
        bound = min((math.log(1 / (delta * w)) + math.log1p(l)) / g + 1
                    for l, w in zip(lams, ws)
                    for g in [p * math.log1p(l * (1 - eps)) + (1 - p) * math.log1p(-l * eps)] if g > 0)
        check(len(times) > 0 and float(np.mean(times)) <= bound,
              f"Prop 5.2(b): mean alarm time at p={p}: {np.mean(times):.1f} <= bound {bound:.1f}")
    p = 0.08
    l_kelly = (p - eps) / (eps * (1 - eps))
    g = p * math.log1p(l_kelly * (1 - eps)) + (1 - p) * math.log1p(-l_kelly * eps)
    klpe = p * math.log(p / eps) + (1 - p) * math.log((1 - p) / (1 - eps))
    check(abs(g - klpe) < 1e-12 and l_kelly < 1 / eps, f"Prop 5.2(b): Kelly growth equals kl(p || eps) = {klpe:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    reps = 1000 if args.quick else 4000
    print("== certificate (exact enumeration)")
    coupling_checks(rng)
    stream_checks(rng)
    print("== estimation and calibration (Monte Carlo)")
    calibration_checks(rng, reps)
    print("== the limit (linear programs)")
    limit_checks(rng)
    noevidence_checks(rng)
    estimated_checks(rng)
    print("== audit and monitoring (Monte Carlo)")
    audit_checks(rng, reps)
    monitor_checks(rng, reps // 2)
    print(f"\n{'PASSED' if not FAILS else 'FAILED: ' + str(len(FAILS))} ({'quick' if args.quick else 'full'})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
