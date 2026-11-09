#!/usr/bin/env python
"""Step p3.10a (Nov 2, PLAN_PAPER3.md): does the linear probe set the price of protection? (CPU, stored features)

    python scripts/score_classes.py --config configs/v3/squad2_q15.yaml --name Squad --variant protect_remedy \
        --protect advbench,general
    python scripts/score_classes.py --config configs/v2/overlap_q15.yaml --name Nq --variant deployed

For one single-task experiment and one reference (the deployed one, or the protected components with or without the
passage component), four scores are fitted on the same fitting prompts and PAC-calibrated with Theorem 3.6 at the same
level on the same calibration prompts:
  probe   the linear probe of the paper (class-balanced logistic regression, as scripts/switch_report.py)
  lda     a Gaussian density ratio with a shared, shrunk covariance
  knn     cosine similarity to the 15 nearest new-task prompts minus that to the 15 nearest reference prompts
  mlp     a one-hidden-layer network (128 tanh units), class-balanced logistic loss, L2 1e-3, L-BFGS; the iterate
          with the lowest loss on 15% of the fitting prompts held aside is kept
Settings are fixed here, before any result: no score is tuned on calibration or evaluation prompts, so Theorem 3.6
holds for each. Reported per score: coverage beta on the unlabeled RL prompts (Theorem 4.1 is priced by the best test,
and every score's beta is a lower bound on it), firing and its 95% Clopper-Pearson bound on every held-out pool, AUROC
of held-out task prompts against held-out reference pools, and Seq-GRPO's gain and refusal behind each gate from
stored responses (every seed). Needs <runs>/analysis/gate_features.npz.
Outputs in --out (default paper/v3_results/<exp>/scores_<variant>/): scores.json, scores_table.tex.
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
from scipy.optimize import minimize  # noqa: E402

import switch_report as sr  # noqa: E402
from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402

SEED = 0
K_NN = 15
HIDDEN = 128
L2_MLP = 1e-3
VAL_SHARE = 0.15


# ----------------------------------------------------------------------------- scores
class Standard:
    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0, ddof=1) + 1e-6

    def __call__(self, X):
        return (np.asarray(X, np.float64) - self.mu) / self.sd


def fit_lda(pos, neg, gamma=0.1):
    st = Standard(np.concatenate([pos, neg]).astype(np.float64))
    P, N = st(pos), st(neg)
    m1, m0 = P.mean(0), N.mean(0)
    C = 0.5 * (np.cov(P.T) + np.cov(N.T))
    d = C.shape[0]
    C = (1 - gamma) * C + gamma * (np.trace(C) / d) * np.eye(d)
    w = np.linalg.solve(C, m1 - m0)
    c = -0.5 * (m1 + m0) @ w
    return lambda X: st(X) @ w + c


def fit_knn(pos, neg, k=K_NN):
    st = Standard(np.concatenate([pos, neg]).astype(np.float64))

    def unit(X):
        Z = st(X)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    P, N = unit(pos), unit(neg)

    def topk_mean(S):
        kk = min(k, S.shape[1])
        return np.sort(S, axis=1)[:, -kk:].mean(1)

    def score(X, chunk=2048):
        U = unit(X)
        out = []
        for i in range(0, len(U), chunk):
            u = U[i:i + chunk]
            out.append(topk_mean(u @ P.T) - topk_mean(u @ N.T))
        return np.concatenate(out) if out else np.zeros(0)
    return score


def fit_mlp(pos, neg, hidden=HIDDEN, l2=L2_MLP, iters=400):
    rng = np.random.default_rng(SEED)
    X = np.concatenate([pos, neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    st = Standard(X)
    Z = st(X)
    val = np.zeros(len(y), bool)
    for lab in (0, 1):   # stratified hold-out of the fitting prompts
        idx = np.flatnonzero(y == lab)
        val[rng.choice(idx, max(1, int(round(VAL_SHARE * len(idx)))), replace=False)] = True
    d = Z.shape[1]

    def weights(m):
        yy = y[m]
        return np.where(yy == 1, 0.5 / max(1, yy.sum()), 0.5 / max(1, (1 - yy).sum()))
    wt, wv = weights(~val), weights(val)

    def unpack(t):
        i = 0
        W1 = t[i:i + d * hidden].reshape(d, hidden); i += d * hidden
        b1 = t[i:i + hidden]; i += hidden
        w2 = t[i:i + hidden]; i += hidden
        return W1, b1, w2, t[i]

    def forward(t, Zs):
        W1, b1, w2, b2 = unpack(t)
        H = np.tanh(Zs @ W1 + b1)
        return H, H @ w2 + b2

    def loss_grad(t):
        W1, b1, w2, b2 = unpack(t)
        Zs, ys = Z[~val], y[~val]
        H, z = forward(t, Zs)
        loss = float((wt * (np.logaddexp(0.0, z) - ys * z)).sum() + l2 * ((W1 ** 2).sum() + (w2 ** 2).sum()))
        g = wt * (1.0 / (1.0 + np.exp(-z)) - ys)
        gw2 = H.T @ g + 2 * l2 * w2
        gb2 = g.sum()
        gH = np.outer(g, w2) * (1 - H ** 2)
        gW1 = Zs.T @ gH + 2 * l2 * W1
        gb1 = gH.sum(0)
        return loss, np.concatenate([gW1.ravel(), gb1, gw2, [gb2]])

    def val_loss(t):
        _, z = forward(t, Z[val])
        return float((wv * (np.logaddexp(0.0, z) - y[val] * z)).sum())

    t0 = np.concatenate([rng.normal(0, 1 / np.sqrt(d), d * hidden), np.zeros(hidden),
                         rng.normal(0, 1 / np.sqrt(hidden), hidden), [0.0]])
    best = {"loss": val_loss(t0), "t": t0.copy(), "it": 0}
    step = {"it": 0}

    def cb(t):
        step["it"] += 1
        v = val_loss(t)
        if v < best["loss"]:
            best.update(loss=v, t=t.copy(), it=step["it"])
    minimize(loss_grad, t0, jac=True, method="L-BFGS-B", callback=cb, options={"maxiter": int(iters)})
    tb = best["t"]
    f = lambda Xq: forward(tb, st(Xq))[1]  # noqa: E731
    f.info = {"best_iter": best["it"], "val_loss": best["loss"]}
    return f


def auroc(pos_s, neg_s):
    """P(score of a positive > score of a negative), ties counted half"""
    pos_s, neg_s = np.asarray(pos_s, np.float64), np.asarray(neg_s, np.float64)
    if not len(pos_s) or not len(neg_s):
        return None
    allv = np.concatenate([pos_s, neg_s])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    s = np.sort(allv)
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[(allv >= s[i]) & (allv <= s[j])] = (i + j + 2) / 2.0
        i = j + 1
    rp = ranks[:len(pos_s)].sum()
    return float((rp - len(pos_s) * (len(pos_s) + 1) / 2) / (len(pos_s) * len(neg_s)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--name", default="")
    ap.add_argument("--variant", default="deployed", choices=["deployed", "protect", "protect_remedy"])
    ap.add_argument("--protect", default="", help="protected components (variants protect, protect_remedy)")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--runs", default="")
    ap.add_argument("--features", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--scores", default="probe,lda,knn,mlp")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    runs = args.runs or os.path.join(cfg["out_dir"], cfg["exp"])
    feat = args.features or os.path.join(runs, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        raise SystemExit(f"{feat} is missing (scripts/gate_features.py)")
    out_dir = args.out or os.path.join("paper", "v3_results", cfg["exp"], f"scores_{args.variant}")
    os.makedirs(out_dir, exist_ok=True)
    ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
    if ex.K != 1:
        raise SystemExit("score_classes.py takes single-task experiments")
    protect = [c.strip() for c in args.protect.split(",") if c.strip()]
    if args.variant != "deployed" and not protect:
        raise SystemExit("--protect is needed for the protect variants")
    if args.variant == "protect_remedy" and not any(ex.has(f"wrap__{t}__fit") for t in ex.tasks):
        raise SystemExit("no stored wrap__<task>__fit features: run step p3.8 first")
    base = "ratio" if args.variant == "deployed" else args.variant
    task, pos, neg, cal, refs_cal = ex.stage_data(1, base, protect)
    delta = float(ex.tc["delta"])
    fitters = {"probe": lambda p, n: sr.fit_probe(p, n, float(ex.tc["l2"]), int(ex.tc["lbfgs_iters"])).score,
               "lda": fit_lda, "knn": fit_knn, "mlp": fit_mlp,
               # p3.22: one more score class, a language model's score computed per prompt before calibration
               # (scripts/llm_gate_scores.py, --features <runs>/analysis/llm_scores.npz); nothing is fitted
               "llm": lambda p, n: (lambda X: np.asarray(X, np.float64)[:, 0])}
    want = [s.strip() for s in args.scores.split(",") if s.strip()]
    eval_keys = ex.eval_keys()
    held_task = f"eval__{task}" if ex.has(f"eval__{task}") else None
    neg_pools = [p for p in ("dolly", "advbench", "st_ctx", "st_ctx_question", "mmlu", "wildchat") if ex.has(f"eval__{p}")]

    # Seq-GRPO runs, every seed (stored greedy responses of the base and the trained model)
    seq_runs = []
    for sd in sorted((d for d in os.listdir(runs) if d.startswith("s") and d[1:].isdigit()), key=lambda d: int(d[1:])):
        md = os.path.join(runs, sd, "seq")
        if os.path.exists(os.path.join(md, "DONE")):
            loaded = sr.load_run(md, 1)
            if loaded is not None:
                seq_runs.append((sd, loaded))

    res = {"exp": cfg["exp"], "label": sr.readable_label(cfg), "task": task, "variant": args.variant,
           "protect": protect, "eps": args.eps, "delta": delta, "components": sorted(cal), "scores": {},
           "settings": {"k_nn": K_NN, "hidden": HIDDEN, "l2_mlp": L2_MLP, "val_share": VAL_SHARE, "seed": SEED},
           "seq_seeds": [sd for sd, _ in seq_runs]}
    own = [sr.metrics(R, B, ex.tasks) for _, (B, R) in seq_runs]
    res["seq_own"] = {k: sr.agg([o.get(k) for o in own]) for k in (f"{task}:correct", "advbench:refused", "dolly:chg")}
    for name in want:
        score = fitters[name](pos, neg)
        cs = {c: np.asarray(score(X), np.float64) for c, X in cal.items()}
        cal_res = calibrate({c: v.tolist() for c, v in cs.items()}, args.eps, delta)
        kappa = float(cal_res["kappa"])
        fire = {p: np.asarray(score(ex.X(f"eval__{p}")), np.float64) > kappa for p in eval_keys}
        rl = np.asarray(score(ex.X(f"rl__{task}")), np.float64) > kappa
        r = {"kappa": kappa, "beta": float(rl.mean()),
             "eps_cert": {c: v["eps_cert"] for c, v in cal_res["components"].items()},
             "fire": {p: float(f.mean()) for p, f in fire.items()},
             "cp": {p: clopper_pearson_upper(int(f.sum()), len(f)) for p, f in fire.items()},
             "auroc": {}}
        if held_task:
            sp = score(ex.X(held_task))
            for p in neg_pools:
                r["auroc"][p] = auroc(sp, score(ex.X(f"eval__{p}")))
        if hasattr(score, "info"):
            r["fit"] = score.info
        sw = [sr.metrics(R, B, ex.tasks, None, fire) for _, (B, R) in seq_runs]
        r["seq_behind"] = {k: sr.agg([m.get(k) for m in sw])
                           for k in (f"{task}:correct", "advbench:refused", "dolly:chg", "st_ctx:refused",
                                     "st_ctx_question:refused")}
        res["scores"][name] = r
        print(f"{cfg['exp']} {args.variant} {name}: beta {100 * r['beta']:.1f}%  "
              + "  ".join(f"{p} {100 * r['fire'][p]:.1f}%" for p in neg_pools if p in r["fire"]), flush=True)
    json.dump(res, open(os.path.join(out_dir, "scores.json"), "w"), indent=1)
    write_table(res, out_dir, neg_pools)
    print(f"wrote {out_dir}/scores.json and scores_table.tex")


def write_table(res, out_dir, pools):
    cols = [p for p in ("advbench", "st_ctx", "st_ctx_question", "dolly", "mmlu") if p in pools]
    head = {"advbench": "AdvBench", "st_ctx": "harm.+pass.", "st_ctx_question": "question+pass.", "dolly": "Dolly",
            "mmlu": "MMLU"}
    names = {"probe": "linear probe", "lda": "Gaussian ratio (LDA)", "knn": "$k$-NN similarity", "mlp": "MLP",
             "llm": "language model"}
    task = res["task"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Scores of " + res["label"].replace("_", r"\_") + f", reference: {res['variant'].replace('_', ' ')}"
             + r", each calibrated at $\varepsilon=" + f"{res['eps']}" + r"$ on the same prompts. $\beta$: coverage of the "
             r"unlabeled RL prompts, a lower bound on $\beta^\ast$ of \cref{thm:limit}; firing on held-out pools (\%); "
             r"AUROC of held-out task prompts against Dolly; Seq-GRPO's gain behind each gate (points).}",
             r"\begin{tabular}{l r " + "r" * len(cols) + r" r r}", r"\toprule",
             "Score & $\\beta\\uparrow$ & " + " & ".join(head[c] for c in cols) + r" & AUROC & Gain \\", r"\midrule"]
    for n, r in res["scores"].items():
        g = r["seq_behind"].get(f"{task}:correct")
        au = r["auroc"].get("dolly")
        lines.append(f"{names.get(n, n)} & {100 * r['beta']:.1f} & "
                     + " & ".join(f"{100 * r['fire'][c]:.1f}" for c in cols)
                     + f" & {'--' if au is None else f'{au:.3f}'} & {'--' if g is None else f'{100 * g[0]:+.1f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "scores_table.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
