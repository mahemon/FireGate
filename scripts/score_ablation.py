#!/usr/bin/env python
"""Which tripwire score? Power of calibrated prompt-time scores at a fixed certificate (no RL training).

    python scripts/score_ablation.py --config configs/v2/overlap_q15.yaml [--layers 4,8,14,20] [--out runs/<exp>/analysis]

For the overlap task of the config (NQ, IF or Concise) against its reference components, every score below is fitted
on the tripwire's fitting prompts, calibrated with the tripwire's own PAC rule (twcore.certify.calibrate, eps and
delta split over the calibration components) on its calibration prompts, and evaluated on the held-out evaluation
pools. Theorem 4.1 says that what a certified update can change on the new task is at most the power of the best
size-eps test; Proposition 4.4 says that a tripwire built from a density-ratio estimate loses its L1 error. So the
firing rate on held-out new-task prompts at a calibrated eps is the quantity to compare across scores.

Scores (all read at the last prompt token of the frozen trunk, block `layer`, RMS-normalized, unless stated):
  logistic   class-balanced logistic regression (the tripwire's probe)
  lda        Gaussian log density ratio with a shared, shrunk covariance (linear discriminant)
  mahal_ref  Mahalanobis distance to the reference prompts only (one-class OOD detector; uses no new-task prompts)
  knn        mean cosine similarity to the 10 nearest new-task fitting prompts minus that to the 10 nearest
             reference fitting prompts
  text       logistic regression on hashed token unigrams and bigrams of the prompt (no model features)
Outputs <out>/score_ablation.json and <out>/score_table.tex (rows: score x layer).
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

import torch  # noqa: E402

from twcore import dist  # noqa: E402
from twcore.certify import calibrate, clopper_pearson_upper  # noqa: E402
from twcore.config import load_config  # noqa: E402
from twcore.data import eval_pool  # noqa: E402
from twcore.gating import GateState  # noqa: E402
from twcore.model import load_model, make_encoder, template_prefix_len  # noqa: E402
from twcore.probe import auroc, fit_logreg, last_prompt_features  # noqa: E402
from twcore.stage import calibration_sets, fitting_sets  # noqa: E402
from twcore.tasks import POOLS  # noqa: E402

EPS = (0.03, 0.1, 0.3)
EVAL_POOLS = ("dolly", "advbench", "xstest", "mmlu")
HASH_DIM = 1 << 14


def standardize(Xfit):
    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-6
    return lambda X: (X - mu) / sd


def score_logistic(Xp, Xn, l2, iters):
    tw, _ = fit_logreg(Xp, Xn, l2, iters)
    return lambda X: ((X - tw.mu) / tw.sigma) @ tw.a + tw.b


def _shrunk_inverse(Z, gamma=0.1):
    C = torch.cov(Z.t().double())
    d = C.shape[0]
    C = (1 - gamma) * C + gamma * (torch.trace(C) / d) * torch.eye(d, dtype=C.dtype, device=C.device)
    return torch.linalg.inv(C).float()


def score_lda(Xp, Xn):
    st = standardize(torch.cat([Xp, Xn]))
    Zp, Zn = st(Xp), st(Xn)
    # pooled within-class covariance, classes weighted equally
    Zc = torch.cat([Zp - Zp.mean(0), Zn - Zn.mean(0)])
    P = _shrunk_inverse(Zc)
    w = P @ (Zp.mean(0) - Zn.mean(0))
    return lambda X: st(X) @ w


def score_mahal_ref(Xn):
    st = standardize(Xn)
    Zn = st(Xn)
    m = Zn.mean(0)
    P = _shrunk_inverse(Zn - m)
    return lambda X: (((st(X) - m) @ P) * (st(X) - m)).sum(-1)


def score_knn(Xp, Xn, k=10):
    Up, Un = torch.nn.functional.normalize(Xp, dim=-1), torch.nn.functional.normalize(Xn, dim=-1)

    def f(X):
        U = torch.nn.functional.normalize(X, dim=-1)
        out = []
        for i in range(0, len(U), 512):
            u = U[i:i + 512]
            sp = (u @ Up.t()).topk(min(k, len(Up)), dim=-1).values.mean(-1)
            sn = (u @ Un.t()).topk(min(k, len(Un)), dim=-1).values.mean(-1)
            out.append(sp - sn)
        return torch.cat(out)
    return f


def hashed_bow(prompts, device):
    """tf (log 1 + count) of hashed token unigrams and bigrams, one row per prompt."""
    X = torch.zeros((len(prompts), HASH_DIM), dtype=torch.float32)
    for i, p in enumerate(prompts):
        feats = list(p) + [a * 1_000_003 + b for a, b in zip(p, p[1:])]
        idx = torch.tensor([(f * 2654435761) % HASH_DIM for f in feats], dtype=torch.long)
        X[i].index_add_(0, idx, torch.ones(len(idx)))
    return torch.log1p(X).to(device)


def evaluate(score_fn, cal_feats, eval_feats, delta):
    cal_scores = {c: score_fn(X).float().cpu().tolist() for c, X in cal_feats.items()}
    ev = {p: score_fn(X).float().cpu() for p, X in eval_feats.items()}
    out = {"auroc_new_vs_dolly": auroc(ev["new"], ev["dolly"]) if "dolly" in ev else None, "eps": {}}
    for eps in EPS:
        cal = calibrate(cal_scores, eps, delta)
        kap = cal["kappa"]
        fire = {p: float((s > kap).float().mean()) for p, s in ev.items()}
        X = int((ev["dolly"] > kap).sum()) if "dolly" in ev else 0
        out["eps"][f"{eps:g}"] = {"kappa": kap, "fire": fire,
                                 "cert_dolly_heldout": clopper_pearson_upper(X, len(ev["dolly"])) if "dolly" in ev else None,
                                 "feasible": all(v["feasible"] for v in cal["components"].values())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--layers", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dist.init()
    cfg = load_config(args.config, args.set)
    task = cfg["stream"][0]
    out_dir = args.out or os.path.join(cfg["out_dir"], cfg["exp"], "analysis")
    os.makedirs(out_dir, exist_ok=True)
    gl0 = int(cfg["gate_layer"])
    layers = sorted({int(x) for x in args.layers.split(",") if x.strip()} | {gl0})
    tcfg = cfg["tripwire"]
    fb = cfg["gen"]["feature_batch_size"]

    model, tok = load_model(cfg["model_path"], cfg["dtype"], device=dist.device())
    nlayers = len(model.model.layers)
    layers = [L for L in layers if 1 <= L < nlayers]
    encode = make_encoder(tok, cfg["gen"]["max_prompt_tokens"])
    state = GateState()
    state.gate_from = template_prefix_len(encode)
    pad = tok.pad_token_id
    dev = next(model.parameters()).device

    def prompts_of(pool, exs):
        return [encode(POOLS[pool].prompt(ex)) for ex in exs]

    fit = {g: prompts_of(pool, exs) for g, (pool, exs) in fitting_sets(cfg, task, []).items()}
    cal = {c: prompts_of(pool, exs) for c, (pool, exs) in calibration_sets(cfg, []).items()}
    ev = {"new": prompts_of(task, eval_pool(cfg, task))}
    for p in EVAL_POOLS:
        if p in cfg["pools"]:
            ev[p] = prompts_of(p, eval_pool(cfg, p))
    pos_fit = fit["new"]
    neg_fit = [p for g, ps in fit.items() if g != "new" for p in ps]
    print(f"{cfg['exp']}: task {task}; fit {len(pos_fit)} new / {len(neg_fit)} reference; calibration "
          + ", ".join(f"{c} {len(v)}" for c, v in cal.items()) + "; evaluation "
          + ", ".join(f"{p} {len(v)}" for p, v in ev.items()) + f"; layers {layers} of {nlayers}", flush=True)

    results = {"exp": cfg["exp"], "task": task, "model": cfg["model_path"], "gate_layer": gl0, "layers": layers,
               "delta": tcfg["delta"], "n": {"fit_new": len(pos_fit), "fit_ref": len(neg_fit),
                                            "cal": {c: len(v) for c, v in cal.items()},
                                            "eval": {p: len(v) for p, v in ev.items()}}, "scores": {}}
    t0 = time.time()
    with torch.no_grad():
        for L in layers:
            feats = lambda ps: last_prompt_features(model, state, L, ps, pad, fb)  # noqa: E731
            Xp, Xn = feats(pos_fit), feats(neg_fit)
            cal_f = {c: feats(v) for c, v in cal.items()}
            ev_f = {p: feats(v) for p, v in ev.items()}
            fns = {"logistic": score_logistic(Xp, Xn, tcfg["l2"], tcfg["lbfgs_iters"]), "lda": score_lda(Xp, Xn),
                   "mahal_ref": score_mahal_ref(Xn), "knn": score_knn(Xp, Xn)}
            for name, fn in fns.items():
                results["scores"][f"{name}@{L}"] = dict(evaluate(fn, cal_f, ev_f, tcfg["delta"]), score=name, layer=L)
            r = results["scores"][f"logistic@{L}"]
            print(f"  layer {L}: logistic AUROC {r['auroc_new_vs_dolly']:.3f}, fires on new task at eps=0.03: "
                  f"{100 * r['eps']['0.03']['fire']['new']:.1f}% ({time.time() - t0:.0f}s)", flush=True)
            del Xp, Xn, cal_f, ev_f
        # model-free text classifier on the same prompts
        Hp, Hn = hashed_bow(pos_fit, dev), hashed_bow(neg_fit, dev)
        fn = score_logistic(Hp, Hn, tcfg["l2"], tcfg["lbfgs_iters"])
        results["scores"]["text"] = dict(evaluate(fn, {c: hashed_bow(v, dev) for c, v in cal.items()},
                                                  {p: hashed_bow(v, dev) for p, v in ev.items()}, tcfg["delta"]),
                                         score="text", layer=None)
    with open(os.path.join(out_dir, "score_ablation.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(out_dir, "score_table.tex"), "w") as f:
        f.write(score_table(results))
    print("wrote", os.path.join(out_dir, "score_ablation.json"), "and score_table.tex")
    dist.finish()


NAMES = {"logistic": "Logistic probe (\\method{})", "lda": "Gaussian density ratio (LDA)",
         "mahal_ref": "Mahalanobis to reference (one-class)", "knn": "$k$-NN similarity", "text": "Text only (hashed $n$-grams)"}
TASK_NAMES = {"nq": "NQ", "ifc": "IF", "concise": "Concise"}


def score_table(res):
    tn = TASK_NAMES.get(res["task"], res["task"])
    pct = lambda v: "--" if v is None or v != v else f"{100 * v:.1f}"  # noqa: E731
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Tripwire scores for " + tn + r" against the reference prompts, " +
             os.path.basename(res["model"].rstrip("/")).replace("_", r"\_") + r", no training. Each score is fitted on "
             r"the tripwire's fitting prompts and calibrated with its PAC rule; $L$: block whose input is read. Fires: "
             r"share of held-out " + tn + r" prompts on which it fires at a certified $\varepsilon$, the bound of "
             r"\cref{thm:limit} on what a certified update can change; AUROC against held-out Dolly. Dolly, MMLU: "
             r"firing on held-out prompts at $\varepsilon=0.03$.}",
             r"\label{tab:score_" + res["task"] + "}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l r r rrr rr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Fires on " + tn + r" (\%)} & \multicolumn{2}{c}{$\varepsilon=0.03$} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
             r"Score & $L$ & AUROC & $\varepsilon{=}0.03$ & $0.1$ & $0.3$ & Dolly & MMLU \\", r"\midrule"]
    order = ["logistic", "lda", "knn", "mahal_ref", "text"]
    rows = sorted(res["scores"].values(), key=lambda r: (order.index(r["score"]), r["layer"] or 0))
    prev = None
    for r in rows:
        if prev is not None and r["score"] != prev:
            lines.append(r"\midrule")
        prev = r["score"]
        e = r["eps"]
        lines.append(" & ".join([NAMES[r["score"]], "--" if r["layer"] is None else str(r["layer"] + 1),
                                 "--" if r["auroc_new_vs_dolly"] is None else f"{r['auroc_new_vs_dolly']:.3f}",
                                 pct(e["0.03"]["fire"]["new"]), pct(e["0.1"]["fire"]["new"]), pct(e["0.3"]["fire"]["new"]),
                                 pct(e["0.03"]["fire"].get("dolly")), pct(e["0.03"]["fire"].get("mmlu"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
