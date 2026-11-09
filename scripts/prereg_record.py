#!/usr/bin/env python
"""Step p3.12 (Nov 3, PLAN_PAPER3.md): a record of what was fixed before any output of the external evaluation was
read. Run once on the login node, before scripts/external_report.py or slurm/p3ext.sbatch:

    python scripts/prereg_record.py --out paper/v3_results/p3_12/PREREG_p3_12.txt

The record holds the UTC time; the sha256 of the three external sets, of AdvBench, of the code that builds and reports
the gates and of every config evaluated; the text of the p3.12 section of PLAN_PAPER3.md (the recipe and every
prediction); the split of every external set into halves A and B (its counts and a digest of the assignment); and, for
honesty, which outputs already exist on disk at that time (their names and times only, nothing read). Commit the file
(git) or post it somewhere with a public timestamp, and paste it into the conversation, before the report runs.

Later steps: --step p3.19, p3.23 (records of their sets) and p3.26 (Nov 8, the frozen recipe on TriviaQA): run once
before any job of p3.26 (inputs and predictions, PREREG_p3_26.txt) and again after step p3.26a, before any training
(the gates' numbers from the features, PREREG_p3_26_gate.txt and pretrain_gate.json).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

FILES = ["data/advbench.jsonl", "data/ext_harmbench.jsonl", "data/ext_harmbench_ctx.jsonl", "data/ext_strongreject.jsonl",
         "twcore/external.py", "twcore/certify.py", "twcore/stress.py", "scripts/prepare_external.py",
         "scripts/external_report.py", "scripts/switch_report.py", "scripts/gate_features.py", "scripts/eval_extra.py",
         "configs/v2/ladder_concise_q15.yaml", "configs/v2/ladder_concisectx_q15.yaml", "configs/v3/squad2_q15.yaml",
         "configs/v2/overlap_q15.yaml", "configs/v2/ladder_ifc_q15.yaml", "configs/v2/ladder_ifc_q7.yaml",
         "configs/v2/stream_q15.yaml", "configs/v2/overlap_olmo7.yaml", "configs/fam/nq_llama3b.yaml"]
SETS = {"ext_hb": "data/ext_harmbench.jsonl", "ext_hb_ctx": "data/ext_harmbench_ctx.jsonl",
        "ext_sr": "data/ext_strongreject.jsonl"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


FILES_P319 = ["data/pr_mj_en.jsonl", "data/pr_mj_hi.jsonl", "data/pr_mj_mid.jsonl", "data/pr_mj_lo.jsonl",
              "data/pr_oa_en.jsonl", "data/pr_oa_ml.jsonl", "data/pr_oa_mt.jsonl", "data/pr_mt_harm.jsonl",
              "data/pr_sys_harm.jsonl", "data/pr_sysprompts.jsonl", "twcore/prospective.py",
              "scripts/prepare_prospective.py", "scripts/prospective_report.py", "scripts/external_report.py",
              "scripts/switch_report.py", "scripts/gate_features.py", "twcore/certify.py"] + FILES[12:]


def record_p319(out):
    """Step p3.19 (Nov 4): the record of the prospective sets, written before any gate sees them."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    L = [f"FireGate step p3.19, pre-registration record, written {now}", "", "sha256 of inputs, code and configs:"]
    for f in FILES_P319:
        L.append(f"  {sha(f) if os.path.exists(f) else 'missing':64s}  {f}")
    L += ["", "rows per set:"]
    for f in FILES_P319[:10]:
        L.append(f"  {f}: {sum(1 for line in open(f) if line.strip()) if os.path.exists(f) else 'missing'}")
    L += ["", "outputs on disk at this time (names only; none was read):"]
    feats = [js for js in sorted(glob.glob("runs/*/analysis/gate_features.json")) if '"eval__pr_' in open(js).read()]
    outs = sorted(glob.glob("paper/v3_results/p3_19/prospective*"))
    L.append(f"  gate features holding the prospective sets: {', '.join(feats) or 'none'}")
    L.append(f"  reports: {', '.join(outs) or 'none'}")
    plan = open("PLAN_PAPER3.md").read()
    m = re.search(r"\*\*p3\.19: .*?(?=\n\*\*p3\.2\d: |\n### |\n## |\Z)", plan, re.S)
    sec = m.group(0) if m else "(p3.19 section not found)"
    L += ["", f"PLAN_PAPER3.md, section p3.19 (sha256 {hashlib.sha256(sec.encode()).hexdigest()}):", sec]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"... wrote {out}; commit or post it before running slurm/p3_prosp.sh")


FILES_P323 = ["data/wd_aya.jsonl", "data/wd_dna.jsonl", "data/ext_harmbench.jsonl", "data/ext_strongreject.jsonl",
              "data/pr_mj_en.jsonl", "data/pr_mj_hi.jsonl", "data/pr_mj_mid.jsonl", "data/pr_mj_lo.jsonl",
              "data/pr_mt_harm.jsonl", "data/pr_sys_harm.jsonl", "data/advbench.jsonl", "twcore/widen.py",
              "scripts/prepare_widen.py", "scripts/widen_report.py", "scripts/fired_audit.py", "scripts/switch_report.py",
              "scripts/gate_features.py", "scripts/eval_extra.py", "twcore/certify.py",
              "configs/v2/ladder_concise_q15.yaml"]


def record_p323(out):
    """Step p3.23 (Nov 5): the record of the wide harmful component, written before any wide gate is fitted."""
    from twcore.widen import SOURCES, group_of, part_of
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    L = [f"FireGate step p3.23, pre-registration record, written {now}", "", "sha256 of inputs, code and configs:"]
    for f in FILES_P323:
        L.append(f"  {sha(f) if os.path.exists(f) else 'missing':64s}  {f}")
    L += ["", "parts of the sources (sha256('p3.23:<request group>') mod 3: fit, cal, test):"]
    files = {"ext_hb": "data/ext_harmbench.jsonl", "ext_sr": "data/ext_strongreject.jsonl",
             "pr_mt_harm": "data/pr_mt_harm.jsonl", "pr_sys_harm": "data/pr_sys_harm.jsonl",
             "pr_mj_en": "data/pr_mj_en.jsonl", "pr_mj_hi": "data/pr_mj_hi.jsonl", "pr_mj_mid": "data/pr_mj_mid.jsonl",
             "pr_mj_lo": "data/pr_mj_lo.jsonl"}
    for name in SOURCES:
        path = files[name]
        if not os.path.exists(path):
            L.append(f"  {name}: missing")
            continue
        ids = [json.loads(line).get("id", str(i)) for i, line in enumerate(open(path)) if line.strip()]
        parts = [part_of(group_of(name, i)) for i in ids]
        dig = hashlib.sha256("".join(p[0] for p in parts).encode()).hexdigest()[:16]
        L.append(f"  {name}: {len(ids)} rows, fit {parts.count('fit')}, cal {parts.count('cal')}, test "
                 f"{parts.count('test')}, assignment digest {dig}")
    L += ["", "rows of the sets never fitted or calibrated:"]
    for f in ("data/wd_aya.jsonl", "data/wd_dna.jsonl"):
        if os.path.exists(f):
            rows = [json.loads(line) for line in open(f) if line.strip()]
            langs = {}
            for r in rows:
                langs[r.get("lang")] = langs.get(r.get("lang"), 0) + 1
            L.append(f"  {f}: {len(rows)} rows ({', '.join(f'{k} {v}' for k, v in sorted(langs.items()))})")
        else:
            L.append(f"  {f}: missing")
    L += ["", "outputs on disk at this time (names only; none was read):"]
    feats = [js for js in sorted(glob.glob("runs/*/analysis/gate_features.json")) if '"eval__wd_' in open(js).read()]
    gens = sorted(glob.glob("runs/*/s*/*/extra/widen/*.jsonl"))
    outs = sorted(glob.glob("paper/v3_results/p3_23/widen*"))
    L.append(f"  gate features holding the new sets: {', '.join(feats) or 'none'}")
    L.append(f"  responses in extra/widen: {len(gens)} files")
    L.append(f"  reports: {', '.join(outs) or 'none'}")
    plan = open("PLAN_PAPER3.md").read()
    m = re.search(r"### Step p3\.23 .*?(?=\n### |\n## |\Z)", plan, re.S)
    sec = m.group(0) if m else "(p3.23 section not found)"
    L += ["", f"PLAN_PAPER3.md, section p3.23 (sha256 {hashlib.sha256(sec.encode()).hexdigest()}):", sec]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"... wrote {out}; keep it (or post it) before running bash slurm/p3_widen.sh")


CFG_P326, CFG_P326_FROM = "configs/p3/trivia_llama3b.yaml", "configs/fam/nq_llama3b.yaml"
FILES_P326 = ["data/trivia_train.jsonl", "data/trivia_test.jsonl", CFG_P326, CFG_P326_FROM, "data/dolly.jsonl",
              "data/advbench.jsonl", "data/general.jsonl", "data/xstest.jsonl", "data/mmlu_eval.jsonl",
              "data/ext_harmbench.jsonl", "data/ext_harmbench_ctx.jsonl", "data/ext_strongreject.jsonl",
              "scripts/prepare_data.py", "scripts/run_stream.py", "scripts/gate_features.py", "scripts/separability.py",
              "scripts/switch_report.py", "scripts/external_report.py", "scripts/prospective_report.py",
              "scripts/addon_report.py", "scripts/trivia_report.py", "scripts/eval_extra.py", "slurm/plans/p3.sh"]
P326_DIR = os.path.join("paper", "v3_results", "p3_26")


def _p326_section():
    plan = open("PLAN_PAPER3.md").read()
    m = re.search(r"### Step p3\.26 .*?(?=\n### |\n## |\Z)", plan, re.S)
    return m.group(0) if m else "(p3.26 section not found)"


def _p326_runs(exp_dir):
    """run folders of the experiment that have started (config.json) and the analysis files, names only"""
    started = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(exp_dir, "s*", "*", "config.json")))
    analysis = sorted(glob.glob(os.path.join(exp_dir, "analysis", "*")))
    return started, analysis


def record_p326(out):
    """Step p3.26 (Nov 8). First call: the record of the inputs and the predictions, before any job. Second call
    (after step p3.26a, before training): the gates' numbers from the features computed before training."""
    from twcore.config import load_config
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cfg = load_config(CFG_P326, [])
    exp_dir = os.path.join(cfg["out_dir"], cfg["exp"])
    started, analysis = _p326_runs(exp_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gate_out = out[:-4] + "_gate.txt"
    if not os.path.exists(out):
        missing = [f for f in FILES_P326[:2] if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"{', '.join(missing)} missing: python scripts/prepare_data.py --only-trivia --data data first")
        L = [f"FireGate step p3.26, pre-registration record, written {now}", "", "sha256 of inputs, code and configs:"]
        for f in FILES_P326:
            L.append(f"  {sha(f) if os.path.exists(f) else 'missing':64s}  {f}")
        h = hashlib.sha256()
        for f in sorted(glob.glob("twcore/*.py")):
            h.update(f.encode() + b"\0" + open(f, "rb").read())
        L.append(f"  {h.hexdigest():64s}  twcore/*.py ({len(glob.glob('twcore/*.py'))} files, names and contents)")
        L += ["", "rows:"]
        for f in FILES_P326[:2]:
            L.append(f"  {f}: {sum(1 for line in open(f) if line.strip()) if os.path.exists(f) else 'missing'}")
        old = load_config(CFG_P326_FROM, [])
        diff = sorted(k for k in set(cfg) | set(old) if cfg.get(k) != old.get(k))
        L += ["", f"settings that differ from {CFG_P326_FROM} (loaded with defaults): {', '.join(diff)}"]
        L += ["", "outputs on disk at this time (names only):",
              f"  started runs: {', '.join(started) or 'none'}", f"  analysis files: {', '.join(analysis) or 'none'}"]
        sec = _p326_section()
        L += ["", f"PLAN_PAPER3.md, section p3.26 (sha256 {hashlib.sha256(sec.encode()).hexdigest()}):", sec]
        open(out, "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        print(f"... wrote {out}; post it, then queue steps p3.26a,p3.26b,p3.26c (PLAN_PAPER3.md, step p3.26)")
        return
    if os.path.exists(gate_out):
        print(f"{out} and {gate_out} exist already; nothing written")
        return
    feat = os.path.join(exp_dir, "analysis", "gate_features.npz")
    if not os.path.exists(feat):
        print(f"{out} exists; {feat} is missing, so the gate record waits for step p3.26a")
        return
    if started:
        raise SystemExit(f"training has started ({', '.join(started)}): the gate record must come before any run")
    import switch_report as sr
    ex = sr.Experiment(cfg, feat, feat[:-4] + ".json")
    eps, alpha = float(cfg["tripwire"]["eps"]), 0.02
    res = {"written": now, "record_sha256": sha(out), "features_sha256": sha(feat),
           "features_json_sha256": sha(feat[:-4] + ".json"), "probes": "deployed" if ex.meta.get("probes") else "refitted",
           "eps": eps, "alpha": alpha, "gates": {}}
    for v in ("ratio", "twosided"):
        st = sr.gate_stats(ex, sr.build_variant(ex, v, eps, alpha))
        res["gates"][f"{v}|{eps}"] = {"power_pre": st["power_pre"], "fire": st["fire"], "cp": st["cp"],
                                      "kappa": st["kappa"], "tau": st["tau"], "eps_cert": st["eps_cert"]}
    sep = os.path.join(exp_dir, "analysis", "separability.json")
    res["separability"] = json.load(open(sep)) if os.path.exists(sep) else None
    jpath = os.path.join(os.path.dirname(out), "pretrain_gate.json")
    json.dump(res, open(jpath, "w"), indent=1, default=float)
    task = cfg["stream"][0]
    L = [f"FireGate step p3.26, gate record before training, written {now}", "",
         f"sha256 of {out}: {res['record_sha256']}", f"sha256 of {feat}: {res['features_sha256']}",
         f"sha256 of {jpath}: {sha(jpath)}", f"classifier: {res['probes']}", "",
         f"firing (%) at eps {eps}, alpha {alpha}, with the 95% Clopper-Pearson bound; coverage on {task} RL prompts:"]
    for key, g in res["gates"].items():
        L.append(f"  {key}: coverage {100 * g['power_pre'][task]:.1f}")
        for p in sorted(g["fire"]):
            L.append(f"    {p:16s} {100 * g['fire'][p]:5.1f}  [{100 * g['cp'][p]:.1f}]")
    if res["separability"]:
        L += ["", "separability.json: " + json.dumps(res["separability"].get("stages"))]
    L += ["", f"started runs: none; analysis files: {', '.join(analysis)}"]
    open(gate_out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"... wrote {gate_out} and {jpath}; post them, then PLAN_QUEUE=logs/plan_queue_p3 bash slurm/plan.sh feed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--step", default="p3.12", choices=["p3.12", "p3.19", "p3.23", "p3.26"])
    args = ap.parse_args()
    if args.step == "p3.26":
        record_p326(args.out or os.path.join(P326_DIR, "PREREG_p3_26.txt"))
        return
    if args.step == "p3.19":
        record_p319(args.out or os.path.join("paper", "v3_results", "p3_19", "PREREG_p3_19.txt"))
        return
    if args.step == "p3.23":
        record_p323(args.out or os.path.join("paper", "v3_results", "p3_23", "PREREG_p3_23.txt"))
        return
    args.out = args.out or os.path.join("paper", "v3_results", "p3_12", "PREREG_p3_12.txt")
    from external_report import half_a
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    L = [f"FireGate step p3.12, pre-registration record, written {now}", ""]
    L.append("sha256 of inputs, code and configs:")
    for f in FILES:
        L.append(f"  {sha(f) if os.path.exists(f) else 'missing':64s}  {f}")
    L.append("")
    L.append("halves of the external sets (sha256('<set>:<id>') even = A):")
    for name, path in SETS.items():
        if not os.path.exists(path):
            L.append(f"  {name}: missing")
            continue
        ids = [json.loads(line).get("id", str(i)) for i, line in enumerate(open(path)) if line.strip()]
        a = [half_a(name, i) for i in ids]
        dig = hashlib.sha256("".join("A" if x else "B" for x in a).encode()).hexdigest()[:16]
        L.append(f"  {name}: {len(ids)} rows, A {sum(a)}, B {len(a) - sum(a)}, assignment digest {dig}")
    L.append("")
    L.append("outputs on disk at this time (names and modification times only; none was read):")
    outs = sorted(glob.glob("runs/*/s*/*/extra/external/*.jsonl")) + \
        sorted(glob.glob("paper/v3_results/p3_12/external*"))
    for f in outs:
        t = datetime.datetime.fromtimestamp(os.path.getmtime(f), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        L.append(f"  {t}  {f}")
    feats = []
    for js in sorted(glob.glob("runs/*/analysis/gate_features.json")):
        if '"eval__ext_' in open(js).read():
            feats.append(js)
    L.append(f"  gate features holding the external sets: {', '.join(feats) or 'none'}")
    if not outs and not feats:
        L.append("  none")
    L.append("")
    plan = open("PLAN_PAPER3.md").read()
    m = re.search(r"\*\*p3\.12: .*?(?=\n\*\*p3\.13: )", plan, re.S)
    sec = m.group(0) if m else "(p3.12 section not found)"
    L.append(f"PLAN_PAPER3.md, section p3.12 (sha256 {hashlib.sha256(sec.encode()).hexdigest()}):")
    L.append(sec)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:len(FILES) + 12]))
    print(f"... wrote {args.out} ({len(L)} lines); commit or post it before running the report")


if __name__ == "__main__":
    main()
