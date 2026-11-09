"""Checks that need no torch: generators, scorers, certificates, table pipeline.

    python tests/test_pure.py            (exits non-zero on failure)
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from twcore.certify import (binom_cdf, binom_sf_ge, calib_index, calibrate,  # noqa: E402
                              clopper_pearson_upper, eps_for_index)
from twcore.gen_countdown import make_instances as make_cd  # noqa: E402
from twcore.gen_kk import make_instances as make_kk, solve  # noqa: E402
import twcore.tasks  # noqa: E402,F401  (load tasks first: tasks -> stress -> ladder)
import twcore.ladder  # noqa: E402,F401  (registers the ifc / concise pools)
from twcore.tasks import (POOLS, countdown_check, extract_letter, has_task_format, is_refusal,  # noqa: E402
                            last_answer_block, last_boxed)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def test_generators():
    cd = make_cd(500, seed=1)
    for ex in cd:
        check(countdown_check(ex["ref_expr"], ex["nums"], ex["target"]), f"countdown reference fails {ex}")
    kk = make_kk(300, seed=2)
    for ex in kk:
        n = len(ex["names"])
        check(len(ex["statements"]) == n, "kk statements")
        txt = "<answer> " + " ".join(f"({i + 1}) {nm} is a {'knight' if s else 'knave'}"
                                     for i, (nm, s) in enumerate(zip(ex["names"], ex["solution"]))) + " </answer>"
        check(POOLS["kk"].score(ex, "reasoning... " + txt)["correct"] == 1, f"kk correct answer scored wrong {ex['id']}")
        wrong = list(ex["solution"])
        wrong[0] = not wrong[0]
        txt2 = "<answer> " + " ".join(f"({i + 1}) {nm} is a {'knight' if s else 'knave'}"
                                      for i, (nm, s) in enumerate(zip(ex["names"], wrong))) + " </answer>"
        check(POOLS["kk"].score(ex, txt2)["correct"] == 0, "kk wrong answer scored right")
    print("generators ok:", len(cd), "countdown,", len(kk), "kk")


def test_scorers():
    ex = {"question": "q", "answer": "blah\n#### 1,234"}
    check(POOLS["gsm8k"].score(ex, "so \\boxed{1234}")["correct"] == 1, "gsm8k boxed")
    check(POOLS["gsm8k"].score(ex, "so \\boxed{\\$1,234}")["correct"] == 1, "gsm8k dollar")
    check(POOLS["gsm8k"].score(ex, "the answer is 1234")["correct"] == 0, "gsm8k needs boxed")
    check(POOLS["gsm8k"].score(ex, "\\boxed{12 \\times 100 + 34 = 1234}")["correct"] == 1, "gsm8k last number")
    check(last_boxed("a \\boxed{\\frac{1}{2}} b") == "\\frac{1}{2}", "nested boxed")
    cd = {"nums": [3, 5, 7], "target": 22}
    check(POOLS["countdown"].score(cd, "<answer> 3 * 5 + 7 </answer>")["correct"] == 1, "countdown ok")
    check(POOLS["countdown"].score(cd, "<answer> 3 * 5 + 7 = 22 </answer>")["correct"] == 1, "countdown with =")
    check(POOLS["countdown"].score(cd, "<answer> 3 * 5 + 7 + 0 </answer>")["correct"] == 0, "countdown extra number")
    check(POOLS["countdown"].score(cd, "<answer> 3 ** 5 </answer>")["correct"] == 0, "countdown power")
    check(POOLS["countdown"].score(cd, "<answer> (3 + 5) * 7 / 0 </answer>")["correct"] == 0, "countdown div0")
    check(last_answer_block("x <answer> a </answer> y <answer> b") == " b", "unclosed answer")
    check(extract_letter("B") == "B", "letter plain")
    check(extract_letter("The answer is (C).") == "C", "letter answer is")
    check(extract_letter("**D**. Because") == "D", "letter bold")
    mm = {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 2, "subject": "x"}
    check(POOLS["mmlu"].score(mm, "C")["correct"] == 1, "mmlu")
    check(is_refusal("I'm sorry, but I can't help with that."), "refusal")
    check(is_refusal("I’m sorry, I cannot do that"), "refusal curly quote")
    check(not is_refusal("Sure! Here is a poem about cats."), "non-refusal")
    check(has_task_format("x \\boxed{3}") and has_task_format("<answer>1</answer>") and not has_task_format("plain"), "leak")
    nq = {"question": "who wrote the novel the old man and the sea", "answers": ["Ernest Hemingway", "Hemingway"]}
    check(POOLS["nq"].prompt(nq) == "Who wrote the novel the old man and the sea?", "nq prompt")
    check(POOLS["nq"].score(nq, "It was written by Ernest Hemingway in 1951.")["correct"] == 1, "nq full name")
    check(POOLS["nq"].score(nq, "HEMINGWAY.")["correct"] == 1, "nq case/punctuation")
    check(POOLS["nq"].score(nq, "Hemingwayesque prose")["correct"] == 0, "nq word boundary")
    check(POOLS["nq"].score(nq, "x" * 300 + " Hemingway")["correct"] == 0, "nq only the first 300 characters")
    yr = {"question": "when did the war end", "answers": ["the 1990s", "1998"]}
    check(POOLS["nq"].score(yr, "In 1998.")["correct"] == 1 and POOLS["nq"].score(yr, "During the 1990s")["correct"] == 1,
          "nq articles stripped")
    print("scorers ok")


def test_certify():
    try:
        from scipy import stats
    except Exception:
        stats = None
    for n, p, k in [(50, 0.3, 10), (300, 0.98, 295), (1000, 0.02, 30)]:
        if stats is not None:
            check(abs(binom_cdf(k, n, p) - stats.binom.cdf(k, n, p)) < 1e-9, f"binom_cdf {n},{p},{k}")
            check(abs(binom_sf_ge(k, n, p) - stats.binom.sf(k - 1, n, p)) < 1e-9, f"binom_sf {n},{p},{k}")
    for x, n in [(0, 250), (3, 400), (10, 100), (50, 50)]:
        u = clopper_pearson_upper(x, n, 0.05)
        if stats is not None and x < n:
            ref = stats.beta.ppf(0.95, x + 1, n - x)
            check(abs(u - ref) < 1e-6, f"CP {x}/{n}: {u} vs {ref}")
    check(abs(clopper_pearson_upper(0, 250, 0.05) - (1 - 0.05 ** (1 / 250))) < 1e-6, "CP zero")
    k = calib_index(1000, 0.02, 0.05)
    check(k is not None and 985 <= k <= 992, f"calib index {k}")
    check(calib_index(100, 0.02, 0.01) is None, "infeasible calib")
    e = eps_for_index(100, 100, 0.01)
    check(abs(e - (1 - 0.01 ** (1 / 100))) < 1e-6, f"eps_for_index {e}")
    # empirical PAC check: fraction of calibration draws whose true firing rate exceeds eps must be <= delta
    rng = random.Random(0)
    bad, trials = 0, 400
    for _ in range(trials):
        sc = {"a": [rng.random() for _ in range(300)], "b": [rng.random() for _ in range(200)]}
        cal = calibrate(sc, 0.03, 0.05)
        true_rate = 1 - cal["kappa"]  # scores ~ U(0,1): P(S > kappa)
        bad += true_rate > 0.03
    check(bad / trials <= 0.05 + 0.03, f"PAC calibration violated: {bad}/{trials}")
    print(f"certify ok (PAC violation rate {bad / trials:.3f}, target <= 0.05)")


def test_tables():
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(1)
        pools = {"gsm8k": 40, "countdown": 40, "kk": 40, "mmlu": 40, "advbench": 30, "xstest": 30, "dolly": 30}
        metric = {"gsm8k": "correct", "countdown": "correct", "kk": "correct", "mmlu": "correct",
                  "advbench": "refused", "xstest": "refused", "dolly": "leak"}
        for method in ("seq", "kl", "replay", "tripwire"):
            base_out = {}
            for k in range(4):
                edir = os.path.join(tmp, method, f"stage{k}", "eval")
                os.makedirs(edir)
                for pool, n in pools.items():
                    rows = []
                    for i in range(n):
                        fire = {}
                        if method == "tripwire":
                            for t in range(1, k + 1):
                                fire[str(t)] = (-3 if (pool == ["gsm8k", "countdown", "kk"][t - 1]) else
                                                (5 if rng.random() < 0.03 else None))
                        prev = base_out.get((pool, i), [1, 2, 3])
                        if method == "tripwire" and k > 0 and fire.get(str(k)) is None:
                            out = prev
                        else:
                            out = prev if rng.random() < 0.5 else [rng.randint(0, 9)]
                        base_out[(pool, i)] = out
                        m = {metric[pool]: int(rng.random() < 0.5)}
                        if pool == "dolly":
                            m["refused"] = 0
                        row = {"i": i, "id": f"{pool}{i}", "plen": 10, "out": out, "text": "", "m": m, "fire": fire, "maxs": {}}
                        if pool == "xstest":
                            row["label"] = "safe" if i % 2 else "unsafe"
                        rows.append(row)
                    with open(os.path.join(edir, f"{pool}.jsonl"), "w") as f:
                        for r in rows:
                            f.write(json.dumps(r) + "\n")
                open(os.path.join(edir, "DONE"), "w").close()
                if method == "tripwire" and k > 0:
                    tdir = os.path.join(tmp, method, f"stage{k}", "tripwire")
                    os.makedirs(tdir)
                    json.dump({"kappa": 0.5, "components": {"dolly": {"eps_cert": 0.03}, "advbench": {"eps_cert": 0.04}},
                               "auroc": 0.99, "posthoc": {"new": [rng.random() + 0.6 for _ in range(30)],
                                                          "dolly": [rng.random() * 0.6 for _ in range(30)],
                                                          "advbench": [rng.random() * 0.5 for _ in range(30)]}},
                              open(os.path.join(tdir, "tripwire.json"), "w"))
        paper = os.path.join(tmp, "paper")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_tables.py"), "--runs", tmp,
                            "--paper", paper], capture_output=True, text=True)
        check(r.returncode == 0, "make_tables failed: " + r.stderr[-2000:])
        for fn in ("numbers.tex", "main_table.tex", "cert_table.tex", "calib_table.tex", "stage_table.tex"):
            check(os.path.exists(os.path.join(paper, "results", fn)), f"missing {fn}")
        nums = open(os.path.join(paper, "results", "numbers.tex")).read()
        check("\\newcommand{\\resExactViol}{0}" in nums, "exactness check should report 0 violations: " + nums)
        # v2 layout: two seed folders -> mean +- sd tables
        seeds = os.path.join(tmp, "seeds")
        for sd in ("s0", "s1"):
            os.makedirs(os.path.join(seeds, sd))
            for method in ("seq", "kl", "replay", "tripwire"):
                md = os.path.join(seeds, sd, method)
                shutil.copytree(os.path.join(tmp, method), md)
                gpu = "NVIDIA A100 80GB PCIe" if method == "replay" else "NVIDIA H200 NVL"
                json.dump({"method": method, "device": gpu}, open(os.path.join(md, "config.json"), "w"))
                if not (sd == "s1" and method == "kl"):      # s1/kl is still running: left out of the tables
                    open(os.path.join(md, "DONE"), "w").close()
        paper2 = os.path.join(tmp, "paper2")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_tables.py"), "--runs", seeds,
                            "--paper", paper2], capture_output=True, text=True)
        check(r.returncode == 0, "make_tables over seeds failed: " + r.stderr[-2000:])
        if r.returncode == 0:
            mt = open(os.path.join(paper2, "results", "main_table.tex")).read()
            nm = open(os.path.join(paper2, "results", "numbers.tex")).read()
            check("(of 2;" in mt and "\\pm" in mt, "seed table shows mean +- sd: " + mt[-600:])
            check("\\newcommand{\\resSeeds}{2}" in nm and "\\newcommand{\\resExactViol}{0}" in nm, "seed numbers: " + nm[:400])
            check("Base model (A100)" in mt and "Base model (H200)" in mt and "$^{\\mathrm{A}}$" in mt
                  and "kl" in r.stdout and "unfinished (left out): ['kl']" in r.stdout,
                  "tables: unfinished runs left out, one base row per GPU model: " + r.stdout[-600:])
        print("tables ok")
    finally:
        shutil.rmtree(tmp)


def test_overlap_report():
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(2)
        pools = {"nq": 40, "mmlu": 40, "advbench": 30, "xstest": 30, "dolly": 30}
        metric = {"nq": "correct", "mmlu": "correct", "advbench": "refused", "xstest": "refused", "dolly": "leak"}
        for method in ("seq", "kl", "router", "tripwire"):
            for k in range(2):
                edir = os.path.join(tmp, method, f"stage{k}", "eval")
                os.makedirs(edir)
                for pool, n in pools.items():
                    with open(os.path.join(edir, f"{pool}.jsonl"), "w") as f:
                        for i in range(n):
                            m = {metric[pool]: int(rng.random() < 0.5)}
                            if pool == "dolly":
                                m["refused"] = 0
                            fire = {"1": (5 if rng.random() < 0.5 else None)} if k and method in ("tripwire", "router") else {}
                            # a gated method keeps the base response wherever its gate did not fire
                            keep = k == 0 or (method in ("tripwire", "router") and fire.get("1") is None)
                            row = {"i": i, "out": [1, 2] if keep else [1], "m": m, "fire": fire}
                            if pool == "xstest":
                                row["label"] = "safe" if i % 2 else "unsafe"
                            f.write(json.dumps(row) + "\n")
                open(os.path.join(edir, "DONE"), "w").close()
        # KL at a second coefficient lives in <runs>_kl0.2/kl; each run records its config
        shutil.copytree(os.path.join(tmp, "kl"), os.path.join(tmp + "_kl0.2", "kl"))
        for root, beta in ((tmp, 0.01), (tmp + "_kl0.2", 0.2)):
            json.dump({"config": {"grpo": {"kl_beta": beta}}}, open(os.path.join(root, "kl", "config.json"), "w"))
        # a second operating point trained at eps 0.10 lives in <runs>_eps0.10/tripwire
        shutil.copytree(os.path.join(tmp, "tripwire"), os.path.join(tmp + "_eps0.10", "tripwire"))
        for root, eps in ((tmp, 0.03), (tmp + "_eps0.10", 0.10)):
            tdir = os.path.join(root, "tripwire", "stage1", "tripwire")
            os.makedirs(tdir)
            json.dump({"kappa": 0.5, "eps": eps, "components": {"dolly": {"eps_cert": eps}},
                       "posthoc": {"new": [rng.random() + 0.3 for _ in range(50)],
                                   "dolly": [rng.random() for _ in range(50)]}},
                      open(os.path.join(tdir, "tripwire.json"), "w"))
        paper = os.path.join(tmp, "paper")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_overlap_report.py"), "--runs", tmp,
                            "--paper", paper], capture_output=True, text=True)
        check(r.returncode == 0, "make_overlap_report failed: " + r.stderr[-2000:])
        for fn in ("overlap_table.tex", "overlap_numbers.tex", "overlap_frontier_tw.dat", "overlap_frontier_base.dat", "overlap_frontier_kl.dat"):
            check(os.path.exists(os.path.join(paper, "results", fn)), f"missing {fn}")
        check("not finished" not in r.stdout and "| 0.030 |" in r.stdout and "eps=0.10" in r.stdout
              and "beta=0.2" in r.stdout and "beta=0.01" in r.stdout,
              "overlap report output: " + r.stdout[-1500:])
        nums = open(os.path.join(paper, "results", "overlap_numbers.tex")).read()
        check("\\ovlTwBEps}{0.10}" in nums and "\\ovlSeqGain}" in nums and "\\ovlTwAExactViol}{0}" in nums,
              "overlap macros: " + nums[:800])
        check(len(open(os.path.join(paper, "results", "overlap_frontier_tw.dat")).read().splitlines()) == 4,
              "frontier data should hold the header, the base point and two operating points")
        print("overlap report ok")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(tmp + "_eps0.10", ignore_errors=True)
        shutil.rmtree(tmp + "_kl0.2", ignore_errors=True)


def test_ladder():
    from twcore import ladder
    good_bad = {"lowercase": ("this is fine.", "This is not."), "no_commas": ("no commas here", "a, b"),
                "three_bullets": ("intro\n* a\n* b\n* c", "* a\n* b"),
                "closing": ("Done. " + ladder.CLOSING, "Done."), "quotes": ('"hello"', "hello"),
                "short": ("A short answer.", " ".join(["w"] * 45)), "certainly": ("Certainly! Yes.", "Sure."),
                "two_paragraphs": ("One.\n***\nTwo.", "One.\n\nTwo.")}
    check(set(good_bad) == set(ladder.CONSTRAINTS), "every ladder constraint is tested")
    for k, (g, b) in good_bad.items():
        ex = {"instruction": "Tell me about whales", "context": "", "constraint": k}
        check(POOLS["ifc"].score(ex, g)["correct"] == 1 and POOLS["ifc"].score(ex, b)["correct"] == 0, f"ifc checker {k}")
    check(POOLS["concise"].score({}, "Whales are large marine mammals.")["correct"] == 1
          and POOLS["concise"].score({}, "I'm sorry, but I can't.")["correct"] == 0
          and POOLS["concise"].score({}, " ".join(["w"] * 31))["correct"] == 0, "concise reward")
    check(ladder.format_leaks("Certainly! Whales are big.") == 1 and ladder.format_leaks("Whales are big.") == 0,
          "format leak detector")
    print("ladder ok")


def test_stream5():
    """MATH answer parsing and scoring; 4x4 Sudoku generator (unique solutions) and answer parsing."""
    from fractions import Fraction as F
    from twcore.gen_sudoku import count_solutions, is_valid, make_instances
    from twcore.stream5 import math_number
    cases = {"12": F(12), "-3": F(-3), "\\frac{3}{4}": F(3, 4), "\\dfrac{3}{4}": F(3, 4), "-\\frac{1}{2}": F(-1, 2),
             "\\frac12": F(1, 2), "0.5": F(1, 2), "1,000": F(1000), "\\$5.50": F(11, 2), "45^\\circ": F(45),
             "10\\text{ inches}": F(10), "3/4": F(3, 4), "25\\%": F(25), "{7}": F(7), "1,2": None, "\\sqrt{2}": None,
             "2\\frac{1}{2}": None, "(1,2)": None, "\\pi": None, "x=3": None}
    bad = [s for s, v in cases.items() if math_number(s) != v]
    check(not bad, f"MATH answer parser: {bad}")
    ex = {"problem": "Compute 1/2.", "answer": "1/2"}
    check(POOLS["math"].score(ex, "so \\boxed{\\frac{1}{2}}")["correct"] == 1
          and POOLS["math"].score(ex, "\\boxed{0.5}")["correct"] == 1
          and POOLS["math"].score(ex, "\\boxed{0.25}")["correct"] == 0
          and POOLS["math"].score(ex, "0.5") == {"correct": 0, "format": 0}, "MATH scorer")
    check("\\boxed{}" in POOLS["math"].prompt(ex), "MATH uses the GSM8K instruction")
    ps = make_instances(600, seed=3)
    check(len({p["puzzle"] for p in ps}) == 600, "distinct Sudoku puzzles")
    ok = True
    for p in ps[:200]:
        g = [[int(p["puzzle"][4 * r + c]) for c in range(4)] for r in range(4)]
        s = [[int(p["solution"][4 * r + c]) for c in range(4)] for r in range(4)]
        ok &= count_solutions(g) == 1 and is_valid(s) and all(g[r][c] in (0, s[r][c]) for r in range(4) for c in range(4))
        ok &= sum(v == 0 for row in g for v in row) == p["blanks"]
    check(ok, "Sudoku puzzles have a unique solution that matches the stored one")
    p = ps[0]
    rows = [p["solution"][i:i + 4] for i in range(0, 16, 4)]
    sc = POOLS["sudoku"].score
    check(sc(p, "<answer>\n" + "\n".join(rows) + "\n</answer>")["correct"] == 1
          and sc(p, "<answer>\n" + "\n".join(f"Row {i + 1}: {' '.join(r)}" for i, r in enumerate(rows)) + "\n</answer>")["correct"] == 1
          and sc(p, "<answer> " + " ".join(" ".join(r) for r in rows) + " </answer>")["correct"] == 1
          and sc(p, "<answer>\n" + "\n".join(rows[::-1]) + "\n</answer>")["correct"] == 0
          and sc(p, "\n".join(rows)) == {"correct": 0, "format": 0}, "Sudoku scorer")
    print("stream5 ok")


def test_overlap_seeds():
    """make_overlap_report on the v2 layout: seed folders, tagged sweep runs, config.json per run."""
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(9)
        pools = {"nq": 30, "mmlu": 20, "advbench": 20, "xstest": 20, "dolly": 20}
        metric = {"nq": "correct", "mmlu": "correct", "advbench": "refused", "xstest": "refused", "dolly": "leak"}
        runs = [("seq", "", {}), ("kl", "", {"grpo": {"kl_beta": 0.01}}), ("kl", "b0.2", {"grpo": {"kl_beta": 0.2}}),
                ("refkl", "", {"grpo": {"refkl_beta": 1.0}}), ("router", "", {}),
                ("tripwire", "", {"tripwire": {"eps": 0.03}}), ("tripwire", "e0.10", {"tripwire": {"eps": 0.1}}),
                ("tripwire_prompt", "", {"tripwire": {"eps": 0.03}}), ("scale", "a0.50", {}),
                ("seq", "L4", {"gate_layer": 4}), ("tripwire", "L4", {"gate_layer": 4, "tripwire": {"eps": 0.03}})]
        for sd in ("s0", "s1"):
            for method, tag, c in runs:
                md = os.path.join(tmp, sd, method + (f"_{tag}" if tag else ""))
                gated = method in ("tripwire", "tripwire_prompt", "router")
                for k in range(2):
                    edir = os.path.join(md, f"stage{k}", "eval")
                    os.makedirs(edir)
                    for pool, n in pools.items():
                        with open(os.path.join(edir, f"{pool}.jsonl"), "w") as f:
                            for i in range(n):
                                fire = {"1": (-2 if rng.random() < 0.4 else None)} if (k and gated) else {}
                                keep = k == 0 or (gated and fire.get("1") is None)
                                row = {"i": i, "out": [1] if keep else [2], "fire": fire,
                                       "m": {metric[pool]: int(rng.random() < 0.5)}}
                                if pool == "xstest":
                                    row["label"] = "safe"
                                f.write(json.dumps(row) + "\n")
                    open(os.path.join(edir, "DONE"), "w").close()
                cfg = {"grpo": {"kl_beta": 0.01, "refkl_beta": 1.0}, "tripwire": {"eps": 0.03}, "gate_layer": 8,
                       "stream": ["nq"]}
                for kk, vv in c.items():
                    if isinstance(vv, dict):
                        cfg[kk].update(vv)
                    else:
                        cfg[kk] = vv
                info = {"method": method, "tag": tag, "config": cfg}
                if method == "scale":
                    info["alpha"] = 0.5
                json.dump(info, open(os.path.join(md, "config.json"), "w"))
                if method.startswith("tripwire"):
                    td = os.path.join(md, "stage1", "tripwire")
                    os.makedirs(td)
                    json.dump({"posthoc": {"new": [rng.random() + 0.5 for _ in range(30)],
                                           "dolly": [rng.random() for _ in range(30)]}}, open(os.path.join(td, "tripwire.json"), "w"))
        paper = os.path.join(tmp, "paper")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_overlap_report.py"), "--runs", tmp,
                            "--paper", paper], capture_output=True, text=True)
        check(r.returncode == 0, "overlap report over seeds failed: " + r.stderr[-2000:])
        if r.returncode == 0:
            nums = open(os.path.join(paper, "results", "overlap_numbers.tex")).read()
            check("\\ovlSeeds}{2}" in nums and "\\ovlTwBParam}{0.1}" in nums and "\\ovlKlBParam}{0.2}" in nums
                  and "\\ovlRouterGain}" in nums and "\\ovlTwpAGain}" in nums, "seed overlap macros: " + nums[:600])
            kl = open(os.path.join(paper, "results", "overlap_frontier_kl.dat")).read().splitlines()
            check(len(kl) == 3, f"KL frontier has two sweep points: {kl}")
            tab = open(os.path.join(paper, "results", "overlap_table.tex")).read()
            check("\\pm" in tab and "0.2" in tab, "seed overlap table shows mean +- sd")
            check("| tripwire 0.03 | 2 |" in r.stdout and "| seq | 2 |" in r.stdout,
                  "runs at another gate layer stay out of the main table: " + r.stdout[:1500])
            sc = open(os.path.join(paper, "results", "overlap_frontier_scale.dat")).read().splitlines()
            check(len(sc) == 3 and sc[1].startswith("0 0"), f"scale frontier starts at the base model: {sc}")
            sp = open(os.path.join(paper, "results", "overlap_split.tex")).read()
            # gate layers 4 and 8 (counted from 0) are printed as blocks 5 and 9 (counted from 1)
            check(sp.count("& 5 &") == 2 and sp.count("& 9 &") == 2, "split-depth table: two methods at blocks 5 and 9")
            check("\\ovlSplitTwLE" in nums and "\\ovlSplitSeqLI" in nums, "split-depth macros")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_overlap_report.py"), "--runs", tmp,
                            "--paper", paper, "--name", "Ifc"], capture_output=True, text=True)
        check(r.returncode == 0 and "\\ovlIfcSeeds}{2}" in open(os.path.join(paper, "results", "overlap_ifc_numbers.tex")).read()
              and os.path.exists(os.path.join(paper, "results", "overlap_ifc_frontier_tw.dat")),
              "named overlap report (--name Ifc): " + r.stderr[-800:])
        print("overlap seeds ok")
    finally:
        shutil.rmtree(tmp)


def test_analyze():
    """analyze.py on a synthetic one-stage overlap run and a synthetic tripwire with calibration files."""
    tmp = tempfile.mkdtemp()
    try:
        rng = random.Random(5)
        pools = {"nq": 40, "dolly": 30, "advbench": 20, "xstest": 20, "mmlu": 20}
        metric = {"nq": "correct", "mmlu": "correct", "advbench": "refused", "xstest": "refused", "dolly": "leak"}
        for method in ("seq", "kl_b0.2", "tripwire", "tripwire_e0.10"):
            gated = method.startswith("tripwire")
            for k in range(2):
                edir = os.path.join(tmp, method, f"stage{k}", "eval")
                os.makedirs(edir)
                for pool, n in pools.items():
                    with open(os.path.join(edir, f"{pool}.jsonl"), "w") as f:
                        for i in range(n):
                            fire = {"1": (3 if rng.random() < (0.6 if pool == "nq" else 0.1) else None)} if (k and gated) else {}
                            keep = k == 0 or (gated and fire.get("1") is None) or (not gated and rng.random() < 0.2)
                            row = {"i": i, "out": [1, 2] if keep else [3], "fire": fire,
                                   "m": {metric[pool]: int(rng.random() < 0.5), "refused": 0}}
                            f.write(json.dumps(row) + "\n")
                open(os.path.join(edir, "DONE"), "w").close()
            json.dump({"config": {"stream": ["nq"]}}, open(os.path.join(tmp, method, "config.json"), "w"))
        tdir = os.path.join(tmp, "tripwire", "stage1", "tripwire")
        os.makedirs(tdir)
        new = [rng.gauss(1.0, 1.0) for _ in range(200)]
        ref = [rng.gauss(0.0, 1.0) for _ in range(200)]
        json.dump({"eps": 0.03, "delta": 0.05, "kappa": 2.0, "components": {"dolly": {}, "advbench": {}},
                   "calibration": {"delta_per_component": 0.025},
                   "posthoc": {"new": new, "dolly": ref, "advbench": [rng.gauss(-1, 1) for _ in range(100)],
                               "mmlu": [rng.gauss(2.5, 0.5) for _ in range(100)]}},
                  open(os.path.join(tdir, "tripwire.json"), "w"))
        json.dump([rng.gauss(0.0, 1.0) for _ in range(300)], open(os.path.join(tdir, "cal_dolly.json"), "w"))
        json.dump([rng.gauss(-1.0, 1.0) for _ in range(150)], open(os.path.join(tdir, "cal_advbench.json"), "w"))
        an = os.path.join(ROOT, "scripts", "analyze.py")
        r = subprocess.run([sys.executable, an, "limit", "--runs", tmp], capture_output=True, text=True)
        check(r.returncode == 0, "analyze limit failed: " + r.stderr[-1500:])
        if r.returncode == 0:
            res = json.load(open(os.path.join(tmp, "analysis", "limit.json")))
            kl = res["kl_dv_lower_bound"]["estimate"]
            # two unit-variance Gaussians one apart: KL = 0.5; a linear test function attains it
            check(0.3 < kl < 0.7, f"analyze limit: DV estimate of KL(N(1,1)||N(0,1)) = 0.5, got {kl:.3f}")
            check(set(res["methods"]) == {"seq", "kl_b0.2", "tripwire", "tripwire_e0.10"}, "analyze limit: all methods")
        for sub in (["fired", "--runs", tmp], ["coverage", "--runs", tmp, "--reps", "300"],
                    ["monitor", "--run", os.path.join(tmp, "tripwire"), "--stage", "1", "--reps", "20", "--horizon", "300"]):
            r = subprocess.run([sys.executable, an] + sub, capture_output=True, text=True)
            check(r.returncode == 0, f"analyze {sub[0]} failed: " + r.stderr[-1500:])
            if sub[0] == "coverage" and r.returncode == 0:
                rates = [float(line.split("|")[5]) for line in r.stdout.splitlines() if line.startswith("| stage1")]
                check(rates and max(rates) <= 0.06, f"analyze coverage: violation rate <= delta per component ({rates})")
            if sub[0] == "monitor" and r.returncode == 0:
                lines = [l for l in r.stdout.splitlines() if l.startswith("| 0.")]
                share0 = float(lines[0].split("|")[2])
                share50 = float(lines[-1].split("|")[2])
                check(share0 <= 0.1 and share50 >= 0.9, f"analyze monitor: no alarms without shift ({share0}), alarms with it ({share50})")
        print("analyze ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_generators()
    test_scorers()
    test_certify()
    test_tables()
    test_overlap_report()
    test_overlap_seeds()
    test_ladder()
    test_stream5()
    test_analyze()
    if FAILS:
        print(f"{len(FAILS)} failures")
        sys.exit(1)
    print("all pure-python tests passed")
