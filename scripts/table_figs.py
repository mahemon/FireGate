"""Fewer appendix tables (Nov 1, paper v53), from the tables the generators write; run after table_style.py:

    python scripts/table_figs.py paper/v2_results

1. Merged tables, in <root>/merged/: tables with the same columns stacked into one, each part under a group row.
     zoo_table.tex          gate comparison on NQ and on Concise-ctx                         (label tab:zoo)
     overlap_concise.tex    overlap tables of Concise-ctx and Concise                        (label tab:overlap_concise)
     overlap_7b.tex         overlap tables of Qwen2.5-7B NQ, OLMo-2-7B NQ and Qwen2.5-7B IF  (label tab:overlap_7b)
2. Figure data, in <root>/figs/: every number of the coverage-sweep and score tables, as pgfplots data files.
     sweep_all.dat, sweep_grid.dat, sweep_eps.dat, sweep_dep.dat            (from v2_overlap_q15/switch/analyses/sweep_table.tex)
     score_{probe,lda,knn,maha}.dat, score_text.dat                         (from v2_overlap_q15/analysis/score_table.tex)
The source tables are left as they are. Idempotent; a missing source skips its output."""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "paper/v2_results"


def read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print("missing", rel)
        return None
    return open(p).read()


def write(rel, text):
    p = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    old = open(p).read() if os.path.exists(p) else None
    if old != text:
        open(p, "w").write(text)
        print("wrote", rel)


def brace(s, i):
    """content of the brace group opening at s[i] == '{', and the index after it"""
    d = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[i + 1:k], k + 1
    raise ValueError("unbalanced braces")


def parts(s):
    """caption, column spec, header and body of a generated table"""
    cap, _ = brace(s, s.index(r"\caption{") + len(r"\caption"))
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", s)
    head_start = s.index("\\toprule\n", m.end()) + len("\\toprule\n")
    head_end = s.index("\n\\midrule\n", head_start)
    body_end = s.index("\n\\bottomrule", head_end)
    sep = re.search(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", s)
    return {"caption": cap, "spec": m.group(1), "ncol": len(re.findall(r"[lcr]", m.group(1))),
            "head": s[head_start:head_end], "body": s[head_end + len("\n\\midrule\n"):body_end],
            "star": r"\begin{table*}" in s, "resize": r"\resizebox" in s,
            "sep": sep.group(0) if sep else r"\setlength{\tabcolsep}{3pt}"}


def merge(out, sources, label, caption, titles, head=lambda h: h):
    ps = []
    for rel in sources:
        s = read(rel)
        if s is None:
            return
        ps.append(parts(s))
    p0 = ps[0]
    assert all(p["ncol"] == p0["ncol"] for p in ps), out
    env = "table*" if any(p["star"] for p in ps) else "table"
    lines = [r"\begin{%s}[t]" % env, r"\centering", r"\small", p0["sep"], r"\caption{%s}" % caption,
             r"\label{%s}" % label]
    if p0["resize"]:
        lines.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    lines += [r"\begin{tabular}{%s}" % p0["spec"], r"\toprule", head(p0["head"]), r"\midrule"]
    for i, (p, t) in enumerate(zip(ps, titles)):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textbf{%s}} \\" % (p0["ncol"], t))
        lines.append(p["body"])
    lines += [r"\bottomrule", r"\end{tabular}}" if p0["resize"] else r"\end{tabular}", r"\end{%s}" % env, ""]
    write(out, "\n".join(lines))


def overlap_head(h):
    h = re.sub(r"^( & & )[^&]+?( & \\multicolumn\{5\})", r"\1New task\2", h)
    return re.sub(r"(& MMLU & )[^&]+?( & Dolly & )", r"\1New\2", h)


def overlap_tail(sources):
    """the common part of the overlap captions, from '; mean' on; the longest one carries the A100 note"""
    tails = []
    for rel in sources:
        s = read(rel)
        if s is None:
            return None
        c = parts(s)["caption"]
        tails.append(c[c.index("; mean"):])
    return max(tails, key=len)


def num(x):
    x = x.replace("$-$", "-").replace("{", "").replace("}", "").strip()
    return float(x)


# ---- merged tables
src = ["v2_overlap_q15/switch/review/zoo_table.tex", "v2_ladder_concisectx_q15/switch/review/zoo_table.tex"]
s0 = read(src[0])
if s0 is not None:
    cap = parts(s0)["caption"].replace("(NQ, Qwen2.5-1.5B-Instruct)",
                                       "(NQ, top, and Concise-ctx, bottom; Qwen2.5-1.5B-Instruct)")
    merge("merged/zoo_table.tex", src, "tab:zoo", cap, ["NQ", "Concise-ctx: Dolly requests with a context passage"])

src = ["v2_ladder_concisectx_q15/results/overlap_concisectx_table.tex",
       "v2_ladder_concise_q15/results/overlap_concise_table.tex"]
tail = overlap_tail(src)
if tail:
    merge("merged/overlap_concise.tex", src, "tab:overlap_concise",
          "New tasks that overlap the reference most, Qwen2.5-1.5B-Instruct, against Dolly as the reference: Concise-ctx, "
          "held-out Dolly requests with a context passage rewarded for short answers, and Concise, plain held-out Dolly "
          "requests rewarded for short answers" + tail,
          ["Concise-ctx: Dolly requests with a context passage", "Concise: plain Dolly requests"], overlap_head)

src = ["v2_overlap_q7/results/overlap_seven_table.tex", "v2_overlap_olmo7/results/overlap_olmo_table.tex",
       "v2_ladder_ifc_q7/results/overlap_ifcseven_table.tex"]
tail = overlap_tail(src)
if tail:
    merge("merged/overlap_7b.tex", src, "tab:overlap_7b",
          "The overlap ladder on 7B models, against Dolly as the reference: Natural Questions asked as plain questions "
          "on Qwen2.5-7B-Instruct and OLMo-2-1124-7B-Instruct, and held-out Dolly requests with one verifiable format "
          "constraint, IF, on Qwen2.5-7B-Instruct" + tail,
          ["Qwen2.5-7B-Instruct, NQ", "OLMo-2-1124-7B-Instruct, NQ", "Qwen2.5-7B-Instruct, IF"], overlap_head)

# ---- coverage sweep
s = read("v2_overlap_q15/switch/analyses/sweep_table.tex")
if s is not None:
    rows = []
    for line in parts(s)["body"].split("\n"):
        c = [x.strip() for x in line.rstrip("\\ ").split(" & ")]
        if len(c) != 7:
            continue
        m = re.match(r"(\S+)\s*\{\\scriptsize\[(.*?),\s*(.*?)\]\}", c[4])
        meas, lo, hi = num(m.group(1)), num(m.group(2)), num(m.group(3))
        kind = "dep" if c[0] == "deployed" else ("eps" if "varepsilon" in c[0] else "grid")
        name = c[0].replace(r"$\varepsilon=", "eps").replace("$", "")
        rows.append((num(c[1]), num(c[2]), num(c[3]), meas, lo, hi, num(c[5]), num(c[6]), kind, name))
    head = "brl beval pred meas lo hi ep em cov dolly name\n"

    def fmt(r):
        brl, bev, pred, meas, lo, hi, cov, dolly, kind, name = r
        return f"{brl} {bev} {pred} {meas} {lo} {hi} {hi - meas:.1f} {meas - lo:.1f} {cov} {dolly} {name}\n"
    write("figs/sweep_all.dat", head + "".join(fmt(r) for r in sorted(rows)))
    for kind in ("grid", "eps", "dep"):
        write(f"figs/sweep_{kind}.dat", head + "".join(fmt(r) for r in rows if r[8] == kind))

# ---- tripwire scores
s = read("v2_overlap_q15/analysis/score_table.tex")
if s is not None:
    by = {}
    for line in parts(s)["body"].split("\n"):
        c = [x.strip() for x in line.rstrip("\\ ").split(" & ")]
        if len(c) != 8:
            continue
        low = c[0].lower()
        key = ("probe" if "logistic" in low else "lda" if "lda" in low else "knn" if "nn" in low
               else "maha" if "mahalanobis" in low else "text" if "gram" in low else None)
        if key is None:
            continue
        by.setdefault(key, []).append([c[1]] + [num(x) for x in c[2:]])
    head = "L auroc f03 f1 f3 dolly mmlu\n"
    for key in ("probe", "lda", "knn", "maha"):
        if key in by:
            write(f"figs/score_{key}.dat", head + "".join(" ".join(str(x) for x in r) + "\n" for r in by[key]))
    if "text" in by:   # no block: a horizontal line across the plot
        r = by["text"][0]
        write("figs/score_text.dat", head + "".join(" ".join(str(x) for x in [L] + r[1:]) + "\n" for L in (3, 23)))
