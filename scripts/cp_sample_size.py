"""How many prompts a certificate needs (no GPU, no data).

Two numbers, both from the binomial distribution only:
  * held-out prompts n such that the one-sided 95% Clopper--Pearson bound of Proposition 3.6 falls below a level with
    probability at least 1/2 (the typical bound) and 0.9, for a given true firing rate;
  * calibration prompts per reference component for Theorem 3.7 to return a finite threshold,
    n >= log(delta/C) / log(1 - eps), for a single stage and for one budget split over K stages (eps/K per stage).

Writes paper/v2_results/overview/cp_sample_size.tex (appendix table) and cp_sample_size_numbers.tex (macros).
Usage: python scripts/cp_sample_size.py [--out paper/v2_results/overview]
"""
import argparse
import math
import os

import numpy as np
from scipy import stats


def cp_upper(x, n, conf=0.95):
    """one-sided Clopper--Pearson upper limit for x firings in n prompts"""
    x = np.asarray(x)
    u = stats.beta.ppf(conf, x + 1, n - x)
    return np.where(x >= n, 1.0, u)


def prob_below(n, p, level, conf=0.95):
    """probability over X ~ Bin(n, p) that the CP bound is at most level (vectorised over n).

    CP(x, n) <= level iff Pr[Bin(n, level) <= x] <= 1 - conf, so the bound is at most level exactly when X is at most
    x_max(n) = the largest such x."""
    n = np.asarray(n)
    xs = stats.binom.ppf(1 - conf, n, level)              # smallest x with cdf >= 1 - conf
    xs = np.where(stats.binom.cdf(xs, n, level) <= 1 - conf, xs, xs - 1)
    return np.where(xs >= 0, stats.binom.cdf(xs, n, p), 0.0)


def n_needed(p, level, want, conf=0.95, nmax=20000):
    """smallest n from which on Pr[CP <= level] stays at least want (the probability is not monotone in n because
    the binomial is a lattice, so a single lucky n is not enough)"""
    if p >= level:
        return None
    ns = np.arange(1, nmax + 1)
    ok = prob_below(ns, p, level, conf) >= want
    if not ok[-1]:
        return None
    bad = np.nonzero(~ok)[0]
    return 1 if len(bad) == 0 else int(ns[bad[-1]] + 1)


def n_calib(eps, delta=0.05, comps=3):
    """calibration prompts per component for a finite threshold in Theorem 3.7 (Dolly, harmful and task-like
    components, as in every single-task gate and in the stream budget of the paper; 4 before paper v3.4)"""
    return math.ceil(math.log(delta / comps) / math.log(1 - eps))


def fmt(n):
    return "--" if n is None else f"{n:,}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/v2_results/overview")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rates = (0.0, 0.01, 0.02)
    levels = (0.03, 0.05, 0.10)
    rows, nums = [], {}
    for p in rates:
        cells = []
        for lv in levels:
            n50, n90 = n_needed(p, lv, 0.5), n_needed(p, lv, 0.9)
            cells.append(f"{fmt(n50)} & {fmt(n90)}")
            key = f"cpN{['Zero', 'One', 'Two'][rates.index(p)]}{['Three', 'Five', 'Ten'][levels.index(lv)]}"
            nums[key + "Typ"], nums[key + "Ninety"] = fmt(n50), fmt(n90)
        rows.append(f"{100 * p:.0f} & " + " & ".join(cells) + r" \\")
    # what our held-out pools give at the observed NQ rate (13 of 400)
    nums["cpNqObserved"] = f"{100 * float(cp_upper(13, 400)):.1f}"
    nums["cpNqAtSixteen"] = f"{100 * float(cp_upper(52, 1600)):.1f}"
    # calibration sizes
    for k, name in ((1, "One"), (3, "Three"), (10, "Ten")):
        nums[f"cpCalib{name}"] = fmt(n_calib(0.03 / k))
    tab = [r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Held-out prompts a certificate needs. For a tripwire whose true firing rate on a distribution is "
           r"$p$, the number $n$ of i.i.d.\ prompts from it for which the one-sided 95\% Clopper--Pearson bound of "
           r"\cref{prop:cp} is at most the level shown with probability $1/2$ (typical) and $0.9$. Calibration "
           r"(\cref{thm:calib}, $\delta=0.05$, three components) needs at least " + nums["cpCalibOne"] +
           r" prompts per component for a finite threshold at $\varepsilon=0.03$, " + nums["cpCalibThree"] +
           r" for one budget over three stages and " + nums["cpCalibTen"] + r" over ten. Computed by "
           r"\texttt{scripts/cp\_sample\_size.py}.}",
           r"\label{tab:cpsize}",
           r"\begin{tabular}{r cc cc cc}", r"\toprule",
           r" & \multicolumn{2}{c}{bound $\le$ 3\%} & \multicolumn{2}{c}{bound $\le$ 5\%} & "
           r"\multicolumn{2}{c}{bound $\le$ 10\%} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"$p$ (\%) & typical & 0.9 & typical & 0.9 & typical & 0.9 \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(a.out, "cp_sample_size.tex"), "w").write("\n".join(tab) + "\n")
    open(os.path.join(a.out, "cp_sample_size_numbers.tex"), "w").write(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(nums.items())))
    for k, v in sorted(nums.items()):
        print(k, v)


if __name__ == "__main__":
    main()
