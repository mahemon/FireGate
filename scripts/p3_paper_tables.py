#!/usr/bin/env python
"""The paper's tables of steps p3.9, p3.12, p3.14 and p3.15 (Appendix K of paper v4.5+; Nov 3-4), CPU, seconds.

    python scripts/p3_paper_tables.py --results paper/v3_results --out paper/v3_results/p3_paper

Reads the reports' JSON files (p3_12/external.json, p3_9/recommended.json, p3_15/budget_*.json, p3_14/np_gap.json)
and writes external_table.tex (tab:external), famrec_table.tex (tab:famrec), matched_table.tex (tab:matched, NQ and
SQuAD 2.0 in one table), npgap_table.tex (tab:npgap) and matched_npgap_table.tex (the last two side by side in one
float, as the paper uses them), in the paper's wording. The Overleaf folder is v2_results/p3/.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join("paper", "v3_results"))
ap.add_argument("--out", default=os.path.join("paper", "v3_results", "p3_paper"))
args = ap.parse_args()
base, out = args.results, args.out
os.makedirs(out, exist_ok=True)
res = json.load(open(f'{base}/p3_12/external.json'))
names = {'v2_ladder_concise_q15': 'Concise, Qwen2.5-1.5B', 'v2_ladder_concisectx_q15': 'Concise-ctx, Qwen2.5-1.5B',
         'v3_squad2_q15': 'SQuAD~2.0, Qwen2.5-1.5B', 'v2_overlap_q15': 'NQ, Qwen2.5-1.5B', 'v2_ladder_ifc_q15': 'IF, Qwen2.5-1.5B',
         'v2_ladder_ifc_q7': 'IF, Qwen2.5-7B', 'v2_stream_q15': '3 tasks, Qwen2.5-1.5B', 'v2_overlap_olmo7': 'NQ, OLMo-2-7B',
         'fam_nq_llama3b': 'NQ, Llama-3.2-3B'}
gname = {'protect_remedy': 'protect + passages', 'twosided': 'two-sided', 'twosided_remedy': 'two-sided + remedy'}
sets = ['ext_hb', 'ext_hb_ctx', 'ext_sr']
def pct(x): return f'{100*x:.1f}'
def pts(x):
    v = 100*x
    s = f'{v:+.1f}'
    if s in ('-0.0',): s = '+0.0'
    return s.replace('-', '$-$')
L = []
L.append(r'''\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{Benchmarks we did not write, every setting frozen: HarmBench (HB, 200 behaviours), its contextual
behaviours (HB ctx., 100, a passage then a request) and StrongREJECT (SR, 288). Firing (\%), before training, of the
gate chosen by the steps of \cref{sec:method}, with its 95\% Clopper--Pearson bound, and of the deployed gate.
Refusal: change of Seq-GRPO's refusal rate (points, seed 0), ungated and behind the chosen gate.}
\label{tab:external}
\begin{tabular}{l l rrr rr rrr rrr}
\toprule
 & & \multicolumn{3}{c}{chosen gate fires (\%)} & \multicolumn{2}{c}{deployed fires (\%)} & \multicolumn{3}{c}{refusal, ungated} & \multicolumn{3}{c}{refusal, behind chosen} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
Experiment & Chosen gate & HB & HB ctx. & SR & HB ctx. & SR & HB & HB ctx. & SR & HB & HB ctx. & SR \\
\midrule''')
for r in res:
    g = r['gates'][r['chosen']]; d = r['gates']['ratio']
    row = [names[r['exp']], gname[r['chosen']]]
    row += [f"{pct(g['fire'][s])} [{pct(g['cp'][s])}]" for s in sets]
    row += [pct(d['fire'][s]) for s in ('ext_hb_ctx', 'ext_sr')]
    sq = {s: r['seq'][s]['s0'] for s in sets}
    row += [pts(sq[s]['ungated'] - sq[s]['base']) for s in sets]
    row += [pts(sq[s].get(r['chosen'], sq[s]['base']) - sq[s]['base']) for s in sets]
    L.append(' & '.join(row) + r' \\')
L.append(r'''\bottomrule
\end{tabular}
\end{table*}''')
open(f'{out}/external_table.tex', 'w').write('\n'.join(L) + '\n')

# step p3.15: every method at a matched budget, NQ and SQuAD 2.0 in one table
lab = {'seq': 'Seq-GRPO', 'scale': 'Seq-GRPO, LoRA $\\times\\alpha$', 'kl': '+ KL', 'refkl': '+ Ref-KL', 'refkl_s': '+ Ref-KL, 4 samples',
       'nullspace': '+ Null-space', 'nullspace_rf': '+ Null-space, refreshed', 'ewc': '+ EWC', 'agem': '+ A-GEM',
       'router': 'Router (uncalibrated)', 'tripwire_prompt': '\\methodp{}', 'tripwire_prompt_seq': '\\methodp{}, whole sequence',
       'tripwire': '\\methoda{}', 'refkl_protect': '+ Ref-KL, protected components', 'safemix': '+ Safety replay',
       'tripwire_prompt_protect': '\\methodp{}, protect'}
orders = {'NQ': ['seq', 'scale', 'kl', 'refkl', 'refkl_s', 'nullspace', 'nullspace_rf', 'ewc', 'agem', 'router',
                 'tripwire_prompt', 'tripwire_prompt_seq', 'tripwire'],
          'SQuAD~2.0': ['seq', 'scale', 'kl', 'refkl_protect', 'safemix', 'tripwire_prompt', 'tripwire_prompt_protect']}
CAP_MATCHED = r"""Every method against the share of held-out Dolly responses it changes (Qwen2.5-1.5B, three seeds),
each setting chosen on one half of the evaluation prompts and reported on the other. Lowest: smallest Dolly change;
within 10\%: largest gain changing at most 10\%; best: largest gain. \#: settings tried; Chg.: Dolly responses
changed (\%); gain in points; --: none within 10\%. The protect gate does not certify Dolly."""
TAB_MATCHED = [r"\begin{tabular}{l r rr rr rr}", r"\toprule",
               r" & & \multicolumn{2}{c}{lowest} & \multicolumn{2}{c}{within 10\%} & \multicolumn{2}{c}{best} \\",
               r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
               r"Method & \# & Chg. & Gain & Gain & Chg. & Gain & Chg. \\"]
for f, title in (('budget_v2_overlap_q15.json', 'NQ'), ('budget_v3_squad2_q15.json', 'SQuAD~2.0')):
    d = json.load(open(f'{base}/p3_15/{f}'))
    ch = d['chosen']
    TAB_MATCHED.append(r'\midrule')
    TAB_MATCHED.append(r'\multicolumn{8}{l}{\emph{' + title + r'}} \\')
    for m in orders[title]:
        ps = [p for p in d['points'] if p['family'] == m]
        low = min(ps, key=lambda p: p['chg_val']['mean'])  # chosen on the first half, as the budgets are
        cells = [str(len(ps)), pct(low['chg_test']['mean']), pts(low['gain_test']['mean'])]
        for b in ('0.1', '1.0'):
            v = ch.get(b, {}).get(m)
            cells += ['--', '--'] if v is None else [pts(v['gain_test']['mean']), pct(v['chg_test']['mean'])]
        TAB_MATCHED.append(lab[m] + ' & ' + ' & '.join(cells) + r' \\')
TAB_MATCHED += [r'\bottomrule', r'\end{tabular}']

# the synthetic study of p3.14: rows scenario x n, columns d = 32 and d = 256
g = json.load(open(f'{base}/p3_14/np_gap.json'))
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
cells = {(c['scenario'], c['d'], c['n_new']): c for c in g['cells']}
CAP_NPGAP = r"""The gate against a known limit (synthetic Gaussian features, 20 repeats per cell, $\varepsilon=0.03$,
$\delta=0.05$): coverage (\%) of the new task. Limit: $\beta^\ast$ of \cref{prop:percomp}; classifier: the paper's
gate, PAC-calibrated; exact: the same score at the threshold whose true firing is $\varepsilon$; plug-in: the
estimated Gaussian likelihood ratio, PAC-calibrated. $d$: dimension; $n$: new-task fitting prompts."""
TAB_NPGAP = [r"\begin{tabular}{l r rrrr rrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{$d=32$} & \multicolumn{4}{c}{$d=256$} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"Scenario & $n$ & Limit & Class. & Exact & Plug-in & Limit & Class. & Exact & Plug-in \\"]
for sc in ('shift', 'spread', 'overlap'):
    TAB_NPGAP.append(r'\midrule')
    for n in (100, 300, 1000, 3000):
        row = [sc if n == 100 else '', str(n)]
        for dd in (32, 256):
            c = cells[(sc, dd, n)]
            row += [pct(c['np']), pct(mean(c['probe'])), pct(mean(c['probe_oracle'])), pct(mean(c['plugin']))]
        TAB_NPGAP.append(' & '.join(row) + r' \\')
TAB_NPGAP += [r'\bottomrule', r'\end{tabular}']


def one_table(cap, label, tab, sep):
    return "\n".join([r"\begin{table}[t]", r"\centering", r"\scriptsize", rf"\setlength{{\tabcolsep}}{{{sep}}}",
                      r"\caption{" + cap + "}", rf"\label{{{label}}}"] + tab + [r"\end{table}"]) + "\n"


open(f'{out}/matched_table.tex', 'w').write(one_table(CAP_MATCHED, "tab:matched", TAB_MATCHED, "3pt"))
open(f'{out}/npgap_table.tex', 'w').write(one_table(CAP_NPGAP, "tab:npgap", TAB_NPGAP, "3.2pt"))
# both side by side in one float (Appendix K.3-K.4 of the paper), which saves a third of a page
open(f'{out}/matched_npgap_table.tex', 'w').write("\n".join(
    [r"\begin{table}[t]", r"\centering", r"\scriptsize",
     r"\begin{minipage}[t]{0.535\linewidth}", r"\centering", r"\setlength{\tabcolsep}{2.2pt}",
     r"\caption{" + CAP_MATCHED + "}", r"\label{tab:matched}"] + TAB_MATCHED +
    [r"\end{minipage}\hfill", r"\begin{minipage}[t]{0.445\linewidth}", r"\centering", r"\setlength{\tabcolsep}{1.6pt}",
     r"\caption{" + CAP_NPGAP + "}", r"\label{tab:npgap}"] + TAB_NPGAP +
    [r"\end{minipage}", r"\end{table}"]) + "\n")

rec = json.load(open(f'{base}/p3_9/recommended.json'))
T4 = [r'''\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{The models of \cref{tab:fam} behind the gate of steps (3) and (4) of \cref{sec:method} (two-sided; with the
remedy on IF) and behind the deployed gate, $\varepsilon=0.03$, three seeds. $\beta$: firing on RL prompts (\%); Kept:
gain behind the gate (points); Chg.: Dolly responses changed (\%); CP: 95\% bound of the held-out Dolly firing (\%);
MMLU, JB: firing on MMLU and jailbreak templates (\%).}
\label{tab:famrec}
\begin{tabular}{l l r rrrrr rrrrrr}
\toprule
 & & & \multicolumn{5}{c}{deployed gate} & \multicolumn{6}{c}{two-sided gate, with the remedy on IF} \\
\cmidrule(lr){4-8}\cmidrule(lr){9-14}
Model & Task & $\bar G$ & $\beta$ & Kept & CP & MMLU & JB & $\beta$ & Kept & Chg. & CP & MMLU & JB \\
\midrule''']
for r in rec:
    dg, rg = r['gates']['deployed'], r['gates']['recommended']
    row = [r['model'], r['task'], pts(r['gain'])]
    row += [pct(dg['beta']), pts(dg['kept']), pct(dg['cp_dolly']), pct(dg['mmlu']), pct(dg['jb'])]
    row += [pct(rg['beta']), pts(rg['kept']), pct(rg['chg']), pct(rg['cp_dolly']), pct(rg['mmlu']), pct(rg['jb'])]
    T4.append(' & '.join(row) + r' \\')
T4.append(r'''\bottomrule
\end{tabular}
\end{table}''')
open(f'{out}/famrec_table.tex', 'w').write('\n'.join(T4) + '\n')
