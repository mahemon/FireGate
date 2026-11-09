#!/bin/bash
# Step p3.26 (Nov 8, PLAN_PAPER3.md): the report of the frozen recipe on TriviaQA (Llama-3.2-3B), CPU only, from the
# stored features, runs and responses; no model is loaded. Submit from the repository folder after steps p3.26b and
# p3.26c:
#
#   sbatch slurm/p3trivrep.sbatch      # the archive name is printed at the end of logs/cpup3tq-<id>.out
#
# Writes paper/v3_results/p3_26/: switch/ (scripts/switch_report.py), external/ (p3.12's report on this experiment),
# prospective/ (p3.19's), trivia_summary.txt with P1-P7 as PASS / FAIL, trivia.json, trivia_table.tex,
# trivia_macros.tex; the records PREREG_p3_26*.txt and pretrain_gate.json are already there and go into the archive.
#SBATCH --job-name=cpup3tq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
cfg=configs/p3/trivia_llama3b.yaml
exp=p3_trivia_llama3b
out=paper/v3_results/p3_26
mkdir -p "${out}"
{
  for f in "${out}/PREREG_p3_26.txt" "${out}/PREREG_p3_26_gate.txt" "${out}/pretrain_gate.json"; do
    [ -s "${f}" ] || echo "WARNING: ${f} is missing (python scripts/prereg_record.py --step p3.26)"
  done
  python scripts/switch_report.py --config "${cfg}" --name Trivia --out "${out}/switch" || echo "  (did not finish: switch_report)"
  cp "runs/${exp}/analysis/separability.json" "${out}/switch/" 2>/dev/null || true
  python scripts/external_report.py --exp "${cfg}:Trivia:" --out "${out}/external" || echo "  (did not finish: external_report)"
  python scripts/prospective_report.py --exp "${cfg}:Trivia:" --out "${out}/prospective" || echo "  (did not finish: prospective_report)"
  python scripts/trivia_report.py --config "${cfg}" --out "${out}" || echo "  (did not finish: trivia_report)"
} 2>&1 | tee "${out}/p3_26_report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_26_${stamp}.tar.gz" "${out}"
echo "send me this file: $(pwd)/p3_26_${stamp}.tar.gz ($(du -h "p3_26_${stamp}.tar.gz" | cut -f1))"
