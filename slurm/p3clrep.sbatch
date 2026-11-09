#!/bin/bash
# Step p3.25 (Nov 5, PLAN_PAPER3.md): every continual-learning baseline on the 3-task streams (1.5B and 7B), CPU
# only, from stored evaluations; no model is loaded. Submit from the repository folder after step p3.25a (EWC and
# A-GEM on the 1.5B stream); finished runs are read as they are:
#
#   sbatch slurm/p3clrep.sbatch      # the archive name is printed at the end of logs/cpup3cl-<id>.out
#
# Writes paper/v3_results/p3_25/stream_<exp>.json, _table.tex, _summary.txt (the predictions) and
# chosen_<exp>.txt (each ungated family's best setting; no step trains more seeds of it).
#SBATCH --job-name=cpup3cl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
out=paper/v3_results/p3_25
mkdir -p "${out}"
{
  for exp in v2_stream_q15 v2_stream_q7; do
    [ -d "runs/${exp}" ] || continue
    python scripts/stream_sweep_report.py --runs "runs/${exp}" --out "${out}" || echo "  (did not finish: ${exp})"
  done
} 2>&1 | tee "${out}/p3_25_report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_25_${stamp}.tar.gz" "${out}"
echo "send me this file: $(pwd)/p3_25_${stamp}.tar.gz ($(du -h "p3_25_${stamp}.tar.gz" | cut -f1))"
