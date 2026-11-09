#!/bin/bash
# Step p3.23 (Nov 5, PLAN_PAPER3.md), job 2 of 2 (CPU): the report, from stored features and responses. Submitted by
# slurm/p3_widen.sh after job 1 (afterok), or on its own from the repository folder:
#
#   sbatch slurm/p3widenrep.sbatch      # the archive name is printed at the end of logs/cpup3widen-<id>.out
#
# Writes paper/v3_results/p3_23/widen.json, widen_table.tex, widen_summary.txt (the predictions) and packs them, with
# the pre-registration record, into p3_23_<date>.tar.gz.
#SBATCH --job-name=cpup3widen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
out=paper/v3_results/p3_23
mkdir -p "${out}"
python scripts/widen_report.py --config configs/v2/ladder_concise_q15.yaml --protect advbench,general --out "${out}" \
  2>&1 | tee "${out}/widen_report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_23_${stamp}.tar.gz" "${out}"
echo "send me this file: $(pwd)/p3_23_${stamp}.tar.gz ($(du -h "p3_23_${stamp}.tar.gz" | cut -f1))"
