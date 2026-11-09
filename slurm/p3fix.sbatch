#!/bin/bash
# Step p3.11 (Nov 2, PLAN_PAPER3.md): the fixes for the weak cells of p3.10, CPU only, from stored features (no GPU,
# no training). One experiment per process, then the tables and an archive to send. Submit from the repository folder:
#
#   sbatch slurm/p3fix.sbatch        # the archive name is printed at the end of logs/cpup3fix-<id>.out
#
# The job name does not start with tw-, so it never takes a slot of the GPU queue, and the autopilot does not track it.
#SBATCH --job-name=cpup3fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
out=paper/v3_results/p3_11
mkdir -p "${out}"
for spec in configs/v2/overlap_q15.yaml:Nq: \
            configs/v3/squad2_q15.yaml:Squad:advbench,general \
            configs/v2/ladder_concise_q15.yaml:Concise:advbench,general \
            configs/v2/ladder_concisectx_q15.yaml:Concisectx:advbench,general; do
  python scripts/score_fixes.py --exp "${spec}" --out "${out}" || echo "  (did not finish: ${spec})"
done
python scripts/score_fixes.py --merge --out "${out}" 2>&1 | tee "${out}/report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_11_${stamp}.tar.gz" "${out}" "logs/cpup3fix-${SLURM_JOB_ID:-local}.out" 2>/dev/null || tar czf "p3_11_${stamp}.tar.gz" "${out}"
echo "send me this file: $(pwd)/p3_11_${stamp}.tar.gz ($(du -h "p3_11_${stamp}.tar.gz" | cut -f1))"
