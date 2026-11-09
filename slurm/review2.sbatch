#!/bin/bash
# Review analyses, round 2 (slurm/plans/review2.sh), as a CPU job on the nova partition: two-sided sensitivity,
# covariance term, stream budget, pooled tables. Reads stored features and outputs only. Submit from the repository
# folder:
#
#   sbatch slurm/review2.sbatch          # the archive name is printed at the end of logs/cpureview2-<id>.out
#
# The job name does not start with tw-, so it never takes a slot of the GPU queue, and the autopilots (cpufam,
# cpusft) do not track it.
#SBATCH --job-name=cpureview2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash slurm/plan.sh review2-collect
