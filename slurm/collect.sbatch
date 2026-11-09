#!/bin/bash
# "bash slurm/plan.sh collect" as a CPU job on the nova partition: every table, figure and analysis (with the v6
# review analyses this takes longer than a login-node session should). Submit from the repository folder:
#
#   sbatch slurm/collect.sbatch          # the archive name is printed at the end of logs/cpucollect-<id>.out
#
# The job name does not start with tw-, so it never takes a slot of the GPU queue (slurm/plan.sh counts tw-* jobs).
#SBATCH --job-name=cpucollect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash slurm/plan.sh collect
