#!/bin/bash
# Paper v3 tables (PLAN_PAPER3.md) as a CPU job on the nova partition: scripts/v3_analyses.py was killed for memory on
# the login node. Reads stored features and finished runs only. Submit from the repository folder:
#
#   sbatch slurm/p3collect.sbatch        # the archive name is printed at the end of logs/cpup3collect-<id>.out
#
# The job name does not start with tw-, so it never takes a slot of the GPU queue, and the autopilot does not track it.
#SBATCH --job-name=cpup3collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash slurm/plan.sh p3-collect
