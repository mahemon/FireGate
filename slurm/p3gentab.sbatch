#!/bin/bash
# The tables of the general-claim runs (slurm/p3_general.sh; Nov 4), CPU only, from stored features and responses.
# Submit from the repository folder once "bash slurm/p3_general.sh status" shows every run and the judge done:
#
#   sbatch slurm/p3gentab.sbatch      # the archive name is printed at the end of logs/cpup3gentab-<id>.out
#
# Runs "bash slurm/plan.sh sft-collect": switch_report.py for every SFT and full-rank experiment and its LoRA
# reference (paper/sft_results/<exp>/switch/), addon_report.py (overview/addon_sft.{md,tex,json} and
# overview/pilot_tabular.tex, the tabular of the paper's Table "Full-rank updates and SFT" with every seed), the run
# list, the training logs, and sft_results_<date>.tar.gz to send. It can be submitted again at any time.
#SBATCH --job-name=cpup3gentab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
MAX_FAILS="${MAX_FAILS:-3}" bash slurm/p3_general.sh status
bash slurm/plan.sh sft-collect
