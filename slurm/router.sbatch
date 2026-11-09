#!/bin/bash
# Router baseline (evaluation only, 1 GPU, ~40 min): reuses <exp>/tripwire's adapters with an uncalibrated
# per-prompt gate. Submit after the Tripwire run of the same config has finished.
#SBATCH --job-name=tw-router
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
CONFIG="${CONFIG:-configs/stream_qwen15_v2.yaml}"
python scripts/eval_router.py --config "${CONFIG}"
bash "${REPO}/slurm/report.sh" "${CONFIG}" "" || true
