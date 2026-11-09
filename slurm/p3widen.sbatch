#!/bin/bash
# Step p3.23 (Nov 5, PLAN_PAPER3.md), job 1 of 2 (GPU): Concise with a wide harmful component. Submitted by
# slurm/p3_widen.sh, which sets the GPU model the stored features were made on (gate_features.py --add refuses any
# other). No training:
#   1. gate_features.py --add: trunk features of the two new sets (data/wd_aya.jsonl, data/wd_dna.jsonl)
#   2. eval_extra.py --what widen, for Seq-GRPO of every finished seed: greedy responses of the trained and the base
#      model on the English harmful test rows (and --what external if its files are missing)
# Can be submitted again: stored features and responses are kept, only what is missing is computed.
#SBATCH --job-name=p3widen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -u
CFG="${CFG:-configs/v2/ladder_concise_q15.yaml}"
EXP=$(grep -E '^exp:' "${CFG}" | awk '{print $2}')
JS="runs/${EXP}/analysis/gate_features.json"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DEV=$(grep -o '"device": "[^"]*"' "${JS}" 2>/dev/null | head -1 | cut -d'"' -f4)
echo "[$(date)] p3.23 GPU job ${SLURM_JOB_ID:-}: ${CFG} on ${GPU_NAME} (features stored on ${DEV:-?})"
rc=0
if { grep -q '"eval__wd_aya"' "${JS}" && grep -q '"eval__wd_dna"' "${JS}"; } 2>/dev/null; then
  echo "done     ${JS} (new sets)"
elif [ "${DEV}" != "${GPU_NAME}" ]; then
  echo "FAILED   ${JS}: stored on ${DEV}, this job has ${GPU_NAME}; bash slurm/p3_widen.sh status lists the GPU model"
  rc=1
else
  python scripts/gate_features.py --config "${CFG}" --add || rc=$?
fi
for s in 0 1 2; do
  d="runs/${EXP}/s${s}/seq"
  [ -f "${d}/DONE" ] || { echo "SKIPPED  ${d}: not finished"; continue; }
  if grep -q '"widen"' "${d}/extra/summary.json" 2>/dev/null && grep -q '"external"' "${d}/extra/summary.json"; then
    echo "done     ${d}/extra/widen"; continue
  fi
  echo "[$(date)] widen ${d}"
  python scripts/eval_extra.py --config "${CFG}" --method seq --what widen --set "seed=${s}" || rc=$?
done
echo "[$(date)] p3.23 GPU job finished (exit ${rc})"
exit ${rc}
