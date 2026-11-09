#!/bin/bash
# Robustness evaluations of one finished run (scripts/eval_extra.py): adversarial task-format wrapping, sampling
# decoding with the coupling check, prompt-format shift. Resumable per pool. At the end the job submits the next
# waiting runs of slurm/plan.sh's queue.
#SBATCH --job-name=tw-extra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
guard_duplicate
ARGS=(scripts/eval_extra.py --config "${CONFIG}" --method "${METHOD}" --what "${WHAT:-all}" --set "seed=${SEED:-0}" ${EXTRA:-})
[ -n "${TAG:-}" ] && ARGS+=(--tag "${TAG}")
rc=0
launch "${NGPU:-1}" "${LAUNCHER:-deepspeed}" "${ARGS[@]}" || rc=$?
echo "[$(date)] evaluations exited with ${rc}"
if [ ${rc} -ne 0 ]; then
  RUN_DIR=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config, run_dir; c = load_config('${CONFIG}', ['seed=${SEED:-0}']); print(run_dir(c, '${METHOD}', '${TAG:-}'))")
  mark_failed "${RUN_DIR}/extra" "${rc}"
fi
plan_feed
