#!/bin/bash
# Adapter-scaling baseline (scripts/eval_scale.py): a finished ungated run (BASE, default seq) evaluated with its
# adapters multiplied by ALPHA. Evaluation only; resumable per pool. 20 minutes before the time limit the job saves
# and resubmits itself; at the end it submits the next waiting runs of slurm/plan.sh's queue.
#SBATCH --job-name=tw-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@1200
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
guard_duplicate
STOP_FILE="${REPO}/logs/STOP_${SLURM_JOB_ID}"
rm -f "${STOP_FILE}"
trap 'echo "[$(date)] USR1 received: asking the evaluation to save and stop"; touch "${STOP_FILE}"' USR1
echo "[$(date)] job ${SLURM_JOB_ID}: ${CONFIG} seed ${SEED:-0} scale ${BASE:-seq} x ${ALPHA}"
ARGS=(scripts/eval_scale.py --config "${CONFIG}" --base "${BASE:-seq}" --alpha "${ALPHA}" --stop-file "${STOP_FILE}"
      --set "seed=${SEED:-0}" ${EXTRA:-})
launch "${NGPU:-1}" "${LAUNCHER:-deepspeed}" "${ARGS[@]}" &
pid=$!
while true; do
  wait "${pid}"; rc=$?
  if [ ${rc} -gt 128 ] && kill -0 "${pid}" 2>/dev/null; then continue; fi  # wait interrupted by USR1
  break
done
echo "[$(date)] evaluation exited with ${rc}"
if [ ${rc} -eq 3 ] && [ -f "${STOP_FILE}" ] && [ "${NO_RESUBMIT:-0}" != "1" ]; then
  echo "[$(date)] time limit reached: resubmitting (continues with the unfinished pools)"
  BEGIN="" DEPEND="" ALPHAS="${ALPHA}" bash "${REPO}/slurm/submit.sh" scale
  rm -f "${STOP_FILE}"
  exit 0
fi
rm -f "${STOP_FILE}"
if [ ${rc} -ne 0 ]; then
  DIR=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config, run_dir; c = load_config('${CONFIG}', ['seed=${SEED:-0}']); b = '${BASE:-seq}'; a = float('${ALPHA}'); print(run_dir(c, 'scale', f'a{a:.2f}' if b == 'seq' else f'{b}_a{a:.2f}'))")
  mark_failed "${DIR}" "${rc}"
fi
plan_feed
