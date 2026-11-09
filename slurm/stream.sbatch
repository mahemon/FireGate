#!/bin/bash
# One method of one experiment over the whole stream, on NGPU GPUs (set by slurm/submit.sh).
# 20 minutes before the time limit SLURM sends USR1: the run saves its progress (training checkpoint, finished
# evaluation pools, tripwire files) and stops, and this job resubmits itself; the new job continues from the last
# checkpoint. When the run is finished (or failed), the job submits the next waiting runs of slurm/plan.sh's queue.
# After a crash, a cancel or the daily restart, run the same step command again: the run continues from what was
# saved.
#SBATCH --job-name=tw-stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
guard_duplicate
CONFIG="${CONFIG}"; METHOD="${METHOD}"; SEED="${SEED:-0}"; TAG="${TAG:-}"; EXTRA="${EXTRA:-}"
NGPU="${NGPU:-1}"; LAUNCHER="${LAUNCHER:-deepspeed}"
STOP_FILE="${REPO}/logs/STOP_${SLURM_JOB_ID}"
rm -f "${STOP_FILE}"
trap 'echo "[$(date)] USR1 received: asking the run to save and stop"; touch "${STOP_FILE}"' USR1
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "[$(date)] job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0}: ${CONFIG} seed ${SEED} ${METHOD}${TAG:+ tag ${TAG}} on ${NGPU} GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})"

ARGS=(scripts/run_stream.py --config "${CONFIG}" --method "${METHOD}" --stop-file "${STOP_FILE}" --set "seed=${SEED}" ${EXTRA})
[ -n "${TAG}" ] && ARGS+=(--tag "${TAG}")
launch "${NGPU}" "${LAUNCHER}" "${ARGS[@]}" &
pid=$!
while true; do
  wait "${pid}"; rc=$?
  if [ ${rc} -gt 128 ] && kill -0 "${pid}" 2>/dev/null; then continue; fi  # wait interrupted by USR1
  break
done
echo "[$(date)] run exited with ${rc}"

RUN_DIR=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config, run_dir; c = load_config('${CONFIG}', ['seed=${SEED}'] + '${EXTRA}'.split()); print(run_dir(c, '${METHOD}', '${TAG}'))")
if [ -f "${RUN_DIR}/DONE" ]; then
  echo "[$(date)] ${RUN_DIR} finished"
  bash "${REPO}/slurm/report.sh" "${CONFIG}" "seed=${SEED} ${EXTRA}" || true
  rm -f "${STOP_FILE}"
  plan_feed
elif [ -f "${STOP_FILE}" ] && [ "${NO_RESUBMIT:-0}" != "1" ]; then
  echo "[$(date)] time limit reached: resubmitting ${METHOD}${TAG:+_${TAG}} (continues from the last checkpoint)"
  BEGIN="" DEPEND="" METHODS="${METHOD}" bash "${REPO}/slurm/submit.sh" stream
  rm -f "${STOP_FILE}"
else
  echo "[$(date)] ${RUN_DIR} is not finished and no stop was requested: the run failed (exit ${rc}); see the log above"
  mark_failed "${RUN_DIR}" "${rc}"
  rm -f "${STOP_FILE}"
  plan_feed
fi
