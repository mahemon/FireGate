#!/bin/bash
# LLM judge of every changed response of an experiment (scripts/judge_changes.py; evaluation only). Labels are
# cached per run, so a job that hits its time limit is continued by submitting the step again. Writes
# runs/<exp>/analysis/judge/ (JUDGE_TAG=olmo: analysis/judge_olmo/, a second judge). At the end the job submits the
# next waiting runs of slurm/plan.sh's queue.
#SBATCH --job-name=tw-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
guard_duplicate
rc=0
python scripts/judge_changes.py --config "${CONFIG}" --judge "${JUDGE_MODEL:-models/Qwen2.5-7B-Instruct}" \
  ${JUDGE_TAG:+--tag ${JUDGE_TAG}} ${JUDGE_ARGS:-} || rc=$?
echo "[$(date)] judge exited with ${rc}"
if [ ${rc} -ne 0 ]; then
  OUT=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config; c = load_config('${CONFIG}'); print(c['out_dir'] + '/' + c['exp'] + '/analysis/judge${JUDGE_TAG:+_${JUDGE_TAG}}')")
  mark_failed "${OUT}" "${rc}"
fi
plan_feed
