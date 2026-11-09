#!/bin/bash
# Score ablation (scripts/score_ablation.py): calibrated prompt-time tripwire scores of the base model, no training.
# Writes runs/<exp>/analysis/score_ablation.json and score_table.tex. At the end the job submits the next waiting
# runs of slurm/plan.sh's queue.
#SBATCH --job-name=tw-ablate
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
python scripts/score_ablation.py --config "${CONFIG}" --layers "${LAYERS:-4,8,14,20}" ${EXTRA:+--set ${EXTRA}} || rc=$?
echo "[$(date)] score ablation exited with ${rc}"
if [ ${rc} -ne 0 ]; then
  OUT=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config; c = load_config('${CONFIG}'); print(c['out_dir'] + '/' + c['exp'] + '/analysis')")
  mkdir -p "${OUT}"
  mark_failed "${OUT}" "${rc}"
fi
plan_feed
