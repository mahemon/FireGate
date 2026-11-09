#!/bin/bash
# Trunk features of every prompt a gate is built from or evaluated on (scripts/gate_features.py; no training).
# Writes runs/<exp>/analysis/gate_features.npz / .json (FEATURES_ARGS=--add: only the prompt sets not stored yet) and
# separability.json (scripts/separability.py); first writes missing ladder-task data (scripts/prepare_ladder.py).
# At the end the job submits the next waiting runs of slurm/plan.sh's queue.
#SBATCH --job-name=tw-features
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
# task data of the ladder tasks that do not exist yet (v6: concisectx); never rewrites an existing file
python scripts/prepare_ladder.py --data "$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config; print(load_config('${CONFIG}')['data_dir'])")" --only-missing \
  || echo "[$(date)] prepare_ladder.py failed (only needed for the ladder tasks)"
python scripts/gate_features.py --config "${CONFIG}" ${EXTRA:+--set ${EXTRA}} ${FEATURES_ARGS:-} || rc=$?
echo "[$(date)] gate features exited with ${rc}"
if [ ${rc} -eq 0 ]; then   # separability of the new task(s) before any training (runs/<exp>/analysis/separability.json)
  python scripts/separability.py --config "${CONFIG}" ${EXTRA:+--set ${EXTRA}} || echo "[$(date)] separability.py failed"
fi
if [ ${rc} -ne 0 ]; then
  OUT=$(python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config; c = load_config('${CONFIG}'); print(c['out_dir'] + '/' + c['exp'] + '/analysis/gate_features')")
  mark_failed "${OUT}" "${rc}"
fi
plan_feed
