#!/bin/bash
# Unit checks, a tiny end-to-end run of every method, real-model speed / bf16 exactness. With NGPU>=2 also the
# multi-GPU checks (distributed results equal single-GPU results).
#SBATCH --job-name=tw-selftest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
nvidia-smi --query-gpu=name,memory.total --format=csv
LEVEL="${LEVEL:-all}"                       # LEVEL=real: only the real-model checks (e.g. a new 7B model)
[ "${NGPU:-1}" -ge 2 ] && [ "${LEVEL}" = all ] && LEVEL=all+dist
rc=0
python scripts/selftest.py --model "${MODEL_DIR:-models/Qwen2.5-1.5B-Instruct}" --data data \
  --config "${CONFIG:-configs/v2/stream_q15.yaml}" --level "${LEVEL}" --launcher "${LAUNCHER:-deepspeed}" || rc=$?
plan_feed     # a free slot: submit waiting runs of slurm/plan.sh's queue (if any)
exit ${rc}
