#!/bin/bash
# Inference cost of the gate (scripts/latency_bench.py): latency, throughput and memory of the base model, gated
# LoRA adapters of a K-stage stream (K = 1 ... 20), a full-rank gated update and the prompt-time switch between two
# whole models; Qwen2.5-1.5B, then Qwen2.5-7B. One GPU, no training, about 1.5-2.5 h. Submit by hand from the
# repository folder (it waits for a free GPU like any other job; its name does not start with tw-, so it takes no
# slot of the autopilots' queue). No checkpoint: it needs its full time limit in one go.
#
#   sbatch --partition=<gpu partition> --gres=gpu:a100:1 --mem=96G --cpus-per-task=8 \
#          --time=03:00:00 slurm/latency.sbatch
#
# Results: paper/v2_results/latency/latency_{small,seven}{.json,_table.tex,_numbers.tex}; log logs/latbench-<id>.out
#SBATCH --job-name=latbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO
cd "${REPO}"
source "${REPO}/slurm/common.sh"
activate_env
nvidia-smi --query-gpu=name,memory.total --format=csv
rc=0
python scripts/latency_bench.py --config configs/v2/overlap_q15.yaml --name Small || rc=$?
python scripts/latency_bench.py --config configs/v2/overlap_q7.yaml --name Seven --repeats 2 --ks 1,5,20 || rc=$?
echo "[$(date)] latency benchmark exited with ${rc}"
exit ${rc}
