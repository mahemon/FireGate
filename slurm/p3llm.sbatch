#!/bin/bash
# Step p3.22 (Nov 4, PLAN_PAPER3.md): an LLM as the prompt-time gate. One GPU job: a judge model scores every stored
# prompt of each experiment once (scripts/llm_gate_scores.py), then the scores are calibrated and reported like any
# other score class (scripts/score_classes.py --scores llm) and merged (scripts/llm_gate_grid.py). No training.
# Submit from the repository folder (any GPU model: the scores do not depend on the stored features' GPU):
#
#   sbatch slurm/p3llm.sbatch                      # the archive name is printed at the end of logs/p3llm-<id>.out
#   JUDGE=models/Qwen2.5-7B-Instruct sbatch slurm/p3llm.sbatch
#
# Can be submitted again: scores are cached per prompt and finished reports are skipped.
#SBATCH --job-name=p3llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
JUDGE="${JUDGE:-models/Qwen2.5-7B-Instruct}"
out=paper/v3_results/p3_22
mkdir -p "${out}"
ok() { "$@" || echo "  (did not finish: $*)"; }
# the 13 model-task pairs of p3.20 (deployed reference), then Concise-ctx and SQuAD 2.0 (also protect + passages)
CONFIGS=(configs/v2/ladder_ifc_q15.yaml configs/v2/overlap_q15.yaml configs/v2/ladder_ifc_q7.yaml
         configs/v2/overlap_q7.yaml configs/v2/overlap_olmo7.yaml configs/fam/ifc_llama3b.yaml configs/fam/nq_llama3b.yaml
         configs/fam/ifc_llama8.yaml configs/fam/nq_llama8.yaml configs/fam/ifc_ds7.yaml configs/fam/nq_ds7.yaml
         configs/fam/ifc_mistral7.yaml configs/fam/nq_mistral7.yaml
         configs/v2/ladder_concisectx_q15.yaml configs/v3/squad2_q15.yaml)
{
  echo "== p3.22 judge ${JUDGE} on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  for c in "${CONFIGS[@]}"; do
    e=$(grep -E '^exp:' "${c}" | awk '{print $2}')
    echo "-- ${e}"
    ok python scripts/llm_gate_scores.py --config "${c}" --judge "${JUDGE}"
    feat="runs/${e}/analysis/llm_scores.npz"
    [ -f "${feat}" ] || continue
    variants=(deployed)
    case "${e}" in v2_ladder_concisectx_q15|v3_squad2_q15) variants+=(protect_remedy) ;; esac
    for v in "${variants[@]}"; do
      [ -f "${out}/${e}/${v}/scores.json" ] && { echo "  done ${e} ${v}"; continue; }
      prot=(); [ "${v}" = protect_remedy ] && prot=(--protect advbench,general)
      ok python scripts/score_classes.py --config "${c}" --variant "${v}" ${prot[@]+"${prot[@]}"} --scores llm \
        --features "${feat}" --out "${out}/${e}/${v}"
    done
  done
  ok python scripts/llm_gate_grid.py --root "${out}" --p320 paper/v3_results/p3_20 --fixes paper/v3_results
} 2>&1 | tee "${out}/p3_22_report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_22_${stamp}.tar.gz" "${out}"
echo "send me this file: $(pwd)/p3_22_${stamp}.tar.gz ($(du -h "p3_22_${stamp}.tar.gz" | cut -f1))"
