#!/bin/bash
# Steps p3.17-p3.19 (Nov 4, PLAN_PAPER3.md): the reports for the last review's questions, CPU only, from stored features
# and responses; no model is loaded. Submit from the repository folder; it can be submitted again at any time (each
# report reads what exists, and p3.19 is skipped until slurm/p3_prosp.sh has stored the prospective sets):
#
#   sbatch slurm/p3more.sbatch        # the archive name is printed at the end of logs/cpup3more-<id>.out
#
#   p3.17  scripts/fired_audit.py         harmful prompts where the gate fires: refusal there, prompts newly answered
#   p3.18  scripts/score_stream.py        the four score classes on the three task streams, stage by stage
#   p3.19  scripts/prospective_report.py  the frozen gates on prompt sets written by others, and the predictions
#   p3.20  scripts/score_classes.py + score_grid.py   linear classifier against Gaussian ratio on all 13 model-task pairs
#SBATCH --job-name=cpup3more
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
out=paper/v3_results
ok() { "$@" || echo "  (did not finish: $*)"; }
mkdir -p "${out}/p3_17" "${out}/p3_18" "${out}/p3_19" "${out}/p3_20"
EXPS=(--exp configs/v2/ladder_concise_q15.yaml:Concise:advbench,general
      --exp configs/v2/ladder_concisectx_q15.yaml:Concisectx:advbench,general
      --exp configs/v3/squad2_q15.yaml:Squad:advbench,general
      --exp configs/v2/overlap_q15.yaml:Nq:
      --exp configs/v2/ladder_ifc_q15.yaml:Ifc:
      --exp configs/v2/ladder_ifc_q7.yaml:Ifcseven:
      --exp configs/v2/stream_q15.yaml:Stream:
      --exp configs/v2/overlap_olmo7.yaml:Olmo:
      --exp configs/fam/nq_llama3b.yaml:Llamathree:)
{
  echo "== p3.17 fired audit"
  ok python scripts/fired_audit.py --out "${out}/p3_17" "${EXPS[@]}"
  echo "== p3.18 score classes on the streams"
  ok python scripts/score_stream.py --out "${out}/p3_18" --config configs/v2/stream_q15.yaml \
    --config configs/v2/stream_q7.yaml --config configs/v2/stream5_q15.yaml
  echo "== p3.20 linear classifier against the Gaussian ratio on every model-task pair"
  for c in configs/v2/ladder_ifc_q15.yaml configs/v2/overlap_q15.yaml configs/v2/ladder_ifc_q7.yaml \
           configs/v2/overlap_q7.yaml configs/v2/overlap_olmo7.yaml configs/fam/ifc_llama3b.yaml \
           configs/fam/nq_llama3b.yaml configs/fam/ifc_llama8.yaml configs/fam/nq_llama8.yaml configs/fam/ifc_ds7.yaml \
           configs/fam/nq_ds7.yaml configs/fam/ifc_mistral7.yaml configs/fam/nq_mistral7.yaml; do
    e=$(grep -E '^exp:' "${c}" | awk '{print $2}')
    if [ -f "${out}/p3_20/${e}/scores.json" ]; then echo "  done ${e}"; continue; fi
    ok python scripts/score_classes.py --config "${c}" --variant deployed --scores probe,lda --out "${out}/p3_20/${e}"
  done
  ok python scripts/score_grid.py --root "${out}/p3_20"
  echo "== p3.19 prospective sets"
  [ -f "${out}/p3_19/PREREG_p3_19.txt" ] || echo "  WARNING: no ${out}/p3_19/PREREG_p3_19.txt"
  ok python scripts/prospective_report.py --out "${out}/p3_19" "${EXPS[@]}"
} 2>&1 | tee "${out}/p3_more_report.txt"
stamp=$(date +%Y%m%d_%H%M)
tar czf "p3_more_${stamp}.tar.gz" "${out}/p3_17" "${out}/p3_18" "${out}/p3_19" "${out}/p3_20" "${out}/p3_more_report.txt"
echo "send me this file: $(pwd)/p3_more_${stamp}.tar.gz ($(du -h "p3_more_${stamp}.tar.gz" | cut -f1))"
