#!/bin/bash
# Steps p3.12-p3.15 (Nov 2, PLAN_PAPER3.md): the reports, CPU only, from stored features, responses and runs. Submit
# from the repository folder after the GPU jobs of p3.12 (and p3.15, if queued) are done; it can be submitted again
# at any time (each report reads what exists):
#
#   sbatch slurm/p3ext.sbatch        # the archive name is printed at the end of logs/cpup3ext-<id>.out
#
#   p3.12  scripts/external_report.py   every gate on HarmBench, HarmBench contextual, StrongREJECT; Seq-GRPO behind it
#   p3.14  scripts/np_gap.py            synthetic, about 1 h on 8 cores (skipped when np_gap.json exists)
#   p3.15  scripts/budget_report.py     NQ 1.5B, every method at a matched budget
#
# The job name does not start with tw-, so it never takes a slot of the GPU queue, and the autopilot does not track it.
#SBATCH --job-name=cpup3ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source slurm/common.sh
set +eu; activate_env >/dev/null; set -eu
out=paper/v3_results
ok() { "$@" || echo "  (did not finish: $*)"; }
mkdir -p "${out}/p3_9" "${out}/p3_12" "${out}/p3_13" "${out}/p3_14" "${out}/p3_15"
{
  echo "== p3.12 external sets"
  for f in data/ext_harmbench.jsonl data/ext_harmbench_ctx.jsonl data/ext_strongreject.jsonl; do
    [ -f "${f}" ] && echo "  ${f}: $(wc -l < "${f}") rows" || echo "  ${f}: missing (python scripts/prepare_external.py --data data)"
  done
  [ -f "${out}/p3_12/PREREG_p3_12.txt" ] || echo "  WARNING: no ${out}/p3_12/PREREG_p3_12.txt (python scripts/prereg_record.py first)"
  ok python scripts/external_report.py --out "${out}/p3_12" \
    --exp configs/v2/ladder_concise_q15.yaml:Concise:advbench,general \
    --exp configs/v2/ladder_concisectx_q15.yaml:Concisectx:advbench,general \
    --exp configs/v3/squad2_q15.yaml:Squad:advbench,general \
    --exp configs/v2/overlap_q15.yaml:Nq: \
    --exp configs/v2/ladder_ifc_q15.yaml:Ifc: \
    --exp configs/v2/ladder_ifc_q7.yaml:Ifcseven: \
    --exp configs/v2/stream_q15.yaml:Stream: \
    --exp configs/v2/overlap_olmo7.yaml:Olmo: \
    --exp configs/fam/nq_llama3b.yaml:Llamathree:
  echo "== p3.9 Table 2 behind the recommended gate"
  ok python scripts/recommended_table.py --out "${out}/p3_9"
  echo "== p3.14 probe against the optimal test (synthetic)"
  if [ -f "${out}/p3_14/np_gap.json" ]; then cat "${out}/p3_14/np_gap_summary.txt"; else ok python scripts/np_gap.py --out "${out}/p3_14"; fi
  echo "== p3.15 matched budget"
  ok python scripts/budget_report.py --runs runs/v2_overlap_q15 --out "${out}/p3_15"
  [ -d runs/v3_squad2_q15 ] && ok python scripts/budget_report.py --runs runs/v3_squad2_q15 --out "${out}/p3_15"
} 2>&1 | tee "${out}/p3_ext_report.txt"
stamp=$(date +%Y%m%d_%H%M)
# the key of the rating sheet stays on the cluster until the sheet is rated (it names method and judge labels)
tar czf "p3_ext_${stamp}.tar.gz" --exclude="${out}/p3_13/human_key.json" "${out}/p3_9" "${out}/p3_12" "${out}/p3_13" "${out}/p3_14" \
  "${out}/p3_15" "${out}/p3_ext_report.txt" "logs/cpup3ext-${SLURM_JOB_ID:-local}.out" 2>/dev/null \
  || tar czf "p3_ext_${stamp}.tar.gz" --exclude="${out}/p3_13/human_key.json" "${out}/p3_9" "${out}/p3_12" "${out}/p3_13" "${out}/p3_14" \
       "${out}/p3_15" "${out}/p3_ext_report.txt"
echo "send me this file: $(pwd)/p3_ext_${stamp}.tar.gz ($(du -h "p3_ext_${stamp}.tar.gz" | cut -f1))"
