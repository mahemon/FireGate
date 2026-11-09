#!/bin/bash
# Step p3.23 (Nov 5, PLAN_PAPER3.md): Concise with a wide harmful component, as two jobs:
#   job 1 (GPU, slurm/p3widen.sbatch)     features of the two new sets, then Seq-GRPO's responses on the English
#                                          harmful test rows (three seeds); on the GPU model of the stored features
#   job 2 (CPU, slurm/p3widenrep.sbatch)  the report and its archive, after job 1 finished without error (afterok)
# Run from the repository folder, after python scripts/prepare_widen.py --data data and
# python scripts/prereg_record.py --step p3.23:
#
#   bash slurm/p3_widen.sh status     what is done, and the GPU model job 1 needs
#   bash slurm/p3_widen.sh dry        the sbatch commands, nothing submitted
#   bash slurm/p3_widen.sh            submit both jobs (a job already queued or running is not submitted twice)
#
# Job 1: PARTITION (default instruction) with the ACCOUNT of slurm/cluster.env on instruction (INSTRUCTION_ACCOUNT
# overrides it); the gres type follows the stored GPU model (the mapping of slurm/p3_prosp.sh), and so does the node
# constraint off the instruction partition (CONSTRAINT sets one anywhere, CONSTRAINT= none). Job 1 checks the GPU
# model itself and stops, without computing features, on any other model.
# Job 2: CPU_PARTITION (default nova). Time limits are those of the two sbatch files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/p3_widen.sh [status|dry]" >&2; exit 1 ;; esac
PARTITION="${PARTITION:-instruction}"
CPU_PARTITION="${CPU_PARTITION:-nova}"
ENVF=slurm/cluster.env; [ -f "${ENVF}" ] || ENVF=slurm/cluster.env.example
INSTRUCTION_ACCOUNT="${INSTRUCTION_ACCOUNT:-$( (unset ACCOUNT; source "${ENVF}" >/dev/null 2>&1; echo "${ACCOUNT:-none}") )}"
ACCOUNT_OTHER="${ACCOUNT_OTHER:-none}"
CFG=configs/v2/ladder_concise_q15.yaml
EXP=$(grep -E '^exp:' "${CFG}" | awk '{print $2}')
JS="runs/${EXP}/analysis/gate_features.json"
mkdir -p logs

for f in data/wd_aya.jsonl data/wd_dna.jsonl; do
  [ -s "${f}" ] || { echo "${f} is missing: run python scripts/prepare_widen.py --data data on the login node" >&2
                     [ "${MODE}" = status ] || exit 1; }
done
[ -f paper/v3_results/p3_23/PREREG_p3_23.txt ] || echo "WARNING: no paper/v3_results/p3_23/PREREG_p3_23.txt (python scripts/prereg_record.py --step p3.23 first)"
DEV=$(grep -o '"device": "[^"]*"' "${JS}" 2>/dev/null | head -1 | cut -d'"' -f4 || true)
[ -n "${DEV}" ] || { echo "${JS}: no stored gate features" >&2; exit 1; }
case "${DEV}" in
  *H200*) GT=h200; CN="" ;;
  *SXM4*) GT=a100; CN=compute-gpu ;;
  *) GT=a100; CN=epyc-7413 ;;
esac
[ "${PARTITION}" = instruction ] && CN=""
CN="${CONSTRAINT-${CN}}"
echo "features of ${EXP} stored on ${DEV}: job 1 asks for gpu:${GT}:1${CN:+, constraint ${CN}}, partition ${PARTITION}"
if { grep -q '"eval__wd_aya"' "${JS}" && grep -q '"eval__wd_dna"' "${JS}"; } 2>/dev/null; then echo "done     features of the new sets"; else echo "to do    features of the new sets"; fi
for s in 0 1 2; do
  d="runs/${EXP}/s${s}/seq"
  if [ ! -f "${d}/DONE" ]; then echo "missing  ${d} (not finished)"
  elif grep -q '"widen"' "${d}/extra/summary.json" 2>/dev/null; then echo "done     ${d}/extra/widen"
  else echo "to do    ${d}/extra/widen"; fi
done
[ -f paper/v3_results/p3_23/widen.json ] && echo "done     paper/v3_results/p3_23/widen.json (job 2 can be submitted again)"
[ "${MODE}" = status ] && exit 0

queued() { squeue --me -h -o '%j' 2>/dev/null | grep -qxF "$1"; }
acct=""
if [ "${PARTITION}" = instruction ]; then acct="${INSTRUCTION_ACCOUNT}"; else acct="${ACCOUNT_OTHER}"; fi
[ "${acct}" = none ] && acct=""
g=(--parsable --partition="${PARTITION}" --gres="gpu:${GT}:1" --chdir="$(pwd)" --export=ALL)
[ -n "${acct}" ] && g+=(--account="${acct}")
[ -n "${CN}" ] && g+=(--constraint="${CN}")
if [ "${MODE}" = dry ]; then
  echo "DRY: sbatch ${g[*]} slurm/p3widen.sbatch"
  echo "DRY: sbatch --parsable --partition=${CPU_PARTITION} --dependency=afterok:<job 1> slurm/p3widenrep.sbatch"
  exit 0
fi
if queued p3widen || queued cpup3widen; then echo "p3widen or cpup3widen is already queued or running: nothing submitted"; exit 0; fi
id1=$(sbatch "${g[@]}" slurm/p3widen.sbatch)
echo "submitted ${id1} p3widen (${PARTITION}, gpu:${GT}${CN:+, ${CN}}; log logs/p3widen-${id1}.out)"
id2=$(sbatch --parsable --partition="${CPU_PARTITION}" --chdir="$(pwd)" --dependency="afterok:${id1}" slurm/p3widenrep.sbatch)
echo "submitted ${id2} cpup3widen (${CPU_PARTITION}, after ${id1}; log logs/cpup3widen-${id2}.out)"
echo "then: bash slurm/p3_widen.sh status; the archive name is at the end of logs/cpup3widen-${id2}.out"
