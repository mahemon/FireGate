#!/bin/bash
# Ref-KL on NQ for Llama-3.1-8B and Mistral-7B, seeds 0-2 (step nov4.1, PLAN_NOV.md), submitted directly to a partition
# of your choice, outside every plan queue, so no autopilot submits or counts them. Run from the repository folder:
#
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/refkl_nova.sh status
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/refkl_nova.sh dry
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/refkl_nova.sh
#
# GPU_TYPE: one GPU model with bfloat16 and at least 40 GB (A100 / H100 / H200; not V100); a run must stay on the GPU
# model it started on, so give the same type every time. ACCOUNT=none: no --account (the partition's default).
# 6 jobs, 64 GB of CPU memory and 15 h each (about 8-10 h per run on an A100); 20 minutes before its limit a job saves
# and resubmits itself to the same partition. The jobs never feed a plan queue (NO_FEED=1), so nothing of the
# autopilots lands on this partition. After a cancel
# or a crash, run the same command again. Finished and queued runs are skipped, the others continue
# from their checkpoint. CONFIGS="..." or SEEDS="..." restrict the list.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/refkl_nova.sh [status|dry]" >&2; exit 1 ;; esac
: "${PARTITION:?set PARTITION, e.g. PARTITION=nova}"
: "${ACCOUNT:?set ACCOUNT (the account for that partition, or ACCOUNT=none)}"
: "${GPU_TYPE:?set GPU_TYPE to one GPU model of that partition (bfloat16, >= 40 GB), e.g. GPU_TYPE=a100}"
ENVF=slurm/cluster.env; [ -f "${ENVF}" ] || ENVF=slurm/cluster.env.example
# the jobs resubmit themselves through slurm/submit.sh, which reads this file: it must not overwrite these settings
if ! ( PARTITION=__p__ ACCOUNT=__a__ GPU_TYPE=__g__; source "${ENVF}" >/dev/null 2>&1
       [ "${PARTITION}" = __p__ ] && [ "${ACCOUNT}" = __a__ ] && [ "${GPU_TYPE}" = __g__ ] ); then
  echo "ERROR: ${ENVF} overwrites PARTITION, ACCOUNT or GPU_TYPE; write them as PARTITION=\"\${PARTITION:-instruction}\"" \
       "(see slurm/cluster.env.example), otherwise a resubmitted job lands on the default partition" >&2
  exit 1
fi
export PARTITION ACCOUNT GPU_TYPE
export NO_FEED=1
CONFIGS="${CONFIGS:-configs/fam/nq_llama8.yaml configs/fam/nq_mistral7.yaml}"
SEEDS="${SEEDS:-0 1 2}"
TL="${TIME_LIMIT_REFKL:-15:00:00}"
GPU_MODEL_CONSTRAINTS="${GPU_MODEL_CONSTRAINTS:-A100 80GB PCIe=epyc-7413;A100-SXM4-80GB=compute-gpu}"
jobs=$(squeue --me -h -o '%j' 2>/dev/null || true)
n_done=0; n_queued=0; n_sub=0; n_todo=0; n_fail=0
for cfg in ${CONFIGS}; do
  exp=$(grep -E '^exp:' "${cfg}" | awk '{print $2}')
  cname=$(basename "${cfg}" .yaml)
  model=$(grep -E '^model_path:' "${cfg}" | awk '{print $2}')
  if [ ! -f "${model}/config.json" ]; then echo "skipped  ${cfg}: ${model} is not downloaded"; continue; fi
  for s in ${SEEDS}; do
    dir="runs/${exp}/s${s}/refkl"
    name="tw-${cname}-s${s}-refkl"     # the name slurm/plan.sh uses for the same run (status, judge waits for it)
    if [ -f "${dir}/DONE" ]; then n_done=$((n_done + 1)); echo "done     ${dir}"; continue; fi
    if grep -qxF "${name}" <<< "${jobs}"; then n_queued=$((n_queued + 1)); echo "queued   ${dir}"; continue; fi
    if [ -f "${dir}/FAILED" ] && [ "$(wc -l < "${dir}/FAILED")" -ge 2 ]; then
      n_fail=$((n_fail + 1)); echo "FAILED   ${dir} (failed twice, see ${dir}/FAILED and logs/${name}-*.out; delete it to retry)"
      continue
    fi
    # a run that has started keeps to nodes with its GPU model (two A100 models share the gpu:a100 type)
    cons=""
    dev=$(grep -o '"device": "[^"]*"' "${dir}/config.json" 2>/dev/null | head -1 | cut -d'"' -f4 || true)
    if [ -n "${dev}" ]; then
      IFS=';' read -ra pairs <<< "${GPU_MODEL_CONSTRAINTS}"
      for pair in "${pairs[@]}"; do [[ -n "${pair}" && "${dev}" == *"${pair%%=*}"* ]] && cons="${pair#*=}"; done
    fi
    if [ "${MODE}" = status ]; then n_todo=$((n_todo + 1)); echo "to do    ${dir}${dev:+ (started on ${dev})}"; continue; fi
    [ "${MODE}" = dry ] && export DRY=1
    CONFIG="${cfg}" SEED="${s}" METHODS=refkl TAG="" EXTRA="" TIME_LIMIT="${TL}" NGPU=1 MEM=64G \
      CONSTRAINT="${cons}" DEPEND="" BEGIN="" bash slurm/submit.sh stream && n_sub=$((n_sub + 1)) || true
  done
done
echo "done ${n_done}, queued or running ${n_queued}, submitted ${n_sub}, to do ${n_todo}, failed ${n_fail}" \
     "(partition ${PARTITION}, GPU ${GPU_TYPE}, account ${ACCOUNT})"
