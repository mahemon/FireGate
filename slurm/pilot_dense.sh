#!/bin/bash
# The full-rank runs of the v7 pilot (step v7.4, PLAN_v7.md), submitted on their own to a partition of your choice,
# outside slurm/plan.sh's queue, so the autopilot never submits them. Run from the repository folder:
#
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/pilot_dense.sh status
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/pilot_dense.sh dry
#   PARTITION=nova ACCOUNT=<account or none> GPU_TYPE=<GPU type on that partition> bash slurm/pilot_dense.sh
#
# GPU_TYPE: one GPU model with bfloat16 and at least 40 GB (A100 / H100 / H200; not V100); a run must stay on the GPU
# model it started on, so give the same type every time. ACCOUNT=none: no --account (the partition's default).
# Each job asks for 64 GB of CPU memory and 6 h; 20 minutes before its limit it saves and resubmits itself to the
# same partition. After a cancel or a
# crash, run the same command again. Finished and queued runs are skipped, the others continue from their checkpoint.
# SEEDS="1 2" or CONFIG=configs/v2/overlap_q15.yaml run the same settings elsewhere (step v7.3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/pilot_dense.sh [status|dry]" >&2; exit 1 ;; esac
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
CONFIG="${CONFIG:-configs/v2/ladder_ifc_q15.yaml}"
SEEDS="${SEEDS:-0}"
TL="${TIME_LIMIT_DENSE:-06:00:00}"
GPU_MODEL_CONSTRAINTS="${GPU_MODEL_CONSTRAINTS:-A100 80GB PCIe=epyc-7413;A100-SXM4-80GB=compute-gpu}"
# the same runs as st_v7_4 in slurm/plan.sh (folder <method>_<tag>, job name tw-<config>-s<seed>-<method>_<tag>)
DENSE=("seq:dense5e6:lora.dense=true grpo.lr=5.0e-6" "seq:dense2e6:lora.dense=true grpo.lr=2.0e-6"
       "tripwire_prompt:dense5e6:lora.dense=true grpo.lr=5.0e-6" "seq:full5e6:lora.dense=true grpo.lr=5.0e-6 gate_layer=1")
exp=$(grep -E '^exp:' "${CONFIG}" | awk '{print $2}')
cname=$(basename "${CONFIG}" .yaml)
jobs=$(squeue --me -h -o '%j' 2>/dev/null || true)
n_done=0; n_queued=0; n_sub=0; n_todo=0; n_fail=0
for s in ${SEEDS}; do
  for spec in "${DENSE[@]}"; do
    IFS=: read -r method tag ovr <<< "${spec}"
    dir="runs/${exp}/s${s}/${method}_${tag}"
    name="tw-${cname}-s${s}-${method}_${tag}"
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
    CONFIG="${CONFIG}" SEED="${s}" METHODS="${method}" TAG="${tag}" EXTRA="${ovr}" TIME_LIMIT="${TL}" NGPU=1 \
      MEM=64G CONSTRAINT="${cons}" DEPEND="" BEGIN="" bash slurm/submit.sh stream && n_sub=$((n_sub + 1)) || true
  done
done
echo "done ${n_done}, queued or running ${n_queued}, submitted ${n_sub}, to do ${n_todo}, failed ${n_fail}" \
     "(partition ${PARTITION}, GPU ${GPU_TYPE}, account ${ACCOUNT})"
