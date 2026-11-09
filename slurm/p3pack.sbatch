#!/bin/bash
# Steps p3.12 and p3.15 packed into one GPU job (PLAN_PAPER3.md), submitted by slurm/p3_nova.sh, never by a plan queue.
# Runs the items listed in ITEMS (one per line: kind config seed method tag overrides), skipping every item already
# done, in LANES lanes on the same GPU (lane 1: training runs, lane 2: everything else; LANES=1: one after the other):
#   features  gate_features.py --add (the external sets), only on the GPU model the stored features come from
#   external  eval_extra.py --what external of a finished run (final and base model, both in this job)
#   scale     eval_scale.py: a finished run's adapter scaled by alpha (the GPU model of the run, as eval_scale checks)
#   train     run_stream.py: a new run
#   judge     judge_changes.py over every finished run of the config (Nov 4; method and tag "-"): labels are cached
#             per run, so a pass judges only new pairs; done when analysis/judge/DONE is newer than every run's DONE
#   prosp     gate_features.py --add for the prospective sets of step p3.19 (Nov 4; data/pr_*.jsonl), only on the
#             GPU model the stored features come from; done when gate_features.json holds eval__pr_*
# 20 minutes before the time limit, training and scaling save and stop; the job then resubmits itself once more per
# limit (the same items; the finished ones are skipped). After a crash or a cancel, run slurm/p3_nova.sh again.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@1200
set -o pipefail
source "${REPO}/slurm/common.sh"
activate_env
set -u
export NO_FEED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation (Nov 3)
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[$(date)] job ${SLURM_JOB_ID}: ${ITEMS} on ${GPU_NAME}, ${LANES:-2} lane(s)"
STOP="${REPO}/logs/STOP_${SLURM_JOB_ID}"
rm -f "${STOP}"
trap 'echo "[$(date)] USR1: asking the running items to save and stop"; touch "${STOP}"' USR1
# cancelled (scancel): no run is marked failed, and no resubmission from here (slurm/p3auto.sh does it)
trap 'echo "[$(date)] TERM: job cancelled"; touch "${STOP}"; NO_RESUBMIT=1' TERM

exp_of() {  # the config's exp: line, else the loaded config's
  local e
  e=$(grep -E '^exp:' "$1" | awk '{print $2}')
  [ -n "${e}" ] && { echo "${e}"; return; }
  python -c "import sys; sys.path.insert(0, '.'); from twcore.config import load_config; print(load_config('$1')['exp'])"
}

run_item() {  # kind config seed method tag overrides; returns 0 when done or skipped
  local kind="$1" cfg="$2" s="$3" m="$4" tag="$5" ovr="$6" exp d dev rc=0
  [ "${tag}" = - ] && tag=""
  [ "${ovr}" = - ] && ovr=""
  exp="$(exp_of "${cfg}")"
  case "${kind}" in
    features|prosp)
      d="runs/${exp}/analysis"
      local want='"eval__ext_' what="external sets"
      [ "${kind}" = prosp ] && want='"eval__pr_' what="prospective sets"
      if grep -q "${want}" "${d}/gate_features.json" 2>/dev/null; then echo "done     ${d} (${what})"; return 0; fi
      dev=$(grep -o '"device": "[^"]*"' "${d}/gate_features.json" 2>/dev/null | head -1 | cut -d'"' -f4)
      if [ -z "${dev}" ]; then echo "SKIPPED  ${d}: no stored gate features"; return 0; fi
      if [ "${dev}" != "${GPU_NAME}" ]; then
        echo "SKIPPED  ${d}: stored on ${dev}, this job has ${GPU_NAME} (p3_nova.sh lists the command for it)"; return 0
      fi
      echo "[$(date)] ${kind} ${cfg}"
      python scripts/gate_features.py --config "${cfg}" --add || rc=$?
      d="${d}/gate_features" ;;
    external)
      d="runs/${exp}/s${s}/${m}${tag:+_${tag}}"
      if grep -q '"external"' "${d}/extra/summary.json" 2>/dev/null; then echo "done     ${d}/extra/external"; return 0; fi
      [ -f "${d}/DONE" ] || { echo "SKIPPED  ${d}: not finished"; return 0; }
      echo "[$(date)] external ${d}"
      # shellcheck disable=SC2086
      python scripts/eval_extra.py --config "${cfg}" --method "${m}" --what external --set "seed=${s}" ${ovr} \
        ${tag:+--tag "${tag}"} || rc=$?
      d="${d}/extra" ;;
    scale)
      d="runs/${exp}/s${s}/scale_$(printf 'a%.2f' "${tag}")"
      if [ -f "${d}/DONE" ]; then echo "done     ${d}"; return 0; fi
      [ -f "runs/${exp}/s${s}/${m}/DONE" ] || { echo "SKIPPED  ${d}: runs/${exp}/s${s}/${m} not finished"; return 0; }
      [ -f "${STOP}" ] && return 0
      echo "[$(date)] scale ${d}"
      python scripts/eval_scale.py --config "${cfg}" --base "${m}" --alpha "${tag}" --stop-file "${STOP}" \
        --set "seed=${s}" || rc=$? ;;
    train)
      d="runs/${exp}/s${s}/${m}${tag:+_${tag}}"
      if [ -f "${d}/DONE" ]; then echo "done     ${d}"; return 0; fi
      [ -f "${STOP}" ] && return 0
      echo "[$(date)] train ${d}"
      # shellcheck disable=SC2086
      python scripts/run_stream.py --config "${cfg}" --method "${m}" ${tag:+--tag "${tag}"} --stop-file "${STOP}" \
        --set "seed=${s}" ${ovr} || rc=$?
      [ -f "${d}/DONE" ] && rc=0 ;;
    judge)
      d="runs/${exp}/analysis/judge"
      if [ -f "${d}/DONE" ] && [ -z "$(find "runs/${exp}" -mindepth 3 -maxdepth 3 -path "runs/${exp}/s*/*/DONE" \
           -newer "${d}/DONE" -print -quit 2>/dev/null)" ]; then echo "done     ${d}"; return 0; fi
      [ -f "${STOP}" ] && return 0
      echo "[$(date)] judge ${cfg}"
      python scripts/judge_changes.py --config "${cfg}" --judge "${JUDGE_MODEL:-models/Qwen2.5-7B-Instruct}" || rc=$? ;;
  esac
  if [ ${rc} -ne 0 ] && [ ! -f "${STOP}" ] && [ ${rc} -ne 143 ] && [ ${rc} -ne 137 ]; then   # 143/137: killed (cancel)
    echo "FAILED   ${kind} ${cfg} s${s} ${m} ${tag} (exit ${rc})"
    mark_failed "${d}" "${rc}"
  fi
  return 0
}

lane() {  # lane <1|2|all>
  local kind cfg s m tag ovr
  while read -r kind cfg s m tag ovr; do
    [ -z "${kind}" ] || [[ "${kind}" == \#* ]] && continue
    case "$1" in 1) [ "${kind}" = train ] || continue ;; 2) [ "${kind}" = train ] && continue ;; esac
    run_item "${kind}" "${cfg}" "${s}" "${m}" "${tag}" "${ovr:--}"
  done < "${ITEMS}"
}

if [ "${LANES:-2}" -ge 2 ]; then
  lane 1 > >(sed -u 's/^/[lane 1] /') 2>&1 &
  p1=$!
  lane 2 > >(sed -u 's/^/[lane 2] /') 2>&1 &
  p2=$!
  for p in ${p1} ${p2}; do
    while true; do wait "${p}"; rc=$?; if [ ${rc} -gt 128 ] && kill -0 "${p}" 2>/dev/null; then continue; fi; break; done
  done
else
  lane all &
  p=$!
  while true; do wait "${p}"; rc=$?; if [ ${rc} -gt 128 ] && kill -0 "${p}" 2>/dev/null; then continue; fi; break; done
fi
echo "[$(date)] items processed"
if [ -f "${STOP}" ] && [ "${NO_RESUBMIT:-0}" != 1 ]; then
  echo "[$(date)] time limit reached: resubmitting (finished items are skipped)"
  sbatch --parsable --partition="${SLURM_JOB_PARTITION}" --gres="${PACK_GRES}" --time="${PACK_TIME}" --mem="${PACK_MEM}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --chdir="${REPO}" --export=ALL --job-name="${SLURM_JOB_NAME}" \
    ${PACK_ACCOUNT:+--account="${PACK_ACCOUNT}"} ${PACK_CONSTRAINT:+--constraint="${PACK_CONSTRAINT}"} \
    "${REPO}/slurm/p3pack.sbatch"
fi
rm -f "${STOP}"
