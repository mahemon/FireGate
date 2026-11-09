# shellcheck shell=bash
# Add-on plan (Oct 30, review round 4), sourced by slurm/plan.sh: drift under other decoding rules. Evaluation only,
# on finished runs: no training, nothing added to the autopilot queues.
#
#   bash slurm/plan.sh decode-status    done / queued / to do for each run below
#   bash slurm/plan.sh decode-submit    one GPU job per run (eval_extra.py --what decode), on the run's own GPU type
#   bash slurm/plan.sh decode-collect   paper/v2_results/decode/ (CPU, login node), packed into decode_results_<date>.tar.gz
#
# Each job decodes Dolly, AdvBench, XSTest and MMLU (1,600 prompts at 1.5B) with the final and the base model under
# three rules (T=0.3; T=1; T=1 with top-p 0.9), one shared seed, plus a serving control (the base model again with an
# eighth of the batch size): 26 pool decodes, about 40-80 min on one GPU. 5 jobs, about 4-6 GPU-hours in all.
# Each job is pinned to the GPU model its run trained on (GPU type and node constraint), so the final and base decodes
# of a pool never run on different GPU models, even after a resubmission. The job name is the one the main plan uses
# for the same run's other evaluations (tw-x-...); those are finished for these runs.
DECODE_RUNS=("${S15}:tripwire_prompt:Stream, \\methodp{}" "${S15}:tripwire:Stream, \\methoda{}"
             "${S15}:seq:Stream, Seq-GRPO" "${O15}:tripwire_prompt:NQ, \\methodp{}" "${O15}:seq:NQ, Seq-GRPO")

decode_state() {  # decode_state <run folder> <job name>: done / queued / waiting / todo
  local d="$1"
  if [ -f "${d}/extra/summary.json" ] && grep -q '"decode_serving"' "${d}/extra/summary.json"; then echo done; return; fi
  if queued "$2"; then echo queued; return; fi
  if [ ! -f "${d}/DONE" ]; then echo waiting; return; fi
  echo todo   # extra/FAILED is shared with the run's other evaluations, so it is not read here; see logs/tw-x-*
}

decode_each() {  # decode_each <submit|status>
  local spec cfg m d job st pinned cons
  refresh_jobs
  for spec in "${DECODE_RUNS[@]}"; do
    cfg="$(cut -d: -f1 <<< "${spec}")"; m="$(cut -d: -f2 <<< "${spec}")"
    d="runs/$(exp_of "${cfg}")/s0/${m}"
    job="tw-x-$(basename "${cfg}" .yaml)-s0-${m}"
    st="$(decode_state "${d}" "${job}")"
    printf '%-8s %s\n' "${st}" "${d}"
    [ "$1" = submit ] && [ "${st}" = todo ] || continue
    if [ -s "${d}/GPU_TYPE" ]; then pinned="$(cat "${d}/GPU_TYPE")"; else choose_gpu "${GPUS_SMALL}" 1; pinned="${CHOSEN}"; fi
    cons="$(constraint_for "${d}")"
    CONFIG="${cfg}" SEED=0 METHODS="${m}" WHAT=decode NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${pinned}" CONSTRAINT="${cons}" MEM="${MEM_SMALL}" EXTRA_TIME=02:30:00 bash slurm/submit.sh extra || true
  done
}
addon_decode_status() { decode_each status; }
addon_decode_submit() { decode_each submit; }

addon_decode_collect() {  # tables in paper/v2_results/decode/, packed to send
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/v2_results/decode stamp spec cfg m args=()
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}"
  for spec in "${DECODE_RUNS[@]}"; do
    cfg="$(cut -d: -f1 <<< "${spec}")"; m="$(cut -d: -f2 <<< "${spec}")"
    args+=(--run "$(cut -d: -f3- <<< "${spec}")=runs/$(exp_of "${cfg}")/s0/${m}")
    mkdir -p "${out}/runs/$(exp_of "${cfg}")_${m}"
    cp "runs/$(exp_of "${cfg}")/s0/${m}/extra/summary.json" "${out}/runs/$(exp_of "${cfg}")_${m}/" 2>/dev/null || true
  done
  python scripts/decode_report.py --out "${out}" "${args[@]}" 2>&1 | tee "${out}/report.txt" || true
  addon_decode_status > "${out}/status.txt" 2>&1 || true
  tar czf "decode_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/decode_results_${stamp}.tar.gz ($(du -h "decode_results_${stamp}.tar.gz" | cut -f1))"
}
