# shellcheck shell=bash
# Add-on plan (Nov 1, PLAN_NOV.md), sourced by slurm/plan.sh: two larger models, a real-user evaluation pool, a code
# task and more baseline coverage. Kept apart from every other plan: configs in configs/nov/, new run folders in
# runs/nov_*/, tables in paper/nov_results/, its own autopilot (job cpunov, queue logs/plan_queue_nov, folder
# logs/autopilot_nov). The baseline runs of nov4.x go into the existing experiments (runs/fam_nq_*, runs/v2_*_q7).
#
#   AP_PLAN=nov AP_COUNT=tw AP_TARGET=12 sbatch --job-name=cpunov slurm/autopilot.sbatch   # nov2.1, nov3.x, nov4.2, nov4.4
#   PLAN_QUEUE=logs/plan_queue_nov bash slurm/plan.sh nov1        # later: the larger models (the autopilot feeds them)
#   PARTITION=nova ACCOUNT=... GPU_TYPE=... bash slurm/refkl_nova.sh   # nov4.1, directly to another partition
#   bash slurm/plan.sh nov1,nov2,nov3,nov4 status    done / queued / waiting / to do
#   bash slurm/plan.sh nov-tables                    tables in paper/nov_results/ (CPU, login node)
#   bash slurm/plan.sh nov-collect                   the same, packed into nov_results_<date>.tar.gz to send
#
# key:folder under models/:time limit per job:CPU memory. Configs: configs/nov/{ifc,nq}_<key>.yaml. A run longer than
# its time limit continues from its checkpoint in the next job.
NOV_MODELS=("llama13:Llama-2-13b-chat-hf:12:00:00:96G" "qwen14:Qwen2.5-14B-Instruct:12:00:00:96G")
NOV_RUNS=(seq:: tripwire_prompt::)
NOV_CODE=configs/nov/code_q15.yaml
# experiments evaluated on WildChat (step nov2.1): the rows of the paper's switch table and the 1.5B stream
NOV_WILD=("${IFC}" "${O15}" "${IF7}" "${O7}" "${OO7}" "${S15}"
          configs/fam/ifc_llama3b.yaml configs/fam/nq_llama3b.yaml configs/fam/ifc_ds7.yaml configs/fam/nq_ds7.yaml
          configs/fam/ifc_llama8.yaml configs/fam/nq_llama8.yaml configs/fam/ifc_mistral7.yaml configs/fam/nq_mistral7.yaml)
NOV_RKL=(configs/fam/nq_llama8.yaml configs/fam/nq_mistral7.yaml)
STEPS+=(
  "nov1.0|later: real-model check of Llama-2-13B-Chat and Qwen2.5-14B (A100: bf16 exactness, peak memory, hours per run); their training waits for it: 2 jobs, under 30 min each"
  "nov1.1|later: IF and NQ, seed 0, Seq-GRPO and Tripwire-prompt (LoRA), both models: 8 runs, about 10-14 h each (12 h jobs, continued)"
  "nov1.2|later: seeds 1-2 of nov1.1: 16 runs, about 10-14 h each"
  "nov1.3|later, last: gate features (after seed 0's Tripwire-prompt) and LLM judge (after the runs) of the 4 experiments: 8 short jobs"
  "nov2.1|WildChat (500 real-user requests, never calibrated on) for 14 finished experiments: gate features added (14 jobs, under 45 min) and seed-0 Seq-GRPO and Tripwire-prompt evaluated (28 jobs, 10-60 min)"
  "nov3.1|code task (Qwen2.5-1.5B), before any training: gate features and the predicted coverage beta (separability.json): 1 job, under 1 h"
  "nov3.2|code task, seeds 0-2, Seq-GRPO and Tripwire-prompt: 6 runs, up to 3 h each (waits for nov3.1)"
  "nov3.3|last: code task, gate features again (after seed 0's Tripwire-prompt) and LLM judge: 2 short jobs"
  "nov4.1|not in the autopilot (slurm/refkl_nova.sh submits it to another partition): Ref-KL (beta 1) on NQ, Llama-3.1-8B and Mistral-7B, seeds 0-2: 6 runs, about 8-10 h each"
  "nov4.2|NQ, Qwen2.5-7B, seeds 1-2: +KL and null-space LoRA (4 runs, up to 8 h) and adapter scaling 0.5 of Seq-GRPO (2 evaluations)"
  "nov4.3|optional, not in the autopilot: 3-task stream, Qwen2.5-7B, seeds 1-2: +KL and router, 4 runs, about 20 h each (12 h jobs, continued)"
  "nov4.4|last: LLM judge of the experiments that got new baseline runs (NQ Llama-3.1-8B, NQ Mistral-7B, NQ and stream Qwen2.5-7B): up to 4 jobs"
)
nov_field() {  # nov_field <key> <n>: field n of the model's entry (2 folder, 3 time limit, 6 memory)
  local m
  for m in "${NOV_MODELS[@]}"; do
    [ "${m%%:*}" = "$1" ] || continue
    case "$2" in 2) cut -d: -f2 <<< "${m}" ;; 3) cut -d: -f3-5 <<< "${m}" ;; 6) cut -d: -f6 <<< "${m}" ;; esac
    return 0
  done
}
nov_model_ok() {  # nov_model_ok <key>: 0 if the model is downloaded; otherwise "skipped" (status) and 1
  local dir
  dir="models/$(nov_field "$1" 2)"
  [ -f "${dir}/config.json" ] && return 0
  say "skipped  ${1}: ${dir} is not downloaded (PLAN_NOV.md); queue the steps again after downloading it"
  return 1
}
nov_data_ok() {  # nov_data_ok <file under data/>...: 0 if every file exists; otherwise "waiting" and 1 (the step stays
                 # in the queue, so an autopilot started before the data were prepared picks it up afterwards)
  local f
  for f in "$@"; do
    if [ ! -s "data/${f}" ]; then
      N_WAIT+=1; say "waiting  data/${f} is missing (PLAN_NOV.md: prepare_data.py on the login node)"
      return 1
    fi
  done
}
nov_mem_of() {  # nov_mem_of <config>: CPU memory of a job on the config's model
  case "$(grep -E '^model_path:' "$1")" in
    *1.5B*) echo "${MEM_SMALL}" ;; *13b*|*14B*) echo 96G ;; *) echo "${MEM_BIG}" ;;
  esac
}
st_nov1_0() {
  local m key dir st
  for m in "${NOV_MODELS[@]}"; do
    key="${m%%:*}"
    nov_model_ok "${key}" || continue
    dir="models/$(nov_field "${key}" 2)"
    st="$(fam_check_state "${key}")"
    case "${st}" in
      passed) N_DONE+=1; say "done     check of ${key}"; continue ;;
      running) N_QUEUED+=1; say "queued   check of ${key}"; continue ;;
      failed) N_FAIL+=1; echo "FAILED   check of ${key}: see the newest logs/tw-selftest-${key}-g1-*.out (delete it to check again)"
              continue ;;
    esac
    item 0 "tw-selftest-${key}-g1" "check of ${key}" "logs/nov_check_${key}" || continue
    if [ "${MODE}" = commands ]; then
      echo "LEVEL=real NAME=${key} MODEL_DIR=${dir} CONFIG=configs/nov/ifc_${key}.yaml GPU_TYPE=a100 MEM=$(nov_field "${key}" 6) bash slurm/submit.sh selftest"
      continue
    fi
    LEVEL=real NAME="${key}" MODEL_DIR="${dir}" CONFIG="configs/nov/ifc_${key}.yaml" NGPU=1 MEM="$(nov_field "${key}" 6)" \
      DEPEND="" BEGIN="" GPU_TYPE=a100 CONSTRAINT="" bash slurm/submit.sh selftest || true
  done
}
nov_train() {  # nov_train <task> "<seeds>": Seq-GRPO and Tripwire-prompt of every downloaded model whose check passed
  local m key s st
  nov_data_ok wildchat_eval.jsonl || return 0
  for m in "${NOV_MODELS[@]}"; do
    key="${m%%:*}"
    nov_model_ok "${key}" || continue
    st="$(fam_check_state "${key}")"
    if [ "${st}" != passed ]; then
      if [ "${st}" = failed ]; then say "held     ${key} $1 runs: its check (step nov1.0) failed"
      else N_WAIT+=1; say "waiting  ${key} $1 runs (needs its check, step nov1.0)"; fi
      continue
    fi
    for s in $(seeds "$2"); do
      GPUS_NOW="${GPUS_BIG}" MEM_NOW="$(nov_field "${key}" 6)" train "configs/nov/$1_${key}.yaml" "${s}" \
        "$(nov_field "${key}" 3)" 1 "${NOV_RUNS[@]}"
    done
  done
}
st_nov1_1() { nov_train ifc "0"; nov_train nq "0"; }
st_nov1_2() { nov_train ifc "1 2"; nov_train nq "1 2"; }
st_nov1_3() {   # gate features once s0/tripwire_prompt is done (its deployed probe, its GPU model), then the judge
  local m key c t
  for m in "${NOV_MODELS[@]}"; do
    key="${m%%:*}"
    nov_model_ok "${key}" || continue
    for t in ifc nq; do
      c="configs/nov/${t}_${key}.yaml"
      want_addon_cfg "${c}" nov1.1 nov1.2 || continue
      GPUS_NOW="${GPUS_BIG}" MEM_NOW="$(nov_field "${key}" 6)" features "${c}" wait
      judge "${c}"
    done
  done
}

# WildChat for finished experiments: their stored gate features get the WildChat prompts (on the GPU model they were
# computed on), and seed 0's Seq-GRPO and Tripwire-prompt runs are evaluated on them (final and base model in one job)
wild_features() {  # wild_features <config>
  local cfg="$1" exp d js dev gt cons isdone=0 ft="00:30:00" mem
  mem="$(nov_mem_of "${cfg}")"
  [ "${mem}" = "${MEM_SMALL}" ] || ft="00:45:00"
  exp="$(exp_of "${cfg}")"
  skipped "features" "features" && return 0
  d="runs/${exp}/analysis/gate_features"
  js="runs/${exp}/analysis/gate_features.json"
  if [ ! -f "${js}" ] || [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
    N_WAIT+=1; say "waiting  ${d} (no stored gate features of ${exp})"; return 0
  fi
  grep -q '"eval__wildchat"' "${js}" && isdone=1
  item "${isdone}" "tw-gf-$(basename "${cfg}" .yaml)" "${d}/wildchat" "${d}" || return 0
  dev=$(grep -o '"device": "[^"]*"' "${js}" | head -1 | cut -d'"' -f4)
  case "${dev}" in *H200*) gt=h200 ;; *A100*) gt=a100 ;; *) gt="" ;; esac
  cons="$(constraint_of_device "${dev}")"
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${gt}${cons:+ CONSTRAINT=${cons}} MEM=${mem} FEATURES_TIME=${ft} FEATURES_ARGS=--add bash slurm/submit.sh features"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${gt}" CONSTRAINT="${cons}" \
    MEM="${mem}" FEATURES_TIME="${ft}" FEATURES_ARGS=--add bash slurm/submit.sh features || true
}
wild_eval() {  # wild_eval <config> <seed> <method>...: eval_extra.py --what wildchat of finished runs
  local cfg="$1" s="$2"; shift 2
  local exp m d isdone xt="00:45:00" mem cons
  mem="$(nov_mem_of "${cfg}")"
  [ "${mem}" = "${MEM_SMALL}" ] || xt="01:30:00"
  exp="$(exp_of "${cfg}")"
  for m in "$@"; do
    skipped "${m}" "${m}" && continue
    d="runs/${exp}/s${s}/${m}"
    if [ ! -f "${d}/DONE" ]; then N_WAIT+=1; say "waiting  ${d}/extra/wildchat (needs ${d})"; continue; fi
    isdone=0
    if [ -f "${d}/extra/summary.json" ] && grep -q '"wildchat"' "${d}/extra/summary.json"; then isdone=1; fi
    item "${isdone}" "tw-x-$(basename "${cfg}" .yaml)-s${s}-${m}" "${d}/extra/wildchat" "${d}/extra" || continue
    if [ -s "${d}/GPU_TYPE" ]; then PINNED="$(cat "${d}/GPU_TYPE")"; else choose_gpu "${GPUS_BIG}" 1; PINNED="${CHOSEN}"; fi
    cons="$(constraint_for "${d}")"
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} METHODS=${m} WHAT=wildchat GPU_TYPE=${PINNED}${cons:+ CONSTRAINT=${cons}} MEM=${mem} EXTRA_TIME=${xt} bash slurm/submit.sh extra"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" METHODS="${m}" TAG="" WHAT=wildchat NGPU=1 EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${PINNED}" CONSTRAINT="${cons}" MEM="${mem}" EXTRA_TIME="${xt}" bash slurm/submit.sh extra || true
  done
}
st_nov2_1() {
  local c
  nov_data_ok wildchat_eval.jsonl || return 0
  for c in "${NOV_WILD[@]}"; do wild_features "${c}"; done
  for c in "${NOV_WILD[@]}"; do wild_eval "${c}" 0 seq tripwire_prompt; done
}

# code task (Qwen2.5-1.5B): the predicted coverage is written before any run is trained
st_nov3_1() { nov_data_ok code_train.jsonl code_test.jsonl wildchat_eval.jsonl || return 0; features "${NOV_CODE}"; }
st_nov3_2() {
  local s js
  nov_data_ok code_train.jsonl code_test.jsonl wildchat_eval.jsonl || return 0
  js="runs/$(exp_of "${NOV_CODE}")/analysis/separability.json"
  if [ ! -f "${js}" ] && ! compgen -G "runs/$(exp_of "${NOV_CODE}")/s[0-9]*/seq" >/dev/null; then
    N_WAIT+=1; say "waiting  runs/$(exp_of "${NOV_CODE}") training (needs ${js}, step nov3.1)"; return 0
  fi
  for s in $(seeds "0 1 2"); do train "${NOV_CODE}" "${s}" 03:00:00 1 "${NOV_RUNS[@]}"; done
}
st_nov3_3() {
  want_addon_cfg "${NOV_CODE}" nov3.2 || return 0
  features "${NOV_CODE}" wait
  judge "${NOV_CODE}"
}

# baseline coverage, in the existing experiments
st_nov4_1() {
  local c s
  for c in "${NOV_RKL[@]}"; do
    for s in $(seeds "0 1 2"); do big train "${c}" "${s}" 12:00:00 1 refkl::; done
  done
}
st_nov4_2() {
  local s
  for s in $(seeds "1 2"); do big train "${O7}" "${s}" 08:00:00 "${NGPU7_F}" kl:: nullspace::; done
  for s in $(seeds "1 2"); do big scale "${O7}" "${s}" seq 0.5; done
}
st_nov4_3() { local s; for s in $(seeds "1 2"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" kl:: router::; done; }
st_nov4_4() { local c; for c in "${NOV_RKL[@]}" "${O7}" "${S7}"; do judge "${c}"; done; }

# ----------------------------------------------------------------------------------------------- tables
addon_nov_tables() {  # bash slurm/plan.sh nov-tables: paper/nov_results/
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/nov_results cfg name exp pair c
  ok() { "$@" || echo "  (did not finish: $*)"; }
  # switch reports (CPU): new experiments, the experiments with new baseline runs, and every WildChat experiment
  local list=("configs/nov/ifc_llama13.yaml:Novllamathirteenifc" "configs/nov/nq_llama13.yaml:Novllamathirteennq"
              "configs/nov/ifc_qwen14.yaml:Novqwenfourteenifc" "configs/nov/nq_qwen14.yaml:Novqwenfourteennq"
              "${NOV_CODE}:Novcode"
              "${IFC}:Refifc" "${O15}:Refnq" "${IF7}:Refifcseven" "${O7}:Refnqseven" "${OO7}:Refolmo" "${S15}:Refstream"
              "configs/fam/ifc_llama3b.yaml:FamllamathreeIfc" "configs/fam/nq_llama3b.yaml:FamllamathreeNq"
              "configs/fam/ifc_ds7.yaml:FamdeepseekIfc" "configs/fam/nq_ds7.yaml:FamdeepseekNq"
              "configs/fam/ifc_llama8.yaml:FamllamaeightIfc" "configs/fam/nq_llama8.yaml:FamllamaeightNq"
              "configs/fam/ifc_mistral7.yaml:FammistralIfc" "configs/fam/nq_mistral7.yaml:FammistralNq")
  for pair in "${list[@]}"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    if [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
      echo "  no gate features yet: left out"; continue
    fi
    mkdir -p "${out}/${exp}/switch"
    [[ "${exp}" == nov_* ]] && ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/switch_report.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch"
    cp "runs/${exp}/analysis/separability.json" "${out}/${exp}/switch/" 2>/dev/null || true
  done
  # per-method tables of the code task and of the experiments with new baseline runs
  for pair in "${NOV_CODE}:Novcode" "${O7}:Seven"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    mkdir -p "${out}/${exp}"
    ok python scripts/make_overlap_report.py --runs "runs/${exp}" --paper "${out}/${exp}" --name "${name}"
  done
  exp="$(exp_of "${S7}")"
  if compgen -G "runs/${exp}/s[12]/kl" >/dev/null || compgen -G "runs/${exp}/s[12]/router" >/dev/null; then
    mkdir -p "${out}/${exp}"
    ok python scripts/make_tables.py --runs "runs/${exp}" --paper "${out}/${exp}"
  fi
  ok python scripts/addon_report.py --plan nov --results "${out}" --runs runs --out "${out}/overview"
  ok python scripts/wildchat_report.py --results "${out}" --runs runs --out "${out}/overview"
  echo "tables in ${out}/"
}

addon_nov_collect() {  # bash slurm/plan.sh nov-collect: tables + status + checks in nov_results_<date>.tar.gz
  local out=paper/nov_results stamp d f
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}/checks"
  addon_nov_tables 2>&1 | tee "${out}/report.txt" || true
  PLAN_QUEUE=logs/plan_queue_nov bash slurm/plan.sh nov1,nov2,nov3,nov4 status > "${out}/status.txt" 2>&1 || true
  for d in runs/nov_*/s*/* runs/fam_nq_*/s*/refkl runs/v2_overlap_q7/s*/* runs/v2_stream_q7/s*/*; do
    [ -f "${d}/config.json" ] || continue
    printf '%s\t%s\t%s\n' "${d}" "$(isfile "${d}/DONE")" \
      "$(grep -o '"device": "[^"]*"' "${d}/config.json" | head -1 | cut -d'"' -f4)"
  done > "${out}/runs_gpu.tsv"
  for f in logs/tw-selftest-llama13-* logs/tw-selftest-qwen14-*; do
    [ -f "${f}" ] && cp "${f}" "${out}/checks/"
  done
  for f in data/wildchat_eval.meta.json data/code_meta.json; do
    [ -f "${f}" ] && cp "${f}" "${out}/checks/"
  done
  cp -r logs/autopilot_nov "${out}/autopilot" 2>/dev/null || true
  tar czf "nov_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/nov_results_${stamp}.tar.gz ($(du -h "nov_results_${stamp}.tar.gz" | cut -f1))"
}
