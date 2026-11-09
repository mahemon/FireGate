# shellcheck shell=bash
# Add-on plan, sourced by slurm/plan.sh (PLAN_SCALE.md): the main LoRA methods on two more model families,
# Llama-3.2-3B-Instruct and DeepSeek-LLM-7B-Chat. Kept apart from every other plan: configs in configs/fam/, runs in
# runs/fam_*/, tables in paper/fam_results/, its own autopilot (job cpufam, queue logs/plan_queue_fam, folder
# logs/autopilot_fam). Tier A = steps fam1.0-fam1.2, tier B = fam2.1, then fam1.3.
#
#   AP_PLAN=fam AP_TARGET=20 sbatch --job-name=cpufam slurm/autopilot.sbatch    # runs fam1.0, 1.1, 1.2, 2.1, 1.3
#   bash slurm/plan.sh fam1,fam2 status   done / queued / waiting / to do
#   bash slurm/plan.sh fam-tables         tables in paper/fam_results/ (CPU, login node)
#   bash slurm/plan.sh fam-collect        the same, packed into fam_results_<date>.tar.gz to send
# key:folder under models/:time limit per job:memory. Each model's configs: configs/fam/{ifc,nq}_<key>.yaml
# Oct 31 addition: llama8 (Llama-3.1-8B-Instruct, the Llama family at 8B) and mistral7 (Mistral-7B-Instruct-v0.3, a fifth
# family). The steps below run every model in this list; the finished llama3b and ds7 runs, checks and judges are skipped.
FAM_MODELS=("llama3b:Llama-3.2-3B-Instruct:06:00:00:32G" "ds7:deepseek-llm-7b-chat:12:00:00:64G"
            "llama8:Llama-3.1-8B-Instruct:12:00:00:64G" "mistral7:Mistral-7B-Instruct-v0.3:12:00:00:64G")
FAM_RUNS=(seq:: tripwire_prompt::)
STEPS+=(
  "fam1.0|tier A: real-model check of each downloaded model (A100: bf16 exactness, peak memory, hours per run); its training waits for it: 2 jobs, under 1 h each"
  "fam1.1|tier A: IF, seed 0, Seq-GRPO and Tripwire-prompt (LoRA), Llama-3.2-3B and DeepSeek-7B: 4 runs (3B 1.5-3 h, 7B 3-6 h)"
  "fam1.2|tier A: NQ, seed 0, the same: 4 runs (3B 1.5-3 h, 7B 3-6 h)"
  "fam2.1|tier B: seeds 1-2 of fam1.1 and fam1.2: 16 runs (3B 1.5-3 h, 7B 3-6 h)"
  "fam1.3|last: gate features (after seed 0's Tripwire-prompt) and LLM judge (after the runs) of each fam experiment: 8 short jobs (a judge runs again if runs finish after it)"
)
fam_field() {  # fam_field <key> <n>: field n of the model's entry (1 key, 2 folder, 3-5 time limit, 6 memory)
  local m
  for m in "${FAM_MODELS[@]}"; do
    [ "${m%%:*}" = "$1" ] || continue
    case "$2" in 2) cut -d: -f2 <<< "${m}" ;; 3) cut -d: -f3-5 <<< "${m}" ;; 6) cut -d: -f6 <<< "${m}" ;; esac
    return 0
  done
}
fam_model_ok() {  # fam_model_ok <key>: 0 if the model is downloaded; otherwise "skipped" (status) and 1
  local dir
  dir="models/$(fam_field "$1" 2)"
  [ -f "${dir}/config.json" ] && return 0
  say "skipped  ${1}: ${dir} is not downloaded (PLAN_SCALE.md); queue the steps again after downloading it"
  return 1
}
fam_check_state() {  # fam_check_state <key>: passed / failed / running / none, from the newest check log
  local log
  if queued "tw-selftest-$1-g1"; then echo running; return 0; fi
  log=$(ls -t logs/tw-selftest-"$1"-g1-*.out 2>/dev/null | head -1 || true)
  if [ -z "${log}" ]; then echo none; return 0; fi
  if ! grep -q "selftest finished" "${log}"; then echo failed; return 0; fi   # ended without finishing (crash, OOM)
  # a check counts as passed when nothing failed except the base model's GSM8K level (not a code problem)
  if grep "  FAIL  " "${log}" | grep -vq "solves GSM8K"; then echo failed; else echo passed; fi
}
st_fam1_0() {
  local m key dir st
  for m in "${FAM_MODELS[@]}"; do
    key="${m%%:*}"
    fam_model_ok "${key}" || continue
    dir="models/$(fam_field "${key}" 2)"
    st="$(fam_check_state "${key}")"
    case "${st}" in
      passed) N_DONE+=1; say "done     check of ${key}"; continue ;;
      running) N_QUEUED+=1; say "queued   check of ${key}"; continue ;;
      failed) N_FAIL+=1; echo "FAILED   check of ${key}: see the newest logs/tw-selftest-${key}-g1-*.out (delete it to check again)"
              continue ;;
    esac
    item 0 "tw-selftest-${key}-g1" "check of ${key}" "logs/fam_check_${key}" || continue
    if [ "${MODE}" = commands ]; then
      echo "LEVEL=real NAME=${key} MODEL_DIR=${dir} CONFIG=configs/fam/ifc_${key}.yaml GPU_TYPE=a100 MEM=$(fam_field "${key}" 6) bash slurm/submit.sh selftest"
      continue
    fi
    LEVEL=real NAME="${key}" MODEL_DIR="${dir}" CONFIG="configs/fam/ifc_${key}.yaml" NGPU=1 MEM="$(fam_field "${key}" 6)" \
      DEPEND="" BEGIN="" GPU_TYPE=a100 CONSTRAINT="" bash slurm/submit.sh selftest || true
  done
}
fam_train() {  # fam_train <task> "<seeds>": Seq-GRPO and Tripwire-prompt of every downloaded model whose check passed
  local m key s st
  for m in "${FAM_MODELS[@]}"; do
    key="${m%%:*}"
    fam_model_ok "${key}" || continue
    st="$(fam_check_state "${key}")"
    if [ "${st}" != passed ]; then
      if [ "${st}" = failed ]; then say "held     ${key} $1 runs: its check (step fam1.0) failed"
      else N_WAIT+=1; say "waiting  ${key} $1 runs (needs its check, step fam1.0)"; fi
      continue
    fi
    for s in $(seeds "$2"); do
      MEM_NOW="$(fam_field "${key}" 6)" train "configs/fam/$1_${key}.yaml" "${s}" "$(fam_field "${key}" 3)" 1 "${FAM_RUNS[@]}"
    done
  done
}
st_fam1_1() { fam_train ifc "0"; }
st_fam1_2() { fam_train nq "0"; }
st_fam2_1() { fam_train ifc "1 2"; fam_train nq "1 2"; }
st_fam1_3() {   # gate features once s0/tripwire_prompt is done (its deployed probe, its GPU model), then the judge
  local m key c t
  for m in "${FAM_MODELS[@]}"; do
    key="${m%%:*}"
    fam_model_ok "${key}" || continue
    for t in ifc nq; do
      c="configs/fam/${t}_${key}.yaml"
      want_addon_cfg "${c}" fam1.1 fam1.2 fam2.1 || continue
      MEM_NOW="$(fam_field "${key}" 6)" features "${c}" wait
      judge "${c}"
    done
  done
}
fam1_check() {  # bash slurm/plan.sh fam1 check: the same real-model checks as step fam1.0, submitted by hand
  refresh_jobs
  MODE=submit FREE=99 st_fam1_0
}

addon_fam_tables() {  # bash slurm/plan.sh fam-tables: paper/fam_results/<exp>/switch/ and overview/addon_fam.{md,tex,json}
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/fam_results t cfg name exp pair
  ok() { "$@" || echo "  (did not finish: $*)"; }
  local list=()
  for t in ifc nq; do
    list+=("configs/fam/${t}_llama3b.yaml:Famllamathree${t^}" "configs/fam/${t}_ds7.yaml:Famdeepseek${t^}"
           "configs/fam/${t}_llama8.yaml:Famllamaeight${t^}" "configs/fam/${t}_mistral7.yaml:Fammistral${t^}")
  done
  # the models of the main plan, for comparison (same tasks, data and settings)
  list+=("${IFC}:Refifc" "${O15}:Refnq" "${IF7}:Refifcseven" "${O7}:Refnqseven" "${OO7}:Refolmo")
  for pair in "${list[@]}"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    if [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
      echo "  no gate features yet (step fam1.3): left out"; continue
    fi
    mkdir -p "${out}/${exp}/switch"
    [[ "${exp}" == fam_* ]] && ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/switch_report.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch"
  done
  ok python scripts/addon_report.py --plan fam --results "${out}" --runs runs --out "${out}/overview"
  echo "tables in ${out}/"
}

addon_fam_collect() {  # bash slurm/plan.sh fam-collect: tables + status + real-model checks in fam_results_<date>.tar.gz
  local out=paper/fam_results stamp d f
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}/checks"
  addon_fam_tables 2>&1 | tee "${out}/report.txt" || true
  PLAN_QUEUE=logs/plan_queue_fam bash slurm/plan.sh fam1,fam2 status > "${out}/status.txt" 2>&1 || true
  for d in runs/fam_*/s*/*; do
    [ -f "${d}/config.json" ] || continue
    printf '%s\t%s\t%s\n' "${d}" "$(isfile "${d}/DONE")" \
      "$(grep -o '"device": "[^"]*"' "${d}/config.json" | head -1 | cut -d'"' -f4)"
  done > "${out}/runs_gpu.tsv"
  for f in logs/tw-selftest-llama3b-* logs/tw-selftest-ds7-* logs/tw-selftest-llama8-* logs/tw-selftest-mistral7-*; do
    [ -f "${f}" ] && cp "${f}" "${out}/checks/"
  done
  cp -r logs/autopilot_fam "${out}/autopilot" 2>/dev/null || true
  tar czf "fam_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/fam_results_${stamp}.tar.gz ($(du -h "fam_results_${stamp}.tar.gz" | cut -f1))"
}
