# shellcheck shell=bash
# Add-on plan for paper version 3 (PLAN_PAPER3.md), sourced by slurm/plan.sh: the stress test of harmful requests
# written to look like each task, a new task that overlaps the reference naturally (SQuAD 2.0), and baselines that
# protect the same behaviours. Kept apart from every other plan: config in configs/v3/, new runs in runs/v3_squad2_q15/
# and new method folders (safemix, refkl_protect, tripwire_prompt_protect) in the existing ladder experiments, tables in
# paper/v3_results/, its own autopilot (job cpup3, queue logs/plan_queue_p3, folder logs/autopilot_p3).
#
#   AP_PLAN=p3 AP_COUNT=tw AP_TARGET=12 sbatch --job-name=cpup3 slurm/autopilot.sbatch
#   bash slurm/plan.sh p3 status                 done / queued / waiting / to do
#   bash slurm/plan.sh p3-tables                 step-1 analyses and every v3 table in paper/v3_results/ (CPU, login node)
#   bash slurm/plan.sh p3-collect                the same, packed into v3_results_<date>.tar.gz to send
P3_SQ=configs/v3/squad2_q15.yaml
P3_PROT="tripwire.protect=advbench,general"
# SQuAD 2.0: Seq-GRPO, Tripwire-prompt (every reference component), Tripwire-prompt trained under a tripwire that
# protects harmful requests and task-like prompts only, Ref-KL on those components, and safety replay
P3_SQ_RUNS=(seq:: tripwire_prompt:: "tripwire_prompt:protect:${P3_PROT}" "refkl:protect:${P3_PROT}" safemix::)
# the overlap ladder's middle and top rungs: the baselines that protect the same behaviours as the protect tripwire
P3_OVL_RUNS=("refkl:protect:${P3_PROT}" safemix:: "tripwire_prompt:protect:${P3_PROT}")
# experiments of the stress test (their tasks have stress families, twcore/stress.py)
P3_STRESS=("${IFC}" "${O15}" "${CCX}" "${CON}" "${S15}")
P3_STRESS_BIG=("${IF7}")
STEPS+=(
  "p3.1|stress test, no training: gate features of the stress families added to 6 experiments (IF 1.5B and 7B, NQ, Concise-ctx, Concise, 3-task stream): 6 jobs, 20-45 min each"
  "p3.2|stress test, no training: finished runs evaluated on the stress families (seed 0 Seq-GRPO and Tripwire-prompt of the 6 experiments, seeds 1-2 Seq-GRPO on IF): 14 jobs, 20-60 min each"
  "p3.3|SQuAD 2.0 (1.5B), before any training: gate features and the coverage beta (separability.json): 1 job, under 1 h"
  "p3.4|SQuAD 2.0, seeds 0-2: Seq-GRPO, Tripwire-prompt, Tripwire-prompt protecting harmful and task-like prompts, Ref-KL on those, safety replay: 15 runs, up to 3 h each (waits for p3.3)"
  "p3.5|should: baselines that protect the same behaviours, seeds 0-2: on Concise and Concise-ctx safety replay, Ref-KL on the protected components and Tripwire-prompt trained under the protect tripwire (18 runs); on IF safety replay (3 runs); up to 3 h each"
  "p3.6|last: SQuAD 2.0 gate features again (after its seed-0 Tripwire-prompt), stress evaluations of the new seed-0 runs, LLM judges of SQuAD 2.0, Concise, Concise-ctx and IF: up to 12 short jobs"
  "p3.7|larger check of the certified level, no training: gate features of 10,000 fresh Dolly requests added to the NQ (Qwen 1.5B and 7B, OLMo-2 7B) and stream (3 tasks 1.5B and 7B, 5 tasks) experiments: 6 jobs, a few minutes each (also fed by p3.1 while p3.1 is queued)"
  "p3.8|the protect gate's remedy, no training (paper v3.4): gate features of harmful fitting and calibration requests followed by a passage added to Concise, Concise-ctx and SQuAD 2.0: 3 jobs, a few minutes each; then sbatch slurm/p3collect.sbatch"
  "p3.10|second-round forms, no training (paper v4.2, step p3.10b): gate features of eight forms of the held-out harmful requests written before any gate sees them (twcore/stress.py st2_*), added to 7 experiments (Concise, Concise-ctx, SQuAD 2.0, NQ, IF 1.5B and 7B, 3-task stream): 7 jobs, a few minutes each; then sbatch slurm/p3collect.sbatch, which also runs the richer scores of p3.10a (CPU)"
  "p3.12|pre-registered external sets, no training: gate features of HarmBench, HarmBench contextual and StrongREJECT (data/ext_*.jsonl, scripts/prepare_external.py on the login node first) added to 7 experiments, and Seq-GRPO seed 0 of each generating on them (final and base model): 14 jobs, a few minutes to 1 h each; then sbatch slurm/p3ext.sbatch"
  "p3.15|matched budget, more ungated operating points on NQ 1.5B: Seq-GRPO's adapter scaled by 0.02, 0.05, 0.1 (seeds 0-2, 9 short evaluation jobs) and + KL at beta 10 and 100 (seeds 0-2, 6 runs up to 3 h each); then sbatch slurm/p3ext.sbatch"
  "p3.25a|EWC and A-GEM on the 3-task stream (1.5B), seed 0 (Nov 5): the two baselines planned for the stream in step w5.4 and missing there, at the config's values; runs already finished are skipped: at most 2 runs, up to 8 h each; then sbatch slurm/p3clrep.sbatch"
  "p3.26a|prospective test on a new task (TriviaQA, Llama-3.2-3B, Nov 8), before any training: gate features and the coverage (waits for data/trivia_*.jsonl and PREREG_p3_26.txt): 1 job, under 1 h"
  "p3.26b|prospective test, seeds 0-2: Seq-GRPO and Tripwire-prompt at the frozen settings (waits for the gate record PREREG_p3_26_gate.txt): 6 runs, 1.5-3 h each"
  "p3.26c|prospective test, last: gate features again (after seed 0's Tripwire-prompt) and seed-0 Seq-GRPO on the external harmful sets: 2 jobs, under 1 h each; then sbatch slurm/p3trivrep.sbatch"
)
# p3.12 (Nov 2): the external sets of twcore/external.py, in the experiments of p3.10b
P3_EXT_DATA=(ext_harmbench.jsonl ext_harmbench_ctx.jsonl ext_strongreject.jsonl)
# p3.15 (Nov 2): + KL with stronger coefficients on NQ 1.5B (the folders kl_b10 and kl_b100)
P3_BUDGET_RUNS=(kl:b10:grpo.kl_beta=10.0 kl:b100:grpo.kl_beta=100.0)
# p3.25 (Nov 5): EWC and A-GEM on the 3-task stream, the two baselines of step w5.4 that the stream lacks (seed 0, the
# config's values). A wider sweep was written first and dropped before any run: NQ already sweeps about 26 ungated
# settings (Table matched), so only what was planned and is missing runs. scripts/stream_sweep_report.py reads the runs.
P3_CL15=(ewc:: agem::)
p3_data_ok() {  # p3_data_ok <file under data/>...: 0 if every file exists; otherwise "waiting" and 1
  local f
  for f in "$@"; do
    if [ ! -s "data/${f}" ]; then
      local how="prepare_data.py --only-squad2"
      [[ "${f}" == ext_* ]] && how="prepare_external.py --data data"
      [[ "${f}" == trivia_* ]] && how="prepare_data.py --only-trivia --data data"
      N_WAIT+=1; say "waiting  data/${f} is missing (PLAN_PAPER3.md: python scripts/${how} on the login node)"
      return 1
    fi
  done
}
p3_features_add() {  # p3_features_add <config> [<key in gate_features.json> <label>]: prompt sets added to the stored
                     # gate features (one short job, no training) on the GPU model they were computed on; default:
                     # the stress families (p3.1); p3.7: '"eval__dolly_fresh"' dolly_fresh
  local cfg="$1" key="${2:-\"eval__st_}" lab="${3:-stress}" exp d js dev gt cons isdone=0 ft="00:30:00"
  [ "${MEM_NOW:-${MEM_SMALL}}" = "${MEM_BIG}" ] && ft="00:45:00"
  exp="$(exp_of "${cfg}")"
  skipped "features" "features" && return 0
  d="runs/${exp}/analysis/gate_features"
  js="runs/${exp}/analysis/gate_features.json"
  if [ ! -f "${js}" ] || [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
    N_WAIT+=1; say "waiting  ${d} (no stored gate features of ${exp})"; return 0
  fi
  grep -q "${key}" "${js}" && isdone=1
  item "${isdone}" "tw-gf-$(basename "${cfg}" .yaml)" "${d}/${lab}" "${d}" || return 0
  dev=$(grep -o '"device": "[^"]*"' "${js}" | head -1 | cut -d'"' -f4)
  case "${dev}" in *H200*) gt=h200 ;; *A100*) gt=a100 ;; *) gt="" ;; esac
  cons="$(constraint_of_device "${dev}")"
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${gt}${cons:+ CONSTRAINT=${cons}} MEM=${MEM_NOW:-${MEM_SMALL}} FEATURES_TIME=${ft} FEATURES_ARGS=--add bash slurm/submit.sh features"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${gt}" CONSTRAINT="${cons}" \
    MEM="${MEM_NOW:-${MEM_SMALL}}" FEATURES_TIME="${ft}" FEATURES_ARGS=--add bash slurm/submit.sh features || true
}
p3_stress_eval() {  # p3_stress_eval <config> <seed> <method[:tag]>...: eval_extra.py --what advwrap,stress of finished runs
                    # (P3_WHAT=external: the external sets of p3.12 instead, done when the summary has "external")
  local cfg="$1" s="$2"; shift 2
  local what="${P3_WHAT:-advwrap,stress}" key="${P3_WHAT:-stress}" jx="x${P3_WHAT:+e}"
  local exp spec m tag name d isdone xt="00:45:00"
  [ "${MEM_NOW:-${MEM_SMALL}}" = "${MEM_BIG}" ] && xt="01:30:00"
  exp="$(exp_of "${cfg}")"
  for spec in "$@"; do
    IFS=: read -r m tag <<< "${spec}"
    name="${m}${tag:+_${tag}}"
    skipped "${name}" "${m}" && continue
    d="runs/${exp}/s${s}/${name}"
    if [ ! -f "${d}/DONE" ]; then N_WAIT+=1; say "waiting  ${d}/extra/${key} (needs ${d})"; continue; fi
    isdone=0
    if [ -f "${d}/extra/summary.json" ] && grep -q "\"${key}\"" "${d}/extra/summary.json"; then isdone=1; fi
    item "${isdone}" "tw-${jx}-$(basename "${cfg}" .yaml)-s${s}-${name}" "${d}/extra/${key}" "${d}/extra" || continue
    if [ -s "${d}/GPU_TYPE" ]; then PINNED="$(cat "${d}/GPU_TYPE")"; else choose_gpu "${GPUS_NOW:-${GPUS_SMALL}}" 1; PINNED="${CHOSEN}"; fi
    cons="$(constraint_for "${d}")"
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} METHODS=${m}${tag:+ TAG=${tag}} WHAT=${what} GPU_TYPE=${PINNED}${cons:+ CONSTRAINT=${cons}} MEM=${MEM_NOW:-${MEM_SMALL}} EXTRA_TIME=${xt} bash slurm/submit.sh extra"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" METHODS="${m}" TAG="${tag}" WHAT="${what}" NGPU=1 EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${PINNED}" CONSTRAINT="${cons}" MEM="${MEM_NOW:-${MEM_SMALL}}" EXTRA_TIME="${xt}" bash slurm/submit.sh extra || true
  done
}
st_p3_1() {
  local c
  for c in "${P3_STRESS[@]}"; do p3_features_add "${c}"; done
  for c in "${P3_STRESS_BIG[@]}"; do big p3_features_add "${c}"; done
  st_p3_7   # the fresh-Dolly features are just as short: fed with p3.1 while it is queued (one job per config at a time)
}
# p3.7: experiments without a ladder task (their Dolly rows 2000-11999 are unused; twcore/data.py FRESH_DOLLY)
P3_FRESH=("${O15}" "${S15}" "${S5}")
P3_FRESH_BIG=("${O7}" "${OO7}" "${S7}")
# p3.8 (Nov 2, paper v3.4): advwrap_<task> of twcore/stress.py, stored as wrap__<task>__fit / __cal and
# eval__advwrap_<task>; scripts/switch_report.py and scripts/v3_analyses.py then build the gate "protect_remedy"
st_p3_8() {
  local c
  for c in "${CON}" "${CCX}" "${P3_SQ}"; do p3_features_add "${c}" '"wrap__' protect_remedy; done
}
# p3.10 (Nov 2, paper v4.2): the second-round forms st2_* (twcore/stress.py), the same in every experiment
P3_ST2=("${CON}" "${CCX}" "${P3_SQ}" "${O15}" "${IFC}" "${S15}")
P3_ST2_BIG=("${IF7}")
st_p3_10() {
  local c
  for c in "${P3_ST2[@]}"; do p3_features_add "${c}" '"eval__st2_' stress2; done
  for c in "${P3_ST2_BIG[@]}"; do big p3_features_add "${c}" '"eval__st2_' stress2; done
}
# p3.12 (Nov 2): external sets, features in the experiments of p3.10b, then Seq-GRPO seed 0 of each on them
st_p3_12() {
  local c s
  p3_data_ok "${P3_EXT_DATA[@]}" || return 0
  for c in "${P3_ST2[@]}"; do p3_features_add "${c}" '"eval__ext_' external; done
  for c in "${P3_ST2_BIG[@]}"; do big p3_features_add "${c}" '"eval__ext_' external; done
  for s in $(seeds "0"); do
    for c in "${P3_ST2[@]}"; do P3_WHAT=external p3_stress_eval "${c}" "${s}" seq; done
    for c in "${P3_ST2_BIG[@]}"; do P3_WHAT=external big p3_stress_eval "${c}" "${s}" seq; done
  done
}
# p3.15 (Nov 2): more operating points of ungated training on NQ 1.5B, closer to the gate's budget
st_p3_15() {
  local s
  for s in $(seeds "0 1 2"); do
    scale "${O15}" "${s}" seq 0.02 0.05 0.1
    train "${O15}" "${s}" 03:00:00 1 "${P3_BUDGET_RUNS[@]}"
  done
}
# p3.25 (Nov 5): EWC and A-GEM on the 1.5B stream (PLAN_PAPER3.md, step p3.25)
st_p3_25a() { local s; for s in $(seeds "0"); do train "${S15}" "${s}" 08:00:00 1 "${P3_CL15[@]}"; done; }
# p3.26 (Nov 8): the frozen recipe on a task no earlier step used (PLAN_PAPER3.md, step p3.26). The predictions and
# inputs are recorded before any job (PREREG_p3_26.txt), the gate's numbers after the features and before training
# (PREREG_p3_26_gate.txt); both by python scripts/prereg_record.py --step p3.26 on the login node.
P3_TQ=configs/p3/trivia_llama3b.yaml
P3_TQ_MEM=32G   # as the Llama-3.2-3B runs of slurm/plans/fam.sh
P3_TQ_REC=paper/v3_results/p3_26/PREREG_p3_26.txt
P3_TQ_GATE=paper/v3_results/p3_26/PREREG_p3_26_gate.txt
p3_tq_ok() {  # 0 if the TriviaQA data and the first record exist; otherwise "waiting" and 1
  p3_data_ok trivia_train.jsonl trivia_test.jsonl || return 1
  if [ ! -s "${P3_TQ_REC}" ]; then
    N_WAIT+=1; say "waiting  ${P3_TQ_REC} (python scripts/prereg_record.py --step p3.26 on the login node)"; return 1
  fi
}
st_p3_26a() { p3_tq_ok || return 0; MEM_NOW="${P3_TQ_MEM}" features "${P3_TQ}"; }
st_p3_26b() {
  local s
  p3_tq_ok || return 0
  if [ ! -s "${P3_TQ_GATE}" ]; then
    N_WAIT+=1; say "waiting  ${P3_TQ_GATE} (after step p3.26a: python scripts/prereg_record.py --step p3.26 again)"
    return 0
  fi
  for s in $(seeds "0 1 2"); do MEM_NOW="${P3_TQ_MEM}" train "${P3_TQ}" "${s}" 06:00:00 1 seq:: tripwire_prompt::; done
}
st_p3_26c() {
  local s
  want_addon_cfg "${P3_TQ}" p3.26b || return 0
  MEM_NOW="${P3_TQ_MEM}" features "${P3_TQ}" wait
  for s in $(seeds "0"); do MEM_NOW="${P3_TQ_MEM}" P3_WHAT=external p3_stress_eval "${P3_TQ}" "${s}" seq; done
}
st_p3_7() {
  local c
  for c in "${P3_FRESH[@]}"; do p3_features_add "${c}" '"eval__dolly_fresh"' dolly_fresh; done
  for c in "${P3_FRESH_BIG[@]}"; do big p3_features_add "${c}" '"eval__dolly_fresh"' dolly_fresh; done
}
st_p3_2() {
  local c s
  for c in "${P3_STRESS[@]}"; do for s in $(seeds "0"); do p3_stress_eval "${c}" "${s}" seq tripwire_prompt; done; done
  for c in "${P3_STRESS_BIG[@]}"; do for s in $(seeds "0"); do big p3_stress_eval "${c}" "${s}" seq tripwire_prompt; done; done
  for s in $(seeds "1 2"); do p3_stress_eval "${IFC}" "${s}" seq; done
}
st_p3_3() { p3_data_ok squad2_train.jsonl squad2_test.jsonl || return 0; features "${P3_SQ}"; }
st_p3_4() {
  local s js
  p3_data_ok squad2_train.jsonl squad2_test.jsonl || return 0
  js="runs/$(exp_of "${P3_SQ}")/analysis/separability.json"
  if [ ! -f "${js}" ] && ! compgen -G "runs/$(exp_of "${P3_SQ}")/s[0-9]*/seq" >/dev/null; then
    N_WAIT+=1; say "waiting  runs/$(exp_of "${P3_SQ}") training (needs ${js}, step p3.3)"; return 0
  fi
  for s in $(seeds "0 1 2"); do train "${P3_SQ}" "${s}" 03:00:00 1 "${P3_SQ_RUNS[@]}"; done
}
st_p3_5() {
  local s
  for s in $(seeds "0 1 2"); do
    train "${CCX}" "${s}" 03:00:00 1 "${P3_OVL_RUNS[@]}"
    train "${CON}" "${s}" 03:00:00 1 "${P3_OVL_RUNS[@]}"
    train "${IFC}" "${s}" 03:00:00 1 safemix::
  done
}
st_p3_6() {
  local c s
  if want_addon_cfg "${P3_SQ}" p3.4; then
    features "${P3_SQ}" wait
    for s in $(seeds "0"); do p3_stress_eval "${P3_SQ}" "${s}" seq tripwire_prompt tripwire_prompt:protect safemix; done
    judge "${P3_SQ}"
  fi
  # the new baselines of p3.5: only if that step was queued or some of its runs exist (it is optional)
  if awk -F'|' '$1=="p3.5"{f=1} END{exit !f}' "${QUEUE}" 2>/dev/null \
     || compgen -G "runs/$(exp_of "${CCX}")/s[0-9]*/safemix" >/dev/null \
     || compgen -G "runs/$(exp_of "${CON}")/s[0-9]*/safemix" >/dev/null; then
    for s in $(seeds "0"); do
      p3_stress_eval "${CCX}" "${s}" safemix tripwire_prompt:protect
      p3_stress_eval "${CON}" "${s}" safemix tripwire_prompt:protect
      p3_stress_eval "${IFC}" "${s}" safemix
    done
    for c in "${CCX}" "${CON}" "${IFC}"; do judge "${c}"; done
  fi
}

# ----------------------------------------------------------------------------------------------- tables
addon_p3_tables() {  # bash slurm/plan.sh p3-tables: paper/v3_results/ (CPU)
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/v3_results exp cfg name pair
  ok() { "$@" || echo "  (did not finish: $*)"; }
  mkdir -p "${out}"
  # step 1 and the stress test: every experiment with stored gate features (protected components after the name)
  local args=()
  for pair in "${CON}:Concise:advbench,general" "${CCX}:Concisectx:advbench,general" "${P3_SQ}:Squad:advbench,general" \
              "${O15}:Nq:advbench,general" "${IFC}:Ifc:" "${S15}:Stream:" "${O7}:Seven:" "${IF7}:Ifcseven:" "${OO7}:Olmo:" \
              "${S7}:Streamseven:" "${S5}:Streamfive:"; do
    cfg="${pair%%:*}"
    [ -f "runs/$(exp_of "${cfg}")/analysis/gate_features.npz" ] && args+=(--exp "${pair}")
  done
  if [ ${#args[@]} -gt 0 ]; then
    # one experiment per process (a single process holding all of them was killed for memory on the login node),
    # then the tables from the per-experiment results; on the login node prefer: sbatch slurm/p3collect.sbatch
    local i
    for ((i = 1; i < ${#args[@]}; i += 2)); do
      ok python scripts/v3_analyses.py --exp "${args[$i]}" --out "${out}" --method-table Squad,Concise,Concisectx
    done
    ok python scripts/v3_analyses.py "${args[@]}" --out "${out}" --method-table Squad,Concise,Concisectx --merge
  fi
  # per-method and switch tables of SQuAD 2.0 and of the ladder rungs with the new baselines
  for pair in "${P3_SQ}:Squad:advbench,general" "${CCX}:Concisectx:advbench,general" "${CON}:Concise:advbench,general" \
              "${IFC}:Ifc:"; do
    IFS=: read -r cfg name prot <<< "${pair}"
    exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    mkdir -p "${out}/${exp}/switch"
    ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/make_overlap_report.py --runs "runs/${exp}" --paper "${out}/${exp}" --name "${name}"
    if [ -f "runs/${exp}/analysis/gate_features.npz" ]; then
      ok python scripts/switch_report.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch" ${prot:+--protect "${prot}"}
    fi
    cp "runs/${exp}/analysis/separability.json" "${out}/${exp}/" 2>/dev/null || true
  done
  # p3.10a (Nov 2): richer scores at the same level, on the references whose coverage the paper reports
  for pair in "${CON}:protect_remedy:advbench,general" "${CCX}:protect_remedy:advbench,general" \
              "${P3_SQ}:protect_remedy:advbench,general" "${P3_SQ}:deployed:" "${O15}:deployed:"; do
    IFS=: read -r cfg name prot <<< "${pair}"
    exp="$(exp_of "${cfg}")"
    [ -f "runs/${exp}/analysis/gate_features.npz" ] || continue
    ok python scripts/score_classes.py --config "${cfg}" --variant "${name}" ${prot:+--protect "${prot}"} \
      --out "${out}/p3_10/${exp}/scores_${name}"
  done
  # p3.10b (Nov 2): firing on the second-round forms, every experiment that has them
  args=()
  for pair in "${CON}:Concise:advbench,general" "${CCX}:Concisectx:advbench,general" "${P3_SQ}:Squad:advbench,general" \
              "${O15}:Nq:" "${IFC}:Ifc:" "${IF7}:Ifcseven:" "${S15}:Stream:"; do
    cfg="${pair%%:*}"
    [ -f "runs/$(exp_of "${cfg}")/analysis/gate_features.npz" ] && args+=(--exp "${pair}")
  done
  [ ${#args[@]} -gt 0 ] && ok python scripts/stress2_report.py "${args[@]}" --out "${out}/p3_10"
  # p3.12 and p3.15 (Nov 2): the external sets and the matched budget (p3.13 and p3.14: sbatch slurm/p3ext.sbatch)
  if compgen -G "data/ext_*.jsonl" >/dev/null; then
    [ ${#args[@]} -gt 0 ] && ok python scripts/external_report.py "${args[@]}" --out "${out}/p3_12"
  fi
  [ -d "runs/$(exp_of "${O15}")" ] && ok python scripts/budget_report.py --runs "runs/$(exp_of "${O15}")" --out "${out}/p3_15"
  # p3.25 (Nov 5): continual-learning baselines at their best settings on the streams
  for c in "${S15}" "${S7}"; do
    [ -d "runs/$(exp_of "${c}")" ] && ok python scripts/stream_sweep_report.py --runs "runs/$(exp_of "${c}")" --out "${out}/p3_25"
  done
  echo "tables in ${out}/"
}

addon_p3_collect() {  # bash slurm/plan.sh p3-collect: tables + status + checks in v3_results_<date>.tar.gz
  local out=paper/v3_results stamp d
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}/checks"
  addon_p3_tables 2>&1 | tee "${out}/report.txt" || true
  PLAN_QUEUE=logs/plan_queue_p3 bash slurm/plan.sh p3 status > "${out}/status.txt" 2>&1 || true
  for d in runs/v3_*/s*/* runs/v2_ladder_*/s*/safemix runs/v2_ladder_*/s*/*_protect; do
    [ -f "${d}/config.json" ] || continue
    printf '%s\t%s\t%s\n' "${d}" "$(isfile "${d}/DONE")" \
      "$(grep -o '"device": "[^"]*"' "${d}/config.json" | head -1 | cut -d'"' -f4)"
  done > "${out}/runs_gpu.tsv"
  cp -r logs/autopilot_p3 "${out}/autopilot" 2>/dev/null || true
  tar czf "v3_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/v3_results_${stamp}.tar.gz ($(du -h "v3_results_${stamp}.tar.gz" | cut -f1))"
}
