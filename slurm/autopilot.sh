#!/bin/bash
# Autopilot: keeps the planned runs going for days without a person at the keyboard.
#
#   mkdir -p logs/autopilot && sbatch slurm/autopilot.sbatch     # a 1-CPU job on the nova partition (10 days)
#   bash slurm/autopilot.sh                                      # or in a terminal (tmux) on the login node
#
# What it does, every minute:
#   * counts your GPU jobs (running + pending, held ones included) in the GPU partition ($PARTITION, from
#     slurm/cluster.env) and keeps that total at AP_TARGET (9): when it is below, it submits the next waiting runs
#     of the plan's queue (slurm/plan.sh feed: steps in queue order, H200 if one is free else A100, a run continues
#     from its checkpoint, a finished or queued run is never submitted twice);
#   * when a submission fails (sbatch errors, e.g. between 04:30 and 07:00) it retries every AP_RETRY_MIN (30)
#     minutes; when nothing could be submitted because the remaining runs wait for others to finish, it tries again
#     as soon as one of your jobs ends, and at the latest after AP_RETRY_MIN minutes;
#   * once a day at AP_REPORT_AT (07:30) writes logs/autopilot/report_<date>.txt: per step done / running / pending /
#     to do; runs stopped in the middle (they continue from their checkpoint when resubmitted); runs marked FAILED;
#     every job of the last 24 hours that ended with an error, with its log copied to logs/autopilot/errors_<date>/.
# At the first start it runs the CPU unit selftest once (the new baselines are new training code) and queues the
# steps of AP_STEPS in that order; steps in AP_GATED are queued only if the selftest passed.
# Diagnosis files in logs/autopilot/: autopilot.log (readable log of every decision), events.tsv (one line per
# event: time, event, job id, job name, detail; events QUEUE, SUBMIT, SUBMIT_FAIL, NEW, START, END, ERROR, BLACKOUT,
# PAUSE, REPORT), feed.log (the full output of every plan.sh call), errors_<date>/ (logs of jobs that ended badly).
# It stops after AP_DAYS (10) days, when every queued step is finished, or when logs/autopilot/STOP exists;
# logs/autopilot/PAUSE pauses submissions (it keeps reporting). As a Slurm job it resubmits itself shortly before
# its time limit, so a partition limit shorter than 7 days is fine.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
if [ -f slurm/cluster.env ]; then source slurm/cluster.env; else source slurm/cluster.env.example; fi
AP_TARGET="${AP_TARGET:-9}"             # running + pending GPU jobs to keep (AP_TARGET=20 to keep more queued)
AP_COUNT="${AP_COUNT:-account}"         # account: all your GPU jobs in $PARTITION count; tw: only tw-* jobs
AP_CHECK_SEC="${AP_CHECK_SEC:-60}"
AP_RETRY_MIN="${AP_RETRY_MIN:-30}"
AP_REPORT_AT="${AP_REPORT_AT:-07:30}"
AP_DAYS="${AP_DAYS:-10}"
# plan v6 (Oct 28, PLAN_v5.md + PLAN_v6.md): the v5 steps and the review add-ons. The stronger NQ baselines first
# (new training code, so they are queued only if the CPU unit selftest passes), then the middle rung (its features,
# then its training if the AUROC is in range), the short v5 jobs (new pools, second judge), the missing seeds, the
# stream robustness evaluations, and the judges last (each waits for its runs; v5.4 is gated with v5.1).
# Earlier plans: AP_PLAN=v5 AP_STEPS=v5.1,v5.2,v5.3,v5.4 AP_GATED=v5.1,v5.4
#                AP_PLAN=v4 AP_STEPS=v4.1,v4.2,v4.3,v4.5,v4.6,v4.4 AP_GATED=""
# plan v7 adds the pilot of PLAN_v7.md (full-rank updates and SFT on IF, seed 0) after the middle rung, gated on a
# second selftest (--level v7), so that a problem in that new code holds back only the pilot.
AP_PLAN="${AP_PLAN:-v7}"
case "${AP_PLAN}" in   # add-on plans (slurm/plans/), each fully separate from the other plans: its own steps,
                       # queue (PLAN_QUEUE), folder (AP_DIR) and Slurm job name; submit them with
                       # sbatch --job-name=cpusft / cpufam (slurm/autopilot.sbatch); the defaults below are plan v7's
  sft)   # PLAN_SFT.md: gated on the unit selftest (the shared trainer) and the sft selftest (the SFT data sources)
    AP_DIR="${AP_DIR:-logs/autopilot_sft}"; PLAN_QUEUE="${PLAN_QUEUE:-logs/plan_queue_sft}"; AP_JOBS_RE="${AP_JOBS_RE-sft_}"
    AP_STEPS="${AP_STEPS:-sft1.1,sft1.2,sft1.3}"; AP_GATED="${AP_GATED:-sft1.1,sft1.2,sft1.3}"
    AP_GATED2="${AP_GATED2:-sft1.1,sft1.2,sft1.3}"
    AP_SELFTEST2="${AP_SELFTEST2:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level sft}" ;;
  fam)   # PLAN_SCALE.md, tiers A and B: gated on the unit selftest and the fam selftest (chat templates); each
         # model's training also waits for its real-model check (step fam1.0) to pass
    AP_DIR="${AP_DIR:-logs/autopilot_fam}"; PLAN_QUEUE="${PLAN_QUEUE:-logs/plan_queue_fam}"
    AP_JOBS_RE="${AP_JOBS_RE-fam_|selftest-(llama3b|ds7|llama8|mistral7)-}"
    AP_STEPS="${AP_STEPS:-fam1.0,fam1.1,fam1.2,fam2.1,fam1.3}"; AP_GATED="${AP_GATED:-fam1.0,fam1.1,fam1.2,fam2.1,fam1.3}"
    AP_GATED2="${AP_GATED2:-fam1.0,fam1.1,fam1.2,fam2.1,fam1.3}"
    AP_SELFTEST2="${AP_SELFTEST2:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level fam}" ;;
  nov)   # PLAN_NOV.md: gated on the unit selftest and the nov selftest (WildChat pool, code task, configs). Not queued
         # here: the larger models (nov1.x, later: PLAN_QUEUE=logs/plan_queue_nov bash slurm/plan.sh nov1), Ref-KL at 8B
         # (nov4.1, submitted directly to another partition: slurm/refkl_nova.sh) and the optional nov4.3
    AP_DIR="${AP_DIR:-logs/autopilot_nov}"; PLAN_QUEUE="${PLAN_QUEUE:-logs/plan_queue_nov}"
    AP_JOBS_RE="${AP_JOBS_RE-_llama13-|_qwen14-|selftest-(llama13|qwen14)-|code_q15|tw-x-|tw-gf-|tw-jg-|overlap_q7-s[0-9]+-(kl|nullspace)$|sc-overlap_q7-}"
    AP_STEPS="${AP_STEPS:-nov2.1,nov3.1,nov3.2,nov4.2,nov3.3,nov4.4}"
    AP_GATED="${AP_GATED:-nov2.1,nov3.1,nov3.2,nov4.2,nov3.3,nov4.4,nov1.0,nov1.1,nov1.2,nov1.3}"
    AP_GATED2="${AP_GATED2:-nov2.1,nov3.1,nov3.2,nov3.3,nov1.0,nov1.1,nov1.2,nov1.3}"
    AP_SELFTEST2="${AP_SELFTEST2:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level nov}" ;;
  p3)    # PLAN_PAPER3.md (paper version 3): gated on the unit selftest (the shared trainer: safety replay, protected
         # components) and the p3 selftest (SQuAD 2.0 task, stress families, safety replay, tripwire.protect, analyses)
    AP_DIR="${AP_DIR:-logs/autopilot_p3}"; PLAN_QUEUE="${PLAN_QUEUE:-logs/plan_queue_p3}"
    AP_JOBS_RE="${AP_JOBS_RE-squad2_q15|tw-x-|tw-gf-|tw-jg-|-(safemix|refkl_protect|tripwire_prompt_protect)$}"
    AP_STEPS="${AP_STEPS:-p3.1,p3.7,p3.2,p3.3,p3.4,p3.5,p3.6}"
    AP_GATED="${AP_GATED:-p3.4,p3.5}"
    AP_GATED2="${AP_GATED2:-p3.1,p3.7,p3.2,p3.3,p3.4,p3.5,p3.6}"
    AP_SELFTEST2="${AP_SELFTEST2:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level p3}" ;;
esac
AP_DIR="${AP_DIR:-logs/autopilot}"
PLAN_QUEUE="${PLAN_QUEUE:-logs/plan_queue}"   # slurm/plan.sh reads it; the jobs inherit it and feed the same queue
export PLAN_QUEUE
MAXF=logs/plan_max_jobs                      # the job limit that slurm/plan.sh (and the jobs' own feeding) uses
[ "${PLAN_QUEUE}" = logs/plan_queue ] || MAXF="${PLAN_QUEUE}.max_jobs"
AP_JOBS_RE="${AP_JOBS_RE-}"   # names of this plan's jobs (regex): the plan is finished when none is left
AP_STEPS="${AP_STEPS:-v5.1,v6.1,v6.2,v7.1,v5.2,v5.3,v6.3,v6.4,v5.4,v6.5,v7.2}"
AP_GATED="${AP_GATED:-v5.1,v5.4}"   # steps queued only if the selftest passed (v5.1 trains with new code)
AP_GATED2="${AP_GATED2:-v7.1,v7.2}"   # steps queued only if the second selftest passed (v7: dense updates, SFT)
AP_SELFTEST2="${AP_SELFTEST2:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level v7}"
AP_RESET_QUEUE="${AP_RESET_QUEUE:-1}"   # first start of a plan: empty the old queue before queuing AP_STEPS
AP_SELFTEST="${AP_SELFTEST:-python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level unit}"
STATE="${AP_DIR}/state_${AP_PLAN}"     # one state per plan: a new plan starts fresh (queue, 10 days, selftest)
mkdir -p "${AP_DIR}"
touch "${STATE}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${AP_DIR}/autopilot.log"; }
event() {  # event <EVENT> <job id> <job name> <detail...>: one tab-separated line in events.tsv
  local ev="$1" id="${2:--}" nm="${3:--}"; shift 3 || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${ev}" "${id}" "${nm}" "$*" >> "${AP_DIR}/events.tsv"
}
state_get() { grep -m1 "^$1=" "${STATE}" 2>/dev/null | cut -d= -f2-; }
state_set() { grep -v "^$1=" "${STATE}" > "${STATE}.tmp" 2>/dev/null; echo "$1=$2" >> "${STATE}.tmp"; mv "${STATE}.tmp" "${STATE}"; }
minutes_of() { echo $(( 10#${1%%:*} * 60 + 10#${1##*:} )); }
now_min() { echo $(( 10#$(date +%H) * 60 + 10#$(date +%M) )); }

# ---------------------------------------------------------------------------------------------- job counts
TOTAL=0; TW=0; TW_RUN=0; TW_PD=0; OTHER=0; OWN=0
TRACKED=0
declare -A CUR=() PREV=() MINE=()   # tripwire jobs now / at the last check (state|name|node|reason|gres); ids we submitted
count_jobs() {  # TOTAL = jobs that count toward AP_TARGET; TW / TW_RUN / TW_PD = tripwire jobs; OTHER = your other GPU jobs
  local out
  if ! out=$(squeue -u "${USER:-$(id -un)}" -h -t PENDING,RUNNING,CONFIGURING,COMPLETING -o '%i|%j|%T|%P|%b|%N|%r' 2>/dev/null); then
    return 1
  fi
  TW=0; TW_RUN=0; TW_PD=0; OTHER=0; OWN=0
  CUR=()
  local id name st part gres node reason
  while IFS='|' read -r id name st part gres node reason; do
    [ -n "${id}" ] || continue
    [ "${part}" = "${PARTITION}" ] || continue
    [[ "${gres}" == *gpu* ]] || continue
    if [[ "${name}" == tw-* ]]; then
      TW=$((TW + 1))
      if [ -z "${AP_JOBS_RE}" ] || [[ "${name}" =~ ${AP_JOBS_RE} ]]; then   # this plan's jobs: tracked and reported
        OWN=$((OWN + 1))
        CUR["${id}"]="${st}|${name}|${node}|${reason}|${gres}"
      fi
      if [ "${st}" = RUNNING ]; then TW_RUN=$((TW_RUN + 1)); else TW_PD=$((TW_PD + 1)); fi
    else
      OTHER=$((OTHER + 1))
    fi
  done <<< "${out}"
  if [ "${AP_COUNT}" = tw ]; then TOTAL=${TW}; else TOTAL=$((TW + OTHER)); fi
}
max_jobs_now() {  # the tripwire share of the target: what plan.sh (and the jobs' own queue feeding) may fill up to
  local m=$((TW + AP_TARGET - TOTAL))
  [ "${m}" -lt 0 ] && m=0
  echo "${m}"
}

# ---------------------------------------------------------------------------------------------- submitting
SUBMITTED=0; ATTEMPTS=0
record_feed() {  # events for every job sbatch accepted or refused in a plan.sh output
  local line id nm
  while read -r line; do
    if [[ "${line}" =~ ^submitted\ ([0-9]+)\ (tw-[^ ]+) ]]; then
      id="${BASH_REMATCH[1]}"; nm="${BASH_REMATCH[2]}"
      MINE["${id}"]=1
      event SUBMIT "${id}" "${nm}" "max_jobs $(max_jobs_now)"
    elif [[ "${line}" =~ ^not\ submitted\ (tw-[^:]+):\ (.*) ]]; then
      event SUBMIT_FAIL - "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ "${line}" =~ [Ee]rror ]]; then
      event SUBMIT_FAIL - - "${line}"
    elif [[ "${line}" =~ ^step\ ([^ ]+)\ is\ finished ]]; then
      event STEP_DONE - - "${BASH_REMATCH[1]}"
    fi
  done <<< "$1"
}
track_jobs() {  # START / NEW / END / ERROR events by comparing this check's tripwire jobs with the last one
  local id st nm node reason gres p_st info fstate ec el end day lf hhmm
  for id in "${!CUR[@]}"; do
    IFS='|' read -r st nm node reason gres <<< "${CUR[${id}]}"
    if [ "${TRACKED}" = 0 ]; then   # first check: what is already in Slurm
      event SEEN "${id}" "${nm}" "${st} ${gres} ${node}"
      continue
    fi
    if [ -z "${PREV[${id}]+x}" ]; then
      [ -n "${MINE[${id}]+x}" ] || event NEW "${id}" "${nm}" "${st} ${gres} (submitted by a job's own queue feeding or by hand)"
    fi
    p_st="${PREV[${id}]-}"; p_st="${p_st%%|*}"
    if [ "${st}" = RUNNING ] && [ "${p_st}" != RUNNING ]; then
      event START "${id}" "${nm}" "node ${node} ${gres}"
    fi
  done
  local -A keep=()
  for id in "${!PREV[@]}"; do
    [ -n "${CUR[${id}]+x}" ] && continue
    IFS='|' read -r st nm node reason gres <<< "${PREV[${id}]}"
    info=$(sacct -j "${id}" -X -n -P -o State,ExitCode,Elapsed,End 2>/dev/null | head -1)
    IFS='|' read -r fstate ec el end <<< "${info}"
    case "${fstate}" in   # not final in the accounting yet: look again at the next check
      RUNNING|PENDING|COMPLETING|CONFIGURING|REQUEUED|"") keep["${id}"]="${PREV[${id}]}"; continue ;;
    esac
    event END "${id}" "${nm}" "${fstate:-unknown} exit ${ec:-?} after ${el:-?} (ended ${end:-?})"
    case "${fstate}" in
      COMPLETED) ;;
      *)
        hhmm=$(sed -E 's/.*T([0-9]{2}):([0-9]{2}).*/\1\2/' <<< "${end}")
        if [[ "${fstate}" == CANCELLED* ]] && [ "${hhmm}" -ge 0415 ] 2>/dev/null && [ "${hhmm}" -le 0425 ] 2>/dev/null; then
          continue   # cancelled: the run continues from its checkpoint
        fi
        day=$(date +%F)
        lf="logs/${nm}-${id}.out"
        if [ -f "${lf}" ]; then mkdir -p "${AP_DIR}/errors_${day}"; cp "${lf}" "${AP_DIR}/errors_${day}/"; fi
        event ERROR "${id}" "${nm}" "${fstate} exit ${ec}; log ${lf} (copy in ${AP_DIR}/errors_${day}/)"
        log "job ${id} ${nm} ended ${fstate} (exit ${ec}); log copied to ${AP_DIR}/errors_${day}/"
        ;;
    esac
  done
  PREV=()
  for id in "${!CUR[@]}"; do PREV["${id}"]="${CUR[${id}]}"; done
  for id in "${!keep[@]}"; do PREV["${id}"]="${keep[${id}]}"; done
  TRACKED=1
}
feed_once() {  # plan.sh feed with the tripwire share; sets SUBMITTED (jobs that sbatch accepted) and ATTEMPTS
  local out
  out=$(MAX_JOBS="$(max_jobs_now)" bash slurm/plan.sh feed 2>&1)
  { echo "===== $(date '+%F %T') plan.sh feed (MAX_JOBS $(max_jobs_now))"; echo "${out}"; } >> "${AP_DIR}/feed.log"
  record_feed "${out}"
  SUBMITTED=$(grep -cE '^submitted [0-9]+ tw-' <<< "${out}" || true)
  ATTEMPTS=$(grep -oE '^submitted [0-9]+ \(at most' <<< "${out}" | grep -oE '[0-9]+' | head -1)
  ATTEMPTS=${ATTEMPTS:-0}
  grep -E '^submitted [0-9]+ tw-' <<< "${out}" | sed 's/^/    /' | tee -a "${AP_DIR}/autopilot.log" >/dev/null
  if [ "${ATTEMPTS}" -gt "${SUBMITTED}" ]; then   # sbatch refused some jobs: keep its messages
    grep -viE '^(submitted|step |queue is empty|no submissions)' <<< "${out}" | tail -n 20 | sed 's/^/    sbatch: /' \
      | tee -a "${AP_DIR}/autopilot.log" >/dev/null
  fi
  tail -n 1 <<< "${out}" | sed 's/^/    plan: /' | tee -a "${AP_DIR}/autopilot.log" >/dev/null
}
queue_empty() { [ ! -s "${PLAN_QUEUE}" ]; }

# ---------------------------------------------------------------------------------------------- first start
first_start() {
  [ -n "$(state_get queued)" ] && return 0
  log "first start: target ${AP_TARGET} GPU jobs (${AP_COUNT}), steps ${AP_STEPS}"
  local steps="${AP_STEPS}" g gated cmd tag
  local t2
  t2=$(grep -oE -- '--level [a-z0-9+]+' <<< "${AP_SELFTEST2}" | awk '{print $2}')   # the second selftest's name in the log
  for tag in unit "${t2:-v7}"; do   # each selftest gates its own steps
    if [ "${tag}" = unit ]; then gated="${AP_GATED}"; cmd="${AP_SELFTEST}"; else gated="${AP_GATED2}"; cmd="${AP_SELFTEST2}"; fi
    [ -n "${gated// /}" ] || continue
    log "running the CPU selftest (${tag}) once (new training code of: ${gated})"
    source slurm/common.sh
    set +u; activate_env >/dev/null 2>&1; set -u
    if ${cmd} > "${AP_DIR}/selftest_${tag}.log" 2>&1; then
      log "selftest (${tag}) PASSED"
      state_set "selftest_${tag}" passed
    else
      log "selftest (${tag}) FAILED: ${gated} are not queued (see ${AP_DIR}/selftest_${tag}.log, last lines below)"
      tail -n 15 "${AP_DIR}/selftest_${tag}.log" | sed 's/^/    /' | tee -a "${AP_DIR}/autopilot.log" >/dev/null
      state_set "selftest_${tag}" failed
      for g in ${gated//,/ }; do steps=$(sed -E "s/(^|,)${g//./\\.}(,|$)/\1\2/; s/,,/,/g; s/^,//; s/,$//" <<< "${steps}"); done
    fi
  done
  count_jobs || true
  if [ "${AP_RESET_QUEUE}" = 1 ]; then
    log "plan ${AP_PLAN}: emptying the old queue (running jobs continue): $(cut -d'|' -f1 "${PLAN_QUEUE}" 2>/dev/null | tr '\n' ' ')"
    bash slurm/plan.sh stop >> "${AP_DIR}/feed.log" 2>&1 || true
  fi
  log "queuing: ${steps}"
  local out st
  for st in ${steps//,/ }; do event QUEUE - - "${st}"; done
  out=$(MAX_JOBS="$(max_jobs_now)" bash slurm/plan.sh "${steps}" 2>&1)
  { echo "===== $(date '+%F %T') plan.sh ${steps} (MAX_JOBS $(max_jobs_now))"; echo "${out}"; } >> "${AP_DIR}/feed.log"
  record_feed "${out}"
  grep -E '^submitted' <<< "${out}" | sed 's/^/    /' | tee -a "${AP_DIR}/autopilot.log" >/dev/null
  state_set queued "$(date +%s)"
  [ -n "$(state_get until)" ] || state_set until $(( $(date +%s) + AP_DAYS * 86400 ))
}

# ---------------------------------------------------------------------------------------------- daily report
report() {
  local day f status errdir since
  day=$(date +%F)
  f="${AP_DIR}/report_${day}.txt"
  errdir="${AP_DIR}/errors_${day}"
  since=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)
  local sel
  case "${AP_PLAN}" in sft) sel=sft1,sft2 ;; fam) sel=fam1,fam2 ;; nov) sel=nov1,nov2,nov3,nov4 ;; p3) sel=p3 ;;
    *) sel=all,w5x.1 ;; esac
  status=$(bash slurm/plan.sh "${sel}" status 2>/dev/null)
  count_jobs || true
  {
    echo "Tripwire runs, $(date '+%F %T')"
    echo "Now in Slurm: ${TW_RUN} running and ${TW_PD} pending tripwire jobs; ${OTHER} other GPU jobs of yours in ${PARTITION}"
    echo "(target ${AP_TARGET} running + pending, counted over: ${AP_COUNT})"
    echo
    echo "== Per step: done / running or pending / to do / waiting for another run / FAILED"
    awk '/^== /{if (s!="") printf "%-6s %4d %4d %4d %4d %4d   %s\n", s, d, q, t, w, f, desc; s=$2; sub(":","",s);
                 desc=$0; sub(/^== [^:]*: /,"",desc); d=q=t=w=f=0; next}
         /^done +[a-z]/{d++} /^queued +[a-z]/{q++} /^to do +[a-z]/{t++} /^waiting +[a-z]/{w++} /^FAILED +[a-z]/{f++}
         END{if (s!="") printf "%-6s %4d %4d %4d %4d %4d   %s\n", s, d, q, t, w, f, desc}' <<< "${status}"
    echo
    echo "== Stopped in the middle (started, not finished, not in Slurm): continue from their checkpoint when resubmitted"
    grep -E '^to do ' <<< "${status}" | awk '{print $3}' | while read -r d; do
      [ -f "${d}/config.json" ] && [ ! -f "${d}/DONE" ] && echo "  ${d}"
    done
    echo
    echo "== FAILED twice (not resubmitted): fix the cause, delete the run's FAILED file, then bash slurm/plan.sh feed"
    grep -E '^FAILED' <<< "${status}" | sed 's/^/  /'
    echo
    echo "== Tripwire jobs of the last 24 hours"
    sacct -u "${USER:-$(id -un)}" -S "${since}" -X -n -P -o JobID,JobName%60,State,ExitCode,Elapsed,End 2>/dev/null \
      | awk -F'|' '$2 ~ /^tw-/ {n[$3]++} END{for (k in n) printf "  %-14s %d\n", k, n[k]}'
    echo
    echo "== Ended with an error in the last 24 hours (logs copied to ${errdir}/)"
    sacct -u "${USER:-$(id -un)}" -S "${since}" -X -n -P -o JobID,JobName%60,State,ExitCode,Elapsed,End 2>/dev/null \
      | awk -F'|' '$2 ~ /^tw-/ && $3 !~ /^(COMPLETED|RUNNING|PENDING|REQUEUED)/' \
      | while IFS='|' read -r id name st ec el end; do
          hhmm=$(sed -E 's/.*T([0-9]{2}):([0-9]{2}).*/\1\2/' <<< "${end}")
          if [[ "${st}" == CANCELLED* ]] && [ "${hhmm}" -ge 0415 ] 2>/dev/null && [ "${hhmm}" -le 0425 ] 2>/dev/null; then
            continue   # cancelled: the run continues from its checkpoint
          fi
          echo "  ${id} ${name} ${st} exit ${ec} after ${el} (ended ${end})"
          for lf in logs/"${name}"-"${id}".out; do
            [ -f "${lf}" ] || continue
            mkdir -p "${errdir}"
            cp "${lf}" "${errdir}/"
            tail -n 8 "${lf}" | sed 's/^/      | /'
          done
        done
    echo
    echo "== Queue (steps not finished yet)"
    bash slurm/plan.sh queue 2>/dev/null | sed 's/^/  /'
  } > "${f}" 2>&1
  log "daily report written: ${f}"
  event REPORT - - "${f}"
}

# ---------------------------------------------------------------------------------------------- self-resubmission
RESUBMIT=0
trap 'RESUBMIT=1' USR1
resubmit_self() {
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    log "time limit near: submitting a new autopilot job"
    sbatch --job-name="${SLURM_JOB_NAME:-cpujob}" slurm/autopilot.sbatch 2>&1 | sed 's/^/    /' | tee -a "${AP_DIR}/autopilot.log" >/dev/null
  fi
}

# ---------------------------------------------------------------------------------------------- main loop
main() {
  local other i
  for i in 1 2 3 4 5 6; do   # a resubmitted autopilot may start while the previous one is still exiting
    other=$(squeue -u "${USER:-$(id -un)}" -h -n "${SLURM_JOB_NAME:-cpujob}" -t RUNNING -o '%i' 2>/dev/null | grep -vx "${SLURM_JOB_ID:-none}" || true)
    [ -z "${other}" ] && break
    sleep 30
  done
  if [ -n "${other}" ]; then
    log "another autopilot job is running (${other//$'\n'/ }); this one exits"
    exit 0
  fi
  log "autopilot started (job ${SLURM_JOB_ID:-terminal}, host $(hostname)); report at ${AP_REPORT_AT}"
  first_start
  local until next_try=0 last_tw=-1 last_beat=0 last_black=0 rep_min now
  until=$(state_get until)
  rep_min=$(minutes_of "${AP_REPORT_AT}")
  [ "$(state_get report_day)" = "$(date +%F)" ] || { report; state_set report_day "$(date +%F)"; }
  while true; do
    now=$(date +%s)
    if [ -f "${AP_DIR}/STOP" ]; then log "STOP file found: exiting"; break; fi
    if [ -n "${until}" ] && [ "${now}" -ge "${until}" ]; then log "${AP_DAYS} days are over: final report and exit"; report; break; fi
    if [ "${RESUBMIT}" = 1 ]; then resubmit_self; exit 0; fi
    if [ "$(now_min)" -ge "${rep_min}" ] && [ "$(state_get report_day)" != "$(date +%F)" ]; then
      report; state_set report_day "$(date +%F)"
    fi
    if ! count_jobs; then sleep "${AP_CHECK_SEC}"; continue; fi
    track_jobs
    if [ -f "${AP_DIR}/PAUSE" ]; then   # paused: neither the autopilot nor the jobs' own queue feeding submits
      echo 0 > "${MAXF}"
      [ $((now - last_beat)) -ge 1800 ] && { log "paused (${AP_DIR}/PAUSE)"; event PAUSE - - "${AP_DIR}/PAUSE"; last_beat=${now}; }
      sleep "${AP_CHECK_SEC}"; continue
    fi
    echo "$(max_jobs_now)" > "${MAXF}"   # the jobs' own queue feeding uses the same share
    if [ $((now - last_beat)) -ge 1800 ]; then
      log "tripwire: ${TW_RUN} running, ${TW_PD} pending; other GPU jobs: ${OTHER}; counted ${TOTAL} of ${AP_TARGET}"
      last_beat=${now}
    fi
    if queue_empty && [ "${OWN}" -eq 0 ]; then log "every queued step is finished: final report and exit"; report; break; fi
    # a job of yours ended since the last check: try again now instead of waiting
    if [ "${last_tw}" -ge 0 ] && [ "${TW}" -lt "${last_tw}" ]; then next_try=0; fi
    if [ "${TOTAL}" -lt "${AP_TARGET}" ] && [ "${now}" -ge "${next_try}" ] && ! queue_empty; then
      log "counted ${TOTAL} of ${AP_TARGET} GPU jobs: submitting up to $((AP_TARGET - TOTAL))"
      feed_once
      if [ "${SUBMITTED}" -gt 0 ] && [ "${SUBMITTED}" -ge "${ATTEMPTS}" ]; then
        next_try=$((now + 60))
      elif [ "${ATTEMPTS}" -gt "${SUBMITTED}" ]; then
        log "sbatch accepted ${SUBMITTED} of ${ATTEMPTS}: trying again in ${AP_RETRY_MIN} min"
        next_try=$((now + AP_RETRY_MIN * 60))
      else
        next_try=$((now + AP_RETRY_MIN * 60))   # nothing ready: the next run waits for another one to finish
      fi
      count_jobs || true
    fi
    last_tw=${TW}
    sleep "${AP_CHECK_SEC}"
  done
}

if [ "${AP_SOURCE_ONLY:-0}" != 1 ]; then main; fi
