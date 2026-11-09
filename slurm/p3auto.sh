#!/bin/bash
# Autopilot for the p3 pack jobs (Nov 3): keeps the jobs of slurm/p3_general.sh (the general-claim runs) and the family
# item of slurm/p3_nova.sh (FAMILIES=1) going until every run is done, with nobody at the keyboard. Started by
# slurm/p3auto.sbatch (a 1-CPU job on nova); stop it with: touch logs/p3auto/STOP
#
# Every P3AUTO_CHECK_MIN (10) minutes it runs both submit scripts unchanged. They submit only a job that has runs left
# and is neither running nor queued, so a job that was cancelled or ended by a crash is submitted again,
# and its runs continue from their last checkpoint. Job time limits are the scripts' own; nothing is shortened.
# A run that fails MAX_FAILS (3) times is left out (its FAILED file lists the jobs), so a broken run cannot loop.
# It exits when nothing is left to run or queued, after P3AUTO_DAYS (7) days, or when logs/p3auto/STOP exists.
# Partitions and other settings come from the environment it was started with (P_IF_DENSE, P_IF_SFT, P_IF_TEACHER,
# P_NQ_TEACHER, FAM_PARTITION, ...). Log: logs/p3auto/p3auto.log (one line per decision); full script output in
# logs/p3auto/submit.log.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
APD=logs/p3auto
mkdir -p "${APD}"
CHECK_MIN="${P3AUTO_CHECK_MIN:-10}"
DAYS="${P3AUTO_DAYS:-7}"
export MAX_FAILS="${MAX_FAILS:-3}"
log() { echo "[$(date '+%F %T')] $*" >> "${APD}/p3auto.log"; }
left() {   # runs still to do (failed MAX_FAILS times: not counted)
  local g f
  g=$(bash slurm/p3_general.sh status 2>/dev/null | grep -c '^to do')
  f=$(FAMILIES=1 bash slurm/p3_nova.sh status 2>/dev/null | sed -n '/two more families/,$p' | grep -c '^to do')
  echo $((g + f))
}
queued() { squeue --me -h -o '%j' 2>/dev/null | grep -cE '^(p3gen-|p3pack-fam$)'; }
start=$(date +%s)
log "start: check every ${CHECK_MIN} min, partitions dense=${P_IF_DENSE:-${PARTITION:-nova}}" \
    "sft=${P_IF_SFT:-${PARTITION:-nova}} if-teacher=${P_IF_TEACHER:-${PARTITION:-nova}}" \
    "nq-teacher=${P_NQ_TEACHER:-${PARTITION:-nova}} fam=${FAM_PARTITION:-${PARTITION:-nova}}, MAX_FAILS=${MAX_FAILS}"
while true; do
  if [ -f "${APD}/STOP" ]; then log "STOP file found: exit"; exit 0; fi
  if [ $(( $(date +%s) - start )) -gt $((DAYS * 86400)) ]; then log "ran ${DAYS} days: exit"; exit 0; fi
  n=$(left); q=$(queued)
  if [ "${n}" -eq 0 ] && [ "${q}" -eq 0 ]; then
    log "nothing left to run and nothing queued: exit (bash slurm/p3_general.sh status for the details)"
    exit 0
  fi
  out=$( { echo "== $(date '+%F %T')"; bash slurm/p3_general.sh; FAMILIES=1 ONLY_FAMILIES=1 bash slurm/p3_nova.sh; } 2>&1 )
  echo "${out}" >> "${APD}/submit.log"
  echo "${out}" | grep -E '^submitted|NOT SUBMITTED|^FAILED' | while read -r line; do log "${line}"; done
  sleep $((CHECK_MIN * 60))
done
