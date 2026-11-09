#!/bin/bash
# What finished, failed or was cancelled, and why: one report to send.
#
#   bash slurm/diagnose.sh [since]          # since: default 2026-10-24T00:00:00
#
# Writes logs/diagnose_<date>.txt with
#   1. every tw-* job since then: count per Slurm state;
#   2. jobs that did not complete (the daily cancel at about 04:19 is listed apart);
#   3. training jobs that "completed" in under 5 minutes (a run that refused to start);
#   4. every run marked FAILED: its FAILED file, pinned GPU type, the GPU model it started on, and for each failed
#      job the node, the error lines and the last 40 lines of its log;
#   5. the GPU nodes of the partition (GPU types and node features), to see which nodes a GPU type covers.
# Read-only: changes nothing in runs/ or in Slurm.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
if [ -f slurm/cluster.env ]; then source slurm/cluster.env; else source slurm/cluster.env.example; fi
SINCE="${1:-2026-10-24T00:00:00}"
mkdir -p logs
OUT="logs/diagnose_$(date +%Y%m%d_%H%M).txt"
JOBS="logs/.diagnose_jobs.tsv"
sacct -u "${USER:-$(id -un)}" -S "${SINCE}" -X -n -P -o JobID,JobName%80,State,ExitCode,Elapsed,End,NodeList 2>/dev/null \
  | awk -F'|' '$2 ~ /^tw-/' > "${JOBS}"

{
  echo "Tripwire jobs since ${SINCE} (report written $(date '+%F %T'))"
  echo
  echo "== 1. jobs per state"
  cut -d'|' -f3 "${JOBS}" | awk '{print $1}' | sort | uniq -c
  echo
  echo "== 2. jobs that did not complete (job, name, state, exit, elapsed, end, node)"
  awk -F'|' '$3 !~ /^(COMPLETED|RUNNING|PENDING)/' "${JOBS}" | while IFS='|' read -r id name st ec el end node; do
    hhmm=$(sed -E 's/.*T([0-9]{2}):([0-9]{2}).*/\1\2/' <<< "${end}")
    tag=""
    if [[ "${st}" == CANCELLED* ]] && [ "${hhmm}" -ge 0415 ] 2>/dev/null && [ "${hhmm}" -le 0425 ] 2>/dev/null; then
      tag="  (daily cancel)"
    fi
    printf '  %-10s %-55s %-12s %-6s %-10s %s %s%s\n' "${id}" "${name}" "${st}" "${ec}" "${el}" "${end}" "${node}" "${tag}"
  done
  echo
  echo "== 3. training jobs that completed in under 5 minutes"
  awk -F'|' '$3 ~ /^COMPLETED/ && $2 !~ /^tw-(gf|jg|x|sc|ab)-/ {split($5, t, ":"); if (length(t) == 3 && $5 !~ /-/ && t[1] + 0 == 0 && t[2] + 0 < 5) print "  " $1 "  " $2 "  " $5 "  " $7}' "${JOBS}"
  echo
  echo "== 4. runs marked FAILED"
  find runs -name FAILED -type f 2>/dev/null | sort | while read -r f; do
    d="$(dirname "${f}")"
    echo "---- ${d}"
    sed 's/^/  FAILED: /' "${f}"
    [ -s "${d}/GPU_TYPE" ] && echo "  pinned GPU type: $(cat "${d}/GPU_TYPE")"
    [ -f "${d}/config.json" ] && echo "  started on: $(grep -o '"device": "[^"]*"' "${d}/config.json" | head -1)"
    echo "  stages present: $(ls -d "${d}"/stage* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
    for id in $(grep -o 'job [0-9]*' "${f}" | awk '{print $2}' | sort -u); do
      node=$(awk -F'|' -v id="${id}" '$1 == id {print $7 " (" $3 ", " $5 ")"}' "${JOBS}")
      lf=$(ls logs/*-"${id}".out 2>/dev/null | head -1)
      echo "  job ${id} on ${node:-?}; log ${lf:-not found}"
      if [ -n "${lf}" ]; then
        grep -n -m 8 -E "Error|error:|Traceback|was started on|refuse|out of memory|OOM|Killed|CANCELLED|DUE TO" "${lf}" \
          | sed 's/^/    ! /'
        tail -n 40 "${lf}" | sed 's/^/    | /'
      fi
    done
  done
  echo
  echo "== 5. GPU nodes of partition ${PARTITION} (node | GPUs | features | state)"
  sinfo -p "${PARTITION}" -N -h -o "%N|%G|%f|%T" 2>/dev/null | sort -u | sed 's/^/  /'
} > "${OUT}" 2>&1

rm -f "${JOBS}"
echo "wrote ${OUT} ($(wc -l < "${OUT}") lines): send me this file"
