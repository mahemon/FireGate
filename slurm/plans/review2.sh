# shellcheck shell=bash
# Review analyses, round 2 (Oct 29), sourced by slurm/plan.sh. CPU only, on stored features and evaluation outputs:
# no steps, no training, nothing queued. Runs as its own CPU job on the nova partition, so it never touches the
# autopilots or the GPU queue:
#
#   sbatch slurm/review2.sbatch           # job cpureview2: bash slurm/plan.sh review2-collect, about 1-2 h
#   bash slurm/plan.sh review2            # the same without the archive (login node: too slow)
#
# Per experiment: scripts/review2_analyses.py (two-sided sensitivity, covariance term, stream budget) into
# paper/v2_results/<exp>/switch/review2/; then the pooled tables (paper/v2_results/overview/review2/) and the overview
# macros again (adds the beta < 1 prediction error). Needs the switch reports of the last collect.
addon_review2() {
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/v2_results pair cfg name exp
  ok() { "$@" || echo "  (did not finish: $*)"; }
  for pair in "${O15}:Nq" "${IFC}:Ifc" "${CON}:Concise" "${S15}:Stream" "${S5}:Streamfive" "${O7}:Seven" \
              "${S7}:Sevenstream" "${OO7}:Olmo" "${IF7}:Ifcseven" "${CCX}:Concisectx"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    if [ ! -f "runs/${exp}/analysis/gate_features.npz" ] || [ ! -f "${out}/${exp}/switch/switch_summary.json" ]; then
      echo "== ${exp}: no features or switch report, left out"; continue
    fi
    echo "== ${exp} ($(date '+%H:%M'))"
    ok python scripts/review2_analyses.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch"
  done
  ok python scripts/review2_analyses.py --aggregate "${out}"
  ok python scripts/make_switch_overview.py --results "${out}"
  echo "tables in ${out}/overview/review2/"
}

addon_review2_collect() {  # bash slurm/plan.sh review2-collect: the above, packed into review2_results_<date>.tar.gz
  local out=paper/v2_results stamp
  stamp=$(date +%Y%m%d_%H%M)
  addon_review2 2>&1 | tee "${out}/review2_report.txt" || true
  # shellcheck disable=SC2046
  tar czf "review2_results_${stamp}.tar.gz" "${out}/review2_report.txt" "${out}/overview/overview_macros.tex" \
    "${out}/overview/review2" $(ls -d "${out}"/*/switch/review2 2>/dev/null) $(ls -d "${out}/latency" 2>/dev/null) || true
  echo "send me this file: $(pwd)/review2_results_${stamp}.tar.gz ($(du -h "review2_results_${stamp}.tar.gz" | cut -f1))"
}
