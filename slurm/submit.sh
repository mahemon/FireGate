#!/bin/bash
# Submit jobs. Every job runs ONE method (or one evaluation) and can use one or several GPUs of one node.
#
#   bash slurm/submit.sh selftest                         # unit + tiny end-to-end + real-model checks, 1 GPU
#   NGPU=2 bash slurm/submit.sh selftest                  # ... plus the multi-GPU checks (2 GPUs)
#   CONFIG=configs/v2/stream_q15.yaml SEED=0 METHODS="tripwire seq kl" bash slurm/submit.sh stream
#                                                         # one job per method
#   CONFIG=configs/v2/overlap_q15.yaml SEED=0 METHODS=kl TAG=b0.2 EXTRA="grpo.kl_beta=0.2" bash slurm/submit.sh stream
#                                                         # a sweep point: run folder kl_b0.2
#   CONFIG=configs/v2/stream_q7.yaml NGPU=4 METHODS=tripwire bash slurm/submit.sh stream
#                                                         # 4 GPUs for one run (deepspeed launcher)
#   CONFIG=configs/v2/stream_q15.yaml SEED=0 METHODS="tripwire seq" WHAT=all bash slurm/submit.sh extra
#                                                         # robustness evaluations of finished runs
#   CONFIG=configs/v2/overlap_q15.yaml SEED=0 BASE=seq ALPHAS="0.25 0.5 0.75" bash slurm/submit.sh scale
#                                                         # adapter-scaling baseline: one evaluation job per alpha
#   CONFIG=configs/v2/overlap_q15.yaml bash slurm/submit.sh ablate
#                                                         # score ablation of the overlap task (no training)
#   LEVEL=real MODEL_DIR=models/Qwen2.5-7B-Instruct CONFIG=configs/v2/stream_q7.yaml NAME=q7 bash slurm/submit.sh selftest
#                                                         # real-model checks of another model (speed, memory, hours)
# Settings (environment variables):
#   NGPU      GPUs per job (default 1). With NGPU>1 the run is started by LAUNCHER (deepspeed, default, or torchrun)
#   GPU_TYPE  h200 / a100 / empty for any. Keep ONE type per experiment: greedy bf16 outputs differ across GPU
#             models, and a run refuses to continue on another model than it started on
#   SEED, TAG, EXTRA ("key=value ..." config overrides), TIME_LIMIT, DEPEND (e.g. afterok:123), BEGIN, DRY=1
# Every submitted job is appended to logs/submitted.tsv (job id, name, command line).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
if [ -f "${REPO}/slurm/cluster.env" ]; then source "${REPO}/slurm/cluster.env"; else source "${REPO}/slurm/cluster.env.example"; fi
mkdir -p "${REPO}/logs"
what="${1:-}"
NGPU="${NGPU:-1}"; export NGPU
LAUNCHER="${LAUNCHER:-deepspeed}"; export LAUNCHER
SEED="${SEED:-0}"; export SEED
TAG="${TAG:-}"; export TAG
EXTRA="${EXTRA:-}"; export EXTRA
CONFIG="${CONFIG:-configs/v2/stream_q15.yaml}"; export CONFIG
CFGNAME="$(basename "${CONFIG}" .yaml)"

to_minutes() {  # SLURM time limit -> minutes (MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS)
  local t="$1" d=0 a b c
  if [[ "${t}" == *-* ]]; then
    d="${t%%-*}"; t="${t#*-}"
    IFS=: read -r a b c <<< "${t}"
    echo $(( 10#${d} * 1440 + 10#${a} * 60 + 10#${b:-0} ))
    return
  fi
  IFS=: read -r a b c <<< "${t}"
  if [ -n "${c:-}" ]; then echo $(( 10#${a} * 60 + 10#${b} )); else echo $(( 10#${a} )); fi
}
# Nov 4: Nova no longer cancels jobs daily, so job time limits are never cut and submissions are never held back.
# The subcommands window / inblackout / windowok stay for older callers: no limit, never a blackout, always ok.
in_blackout() { return 1; }
window() { echo 99999; }

submit_one() {   # $1 job name, $2 sbatch script, $3 time limit; METHOD etc. exported by the caller
  local name="$1" script="$2" tl="$3"
  local gres="gpu:${NGPU}"
  [ -n "${GPU_TYPE}" ] && gres="gpu:${GPU_TYPE}:${NGPU}"
  local args=(--parsable --partition="${PARTITION}" --gres="${gres}" --time="${tl}" --mem="${MEM}"
              --cpus-per-task="${CPUS}" --chdir="${REPO}" --export=ALL --job-name="${name}")
  [ -n "${ACCOUNT}" ] && [ "${ACCOUNT}" != none ] && args+=(--account="${ACCOUNT}")   # ACCOUNT=none: the default account
  [ -n "${BEGIN:-}" ] && args+=(--begin="${BEGIN}")
  [ -n "${DEPEND:-}" ] && args+=(--dependency="${DEPEND}")
  [ -n "${CONSTRAINT:-}" ] && args+=(--constraint="${CONSTRAINT}")   # node feature: the GPU model a run started on
  if [ "${DRY:-0}" = "1" ]; then
    echo "DRY: sbatch ${args[*]} ${REPO}/${script}"
    return
  fi
  local id
  id=$(sbatch "${args[@]}" "${REPO}/${script}")
  echo "submitted ${id} ${name}"
  printf '%s\t%s\t%s\t%s\n' "${id}" "${name}" "$(date '+%F %T')" "CONFIG=${CONFIG} METHOD=${METHOD:-} SEED=${SEED} TAG=${TAG} NGPU=${NGPU} GPU_TYPE=${GPU_TYPE} EXTRA=${EXTRA}${ALPHA:+ BASE=${BASE:-} ALPHA=${ALPHA}}" >> "${REPO}/logs/submitted.tsv"
}

case "${what}" in
  window) window ;;
  inblackout) in_blackout ;;   # never a blackout (Nov 4)
  windowok) exit 0 ;;          # always ok (Nov 4)
  selftest)
    submit_one "tw-selftest${NAME:+-${NAME}}-g${NGPU}" slurm/selftest.sbatch "${SELFTEST_TIME:-01:30:00}" ;;
  stream)
    METHODS="${METHODS:-${METHOD:-tripwire}}"
    for m in ${METHODS}; do
      export METHOD="${m}"
      submit_one "tw-${CFGNAME}-s${SEED}-${m}${TAG:+_${TAG}}" slurm/stream.sbatch "${TIME_LIMIT}"
    done ;;
  extra)
    METHODS="${METHODS:-${METHOD:-tripwire}}"
    export WHAT="${WHAT:-all}"
    for m in ${METHODS}; do
      export METHOD="${m}"
      submit_one "tw-x-${CFGNAME}-s${SEED}-${m}${TAG:+_${TAG}}" slurm/extra.sbatch "${EXTRA_TIME:-03:00:00}"
    done ;;
  scale)
    export BASE="${BASE:-seq}"
    for a in ${ALPHAS:-${ALPHA:-0.5}}; do
      export ALPHA="${a}" METHOD="scale"
      submit_one "tw-sc-${CFGNAME}-s${SEED}-${BASE}-a${a}" slurm/scale.sbatch "${SCALE_TIME:-03:00:00}"
    done ;;
  ablate)   # score ablation of a config's overlap task (no training): CONFIG, LAYERS (default 4,8,14,20)
    export LAYERS="${LAYERS:-4,8,14,20}"
    submit_one "tw-ab-${CFGNAME}" slurm/ablate.sbatch "${ABLATE_TIME:-01:30:00}" ;;
  features)   # trunk features of every gate prompt of a config (no training): CONFIG; FEATURES_ARGS=--add
    export FEATURES_ARGS="${FEATURES_ARGS:-}"
    submit_one "tw-gf-${CFGNAME}" slurm/features.sbatch "${FEATURES_TIME:-01:30:00}" ;;
  judge)      # LLM judge of the changed responses of a config's runs: CONFIG, JUDGE_MODEL, JUDGE_ARGS;
              # JUDGE_TAG (second judge): labels in analysis/judge_<tag>/, job tw-jg-<tag>-<config>
    export JUDGE_MODEL="${JUDGE_MODEL:-models/Qwen2.5-7B-Instruct}" JUDGE_ARGS="${JUDGE_ARGS:-}" JUDGE_TAG="${JUDGE_TAG:-}"
    submit_one "tw-jg${JUDGE_TAG:+-${JUDGE_TAG}}-${CFGNAME}" slurm/judge.sbatch "${JUDGE_TIME:-06:00:00}" ;;
  router)   # legacy evaluation-only router (v1 experiments)
    submit_one "tw-router" slurm/router.sbatch "${ROUTER_TIME:-02:00:00}" ;;
  *) echo "usage: bash slurm/submit.sh selftest|stream|extra|scale|ablate|features|judge|router|window|windowok|inblackout" >&2; exit 1 ;;
esac
