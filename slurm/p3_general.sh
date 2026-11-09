#!/bin/bash
# Optional (Nov 3; claude/tripwire_weakness_strategy_nov3.md): more seeds for the claim that the certificate does not
# depend on LoRA or on RL. Turns the one-seed pilot of Table "Full-rank updates and SFT" (IF, Qwen2.5-1.5B) into three
# seeds, and adds SFT on a teacher's answers on NQ, the task with overlap (beta < 1). Only needed if the paper's title
# drops "RL"; written Nov 3, not to be submitted before the p3.12-p3.15 reports are in. Four GPU jobs, one GPU each,
# outside every plan queue, each skipping what is done (slurm/p3pack.sbatch):
#   p3gen-if-dense     IF seeds 1-2: full-rank Seq-GRPO, full-rank FireGate-prompt, full-rank Seq-GRPO on every block
#                      but the first (the v7.4 settings, lr 5e-6): 6 runs, 1-1.5 h each
#   p3gen-if-sft       IF seeds 1-2: LoRA SFT on the model's own accepted samples, ungated and gated (v7.1): 4 runs,
#                      under 1 h each
#   p3gen-if-teacher   IF seeds 1-2: full-rank SFT on Qwen2.5-7B answers, ungated, gated, every block but the first
#                      (configs/sft/sft_ifc_q15.yaml): 6 runs, about 1.5 h each
#   p3gen-nq-teacher   NQ seeds 0-2: full-rank SFT on Qwen2.5-7B answers the reward accepts (instead of the gold
#                      answers, which lowered NQ accuracy), ungated and gated: 6 runs, about 1.5 h each
#   p3gen-judge        (Nov 4) the LLM judge (Qwen2.5-7B) of the new runs, for the "Subst." column: one pass per
#                      config (IF, the IF teacher SFT, the NQ SFT), only new pairs (labels are cached), a few hours.
#                      Submitted only when no training run above is left to do and none of their jobs is queued.
# Run from the repository folder:
#
#   bash slurm/p3_general.sh status     what is done
#   bash slurm/p3_general.sh dry        the sbatch commands, nothing submitted
#   bash slurm/p3_general.sh            submit (a job already queued or running is not submitted again)
#   JOBS="p3gen-nq-teacher" bash slurm/p3_general.sh     only some of the jobs
#
# Partition: PARTITION (default nova) for every job, or per job P_IF_DENSE, P_IF_SFT, P_IF_TEACHER, P_NQ_TEACHER.
# Account: on instruction the ACCOUNT of slurm/cluster.env, elsewhere ACCOUNT (default none). GPU_TYPE (a100) and
# CONSTRAINT (epyc-7413: the A100 80GB PCIe nodes, the GPU model of the 1.5B runs). PACK_TIME (20:00:00).
# Afterwards (CPU, nova): sbatch slurm/p3gentab.sbatch, the tables of every SFT and full-rank run with all seeds
# (bash slurm/plan.sh sft-collect: paper/sft_results/, overview/pilot_table.tex for the paper's Table 21), packed into
# sft_results_<date>.tar.gz. Partition of the judge: P_JUDGE (default PARTITION); it runs on any A100
# (JUDGE_CONSTRAINT, default none: the judge reads text, so the GPU model does not matter).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/p3_general.sh [status|dry]" >&2; exit 1 ;; esac
PARTITION="${PARTITION:-nova}"
ACCOUNT="${ACCOUNT:-none}"
ENVF=slurm/cluster.env; [ -f "${ENVF}" ] || ENVF=slurm/cluster.env.example
INSTRUCTION_ACCOUNT="${INSTRUCTION_ACCOUNT:-$( (unset ACCOUNT; source "${ENVF}" >/dev/null 2>&1; echo "${ACCOUNT:-none}") )}"
acct_for() { if [ "$1" = instruction ]; then echo "${INSTRUCTION_ACCOUNT}"; else echo "${ACCOUNT}"; fi; }
GPU_TYPE="${GPU_TYPE:-a100}"
CONSTRAINT="${CONSTRAINT-epyc-7413}"
PACK_TIME="${PACK_TIME:-20:00:00}"
PACK_MEM="${PACK_MEM:-64G}"
TRAIN_JOBS="p3gen-if-dense p3gen-if-sft p3gen-if-teacher p3gen-nq-teacher"
JOBS="${JOBS:-${TRAIN_JOBS} p3gen-judge}"
JUDGE_CONSTRAINT="${JUDGE_CONSTRAINT-}"
REPO="$(pwd)"
export REPO
mkdir -p logs

IFC=configs/v2/ladder_ifc_q15.yaml; SFT_IF=configs/sft/sft_ifc_q15.yaml; SFT_NQ=configs/sft/sft_nq_q15.yaml
TEACH="grpo.sft_source=teacher grpo.sft_teacher=models/Qwen2.5-7B-Instruct grpo.sft_samples=2"
exp_of() { grep -E '^exp:' "$1" | awk '{print $2}'; }

# job name | partition | items (kind config seed method tag overrides)
items_of() {
  local s
  case "$1" in
    p3gen-if-dense) for s in 1 2; do
        echo "train ${IFC} ${s} seq dense5e6 lora.dense=true grpo.lr=5.0e-6"
        echo "train ${IFC} ${s} tripwire_prompt dense5e6 lora.dense=true grpo.lr=5.0e-6"
        echo "train ${IFC} ${s} seq full5e6 lora.dense=true grpo.lr=5.0e-6 gate_layer=1"
      done ;;
    p3gen-if-sft) for s in 1 2; do
        echo "train ${IFC} ${s} seq sft grpo.objective=sft"
        echo "train ${IFC} ${s} tripwire_prompt sft grpo.objective=sft"
      done ;;
    p3gen-if-teacher) for s in 1 2; do
        echo "train ${SFT_IF} ${s} seq - -"
        echo "train ${SFT_IF} ${s} tripwire_prompt - -"
        echo "train ${SFT_IF} ${s} seq full gate_layer=1"
      done ;;
    p3gen-nq-teacher) for s in 0 1 2; do
        echo "train ${SFT_NQ} ${s} seq teacher ${TEACH}"
        echo "train ${SFT_NQ} ${s} tripwire_prompt teacher ${TEACH}"
      done ;;
    p3gen-judge) for c in "${IFC}" "${SFT_IF}" "${SFT_NQ}"; do echo "judge ${c} - - - -"; done ;;
  esac
}
part_of() {
  case "$1" in
    p3gen-if-dense) echo "${P_IF_DENSE:-${PARTITION}}" ;;
    p3gen-if-sft) echo "${P_IF_SFT:-${PARTITION}}" ;;
    p3gen-if-teacher) echo "${P_IF_TEACHER:-${PARTITION}}" ;;
    p3gen-nq-teacher) echo "${P_NQ_TEACHER:-${PARTITION}}" ;;
    p3gen-judge) echo "${P_JUDGE:-${PARTITION}}" ;;
  esac
}
judge_done() {  # judge_done <exp>: analysis/judge/DONE newer than every finished run of the experiment
  local d="runs/$1/analysis/judge"
  [ -f "${d}/DONE" ] && [ -z "$(find "runs/$1" -mindepth 3 -maxdepth 3 -path "runs/$1/s*/*/DONE" -newer "${d}/DONE" \
    -print -quit 2>/dev/null)" ]
}
train_left=0   # training runs of every training job still to do (failed MAX_FAILS times: not counted); the judge waits
for tj in ${TRAIN_JOBS}; do
  while read -r kind cfg s m tag ovr; do
    [ "${tag}" = - ] && tag=""
    d="runs/$(exp_of "${cfg}")/s${s}/${m}${tag:+_${tag}}"
    [ -f "${d}/DONE" ] && continue
    [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge "${MAX_FAILS:-3}" ] && continue
    train_left=$((train_left + 1))
  done < <(items_of "${tj}")
done

queued() { squeue --me -h -o '%j' 2>/dev/null | grep -qxF "$1"; }
gres_avail() { sinfo -p "$1" -h -o '%G' 2>/dev/null | grep -q "gpu:$2"; }
for job in ${JOBS}; do
  part="$(part_of "${job}")"
  items="logs/${job}.items"
  { echo "# kind config seed method tag overrides (written by slurm/p3_general.sh)"; items_of "${job}"; } > "${items}"
  todo=0
  cons="${CONSTRAINT}"; [ "${job}" = p3gen-judge ] && cons="${JUDGE_CONSTRAINT}"
  echo "== ${job} (partition ${part}, ${GPU_TYPE}${cons:+, nodes ${cons}})"
  while read -r kind cfg s m tag ovr; do
    [[ "${kind}" == \#* ]] && continue
    [ "${tag}" = - ] && tag=""
    if [ "${kind}" = judge ]; then
      d="runs/$(exp_of "${cfg}")/analysis/judge"
      if judge_done "$(exp_of "${cfg}")"; then echo "done     ${d}"
      elif [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge "${MAX_FAILS:-3}" ]; then echo "FAILED   ${d} (see ${d}/FAILED)"
      else echo "to do    ${d}"; todo=$((todo + 1)); fi
      continue
    fi
    d="runs/$(exp_of "${cfg}")/s${s}/${m}${tag:+_${tag}}"
    if [ -f "${d}/DONE" ]; then echo "done     ${d}"
    elif [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge "${MAX_FAILS:-3}" ]; then echo "FAILED   ${d} (see ${d}/FAILED)"
    else echo "to do    ${d}"; todo=$((todo + 1)); fi
  done < "${items}"
  [ "${MODE}" = status ] && continue
  [ "${todo}" -gt 0 ] || continue
  if queued "${job}"; then echo "queued   ${job} (not submitted again)"; continue; fi
  if [ "${job}" = p3gen-judge ]; then
    busy=$(squeue --me -h -o '%j' 2>/dev/null | grep -cE '^p3gen-(if|nq)-' || true)
    if [ "${train_left}" -gt 0 ] || [ "${busy}" -gt 0 ]; then
      echo "waiting  ${job}: ${train_left} training runs left, ${busy} training jobs queued or running (submit again later)"
      continue
    fi
  fi
  if ! gres_avail "${part}" "${GPU_TYPE}"; then echo "NOT SUBMITTED ${job}: partition ${part} lists no gpu:${GPU_TYPE}"; continue; fi
  acct="$(acct_for "${part}")"; [ "${acct}" = none ] && acct=""
  args=(--parsable --partition="${part}" --gres="gpu:${GPU_TYPE}:1" --time="${PACK_TIME}" --mem="${PACK_MEM}"
        --cpus-per-task=8 --chdir="${REPO}" --export=ALL --job-name="${job}")
  [ -n "${acct}" ] && args+=(--account="${acct}")
  [ -n "${cons}" ] && args+=(--constraint="${cons}")
  if [ "${MODE}" = dry ]; then echo "DRY: ITEMS=${REPO}/${items} LANES=1 sbatch ${args[*]} slurm/p3pack.sbatch"; continue; fi
  id=$(ITEMS="${REPO}/${items}" LANES=1 PACK_GRES="gpu:${GPU_TYPE}:1" PACK_TIME="${PACK_TIME}" PACK_MEM="${PACK_MEM}" \
       PACK_ACCOUNT="${acct}" PACK_CONSTRAINT="${cons}" sbatch "${args[@]}" slurm/p3pack.sbatch)
  echo "submitted ${id} ${job} (${part}, ${todo} runs; log logs/${job}-${id}.out)"
done
