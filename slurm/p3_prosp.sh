#!/bin/bash
# Step p3.19 (Nov 4, PLAN_PAPER3.md): the trunk features of the prospective sets (data/pr_*.jsonl) for the nine
# experiments of p3.12, as one GPU job per GPU model the stored features were made on (gate_features.py --add refuses
# any other model). Feature passes only, no generation, no training: minutes per experiment.
# Run from the repository folder, after python scripts/prepare_prospective.py --data data and
# python scripts/prereg_record.py --step p3.19:
#
#   bash slurm/p3_prosp.sh status     what is done, and the GPU model each experiment needs
#   bash slurm/p3_prosp.sh dry        the sbatch commands, nothing submitted
#   bash slurm/p3_prosp.sh            submit (a job already queued or running is not submitted twice)
#
# Partitions: PARTITION (default nova) for every job, H200_PARTITION for the H200 group (default PARTITION). Account:
# on instruction the ACCOUNT of slurm/cluster.env (INSTRUCTION_ACCOUNT overrides it), elsewhere ACCOUNT (default none).
# PACK_TIME (03:00:00), PACK_MEM (64G). After a cancel or a crash, run the same command again: done
# experiments are skipped. When everything is done: sbatch slurm/p3more.sbatch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/p3_prosp.sh [status|dry]" >&2; exit 1 ;; esac
PARTITION="${PARTITION:-nova}"
H200_PARTITION="${H200_PARTITION:-${PARTITION}}"
ACCOUNT="${ACCOUNT:-none}"
ENVF=slurm/cluster.env; [ -f "${ENVF}" ] || ENVF=slurm/cluster.env.example
INSTRUCTION_ACCOUNT="${INSTRUCTION_ACCOUNT:-$( (unset ACCOUNT; source "${ENVF}" >/dev/null 2>&1; echo "${ACCOUNT:-none}") )}"
acct_for() { if [ "$1" = instruction ]; then echo "${INSTRUCTION_ACCOUNT}"; else echo "${ACCOUNT}"; fi; }
PACK_TIME="${PACK_TIME:-03:00:00}"
PACK_MEM="${PACK_MEM:-64G}"
REPO="$(pwd)"
export REPO NO_FEED=1
mkdir -p logs

CONFIGS=(configs/v2/ladder_concise_q15.yaml configs/v2/ladder_concisectx_q15.yaml configs/v3/squad2_q15.yaml
         configs/v2/overlap_q15.yaml configs/v2/ladder_ifc_q15.yaml configs/v2/ladder_ifc_q7.yaml
         configs/v2/stream_q15.yaml configs/v2/overlap_olmo7.yaml configs/fam/nq_llama3b.yaml)
exp_of() { grep -E '^exp:' "$1" | awk '{print $2}'; }
devof() { grep -o '"device": "[^"]*"' "$1" 2>/dev/null | head -1 | cut -d'"' -f4 || true; }
group_of() {  # GPU model -> group name, gres type, node constraint (the mapping of slurm/p3_nova.sh)
  case "$1" in
    *H200*) echo "h200 h200 -" ;;
    *SXM4*) echo "sxm4 a100 compute-gpu" ;;
    *) echo "pcie a100 epyc-7413" ;;
  esac
}

n=0; for f in data/pr_*.jsonl; do [ -s "${f}" ] && n=$((n + 1)); done
if [ "${n}" -lt 9 ]; then
  echo "data/pr_*.jsonl: ${n} of 9 sets present; run python scripts/prepare_prospective.py --data data on the login node" >&2
  [ "${MODE}" = status ] || exit 1
fi
[ -f paper/v3_results/p3_19/PREREG_p3_19.txt ] || echo "WARNING: no paper/v3_results/p3_19/PREREG_p3_19.txt (python scripts/prereg_record.py --step p3.19 first)"

declare -A TODO GT CN
for cfg in "${CONFIGS[@]}"; do
  exp="$(exp_of "${cfg}")"; js="runs/${exp}/analysis/gate_features.json"; dev="$(devof "${js}")"
  if [ -z "${dev}" ]; then echo "missing  ${js}: no stored gate features"; continue; fi
  read -r g gt cn <<< "$(group_of "${dev}")"
  if grep -q '"eval__pr_' "${js}" 2>/dev/null; then echo "done     ${js} (prospective sets)"; continue; fi
  echo "to do    ${js} (stored on ${dev}: group ${g})"
  TODO[${g}]="${TODO[${g}]:-}prosp ${cfg} 0 - - -"$'\n'
  GT[${g}]="${gt}"; CN[${g}]="${cn}"
done
[ "${MODE}" = status ] && exit 0
if [ "${#TODO[@]}" -eq 0 ]; then echo "nothing to submit"; echo "next: sbatch slurm/p3more.sbatch"; exit 0; fi

queued() { squeue --me -h -o '%j' 2>/dev/null | grep -qxF "$1"; }
for g in "${!TODO[@]}"; do
  name="p3prosp-${g}"; items="${REPO}/logs/p3prosp_${g}.items"
  printf '# kind config seed method tag overrides (written by slurm/p3_prosp.sh)\n%s' "${TODO[${g}]}" > "${items}"
  part="${PARTITION}"; [ "${g}" = h200 ] && part="${H200_PARTITION}"
  gt="${GT[${g}]}"; cn="${CN[${g}]}"; [ "${cn}" = - ] && cn=""
  if queued "${name}"; then echo "queued   ${name} (not submitted again)"; continue; fi
  acct="$(acct_for "${part}")"; [ "${acct}" = none ] && acct=""
  args=(--parsable --partition="${part}" --gres="gpu:${gt}:1" --time="${PACK_TIME}" --mem="${PACK_MEM}" --cpus-per-task=8
        --chdir="${REPO}" --export=ALL --job-name="${name}")
  [ -n "${acct}" ] && args+=(--account="${acct}")
  [ -n "${cn}" ] && args+=(--constraint="${cn}")
  if [ "${MODE}" = dry ]; then echo "DRY: ITEMS=${items} LANES=1 sbatch ${args[*]} slurm/p3pack.sbatch"; continue; fi
  id=$(ITEMS="${items}" LANES=1 PACK_GRES="gpu:${gt}:1" PACK_TIME="${PACK_TIME}" PACK_MEM="${PACK_MEM}" PACK_ACCOUNT="${acct}" \
       PACK_CONSTRAINT="${cn}" sbatch "${args[@]}" slurm/p3pack.sbatch)
  echo "submitted ${id} ${name} (${part}, gpu:${gt}${cn:+, ${cn}}; log logs/${name}-${id}.out)"
done
echo "then: bash slurm/p3_prosp.sh status; when every experiment is done: sbatch slurm/p3more.sbatch"
