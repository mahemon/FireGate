#!/bin/bash
# Steps p3.12 and p3.15 (PLAN_PAPER3.md) as four GPU jobs, one GPU each, outside every plan queue; each job can go to
# nova or instruction (or any partition):
#   p3pack-q15a          Qwen2.5-1.5B, 15 items: external-set features of 6 experiments, Seq-GRPO seed 0 of each on
#                        the external sets, + KL at beta 10 on NQ (seeds 0-2)
#   p3pack-q15b          Qwen2.5-1.5B, 12 items: Seq-GRPO's adapter scaled by 0.02/0.05/0.1 (NQ, seeds 0-2), + KL at
#                        beta 100 on NQ (seeds 0-2)
#                        (both: two lanes on the GPU, training / the rest)
#   p3pack-q7-features   one GPU, IF on Qwen2.5-7B: external-set features (on the GPU model of its stored features)
#   p3pack-q7-external   one GPU, IF on Qwen2.5-7B: Seq-GRPO seed 0 on the external sets
#   p3pack-fam           only with FAMILIES=1 (Nov 3 addendum): one GPU, NQ on OLMo-2-7B and on Llama-3.2-3B,
#                        external-set features and Seq-GRPO seed 0 on the external sets (FAM_PARTITION)
#   p3pack-sq            only with SQUAD=1 (Nov 3): one GPU, p3.15 on SQuAD 2.0 (gain +24): Seq-GRPO's adapter scaled
#                        by 0.02/0.05/0.1 and + KL at beta 10 and 100, seeds 0-2 (SQ_PARTITION; two lanes)
# Run from the repository folder, after python scripts/prepare_external.py --data data:
#
#   bash slurm/p3_nova.sh status     what is done, and which items need another GPU model
#   bash slurm/p3_nova.sh dry        the sbatch commands, nothing submitted
#   bash slurm/p3_nova.sh            submit (a job already queued or running is not submitted twice)
#
# Partitions: PARTITION (default nova) for every job, or per job Q15A_PARTITION, Q15B_PARTITION, Q7_PARTITION (both
# 7B jobs) and Q7F_PARTITION (the 7B features job alone), e.g.
#   Q15B_PARTITION=instruction Q7F_PARTITION=instruction bash slurm/p3_nova.sh
# Account: on instruction the ACCOUNT of slurm/cluster.env (INSTRUCTION_ACCOUNT overrides it); elsewhere ACCOUNT
# (default none: the partition's default account).
# Other settings (environment): GPU_TYPE (a100) and CONSTRAINT
# (epyc-7413: the "A100 80GB PCIe" nodes, the GPU model of the 1.5B runs), LANES (2), PACK_TIME (24:00:00),
# PACK_MEM (64G), Q7_PARTITION (the partition of the two 7B jobs, default PARTITION), Q7F_PARTITION (the 7B
# features job alone, default Q7_PARTITION: it needs the GPU model of IF 7B's stored features, H200 if they were made
# there). Every item is skipped when done,
# so after a cancel or a crash, run the same command again. Do not also queue
# steps p3.12 / p3.15 of slurm/plan.sh while these jobs run: both would work on the same items.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-submit}"
case "${MODE}" in submit|status|dry) ;; *) echo "usage: bash slurm/p3_nova.sh [status|dry]" >&2; exit 1 ;; esac
PARTITION="${PARTITION:-nova}"
ACCOUNT="${ACCOUNT:-none}"
ENVF=slurm/cluster.env; [ -f "${ENVF}" ] || ENVF=slurm/cluster.env.example
INSTRUCTION_ACCOUNT="${INSTRUCTION_ACCOUNT:-$( (unset ACCOUNT; source "${ENVF}" >/dev/null 2>&1; echo "${ACCOUNT:-none}") )}"
acct_for() { if [ "$1" = instruction ]; then echo "${INSTRUCTION_ACCOUNT}"; else echo "${ACCOUNT}"; fi; }
GPU_TYPE="${GPU_TYPE:-a100}"
CONSTRAINT="${CONSTRAINT-epyc-7413}"
TARGET="${TARGET:-NVIDIA A100 80GB PCIe}"     # the GPU model the 1.5B items need (their runs and stored features)
LANES="${LANES:-2}"
PACK_TIME="${PACK_TIME:-24:00:00}"
PACK_MEM="${PACK_MEM:-64G}"
Q15A_PARTITION="${Q15A_PARTITION:-${PARTITION}}"
Q15B_PARTITION="${Q15B_PARTITION:-${PARTITION}}"
Q7_PARTITION="${Q7_PARTITION:-${PARTITION}}"
Q7F_PARTITION="${Q7F_PARTITION:-${Q7_PARTITION}}"
FAMILIES="${FAMILIES:-0}"
FAM_PARTITION="${FAM_PARTITION:-${PARTITION}}"
SQUAD="${SQUAD:-0}"
SQ_PARTITION="${SQ_PARTITION:-${PARTITION}}"
REPO="$(pwd)"
export REPO
mkdir -p logs

O15=configs/v2/overlap_q15.yaml; S15=configs/v2/stream_q15.yaml; IFC=configs/v2/ladder_ifc_q15.yaml
CON=configs/v2/ladder_concise_q15.yaml; CCX=configs/v2/ladder_concisectx_q15.yaml; SQ=configs/v3/squad2_q15.yaml
IF7=configs/v2/ladder_ifc_q7.yaml
exp_of() { grep -E '^exp:' "$1" | awk '{print $2}'; }
devof() { grep -o '"device": "[^"]*"' "$1" 2>/dev/null | head -1 | cut -d'"' -f4 || true; }

for f in ext_harmbench.jsonl ext_harmbench_ctx.jsonl ext_strongreject.jsonl; do
  if [ ! -s "data/${f}" ]; then
    echo "data/${f} is missing: run python scripts/prepare_external.py --data data on the login node first" >&2
    [ "${MODE}" = status ] || exit 1
  fi
done

# ----------------------------------------------------------------------------------------- the items
Q15A_ITEMS=logs/p3pack_q15a.items
Q15B_ITEMS=logs/p3pack_q15b.items
{
  echo "# kind config seed method tag overrides (written by slurm/p3_nova.sh)"
  for c in "${CON}" "${CCX}" "${SQ}" "${O15}" "${IFC}" "${S15}"; do echo "features ${c} 0 - - -"; done
  for c in "${CON}" "${CCX}" "${SQ}" "${O15}" "${IFC}" "${S15}"; do echo "external ${c} 0 seq - -"; done
  for s in 0 1 2; do echo "train ${O15} ${s} kl b10 grpo.kl_beta=10.0"; done
} > "${Q15A_ITEMS}"
{
  echo "# kind config seed method tag overrides (written by slurm/p3_nova.sh)"
  for s in 0 1 2; do for a in 0.02 0.05 0.1; do echo "scale ${O15} ${s} seq ${a} -"; done; done
  for s in 0 1 2; do echo "train ${O15} ${s} kl b100 grpo.kl_beta=100.0"; done
} > "${Q15B_ITEMS}"
echo "features ${IF7} 0 - - -" > logs/p3pack_q7_features.items
echo "external ${IF7} 0 seq - -" > logs/p3pack_q7_external.items
OO7=configs/v2/overlap_olmo7.yaml; L3=configs/fam/nq_llama3b.yaml
# OLMo-2-7B generates the external sets at batch 16 (at its batch of 64 the passages ran out of memory on an A100)
printf '%s\n' "features ${OO7} 0 - - -" "features ${L3} 0 - - -" "external ${OO7} 0 seq - extra.external_batch_size=16" \
  "external ${L3} 0 seq - -" \
  > logs/p3pack_fam.items
{
  echo "# kind config seed method tag overrides (written by slurm/p3_nova.sh)"
  for s in 0 1 2; do for a in 0.02 0.05 0.1; do echo "scale ${SQ} ${s} seq ${a} -"; done; done
  for s in 0 1 2; do echo "train ${SQ} ${s} kl b10 grpo.kl_beta=10.0"; echo "train ${SQ} ${s} kl b100 grpo.kl_beta=100.0"; done
} > logs/p3pack_sq.items

# ----------------------------------------------------------------------------------------- status
n_done=0; n_todo=0; n_other=0; todo_a=0; todo_b=0
for job in a b; do
if [ "${job}" = a ]; then items="${Q15A_ITEMS}"; part="${Q15A_PARTITION}"; else items="${Q15B_ITEMS}"; part="${Q15B_PARTITION}"; fi
echo "== p3pack-q15${job} (partition ${part}, ${GPU_TYPE}${CONSTRAINT:+, nodes ${CONSTRAINT}} = ${TARGET})"
t0="${n_todo}"
while read -r kind cfg s m tag ovr; do
  [[ "${kind}" == \#* ]] && continue
  exp="$(exp_of "${cfg}")"
  case "${kind}" in
    features)
      js="runs/${exp}/analysis/gate_features.json"; dev="$(devof "${js}")"
      if grep -q '"eval__ext_' "${js}" 2>/dev/null; then n_done=$((n_done + 1)); echo "done     ${js} (external sets)"
      elif [ -z "${dev}" ]; then n_other=$((n_other + 1)); echo "missing  ${js}: no stored gate features"
      elif [ "${dev}" != "${TARGET}" ]; then
        n_other=$((n_other + 1))
        case "${dev}" in *H200*) gt=h200; cn="" ;; *SXM4*) gt=a100; cn=compute-gpu ;; *) gt=a100; cn=epyc-7413 ;; esac
        echo "OTHER    ${js}: stored on ${dev}; the pack skips it. Submit it on its own GPU model:"
        echo "         PARTITION=<partition with ${gt}> CONFIG=${cfg} GPU_TYPE=${gt}${cn:+ CONSTRAINT=${cn}} MEM=16G FEATURES_TIME=00:30:00 FEATURES_ARGS=--add bash slurm/submit.sh features"
      else n_todo=$((n_todo + 1)); echo "to do    ${js} (external sets)"; fi ;;
    external)
      d="runs/${exp}/s${s}/${m}"
      if grep -q '"external"' "${d}/extra/summary.json" 2>/dev/null; then n_done=$((n_done + 1)); echo "done     ${d}/extra/external"
      elif [ ! -f "${d}/DONE" ]; then n_other=$((n_other + 1)); echo "missing  ${d} is not finished"
      else n_todo=$((n_todo + 1)); echo "to do    ${d}/extra/external"; fi ;;
    scale)
      d="runs/${exp}/s${s}/scale_$(printf 'a%.2f' "${tag}")"; src="runs/${exp}/s${s}/${m}"; dev="$(devof "${src}/config.json")"
      if [ -f "${d}/DONE" ]; then n_done=$((n_done + 1)); echo "done     ${d}"
      elif [ -n "${dev}" ] && [ "${dev}" != "${TARGET}" ]; then
        n_other=$((n_other + 1)); echo "OTHER    ${d}: ${src} ran on ${dev}; the pack would compare across GPU models"
      else n_todo=$((n_todo + 1)); echo "to do    ${d}"; fi ;;
    train)
      d="runs/${exp}/s${s}/${m}_${tag}"; dev="$(devof "${d}/config.json")"
      if [ -f "${d}/DONE" ]; then n_done=$((n_done + 1)); echo "done     ${d}"
      elif [ -n "${dev}" ] && [ "${dev}" != "${TARGET}" ]; then
        n_other=$((n_other + 1)); echo "OTHER    ${d}: started on ${dev}; it must continue on that GPU model"
      else n_todo=$((n_todo + 1)); echo "to do    ${d}${dev:+ (continues)}"; fi ;;
  esac
done < "${items}"
if [ "${job}" = a ]; then todo_a=$((n_todo - t0)); else todo_b=$((n_todo - t0)); fi
done
e7="$(exp_of "${IF7}")"; js7="runs/${e7}/analysis/gate_features.json"; dev7="$(devof "${js7}")"
case "${dev7}" in *H200*) gt7=h200; cn7="" ;; *SXM4*) gt7=a100; cn7=compute-gpu ;; *) gt7=a100; cn7=epyc-7413 ;; esac
echo "== 7B (features: partition ${Q7F_PARTITION}; generation: ${Q7_PARTITION})"
if grep -q '"eval__ext_' "${js7}" 2>/dev/null; then echo "done     ${js7} (external sets)"; q7f=0
elif [ -z "${dev7}" ]; then echo "missing  ${js7}: no stored gate features"; q7f=0
else echo "to do    ${js7} (external sets; stored on ${dev7}: GPU ${gt7}${cn7:+, nodes ${cn7}})"; q7f=1; fi
d7="runs/${e7}/s0/seq"
if grep -q '"external"' "${d7}/extra/summary.json" 2>/dev/null; then echo "done     ${d7}/extra/external"; q7x=0
elif [ ! -f "${d7}/DONE" ]; then echo "missing  ${d7} is not finished"; q7x=0
else echo "to do    ${d7}/extra/external"; q7x=1; fi
sqtodo=0
esq="$(exp_of "${SQ}")"
echo "== p3.15 on SQuAD 2.0 (SQUAD=1 to submit; partition ${SQ_PARTITION})"
for s in 0 1 2; do
  src="runs/${esq}/s${s}/seq"; dev="$(devof "${src}/config.json")"
  for a in 0.02 0.05 0.1; do
    d="runs/${esq}/s${s}/scale_$(printf 'a%.2f' "${a}")"
    if [ -f "${d}/DONE" ]; then echo "done     ${d}"
    elif [ ! -f "${src}/DONE" ]; then echo "missing  ${src} is not finished"
    elif [ -n "${dev}" ] && [ "${dev}" != "${TARGET}" ]; then echo "OTHER    ${d}: ${src} ran on ${dev}, not ${TARGET}"
    else echo "to do    ${d}"; sqtodo=$((sqtodo + 1)); fi
  done
  for b in b10 b100; do
    d="runs/${esq}/s${s}/kl_${b}"
    if [ -f "${d}/DONE" ]; then echo "done     ${d}"; else echo "to do    ${d}"; sqtodo=$((sqtodo + 1)); fi
  done
done
famtodo=0
echo "== two more families (FAMILIES=1 to submit; partition ${FAM_PARTITION})"
for c in "${OO7}" "${L3}"; do
  e="$(exp_of "${c}")"; js="runs/${e}/analysis/gate_features.json"; dev="$(devof "${js}")"; d="runs/${e}/s0/seq"
  if grep -q '"eval__ext_' "${js}" 2>/dev/null; then echo "done     ${js} (external sets)"
  elif [ -z "${dev}" ]; then echo "missing  ${js}: no stored gate features"
  elif [ "${dev}" != "${TARGET}" ]; then echo "OTHER    ${js}: stored on ${dev}, not ${TARGET}"
  else echo "to do    ${js} (external sets)"; famtodo=$((famtodo + 1)); fi
  if grep -q '"external"' "${d}/extra/summary.json" 2>/dev/null; then echo "done     ${d}/extra/external"
  elif [ ! -f "${d}/DONE" ]; then echo "missing  ${d} is not finished"
  elif [ -f "${d}/extra/FAILED" ] && [ "$(wc -l < "${d}/extra/FAILED")" -ge "${MAX_FAILS:-3}" ]; then
    echo "FAILED   ${d}/extra/external (failed ${MAX_FAILS:-3} times, see ${d}/extra/FAILED)"
  else echo "to do    ${d}/extra/external"; famtodo=$((famtodo + 1)); fi
done
echo "1.5B: done ${n_done}, to do ${n_todo}, needs another GPU model or a missing run ${n_other}"
[ "${MODE}" = status ] && exit 0

# ----------------------------------------------------------------------------------------- submit
queued() { squeue --me -h -o '%j' 2>/dev/null | grep -qxF "$1"; }
gres_avail() { sinfo -p "$1" -h -o '%G' 2>/dev/null | grep -q "gpu:$2"; }
sub() {  # sub <name> <partition> <gpu type> <constraint> <time> <mem> <items> <lanes>
  local name="$1" part="$2" gt="$3" cn="$4" tl="$5" mem="$6" items="$7" lanes="$8"
  if queued "${name}"; then echo "queued   ${name} (not submitted again)"; return 0; fi
  if ! gres_avail "${part}" "${gt}"; then
    echo "NOT SUBMITTED ${name}: partition ${part} lists no gpu:${gt} (sinfo -p ${part} -o %G); choose another" \
         "partition for it (Q15A_PARTITION, Q15B_PARTITION, Q7_PARTITION, Q7F_PARTITION)"
    return 0
  fi
  local acct=""; acct="$(acct_for "${part}")"; [ "${acct}" = none ] && acct=""
  local args=(--parsable --partition="${part}" --gres="gpu:${gt}:1" --time="${tl}" --mem="${mem}" --cpus-per-task=8
              --chdir="${REPO}" --export=ALL --job-name="${name}")
  [ -n "${acct}" ] && args+=(--account="${acct}")
  [ -n "${cn}" ] && args+=(--constraint="${cn}")
  if [ "${MODE}" = dry ]; then
    echo "DRY: ITEMS=${items} LANES=${lanes} sbatch ${args[*]} slurm/p3pack.sbatch"; return 0
  fi
  local id
  id=$(ITEMS="${items}" LANES="${lanes}" PACK_GRES="gpu:${gt}:1" PACK_TIME="${tl}" PACK_MEM="${mem}" PACK_ACCOUNT="${acct}" \
       PACK_CONSTRAINT="${cn}" sbatch "${args[@]}" slurm/p3pack.sbatch)
  echo "submitted ${id} ${name} (${part}, gpu:${gt}${cn:+, ${cn}}, ${tl}; log logs/${name}-${id}.out)"
}
if [ "${ONLY_FAMILIES:-0}" = 1 ]; then   # slurm/p3auto.sh: only the family job is managed from here
  todo_a=0; todo_b=0; q7f=0; q7x=0; SQUAD=0
fi
[ "${todo_a}" -gt 0 ] && sub p3pack-q15a "${Q15A_PARTITION}" "${GPU_TYPE}" "${CONSTRAINT}" "${PACK_TIME}" "${PACK_MEM}" "${REPO}/${Q15A_ITEMS}" "${LANES}"
[ "${todo_b}" -gt 0 ] && sub p3pack-q15b "${Q15B_PARTITION}" "${GPU_TYPE}" "${CONSTRAINT}" "${PACK_TIME}" "${PACK_MEM}" "${REPO}/${Q15B_ITEMS}" "${LANES}"
[ "${q7f}" = 1 ] && sub p3pack-q7-features "${Q7F_PARTITION}" "${gt7}" "${cn7}" 01:00:00 64G "${REPO}/logs/p3pack_q7_features.items" 1
[ "${q7x}" = 1 ] && sub p3pack-q7-external "${Q7_PARTITION}" a100 "" 03:00:00 64G "${REPO}/logs/p3pack_q7_external.items" 1
if [ "${SQUAD}" = 1 ] && [ "${sqtodo}" -gt 0 ]; then
  sub p3pack-sq "${SQ_PARTITION}" "${GPU_TYPE}" "${CONSTRAINT}" "${PACK_TIME}" "${PACK_MEM}" "${REPO}/logs/p3pack_sq.items" "${LANES}"
fi
if [ "${FAMILIES}" = 1 ] && [ "${famtodo}" -gt 0 ]; then
  sub p3pack-fam "${FAM_PARTITION}" "${GPU_TYPE}" "${CONSTRAINT}" 06:00:00 64G "${REPO}/logs/p3pack_fam.items" 1
fi
echo "then: bash slurm/p3_nova.sh status; when everything is done: sbatch slurm/p3ext.sbatch"
