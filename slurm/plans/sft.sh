# shellcheck shell=bash
# Add-on plan, sourced by slurm/plan.sh (PLAN_SFT.md): supervised fine-tuning without LoRA, i.e. full-rank updates.
# Kept apart from every other plan: configs in configs/sft/, runs in runs/sft_*/, tables in paper/sft_results/. The code
# is the shared trainer: twcore/grpo.py (grpo.objective=sft, grpo.sft_source), twcore/gating.py (lora.dense).
#
#   python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level sft   # CPU check of the new code, ~2 min
#   bash slurm/plan.sh sft1 status      done / queued / to do
#   AP_PLAN=sft AP_TARGET=20 sbatch --job-name=cpusft slurm/autopilot.sbatch   # its own autopilot: sft1.1-sft1.3,
#                                       queue logs/plan_queue_sft, folder logs/autopilot_sft (nothing shared with fam)
#   bash slurm/plan.sh sft-tables       tables in paper/sft_results/ (CPU, login node)
#   bash slurm/plan.sh sft-collect      the same, packed into sft_results_<date>.tar.gz to send
SFT_NQ=configs/sft/sft_nq_q15.yaml     # NQ, the gold answers, Qwen2.5-1.5B
SFT_IF=configs/sft/sft_ifc_q15.yaml    # IF, Qwen2.5-7B demonstrations, Qwen2.5-1.5B
SFT_NQ7=configs/sft/sft_nq_q7.yaml     # NQ, the gold answers, Qwen2.5-7B (optional: H200 only)
# method:tag:overrides. The configs make every run full-rank SFT (lora.dense, grpo.objective=sft): seq = ungated,
# tripwire_prompt = behind the prompt-time tripwire, seq_full = every block but the first (gate_layer=1), ungated
SFT_RUNS=(seq:: tripwire_prompt:: "seq:full:gate_layer=1")
SFT_RUNS7=(seq:: tripwire_prompt::)
STEPS+=(
  "sft1.1|SFT without LoRA, NQ (1.5B), seed 0: gold answers; full-rank Seq, Tripwire-prompt, Seq on every block but the first: 3 runs, about 1 h each"
  "sft1.2|SFT without LoRA, IF (1.5B), seed 0: Qwen2.5-7B demonstrations; the same 3 runs, about 1.5 h each"
  "sft1.3|last: gate features and LLM judge of every SFT experiment (each waits for its runs): up to 6 short jobs"
  "sft2.1|only if sft1 looks right: seeds 1-2 of sft1.1 and sft1.2: 12 runs, 1-1.5 h each (then queue sft1.3 again)"
  "sft2.2|optional, H200 only: NQ gold-answer SFT at Qwen2.5-7B, full-rank blocks 8-27, Seq and Tripwire-prompt, seed 0: 2 runs, about 2 h each (then queue sft1.3 again)"
)
h200() { local GPUS_NOW="h200" MEM_NOW="128G"; "$@"; }   # 7B full-rank: about 100 GB on the GPU, a 56 GB checkpoint

st_sft1_1() { local s; for s in $(seeds "0"); do big train "${SFT_NQ}" "${s}" 03:00:00 1 "${SFT_RUNS[@]}"; done; }
st_sft1_2() { local s; for s in $(seeds "0"); do big train "${SFT_IF}" "${s}" 04:00:00 1 "${SFT_RUNS[@]}"; done; }
st_sft2_1() {
  local s
  for s in $(seeds "1 2"); do
    big train "${SFT_NQ}" "${s}" 03:00:00 1 "${SFT_RUNS[@]}"
    big train "${SFT_IF}" "${s}" 04:00:00 1 "${SFT_RUNS[@]}"
  done
}
st_sft2_2() { local s; for s in $(seeds "0"); do h200 train "${SFT_NQ7}" "${s}" 04:00:00 1 "${SFT_RUNS7[@]}"; done; }
st_sft1_3() {   # gate features once s0/tripwire_prompt is done (its deployed probe, its GPU model), then the judge
  if want_addon_cfg "${SFT_NQ}" sft1.1 sft2.1; then features "${SFT_NQ}" wait; judge "${SFT_NQ}"; fi
  if want_addon_cfg "${SFT_IF}" sft1.2 sft2.1; then features "${SFT_IF}" wait; judge "${SFT_IF}"; fi
  if want_addon_cfg "${SFT_NQ7}" sft2.2; then big features "${SFT_NQ7}" wait; judge "${SFT_NQ7}"; fi
}
sft1_check() {  # bash slurm/plan.sh sft1 check: the CPU selftest of the SFT data sources, here (about 2 minutes)
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  python scripts/selftest.py --model models/Qwen2.5-1.5B-Instruct --level sft
}

addon_sft_tables() {  # bash slurm/plan.sh sft-tables: paper/sft_results/<exp>/switch/ and overview/addon_sft.{md,tex,json}
  source slurm/common.sh
  set +eu; activate_env >/dev/null; set -eu
  local out=paper/sft_results pair cfg name exp
  ok() { "$@" || echo "  (did not finish: $*)"; }
  # the SFT experiments and the LoRA experiments they are compared with (the same base model, data and gate)
  for pair in "${SFT_NQ}:Sftnq" "${SFT_IF}:Sftifc" "${SFT_NQ7}:Sftnqseven" "${O15}:Refnq" "${IFC}:Refifc" "${O7}:Refnqseven"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    if [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
      echo "  no gate features yet (step sft1.3): left out"; continue
    fi
    mkdir -p "${out}/${exp}/switch"
    [[ "${exp}" == sft_* ]] && ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/switch_report.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch"
  done
  ok python scripts/addon_report.py --plan sft --results "${out}" --runs runs --out "${out}/overview"
  echo "tables in ${out}/"
}

addon_sft_collect() {  # bash slurm/plan.sh sft-collect: tables + status + training logs in sft_results_<date>.tar.gz
  local out=paper/sft_results stamp d f
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}"
  addon_sft_tables 2>&1 | tee "${out}/report.txt" || true
  PLAN_QUEUE=logs/plan_queue_sft bash slurm/plan.sh sft1,sft2 status > "${out}/status.txt" 2>&1 || true
  # run folder, finished, GPU model, SFT examples kept (train/sft_samples.jsonl), the data log line
  for d in runs/sft_*/s*/*; do
    [ -f "${d}/config.json" ] || continue
    printf '%s\t%s\t%s\t%s\t%s\n' "${d}" "$(isfile "${d}/DONE")" \
      "$(grep -o '"device": "[^"]*"' "${d}/config.json" | head -1 | cut -d'"' -f4)" \
      "$(cat "${d}/stage1/train/sft_samples.jsonl" 2>/dev/null | wc -l)" \
      "$(grep -ho 'sft ([a-z]*): [^"]*' "${d}/log.jsonl" 2>/dev/null | tail -1)"
  done > "${out}/runs.tsv"
  for f in runs/sft_*/s*/*/stage1/train/log.jsonl; do   # loss per step
    [ -f "${f}" ] || continue
    mkdir -p "${out}/train/$(dirname "${f#runs/}")"
    cp "${f}" "${out}/train/${f#runs/}"
  done
  cp -r logs/autopilot_sft "${out}/autopilot" 2>/dev/null || true
  tar czf "sft_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/sft_results_${stamp}.tar.gz ($(du -h "sft_results_${stamp}.tar.gz" | cut -f1))"
}
