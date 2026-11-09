#!/bin/bash
# The v2 experiments as numbered steps; RUN_v2.md lists the commands in order. At most MAX_JOBS of your tw-* jobs
# are queued or running at any time (like %8 for a job array): a step command submits what fits now and puts the
# step in a queue (logs/plan_queue); every job that ends submits the next waiting run of the queue itself. A
# finished or queued run is never submitted twice; an unfinished run continues from its last checkpoint.
#
#   bash slurm/plan.sh steps            list all steps
#   bash slurm/plan.sh w1.1             step 1 of week 1 (NQ overlap, seed 0)
#   bash slurm/plan.sh w1               all steps of week 1, in order;  w1.1,w1.2 = several steps;  all = weeks 1-5
#   bash slurm/plan.sh w1 check         selftests of the week (w1: 1.5B on 1 and 2 GPUs; w2: the 7-8B models)
#   bash slurm/plan.sh w1 status        every run of the step(s): done / queued / FAILED / to do
#   bash slurm/plan.sh w1.1 commands    the plain submit command of every remaining run (to submit by hand instead)
#   bash slurm/plan.sh w1.1 dry         the sbatch commands, without submitting
#   bash slurm/plan.sh queue            steps in the queue;   bash slurm/plan.sh stop   empties the queue
#   bash slurm/plan.sh feed             fill free slots from the queue (jobs do this themselves when they end)
#   bash slurm/plan.sh tables           tables, figures and analyses of every finished experiment (login node)
#   bash slurm/plan.sh collect          the same, plus run status, packed into v2_results_<date>.tar.gz to send
#   add-on plans (slurm/plans/*.sh): steps sft* (PLAN_SFT.md), fam* (PLAN_SCALE.md), nov* (PLAN_NOV.md) and p3* (paper
#   version 3, PLAN_PAPER3.md: bash slurm/plan.sh p3-tables / p3-collect), never part
#   of "all"; their autopilots use their own queues: PLAN_QUEUE=logs/plan_queue_sft (or _fam, _nov) bash slurm/plan.sh
#   feed / queue; their tables: bash slurm/plan.sh sft-tables / sft-collect / fam-tables / fam-collect / nov-collect
#
# Settings (environment variables; SEEDS, SKIP, NGPU_7B, GPU_TYPE and SELFTEST_JOB are stored with the step):
#   MAX_JOBS  tw-* jobs queued or running at once (default 8; the last value given is kept in logs/plan_max_jobs)
#   GPU_TYPE  unset (default): a run takes an H200 if one is free when it is first submitted, otherwise an A100
#             (also 7-8B runs: the A100s have 80 GB). A run then always stays on the GPU model it started on.
#             GPU_TYPE=h200 or GPU_TYPE=a100 forces one model.
#   NGPU_7B   GPUs per 7-8B job (default 1; 2 or 4 after the 2-GPU selftest passed)
#   SEEDS     only these seeds of the step, e.g. SEEDS=0
#   SKIP      runs to leave out, e.g. SKIP="tripwire_prompt kl_b1" ("kl" leaves out every kl run)
#   SELFTEST_JOB  id of the 1-GPU selftest job: runs wait for it to pass
# A run that failed twice is not resubmitted (its FAILED file says when; delete the file to retry).
# Job time limits are never cut and submissions are never held back by the hour (Nov 4: no daily cancel any more).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
mkdir -p logs
SEL="${1:-}"
MODE="${2:-submit}"
QUEUE="${PLAN_QUEUE:-logs/plan_queue}"   # the add-on plans' autopilots use their own (logs/plan_queue_sft, ..._fam)
export PLAN_QUEUE="${QUEUE}"              # jobs inherit it, so a job that ends feeds the queue it came from
MAXF=logs/plan_max_jobs
[ "${QUEUE}" = logs/plan_queue ] || MAXF="${QUEUE}.max_jobs"
case "${MODE}" in submit|status|dry|commands|check) ;;
  *) echo "unknown action ${MODE} (submit, status, commands, dry, check)" >&2; exit 1 ;; esac
[ "${MODE}" = dry ] && export DRY=1
if [ -n "${MAX_JOBS:-}" ]; then echo "${MAX_JOBS}" > "${MAXF}"; fi   # jobs call feed without MAX_JOBS
MAXJ="${MAX_JOBS:-$(cat "${MAXF}" 2>/dev/null || echo 8)}"
# options of this command (stored with the steps it queues)
OPT_SEEDS="${SEEDS:-}"; OPT_SKIP="${SKIP:-}"; OPT_NGPU7="${NGPU_7B:-1}"; OPT_GPU="${GPU_TYPE:-auto}"
OPT_ST="${SELFTEST_JOB:-}"
if [ -f slurm/cluster.env ]; then source slurm/cluster.env; else source slurm/cluster.env.example; fi   # PARTITION
if ! grep -q windowok slurm/submit.sh; then   # the two scripts come from different updates
  echo "ERROR: slurm/submit.sh is older than slurm/plan.sh; unzip tripwire_update_all_since_v6.zip again" >&2; exit 1
fi
GPUS_SMALL="h200 a100"   # 1.5B runs: first choice, then the fallback (also used when neither is free right now)
GPUS_BIG="h200 a100"     # 7-8B runs: the same order (the A100s have 80 GB, "NVIDIA A100 80GB PCIe" in the job logs)
# A started run keeps to nodes with the GPU model it started on: GPU types alone do not tell the two A100 models
# apart (nova22-gpu-1..3: A100 80GB PCIe, feature epyc-7413; nova22-amp-7: A100-SXM4-80GB, feature compute-gpu;
# both are gpu:a100). "<part of the device name>=<node feature>;..."; a run that has not started gets no constraint.
GPU_MODEL_CONSTRAINTS="${GPU_MODEL_CONSTRAINTS:-A100 80GB PCIe=epyc-7413;A100-SXM4-80GB=compute-gpu}"
unset CONSTRAINT   # set per run below; never inherited from a job's environment
MEM_SMALL="${MEM_SMALL:-16G}"   # CPU memory per 1.5B job (sacct: peak 2.4 GB); more blocks jobs on busy nodes
MEM_BIG="${MEM_BIG:-64G}"       # per 7-8B job (a 7B model alone is about 15 GB in bfloat16)
# options of the step being processed
SEEDS_F="${OPT_SEEDS}"; SKIP_F="${OPT_SKIP}"; NGPU7_F="${OPT_NGPU7}"; GPU_F="${OPT_GPU}"; ST_F="${OPT_ST}"

# ----------------------------------------------------------------------------------------------- experiments
O15=configs/v2/overlap_q15.yaml        # NQ against Dolly, Qwen2.5-1.5B (partial overlap)
S15=configs/v2/stream_q15.yaml         # GSM8K -> Countdown -> K&K, Qwen2.5-1.5B
IFC=configs/v2/ladder_ifc_q15.yaml     # format-constrained Dolly requests (separable)
CON=configs/v2/ladder_concise_q15.yaml # plain Dolly requests, short answers (no separability)
S5=configs/v2/stream5_q15.yaml         # 5-task stream (+ MATH, Sudoku)
O7=configs/v2/overlap_q7.yaml          # NQ, Qwen2.5-7B
S7=configs/v2/stream_q7.yaml           # 3-task stream, Qwen2.5-7B
OO7=configs/v2/overlap_olmo7.yaml      # NQ, OLMo-2-7B (second model family)
SO7=configs/v2/stream_olmo7.yaml       # 3-task stream, OLMo-2-7B
IF7=configs/v2/ladder_ifc_q7.yaml      # IF (format-constrained Dolly requests), Qwen2.5-7B (v4)
CCX=configs/v2/ladder_concisectx_q15.yaml  # Concise on Dolly requests with a context passage: the middle rung (v6)

# method:tag:overrides (tag "" = the config's own value)
OVERLAP=(seq:: kl:: kl:b0.05:grpo.kl_beta=0.05 kl:b0.2:grpo.kl_beta=0.2 kl:b1:grpo.kl_beta=1.0
         refkl:b0.1:grpo.refkl_beta=0.1 refkl:: refkl:b10:grpo.refkl_beta=10.0
         nullspace:e0.80:grpo.nullspace_energy=0.8 nullspace:: nullspace:e0.99:grpo.nullspace_energy=0.99 router::
         tripwire:: tripwire:e0.10:tripwire.eps=0.10 tripwire:e0.30:tripwire.eps=0.30
         tripwire_prompt:: tripwire_prompt:e0.10:tripwire.eps=0.10 tripwire_prompt:e0.30:tripwire.eps=0.30)
STREAM=(seq:: kl:: kl:b0.05:grpo.kl_beta=0.05 replay:: refkl:: olora:: nullspace:: router:: tripwire:: tripwire_prompt::)
LADDER=(seq:: kl:: refkl:: nullspace:: router:: tripwire:: tripwire_prompt::)
LADDER_SWEEP=(kl:b0.2:grpo.kl_beta=0.2 refkl:b10:grpo.refkl_beta=10.0 tripwire:e0.10:tripwire.eps=0.10
              tripwire:e0.30:tripwire.eps=0.30 tripwire_prompt:e0.10:tripwire.eps=0.10)
SPLIT=(seq:L4:gate_layer=4 seq:L14:gate_layer=14 tripwire:L4:gate_layer=4 tripwire:L14:gate_layer=14
       tripwire_prompt:L4:gate_layer=4 tripwire_prompt:L14:gate_layer=14)
BIG_STREAM=(seq:: kl:: refkl:: router:: tripwire:: tripwire_prompt::)
BIG_OVERLAP=(seq:: kl:: refkl:: nullspace:: router:: tripwire:: tripwire_prompt:: tripwire_prompt:e0.10:tripwire.eps=0.10)
CORE=(seq:: refkl:: router:: tripwire:: tripwire_prompt::)
NQ7=(seq:: refkl:: router:: tripwire_prompt:: tripwire_prompt:e0.10:tripwire.eps=0.10)   # 7B NQ after decision B
# weight-importance penalties (lambda roughly on Ref-KL's beta scale; MAS and L2 rescaled to the Fisher's mass) and A-GEM
REG_NQ=(ewc:l0.1:grpo.ewc_lambda=0.1 ewc:: ewc:l10:grpo.ewc_lambda=10.0
        mas:l0.1:grpo.mas_lambda=0.1 mas:: mas:l10:grpo.mas_lambda=10.0
        l2:l0.1:grpo.l2_lambda=0.1 l2:: l2:l10:grpo.l2_lambda=10.0 agem::)
REG_STREAM=(ewc:: mas:: l2:: agem::)
S5_ALL=(seq:: kl:: refkl:: olora:: nullspace:: router:: tripwire:: tripwire_prompt::)
S5_CORE=(seq:: refkl:: tripwire:: tripwire_prompt::)
# v3 (Oct 24, PLAN_v3.md): the prompt-time tripwire with whole-sequence activation (run folders tripwire_prompt_seq,
# tripwire_prompt_seq_e0.10); each spec is one quoted word
TWS="tripwire_prompt:seq:tripwire.activation=sequence"
TWS10="tripwire_prompt:seq_e0.10:tripwire.activation=sequence tripwire.eps=0.10"
TWP10="tripwire_prompt:e0.10:tripwire.eps=0.10"

# ----------------------------------------------------------------------------------------------- steps
STEPS=(
  "w1.1|NQ overlap (1.5B), seed 0: 18 runs, up to 3 h each"
  "w1.2|3-task stream (1.5B), seed 0: 10 runs, up to 8 h each"
  "w1.3|NQ overlap (1.5B), seeds 1-2: 36 runs, up to 3 h each"
  "w1.4|3-task stream (1.5B), seeds 1-2: 20 runs, up to 8 h each"
  "w1.5|evaluations of finished week-1 runs: 12 adapter-scaling + 6 robustness jobs, up to 3 h each"
  "w2.1|IF overlap task (1.5B): 7 methods x 3 seeds + 5 sweep points = 26 runs, up to 3 h each"
  "w2.2|Concise overlap task (1.5B): 26 runs, up to 3 h each"
  "w2.3|split depth on NQ (gate at block 4 / 14; seq, tripwire, tripwire_prompt; seeds 0-1): 12 runs, up to 3 h"
  "w2.4|adapter-scaling evaluations of the IF and Concise runs: 18 jobs, up to 3 h each"
  "w3.1|NQ overlap, Qwen2.5-7B, seed 0: 8 runs, up to 8 h each"
  "w3.2|3-task stream, Qwen2.5-7B, seed 0: 6 runs, up to 12 h each (a run resubmits itself until done)"
  "w3.3|evaluations of the 7B runs: 1 adapter-scaling + 2 robustness jobs"
  "w4.1|5-task stream (1.5B), seed 0: 8 runs, up to 8 h each"
  "w4.2|NQ overlap, OLMo-2-7B, seed 0: 5 runs, up to 8 h each"
  "w4.3|3-task stream, OLMo-2-7B, seed 0: 5 runs, up to 12 h each"
  "w4.4|NQ overlap, Qwen2.5-7B, seed 1: 5 runs, up to 8 h each"
  "w4.5|robustness evaluations of the 5-task stream: 2 jobs"
  "w5.1|5-task stream (1.5B), seeds 1-2: 8 runs, up to 8 h each"
  "w5.2|NQ overlap, Qwen2.5-7B, seed 2: 5 runs, up to 8 h each"
  "w5x.1|optional: 3-task stream, Qwen2.5-7B, seeds 1-2: 8 runs, up to 12 h each"
  "w5.3|EWC, MAS and L2 (lambda 0.1 / 1 / 10) and A-GEM on NQ (1.5B), seeds 0-2: 30 runs, up to 3 h each"
  "w5.4|EWC, MAS, L2 and A-GEM on the 3-task stream (1.5B), seeds 0-2: 12 runs, up to 8 h each"
  "w5.5|score ablation, no training: NQ, IF and Concise (1.5B) and NQ (7B): 4 jobs, under 1 h each"
  "w5.6|NQ overlap, OLMo-2-7B, seeds 1-2: 10 runs, up to 8 h each"
  "v3.1|gate features of every experiment (no training): 7 jobs, under 1 h each"
  "v3.2|NQ overlap, OLMo-2-7B, seeds 1-2: Seq-GRPO and Ref-KL, 4 runs, up to 8 h each"
  "v3.3|3-task stream, Qwen2.5-7B, seeds 1-2: Seq-GRPO and Ref-KL, 4 runs, about 20 h each (12 h jobs, resubmitted)"
  "v3.4.1|whole-sequence Tripwire-prompt at 7B, seeds 0-2: 3-task stream (3 runs, about 20 h each), NQ (3 runs, up to 8 h)"
  "v3.4.2|whole-sequence Tripwire-prompt at 1.5B: NQ eps 0.03 / 0.1 and 3-task stream, seeds 0-2; IF eps 0.03 / 0.1, seed 0: 11 runs (+2 short evaluations)"
  "v3.5|LLM judge of the changed responses (each config waits until its training runs are done): 7 jobs, 1-3 h each"
  "v3.6|5-task stream (1.5B), Seq-GRPO and whole-sequence Tripwire, seeds 0-2, and its gate features: 7 jobs"
  "v3.7|Tripwire-prompt (decision-token activation), 3-task stream, Qwen2.5-7B, seeds 1-2: 2 runs, about 20 h each"
  "v3.8|EWC (lambda 0.1 / 1 / 10) on NQ (1.5B), seeds 0-2: 9 runs, up to 3 h each"
  "v3.9|harmful requests in the IF format: evaluations of 7 finished IF runs (no training), under 15 min each"
  # v4 (Oct 27, PLAN_v4.md): only what the paper needs. Must: v4.1-v4.4 (v4.4 last); should: v4.5, v4.6
  "v4.1|must: IF at Qwen2.5-7B, Seq-GRPO and Tripwire-prompt seeds 0-2 (6 runs, up to 8 h each, resubmitted), then gate features and harmful-request evaluations (7 short jobs)"
  "v4.2|must: 5-task stream (1.5B), Tripwire-prompt (decision token) seeds 0-2 (3 runs, up to 8 h each, resubmitted), then gate features again (1 short job)"
  "v4.3|must: NQ, OLMo-2-7B, Tripwire-prompt seeds 1-2: 2 runs, up to 7 h each"
  "v4.4|must, last: LLM judge of the new runs (5-task stream, OLMo-2 NQ, IF 7B, IF and Concise 1.5B): up to 5 jobs, 1-3 h each"
  "v4.5|should: Tripwire-prompt eps 0.1 on IF and Concise (1.5B), seeds 1-2: 4 runs up to 3 h, + 2 harmful-request evaluations"
  "v4.6|should: NQ, OLMo-2-7B, router seeds 1-2: 2 runs, up to 7 h each"
  # v5 (Oct 28, PLAN_v5.md): review points that need the GPU; the CPU analyses run in "tables" / "collect"
  "v5.1|stronger baselines, NQ (1.5B), seeds 0-2 (new training code: after the selftest): Ref-KL beta 10 and 100 with 4 samples per reference prompt and 64 per step, null-space refreshed every 25 steps: 9 runs, up to 3 h each"
  "v5.2|gate firing on new pools (AdvBench after an earlier exchange and in 3 jailbreak templates): features added to 9 experiments, no training, 9 jobs, 30-45 min each"
  "v5.3|second judge (OLMo-2-7B-Instruct) of Seq-GRPO and Tripwire-prompt: NQ, IF, Concise and the 3-task stream (1.5B), NQ and IF (7B): 6 jobs, up to 1 h each"
  "v5.4|last: LLM judge (Qwen2.5-7B) of the new NQ runs of v5.1 (1 job, waits for them)"
  # v6 (Oct 28, PLAN_v6.md): add-ons from the two reviews; the CPU add-ons run in "tables" / "collect"
  "v6.1|middle rung (Concise on Dolly requests with a context passage, 1.5B): data, gate features and separability, no training: 1 job, under 1 h"
  "v6.2|middle rung, trained only if its probe AUROC (v6.1) is in [0.70, 0.95]: Seq-GRPO and Tripwire-prompt, seeds 0-2 (6 runs, up to 3 h), then gate features again (1 job)"
  "v6.3|three seeds for the 1.5B rows that have fewer: A-GEM and split depth on NQ, whole-sequence Tripwire-prompt on IF: up to 9 runs, up to 3 h each"
  "v6.4|robustness evaluations (sampling exactness, harmful requests in task formats, format shift) of the 3-task stream, seeds 1-2: Tripwire, Tripwire-prompt, Seq-GRPO: 6 jobs, 1-2 h each"
  "v6.5|last: LLM judge of the middle rung's runs (1 job, waits for them; nothing if v6.2 was skipped)"
  # v7 (Oct 28, PLAN_v7.md): pilot - does anything depend on LoRA or on RL? Full-rank updates and SFT, IF 1.5B
  "v7.1|pilot, IF (1.5B), seed 0 (new training code: after the v7 selftest): SFT on rejection-sampled data (Seq, Tripwire-prompt): 2 runs, up to 3 h each"
  "v7.2|last: LLM judge of the pilot runs (waits for the SFT runs and the full-rank runs of v7.4)"
  "v7.4|pilot, full-rank runs, not in the autopilot (submit them yourself, e.g. on nova: slurm/pilot_dense.sh): Seq-GRPO lr 5e-6 and 2e-6, Tripwire-prompt 5e-6, every block but the first 5e-6, IF seed 0: 4 runs, up to 6 h each"
  "v7.3|not in the autopilot, only if the pilot looks right: the pilot settings on IF seeds 1-2 and NQ seeds 0-2: 30 runs"
)
st_w1_1() { for s in $(seeds "0"); do train "${O15}" "${s}" 03:00:00 1 "${OVERLAP[@]}"; done; }
st_w1_2() { for s in $(seeds "0"); do train "${S15}" "${s}" 08:00:00 1 "${STREAM[@]}"; done; }
st_w1_3() { for s in $(seeds "1 2"); do train "${O15}" "${s}" 03:00:00 1 "${OVERLAP[@]}"; done; }
st_w1_4() { for s in $(seeds "1 2"); do train "${S15}" "${s}" 08:00:00 1 "${STREAM[@]}"; done; }
st_w1_5() {
  for s in $(seeds "0 1 2"); do
    scale "${O15}" "${s}" seq 0.25 0.5 0.75
    scale "${S15}" "${s}" seq 0.5
  done
  for s in $(seeds "0"); do extra "${S15}" "${s}" tripwire tripwire_prompt router seq refkl nullspace; done
}
st_w2_1() {
  for s in $(seeds "0 1 2"); do train "${IFC}" "${s}" 03:00:00 1 "${LADDER[@]}"; done
  for s in $(seeds "0"); do train "${IFC}" "${s}" 03:00:00 1 "${LADDER_SWEEP[@]}"; done
}
st_w2_2() {
  for s in $(seeds "0 1 2"); do train "${CON}" "${s}" 03:00:00 1 "${LADDER[@]}"; done
  for s in $(seeds "0"); do train "${CON}" "${s}" 03:00:00 1 "${LADDER_SWEEP[@]}"; done
}
st_w2_3() { for s in $(seeds "0 1"); do train "${O15}" "${s}" 03:00:00 1 "${SPLIT[@]}"; done; }
st_w2_4() {
  for s in $(seeds "0 1 2"); do
    scale "${IFC}" "${s}" seq 0.25 0.5 0.75
    scale "${CON}" "${s}" seq 0.25 0.5 0.75
  done
}
st_w3_1() { for s in $(seeds "0"); do big train "${O7}" "${s}" 08:00:00 "${NGPU7_F}" "${BIG_OVERLAP[@]}"; done; }
st_w3_2() { for s in $(seeds "0"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" "${BIG_STREAM[@]}"; done; }
st_w3_3() {
  for s in $(seeds "0"); do scale "${O7}" "${s}" seq 0.5; extra "${S7}" "${s}" tripwire seq; done
}
st_w4_1() { for s in $(seeds "0"); do train "${S5}" "${s}" 08:00:00 1 "${S5_ALL[@]}"; done; }
st_w4_2() { for s in $(seeds "0"); do big train "${OO7}" "${s}" 08:00:00 "${NGPU7_F}" "${NQ7[@]}"; done; }
st_w4_3() { for s in $(seeds "0"); do big train "${SO7}" "${s}" 12:00:00 "${NGPU7_F}" "${CORE[@]}"; done; }
st_w4_4() { for s in $(seeds "1"); do big train "${O7}" "${s}" 08:00:00 "${NGPU7_F}" "${NQ7[@]}"; done; }
st_w4_5() { for s in $(seeds "0"); do extra "${S5}" "${s}" tripwire seq; done; }
st_w5_1() { for s in $(seeds "1 2"); do train "${S5}" "${s}" 08:00:00 1 "${S5_CORE[@]}"; done; }
st_w5_2() { for s in $(seeds "2"); do big train "${O7}" "${s}" 08:00:00 "${NGPU7_F}" "${NQ7[@]}"; done; }
st_w5x_1() { for s in $(seeds "1 2"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" "${S5_CORE[@]}"; done; }
st_w5_3() { for s in $(seeds "0 1 2"); do train "${O15}" "${s}" 03:00:00 1 "${REG_NQ[@]}"; done; }
st_w5_4() { for s in $(seeds "0 1 2"); do train "${S15}" "${s}" 08:00:00 1 "${REG_STREAM[@]}"; done; }
st_w5_5() { ablate "${O15}"; ablate "${IFC}"; ablate "${CON}"; big ablate "${O7}"; }
st_w5_6() { for s in $(seeds "1 2"); do big train "${OO7}" "${s}" 08:00:00 "${NGPU7_F}" "${NQ7[@]}"; done; }
st_v3_1() {
  features "${O15}"; features "${IFC}"; features "${CON}"; features "${S15}"
  big features "${O7}"; big features "${S7}"; big features "${OO7}"
}
st_v3_2() { for s in $(seeds "1 2"); do big train "${OO7}" "${s}" 08:00:00 "${NGPU7_F}" seq:: refkl::; done; }
st_v3_3() { for s in $(seeds "1 2"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" seq:: refkl::; done; }
st_v3_4_1() {   # the longest runs first
  for s in $(seeds "0 1 2"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" "${TWS}"; done
  for s in $(seeds "0 1 2"); do big train "${O7}" "${s}" 08:00:00 "${NGPU7_F}" "${TWS}"; done
}
st_v3_4_2() {
  for s in $(seeds "0 1 2"); do train "${S15}" "${s}" 08:00:00 1 "${TWS}"; done
  for s in $(seeds "0 1 2"); do train "${O15}" "${s}" 03:00:00 1 "${TWS}" "${TWS10}"; done
  for s in $(seeds "0"); do train "${IFC}" "${s}" 03:00:00 1 "${TWS}" "${TWS10}"; done
  for s in $(seeds "0"); do advx "${IFC}" "${s}" tripwire_prompt:seq tripwire_prompt:seq_e0.10; done   # as in v3.9
}
st_v3_5() {
  judge "${O15}"; judge "${IFC}"; judge "${CON}"; judge "${S15}"; judge "${O7}"; judge "${S7}"; judge "${OO7}"
}
st_v3_6() { features "${S5}"; for s in $(seeds "0 1 2"); do train "${S5}" "${s}" 08:00:00 1 seq:: "${TWS}"; done; }
st_v3_7() { for s in $(seeds "1 2"); do big train "${S7}" "${s}" 12:00:00 "${NGPU7_F}" tripwire_prompt::; done; }
st_v3_8() {
  for s in $(seeds "0 1 2"); do
    train "${O15}" "${s}" 03:00:00 1 ewc:l0.1:grpo.ewc_lambda=0.1 ewc:: ewc:l10:grpo.ewc_lambda=10.0
  done
}
st_v3_9() {   # inputs of the remedy for harmful requests in the IF format (scripts/switch_report.py)
  for s in $(seeds "0 1 2"); do advx "${IFC}" "${s}" seq tripwire_prompt; done
  for s in $(seeds "0"); do advx "${IFC}" "${s}" tripwire_prompt:e0.10; done
}
# v4 (PLAN_v4.md)
st_v4_1() {   # IF at 7B: the switch and the two-sided tripwire at 7B on a task with a large gain
  for s in $(seeds "0 1 2"); do big train "${IF7}" "${s}" 08:00:00 "${NGPU7_F}" seq:: tripwire_prompt::; done
  big features "${IF7}" wait   # after s0/tripwire_prompt: copies its deployed probe, same GPU model
  for s in $(seeds "0 1 2"); do big advx "${IF7}" "${s}" seq tripwire_prompt; done   # remedy inputs, as v3.9
}
st_v4_2() {   # the main method on the longest stream (only the whole-sequence variant has run there)
  for s in $(seeds "0 1 2"); do train "${S5}" "${s}" 08:00:00 1 tripwire_prompt::; done
  features "${S5}" wait        # redone once s0/tripwire_prompt exists: the switch then uses its deployed probe
}
st_v4_3() { for s in $(seeds "1 2"); do big train "${OO7}" "${s}" 07:00:00 "${NGPU7_F}" tripwire_prompt::; done; }
st_v4_4() { judge "${S5}"; judge "${OO7}"; judge "${IF7}"; judge "${IFC}"; judge "${CON}"; }
st_v4_5() {
  for s in $(seeds "1 2"); do train "${IFC}" "${s}" 03:00:00 1 "${TWP10}"; train "${CON}" "${s}" 03:00:00 1 "${TWP10}"; done
  for s in $(seeds "1 2"); do advx "${IFC}" "${s}" tripwire_prompt:e0.10; done
}
st_v4_6() { for s in $(seeds "1 2"); do big train "${OO7}" "${s}" 07:00:00 "${NGPU7_F}" router::; done; }
# v5 (PLAN_v5.md)
NQB=("refkl:b10s4:grpo.refkl_beta=10.0 grpo.refkl_batch=64 grpo.refkl_samples=4"
     "refkl:b100s4:grpo.refkl_beta=100.0 grpo.refkl_batch=64 grpo.refkl_samples=4"
     "nullspace:rf25:grpo.nullspace_refresh=25")
st_v5_1() { for s in $(seeds "0 1 2"); do train "${O15}" "${s}" 03:00:00 1 "${NQB[@]}"; done; }
st_v5_2() {
  local c
  for c in "${O15}" "${IFC}" "${CON}" "${S15}" "${S5}"; do features_add "${c}"; done
  for c in "${O7}" "${IF7}" "${S7}" "${OO7}"; do big features_add "${c}"; done
}
st_v5_3() { local c; for c in "${O15}" "${IFC}" "${CON}" "${S15}" "${O7}" "${IF7}"; do judge2 "${c}"; done; }
# v6 (PLAN_v6.md)
MID=(seq:: tripwire_prompt::)
MIDOK=""
mid_decision() {  # MIDOK = 1 if the middle rung's probe AUROC is inside the range, 0 if outside, "" if not measured yet;
                  # once a run of it was submitted the decision stays 1 (the features are redone after training)
  local exp js
  exp="$(exp_of "${CCX}")"
  js="runs/${exp}/analysis/separability.json"
  MIDOK=""
  if compgen -G "runs/${exp}/s[0-9]*/seq" >/dev/null || compgen -G "runs/${exp}/s[0-9]*/tripwire_prompt" >/dev/null; then
    MIDOK=1; return 0
  fi
  [ -f "${js}" ] || return 0
  if grep -q '"in_range": true' "${js}"; then MIDOK=1; else MIDOK=0; fi
}
st_v6_1() { features "${CCX}"; }   # the job also writes the task data (prepare_ladder.py) and separability.json
st_v6_2() {
  local s
  mid_decision
  if [ -z "${MIDOK}" ]; then N_WAIT+=1; say "waiting  runs/$(exp_of "${CCX}") (separability.json of step v6.1)"; return 0; fi
  if [ "${MIDOK}" = 0 ]; then
    say "skipped  runs/$(exp_of "${CCX}"): probe AUROC outside the middle range (analysis/separability.json)"; return 0
  fi
  for s in $(seeds "0 1 2"); do train "${CCX}" "${s}" 03:00:00 1 "${MID[@]}"; done
  features "${CCX}" wait   # again once s0/tripwire_prompt is done: the switch then uses its deployed probe
}
st_v6_3() {
  local s
  for s in $(seeds "0 1 2"); do train "${O15}" "${s}" 03:00:00 1 agem::; done          # the seed that is missing
  for s in $(seeds "2"); do train "${O15}" "${s}" 03:00:00 1 "${SPLIT[@]}"; done       # split depth: seeds 0-1 exist
  for s in $(seeds "1 2"); do train "${IFC}" "${s}" 03:00:00 1 "${TWS}"; done          # whole sequence on IF: seed 0 exists
}
st_v6_4() { local s; for s in $(seeds "1 2"); do extra "${S15}" "${s}" tripwire tripwire_prompt seq; done; }
st_v6_5() {   # waits for the middle rung's runs (done or failed twice), then judges them
  local s spec method tag ovr d
  mid_decision
  if [ -z "${MIDOK}" ]; then N_WAIT+=1; say "waiting  runs/$(exp_of "${CCX}")/analysis/judge (step v6.1)"; return 0; fi
  [ "${MIDOK}" = 0 ] && return 0
  for s in $(seeds "0 1 2"); do
    for spec in "${MID[@]}"; do
      IFS=: read -r method tag ovr <<< "${spec}"
      skipped "${method}${tag:+_${tag}}" "${method}" && continue
      d="runs/$(exp_of "${CCX}")/s${s}/${method}${tag:+_${tag}}"
      if [ ! -f "${d}/DONE" ] && ! { [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge 2 ]; }; then
        N_WAIT+=1; say "waiting  runs/$(exp_of "${CCX}")/analysis/judge (needs ${d})"; return 0
      fi
    done
  done
  judge "${CCX}"
}
# v7 (PLAN_v7.md): the pilot. Full-rank updates use about a tenth of the LoRA learning rate (LoRA Without Regret)
DENSE=("seq:dense5e6:lora.dense=true grpo.lr=5.0e-6" "seq:dense2e6:lora.dense=true grpo.lr=2.0e-6"
       "tripwire_prompt:dense5e6:lora.dense=true grpo.lr=5.0e-6" "seq:full5e6:lora.dense=true grpo.lr=5.0e-6 gate_layer=1")
SFT=("seq:sft:grpo.objective=sft" "tripwire_prompt:sft:grpo.objective=sft")
st_v7_1() { local s; for s in $(seeds "0"); do train "${IFC}" "${s}" 03:00:00 1 "${SFT[@]}"; done; }
st_v7_4() {   # big: 64 GB of CPU memory, since a dense run's checkpoint (12-17 GB) is loaded on the CPU when it resumes.
              # Not in the autopilot: slurm/pilot_dense.sh submits these to another partition (status / commands work here)
  local s
  for s in $(seeds "0"); do big train "${IFC}" "${s}" 06:00:00 1 "${DENSE[@]}"; done
}
st_v7_2() {   # waits until every pilot run is done (or failed twice, or left out by SKIP), then judges the new pairs
  local s spec method tag ovr d
  for s in $(seeds "0"); do
    for spec in "${DENSE[@]}" "${SFT[@]}"; do
      IFS=: read -r method tag ovr <<< "${spec}"
      skipped "${method}_${tag}" "${method}" && continue
      d="runs/$(exp_of "${IFC}")/s${s}/${method}_${tag}"
      if [ ! -f "${d}/DONE" ] && ! { [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge 2 ]; }; then
        N_WAIT+=1; say "waiting  runs/$(exp_of "${IFC}")/analysis/judge (needs ${d})"; return 0
      fi
    done
  done
  judge "${IFC}"
}
st_v7_3() {
  local s
  for s in $(seeds "1 2"); do big train "${IFC}" "${s}" 06:00:00 1 "${DENSE[@]}"; train "${IFC}" "${s}" 03:00:00 1 "${SFT[@]}"; done
  for s in $(seeds "0 1 2"); do big train "${O15}" "${s}" 06:00:00 1 "${DENSE[@]}"; train "${O15}" "${s}" 03:00:00 1 "${SFT[@]}"; done
}
st_v5_4() {   # waits until every v5.1 run is done (or failed twice), then judges the new pairs of the config
  local s spec method tag ovr d
  for s in $(seeds "0 1 2"); do
    for spec in "${NQB[@]}"; do
      IFS=: read -r method tag ovr <<< "${spec}"
      skipped "${method}_${tag}" "${method}" && continue   # left out of v5.1 by SKIP: nothing to wait for
      d="runs/$(exp_of "${O15}")/s${s}/${method}_${tag}"
      if [ ! -f "${d}/DONE" ] && ! { [ -f "${d}/FAILED" ] && [ "$(wc -l < "${d}/FAILED")" -ge 2 ]; }; then
        N_WAIT+=1; say "waiting  runs/$(exp_of "${O15}")/analysis/judge (needs ${d})"; return 0
      fi
    done
  done
  judge "${O15}"
}

w1_check() {
  choose_gpu "${GPUS_SMALL}" 1
  NGPU=1 NAME=q15 DEPEND="" BEGIN="" GPU_TYPE="${CHOSEN}" bash slurm/submit.sh selftest || true
  choose_gpu "${GPUS_SMALL}" 2
  NGPU=2 NAME=q15 DEPEND="" BEGIN="" GPU_TYPE="${CHOSEN}" bash slurm/submit.sh selftest || true
  echo "next, when logs/tw-selftest-q15-g1-<id>.out ends with PASSED: bash slurm/plan.sh w1.1"
}
w2_check() {  # real-model checks of the 7-8B models (download them first, RUN_v2.md week 1)
  local m name dir cfg
  for m in "q7:models/Qwen2.5-7B-Instruct:${S7}" "olmo7:models/OLMo-2-1124-7B-Instruct:${SO7}"; do
    IFS=: read -r name dir cfg <<< "${m}"
    if [ ! -d "${dir}" ]; then echo "skip ${name}: ${dir} not downloaded yet"; continue; fi
    choose_gpu "${GPUS_BIG}" 1
    LEVEL=real NAME="${name}" MODEL_DIR="${dir}" CONFIG="${cfg}" NGPU=1 DEPEND="" BEGIN="" \
      GPU_TYPE="${CHOSEN}" bash slurm/submit.sh selftest || true
  done
  echo "the logs (logs/tw-selftest-q7-g1-*.out, ...-olmo7-...) estimate the GPU-hours of one run per method"
}

tables() {  # every table, figure and analysis from the finished runs; output in paper/v2_results/<exp>/
  source slurm/common.sh
  set +eu                     # conda's activation scripts do not run under set -u
  activate_env >/dev/null
  set -eu
  local out=paper/v2_results cfg pair name exp tw
  ok() { "$@" || echo "  (did not finish: $*)"; }
  for cfg in "${S15}" "${S5}" "${S7}" "${SO7}"; do
    exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    mkdir -p "${out}/${exp}/analysis"
    ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/make_tables.py --runs "runs/${exp}" --paper "${out}/${exp}"
    ok python scripts/analyze.py fired --runs "runs/${exp}/s0" --out "${out}/${exp}/analysis"
    ok python scripts/analyze.py coverage --runs "runs/${exp}/s0" --method tripwire
  done
  for pair in "${O15}:" "${IFC}:Ifc" "${CON}:Concise" "${O7}:Seven" "${OO7}:Olmo" "${IF7}:Ifcseven" "${CCX}:Concisectx"; do
    cfg="${pair%%:*}"; name="${pair##*:}"
    exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] || continue
    echo "== ${exp}"
    mkdir -p "${out}/${exp}/analysis"
    ok python scripts/check_devices.py --runs "runs/${exp}"
    ok python scripts/make_overlap_report.py --runs "runs/${exp}" --paper "${out}/${exp}" --name "${name}"
    ok python scripts/analyze.py limit --runs "runs/${exp}/s0" --out "${out}/${exp}/analysis"
  done
  for f in runs/*/analysis/score_ablation.json; do   # score ablation (step w5.5): table and numbers
    [ -f "${f}" ] || continue
    exp="$(basename "$(dirname "$(dirname "${f}")")")"
    mkdir -p "${out}/${exp}/analysis"
    cp "${f}" "$(dirname "${f}")/score_table.tex" "${out}/${exp}/analysis/" 2>/dev/null || true
  done
  # certified switches (step v3.1 features + finished runs; CPU only): every trained model behind every gate variant
  for pair in "${O15}:Nq:" "${IFC}:Ifc:" "${CON}:Concise:advbench,general" "${S15}:Stream:" "${S5}:Streamfive:" \
              "${O7}:Seven:" "${S7}:Sevenstream:" "${OO7}:Olmo:" "${IF7}:Ifcseven:" "${CCX}:Concisectx:advbench,general"; do
    IFS=: read -r cfg name prot <<< "${pair}"
    exp="$(exp_of "${cfg}")"
    [ -f "runs/${exp}/analysis/gate_features.npz" ] || continue
    echo "== ${exp}: certified switches"
    mkdir -p "${out}/${exp}/switch"
    ok python scripts/switch_report.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch" ${prot:+--protect ${prot}}
    # v5 review analyses (CPU): judged drift, beta sweep, calibration vs held-out, numerics floor, forfeited transfer
    ok python scripts/switch_analyses.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch" ${prot:+--protect ${prot}}
    # v6 review add-ons (CPU): other gates natural vs calibrated, simulated middle, gate re-draws, region coverage, CIs
    ok python scripts/review_analyses.py --config "${cfg}" --name "${name}" --out "${out}/${exp}/switch"
    cp "runs/${exp}/analysis/separability.json" "${out}/${exp}/switch/" 2>/dev/null || true
    cp "runs/${exp}/analysis/judge/sample_for_manual.tsv" "${out}/${exp}/switch/" 2>/dev/null || true
  done
  # paper inputs across experiments: switch and gate overview tables, predicted-vs-actual points, \ovw / \ovg macros
  ok python scripts/make_switch_overview.py --results "${out}"
  # second judge against the first (v5.3); your labels where filled in: save a filled copy of
  # <exp>/switch/sample_for_manual.tsv as <exp>/switch/sample_labeled.tsv (column your_label: SAME / STYLE / DIFFERENT)
  local ja="" man=()
  for pair in "${O15}:Nq" "${IFC}:Ifc" "${CON}:Concise" "${S15}:Stream" "${O7}:Seven" "${IF7}:Ifcseven"; do
    cfg="${pair%%:*}"; name="${pair##*:}"; exp="$(exp_of "${cfg}")"
    [ -d "runs/${exp}" ] && ja+="${name}:runs/${exp},"
    [ -f "${out}/${exp}/switch/sample_labeled.tsv" ] && man+=("${out}/${exp}/switch/sample_labeled.tsv")
  done
  if [ -n "${ja}" ]; then
    ok python scripts/judge_agreement.py --exps "${ja%,}" --tag olmo --out "${out}/overview" ${man[@]+--manual "${man[@]}"}
  fi
  exp="$(exp_of "${S7}")"   # the collapsed whole-sequence seed of the 7B stream: per-seed accuracy and reward curves
  if [ -d "runs/${exp}" ]; then
    mkdir -p "${out}/${exp}/analysis"
    ok python scripts/seed_report.py --runs "runs/${exp}" --methods tripwire_prompt_seq,tripwire_prompt,seq \
      > "${out}/${exp}/analysis/seed_report.txt"
  fi
  tw="runs/$(exp_of "${S15}")/s0/tripwire"
  if [ -f "${tw}/DONE" ]; then
    ok python scripts/analyze.py monitor --run "${tw}" --stage 3 --shift mmlu --shares 0 0.05 0.2 0.5 1 \
      --out "${out}/$(exp_of "${S15}")/analysis"
  fi
  ok python scripts/table_style.py "${out}"   # last: shrink-only widths, short labels, arrows (paper v52)
  ok python scripts/table_figs.py "${out}"    # then: merged appendix tables and figure data (paper v53)
  echo "tables and analyses in ${out}/"
}

collect() {  # tables + analyses + status of every run, in one archive
  local out=paper/v2_results stamp f
  stamp=$(date +%Y%m%d_%H%M)
  mkdir -p "${out}"
  tables 2>&1 | tee "${out}/report.txt" || true
  bash slurm/plan.sh all status > "${out}/status.txt" 2>&1 || true
  for f in runs/*/s*/*/config.json; do
    [ -f "${f}" ] && printf '%s\t%s\t%s\n' "$(dirname "${f}")" "$(isfile "$(dirname "${f}")/DONE")" \
      "$(grep -o '"device": "[^"]*"' "${f}" | head -1 | cut -d'"' -f4)"
  done > "${out}/runs_gpu.tsv"
  cp logs/submitted.tsv "${out}/" 2>/dev/null || true
  tar czf "v2_results_${stamp}.tar.gz" "${out}"
  echo "send me this file: $(pwd)/v2_results_${stamp}.tar.gz ($(du -h "v2_results_${stamp}.tar.gz" | cut -f1))"
}

# ----------------------------------------------------------------------------------------------- helpers
exp_of() { grep -E '^exp:' "$1" | awk '{print $2}'; }
seeds() {  # the step's seeds, restricted to SEEDS when given
  local s out=""
  for s in $1; do
    if [ -z "${SEEDS_F}" ] || [[ " ${SEEDS_F} " == *" ${s} "* ]]; then out+="${s} "; fi
  done
  echo "${out}"
}
skipped() { case " ${SKIP_F} " in *" $1 "*|*" $2 "*) return 0 ;; esac; return 1; }   # run name, method
want_addon_cfg() {  # want_addon_cfg <config> <step>...: 0 if the config has run folders or one of the steps is queued
                  # (add-on plans: their features / judge steps wait only for experiments that were started)
  local cfg="$1" st
  shift
  [ -d "runs/$(exp_of "${cfg}")" ] && return 0
  for st in "$@"; do
    awk -F'|' -v id="${st}" '$1==id{f=1} END{exit !f}' "${QUEUE}" 2>/dev/null && return 0
  done
  return 1
}
say() { if [ "${MODE}" = status ]; then echo "$@"; fi; return 0; }

JOBS=""
refresh_jobs() { JOBS="$(squeue --me -h -o '%j' 2>/dev/null || true)"; }
queued() { grep -qxF "$1" <<< "${JOBS}"; }
free_slots() {   # MAX_JOBS minus your tw-* jobs (queued or running); a job feeding the queue does not count itself
  local n
  n=$(grep -c '^tw-' <<< "${JOBS}" || true)
  if [ -n "${SLURM_JOB_ID:-}" ] && [[ "${SLURM_JOB_NAME:-}" == tw-* ]]; then n=$((n - 1)); fi
  echo $((MAXJ - n))
}

declare -A GFREE=()
GFREE_DONE=0
gpu_add() {  # gpu_add <text> <sign>: add (or subtract) every gpu:<type>:<n> in the text to GFREE
  local g t n
  for g in $(grep -oE 'gpu:[A-Za-z0-9_.-]+:[0-9]+' <<< "$1" || true); do
    t="${g#gpu:}"; n="${t##*:}"; t="${t%:*}"
    GFREE[$t]=$(( ${GFREE[$t]:-0} + $2 * n ))
  done
}
gpu_free_init() {  # free GPUs per type in the partition now, minus the GPUs that pending jobs wait for
  [ "${GFREE_DONE}" = 1 ] && return 0
  GFREE_DONE=1
  local node gres used state
  while read -r node gres used state; do
    case "${state}" in idle*|mix*) gpu_add "${gres}" 1; gpu_add "${used}" -1 ;; esac
  done < <(sinfo -p "${PARTITION}" -h -N -O "NodeList:40,Gres:100,GresUsed:120,StateLong:20" 2>/dev/null || true)
  gpu_add "$(squeue -p "${PARTITION}" -t PENDING -h -o '%b' 2>/dev/null || true)" -1
}
CHOSEN=""
choose_gpu() {  # choose_gpu "<types in order of preference>" <GPUs>: CHOSEN = the first type with enough free GPUs
  local t last=""
  if [ "${GPU_F}" != auto ]; then CHOSEN="${GPU_F}"; return 0; fi
  gpu_free_init
  for t in $1; do
    last="${t}"
    if [ "${GFREE[$t]:-0}" -ge "$2" ]; then GFREE[$t]=$(( GFREE[$t] - $2 )); CHOSEN="${t}"; return 0; fi
  done
  CHOSEN="${last}"   # nothing free now: queue for the last type (the A100s: many more of them)
}
PINNED=""
pin_run() {  # pin_run <run folder> "<types>" <GPUs> [<source run>]: PINNED = the run's GPU type, chosen once;
             # a run that has started (config.json) keeps it; evaluations use their source run's type
  local f="$1/GPU_TYPE"
  if [ -s "${f}" ] && [ -f "$1/config.json" ]; then PINNED="$(cat "${f}")"; return 0; fi
  if [ -n "${4:-}" ] && [ -s "$4/GPU_TYPE" ]; then PINNED="$(cat "$4/GPU_TYPE")"
  else choose_gpu "$2" "$3"; PINNED="${CHOSEN}"; fi
  if [ "${MODE}" = submit ]; then mkdir -p "$1"; echo "${PINNED}" > "${f}"; fi
}
big() { local GPUS_NOW="${GPUS_BIG}" MEM_NOW="${MEM_BIG}"; "$@"; }   # big train ...: a 7-8B step
constraint_for() {  # constraint_for <run folder>: node feature of the GPU model the run started on (empty if new)
  local dev pair pairs
  dev=$(grep -o '"device": "[^"]*"' "$1/config.json" 2>/dev/null | head -1 | cut -d'"' -f4)
  [ -n "${dev}" ] || return 0
  IFS=';' read -ra pairs <<< "${GPU_MODEL_CONSTRAINTS}"
  for pair in "${pairs[@]}"; do
    if [ -n "${pair}" ] && [[ "${dev}" == *"${pair%%=*}"* ]]; then echo "${pair#*=}"; return 0; fi
  done
}

DEP=""
selftest_dep() {  # DEP = dependency on the selftest while it is queued; returns 1 if the selftest failed
  DEP=""
  [ -n "${ST_F}" ] || return 0
  if squeue -h -j "${ST_F}" 2>/dev/null | grep -q .; then DEP="afterok:${ST_F}"; return 0; fi
  local st
  st=$(sacct -j "${ST_F}" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)
  case "${st}" in COMPLETED|"") return 0 ;; *) echo "selftest job ${ST_F} ended ${st}: not submitting (fix it first)"; return 1 ;; esac
}

declare -i N_DONE=0 N_QUEUED=0 N_SUB=0 N_TODO=0 N_WAIT=0 N_HELD=0 N_FAIL=0
FREE=0
item() {  # item <done 0/1> <job name> <label> <folder of the FAILED file>: returns 0 when it is to be submitted now
  if [ "$1" = 1 ]; then N_DONE+=1; say "done     $3"; return 1; fi
  if queued "$2"; then N_QUEUED+=1; say "queued   $3"; return 1; fi
  if [ -f "$4/FAILED" ] && [ "$(wc -l < "$4/FAILED")" -ge 2 ]; then
    N_FAIL+=1; echo "FAILED   $3 (failed twice, see $4/FAILED and logs/; delete that file to retry)"; return 1
  fi
  case "${MODE}" in
    status) N_TODO+=1; echo "to do    $3"; return 1 ;;
    commands|dry) N_SUB+=1; return 0 ;;
  esac
  if [ "${FREE}" -le 0 ]; then N_HELD+=1; return 1; fi
  FREE=$((FREE - 1)); N_SUB+=1
  JOBS+=$'\n'"$2"
  return 0
}
isfile() { if [ -f "$1" ]; then echo 1; else echo 0; fi; }

train() {  # train <config> <seed> <time limit> <GPUs> <spec>...
  local cfg="$1" s="$2" tl="$3" ng="$4"; shift 4
  local exp spec method tag ovr dir name cons
  exp="$(exp_of "${cfg}")"
  for spec in "$@"; do
    IFS=: read -r method tag ovr <<< "${spec}"
    skipped "${method}${tag:+_${tag}}" "${method}" && continue
    dir="runs/${exp}/s${s}/${method}${tag:+_${tag}}"
    name="tw-$(basename "${cfg}" .yaml)-s${s}-${method}${tag:+_${tag}}"
    item "$(isfile "${dir}/DONE")" "${name}" "${dir}" "${dir}" || continue
    pin_run "${dir}" "${GPUS_NOW:-${GPUS_SMALL}}" "${ng}"
    cons="$(constraint_for "${dir}")"
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} METHODS=${method}${tag:+ TAG=${tag}}${ovr:+ EXTRA=\"${ovr}\"} TIME_LIMIT=${tl}" \
           "NGPU=${ng} GPU_TYPE=${PINNED}${cons:+ CONSTRAINT=${cons}} MEM=${MEM_NOW:-${MEM_SMALL}} bash slurm/submit.sh stream"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" METHODS="${method}" TAG="${tag}" EXTRA="${ovr}" TIME_LIMIT="${tl}" NGPU="${ng}" \
      GPU_TYPE="${PINNED}" CONSTRAINT="${cons}" MEM="${MEM_NOW:-${MEM_SMALL}}" DEPEND="${DEP}" BEGIN="" \
      bash slurm/submit.sh stream || true
  done
}

scale() {  # scale <config> <seed> <base> <alpha>...: adapter-scaling evaluations of a finished ungated run
  local cfg="$1" s="$2" base="$3"; shift 3
  local exp a dir src
  exp="$(exp_of "${cfg}")"
  src="runs/${exp}/s${s}/${base}"
  for a in "$@"; do
    skipped "scale" "scale" && continue
    dir="runs/${exp}/s${s}/scale_$([ "${base}" = seq ] || echo "${base}_")a$(printf '%.2f' "${a}")"
    if [ ! -f "${src}/DONE" ]; then N_WAIT+=1; say "waiting  ${dir} (needs ${src})"; continue; fi
    item "$(isfile "${dir}/DONE")" "tw-sc-$(basename "${cfg}" .yaml)-s${s}-${base}-a${a}" "${dir}" "${dir}" || continue
    pin_run "${dir}" "${GPUS_NOW:-${GPUS_SMALL}}" 1 "${src}"
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} BASE=${base} ALPHAS=${a} GPU_TYPE=${PINNED} MEM=${MEM_NOW:-${MEM_SMALL}} bash slurm/submit.sh scale"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" BASE="${base}" ALPHAS="${a}" NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${PINNED}" MEM="${MEM_NOW:-${MEM_SMALL}}" bash slurm/submit.sh scale || true
  done
}

extra() {  # extra <config> <seed> <method>...: robustness evaluations of finished runs
  local cfg="$1" s="$2"; shift 2
  local exp m d isdone
  exp="$(exp_of "${cfg}")"
  for m in "$@"; do
    skipped "${m}" "${m}" && continue
    d="runs/${exp}/s${s}/${m}"
    if [ ! -f "${d}/DONE" ]; then N_WAIT+=1; say "waiting  ${d}/extra (needs ${d})"; continue; fi
    isdone=0
    if [ -f "${d}/extra/summary.json" ] && grep -q '"sample"' "${d}/extra/summary.json"; then isdone=1; fi
    item "${isdone}" "tw-x-$(basename "${cfg}" .yaml)-s${s}-${m}" "${d}/extra" "${d}/extra" || continue
    if [ -s "${d}/GPU_TYPE" ]; then PINNED="$(cat "${d}/GPU_TYPE")"; else choose_gpu "${GPUS_NOW:-${GPUS_SMALL}}" 1; PINNED="${CHOSEN}"; fi
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} METHODS=${m} WHAT=all GPU_TYPE=${PINNED} MEM=${MEM_NOW:-${MEM_SMALL}} bash slurm/submit.sh extra"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" METHODS="${m}" WHAT=all NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${PINNED}" MEM="${MEM_NOW:-${MEM_SMALL}}" bash slurm/submit.sh extra || true
  done
}

advx() {  # advx <config> <seed> <method[:tag]>...: harmful requests written in the task's format only
          # (eval_extra.py --what advwrap) for finished runs; final and base model in the same job
  local cfg="$1" s="$2"; shift 2
  local exp spec m tag name d isdone xt="00:30:00"
  [ "${MEM_NOW:-${MEM_SMALL}}" = "${MEM_BIG}" ] && xt="01:00:00"   # time limit: 1.5B about 15 min, 7B longer
  exp="$(exp_of "${cfg}")"
  for spec in "$@"; do
    IFS=: read -r m tag <<< "${spec}"
    name="${m}${tag:+_${tag}}"
    skipped "${name}" "${m}" && continue
    d="runs/${exp}/s${s}/${name}"
    if [ ! -f "${d}/DONE" ]; then N_WAIT+=1; say "waiting  ${d}/extra/advwrap (needs ${d})"; continue; fi
    isdone=0
    if [ -f "${d}/extra/summary.json" ] && grep -q '"advwrap_' "${d}/extra/summary.json"; then isdone=1; fi
    item "${isdone}" "tw-x-$(basename "${cfg}" .yaml)-s${s}-${name}" "${d}/extra/advwrap" "${d}/extra" || continue
    if [ -s "${d}/GPU_TYPE" ]; then PINNED="$(cat "${d}/GPU_TYPE")"; else choose_gpu "${GPUS_NOW:-${GPUS_SMALL}}" 1; PINNED="${CHOSEN}"; fi
    if [ "${MODE}" = commands ]; then
      echo "CONFIG=${cfg} SEED=${s} METHODS=${m}${tag:+ TAG=${tag}} WHAT=advwrap GPU_TYPE=${PINNED} MEM=${MEM_NOW:-${MEM_SMALL}} EXTRA_TIME=${xt} bash slurm/submit.sh extra"
      continue
    fi
    CONFIG="${cfg}" SEED="${s}" METHODS="${m}" TAG="${tag}" WHAT=advwrap NGPU=1 EXTRA="" DEPEND="" BEGIN="" \
      GPU_TYPE="${PINNED}" MEM="${MEM_NOW:-${MEM_SMALL}}" EXTRA_TIME="${xt}" bash slurm/submit.sh extra || true
  done
}

ablate() {  # ablate <config>: calibrated tripwire scores of the config's overlap task, no training (one job)
  local cfg="$1" exp d
  exp="$(exp_of "${cfg}")"
  skipped "ablate" "ablate" && return 0
  d="runs/${exp}/analysis"
  item "$(isfile "${d}/score_ablation.json")" "tw-ab-$(basename "${cfg}" .yaml)" "${d}/score_ablation" "${d}" || return 0
  choose_gpu "${GPUS_NOW:-${GPUS_SMALL}}" 1; PINNED="${CHOSEN}"
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${PINNED} MEM=${MEM_NOW:-${MEM_SMALL}} bash slurm/submit.sh ablate"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${PINNED}" MEM="${MEM_NOW:-${MEM_SMALL}}" \
    bash slurm/submit.sh ablate || true
}

features() {  # features <config> [wait]: trunk features of every gate prompt of the config (one job, no training).
               # wait: only after runs/<exp>/s0/tripwire_prompt is done (its deployed probe is copied into the
               # features), on that run's GPU model, and again whenever that run finished after the features
  local cfg="$1" how="${2:-}" exp d npz src isdone cons="" ft="01:00:00"
  [ "${MEM_NOW:-${MEM_SMALL}}" = "${MEM_BIG}" ] && ft="01:30:00"   # time limit: 1.5B under 1 h, 7B kept at 1.5 h
  exp="$(exp_of "${cfg}")"
  skipped "features" "features" && return 0
  d="runs/${exp}/analysis/gate_features"
  npz="runs/${exp}/analysis/gate_features.npz"
  src="runs/${exp}/s0/tripwire_prompt"
  isdone="$(isfile "${npz}")"
  if [ "${how}" = wait ]; then
    if [ ! -f "${src}/DONE" ]; then N_WAIT+=1; say "waiting  ${d} (needs ${src})"; return 0; fi
    if [ "${isdone}" = 1 ] && [ "${src}/DONE" -nt "${npz}" ]; then isdone=0; fi
    cons="$(constraint_for "${src}")"
  fi
  item "${isdone}" "tw-gf-$(basename "${cfg}" .yaml)" "${d}" "${d}" || return 0
  pin_run "${d}" "${GPUS_NOW:-${GPUS_SMALL}}" 1 "${src}"   # the deployed tripwire's GPU model
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${PINNED}${cons:+ CONSTRAINT=${cons}} MEM=${MEM_NOW:-${MEM_SMALL}} FEATURES_TIME=${ft} bash slurm/submit.sh features"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${PINNED}" CONSTRAINT="${cons}" \
    MEM="${MEM_NOW:-${MEM_SMALL}}" FEATURES_TIME="${ft}" bash slurm/submit.sh features || true
}

features_add() {  # features_add <config>: features of the v5 pools added to the stored gate features (FEATURES_ARGS=--add;
                  # one short job, no training), on the GPU model the stored features were computed on
  local cfg="$1" exp d js dev gt cons isdone=0 ft="00:30:00"
  [ "${MEM_NOW:-${MEM_SMALL}}" = "${MEM_BIG}" ] && ft="00:45:00"   # 1.5B about 10 min, 7B about 20 min
  exp="$(exp_of "${cfg}")"
  skipped "features" "features" && return 0
  d="runs/${exp}/analysis/gate_features"
  js="runs/${exp}/analysis/gate_features.json"
  if [ ! -f "${js}" ] || [ ! -f "runs/${exp}/analysis/gate_features.npz" ]; then
    N_WAIT+=1; say "waiting  ${d} (no stored gate features of ${exp})"; return 0
  fi
  grep -q '"eval__jb_hypo"' "${js}" && isdone=1
  item "${isdone}" "tw-gf-$(basename "${cfg}" .yaml)" "${d}" "${d}" || return 0
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
constraint_of_device() {  # constraint_of_device <GPU model name>: its node feature (empty if none is needed)
  local pair pairs
  [ -n "$1" ] || return 0
  IFS=';' read -ra pairs <<< "${GPU_MODEL_CONSTRAINTS}"
  for pair in "${pairs[@]}"; do
    if [ -n "${pair}" ] && [[ "$1" == *"${pair%%=*}"* ]]; then echo "${pair#*=}"; return 0; fi
  done
}

JUDGE2_MODEL="${JUDGE2_MODEL:-models/OLMo-2-1124-7B-Instruct}"
judge2() {  # judge2 <config>: a second judge of another model family (JUDGE2_MODEL, labels in analysis/judge_olmo/) on
            # the Seq-GRPO and Tripwire-prompt runs; done unless such a run finished after its last pass
  local cfg="$1" exp d c isdone=0 args="--families seq,tripwire_prompt --pools dolly,xstest,advbench --sample 0"
  exp="$(exp_of "${cfg}")"
  c="$(basename "${cfg}" .yaml)"
  skipped "judge2" "judge2" && return 0
  [ -d "runs/${exp}" ] || return 0
  d="runs/${exp}/analysis/judge_olmo"
  # the judge takes every run folder named seq[_*] or tripwire_prompt[_*] (prefix match), so both checks do too
  if [ -f "${d}/DONE" ] && [ -z "$(find "runs/${exp}" -mindepth 3 -maxdepth 3 \( -path "runs/${exp}/s*/seq*/DONE" -o \
       -path "runs/${exp}/s*/tripwire_prompt*/DONE" \) -newer "${d}/DONE" -print -quit 2>/dev/null)" ]; then isdone=1; fi
  if [ "${isdone}" = 0 ] && grep -qE "^tw-${c}-s[0-9]+-(seq|tripwire_prompt)(_.*)?$" <<< "${JOBS}"; then
    N_WAIT+=1; say "waiting  ${d} (Seq-GRPO / Tripwire-prompt runs of ${c} are queued or running)"; return 0
  fi
  item "${isdone}" "tw-jg-olmo-${c}" "${d}" "${d}" || return 0
  choose_gpu "${GPUS_BIG}" 1; PINNED="${CHOSEN}"
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${PINNED} MEM=${MEM_BIG} JUDGE_TIME=01:00:00 JUDGE_MODEL=${JUDGE2_MODEL} JUDGE_TAG=olmo JUDGE_ARGS=\"${args}\" bash slurm/submit.sh judge"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${PINNED}" MEM="${MEM_BIG}" \
    JUDGE_TIME=01:00:00 JUDGE_MODEL="${JUDGE2_MODEL}" JUDGE_TAG=olmo JUDGE_ARGS="${args}" \
    bash slurm/submit.sh judge || true
}

judge() {  # judge <config>: LLM judge of the changed responses of the config's finished runs (one job, 7B judge);
           # waits while a training run of the config is queued or running (put the step last in the queue, so
           # that every planned run has been submitted before it). It counts as done only if no run of the config
           # finished after its last pass; a new pass judges only the new pairs (labels are cached).
  local cfg="$1" exp d c isdone=0
  exp="$(exp_of "${cfg}")"
  c="$(basename "${cfg}" .yaml)"
  skipped "judge" "judge" && return 0
  [ -d "runs/${exp}" ] || return 0
  d="runs/${exp}/analysis/judge"
  if [ -f "${d}/DONE" ] && [ -z "$(find "runs/${exp}" -mindepth 3 -maxdepth 3 -path "runs/${exp}/s*/*/DONE" \
       -newer "${d}/DONE" -print -quit 2>/dev/null)" ]; then isdone=1; fi
  if [ "${isdone}" = 0 ] && grep -qE "^tw-${c}-s[0-9]+-" <<< "${JOBS}"; then
    N_WAIT+=1; say "waiting  ${d} (training runs of ${c} are queued or running)"; return 0
  fi
  item "${isdone}" "tw-jg-${c}" "${d}" "${d}" || return 0
  choose_gpu "${GPUS_BIG}" 1; PINNED="${CHOSEN}"
  if [ "${MODE}" = commands ]; then
    echo "CONFIG=${cfg} GPU_TYPE=${PINNED} MEM=${MEM_BIG} JUDGE_TIME=${JUDGE_TIME:-03:00:00} JUDGE_MODEL=${JUDGE_MODEL:-models/Qwen2.5-7B-Instruct} bash slurm/submit.sh judge"
    return 0
  fi
  CONFIG="${cfg}" SEED=0 NGPU=1 TAG="" EXTRA="" DEPEND="" BEGIN="" GPU_TYPE="${PINNED}" MEM="${MEM_BIG}" \
    JUDGE_TIME="${JUDGE_TIME:-03:00:00}" JUDGE_MODEL="${JUDGE_MODEL:-models/Qwen2.5-7B-Instruct}" JUDGE_TAG="" JUDGE_ARGS="" \
    bash slurm/submit.sh judge || true   # 1-3 h; labels are cached, so a pass cut by the limit continues
}

desc_of() { local d; for d in "${STEPS[@]}"; do [ "${d%%|*}" = "$1" ] && { echo "${d#*|}"; return; }; done; }
expand() {  # selector -> step ids in plan order
  local tok d id found
  for tok in ${1//,/ }; do
    case "${tok}" in
      all) for d in "${STEPS[@]}"; do   # the add-on plans (slurm/plans/) are never part of "all"
             id="${d%%|*}"; [[ "${id}" == w5x.* || "${id}" == sft* || "${id}" == fam* || "${id}" == nov* || "${id}" == p3* ]] \
               || echo "${id}"; done ;;
      w6|w7|w8) echo "${tok}: no new runs (RUN_v2.md)" >&2 ;;
      *) found=0
         for d in "${STEPS[@]}"; do
           id="${d%%|*}"
           if [ "${id}" = "${tok}" ] || [[ "${id}" == "${tok}".* ]]; then echo "${id}"; found=1; fi
         done
         [ "${found}" = 1 ] || { echo "unknown step or week: ${tok} (bash slurm/plan.sh steps)" >&2; exit 1; } ;;
    esac
  done
}

run_step() {  # run_step <id>: walk the step's runs with the current options
  selftest_dep || { N_HELD+=1; return 0; }
  "st_${1//./_}"
}

queue_add() {  # put a step in the queue with this command's options (replacing an earlier entry of the step)
  local line="$1|${OPT_SEEDS}|${OPT_SKIP}|${OPT_NGPU7}|${OPT_GPU}|${OPT_ST}"
  touch "${QUEUE}"
  awk -F'|' -v id="$1" -v l="${line}" 'BEGIN{done=0} $1==id{print l; done=1; next} {print} END{if(!done) print l}' \
    "${QUEUE}" > "${QUEUE}.tmp"
  mv "${QUEUE}.tmp" "${QUEUE}"
}

feed() {  # submit waiting runs of the queued steps, in queue order, while slots are free; drop finished steps
  [ -s "${QUEUE}" ] || { echo "queue is empty"; return 0; }
  refresh_jobs
  FREE=$(free_slots)
  local lines keep=() line id before
  mapfile -t lines < "${QUEUE}"
  for line in "${lines[@]}"; do
    [ -n "${line}" ] || continue
    IFS='|' read -r id SEEDS_F SKIP_F NGPU7_F GPU_F ST_F <<< "${line}"
    before=$((N_QUEUED + N_HELD + N_WAIT + N_SUB))
    run_step "${id}"
    if [ $((N_QUEUED + N_HELD + N_WAIT + N_SUB)) -gt "${before}" ] || [ -n "${DEP}" ]; then
      keep+=("${line}")
    else
      echo "step ${id} is finished: removed from the queue"
    fi
  done
  printf '%s\n' ${keep[@]+"${keep[@]}"} | sed '/^$/d' > "${QUEUE}.tmp"
  mv "${QUEUE}.tmp" "${QUEUE}"
}

lock() {  # one plan.sh at a time changes the queue (best effort: needs flock on the file system)
  exec 9>> logs/.plan.lock
  command -v flock >/dev/null 2>&1 && flock -w 300 9 || true
}

summary() {
  case "${MODE}" in
    status) echo "done ${N_DONE}, queued or running ${N_QUEUED}, to do ${N_TODO}, waiting for a run ${N_WAIT}, failed ${N_FAIL}" ;;
    commands|dry) echo "# ${N_SUB} to submit; done ${N_DONE}, queued ${N_QUEUED}, waiting for a run ${N_WAIT}, failed ${N_FAIL}" ;;
    *) echo "submitted ${N_SUB} (at most ${MAXJ} tw-* jobs at once); waiting for a free slot ${N_HELD}," \
            "waiting for a run to finish ${N_WAIT}; done ${N_DONE}, queued ${N_QUEUED}, failed ${N_FAIL}" ;;
  esac
}

# add-on plans, kept apart from the main plan: slurm/plans/sft.sh (PLAN_SFT.md: SFT without LoRA) and
# slurm/plans/fam.sh (PLAN_SCALE.md: larger models of other families). Each adds its own steps (sft*, fam*), configs,
# run folders and tables ("bash slurm/plan.sh sft-tables", "... sft-collect", "... fam-tables", "... fam-collect")
for f in slurm/plans/*.sh; do [ -f "${f}" ] && source "${f}"; done

# ----------------------------------------------------------------------------------------------- main
if [ -n "${SEL}" ] && declare -F "addon_${SEL//-/_}" >/dev/null; then "addon_${SEL//-/_}"; exit 0; fi
case "${SEL}" in
  "") sed -n '2,29p' "$0"; exit 1 ;;
  tables) tables; exit 0 ;;
  collect) collect; exit 0 ;;
  steps)
    touch "${QUEUE}"
    for d in "${STEPS[@]}"; do
      id="${d%%|*}"
      q=""; awk -F'|' -v id="${id}" '$1==id{f=1} END{exit !f}' "${QUEUE}" && q="  [in queue]"
      printf '%-6s %s%s\n' "${id}" "${d#*|}" "${q}"
    done
    exit 0 ;;
  queue) if [ -s "${QUEUE}" ]; then cut -d'|' -f1 "${QUEUE}" | while read -r id; do echo "${id}  $(desc_of "${id}")"; done
         else echo "queue is empty"; fi; exit 0 ;;
  stop) lock; : > "${QUEUE}"; echo "queue emptied: no new jobs will be submitted. Jobs already queued keep running" \
          "(scancel them to stop: squeue --me)"; exit 0 ;;
  feed) lock; feed; summary; exit 0 ;;
esac

IDS="$(expand "${SEL}")"
if [ "${MODE}" = check ]; then
  for w in $(for id in ${IDS}; do echo "${id%%.*}"; done | uniq); do
    if declare -F "${w}_check" >/dev/null; then "${w}_check"; else echo "${w}: no selftest needed"; fi
  done
  exit 0
fi
if [ "${MODE}" = submit ]; then
  lock
  for id in ${IDS}; do queue_add "${id}"; done
  feed
else
  refresh_jobs
  for id in ${IDS}; do
    [ "${MODE}" = commands ] && echo "# ${id}: $(desc_of "${id}")" || echo "== ${id}: $(desc_of "${id}")"
    run_step "${id}"
  done
fi
summary
