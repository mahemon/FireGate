# FireGate: Bounding Behaviour Change Before Training in Continual RL Fine-Tuning

Code, configurations and result tables for the paper *FireGate: Bounding Behaviour Change Before Training in
Continual RL Fine-Tuning* (Md Anwar Hossen and Ali Jannesari, Iowa State University).

## Overview

Fine-tuning a deployed language model on one task after another can change behaviour that should stay fixed, such
as earlier skills, general chat or refusals. FireGate places each task's update (for example a LoRA) behind a gate:
a linear classifier on the model's frozen lower layers that decides, for each prompt, whether the update is used.
Where no gate fires, the model computes exactly what the old model computes, so on any prompt distribution every
bounded behaviour score changes by at most the gate's firing rate there. Each gate is calibrated on unlabeled
prompts of what must not change, which fixes that rate before training with a PAC guarantee, whatever weights
training produces.

The update can be trained under its gate (gated training) or trained by any method and placed behind a
prompt-time gate afterwards (a certified switch).

## Requirements

- Python 3.11, PyTorch 2.6 with CUDA 12.4, and the packages in `requirements.txt` (tested with transformers 4.51 and 5.x)
- One GPU with 80 GB or more per run (the paper's runs used H200 and A100 GPUs)
- A Slurm cluster for the job scripts; single runs also work without Slurm

## Setup

1. Create a Python environment and install the requirements.
2. Copy `slurm/cluster.env.example` to `slurm/cluster.env` and set your partition, account and environment path.
3. Download the data and the base model:

   ```bash
   python scripts/prepare_data.py --data data --models models
   ```

   Other task data come from `scripts/prepare_ladder.py`, `prepare_stream5.py`, `prepare_external.py`,
   `prepare_prospective.py` and `prepare_widen.py`; each script's header says what it downloads. Gated models
   (Llama) need your own Hugging Face access.

## Quick check

```bash
python tests/test_pure.py          
bash slurm/submit.sh selftest     
```

## Running an experiment

Each run trains one method on one configuration and seed, for example Seq-GRPO and FireGate-prompt on NQ:

```bash
CONFIG=configs/v2/overlap_q15.yaml SEED=0 METHODS="seq tripwire_prompt" bash slurm/submit.sh stream
```

Runs are saved in `runs/<experiment>/s<seed>/<method>` and resume from their last checkpoint. To run the paper's
experiments as numbered steps, see `bash slurm/plan.sh steps`.

## Experiments in the paper

| Paper | Configs |
|---|---|
| Three-task stream (GSM8K, Countdown, K&K) | `configs/v2/stream_q15.yaml`, `stream_q7.yaml` |
| Five-task stream | `configs/v2/stream5_q15.yaml` |
| Overlap ladder (NQ, IF, Concise, Concise-ctx) | `configs/v2/overlap_*.yaml`, `ladder_*.yaml` |
| SQuAD 2.0 | `configs/v3/squad2_q15.yaml` |
| Other model families (Llama, DeepSeek, Mistral) | `configs/fam/` |
| Full-rank updates and SFT | `configs/sft/` |
| New task with the recipe frozen (TriviaQA) | `configs/p3/trivia_llama3b.yaml` |

## Results

`paper/latex_v2_results/` contains the result files exactly as used in the paper; `paper/` compiles with pdflatex.

## Citation

```bibtex
@misc{hossen2026firegate,
  title  = {FireGate: Bounding Behaviour Change Before Training in Continual RL Fine-Tuning},
  author = {Hossen, Md Anwar and Jannesari, Ali},
  year   = {2026}
}
```
