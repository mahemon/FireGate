# FireGate: Bounding Behaviour Change Before Training in Continual RL Fine-Tuning

**Paper:** [arXiv:2611.11979](https://arxiv.org/abs/2611.11979)

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

## Results

`paper/latex_v2_results/` contains the result files exactly as used in the paper; `paper/` compiles with pdflatex.

## Citation

```bibtex
@misc{hossen2026firegate,
  title         = {FireGate: Bounding Behaviour Change Before Training in Continual RL Fine-Tuning},
  author        = {Hossen, Md Anwar and Jannesari, Ali},
  year          = {2026},
  eprint        = {2611.11979},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.11979}
}
```
