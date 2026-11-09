#!/bin/bash
# Environment for every job: site settings, CUDA module, the Python env, Hugging Face cache.
# ENV_PATH may be a conda env (has conda-meta/, e.g. envs_d/hybridagent) or a venv (has bin/activate).
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -f "${REPO}/slurm/cluster.env" ]; then source "${REPO}/slurm/cluster.env"; else source "${REPO}/slurm/cluster.env.example"; fi

activate_env() {
  # environment modules: batch shells are not always login shells
  if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      if [ -f "$f" ]; then source "$f"; break; fi
    done
  fi
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" 2>/dev/null || echo "[env] WARNING: 'module load ${CUDA_MODULE}' failed"
  fi
  if [ -d "${ENV_PATH}/conda-meta" ]; then            # conda env
    if [ -f "${CONDA_SH}" ]; then
      source "${CONDA_SH}"
      conda activate "${ENV_PATH}" || echo "[env] WARNING: 'conda activate ${ENV_PATH}' failed"
    fi
    export PATH="${ENV_PATH}/bin:${PATH}"
  elif [ -f "${ENV_PATH}/bin/activate" ]; then         # venv
    source "${ENV_PATH}/bin/activate"
  else
    echo "ERROR: ${ENV_PATH} is neither a conda env nor a venv; set ENV_PATH in slurm/cluster.env" >&2
    return 1
  fi
  export HF_HOME
  mkdir -p "${HF_HOME}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  cd "${REPO}" || return 1
  echo "[env] python: $(command -v python)"
  python -c "import torch, transformers; print('[env] torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
}

# launch NGPU LAUNCHER script.py args...   one process per GPU (one node); plain python for one GPU.
# deepspeed is started without --num_gpus so that it uses exactly the GPUs in CUDA_VISIBLE_DEVICES (the ones
# SLURM gave the job); torchrun keeps CUDA_VISIBLE_DEVICES and numbers the local ranks inside it.
launch() {
  local ngpu="$1" launcher="$2"; shift 2
  if [ "${ngpu}" -le 1 ]; then
    python "$@"
    return $?
  fi
  local port=$(( 29500 + ${SLURM_JOB_ID:-$$} % 2000 ))
  if [ "${launcher}" = "torchrun" ]; then
    torchrun --standalone --nproc_per_node "${ngpu}" "$@"
  else
    deepspeed --master_port "${port}" "$@"
  fi
}

# ---- used by the job scripts (stream / scale / extra / selftest)
# guard_duplicate: two jobs of the same run must not run at once (e.g. two queue feeds racing): if another job with
# this job's name is running after 2 minutes, this one exits
guard_duplicate() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    squeue --me -h -t RUNNING -n "${SLURM_JOB_NAME}" -o '%i' 2>/dev/null | grep -vqx "${SLURM_JOB_ID}" || return 0
    sleep 10
  done
  echo "[$(date)] another job named ${SLURM_JOB_NAME} is running; this one exits"
  exit 0
}

# mark_failed <folder> <exit code>: slurm/plan.sh resubmits a failed run once, then leaves it (FAILED has 2 lines)
mark_failed() { mkdir -p "$1"; echo "$(date '+%F %T') job ${SLURM_JOB_ID:-?} exit $2" >> "$1/FAILED"; }

# plan_feed: when a job ends, submit the next waiting runs of the queued steps (slurm/plan.sh, logs/plan_queue)
plan_feed() {
  [ "${NO_FEED:-0}" = 1 ] && return 0
  env -u MAX_JOBS -u SEEDS -u SKIP -u NGPU_7B -u SELFTEST_JOB -u DRY -u CONFIG -u SEED -u METHOD -u METHODS -u TAG \
      -u EXTRA -u DEPEND -u BEGIN -u ALPHA -u ALPHAS -u BASE -u WHAT -u TIME_LIMIT -u NGPU -u GPU_TYPE -u LEVEL \
      -u NAME -u MODEL_DIR -u LAYERS -u CONSTRAINT -u JUDGE_MODEL -u JUDGE_ARGS -u JUDGE_TAG -u JUDGE_TIME \
      -u FEATURES_ARGS -u FEATURES_TIME -u EXTRA_TIME bash "${REPO}/slurm/plan.sh" feed \
    || echo "[$(date)] could not submit the next runs; run on the login node: bash slurm/plan.sh feed"
}
