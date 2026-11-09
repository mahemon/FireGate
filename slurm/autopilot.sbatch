#!/bin/bash
# The autopilot (slurm/autopilot.sh) as a 1-CPU job on the nova partition (10 days); it submits the GPU jobs to the partition
# and account of slurm/cluster.env. Submit from the repository folder:
#
#   mkdir -p logs/autopilot && sbatch slurm/autopilot.sbatch
#   sbatch --time=2-00:00:00 slurm/autopilot.sbatch        # if the partition allows less than 7 days
#
# It resubmits itself 15 minutes before its time limit and stops after 10 days (AP_DAYS), when every queued step is
# finished, or when logs/autopilot/STOP exists.
# The add-on plans run as their own autopilots, each with its own job name, queue and folder (logs/autopilot_<plan>/):
#   AP_PLAN=sft AP_TARGET=20 sbatch --job-name=cpusft slurm/autopilot.sbatch     # PLAN_SFT.md
#   AP_PLAN=fam AP_TARGET=20 sbatch --job-name=cpufam slurm/autopilot.sbatch     # PLAN_SCALE.md
#SBATCH --job-name=cpujob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/autopilot/%x-%j.out
#SBATCH --open-mode=append
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/autopilot
APD="${AP_DIR:-logs/autopilot}"   # the autopilot's folder (the add-on plans have their own)
case "${AP_PLAN:-}" in sft|fam) APD="${AP_DIR:-logs/autopilot_${AP_PLAN}}" ;; esac
mkdir -p "${APD}"
# run the autopilot; if it ever exits with an error, note it and start it again (at most 20 times); the time-limit
# signal is passed on so that it resubmits itself
n=0
while true; do
  bash slurm/autopilot.sh &
  child=$!
  trap 'kill -USR1 ${child} 2>/dev/null' USR1
  wait "${child}"; rc=$?
  while kill -0 "${child}" 2>/dev/null; do wait "${child}"; rc=$?; done   # wait was interrupted by the signal
  [ "${rc}" -eq 0 ] && break
  [ -f "${APD}/STOP" ] && break
  n=$((n + 1))
  echo "[$(date '+%F %T')] autopilot exited with code ${rc}; restarting in 60 s (restart ${n} of 20; error output in logs/autopilot/${SLURM_JOB_NAME:-cpujob}-${SLURM_JOB_ID:-x}.out)" \
    >> "${APD}/autopilot.log"
  printf '%s\t%s\t-\t-\t%s\n' "$(date '+%F %T')" CRASH "exit ${rc}, restart ${n}" >> "${APD}/events.tsv"
  [ "${n}" -ge 20 ] && break
  sleep 60
done
