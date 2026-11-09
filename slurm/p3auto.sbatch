#!/bin/bash
# The autopilot of the p3 pack jobs (slurm/p3auto.sh) as a 1-CPU job on nova, 7 days; it resubmits itself 10 minutes
# before its time limit. Submit from the repository folder, with the partitions the jobs should use, e.g.:
#
#   P_IF_TEACHER=instruction P_NQ_TEACHER=instruction sbatch slurm/p3auto.sbatch
#
# Stop it with: touch logs/p3auto/STOP   (the GPU jobs it submitted keep running)
#SBATCH --job-name=cpup3auto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/p3auto/%x-%j.out
#SBATCH --open-mode=append
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/p3auto
bash slurm/p3auto.sh &
child=$!
trap 'echo "[$(date)] time limit near: resubmitting the autopilot"; [ -f logs/p3auto/STOP ] || sbatch --export=ALL slurm/p3auto.sbatch; kill ${child} 2>/dev/null; exit 0' USR1
while kill -0 "${child}" 2>/dev/null; do wait "${child}"; done
