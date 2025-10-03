#!/bin/bash
#SBATCH --job-name=probe_layers
#SBATCH --output=/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/logs/array_%A_%a.out
#SBATCH --error=/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/logs/array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

set -euo pipefail
IFS=$'\n\t'

# --- Paths ---
REPO="/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing"
JOBLIST="${JOBLIST:-$REPO/experiments/layers/joblist.txt}"

mkdir -p "$REPO/logs"

[[ -f "$JOBLIST" ]] || { echo "ERROR: $JOBLIST not found"; exit 1; }

# Print some diagnostics
echo "Node: ${SLURMD_NODENAME}"
echo "Array: ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"

# Environment (match your single-job runner)
source /mnt/polished-lake/home/annabelma/.cache/pypoetry/virtualenvs/disentangling-computation-from-cot-y7e-4Qh5-py3.10/bin/activate
echo "Virtual environment activated"
nvidia-smi || true

# Which command to run
NLINES=$(wc -l < "$JOBLIST")
LIDX=${SLURM_ARRAY_TASK_ID:?need array id}
(( LIDX>=1 && LIDX<=NLINES )) || { echo "Index $LIDX out of range (1..$NLINES)"; exit 0; }

read -r CMD < <(sed -n "${LIDX}p" "$JOBLIST")
echo "[CMD] $CMD"

# Run it
set +e
eval "$CMD"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  echo "[ERROR] Command failed with rc=$RC"
  exit $RC
fi

echo "=== [FINISH] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($LIDX/$NLINES) ==="