#!/bin/bash
#SBATCH --job-name=causal_probe_sweep
#SBATCH --output=/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/logs/causal_probe_sweep_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=1000GB
#SBATCH --time=24:00:00

# Print configuration for verification
echo "Job Configuration:"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Master Node: ${SLURMD_NODENAME}"
echo "Node List: ${SLURM_NODELIST}"

nvidia-smi

# Activate the virtual environment
source /mnt/polished-lake/home/annabelma/.cache/pypoetry/virtualenvs/disentangling-computation-from-cot-y7e-4Qh5-py3.10/bin/activate
echo "Virtual environment activated"

# Clear corrupted dataset cache
echo "Clearing corrupted dataset cache..."
rm -rf /mnt/polished-lake/home/annabelma/.cache/huggingface/datasets/edinburgh-dawg___mmlu-redux-2.0

# Run the causal probe sweep
echo "Starting causal probe sweep at $(date)"
echo "Job ID: $SLURM_JOB_ID"

python /mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/code/train_pooling_probe.py \
    --config /mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/configs/causal_probe_layer60.yaml \
    --mode sweep

echo "Causal probe sweep completed at $(date)"