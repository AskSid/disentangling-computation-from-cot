#!/bin/bash
#SBATCH --job-name=probe_training
#SBATCH --output=/mnt/polished-lake/home/sidboppana/disentangling-computation-from-cot/probing/logs/probe_training_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

# Run with: sbatch probing/run_train_probe.sh [config_file] [mode]
# Example: sbatch probing/run_train_probe.sh probing/experiments/layerwise_experiment.yaml all_layers
# Or use defaults: sbatch probing/run_train_probe.sh

# Print configuration for verification
echo "Job Configuration:"
echo "Number of Nodes: ${SLURM_NNODES}"
echo "GPUs per Node: ${SLURM_GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Master Node: ${SLURMD_NODENAME}"
echo "Node List: ${SLURM_NODELIST}"

nvidia-smi

# Activate the virtual environment
source /mnt/polished-lake/home/sidboppana/.cache/pypoetry/virtualenvs/disentangling-computation-from-cot-DeLAc2Hh-py3.10/bin/activate
echo "Virtual environment activated"

# default arguments if none provided
CONFIG_FILE=${1:-"probing/experiments/layerwise_experiment.yaml"}
MODE=${2:-"all_layers"}

echo "Using config: $CONFIG_FILE"
echo "Using mode: $MODE"

# Run the training script with arguments
python /mnt/polished-lake/home/sidboppana/disentangling-computation-from-cot/probing/train_probe.py --config "$CONFIG_FILE" --mode "$MODE"

echo "Training script completed"