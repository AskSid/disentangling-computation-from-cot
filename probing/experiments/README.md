# Pooling Probe Experiments

This directory contains configuration files for running pooling probe experiments.

## Available Configurations

### 1. `pooling_layerwise_experiment.yaml`
- **Purpose**: Train probes for all layers (0-60) with a single pooling type
- **Usage**: `sbatch run_train_probe_pooled.sh pooling_layerwise_experiment.yaml all_layers`
- **Default**: Uses mean pooling for all layers

### 2. `single_layer_pooling_sweep.yaml`
- **Purpose**: Comprehensive sweep over all probe types and their hyperparameters on a single layer
- **Usage**: `sbatch run_train_probe_pooled.sh single_layer_pooling_sweep.yaml sweep`
- **Sweep**: 6 probe types × 4 rolling windows × 3 temperatures = 72 combinations
- **Note**: Only rolling_means uses rolling_window, only softmax uses temperature

### 3. `rolling_means_sweep.yaml`
- **Purpose**: Focused sweep on rolling window lengths for rolling_means probe
- **Usage**: `sbatch run_train_probe_pooled.sh rolling_means_sweep.yaml sweep`
- **Sweep**: 7 different window lengths (2, 3, 5, 7, 10, 15, 20)

### 4. `softmax_temperature_sweep.yaml`
- **Purpose**: Focused sweep on temperature parameter for softmax probe
- **Usage**: `sbatch run_train_probe_pooled.sh softmax_temperature_sweep.yaml sweep`
- **Sweep**: 7 different temperatures (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)

### 5. `probe_comparison_sweep.yaml`
- **Purpose**: Balanced comparison of all probe types with reasonable hyperparameter ranges
- **Usage**: `sbatch run_train_probe_pooled.sh probe_comparison_sweep.yaml sweep`
- **Sweep**: 6 probe types × 3 rolling windows × 3 temperatures = 54 combinations
- **Note**: Uses more epochs (15) for better comparison

## Usage Examples

### Single Layer Experiments
```bash
# Test all probe types on layer 32
sbatch run_train_probe_pooled.sh single_layer_pooling_sweep.yaml sweep

# Focus on rolling window optimization
sbatch run_train_probe_pooled.sh rolling_means_sweep.yaml sweep

# Focus on temperature optimization
sbatch run_train_probe_pooled.sh softmax_temperature_sweep.yaml sweep
```

### Layer-wise Experiments
```bash
# Train mean pooling probes for all layers
sbatch run_train_probe_pooled.sh pooling_layerwise_experiment.yaml all_layers

# Train max pooling probes for all layers (modify config first)
# Edit pooling_layerwise_experiment.yaml: probe_type: "max"
sbatch run_train_probe_pooled.sh pooling_layerwise_experiment.yaml all_layers
```

### Custom Experiments
```bash
# Train single probe with custom config
python train_probe_pool.py --config example_pooling_config.yaml --mode single
```

## Expected Output

Results will be saved to `results/[run_name]/` with:
- `models/`: Trained probe weights
- `[run_name].jsonl`: Training logs and final accuracies

## Performance Expectations

Based on the paper, you should expect:
- **Mean/Max/Last Token**: Fastest training, baseline performance
- **Rolling Means**: Moderate overhead, may outperform others with optimal window size
- **Softmax**: Higher computational cost, may provide better performance
- **Attention**: Highest computational cost, potentially best performance

## Monitoring Progress

Check the SLURM logs:
```bash
tail -f /mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/logs/pooling_probe_training_*.log
```
