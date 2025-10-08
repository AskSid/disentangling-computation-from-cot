# Pooling Probe Implementation

This implementation adds support for different pooling mechanisms for activation probes, as described in the paper "Detecting High-Stakes Interactions with Activation Probes" (https://arxiv.org/pdf/2506.10805).

## Supported Pooling Types

The implementation supports the following pooling mechanisms:

### 1. Mean Pooling
- **Description**: Averages activations across all sequence positions
- **Formula**: `1/|S| * Σ(θ^T * a_s)` for s ∈ S
- **Use case**: When you want to consider all tokens equally

### 2. Max Pooling
- **Description**: Takes the maximum score across sequence positions
- **Formula**: `max{θ^T * a_1, ..., θ^T * a_S}`
- **Use case**: When you want to focus on the most salient tokens

### 3. Last Token
- **Description**: Only uses the activation from the last sequence position
- **Formula**: `θ^T * a_S`
- **Use case**: When the final token representation is most relevant

### 4. Max of Rolling Means
- **Description**: Applies rolling mean with window size T and takes maximum
- **Formula**: `max{1/T * Σ(θ^T * a_{i+t}) for t=0 to T-1}` for i=0 to S-T
- **Use case**: When you want local averaging with global maximum selection

### 5. Softmax Pooling
- **Description**: Uses softmax-weighted sum with temperature
- **Formula**: `softmax(Aθ/φ)^T * Aθ`
- **Use case**: When you want attention-like weighting with temperature control

### 6. Attention Pooling
- **Description**: Uses attention weights from queries to weight values
- **Formula**: `softmax(Aθ_q)^T * Aθ_v`
- **Use case**: When you want learned attention patterns

## Configuration

Add these parameters to your YAML configuration:

```yaml
params:
  probe_type: "mean"  # Required: one of the above types
  rolling_window: 5   # Optional: for rolling_means (default: 5)
  softmax_temperature: 1.0  # Optional: for softmax (default: 1.0)
  # ... other existing parameters
```

## Usage Examples

### Single Probe Training
```bash
python train_probe_pool.py --config example_pooling_config.yaml --mode single
```

### Sweep Over Probe Types
```yaml
# In your config file
sweep:
  probe_type: ["mean", "max", "last_token", "softmax"]
  softmax_temperature: [0.5, 1.0, 2.0]
```
```bash
python train_probe_pool.py --config your_config.yaml --mode sweep
```

### Train All Layers
```bash
python train_probe_pool.py --config example_pooling_config.yaml --mode all_layers
```

## Key Changes from Original Implementation

1. **Sequence-level processing**: Instead of token-level classification, the probe now processes entire sequences
2. **Pooling strategies**: Multiple ways to aggregate sequence information
3. **Flexible architecture**: Easy to add new pooling mechanisms
4. **Parameter control**: Configurable window sizes and temperatures

## Performance Considerations

- **Mean/Max/Last Token**: Fastest, minimal computation
- **Rolling Means**: Moderate overhead, depends on window size
- **Softmax**: Higher overhead due to softmax computation
- **Attention**: Highest overhead, requires separate query/value projections

## Implementation Details

The `PoolingProbe` class handles all pooling mechanisms in a unified interface:

```python
model = PoolingProbe(
    input_dim=7168,
    output_dim=4,  # or 1 for binary classification
    probe_type="mean",
    rolling_window=5,
    softmax_temperature=1.0,
    dtype=torch.bfloat16
)
```

The forward pass automatically applies the appropriate pooling strategy based on the `probe_type` parameter.
