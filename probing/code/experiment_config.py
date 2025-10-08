from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Data params
    data_dir: str                 # Root path containing stage1_responses/ and stage2_activations/
    cache_dir: str                # HF datasets cache directory; empty uses default (~/.cache/huggingface/datasets)
    question_split_path: str      # Optional path to JSON split for split_mode='question' ('' to disable)
    label_type: str              # One of: 'model_ans', 'correct_ans', 'model_correct', 'absolute_pos_front', 'absolute_pos_back', 'relative_position'
    layer_idx: int               # Layer index whose activations to load (0-based)
    split_mode: str              # 'token' (random tokens) or 'question' (by question_id)
    filter_pos_mode: str         # 'none' | 'relative_pos' | 'absolute_pos_front' | 'absolute_pos_back'
    bin_size: float              # For relative_pos: fraction (0,1]; for absolute: tokens per bin (int)
    bin_idx: int                 # Which bin index to keep (0-based)
    last_n_tokens: int           # Last n tokens' activations to include in a single sample (1 for single token)
    train_fraction: float        # Fraction of data in train split (0-1)
    randomize_labels: bool       # If true, shuffle labels to create a random baseline

    # Model params
    probe_type: str              # 'linear' or 'mlp'
    mlp_hidden_dim: int          # Hidden size if probe_type == 'mlp'
    r1_model_type: str           # 'full' or 'distilled'

    # Training params
    batch_size: int              # Global batch size per step
    learning_rate: float         # Optimizer learning rate
    weight_decay: float          # L2 weight decay
    num_epochs: int              # Number of training epochs
    optimizer_type: str          # 'Adam' or 'SGD'

    # Output params
    output_dir: str              # Directory for results.jsonl and models/
    run_name: str                # Run identifier; used to name saved model files

    # Other
    seed: int                    # Random seed for reproducibility
    disable_tqdm: bool           # If true, disables progress bars
    device: str                  # 'cuda' or 'cpu'


def _build_config_from_params(params: dict) -> ExperimentConfig:
    """Construct ExperimentConfig from dict params."""
    return ExperimentConfig(
        data_dir=str(params.get('data_dir', '/mnt/polished-lake/data/mmlu_activations')),
        layer_idx=int(params.get('layer_idx', 0)),
        label_type=str(params.get('label_type', 'model_ans')),
        train_fraction=float(params.get('train_fraction', 0.8)),
        randomize_labels=bool(params.get('randomize_labels', False)),
        seed=int(params.get('seed', 42)),
        split_mode=str(params.get('split_mode', 'token')),
        filter_pos_mode=str(params.get('filter_pos_mode', 'none')),
        bin_size=float(params.get('bin_size', 0.05)),
        bin_idx=params.get('bin_idx', 0),
        disable_tqdm=bool(params.get('disable_tqdm', False)),
        cache_dir=params.get('cache_dir', ''),
        question_split_path=str(params.get('question_split_path', '')),
        batch_size=int(params.get('batch_size', 512)),
        learning_rate=float(params.get('learning_rate', 1e-3)),
        weight_decay=float(params.get('weight_decay', 0.0)),
        num_epochs=int(params.get('num_epochs', 10)),
        optimizer_type=str(params.get('optimizer_type', 'Adam')),
        device=str(params.get('device', 'cuda')),
        probe_type=str(params.get('probe_type', 'linear')),
        mlp_hidden_dim=int(params.get('mlp_hidden_dim', 512)),
        r1_model_type=str(params.get('r1_model_type', 'full')),
        output_dir=str(params.get('output_dir', 'results/')),
        run_name=str(params.get('run_name', 'test')),
        last_n_tokens=int(params.get('last_n_tokens', 1)),
    )