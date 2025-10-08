"""
Configuration for pooling probe experiments.
Consolidated config for all pooling probe types.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class PoolingConfig:
    # Data params
    data_dir: str = "/mnt/polished-lake/data/mmlu_activations"
    cache_dir: str = ""  # Empty uses default HF cache
    label_type: str = "model_correct"  # 'model_answer', 'correct_answer', 'model_correct'
    layer_idx: int = 60  # Layer to probe (0-60)
    train_fraction: float = 0.8  # Train/test split
    
    # Pooling params
    probe_type: str = "mean"  # 'mean', 'max', 'last_token', 'rolling_means', 'softmax', 'attention'
    rolling_window: int = 5  # For rolling_means probe
    softmax_temperature: float = 1.0  # For softmax probe
    
    # Position-based strategies (all probe types are now causal)
    use_position_based: bool = True  # Enable position-based training for all probe types
    filter_pos_mode: str = "relative_pos"  # 'relative_pos', 'absolute_pos_front', 'absolute_pos_back'
    bin_size: float = 0.01  # For relative_pos: fraction (0,1]; for absolute: tokens per bin (int)
    bin_idx: int = 0  # Which bin index to keep (0-based)
    stride: int = 1  # Stride for position sampling
    
    # Training params
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    num_epochs: int = 15
    optimizer_type: str = "Adam"
    
    # Output params
    output_dir: str = "results/"
    run_name: str = "pooling_experiment"
    
    # Other
    seed: int = 42
    disable_tqdm: bool = False
    device: str = "cuda"

    @classmethod
    def from_dict(cls, params: dict) -> 'PoolingConfig':
        """Create config from dictionary."""
        return cls(
            data_dir=str(params.get('data_dir', cls.data_dir)),
            cache_dir=str(params.get('cache_dir', cls.cache_dir)),
            label_type=str(params.get('label_type', cls.label_type)),
            layer_idx=int(params.get('layer_idx', cls.layer_idx)),
            train_fraction=float(params.get('train_fraction', cls.train_fraction)),
            probe_type=str(params.get('probe_type', cls.probe_type)),
            rolling_window=int(params.get('rolling_window', cls.rolling_window)),
            softmax_temperature=float(params.get('softmax_temperature', cls.softmax_temperature)),
            use_position_based=bool(params.get('use_position_based', cls.use_position_based)),
            filter_pos_mode=str(params.get('filter_pos_mode', cls.filter_pos_mode)),
            bin_size=float(params.get('bin_size', cls.bin_size)),
            bin_idx=int(params.get('bin_idx', cls.bin_idx)),
            stride=int(params.get('stride', cls.stride)),
            batch_size=int(params.get('batch_size', cls.batch_size)),
            learning_rate=float(params.get('learning_rate', cls.learning_rate)),
            weight_decay=float(params.get('weight_decay', cls.weight_decay)),
            num_epochs=int(params.get('num_epochs', cls.num_epochs)),
            optimizer_type=str(params.get('optimizer_type', cls.optimizer_type)),
            output_dir=str(params.get('output_dir', cls.output_dir)),
            run_name=str(params.get('run_name', cls.run_name)),
            seed=int(params.get('seed', cls.seed)),
            disable_tqdm=bool(params.get('disable_tqdm', cls.disable_tqdm)),
            device=str(params.get('device', cls.device)),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data_dir': self.data_dir,
            'cache_dir': self.cache_dir,
            'label_type': self.label_type,
            'layer_idx': self.layer_idx,
            'train_fraction': self.train_fraction,
            'probe_type': self.probe_type,
            'rolling_window': self.rolling_window,
            'softmax_temperature': self.softmax_temperature,
            'use_position_based': self.use_position_based,
            'filter_pos_mode': self.filter_pos_mode,
            'bin_size': self.bin_size,
            'bin_idx': self.bin_idx,
            'stride': self.stride,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'optimizer_type': self.optimizer_type,
            'output_dir': self.output_dir,
            'run_name': self.run_name,
            'seed': self.seed,
            'disable_tqdm': self.disable_tqdm,
            'device': self.device,
        }
