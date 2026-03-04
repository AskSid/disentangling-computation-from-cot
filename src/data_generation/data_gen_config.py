"""Configuration for data generation pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class DataGenerationConfig:
    """Configuration for the data generation pipeline."""

    model_id: str
    dataset_name: str
    output_dir: Path

    # vLLM settings
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    tensor_parallel_size: int = 1

    # Activation settings
    layers_to_extract: Union[List[int], str] = "all"
    dtype: str = "bfloat16"
    num_layers: int = 28

    limit: Optional[int] = None
    existing_responses_dir: Optional[Path] = None
    system_prompt: str = "The assistant is DeepSeek-R1, created by DeepSeek."
    _activations_dir_override: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.existing_responses_dir, str):
            self.existing_responses_dir = Path(self.existing_responses_dir)

    def get_layers_list(self) -> List[int]:
        if isinstance(self.layers_to_extract, str) and self.layers_to_extract == "all":
            return list(range(self.num_layers))
        return self.layers_to_extract

    @property
    def responses_dir(self) -> Path:
        return self.output_dir / "stage1_responses"

    @property
    def activations_dir(self) -> Path:
        if self._activations_dir_override is not None:
            return self._activations_dir_override
        return self.output_dir / "stage2_activations"

