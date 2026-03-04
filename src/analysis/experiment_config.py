"""Configuration loader for the unified early-decoder pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
    run_name: str
    results_dir: Path = Path("results")
    seed: int = 42

    @property
    def root(self) -> Path:
        return self.results_dir / self.run_name


@dataclass
class DataConfig:
    responses_dir: Path
    activations_dir: Path
    tokenizer_model: str = "deepseek-ai/DeepSeek-R1-0528"
    forced_answers_path: Optional[Path] = None
    split_path: Optional[Path] = None
    dataset_name: Optional[str] = None


@dataclass
class ProbeConfig:
    num_layers: int  # Total number of layers in the model (required, no default)
    enabled: bool = True
    train: bool = True
    eval: bool = True
    checkpoint: Optional[Path] = None
    reuse_run_root: Optional[Path] = None  # path to an existing run with trained probes
    selected_layer: int = -1  # -1 for all layers, otherwise train just this layer (0 to num_layers-1)
    probe_type: str = "attention"
    label_type: str = "model_ans"
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 5
    mlp_hidden_dim: int = 32
    randomize_labels: bool = False
    disable_tqdm: bool = True
    normalize_acts: bool = False  # Normalize activations per-layer (standardization)
    recompute_norm_stats: bool = False  # Recompute normalization stats even if reuse_run_root is set


@dataclass
class SetupConfig:
    enabled: bool = True
    train_val_test_split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])


@dataclass
class ForcedAnswerConfig:
    enabled: bool = True
    generate: bool = False  # If True, run vLLM generation before injection
    completions_path: Optional[Path] = None
    tensor_parallel_size: int = 1
    batch_size: int = 100


@dataclass
class CotMonitorConfig:
    enabled: bool = True
    model: str = "google/gemini-2.5-flash"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 8
    temperature: float = 0.0
    per_question_concurrency: int = 8
    disable_tqdm: bool = False
    sequences_path: Optional[Path] = None  # defaults to results/<run>/cot_monitor_sequences.json
    completions_path: Optional[Path] = None  # defaults to results/<run>/cot_monitor_completions.json


@dataclass
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    setup: SetupConfig
    probe: ProbeConfig
    forced_answer: ForcedAnswerConfig
    cot_monitor: CotMonitorConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        def _section(name: str, model, defaults: Dict[str, Any] | None = None):
            defaults = defaults or {}
            values = dict(defaults)
            values.update(payload.get(name, {}))
            return values

        run_raw = _section("run", RunConfig)
        run = RunConfig(
            run_name=str(run_raw["run_name"]),
            results_dir=Path(run_raw.get("results_dir", "results")),
            seed=int(run_raw.get("seed", 42)),
        )

        data_raw = payload.get("data", {})
        data = DataConfig(
            responses_dir=Path(data_raw["responses_dir"]),
            activations_dir=Path(data_raw["activations_dir"]),
            tokenizer_model=data_raw.get("tokenizer_model", "deepseek-ai/DeepSeek-R1-0528"),
            forced_answers_path=Path(data_raw["forced_answers_path"]) if data_raw.get("forced_answers_path") else None,
            split_path=Path(data_raw["split_path"]) if data_raw.get("split_path") else None,
            dataset_name=data_raw.get("dataset_name"),
        )

        setup_raw = _section("setup", SetupConfig)
        split_ratio = setup_raw.get("train_val_test_split_ratio", [0.8, 0.1, 0.1])
        if isinstance(split_ratio, list) and len(split_ratio) == 3:
            split_ratio = [float(x) for x in split_ratio]
        else:
            raise ValueError("setup.train_val_test_split_ratio must be a list of 3 floats")
        setup = SetupConfig(
            enabled=bool(setup_raw.get("enabled", True)),
            train_val_test_split_ratio=split_ratio,
        )

        probe_raw = _section("probe", ProbeConfig)
        if "num_layers" not in probe_raw:
            raise ValueError("probe.num_layers is required")
        probe = ProbeConfig(
            enabled=bool(probe_raw.get("enabled", True)),
            train=bool(probe_raw.get("train", True)),
            eval=bool(probe_raw.get("eval", True)),
            checkpoint=Path(probe_raw["checkpoint"]) if probe_raw.get("checkpoint") else None,
            reuse_run_root=Path(probe_raw["reuse_run_root"]) if probe_raw.get("reuse_run_root") else None,
            num_layers=int(probe_raw["num_layers"]),
            selected_layer=int(probe_raw.get("selected_layer", -1)),
            probe_type=str(probe_raw.get("probe_type", "attention")),
            label_type=str(probe_raw.get("label_type", "model_ans")),
            batch_size=int(probe_raw.get("batch_size", 16)),
            learning_rate=float(probe_raw.get("learning_rate", 1e-3)),
            weight_decay=float(probe_raw.get("weight_decay", 0.01)),
            num_epochs=int(probe_raw.get("num_epochs", 5)),
            mlp_hidden_dim=int(probe_raw.get("mlp_hidden_dim", 32)),
            randomize_labels=bool(probe_raw.get("randomize_labels", False)),
            disable_tqdm=bool(probe_raw.get("disable_tqdm", True)),
            normalize_acts=bool(probe_raw.get("normalize_acts", False)),
            recompute_norm_stats=bool(probe_raw.get("recompute_norm_stats", False)),
        )

        forced_raw = payload.get("forced_answer", {})
        forced = ForcedAnswerConfig(
            enabled=forced_raw.get("enabled", True),
            generate=bool(forced_raw.get("generate", False)),
            completions_path=Path(forced_raw["completions_path"]) if forced_raw.get("completions_path") else None,
            tensor_parallel_size=int(forced_raw.get("tensor_parallel_size", 1)),
            batch_size=int(forced_raw.get("batch_size", 100)),
        )

        cot_monitor_raw = payload.get("cot_monitor", {})
        cot_monitor = CotMonitorConfig(
            enabled=cot_monitor_raw.get("enabled", True),
            model=cot_monitor_raw.get("model", "google/gemini-2.5-flash"),
            api_key_env=cot_monitor_raw.get("api_key_env", "OPENROUTER_API_KEY"),
            max_tokens=int(cot_monitor_raw.get("max_tokens", 8)),
            temperature=float(cot_monitor_raw.get("temperature", 0.0)),
            per_question_concurrency=int(cot_monitor_raw.get("per_question_concurrency", 8)),
            disable_tqdm=bool(cot_monitor_raw.get("disable_tqdm", False)),
            sequences_path=Path(cot_monitor_raw["sequences_path"]) if cot_monitor_raw.get("sequences_path") else None,
            completions_path=Path(cot_monitor_raw["completions_path"]) if cot_monitor_raw.get("completions_path") else None,
        )

        return cls(run=run, data=data, setup=setup, probe=probe, forced_answer=forced, cot_monitor=cot_monitor)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def resolved_paths(self) -> Dict[str, Path]:
        """Return canonical paths for the run to keep downstream scripts consistent."""
        root = self.run.root.resolve()
        paths = {
            "root": root,
            "metadata": (root / "predictions_metadata.csv").resolve(),
            "token_level": (root / "token_level").resolve(),
            "step_level": (root / "step_level").resolve(),
            "models": (root / "models").resolve(),
            "cot_monitor_sequences": (self.cot_monitor.sequences_path or root / "cot_monitor_sequences.json").resolve(),
            "cot_monitor_completions": (self.cot_monitor.completions_path or root / "cot_monitor_completions.json").resolve(),
            "normalization_stats": (root / "normalization_stats.json").resolve(),
        }
        return paths
