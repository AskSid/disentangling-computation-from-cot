"""Data loading utilities for early decoder plotting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RunData:
    """Container for all data from a single probing run."""

    model_name: str
    dataset_name: str
    results_dir: Path
    metadata_df: pd.DataFrame
    token_level_df: pd.DataFrame
    step_level_df: pd.DataFrame
    token_tables: dict

    # Cached computations
    _best_layer: Optional[int] = field(default=None, repr=False)
    _best_accuracy: Optional[float] = field(default=None, repr=False)
    _accuracy_by_layer: Optional[pd.Series] = field(default=None, repr=False)

    # Custom plots directory (if None, uses default computed path)
    _custom_plots_dir: Optional[Path] = field(default=None, repr=False)

    @property
    def best_layer(self) -> int:
        """Get best probe layer (cached)."""
        if self._best_layer is None:
            self._compute_best_layer()
        return self._best_layer

    @property
    def best_accuracy(self) -> float:
        """Get best probe accuracy (cached)."""
        if self._best_accuracy is None:
            self._compute_best_layer()
        return self._best_accuracy

    @property
    def accuracy_by_layer(self) -> pd.Series:
        """Get accuracy by layer Series (cached)."""
        if self._accuracy_by_layer is None:
            self._compute_best_layer()
        return self._accuracy_by_layer

    def _compute_best_layer(self):
        """Compute and cache best layer info."""
        self._best_layer, self._best_accuracy, self._accuracy_by_layer = find_best_probe_layer(
            self.step_level_df, self.metadata_df
        )

    @property
    def num_questions(self) -> int:
        return len(self.metadata_df)

    @property
    def num_layers(self) -> int:
        probe_layers = self.step_level_df[self.step_level_df['layer_idx'] >= 0]['layer_idx']
        return probe_layers.nunique()

    @property
    def run_name(self) -> str:
        """Derive run name from results_dir path."""
        return self.results_dir.name

    @property
    def plots_dir(self) -> Path:
        """Get plots directory for this run."""
        if self._custom_plots_dir is not None:
            return self._custom_plots_dir
        repo_root = self.results_dir.parent.parent
        return repo_root / "plots" / self.run_name

    def __repr__(self) -> str:
        return (
            f"RunData({self.model_name} on {self.dataset_name}, "
            f"{self.num_questions} questions, {self.num_layers} layers)"
        )


COLUMN_RENAMES = {
    'decoder_pred': 'probe_pred',
    'decoder_output': 'probe_output',
    'token_str': 'token_text',
    'decoder_type': 'early_decoder',
}

EARLY_DECODER_RENAMES = {
    'probe': 'attention_probe',
}


def parse_probe_output_fast(series: pd.Series) -> pd.Series:
    """Parse probe_output strings to numpy arrays using vectorized string ops."""
    stripped = series.str.strip('[]')
    valid_mask = stripped.notna() & (stripped != '')

    result = pd.Series([None] * len(series), index=series.index)

    if valid_mask.any():
        valid_strings = stripped[valid_mask]
        parsed = valid_strings.str.split(',').apply(
            lambda x: np.array([float(v.strip()) for v in x]) if x else None
        )
        result[valid_mask] = parsed

    return result


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names."""
    rename_map = {k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def standardize_early_decoder(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize early_decoder values and infer from layer_idx if missing."""
    if 'early_decoder' not in df.columns:
        if 'layer_idx' in df.columns:
            df['early_decoder'] = np.where(
                df['layer_idx'] == -1, 'forced_answer',
                np.where(df['layer_idx'] == -2, 'cot_monitor', 'attention_probe')
            )
        else:
            df['early_decoder'] = 'attention_probe'
    else:
        df['early_decoder'] = df['early_decoder'].replace(EARLY_DECODER_RENAMES)

        if 'layer_idx' in df.columns:
            df.loc[df['layer_idx'] == -1, 'early_decoder'] = 'forced_answer'
            df.loc[df['layer_idx'] == -2, 'early_decoder'] = 'cot_monitor'

    return df


def load_csv_directory(
    directory: Path,
    hash_to_idx: Optional[dict] = None,
    parse_probe_output: bool = False,
    desc: str = "Loading",
) -> tuple[pd.DataFrame, dict]:
    """Load all CSVs from a directory and concatenate."""
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = []
    tables = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if df.empty:
            continue

        df = standardize_columns(df)

        question_hash_key = None
        if 'question_idx' not in df.columns:
            if 'question_hash' in df.columns and hash_to_idx:
                df['question_idx'] = df['question_hash'].map(hash_to_idx)
                question_hash_key = str(df['question_hash'].iloc[0])
            else:
                file_stem = csv_file.stem
                if file_stem.isdigit():
                    df['question_idx'] = int(file_stem)
                elif hash_to_idx and file_stem in hash_to_idx:
                    df['question_idx'] = hash_to_idx[file_stem]
                    if 'question_hash' not in df.columns:
                        df['question_hash'] = file_stem
                    question_hash_key = file_stem
        else:
            if 'question_hash' in df.columns:
                question_hash_key = str(df['question_hash'].iloc[0])
            else:
                question_hash_key = csv_file.stem

        if question_hash_key:
            tables[question_hash_key] = df

        dfs.append(df)

    if not dfs:
        raise ValueError(f"All CSV files in {directory} were empty")

    combined = pd.concat(dfs, ignore_index=True)
    combined = standardize_early_decoder(combined)

    # Only parse attention_probe rows — forced_answer and cot_monitor have different formats
    if parse_probe_output and 'probe_output' in combined.columns:
        probe_mask = combined['early_decoder'] == 'attention_probe'
        combined['probe_output_parsed'] = None
        if probe_mask.any():
            combined.loc[probe_mask, 'probe_output_parsed'] = parse_probe_output_fast(
                combined.loc[probe_mask, 'probe_output']
            )

    return combined, tables


def find_best_probe_layer(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    level: str = "step",
) -> tuple[int, float, pd.Series]:
    """Find the probe layer with highest accuracy for predicting model's answer."""
    if 'question_hash' in metadata_df.columns:
        model_answers = (
            metadata_df.set_index('question_hash')['model_answer']
            .astype(str).str.strip().str.upper()
            .to_dict()
        )
        key_col = 'question_hash'
    else:
        model_answers = (
            metadata_df.set_index('question_idx')['model_answer']
            .astype(str).str.strip().str.upper()
            .to_dict()
        )
        key_col = 'question_idx'

    probe_df = df[df['layer_idx'] >= 0].copy()

    if probe_df.empty:
        raise ValueError("No probe layers found in data (layer_idx >= 0)")

    probe_df['target'] = probe_df[key_col].map(model_answers)
    probe_df = probe_df[probe_df['target'].notna()]

    probe_df['probe_pred_norm'] = (
        probe_df['probe_pred'].astype(str).str.strip().str.upper()
    )

    probe_df['is_correct'] = probe_df['probe_pred_norm'] == probe_df['target']
    accuracy_by_layer = probe_df.groupby('layer_idx')['is_correct'].mean()
    best_layer = int(accuracy_by_layer.idxmax())
    best_accuracy = accuracy_by_layer[best_layer]

    return best_layer, best_accuracy, accuracy_by_layer


def load_results(
    results_dir: str | Path,
    model_name: str,
    dataset_name: str,
    parse_probe_output: bool = False,
    plots_dir: str | Path | None = None,
) -> RunData:
    """Load question-, token-, and step-level data from a probing results directory."""
    results_dir = Path(results_dir)

    metadata_paths = [
        results_dir / "eval_results" / "predictions_metadata.csv",
        results_dir / "predictions_metadata.csv",
    ]
    metadata_path = next((p for p in metadata_paths if p.exists()), None)

    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find predictions_metadata.csv in {results_dir}. "
            f"Tried: {[str(p) for p in metadata_paths]}"
        )

    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(metadata_df):,} questions from {metadata_path}")

    if (results_dir / "eval_results" / "token_level").exists():
        token_level_dir = results_dir / "eval_results" / "token_level"
        step_level_dir = results_dir / "eval_results" / "step_level"
    elif (results_dir / "token_level").exists():
        token_level_dir = results_dir / "token_level"
        step_level_dir = results_dir / "step_level"
    else:
        raise FileNotFoundError(
            f"Could not find token_level directory in {results_dir}"
        )

    hash_to_idx = None
    if 'question_hash' in metadata_df.columns and 'question_idx' in metadata_df.columns:
        hash_to_idx = metadata_df.set_index('question_hash')['question_idx'].to_dict()

    print("Loading token-level data...")
    token_level_df, token_tables = load_csv_directory(
        token_level_dir,
        hash_to_idx=hash_to_idx,
        parse_probe_output=parse_probe_output,
        desc="Loading token-level",
    )
    print(f"  Loaded {len(token_level_df):,} token-level rows")

    print("Loading step-level data...")
    step_level_df, _ = load_csv_directory(
        step_level_dir,
        hash_to_idx=hash_to_idx,
        parse_probe_output=parse_probe_output,
        desc="Loading step-level",
    )
    print(f"  Loaded {len(step_level_df):,} step-level rows")

    if 'early_decoder' in step_level_df.columns:
        print(f"\nStep-level early_decoder distribution:")
        for decoder, count in step_level_df['early_decoder'].value_counts().items():
            print(f"  {decoder}: {count:,}")

    return RunData(
        model_name=model_name,
        dataset_name=dataset_name,
        results_dir=results_dir,
        metadata_df=metadata_df,
        token_level_df=token_level_df,
        step_level_df=step_level_df,
        token_tables=token_tables,
        _custom_plots_dir=Path(plots_dir) if plots_dir else None,
    )


def load_multiple_runs(
    run_configs: dict[str, dict],
    parse_probe_output: bool = False,
) -> dict[str, RunData]:
    """Load multiple runs at once."""
    runs = {}
    for run_name, config in run_configs.items():
        print(f"\n{'='*60}")
        print(f"Loading run: {run_name}")
        print(f"{'='*60}")
        runs[run_name] = load_results(
            results_dir=config['results_dir'],
            model_name=config['model_name'],
            dataset_name=config['dataset_name'],
            parse_probe_output=parse_probe_output,
        )
    return runs
