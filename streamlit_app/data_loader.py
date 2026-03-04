"""Data loading utilities for the Streamlit probe viewer."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from typing import Literal

import fsspec
import pandas as pd

from constants import (
    CHOICE_LABELS,
    build_default_class_labels,
    cycle_palette,
    label_for_choice,
)

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parent  # repository root

DATASETS = [
    {
        "key": "r1_mmlu",
        "label": "MMLU-Redux (DeepSeek-R1)",
        "local_root": REPO_ROOT / "results" / "canonical_runs" / "r1_mmlu",
        "remote_roots": [],
        "remote_subdir": "results/canonical_runs/r1_mmlu",

        "defaults": {
            "probe_layer": 34,
            "decoder": "probe_34",
        },
    },
    {
        "key": "r1_gpqa",
        "label": "GPQA-Diamond (DeepSeek-R1)",
        "local_root": REPO_ROOT / "results" / "canonical_runs" / "r1_gpqa",
        "remote_roots": [],
        "remote_subdir": "results/canonical_runs/r1_gpqa",

        "defaults": {
            "probe_layer": 30,
            "decoder": "probe_30",
        },
    },
    {
        "key": "gpt_oss_mmlu",
        "label": "MMLU-Redux (GPT-OSS)",
        "local_root": REPO_ROOT / "results" / "canonical_runs" / "gpt_oss_mmlu",
        "remote_roots": [],
        "remote_subdir": "results/canonical_runs/gpt_oss_mmlu",

        "defaults": {
            "probe_layer": 17,
            "decoder": "probe_17",
        },
    },
    {
        "key": "gpt_oss_gpqa",
        "label": "GPQA-Diamond (GPT-OSS)",
        "local_root": REPO_ROOT / "results" / "canonical_runs" / "gpt_oss_gpqa",
        "remote_roots": [],
        "remote_subdir": "results/canonical_runs/gpt_oss_gpqa",

        "defaults": {
            "probe_layer": 12,
            "decoder": "probe_12",
        },
    },
]

def _get_all_datasets() -> List[Dict[str, Any]]:
    """Return dataset configs. If RESULTS_DIR is set, only show the local dataset."""
    results_dir = os.getenv("RESULTS_DIR")
    if results_dir:
        p = Path(results_dir).resolve()
        return [{
            "key": f"local_{p.name}",
            "label": p.name,
            "local_root": p,
            "remote_roots": [],
            "remote_subdir": "",
            "defaults": {},
        }]
    return list(DATASETS)


def list_dataset_options() -> List[Dict[str, str]]:
    """Return display options for the dataset selector."""
    return [{"label": ds["label"], "value": ds["key"]} for ds in _get_all_datasets()]


def get_dataset_config(key: str) -> Dict[str, Any]:
    for ds in _get_all_datasets():
        if ds.get("key") == key:
            return ds
    raise KeyError(f"Unknown dataset key: {key}")


def normalize_remote_roots(
    dataset: Dict[str, Any], r2_config: Optional[Dict[str, str]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """Build candidate remote roots and storage options."""
    r2_config = r2_config or {}
    bucket = r2_config.get("bucket") or os.getenv("R2_BUCKET")
    endpoint = r2_config.get("endpoint") or os.getenv("R2_ENDPOINT")
    access_key = r2_config.get("access_key") or os.getenv("R2_ACCESS_KEY")
    secret_key = r2_config.get("secret_key") or os.getenv("R2_SECRET_KEY")
    root_prefix = r2_config.get("root_prefix") or os.getenv("R2_ROOT_PREFIX") or ""

    storage_options: Dict[str, Any] = {}
    if bucket and endpoint and access_key and secret_key:
        storage_options = {
            "key": access_key,
            "secret": secret_key,
            "client_kwargs": {"endpoint_url": endpoint},
        }

    remote_roots: List[str] = []
    configured = dataset.get("remote_roots") or []
    for root in configured:
        if isinstance(root, str) and root.rstrip("/"):
            remote_roots.append(root.rstrip("/"))

    remote_subdir = dataset.get("remote_subdir") or dataset.get("key") or ""
    candidate_subdirs: List[str] = []
    if remote_subdir:
        candidate_subdirs.append(remote_subdir)
        base_name = Path(remote_subdir).name
        if base_name and base_name not in candidate_subdirs:
            candidate_subdirs.append(base_name)
    if dataset.get("local_root"):
        local_base = Path(dataset["local_root"]).name
        if local_base and local_base not in candidate_subdirs:
            candidate_subdirs.append(local_base)

    def _build_s3_uri(prefix: str, subdir: str) -> str:
        prefix_clean = prefix.strip("/")
        subdir_clean = subdir.strip("/")
        parts = [p for p in (prefix_clean, subdir_clean) if p]
        path = "/".join(parts)
        return f"s3://{bucket}/{path}" if path else f"s3://{bucket}"

    if bucket and candidate_subdirs:
        for subdir in candidate_subdirs:
            remote_roots.append(_build_s3_uri(root_prefix, subdir))
            if root_prefix.strip("/") and root_prefix.strip("/") != root_prefix:
                remote_roots.append(_build_s3_uri(root_prefix.strip("/"), subdir))

    seen: set[str] = set()
    unique_roots: List[str] = []
    for root in remote_roots:
        normalized = root.rstrip("/")
        if normalized and normalized not in seen:
            unique_roots.append(normalized)
            seen.add(normalized)

    return unique_roots, storage_options

ViewMode = Literal["token", "sentence"]


_ANSWERS = {"A", "B", "C", "D"}
_ANSWER_PATTERN = re.compile(
    r'(?:'
    r'\\boxed\{([ABCD])\}'
    r'|\*\*Answer:?\*\*\s*:?\s*([ABCD])'
    r'|\bAnswer\s*:\s*([ABCD])\b'
    r'|"answer"\s*:\s*"([ABCD])"'
    r'|final answer[^ABCD]*([ABCD])'
    r')',
    re.IGNORECASE,
)

_ASSISTANT_MARKERS = {
    "<｜assistant｜>",
    "<|assistant|>",
    "<assistant>",
}

# Column rename maps for normalizing old column names to the canonical names
# used internally by the app.
_TOKEN_RENAMES = {
    "decoder_pred": "probe_pred",
    "decoder_output": "probe_output",
    "decoder_type": "early_decoder",
    "token_str": "token_text",
    "step_idx": "sentence_idx",
}
_STEP_RENAMES = {
    "decoder_pred": "probe_pred",
    "decoder_output": "probe_output",
    "decoder_type": "early_decoder",
    "step_text": "sentence_text",
    "step_idx": "sentence_idx",
}


def normalize_choice_label(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        pass

    if isinstance(value, str):
        cleaned = value.strip().strip("()[]{}<>.:;,-\"'` ").upper()
        if cleaned in _ANSWERS:
            return cleaned
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        int_val = int(value)
        if 0 <= int_val < len(CHOICE_LABELS):
            return CHOICE_LABELS[int_val]
    return None


def derive_predicted_answer(row: pd.Series) -> str:
    for key in ("predicted_answer", "parsed_answer"):
        letter = normalize_choice_label(row.get(key))
        if letter:
            return letter

    cot = row.get("full_cot")
    if isinstance(cot, str) and cot.strip():
        match = _ANSWER_PATTERN.search(cot)
        if match:
            letter = normalize_choice_label(next(g for g in match.groups() if g))
            if letter:
                return letter
    return "N/A"


@dataclass(frozen=True)
class QuestionRecord:
    """Container for per-question metadata."""

    question_idx: int
    question: str
    question_hash: Optional[str]
    formatted_question: Optional[str]
    answer_choices: List[str]
    full_prompt: str
    correct_answer: str
    full_cot: str
    predicted_answer: str
    category: str

    def enumerated_choices(self) -> List[Tuple[str, str]]:
        return [(label_for_choice(idx), choice) for idx, choice in enumerate(self.answer_choices)]

    def display_question(self) -> str:
        if isinstance(self.formatted_question, str) and self.formatted_question.strip():
            return self.formatted_question
        return self.question


@dataclass(frozen=True)
class HeatmapPayload:
    pivot: pd.DataFrame
    x_labels: List[str]
    layer_labels: List[str]
    customdata: List[List[List[str]]]
    class_labels: Dict[int, str]
    probe_name: str
    view_mode: ViewMode
    x_axis_title: str
    hover_template: str
    colorscale: List[List[float | str]]
    zmin: float
    zmax: float
    colorbar: Dict[str, Any]
    prompt_boundary_label: Optional[str] = None


@dataclass(frozen=True)
class DecoderStepPayload:
    """Per-step probabilities for a single decoder."""

    step_indices: List[int]
    probabilities: List[List[float]]
    class_labels: Dict[int, str]
    x_labels: List[str]
    tick_labels: List[str]


class ProbeDataRepository:
    """Lazy-loading repository for per-question probe outputs."""

    def __init__(
        self,
        dataset: Dict[str, Any],
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.storage_options: Dict[str, Any] = storage_options or {}
        self.dataset = dataset

        base_root = dataset.get("local_root")
        self.local_root: Optional[Path] = Path(base_root) if base_root else None
        self.remote_roots: List[str] = []
        for root in dataset.get("remote_roots", []):
            if isinstance(root, Path):
                root = root.as_posix()
            if isinstance(root, str) and root.rstrip("/"):
                self.remote_roots.append(root.rstrip("/"))

        self.predictions_path, self.token_dir, self.sentence_dir = self._resolve_paths()

        self.predictions_df = self._load_predictions(self.predictions_path)
        self.hash_to_idx: Dict[str, int] = self._build_hash_to_idx()
        self.questions: Dict[int, QuestionRecord] = self._build_questions_index()
        self.question_file_map: Dict[int, Path | str] = self._build_question_file_map()
        self.available_question_indices: List[int] = self._filter_questions_with_step_level()
        self.available_probes: List[str] = self._discover_probes()
        self.inflection_data: Optional[Dict[str, Any]] = self._load_inflection_data()

    # ---- Path resolution ----

    def _resolve_paths(self) -> Tuple[Path | str, Path | str, Optional[Path | str]]:
        predictions = self._first_existing(self._build_candidates("predictions"))
        token_dir = self._first_existing(self._build_candidates("token"))
        sentence_dir = self._first_existing(self._build_candidates("step"), optional=True)
        return predictions, token_dir, sentence_dir

    def _build_candidates(self, kind: str) -> List[Path | str]:
        suffixes: Dict[str, List[str]] = {
            "predictions": [
                "eval_results/predictions_metadata.csv",
                "predictions_metadata.csv",
                "overall/predictions_metadata.csv",
            ],
            "token": [
                "eval_results/token_level",
                "token_level",
                "eval_results/question_level",
                "question_level",
            ],
            "step": [
                "eval_results/step_level",
                "step_level",
                "eval_results/sentence_level",
                "sentence_level",
            ],
        }
        roots: List[Path | str] = []
        if self.local_root:
            roots.append(self.local_root)
        roots.extend(self.remote_roots)

        candidates: List[Path | str] = []
        for root in roots:
            for suffix in suffixes.get(kind, []):
                candidates.append(self._join_path(root, suffix))
        return candidates

    def _first_existing(
        self, candidates: Sequence[Path | str], *, optional: bool = False
    ) -> Optional[Path | str]:
        for candidate in candidates:
            if candidate is not None and self._path_exists(candidate):
                return candidate
        if optional:
            return None
        raise FileNotFoundError(
            "Could not locate required data. Please check that the dataset is available."
        )

    # ---- Filesystem helpers ----

    @staticmethod
    def _is_remote_path(path: Optional[Path | str]) -> bool:
        return isinstance(path, str) and "://" in path

    def _path_exists(self, path: Path | str) -> bool:
        if self._is_remote_path(path):
            fs, inner = fsspec.core.url_to_fs(path, **self.storage_options)
            return fs.exists(inner)
        return Path(path).exists()

    def _list_csv_files(self, base: Optional[Path | str]) -> Dict[str, str | Path]:
        if base is None:
            return {}
        if self._is_remote_path(base):
            fs, inner = fsspec.core.url_to_fs(base, **self.storage_options)
            pattern = inner.rstrip("/") + "/*.csv"
            files = fs.glob(pattern)
            return {Path(name).name: f"s3://{name}" for name in files}
        path_base = Path(base)
        return {path.name: path for path in path_base.glob("*.csv")}

    def _join_path(self, base: Path | str, name: str) -> Path | str:
        if self._is_remote_path(base):
            return f"{str(base).rstrip('/')}/{name}"
        return Path(base) / name

    def _read_csv(self, path: Path | str) -> pd.DataFrame:
        if self._is_remote_path(path):
            return pd.read_csv(path, storage_options=self.storage_options)
        return pd.read_csv(Path(path))

    # ---- Predictions / metadata loading ----

    @staticmethod
    def _parse_answer_choices(raw_value: Any) -> List[str]:
        if isinstance(raw_value, list):
            return raw_value
        if isinstance(raw_value, str):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(raw_value)
                if isinstance(parsed, list):
                    return list(parsed)
        raise ValueError(f"Unexpected answer_choices format: {raw_value!r}")

    def _load_predictions(self, path: Path | str) -> pd.DataFrame:
        df = self._read_csv(path)
        required = {
            "question_idx",
            "question",
            "answer_choices",
            "full_prompt",
            "correct_answer",
            "full_cot",
            "predicted_answer",
            "category",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Predictions CSV missing expected columns: {sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        if "question_hash" in df.columns:
            df["question_hash"] = df["question_hash"].astype(str)
        df["answer_choices"] = df["answer_choices"].apply(self._parse_answer_choices)
        df["correct_answer"] = df["correct_answer"].astype(str)
        df["category"] = df["category"].astype(str)
        df["predicted_answer"] = df.apply(derive_predicted_answer, axis=1)
        return df

    def _build_hash_to_idx(self) -> Dict[str, int]:
        if "question_hash" not in self.predictions_df.columns:
            return {}
        return (
            self.predictions_df.set_index("question_hash")["question_idx"]
            .astype(int)
            .to_dict()
        )

    def _build_questions_index(self) -> Dict[int, QuestionRecord]:
        records: Dict[int, QuestionRecord] = {}
        for row in self.predictions_df.itertuples(index=False):
            records[int(row.question_idx)] = QuestionRecord(
                question_idx=int(row.question_idx),
                question=str(row.question),
                question_hash=str(row.question_hash)
                if hasattr(row, "question_hash") and pd.notna(row.question_hash)
                else None,
                formatted_question=str(row.formatted_question)
                if hasattr(row, "formatted_question") and pd.notna(row.formatted_question)
                else None,
                answer_choices=list(row.answer_choices),
                full_prompt=str(row.full_prompt),
                correct_answer=str(row.correct_answer),
                full_cot=str(row.full_cot),
                predicted_answer=str(row.predicted_answer),
                category=str(row.category),
            )
        return records

    def _build_question_file_map(self) -> Dict[int, Path | str]:
        mapping: Dict[int, Path | str] = {}
        if self.token_dir is None:
            return mapping

        available_files = self._list_csv_files(self.token_dir)

        for record in self.questions.values():
            candidates: List[str] = []
            if record.question_hash:
                candidates.append(f"{record.question_hash}.csv")
            candidates.append(f"question_{record.question_idx}.csv")

            for candidate in candidates:
                if candidate in available_files:
                    mapping[record.question_idx] = available_files[candidate]
                    break

        return mapping

    def _filter_questions_with_step_level(self) -> List[int]:
        """Return question indices that have step_level files available."""
        if self.sentence_dir is None:
            return sorted(self.questions.keys())

        available_files = self._list_csv_files(self.sentence_dir)
        if not available_files:
            return sorted(self.questions.keys())

        result: set[int] = set()
        for record in self.questions.values():
            for name in (f"{record.question_hash}.csv", f"question_{record.question_idx}.csv"):
                if name in available_files:
                    result.add(record.question_idx)
                    break

        return sorted(result) if result else sorted(self.questions.keys())

    # ---- Public query methods ----

    def list_question_options(self) -> List[Dict[str, str | int]]:
        options: List[Dict[str, str | int]] = []
        for idx in self.available_question_indices:
            record = self.questions[idx]
            preview = record.display_question().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            options.append({"label": f"#{idx} — {preview}", "value": idx})
        return options

    def list_probe_options(self) -> List[Dict[str, str]]:
        return [{"label": probe, "value": probe} for probe in self.available_probes]

    def list_probe_layers(self, question_idx: int, probe_name: Optional[str] = None) -> List[int]:
        try:
            df = self.load_sentence_df(question_idx)
        except FileNotFoundError:
            return []
        subset = df[df["layer_idx"] >= 0]
        if subset.empty:
            return []
        return sorted(subset["layer_idx"].astype(int).unique())

    def list_decoder_options(self, question_idx: int) -> List[Dict[str, str]]:
        """Return available decoders (forced, CoT monitor, probe layers) for a question."""
        try:
            df = self.load_sentence_df(question_idx)
        except FileNotFoundError:
            return []

        options: List[Dict[str, str]] = []
        if (df["layer_idx"] == -1).any():
            options.append({"label": "Forced Answer", "value": "forced_answer"})
        if (df["layer_idx"] == -2).any():
            options.append({"label": "CoT Monitor LLM", "value": "cot_monitor_llm"})

        for layer in self.list_probe_layers(question_idx):
            options.append(
                {"label": f"Probe Layer {layer}", "value": f"probe_{layer}"}
            )
        return options

    def list_view_modes(self) -> List[Dict[str, str]]:
        modes = [
            {"label": "Token-Level", "value": "token"},
            {"label": "Step-Level", "value": "sentence"},
            {"label": "Individual Decoder View", "value": "decoder"},
        ]
        if self.has_inflection_data():
            modes.append({"label": "Inflection Points", "value": "inflection"})
        return modes

    def get_question(self, question_idx: int) -> QuestionRecord:
        if question_idx not in self.questions:
            raise KeyError(f"Unknown question_idx: {question_idx}")
        return self.questions[question_idx]

    # ---- Probe discovery ----

    def _discover_probes(self) -> List[str]:
        probes: set[str] = set()
        for idx in self.available_question_indices:
            try:
                token_df = self._load_token_df(idx)
                probes.update(token_df["early_decoder"].astype(str).unique())
            except (FileNotFoundError, ValueError):
                pass
            try:
                step_df = self.load_sentence_df(idx)
                probes.update(step_df["early_decoder"].astype(str).unique())
            except (FileNotFoundError, ValueError):
                pass
            if probes:
                break
        return sorted(probes)

    # ---- Inflection data ----

    def _load_inflection_data(self) -> Optional[Dict[str, Any]]:
        """Load inflection results if available, checking local and remote roots."""
        candidates = []
        if self.local_root:
            candidates.append(self.local_root / "inflection_results.json")
        for root in self.remote_roots:
            candidates.append(self._join_path(root, "inflection_results.json"))

        inflection_path = self._first_existing(candidates, optional=True)
        if inflection_path is None:
            return None
        try:
            raw_text = self._read_text(inflection_path)
            data = json.loads(raw_text)
            # Build hash-keyed lookup
            lookup: Dict[str, Dict] = {}
            for result in data.get("results", []):
                qhash = result.get("question_hash")
                if qhash:
                    lookup[qhash] = result
                else:
                    # Old format with question_idx
                    q_idx = result.get("question_idx")
                    if q_idx is not None:
                        for h, idx in self.hash_to_idx.items():
                            if idx == int(q_idx):
                                lookup[h] = result
                                break
            return lookup
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load inflection data: {exc} (candidates: {candidates})")
            return None

    def _read_text(self, path: "Path | str") -> str:
        """Read a text file from local or remote path."""
        if self._is_remote_path(path):
            fs_obj, inner = fsspec.core.url_to_fs(path, **self.storage_options)
            with fs_obj.open(inner, "r", encoding="utf-8") as f:
                return f.read()
        with Path(path).open("r", encoding="utf-8") as f:
            return f.read()

    def has_inflection_data(self) -> bool:
        return self.inflection_data is not None and len(self.inflection_data) > 0

    def get_inflection_info(self, question_idx: int) -> Optional[Dict]:
        """Get inflection data for a question by index."""
        if not self.inflection_data:
            return None
        record = self.questions.get(question_idx)
        if record is None or record.question_hash is None:
            return None
        return self.inflection_data.get(record.question_hash)

    # ---- DataFrame normalization ----

    @staticmethod
    def _apply_renames(df: pd.DataFrame, renames: Dict[str, str]) -> pd.DataFrame:
        """Apply column renames, only for columns that exist and whose target doesn't."""
        actual = {}
        for old, new in renames.items():
            if old in df.columns and new not in df.columns:
                actual[old] = new
        if actual:
            df.rename(columns=actual, inplace=True)
        return df

    def _normalize_common(self, df: pd.DataFrame, question_idx: int) -> pd.DataFrame:
        """Shared normalization logic for both token and step DataFrames."""
        if "question_idx" not in df.columns and "question_hash" in df.columns:
            df["question_idx"] = df["question_hash"].map(self.hash_to_idx)
        if "question_idx" not in df.columns:
            df["question_idx"] = question_idx
        df["question_idx"] = df["question_idx"].fillna(question_idx).astype(int)

        if "probe_ans" not in df.columns and "probe_pred" in df.columns:
            df["probe_ans"] = df["probe_pred"]

        if "early_decoder" not in df.columns:
            if "layer_idx" in df.columns:
                df["early_decoder"] = df["layer_idx"].apply(
                    lambda l: "forced_answer" if l == -1 else ("cot_monitor" if l == -2 else "attention_probe")
                )
            else:
                df["early_decoder"] = "attention_probe"
        df["early_decoder"] = df["early_decoder"].astype(str).replace(
            {"probe": "attention_probe"}
        )

        if "probe_output" in df.columns:
            df["probe_output"] = df["probe_output"].apply(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else v
            )

        for col, default in [("layer_idx", -1), ("sentence_idx", 0)]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

        return df

    @staticmethod
    def _infer_sentence_indices(df: pd.DataFrame) -> Dict[int, int]:
        unique_tokens = (
            df[["token_idx", "token_text"]]
            .drop_duplicates(subset=["token_idx"], keep="first")
            .sort_values("token_idx")
        )

        mapping: Dict[int, int] = {}
        current_idx = 0
        consecutive_newlines = 0

        for token_idx, token_text in unique_tokens.itertuples(index=False):
            text = str(token_text) if pd.notna(token_text) else ""
            mapping[int(token_idx)] = current_idx
            for ch in text:
                if ch == "\n":
                    consecutive_newlines += 1
                    if consecutive_newlines == 2:
                        current_idx += 1
                        consecutive_newlines = 0
                elif ch != "\r":
                    consecutive_newlines = 0

        return mapping

    def _normalize_token_df(self, df: pd.DataFrame, question_idx: int) -> pd.DataFrame:
        df = df.copy()
        self._apply_renames(df, _TOKEN_RENAMES)

        if "token_text" not in df.columns:
            df["token_text"] = ""

        missing_sentence_idx = "sentence_idx" not in df.columns or df["sentence_idx"].isna().all()

        df = self._normalize_common(df, question_idx)

        df["token_idx"] = pd.to_numeric(
            df.get("token_idx", pd.Series([pd.NA] * len(df))),
            errors="coerce",
        ).fillna(-1).astype(int)
        df["token_text"] = df["token_text"].fillna("").astype(str)

        if missing_sentence_idx:
            inferred = self._infer_sentence_indices(df)
            df["sentence_idx"] = df["token_idx"].map(inferred).fillna(0).astype(int)

        if "probe_ans" not in df.columns:
            raise ValueError(
                "Token CSV missing probe prediction column; expected 'probe_ans' or 'probe_pred'."
            )
        return df

    def _normalize_step_df(self, df: pd.DataFrame, question_idx: int) -> pd.DataFrame:
        df = df.copy()
        self._apply_renames(df, _STEP_RENAMES)

        df = self._normalize_common(df, question_idx)

        if "sentence_text" in df.columns:
            df["sentence_text"] = df["sentence_text"].fillna("").astype(str)
        else:
            df["sentence_text"] = ""

        if "probe_ans" not in df.columns:
            raise ValueError(
                "Step CSV missing probe prediction column; expected 'probe_ans' or 'probe_pred'."
            )
        return df

    # ---- Per-question CSV loading ----

    @lru_cache(maxsize=32)
    def _load_token_df(self, question_idx: int) -> pd.DataFrame:
        if self.token_dir is None:
            raise FileNotFoundError("Token-level data directory is not configured.")
        path = self.question_file_map.get(question_idx)
        if path is None:
            path = self._join_path(self.token_dir, f"question_{question_idx}.csv")
        if not self._path_exists(path):
            raise FileNotFoundError(
                f"Token-level CSV not found for question {question_idx}."
            )
        df = self._read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
        df = self._normalize_token_df(df, question_idx)

        required = {
            "question_idx",
            "token_idx",
            "layer_idx",
            "early_decoder",
            "probe_output",
            "probe_ans",
            "token_text",
        }
        missing_required = required - set(df.columns)
        if missing_required:
            raise ValueError(
                f"Token CSV for question {question_idx} missing required columns: {sorted(missing_required)}"
            )
        return df

    @lru_cache(maxsize=32)
    def load_sentence_df(self, question_idx: int) -> pd.DataFrame:
        if self.sentence_dir is None:
            raise FileNotFoundError("Sentence-level data directory is not configured.")
        path = self._join_path(self.sentence_dir, f"question_{question_idx}.csv")
        if not self._path_exists(path):
            record = self.get_question(question_idx)
            if record.question_hash:
                alt_path = self._join_path(self.sentence_dir, f"{record.question_hash}.csv")
                if self._path_exists(alt_path):
                    path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Sentence-level CSV not found for question {question_idx}."
                    )
            else:
                raise FileNotFoundError(
                    f"Sentence-level CSV not found for question {question_idx}."
                )
        df = self._read_csv(path)
        df = self._normalize_step_df(df, question_idx)

        expected = {
            "question_idx",
            "layer_idx",
            "early_decoder",
            "probe_output",
            "sentence_idx",
            "probe_ans",
        }
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"Step CSV for question {question_idx} missing columns: {sorted(missing)}"
            )
        return df

    # ---- Label / display helpers ----

    def _collect_token_labels(self, subset: pd.DataFrame) -> List[str]:
        token_series = (
            subset[["token_idx", "token_text"]]
            .drop_duplicates(subset=["token_idx"])
            .set_index("token_idx")
            .sort_index()["token_text"]
        )
        labels: List[str] = []
        for token_idx, token_text in token_series.items():
            display = token_text if token_text else "␀"
            if len(display) > 14:
                display = display[:11] + "…"
            labels.append(f"{token_idx}: {display}")
        return labels

    def _build_sentence_previews(
        self,
        question_idx: int,
        sentence_indices: List[int],
        probe_name: str,
        sentence_text_lookup: Optional[Dict[int, str]] = None,
        max_tokens: int = 10,
        max_chars: int = 60,
    ) -> Dict[int, str]:
        text_map: Dict[int, str] = {}
        if sentence_text_lookup:
            for idx, text in sentence_text_lookup.items():
                text_map[int(idx)] = str(text)

        missing_indices = [idx for idx in sentence_indices if idx not in text_map]

        if missing_indices:
            try:
                token_df = self._load_token_df(question_idx)
            except FileNotFoundError:
                token_df = None
            if token_df is not None:
                base = token_df[token_df["sentence_idx"].isin(missing_indices)].copy()
                if not base.empty:
                    layer_min = int(base["layer_idx"].min())
                    preferred = base[
                        (base["early_decoder"] == probe_name)
                        & (base["layer_idx"] == layer_min)
                    ]
                    if preferred.empty:
                        preferred = base[base["layer_idx"] == layer_min]
                    if preferred.empty:
                        preferred = base

                    preferred.sort_values(["sentence_idx", "token_idx"], inplace=True)

                    for sentence_idx in missing_indices:
                        subset = preferred[preferred["sentence_idx"] == sentence_idx]
                        if subset.empty:
                            continue
                        subset = subset.drop_duplicates(subset=["token_idx"])
                        tokens = [str(text) for text in subset["token_text"] if str(text)]
                        summary = " ".join(tokens[:max_tokens]).strip()
                        if len(tokens) > max_tokens:
                            summary += " …"
                        text_map[sentence_idx] = summary

        previews: Dict[int, str] = {}
        for sentence_idx in sentence_indices:
            collapsed = " ".join(text_map.get(sentence_idx, "").split())
            truncated = collapsed[:max_chars - 1].rstrip() + "…" if len(collapsed) > max_chars else collapsed
            previews[sentence_idx] = f'Step {sentence_idx}: "{truncated}"'

        return previews

    @staticmethod
    def _format_layer_labels(layer_indices: Iterable[int]) -> List[str]:
        _SPECIAL = {-2: "cot_monitor_llm", -1: "forced_answer"}
        labels: List[str] = []
        for layer in layer_indices:
            try:
                layer_int = int(layer)
            except (TypeError, ValueError):
                labels.append(str(layer))
                continue
            labels.append(_SPECIAL.get(layer_int, str(layer_int)))
        return labels

    # ---- Colorscale / heat value helpers ----

    @staticmethod
    def _blend_confidence_colorscale(colors: List[str]) -> List[List[float | str]]:
        if not colors:
            return [[0.0, "#FFFFFF"], [1.0, "#4E79A7"]]

        total = len(colors)
        eps = max(1e-6, 1.0 / (total * 1000))
        stops: List[List[float | str]] = []

        for idx, color in enumerate(colors):
            start = idx / total
            end = (idx + 1) / total

            stops.append([start, "#FFFFFF"])
            stops.append([min(start + eps, end), "#FFFFFF"])
            stops.append([max(end - eps, start), color])
            stops.append([end, color])

        stops.sort(key=lambda pair: pair[0])
        if stops[-1][0] < 1.0:
            stops.append([1.0, colors[-1]])
        return stops

    @staticmethod
    def _format_predicted_label(raw_value: Any, class_labels: Dict[int, str]) -> str:
        if raw_value is None:
            return "N/A"
        try:
            if pd.isna(raw_value):  # type: ignore[arg-type]
                return "N/A"
        except TypeError:
            pass
        try:
            idx = ProbeDataRepository._to_category_idx(raw_value)
        except ValueError:
            text = str(raw_value).strip()
            return text if text else "N/A"
        return class_labels.get(idx, str(raw_value))

    def build_class_labels(
        self,
        probe_name: str,
        num_classes: int,
        question: QuestionRecord,
    ) -> Dict[int, str]:
        if num_classes == 2 and probe_name.endswith("answer_correct"):
            return {0: "Incorrect", 1: "Correct"}

        enumerated = question.enumerated_choices()
        if enumerated and num_classes <= len(enumerated):
            labels: Dict[int, str] = {}
            for idx in range(num_classes):
                if idx < len(enumerated):
                    option_label, choice_text = enumerated[idx]
                    summary = choice_text.replace("\n", " ")
                    if len(summary) > 48:
                        summary = summary[:47].rstrip() + "…"
                    suffix = f" — {summary}" if summary else ""
                    labels[idx] = f"{option_label}{suffix}"
                else:
                    labels[idx] = label_for_choice(idx)
            return labels

        return build_default_class_labels(num_classes)

    @staticmethod
    def _to_category_idx(value: Any) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = str(value).strip().upper()
        if text.isdigit():
            return int(text)
        if len(text) == 1 and text in CHOICE_LABELS:
            return CHOICE_LABELS.index(text)
        raise ValueError(f"Unsupported probe_ans value: {value!r}")

    def _build_customdata(
        self,
        layer_indices: List[int],
        column_indices: List[int],
        lookup: Dict[Tuple[int, int], Dict[str, str]],
        fields: List[str],
    ) -> List[List[List[str]]]:
        return [
            [
                [lookup.get((layer, col), {}).get(field, "") for field in fields]
                for col in column_indices
            ]
            for layer in layer_indices
        ]

    def _prepare_heat_values(
        self,
        subset: pd.DataFrame,
        class_labels: Dict[int, str],
        num_classes: int,
    ) -> Tuple[pd.DataFrame, List[List[float | str]], float, float, Dict[str, Any]]:
        if num_classes == 2:
            subset = subset.copy()
            subset["heat_value"] = subset["probe_output"].apply(
                lambda arr: float(arr[1]) if len(arr) > 1 else float(arr[0])
            )
            colorscale = [
                [0.0, "#8B0000"],
                [0.5, "#FFFFFF"],
                [1.0, "#00008B"],
            ]
            zmin, zmax = 0.0, 1.0
            colorbar = {
                "title": "P(class=1)",
                "tickmode": "array",
                "tickvals": [0.0, 0.5, 1.0],
                "ticktext": ["0.0", "0.5", "1.0"],
            }
            return subset, colorscale, zmin, zmax, colorbar

        subset = subset.copy()

        def _resolve_category(row: pd.Series) -> Optional[int]:
            value = row.get("probe_ans")
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except TypeError:
                pass

            text = str(value).strip()
            if not text:
                return None

            try:
                return self._to_category_idx(value)
            except ValueError:
                if text.upper() == "N/A":
                    return None
                outputs = row.get("probe_output")
                if isinstance(outputs, (list, tuple)) and outputs:
                    return int(max(range(len(outputs)), key=lambda idx: outputs[idx]))
                return None

        subset["category_idx"] = subset.apply(_resolve_category, axis=1)
        subset["confidence"] = subset["probe_output"].apply(
            lambda arr: float(max(arr)) if arr else 0.0
        )
        subset["confidence"] = subset["confidence"].clip(
            lower=1e-3, upper=1.0 - 1e-6
        )
        subset.loc[subset["category_idx"].isna(), "confidence"] = 0.0
        subset["heat_value"] = subset["category_idx"].astype(float) + subset["confidence"]
        subset.loc[subset["category_idx"].isna(), "heat_value"] = float("nan")
        colors = cycle_palette(num_classes)
        colorscale = self._blend_confidence_colorscale(colors)
        zmin = 0.0
        zmax = float(num_classes)
        tickvals = [idx + 0.5 for idx in range(num_classes)]
        ticktext = [class_labels.get(idx, str(idx)) for idx in range(num_classes)]
        colorbar = {
            "title": "Predicted (shade = confidence)",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }
        return subset, colorscale, zmin, zmax, colorbar

    # ---- Heatmap payload builders ----

    def _build_hover_lookup(
        self,
        subset: pd.DataFrame,
        class_labels: Dict[int, str],
        index_col: str,
    ) -> Dict[Tuple[int, int], Dict[str, str]]:
        """Build a (layer_idx, index_col) -> field dict for customdata."""
        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            predicted_label = self._format_predicted_label(row.probe_ans, class_labels)
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            confidence_str = f"{float(row.confidence):.3f}" if hasattr(row, "confidence") else ""
            entry: Dict[str, str] = {
                "confidence": confidence_str,
                "argmax_label": predicted_label,
                "logits": f"[{logits_str}]",
            }
            if index_col == "token_idx":
                entry["token_pos"] = str(int(row.token_idx))
                entry["token_text"] = str(row.token_text) if row.token_text else "␀"
            lookup[(int(row.layer_idx), int(getattr(row, index_col)))] = entry
        return lookup

    def _build_token_heatmap(
        self, question_idx: int, probe_name: str
    ) -> HeatmapPayload:
        df = self._load_token_df(question_idx)
        subset = df[df["early_decoder"] == probe_name]
        if subset.empty:
            raise ValueError(
                f"No token-level probe data found for question {question_idx} and probe '{probe_name}'."
            )
        subset = subset.copy().sort_values(["layer_idx", "token_idx"])
        question = self.get_question(question_idx)
        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self.build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="token_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)
        token_labels = self._collect_token_labels(subset)

        # Find prompt/response boundary
        token_series = (
            subset[["token_idx", "token_text"]]
            .drop_duplicates(subset=["token_idx"])
            .set_index("token_idx")
            .sort_index()["token_text"]
        )
        prompt_boundary_label: Optional[str] = None
        for label_idx, (_, token_text) in enumerate(token_series.items()):
            if str(token_text).strip().lower() in _ASSISTANT_MARKERS:
                if label_idx < len(token_labels):
                    prompt_boundary_label = token_labels[label_idx]
                break

        layer_labels = self._format_layer_labels(list(pivot.index))
        lookup = self._build_hover_lookup(subset, class_labels, "token_idx")
        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["token_pos", "token_text", "confidence", "argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Token Pos: %{customdata[0]}<br>"
            "Token: %{customdata[1]}<br>"
            "Confidence: %{customdata[2]}<br>"
            "Predicted: %{customdata[3]}<br>"
            "Logits: %{customdata[4]}<extra></extra>"
        )

        return HeatmapPayload(
            pivot=pivot,
            x_labels=token_labels,
            layer_labels=layer_labels,
            customdata=customdata,
            class_labels=class_labels,
            probe_name=probe_name,
            view_mode="token",
            x_axis_title="Token index",
            hover_template=hover_template,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=colorbar,
            prompt_boundary_label=prompt_boundary_label,
        )

    def _build_sentence_heatmap(
        self,
        question_idx: int,
        probe_name: str,
        probe_layer: Optional[int] = None,
        include_baselines: bool = False,
    ) -> HeatmapPayload:
        df = self.load_sentence_df(question_idx)

        target_layers: List[int] = []
        if probe_layer is not None:
            target_layers.append(probe_layer)
        if include_baselines:
            target_layers.extend([-1, -2])

        if target_layers:
            subset = df[df["layer_idx"].isin(target_layers)]
        else:
            subset = df[df["layer_idx"] >= 0]

        if subset.empty:
            raise ValueError(
                f"No sentence-level data found for question {question_idx} and layer(s) {target_layers or 'probe layers'}."
            )
        subset = subset.copy().sort_values(["layer_idx", "sentence_idx"])
        question = self.get_question(question_idx)

        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self.build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="sentence_idx", values="heat_value"
        ).sort_index(axis=1)

        pivot = pivot.loc[~pivot.index.isna()]
        pivot = pivot.loc[:, ~pivot.columns.isna()]

        # Reorder rows: probe layers first, then forced_answer, then cot_monitor
        def _layer_sort_key(idx):
            try:
                v = int(idx)
            except (TypeError, ValueError):
                return (2, str(idx))
            if v >= 0:
                return (0, v)
            return (1, -v)  # -1 before -2

        pivot = pivot.reindex(sorted(pivot.index, key=_layer_sort_key))

        sentence_indices = [int(idx) for idx in pivot.columns if not pd.isna(idx)]
        text_lookup: Dict[int, str] = {}
        if "sentence_text" in df.columns:
            ordered_for_text = df.sort_values(
                ["sentence_idx", "layer_idx"], ascending=[True, False]
            )
            text_lookup = (
                ordered_for_text[["sentence_idx", "sentence_text"]]
                .drop_duplicates(subset=["sentence_idx"])
                .set_index("sentence_idx")["sentence_text"]
                .astype(str)
                .to_dict()
            )
        previews = self._build_sentence_previews(
            question_idx, sentence_indices, probe_name, text_lookup
        )
        sentence_labels = [previews.get(int(idx), str(idx)) for idx in sentence_indices]
        layer_labels = self._format_layer_labels(list(pivot.index))
        layer_labels = [
            f"Probe Layer {label}" if str(label).lstrip("-").isdigit() and int(str(label)) >= 0 else str(label)
            for label in layer_labels
        ]

        lookup = self._build_hover_lookup(subset, class_labels, "sentence_idx")
        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["argmax_label", "confidence", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Step: %{x}<br>"
            "Predicted: %{customdata[0]}<br>"
            "Confidence: %{customdata[1]}<br>"
            "Logits: %{customdata[2]}<extra></extra>"
        )

        return HeatmapPayload(
            pivot=pivot,
            x_labels=sentence_labels,
            layer_labels=layer_labels,
            customdata=customdata,
            class_labels=class_labels,
            probe_name=probe_name,
            view_mode="sentence",
            x_axis_title="Step index",
            hover_template=hover_template,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=colorbar,
        )

    def get_heatmap_payload(
        self,
        question_idx: int,
        probe_name: str,
        view_mode: ViewMode = "token",
        probe_layer: Optional[int] = None,
        include_baselines: bool = False,
    ) -> HeatmapPayload:
        if view_mode == "token":
            return self._build_token_heatmap(question_idx, probe_name)
        if view_mode == "sentence":
            return self._build_sentence_heatmap(
                question_idx,
                probe_name,
                probe_layer=probe_layer,
                include_baselines=include_baselines,
            )
        raise ValueError(f"Unsupported view mode: {view_mode}")

    def list_sentence_options(
        self,
        question_idx: int,
        probe_name: str,
        probe_layer: Optional[int] = None,
    ) -> List[Dict[str, int]]:
        try:
            df = self.load_sentence_df(question_idx)
        except FileNotFoundError:
            return []
        subset = df[df["layer_idx"] == probe_layer] if probe_layer is not None else df[df["layer_idx"] >= 0]
        if subset.empty:
            return []
        values = sorted(subset["sentence_idx"].astype(int).unique())
        text_lookup: Dict[int, str] = {}
        if "sentence_text" in df.columns:
            text_lookup = (
                df[["sentence_idx", "sentence_text"]]
                .drop_duplicates(subset=["sentence_idx"])
                .set_index("sentence_idx")["sentence_text"]
                .astype(str)
                .to_dict()
            )
        previews = self._build_sentence_previews(
            question_idx, values, probe_name, text_lookup
        )
        return [{"label": previews.get(val, f"Step {val}"), "value": int(val)} for val in values]

    def get_sentence_text(self, question_idx: int, sentence_idx: int) -> str:
        try:
            df = self.load_sentence_df(question_idx)
        except FileNotFoundError:
            return ""

        subset = df[df["sentence_idx"] == sentence_idx]
        if subset.empty:
            return ""

        if "sentence_text" in subset.columns:
            for raw_text in subset["sentence_text"].dropna():
                if raw_text:
                    return str(raw_text)

        try:
            token_df = self._load_token_df(question_idx)
        except FileNotFoundError:
            return ""

        tokens = [
            str(tok)
            for tok in token_df[token_df["sentence_idx"] == sentence_idx]["token_text"]
        ]
        return "".join(token for token in tokens if token)

    def get_decoder_step_payload(
        self,
        question_idx: int,
        decoder_key: str,
    ) -> DecoderStepPayload:
        try:
            df = self.load_sentence_df(question_idx)
        except FileNotFoundError:
            raise ValueError("No sentence-level data available for this dataset.")

        if decoder_key == "forced_answer":
            layer_idx = -1
        elif decoder_key == "cot_monitor_llm":
            layer_idx = -2
        elif decoder_key.startswith("probe_"):
            try:
                layer_idx = int(decoder_key.split("_", 1)[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid probe layer key: {decoder_key}")
        else:
            raise ValueError(f"Unknown decoder key: {decoder_key}")

        subset = df[df["layer_idx"] == layer_idx]
        if subset.empty:
            raise ValueError(f"No data for decoder '{decoder_key}' on this question.")

        subset = subset.sort_values("sentence_idx")

        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self.build_class_labels(
            decoder_key,
            num_classes,
            self.get_question(question_idx),
        )

        step_indices = subset["sentence_idx"].astype(int).tolist()
        probabilities: List[List[float]] = []
        for outputs in subset["probe_output"]:
            if isinstance(outputs, (list, tuple)):
                probabilities.append([float(val) for val in outputs])
            else:
                probabilities.append([])

        x_labels = [f"Step {idx}" for idx in step_indices]
        text_lookup: Dict[int, str] = {}
        if "sentence_text" in subset.columns:
            text_lookup = (
                subset[["sentence_idx", "sentence_text"]]
                .drop_duplicates(subset=["sentence_idx"])
                .set_index("sentence_idx")["sentence_text"]
                .astype(str)
                .to_dict()
            )
        previews = self._build_sentence_previews(
            question_idx,
            step_indices,
            decoder_key,
            sentence_text_lookup=text_lookup,
            max_chars=30,
        )
        tick_labels = [previews.get(idx, f"Step {idx}") for idx in step_indices]
        return DecoderStepPayload(
            step_indices=step_indices,
            probabilities=probabilities,
            class_labels=class_labels,
            x_labels=x_labels,
            tick_labels=tick_labels,
        )

    def get_sentence_token_payload(
        self,
        question_idx: int,
        probe_name: str,
        sentence_idx: int,
        probe_layer: Optional[int] = None,
    ) -> HeatmapPayload:
        df = self._load_token_df(question_idx)
        if probe_layer is not None:
            subset = df[
                (df["sentence_idx"] == sentence_idx)
                & (df["layer_idx"] == probe_layer)
            ]
        else:
            subset = df[
                (df["sentence_idx"] == sentence_idx)
                & (df["layer_idx"] >= 0)
            ]
        if subset.empty:
            raise ValueError(
                f"No token-level data for question {question_idx}, sentence {sentence_idx}, probe '{probe_name}'."
            )
        subset = subset.copy().sort_values(["layer_idx", "token_idx"])
        question = self.get_question(question_idx)
        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self.build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="token_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)
        token_labels = self._collect_token_labels(subset)
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup = self._build_hover_lookup(subset, class_labels, "token_idx")
        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["token_pos", "token_text", "confidence", "argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Token Pos: %{customdata[0]}<br>"
            "Token: %{customdata[1]}<br>"
            "Confidence: %{customdata[2]}<br>"
            "Predicted: %{customdata[3]}<br>"
            "Logits: %{customdata[4]}<extra></extra>"
        )

        return HeatmapPayload(
            pivot=pivot,
            x_labels=token_labels,
            layer_labels=layer_labels,
            customdata=customdata,
            class_labels=class_labels,
            probe_name=probe_name,
            view_mode="token",
            x_axis_title=f"Token index (sentence {sentence_idx})",
            hover_template=hover_template,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=colorbar,
        )
