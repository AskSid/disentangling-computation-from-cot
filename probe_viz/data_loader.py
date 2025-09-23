"""Data loading and preprocessing utilities for the probe visualization app."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from typing import Literal

import pandas as pd

from constants import (
    CHOICE_LABELS,
    build_default_class_labels,
    cycle_palette,
    label_for_choice,
)

MODULE_ROOT = Path(__file__).resolve().parent
DATA_DIR = MODULE_ROOT / "data"

DEFAULT_TOKEN_LEVEL_RESULTS = DATA_DIR / (
    "deepseekr1_anatomy_results_token_level_questions_0_1.csv"
)
DEFAULT_SENTENCE_LEVEL_RESULTS = DATA_DIR / (
    "deepseekr1_anatomy_probe_results_sentence_level_questions_0_1.csv"
)
DEFAULT_PREDICTIONS_FILE = DATA_DIR / (
    "deepseekr1_anatomy_predictions_questions_0_1.csv"
)

ViewMode = Literal["token", "sentence"]


@dataclass(frozen=True)
class QuestionRecord:
    """Container for per-question metadata."""

    question_idx: int
    question: str
    answer_choices: List[str]
    full_prompt: str
    correct_answer: str
    full_cot: str
    predicted_answer: str
    category: str

    def enumerated_choices(self) -> List[Tuple[str, str]]:
        """Return answer choices paired with alphabetical labels."""
        labels = []
        for idx, choice in enumerate(self.answer_choices):
            labels.append((label_for_choice(idx), choice))
        return labels


@dataclass(frozen=True)
class HeatmapPayload:
    """Data required to render a heatmap for a single probe/question pair."""

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


class ProbeDataRepository:
    """Provides convenient accessors around prediction and probe data."""

    def __init__(
        self,
        predictions_path: Optional[Path | str] = None,
        probe_results_path: Optional[Path | str] = None,
        sentence_results_path: Optional[Path | str] = None,
    ) -> None:
        self.predictions_path = self._resolve_predictions_path(predictions_path)
        self.probe_results_path = self._resolve_token_results_path(probe_results_path)
        self.sentence_results_path = self._resolve_sentence_results_path(
            sentence_results_path
        )

        self.predictions_df = self._load_predictions(self.predictions_path)
        self.token_probe_df = self._load_probe_results(self.probe_results_path)
        self.sentence_probe_df = (
            self._load_sentence_probe_results(self.sentence_results_path)
            if self.sentence_results_path is not None
            else None
        )

        self.questions: Dict[int, QuestionRecord] = self._build_questions_index()
        self.available_question_indices: List[int] = sorted(self.questions.keys())

        probe_sources = [self.token_probe_df]
        if self.sentence_probe_df is not None:
            probe_sources.append(self.sentence_probe_df)
        probe_names = {
            probe
            for df in probe_sources
            for probe in df["early_decoder"].astype(str).unique()
        }
        self.available_probes: List[str] = sorted(probe_names)

    @staticmethod
    def _resolve_predictions_path(overrides: Optional[Path | str]) -> Path:
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"Predictions CSV not found: {path}")
            return path
        if DEFAULT_PREDICTIONS_FILE.exists():
            return DEFAULT_PREDICTIONS_FILE
        raise FileNotFoundError(
            "Could not locate predictions CSV. Expected to find "
            f"'{DEFAULT_PREDICTIONS_FILE}'."
        )

    @staticmethod
    def _resolve_token_results_path(overrides: Optional[Path | str]) -> Path:
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"Probe results CSV not found: {path}")
            return path
        if DEFAULT_TOKEN_LEVEL_RESULTS.exists():
            return DEFAULT_TOKEN_LEVEL_RESULTS
        raise FileNotFoundError(
            "Could not locate token-level probe results CSV. Expected to find "
            f"'{DEFAULT_TOKEN_LEVEL_RESULTS}'."
        )

    @staticmethod
    def _resolve_sentence_results_path(
        overrides: Optional[Path | str],
    ) -> Optional[Path]:
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"Sentence-level probe results CSV not found: {path}")
            return path
        if DEFAULT_SENTENCE_LEVEL_RESULTS.exists():
            return DEFAULT_SENTENCE_LEVEL_RESULTS
        return None

    @staticmethod
    def _parse_answer_choices(raw_value: str) -> List[str]:
        """Handle either JSON or Python list string representations."""
        if isinstance(raw_value, list):
            return raw_value
        try:
            # Prefer JSON parsing when possible.
            return json.loads(raw_value)
        except (json.JSONDecodeError, TypeError):
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, list):
                return parsed
            raise ValueError(f"Unexpected answer_choices format: {raw_value!r}")

    @classmethod
    def _load_predictions(cls, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        expected_columns = {
            "question_idx",
            "question",
            "answer_choices",
            "full_prompt",
            "correct_answer",
            "full_cot",
            "predicted_answer",
            "category",
        }
        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Predictions CSV missing expected columns: {sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        df["answer_choices"] = df["answer_choices"].apply(cls._parse_answer_choices)
        df["correct_answer"] = df["correct_answer"].astype(str)
        df["predicted_answer"] = df["predicted_answer"].astype(str)
        df["category"] = df["category"].astype(str)
        return df

    @staticmethod
    def _load_probe_results(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        expected_columns = {
            "question_idx",
            "token_idx",
            "sentence_idx",
            "layer_idx",
            "token",
            "token_text",
            "early_decoder",
            "probe_output",
            "probe_ans",
        }
        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Probe results CSV missing expected columns: {sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        df["token_idx"] = df["token_idx"].astype(int)
        df["sentence_idx"] = df["sentence_idx"].astype(int)
        df["layer_idx"] = df["layer_idx"].astype(int)
        df["probe_ans"] = df["probe_ans"].apply(ProbeDataRepository._parse_probe_ans)
        df["token_text"] = df["token_text"].fillna("")
        df["early_decoder"] = df["early_decoder"].astype(str)
        # Parse textual list representations into Python lists of floats.
        df["probe_output"] = df["probe_output"].apply(ast.literal_eval)
        return df

    @staticmethod
    def _load_sentence_probe_results(path: Optional[Path]) -> Optional[pd.DataFrame]:
        if path is None:
            return None
        df = pd.read_csv(path)
        expected_columns = {
            "question_idx",
            "sentence_idx",
            "layer_idx",
            "early_decoder",
            "probe_output",
            "probe_ans",
        }
        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(
                "Sentence-level probe results CSV missing expected columns: "
                f"{sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        df["sentence_idx"] = df["sentence_idx"].astype(int)
        df["layer_idx"] = df["layer_idx"].astype(int)
        df["probe_ans"] = df["probe_ans"].apply(ProbeDataRepository._parse_probe_ans)
        df["early_decoder"] = df["early_decoder"].astype(str)
        df["probe_output"] = df["probe_output"].apply(ast.literal_eval)
        return df

    @staticmethod
    def _parse_probe_ans(value) -> int:
        """Normalise probe argmax outputs into integer class indices."""

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)

        text = str(value).strip()
        if text.isdigit():
            return int(text)

        if len(text) == 1 and text.upper() in CHOICE_LABELS:
            return CHOICE_LABELS.index(text.upper())

        lowered = text.lower()
        if lowered in {"correct", "true"}:
            return 1
        if lowered in {"incorrect", "false"}:
            return 0

        raise ValueError(f"Unsupported probe_ans format: {value!r}")

    def _build_questions_index(self) -> Dict[int, QuestionRecord]:
        records: Dict[int, QuestionRecord] = {}
        for row in self.predictions_df.itertuples(index=False):
            records[int(row.question_idx)] = QuestionRecord(
                question_idx=int(row.question_idx),
                question=str(row.question),
                answer_choices=list(row.answer_choices),
                full_prompt=str(row.full_prompt),
                correct_answer=str(row.correct_answer),
                full_cot=str(row.full_cot),
                predicted_answer=str(row.predicted_answer),
                category=str(row.category),
            )
        return records

    def list_question_options(self) -> List[Dict[str, str | int]]:
        """Return dropdown-friendly metadata for all questions."""
        options: List[Dict[str, str | int]] = []
        for question_idx in self.available_question_indices:
            record = self.questions[question_idx]
            preview = record.question.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            label = f"#{question_idx} — {preview}"
            options.append({"label": label, "value": question_idx})
        return options

    def list_probe_options(self) -> List[Dict[str, str]]:
        return [{"label": probe, "value": probe} for probe in self.available_probes]

    def get_question(self, question_idx: int) -> QuestionRecord:
        if question_idx not in self.questions:
            raise KeyError(f"Unknown question_idx: {question_idx}")
        return self.questions[question_idx]

    def _collect_token_labels(self, subset: pd.DataFrame) -> List[str]:
        """Create human-friendly labels for each token index."""
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

    @staticmethod
    def _format_layer_labels(layer_indices: Iterable[int]) -> List[str]:
        return [str(layer) for layer in layer_indices]

    @staticmethod
    def _build_discrete_colorscale(colors: List[str]) -> List[List[float | str]]:
        """Create a stepwise colorscale for categorical outputs."""

        if not colors:
            return [[0.0, "#4E79A7"], [1.0, "#4E79A7"]]

        stops: List[List[float | str]] = []
        total = len(colors)
        for idx, color in enumerate(colors):
            start = idx / total
            end = (idx + 1) / total
            stops.append([start, color])
            stops.append([end, color])
        stops[0][0] = 0.0
        stops[-1][0] = 1.0
        return stops

    @staticmethod
    def _summarize_choice(choice: str, limit: int = 48) -> str:
        choice = choice.replace("\n", " ").strip()
        if not choice:
            return ""
        if len(choice) > limit:
            return choice[: limit - 1].rstrip() + "…"
        return choice

    def _build_class_labels(
        self,
        probe_name: str,
        num_classes: int,
        question: QuestionRecord,
    ) -> Dict[int, str]:
        """Create descriptive labels for the specified probe output classes."""

        if num_classes == 2 and probe_name.endswith("answer_correct"):
            return {0: "Incorrect", 1: "Correct"}

        if "model_answer" in probe_name or "correct_answer" in probe_name:
            enumerated = question.enumerated_choices()
            labels: Dict[int, str] = {}
            for idx in range(num_classes):
                if idx < len(enumerated):
                    option_label, choice_text = enumerated[idx]
                    summary = self._summarize_choice(choice_text)
                    suffix = f" — {summary}" if summary else ""
                    labels[idx] = f"{option_label}{suffix}"
                else:
                    labels[idx] = label_for_choice(idx)
            return labels

        return build_default_class_labels(num_classes)

    @staticmethod
    def _build_customdata(
        layer_indices: List[int],
        column_indices: List[int],
        lookup: Dict[Tuple[int, int], Dict[str, str]],
        fields: List[str],
    ) -> List[List[List[str]]]:
        customdata: List[List[List[str]]] = []
        for layer in layer_indices:
            row: List[List[str]] = []
            for column in column_indices:
                cell = lookup.get((layer, column), {})
                row.append([cell.get(field, "") for field in fields])
            customdata.append(row)
        return customdata

    def _prepare_heat_values(
        self,
        subset: pd.DataFrame,
        class_labels: Dict[int, str],
        num_classes: int,
    ) -> Tuple[pd.DataFrame, List[List[float | str]], float, float, Dict[str, Any]]:
        """Attach heat values and colour metadata to the subset."""

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
        subset["heat_value"] = subset["probe_ans"].astype(float)
        colors = cycle_palette(num_classes)
        colorscale = self._build_discrete_colorscale(colors)
        zmin = -0.5
        zmax = num_classes - 0.5
        tickvals = list(range(num_classes))
        ticktext = [class_labels.get(idx, str(idx)) for idx in tickvals]
        colorbar = {
            "title": "Predicted Answer",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }
        return subset, colorscale, zmin, zmax, colorbar

    def _build_token_heatmap_from_subset(
        self,
        subset: pd.DataFrame,
        question: QuestionRecord,
        probe_name: str,
        x_axis_title: str,
    ) -> HeatmapPayload:
        if subset.empty:
            raise ValueError(
                f"No token-level probe data found for question {question.question_idx} "
                f"and probe '{probe_name}'."
            )

        subset = subset.copy()
        subset.sort_values(["layer_idx", "token_idx"], inplace=True)

        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self._build_class_labels(probe_name, num_classes, question)

        (
            subset,
            colorscale,
            zmin,
            zmax,
            colorbar,
        ) = self._prepare_heat_values(subset, class_labels, num_classes)

        pivot = subset.pivot(
            index="layer_idx", columns="token_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)

        token_labels = self._collect_token_labels(subset)
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            lookup[(int(row.layer_idx), int(row.token_idx))] = {
                "token_text": str(row.token_text) if row.token_text else "␀",
                "argmax_label": class_labels.get(int(row.probe_ans), str(row.probe_ans)),
                "logits": f"[{logits_str}]",
            }

        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["token_text", "argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Token: %{x}<br>"
            "Token text: %{customdata[0]}<br>"
            "Predicted: %{customdata[1]}<br>"
            "Logits: %{customdata[2]}<extra></extra>"
        )

        return HeatmapPayload(
            pivot=pivot,
            x_labels=token_labels,
            layer_labels=layer_labels,
            customdata=customdata,
            class_labels=class_labels,
            probe_name=probe_name,
            view_mode="token",
            x_axis_title=x_axis_title,
            hover_template=hover_template,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=colorbar,
        )

    def get_heatmap_payload(
        self, question_idx: int, probe_name: str, view_mode: ViewMode = "token"
    ) -> HeatmapPayload:
        if view_mode == "token":
            return self._build_token_heatmap(question_idx, probe_name)
        if view_mode == "sentence":
            if self.sentence_probe_df is None:
                raise ValueError("Sentence-level probe data is not available.")
            return self._build_sentence_heatmap(question_idx, probe_name)
        raise ValueError(f"Unsupported view mode: {view_mode}")

    def _build_token_heatmap(self, question_idx: int, probe_name: str) -> HeatmapPayload:
        subset = self.token_probe_df[
            (self.token_probe_df["question_idx"] == question_idx)
            & (self.token_probe_df["early_decoder"] == probe_name)
        ]
        question = self.get_question(question_idx)
        return self._build_token_heatmap_from_subset(
            subset,
            question,
            probe_name,
            x_axis_title="Token index",
        )

    def _build_sentence_heatmap(
        self, question_idx: int, probe_name: str
    ) -> HeatmapPayload:
        if self.sentence_probe_df is None:
            raise ValueError("Sentence-level probe data is not available.")

        subset = self.sentence_probe_df[
            (self.sentence_probe_df["question_idx"] == question_idx)
            & (self.sentence_probe_df["early_decoder"] == probe_name)
        ]
        if subset.empty:
            raise ValueError(
                f"No sentence-level probe data found for question {question_idx} "
                f"and probe '{probe_name}'."
            )
        subset = subset.copy()
        subset.sort_values(["layer_idx", "sentence_idx"], inplace=True)

        question = self.get_question(question_idx)

        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self._build_class_labels(probe_name, num_classes, question)

        (
            subset,
            colorscale,
            zmin,
            zmax,
            colorbar,
        ) = self._prepare_heat_values(subset, class_labels, num_classes)

        pivot = subset.pivot(
            index="layer_idx", columns="sentence_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)

        sentence_labels = [str(idx) for idx in pivot.columns]
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            lookup[(int(row.layer_idx), int(row.sentence_idx))] = {
                "argmax_label": class_labels.get(int(row.probe_ans), str(row.probe_ans)),
                "logits": f"[{logits_str}]",
            }

        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Sentence: %{x}<br>"
            "Predicted: %{customdata[0]}<br>"
            "Logits: %{customdata[1]}<extra></extra>"
        )

        return HeatmapPayload(
            pivot=pivot,
            x_labels=sentence_labels,
            layer_labels=layer_labels,
            customdata=customdata,
            class_labels=class_labels,
            probe_name=probe_name,
            view_mode="sentence",
            x_axis_title="Sentence index",
            hover_template=hover_template,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=colorbar,
        )

    def list_sentence_options(
        self, question_idx: int, probe_name: str
    ) -> List[Dict[str, Union[str, int]]]:
        if self.sentence_probe_df is None:
            return []

        subset = self.sentence_probe_df[
            (self.sentence_probe_df["question_idx"] == question_idx)
            & (self.sentence_probe_df["early_decoder"] == probe_name)
        ]
        if subset.empty:
            return []

        options: List[Dict[str, Union[str, int]]] = []
        for value in sorted(subset["sentence_idx"].astype(int).unique()):
            options.append({"label": f"Sentence {value}", "value": int(value)})
        return options

    def get_sentence_token_payload(
        self, question_idx: int, probe_name: str, sentence_idx: int
    ) -> HeatmapPayload:
        subset = self.token_probe_df[
            (self.token_probe_df["question_idx"] == question_idx)
            & (self.token_probe_df["early_decoder"] == probe_name)
            & (self.token_probe_df["sentence_idx"] == sentence_idx)
        ]

        question = self.get_question(question_idx)
        return self._build_token_heatmap_from_subset(
            subset,
            question,
            probe_name,
            x_axis_title=f"Token index (sentence {sentence_idx})",
        )

    def list_view_modes(self) -> List[Dict[str, str]]:
        """Return the available view modes for the heatmap toggle."""

        modes = [{"label": "Token-Level", "value": "token"}]
        if self.sentence_probe_df is not None and not self.sentence_probe_df.empty:
            modes.append({"label": "Sentence-Level", "value": "sentence"})
        return modes
