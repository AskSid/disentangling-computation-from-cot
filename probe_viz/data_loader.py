"""Data loading utilities for the Streamlit probe viewer."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
TOKEN_DIR_DEFAULT = DATA_DIR / "token_level"
SENTENCE_DIR_DEFAULT = DATA_DIR / "sentence_level"
PREDICTIONS_DEFAULT = DATA_DIR / "deepseekr1_anatomy_predictions_questions_0_to_10.csv"

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
        labels: List[Tuple[str, str]] = []
        for idx, choice in enumerate(self.answer_choices):
            labels.append((label_for_choice(idx), choice))
        return labels


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


class ProbeDataRepository:
    """Lazy-loading repository for per-question probe outputs."""

    def __init__(
        self,
        predictions_path: Optional[Path | str] = None,
        token_dir: Optional[Path | str] = None,
        sentence_dir: Optional[Path | str] = None,
    ) -> None:
        self.predictions_path = self._resolve_predictions_path(predictions_path)
        self.token_dir = self._resolve_dir(token_dir, TOKEN_DIR_DEFAULT)
        self.sentence_dir = self._resolve_dir(sentence_dir, SENTENCE_DIR_DEFAULT, optional=True)

        self.predictions_df = self._load_predictions(self.predictions_path)
        self.questions: Dict[int, QuestionRecord] = self._build_questions_index()
        self.available_question_indices: List[int] = sorted(self.questions.keys())
        self.available_probes: List[str] = self._discover_probes()

    @staticmethod
    def _resolve_predictions_path(overrides: Optional[Path | str]) -> Path:
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"Predictions CSV not found: {path}")
            return path
        if not PREDICTIONS_DEFAULT.exists():
            raise FileNotFoundError(
                "Could not locate predictions CSV. Expected to find "
                f"'{PREDICTIONS_DEFAULT}'."
            )
        return PREDICTIONS_DEFAULT

    @staticmethod
    def _resolve_dir(
        overrides: Optional[Path | str], default: Path, optional: bool = False
    ) -> Optional[Path]:
        if overrides:
            path = Path(overrides)
            if not path.exists():
                raise FileNotFoundError(f"Data path not found: {path}")
            return path
        if default.exists():
            return default
        if optional:
            return None
        raise FileNotFoundError(f"Expected data at '{default}'.")

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

    @classmethod
    def _load_predictions(cls, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
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
        df["answer_choices"] = df["answer_choices"].apply(cls._parse_answer_choices)
        df["correct_answer"] = df["correct_answer"].astype(str)
        df["predicted_answer"] = df["predicted_answer"].astype(str)
        df["category"] = df["category"].astype(str)
        return df

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
        options: List[Dict[str, str | int]] = []
        for idx in self.available_question_indices:
            record = self.questions[idx]
            preview = record.question.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            options.append({"label": f"#{idx} — {preview}", "value": idx})
        return options

    def list_probe_options(self) -> List[Dict[str, str]]:
        return [{"label": probe, "value": probe} for probe in self.available_probes]

    def list_view_modes(self) -> List[Dict[str, str]]:
        modes = [{"label": "Token-Level", "value": "token"}]
        if self.sentence_dir is not None and any(self.sentence_dir.glob("question_*.csv")):
            modes.append({"label": "Sentence-Level", "value": "sentence"})
        return modes

    def get_question(self, question_idx: int) -> QuestionRecord:
        if question_idx not in self.questions:
            raise KeyError(f"Unknown question_idx: {question_idx}")
        return self.questions[question_idx]

    def _discover_probes(self) -> List[str]:
        probes: set[str] = set()
        if self.token_dir is None:
            return []
        for idx in self.available_question_indices:
            try:
                df = self._load_token_df(idx)
            except (FileNotFoundError, ValueError):
                continue
            probes.update(df["early_decoder"].astype(str).unique())
            if probes:
                break
        return sorted(probes)

    @lru_cache(maxsize=32)
    def _load_token_df(self, question_idx: int) -> pd.DataFrame:
        if self.token_dir is None:
            raise FileNotFoundError("Token-level data directory is not configured.")
        path = self.token_dir / f"question_{question_idx}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Token-level CSV not found for question {question_idx}: {path}"
            )
        df = pd.read_csv(path)
        expected = {
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
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"Token CSV for question {question_idx} missing columns: {sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        df["token_idx"] = df["token_idx"].astype(int)
        df["sentence_idx"] = df["sentence_idx"].astype(int)
        df["layer_idx"] = df["layer_idx"].astype(int)
        df["early_decoder"] = df["early_decoder"].astype(str)
        df["token_text"] = df["token_text"].fillna("")
        df["probe_output"] = df["probe_output"].apply(ast.literal_eval)
        return df

    @lru_cache(maxsize=32)
    def _load_sentence_df(self, question_idx: int) -> pd.DataFrame:
        if self.sentence_dir is None:
            raise FileNotFoundError("Sentence-level data directory is not configured.")
        path = self.sentence_dir / f"question_{question_idx}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Sentence-level CSV not found for question {question_idx}: {path}"
            )
        df = pd.read_csv(path)
        expected = {
            "question_idx",
            "sentence_idx",
            "layer_idx",
            "early_decoder",
            "probe_output",
            "probe_ans",
        }
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"Sentence CSV for question {question_idx} missing columns: {sorted(missing)}"
            )
        df["question_idx"] = df["question_idx"].astype(int)
        df["sentence_idx"] = df["sentence_idx"].astype(int)
        df["layer_idx"] = df["layer_idx"].astype(int)
        df["early_decoder"] = df["early_decoder"].astype(str)
        df["probe_output"] = df["probe_output"].apply(ast.literal_eval)
        if "sentence_text" not in df.columns:
            df["sentence_text"] = ""
        else:
            df["sentence_text"] = df["sentence_text"].fillna("").astype(str)
        return df

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
        previews: Dict[int, str] = {}
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
                        summary_tokens = tokens[:max_tokens]
                        summary = " ".join(summary_tokens).strip()
                        if len(tokens) > max_tokens:
                            summary = summary.rstrip() + " …"
                        text_map[sentence_idx] = summary

        for sentence_idx in sentence_indices:
            raw_text = text_map.get(sentence_idx, "")
            collapsed = " ".join(raw_text.split())
            if not collapsed:
                collapsed = ""
            truncated = collapsed
            ellipsis_needed = len(collapsed) > max_chars
            if len(truncated) > max_chars:
                truncated = collapsed[: max_chars - 1].rstrip()
            if ellipsis_needed:
                truncated = truncated.rstrip("…") + "…"
            previews[sentence_idx] = f'{sentence_idx}: "{truncated}"'

        return previews

    @staticmethod
    def _format_layer_labels(layer_indices: Iterable[int]) -> List[str]:
        return [str(layer) for layer in layer_indices]

    @staticmethod
    def _build_discrete_colorscale(colors: List[str]) -> List[List[float | str]]:
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

    def _build_class_labels(
        self,
        probe_name: str,
        num_classes: int,
        question: QuestionRecord,
    ) -> Dict[int, str]:
        if num_classes == 2 and probe_name.endswith("answer_correct"):
            return {0: "Incorrect", 1: "Correct"}

        if "model_answer" in probe_name or "correct_answer" in probe_name:
            enumerated = question.enumerated_choices()
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
        subset["category_idx"] = subset["probe_ans"].apply(self._to_category_idx)
        subset["heat_value"] = subset["category_idx"].astype(float)
        colors = cycle_palette(num_classes)
        colorscale = self._build_discrete_colorscale(colors)
        zmin = -0.5
        zmax = num_classes - 0.5
        tickvals = list(range(num_classes))
        ticktext = [class_labels.get(idx, str(idx)) for idx in tickvals]
        colorbar = {
            "title": "Predicted",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }
        return subset, colorscale, zmin, zmax, colorbar

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
        class_labels = self._build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="token_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)
        token_labels = self._collect_token_labels(subset)
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            label_idx = self._to_category_idx(row.probe_ans)
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            lookup[(int(row.layer_idx), int(row.token_idx))] = {
                "token_pos": str(int(row.token_idx)),
                "token_text": str(row.token_text) if row.token_text else "␀",
                "argmax_label": class_labels.get(label_idx, str(row.probe_ans)),
                "logits": f"[{logits_str}]",
            }
        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["token_pos", "token_text", "argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Token Pos: %{customdata[0]}<br>"
            "Token: %{customdata[1]}<br>"
            "Probe Predicted: %{customdata[2]}<br>"
            "Probe Pred Logits: %{customdata[3]}<extra></extra>"
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
        )

    def _build_sentence_heatmap(
        self, question_idx: int, probe_name: str
    ) -> HeatmapPayload:
        df = self._load_sentence_df(question_idx)
        subset = df[df["early_decoder"] == probe_name]
        if subset.empty:
            raise ValueError(
                f"No sentence-level probe data found for question {question_idx} and probe '{probe_name}'."
            )
        subset = subset.copy().sort_values(["layer_idx", "sentence_idx"])
        question = self.get_question(question_idx)
        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self._build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="sentence_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)
        sentence_indices = [int(idx) for idx in pivot.columns]
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
            question_idx, sentence_indices, probe_name, text_lookup
        )
        sentence_labels = [previews.get(int(idx), str(idx)) for idx in sentence_indices]
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            label_idx = self._to_category_idx(row.probe_ans)
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            lookup[(int(row.layer_idx), int(row.sentence_idx))] = {
                "argmax_label": class_labels.get(label_idx, str(row.probe_ans)),
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
            "Probe Predicted: %{customdata[0]}<br>"
            "Probe Pred Logits: %{customdata[1]}<extra></extra>"
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

    def get_heatmap_payload(
        self, question_idx: int, probe_name: str, view_mode: ViewMode = "token"
    ) -> HeatmapPayload:
        if view_mode == "token":
            return self._build_token_heatmap(question_idx, probe_name)
        if view_mode == "sentence":
            return self._build_sentence_heatmap(question_idx, probe_name)
        raise ValueError(f"Unsupported view mode: {view_mode}")

    def list_sentence_options(
        self, question_idx: int, probe_name: str
    ) -> List[Dict[str, int]]:
        try:
            df = self._load_sentence_df(question_idx)
        except FileNotFoundError:
            return []
        subset = df[df["early_decoder"] == probe_name]
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
        return [{"label": previews.get(val, f"Sentence {val}"), "value": int(val)} for val in values]

    def get_sentence_text(self, question_idx: int, sentence_idx: int) -> str:
        try:
            df = self._load_sentence_df(question_idx)
        except FileNotFoundError:
            return ""

        subset = df[df["sentence_idx"] == sentence_idx]
        if subset.empty:
            return ""

        if "sentence_text" in subset.columns:
            for raw_text in subset["sentence_text"].dropna().astype(str):
                collapsed = " ".join(raw_text.split())
                if collapsed:
                    return collapsed

        try:
            token_df = self._load_token_df(question_idx)
        except FileNotFoundError:
            return ""

        tokens = [str(tok) for tok in token_df[token_df["sentence_idx"] == sentence_idx]["token_text"]]
        collapsed_tokens = " ".join(token for token in tokens if token).strip()
        return collapsed_tokens

    def get_sentence_token_payload(
        self, question_idx: int, probe_name: str, sentence_idx: int
    ) -> HeatmapPayload:
        df = self._load_token_df(question_idx)
        subset = df[
            (df["early_decoder"] == probe_name)
            & (df["sentence_idx"] == sentence_idx)
        ]
        if subset.empty:
            raise ValueError(
                f"No token-level data for question {question_idx}, sentence {sentence_idx}, probe '{probe_name}'."
            )
        subset = subset.copy().sort_values(["layer_idx", "token_idx"])
        question = self.get_question(question_idx)
        sample_output = subset.iloc[0]["probe_output"]
        num_classes = len(sample_output)
        class_labels = self._build_class_labels(probe_name, num_classes, question)
        subset, colorscale, zmin, zmax, colorbar = self._prepare_heat_values(
            subset, class_labels, num_classes
        )
        pivot = subset.pivot(
            index="layer_idx", columns="token_idx", values="heat_value"
        ).sort_index().sort_index(axis=1)
        token_labels = self._collect_token_labels(subset)
        layer_labels = self._format_layer_labels(list(pivot.index))

        lookup: Dict[Tuple[int, int], Dict[str, str]] = {}
        for row in subset.itertuples(index=False):
            label_idx = self._to_category_idx(row.probe_ans)
            logits_str = ", ".join(f"{float(val):.3f}" for val in row.probe_output)
            lookup[(int(row.layer_idx), int(row.token_idx))] = {
                "token_pos": str(int(row.token_idx)),
                "token_text": str(row.token_text) if row.token_text else "␀",
                "argmax_label": class_labels.get(label_idx, str(row.probe_ans)),
                "logits": f"[{logits_str}]",
            }
        customdata = self._build_customdata(
            list(pivot.index),
            list(pivot.columns),
            lookup,
            ["token_pos", "token_text", "argmax_label", "logits"],
        )

        hover_template = (
            "Layer: %{y}<br>"
            "Token Pos: %{customdata[0]}<br>"
            "Token: %{customdata[1]}<br>"
            "Probe Predicted: %{customdata[2]}<br>"
            "Probe Pred Logits: %{customdata[3]}<extra></extra>"
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
