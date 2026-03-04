"""Analyze correlation between probe prediction shifts and LLM-detected inflection points."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WINDOW_STEPS = [1, 2, 4, 10, 20, 50, 100]
INFLECTION_TYPES = ["backtrack", "realization", "reconsideration"]
COLORBAR_VMAX = 0.5
LOG2_RATIO_VMAX = 2.0  # log2 scale: [-2, 2] = [0.25x, 4x]


# --- Metrics: each maps a probability vector to a scalar ---

def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def top_prob(probs: np.ndarray) -> float:
    """Probability of the highest-confidence class."""
    return float(np.max(probs))


METRICS: dict[str, dict] = {
    "entropy": {
        "fn": entropy,
        "thresholds": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
        "label": "Entropy",
    },
    "top_prob": {
        "fn": top_prob,
        "thresholds": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
        "label": "Top-Class Prob",
    },
}


# --- Core computation ---

def parse_probs(output_str) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(output_str))
    except Exception:
        return np.array([0.25, 0.25, 0.25, 0.25])


def get_predictions(df: pd.DataFrame, layer_idx: int) -> Dict[int, np.ndarray]:
    """Get step_idx -> probability array for a given layer."""
    subset = df[df["layer_idx"] == layer_idx]
    preds = {}
    for _, row in subset.iterrows():
        preds[int(row["step_idx"])] = parse_probs(row["decoder_output"])
    return preds


def get_shift_steps(
    preds: Dict[int, np.ndarray], metric_fn: Callable, threshold: float,
    direction: str = "any",
) -> set:
    """Find steps where a metric changes by more than threshold.

    Args:
        direction: "any" (absolute change), "up" (increase only), "down" (decrease only).
    """
    sorted_steps = sorted(preds.keys())
    shifts = set()
    for i in range(1, len(sorted_steps)):
        prev, curr = sorted_steps[i - 1], sorted_steps[i]
        delta = metric_fn(preds[curr]) - metric_fn(preds[prev])
        if direction == "up" and delta > threshold:
            shifts.add(curr)
        elif direction == "down" and delta < -threshold:
            shifts.add(curr)
        elif direction == "any" and abs(delta) > threshold:
            shifts.add(curr)
    return shifts


def load_data(
    inflection_path: Path,
    step_level_dir: Path,
    metadata_path: Path,
    layer_idx: int,
) -> Dict[int, dict]:
    """Load and join inflection results with step-level probe data."""
    with inflection_path.open("r", encoding="utf-8") as f:
        inflection_data = json.load(f)

    metadata_df = pd.read_csv(metadata_path)
    hash_to_idx = dict(zip(metadata_df["question_hash"].astype(str), metadata_df["question_idx"].astype(int)))
    idx_to_hash = {v: k for k, v in hash_to_idx.items()}

    data = {}
    for result in inflection_data["results"]:
        qhash = result.get("question_hash")
        if not qhash:
            q_idx = result.get("question_idx")
            if q_idx is None:
                continue
            qhash = idx_to_hash.get(int(q_idx))
            if not qhash:
                continue

        csv_path = step_level_dir / f"{qhash}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        preds = get_predictions(df, layer_idx)
        if not preds:
            continue

        q_idx = hash_to_idx.get(qhash, -1)
        data[q_idx] = {"preds": preds, "inflection_info": result}

    logger.info(f"Loaded {len(data)} questions with step-level data")
    return data


def compute_correlation(
    data: Dict[int, dict],
    metric_fn: Callable,
    threshold: float,
    window_steps: int,
    inflection_type: Optional[str] = None,
    direction: str = "any",
) -> dict:
    """Compute conditional probability differences between probe shifts and inflections.

    Left panel (Probe Shift → Inflection): forward — does an inflection follow this shift?
    Right panel (Inflection → Probe Shift): forward — does a shift follow this inflection?
    """
    s_with_inf, s_without_inf = 0, 0
    no_s_with_inf, no_s_without_inf = 0, 0
    inf_with_s, inf_without_s = 0, 0
    no_inf_with_s, no_inf_without_s = 0, 0

    for q_idx, entry in data.items():
        preds = entry["preds"]
        info = entry["inflection_info"]
        sorted_steps = sorted(preds.keys())

        shift_steps = get_shift_steps(preds, metric_fn, threshold, direction)

        inflections = info.get("inflections", [])
        if inflection_type:
            inflections = [i for i in inflections if i.get("inflection_type") == inflection_type]
        inf_steps = {i["step_number"] for i in inflections}

        for step in sorted_steps:
            is_shift = step in shift_steps
            is_inf = step in inf_steps
            has_inf_ahead = any(0 < s - step <= window_steps for s in inf_steps)
            has_shift_ahead = any(0 < s - step <= window_steps for s in shift_steps)

            # Left panel: Probe Shift → Inflection (forward)
            if is_shift:
                if has_inf_ahead:
                    s_with_inf += 1
                else:
                    s_without_inf += 1
            else:
                if has_inf_ahead:
                    no_s_with_inf += 1
                else:
                    no_s_without_inf += 1

            # Right panel: Inflection → Probe Shift (forward)
            if is_inf:
                if has_shift_ahead:
                    inf_with_s += 1
                else:
                    inf_without_s += 1
            else:
                if has_shift_ahead:
                    no_inf_with_s += 1
                else:
                    no_inf_without_s += 1

    total_s = s_with_inf + s_without_inf
    total_no_s = no_s_with_inf + no_s_without_inf
    total_inf = inf_with_s + inf_without_s
    total_no_inf = no_inf_with_s + no_inf_without_s

    p_inf_s = s_with_inf / total_s if total_s else 0
    p_inf_no_s = no_s_with_inf / total_no_s if total_no_s else 0
    p_s_inf = inf_with_s / total_inf if total_inf else 0
    p_s_no_inf = no_inf_with_s / total_no_inf if total_no_inf else 0

    def _safe_ratio(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "diff_shift_to_inf": p_inf_s - p_inf_no_s,
        "diff_inf_to_shift": p_s_inf - p_s_no_inf,
        "ratio_shift_to_inf": _safe_ratio(p_inf_s, p_inf_no_s),
        "ratio_inf_to_shift": _safe_ratio(p_s_inf, p_s_no_inf),
        "n_hits_shift": s_with_inf,
        "n_shifts": total_s,
        "n_hits_no_shift": no_s_with_inf,
        "n_no_shifts": total_no_s,
        "n_hits_inf": inf_with_s,
        "n_inflections": total_inf,
        "n_hits_no_inf": no_inf_with_s,
        "n_no_inflections": total_no_inf,
    }


def compute_cooccurrence(
    data: Dict[int, dict],
    metric_fn: Callable,
    threshold: float,
    window_steps: int,
    inflection_type: Optional[str] = None,
    direction: str = "any",
) -> dict:
    """Compute co-occurrence: symmetric window (either side).

    Left panel: given a shift, is there an inflection within ±window?
    Right panel: given an inflection, is there a shift within ±window?
    """
    s_with_inf, s_without_inf = 0, 0
    no_s_with_inf, no_s_without_inf = 0, 0
    inf_with_s, inf_without_s = 0, 0
    no_inf_with_s, no_inf_without_s = 0, 0

    for q_idx, entry in data.items():
        preds = entry["preds"]
        info = entry["inflection_info"]
        sorted_steps = sorted(preds.keys())

        shift_steps = get_shift_steps(preds, metric_fn, threshold, direction)

        inflections = info.get("inflections", [])
        if inflection_type:
            inflections = [i for i in inflections if i.get("inflection_type") == inflection_type]
        inf_steps = {i["step_number"] for i in inflections}

        for step in sorted_steps:
            is_shift = step in shift_steps
            is_inf = step in inf_steps
            has_inf_nearby = any(0 < abs(s - step) <= window_steps for s in inf_steps)
            has_shift_nearby = any(0 < abs(s - step) <= window_steps for s in shift_steps)

            if is_shift:
                if has_inf_nearby:
                    s_with_inf += 1
                else:
                    s_without_inf += 1
            else:
                if has_inf_nearby:
                    no_s_with_inf += 1
                else:
                    no_s_without_inf += 1

            if is_inf:
                if has_shift_nearby:
                    inf_with_s += 1
                else:
                    inf_without_s += 1
            else:
                if has_shift_nearby:
                    no_inf_with_s += 1
                else:
                    no_inf_without_s += 1

    total_s = s_with_inf + s_without_inf
    total_no_s = no_s_with_inf + no_s_without_inf
    total_inf = inf_with_s + inf_without_s
    total_no_inf = no_inf_with_s + no_inf_without_s

    p_inf_s = s_with_inf / total_s if total_s else 0
    p_inf_no_s = no_s_with_inf / total_no_s if total_no_s else 0
    p_s_inf = inf_with_s / total_inf if total_inf else 0
    p_s_no_inf = no_inf_with_s / total_no_inf if total_no_inf else 0

    def _safe_ratio(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "diff_shift_to_inf": p_inf_s - p_inf_no_s,
        "diff_inf_to_shift": p_s_inf - p_s_no_inf,
        "ratio_shift_to_inf": _safe_ratio(p_inf_s, p_inf_no_s),
        "ratio_inf_to_shift": _safe_ratio(p_s_inf, p_s_no_inf),
        "n_hits_shift": s_with_inf,
        "n_shifts": total_s,
        "n_hits_no_shift": no_s_with_inf,
        "n_no_shifts": total_no_s,
        "n_hits_inf": inf_with_s,
        "n_inflections": total_inf,
        "n_hits_no_inf": no_inf_with_s,
        "n_no_inflections": total_no_inf,
    }


def sweep(
    data: Dict[int, dict], metric_name: str,
    inflection_type: Optional[str] = None, direction: str = "any",
    mode: str = "causal",
) -> pd.DataFrame:
    """Sweep over thresholds and window sizes for a given metric.

    Args:
        mode: "causal" (forward-looking) or "cooccurrence" (symmetric ±window).
    """
    metric = METRICS[metric_name]
    compute_fn = compute_cooccurrence if mode == "cooccurrence" else compute_correlation
    rows = []
    for thresh in metric["thresholds"]:
        for window in WINDOW_STEPS:
            result = compute_fn(data, metric["fn"], thresh, window, inflection_type, direction)
            result["threshold"] = thresh
            result["window_steps"] = window
            rows.append(result)
    return pd.DataFrame(rows)


# --- Plotting ---

def annotate_heatmap(ax, pivot, n_pivots=None, fmt="+.2f", raw_pivot=None):
    """Add value and optional N annotations to each heatmap cell.

    Args:
        n_pivots: List of (hits_pivot, total_pivot) tuples, displayed as "hits/total".
        raw_pivot: If provided, use these values for annotation text instead of pivot
            (useful for showing raw ratios while plotting log-transformed values).
    """
    display = raw_pivot if raw_pivot is not None else pivot
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = display.values[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = f"{val:{fmt}}"
            if n_pivots:
                parts = []
                for hits_pivot, total_pivot in n_pivots:
                    h = int(hits_pivot.values[i, j])
                    t = int(total_pivot.values[i, j])
                    parts.append(f"{h}/{t}")
                ax.text(j, i, text, ha="center", va="center", fontsize=6)
                ax.text(j, i + 0.3, ", ".join(parts), ha="center", va="center", fontsize=4.5, color="0.3")
            else:
                ax.text(j, i, text, ha="center", va="center", fontsize=6)


def render_heatmap(ax, pivot, stat, n_pivots=None):
    """Render a heatmap with appropriate scale and annotation for diff or ratio stat."""
    if stat == "ratio":
        log2_pivot = pivot.apply(lambda x: np.log2(x.replace(0, np.nan)))
        im = ax.imshow(log2_pivot.values, cmap="RdBu", aspect="auto",
                        vmin=-LOG2_RATIO_VMAX, vmax=LOG2_RATIO_VMAX)
        annotate_heatmap(ax, log2_pivot, n_pivots=n_pivots, fmt=".2f", raw_pivot=pivot)
    else:
        im = ax.imshow(pivot.values, cmap="RdBu", aspect="auto",
                        vmin=-COLORBAR_VMAX, vmax=COLORBAR_VMAX)
        annotate_heatmap(ax, pivot, n_pivots=n_pivots, fmt="+.2f")
    cbar_label = "log₂(ratio)" if stat == "ratio" else "Difference"
    plt.colorbar(im, ax=ax, label=cbar_label)


def setup_heatmap_axes(ax, title, thresholds):
    """Set standard tick labels and axis labels for a heatmap."""
    ax.set_xticks(range(len(WINDOW_STEPS)))
    ax.set_xticklabels(WINDOW_STEPS)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels(thresholds)
    ax.set_xlabel("Window Steps")
    ax.set_ylabel("Threshold")
    ax.set_title(title)


def plot_heatmap_pair(
    sweep_df: pd.DataFrame,
    thresholds: list,
    title: str,
    stat: str = "diff",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 2-panel heatmap for a single metric. stat='diff' or 'ratio'."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prefix = stat  # "diff" or "ratio"

    def _pivot(col):
        return sweep_df.pivot(index="threshold", columns="window_steps", values=col)

    # Left panel: Probe Shift → Inflection (forward).
    pivot = _pivot(f"{prefix}_shift_to_inf")
    ax = axes[0]
    setup_heatmap_axes(ax, "P(Inflection follows | Probe Shift) / P(Inflection follows | No Probe Shift)" if stat == "ratio"
        else "P(Inflection follows | Probe Shift) - P(Inflection follows | No Probe Shift)", thresholds)
    render_heatmap(ax, pivot, stat, n_pivots=[
        (_pivot("n_hits_shift"), _pivot("n_shifts")),
        (_pivot("n_hits_no_shift"), _pivot("n_no_shifts")),
    ])

    # Right panel: Inflection → Probe Shift (forward).
    pivot = _pivot(f"{prefix}_inf_to_shift")
    ax = axes[1]
    setup_heatmap_axes(ax, "P(Probe Shift follows | Inflection) / P(Probe Shift follows | No Inflection)" if stat == "ratio"
        else "P(Probe Shift follows | Inflection) - P(Probe Shift follows | No Inflection)", thresholds)
    render_heatmap(ax, pivot, stat, n_pivots=[
        (_pivot("n_hits_inf"), _pivot("n_inflections")),
        (_pivot("n_hits_no_inf"), _pivot("n_no_inflections")),
    ])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    return fig


def plot_per_type_heatmaps(
    data: Dict[int, dict],
    metric_name: str,
    title: str,
    stat: str = "diff",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot per-inflection-type heatmaps (2 rows x 3 columns). stat='diff' or 'ratio'."""
    metric = METRICS[metric_name]
    thresholds = metric["thresholds"]
    prefix = stat
    op = "/" if stat == "ratio" else "-"

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    type_sweeps = {t: sweep(data, metric_name, inflection_type=t) for t in INFLECTION_TYPES}

    for col, inf_type in enumerate(INFLECTION_TYPES):
        sweep_df = type_sweeps[inf_type]

        def _pivot(c):
            return sweep_df.pivot(index="threshold", columns="window_steps", values=c)

        # Row 0: Probe Shift → Inflection.
        pivot = _pivot(f"{prefix}_shift_to_inf")
        ax = axes[0, col]
        setup_heatmap_axes(
            ax,
            f"Probe Shift → {inf_type}\nP({inf_type} follows | Probe Shift) {op} P({inf_type} follows | No Probe Shift)",
            thresholds,
        )
        render_heatmap(ax, pivot, stat, n_pivots=[
            (_pivot("n_hits_shift"), _pivot("n_shifts")),
            (_pivot("n_hits_no_shift"), _pivot("n_no_shifts")),
        ])

        # Row 1: Inflection → Probe Shift.
        pivot = _pivot(f"{prefix}_inf_to_shift")
        ax = axes[1, col]
        setup_heatmap_axes(
            ax,
            f"{inf_type} → Probe Shift\nP(Probe Shift follows | {inf_type}) {op} P(Probe Shift follows | No {inf_type})",
            thresholds,
        )
        render_heatmap(ax, pivot, stat, n_pivots=[
            (_pivot("n_hits_inf"), _pivot("n_inflections")),
            (_pivot("n_hits_no_inf"), _pivot("n_no_inflections")),
        ])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    return fig


def plot_cooccurrence_by_type(
    data: Dict[int, dict],
    metric_name: str,
    title: str,
    stat: str = "diff",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot co-occurrence heatmaps (symmetric ±window) per inflection type (2 rows x 3 cols)."""
    metric = METRICS[metric_name]
    thresholds = metric["thresholds"]
    prefix = stat
    op = "/" if stat == "ratio" else "-"

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    type_sweeps = {t: sweep(data, metric_name, inflection_type=t, mode="cooccurrence") for t in INFLECTION_TYPES}

    for col, inf_type in enumerate(INFLECTION_TYPES):
        sweep_df = type_sweeps[inf_type]

        def _pivot(c):
            return sweep_df.pivot(index="threshold", columns="window_steps", values=c)

        # Row 0: Given a shift, is there an inflection within ±window?
        pivot = _pivot(f"{prefix}_shift_to_inf")
        ax = axes[0, col]
        setup_heatmap_axes(
            ax,
            f"Probe Shift ↔ {inf_type}\nP({inf_type} nearby | Shift) {op} P({inf_type} nearby | No Shift)",
            thresholds,
        )
        render_heatmap(ax, pivot, stat, n_pivots=[
            (_pivot("n_hits_shift"), _pivot("n_shifts")),
            (_pivot("n_hits_no_shift"), _pivot("n_no_shifts")),
        ])

        # Row 1: Given an inflection, is there a shift within ±window?
        pivot = _pivot(f"{prefix}_inf_to_shift")
        ax = axes[1, col]
        setup_heatmap_axes(
            ax,
            f"{inf_type} ↔ Probe Shift\nP(Shift nearby | {inf_type}) {op} P(Shift nearby | No {inf_type})",
            thresholds,
        )
        render_heatmap(ax, pivot, stat, n_pivots=[
            (_pivot("n_hits_inf"), _pivot("n_inflections")),
            (_pivot("n_hits_no_inf"), _pivot("n_no_inflections")),
        ])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    return fig


# --- High-confidence vs inflection analysis ---

HIGH_CONFIDENCE_THRESHOLD = 0.9


def classify_confidence(data: Dict[int, dict]) -> Dict[int, bool]:
    """Classify each question as high-confidence or not.

    High-confidence: the probe's probability on its final-step predicted class
    starts at ≥ HIGH_CONFIDENCE_THRESHOLD and never drops below it.
    """
    labels = {}
    for q_idx, entry in data.items():
        preds = entry["preds"]
        sorted_steps = sorted(preds.keys())
        if not sorted_steps:
            labels[q_idx] = False
            continue

        # Final answer = argmax at last step
        final_probs = preds[sorted_steps[-1]]
        final_class = int(np.argmax(final_probs))

        # Check if probability of that class stays ≥ threshold throughout
        high = True
        for step in sorted_steps:
            if preds[step][final_class] < HIGH_CONFIDENCE_THRESHOLD:
                high = False
                break
        labels[q_idx] = high
    return labels


def compute_confidence_inflection_stats(data: Dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute inflection counts and per-step rates by confidence level.

    Returns (counts_df, rates_df) with columns:
        [, Traces, Steps, Backtrack, Realization, Reconsideration, Total]
    """
    confidence_labels = classify_confidence(data)

    # Gather per-group stats
    group_stats = {}
    for label in ["High Confidence", "Other", "Total"]:
        n_traces, n_steps = 0, 0
        type_counts = {t: 0 for t in INFLECTION_TYPES}
        for q_idx, entry in data.items():
            is_high = confidence_labels.get(q_idx, False)
            if label == "High Confidence" and not is_high:
                continue
            if label == "Other" and is_high:
                continue
            n_traces += 1
            n_steps += len(entry["preds"])
            for inf in entry["inflection_info"].get("inflections", []):
                itype = inf.get("inflection_type")
                if itype in type_counts:
                    type_counts[itype] += 1
        group_stats[label] = (n_traces, n_steps, type_counts)

    # Build raw counts table
    count_rows = []
    for label in ["High Confidence", "Other", "Total"]:
        n_traces, n_steps, tc = group_stats[label]
        total = sum(tc.values())
        count_rows.append({
            "": label, "Traces": n_traces, "Steps": n_steps,
            "Backtrack": tc["backtrack"], "Realization": tc["realization"],
            "Reconsideration": tc["reconsideration"], "Total": total,
        })
    counts_df = pd.DataFrame(count_rows)

    # Build per-step rates table
    rate_rows = []
    for label in ["High Confidence", "Other", "Total"]:
        n_traces, n_steps, tc = group_stats[label]
        total = sum(tc.values())
        def fmt_rate(count):
            return f"{count / n_steps:.4f}" if n_steps else "—"
        rate_rows.append({
            "": label, "Traces": n_traces, "Steps": n_steps,
            "Backtrack": fmt_rate(tc["backtrack"]),
            "Realization": fmt_rate(tc["realization"]),
            "Reconsideration": fmt_rate(tc["reconsideration"]),
            "Total": fmt_rate(total),
        })
    rates_df = pd.DataFrame(rate_rows)

    col_order = ["", "Traces", "Steps", "Backtrack", "Realization", "Reconsideration", "Total"]
    return counts_df[col_order], rates_df[col_order]


def style_table(table, n_rows, n_cols):
    """Apply consistent styling to a matplotlib table."""
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(n_cols):
        table[0, j].set_facecolor("#d9e2f3")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(1, n_rows + 1):
        table[i, 0].set_text_props(fontweight="bold")
        color = "#f2f2f2" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(color)
    for j in range(n_cols):
        table[n_rows, j].set_facecolor("#e6e6e6")


def plot_confidence_inflection_table(
    data: Dict[int, dict],
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot tables showing inflection counts and per-step rates by confidence level."""
    counts_df, rates_df = compute_confidence_inflection_stats(data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 3.8))

    # Top: raw counts
    ax1.axis("off")
    ax1.set_title("Raw Inflection Counts", fontsize=10, fontweight="bold", pad=8)
    t1 = ax1.table(cellText=counts_df.values, colLabels=counts_df.columns, cellLoc="center", loc="center")
    style_table(t1, len(counts_df), len(counts_df.columns))

    # Bottom: per-step rates
    ax2.axis("off")
    ax2.set_title("Inflections per Step", fontsize=10, fontweight="bold", pad=8)
    t2 = ax2.table(cellText=rates_df.values, colLabels=rates_df.columns, cellLoc="center", loc="center")
    style_table(t2, len(rates_df), len(rates_df.columns))

    fig.suptitle(
        f"{title}\n(High Confidence = probe ≥ {HIGH_CONFIDENCE_THRESHOLD:.0%} on final answer throughout)",
        fontsize=11, y=1.05,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    return fig


# --- Bar chart: P(shift nearby | inflection type) ---

BAR_WINDOW_STEPS = 10
BAR_THRESHOLD = 0.2

BAR_COLORS = {
    "backtrack": "#D32F2F",
    "realization": "#1976D2",
    "reconsideration": "#388E3C",
}


N_BOOTSTRAP = 10_000


def cluster_bootstrap_ci(
    outcomes: list[tuple[int, int]], n_boot: int = N_BOOTSTRAP,
) -> tuple[float, float, float, int]:
    """Proportion and 95% CI via cluster bootstrap (resampling by question).

    Args:
        outcomes: list of (q_idx, 0/1) tuples.

    Returns:
        (proportion, ci_lo, ci_hi, n_observations)
    """
    if not outcomes:
        return 0.0, 0.0, 0.0, 0

    by_question: dict[int, list[int]] = {}
    for q_idx, hit in outcomes:
        by_question.setdefault(q_idx, []).append(hit)

    q_ids = list(by_question.keys())
    n_questions = len(q_ids)
    n_obs = len(outcomes)

    p = sum(h for _, h in outcomes) / n_obs

    rng = np.random.default_rng(42)
    boot_props = np.empty(n_boot)
    for b in range(n_boot):
        sampled_qs = rng.choice(q_ids, size=n_questions, replace=True)
        total, hits = 0, 0
        for q in sampled_qs:
            obs = by_question[q]
            total += len(obs)
            hits += sum(obs)
        boot_props[b] = hits / total if total > 0 else 0.0

    lo = float(np.percentile(boot_props, 2.5))
    hi = float(np.percentile(boot_props, 97.5))
    return p, lo, hi, n_obs


def collect_shift_to_inf_outcomes(
    data: Dict[int, dict],
    metric_name: str = "top_prob",
    threshold: float = BAR_THRESHOLD,
    window_steps: int = BAR_WINDOW_STEPS,
) -> tuple[dict, dict]:
    """Given a shift step, does an inflection follow within the window?

    Returns (type_yes, type_no):
        type_yes[t]: for each shift step, did an inflection of type t follow?
        type_no[t]: for each non-shift step, did an inflection of type t follow?
    """
    metric_fn = METRICS[metric_name]["fn"]
    type_yes = {t: [] for t in INFLECTION_TYPES}
    type_no = {t: [] for t in INFLECTION_TYPES}

    for q_idx, entry in data.items():
        preds = entry["preds"]
        sorted_steps = sorted(preds.keys())
        shift_steps = get_shift_steps(preds, metric_fn, threshold)

        inf_by_type = {t: set() for t in INFLECTION_TYPES}
        for inf in entry["inflection_info"].get("inflections", []):
            itype = inf.get("inflection_type")
            if itype in inf_by_type:
                inf_by_type[itype].add(inf["step_number"])

        for step in sorted_steps:
            is_shift = step in shift_steps
            for t in INFLECTION_TYPES:
                hit = 1 if any(0 <= s - step <= window_steps for s in inf_by_type[t]) else 0
                if is_shift:
                    type_yes[t].append((q_idx, hit))
                else:
                    type_no[t].append((q_idx, hit))

    return type_yes, type_no


def collect_window_inf_outcomes(
    data: Dict[int, dict],
    metric_name: str = "top_prob",
    threshold: float = BAR_THRESHOLD,
    window_steps: int = BAR_WINDOW_STEPS,
) -> tuple[dict, dict]:
    """Given a shift-step window, does an inflection occur? Baseline excludes windows with any shift.

    Returns (type_yes, type_no):
        type_yes[t]: for each window starting at a shift step, does an inflection of type t occur?
        type_no[t]: for each window with NO shift inside it, does an inflection of type t occur?
    """
    metric_fn = METRICS[metric_name]["fn"]
    type_yes = {t: [] for t in INFLECTION_TYPES}
    type_no = {t: [] for t in INFLECTION_TYPES}

    for q_idx, entry in data.items():
        preds = entry["preds"]
        sorted_steps = sorted(preds.keys())
        shift_steps = get_shift_steps(preds, metric_fn, threshold)

        inf_by_type = {t: set() for t in INFLECTION_TYPES}
        for inf in entry["inflection_info"].get("inflections", []):
            itype = inf.get("inflection_type")
            if itype in inf_by_type:
                inf_by_type[itype].add(inf["step_number"])

        for step in sorted_steps:
            window_end = step + window_steps

            if step in shift_steps:
                for t in INFLECTION_TYPES:
                    hit = 1 if any(step <= s <= window_end for s in inf_by_type[t]) else 0
                    type_yes[t].append((q_idx, hit))
            else:
                has_shift_in_window = any(
                    step <= s <= window_end for s in shift_steps
                )
                if not has_shift_in_window:
                    for t in INFLECTION_TYPES:
                        hit = 1 if any(step <= s <= window_end for s in inf_by_type[t]) else 0
                        type_no[t].append((q_idx, hit))

    return type_yes, type_no


def collect_inf_to_shift_outcomes(
    data: Dict[int, dict],
    metric_name: str = "top_prob",
    threshold: float = BAR_THRESHOLD,
    window_steps: int = BAR_WINDOW_STEPS,
) -> tuple[dict, dict]:
    """Given an inflection step, does a probe shift occur within the window?

    Returns (type_yes, type_no):
        type_yes[t]: for each step with an inflection of type t, does a probe shift occur
                     within [step, step + window_steps]?
        type_no[t]: for each step with no inflection (and none within the window),
                    does a probe shift occur?
    """
    metric_fn = METRICS[metric_name]["fn"]
    type_yes = {t: [] for t in INFLECTION_TYPES}
    type_no = {t: [] for t in INFLECTION_TYPES}

    for q_idx, entry in data.items():
        preds = entry["preds"]
        sorted_steps = sorted(preds.keys())
        shift_steps = get_shift_steps(preds, metric_fn, threshold)

        inf_by_type = {t: set() for t in INFLECTION_TYPES}
        all_inf_steps = set()
        for inf in entry["inflection_info"].get("inflections", []):
            itype = inf.get("inflection_type")
            if itype in inf_by_type:
                inf_by_type[itype].add(inf["step_number"])
                all_inf_steps.add(inf["step_number"])

        for step in sorted_steps:
            window_end = step + window_steps
            has_shift = 1 if any(step <= s <= window_end for s in shift_steps) else 0

            for t in INFLECTION_TYPES:
                if step in inf_by_type[t]:
                    type_yes[t].append((q_idx, has_shift))
                else:
                    has_any_inf_in_window = any(
                        step <= s <= window_end for s in all_inf_steps
                    )
                    if not has_any_inf_in_window:
                        type_no[t].append((q_idx, has_shift))

    return type_yes, type_no


def plot_bars(
    type_yes: dict,
    type_no: dict,
    title: str,
    ylabel: str,
    legend_labels: tuple[str, str] = ("Given Probe Shift", "Given No Probe Shift"),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Paired bar chart with cluster-bootstrap 95% CIs."""
    import seaborn as sns
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(INFLECTION_TYPES))

    for offset, (group, alpha) in enumerate([(type_yes, 0.85), (type_no, 0.4)]):
        props, err_lo, err_hi = [], [], []
        for t in INFLECTION_TYPES:
            p, lo, hi, n = cluster_bootstrap_ci(group[t])
            props.append(p)
            err_lo.append(p - lo)
            err_hi.append(hi - p)

        positions = x + (offset - 0.5) * bar_width
        colors = [BAR_COLORS[t] for t in INFLECTION_TYPES]
        ax.bar(
            positions, props, bar_width,
            yerr=[err_lo, err_hi], capsize=5,
            color=colors, alpha=alpha, edgecolor="white", linewidth=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in INFLECTION_TYPES], fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax)

    legend_elements = [
        Patch(facecolor="gray", alpha=0.85, label=legend_labels[0]),
        Patch(facecolor="gray", alpha=0.4, label=legend_labels[1]),
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc="upper right")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved {save_path}")

    return fig


# --- Entry points ---

def run_inflection_analysis(
    inflection_path: Path,
    step_level_dir: Path,
    metadata_path: Path,
    layer_idx: int,
    model_name: str = "",
    dataset_name: str = "",
    plots_dir_override: Path | None = None,
) -> None:
    """Run the full inflection point analysis for all metrics and save heatmaps."""
    if plots_dir_override is not None:
        plots_dir = plots_dir_override
    else:
        # Derive plots dir: step_level_dir is results/<run>/step_level → plots/<run>/
        run_root = step_level_dir.parent
        plots_dir = run_root.parent.parent / "plots" / run_root.name
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(inflection_path, step_level_dir, metadata_path, layer_idx)
    if not data:
        logger.warning("No data to analyze")
        return

    subtitle = f"({model_name} on {dataset_name}, Probe L{layer_idx})" if model_name else f"(Probe L{layer_idx})"

    metric_name = "top_prob"
    metric = METRICS[metric_name]
    stat = "ratio"

    sweep_df = sweep(data, metric_name)
    plot_heatmap_pair(
        sweep_df,
        thresholds=metric["thresholds"],
        title=f"Probe Shift and Inflection Point Correlation (Ratio)\n{subtitle}",
        stat=stat,
        save_path=plots_dir / f"{metric_name}_inflection_correlation.pdf",
    )
    plot_per_type_heatmaps(
        data,
        metric_name,
        title=f"Probe Shift and Inflection Point Correlation by Type (Ratio)\n{subtitle}",
        stat=stat,
        save_path=plots_dir / f"{metric_name}_inflection_by_type.pdf",
    )
    plot_cooccurrence_by_type(
        data,
        metric_name,
        title=f"Probe Shift and Inflection Point Co-occurrence by Type (Ratio)\n{subtitle}",
        stat=stat,
        save_path=plots_dir / f"{metric_name}_inflection_cooccurrence.pdf",
    )
    bar_params = f"window={BAR_WINDOW_STEPS}, threshold={BAR_THRESHOLD}"
    bar_configs = [
        (collect_shift_to_inf_outcomes, "P(Inflection Follows)", "shift_to_inf_bars",
         ("Given Probe Shift", "Given No Probe Shift")),
        (collect_window_inf_outcomes, "P(Inflection in Window)", "window_inf_bars",
         ("Given Probe Shift", "Given No Probe Shift")),
        (collect_inf_to_shift_outcomes, "P(Probe Shift in Window)", "inf_to_shift_bars",
         ("Given Inflection", "Given No Inflection")),
    ]
    for collect_fn, ylabel, bar_file, legend_labels in bar_configs:
        type_yes, type_no = collect_fn(data, metric_name)
        plot_bars(
            type_yes, type_no,
            title=f"{ylabel}\n{subtitle}, {bar_params}",
            ylabel=ylabel,
            legend_labels=legend_labels,
            save_path=plots_dir / f"{metric_name}_{bar_file}.pdf",
        )
    plt.close("all")

    plot_confidence_inflection_table(
        data,
        title=f"Inflection Points by Probe Confidence Level\n{subtitle}",
        save_path=plots_dir / "confidence_inflection_table.pdf",
    )

    plt.close("all")
    logger.info(f"Analysis complete. Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze probe shift / inflection correlations")
    parser.add_argument("--inflection-path", type=Path, required=True, help="Path to inflection_results.json")
    parser.add_argument("--step-level-dir", type=Path, required=True, help="Path to step_level/ directory")
    parser.add_argument("--metadata-path", type=Path, required=True, help="Path to predictions_metadata.csv")
    parser.add_argument("--layer", type=int, required=True, help="Probe layer to analyze")
    parser.add_argument("--model-name", type=str, default="", help="Model name for plot titles")
    parser.add_argument("--dataset-name", type=str, default="", help="Dataset name for plot titles")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Override output directory for plots")
    args = parser.parse_args()

    run_inflection_analysis(
        inflection_path=args.inflection_path,
        step_level_dir=args.step_level_dir,
        metadata_path=args.metadata_path,
        layer_idx=args.layer,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        plots_dir_override=args.plots_dir,
    )


if __name__ == "__main__":
    main()
