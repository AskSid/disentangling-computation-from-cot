"""Visualization helpers for the probe viewer."""

from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go

import math

from data_loader import HeatmapPayload, DecoderStepPayload, cycle_palette


def build_heatmap_figure(
    payload: HeatmapPayload,
    show_x_labels: bool = True,
    theme_config: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Return a Plotly heatmap for the provided payload."""

    z = payload.pivot.values.astype(float)
    layer_labels = payload.layer_labels
    x_labels = payload.x_labels
    hovertemplate = payload.hover_template

    heatmap = go.Heatmap(
        z=z,
        x=x_labels,
        y=layer_labels,
        colorscale=payload.colorscale,
        zmin=payload.zmin,
        zmax=payload.zmax,
        customdata=payload.customdata,
        hovertemplate=hovertemplate,
        colorbar=payload.colorbar,
    )

    fig = go.Figure(data=[heatmap])

    predicted_idx: Optional[int] = None
    if payload.customdata:
        sample_cell: Optional[Sequence[str]] = None
        for row in payload.customdata:
            if row:
                sample_cell = row[0]
                break
        if sample_cell:
            if payload.view_mode == "sentence" and len(sample_cell) >= 1:
                predicted_idx = 0
            elif len(sample_cell) >= 4:
                predicted_idx = 3

    white_overlay: list[list[int]] = []
    has_na_prediction = False
    if predicted_idx is not None:
        for row in payload.customdata:
            overlay_row: list[int] = []
            for cell in row:
                predicted_value = ""
                if isinstance(cell, (list, tuple)) and len(cell) > predicted_idx:
                    predicted_value = str(cell[predicted_idx]).strip()
                is_na = predicted_value.upper() == "N/A"
                overlay_row.append(1 if is_na else 0)
                if is_na:
                    has_na_prediction = True
            white_overlay.append(overlay_row)

    if has_na_prediction:
        overlay_trace = go.Heatmap(
            z=white_overlay,
            x=x_labels,
            y=layer_labels,
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [0.5, "rgba(0,0,0,0)"],
                [1.0, "#FFFFFF"],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            hoverinfo="skip",
            hovertemplate=None,
        )
        fig.add_trace(overlay_trace)

    xaxis_config = dict(
        title=dict(text=payload.x_axis_title, font=dict(size=18)),
        tickangle=45,
        automargin=True,
        tickfont=dict(size=14),
    )
    axis_title_lower = str(payload.x_axis_title).lower()
    if show_x_labels:
        if "sentence" in axis_title_lower or payload.view_mode == "sentence":
            xaxis_config.update(
                tickmode="array",
                tickvals=x_labels,
                ticktext=x_labels,
            )
        else:
            xaxis_config.update(
                tickmode="array",
                tickvals=x_labels,
                ticktext=x_labels,
            )
    else:
        xaxis_config["showticklabels"] = False

    yaxis_title = "Decoder Type" if payload.view_mode == "sentence" else "Layer"

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=50, r=20, b=120, l=80),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=18)),
            tickmode="array",
            tickvals=list(range(len(layer_labels))),
            ticktext=layer_labels,
            tickfont=dict(size=14),
        ),
        height=1050,
    )

    def _hex_luminance(hex_color: str) -> float:
        if not isinstance(hex_color, str) or not hex_color.startswith("#"):
            return 1.0
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return 1.0
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
        except ValueError:
            return 1.0
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    boundary_label = None
    if isinstance(payload, dict):
        boundary_label = payload.get("prompt_boundary_label")
        view_mode = payload.get("view_mode")
    else:
        boundary_label = getattr(payload, "prompt_boundary_label", None)
        view_mode = getattr(payload, "view_mode", None)

    if boundary_label and view_mode == "token":
        fig.add_vline(
            x=boundary_label,
            line_width=2,
            line_dash="dash",
            line_color="#6c757d",
            opacity=0.7,
        )
        # Choose annotation colors based on Streamlit theme.
        base = ""
        background_color = ""
        if theme_config:
            base = theme_config.get("base", "")
            background_color = theme_config.get("backgroundColor", "")
        is_dark = str(base).lower() == "dark"
        if background_color:
            luminance = _hex_luminance(background_color)
            is_dark = luminance < 0.5
        text_color = "#f8f9fa" if is_dark else "#212529"
        if is_dark:
            bg_color = "rgba(0,0,0,0)"
            border_color = "rgba(0,0,0,0)"
        else:
            bg_color = "#ffffff"
            border_color = "#dee2e6"

        annotation = dict(
            x=boundary_label,
            y=1.04,
            xref="x",
            yref="paper",
            text="Start of Model Response",
            showarrow=False,
            bgcolor=bg_color,
            borderpad=4,
            bordercolor=border_color,
            opacity=0.9,
            font=dict(color=text_color, size=13),
        )
        fig.add_annotation(**annotation)
    return fig


def build_decoder_bar_figure(
    payload: DecoderStepPayload,
    title: Optional[str] = None,
    show_x_labels: bool = True,
    theme_config: Optional[Dict[str, str]] = None,
) -> go.Figure:
    is_dark = theme_config and theme_config.get("base", "light").lower() == "dark"
    text_color = "white" if is_dark else "black"

    colors = cycle_palette(len(payload.class_labels))
    traces = []
    x = payload.x_labels
    for idx in range(len(payload.class_labels)):
        y = [probs[idx] if idx < len(probs) else 0.0 for probs in payload.probabilities]
        traces.append(
            go.Bar(
                x=x,
                y=y,
                name=payload.class_labels.get(idx, str(idx)),
                marker_color=colors[idx % len(colors)],
            )
        )

    fig = go.Figure(data=traces)

    xaxis_config = dict(
        title=dict(text="Step", font=dict(size=30, color=text_color)),
        tickangle=45,
        tickmode="array",
        tickvals=x,
        tickfont=dict(size=24, color=text_color),
    )
    if show_x_labels:
        xaxis_config["ticktext"] = [
            payload.tick_labels[i] if payload.tick_labels else x[i]
            for i in range(len(x))
        ]
    else:
        xaxis_config["showticklabels"] = False

    fig.update_layout(
        barmode="stack",
        title=dict(text=""),
        xaxis=xaxis_config,
        yaxis=dict(
            title=dict(text="Probability", font=dict(size=30, color=text_color)),
            range=[0, 1],
            tickfont=dict(size=24, color=text_color),
        ),
        legend=dict(title=dict(text="Choices", font=dict(size=24, color=text_color)), font=dict(size=24, color=text_color)),
        bargap=0.15,
        margin=dict(t=120, r=300, b=180, l=70),
        height=1000,
        width=1600,
    )
    if title:
        fig.add_annotation(
            text=title,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.15,
            showarrow=False,
            font=dict(size=40, color=text_color),
            xanchor="center",
        )
    return fig


def build_inflection_figure(
    step_probs: dict,
    inflections: list,
    class_labels: dict,
    tick_labels: Optional[list] = None,
) -> go.Figure:
    """Build a two-subplot figure: probabilities and entropy over steps, with inflection markers.

    Args:
        step_probs: {step_idx: [p_A, p_B, p_C, p_D]} ordered by step
        inflections: list of dicts with step_number, inflection_type, description
        class_labels: {0: "A — ...", 1: "B — ...", ...}
        tick_labels: optional list of step text labels (same order as sorted steps)
    """
    from plotly.subplots import make_subplots

    sorted_steps = sorted(step_probs.keys())
    x_vals = [f"Step {s}" for s in sorted_steps]
    probs_matrix = [step_probs[s] for s in sorted_steps]
    num_classes = len(probs_matrix[0]) if probs_matrix else 4

    colors = cycle_palette(num_classes)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Probe Probabilities by Step", "Prediction Entropy by Step"),
    )

    # Top: probability traces
    for cls_idx in range(num_classes):
        label = class_labels.get(cls_idx, f"Class {cls_idx}")
        y_vals = [p[cls_idx] if cls_idx < len(p) else 0.0 for p in probs_matrix]
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                name=label,
                line=dict(color=colors[cls_idx % len(colors)], width=2),
                marker=dict(size=5),
            ),
            row=1, col=1,
        )

    # Bottom: entropy (no hover)
    def _entropy(probs):
        return -sum(max(p, 1e-10) * math.log2(max(p, 1e-10)) for p in probs)

    entropies = [_entropy(p) for p in probs_matrix]
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=entropies,
            mode="lines+markers",
            name="Entropy",
            line=dict(color="black", width=2),
            marker=dict(size=5),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # Mark inflection points as hoverable invisible scatter traces
    type_colors = {
        "backtrack": "#e41a1c",
        "realization": "#377eb8",
        "reconsideration": "#4daf4a",
    }
    for inf in inflections:
        step_num = inf.get("step_number", 0)
        inf_type = inf.get("inflection_type", "unknown")
        desc = inf.get("description", "")
        color = type_colors.get(inf_type, "#999999")
        x_val = f"Step {step_num}"

        # Vertical lines on both subplots
        for row_idx in (1, 2):
            fig.add_vline(
                x=x_val, line_width=2, line_dash="dash",
                line_color=color, opacity=0.7,
                row=row_idx, col=1,
            )

        # Hoverable marker on the probability subplot
        fig.add_trace(
            go.Scatter(
                x=[x_val], y=[1.0],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                name=inf_type,
                showlegend=False,
                hovertemplate=(
                    f"<b>{inf_type}</b> (Step {step_num})<br>"
                    f"{desc}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Probability", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Entropy (bits)", range=[0, 2.1], row=2, col=1)

    # Use step text as tick labels if provided
    xaxis_config = dict(tickangle=45)
    if tick_labels and len(tick_labels) == len(x_vals):
        xaxis_config.update(
            tickmode="array",
            tickvals=x_vals,
            ticktext=tick_labels,
        )
    fig.update_xaxes(**xaxis_config, row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_white",
        margin=dict(t=60, r=20, b=120, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
