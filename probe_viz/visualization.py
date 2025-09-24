"""Visualization helpers for the probe viewer."""

from __future__ import annotations

import plotly.graph_objects as go

from data_loader import HeatmapPayload


def build_heatmap_figure(payload: HeatmapPayload, show_x_labels: bool = True) -> go.Figure:
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

    xaxis_config = dict(
        title=payload.x_axis_title,
        tickangle=45,
        automargin=True,
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

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=50, r=20, b=120, l=80),
        xaxis=xaxis_config,
        yaxis=dict(title="Layer"),
        height=700,
    )
    return fig
