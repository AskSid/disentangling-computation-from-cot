"""Streamlit application for visualizing DeepSeek-R1 probe outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from data_loader import (
    ProbeDataRepository,
    get_dataset_config,
    list_dataset_options,
    normalize_remote_roots,
)
from visualization import build_heatmap_figure, build_decoder_bar_figure, build_inflection_figure


def normalize_full_response(raw_text: str) -> str:
    return raw_text if isinstance(raw_text, str) else str(raw_text)


def render_question_text(question_record) -> str:
    """Use formatted question when available; fallback to raw question."""
    if hasattr(question_record, "display_question"):
        try:
            text = question_record.display_question()
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
    return getattr(question_record, "question", "")


@st.cache_resource
def load_repository(dataset_key: str):
    r2_config = {}
    if hasattr(st, "secrets"):
        try:
            r2_config = dict(st.secrets.get("r2", {}))
        except Exception:
            r2_config = {}

    dataset = dict(get_dataset_config(dataset_key))
    remote_roots, storage_options = normalize_remote_roots(dataset, r2_config)
    dataset["remote_roots"] = remote_roots

    return ProbeDataRepository(
        dataset=dataset,
        storage_options=storage_options,
    )


def resolve_theme_config() -> Dict[str, str]:
    if not hasattr(st, "get_option"):
        return {"base": "light"}
    base = st.get_option("theme.base") or "light"
    background = st.get_option("theme.backgroundColor")
    return {
        "base": str(base),
        "backgroundColor": str(background) if background else "",
    }


def build_question_lookup(options) -> Dict[int, str]:
    return {int(option["value"]): str(option["label"]) for option in options}


def ensure_payload(
    repo: ProbeDataRepository,
    question_idx: int,
    probe: str,
    view: str,
    show_x_labels: bool,
    theme_config: Dict[str, str],
    probe_layer: int | None = None,
    include_baselines: bool = False,
):
    try:
        payload = repo.get_heatmap_payload(
            question_idx,
            probe,
            view,
            probe_layer=probe_layer,
            include_baselines=include_baselines,
        )
        figure = build_heatmap_figure(
            payload,
            show_x_labels=show_x_labels,
            theme_config=theme_config,
        )
        return payload, figure, None
    except (ValueError, FileNotFoundError) as exc:
        return None, None, str(exc)


def main() -> None:
    st.set_page_config(page_title="Early Decoder Viewer", layout="wide")
    st.title("Early Decoder Viewer")

    dataset_options = list_dataset_options()
    dataset_lookup = {opt["value"]: opt["label"] for opt in dataset_options}
    if not dataset_options:
        st.error("No datasets configured.")
        return

    selected_dataset = st.selectbox(
        "Dataset",
        options=[opt["value"] for opt in dataset_options],
        format_func=lambda value: dataset_lookup.get(value, str(value)),
    )

    repository = load_repository(selected_dataset)
    theme_config = resolve_theme_config()
    dataset_defaults = get_dataset_config(selected_dataset).get("defaults", {})

    st.markdown(
        """
        <style>
        .full-sequence-box {
            background-color: #f5f7ff;
            border: 1px solid #dce3ff;
            border-radius: 0.6rem;
            padding: 1.5rem;
            margin-top: 1.75rem;
            color: #17203c;
        }
        .full-sequence-title {
            font-weight: 700;
            font-size: 1.35rem;
            margin: 0 0 0.75rem 0;
        }
        .full-sequence-content {
            white-space: pre-wrap;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #17203c;
        }
        @media (prefers-color-scheme: dark) {
            .full-sequence-box {
                background-color: #202a3f !important;
                border-color: #334162 !important;
                color: #f4f7ff !important;
            }
            .full-sequence-content {
                color: #f4f7ff !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    question_options = repository.list_question_options()
    probe_options = repository.list_probe_options()
    view_options = repository.list_view_modes()

    question_lookup = build_question_lookup(question_options)
    view_lookup = {opt["value"]: opt["label"] for opt in view_options}

    # Selection controls
    col_question, col_view = st.columns((3, 2))

    with col_question:
        selected_question = st.selectbox(
            "Question",
            options=[opt["value"] for opt in question_options],
            format_func=lambda value: question_lookup[int(value)],
        )
    with col_view:
        default_view_idx = 0
        for idx, opt in enumerate(view_options):
            if opt["value"] == "decoder":
                default_view_idx = idx
                break
        selected_view = st.radio(
            "View",
            options=[opt["value"] for opt in view_options],
            format_func=lambda value: view_lookup[value],
            horizontal=True,
            index=default_view_idx,
        )

    selected_probe = next(
        (
            opt["value"]
            for opt in probe_options
            if "attention" in str(opt["value"]).lower()
        ),
        probe_options[0]["value"] if probe_options else "attention_probe",
    )

    SHOW_X_LABELS_KEY = "show_x_axis_labels_toggle"
    SENTENCE_SHOW_X_LABELS_KEY = "show_sentence_x_axis_labels_toggle"
    DECODER_SHOW_X_LABELS_KEY = "show_decoder_x_axis_labels_toggle"

    show_x_labels = bool(st.session_state.get(SHOW_X_LABELS_KEY, True))

    question_record = repository.get_question(int(selected_question))

    # Question panel
    st.markdown(render_question_text(question_record))

    st.markdown("**Answer Choices**")
    for label, choice in question_record.enumerated_choices():
        st.write(f"{label}. {choice}")

    meta_cols = st.columns(3)
    meta_cols[0].info(f"Correct Answer: {question_record.correct_answer}")
    meta_cols[1].info(f"Model Answer: {question_record.predicted_answer}")

    selected_probe_layer: int | None = None
    decoder_options = []
    selected_decoder = None

    if selected_view == "sentence":
        probe_layers = repository.list_probe_layers(int(selected_question), selected_probe)
        if probe_layers:
            default_layer = dataset_defaults.get("probe_layer")
            default_idx = probe_layers.index(default_layer) if default_layer in probe_layers else len(probe_layers) - 1
            selected_probe_layer = st.selectbox(
                "Probe layer (compared against forced-answer and CoT monitor)",
                options=probe_layers,
                index=default_idx,
            )
            meta_cols[2].info(f"Decoder: {selected_probe} (L{selected_probe_layer})")
        else:
            meta_cols[2].warning("No probe layers found.")
    elif selected_view == "decoder":
        decoder_options = repository.list_decoder_options(int(selected_question))
        if decoder_options:
            default_decoder = dataset_defaults.get("decoder")
            default_idx = 0
            if default_decoder:
                for idx, opt in enumerate(decoder_options):
                    if opt["value"] == default_decoder:
                        default_idx = idx
                        break
            selected_decoder = st.selectbox(
                "Decoder",
                options=[opt["value"] for opt in decoder_options],
                format_func=lambda v: next(
                    (opt["label"] for opt in decoder_options if opt["value"] == v),
                    str(v),
                ),
                index=default_idx,
            )
            meta_cols[2].info(f"Decoder: {selected_decoder}")
        else:
            meta_cols[2].warning("No decoders available.")
    else:
        meta_cols[2].info(f"Decoder: {selected_probe}")

    if selected_view == "inflection":
        # Inflection points view
        inflection_info = repository.get_inflection_info(int(selected_question))
        if inflection_info is None:
            payload, figure, error_message = None, None, "No inflection data available for this question."
        else:
            try:
                # Get step-level probe data for this question at selected layer
                step_df = repository.load_sentence_df(int(selected_question))
                if selected_probe_layer is not None:
                    layer = selected_probe_layer
                elif dataset_defaults.get("probe_layer") is not None:
                    layer = dataset_defaults["probe_layer"]
                else:
                    available_layers = sorted(step_df["layer_idx"].unique())
                    available_layers = [l for l in available_layers if l >= 0]
                    layer = available_layers[-1] if available_layers else 0
                layer_data = step_df[step_df["layer_idx"] == layer].sort_values("sentence_idx")
                if layer_data.empty:
                    payload, figure, error_message = None, None, f"No step-level data for layer {layer}."
                else:
                    import ast as _ast
                    step_probs = {}
                    for _, row in layer_data.iterrows():
                        probs = row["probe_output"]
                        if isinstance(probs, str):
                            probs = _ast.literal_eval(probs)
                        step_probs[int(row["sentence_idx"])] = [float(v) for v in probs]

                    num_classes = len(next(iter(step_probs.values())))
                    class_labels = repository.build_class_labels(selected_probe, num_classes, question_record)
                    inflections = inflection_info.get("inflections", [])

                    # Build step text tick labels
                    sorted_step_indices = sorted(step_probs.keys())
                    tick_labels = []
                    for s_idx in sorted_step_indices:
                        text = repository.get_sentence_text(int(selected_question), s_idx).strip()
                        text = " ".join(text.split())
                        if len(text) > 30:
                            text = text[:27] + "..."
                        tick_labels.append(f'Step {s_idx}: "{text}"' if text else f"Step {s_idx}")

                    figure = build_inflection_figure(step_probs, inflections, class_labels, tick_labels=tick_labels)
                    payload = None
                    error_message = None
            except (ValueError, FileNotFoundError) as exc:
                payload, figure, error_message = None, None, str(exc)
    elif selected_view != "decoder":
        payload, figure, error_message = ensure_payload(
            repository,
            int(selected_question),
            selected_probe,
            selected_view,
            show_x_labels=show_x_labels,
            theme_config=theme_config,
            probe_layer=selected_probe_layer,
            include_baselines=True if selected_view == "sentence" else False,
        )
    else:
        try:
            decoder_payload = (
                repository.get_decoder_step_payload(int(selected_question), selected_decoder)
                if selected_decoder
                else None
            )
            if decoder_payload and selected_decoder:
                decoder_show_x_labels = bool(st.session_state.get(DECODER_SHOW_X_LABELS_KEY, True))
                figure = build_decoder_bar_figure(decoder_payload, show_x_labels=decoder_show_x_labels, theme_config=theme_config)
            else:
                figure = None
            payload = None
            error_message = None if decoder_payload else "No decoder selected."
        except ValueError as exc:
            figure = None
            payload = None
            error_message = str(exc)

    title_text = "Probe Output:" if selected_view == "token" else "Probe / Forced Answer / CoT Monitor LLM Comparison"
    if selected_view == "inflection":
        title_text = "Probe Predictions & Inflection Points"
    st.subheader(title_text)
    if error_message:
        st.warning(error_message)
    else:
        st.plotly_chart(figure, use_container_width=True)

        # Show inflection details below the plot
        if selected_view == "inflection" and inflection_info:
            inflections = inflection_info.get("inflections", [])
            if inflections:
                st.markdown("### Inflection Points")
                for inf in inflections:
                    step_num = inf.get("step_number", "?")
                    inf_type = inf.get("inflection_type", "unknown")
                    desc = inf.get("description", "")
                    type_emoji = {"backtrack": "backtrack", "realization": "realization", "reconsideration": "reconsideration"}.get(inf_type, inf_type)
                    st.markdown(f"**Step {step_num}** ({type_emoji}): {desc}")
            else:
                st.info("No inflection points detected in this reasoning trace.")

        if selected_view == "decoder":
            st.checkbox(
                "Show x-axis labels",
                value=bool(st.session_state.get(DECODER_SHOW_X_LABELS_KEY, True)),
                key=DECODER_SHOW_X_LABELS_KEY,
            )
        elif selected_view not in ("inflection",):
            st.checkbox(
                "Show x-axis labels",
                value=show_x_labels,
                key=SHOW_X_LABELS_KEY,
            )

    # Step view drilldown
    if selected_view == "sentence" and not error_message:
        sentence_show_x_labels = bool(
            st.session_state.get(SENTENCE_SHOW_X_LABELS_KEY, True)
        )

        sentence_options = repository.list_sentence_options(
            int(selected_question),
            selected_probe,
            probe_layer=selected_probe_layer,
        )
        if sentence_options:
            sentence_lookup = {
                int(option["value"]): str(option["label"]) for option in sentence_options
            }
            st.subheader("Individual Step Breakdown:")
            selected_sentence = st.selectbox(
                "Step",
                options=[opt["value"] for opt in sentence_options],
                format_func=lambda value: sentence_lookup[int(value)],
            )

            try:
                drilldown_payload = repository.get_sentence_token_payload(
                    int(selected_question),
                    selected_probe,
                    int(selected_sentence),
                    probe_layer=selected_probe_layer,
                )
                drilldown_fig = build_heatmap_figure(
                    drilldown_payload,
                    show_x_labels=sentence_show_x_labels,
                    theme_config=theme_config,
                )
                st.plotly_chart(drilldown_fig, use_container_width=True)
                st.checkbox(
                    "Show x-axis labels",
                    value=sentence_show_x_labels,
                    key=SENTENCE_SHOW_X_LABELS_KEY,
                )

                sentence_text = repository.get_sentence_text(
                    int(selected_question), int(selected_sentence)
                ).strip()
                st.markdown("### Currently Selected Step:")
                if sentence_text:
                    st.code(sentence_text, language="text")
                else:
                    st.markdown("_Not available._")
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.info("No sentence-level data available for this probe and question.")

    full_response_text = normalize_full_response(question_record.full_cot)
    st.markdown("### Full Response:")
    if full_response_text:
        is_dark = theme_config.get("base", "light").lower() == "dark"
        if is_dark:
            fr_bg = "#1e1e2e"
            fr_border = "#334162"
            fr_color = "#f4f7ff"
        else:
            fr_bg = "#f5f7ff"
            fr_border = "#dce3ff"
            fr_color = "#17203c"
        st.markdown(
            f'<div style="overflow-x:auto; white-space:pre-wrap; max-height:500px; overflow-y:auto; '
            f'background-color:{fr_bg}; border:1px solid {fr_border}; border-radius:0.6rem; '
            f'padding:1rem; font-family:monospace; font-size:0.9rem; line-height:1.5; color:{fr_color};">'
            f'{full_response_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("_Not available._")


if __name__ == "__main__":
    main()
