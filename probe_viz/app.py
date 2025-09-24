"""Streamlit application for visualizing DeepSeek-R1 probe outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from data_loader import ProbeDataRepository
from visualization import build_heatmap_figure


@st.cache_resource
def _load_repository() -> ProbeDataRepository:
    return ProbeDataRepository()


def _build_question_lookup(options) -> Dict[int, str]:
    return {int(option["value"]): str(option["label"]) for option in options}


def _ensure_payload(
    repo: ProbeDataRepository,
    question_idx: int,
    probe: str,
    view: str,
    show_x_labels: bool,
):
    try:
        payload = repo.get_heatmap_payload(question_idx, probe, view)
        figure = build_heatmap_figure(payload, show_x_labels=show_x_labels)
        return payload, figure, None
    except ValueError as exc:
        return None, None, str(exc)


def main() -> None:
    st.set_page_config(page_title="DeepSeek-R1 Probe Viewer", layout="wide")
    st.title("DeepSeek-R1 Probe Viewer")

    repository = _load_repository()

    question_options = repository.list_question_options()
    probe_options = repository.list_probe_options()
    view_options = repository.list_view_modes()

    question_lookup = _build_question_lookup(question_options)
    probe_lookup = {opt["value"]: opt["label"] for opt in probe_options}
    view_lookup = {opt["value"]: opt["label"] for opt in view_options}

    # Selection controls
    col_question, col_probe, col_view = st.columns((3, 2, 2))

    with col_question:
        selected_question = st.selectbox(
            "Question",
            options=[opt["value"] for opt in question_options],
            format_func=lambda value: question_lookup[int(value)],
        )
    with col_probe:
        selected_probe = st.selectbox(
            "Probe",
            options=[opt["value"] for opt in probe_options],
            format_func=lambda value: probe_lookup[value],
        )
    with col_view:
        selected_view = st.radio(
            "View",
            options=[opt["value"] for opt in view_options],
            format_func=lambda value: view_lookup[value],
            horizontal=True,
        )

    SHOW_X_LABELS_KEY = "show_x_axis_labels_toggle"
    SENTENCE_SHOW_X_LABELS_KEY = "show_sentence_x_axis_labels_toggle"

    show_x_labels = bool(st.session_state.get(SHOW_X_LABELS_KEY, True))

    question_record = repository.get_question(int(selected_question))

    # Question panel
    st.subheader("Question")
    st.markdown(question_record.question)

    st.markdown("**Answer Choices**")
    for label, choice in question_record.enumerated_choices():
        st.write(f"{label}. {choice}")

    meta_cols = st.columns(4)
    meta_cols[0].info(f"Correct Answer: {question_record.correct_answer}")
    meta_cols[1].info(f"Model Answer: {question_record.predicted_answer}")
    meta_cols[2].info(f"Category: {question_record.category}")
    meta_cols[3].info(f"Probe: {selected_probe}")

    payload, figure, error_message = _ensure_payload(
        repository,
        int(selected_question),
        selected_probe,
        selected_view,
        show_x_labels=show_x_labels,
    )

    st.subheader("Probe Output")
    if error_message:
        st.warning(error_message)
    else:
        st.plotly_chart(figure, use_container_width=True)
        st.checkbox(
            "Show x-axis labels",
            value=show_x_labels,
            key=SHOW_X_LABELS_KEY,
        )

        if selected_view == "token":
            full_response = question_record.full_cot.strip()
            st.markdown("### Full Model Response:")
            if full_response:
                st.write(full_response)
            else:
                st.markdown("_Not available._")

    # Sentence view drilldown
    if selected_view == "sentence" and not error_message:
        sentence_show_x_labels = bool(
            st.session_state.get(SENTENCE_SHOW_X_LABELS_KEY, True)
        )

        sentence_options = repository.list_sentence_options(
            int(selected_question), selected_probe
        )
        if sentence_options:
            sentence_lookup = {
                int(option["value"]): str(option["label"]) for option in sentence_options
            }
            selected_sentence = st.selectbox(
                "Individual Sentence Breakdown",
                options=[opt["value"] for opt in sentence_options],
                format_func=lambda value: sentence_lookup[int(value)],
            )

            try:
                drilldown_payload = repository.get_sentence_token_payload(
                    int(selected_question), selected_probe, int(selected_sentence)
                )
                drilldown_fig = build_heatmap_figure(
                    drilldown_payload, show_x_labels=sentence_show_x_labels
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
                st.markdown("### Currently Selected Sentence:")
                if sentence_text:
                    st.write(sentence_text)
                else:
                    st.markdown("_Not available._")
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.info("No sentence-level data available for this probe and question.")


if __name__ == "__main__":
    main()
