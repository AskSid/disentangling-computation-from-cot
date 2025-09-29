from __future__ import annotations

import ast

from transformers import AutoTokenizer
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from collections import defaultdict

# Assuming DeepSeek-R1-0528_mmlu-redux_results.csv exists with the following columns:
# question_idx, question, answer_choices, full_prompt, correct_answer, full_cot, predicted_answer, category

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_answer_choices(raw_value):
    if isinstance(raw_value, list):
        return list(raw_value)
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, list):
                return list(parsed)
        except (SyntaxError, ValueError):
            pass
    return []


def _label_for_choice(index: int) -> str:
    if 0 <= index < len(CHOICE_LABELS):
        return CHOICE_LABELS[index]
    return f"Choice {index}"


def generate_fake_probing_data(output_dir, questions_csv, N_layers, N_questions):

    early_decoders = [
        "probe_model_answer_correct", "probe_model_answer", "probe_correct_answer",
        "observer_model_answer_correct", "observer_model_answer", "observer_correct_answer",
    ]

    def generate_fake_data(early_decoder: str, num_choices: int) -> list[float]:
        if early_decoder.endswith('model_answer_correct'):
            return np.random.dirichlet(np.ones(2)).tolist()
        return np.random.dirichlet(np.ones(num_choices)).tolist()

    questions_df = pd.read_csv(questions_csv)
    sample_size = min(N_questions, len(questions_df))
    questions_df = questions_df.sample(sample_size, replace=False, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

    delimiter = '.\n\n'

    output_dir = Path(output_dir)
    token_dir = output_dir / 'token_level'
    sentence_dir = output_dir / 'sentence_level'
    token_dir.mkdir(parents=True, exist_ok=True)
    sentence_dir.mkdir(parents=True, exist_ok=True)

    for _, row in questions_df.iterrows():
        question_idx = int(row['question_idx'])
        full_cot = row['full_cot']
        answer_choices = _parse_answer_choices(row.get('answer_choices'))
        num_choices = max(2, min(len(answer_choices) or 4, len(CHOICE_LABELS)))

        token_level_data = []
        sentence_groups = defaultdict(list)

        raw_sentences = full_cot.split(delimiter)
        sentence_texts: List[str] = []
        for idx, segment in enumerate(raw_sentences):
            text = segment
            if idx < len(raw_sentences) - 1:
                text = f"{segment}{delimiter}"
            sentence_texts.append(text)

        flat_tokens = []  # (sentence_idx, token_id, cleaned_token_text)
        sentence_idx_counter = 0
        for sentence_text in sentence_texts:
            if not sentence_text.strip():
                sentence_idx_counter += 1
                continue
            sentence_tokens = tokenizer.encode(sentence_text, add_special_tokens=False)
            if not sentence_tokens:
                sentence_idx_counter += 1
                continue
            for token in sentence_tokens:
                token_text = tokenizer.decode([token], skip_special_tokens=True)
                cleaned_token_text = (
                    token_text.replace('\n', '\\n')
                    .replace('\r', '\\r')
                    .replace('\t', '\\t')
                )
                flat_tokens.append((sentence_idx_counter, token, cleaned_token_text))
            sentence_idx_counter += 1

        for early_decoder in early_decoders:
            for token_idx, (sentence_idx, token, cleaned_token_text) in enumerate(flat_tokens):
                for layer_idx in range(N_layers):
                    probe_output = generate_fake_data(early_decoder, num_choices)
                    argmax_idx = int(np.argmax(probe_output))
                    probe_ans = (
                        argmax_idx
                        if len(probe_output) == 2
                        else _label_for_choice(argmax_idx)
                    )
                    token_level_data.append([
                        question_idx,
                        token_idx,
                        sentence_idx,
                        layer_idx,
                        token,
                        cleaned_token_text,
                        early_decoder,
                        probe_output,
                        probe_ans,
                    ])
                    sentence_groups[
                        (question_idx, early_decoder, sentence_idx, layer_idx)
                    ].append((probe_output, probe_ans))

        sentence_level_data = []
        for (q_idx, ed, s_idx, l_idx), vals in sentence_groups.items():
            outputs = [v[0] for v in vals]
            avg_output_arr = np.mean(np.array([np.array(o) for o in outputs]), axis=0)
            avg_output = avg_output_arr.tolist()
            argmax_idx = int(np.argmax(avg_output_arr))
            avg_ans = (
                argmax_idx if len(avg_output) == 2 else _label_for_choice(argmax_idx)
            )
            sentence_text = ""
            if 0 <= s_idx < len(sentence_texts):
                sentence_text = sentence_texts[s_idx]
            sentence_level_data.append([
                q_idx,
                s_idx,
                l_idx,
                ed,
                avg_output,
                avg_ans,
                sentence_text,
            ])

        token_level_df = pd.DataFrame(
            token_level_data,
            columns=[
                'question_idx',
                'token_idx',
                'sentence_idx',
                'layer_idx',
                'token',
                'token_text',
                'early_decoder',
                'probe_output',
                'probe_ans',
            ],
        )
        token_path = token_dir / f'question_{question_idx}.csv'
        token_level_df.to_csv(token_path, index=False, quoting=csv.QUOTE_ALL)

        sentence_level_df = pd.DataFrame(
            sentence_level_data,
            columns=[
                'question_idx',
                'sentence_idx',
                'layer_idx',
                'early_decoder',
                'probe_output',
                'probe_ans',
                'sentence_text',
            ],
        )
        sentence_path = sentence_dir / f'question_{question_idx}.csv'
        sentence_level_df.to_csv(sentence_path, index=False, quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    generate_fake_probing_data(
        './probe_viz/data',
        './probe_viz/data/deepseekr1_anatomy_predictions_questions_0_to_10.csv',
        28,
        10,
    )
