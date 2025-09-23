from datasets import load_dataset
import random
from transformers import AutoTokenizer
import csv
import pandas as pd
import numpy as np
import os

# Assuming DeepSeek-R1-0528_mmlu-redux_results.csv exists with the following columns:
# question_idx, question, answer_choices, full_prompt, correct_answer, full_cot, predicted_answer, category

from collections import defaultdict, Counter

def generate_fake_probing_data(output_dir, questions_csv, N_layers, N_questions):
    CHOICE_LABELS_REVERSED = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D'
    }

    early_decoders = [
        "probe_model_answer_correct", "probe_model_answer", "probe_correct_answer",
        "observer_model_answer_correct", "observer_model_answer", "observer_correct_answer",
    ]

    def generate_fake_data(early_decoder):
        if early_decoder.endswith('model_answer_correct'): # did the model answer correctly?
            return np.random.dirichlet(np.ones(2)).tolist()
        elif early_decoder.endswith('model_answer'): # what did the model answer?
            return np.random.dirichlet(np.ones(4)).tolist() 
        elif early_decoder.endswith('correct_answer'): # what was the correct answer?
            return np.random.dirichlet(np.ones(4)).tolist()

    questions_df = pd.read_csv(questions_csv).sample(N_questions)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

    token_level_data = []
    # key: (question_idx, early_decoder, sentence_idx, layer_idx) -> list of (probe_output, probe_ans)
    sentence_groups = defaultdict(list)

    for _, row in questions_df.iterrows():
        question_idx = row['question_idx']
        full_cot = row['full_cot']
        tokens_per_sentence = random.sample(range(10, 20), 1)[0]
        full_cot_tokens = tokenizer.encode(full_cot)

        for early_decoder in early_decoders:
            for token_idx, token in enumerate(full_cot_tokens):
                sentence_idx = token_idx // tokens_per_sentence
                token_text = tokenizer.decode(token, skip_special_tokens=True)
                cleaned_token_text = token_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                for layer_idx in range(N_layers):
                    probe_output = generate_fake_data(early_decoder)
                    # For 2-way predictions keep 0/1, for 4-way map to A-D
                    argmax_idx = int(np.argmax(probe_output))
                    probe_ans = argmax_idx if len(probe_output) == 2 else CHOICE_LABELS_REVERSED[argmax_idx]
                    token_level_data.append([
                        question_idx, token_idx, sentence_idx, layer_idx, token,
                        cleaned_token_text, early_decoder, probe_output, probe_ans
                    ])
                    sentence_groups[(question_idx, early_decoder, sentence_idx, layer_idx)].append((probe_output, probe_ans))

    # build sentence-level rows: mean of probe_output, argmax(mean) as probe_ans
    sentence_level_data = []
    for (q_idx, ed, s_idx, l_idx), vals in sentence_groups.items():
        outputs = [v[0] for v in vals]
        avg_output_arr = np.mean(np.array([np.array(o) for o in outputs]), axis=0)
        avg_output = avg_output_arr.tolist()
        argmax_idx = int(np.argmax(avg_output_arr))
        # 2-way -> 0/1, 4-way -> A-D
        avg_ans = argmax_idx if len(avg_output) == 2 else CHOICE_LABELS_REVERSED[argmax_idx]
        sentence_level_data.append([
            q_idx, s_idx, l_idx, ed, avg_output, avg_ans
        ])

    token_level_df = pd.DataFrame(
        token_level_data,
        columns=['question_idx', 'token_idx', 'sentence_idx', 'layer_idx', 'token', 'token_text', 'early_decoder', 'probe_output', 'probe_ans']
    )
    token_level_df.to_csv(os.path.join(output_dir, 'deepseekr1_anatomy_results_token_level_questions_0_1.csv'), index=False, quoting=csv.QUOTE_ALL)

    sentence_level_df = pd.DataFrame(
        sentence_level_data,
        columns=['question_idx', 'sentence_idx', 'layer_idx', 'early_decoder', 'probe_output', 'probe_ans']
    )
    sentence_level_df.to_csv(os.path.join(output_dir, 'deepseekr1_anatomy_probe_results_sentence_level_questions_0_1.csv'), index=False, quoting=csv.QUOTE_ALL)

generate_fake_probing_data('./probe_viz/data', './probe_viz/data/deepseekr1_anatomy_predictions_questions_0_1.csv', 28, 2)