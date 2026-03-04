"""Utilities shared across the analysis pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DELIMITERS = ["\n\n", ".\n\n", ":\n\n", "<think>\n", "</think>\n"]


def split_response_text(tokens: List[int], tokenizer: AutoTokenizer, delimiters: List[str]) -> Tuple[List[int], List[List[int]]]:
    """Split response tokens into steps using delimiters."""
    delimiter_token_sequences = [tokenizer.encode(d, add_special_tokens=False) for d in delimiters]
    delimiter_token_sequences = [seq for seq in delimiter_token_sequences if seq]
    sentence_end_indices = []
    sentence_token_sequences = []
    current_tokens = []
    idx = 0
    num_tokens = len(tokens)
    while idx < num_tokens:
        matched_length = None
        for delimiter_tokens in delimiter_token_sequences:
            length = len(delimiter_tokens)
            if length == 0 or idx + length > num_tokens:
                continue
            if list(tokens[idx : idx + length]) == delimiter_tokens:
                matched_length = length
                break
        if matched_length is None:
            current_tokens.append(tokens[idx])
            idx += 1
            continue
        current_tokens.extend(tokens[idx : idx + matched_length])
        idx += matched_length
        if current_tokens:
            sentence_token_sequences.append(current_tokens)
            sentence_end_indices.append(idx)
            current_tokens = []
    if current_tokens or not sentence_token_sequences:
        sentence_token_sequences.append(current_tokens)
        sentence_end_indices.append(num_tokens)
    return sentence_end_indices, sentence_token_sequences


def compute_reasoning_token_start(
    system_prompt: str,
    formatted_question: str,
    tokenizer: AutoTokenizer,
) -> int:
    """Compute the token index where assistant reasoning content begins."""
    if not system_prompt and not formatted_question:
        return 0

    try:
        messages_prefix = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question},
        ]

        prefix_text = tokenizer.apply_chat_template(
            messages_prefix,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

        return len(prefix_tokens)
    except Exception as e:
        logger.warning(f"Could not compute reasoning token start: {e}")
        return 0


def is_gpt_oss_model(cfg: "ExperimentConfig") -> bool:
    """Check if the current model is GPT-OSS (vs DeepSeekR1 etc)."""
    tokenizer_model = cfg.data.tokenizer_model or ""
    dataset_name = cfg.data.dataset_name or ""
    return "gpt-oss" in tokenizer_model.lower() or "gpt-oss" in dataset_name.lower()


def load_forced_step_boundaries(cfg: "ExperimentConfig") -> Dict[str, List[int]]:
    """Load forced answering step boundaries for GPT-OSS probe/forced-answer alignment."""
    completions_path = None
    if cfg.forced_answer.completions_path:
        completions_path = Path(cfg.forced_answer.completions_path)
    elif cfg.data.forced_answers_path:
        completions_path = Path(cfg.data.forced_answers_path)

    if completions_path is None or not completions_path.exists():
        return {}

    try:
        with completions_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        boundaries = {}
        for qhash, value in data.items():
            if isinstance(value, list):
                steps = value
            elif isinstance(value, dict) and "results" in value:
                steps = value["results"]
            else:
                continue

            lengths = []
            for step in steps:
                if isinstance(step, dict) and "partial_reasoning_length" in step:
                    lengths.append(int(step["partial_reasoning_length"]))
            if lengths:
                boundaries[str(qhash)] = lengths
        logger.info(f"Loaded forced step boundaries for {len(boundaries)} questions from {completions_path}")
        return boundaries
    except Exception as e:
        logger.warning(f"Could not load forced step boundaries: {e}")
        return {}
