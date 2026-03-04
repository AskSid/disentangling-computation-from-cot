"""Utility functions for data generation pipeline."""

import json
import logging
import re
from typing import List, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def parse_answer_from_response(response: str) -> Optional[str]:
    """Parse the answer letter from a model response."""
    # Strategy 1: JSON format
    json_patterns = [
        r'\{\s*"answer"\s*:\s*"([A-D])"\s*\}',
        r'\{\s*"answer"\s*:\s*\'([A-D])\'\s*\}',
        r'"answer"\s*:\s*"([A-D])"',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Strategy 2: Answer after </think>
    think_match = re.search(r"</think>\s*\{?\s*[\"']?answer[\"']?\s*[:=]\s*[\"']?([A-D])", response, re.IGNORECASE)
    if think_match:
        return think_match.group(1).upper()

    # Strategy 3: \boxed{X}
    boxed_match = re.search(r"\\boxed\{([A-D])\}", response, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()

    # Strategy 4: "Answer: X" patterns (in last 500 chars)
    answer_patterns = [
        r"[Aa]nswer\**:?\**\s*\n?\s*[-*]*\s*\(?([A-D])\)?",
        r"[Aa]nswer\**:?\**\s*\n?\s*\*\*([A-D])\*\*",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response[-500:])
        if match:
            return match.group(1).upper()

    # Strategy 5: "The answer is X"
    answer_is_match = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if answer_is_match:
        return answer_is_match.group(1).upper()

    # Strategy 6: Standalone letter at end
    end_match = re.search(r"\b([A-D])\b(?:\s|[.!)\*])*$", response.strip()[-100:], re.IGNORECASE)
    if end_match:
        return end_match.group(1).upper()

    return None


def extract_reasoning_from_response(response: str) -> str:
    """Extract the reasoning content from between <think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response


def build_chat_messages(
    system_prompt: str,
    formatted_question: str,
) -> List[dict]:
    """Build chat messages for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_question},
    ]


def format_response_json(
    question_id: str,
    question_data: dict,
    response: str,
    tokenizer: AutoTokenizer,
    system_prompt: str,
) -> dict:
    """Format a response into the expected JSON structure."""
    full_response = "<think>\n" + response

    parsed_answer = parse_answer_from_response(full_response)
    reasoning = extract_reasoning_from_response(full_response)

    messages = build_chat_messages(system_prompt, question_data["formatted_question"])

    try:
        prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        complete_rollout = prefix + response + "<｜end▁of▁sentence｜>"
    except Exception:
        complete_rollout = f"{system_prompt}\n\n{question_data['formatted_question']}\n\n<think>\n{response}"

    complete_rollout_tokens = tokenizer.encode(complete_rollout, add_special_tokens=False)
    complete_rollout_tokenized = [tokenizer.decode([t]) for t in complete_rollout_tokens]

    return {
        "question_id": question_id,
        "parsed_answer": parsed_answer,
        "correct_answer": question_data["correct_answer"],
        "is_correct": parsed_answer == question_data["correct_answer"] if parsed_answer else False,
        "reasoning": reasoning,
        "formatted_question": question_data["formatted_question"],
        "system_prompt": system_prompt,
        "complete_rollout": complete_rollout,
        "complete_rollout_tokenized": complete_rollout_tokenized,
        "complete_rollout_tokens": complete_rollout_tokens,
        "category": question_data.get("category", "unknown"),
    }
