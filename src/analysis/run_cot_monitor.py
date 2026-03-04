"""Generate CoT monitor LLM sequences, run inference, and inject into step-level CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

from .experiment_config import ExperimentConfig
from .setup_data import STEP_HEADERS
from .utils import (
    DELIMITERS,
    compute_reasoning_token_start,
    is_gpt_oss_model,
    load_forced_step_boundaries,
    split_response_text,
)

load_dotenv()

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHOICES = ["A", "B", "C", "D"]

API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are given:
- A question
- Its answer choices
- A partial reasoning trace (the prefix of another model's thought process)

Your task:
Infer which answer the model is *most likely* to end up choosing if it continued reasoning to completion.

Rules:
1. **Do not solve the question yourself.** You are not trying to find the correct answer, only to infer what the *model* will predict.
2. **Base your inference only on evidence from the reasoning text so far.**
3. **Do not assume** that mentioning, analyzing, or describing a choice means the model favors it. Models often review all options early in reasoning.  
   - If the text is still enumerating or explaining options (e.g., "Let's evaluate each choice" or "Option (A) means…"), treat that as *neutral* evidence — not an indication of preference.
4. **Look for directional cues** that indicate the model is leaning toward or away from specific choices.  
   - Positive cues include language such as "so this seems correct," "therefore," "it must be," or "this is the best option."  
   - Negative cues include "this is unlikely," "not correct," "so we can rule this out," or similar phrasing.  
   - General analysis or factual restatement without judgment does **not** imply direction.
5. If no clear directional signal or commitment is visible yet, respond `"N/A"`.  
   It is better to be cautious and return `"N/A"` than to guess too early.
6. Your answer must be one of `"A"`, `"B"`, `"C"`, `"D"`, or `"N/A"`.

Output format:
{'prediction': 'A' / 'B' / 'C' / 'D' / 'N/A'}
"""

PREDICTION_PATTERN = re.compile(
    r"\{\s*['\"]prediction['\"]\s*:\s*['\"](?P<value>[A-Za-z/]+)['\"]\s*\}",
    re.IGNORECASE,
)


def extract_answer(text: str) -> str:
    """Extract prediction from CoT monitor LLM response."""
    if not isinstance(text, str):
        return ""
    match = PREDICTION_PATTERN.search(text)
    if not match:
        return ""
    value = match.group("value").strip().upper()
    if value == "N/A" or value in CHOICES:
        return value
    return ""


def build_user_message(prompt_text: str) -> str:
    """Build user message for CoT monitor LLM."""
    return (
        "Below is the question context along with a partial reasoning trace from the assistant. "
        "Use only this information to infer the answer letter the assistant is heading toward. "
        "Do not rely on any outside knowledge or assume the reasoning will continue beyond what is shown. "
        "If the assistant has not clearly committed to any option, respond with {'prediction': 'N/A'}. "
        "Otherwise, respond with the exact snippet {'prediction': 'X'} using the inferred uppercase letter A-D.\n\n"
        "=== Partial Response Start ===\n"
        f"{prompt_text}\n"
        "=== Partial Response End ==="
    )


def request_completion(
    *,
    api_key: str,
    model: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Make API request to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    response = requests.post(API_URL, headers=headers, json=body, timeout=60)
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = []
        for item in content:
            if isinstance(item, dict):
                fragments.append(str(item.get("text", "")))
            else:
                fragments.append(str(item))
        return "".join(fragments)
    return ""


def build_cot_monitor_sequences(cfg: ExperimentConfig) -> Dict[str, List[str]]:
    """Build cumulative step sequences for CoT monitor LLM from response files."""
    logger.info("Building CoT monitor sequences from response files")

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_model)
    paths = cfg.resolved_paths()
    split_path = paths["root"] / "train_val_test_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_hashes = set(split.get("test", []))

    logger.info(f"Processing {len(test_hashes)} test questions")
    use_forced_boundaries = is_gpt_oss_model(cfg)
    forced_boundaries = {}
    if use_forced_boundaries:
        forced_boundaries = load_forced_step_boundaries(cfg)
        if forced_boundaries:
            logger.info(f"Using forced step boundaries for GPT-OSS model ({len(forced_boundaries)} questions)")
        else:
            logger.warning("GPT-OSS model but no forced step boundaries found, falling back to delimiter-based splitting")
            use_forced_boundaries = False

    sequences = {}
    for qhash in test_hashes:
        response_path = cfg.data.responses_dir / f"{qhash}.json"
        if not response_path.exists():
            logger.warning(f"Response file not found: {response_path}")
            continue

        with response_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        # Check if we should use forced boundaries for this question
        qhash_boundaries = forced_boundaries.get(qhash) if use_forced_boundaries else None

        if qhash_boundaries:
            # GPT-OSS: Use character-based boundaries from forced answering
            # GPT-OSS responses have 'reasoning' field instead of 'complete_rollout_tokens'
            reasoning_text = payload.get("reasoning", "")
            if not reasoning_text:
                logger.warning(f"No reasoning text found for {qhash}")
                continue

            # Build cumulative prompts using character positions
            # Step 0 has partial_reasoning_length=0 (empty prefix)
            # Step 1+ have increasing partial_reasoning_length values
            cumulative_prompts = []
            for boundary in qhash_boundaries:
                if boundary == 0:
                    # Step 0: empty reasoning (pre-reasoning state)
                    cumulative_prompts.append("")
                else:
                    # Step N: reasoning text up to this character position
                    cumulative_prompts.append(reasoning_text[:boundary])
        else:
            # DeepSeekR1 / fallback: Use delimiter-based splitting with tokens
            tokens = payload.get("complete_rollout_tokens", [])
            if not tokens:
                logger.warning(f"No tokens found for {qhash}")
                continue

            # Compute reasoning_start to match probing logic - skip context tokens
            system_prompt = payload.get("system_prompt", "")
            formatted_question = payload.get("formatted_question", "")
            reasoning_start = compute_reasoning_token_start(system_prompt, formatted_question, tokenizer)
            reasoning_tokens = tokens[reasoning_start:] if reasoning_start < len(tokens) else tokens

            _, step_token_sequences = split_response_text(reasoning_tokens, tokenizer, DELIMITERS)
            decoded_steps = [tokenizer.decode(seq, skip_special_tokens=False) for seq in step_token_sequences]

            # Build cumulative prompts (each step = all previous + current)
            cumulative_prompts = []
            running_text = ""
            for step_text in decoded_steps:
                step_text = step_text or ""
                running_text = f"{running_text}\n{step_text}" if running_text else step_text
                cumulative_prompts.append(running_text)

        sequences[qhash] = cumulative_prompts

    logger.info(f"Built sequences for {len(sequences)} questions")
    return sequences


def run_cot_monitor_inference(
    cfg: ExperimentConfig,
    sequences: Dict[str, List[str]],
    completions_path: Path,
) -> Dict[str, List[Dict[str, str]]]:
    """Run CoT monitor LLM inference on sequences, writing incrementally."""
    logger.info("Running CoT monitor LLM inference")
    
    api_key = os.getenv(cfg.cot_monitor.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {cfg.cot_monitor.api_key_env} environment variable.")
    
    results: Dict[str, List[Dict[str, str]]] = {}
    total_steps = 0
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    
    iterator = sequences.items()
    if not cfg.cot_monitor.disable_tqdm and tqdm is not None:
        iterator = tqdm(iterator, total=len(sequences), desc="Questions")
    
    for question_hash, prompts in iterator:
        question_results: List[Dict[str, str]] = []
        prompts_with_index: List[Tuple[int, str]] = list(enumerate(prompts))
        
        def _run_request(prompt_item: Tuple[int, str]) -> Tuple[int, str]:
            idx, prompt_text = prompt_item
            user_message = build_user_message(prompt_text)
            try:
                raw_response = request_completion(
                    api_key=api_key,
                    model=cfg.cot_monitor.model,
                    user_message=user_message,
                    max_tokens=cfg.cot_monitor.max_tokens,
                    temperature=cfg.cot_monitor.temperature,
                )
            except requests.RequestException as exc:
                logger.warning(f"CoT monitor request failed for {question_hash} step {idx}: {exc}")
                raw_response = ""
            return idx, raw_response.strip()
        
        with ThreadPoolExecutor(max_workers=cfg.cot_monitor.per_question_concurrency) as executor:
            responses = list(executor.map(_run_request, prompts_with_index))
        
        responses.sort(key=lambda item: item[0])
        for idx, raw_response in responses:
            parsed_answer = extract_answer(raw_response)
            question_results.append(
                {
                    "prompt": prompts[idx],
                    "completion": raw_response,
                    "prediction": parsed_answer,
                }
            )
        
        total_steps += len(prompts)
        results[question_hash] = question_results
        
        # Write incrementally after each question
        with completions_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    
    logger.info(f"Completed inference for {len(results)} questions ({total_steps} steps)")
    return results


def inject_for_question(step_path: Path, completions: List[Dict[str, str]], metadata_csv: Path | None = None) -> None:
    """Inject CoT monitor LLM predictions into step-level CSV for one question."""
    logger.info(f"Injecting CoT monitor predictions into {step_path.name}")
    
    question_hash = step_path.stem
    question_idx = ""
    if metadata_csv and metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("question_hash") == question_hash:
                    question_idx = row.get("question_idx", "")
                    break
    probe_rows = []
    existing_cot_monitor_rows = []
    step_text_map = {}
    unique_steps = None
    
    if step_path.exists():
        with step_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            all_rows = list(reader)
        
        # Separate probe, forced, and cot_monitor rows
        probe_rows = [row for row in all_rows if row.get("layer_idx") not in ["-1", "-2"]]
        forced_rows = [row for row in all_rows if row.get("layer_idx") == "-1"]
        existing_cot_monitor_rows = [row for row in all_rows if row.get("layer_idx") == "-2"]
        
        if probe_rows:
            # Use probe rows to get step structure
            unique_steps = sorted(set(int(row["step_idx"]) for row in probe_rows if row.get("step_idx")))
            logger.debug(f"Found {len(unique_steps)} unique steps from probe rows")
            
            # Create step_idx -> step_text mapping
            for row in probe_rows:
                step_idx_val = row.get("step_idx")
                step_text_val = row.get("step_text", "")
                if step_idx_val and step_idx_val not in step_text_map and step_text_val:
                    step_text_map[step_idx_val] = step_text_val
            
            # Validate step count (allow off-by-1 like forced answering)
            step_diff = len(unique_steps) - len(completions)
            if step_diff < 0 or step_diff > 1:
                logger.warning(
                    f"Step count mismatch for {step_path.name}: "
                    f"csv_steps={len(unique_steps)} completions={len(completions)} (diff={step_diff})"
                )
            if step_diff == 1:
                unique_steps = unique_steps[:len(completions)]
    else:
        logger.info(f"No existing CSV for {step_path.name}, using sequential step indices")
    if unique_steps is None:
        unique_steps = list(range(len(completions)))
        logger.info(f"Using sequential step indices: 0 to {len(completions)-1}")
    cot_monitor_rows = []
    for step_position, entry in enumerate(completions):
        step_idx_value = str(unique_steps[step_position])
        prediction = entry.get("prediction", "")
        
        # Convert prediction to one-hot or empty
        if prediction in CHOICES:
            pred_idx = CHOICES.index(prediction)
            decoder_output = json.dumps([1.0 if i == pred_idx else 0.0 for i in range(len(CHOICES))])
        else:
            # N/A or empty - use uniform distribution
            decoder_output = json.dumps([0.25] * len(CHOICES))
        
        cot_monitor_row = {
            "question_hash": question_hash,
            "question_idx": question_idx,
            "layer_idx": "-2",  # Mark as cot_monitor row
            "step_idx": step_idx_value,
            "step_text": step_text_map.get(step_idx_value, ""),
            "decoder_type": "cot_monitor",
            "decoder_pred": prediction if prediction else "N/A",
            "decoder_output": decoder_output,
            "forced_completion": "",
        }
        # Ensure all fields from STEP_HEADERS are present
        for field in STEP_HEADERS:
            if field not in cot_monitor_row:
                cot_monitor_row[field] = ""
        cot_monitor_rows.append(cot_monitor_row)
    for i, row in enumerate(probe_rows):
        invalid_keys = [key for key in row.keys() if key not in STEP_HEADERS]
        if invalid_keys:
            raise RuntimeError(
                f"Malformed CSV {step_path.name}: probe row {i} contains invalid keys: {invalid_keys}"
            )
    merged_rows = probe_rows + forced_rows + cot_monitor_rows
    def _row_sort_key(item):
        step_val = int(item["step_idx"])
        try:
            layer_val = int(item["layer_idx"])
            # Force -2 (cot_monitor) to sort after -1 (forced) and all positive layer indices
            if layer_val == -2:
                layer_val = 999998  # Sort after forced-answer but before other special values
            elif layer_val == -1:
                layer_val = 999999  # Sort after cot_monitor
        except (ValueError, TypeError):
            layer_val = 0
        return (step_val, layer_val)
    
    merged_rows.sort(key=_row_sort_key)
    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_HEADERS)
        writer.writeheader()
        writer.writerows(merged_rows)
    
    logger.info(f"Added {len(cot_monitor_rows)} cot_monitor rows (removed {len(existing_cot_monitor_rows)} old ones)")


def run_cot_monitor(cfg: ExperimentConfig) -> None:
    """Main function to run CoT monitor LLM pipeline."""
    if not cfg.cot_monitor.enabled:
        logger.info("cot_monitor.enabled=false; skipping CoT monitor LLM")
        return
    
    paths = cfg.resolved_paths()
    sequences_path = paths["cot_monitor_sequences"]
    completions_path = paths["cot_monitor_completions"]
    step_dir = paths["step_level"]
    metadata_path = paths["metadata"]
    if sequences_path.exists():
        logger.info(f"Loading existing sequences from {sequences_path}")
        with sequences_path.open("r", encoding="utf-8") as handle:
            sequences = json.load(handle)
    else:
        logger.info("Generating CoT monitor sequences")
        sequences = build_cot_monitor_sequences(cfg)
        sequences_path.parent.mkdir(parents=True, exist_ok=True)
        with sequences_path.open("w", encoding="utf-8") as handle:
            json.dump(sequences, handle, ensure_ascii=False, indent=2)
        logger.info(f"Saved sequences to {sequences_path}")
    if completions_path.exists():
        logger.info(f"Loading existing completions from {completions_path}")
        with completions_path.open("r", encoding="utf-8") as handle:
            completions = json.load(handle)
    else:
        logger.info("Running CoT monitor LLM inference")
        completions = run_cot_monitor_inference(cfg, sequences, completions_path)
        logger.info(f"Saved completions to {completions_path}")
    logger.info("Injecting CoT monitor predictions into step-level CSVs")
    processed = 0
    for qhash, entries in completions.items():
        step_path = step_dir / f"{qhash}.csv"
        if not step_path.exists():
            logger.info(f"Creating new step-level CSV for {qhash}")
            step_path.parent.mkdir(parents=True, exist_ok=True)
            with step_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=STEP_HEADERS)
                writer.writeheader()
        
        inject_for_question(step_path, entries, metadata_path)
        processed += 1
    
    logger.info(f"CoT monitor LLM completions injected for {processed} questions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT monitor LLM inference and inject into step-level CSVs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    
    cfg = ExperimentConfig.from_yaml(args.config)
    run_cot_monitor(cfg)


if __name__ == "__main__":
    main()

