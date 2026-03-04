"""Generate forced-answer completions and inject into step-level CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from .experiment_config import ExperimentConfig
from .setup_data import STEP_HEADERS
from .utils import DELIMITERS, compute_reasoning_token_start, split_response_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHOICES = ["A", "B", "C", "D"]


def generate_forced_completions(cfg: ExperimentConfig) -> Path:
    """Generate forced-answer completions using vLLM and save to JSON."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    responses_dir = cfg.data.responses_dir
    output_path = cfg.forced_answer.completions_path or cfg.data.forced_answers_path
    if output_path is None:
        output_path = cfg.resolved_paths()["root"] / "forced_answer_completions.json"

    response_files = sorted(Path(responses_dir).glob("*.json"))
    logger.info(f"Found {len(response_files)} response files")

    if not response_files:
        raise RuntimeError("No response files found. Run data generation first.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_model)

    logger.info(f"Loading model {cfg.data.tokenizer_model}")
    llm = LLM(
        model=cfg.data.tokenizer_model,
        tensor_parallel_size=cfg.forced_answer.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
    )

    existing_results = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_results = json.load(f)
        logger.info(f"Loaded {len(existing_results)} existing results")

    all_prompts = []
    prompt_info = []  # (question_hash, step_idx, partial_reasoning_length)

    for response_path in response_files:
        question_hash = response_path.stem
        if question_hash in existing_results:
            continue

        with response_path.open("r", encoding="utf-8") as f:
            response_data = json.load(f)

        complete_rollout = response_data.get("complete_rollout", "")
        system_prompt = response_data.get("system_prompt", "")
        formatted_question = response_data.get("formatted_question", "")

        if not complete_rollout:
            continue

        complete_tokens = tokenizer.encode(complete_rollout, add_special_tokens=False)
        reasoning_start = compute_reasoning_token_start(
            system_prompt, formatted_question, tokenizer
        )
        reasoning_tokens = complete_tokens[reasoning_start:]

        step_end_indices, _ = split_response_text(reasoning_tokens, tokenizer, DELIMITERS)

        for step_idx, end_idx in enumerate(step_end_indices):
            prefix_tokens = reasoning_tokens[:end_idx]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=False)
            context_tokens = complete_tokens[:reasoning_start]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)

            if "</think>" in prefix_text:
                forced_prompt = f"{context_text}{prefix_text}\n{{\n  \"answer\": \""
            else:
                forced_prompt = f"{context_text}{prefix_text}</think>\n{{\n  \"answer\": \""

            all_prompts.append(forced_prompt)
            prompt_info.append((question_hash, step_idx, end_idx))

    logger.info(f"Collected {len(all_prompts)} prompts for forced completion")

    all_results = dict(existing_results)
    batch_size = cfg.forced_answer.batch_size
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_info = prompt_info[i:i + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for output, (question_hash, step_idx, partial_len) in zip(outputs, batch_info):
            result = output.outputs[0]
            completion = result.text.strip()

            logprobs_dict = {}
            if result.logprobs:
                token_logprobs = result.logprobs[0]
                for token_id, logprob_obj in token_logprobs.items():
                    decoded = tokenizer.decode([token_id]).strip()
                    if decoded in CHOICES:
                        logprobs_dict[decoded] = logprob_obj.logprob

            for choice in CHOICES:
                if choice not in logprobs_dict:
                    logprobs_dict[choice] = -100.0

            if question_hash not in all_results:
                all_results[question_hash] = []

            all_results[question_hash].append({
                "partial_reasoning_length": partial_len,
                "step_idx": step_idx,
                "completion": completion,
                "logprobs": logprobs_dict,
            })

        logger.info(f"Processed {min(i + batch_size, len(all_prompts))}/{len(all_prompts)} prompts")

    for question_hash in all_results:
        all_results[question_hash].sort(key=lambda x: x["step_idx"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Forced answering generation complete: {len(all_results)} questions")
    return output_path


def select_logprobs(raw_logprobs):
    """Accept either a dict or a list of dicts (first element wins)."""
    if isinstance(raw_logprobs, dict):
        return raw_logprobs
    if isinstance(raw_logprobs, (list, tuple)) and raw_logprobs and isinstance(raw_logprobs[0], dict):
        return raw_logprobs[0]
    raise RuntimeError("logprobs must be a dict or list[dict] with A/B/C/D keys")


def normalize_logprobs(logprob_dict):
    """Return normalized probabilities for A/B/C/D. Missing logprobs default to 0."""
    values = []
    for choice in CHOICES:
        if choice not in logprob_dict:
            values.append(0.0)  # Default missing logprobs to 0
        else:
            values.append(float(logprob_dict[choice]))
    max_lp = max(values)
    exp_vals = [math.exp(v - max_lp) for v in values]
    denom = sum(exp_vals)
    if denom == 0:
        raise RuntimeError("Degenerate logprobs: sum(exp(logprobs)) == 0")
    return [v / denom for v in exp_vals]


def load_completions(path: Path):
    """Load forced-answer completions from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Forced-answer completions file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("Forced-answer completions must be a JSON object keyed by question_hash")
    completions = {}
    for qhash, value in payload.items():
        # Handle both formats
        if isinstance(value, list):
            entries = value
        elif isinstance(value, dict) and "results" in value:
            entries = value["results"]
        else:
            raise RuntimeError(f"Expected list or dict with 'results' key for {qhash}, got {type(value)}")
        completions[str(qhash)] = entries
    return completions


def inject_for_question(step_path: Path, completions, metadata_csv: Path | None = None) -> None:
    logger.info(f"Injecting forced answers into {step_path.name}")
    
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
    existing_forced_rows = []
    step_text_map = {}
    unique_steps = None
    
    if step_path.exists():
        with step_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            all_rows = list(reader)
        
        # Separate probe, cot_monitor, and forced-answer rows
        probe_rows = [row for row in all_rows if row.get("layer_idx") not in ["-1", "-2"]]
        cot_monitor_rows = [row for row in all_rows if row.get("layer_idx") == "-2"]
        existing_forced_rows = [row for row in all_rows if row.get("layer_idx") == "-1"]
        
        if probe_rows:
            # Use probe rows to get step structure
            unique_steps = sorted(set(int(row["step_idx"]) for row in probe_rows if row.get("step_idx")))
            logger.debug(f"Found {len(unique_steps)} unique steps from probe rows: {unique_steps[:5]}...")
            
            # Create step_idx -> step_text mapping from probe rows
            for row in probe_rows:
                step_idx_val = row.get("step_idx")
                step_text_val = row.get("step_text", "")
                if step_idx_val and step_idx_val not in step_text_map and step_text_val:
                    step_text_map[step_idx_val] = step_text_val
            
            # Align step counts between probe CSV and forced completions
            # Truncate to the minimum of both to ensure alignment
            step_diff = len(unique_steps) - len(completions)
            if step_diff != 0:
                if abs(step_diff) <= 2:
                    logger.debug(
                        f"Step count adjustment for {step_path.name}: "
                        f"csv_steps={len(unique_steps)} completions={len(completions)} (diff={step_diff})"
                    )
                else:
                    logger.warning(
                        f"Step count mismatch for {step_path.name}: "
                        f"csv_steps={len(unique_steps)} completions={len(completions)} (diff={step_diff})"
                    )
                # Truncate to minimum of both
                min_steps = min(len(unique_steps), len(completions))
                unique_steps = unique_steps[:min_steps]
                completions = completions[:min_steps]
        else:
            logger.info(f"No probe rows found in {step_path.name}, using sequential step indices")
    
    if unique_steps is None:
        unique_steps = list(range(len(completions)))
        logger.info(f"Using sequential step indices: 0 to {len(completions)-1}")

    forced_rows = []
    for step_position, entry in enumerate(completions):
        step_idx_value = str(unique_steps[step_position])
        completion = entry.get("completion", "")
        logprobs_raw = entry.get("logprobs")
        if logprobs_raw is None:
            raise RuntimeError(f"Missing logprobs for {step_path.name} step {step_position}")
        
        probs = normalize_logprobs(select_logprobs(logprobs_raw))
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        
        forced_row = {
            "question_hash": question_hash,
            "question_idx": question_idx,
            "layer_idx": "-1",  # Mark as forced-answer row
            "step_idx": step_idx_value,
            "step_text": step_text_map.get(step_idx_value, ""),
            "decoder_type": "forced_answer",
            "decoder_pred": CHOICES[pred_idx],
            "decoder_output": json.dumps(probs),
            "forced_completion": completion if completion in CHOICES else "",
        }
        # Ensure all fields from STEP_HEADERS are present
        for field in STEP_HEADERS:
            if field not in forced_row:
                forced_row[field] = ""
        forced_rows.append(forced_row)

    step_to_completion = {}
    for forced_row in forced_rows:
        step_idx_val = forced_row["step_idx"]
        fc = forced_row.get("forced_completion", "")
        if fc:
            step_to_completion[step_idx_val] = fc

    for i, row in enumerate(probe_rows):
        invalid_keys = [key for key in row.keys() if key not in STEP_HEADERS]
        if invalid_keys:
            raise RuntimeError(
                f"Malformed CSV {step_path.name}: probe row {i} contains invalid keys: {invalid_keys}. "
                f"Expected keys: {STEP_HEADERS}. Row keys: {list(row.keys())}"
            )
        # Check for None keys (indicates malformed CSV header)
        if None in row:
            raise RuntimeError(
                f"Malformed CSV {step_path.name}: probe row {i} contains None key. "
                f"This indicates the CSV header is malformed. Row: {row}"
            )
        # Backfill forced_completion from forced_answer row with matching step_idx
        step_idx_val = row.get("step_idx")
        if step_idx_val in step_to_completion:
            row["forced_completion"] = step_to_completion[step_idx_val]

    merged_rows = probe_rows + cot_monitor_rows + forced_rows

    def _row_sort_key(item):
        step_val = int(item["step_idx"])
        try:
            layer_val = int(item["layer_idx"])
            # Force -1 (forced-answer) to sort after all positive layer indices
            # Force -2 (cot_monitor) to sort after forced-answer
            if layer_val == -1:
                layer_val = 999999  # Sort after all probe layers
            elif layer_val == -2:
                layer_val = 999998  # Sort after forced-answer
        except (ValueError, TypeError):
            layer_val = 0
        return (step_val, layer_val)

    merged_rows.sort(key=_row_sort_key)

    with step_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_HEADERS)
        writer.writeheader()
        writer.writerows(merged_rows)
    
    logger.info(f"Added {len(forced_rows)} forced-answer rows (removed {len(existing_forced_rows)} old ones)")


def run_forced_answering(cfg: ExperimentConfig) -> None:
    if not cfg.forced_answer.enabled:
        logger.info("forced_answer.enabled=false; skipping forced-answer injection")
        return

    if cfg.forced_answer.generate:
        logger.info("forced_answer.generate=true; generating forced completions via vLLM")
        generated_path = generate_forced_completions(cfg)
        completions_path = generated_path
    else:
        completions_path = cfg.forced_answer.completions_path or cfg.data.forced_answers_path

    if completions_path is None:
        raise RuntimeError("No completions path provided (set forced_answer.completions_path or data.forced_answers_path)")

    completions = load_completions(completions_path)
    paths = cfg.resolved_paths()
    step_dir = paths["step_level"]
    metadata_path = paths["metadata"]
    split_path = cfg.data.split_path or paths["root"] / "train_val_test_split.json"

    logger.info(f"Loaded {len(completions)} questions from {completions_path}")
    logger.info(f"Step-level directory: {step_dir}")

    test_set_hashes = None
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        test_set_hashes = set(str(h) for h in split_data.get("test", []))
        logger.info(f"Filtering to {len(test_set_hashes)} test set questions (split file: {split_path})")
    else:
        logger.warning("No split file found - processing all questions (this may not be intended)")

    processed = 0
    skipped = 0
    questions_with_missing_logprobs = 0
    for qhash, entries in completions.items():
        # Skip non-test questions
        if test_set_hashes is not None and str(qhash) not in test_set_hashes:
            skipped += 1
            continue
        step_path = step_dir / f"{qhash}.csv"
        # Create empty CSV if it doesn't exist (for cases without probe rows)
        if not step_path.exists():
            logger.info(f"Creating new step-level CSV for {qhash}")
            step_path.parent.mkdir(parents=True, exist_ok=True)
            with step_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=STEP_HEADERS)
                writer.writeheader()
        
        # Check if this question has any missing logprobs
        has_missing = False
        for entry in entries:
            logprobs_raw = entry.get("logprobs")
            if logprobs_raw:
                logprob_dict = select_logprobs(logprobs_raw)
                for choice in CHOICES:
                    if choice not in logprob_dict:
                        has_missing = True
                        break
                if has_missing:
                    break
        
        if has_missing:
            questions_with_missing_logprobs += 1
        
        inject_for_question(step_path, entries, metadata_path)
        processed += 1

    logger.info(f"Forced-answer completions injected for {processed} questions (skipped {skipped} non-test questions)")
    if questions_with_missing_logprobs > 0:
        logger.warning(f"{questions_with_missing_logprobs} questions had at least one missing logprob (set to 0)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject forced-answer completions into step-level CSVs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_forced_answering(cfg)


if __name__ == "__main__":
    main()
