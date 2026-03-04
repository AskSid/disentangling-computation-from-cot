"""Find inflection points in reasoning traces via OpenRouter LLM."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert at analyzing reasoning traces from language models.

Identify "inflection points" — moments where the model:
1. **Backtracks**: Explicitly corrects earlier reasoning ("Wait,", "Actually,", "No, that's wrong")
2. **Realizes**: Has a new insight that changes direction ("I just realized", "Oh,", "I see now")
3. **Reconsiders**: Questions previous reasoning ("Let me reconsider", "Hmm", "But wait")

Output JSON with this schema:
{
  "has_inflection": boolean,
  "reasoning": "Brief analysis of the reasoning flow",
  "inflections": [
    {"step_number": int, "inflection_type": "backtrack"|"realization"|"reconsideration", "description": "What changed"}
  ]
}

Be conservative — only flag genuine course changes, not normal step-by-step reasoning."""


def extract_reasoning(response_data: dict) -> Optional[str]:
    """Extract reasoning text from a stage 1 response JSON."""
    full_text = response_data.get("complete_rollout", "")
    match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL) if full_text else None
    if match:
        content = match.group(1).strip()
        if content and content != "...":
            return content
    return response_data.get("reasoning", "") or None


def format_steps(reasoning: str) -> tuple[List[str], str]:
    """Split reasoning into steps and format for the prompt."""
    steps = [s.strip() for s in reasoning.split("\n\n") if s.strip()]
    numbered = []
    for i, step in enumerate(steps, 1):
        text = step[:500] + "..." if len(step) > 500 else step
        numbered.append(f"[Step {i}]: {text}")
    return steps, "\n\n".join(numbered)


def call_api(api_key: str, model: str, user_prompt: str, max_retries: int = 5) -> dict:
    """Call OpenRouter with retries."""
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(backoff)
            backoff *= 2
    return {"has_inflection": False, "reasoning": "", "inflections": []}


def process_one(api_key: str, model: str, question_hash: str, response_data: dict) -> dict:
    """Analyze a single response for inflection points."""
    reasoning = extract_reasoning(response_data)
    if not reasoning:
        return {"question_hash": question_hash, "has_inflection": False, "inflections": [], "total_steps": 0}

    steps, numbered_text = format_steps(reasoning)
    if not steps:
        return {"question_hash": question_hash, "has_inflection": False, "inflections": [], "total_steps": 0}

    user_prompt = (
        "Analyze the following reasoning trace for inflection points.\n\n"
        f"REASONING TRACE:\n{numbered_text}\n\n"
        "Return your analysis as JSON."
    )

    result = call_api(api_key, model, user_prompt)

    for inflection in result.get("inflections", []):
        step_num = inflection.get("step_number", 0)
        if 1 <= step_num <= len(steps):
            text = steps[step_num - 1]
            inflection["step_text"] = text[:200] + "..." if len(text) > 200 else text

    return {
        "question_hash": question_hash,
        "has_inflection": result.get("has_inflection", False),
        "reasoning": result.get("reasoning", ""),
        "inflections": result.get("inflections", []),
        "total_steps": len(steps),
    }


def find_inflection_points(
    responses_dir: Path,
    output_path: Optional[Path] = None,
    model: str = "google/gemini-2.5-flash",
    api_key_env: str = "OPENROUTER_API_KEY",
    workers: int = 10,
    limit: Optional[int] = None,
    split_path: Optional[Path] = None,
) -> Path:
    """Run inflection point analysis on stage 1 responses."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {api_key_env} environment variable.")

    if output_path is None:
        output_path = responses_dir.parent / "inflection_results.json"
    checkpoint_path = output_path.parent / "inflection_checkpoint.json"

    response_files = sorted(responses_dir.glob("*.json"))
    responses = []
    for path in response_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_question_hash"] = path.stem
        responses.append(data)
    logger.info(f"Loaded {len(responses)} responses from {responses_dir}")

    if split_path and split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            test_hashes = set(str(h) for h in json.load(f).get("test", []))
        responses = [r for r in responses if r["_question_hash"] in test_hashes]
        logger.info(f"Filtered to {len(responses)} test set responses")

    if limit:
        responses = responses[:limit]
        logger.info(f"Limited to {limit} responses")

    results: List[dict] = []
    processed: set = set()
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        processed = set(ckpt.get("processed_hashes", []))
        logger.info(f"Resuming: {len(processed)} already processed")

    to_process = [r for r in responses if r["_question_hash"] not in processed]
    if not to_process:
        logger.info("All responses already processed.")
    else:
        logger.info(f"Processing {len(to_process)} responses with {model} ({workers} workers)")
        lock = threading.Lock()
        count = [0]

        def do_one(resp):
            return process_one(api_key, model, resp["_question_hash"], resp)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(do_one, r): r["_question_hash"] for r in to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Finding inflections"):
                qhash = futures[future]
                try:
                    result = future.result()
                    with lock:
                        results.append(result)
                        processed.add(qhash)
                        count[0] += 1
                        if count[0] % 50 == 0:
                            with checkpoint_path.open("w", encoding="utf-8") as f:
                                json.dump({"results": results, "processed_hashes": list(processed)}, f)
                except Exception as e:
                    logger.error(f"Error processing {qhash}: {e}")

    total = len(results)
    with_inflections = sum(1 for r in results if r.get("has_inflection"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "metadata": {
                "model": model,
                "total_responses": total,
                "responses_with_inflections": with_inflections,
                "inflection_rate": with_inflections / total if total > 0 else 0,
            },
        }, f, indent=2)

    if total:
        logger.info(f"Saved to {output_path}: {with_inflections}/{total} ({with_inflections/total*100:.1f}%) have inflections")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Find inflection points in reasoning traces")
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    args = parser.parse_args()

    find_inflection_points(
        responses_dir=args.responses_dir,
        output_path=args.output,
        model=args.model,
        api_key_env=args.api_key_env,
        workers=args.workers,
        limit=args.limit,
        split_path=args.split_path,
    )


if __name__ == "__main__":
    main()
