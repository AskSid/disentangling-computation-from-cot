import argparse
import csv
import json
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CHOICE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D"}
CHOICE_LABELS_REVERSED = {v: k for k, v in CHOICE_LABELS.items()}

SYSTEM_PROMPT = (
    "You are DeepSeek-R1, an AI assistant developed by DeepSeek. "
    "You are analytical, detail-oriented, and skilled at multi-step reasoning. "
    "Work through the problem carefully, showing all intermediate reasoning before "
    "stating the final answer."
)

USER_INSTRUCTION = (
    "Think step by step. When you are confident in the answer, explicitly conclude "
    "with `Final answer: <letter>`."
)


def build_prompt(tokenizer: AutoTokenizer, question: str, choices: List[str]) -> str:
    choice_lines = [f"{CHOICE_LABELS[idx]}. {choice}" for idx, choice in enumerate(choices)]
    joined_choices = "\n".join(choice_lines)
    user_content = (
        f"{USER_INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        f"Answer Choices:\n{joined_choices}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except AttributeError:
        # Fallback in case the tokenizer lacks a chat template.
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    return prompt


def extract_answer(generated_text: str, num_choices: int) -> str:
    pattern = re.compile(r"final answer[^A-Z]*([A-Z])", re.IGNORECASE)
    match = pattern.search(generated_text)
    if match:
        candidate = match.group(1).upper()
        if candidate in CHOICE_LABELS_REVERSED:
            return candidate
    for char in reversed(generated_text.strip()):
        upper_char = char.upper()
        if upper_char in CHOICE_LABELS_REVERSED:
            return upper_char
    return ""


def run_inference(output_csv: Path, split: str = "test", limit: Optional[int] = None, category: str = "anatomy") -> None:
    dataset = load_dataset("edinburgh-dawg/mmlu-redux", category)[split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.to(device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "question_idx",
                "question",
                "answer_choices",
                "full_prompt",
                "correct_answer",
                "full_cot",
                "predicted_answer",
                "category",
            ],
        )
        writer.writeheader()

        for idx, example in enumerate(tqdm(dataset)):
            if limit is not None and idx >= limit:
                break
            question = example["question"]
            choices = list(example["choices"])
            answer_letter = example["answer"]
            prompt = build_prompt(tokenizer, question, choices)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            predicted_letter = CHOICE_LABELS_REVERSED[extract_answer(generated_text, len(choices))]


            writer.writerow(
                {
                    "question_idx": idx,
                    "question": question,
                    "answer_choices": json.dumps(choices),
                    "full_prompt": prompt,
                    "correct_answer": answer_letter,
                    "full_cot": generated_text.strip(),
                    "predicted_answer": predicted_letter,
                    "category": category,
                }
            )


def parse_args() -> argparse.Namespace:
    # example usage: uv run run_deepseekr1_eval.py --output "./deepseekr1_anatomy_predictions.csv" --category "anatomy"
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepSeek-R1 Distill Qwen 1.5B on a category's split of MMLU Redux "
            "and export a CSV with prompts, CoT, and predictions."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deepseekr1_predictions.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of questions to process.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="anatomy",
        help="Category to evaluate (default: anatomy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(output_csv=args.output, split=args.split, limit=args.limit, category=args.category)


if __name__ == "__main__":
    main()
