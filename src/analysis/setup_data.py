import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from .experiment_config import ExperimentConfig
import argparse


METADATA_FIELDS = [
    "question_idx",
    "question_hash",
    "question",
    "answer_choices",
    "full_prompt",
    "correct_answer",
    "full_cot",
    "predicted_answer",
    "category",
    "system_prompt",
    "formatted_question",
    "model_answer",
    "is_correct",
    "reasoning",
    "complete_rollout",
]

TOKEN_HEADERS = [
    "question_hash",
    "question_idx",
    "layer_idx",
    "token_idx",
    "step_idx",
    "token_text",
    "decoder_type",
    "decoder_pred",
    "decoder_output",
]

STEP_HEADERS = [
    "question_hash",
    "question_idx",
    "layer_idx",
    "step_idx",
    "step_text",
    "decoder_type",
    "decoder_pred",
    "decoder_output",
    "forced_completion",
]


def extract_answer_choices(formatted_question: str) -> List[str]:
    choices = []
    for line in formatted_question.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")) and "(" in line and ")" in line:
            after_paren = line.split(")", 1)[-1].strip()
            if after_paren:
                choices.append(after_paren)
    return choices


def build_full_prompt(system_prompt: str, formatted_question: str) -> str:
    parts = [part.strip() for part in (system_prompt, formatted_question) if part and part.strip()]
    return "\n\n".join(parts)


def load_responses(responses_dir: Path) -> List[Dict]:
    payloads = []
    files = sorted(responses_dir.glob("*.json"))
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["_question_hash"] = path.stem
        payloads.append(data)
    return payloads


def build_metadata_rows(responses: Sequence[Dict]) -> List[Dict]:
    rows = []
    for idx, payload in enumerate(responses):
        qhash = str(payload.get("_question_hash", idx))
        formatted_question = payload.get("formatted_question", "")
        system_prompt = payload.get("system_prompt", "")
        question_text = payload.get("question", "") or formatted_question
        answer_choices = payload.get("answer_choices") or extract_answer_choices(formatted_question)
        full_cot = payload.get("complete_rollout") or payload.get("reasoning", "")
        predicted_answer = payload.get("parsed_answer", "")
        correct_answer = payload.get("correct_answer", "")
        category = payload.get("category", "unknown")

        rows.append(
            {
                "question_idx": idx,
                "question_hash": qhash,
                "question": question_text,
                "answer_choices": json.dumps(answer_choices),
                "full_prompt": build_full_prompt(system_prompt, formatted_question),
                "correct_answer": correct_answer,
                "full_cot": full_cot,
                "predicted_answer": predicted_answer,
                "category": category,
                "system_prompt": system_prompt,
                "formatted_question": formatted_question,
                "model_answer": predicted_answer,
                "is_correct": payload.get("is_correct", ""),
                "reasoning": payload.get("reasoning", ""),
                "complete_rollout": full_cot,
            }
        )
    return rows


def write_metadata_csv(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def setup_data(cfg: ExperimentConfig) -> None:
    paths = cfg.resolved_paths()
    root = paths["root"]

    (paths["token_level"]).mkdir(parents=True, exist_ok=True)
    (paths["step_level"]).mkdir(parents=True, exist_ok=True)
    (paths["models"]).mkdir(parents=True, exist_ok=True)

    responses = load_responses(cfg.data.responses_dir)
    metadata_rows = build_metadata_rows(responses)
    write_metadata_csv(metadata_rows, paths["metadata"])

    split_path = root / "train_val_test_split.json"
    if cfg.data.split_path:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.data.split_path, split_path)
    else:
        if not split_path.exists():
            train_ratio, val_ratio, _ = cfg.setup.train_val_test_split_ratio
            rng = random.Random(cfg.run.seed)
            hashes = [row["question_hash"] for row in metadata_rows]
            rng.shuffle(hashes)
            n = len(hashes)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            split = {
                "train": hashes[:train_end],
                "val": hashes[train_end:val_end],
                "test": hashes[val_end:],
            }
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_hashes = set(split.get("test", []))

    for row in metadata_rows:
        qhash = row["question_hash"]
        if qhash not in test_hashes:
            continue
        token_path = paths["token_level"] / f"{qhash}.csv"
        step_path = paths["step_level"] / f"{qhash}.csv"
        for path, headers in ((token_path, TOKEN_HEADERS), (step_path, STEP_HEADERS)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()


def main():
    parser = argparse.ArgumentParser(description="Setup data for early decoder run")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    setup_data(cfg)


if __name__ == "__main__":
    main()
