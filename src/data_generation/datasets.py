"""Dataset loading and formatting for data generation pipeline."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

from .data_gen_config import DataGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class QuestionData:
    """Structured representation of a question."""

    question_hash: str
    question: str
    choices: List[str]
    correct_answer: str
    formatted_question: str
    category: str = "unknown"


def compute_question_hash(question_text: str) -> str:
    """Compute a stable MD5-based hash for a question."""
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()[:12]


def build_question_id_lookup(existing_responses_dir: Path) -> Dict[str, str]:
    """Build lookup from question content to existing question hash."""
    lookup = {}
    if not existing_responses_dir.exists():
        logger.warning(f"Existing responses directory not found: {existing_responses_dir}")
        return lookup

    for path in existing_responses_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fq = data.get("formatted_question", "")
            q_key = fq.split("## Instruction")[0].strip()
            lookup[q_key] = path.stem
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    logger.info(f"Built lookup with {len(lookup)} existing questions")
    return lookup


def format_r1_question(
    question: str,
    choices: List[str],
    choice_labels: List[str] = None,
) -> str:
    """Format a question in the R1 prompt style."""
    if choice_labels is None:
        choice_labels = ["A", "B", "C", "D"]

    choices_text = "\n".join(
        f"- ({label}) {choice}" for label, choice in zip(choice_labels, choices)
    )

    formatted = f"""## Question:
{question}

## Choices:
{choices_text}

## Instruction:
Please analyze the question step by step in <think>...</think> tags, then provide your final answer in JSON format with the key "answer" containing only the letter (A, B, C, or D) of the correct choice."""

    return formatted


def load_gpqa_questions(
    config: DataGenerationConfig,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load GPQA diamond split questions."""
    logger.info("Loading GPQA diamond split from Idavidrein/gpqa")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    questions = []
    for item in dataset:
        question_text = item["Question"]
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]

        correct_text = item["Correct Answer"]
        formatted = format_r1_question(question_text, choices)

        q_key = formatted.split("## Instruction")[0].strip()
        if question_lookup and q_key in question_lookup:
            q_hash = question_lookup[q_key]
        else:
            q_hash = compute_question_hash(question_text)

        # GPQA choices are ordered correct-first, so correct is always A
        correct_answer = "A"

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category=item.get("Subdomain", "unknown"),
            )
        )

    logger.info(f"Loaded {len(questions)} GPQA questions")
    return questions


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]


def load_mmlu_questions(
    config: DataGenerationConfig,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load MMLU-redux questions with error_type == 'ok' filtering."""
    logger.info("Loading MMLU-redux from edinburgh-dawg/mmlu-redux-2.0 (all 57 subjects)")

    from datasets import concatenate_datasets

    all_datasets = []
    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
            ds = ds.add_column("subject", [subject] * len(ds))
            all_datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load subject {subject}: {e}")

    if not all_datasets:
        raise ValueError("Failed to load any MMLU subjects")

    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Loaded {len(dataset)} total MMLU questions before filtering")

    dataset = dataset.filter(lambda x: x.get("error_type") == "ok")

    questions = []
    choice_labels = ["A", "B", "C", "D"]

    for item in dataset:
        question_text = item["question"]
        choices = item["choices"]

        formatted = format_r1_question(question_text, choices)

        q_key = formatted.split("## Instruction")[0].strip()
        if question_lookup and q_key in question_lookup:
            q_hash = question_lookup[q_key]
        else:
            q_hash = compute_question_hash(question_text)

        answer_idx = item["answer"]
        correct_answer = choice_labels[answer_idx]

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category=item.get("subject", "unknown"),
            )
        )

    logger.info(f"Loaded {len(questions)} MMLU questions")
    return questions


def load_dataset_questions(config: DataGenerationConfig) -> List[QuestionData]:
    """Load questions from the specified dataset."""
    question_lookup = None
    if config.existing_responses_dir:
        question_lookup = build_question_id_lookup(config.existing_responses_dir)

    if config.dataset_name.lower() == "gpqa":
        questions = load_gpqa_questions(config, question_lookup)
    elif config.dataset_name.lower() == "mmlu":
        questions = load_mmlu_questions(config, question_lookup)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}. Expected 'gpqa' or 'mmlu'")

    if config.limit is not None:
        questions = questions[:config.limit]
        logger.info(f"Limited to {len(questions)} questions")

    return questions
