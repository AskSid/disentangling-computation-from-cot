"""Stage 1: Response generation using vLLM."""

import json
import logging
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .data_gen_config import DataGenerationConfig
from .datasets import QuestionData, load_dataset_questions
from .utils import format_response_json, build_chat_messages

logger = logging.getLogger(__name__)


def generate_responses(config: DataGenerationConfig) -> None:
    """Generate full reasoning responses using vLLM."""
    output_dir = config.responses_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_dataset_questions(config)
    logger.info(f"Loaded {len(questions)} questions for {config.dataset_name}")

    existing_files = set(p.stem for p in output_dir.glob("*.json"))
    remaining_questions = [q for q in questions if q.question_hash not in existing_files]
    logger.info(f"Found {len(existing_files)} existing responses, {len(remaining_questions)} remaining")

    if not remaining_questions:
        logger.info("All questions already processed")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    logger.info(f"Loading model {config.model_id} with tensor_parallel_size={config.tensor_parallel_size}")
    llm = LLM(
        model=config.model_id,
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=config.dtype,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
    )

    prompts = []
    prompt_question_map = []

    for question in remaining_questions:
        messages = build_chat_messages(config.system_prompt, question.formatted_question)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template for {question.question_hash}: {e}")
            prompt = f"{config.system_prompt}\n\n{question.formatted_question}\n\nAssistant: <think>"

        prompts.append(prompt)
        prompt_question_map.append(question)

    logger.info(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    successful = 0
    failed = 0
    for output, question in zip(outputs, prompt_question_map):
        response_text = output.outputs[0].text

        try:
            response_data = format_response_json(
                question_id=question.question_hash,
                question_data={
                    "formatted_question": question.formatted_question,
                    "correct_answer": question.correct_answer,
                    "category": question.category,
                },
                response=response_text,
                tokenizer=tokenizer,
                system_prompt=config.system_prompt,
            )

            output_path = output_dir / f"{question.question_hash}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)

            successful += 1

        except Exception as e:
            logger.error(f"Failed to process response for {question.question_hash}: {e}")
            failed += 1

    logger.info(f"Stage 1 complete: {successful} successful, {failed} failed")
