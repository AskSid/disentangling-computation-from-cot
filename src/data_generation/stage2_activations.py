"""Stage 2: Activation harvesting using nnsight."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from nnsight import LanguageModel

from .data_gen_config import DataGenerationConfig

logger = logging.getLogger(__name__)


def load_model(config: DataGenerationConfig) -> LanguageModel:
    """Load model via nnsight."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.bfloat16)

    logger.info(f"Loading model {config.model_id} with dtype={config.dtype}")
    model = LanguageModel(
        config.model_id,
        device_map="cuda",
        dtype=dtype,
        trust_remote_code=True,
    )
    return model


def extract_layer_activations(
    model: LanguageModel,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Extract hidden states from a single layer using nnsight."""
    with model.trace(input_ids, scan=False, validate=False):
        acts = model.model.layers[layer_idx].output[0].save()

    return acts.squeeze(0).cpu()


def harvest_activations(
    config: DataGenerationConfig,
    responses_dir: Optional[Path] = None,
    shard_idx: Optional[int] = None,
    total_shards: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> None:
    """Extract layer-wise activations using nnsight."""
    responses_dir = responses_dir or config.responses_dir
    output_dir = config.activations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    response_files = sorted(responses_dir.glob("*.json"))
    logger.info(f"Found {len(response_files)} response files")

    if shard_idx is not None and total_shards is not None:
        response_files = [f for i, f in enumerate(response_files) if i % total_shards == shard_idx]
        logger.info(f"Shard {shard_idx + 1}/{total_shards}: processing {len(response_files)} files")

    if not response_files:
        logger.warning("No response files found.")
        return

    if layer_idx is not None:
        layers = [layer_idx]
    else:
        layers = config.get_layers_list()
    logger.info(f"Will extract activations for {len(layers)} layers: {layers[:5]}...")

    for l in layers:
        (output_dir / f"layer_{l}").mkdir(parents=True, exist_ok=True)

    model = load_model(config)

    processed = 0
    skipped = 0
    failed = 0

    desc = "Harvesting activations"
    if shard_idx is not None:
        desc = f"Shard {shard_idx + 1}/{total_shards}"
    elif layer_idx is not None:
        desc = f"Layer {layer_idx}"

    for response_path in tqdm(response_files, desc=desc):
        question_hash = response_path.stem

        first_out = output_dir / f"layer_{layers[0]}" / question_hash
        if first_out.exists() and list(first_out.glob("*.pt")):
            skipped += 1
            continue

        try:
            with response_path.open("r", encoding="utf-8") as f:
                response_data = json.load(f)

            complete_rollout = response_data.get("complete_rollout", "")
            if not complete_rollout:
                failed += 1
                continue

            input_ids = model.tokenizer.encode(
                complete_rollout,
                return_tensors="pt",
                add_special_tokens=False,
            ).to("cuda")

            num_tokens = input_ids.shape[1]

            for l in layers:
                question_output_dir = output_dir / f"layer_{l}" / question_hash
                question_output_dir.mkdir(parents=True, exist_ok=True)

                acts_cpu = extract_layer_activations(model, input_ids, l)

                output_path = question_output_dir / f"{num_tokens}_tokens.pt"
                torch.save(acts_cpu, output_path)

            processed += 1

        except Exception as e:
            logger.error(f"Failed to process {question_hash}: {e}")
            failed += 1

    logger.info(f"Complete: {processed} processed, {skipped} skipped, {failed} failed")
