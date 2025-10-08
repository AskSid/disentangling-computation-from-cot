"""Utilities for loading activation data and building datasets for probing."""

import hashlib
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from probing.code.experiment_config import ExperimentConfig
from probing.code.utils import set_seed

_DATASET_NAME = "edinburgh-dawg/mmlu-redux-2.0"
_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")


def _list_cached_configs(cache_dir: str) -> List[str]:
    """Return dataset config names available locally in the HF cache.

    Expected layout:
      {cache_dir}/edinburgh-dawg___mmlu-redux-2.0/{config}/{version}/{hash}/...
    """
    dataset_cache_root = os.path.join(cache_dir, "edinburgh-dawg___mmlu-redux-2.0")
    if not os.path.isdir(dataset_cache_root):
        return []
    try:
        entries = os.listdir(dataset_cache_root)
    except OSError:
        return []
    configs: List[str] = []
    for entry in entries:
        entry_path = os.path.join(dataset_cache_root, entry)
        if os.path.isdir(entry_path) and not entry.startswith("."):
            configs.append(entry)
    configs.sort()
    return configs


def write_global_question_split_json(
    output_path: str,
    train_fraction: float,
    seed: int,
    cache_dir: Optional[str],
    dataset_name: str,
) -> None:
    """Create a deterministic question-level split across all sections and save to JSON.

    Output schema:
    {
      "train": {"question_ids": [...]},
      "test": {"question_ids": [...]},
      "meta": {dataset_name, cache_dir, train_fraction, seed}
    }
    """
    cache = cache_dir or _DEFAULT_CACHE
    sections = _list_cached_configs(cache)

    all_train_ids: List[str] = []
    all_test_ids: List[str] = []

    for section in tqdm(sections):
        ds = load_dataset(dataset_name, section, cache_dir=cache)["test"]
        indices = list(range(len(ds)))
        question_ids = {
            idx: _question_hash(section, idx, ds[idx]["question"])
            for idx in indices
        }
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - float(train_fraction),
            stratify=None,
            random_state=seed,
        )
        train_sorted = sorted(train_idx)
        test_sorted = sorted(test_idx)
        all_train_ids.extend([question_ids[idx] for idx in train_sorted])
        all_test_ids.extend([question_ids[idx] for idx in test_sorted])

    split = {
        "train": {"question_ids": all_train_ids},
        "test": {"question_ids": all_test_ids},
    }

    payload: Dict[str, Any] = {
        "train": split["train"],
        "test": split["test"],
        "meta": {
            "dataset_name": dataset_name,
            "cache_dir": cache,
            "train_fraction": float(train_fraction),
            "seed": seed,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _question_hash(section: str, idx: int, question: str) -> str:
    seed = f"{section}_{idx}_{question}"
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def _parse_model_answer(metadata: Dict, question_id: str, verbose: bool = False) -> Optional[str]:
    if metadata.get("parsed_answer") in {"A", "B", "C", "D"}:
        return metadata["parsed_answer"]

    pattern = r"\\boxed\{[^}]*\}"
    response = metadata["full_response"]["choices"][0]["message"]["content"]
    match = re.search(pattern, response)
    if match:
        if verbose:
            print(f"CAUGHT and EXTRACTED (question {question_id}):\n{match.group(0)}")
        return re.sub(r"\\boxed\{|\\text\{|\}", "", match.group(0))

    # Try additional rescue patterns commonly seen in responses
    # 1) Markdown/Plaintext variants like **Answer: D**, **Answer:** D, **Answer**: D, Answer: D
    answer_letter_match = (
        re.search(r"\*\*Answer:\s*([ABCD])\*\*", response)
        or re.search(r"\*\*Answer:\*\*\s*([ABCD])", response)
        or re.search(r"\*\*Answer\*\*\s*:\s*([ABCD])", response)
        or re.search(r"\bAnswer\s*:\s*([ABCD])\b", response, flags=re.IGNORECASE)
    )
    if answer_letter_match:
        letter = answer_letter_match.group(1)
        if verbose:
            print(f"RESCUED plaintext answer (question {question_id}): {letter}")
        return letter

    # 2) JSON-like key with straight or curly quotes: "answer": "D"
    json_like_match = re.search(r"[\"\'\u201C\u201D]answer[\"\'\u201C\u201D]\s*:\s*[\"\'\u201C\u201D]([ABCD])[\"\'\u201C\u201D]", response, flags=re.IGNORECASE)
    if json_like_match:
        letter = json_like_match.group(1)
        if verbose:
            print(f"RESCUED JSON-like answer (question {question_id}): {letter}")
        return letter

    if verbose:
        print(f"CAUGHT and SKIPPED (question {question_id}): {response[-20:]}")
    return None


def _load_question_tokens(
    data_dir: str,
    section: str,
    idx: int,
    question_text: str,
    layer_idx: int,
    filter_pos_mode: str,
    bin_size: float,
    last_n_tokens: Optional[int],
) -> Optional[Tuple[List[torch.Tensor], List[Dict[str, Any]]]]:
    question_id = _question_hash(section, idx, question_text)
    activation_dir = os.path.join(data_dir, "stage2_activations", f"layer_{layer_idx}", question_id)
    if not os.path.exists(activation_dir):
        return None

    activation_files = os.listdir(activation_dir)
    if not activation_files:
        return None
    activations = torch.load(os.path.join(activation_dir, activation_files[0]))

    metadata_path = os.path.join(data_dir, "stage1_responses", f"{question_id}.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    raw_model_answer = _parse_model_answer(metadata, question_id)
    if raw_model_answer is None or raw_model_answer not in {"A", "B", "C", "D"}:
        return None

    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    model_ans = answer_map[raw_model_answer]
    correct_ans = answer_map[metadata["correct_answer"]]

    seq_len = activations.shape[0]
    
    # Compute bins based on filter_pos_mode
    if filter_pos_mode == "none":
        edges: Optional[np.ndarray] = None
    elif filter_pos_mode == "relative_pos":
        if not (0.0 < bin_size <= 1.0):
            raise ValueError("bin_size must be in (0, 1] for filter_pos_mode='relative_pos'")
        num_bins = max(1, int(np.ceil(1.0 / bin_size)))
        edges = np.linspace(0, seq_len, num_bins + 1)
    elif filter_pos_mode == "absolute_pos_front":
        width = int(bin_size)
        if width <= 0:
            raise ValueError("bin_size must be positive for absolute position modes")
        edges = np.arange(0, seq_len + width, width, dtype=float)
        if edges[-1] < seq_len:
            edges = np.append(edges, float(seq_len))
    elif filter_pos_mode == "absolute_pos_back":
        width = int(bin_size)
        if width <= 0:
            raise ValueError("bin_size must be positive for absolute position modes")
        # For absolute_pos_back, create bins based on the actual position values (0 to seq_len-1)
        # but in reverse order since absolute_pos_back = seq_len - 1 - token_idx
        max_pos_back = seq_len - 1
        edges = np.arange(0, max_pos_back + width, width, dtype=float)
        if edges[-1] < max_pos_back:
            edges = np.append(edges, float(max_pos_back + 1))
    else:
        raise ValueError("filter_pos_mode must be one of 'none', 'relative_pos', 'absolute_pos_front', or 'absolute_pos_back'")

    activations_list: List[torch.Tensor] = []
    metadata_list: List[Dict[str, Any]] = []

    # No averaging: we will emit concatenated last-N windows
    avg_windows: Optional[torch.Tensor] = None

    for token_idx in range(seq_len):
        if edges is None or seq_len <= 1:
            relative_pos = token_idx
        else:
            # Use the appropriate position value for binning based on filter_pos_mode
            if filter_pos_mode == "absolute_pos_back":
                # For absolute_pos_back, use the actual position value for binning
                pos_value = seq_len - 1 - token_idx
                relative_pos = int(np.digitize(pos_value, edges, right=False)) - 1
            else:
                # For other modes, use token_idx for binning
                relative_pos = int(np.digitize(token_idx, edges, right=False)) - 1
        
        # Concatenate last N tokens into a single feature vector [last_n_tokens * D]
        if last_n_tokens is not None:
            if token_idx >= last_n_tokens - 1:
                window = activations[token_idx - last_n_tokens + 1 : token_idx + 1]
                activations_list.append(window.reshape(-1))
            else:
                continue
        else:
            # If last_n_tokens is None, fall back to per-token activations as-is
            activations_list.append(activations[token_idx].reshape(-1))

        # Append metadata only for samples we actually emitted
        metadata_list.append(
            {
                "relative_pos": relative_pos,
                "absolute_pos_front": token_idx,
                "absolute_pos_back": seq_len - 1 - token_idx,
                "model_ans": model_ans,
                "correct_ans": correct_ans,
                "model_correct": int(model_ans == correct_ans),
                "category": section,
                "question_id": question_id,
            }
        )

    if not activations_list:
        return None
    return activations_list, metadata_list


class ActivationDataset(Dataset):
    """Torch dataset over activation tensors and precomputed labels."""

    def __init__(
        self,
        activations: Sequence[torch.Tensor],
        labels: Sequence[int],
        label_type: str,
        *,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if len(activations) != len(labels):
            raise ValueError("activations and labels must have the same length")
        self.activations: List[torch.Tensor] = list(activations)
        self.labels: List[int] = list(labels)
        self.label_type = label_type
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int):
        activation = self.activations[idx]
        label_value = self.labels[idx]
        if self.label_type in ["model_correct", "relative_position"]:
            label_tensor = torch.tensor(label_value, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label_value, dtype=torch.long)
        return activation, label_tensor


def create_activation_datasets(cfg: ExperimentConfig) -> Tuple[ActivationDataset, ActivationDataset]:
    """Load activation records and build train/test datasets for probing."""

    set_seed(cfg.seed)

    cache = cfg.cache_dir or _DEFAULT_CACHE

    categories = _list_cached_configs(cache)

    all_activations: List[torch.Tensor] = []
    all_metadata: List[Dict[str, Any]] = []
    question_indices: Dict[str, List[int]] = {}
    section_index_to_qid: Dict[str, Dict[int, str]] = {}

    for section in tqdm(categories, disable=cfg.disable_tqdm):
        dataset = load_dataset(
            _DATASET_NAME,
            section,
            cache_dir=cache,
        )["test"]
        allowed_indices: Optional[Set[int]] = None
        for idx in range(len(dataset)):
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            result = _load_question_tokens(
                data_dir=cfg.data_dir,
                section=section,
                idx=idx,
                question_text=dataset[idx]["question"],
                layer_idx=cfg.layer_idx,
                filter_pos_mode=cfg.filter_pos_mode,
                bin_size=cfg.bin_size,
                last_n_tokens=cfg.last_n_tokens,
            )
            if result is None:
                continue
            token_activations, token_metadata = result
            filtered_acts: List[torch.Tensor] = []
            filtered_meta: List[Dict[str, Any]] = []
            for act, meta in zip(token_activations, token_metadata):
                # Filter based on bin index
                if cfg.bin_idx is not None and cfg.filter_pos_mode != "none":
                    # Use the correct position value for filtering based on filter_pos_mode
                    if cfg.filter_pos_mode == "relative_pos":
                        # For relative_pos, use the relative_pos field
                        if int(meta["relative_pos"]) != int(cfg.bin_idx):
                            continue
                    elif cfg.filter_pos_mode in ["absolute_pos_front", "absolute_pos_back"]:
                        # For absolute position modes, recalculate edges and bin assignment
                        pos_value = meta[cfg.filter_pos_mode]
                        width = int(cfg.bin_size)
                        seq_len = len(token_activations)
                        
                        if cfg.filter_pos_mode == "absolute_pos_front":
                            edges = np.arange(0, seq_len + width, width, dtype=float)
                            if edges[-1] < seq_len:
                                edges = np.append(edges, float(seq_len))
                        elif cfg.filter_pos_mode == "absolute_pos_back":
                            max_pos_back = seq_len - 1
                            edges = np.arange(0, max_pos_back + width, width, dtype=float)
                            if edges[-1] < max_pos_back:
                                edges = np.append(edges, float(max_pos_back + 1))
                        
                        if edges is None or seq_len <= 1:
                            bin_idx = 0
                        else:
                            bin_idx = int(np.digitize(pos_value, edges, right=False)) - 1
                        if bin_idx != int(cfg.bin_idx):
                            continue
                filtered_acts.append(act)
                filtered_meta.append(meta)

            if not filtered_meta:
                continue

            question_id = filtered_meta[0]["question_id"]
            section_index_to_qid.setdefault(section, {})[idx] = question_id
            start_idx = len(all_activations)
            all_activations.extend(filtered_acts)
            all_metadata.extend(filtered_meta)
            question_indices[question_id] = list(range(start_idx, start_idx + len(filtered_meta)))

    if not all_metadata:
        raise RuntimeError("No activation records loaded; check data_dir and layer_idx")
    
    original_labels: List[int] = [int(meta[cfg.label_type]) for meta in all_metadata]
    unique_label_values = sorted(set(original_labels))
    label_to_index = {value: idx for idx, value in enumerate(unique_label_values)}
    all_labels: List[int] = [label_to_index[value] for value in original_labels]

    if cfg.label_type not in {"absolute_pos_front", "absolute_pos_back", "relative_position"}:
        display_counts = {value: 0 for value in unique_label_values}
        for value in original_labels:
            display_counts[value] += 1
        print(f"\nLabel distribution ({cfg.label_type}): {display_counts}")
    print(f"Total samples: {len(all_labels)}")

    def _should_stratify(labels: Sequence[int]) -> bool:
        """Return True when stratified sampling is safe for the provided labels."""
        if not labels:
            return False
        counts = Counter(labels)
        if len(counts) <= 1:
            return False
        return min(counts.values()) >= 2

    if cfg.split_mode == "token":
        indices = list(range(len(all_activations)))
        stratify = all_labels if _should_stratify(all_labels) else None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - cfg.train_fraction,
            stratify=stratify,
            random_state=cfg.seed,
        )
    elif cfg.split_mode == "question":
        with open(cfg.question_split_path, "r", encoding="utf-8") as f:
            question_split_config = json.load(f)

        train_id_targets = set(question_split_config["train"]["question_ids"])
        test_id_targets = set(question_split_config["test"]["question_ids"])
        
        train_idx = [idx for q in train_id_targets for idx in question_indices.get(q, [])]
        test_idx = [idx for q in test_id_targets for idx in question_indices.get(q, [])]
    else:
        raise ValueError("split_mode must be 'token' or 'question'")

    def _gather(indices: List[int]):
        activations_slice = [all_activations[i] for i in indices]
        labels_slice = [all_labels[i] for i in indices]
        if cfg.randomize_labels:
            random.shuffle(labels_slice)
        metadata_slice = [all_metadata[i] for i in indices]
        return activations_slice, labels_slice, metadata_slice

    train_activations, train_labels, train_metadata = _gather(train_idx)
    test_activations, test_labels, test_metadata = _gather(test_idx)

    train_dataset = ActivationDataset(
        train_activations,
        train_labels,
        cfg.label_type,
        metadata=train_metadata,
    )

    test_dataset = ActivationDataset(
        test_activations,
        test_labels,
        cfg.label_type,
        metadata=test_metadata,
    )

    print(f'    Number of train samples: {len(train_dataset)}')
    print(f'    Number of test samples: {len(test_dataset)}')

    return train_dataset, test_dataset

def create_dummy_datasets() -> Tuple[ActivationDataset, ActivationDataset]:
    """Create dummy datasets for testing."""
    train_dataset = ActivationDataset(
        [torch.randn(10, 7168, dtype=torch.bfloat16) for _ in range(100)],
        [torch.tensor([0, 0, 1, 0]) for _ in range(100)],
        'model_ans',
    )
    test_dataset = ActivationDataset(
        [torch.randn(10, 7168, dtype=torch.bfloat16) for _ in range(100)],
        [torch.tensor([0, 0, 1, 0]) for _ in range(100)],
        'model_ans',
    )
    return train_dataset, test_dataset

def main():
    write_global_question_split_json(
        output_path="/mnt/polished-lake/home/annabelma/disentangling-computation-from-cot/probing/data/question_split.json",
        train_fraction=0.8,
        seed=42,
        cache_dir=None,
        dataset_name=_DATASET_NAME,
    )

if __name__ == "__main__":
    main()
