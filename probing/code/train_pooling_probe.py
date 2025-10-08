#!/usr/bin/env python3
"""
Consolidated pooling probe training script.
Supports all pooling types from the paper with proper padding handling.
"""

import json
import os
import argparse
import hashlib
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
import yaml

from pooling_config import PoolingConfig


class PoolingProbe(nn.Module):
    """Pooling probe that aggregates sequence-level activations according to different strategies."""
    
    def __init__(self, input_dim: int, output_dim: int, probe_type: str, 
                 rolling_window: int = 5, softmax_temperature: float = 1.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.probe_type = probe_type
        self.rolling_window = rolling_window
        self.softmax_temperature = softmax_temperature
        self.dtype = dtype
        
        if probe_type == 'attention':
            # Attention probe needs separate query and value projections
            self.theta_q = nn.Linear(input_dim, input_dim, bias=False, dtype=dtype)
            self.theta_v = nn.Linear(input_dim, output_dim, bias=False, dtype=dtype)
        else:
            # All other probe types use a single linear projection
            self.theta = nn.Linear(input_dim, output_dim, bias=False, dtype=dtype)
    
    def forward(self, A, positions=None):
        """
        Forward pass for different pooling strategies.
        A: (batch_size, seq_len, hidden_dim) activation tensor (may be padded)
        positions: (batch_size,) tensor of positions for position-based training
        """
        batch_size, seq_len, hidden_dim = A.shape
        
        # Create attention mask for padded sequences (assumes padding value is 0.0)
        attention_mask = torch.any(A != 0.0, dim=-1)  # (batch_size, seq_len)
        
        # Handle position-based training
        if positions is not None:
            return self._forward_position_based(A, attention_mask, positions)
        else:
            return self._forward_sequence_based(A, attention_mask)
    
    def _forward_position_based(self, A, attention_mask, positions):
        """Forward pass for position-based training (causal attention)."""
        batch_size, seq_len, hidden_dim = A.shape
        
        if self.probe_type == 'mean':
            # Causal mean: average only up to current position
            batch_outputs = []
            for b in range(batch_size):
                pos = positions[b].item()
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if pos >= valid_length:
                    pos = valid_length - 1
                
                # Only use activations up to current position
                causal_activations = A[b, :pos+1, :]
                mean_activation = causal_activations.mean(dim=0)
                output = self.theta(mean_activation)
                batch_outputs.append(output)
            
            return torch.stack(batch_outputs, dim=0)
        
        elif self.probe_type == 'max':
            # Causal max: maximum only up to current position
            batch_outputs = []
            for b in range(batch_size):
                pos = positions[b].item()
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if pos >= valid_length:
                    pos = valid_length - 1
                
                # Only use activations up to current position
                causal_activations = A[b, :pos+1, :]
                scores = self.theta(causal_activations)
                max_score = scores.max(dim=0)[0]
                batch_outputs.append(max_score)
            
            return torch.stack(batch_outputs, dim=0)
        
        elif self.probe_type == 'rolling_means':
            # For rolling means at a specific position, use causal window
            batch_outputs = []
            for b in range(batch_size):
                pos = positions[b].item()
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if pos >= valid_length:
                    pos = valid_length - 1
                
                # Get causal window ending at position
                window_start = max(0, pos - self.rolling_window + 1)
                causal_activations = A[b, window_start:pos+1, :]  # Only up to current position
                
                if causal_activations.shape[0] < self.rolling_window:
                    # If window is smaller, just use mean
                    output = self.theta(causal_activations.mean(dim=0))
                else:
                    # Apply rolling means to the causal window
                    scores = self.theta(causal_activations)
                    rolling_means = []
                    for i in range(causal_activations.shape[0] - self.rolling_window + 1):
                        window_scores = scores[i:i+self.rolling_window, :]
                        mean_score = window_scores.mean(dim=0)
                        rolling_means.append(mean_score)
                    rolling_means = torch.stack(rolling_means, dim=0)
                    output = rolling_means.max(dim=0)[0]
                
                batch_outputs.append(output)
            
            return torch.stack(batch_outputs, dim=0)
        
        elif self.probe_type == 'softmax':
            # Causal softmax: only attend to positions up to current position
            batch_outputs = []
            for b in range(batch_size):
                pos = positions[b].item()
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if pos >= valid_length:
                    pos = valid_length - 1
                
                # Only use activations up to current position
                causal_activations = A[b, :pos+1, :]
                scores = self.theta(causal_activations)
                
                # Apply softmax with temperature (causal)
                weights = torch.softmax(scores / self.softmax_temperature, dim=0)
                weighted_sum = (weights * scores).sum(dim=0)
                batch_outputs.append(weighted_sum)
            
            return torch.stack(batch_outputs, dim=0)
        
        elif self.probe_type == 'attention':
            # Causal attention: only attend to previous positions
            batch_outputs = []
            for b in range(batch_size):
                pos = positions[b].item()
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if pos >= valid_length:
                    pos = valid_length - 1
                
                # Only use activations up to current position
                causal_activations = A[b, :pos+1, :]
                queries = self.theta_q(causal_activations)
                values = self.theta_v(causal_activations)
                
                # Compute causal attention scores
                attention_scores = torch.sum(queries * queries, dim=-1, keepdim=True)
                attention_weights = torch.softmax(attention_scores, dim=0)  # Causal softmax
                attended_values = (attention_weights * values).sum(dim=0)
                batch_outputs.append(attended_values)
            
            return torch.stack(batch_outputs, dim=0)
        
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")
    
    def _forward_sequence_based(self, A, attention_mask):
        """Original sequence-based forward pass."""
        batch_size, seq_len, hidden_dim = A.shape
        
        if self.probe_type == 'mean':
            # Mean pooling: average across sequence positions (excluding padding)
            masked_A = A * attention_mask.unsqueeze(-1).float()
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            mean_activations = masked_A.sum(dim=1) / seq_lengths
            return self.theta(mean_activations)
        
        elif self.probe_type == 'max':
            # Max pooling: maximum across sequence positions (excluding padding)
            scores = self.theta(A)
            masked_scores = scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            return masked_scores.max(dim=1)[0]
        
        
        elif self.probe_type == 'rolling_means':
            # Max of rolling means with window size T (excluding padding)
            scores = self.theta(A)
            
            batch_rolling_means = []
            for b in range(batch_size):
                seq_mask = attention_mask[b]
                valid_length = seq_mask.sum().item()
                
                if valid_length < self.rolling_window:
                    valid_scores = scores[b, seq_mask, :]
                    mean_score = valid_scores.mean(dim=0)
                    batch_rolling_means.append(mean_score)
                else:
                    valid_scores = scores[b, seq_mask, :]
                    rolling_means = []
                    for i in range(valid_length - self.rolling_window + 1):
                        window_scores = valid_scores[i:i+self.rolling_window, :]
                        mean_score = window_scores.mean(dim=0)
                        rolling_means.append(mean_score)
                    
                    if rolling_means:
                        rolling_means = torch.stack(rolling_means, dim=0)
                        max_score = rolling_means.max(dim=0)[0]
                    else:
                        max_score = valid_scores.mean(dim=0)
                    batch_rolling_means.append(max_score)
            
            return torch.stack(batch_rolling_means, dim=0)
        
        elif self.probe_type == 'softmax':
            # Softmax weighted sum with temperature (excluding padding)
            scores = self.theta(A)
            masked_scores = scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            weights = torch.softmax(masked_scores / self.softmax_temperature, dim=1)
            weighted_sum = (weights * scores).sum(dim=1)
            return weighted_sum
        
        elif self.probe_type == 'attention':
            # Attention mechanism: use query to weight values (excluding padding)
            queries = self.theta_q(A)
            values = self.theta_v(A)
            
            attention_scores = torch.sum(queries * queries, dim=-1, keepdim=True)
            masked_attention_scores = attention_scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            attention_weights = torch.softmax(masked_attention_scores, dim=1)
            attended_values = (attention_weights * values).sum(dim=1)
            return attended_values
        
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")


class ActivationDataset(Dataset):
    """Dataset for sequence-level activations with labels."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        activation = self.data[idx]['activations']
        label = torch.tensor(self.data[idx]['label'], dtype=torch.long if isinstance(self.data[idx]['label'], int) else torch.float32)
        
        # Include position information if available
        if 'position' in self.data[idx]:
            position = torch.tensor(self.data[idx]['position'], dtype=torch.long)
            return activation, label, position
        else:
            return activation, label


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    if len(batch[0]) == 3:  # Has position information
        activations, labels, positions = zip(*batch)
        activations_padded = pad_sequence(activations, batch_first=True, padding_value=0.0)
        labels_stacked = torch.stack(labels)
        positions_stacked = torch.stack(positions)
        return activations_padded, labels_stacked, positions_stacked
    else:  # No position information
        activations, labels = zip(*batch)
        activations_padded = pad_sequence(activations, batch_first=True, padding_value=0.0)
        labels_stacked = torch.stack(labels)
        return activations_padded, labels_stacked


def get_question_data(data_dir: str, section: str, idx: int, question_name: str, 
                     layer_idx: int, config: PoolingConfig) -> Optional[List[Dict]]:
    """Load activations and labels for a single MMLU question."""
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    question_id = hashlib.md5(f"{section}_{idx}_{question_name}".encode()).hexdigest()[:12]

    activation_dir = f'{data_dir}/stage2_activations/layer_{layer_idx}/{question_id}'
    if not os.path.exists(activation_dir):
        return None
    
    activation_file = os.listdir(activation_dir)[0]
    path = f'{activation_dir}/{activation_file}'
    activations = torch.load(path).float()  # Convert to float32

    metadata_file = f'{data_dir}/stage1_responses/{question_id}.json'
    metadata = json.load(open(metadata_file))
    
    # Parse model answer
    if metadata['parsed_answer'] in answer_mapping:
        model_answer = metadata['parsed_answer']
    else:
        pattern = r'\\boxed\{[^}]*\}'
        model_response = metadata["full_response"]["choices"][0]["message"]["content"]
        match = re.search(pattern, model_response)
        if match:
            boxed_content = match.group(0)
            model_answer = re.sub(r'\\boxed\{|\\text\{|\}', '', boxed_content)
        else:
            return None

    model_answer = answer_mapping[model_answer]
    correct_answer = answer_mapping[metadata['correct_answer']]
    
    # If using position-based training, create multiple samples per question (all probe types now)
    if config.use_position_based:
        seq_len = activations.shape[0]
        samples = []
        
        # For causal probing, we only need ONE sample per question at the target position
        if config.filter_pos_mode == 'relative_pos':
            # Relative position binning - use the middle of the bin
            bin_size_tokens = int(config.bin_size * seq_len)
            bin_start = config.bin_idx * bin_size_tokens
            bin_end = min(bin_start + bin_size_tokens, seq_len)
            target_pos = (bin_start + bin_end) // 2  # Middle of the bin
            
        elif config.filter_pos_mode == 'absolute_pos_front':
            # Absolute position from front - use the middle of the bin
            bin_size_tokens = int(config.bin_size)
            bin_start = config.bin_idx * bin_size_tokens
            bin_end = min(bin_start + bin_size_tokens, seq_len)
            target_pos = (bin_start + bin_end) // 2  # Middle of the bin
            
        elif config.filter_pos_mode == 'absolute_pos_back':
            # Absolute position from back - use the middle of the bin
            bin_size_tokens = int(config.bin_size)
            bin_end = seq_len - (config.bin_idx * bin_size_tokens)
            bin_start = max(bin_end - bin_size_tokens, 0)
            target_pos = (bin_start + bin_end) // 2  # Middle of the bin
            
        else:  # 'none'
            # Use a specific position (e.g., middle of sequence)
            target_pos = seq_len // 2
        
        # Create ONE sample per question at the target position
        if target_pos < seq_len:
            samples.append({
                'activations': activations,
                'position': target_pos,  # Track the position
                'model_answer': model_answer, 
                'correct_answer': correct_answer, 
                'category': section,
                'question_id': question_id
            })
        
        return samples
    
    else:
        # Standard sequence-level training
        return [{
            'activations': activations,
            'model_answer': model_answer, 
            'correct_answer': correct_answer, 
            'category': section,
            'question_id': question_id
        }]



def load_data(config: PoolingConfig) -> Tuple[ActivationDataset, ActivationDataset]:
    """Load and prepare datasets for pooling probes."""
    categories = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
    activation_data = []
    
    # Set up cache directory
    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)
    
    for category in tqdm(categories, desc="Loading categories", disable=config.disable_tqdm):
        ds = load_dataset('edinburgh-dawg/mmlu-redux-2.0', category, 
                         cache_dir=config.cache_dir if config.cache_dir else None)['test']
        
        for i in range(len(ds)):
            question_samples = get_question_data(
                config.data_dir, category, i, ds[i]['question'], config.layer_idx, config
            )
            if question_samples is not None:
                activation_data.extend(question_samples)
    
    # Prepare labels and group samples by question to avoid leakage across splits
    for item in activation_data:
        if config.label_type == 'model_answer':
            item['label'] = item['model_answer']
        elif config.label_type == 'correct_answer':
            item['label'] = item['correct_answer']
        elif config.label_type == 'model_correct':
            item['label'] = item['model_answer'] == item['correct_answer']

    # Group by question_id
    question_id_to_items: Dict[str, list] = {}
    for item in activation_data:
        qid = item['question_id']
        question_id_to_items.setdefault(qid, []).append(item)

    # Build per-question labels for stratification
    question_ids = list(question_id_to_items.keys())
    question_labels = []
    for qid in question_ids:
        labels_in_q = [it['label'] for it in question_id_to_items[qid]]
        question_labels.append(labels_in_q[0])

    # Print distribution
    label_counts = Counter(question_labels)
    total = len(question_labels)
    print(f"\nQuestion-level label distribution ({config.label_type}): {dict(label_counts)}")
    print(f"Total questions: {total}")

    # Split by question_id to avoid leakage
    q_train, q_test = train_test_split(
        question_ids,
        test_size=1 - config.train_fraction,
        stratify=question_labels,
        random_state=config.seed,
    )

    # Expand question split back to samples
    train_data = [it for qid in q_train for it in question_id_to_items[qid]]
    test_data = [it for qid in q_test for it in question_id_to_items[qid]]
    
    train_dataset = ActivationDataset(train_data)
    test_dataset = ActivationDataset(test_data)
    
    return train_dataset, test_dataset


def create_model(config: PoolingConfig) -> PoolingProbe:
    """Create the pooling probe model."""
    if config.label_type in ['model_answer', 'correct_answer']:
        output_dim = 4
    elif config.label_type == 'model_correct':
        output_dim = 1
    else:
        raise ValueError(f"Unknown label_type: {config.label_type}")
    
    model = PoolingProbe(
        input_dim=7168,
        output_dim=output_dim,
        probe_type=config.probe_type,
        rolling_window=config.rolling_window,
        softmax_temperature=config.softmax_temperature,
        dtype=torch.float32  # Ensure consistent dtype
    )
    model.to(config.device)
    return model


def train_model(model: PoolingProbe, train_dataset: ActivationDataset, 
               test_dataset: ActivationDataset, config: PoolingConfig) -> Tuple[Dict, float]:
    """Train and evaluate the model."""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                            num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                           num_workers=0, collate_fn=custom_collate_fn)

    # Setup optimizer
    if config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                   weight_decay=config.weight_decay)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")

    # Setup loss function
    if config.label_type in ['model_answer', 'correct_answer']:
        criterion = nn.CrossEntropyLoss()
        is_binary = False
    elif config.label_type == 'model_correct':
        criterion = nn.BCEWithLogitsLoss()
        is_binary = True
    
    # Create output directory
    run_output_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, 'models'), exist_ok=True)
    
    best_test_acc = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        
        for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=config.disable_tqdm):
            if len(batch_data) == 3:  # Has position information
                batch_inputs, batch_labels, batch_positions = batch_data
                batch_inputs = batch_inputs.to(config.device)
                batch_labels = batch_labels.to(config.device)
                batch_positions = batch_positions.to(config.device)
                
                optimizer.zero_grad()
                
                if is_binary:
                    logits = model(batch_inputs, batch_positions).float()
                    loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                        train_correct += (preds == batch_labels.view(-1)).sum().item()
                        train_samples += batch_labels.numel()
                else:
                    logits = model(batch_inputs, batch_positions)
                    loss = criterion(logits.float(), batch_labels.long())
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        train_correct += (preds == batch_labels).sum().item()
                        train_samples += batch_labels.size(0)
            else:  # No position information
                batch_inputs, batch_labels = batch_data
                batch_inputs = batch_inputs.to(config.device)
                batch_labels = batch_labels.to(config.device)

                optimizer.zero_grad()
                
                if is_binary:
                    logits = model(batch_inputs).float()
                    loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                        train_correct += (preds == batch_labels.view(-1)).sum().item()
                        train_samples += batch_labels.numel()
                else:
                    logits = model(batch_inputs)
                    loss = criterion(logits.float(), batch_labels.long())
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        train_correct += (preds == batch_labels).sum().item()
                        train_samples += batch_labels.size(0)

            train_loss_sum += loss.item() * (batch_labels.numel() if is_binary else batch_labels.size(0))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_loss_sum = 0.0
            test_correct = 0
            test_samples = 0
            
            for batch_data in tqdm(test_loader, desc=f"Testing Epoch {epoch}", disable=config.disable_tqdm):
                if len(batch_data) == 3:  # Has position information
                    batch_inputs, batch_labels, batch_positions = batch_data
                    batch_inputs = batch_inputs.to(config.device)
                    batch_labels = batch_labels.to(config.device)
                    batch_positions = batch_positions.to(config.device)

                    if is_binary:
                        logits = model(batch_inputs, batch_positions).float()
                        loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                        preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                        test_correct += (preds == batch_labels.view(-1)).sum().item()
                        test_samples += batch_labels.numel()
                        test_loss_sum += loss.item() * batch_labels.numel()
                    else:
                        logits = model(batch_inputs, batch_positions)
                        loss = criterion(logits.float(), batch_labels.long())
                        preds = torch.argmax(logits, dim=1)
                        test_correct += (preds == batch_labels).sum().item()
                        test_samples += batch_labels.size(0)
                        test_loss_sum += loss.item() * batch_labels.size(0)
                else:  # No position information
                    batch_inputs, batch_labels = batch_data
                    batch_inputs = batch_inputs.to(config.device)
                    batch_labels = batch_labels.to(config.device)

                    if is_binary:
                        logits = model(batch_inputs).float()
                        loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                        preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                        test_correct += (preds == batch_labels.view(-1)).sum().item()
                        test_samples += batch_labels.numel()
                        test_loss_sum += loss.item() * batch_labels.numel()
                    else:
                        logits = model(batch_inputs)
                        loss = criterion(logits.float(), batch_labels.long())
                        preds = torch.argmax(logits, dim=1)
                        test_correct += (preds == batch_labels).sum().item()
                        test_samples += batch_labels.size(0)
                        test_loss_sum += loss.item() * batch_labels.size(0)

        # Calculate metrics
        avg_train_loss = train_loss_sum / max(1, train_samples)
        avg_test_loss = test_loss_sum / max(1, test_samples)
        test_acc = (test_correct / max(1, test_samples)) * 100.0
        
        if test_acc > best_test_acc:
            best_model_state = model.state_dict()
            best_test_acc = test_acc
        
        end_time = time.time()
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | Time: {end_time - start_time:.2f}s")

    # Save results
    results_path = os.path.join(run_output_dir, f'{config.run_name}.jsonl')
    log_record = {
        'run_name': config.run_name,
        'layer_idx': config.layer_idx,
        'best_test_acc': best_test_acc,
        'config': config.to_dict(),
    }
    
    with open(results_path, 'a') as f:
        f.write(json.dumps(log_record) + "\n")
    
    # Save best model
    if best_model_state is not None:
        model_path = os.path.join(run_output_dir, 'models', f'{config.run_name}_best.pt')
        torch.save(best_model_state, model_path)
    
    return best_model_state, best_test_acc


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train pooling probe")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, choices=["single", "sweep", "all_layers"], 
                       default="single", help="Training mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    params = raw_cfg.get('params', {})
    config = PoolingConfig.from_dict(params)
    
    set_seed(config.seed)
    print(f"Starting {config.run_name} with mode {args.mode}")
    print(f"Config: {config}")

    if args.mode == 'single':
        # Single training run
        train_dataset, test_dataset = load_data(config)
        print(f"Train len: {len(train_dataset)} | Test len: {len(test_dataset)}")
        
        model = create_model(config)
        best_model, best_acc = train_model(model, train_dataset, test_dataset, config)
        print(f"Best accuracy: {best_acc:.2f}%")
        
    elif args.mode == 'sweep':
        # Hyperparameter sweep
        sweep_grid = raw_cfg.get('sweep', {})
        if not sweep_grid:
            print("No sweep grid provided.")
            return
        
        # Load data once
        train_dataset, test_dataset = load_data(config)
        print(f"Train len: {len(train_dataset)} | Test len: {len(test_dataset)}")
        
        # Generate all combinations
        import itertools
        param_names = list(sweep_grid.keys())
        values_product = list(itertools.product(*[sweep_grid[name] for name in param_names]))
        print(f"Total combinations: {len(values_product)}")
        
        best_acc = 0.0
        best_config = None
        
        for i, combination in enumerate(values_product):
            print(f"\n{'='*50}")
            print(f"Combination {i+1}/{len(values_product)}: {param_names} = {combination}")
            print(f"{'='*50}")
            
            # Update config
            sweep_params = dict(params)
            for name, value in zip(param_names, combination):
                sweep_params[name] = value
            
            sweep_config = PoolingConfig.from_dict(sweep_params)
            sweep_config.run_name = f"{config.run_name}_{i+1}"
            
            model = create_model(sweep_config)
            _, test_acc = train_model(model, train_dataset, test_dataset, sweep_config)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_config = sweep_config
            
            print(f"Combination {i+1} completed with accuracy: {test_acc:.2f}%")
        
        print(f"\nSweep completed. Best accuracy: {best_acc:.2f}%")
        
    elif args.mode == 'all_layers':
        # Train on all layers
        for layer_idx in range(61):
            print(f"\n{'='*50}")
            print(f"Training probe for layer {layer_idx}")
            print(f"{'='*50}")
            
            layer_params = dict(params)
            layer_params['layer_idx'] = layer_idx
            layer_config = PoolingConfig.from_dict(layer_params)
            
            try:
                train_dataset, test_dataset = load_data(layer_config)
                if len(train_dataset) == 0:
                    print(f"No data found for layer {layer_idx}, skipping...")
                    continue
                
                model = create_model(layer_config)
                _, best_acc = train_model(model, train_dataset, test_dataset, layer_config)
                print(f"Layer {layer_idx} completed with accuracy: {best_acc:.2f}%")
                
            except Exception as e:
                print(f"Error training layer {layer_idx}: {e}")
                continue


if __name__ == "__main__":
    main()
