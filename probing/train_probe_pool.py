import json
import itertools
import hashlib
import os
import re
import argparse
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from dataclasses import dataclass, replace, asdict
import random
import yaml
import time
import numpy as np

@dataclass
class TrainingConfig:
    train_test_split: float
    batch_size: int
    device: str
    label_type: str
    learning_rate: float
    weight_decay: float
    optimizer_type: str
    dtype: torch.dtype
    num_epochs: int
    output_dir: str
    run_name: str
    disable_tqdm: bool
    seed: int
    layer_idx: int
    probe_type: str  # 'mean', 'max', 'last_token', 'rolling_means', 'softmax', 'attention'
    rolling_window: int  # For rolling means probe
    softmax_temperature: float  # For softmax probe

class PoolingProbe(torch.nn.Module):
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
            self.theta_q = torch.nn.Linear(input_dim, input_dim, dtype=dtype, bias=False)
            self.theta_v = torch.nn.Linear(input_dim, output_dim, dtype=dtype, bias=False)
        else:
            # All other probe types use a single linear projection
            self.theta = torch.nn.Linear(input_dim, output_dim, dtype=dtype, bias=False)
    
    def forward(self, A):
        """
        Forward pass for different pooling strategies.
        A: (batch_size, seq_len, hidden_dim) activation tensor (may be padded)
        """
        batch_size, seq_len, hidden_dim = A.shape
        
        # Create attention mask for padded sequences (assumes padding value is 0.0)
        # For sequences, we can detect padding by checking if all activations in a position are zero
        attention_mask = torch.any(A != 0.0, dim=-1)  # (batch_size, seq_len)
        
        if self.probe_type == 'mean':
            # Mean pooling: average across sequence positions (excluding padding)
            # Mask out padding positions
            masked_A = A * attention_mask.unsqueeze(-1).float()
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            mean_activations = masked_A.sum(dim=1) / seq_lengths  # (batch_size, hidden_dim)
            return self.theta(mean_activations)
        
        elif self.probe_type == 'max':
            # Max pooling: maximum across sequence positions (excluding padding)
            # Apply linear transformation first, then take max
            scores = self.theta(A)  # (batch_size, seq_len, output_dim)
            # Mask out padding positions with large negative values
            masked_scores = scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            return masked_scores.max(dim=1)[0]  # (batch_size, output_dim)
        
        elif self.probe_type == 'last_token':
            # Last token: use the last non-padded token
            # Find the last non-padded position for each sequence
            last_indices = attention_mask.sum(dim=1) - 1  # (batch_size,)
            batch_indices = torch.arange(batch_size, device=A.device)
            last_activations = A[batch_indices, last_indices, :]  # (batch_size, hidden_dim)
            return self.theta(last_activations)
        
        elif self.probe_type == 'rolling_means':
            # Max of rolling means with window size T (excluding padding)
            # Apply linear transformation first
            scores = self.theta(A)  # (batch_size, seq_len, output_dim)
            
            # For each sequence, compute rolling means only for valid positions
            batch_rolling_means = []
            for b in range(batch_size):
                seq_mask = attention_mask[b]  # (seq_len,)
                valid_length = seq_mask.sum().item()
                
                if valid_length < self.rolling_window:
                    # If sequence is shorter than window, use mean pooling
                    valid_scores = scores[b, seq_mask, :]  # (valid_length, output_dim)
                    mean_score = valid_scores.mean(dim=0)  # (output_dim,)
                    batch_rolling_means.append(mean_score)
                else:
                    # Compute rolling means for valid positions
                    valid_scores = scores[b, seq_mask, :]  # (valid_length, output_dim)
                    rolling_means = []
                    for i in range(valid_length - self.rolling_window + 1):
                        window_scores = valid_scores[i:i+self.rolling_window, :]  # (rolling_window, output_dim)
                        mean_score = window_scores.mean(dim=0)  # (output_dim,)
                        rolling_means.append(mean_score)
                    
                    if rolling_means:
                        rolling_means = torch.stack(rolling_means, dim=0)  # (num_windows, output_dim)
                        max_score = rolling_means.max(dim=0)[0]  # (output_dim,)
                    else:
                        max_score = valid_scores.mean(dim=0)  # (output_dim,)
                    batch_rolling_means.append(max_score)
            
            return torch.stack(batch_rolling_means, dim=0)  # (batch_size, output_dim)
        
        elif self.probe_type == 'softmax':
            # Softmax weighted sum with temperature (excluding padding)
            scores = self.theta(A)  # (batch_size, seq_len, output_dim)
            
            # Mask out padding positions with large negative values for softmax
            masked_scores = scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            
            # Apply softmax with temperature across sequence dimension
            weights = torch.softmax(masked_scores / self.softmax_temperature, dim=1)  # (batch_size, seq_len, output_dim)
            
            # Weighted sum
            weighted_sum = (weights * scores).sum(dim=1)  # (batch_size, output_dim)
            return weighted_sum
        
        elif self.probe_type == 'attention':
            # Attention mechanism: use query to weight values (excluding padding)
            queries = self.theta_q(A)  # (batch_size, seq_len, hidden_dim)
            values = self.theta_v(A)  # (batch_size, seq_len, output_dim)
            
            # Compute attention weights (using dot product attention)
            attention_scores = torch.sum(queries * queries, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
            
            # Mask out padding positions with large negative values
            masked_attention_scores = attention_scores.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            
            # Apply softmax to get attention weights
            attention_weights = torch.softmax(masked_attention_scores, dim=1)  # (batch_size, seq_len, 1)
            
            # Apply attention weights to values
            attended_values = (attention_weights * values).sum(dim=1)  # (batch_size, output_dim)
            return attended_values
        
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

def set_seed(seed: int):
    """Set random seeds for reproducibility across python, numpy, and torch."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_question_data(data_dir: str, section: str, idx: int, question_name: str, layer_idx: int, verbose: bool = False, disable_tqdm: bool = False) -> dict:
    """
    Load activations and labels for a single MMLU question.

    Returns a single sequence-level record with 'activations', 'model_answer', 
    and 'correct_answer', or None if parsing fails.
    """
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    question_id = hashlib.md5(f"{section}_{idx}_{question_name}".encode()).hexdigest()[:12]

    activation_dir = f'{data_dir}/stage2_activations/layer_{layer_idx}/{question_id}'
    if not os.path.exists(activation_dir):
        return None
    activation_file = os.listdir(activation_dir)[0]
    path = f'{activation_dir}/{activation_file}'
    activations = torch.load(path)

    metadata_file = f'{data_dir}/stage1_responses/{question_id}.json'
    metadata = json.load(open(metadata_file))
    # Try to further extract the answer from the model's response if it's not in the metadata, otherwise don't include it in the dataset
    if metadata['parsed_answer'] in answer_mapping:
        model_answer = metadata['parsed_answer']
    else:
        pattern = r'\\boxed\{[^}]*\}'
        model_response = metadata["full_response"]["choices"][0]["message"]["content"]
        match = re.search(pattern, model_response)
        if match:
            boxed_content = match.group(0)
            print(f"CAUGHT and EXTRACTED:\n{boxed_content}") if verbose else None
            model_answer = re.sub(r'\\boxed\{|\\text\{|\}', '', boxed_content)
        else:
            print(f"CAUGHT and SKIPPED:\n{model_response[-20:]}") if verbose else None
            return None

    model_answer = answer_mapping[model_answer]
    correct_answer = answer_mapping[metadata['correct_answer']]
    
    # Return sequence-level data instead of token-level
    return {
        'activations': activations,  # Full sequence of activations
        'model_answer': model_answer, 
        'correct_answer': correct_answer, 
        'category': section,
        'question_id': question_id
    }

class ActivationDataset(Dataset):
    """Torch dataset for pairing sequence-level activations with relevant label type."""
    def __init__(self, ds: list, include_metadata: bool = False):
        self.ds = ds
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        activation = self.ds[idx]['activations']  # This is now a full sequence
        label = torch.tensor(self.ds[idx]['label'], dtype=torch.long if isinstance(self.ds[idx]['label'], int) else torch.float32)
        if self.include_metadata:
            category = self.ds[idx]['category']  # Keep as string for now
            question_id = self.ds[idx]['question_id']
            return activation, label, category, question_id
        return activation, label

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the same length in each batch.
    """
    activations, labels = zip(*batch)
    
    # Pad sequences to the same length
    activations_padded = pad_sequence(activations, batch_first=True, padding_value=0.0)
    
    # Stack labels
    labels_stacked = torch.stack(labels)
    
    return activations_padded, labels_stacked

def get_datasets(cfg: TrainingConfig, data_dir: str, layer_idx: int):
    """Build train/test ActivationDatasets on CPU for all MMLU categories with stratified split."""
    categories = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
    activation_data = []
    for category in tqdm(categories, disable=cfg.disable_tqdm):
        # changing the cache dir so that i can run as well
        user = os.environ.get("USER", os.path.basename(os.path.expanduser("~")))
        cache_dir = f"/mnt/polished-lake/home/{user}/.cache/huggingface/datasets"
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset('edinburgh-dawg/mmlu-redux-2.0', category, cache_dir=cache_dir)['test']
        for i in range(len(ds)):
            question_data = get_question_data(data_dir, category, i, ds[i]['question'], layer_idx, verbose=False, disable_tqdm=cfg.disable_tqdm)
            if question_data is not None:
                activation_data.append(question_data)  # Append single sequence instead of extending
    
    # Prepare labels for stratified split
    labels = []
    for item in activation_data:
        if cfg.label_type == 'model_answer':
            item['label'] = item['model_answer']
        elif cfg.label_type == 'correct_answer':
            item['label'] = item['correct_answer']
        elif cfg.label_type == 'model_correct':
            item['label'] = item['model_answer'] == item['correct_answer']
        labels.append(item['label'])
    
    # Print simple label distribution
    label_counts = Counter(labels)
    total = len(labels)
    print(f"\nLabel distribution ({cfg.label_type}): {dict(label_counts)}")
    print(f"Total samples: {total}\n")
    
    # Stratified split
    train_data, test_data = train_test_split(
        activation_data, 
        test_size=1-cfg.train_test_split, 
        stratify=labels,
        random_state=cfg.seed
    )
    
    train_dataset = ActivationDataset(train_data)
    test_dataset = ActivationDataset(test_data)
    return train_dataset, test_dataset

def create_model(cfg: TrainingConfig):
    """
    Create the pooling probe model. 
    'model_answer' and 'correct_answer' are one of the four options, and 'model_correct' is a yes/no label so a single probability.
    """
    if cfg.label_type in ['model_answer', 'correct_answer']:
        output_dim = 4
    elif cfg.label_type == 'model_correct':
        output_dim = 1
    else:
        raise ValueError(f"Unknown label_type: {cfg.label_type}")
    
    model = PoolingProbe(
        input_dim=7168,
        output_dim=output_dim,
        probe_type=cfg.probe_type,
        rolling_window=cfg.rolling_window,
        softmax_temperature=cfg.softmax_temperature,
        dtype=torch.float32  # Use float32 for compatibility
    )
    model.to(cfg.device)
    return model

def train_probe(model: torch.nn.Module, train_dataset: ActivationDataset, test_dataset: ActivationDataset, cfg: TrainingConfig):
    """Train and evaluate the probe using DataLoaders."""
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    if cfg.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    # Define loss based on task
    if cfg.label_type in ['model_answer', 'correct_answer']:
        criterion = torch.nn.CrossEntropyLoss()
        is_binary = False
    elif cfg.label_type == 'model_correct':
        criterion = torch.nn.BCEWithLogitsLoss()
        is_binary = True
    best_test_acc = 0.0
    best_model = None

    # Prepare output dirs once
    run_output_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, 'models'), exist_ok=True)
    results_path = os.path.join(run_output_dir, f'{cfg.run_name}.jsonl')

    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=cfg.disable_tqdm):
            batch_inputs = batch_inputs.to(device=cfg.device)
            batch_labels = batch_labels.to(cfg.device)

            if is_binary:
                logits = model(batch_inputs).float()  # Shape: (batch_size, 1)
                loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                with torch.no_grad():
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                    train_correct += (preds == batch_labels.view(-1)).sum().item()
                    train_samples += batch_labels.numel()
            else:
                logits = model(batch_inputs)  # Shape: (batch_size, num_classes)
                loss = criterion(logits.float(), batch_labels.long())
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    train_correct += (preds == batch_labels).sum().item()
                    train_samples += batch_labels.size(0)

            train_loss_sum += loss.item() * (batch_labels.numel() if is_binary else batch_labels.size(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            test_loss_sum = 0.0
            test_correct = 0
            test_samples = 0
            
            for batch_inputs, batch_labels in tqdm(test_loader, desc=f"Testing Epoch {epoch}", disable=cfg.disable_tqdm):
                batch_inputs = batch_inputs.to(device=cfg.device)
                batch_labels = batch_labels.to(cfg.device)

                if is_binary:
                    logits = model(batch_inputs).float()  # Shape: (batch_size, 1)
                    loss = criterion(logits.view(-1), batch_labels.view(-1).float())
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                    test_correct += (preds == batch_labels.view(-1)).sum().item()
                    test_samples += batch_labels.numel()
                    test_loss_sum += loss.item() * batch_labels.numel()
                else:
                    logits = model(batch_inputs)  # Shape: (batch_size, num_classes)
                    loss = criterion(logits.float(), batch_labels.long())
                    preds = torch.argmax(logits, dim=1)
                    test_correct += (preds == batch_labels).sum().item()
                    test_samples += batch_labels.size(0)
                    test_loss_sum += loss.item() * batch_labels.size(0)

        avg_train_loss = train_loss_sum / max(1, train_samples)
        avg_test_loss = test_loss_sum / max(1, test_samples)
        test_acc = (test_correct / max(1, test_samples)) * 100.0
        if test_acc > best_test_acc:
            best_model = model.state_dict()
            best_test_acc = test_acc
        end_time = time.time()
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {end_time - start_time:.2f}s")

    # Append JSONL record
    cfg_dict = asdict(cfg)
    cfg_dict['dtype'] = str(cfg.dtype)
    log_record = {
        'run_name': cfg.run_name,
        'layer_idx': cfg.layer_idx,
        'best_test_acc': best_test_acc,
        'config': cfg_dict,
    }
    with open(results_path, 'a') as f:
        f.write(json.dumps(log_record) + "\n")

    return best_model, best_test_acc

def _parse_dtype_name(dtype_name: str) -> torch.dtype:
    """Map dtype string to corresponding torch.dtype."""
    name = dtype_name.lower()
    if name in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if name in ["float16", "fp16", "half"]:
        return torch.float16
    if name in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _build_config_from_params(params: dict) -> TrainingConfig:
    """Construct TrainingConfig from dict params with proper types."""
    return TrainingConfig(
        train_test_split=float(params.get('train_test_split', 0.8)),
        batch_size=int(params.get('batch_size', 512)),
        device=str(params.get('device', 'cuda')),
        label_type=str(params.get('label_type', 'model_answer')),
        learning_rate=float(params.get('learning_rate', 1e-3)),
        weight_decay=float(params.get('weight_decay', 0.0)),
        optimizer_type=str(params.get('optimizer_type', 'Adam')),
        dtype=_parse_dtype_name(str(params.get('dtype', 'bfloat16'))),
        num_epochs=int(params.get('num_epochs', 10)),
        output_dir=str(params.get('output_dir', 'results/')),
        run_name=str(params.get('run_name', 'test')),
        disable_tqdm=bool(params.get('disable_tqdm', False)),
        seed=int(params.get('seed', 42)),
        layer_idx=int(params.get('layer_idx', 0)),
        probe_type=str(params.get('probe_type', 'mean')),
        rolling_window=int(params.get('rolling_window', 5)),
        softmax_temperature=float(params.get('softmax_temperature', 1.0)),
    )


def train_all_layers(base_params: dict):
    """
    Train probes for all layers (0-60) using the same settings from base_params.
    Saves each model to output_dir/run_name/models/run_name_layer{layer}.pt
    """
    data_dir = str(base_params.get('data_dir', '/mnt/polished-lake/data/mmlu_activations'))
    base_cfg = _build_config_from_params(base_params)
    
    # Create layer-wise output directory using run_name structure
    layer_wise_dir = os.path.join(base_cfg.output_dir, base_cfg.run_name)
    os.makedirs(layer_wise_dir, exist_ok=True)
    os.makedirs(os.path.join(layer_wise_dir, 'models'), exist_ok=True)
    
    layer_results = []
    
    for layer_idx in range(61):  # 0-60 inclusive
        print(f"\n{'='*50}")
        print(f"Training probe for layer {layer_idx}")
        print(f"{'='*50}")
        
        # Create config for this layer - use same run_name for all layers
        layer_cfg = replace(base_cfg, run_name=base_cfg.run_name, layer_idx=layer_idx)
        
        # Load datasets for this layer
        train_dataset, test_dataset = get_datasets(layer_cfg, data_dir, layer_idx)
        print(f"Train len: {len(train_dataset)} | Test len: {len(test_dataset)}")
        
        if len(train_dataset) == 0:
            print(f"No data found for layer {layer_idx}, skipping...")
            continue
            
        # Create and train model
        model = create_model(layer_cfg)
        
        # For layer-wise training, write over the JSONL file at the end with the final results
        best_model, best_acc = train_probe(model, train_dataset, test_dataset, layer_cfg)
        
        # Save model with layer index in filename
        model_path = os.path.join(layer_wise_dir, 'models', f'{layer_cfg.run_name}_layer{layer_idx}.pt')
        torch.save(best_model, model_path)
        
        layer_results.append({
            'layer': layer_idx,
            'accuracy': best_acc,
            'model_path': model_path
        })
        
        print(f"Layer {layer_idx} completed with accuracy: {best_acc:.2f}%")
        del train_dataset, test_dataset, model
        import gc; gc.collect()
        torch.cuda.empty_cache()
            
    
    print(f"\nLayer-wise training completed. Results saved to {layer_wise_dir}")
    print(f"JSONL results appended to: {os.path.join(layer_wise_dir, f'{base_cfg.run_name}.jsonl')}")
    
    return layer_results

def run_sweep(base_params: dict, sweep_grid: dict):
    """
    Iterate over all combinations of sweep_grid's params and train once per combo.
    Loads datasets once using base params, then reuses them for all combos.
    """
    if not sweep_grid:
        print("No sweep grid provided.")
        return

    # Load datasets once from base params
    base_cfg = _build_config_from_params(base_params)
    data_dir = str(base_params.get('data_dir', '/mnt/polished-lake/data/mmlu_activations'))
    layer_idx = int(base_params.get('layer_idx', 0))
    train_dataset, test_dataset = get_datasets(base_cfg, data_dir, layer_idx)
    print(f"Train len: {len(train_dataset)} | Test len: {len(test_dataset)}")

    param_names = list(sweep_grid.keys())
    values_product = list(itertools.product(*[sweep_grid[name] for name in param_names]))
    print(f"Total combinations: {len(values_product)} | Params: {param_names}")

    best_acc = 0.0
    best_params = None
    for i, combination in enumerate(values_product):
        print(f"\n{'='*50}")
        print(f"Training probe for combination {i+1}/{len(values_product)}: {param_names} = {combination}")
        print(f"{'='*50}")
        
        merged_params = dict(base_params)
        for name, value in zip(param_names, combination):
            merged_params[name] = value

        cfg = _build_config_from_params(merged_params)
        model = create_model(cfg)
        model, test_acc = train_probe(model, train_dataset, test_dataset, cfg)
        
        # Save model for this combination
        acc_str = f"{test_acc:.2f}".replace(".", "_")
        combo_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, combination)])
        model_path = os.path.join(cfg.output_dir, cfg.run_name, 'models', f'{cfg.run_name}_{combo_str}_{acc_str}.pt')
        torch.save(model, model_path)
        print(f"Combination {i+1} completed with accuracy: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_params = cfg
    
    print(f"\nSweep completed. Best params: {best_params} | Best acc: {best_acc}")

def main():
    """Parse args, load YAML, run single training run, sweep, or layer-wise training."""
    parser = argparse.ArgumentParser(description="Train linear probe, run hyperparameter sweep, or train all layers.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config containing params and optional sweep grid.")
    parser.add_argument("--mode", type=str, choices=["single", "sweep", "all_layers"], default="single", help="Run mode: single, sweep, or all_layers.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    params = raw_cfg.get('params', {})
    data_dir = str(params.get('data_dir', '/mnt/polished-lake/data/mmlu_activations'))
    layer_idx = int(params.get('layer_idx', 0))

    cfg = _build_config_from_params(params)
    set_seed(cfg.seed)
    print(f"Starting {cfg.run_name} with mode {args.mode}")

    if args.mode == 'single':
        model = create_model(cfg)
        train_dataset, test_dataset = get_datasets(cfg, data_dir, layer_idx)
        print(f"Train len: {len(train_dataset)}")
        print(f"Test len: {len(test_dataset)}")
        best_model, best_acc = train_probe(model, train_dataset, test_dataset, cfg)
        acc_str = f"{best_acc:.2f}".replace(".", "_")
        torch.save(best_model, os.path.join(cfg.output_dir, cfg.run_name, 'models', f'{cfg.run_name}_best_model_{acc_str}.pt'))
    elif args.mode == 'all_layers':
        train_all_layers(params)
    else:
        sweep_grid = raw_cfg.get('sweep', {})
        run_sweep(params, sweep_grid)

if __name__ == "__main__":
    main()