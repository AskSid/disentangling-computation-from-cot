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

def set_seed(seed: int):
    """Set random seeds for reproducibility across python, numpy, and torch."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_question_data(data_dir: str, section: str, idx: int, question_name: str, layer_idx: int, verbose: bool = False, disable_tqdm: bool = False) -> list:
    """
    Load activations and labels for a single MMLU question.

    Returns a list of token-level records with 'activations', 'seq_idx',
    'model_answer', and 'correct_answer', or None if parsing fails.
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
    seq_len = activations.shape[0]
    bins = np.linspace(0, seq_len, 21)
    data = []
    for i in range(seq_len):
        seq_bin = np.digitize(i, bins)
        data.append({'activations': activations[i], 'pos_bin': seq_bin, 'model_answer': model_answer, 'correct_answer': correct_answer, 'category': section})
    return data

class ActivationDataset(Dataset):
    """Torch dataset for pairing token-level activations with relevant label type."""
    def __init__(self, ds: list, include_metadata: bool = False):
        self.ds = ds
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        activation = self.ds[idx]['activations']
        label = torch.tensor(self.ds[idx]['label'], dtype=torch.long if isinstance(self.ds[idx]['label'], int) else torch.float32)
        if self.include_metadata:
            pos_bin = torch.tensor(self.ds[idx]['pos_bin'], dtype=torch.long)
            category = torch.tensor(self.ds[idx]['category'], dtype=torch.long)
            return activation, label, pos_bin, category
        return activation, label

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
                activation_data.extend(question_data)
    
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
    Create the linear probe head. 
    'model_answer' and 'correct_answer' are one of the four options, and 'model_correct' is a yes/no label so a single probability.
    """
    if cfg.label_type == 'model_answer':
        model = torch.nn.Linear(7168, 4, dtype=cfg.dtype)
    elif cfg.label_type == 'correct_answer':
        model = torch.nn.Linear(7168, 4, dtype=cfg.dtype)
    elif cfg.label_type == 'model_correct':
        model = torch.nn.Linear(7168, 1, dtype=cfg.dtype)
    model.to(cfg.device)
    return model

def train_probe(model: torch.nn.Module, train_dataset: ActivationDataset, test_dataset: ActivationDataset, cfg: TrainingConfig):
    """Train and evaluate the probe using DataLoaders."""
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

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
            batch_inputs = batch_inputs.to(device=cfg.device, dtype=cfg.dtype)
            batch_labels = batch_labels.to(cfg.device)

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
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            test_loss_sum = 0.0
            test_correct = 0
            test_samples = 0
            
            for batch_inputs, batch_labels in tqdm(test_loader, desc=f"Testing Epoch {epoch}", disable=cfg.disable_tqdm):
                batch_inputs = batch_inputs.to(device=cfg.device, dtype=cfg.dtype)
                batch_labels = batch_labels.to(cfg.device)

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