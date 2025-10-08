import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import yaml
import time
from typing import Tuple

from probing.code.experiment_config import ExperimentConfig, _build_config_from_params
from probing.code.load_activation_data import create_activation_datasets, create_dummy_datasets
from probing.code.utils import set_seed


def create_model(cfg: ExperimentConfig) -> torch.nn.Module:
    input_dim = 7168 if cfg.r1_model_type == "full" else 4096
    input_dim *= cfg.last_n_tokens
    # input_dim *= cfg.last_n_tokens
    
    output_dims = {
        "absolute_pos_front": 1,
        "absolute_pos_back": 1,
        "relative_position": 1,
        "model_ans": 4,
        "correct_ans": 4,
        "model_correct": 1,
    }
    output_dim = output_dims[cfg.label_type]
    dtype = torch.bfloat16
    
    if cfg.probe_type == "linear":
        model = torch.nn.Linear(input_dim, output_dim, dtype=dtype)
    elif cfg.probe_type == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, cfg.mlp_hidden_dim, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.mlp_hidden_dim, output_dim, dtype=dtype)
        )
    
    model.to(cfg.device)
    return model
    
def setup_training(cfg: ExperimentConfig) -> Tuple[torch.nn.Module, DataLoader, DataLoader, torch.optim.Optimizer, torch.nn.Module]:
    model = create_model(cfg)
    
    train_dataset, test_dataset = create_activation_datasets(cfg)
    # train_dataset, test_dataset = create_dummy_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    
    if cfg.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    
    if cfg.label_type in ['model_ans', 'correct_ans']:
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.label_type == 'model_correct':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.label_type in ['absolute_pos_front', 'absolute_pos_back', 'relative_position']:
        criterion = torch.nn.MSELoss()

    return model, train_loader, test_loader, optimizer, criterion


def train_probe(cfg: ExperimentConfig):
    """Train and evaluate the probe using DataLoaders."""
    model, train_loader, test_loader, optimizer, loss_fn = setup_training(cfg)
    is_binary = cfg.label_type == 'model_correct'
    is_regression = cfg.label_type in ['absolute_pos_front', 'absolute_pos_back', 'relative_position']
    
    best_test_loss = float('inf')
    best_test_acc = None
    best_model = None

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'models'), exist_ok=True)
    results_path = os.path.join(cfg.output_dir, 'results.jsonl')

    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        train_loss_sum, train_correct, train_samples = 0.0, 0, 0
        model.train()
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=cfg.disable_tqdm):
            batch_inputs = batch_inputs.to(device=cfg.device, non_blocking=True)
            batch_labels = batch_labels.to(cfg.device, non_blocking=True)

            if is_binary:
                logits = model(batch_inputs).float()
                loss = loss_fn(logits.view(-1), batch_labels.view(-1).float())
                with torch.no_grad():
                    preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                    train_correct += (preds == batch_labels.view(-1)).sum().item()
                    train_samples += batch_labels.numel()
            elif is_regression:
                logits = model(batch_inputs).float()
                loss = loss_fn(logits.view(-1), batch_labels.view(-1).float())
                with torch.no_grad():
                    preds = logits.view(-1)
                    targets = batch_labels.view(-1).float()
                    train_correct += (torch.abs(preds - targets) <= 1.0).sum().item()
                    train_samples += batch_labels.numel()
            else:
                logits = model(batch_inputs)
                loss = loss_fn(logits.float(), batch_labels.long())
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    train_correct += (preds == batch_labels).sum().item()
                    train_samples += batch_labels.size(0)

            train_loss_sum += loss.item() * (batch_labels.numel() if (is_binary or is_regression) else batch_labels.size(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            test_loss_sum = 0.0
            test_correct = 0
            test_samples = 0
            
            for batch_inputs, batch_labels in tqdm(test_loader, desc=f"Testing", disable=cfg.disable_tqdm):
                batch_inputs = batch_inputs.to(device=cfg.device, non_blocking=True)
                batch_labels = batch_labels.to(cfg.device, non_blocking=True)

                if is_binary:
                    logits = model(batch_inputs).float()
                    loss = loss_fn(logits.view(-1), batch_labels.view(-1).float())
                    with torch.no_grad():
                        preds = (torch.sigmoid(logits.view(-1)) > 0.5).to(batch_labels.dtype)
                        test_correct += (preds == batch_labels.view(-1)).sum().item()
                        test_samples += batch_labels.numel()
                elif is_regression:
                    logits = model(batch_inputs).float()
                    loss = loss_fn(logits.view(-1), batch_labels.view(-1).float())
                    with torch.no_grad():
                        preds = logits.view(-1)
                        targets = batch_labels.view(-1).float()
                        test_correct += (torch.abs(preds - targets) <= 1.0).sum().item()
                        test_samples += batch_labels.numel()
                else:
                    logits = model(batch_inputs)
                    loss = loss_fn(logits.float(), batch_labels.long())
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        test_correct += (preds == batch_labels).sum().item()
                        test_samples += batch_labels.size(0)

                test_loss_sum += loss.item() * (batch_labels.numel() if (is_binary or is_regression) else batch_labels.size(0))

        avg_train_loss = train_loss_sum / max(1, train_samples)
        avg_test_loss = test_loss_sum / max(1, test_samples)
        test_acc = (test_correct / max(1, test_samples)) * 100.0
        
        # Log this epoch's results
        epoch_data = {
            'run_name': cfg.run_name,
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_acc': test_acc,
            'timestamp': time.time()
        }
        with open(results_path, 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model = model.state_dict()
            best_test_acc = test_acc
        end_time = time.time()
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {end_time - start_time:.2f}s")
    
    # save best results + model
    with open(results_path, 'a') as f:
        f.write(json.dumps({'run_name': cfg.run_name, 'epoch': 'best', 'test_loss': best_test_loss, 'test_acc': best_test_acc, 'timestamp': time.time()}) + '\n')
    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'models', f'{cfg.run_name}_best_model.pth'))

def main():
    parser = argparse.ArgumentParser(description="Train linear probe, run hyperparameter sweep, or train all layers.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config containing params and optional sweep grid.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        params = yaml.safe_load(f) or {}

    cfg = _build_config_from_params(params)
    set_seed(cfg.seed)
    print(f"Starting {cfg.run_name}")

    train_probe(cfg)

if __name__ == "__main__":
    main()
