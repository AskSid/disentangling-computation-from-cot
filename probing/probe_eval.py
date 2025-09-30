import argparse
import os
import yaml
import torch
from collections import defaultdict
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_probe import (
    ActivationDataset,
    get_question_data,
    get_datasets,
    _build_config_from_params,
    create_model,
    set_seed,
)

def get_test_dataset(cfg, data_dir: str, layer_idx: int):
    """
    Get the test dataset from the data directory.
    """
    print(f"Getting test dataset for layer {layer_idx}")
    _, test_dataset = get_datasets(cfg, data_dir, layer_idx)
    return test_dataset

def get_layerwise_models(models_dir: str, run_name: str) -> Dict[int, str]:
    """
    Get the layerwise models from the model path.
    Expects files named '{run_name}_layer{idx}.pt' in the directory.
    """
    layer_to_path: Dict[int, str] = {}
    if os.path.isfile(models_dir):
        models_dir = os.path.dirname(models_dir)
    for layer_idx in range(61):
        expected = os.path.join(models_dir, f"{run_name}_layer{layer_idx}.pt")
        if os.path.exists(expected):
            layer_to_path[layer_idx] = expected
    return layer_to_path

def plot_accuracy_by_category(accuracy_per_category: Dict[str, Tuple[int, int]], save_path: str, layer_idx: int = None):
    """
    Plot the accuracy by category.
    """
    # Compute accuracy per category and sort descending by accuracy
    cat_acc_pairs = []
    for c, (correct, total) in accuracy_per_category.items():
        acc = 100.0 * correct / max(1, total)
        cat_acc_pairs.append((c, acc))
    cat_acc_pairs.sort(key=lambda x: x[1], reverse=True)

    categories = [c for c, _ in cat_acc_pairs]
    accs = [acc for _, acc in cat_acc_pairs]
    plt.figure(figsize=(12, 6))
    plt.bar(categories, accs)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel('Accuracy (%)')
    if layer_idx is not None:
        plt.title(f'Accuracy by Question Category (Layer {layer_idx})')
    else:
        plt.title('Accuracy by Question Category')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_by_position(accuracy_by_position: Dict[int, Tuple[int, int]], save_path: str, layer_idx: int = None):
    """
    Plot the accuracy by position bin (averaged per bin) as a line plot.
    X-axis is labeled as percentage through the sequence.
    """
    # Sort bins numerically
    bins_sorted = sorted(accuracy_by_position.keys())

    # Compute bin-center percentages and accuracies
    # Handle either 0-based bins (0..19) or 1-based bins (1..20)
    bin_keys = list(bins_sorted)
    is_one_based = (0 not in bin_keys) and (1 in bin_keys)
    index_offset = 1 if is_one_based else 0

    xs_pct = []  # numeric x positions as percent-of-sequence
    labels = []  # human-readable tick labels like "0–5%"
    accs = []
    for b in bins_sorted:
        correct, total = accuracy_by_position[b]
        accs.append(100.0 * correct / max(1, total))
        # Each bin is 5% wide; label as range and position at bin center
        zero_based_index = b - index_offset
        start_pct = 5 * zero_based_index
        end_pct = 5 * (zero_based_index + 1)
        center_pct = (start_pct + end_pct) / 2.0
        xs_pct.append(center_pct)
        labels.append(f"{start_pct}\u2013{end_pct}%")

    plt.figure(figsize=(10, 5))
    plt.plot(xs_pct, accs, marker='o')
    plt.xlabel('Percent of sequence (%)')
    plt.ylabel('Accuracy (%)')
    if layer_idx is not None:
        plt.title(f'Accuracy by Position (Layer {layer_idx})')
    else:
        plt.title('Accuracy by Position')
    plt.grid(True, alpha=0.3)
    plt.xticks(xs_pct, labels, rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def eval_probe(model: torch.nn.Module, test_dataset: ActivationDataset, cfg, plots_dir: str, plot: bool = True, layer_idx: int = None):
    """
    Evaluate the probe on the test dataset.
    Returns the total test accuracy, accuracy by category, and accuracy by position.
    Plots accuracy by category and accuracy by position.
    """
    device = cfg.device
    is_binary = cfg.label_type == 'model_correct'

    model.eval()
    total_correct = 0
    total_count = 0
    acc_per_cat: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    acc_per_pos: Dict[int, List[int]] = defaultdict(lambda: [0, 0])

    with torch.no_grad():
        batch_size = 4096 # or use cfg.batch_size
        # iterate directly over underlying list to access metadata
        ds_list = test_dataset.ds if isinstance(test_dataset, ActivationDataset) else test_dataset
        for start in tqdm(range(0, len(ds_list), batch_size), desc="Evaluating probe", disable=cfg.disable_tqdm):
            batch = ds_list[start:start+batch_size]
            activations = torch.stack([item['activations'] for item in batch]).to(device=device, dtype=cfg.dtype)
            labels = [item['label'] for item in batch]
            if is_binary:
                labels_t = torch.tensor(labels, dtype=torch.float32, device=device).view(-1)
                logits = model(activations).float().view(-1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_tensor = (preds == labels_t).cpu()
            else:
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                logits = model(activations)
                preds = torch.argmax(logits, dim=1)
                correct_tensor = (preds == labels_t).cpu()

            correct_list = correct_tensor.tolist()
            for idx_local, item in enumerate(batch):
                is_correct = 1 if correct_list[idx_local] else 0
                total_correct += is_correct
                total_count += 1
                # category and position metadata
                category = item.get('category', 'unknown')
                # Use standardized position bin (required)
                pos_bin = int(item['pos_bin'])
                acc_per_cat[category][0] += is_correct
                acc_per_cat[category][1] += 1
                acc_per_pos[pos_bin][0] += is_correct
                acc_per_pos[pos_bin][1] += 1

    overall_acc = 100.0 * total_correct / max(1, total_count)

    if plot:
        os.makedirs(plots_dir, exist_ok=True)
        prefix = f"layer{layer_idx}_" if layer_idx is not None else ""
        plot_accuracy_by_category(
            {k: tuple(v) for k, v in acc_per_cat.items()},
            os.path.join(plots_dir, f"{prefix}accuracy_by_category.png"),
            layer_idx=layer_idx,
        )
        plot_accuracy_by_position(
            {k: tuple(v) for k, v in acc_per_pos.items()},
            os.path.join(plots_dir, f"{prefix}accuracy_by_position.png"),
            layer_idx=layer_idx,
        )

    return overall_acc, {k: tuple(v) for k, v in acc_per_cat.items()}, {k: tuple(v) for k, v in acc_per_pos.items()}

def eval_layerwise(layerwise_models: Dict[int, str], cfg, data_dir: str, individual_plots: bool = True, base_plots_dir: str = None):
    """
    Evaluate the layerwise models on the test dataset.
    Returns the total test accuracy, accuracy per category, and accuracy per position for each layer.
    Plots accuracy by layer.
    """
    results_by_layer: Dict[int, float] = {}
    per_layer_stats = {}
    acc_vs_layer_x = []
    acc_vs_layer_y = []

    base_plots_dir = base_plots_dir or os.path.join(cfg.output_dir, cfg.run_name, 'plots')
    os.makedirs(base_plots_dir, exist_ok=True)

    for layer_idx, model_path in sorted(layerwise_models.items()):
        print(f"Evaluating layer {layer_idx} using model at: {model_path}")
        # rebuild dataset for this layer
        test_dataset = get_test_dataset(cfg, data_dir, layer_idx)
        # build model and load state_dict
        model = create_model(cfg)
        state_dict = torch.load(model_path, map_location=cfg.device)
        # tolerate both state_dict and full model save
        if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
            model.load_state_dict(state_dict)
        else:
            model = state_dict  # fallback if full module saved
        model.to(cfg.device)

        layer_plots_dir = os.path.join(base_plots_dir, f'layer_{layer_idx}')
        acc, acc_cat, acc_pos = eval_probe(model, test_dataset, cfg, plots_dir=layer_plots_dir, plot=individual_plots, layer_idx=layer_idx)
        results_by_layer[layer_idx] = acc
        per_layer_stats[layer_idx] = {
            'overall_acc': acc,
            'accuracy_by_category': acc_cat,
            'accuracy_by_position': acc_pos,
        }
        acc_vs_layer_x.append(layer_idx)
        acc_vs_layer_y.append(acc)
        print(f"Layer {layer_idx} accuracy: {acc:.2f}%")

    # Plot accuracy vs layer
    plt.figure(figsize=(10, 5))
    plt.plot(acc_vs_layer_x, acc_vs_layer_y, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy (%)')
    plt.title('Probe Accuracy by Layer')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(base_plots_dir, 'accuracy_by_layer.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved 'accuracy by layer' plot to: {plot_path}")

    return results_by_layer, per_layer_stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained probes (single or layerwise)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for training")
    parser.add_argument("--mode", type=str, choices=["single", "layerwise"], required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file (single) or models dir (layerwise)")
    parser.add_argument("--individual_plots", action='store_true', help="For layerwise, save per-layer plots as well")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    params = raw_cfg.get('params', {})
    cfg = _build_config_from_params(params)
    set_seed(cfg.seed)

    data_dir = str(params.get('data_dir', '/mnt/polished-lake/data/mmlu_activations'))
    plots_base_dir = os.path.join(cfg.output_dir, cfg.run_name, 'plots')
    print(f"Starting in mode='{args.mode}'. Config='{args.config}'. Plots dir='{plots_base_dir}'")

    if args.mode == "single":
        layer_idx = int(params.get('layer_idx', 0))
        print(f"Single mode: evaluating layer {layer_idx} with model: {args.model_path}")
        # Rebuild test set for specified layer
        test_dataset = get_test_dataset(cfg, data_dir, layer_idx)
        # Build and load model
        model = create_model(cfg)
        state = torch.load(args.model_path, map_location=cfg.device)
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            model.load_state_dict(state)
        else:
            model = state
        model.to(cfg.device)
        # Evaluate
        overall_acc, _, _ = eval_probe(model, test_dataset, cfg, plots_dir=plots_base_dir, plot=True, layer_idx=layer_idx)
        print(f"Single mode: overall accuracy: {overall_acc:.2f}%")
        print(f"Saved plots to: {plots_base_dir}")
    else:
        # args.model_path should point to the models directory
        models_dir = args.model_path
        print(f"Layerwise mode: models directory: {models_dir}")
        layerwise_models = get_layerwise_models(models_dir, cfg.run_name)
        eval_layerwise(layerwise_models, cfg, data_dir, individual_plots=args.individual_plots, base_plots_dir=plots_base_dir)

if __name__ == "__main__":
    main()