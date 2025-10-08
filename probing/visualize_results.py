#!/usr/bin/env python3
"""
Visualization script for pooling probe results.
Generates graphs similar to those in the paper.
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results_from_jsonl(filepath):
    """Load results from JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def load_all_results(results_dir="results"):
    """Load all results from the results directory."""
    all_results = []
    
    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(results_dir, "**/*.jsonl"), recursive=True)
    
    for filepath in jsonl_files:
        results = load_results_from_jsonl(filepath)
        all_results.extend(results)
    
    return all_results

def create_probe_comparison_plot(results, save_path="probe_comparison.png"):
    """Create a plot comparing different probe types."""
    # Filter results for single layer experiments
    single_layer_results = [r for r in results if 'layer_idx' in r['config']]
    
    if not single_layer_results:
        print("No single layer results found")
        return
    
    # Create DataFrame
    data = []
    for result in single_layer_results:
        config = result['config']
        data.append({
            'probe_type': config.get('probe_type', 'unknown'),
            'accuracy': result['best_test_acc'],
            'layer': config.get('layer_idx', 0),
            'rolling_window': config.get('rolling_window', None),
            'temperature': config.get('softmax_temperature', None),
            'run_name': result['run_name']
        })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy by probe type
    probe_types = df['probe_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(probe_types)))
    
    for i, probe_type in enumerate(probe_types):
        probe_data = df[df['probe_type'] == probe_type]
        plt.scatter([i] * len(probe_data), probe_data['accuracy'], 
                   alpha=0.7, label=probe_type, color=colors[i], s=100)
    
    plt.xlabel('Probe Type')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Probe Performance Comparison')
    plt.xticks(range(len(probe_types)), probe_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_layer_wise_plot(results, save_path="layer_wise_accuracy.png"):
    """Create a plot showing accuracy vs layer position."""
    # Filter results for layer-wise experiments
    layer_results = [r for r in results if 'layer_idx' in r['config']]
    
    if not layer_results:
        print("No layer-wise results found")
        return
    
    # Create DataFrame
    data = []
    for result in layer_results:
        config = result['config']
        data.append({
            'layer': config.get('layer_idx', 0),
            'accuracy': result['best_test_acc'],
            'probe_type': config.get('probe_type', 'unknown'),
            'run_name': result['run_name']
        })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot accuracy vs layer for each probe type
    probe_types = df['probe_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(probe_types)))
    
    for i, probe_type in enumerate(probe_types):
        probe_data = df[df['probe_type'] == probe_type].sort_values('layer')
        if len(probe_data) > 1:  # Only plot if we have multiple layers
            plt.plot(probe_data['layer'], probe_data['accuracy'], 
                    marker='o', label=probe_type, color=colors[i], linewidth=2, markersize=6)
        else:
            # For single layer results, just scatter
            plt.scatter(probe_data['layer'], probe_data['accuracy'], 
                       label=probe_type, color=colors[i], s=100)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Layer Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_hyperparameter_analysis(results, save_path="hyperparameter_analysis.png"):
    """Create plots for hyperparameter analysis (rolling window, temperature)."""
    # Filter results with hyperparameters
    param_results = [r for r in results if 'rolling_window' in r['config'] or 'softmax_temperature' in r['config']]
    
    if not param_results:
        print("No hyperparameter results found")
        return
    
    # Create DataFrame
    data = []
    for result in param_results:
        config = result['config']
        data.append({
            'probe_type': config.get('probe_type', 'unknown'),
            'accuracy': result['best_test_acc'],
            'rolling_window': config.get('rolling_window', None),
            'temperature': config.get('softmax_temperature', None),
            'run_name': result['run_name']
        })
    
    df = pd.DataFrame(data)
    
    # Create subplots for different hyperparameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Rolling window analysis
    rolling_data = df[df['rolling_window'].notna()]
    if not rolling_data.empty:
        for probe_type in rolling_data['probe_type'].unique():
            probe_rolling = rolling_data[rolling_data['probe_type'] == probe_type]
            ax1.plot(probe_rolling['rolling_window'], probe_rolling['accuracy'], 
                    marker='o', label=probe_type, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Rolling Window Size')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Accuracy vs Rolling Window Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Temperature analysis
    temp_data = df[df['temperature'].notna()]
    if not temp_data.empty:
        for probe_type in temp_data['probe_type'].unique():
            probe_temp = temp_data[temp_data['probe_type'] == probe_type]
            ax2.plot(probe_temp['temperature'], probe_temp['accuracy'], 
                    marker='o', label=probe_type, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Softmax Temperature')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Accuracy vs Softmax Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_summary_table(results, save_path="results_summary.csv"):
    """Create a summary table of all results."""
    if not results:
        print("No results found")
        return
    
    # Create DataFrame
    data = []
    for result in results:
        config = result['config']
        data.append({
            'run_name': result['run_name'],
            'layer_idx': config.get('layer_idx', 'unknown'),
            'probe_type': config.get('probe_type', 'unknown'),
            'rolling_window': config.get('rolling_window', None),
            'softmax_temperature': config.get('softmax_temperature', None),
            'best_accuracy': result['best_test_acc'],
            'label_type': config.get('label_type', 'unknown'),
            'num_epochs': config.get('num_epochs', 'unknown'),
            'batch_size': config.get('batch_size', 'unknown'),
            'learning_rate': config.get('learning_rate', 'unknown')
        })
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    df = df.sort_values('best_accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    # Print summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total experiments: {len(df)}")
    print(f"Best accuracy: {df['best_accuracy'].max():.2f}%")
    print(f"Probe types tested: {df['probe_type'].unique()}")
    print(f"\nTop 10 Results:")
    print(df.head(10)[['probe_type', 'layer_idx', 'rolling_window', 'softmax_temperature', 'best_accuracy']].to_string(index=False))
    
    return df

def main():
    """Main function to generate all visualizations."""
    print("Loading results...")
    results = load_all_results()
    
    if not results:
        print("No results found in results/ directory")
        return
    
    print(f"Loaded {len(results)} results")
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate visualizations
    print("\nCreating probe comparison plot...")
    probe_df = create_probe_comparison_plot(results, "plots/probe_comparison.png")
    
    print("Creating layer-wise plot...")
    layer_df = create_layer_wise_plot(results, "plots/layer_wise_accuracy.png")
    
    print("Creating hyperparameter analysis...")
    param_df = create_hyperparameter_analysis(results, "plots/hyperparameter_analysis.png")
    
    print("Creating summary table...")
    summary_df = create_summary_table(results, "plots/results_summary.csv")
    
    print("\nAll visualizations saved to plots/ directory!")

if __name__ == "__main__":
    main()
