#!/usr/bin/env python3
"""
Visualization script for pooling probe results.
Generates publication-ready plots similar to the paper.
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all results from JSONL files."""
    all_results = []
    
    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(results_dir, "**/*.jsonl"), recursive=True)
    
    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    all_results.append(result)
    
    if not all_results:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    data = []
    for result in all_results:
        config = result['config']
        data.append({
            'run_name': result['run_name'],
            'layer_idx': config.get('layer_idx', 0),
            'probe_type': config.get('probe_type', 'unknown'),
            'rolling_window': config.get('rolling_window', None),
            'softmax_temperature': config.get('softmax_temperature', None),
            'accuracy': result['best_test_acc'],
            'label_type': config.get('label_type', 'unknown'),
            'num_epochs': config.get('num_epochs', 0),
            'batch_size': config.get('batch_size', 0),
            'learning_rate': config.get('learning_rate', 0),
        })
    
    return pd.DataFrame(data)


def plot_probe_comparison(df: pd.DataFrame, save_path: str = "plots/probe_comparison.png"):
    """Plot accuracy comparison across different probe types."""
    if df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Get unique probe types and assign colors
    probe_types = df['probe_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(probe_types)))
    
    # Create box plot
    box_data = [df[df['probe_type'] == pt]['accuracy'].values for pt in probe_types]
    
    bp = plt.boxplot(box_data, labels=probe_types, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, probe_type in enumerate(probe_types):
        probe_data = df[df['probe_type'] == probe_type]['accuracy']
        x_pos = np.random.normal(i + 1, 0.04, size=len(probe_data))
        plt.scatter(x_pos, probe_data, alpha=0.6, s=50, color=colors[i])
    
    plt.xlabel('Probe Type')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Pooling Probe Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nProbe Type Summary:")
    summary = df.groupby('probe_type')['accuracy'].agg(['mean', 'std', 'count']).round(2)
    print(summary)


def plot_layer_wise_accuracy(df: pd.DataFrame, save_path: str = "plots/layer_wise_accuracy.png"):
    """Plot accuracy vs layer position."""
    if df.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot each probe type separately
    probe_types = df['probe_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(probe_types)))
    
    for i, probe_type in enumerate(probe_types):
        probe_data = df[df['probe_type'] == probe_type].sort_values('layer_idx')
        if len(probe_data) > 1:
            plt.plot(probe_data['layer_idx'], probe_data['accuracy'], 
                    marker='o', label=probe_type, color=colors[i], linewidth=2, markersize=6)
        else:
            plt.scatter(probe_data['layer_idx'], probe_data['accuracy'], 
                       label=probe_type, color=colors[i], s=100)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Layer Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hyperparameter_analysis(df: pd.DataFrame, save_path: str = "plots/hyperparameter_analysis.png"):
    """Plot hyperparameter sensitivity analysis."""
    if df.empty:
        print("No data to plot")
        return
    
    # Check if we have hyperparameter data
    rolling_data = df[df['rolling_window'].notna()]
    temp_data = df[df['softmax_temperature'].notna()]
    
    if rolling_data.empty and temp_data.empty:
        print("No hyperparameter data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Rolling window analysis
    if not rolling_data.empty:
        for probe_type in rolling_data['probe_type'].unique():
            probe_data = rolling_data[rolling_data['probe_type'] == probe_type].sort_values('rolling_window')
            axes[0].plot(probe_data['rolling_window'], probe_data['accuracy'], 
                        marker='o', label=probe_type, linewidth=2, markersize=6)
        
        axes[0].set_xlabel('Rolling Window Size')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Accuracy vs Rolling Window Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No rolling window data', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Rolling Window Analysis')
    
    # Temperature analysis
    if not temp_data.empty:
        for probe_type in temp_data['probe_type'].unique():
            probe_data = temp_data[temp_data['probe_type'] == probe_type].sort_values('softmax_temperature')
            axes[1].plot(probe_data['softmax_temperature'], probe_data['accuracy'], 
                        marker='o', label=probe_type, linewidth=2, markersize=6)
        
        axes[1].set_xlabel('Softmax Temperature')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Accuracy vs Softmax Temperature')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
    else:
        axes[1].text(0.5, 0.5, 'No temperature data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Temperature Analysis')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table(df: pd.DataFrame, save_path: str = "plots/results_summary.csv"):
    """Create and save summary table."""
    if df.empty:
        print("No data for summary")
        return
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sorted.to_csv(save_path, index=False)
    
    # Print summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Total experiments: {len(df)}")
    print(f"Best accuracy: {df['accuracy'].max():.2f}%")
    print(f"Probe types tested: {df['probe_type'].unique()}")
    print(f"\nTop 10 Results:")
    top_10 = df_sorted.head(10)[['probe_type', 'layer_idx', 'rolling_window', 'softmax_temperature', 'accuracy']]
    print(top_10.to_string(index=False))
    
    return df_sorted


def main():
    parser = argparse.ArgumentParser(description="Visualize pooling probe results")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default="plots", 
                       help="Output directory for plots")
    args = parser.parse_args()
    
    print("Loading results...")
    df = load_results(args.results_dir)
    
    if df.empty:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Loaded {len(df)} results")
    
    # Generate visualizations
    print("\nCreating probe comparison plot...")
    plot_probe_comparison(df, os.path.join(args.output_dir, "probe_comparison.png"))
    
    print("Creating layer-wise plot...")
    plot_layer_wise_accuracy(df, os.path.join(args.output_dir, "layer_wise_accuracy.png"))
    
    print("Creating hyperparameter analysis...")
    plot_hyperparameter_analysis(df, os.path.join(args.output_dir, "hyperparameter_analysis.png"))
    
    print("Creating summary table...")
    create_summary_table(df, os.path.join(args.output_dir, "results_summary.csv"))
    
    print(f"\nAll visualizations saved to {args.output_dir}/ directory!")


if __name__ == "__main__":
    main()
