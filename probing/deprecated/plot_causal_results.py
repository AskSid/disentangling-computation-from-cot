#!/usr/bin/env python3
"""
Plot causal probe results per position, similar to plot.ipynb
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List

def load_causal_results(results_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """Load causal probe results and organize by probe type."""
    results_by_probe = {}
    
    # Find all JSONL files
    jsonl_files = list(Path(results_dir).glob("**/*.jsonl"))
    
    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    config = result['config']
                    
                    # Check if this is a causal probe result
                    if config.get('use_position_based', False) and 'bin_idx' in config:
                        probe_type = config.get('probe_type', 'unknown')
                        bin_idx = config.get('bin_idx', 0)
                        accuracy = result['best_test_acc']
                        
                        if probe_type not in results_by_probe:
                            results_by_probe[probe_type] = []
                        
                        results_by_probe[probe_type].append({
                            'bin_idx': bin_idx,
                            'accuracy': accuracy,
                            'run_name': result['run_name']
                        })
    
    # Convert to DataFrames and sort by bin_idx
    for probe_type in results_by_probe:
        df = pd.DataFrame(results_by_probe[probe_type])
        df = df.sort_values('bin_idx')
        results_by_probe[probe_type] = df
    
    return results_by_probe

def plot_causal_accuracy(results_by_probe: Dict[str, pd.DataFrame], save_path: str = "plots/causal_probe_accuracy.png"):
    """Plot accuracy vs position bin for each probe type."""
    
    if not results_by_probe:
        print("No causal probe results found")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Define colors for each probe type
    probe_colors = {
        'mean': 'blue',
        'max': 'red', 
        'rolling_means': 'green',
        'softmax': 'orange',
        'attention': 'purple'
    }
    
    # Plot each probe type
    for probe_type, df in results_by_probe.items():
        if df.empty:
            continue
            
        color = probe_colors.get(probe_type, 'gray')
        plt.plot(df['bin_idx'], df['accuracy'], 
                marker='o', label=f'{probe_type}', 
                color=color, linewidth=2, markersize=4)
    
    # Add baseline line (random chance for binary classification)
    plt.axhline(y=50.0, color='black', linestyle='--', linewidth=1, 
                alpha=0.7, label='random baseline (50%)')
    
    plt.xlabel('Position Bin Index (1% of sequence length)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Causal Probe Accuracy vs Position (Layer 60)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis to show percentage of sequence
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.linspace(0, 100, 11))
    ax2.set_xticklabels([f'{int(x)}%' for x in np.linspace(0, 100, 11)])
    ax2.set_xlabel('Position in Sequence (%)')
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== CAUSAL PROBE RESULTS SUMMARY ===")
    for probe_type, df in results_by_probe.items():
        if not df.empty:
            best_acc = df['accuracy'].max()
            best_bin = df.loc[df['accuracy'].idxmax(), 'bin_idx']
            mean_acc = df['accuracy'].mean()
            print(f"{probe_type}: Best {best_acc:.2f}% at bin {best_bin}, Mean {mean_acc:.2f}%")

def plot_probe_comparison(results_by_probe: Dict[str, pd.DataFrame], save_path: str = "plots/probe_comparison_causal.png"):
    """Plot comparison of probe types at different positions."""
    
    if not results_by_probe:
        print("No causal probe results found")
        return
    
    # Find common bin indices
    all_bins = set()
    for df in results_by_probe.values():
        all_bins.update(df['bin_idx'].tolist())
    
    common_bins = sorted(list(all_bins))
    
    if len(common_bins) < 2:
        print("Not enough common bins for comparison")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Early positions (0-20%)
    early_bins = [b for b in common_bins if b <= 20]
    for probe_type, df in results_by_probe.items():
        if df.empty:
            continue
        early_data = df[df['bin_idx'].isin(early_bins)]
        if not early_data.empty:
            ax1.plot(early_data['bin_idx'], early_data['accuracy'], 
                    marker='o', label=probe_type, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Position Bin Index')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Early Positions (0-20%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Late positions (80-100%)
    late_bins = [b for b in common_bins if b >= 80]
    for probe_type, df in results_by_probe.items():
        if df.empty:
            continue
        late_data = df[df['bin_idx'].isin(late_bins)]
        if not late_data.empty:
            ax2.plot(late_data['bin_idx'], late_data['accuracy'], 
                    marker='o', label=probe_type, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Position Bin Index')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Late Positions (80-100%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results_csv(results_by_probe: Dict[str, pd.DataFrame], save_path: str = "plots/causal_results.csv"):
    """Save results to CSV for further analysis."""
    
    all_results = []
    for probe_type, df in results_by_probe.items():
        for _, row in df.iterrows():
            all_results.append({
                'probe_type': probe_type,
                'bin_idx': row['bin_idx'],
                'accuracy': row['accuracy'],
                'run_name': row['run_name']
            })
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(['probe_type', 'bin_idx'])
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")
        
        # Print top results
        print("\n=== TOP 10 RESULTS ===")
        top_results = results_df.nlargest(10, 'accuracy')
        print(top_results[['probe_type', 'bin_idx', 'accuracy']].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Plot causal probe results")
    parser.add_argument("--results_dir", type=str, default="results", 
                       help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default="plots", 
                       help="Output directory for plots")
    args = parser.parse_args()
    
    print("Loading causal probe results...")
    results_by_probe = load_causal_results(args.results_dir)
    
    if not results_by_probe:
        print(f"No causal probe results found in {args.results_dir}")
        return
    
    print(f"Found results for probe types: {list(results_by_probe.keys())}")
    
    # Generate plots
    print("\nGenerating causal accuracy plot...")
    plot_causal_accuracy(results_by_probe, 
                        os.path.join(args.output_dir, "causal_probe_accuracy.png"))
    
    print("Generating probe comparison plot...")
    plot_probe_comparison(results_by_probe, 
                         os.path.join(args.output_dir, "probe_comparison_causal.png"))
    
    print("Saving results to CSV...")
    save_results_csv(results_by_probe, 
                    os.path.join(args.output_dir, "causal_results.csv"))
    
    print(f"\nAll plots saved to {args.output_dir}/ directory!")

if __name__ == "__main__":
    import os
    main()
