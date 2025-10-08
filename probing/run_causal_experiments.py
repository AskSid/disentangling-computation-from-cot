#!/usr/bin/env python3
"""
Run causal probe experiments on layer 60.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_causal_experiment():
    """Run the causal probe sweep experiment."""
    script_path = Path(__file__).parent / "code" / "train_pooling_probe.py"
    config_path = Path(__file__).parent / "configs" / "causal_probe_layer60.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    cmd = [
        "python", str(script_path),
        "--config", str(config_path),
        "--mode", "sweep"
    ]
    
    print(f"Running causal probe experiment: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def plot_results():
    """Generate plots from the results."""
    script_path = Path(__file__).parent / "code" / "plot_causal_results.py"
    
    cmd = ["python", str(script_path)]
    print(f"Generating plots: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_causal_experiments.py <command>")
        print("Commands:")
        print("  train  - Run causal probe training")
        print("  plot   - Generate plots from results")
        print("  all    - Run training then plotting")
        return
    
    command = sys.argv[1]
    
    if command == "train":
        success = run_causal_experiment()
        if success:
            print("Causal probe training completed successfully!")
        else:
            print("Causal probe training failed!")
    
    elif command == "plot":
        success = plot_results()
        if success:
            print("Plotting completed successfully!")
        else:
            print("Plotting failed!")
    
    elif command == "all":
        print("Running full causal probe experiment...")
        train_success = run_causal_experiment()
        if train_success:
            print("Training completed, now generating plots...")
            plot_success = plot_results()
            if plot_success:
                print("Full experiment completed successfully!")
            else:
                print("Training succeeded but plotting failed!")
        else:
            print("Training failed, skipping plotting!")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
