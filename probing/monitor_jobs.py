#!/usr/bin/env python3
"""
Monitor SLURM jobs and automatically generate visualizations when they complete.
"""

import subprocess
import time
import os
import json
from pathlib import Path

def get_user_jobs():
    """Get list of running jobs for current user."""
    try:
        result = subprocess.run(['squeue', '-u', os.environ.get('USER', '')], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            jobs = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 7:
                        jobs.append({
                            'jobid': parts[0],
                            'partition': parts[1],
                            'name': parts[2],
                            'user': parts[3],
                            'status': parts[4],
                            'time': parts[5],
                            'nodes': parts[6]
                        })
            return jobs
        else:
            print(f"Error getting job list: {result.stderr}")
            return []
    except Exception as e:
        print(f"Error running squeue: {e}")
        return []

def check_job_completion(jobid):
    """Check if a specific job has completed."""
    try:
        result = subprocess.run(['sacct', '-j', jobid, '--format=JobID,State,ExitCode'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if jobid in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        state = parts[1]
                        exit_code = parts[2]
                        return state, exit_code
        return None, None
    except Exception as e:
        print(f"Error checking job {jobid}: {e}")
        return None, None

def count_results():
    """Count number of result files."""
    results_dir = Path("results")
    if not results_dir.exists():
        return 0
    
    jsonl_files = list(results_dir.glob("**/*.jsonl"))
    return len(jsonl_files)

def run_visualizations():
    """Run the visualization script."""
    try:
        print("Running visualizations...")
        result = subprocess.run(['python', 'visualize_results.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Visualizations completed successfully!")
            print(result.stdout)
        else:
            print(f"Error running visualizations: {result.stderr}")
    except Exception as e:
        print(f"Error running visualization script: {e}")

def main():
    """Main monitoring loop."""
    print("Starting job monitoring...")
    print("Press Ctrl+C to stop")
    
    last_result_count = 0
    monitored_jobs = set()
    
    try:
        while True:
            # Get current jobs
            jobs = get_user_jobs()
            current_jobids = {job['jobid'] for job in jobs}
            
            # Check for new jobs
            new_jobs = current_jobids - monitored_jobs
            if new_jobs:
                print(f"\nNew jobs detected: {new_jobs}")
                monitored_jobs.update(new_jobs)
            
            # Check for completed jobs
            completed_jobs = monitored_jobs - current_jobids
            if completed_jobs:
                print(f"\nCompleted jobs: {completed_jobs}")
                monitored_jobs -= completed_jobs
                
                # Check if we have new results
                current_result_count = count_results()
                if current_result_count > last_result_count:
                    print(f"New results detected ({current_result_count} files)")
                    run_visualizations()
                    last_result_count = current_result_count
            
            # Print current status
            if jobs:
                print(f"\nCurrent jobs: {len(jobs)}")
                for job in jobs:
                    print(f"  {job['jobid']}: {job['name']} ({job['status']})")
            else:
                print("\nNo running jobs")
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error in monitoring loop: {e}")

if __name__ == "__main__":
    main()
