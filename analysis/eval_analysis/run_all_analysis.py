#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import time
from pathlib import Path

def run_analysis(run_dir: str, output_dir: str, add_problem_details: bool = False, problems_per_task: int = 45):
    """
    Run all analysis scripts on the specified run directory.
    
    Args:
        run_dir: Directory containing evaluation output files
        output_dir: Directory to save analysis results
        add_problem_details: Whether to add problem details to heatmap output
        problems_per_task: Number of problems per task type (default: 45)
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the directory where the analysis scripts are located
    script_dir = Path(__file__).parent
    
    # Command 1: Run visualize_eval_progress.py
    print(f"Running visualize_eval_progress.py on {run_dir}...")
    cmd1 = [
        "python", 
        str(script_dir / "visualize_eval_progress.py"),
        run_dir,
        "--output-dir", output_dir,
        "--problems-per-task", str(problems_per_task)
    ]
    subprocess.run(cmd1, check=True)
    
    # Command 2: Run problem_difficulty_analysis.py
    print(f"\nRunning problem_difficulty_analysis.py on {run_dir}...")
    cmd2 = [
        "python", 
        str(script_dir / "problem_difficulty_analysis.py"),
        run_dir,
        "--output-dir", output_dir,
        "--problems-per-task", str(problems_per_task)
    ]
    subprocess.run(cmd2, check=True)
    
    # Command 3: Run problem_correctness_heatmap.py
    print(f"\nRunning problem_correctness_heatmap.py on {run_dir}...")
    cmd3 = [
        "python", 
        str(script_dir / "problem_correctness_heatmap.py"),
        run_dir,
        "--output-dir", output_dir,
        "--problems-per-task", str(problems_per_task)
    ]
    
    if add_problem_details:
        cmd3.append("--add-problem-details")
        
    subprocess.run(cmd3, check=True)
    
    elapsed_time = time.time() - start_time
    print(f"\nAll analysis completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run all evaluation analysis scripts')
    parser.add_argument('run_dir', help='Directory containing evaluation output files')
    parser.add_argument('--output-dir', default=None, 
                      help='Directory to save analysis results (default: analysis/eval_analysis/output/RUN_DIR_NAME)')
    parser.add_argument('--add-problem-details', action='store_true',
                      help='Add problem details to heatmap output')
    parser.add_argument('--problems-per-task', type=int, default=45,
                      help='Number of problems per task type (default: 45)')
    
    args = parser.parse_args()
    
    # If output_dir is not specified, use a default based on run_dir name
    if args.output_dir is None:
        run_name = os.path.basename(os.path.normpath(args.run_dir))
        args.output_dir = os.path.join("analysis/eval_analysis/output", run_name)
    
    run_analysis(args.run_dir, args.output_dir, args.add_problem_details, args.problems_per_task)

if __name__ == "__main__":
    main() 