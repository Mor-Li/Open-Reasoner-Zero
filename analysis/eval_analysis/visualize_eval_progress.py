#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def parse_eval_filename(filename: str) -> Tuple[int, Dict[str, float]]:
    """
    Parse the evaluation output filename to extract the iteration and scores.
    
    Example filename format:
    eval_output_iter224_ea_zh0.2667_ea_en0.2444_na_zh0.2667_na_en0.2889_nd_zh0.3111_nd_en0.2889_rd_zh0.1556_rd_en0.1333.jsonl
    
    Returns:
        Tuple containing iteration number and a dictionary of scores
    """
    # Extract iteration number
    iter_match = re.search(r'iter(\d+)', filename)
    if not iter_match:
        return None, {}
    
    iteration = int(iter_match.group(1))
    
    # Extract metrics
    metrics = {}
    pattern = r'([a-z]+)_([a-z]+)([\d\.]+)'
    matches = re.findall(pattern, filename)
    
    for task_type, lang, score in matches:
        key = f"{task_type}_{lang}"
        # Remove any trailing dots from the score before converting to float
        clean_score = score.rstrip('.')
        try:
            metrics[key] = float(clean_score)
        except ValueError:
            print(f"Warning: Could not convert '{score}' to float in filename: {filename}")
            metrics[key] = 0.0
    
    return iteration, metrics

def get_task_types_from_filename(filename: str) -> List[str]:
    """
    Extract the task types from the filename in the correct order.
    
    Example filename format:
    eval_output_iter224_ea_zh0.2667_ea_en0.2444_na_zh0.2667_na_en0.2889_nd_zh0.3111_nd_en0.2889_rd_zh0.1556_rd_en0.1333.jsonl
    
    Returns:
        List of task types in the order they appear in the filename
    """
    pattern = r'([a-z]+)_([a-z]+)([\d\.]+)'
    matches = re.findall(pattern, filename)
    task_types = []
    
    valid_prefixes = ["ea", "na", "nd", "rd"]
    valid_languages = ["zh", "en"]
    
    for task_prefix, lang, _ in matches:
        # Only add valid task type combinations
        if task_prefix in valid_prefixes and lang in valid_languages:
            task_type = f"{task_prefix}_{lang}"
            task_types.append(task_type)
    
    return task_types

def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read jsonl file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def create_correctness_heatmap(data_dict: Dict[int, Dict[str, List[bool]]], output_dir: Path):
    """
    Create a heatmap showing correctness of each problem across iterations.
    
    Args:
        data_dict: Dictionary mapping iteration to dictionary of problem correctness
        output_dir: Directory to save the output figures
    """
    # Sort iterations
    iterations = sorted(data_dict.keys())
    
    # Get all task types
    all_task_types = set()
    for iter_data in data_dict.values():
        all_task_types.update(iter_data.keys())
    
    # Create figure for each task type
    for task_type in all_task_types:
        # Check if we have data for this task type across iterations
        n_problems = None
        for iter_num in iterations:
            if task_type in data_dict[iter_num]:
                if n_problems is None:
                    n_problems = len(data_dict[iter_num][task_type])
                elif len(data_dict[iter_num][task_type]) != n_problems:
                    print(f"Warning: Inconsistent number of problems for {task_type} across iterations")
        
        if n_problems is None:
            continue  # Skip if no data
            
        # Create a matrix for heatmap
        n_iters = len(iterations)
        correctness_matrix = np.zeros((n_problems, n_iters))
        
        for i, iter_num in enumerate(iterations):
            if task_type in data_dict[iter_num]:
                for j in range(min(n_problems, len(data_dict[iter_num][task_type]))):
                    correctness_matrix[j, i] = 1 if data_dict[iter_num][task_type][j] else 0
        
        # Plot heatmap
        plt.figure(figsize=(12, max(8, n_problems // 4)))
        sns.heatmap(
            correctness_matrix, 
            cmap=['#ffcccc', '#99ff99'],  # Red for wrong, green for correct
            cbar=False,
            xticklabels=[f"{iter_num}" for iter_num in iterations],
            yticklabels=range(1, n_problems + 1)
        )
        plt.xlabel('Iteration')
        plt.ylabel('Problem Number')
        plt.title(f'Problem Correctness Evolution - {task_type}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"heatmap_{task_type}.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved heatmap to {output_path}")

def create_accuracy_line_chart(iter_metrics: Dict[int, Dict[str, float]], output_dir: Path):
    """
    Create a line chart showing accuracy trends over iterations.
    
    Args:
        iter_metrics: Dictionary mapping iteration to metrics
        output_dir: Directory to save the output figures
    """
    iterations = sorted(iter_metrics.keys())
    metrics = defaultdict(list)
    
    for iter_num in iterations:
        for metric_name, value in iter_metrics[iter_num].items():
            metrics[metric_name].append(value)
    
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(iterations, values, marker='o', label=metric_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trends Across Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "accuracy_trends.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved accuracy trends to {output_path}")

def sort_task_types(task_types):
    """Sort task types in a logical order."""
    # Define order: ea -> na -> nd -> rd, then zh -> en
    order_prefix = {"ea": 0, "na": 1, "nd": 2, "rd": 3}
    order_suffix = {"zh": 0, "en": 1}
    
    def sort_key(task):
        prefix, suffix = task.split('_')
        return (order_prefix.get(prefix, 999), order_suffix.get(suffix, 999))
    
    return sorted(task_types, key=sort_key)

def analyze_eval_outputs(run_dir: str, output_dir: str, problems_per_task: int = 45):
    """
    Analyze evaluation outputs from the specified run directory.
    
    Args:
        run_dir: Directory containing evaluation output files
        output_dir: Directory to save analysis results
        problems_per_task: Number of problems per task type (default: 45)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all evaluation output files
    eval_files = []
    for file in os.listdir(run_dir):
        if file.startswith("eval_output_iter") and file.endswith(".jsonl"):
            eval_files.append(file)
    
    if not eval_files:
        print(f"No evaluation output files found in {run_dir}")
        return
    
    # Parse filenames and sort by iteration
    iter_data = {}
    iter_metrics = {}
    
    for file in eval_files:
        try:
            iteration, metrics = parse_eval_filename(file)
            if iteration is None:
                continue
                
            # Get expected task types from filename in order
            expected_task_types = get_task_types_from_filename(file)
            
            if not expected_task_types:
                print(f"Warning: Could not extract task types from filename: {file}")
                continue
                
            # Read the evaluation results
            filepath = os.path.join(run_dir, file)
            results = read_jsonl(filepath)
            
            # Group correctness by task type based on index in the results
            correctness_by_type = defaultdict(list)
            
            for i, item in enumerate(results):
                # Determine task type from position in the file
                # Assuming results are ordered by task type as they appear in the filename
                task_idx = i // problems_per_task
                if task_idx < len(expected_task_types):
                    task_type = expected_task_types[task_idx]
                    is_correct = item.get("iscorrect", False)
                    correctness_by_type[task_type].append(is_correct)
            
            iter_data[iteration] = dict(correctness_by_type)
            iter_metrics[iteration] = metrics
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if not iter_data:
        print("No valid data could be processed. Exiting.")
        return
        
    # Create visualizations
    create_correctness_heatmap(iter_data, output_dir)
    create_accuracy_line_chart(iter_metrics, output_dir)
    
    # Generate summary statistics
    task_types = sort_task_types(set().union(*[d.keys() for d in iter_data.values()]))
    iterations = sorted(iter_data.keys())
    
    # Create a DataFrame for iteration progress
    progress_data = []
    for iter_num in iterations:
        row = {"iteration": iter_num}
        for task_type in task_types:
            if task_type in iter_data[iter_num]:
                correctness = iter_data[iter_num][task_type]
                accuracy = sum(correctness) / len(correctness) if correctness else 0
                row[f"{task_type}_accuracy"] = accuracy
        progress_data.append(row)
    
    progress_df = pd.DataFrame(progress_data)
    
    # Save to CSV
    csv_path = output_dir / "iteration_progress.csv"
    progress_df.to_csv(csv_path, index=False)
    print(f"Saved progress data to {csv_path}")
    
    # Print summary
    print("\nSummary of evaluation progress:")
    print(f"Analyzed {len(eval_files)} evaluation files across {len(iterations)} iterations")
    print(f"Task types found: {', '.join(task_types)}")
    
    if iterations:
        first_iter = min(iterations)
        last_iter = max(iterations)
        print(f"\nAccuracy comparison between first (iter {first_iter}) and last (iter {last_iter}) iterations:")
        
        for task_type in task_types:
            first_acc = metrics_for_task(iter_metrics[first_iter], task_type)
            last_acc = metrics_for_task(iter_metrics[last_iter], task_type)
            change = last_acc - first_acc
            print(f"  {task_type}: {first_acc:.4f} → {last_acc:.4f} ({change:+.4f})")

def metrics_for_task(metrics, task_type):
    """Get the metric value for a specific task type."""
    return metrics.get(task_type, 0.0)

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation outputs')
    parser.add_argument('run_dir', help='Directory containing evaluation output files')
    parser.add_argument('--output-dir', default='analysis/eval_analysis/output', 
                      help='Directory to save analysis results')
    parser.add_argument('--problems-per-task', type=int, default=45,
                      help='Number of problems per task type (default: 45)')
    
    args = parser.parse_args()
    analyze_eval_outputs(args.run_dir, args.output_dir, args.problems_per_task)

if __name__ == "__main__":
    main() 