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

def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read jsonl file and return a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

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

def create_correctness_heatmap(run_dir: str, output_dir: str, add_problem_details: bool = False, problems_per_task: int = 45):
    """
    Create a detailed heatmap showing correctness of problems across iterations.
    
    Args:
        run_dir: Directory containing evaluation output files
        output_dir: Directory to save analysis results
        add_problem_details: Whether to add problem details to the heatmap
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
    
    # Build problem tracking across iterations
    all_problems = {}  # Maps fingerprint to problem_id
    problem_data = defaultdict(dict)  # Maps (problem_id, iteration) to correctness
    task_types = defaultdict(str)  # Maps problem_id to task_type
    problem_prompts = {}  # Maps problem_id to prompt (for problem details)
    
    iterations = []
    
    for file in eval_files:
        try:
            iteration, _ = parse_eval_filename(file)
            if iteration is None:
                continue
            
            # Get expected task types from filename in order
            expected_task_types = get_task_types_from_filename(file)
            
            if not expected_task_types:
                print(f"Warning: Could not extract task types from filename: {file}")
                continue
                
            iterations.append(iteration)
            filepath = os.path.join(run_dir, file)
            results = read_jsonl(filepath)
            
            for i, item in enumerate(results):
                prompt = item.get("prompt", "")
                # Extract a short snippet to use as fingerprint (first 100 chars)
                fingerprint = prompt[:100]
                
                if fingerprint not in all_problems:
                    problem_id = len(all_problems)
                    all_problems[fingerprint] = problem_id
                    problem_prompts[problem_id] = prompt
                else:
                    problem_id = all_problems[fingerprint]
                
                # Determine task type from position in the file
                task_idx = i // problems_per_task
                if task_idx < len(expected_task_types):
                    task_type = expected_task_types[task_idx]
                    language = task_type.split('_')[1]  # Extract language from task_type
                else:
                    # Fallback to detection from file name
                    if "zh" in file:
                        language = "zh"
                    else:
                        language = "en"
                        
                    # Try to determine task type based on prompt content
                    if any(x in prompt.lower() for x in ["elementary", "arithmetic", "简单", "初等"]):
                        task_prefix = "ea"
                    elif any(x in prompt.lower() for x in ["natural", "自然"]):
                        task_prefix = "na" 
                    elif any(x in prompt.lower() for x in ["decomposition", "分解"]):
                        task_prefix = "nd"
                    elif any(x in prompt.lower() for x in ["difficult", "reasoning", "困难"]):
                        task_prefix = "rd"
                    else:
                        task_prefix = "unknown"
                    
                    task_type = f"{task_prefix}_{language}"
                
                task_types[problem_id] = task_type
                is_correct = item.get("iscorrect", False)
                problem_data[(problem_id, iteration)] = is_correct
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
            
    if not all_problems:
        print("No valid data could be processed. Exiting.")
        return
    
    # Group problems by task type
    problems_by_task = defaultdict(list)
    for problem_id, task_type in task_types.items():
        problems_by_task[task_type].append(problem_id)
    
    # Create a comprehensive dataframe for all problems and iterations
    rows = []
    for problem_id in range(len(all_problems)):
        task_type = task_types[problem_id]
        for iteration in sorted(iterations):
            is_correct = problem_data.get((problem_id, iteration), None)
            if is_correct is not None:
                rows.append({
                    "problem_id": problem_id,
                    "iteration": iteration,
                    "task_type": task_type,
                    "is_correct": is_correct
                })
    
    df = pd.DataFrame(rows)
    
    # Create visualizations by task type
    for task_type, problem_ids in problems_by_task.items():
        if not problem_ids:
            continue
            
        # Filter the dataframe for this task type
        task_df = df[df["task_type"] == task_type]
        
        # Create a matrix for the heatmap
        iterations_sorted = sorted(iterations)
        problem_ids_sorted = sorted(problem_ids)
        
        matrix = np.full((len(problem_ids_sorted), len(iterations_sorted)), np.nan)
        
        for i, problem_id in enumerate(problem_ids_sorted):
            for j, iteration in enumerate(iterations_sorted):
                is_correct = problem_data.get((problem_id, iteration), None)
                if is_correct is not None:
                    matrix[i, j] = 1 if is_correct else 0
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(problem_ids_sorted) // 3)))
        
        # Use a custom colormap with NaN values as white
        cmap = plt.cm.RdYlGn
        cmap.set_bad('white')
        
        # Create the heatmap
        ax = sns.heatmap(
            matrix,
            cmap=['#ffcccc', '#99ff99'],  # Red for wrong, green for correct
            cbar=False,
            xticklabels=[str(iter_num) for iter_num in iterations_sorted],
            yticklabels=problem_ids_sorted,
            mask=np.isnan(matrix)
        )
        
        plt.xlabel('Iteration')
        plt.ylabel('Problem ID')
        plt.title(f'Problem Correctness Across Iterations - {task_type}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(output_dir / f"detailed_heatmap_{task_type}.png")
        plt.close()
        
    # Create a combined visualization with all problems
    all_problem_ids = sorted(range(len(all_problems)))
    iterations_sorted = sorted(iterations)
    
    # Group problem IDs by task type for better visualization
    grouped_problem_ids = []
    task_boundaries = []
    current_boundary = 0
    
    for task_type in ["ea_zh", "ea_en", "na_zh", "na_en", "nd_zh", "nd_en", "rd_zh", "rd_en"]:
        problem_ids = [pid for pid in all_problem_ids if task_types[pid] == task_type]
        if problem_ids:
            grouped_problem_ids.extend(problem_ids)
            current_boundary += len(problem_ids)
            task_boundaries.append((current_boundary, task_type))
    
    # Create the matrix for visualization
    matrix = np.full((len(grouped_problem_ids), len(iterations_sorted)), np.nan)
    
    for i, problem_id in enumerate(grouped_problem_ids):
        for j, iteration in enumerate(iterations_sorted):
            is_correct = problem_data.get((problem_id, iteration), None)
            if is_correct is not None:
                matrix[i, j] = 1 if is_correct else 0
    
    # Create the combined heatmap
    plt.figure(figsize=(14, max(10, len(grouped_problem_ids) // 2)))
    
    # Use a custom colormap with NaN values as white
    cmap = plt.cm.RdYlGn
    cmap.set_bad('white')
    
    # Create the heatmap
    ax = sns.heatmap(
        matrix,
        cmap=['#ffcccc', '#99ff99'],  # Red for wrong, green for correct
        cbar=False,
        xticklabels=[str(iter_num) for iter_num in iterations_sorted],
        yticklabels=grouped_problem_ids,
        mask=np.isnan(matrix)
    )
    
    # Add horizontal lines to separate task types
    prev_boundary = 0
    for boundary, task_type in task_boundaries:
        if boundary > 0:
            plt.axhline(y=boundary, color='black', linestyle='-', linewidth=1)
            # Add a task type label in the middle of each section
            midpoint = (prev_boundary + boundary) / 2
            plt.text(-0.05, midpoint, task_type, rotation=90, 
                     verticalalignment='center', horizontalalignment='right',
                     transform=ax.get_yaxis_transform(), fontsize=10)
            prev_boundary = boundary
    
    plt.xlabel('Iteration')
    plt.ylabel('Problem ID')
    plt.title('All Problems Correctness Across Iterations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the combined visualization
    plt.savefig(output_dir / f"detailed_heatmap_all_problems.png")
    plt.close()
    
    # Create a CSV file with detailed problem information
    if add_problem_details:
        problem_details = []
        for problem_id in all_problem_ids:
            prompt = problem_prompts[problem_id]
            
            # Extract a simplified version of the prompt (first 100 chars)
            simplified_prompt = prompt[:100].replace('\n', ' ') + '...'
            
            # Calculate the success rate across iterations
            correct_count = sum(1 for iter_num in iterations if problem_data.get((problem_id, iter_num), False))
            total_count = sum(1 for iter_num in iterations if (problem_id, iter_num) in problem_data)
            success_rate = correct_count / total_count if total_count > 0 else 0
            
            # Find earliest iteration where the problem was solved correctly
            solved_iters = [iter_num for iter_num in iterations_sorted 
                          if problem_data.get((problem_id, iter_num), False)]
            first_solved = min(solved_iters) if solved_iters else None
            
            problem_details.append({
                "problem_id": problem_id,
                "task_type": task_types[problem_id],
                "simplified_prompt": simplified_prompt,
                "success_rate": success_rate,
                "first_solved_iter": first_solved,
                "evaluated_count": total_count
            })
        
        # Save to CSV
        details_df = pd.DataFrame(problem_details)
        details_df.to_csv(output_dir / "problem_details.csv", index=False)
        print(f"Saved problem details to {output_dir / 'problem_details.csv'}")
    
    print(f"Created detailed heatmaps in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create detailed problem correctness heatmaps')
    parser.add_argument('run_dir', help='Directory containing evaluation output files')
    parser.add_argument('--output-dir', default='analysis/eval_analysis/output', 
                      help='Directory to save analysis results')
    parser.add_argument('--add-problem-details', action='store_true',
                      help='Add problem details to output')
    parser.add_argument('--problems-per-task', type=int, default=45,
                      help='Number of problems per task type (default: 45)')
    
    args = parser.parse_args()
    create_correctness_heatmap(args.run_dir, args.output_dir, args.add_problem_details, args.problems_per_task)

if __name__ == "__main__":
    main() 