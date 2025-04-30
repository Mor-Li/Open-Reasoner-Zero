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

def analyze_problem_difficulty(run_dir: str, output_dir: str, problems_per_task: int = 45):
    """
    Analyze problem difficulty based on correctness across iterations.
    
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
    
    # Build problem tracking across iterations
    problems_by_iter = {}
    problem_fingerprints = {}  # To identify same problems across iterations
    
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
                
            filepath = os.path.join(run_dir, file)
            results = read_jsonl(filepath)
            
            problems_by_iter[iteration] = []
            
            for i, item in enumerate(results):
                prompt = item.get("prompt", "")
                # Extract a short snippet to use as fingerprint (first 100 chars)
                fingerprint = prompt[:100]
                
                if fingerprint not in problem_fingerprints:
                    problem_fingerprints[fingerprint] = len(problem_fingerprints)
                
                problem_id = problem_fingerprints[fingerprint]
                
                # Determine task type from position in the file
                task_idx = i // problems_per_task
                if task_idx < len(expected_task_types):
                    task_type = expected_task_types[task_idx]
                    language = task_type.split('_')[1]  # Extract language from task_type
                else:
                    # Fallback to heuristic detection if task_idx is out of range
                    if "zh" in file and any(x in prompt for x in ["中文", "汉语", "简体", "繁体"]):
                        language = "zh"
                    else:
                        language = "en"
                        
                    # Determine task type based on prompt characteristics
                    if any(x in prompt.lower() for x in ["elementary", "arithmetic", "简单", "初等"]):
                        task_prefix = "ea"
                    elif any(x in prompt.lower() for x in ["natural", "自然"]):
                        task_prefix = "na"
                    elif any(x in prompt.lower() for x in ["decomposition", "分解"]):
                        task_prefix = "nd"
                    elif any(x in prompt.lower() for x in ["difficult", "reasoning", "困难"]):
                        task_prefix = "rd"
                    else:
                        # Default to "unknown"
                        task_prefix = "unknown"
                    
                    task_type = f"{task_prefix}_{language}"
                
                # Store problem data
                problems_by_iter[iteration].append({
                    "problem_id": problem_id,
                    "task_type": task_type,
                    "language": language,
                    "is_correct": item.get("iscorrect", False),
                    "index": i
                })
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if not problems_by_iter:
        print("No valid data could be processed. Exiting.")
        return
    
    # Create a unified dataframe for all problems across iterations
    all_data = []
    for iteration, problems in problems_by_iter.items():
        for problem in problems:
            all_data.append({
                "iteration": iteration,
                "problem_id": problem["problem_id"],
                "task_type": problem["task_type"],
                "language": problem["language"],
                "is_correct": problem["is_correct"],
                "index": problem["index"]
            })
    
    df = pd.DataFrame(all_data)
    
    # Save the dataframe
    csv_path = output_dir / "problem_tracking.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved problem tracking data to {csv_path}")
    
    # Analyze problem difficulty (percentage of times correctly solved)
    problem_difficulty = df.groupby("problem_id")["is_correct"].agg(
        difficulty=lambda x: 1 - x.mean(),  # 1 - correctness_rate (higher = more difficult)
        count=lambda x: len(x)
    ).reset_index()
    
    # Add task type for each problem
    problem_task_types = df.groupby("problem_id")["task_type"].first().reset_index()
    problem_difficulty = problem_difficulty.merge(problem_task_types, on="problem_id")
    
    # Save problem difficulty
    difficulty_path = output_dir / "problem_difficulty.csv"
    problem_difficulty.to_csv(difficulty_path, index=False)
    print(f"Saved problem difficulty analysis to {difficulty_path}")
    
    # Analyze transitions (wrong to right, right to wrong)
    # Get problems that appear in at least 2 iterations
    problem_counts = df.groupby("problem_id").size()
    multi_iter_problems = problem_counts[problem_counts >= 2].index
    
    transitions = []
    for problem_id in multi_iter_problems:
        problem_data = df[df["problem_id"] == problem_id].sort_values("iteration")
        
        for i in range(len(problem_data) - 1):
            current = problem_data.iloc[i]
            next_iter = problem_data.iloc[i+1]
            
            transition = {
                "problem_id": problem_id,
                "from_iter": current["iteration"],
                "to_iter": next_iter["iteration"],
                "from_correct": current["is_correct"],
                "to_correct": next_iter["is_correct"],
                "task_type": current["task_type"]
            }
            
            if current["is_correct"] != next_iter["is_correct"]:
                if current["is_correct"]:
                    transition["transition_type"] = "right_to_wrong"
                else:
                    transition["transition_type"] = "wrong_to_right"
            else:
                if current["is_correct"]:
                    transition["transition_type"] = "right_to_right"
                else:
                    transition["transition_type"] = "wrong_to_wrong"
                    
            transitions.append(transition)
    
    transitions_df = pd.DataFrame(transitions)
    
    # Save transitions
    transitions_path = output_dir / "problem_transitions.csv"
    transitions_df.to_csv(transitions_path, index=False)
    print(f"Saved problem transitions to {transitions_path}")
    
    # Create visualizations
    
    # 1. Problem difficulty distribution by task type
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="task_type", y="difficulty", data=problem_difficulty)
    plt.title("Problem Difficulty Distribution by Task Type")
    plt.xlabel("Task Type")
    plt.ylabel("Difficulty (1 - correctness rate)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_by_task_type.png")
    plt.close()
    
    # 2. Transition sankey diagram (simplified as stacked bar)
    transition_counts = transitions_df.groupby(["task_type", "transition_type"]).size().reset_index(name="count")
    pivot_df = transition_counts.pivot(index="task_type", columns="transition_type", values="count").fillna(0)
    
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Transition Types by Task Category")
    plt.xlabel("Task Type")
    plt.ylabel("Number of Transitions")
    plt.legend(title="Transition Type")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "transition_types.png")
    plt.close()
    
    # 3. Learning curves - show how correctness improves over iterations
    # Group by iteration and task_type, calculate percentage correct
    learning_curve = df.groupby(["iteration", "task_type"])["is_correct"].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    task_types = df["task_type"].unique()
    
    for task_type in task_types:
        task_data = learning_curve[learning_curve["task_type"] == task_type]
        if not task_data.empty:
            plt.plot(task_data["iteration"], task_data["is_correct"], 
                    marker='o', label=task_type)
    
    plt.title("Learning Curves by Task Type")
    plt.xlabel("Iteration")
    plt.ylabel("Percentage Correct")
    plt.legend(title="Task Type")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png")
    plt.close()
    
    # 4. Consistency Analysis - how consistently problems are solved correctly
    # Create a heatmap of consistency by problem_id and iteration range
    # First, identify iterations with enough data
    iterations = sorted(df["iteration"].unique())
    
    # We'll split iterations into 3 ranges: early, middle, late
    if len(iterations) >= 3:
        iter_ranges = [
            (iterations[0], iterations[len(iterations)//3]),
            (iterations[len(iterations)//3], iterations[2*len(iterations)//3]),
            (iterations[2*len(iterations)//3], iterations[-1])
        ]
        range_names = ["Early", "Middle", "Late"]
        
        # Calculate correctness rate for each problem in each range
        consistency_data = []
        
        for problem_id in df["problem_id"].unique():
            problem_data = df[df["problem_id"] == problem_id]
            task_type = problem_data["task_type"].iloc[0]
            
            for i, (start_iter, end_iter) in enumerate(iter_ranges):
                range_data = problem_data[
                    (problem_data["iteration"] >= start_iter) & 
                    (problem_data["iteration"] <= end_iter)
                ]
                
                if not range_data.empty:
                    correctness_rate = range_data["is_correct"].mean()
                    consistency_data.append({
                        "problem_id": problem_id,
                        "range": range_names[i],
                        "correctness_rate": correctness_rate,
                        "task_type": task_type
                    })
        
        consistency_df = pd.DataFrame(consistency_data)
        
        # Create heatmap for each task type
        for task_type in df["task_type"].unique():
            task_consistency = consistency_df[consistency_df["task_type"] == task_type]
            
            if len(task_consistency) > 1:  # Only plot if we have enough data
                # Pivot the data for the heatmap
                pivot_data = task_consistency.pivot(
                    index="problem_id", 
                    columns="range", 
                    values="correctness_rate"
                ).fillna(0)
                
                if not pivot_data.empty and pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
                    plt.figure(figsize=(10, max(8, len(pivot_data) // 3)))
                    sns.heatmap(
                        pivot_data, 
                        cmap="RdYlGn", 
                        vmin=0, 
                        vmax=1,
                        annot=True,
                        fmt=".2f"
                    )
                    plt.title(f"Problem Consistency Over Time - {task_type}")
                    plt.ylabel("Problem ID")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"consistency_{task_type}.png")
                    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze problem difficulty')
    parser.add_argument('run_dir', help='Directory containing evaluation output files')
    parser.add_argument('--output-dir', default='analysis/eval_analysis/output', 
                      help='Directory to save analysis results')
    parser.add_argument('--problems-per-task', type=int, default=45,
                      help='Number of problems per task type (default: 45)')
    
    args = parser.parse_args()
    analyze_problem_difficulty(args.run_dir, args.output_dir, args.problems_per_task)

if __name__ == "__main__":
    main() 