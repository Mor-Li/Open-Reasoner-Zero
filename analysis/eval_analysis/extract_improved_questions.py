#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import shutil

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
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {filepath}: {line}")
    return data

def extract_improved_questions(run_dir: str, output_dir: str, problems_per_task: int = 45):
    """
    Extract questions that were initially wrong but became correct in the final iteration.
    
    Args:
        run_dir: Directory containing evaluation output files
        output_dir: Directory to save extracted questions
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
    
    # Parse filenames and extract iterations
    iterations_data = {}
    
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
            
            # Read the evaluation results
            filepath = os.path.join(run_dir, file)
            results = read_jsonl(filepath)
            
            # Store results by iteration for reference
            iterations_data[iteration] = {
                "file": file,
                "results": results,
                "task_types": expected_task_types
            }
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    if not iterations_data:
        print("No valid data could be processed. Exiting.")
        return
    
    # Sort iterations to find first and last
    iterations = sorted(iterations_data.keys())
    first_iter = iterations[0]
    last_iter = iterations[-1]
    
    print(f"Analyzing data from iteration {first_iter} to {last_iter}")
    
    # Create a directory for tracking progress across iterations
    tracking_dir = output_dir / "question_tracking"
    if tracking_dir.exists():
        shutil.rmtree(tracking_dir)
    tracking_dir.mkdir(parents=True)
    
    # Identify improved questions (wrong in first iteration, correct in last iteration)
    improved_questions = []
    questions_by_task = defaultdict(list)
    
    first_iter_data = iterations_data[first_iter]
    last_iter_data = iterations_data[last_iter]
    
    first_results = first_iter_data["results"]
    last_results = last_iter_data["results"]
    task_types = first_iter_data["task_types"]
    
    for i, (first_item, last_item) in enumerate(zip(first_results, last_results)):
        # Determine task type from position in the file
        task_idx = i // problems_per_task
        if task_idx >= len(task_types):
            continue
        
        task_type = task_types[task_idx]
        
        # Calculate local index within the task type dataset
        local_index = i % problems_per_task
        
        # Check if question improved
        if first_item.get("iscorrect", False) == False and last_item.get("iscorrect", False) == True:
            question_info = {
                "index": i,  # Global index across all datasets
                "local_index": local_index,  # Local index within this specific task type
                "task_type": task_type,
                "question_id": local_index,
                "prompt": first_item.get("prompt", ""),
                "first_answer": first_item.get("output", ""),
                "last_answer": last_item.get("output", ""),
                "reference_answer": first_item.get("answer", "")
            }
            improved_questions.append(question_info)
            questions_by_task[task_type].append(question_info)
    
    # Print summary of improved questions
    print(f"Found {len(improved_questions)} questions that improved from wrong to correct")
    for task_type, questions in questions_by_task.items():
        print(f"  {task_type}: {len(questions)} questions")
    
    # Save improved questions summary
    improved_summary = {
        "total_improved": len(improved_questions),
        "by_task_type": {task: len(questions) for task, questions in questions_by_task.items()},
        "questions": improved_questions
    }
    
    with open(output_dir / "improved_questions_summary.json", "w", encoding="utf-8") as f:
        json.dump(improved_summary, f, ensure_ascii=False, indent=2)
    
    print(f"Saved summary to {output_dir / 'improved_questions_summary.json'}")
    
    # Track each improved question across all iterations
    question_progress = defaultdict(dict)
    
    for iteration, iter_data in iterations_data.items():
        results = iter_data["results"]
        
        for question_info in improved_questions:
            idx = question_info["index"]
            if idx < len(results):
                item = results[idx]
                question_progress[idx][iteration] = {
                    "iscorrect": item.get("iscorrect", False),
                    "output": item.get("output", ""),
                    "final_answer": item.get("final_answer", "")
                }
    
    # Save individual question progress files
    for idx, iterations_info in question_progress.items():
        question_info = next(q for q in improved_questions if q["index"] == idx)
        
        # Create a more descriptive filename
        task_type = question_info["task_type"]
        local_index = question_info["local_index"]
        
        filename = f"{task_type}_q{local_index:02d}_idx{idx:03d}.json"
        
        # Prepare question data with all iterations
        question_data = {
            "index": idx,  # Global index
            "local_index": local_index,  # Local index within task type
            "task_type": task_type,
            "question_id": local_index,
            "prompt": question_info["prompt"],
            "reference_answer": question_info["reference_answer"],
            "iterations": {}
        }
        
        # Add data for each iteration
        for iteration in sorted(iterations_info.keys()):
            question_data["iterations"][str(iteration)] = iterations_info[iteration]
        
        # Save to file
        with open(tracking_dir / filename, "w", encoding="utf-8") as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved detailed progress for each improved question to {tracking_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract questions that improved from wrong to correct')
    parser.add_argument('run_dir', help='Directory containing evaluation output files')
    parser.add_argument('--output-dir', default='analysis/eval_analysis/improved_questions', 
                      help='Directory to save extracted questions')
    parser.add_argument('--problems-per-task', type=int, default=45,
                      help='Number of problems per task type (default: 45)')
    
    args = parser.parse_args()
    extract_improved_questions(args.run_dir, args.output_dir, args.problems_per_task)

if __name__ == "__main__":
    main() 