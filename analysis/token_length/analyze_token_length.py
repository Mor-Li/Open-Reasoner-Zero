#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze the token length of prompts in training data using Qwen2.5-7B tokenizer.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path

def analyze_file(file_path, tokenizer, desc=None):
    """Analyze token lengths in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lengths = []
    
    def process_item(item):
        # Handle cases where item might be a dictionary
        if isinstance(item, dict):
            if "prompt" in item:
                return item["prompt"]
            elif "question" in item:
                return item["question"]
            elif "value" in item and isinstance(item["value"], str):
                return item["value"]
            else:
                # Try to find any long string
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        return value
            return None
        
        # Handle cases where item might be a list that contains dictionaries
        elif isinstance(item, list):
            for element in item:
                if isinstance(element, dict) and "from" in element and element.get("from") == "human":
                    if "value" in element and isinstance(element["value"], str):
                        return element["value"]
            return None
        
        return None
    
    # Handle different data formats (list of lists, list of dicts, etc.)
    if isinstance(data, list):
        for item in tqdm(data, desc=desc or f"Processing {os.path.basename(file_path)}"):
            prompt = None
            
            # Case 1: Data is a list of conversations (each a list of messages)
            if isinstance(item, list):
                prompt = process_item(item)
            # Case 2: Data is a list of dictionaries
            elif isinstance(item, dict):
                prompt = process_item(item)
                
            if prompt:
                tokens = tokenizer.encode(prompt)
                lengths.append(len(tokens))
    
    return lengths

def main():
    parser = argparse.ArgumentParser(description='Analyze token lengths in training data')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Directory containing training data files')
    parser.add_argument('--output_dir', type=str, default='analysis/token_length',
                        help='Directory to save analysis results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B',
                        help='Model name for tokenizer')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Get list of JSON files in data directory
    json_files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Analyze each file
    all_lengths = []
    file_results = {}
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\nAnalyzing {file_name}...")
        
        lengths = analyze_file(file_path, tokenizer)
        if not lengths:
            print(f"No valid prompts found in {file_name}")
            continue
            
        all_lengths.extend(lengths)
        
        # Calculate statistics
        stats = {
            'min': np.min(lengths),
            'max': np.max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'p90': np.percentile(lengths, 90),
            'p95': np.percentile(lengths, 95),
            'p99': np.percentile(lengths, 99),
            'std': np.std(lengths),
            'count': len(lengths)
        }
        
        file_results[file_name] = stats
        
        # Print statistics for this file
        print(f"File: {file_name}")
        print(f"  Count: {stats['count']}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  90th percentile: {stats['p90']:.2f}")
        print(f"  95th percentile: {stats['p95']:.2f}")
        print(f"  99th percentile: {stats['p99']:.2f}")
        print(f"  Standard deviation: {stats['std']:.2f}")
        
        # Create histogram for this file
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7)
        plt.title(f'Token Length Distribution - {file_name}')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.axvline(stats['mean'], color='r', linestyle='dashed', linewidth=1, label=f'Mean: {stats["mean"]:.2f}')
        plt.axvline(stats['median'], color='g', linestyle='dashed', linewidth=1, label=f'Median: {stats["median"]:.2f}')
        plt.axvline(stats['p95'], color='b', linestyle='dashed', linewidth=1, label=f'95th %: {stats["p95"]:.2f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, f'{os.path.splitext(file_name)[0]}_histogram.png'))
        plt.close()
    
    # Calculate overall statistics
    if all_lengths:
        overall_stats = {
            'min': np.min(all_lengths),
            'max': np.max(all_lengths),
            'mean': np.mean(all_lengths),
            'median': np.median(all_lengths),
            'p90': np.percentile(all_lengths, 90),
            'p95': np.percentile(all_lengths, 95),
            'p99': np.percentile(all_lengths, 99),
            'std': np.std(all_lengths),
            'count': len(all_lengths)
        }
        
        # Print overall statistics
        print("\nOverall Statistics:")
        print(f"  Total prompts: {overall_stats['count']}")
        print(f"  Min: {overall_stats['min']}")
        print(f"  Max: {overall_stats['max']}")
        print(f"  Mean: {overall_stats['mean']:.2f}")
        print(f"  Median: {overall_stats['median']:.2f}")
        print(f"  90th percentile: {overall_stats['p90']:.2f}")
        print(f"  95th percentile: {overall_stats['p95']:.2f}")
        print(f"  99th percentile: {overall_stats['p99']:.2f}")
        print(f"  Standard deviation: {overall_stats['std']:.2f}")
        
        # Create overall histogram
        plt.figure(figsize=(12, 7))
        plt.hist(all_lengths, bins=50, alpha=0.7)
        plt.title('Overall Token Length Distribution')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.axvline(overall_stats['mean'], color='r', linestyle='dashed', linewidth=1, label=f'Mean: {overall_stats["mean"]:.2f}')
        plt.axvline(overall_stats['median'], color='g', linestyle='dashed', linewidth=1, label=f'Median: {overall_stats["median"]:.2f}')
        plt.axvline(overall_stats['p95'], color='b', linestyle='dashed', linewidth=1, label=f'95th %: {overall_stats["p95"]:.2f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'overall_histogram.png'))
        plt.close()
        
        # Save all results to a JSON file
        results = {
            'overall': overall_stats,
            'files': file_results
        }
        
        with open(os.path.join(args.output_dir, 'token_length_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {os.path.join(args.output_dir, 'token_length_analysis.json')}")

if __name__ == '__main__':
    main() 
    
    
    
# (base) root@di-20250312114317-s565s /fs-computility/llmeval/limo/stepfun/Open-Reasoner-Zero ➜ git:(main) [04-29 19:35:06] ./analysis/token_length/run_analysis.sh
# Loading tokenizer from Qwen/Qwen2.5-7B...

# Analyzing rd_en.json...
# Processing rd_en.json: 100%|███████████████████████████████████████████████| 90/90 [00:00<00:00, 213.84it/s]
# File: rd_en.json
#   Count: 90
#   Min: 423
#   Max: 9874
#   Mean: 2434.68
#   Median: 982.50
#   90th percentile: 9415.50
#   95th percentile: 9525.00
#   99th percentile: 9675.53
#   Standard deviation: 2876.01

# Analyzing na_en.json...
# Processing na_en.json: 100%|███████████████████████████████████████████████| 90/90 [00:00<00:00, 225.21it/s]
# File: na_en.json
#   Count: 90
#   Min: 378
#   Max: 9720
#   Mean: 2371.60
#   Median: 928.50
#   90th percentile: 9320.60
#   95th percentile: 9445.85
#   99th percentile: 9598.07
#   Standard deviation: 2867.29
