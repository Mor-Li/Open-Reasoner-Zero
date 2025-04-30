# Evaluation Output Analysis Tools

This directory contains scripts to analyze the progress of model training through evaluation output files.

## Overview

The analysis scripts in this directory help you visualize and understand how the model's performance evolves across training iterations. The scripts analyze the JSONL evaluation output files produced during model training.

## Scripts

### 1. `visualize_eval_progress.py`

This script analyzes evaluation output files and creates visualizations of model performance across iterations.

**Features:**
- Generates heatmaps showing which problems are solved correctly across iterations
- Creates line charts of accuracy trends for different task types
- Produces summary statistics about performance improvements

**Usage:**
```bash
python visualize_eval_progress.py [run_dir] --output-dir [output_dir] --problems-per-task [num_problems]
```

Arguments:
- `run_dir`: Directory containing evaluation output files (e.g., `orz_ckpt/debug_orz_7b_ppo_atc_prm/`)
- `--output-dir`: Directory to save analysis results (default: `analysis/eval_analysis/output`)
- `--problems-per-task`: Number of problems per task type (default: 45)

### 2. `problem_difficulty_analysis.py`

This script analyzes the difficulty of problems and how the model's ability to solve them changes across iterations.

**Features:**
- Analyzes problem difficulty based on correctness across iterations
- Tracks problem transitions (wrong to right, right to wrong)
- Creates visualizations of problem difficulty distributions
- Shows how problem correctness evolves across training

**Usage:**
```bash
python problem_difficulty_analysis.py [run_dir] --output-dir [output_dir] --problems-per-task [num_problems]
```

Arguments:
- `run_dir`: Directory containing evaluation output files (e.g., `orz_ckpt/debug_orz_7b_ppo_atc_prm/`)
- `--output-dir`: Directory to save analysis results (default: `analysis/eval_analysis/output`)
- `--problems-per-task`: Number of problems per task type (default: 45)

### 3. `problem_correctness_heatmap.py`

This script creates detailed heatmap visualizations showing the correctness of individual problems across training iterations.

**Features:**
- Generates detailed heatmaps for each task type
- Creates a combined heatmap of all problems grouped by task type
- Optionally exports detailed problem information including success rates and first solved iteration

**Usage:**
```bash
python problem_correctness_heatmap.py [run_dir] --output-dir [output_dir] [--add-problem-details] --problems-per-task [num_problems]
```

Arguments:
- `run_dir`: Directory containing evaluation output files (e.g., `orz_ckpt/debug_orz_7b_ppo_atc_prm/`)
- `--output-dir`: Directory to save analysis results (default: `analysis/eval_analysis/output`)
- `--add-problem-details`: Flag to generate a CSV file with detailed problem information
- `--problems-per-task`: Number of problems per task type (default: 45)

### 4. `extract_improved_questions.py`

This script identifies questions that were initially answered incorrectly but were later answered correctly in the final iteration, and tracks their progress across all iterations.

**Features:**
- Identifies questions that improved from wrong to correct
- Provides a summary of improved questions by task type
- Creates individual JSON files for each improved question, showing its progress across all iterations
- Preserves the full prompt, outputs, and answers for detailed analysis

**Usage:**
```bash
python extract_improved_questions.py [run_dir] --output-dir [output_dir] --problems-per-task [num_problems]
```

Arguments:
- `run_dir`: Directory containing evaluation output files (e.g., `orz_ckpt/debug_orz_7b_ppo_atc_prm/`)
- `--output-dir`: Directory to save extracted questions (default: `analysis/eval_analysis/improved_questions`)
- `--problems-per-task`: Number of problems per task type (default: 45)

### 5. `run_all_analysis.py`

This wrapper script runs all three analysis tools sequentially, making it easy to perform a complete analysis in one command.

**Features:**
- Runs all three analysis scripts in sequence
- Automatically creates a suitable output directory based on the run directory name
- Provides timing information for the complete analysis process

**Usage:**
```bash
python run_all_analysis.py [run_dir] [--output-dir output_dir] [--add-problem-details] --problems-per-task [num_problems]
```

Arguments:
- `run_dir`: Directory containing evaluation output files (e.g., `orz_ckpt/debug_orz_7b_ppo_atc_prm/`)
- `--output-dir`: Directory to save analysis results (default: `analysis/eval_analysis/output/RUN_DIR_NAME`)
- `--add-problem-details`: Flag to generate a CSV file with detailed problem information
- `--problems-per-task`: Number of problems per task type (default: 45)

## Examples

### Basic Analysis

To analyze a specific model run:

```bash
# Run all analysis at once
python analysis/eval_analysis/run_all_analysis.py orz_ckpt/debug_orz_7b_ppo_atc_prm/ --add-problem-details

# Extract questions that improved during training
python analysis/eval_analysis/extract_improved_questions.py orz_ckpt/debug_orz_7b_ppo_atc_prm/

# Or run individual scripts
python analysis/eval_analysis/visualize_eval_progress.py orz_ckpt/debug_orz_7b_ppo_atc_prm/
python analysis/eval_analysis/problem_difficulty_analysis.py orz_ckpt/debug_orz_7b_ppo_atc_prm/
python analysis/eval_analysis/problem_correctness_heatmap.py orz_ckpt/debug_orz_7b_ppo_atc_prm/ --add-problem-details
```

### Comparing Multiple Runs

To compare multiple runs, run the analysis for each and then examine the output files.

```bash
# Analyze multiple runs using the wrapper script
python analysis/eval_analysis/run_all_analysis.py orz_ckpt/run1/
python analysis/eval_analysis/run_all_analysis.py orz_ckpt/run2/

# The results will be saved to:
# - analysis/eval_analysis/output/run1/
# - analysis/eval_analysis/output/run2/
```

## Output Files

The scripts generate several files in the output directory:

### From `visualize_eval_progress.py`:
- `heatmap_*.png`: Heatmaps showing problem correctness across iterations
- `accuracy_trends.png`: Line chart of accuracy trends across iterations
- `iteration_progress.csv`: CSV file with accuracy data for each iteration

### From `problem_difficulty_analysis.py`:
- `problem_tracking.csv`: CSV file tracking problems across iterations
- `problem_difficulty.csv`: CSV file with difficulty scores for each problem
- `problem_transitions.csv`: CSV file tracking transitions between correct/incorrect answers
- `learning_curves.png`: Line chart showing how correctness improves over iterations
- `difficulty_by_task_type.png`: Box plot showing problem difficulty by task type
- `transition_types.png`: Bar chart showing transition types by task category
- `consistency_*.png`: Heatmaps showing problem consistency over time

### From `problem_correctness_heatmap.py`:
- `detailed_heatmap_*.png`: Detailed heatmaps for each task type
- `detailed_heatmap_all_problems.png`: Combined heatmap of all problems
- `problem_details.csv`: CSV file with detailed information about each problem (when using --add-problem-details)

### From `extract_improved_questions.py`:
- `improved_questions_summary.json`: Summary of all questions that improved from wrong to correct
- `question_tracking/`: Directory containing individual JSON files for each improved question
- `question_tracking/{task_type}_q{question_id}_idx{index}.json`: Detailed tracking of a specific question across all iterations

## Requirements

These scripts require the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- pathlib 