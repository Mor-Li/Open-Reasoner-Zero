# Question Progress Visualization

This tool visualizes how model outputs change across iterations for improved questions, highlighting person names with background colors to help analyze reasoning progress.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the improved questions data:

```bash
python extract_improved_questions.py <run_dir> --output-dir analysis/eval_analysis/improved_questions --eval-data-dir data/eval_data
```

Where:
- `<run_dir>` is the directory containing evaluation output files (jsonl files)
- `--output-dir` specifies where to save the extracted questions
- `--eval-data-dir` points to the original evaluation data files

3. Launch the visualization interface:

```bash
python visualize_progress.py
```

## Features

- Visualize model outputs across iterations for questions that improved from wrong to correct
- Highlight person names with color coding based on generational relationships
- Red colors represent older generations, blue colors represent younger generations
- Track the model's reasoning progress throughout iterations
- See at a glance when the model starts reasoning correctly

## How to Use

1. Select a question from the dropdown menu
2. View the original prompt with highlighted names
3. Examine each iteration's output with highlighted names
4. Use the color legend to identify which names represent which generations
5. Analyze how the model's reasoning about familial relationships improved over iterations

## Example

For each question, you'll see:
- The original prompt with highlighted person names
- The iterations of model outputs, with correct/incorrect status indicated
- A color legend showing the generational relationships 