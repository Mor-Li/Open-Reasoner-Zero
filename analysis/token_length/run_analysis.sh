#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p analysis/token_length

# Run the token length analysis script
python analysis/token_length/analyze_token_length.py \
  --data_dir data \
  --output_dir analysis/token_length \
  --model_name Qwen/Qwen2.5-7B

echo "Analysis complete. Results are in analysis/token_length directory." 