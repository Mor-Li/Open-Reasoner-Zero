#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import gradio as gr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from typing import Dict, List, Any, Tuple, Optional

# Directory containing the tracking files
TRACKING_DIR = Path("analysis/eval_analysis/improved_questions/question_tracking")

def load_question_files() -> Dict[str, str]:
    """
    Load all question tracking files and return a dictionary mapping 
    question IDs to file paths
    """
    question_files = {}
    for file_path in TRACKING_DIR.glob("*.json"):
        question_id = file_path.stem
        question_files[question_id] = str(file_path)
    return question_files

def load_question_data(file_path: str) -> Dict[str, Any]:
    """Load question data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_colors_for_names(names: List[str], total_generations: int) -> Dict[str, str]:
    """
    Generate a color mapping for person names based on their generational index
    
    Args:
        names: List of person names
        total_generations: Total number of generations in the family tree
        
    Returns:
        Dictionary mapping names to HTML color codes
    """
    if not names or total_generations <= 0:
        return {}
    
    # Create a red gradient from darkest (oldest) to lightest (youngest)
    colors = {}
    
    # Process names in order (oldest first in the names list)
    for i, name in enumerate(names):
        # Calculate intensity based on position (darkest red for eldest, lightest red for youngest)
        intensity = 0.3 + 0.7 * (i / (len(names) - 1 or 1))
        
        # Create red gradient (darker = older, lighter = younger)
        r = 255
        g = int(255 * intensity)
        b = int(255 * intensity)
        
        # Convert to hex color string
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        colors[name] = hex_color
    
    return colors

def create_color_bar(name_colors: Dict[str, str]) -> str:
    """
    Create an HTML color bar showing the gradient from oldest to youngest
    
    Args:
        name_colors: Dictionary mapping names to color hex codes
        
    Returns:
        HTML string representing a color bar
    """
    if not name_colors:
        return ""
    
    # Sort names by colors (darkest to lightest red)
    sorted_names = sorted(name_colors.items(), 
                         key=lambda x: sum(int(x[1][i:i+2], 16) for i in (1, 3, 5)))
    
    # Create color bar HTML
    html = """
    <div style="margin-bottom: 20px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Age Relationship:</div>
        <div style="display: flex; flex-direction: row; align-items: center; width: 100%; height: 30px; border-radius: 5px; overflow: hidden;">
    """
    
    # Calculate width for each color segment
    width_per_name = 100 / len(sorted_names)
    
    # Add color segments
    for name, color in sorted_names:
        html += f'<div style="background-color: {color}; height: 100%; width: {width_per_name}%;"></div>'
    
    html += """
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <div>Eldest</div>
            <div>Youngest</div>
        </div>
    </div>
    """
    
    return html

def highlight_names_in_text(text: str, name_colors: Dict[str, str], generation_ranks: Optional[Dict[str, int]] = None) -> str:
    """
    Highlight person names in text with background colors
    
    Args:
        text: The text to highlight
        name_colors: Dictionary mapping names to color hex codes
        generation_ranks: Optional dictionary mapping names to their generation rank
        
    Returns:
        HTML formatted text with highlighted names
    """
    if not text or not name_colors:
        return text
    
    # Sort names by length (longest first) to avoid partial matches
    sorted_names = sorted(name_colors.keys(), key=len, reverse=True)
    
    # Replace each name with a colored span
    for name in sorted_names:
        if name in text:
            # Create tooltip text if generation ranks are provided
            tooltip = ""
            if generation_ranks and name in generation_ranks:
                tooltip = f"Generation rank: {generation_ranks[name]}"
            
            # Create HTML span with background color and optional tooltip
            highlighted = f'<span style="background-color: {name_colors[name]}; padding: 0px 2px; border-radius: 3px;" title="{tooltip}">{name}</span>'
            
            # Replace the name with the highlighted version
            text = text.replace(name, highlighted)
    
    return text

def visualize_question_progress(question_id: str) -> Tuple[str, str, str]:
    """
    Visualize the progress of a question across iterations
    
    Args:
        question_id: The ID of the question to visualize
        
    Returns:
        Tuple of (question prompt, iteration visualizations, generation legend)
    """
    # Load question files
    question_files = load_question_files()
    if question_id not in question_files:
        return "Question not found", "", ""
    
    # Load question data
    question_data = load_question_data(question_files[question_id])
    
    # Extract key information
    prompt = question_data.get("prompt", "")
    reference_answer = question_data.get("reference_answer", "")
    details = question_data.get("details", {})
    iterations = question_data.get("iterations", {})
    
    # Get name and generation information
    names = details.get("names", [])
    total_generations = details.get("total_generations", len(names) if names else 1)
    
    # Generate colors for names
    name_colors = generate_colors_for_names(names, total_generations)
    
    # Create generation rank dictionary
    generation_ranks = {name: i+1 for i, name in enumerate(names)}
    
    # Format prompt with highlighted names
    highlighted_prompt = highlight_names_in_text(prompt, name_colors, generation_ranks)
    prompt_html = f"""
    <div style="white-space: pre-wrap; font-family: monospace; margin-bottom: 20px;">
        {highlighted_prompt}
    </div>
    <div style="margin-bottom: 20px;">
        <b>Reference Answer:</b> {highlight_names_in_text(reference_answer, name_colors, generation_ranks)}
    </div>
    """
    
    # Build HTML for each iteration
    iterations_html = "<h3>Iterations</h3>"
    sorted_iterations = sorted([int(i) for i in iterations.keys()])
    
    for iter_num in sorted_iterations:
        iter_data = iterations.get(str(iter_num), {})
        iterations_html += f"<h4>Iteration {iter_num}</h4>"
        
        # Format iteration output
        output = iter_data.get("output", "")
        is_correct = iter_data.get("iscorrect", False)
        final_answer = iter_data.get("final_answer", "")
        
        highlighted_output = highlight_names_in_text(output, name_colors, generation_ranks)
        
        # Format as HTML with correct/incorrect indication
        status_color = "#4CAF50" if is_correct else "#F44336"  # Green if correct, red if incorrect
        
        output_html = f"""
        <div style="border: 1px solid {status_color}; border-radius: 5px; padding: 10px; margin-bottom: 15px;">
            <div style="font-weight: bold; color: {status_color}; margin-bottom: 5px;">
                {'✓ Correct' if is_correct else '✗ Incorrect'}
            </div>
            <div style="white-space: pre-wrap; font-family: monospace;">
                {highlighted_output}
            </div>
        """
        
        if final_answer:
            highlighted_final = highlight_names_in_text(final_answer, name_colors, generation_ranks)
            output_html += f"""
            <div style="margin-top: 10px; border-top: 1px dotted #ccc; padding-top: 5px;">
                <b>Final Answer:</b> {highlighted_final}
            </div>
            """
        
        output_html += "</div>"
        iterations_html += output_html
    
    # Create legend for name colors with color bar
    legend_html = "<h3>Person Color Legend</h3>"
    
    # Add color bar
    legend_html += create_color_bar(name_colors)
    
    # Add individual name colors with generation rank
    legend_html += "<div style='display: flex; flex-direction: column;'>"
    
    # Sort by color intensity (darkest to lightest - oldest to youngest)
    sorted_names = sorted(name_colors.items(), 
                         key=lambda x: sum(int(x[1][i:i+2], 16) for i in (1, 3, 5)))
    
    for i, (name, color) in enumerate(sorted_names):
        generation_rank = generation_ranks.get(name, i+1)
        legend_html += f"""
        <div style="margin: 5px 0; display: flex; align-items: center;">
            <div style="background-color: {color}; width: 20px; height: 20px; margin-right: 10px; border-radius: 3px;"></div>
            <span>{name} <small>(Generation {generation_rank})</small></span>
        </div>
        """
    
    legend_html += "</div>"
    
    return prompt_html, iterations_html, legend_html

def create_interface():
    """Create and launch the Gradio interface"""
    question_files = load_question_files()
    question_ids = sorted(question_files.keys())
    
    with gr.Blocks(title="Question Progress Visualization") as interface:
        gr.Markdown("# Question Progress Visualization")
        gr.Markdown("Visualize how model outputs change across iterations for improved questions")
        
        with gr.Row():
            with gr.Column(scale=1):
                question_dropdown = gr.Dropdown(
                    choices=question_ids, 
                    label="Select Question",
                    info="Choose a question to visualize"
                )
                
                with gr.Accordion("Color Legend", open=False):
                    legend_html = gr.HTML(label="Color Legend")
            
            with gr.Column(scale=3):
                prompt_html = gr.HTML(label="Question Prompt")
                iterations_html = gr.HTML(label="Iterations")
        
        question_dropdown.change(
            fn=visualize_question_progress,
            inputs=[question_dropdown],
            outputs=[prompt_html, iterations_html, legend_html]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=True) 