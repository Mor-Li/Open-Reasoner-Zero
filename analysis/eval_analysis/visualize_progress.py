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
    
    # Create a colormap from oldest (red) to youngest (blue)
    colors = {}
    
    # If we have the names in order of generation (eldest first)
    for i, name in enumerate(names):
        # Calculate hue based on position in the list (red for eldest, blue for youngest)
        # Hue values in HSV: 0 = red, 0.33 = green, 0.66 = blue, 1 = red again
        hue = 0.0 if i == 0 else 0.6 * (i / (len(names) - 1 or 1))
        
        # Saturation and value remain constant for better visibility
        saturation = 0.8
        value = 0.9
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB to hex color string
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r*255), int(g*255), int(b*255)
        )
        colors[name] = hex_color
    
    return colors

def highlight_names_in_text(text: str, name_colors: Dict[str, str]) -> str:
    """
    Highlight person names in text with background colors
    
    Args:
        text: The text to highlight
        name_colors: Dictionary mapping names to color hex codes
        
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
            # Create HTML span with background color
            highlighted = f'<span style="background-color: {name_colors[name]}; padding: 0px 2px; border-radius: 3px;">{name}</span>'
            # Replace the name with the highlighted version
            text = text.replace(name, highlighted)
    
    return text

def format_iteration_output(iteration_data: Dict[str, Any], 
                           name_colors: Dict[str, str]) -> str:
    """
    Format the output of a single iteration with highlighted names
    
    Args:
        iteration_data: Data for a single iteration
        name_colors: Dictionary mapping names to color hex codes
        
    Returns:
        Formatted HTML string with highlighted names
    """
    output = iteration_data.get("output", "")
    is_correct = iteration_data.get("iscorrect", False)
    final_answer = iteration_data.get("final_answer", "")
    
    highlighted_output = highlight_names_in_text(output, name_colors)
    
    # Format as HTML with correct/incorrect indication
    status_color = "#4CAF50" if is_correct else "#F44336"  # Green if correct, red if incorrect
    
    html = f"""
    <div style="border: 1px solid {status_color}; border-radius: 5px; padding: 10px; margin-bottom: 15px;">
        <div style="font-weight: bold; color: {status_color}; margin-bottom: 5px;">
            {'✓ Correct' if is_correct else '✗ Incorrect'}
        </div>
        <div style="white-space: pre-wrap; font-family: monospace;">
            {highlighted_output}
        </div>
    """
    
    if final_answer:
        highlighted_final = highlight_names_in_text(final_answer, name_colors)
        html += f"""
        <div style="margin-top: 10px; border-top: 1px dotted #ccc; padding-top: 5px;">
            <b>Final Answer:</b> {highlighted_final}
        </div>
        """
    
    html += "</div>"
    return html

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
    
    # Format prompt with highlighted names
    highlighted_prompt = highlight_names_in_text(prompt, name_colors)
    prompt_html = f"""
    <div style="white-space: pre-wrap; font-family: monospace; margin-bottom: 20px;">
        {highlighted_prompt}
    </div>
    <div style="margin-bottom: 20px;">
        <b>Reference Answer:</b> {highlight_names_in_text(reference_answer, name_colors)}
    </div>
    """
    
    # Build HTML for each iteration
    iterations_html = "<h3>Iterations</h3>"
    sorted_iterations = sorted([int(i) for i in iterations.keys()])
    
    for iter_num in sorted_iterations:
        iter_data = iterations.get(str(iter_num), {})
        iterations_html += f"<h4>Iteration {iter_num}</h4>"
        iterations_html += format_iteration_output(iter_data, name_colors)
    
    # Create legend for name colors
    legend_html = "<h3>Person Color Legend</h3>"
    legend_html += "<div style='display: flex; flex-direction: column;'>"
    
    for name, color in name_colors.items():
        legend_html += f"""
        <div style="margin: 5px 0;">
            <span style="background-color: {color}; padding: 2px 5px; border-radius: 3px;">{name}</span>
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