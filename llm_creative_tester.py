#!/usr/bin/env python3
# llm_creative_tester.py

import os
import argparse
from utils import (
    # Text analysis
    analyze_responses,
    calculate_advanced_metrics,
    
    # Prompt and I/O
    load_parameters,
    create_prompt,
    ensure_output_dir,
    generate_timestamp,
    write_model_results,
    write_json_results,
    write_summary,
    
    # LLM tester
    test_model,
    extract_response_texts,
    count_successful_responses
)

# Default settings
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_REPEATS = 3
DEFAULT_MODELS = ["gpt-4o", "gemini-1.5-pro"]
DEFAULT_WORD_COUNT = 1500
DEFAULT_PAUSE = 1  # seconds between API calls

def run_tests(models, parameters_text, repeats=3, word_count=500, output_dir=DEFAULT_OUTPUT_DIR, pause_seconds=DEFAULT_PAUSE):
    """Run tests against specified models with given parameters.
    
    Args:
        models: List of model names to test
        parameters_text: Text containing story parameters
        repeats: Number of times to repeat each test
        word_count: Target word count for generated content
        output_dir: Directory to store results
        pause_seconds: Seconds to pause between API calls
        
    Returns:
        Tuple of (results dictionary, timestamp string)
    """
    # Create the output directory if it doesn't exist
    ensure_output_dir(output_dir)
    
    # Create timestamp for this run
    timestamp = generate_timestamp()
    
    # Create the prompt
    prompt = create_prompt(parameters_text, word_count)
    
    results = {}
    
    for model in models:
        # Test the model
        model_responses = test_model(model, prompt, repeats, pause_seconds)
        
        # Extract just the response texts for analysis
        response_texts = extract_response_texts(model_responses)
        
        # Analyze the responses
        basic_analysis = analyze_responses(response_texts)
        advanced_analysis = calculate_advanced_metrics(response_texts)
        
        # Combine analyses
        analysis = {**basic_analysis, **advanced_analysis}
        
        # Add to results
        results[model] = {
            "responses": model_responses,
            "analysis": analysis
        }
        
        # Write model-specific results to a text file
        write_model_results(model, model_responses, analysis, prompt, timestamp, output_dir)
    
    # Write overall results to a JSON file
    write_json_results(results, timestamp, output_dir)
    
    # Create a summary file
    summary_file = write_summary(results, models, repeats, word_count, timestamp, output_dir)
    
    return results, timestamp

def main():
    parser = argparse.ArgumentParser(description="Test LLMs for creative writing abilities")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, 
                        help="List of models to test (e.g., gpt-4o gemini-1.5-pro)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Number of times to repeat each test")
    parser.add_argument("--word-count", type=int, default=DEFAULT_WORD_COUNT,
                        help="Target word count for generated content")
    parser.add_argument("--params-file", default="parameters.txt",
                        help="File containing story parameters")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to store results")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE,
                        help="Seconds to pause between API calls")
    
    args = parser.parse_args()
    
    try:
        # Load parameters
        parameters_text = load_parameters(args.params_file)
        
        # Run the tests
        results, timestamp = run_tests(
            models=args.models,
            parameters_text=parameters_text,
            repeats=args.repeats,
            word_count=args.word_count,
            output_dir=args.output_dir,
            pause_seconds=args.pause
        )
        
        print(f"\nTests completed successfully.")
        print(f"Results saved to {args.output_dir}/")
        print(f"Summary file: {args.output_dir}/summary_{timestamp}.txt")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 