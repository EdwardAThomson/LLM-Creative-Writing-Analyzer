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

def run_tests(models, parameters_text, repeats=3, word_count=500, 
              output_dir=DEFAULT_OUTPUT_DIR, pause_seconds=DEFAULT_PAUSE,
              run_structure=True, run_semantic=True, run_entities=True, run_entity_overlap=True,
              system_prompt_file=None):
    """Run tests against specified models with given parameters.
    
    Args:
        models: List of model names to test
        parameters_text: Text containing story parameters
        repeats: Number of times to repeat each test
        word_count: Target word count for generated content
        output_dir: Directory to store results
        pause_seconds: Seconds to pause between API calls
        run_structure (bool): Run text structure analysis
        run_semantic (bool): Run semantic similarity analysis
        run_entities (bool): Run named entity analysis
        run_entity_overlap (bool): Run detailed entity overlap calculation
        system_prompt_file (str, optional): Path to a file containing an additional system prompt.
        
    Returns:
        Tuple of (results dictionary, timestamp string)
    """
    # Create the output directory if it doesn't exist
    ensure_output_dir(output_dir)
    
    # Create timestamp for this run
    timestamp = generate_timestamp()
    
    # Create the prompt, potentially including the system prompt
    prompt = create_prompt(parameters_text, word_count, system_prompt_file=system_prompt_file)
    
    results = {}
    
    for model in models:
        # Test the model
        model_responses = test_model(model, prompt, repeats, pause_seconds)
        
        # Extract just the response texts for analysis
        response_texts = extract_response_texts(model_responses)
        
        # --- DEBUGGING START ---
        print(f"DEBUG: Analyzing {len(response_texts)} responses for model {model}...")
        print(f"DEBUG: Type of response_texts: {type(response_texts)}")
        if response_texts:
            print(f"DEBUG: Type of first response: {type(response_texts[0])}")
        # --- DEBUGGING END ---

        # Analyze the responses
        print("DEBUG: Calling analyze_responses...")
        basic_analysis = analyze_responses(response_texts)
        print(f"DEBUG: Returned from analyze_responses. Type: {type(basic_analysis)}")
        print(f"DEBUG: Basic Analysis Keys: {list(basic_analysis.keys()) if isinstance(basic_analysis, dict) else 'Not a dict'}")

        print("DEBUG: Calling calculate_advanced_metrics...")
        advanced_analysis = calculate_advanced_metrics(
            response_texts,
            run_structure_analysis=run_structure, 
            run_semantic_analysis=run_semantic, 
            run_entity_analysis=run_entities,
            run_entity_overlap_calculation=run_entity_overlap
        )
        print(f"DEBUG: Returned from calculate_advanced_metrics. Type: {type(advanced_analysis)}")
        print(f"DEBUG: Advanced Analysis Keys: {list(advanced_analysis.keys()) if isinstance(advanced_analysis, dict) else 'Not a dict'}")
        
        # Combine analyses
        analysis = {**basic_analysis, **advanced_analysis}
        print(f"DEBUG: Combined analysis created. Type: {type(analysis)}")
        print(f"DEBUG: Combined Analysis Keys: {list(analysis.keys()) if isinstance(analysis, dict) else 'Not a dict'}")
        
        # Add to results
        results[model] = {
            "responses": model_responses,
            "analysis": analysis
        }
        
        # Write model-specific results to a text file
        write_model_results(model, model_responses, analysis, prompt, timestamp, output_dir)
        print(f"DEBUG: Finished writing results for model {model}. Proceeding...")
    
    # Write overall results to a JSON file
    print("DEBUG: Calling write_json_results...")
    write_json_results(results, timestamp, output_dir)
    print("DEBUG: Returned from write_json_results.")
    
    # Create a summary file
    print("DEBUG: Calling write_summary...")
    summary_file = write_summary(results, models, repeats, word_count, timestamp, output_dir)
    print("DEBUG: Returned from write_summary.")
    
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
    
    # Add argument for optional system prompt file
    parser.add_argument("--system-prompt", default=None,
                        help="Path to a file containing an additional system prompt/context")
    
    # Add arguments for optional analyses (default to True, use action='store_false' to disable)
    parser.add_argument("--no-structure", action="store_false", dest="run_structure",
                        help="Disable text structure analysis")
    parser.add_argument("--no-semantic", action="store_false", dest="run_semantic",
                        help="Disable semantic similarity analysis")
    parser.add_argument("--no-entities", action="store_false", dest="run_entities",
                        help="Disable named entity analysis (implies --no-entity-overlap)")
    parser.add_argument("--no-entity-overlap", action="store_false", dest="run_entity_overlap",
                        help="Disable detailed entity overlap calculation")
    
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
            pause_seconds=args.pause,
            run_structure=args.run_structure,
            run_semantic=args.run_semantic,
            run_entities=args.run_entities,
            run_entity_overlap=args.run_entity_overlap and args.run_entities, # Overlap requires entity analysis
            system_prompt_file=args.system_prompt # Pass the system prompt file path
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