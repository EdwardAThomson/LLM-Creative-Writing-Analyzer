#!/usr/bin/env python3
# utils/prompt_io.py

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_parameters(params_file="parameters.txt"):
    """Load the creative writing parameters from file.
    
    Args:
        params_file: Path to the parameters file
        
    Returns:
        The content of the parameters file as a string
        
    Raises:
        FileNotFoundError: If the parameters file doesn't exist
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameters file '{params_file}' not found.")
    
    with open(params_file, "r") as f:
        params_text = f.read()
    
    return params_text

def create_prompt(parameters_text, word_count=1500, system_prompt_file=None):
    """Create a complete prompt for the LLM based on parameters and word count,
    optionally including a system prompt from a file.
    
    Args:
        parameters_text: The text containing the story parameters
        word_count: Target word count for the generated content
        system_prompt_file: Path to a file containing an additional system prompt/context (optional)
        
    Returns:
        A formatted prompt string ready to be sent to the LLM
    """
    
    system_prompt_content = ""
    if system_prompt_file and os.path.exists(system_prompt_file):
        try:
            with open(system_prompt_file, 'r') as f:
                system_prompt_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read system prompt file {system_prompt_file}: {e}")
            system_prompt_content = f"[Error reading system prompt file: {system_prompt_file}]"
    elif system_prompt_file:
        print(f"Warning: System prompt file {system_prompt_file} not found.")
        system_prompt_content = f"[System prompt file not found: {system_prompt_file}]"

    # Construct the main prompt
    prompt_parts = [
        f"You are a professional science fiction author. Write the engaging opening section of a novel using \n"
        f"the parameters below. The opening should hook the reader, establish the setting, and introduce \n"
        f"at least one main character.\n"
    ]

    # Add the system prompt content if it exists
    if system_prompt_content:
        prompt_parts.append(f"\nADDITIONAL SYSTEM CONTEXT / PERSONA:\n{'='*30}\n{system_prompt_content}\n{'='*30}\n\n")

    # Add the rest of the original prompt
    prompt_parts.extend([
        f"\nYour writing should be original, creative, and high-quality. Aim for approximately {word_count} words.\n",
        f"\nPARAMETERS:\n{parameters_text}",
        f"\nWrite the opening section now:\n"
    ])

    prompt = "".join(prompt_parts)
    return prompt

def ensure_output_dir(output_dir):
    """Ensure the output directory exists.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        The path object for the directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def generate_timestamp():
    """Generate a timestamp string for the current time.
    
    Returns:
        A formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def write_model_results(model, responses, analysis, prompt, timestamp, output_dir):
    """Write the results for a specific model to a text file.
    
    Args:
        model: The name of the model
        responses: List of response dictionaries
        analysis: Analysis results dictionary
        prompt: The prompt used
        timestamp: Timestamp string
        output_dir: Output directory path
    """
    model_output_file = os.path.join(output_dir, f"{model}_{timestamp}.txt")
    with open(model_output_file, "w") as f:
        f.write(f"MODEL: {model}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"RESPONSES ({len(responses)}):\n\n")
        
        for i, resp in enumerate(responses):
            f.write(f"RESPONSE {i+1}:\n")
            if "error" in resp:
                f.write(f"ERROR: {resp['error']}\n\n")
            else:
                f.write(f"Time taken: {resp['time_taken']:.2f} seconds\n")
                f.write(f"Word count: {len(resp['response_text'].split())}\n\n")
                f.write(f"{resp['response_text']}\n\n")
                f.write("-" * 80 + "\n\n")
        
        f.write("\nANALYSIS:\n")
        
        # Basic similarity metrics
        f.write("TEXT SIMILARITY (based on exact text matches):\n")
        f.write(f"  Average similarity: {analysis['average_similarity']:.4f}\n")
        f.write(f"  Median similarity: {analysis['median_similarity']:.4f}\n")
        f.write(f"  Min similarity: {analysis['min_similarity']:.4f}\n")
        f.write(f"  Max similarity: {analysis['max_similarity']:.4f}\n")
        f.write(f"  Average word count: {analysis['average_word_count']:.2f}\n\n")
        
        # Vocabulary metrics
        f.write("VOCABULARY METRICS:\n")
        f.write(f"  Vocabulary diversity: {analysis.get('vocabulary_diversity', 'N/A'):.4f}\n")
        f.write(f"  Unique words: {analysis.get('unique_word_count', 'N/A')}\n")
        f.write(f"  Total words: {analysis.get('total_word_count', 'N/A')}\n\n")
        
        # Text structure metrics if available
        if 'structure_metrics' in analysis:
            f.write("TEXT STRUCTURE METRICS:\n")
            struct = analysis['structure_metrics'].get('aggregate', {})
            
            # Paragraph metrics
            if 'paragraph_count' in struct:
                para_metrics = struct['paragraph_count']
                f.write(f"  Average paragraphs per response: {para_metrics.get('mean', 'N/A'):.2f}\n")
                
            if 'paragraphs_per_1000_words' in struct:
                para_density = struct['paragraphs_per_1000_words']
                f.write(f"  Paragraphs per 1000 words: {para_density.get('mean', 'N/A'):.2f}\n")
                f.write(f"  Paragraph density range: {para_density.get('min', 'N/A'):.2f} - {para_density.get('max', 'N/A'):.2f}\n")
            
            # Sentence metrics
            if 'avg_sentences_per_paragraph' in struct:
                sent_per_para = struct['avg_sentences_per_paragraph']
                f.write(f"  Average sentences per paragraph: {sent_per_para.get('mean', 'N/A'):.2f}\n")
                f.write(f"  Sentences per paragraph range: {sent_per_para.get('min', 'N/A'):.2f} - {sent_per_para.get('max', 'N/A'):.2f}\n")
            
            # Word metrics
            if 'avg_words_per_sentence' in struct:
                words_per_sent = struct['avg_words_per_sentence']
                f.write(f"  Average words per sentence: {words_per_sent.get('mean', 'N/A'):.2f}\n")
                f.write(f"  Words per sentence range: {words_per_sent.get('min', 'N/A'):.2f} - {words_per_sent.get('max', 'N/A'):.2f}\n")
                
            if 'avg_words_per_paragraph' in struct:
                words_per_para = struct['avg_words_per_paragraph']
                f.write(f"  Average words per paragraph: {words_per_para.get('mean', 'N/A'):.2f}\n\n")
        
        # Semantic similarity analysis if available
        if 'semantic_similarity' in analysis:
            sem = analysis['semantic_similarity']
            f.write("SEMANTIC SIMILARITY (based on meaning):\n")
            f.write(f"  Average semantic similarity: {sem.get('average', 'N/A'):.4f}\n")
            f.write(f"  Median semantic similarity: {sem.get('median', 'N/A'):.4f}\n")
            f.write(f"  Min semantic similarity: {sem.get('min', 'N/A'):.4f}\n")
            f.write(f"  Max semantic similarity: {sem.get('max', 'N/A'):.4f}\n\n")
        elif 'semantic_error' in analysis:
            f.write(f"SEMANTIC SIMILARITY: Not available - {analysis['semantic_error']}\n\n")
        
        # Named entity analysis if available
        if 'entity_analysis' in analysis:
            ent = analysis['entity_analysis']
            print(f"DEBUG (write_results): Type of ent (entity_analysis): {type(ent)}") # DEBUG
            f.write("NAMED ENTITY ANALYSIS:\n")
            # Add checks before accessing potentially missing keys with default values
            f.write(f"  Total entities detected: {ent.get('total_entities', 'N/A')}\n")
            f.write(f"  Unique entities: {ent.get('unique_entities', 'N/A')}\n")
            
            # Entity types summary
            entity_types_data = ent.get('entity_types')
            print(f"DEBUG (write_results): Type of entity_types_data: {type(entity_types_data)}") # DEBUG
            if isinstance(entity_types_data, dict):
                f.write("\n  ENTITY TYPES:\n")
                for entity_type, count in entity_types_data.items():
                    f.write(f"    {entity_type}: {count}\n")
            
            # Entity similarity
            ent_sim = ent.get('entity_similarity')
            print(f"DEBUG (write_results): Type of ent_sim (entity_similarity): {type(ent_sim)}") # DEBUG
            if isinstance(ent_sim, dict):
                f.write(f"\n  ENTITY SIMILARITY:\n")
                f.write(f"    Average entity overlap: {ent_sim.get('average', 'N/A'):.4f}\n")
                f.write(f"    Max entity overlap: {ent_sim.get('max', 'N/A'):.4f}\n")
                
                # If there are repeated entities, list them
                repeated_entities_data = ent.get('repeated_entities')
                print(f"DEBUG (write_results): Type of repeated_entities_data: {type(repeated_entities_data)}") # DEBUG
                if isinstance(repeated_entities_data, list) and repeated_entities_data:
                    f.write("\n  REPEATED ENTITIES (appearing in multiple responses):\n")
                    # Group by entity type
                    repeated_by_type = defaultdict(list)
                    # Add check for dict type within the list
                    for entity in repeated_entities_data:
                        if isinstance(entity, dict) and 'type' in entity and 'text' in entity:
                            repeated_by_type[entity['type']].append(entity['text'])
                        else:
                            print(f"DEBUG (write_results): Skipping non-dict or malformed item in repeated_entities_data: {entity}") # DEBUG
                        
                    for entity_type, entities in repeated_by_type.items():
                        f.write(f"    {entity_type}:\n")
                        for entity in entities:
                            f.write(f"      - {entity}\n")
                
                # Name component analysis
                name_comp = ent.get('name_components')
                print(f"DEBUG (write_results): Type of name_comp (name_components): {type(name_comp)}") # DEBUG
                if isinstance(name_comp, dict):
                    f.write(f"\n  NAME COMPONENT ANALYSIS:\n")
                    f.write(f"    Total name components: {name_comp.get('total', 'N/A')}\n")
                    f.write(f"    Unique name components: {name_comp.get('unique', 'N/A')}\n")
                    
                    name_sim = name_comp.get('similarity')
                    print(f"DEBUG (write_results): Type of name_sim (name_components[similarity]): {type(name_sim)}") # DEBUG
                    if isinstance(name_sim, dict):
                        f.write(f"    Average name component overlap: {name_sim.get('average', 'N/A'):.4f}\n")
                        f.write(f"    Max name component overlap: {name_sim.get('max', 'N/A'):.4f}\n")
                    
                    # If there are repeated name components, list them
                    repeated_name_comp_data = name_comp.get('repeated')
                    print(f"DEBUG (write_results): Type of repeated_name_comp_data: {type(repeated_name_comp_data)}") # DEBUG
                    if isinstance(repeated_name_comp_data, list) and repeated_name_comp_data:
                        f.write("\n  REPEATED NAME COMPONENTS (appearing across multiple responses):\n")
                        # Sort by number of responses the component appears in (descending)
                        try:
                            # Add check for dict type within the list during sort key access
                            sorted_components = sorted(
                                [item for item in repeated_name_comp_data if isinstance(item, dict)], # Filter for dicts first
                                key=lambda x: (x.get('response_count', 0), x.get('total_count', 0)), 
                                reverse=True)
                            
                            for comp in sorted_components:
                                # Use .get() for safer access
                                response_count = comp.get('response_count', 'N/A')
                                total_count = comp.get('total_count', 'N/A')
                                text = comp.get('text', '{Missing Text}')
                                f.write(f"    - {text} (appears in {response_count}/{len(responses)} responses, {total_count} total occurrences)\n")
                        except TypeError as sort_err:
                             print(f"DEBUG (write_results): TypeError during sorting/processing repeated name components: {sort_err}") # DEBUG
                             f.write("    Error processing repeated name components.\n")
                
                # Detailed entity overlap
                # Check if detailed_overlap exists, is a list (not the 'Not Calculated' string), and is not empty
                detailed_overlap_data = ent_sim.get('detailed_overlap')
                if isinstance(detailed_overlap_data, list) and detailed_overlap_data:
                    f.write("\n  DETAILED ENTITY OVERLAP:\n")
                    for comparison in detailed_overlap_data:
                        f.write(f"    Responses {comparison['responses']}:\n")
                        
                        # For each entity type
                        for entity_type, details in comparison['overlap'].items():
                            if details['shared_entities']:
                                f.write(f"      {entity_type} (similarity: {details['similarity']:.4f}):\n")
                                for entity in details['shared_entities']:
                                    f.write(f"        - {entity}\n")
            
        elif 'entity_error' in analysis:
            f.write(f"NAMED ENTITY ANALYSIS: Not available - {analysis['entity_error']}\n")

def write_json_results(results, timestamp, output_dir):
    """Write overall results to a JSON file.
    
    Args:
        results: Dictionary containing all results
        timestamp: Timestamp string
        output_dir: Output directory path
    """
    json_output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    # Process the results to convert any non-serializable objects
    try:
        with open(json_output_file, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    except TypeError as e:
        # If we still have serialization issues, try more aggressive conversion
        print(f"Warning: {e}. Attempting more aggressive type conversion...")
        # Create a deep copy with all numpy and problematic types converted
        sanitized_results = sanitize_for_json(results)
        with open(json_output_file, "w") as f:
            json.dump(sanitized_results, f, indent=2)

def sanitize_for_json(obj):
    """Convert a complex object with potential non-JSON serializable types to JSON serializable types.
    
    Args:
        obj: The object to sanitize
        
    Returns:
        A JSON serializable version of the object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(i) for i in obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        try:
            # Try to convert to a simple type
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # If we can't serialize it, convert to string
            return str(obj)

def count_successful_responses(responses):
    """Count the number of successful responses (without errors).
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        Number of successful responses
    """
    return sum(1 for resp in responses if "error" not in resp)

def write_summary(results, models, repeats, word_count, timestamp, output_dir):
    """Write a summary of all results to a text file.
    
    Args:
        results: Dictionary containing all results
        models: List of tested models
        repeats: Number of repeats per model
        word_count: Target word count
        timestamp: Timestamp string
        output_dir: Output directory path
    """
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write(f"CREATIVE WRITING TEST SUMMARY\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Models tested: {', '.join(models)}\n")
        f.write(f"Repeats per model: {repeats}\n")
        f.write(f"Target word count: {word_count}\n\n")
        
        f.write("RESULTS SUMMARY:\n\n")
        
        for model in models:
            if model in results:
                analysis = results[model]["analysis"]
                f.write(f"MODEL: {model}\n")
                
                # Basic similarity
                f.write(f"  Text Similarity (exact matches):\n")
                f.write(f"    Average similarity: {analysis['average_similarity']:.4f}\n")
                f.write(f"    Median similarity: {analysis['median_similarity']:.4f}\n")
                
                # Semantic similarity if available
                if 'semantic_similarity' in analysis:
                    sem = analysis['semantic_similarity']
                    f.write(f"  Semantic Similarity (meaning):\n")
                    f.write(f"    Average: {sem.get('average', 'N/A'):.4f}\n")
                    f.write(f"    Median: {sem.get('median', 'N/A'):.4f}\n")
                
                # Text structure metrics if available
                if 'structure_metrics' in analysis:
                    struct = analysis['structure_metrics'].get('aggregate', {})
                    f.write(f"  Text Structure:\n")
                    
                    if 'paragraphs_per_1000_words' in struct:
                        para_density = struct['paragraphs_per_1000_words']
                        f.write(f"    Paragraphs per 1000 words: {para_density.get('mean', 'N/A'):.2f}\n")
                    
                    if 'avg_sentences_per_paragraph' in struct:
                        sent_per_para = struct['avg_sentences_per_paragraph']
                        f.write(f"    Sentences per paragraph: {sent_per_para.get('mean', 'N/A'):.2f}\n")
                    
                    if 'avg_words_per_sentence' in struct:
                        words_per_sent = struct['avg_words_per_sentence']
                        f.write(f"    Words per sentence: {words_per_sent.get('mean', 'N/A'):.2f}\n")
                
                # Entity similarity if available
                if 'entity_analysis' in analysis and 'entity_similarity' in analysis['entity_analysis']:
                    ent_sim = analysis['entity_analysis']['entity_similarity']
                    f.write(f"  Entity Similarity (names, places, etc):\n")
                    f.write(f"    Average overlap: {ent_sim.get('average', 'N/A'):.4f}\n")
                    
                    # Count repeated entities if available
                    if 'repeated_entities' in analysis['entity_analysis']:
                        repeated = analysis['entity_analysis']['repeated_entities']
                        f.write(f"    Repeated entities: {len(repeated)}\n")
                    
                    # Name component analysis if available
                    if 'name_components' in analysis['entity_analysis']:
                        name_comp = analysis['entity_analysis']['name_components']
                        
                        # Count repeated name components if available
                        if 'repeated' in name_comp:
                            repeated_names = name_comp['repeated']
                            # Only count components that appear in multiple responses
                            cross_response_names = [comp for comp in repeated_names if comp.get('response_count', 0) > 1]
                            if cross_response_names:
                                f.write(f"    Repeated name components: {len(cross_response_names)}\n")
                                
                                # Sort by number of responses the component appears in (descending)
                                sorted_components = sorted(cross_response_names, 
                                                        key=lambda x: x.get('response_count', 0), 
                                                        reverse=True)
                                
                                # Format the shared names with response counts
                                shared_names = []
                                for comp in sorted_components[:5]:
                                    shared_names.append(f"{comp['text']} ({comp.get('response_count', 0)}/{repeats})")
                                
                                f.write(f"    Shared name parts: {', '.join(shared_names)}")
                                if len(sorted_components) > 5:
                                    f.write(f" and {len(sorted_components) - 5} more")
                                f.write("\n")
                
                f.write(f"  Average word count: {analysis['average_word_count']:.2f}\n")
                
                # Count successful responses
                successful = count_successful_responses(results[model]["responses"])
                f.write(f"  Successful responses: {successful}/{repeats}\n\n")
    
    return summary_file 