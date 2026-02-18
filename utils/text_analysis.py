#!/usr/bin/env python3
# utils/text_analysis.py

from difflib import SequenceMatcher
import statistics
import re
from collections import Counter, defaultdict
import sys
import spacy
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def calculate_similarity(text1, text2):
    """Calculate similarity between two text responses using SequenceMatcher."""
    return SequenceMatcher(None, text1, text2).ratio()

def analyze_responses(responses):
    """Analyze a list of responses for similarity and other metrics.
    
    Args:
        responses: List of text responses to analyze
        
    Returns:
        Dictionary containing various similarity metrics and word counts
    """
    if len(responses) < 2:
        return {
            "similarity_scores": [],
            "average_similarity": 0,
            "median_similarity": 0,
            "min_similarity": 0,
            "max_similarity": 0,
            "word_counts": [len(r.split()) for r in responses],
            "average_word_count": statistics.mean([len(r.split()) for r in responses]) if responses else 0
        }
    
    similarity_scores = []
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            similarity_scores.append(calculate_similarity(responses[i], responses[j]))
    
    return {
        "similarity_scores": similarity_scores,
        "average_similarity": statistics.mean(similarity_scores) if similarity_scores else 0,
        "median_similarity": statistics.median(similarity_scores) if similarity_scores else 0,
        "min_similarity": min(similarity_scores) if similarity_scores else 0,
        "max_similarity": max(similarity_scores) if similarity_scores else 0,
        "word_counts": [len(r.split()) for r in responses],
        "average_word_count": statistics.mean([len(r.split()) for r in responses]) if responses else 0
    }

def get_word_count(text):
    """Get the number of words in a text."""
    return len(text.split())

def analyze_text_structure(responses):
    """Analyze the structural characteristics of text responses.
    
    Args:
        responses: List of text responses to analyze
        
    Returns:
        Dictionary containing structural metrics like paragraph counts, 
        sentences per paragraph, and words per sentence
    """
    if not responses:
        return {"structure_metrics": "No responses to analyze"}
    
    # Initialize metrics collections
    all_metrics = []
    
    for response in responses:
        # Skip empty responses
        if not response.strip():
            continue
            
        # Count paragraphs (split by double newlines or single newlines)
        paragraphs = re.split(r'\n\s*\n|\n', response.strip())
        paragraphs = [p for p in paragraphs if p.strip()]  # Remove empty paragraphs
        paragraph_count = len(paragraphs)
        
        # Calculate paragraphs per 1000 words
        word_count = len(response.split())
        paragraphs_per_1000 = (paragraph_count / word_count) * 1000 if word_count > 0 else 0
        
        # Count sentences and words per paragraph
        paragraph_sentence_counts = []
        paragraph_word_counts = []
        sentence_word_counts = []
        
        for paragraph in paragraphs:
            # Split into sentences (basic split by ., !, ?)
            sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
            sentences = [s for s in sentences if s.strip()]  # Remove empty sentences
            
            paragraph_sentence_counts.append(len(sentences))
            paragraph_word_counts.append(len(paragraph.split()))
            
            # Count words per sentence
            for sentence in sentences:
                words = len(sentence.split())
                if words > 0:  # Only count non-empty sentences
                    sentence_word_counts.append(words)
        
        # Calculate averages
        avg_sentences_per_paragraph = statistics.mean(paragraph_sentence_counts) if paragraph_sentence_counts else 0
        avg_words_per_paragraph = statistics.mean(paragraph_word_counts) if paragraph_word_counts else 0
        avg_words_per_sentence = statistics.mean(sentence_word_counts) if sentence_word_counts else 0
        
        # Store metrics for this response
        response_metrics = {
            "paragraph_count": paragraph_count,
            "paragraphs_per_1000_words": float(paragraphs_per_1000),
            "avg_sentences_per_paragraph": float(avg_sentences_per_paragraph),
            "avg_words_per_paragraph": float(avg_words_per_paragraph),
            "avg_words_per_sentence": float(avg_words_per_sentence),
            "word_count": word_count
        }
        
        all_metrics.append(response_metrics)
    
    # Calculate aggregate metrics across all responses
    aggregate_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics]
            aggregate_metrics[key] = {
                "mean": float(statistics.mean(values)),
                "median": float(statistics.median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0
            }
    
    return {
        "structure_metrics": {
            "individual_responses": all_metrics,
            "aggregate": aggregate_metrics
        }
    }

def calculate_advanced_metrics(
    responses, 
    run_structure_analysis=True, 
    run_semantic_analysis=True, 
    run_entity_analysis=True,
    run_entity_overlap_calculation=True
):
    """Calculate more advanced metrics on the responses.
    
    Args:
        responses: List of text responses
        run_structure_analysis (bool): Whether to perform text structure analysis
        run_semantic_analysis (bool): Whether to perform semantic similarity analysis
        run_entity_analysis (bool): Whether to perform named entity analysis
        run_entity_overlap_calculation (bool): Whether to calculate detailed entity overlap (requires run_entity_analysis=True)
        
    Returns:
        Dictionary with additional metrics beyond basic similarity
    """
    # This could be extended with more sophisticated metrics like:
    # - Vocabulary diversity (unique words / total words)
    # - Sentiment analysis
    # - Named entity recognition
    # - etc.
    
    if not responses:
        return {"advanced_metrics": "No responses to analyze"}
    
    # For now, just calculate vocabulary diversity as an example
    all_words = []
    unique_words = set()
    
    for response in responses:
        words = response.lower().split()
        all_words.extend(words)
        unique_words.update(words)
    
    vocabulary_diversity = len(unique_words) / len(all_words) if all_words else 0
    
    # Initialize result dictionaries
    entity_metrics = {}
    semantic_metrics = {}
    structure_metrics = {}
    
    # Try to calculate named entity metrics if requested and spaCy is available
    if run_entity_analysis:
        print("DEBUG (calc_adv): Calling analyze_named_entities...") # DEBUG
        try:
            entity_metrics = analyze_named_entities(responses, run_entity_overlap_calculation)
            print(f"DEBUG (calc_adv): Returned from analyze_named_entities. Type: {type(entity_metrics)}") # DEBUG
        except ImportError:
            entity_metrics = {"entity_error": "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"}
            print("DEBUG (calc_adv): analyze_named_entities raised ImportError") # DEBUG
        except Exception as e:
            entity_metrics = {"entity_error": f"Error analyzing named entities: {str(e)}"}
            print(f"DEBUG (calc_adv): analyze_named_entities raised Exception: {e}") # DEBUG
    else:
        print("DEBUG (calc_adv): Skipping analyze_named_entities.") # DEBUG
    
    # Try to calculate semantic similarity if requested and sentence-transformers is available
    if run_semantic_analysis:
        print("DEBUG (calc_adv): Calling calculate_semantic_similarities...") # DEBUG
        try:
            semantic_metrics = calculate_semantic_similarities(responses)
            print(f"DEBUG (calc_adv): Returned from calculate_semantic_similarities. Type: {type(semantic_metrics)}") # DEBUG
        except ImportError:
            semantic_metrics = {"semantic_error": "sentence-transformers not installed. Run: pip install sentence-transformers"}
            print("DEBUG (calc_adv): calculate_semantic_similarities raised ImportError") # DEBUG
        except Exception as e:
            semantic_metrics = {"semantic_error": f"Error calculating semantic similarity: {str(e)}"}
            print(f"DEBUG (calc_adv): calculate_semantic_similarities raised Exception: {e}") # DEBUG
    else:
        print("DEBUG (calc_adv): Skipping calculate_semantic_similarities.") # DEBUG
    
    # Calculate structure metrics if requested
    if run_structure_analysis:
        print("DEBUG (calc_adv): Calling analyze_text_structure...") # DEBUG
        # Add try-except block here for better debugging
        try:
            structure_metrics = analyze_text_structure(responses)
            print(f"DEBUG (calc_adv): Returned from analyze_text_structure. Type: {type(structure_metrics)}") # DEBUG
        except Exception as e:
            print(f"DEBUG (calc_adv): analyze_text_structure raised Exception: {e}") # DEBUG
            # Assign an error dictionary if it fails, similar to other analyses
            structure_metrics = {"structure_error": f"Error analyzing text structure: {str(e)}"}
    else:
        print("DEBUG (calc_adv): Skipping analyze_text_structure.") # DEBUG

    print("DEBUG (calc_adv): Combining results...") # DEBUG
    return {
        "vocabulary_diversity": vocabulary_diversity,
        "unique_word_count": len(unique_words),
        "total_word_count": len(all_words),
        **entity_metrics,
        **semantic_metrics,
        **structure_metrics
    }

def analyze_named_entities(responses, calculate_overlap=True):
    """Analyze named entities across responses to find similarities.
    
    Args:
        responses: List of text responses to analyze
        calculate_overlap (bool): Whether to calculate detailed entity overlap
        
    Returns:
        Dictionary containing named entity analysis
    """
    try:
        
        
        # Load the English model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not downloaded, try to download it
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                           check=True, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
    except ImportError:
        return {"entity_error": "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"}
    except Exception as e:
        return {"entity_error": f"Error loading spaCy model: {str(e)}"}
    
    # Process each response and extract entities
    all_entities = []
    response_entities = []
    
    # Track name components separately
    all_name_components = []
    response_name_components = []
    
    # Track which responses each name component appears in
    name_component_responses = defaultdict(set)
    entity_responses = defaultdict(set)
    
    for i, response in enumerate(responses):
        doc = nlp(response)
        entities = defaultdict(list)
        name_components = defaultdict(list)
        
        # Track unique components in this response to avoid double-counting
        unique_components_in_response = set()
        unique_entities_in_response = set()
        
        for ent in doc.ents:
            # Convert to string to ensure JSON serialization works
            label = str(ent.label_)
            text = str(ent.text)
            entities[label].append(text)
            
            # Track which response this entity appears in
            entity_key = (label, text)
            all_entities.append(entity_key)
            unique_entities_in_response.add(entity_key)
            
            # Extract name components for PERSON entities
            if label == "PERSON":
                # Split names into components and store them
                components = text.split()
                for component in components:
                    if len(component) > 1:  # Ignore single-character components
                        name_components["NAME_PART"].append(component)
                        comp_key = ("NAME_PART", component)
                        all_name_components.append(comp_key)
                        unique_components_in_response.add(comp_key)
        
        # Record which response each unique component appears in
        for comp in unique_components_in_response:
            name_component_responses[comp].add(i)
            
        # Record which response each unique entity appears in
        for entity in unique_entities_in_response:
            entity_responses[entity].add(i)
        
        response_entities.append(entities)
        response_name_components.append(name_components)
    
    # Count total occurrences for all components and entities
    entity_counts = Counter(all_entities)
    name_component_counts = Counter(all_name_components)
    
    # Find entities and components that appear in MULTIPLE RESPONSES
    repeated_entities = [{"type": str(entity[0]), "text": str(entity[1]), "response_count": len(responses_set)} 
                         for entity, responses_set in entity_responses.items() 
                         if len(responses_set) > 1]
    
    repeated_name_components = [{"type": "NAME_PART", "text": str(component[1]), "response_count": len(responses_set), "total_count": name_component_counts[component]} 
                               for component, responses_set in name_component_responses.items() 
                               if len(responses_set) > 1]
    
    # Group entities by type
    entity_types = defaultdict(list)
    for label, text in all_entities:
        entity_types[label].append(text)
    
    # Make sure everything is serializable
    serializable_entity_types = {str(label): len(entities) for label, entities in entity_types.items()}
    
    # Compare entities between responses if requested
    entity_similarities = []
    entity_overlap_details = []
    if calculate_overlap:
        for i in range(len(response_entities)):
            for j in range(i+1, len(response_entities)):
                # Calculate similarity based on shared entities
                overlap = calculate_entity_overlap(response_entities[i], response_entities[j])
                entity_similarities.append(float(overlap["similarity"]))
                entity_overlap_details.append({
                    "responses": f"{i+1} vs {j+1}",
                    "overlap": overlap["details"]
                })
    
    # Compare name components between responses if requested
    name_component_similarities = []
    name_component_overlap_details = []
    if calculate_overlap:
        for i in range(len(response_name_components)):
            for j in range(i+1, len(response_name_components)):
                # Calculate similarity based on shared name components
                overlap = calculate_entity_overlap(response_name_components[i], response_name_components[j])
                name_component_similarities.append(float(overlap["similarity"]))
                name_component_overlap_details.append({
                    "responses": f"{i+1} vs {j+1}",
                    "overlap": overlap["details"]
                })
    
    avg_similarity = 0
    max_similarity = 0
    if entity_similarities:
        avg_similarity = float(statistics.mean(entity_similarities))
        max_similarity = float(max(entity_similarities))
    
    avg_name_similarity = 0
    max_name_similarity = 0
    if name_component_similarities:
        avg_name_similarity = float(statistics.mean(name_component_similarities))
        max_name_similarity = float(max(name_component_similarities))
    
    return {
        "entity_analysis": {
            "total_entities": len(all_entities),
            "unique_entities": len(entity_responses),
            "entity_types": serializable_entity_types,
            "repeated_entities": repeated_entities,
            "entity_similarity": {
                "average": avg_similarity,
                "max": max_similarity,
                "detailed_overlap": entity_overlap_details if calculate_overlap else "Not Calculated"
            },
            "name_components": {
                "total": len(all_name_components),
                "unique": len(name_component_responses),
                "repeated": repeated_name_components,
                "similarity": {
                    "average": avg_name_similarity,
                    "max": max_name_similarity,
                    "detailed_overlap": name_component_overlap_details if calculate_overlap else "Not Calculated"
                }
            }
        }
    }

def calculate_entity_overlap(entities1, entities2):
    """Calculate the overlap between named entities in two texts.
    
    Args:
        entities1: Dictionary of entities from the first text by entity type
        entities2: Dictionary of entities from the second text by entity type
        
    Returns:
        Dictionary containing similarity score and detailed overlap
    """
    overlap_details = {}
    total_overlap = 0
    total_entities = 0
    
    # Get all entity types present in either text
    all_types = set(entities1.keys()) | set(entities2.keys())
    
    for entity_type in all_types:
        set1 = set(entities1.get(entity_type, []))
        set2 = set(entities2.get(entity_type, []))
        
        # Calculate Jaccard similarity for this entity type
        intersection = set1 & set2
        union = set1 | set2
        
        if union:
            type_similarity = len(intersection) / len(union)
        else:
            type_similarity = 0
            
        overlap_details[str(entity_type)] = {
            "similarity": float(type_similarity),
            "shared_entities": list(intersection),
            "count1": len(set1),
            "count2": len(set2)
        }
        
        # Add to totals for overall similarity
        total_overlap += len(intersection)
        total_entities += len(union)
    
    # Overall similarity (Jaccard index of all entities)
    similarity = total_overlap / total_entities if total_entities > 0 else 0
    
    return {
        "similarity": float(similarity),
        "details": overlap_details
    }

def calculate_semantic_similarities(responses):
    """Calculate semantic similarity between responses using embeddings.
    
    Args:
        responses: List of text responses
        
    Returns:
        Dictionary with semantic similarity metrics
    """
    if len(responses) < 2:
        return {"semantic_similarity": "Needs at least 2 responses"}
    
    try:
        
        # Load model (first time will download it)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all responses
        embeddings = model.encode(responses)
        
        # Calculate pairwise cosine similarities
        semantic_scores = []
        
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                # Convert numpy float32 to Python float
                semantic_scores.append(float(sim))
        
        # Convert all numpy values to Python native types
        return {
            "semantic_similarity": {
                "average": float(statistics.mean(semantic_scores)),
                "median": float(statistics.median(semantic_scores)),
                "min": float(min(semantic_scores)),
                "max": float(max(semantic_scores)),
                "scores": [float(score) for score in semantic_scores]
            }
        }
    except ImportError:
        return {"semantic_error": "Required libraries not installed. Run: pip install sentence-transformers scikit-learn"}
    except Exception as e:
        return {"semantic_error": f"Error calculating semantic similarity: {str(e)}"}

def extract_responses_from_file(file_path):
    """Extract response texts from an existing output file.
    
    Args:
        file_path: Path to the output file
        
    Returns:
        List of response texts
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find all responses using regex pattern
        # Look for "RESPONSE N:" followed by text up until another "RESPONSE" or "ANALYSIS"
        response_pattern = r"RESPONSE \d+:\n(?:Time taken:.*?\n)?(?:Word count:.*?\n\n)(.*?)(?=(?:\n-{80}\n\n)|(?:\nANALYSIS:))"
        responses = re.findall(response_pattern, content, re.DOTALL)
        
        # Clean up responses by removing trailing whitespace
        cleaned_responses = [resp.strip() for resp in responses]
        
        return cleaned_responses
    except Exception as e:
        print(f"Error extracting responses: {e}")
        return []

def print_analysis_results(analysis):
    """Print analysis results in a formatted way.
    
    Args:
        analysis: Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80 + "\n")
    
    # Basic similarity metrics
    print("TEXT SIMILARITY:")
    print(f"  Average similarity: {format_float_or_str(analysis.get('average_similarity', 'N/A'))}")
    print(f"  Median similarity: {format_float_or_str(analysis.get('median_similarity', 'N/A'))}")
    print(f"  Range: {format_float_or_str(analysis.get('min_similarity', 'N/A'))} - {format_float_or_str(analysis.get('max_similarity', 'N/A'))}")
    print(f"  Average word count: {format_float_or_str(analysis.get('average_word_count', 'N/A'), '.2f')}\n")
    
    # Vocabulary metrics
    print("VOCABULARY METRICS:")
    print(f"  Vocabulary diversity: {format_float_or_str(analysis.get('vocabulary_diversity', 'N/A'))}")
    print(f"  Unique words: {analysis.get('unique_word_count', 'N/A')}")
    print(f"  Total words: {analysis.get('total_word_count', 'N/A')}\n")
    
    # Text structure metrics
    if 'structure_metrics' in analysis:
        print("TEXT STRUCTURE METRICS:")
        struct = analysis['structure_metrics'].get('aggregate', {})
        
        if 'paragraph_count' in struct:
            para_metrics = struct['paragraph_count']
            print(f"  Average paragraphs per response: {format_float_or_str(para_metrics.get('mean', 'N/A'), '.2f')}")
            
        if 'paragraphs_per_1000_words' in struct:
            para_density = struct['paragraphs_per_1000_words']
            print(f"  Paragraphs per 1000 words: {format_float_or_str(para_density.get('mean', 'N/A'), '.2f')}")
            print(f"  Paragraph density range: {format_float_or_str(para_density.get('min', 'N/A'), '.2f')} - {format_float_or_str(para_density.get('max', 'N/A'), '.2f')}")
        
        if 'avg_sentences_per_paragraph' in struct:
            sent_per_para = struct['avg_sentences_per_paragraph']
            print(f"  Average sentences per paragraph: {format_float_or_str(sent_per_para.get('mean', 'N/A'), '.2f')}")
            print(f"  Range: {format_float_or_str(sent_per_para.get('min', 'N/A'), '.2f')} - {format_float_or_str(sent_per_para.get('max', 'N/A'), '.2f')}")
        
        if 'avg_words_per_sentence' in struct:
            words_per_sent = struct['avg_words_per_sentence']
            print(f"  Average words per sentence: {format_float_or_str(words_per_sent.get('mean', 'N/A'), '.2f')}")
            print(f"  Range: {format_float_or_str(words_per_sent.get('min', 'N/A'), '.2f')} - {format_float_or_str(words_per_sent.get('max', 'N/A'), '.2f')}")
            
        if 'avg_words_per_paragraph' in struct:
            words_per_para = struct['avg_words_per_paragraph']
            print(f"  Average words per paragraph: {format_float_or_str(words_per_para.get('mean', 'N/A'), '.2f')}\n")
    
    # Semantic similarity
    if 'semantic_similarity' in analysis:
        sem = analysis['semantic_similarity']
        print("SEMANTIC SIMILARITY:")
        print(f"  Average: {format_float_or_str(sem.get('average', 'N/A'))}")
        print(f"  Median: {format_float_or_str(sem.get('median', 'N/A'))}")
        print(f"  Range: {format_float_or_str(sem.get('min', 'N/A'))} - {format_float_or_str(sem.get('max', 'N/A'))}\n")
    elif 'semantic_error' in analysis:
        print(f"SEMANTIC SIMILARITY: Not available - {analysis['semantic_error']}\n")
    
    # Entity analysis
    if 'entity_analysis' in analysis:
        ent = analysis['entity_analysis']
        print("NAMED ENTITY ANALYSIS:")
        print(f"  Total entities: {ent.get('total_entities', 'N/A')}")
        print(f"  Unique entities: {ent.get('unique_entities', 'N/A')}")
        
        # Entity types
        if 'entity_types' in ent:
            print("\n  ENTITY TYPES:")
            for entity_type, count in ent['entity_types'].items():
                print(f"    {entity_type}: {count}")
                
        # Entity similarity
        if 'entity_similarity' in ent:
            ent_sim = ent['entity_similarity']
            print(f"\n  ENTITY SIMILARITY:")
            print(f"    Average overlap: {format_float_or_str(ent_sim.get('average', 'N/A'))}")
            print(f"    Max overlap: {format_float_or_str(ent_sim.get('max', 'N/A'))}")
            
        # Repeated entities
        if 'repeated_entities' in ent:
            repeated = ent['repeated_entities']
            if repeated:
                print("\n  REPEATED ENTITIES:")
                for entity in sorted(repeated, key=lambda x: x.get('response_count', 0), reverse=True):
                    print(f"    - {entity['text']} ({entity['type']}): appears in {entity.get('response_count', '?')} responses")
                
        # Name components
        if 'name_components' in ent:
            name_comp = ent['name_components']
            print(f"\n  NAME COMPONENT ANALYSIS:")
            print(f"    Total components: {name_comp.get('total', 'N/A')}")
            print(f"    Unique components: {name_comp.get('unique', 'N/A')}")
            
            if 'repeated' in name_comp:
                repeated_names = [comp for comp in name_comp['repeated'] if comp.get('response_count', 0) > 1]
                if repeated_names:
                    print("\n    REPEATED NAME COMPONENTS:")
                    for comp in sorted(repeated_names, key=lambda x: x.get('response_count', 0), reverse=True):
                        print(f"      - {comp['text']}: appears in {comp.get('response_count', '?')} responses, {comp.get('total_count', '?')} total occurrences")
    elif 'entity_error' in analysis:
        print(f"ENTITY ANALYSIS: Not available - {analysis['entity_error']}")

def format_float_or_str(value, format_spec='.4f'):
    """Format a value as float if possible, otherwise return as is.
    
    Args:
        value: The value to format
        format_spec: Format specification for float values
        
    Returns:
        Formatted string
    """
    if isinstance(value, (int, float)):
        return f"{value:{format_spec}}"
    else:
        return value

def main():
    """Main function to run analysis on existing output files."""
    parser = argparse.ArgumentParser(description='Analyze LLM output files for various metrics.')
    parser.add_argument('file', help='Path to the output file to analyze')
    parser.add_argument('--text-only', action='store_true', help='Only run basic text similarity analysis')
    parser.add_argument('--semantic', action='store_true', help='Include semantic similarity analysis')
    parser.add_argument('--entity', action='store_true', help='Include named entity analysis')
    parser.add_argument('--structure', action='store_true', help='Include text structure analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--output', help='Path to save results as a text file')
    
    args = parser.parse_args()
    
    # Extract responses from file
    print(f"Extracting responses from {args.file}...")
    responses = extract_responses_from_file(args.file)
    
    if not responses:
        print("No valid responses found in the file. Please check the file format.")
        return
    
    print(f"Found {len(responses)} responses.")
    
    # Run analysis
    print("Running analysis...")
    
    # Basic text similarity analysis (always run)
    analysis = analyze_responses(responses)
    
    print("Calculating vocabulary diversity...")
    # We call calculate_advanced_metrics here mainly for vocabulary metrics,
    # but pass flags so it *could* run other analyses if needed (though we run them separately below).
    # This structure is a bit redundant now but preserves the original logic flow.
    # We specifically disable entity overlap calculation here as it's handled conditionally later.
    adv_metrics = calculate_advanced_metrics(
        responses, 
        run_structure_analysis=args.structure or args.all,
        run_semantic_analysis=args.semantic or args.all,
        run_entity_analysis=args.entity or args.all,
        run_entity_overlap_calculation=False # Handled below
    )
    # Only take vocabulary metrics from this call
    if 'vocabulary_diversity' in adv_metrics:
        analysis['vocabulary_diversity'] = adv_metrics['vocabulary_diversity']
    if 'unique_word_count' in adv_metrics:
        analysis['unique_word_count'] = adv_metrics['unique_word_count']
    if 'total_word_count' in adv_metrics:
        analysis['total_word_count'] = adv_metrics['total_word_count']
    
    # Determine which analyses to run based on flags
    run_all = args.all or (not args.text_only and not args.semantic and not args.entity and not args.structure)
    run_structure = run_all or args.structure
    run_semantic = run_all or args.semantic
    run_entity = run_all or args.entity
    # For standalone analysis, let's assume if --entity is passed, we want overlap unless explicitly disabled.
    # We don't have a separate --no-entity-overlap flag here, so entity implies overlap.
    run_overlap = run_entity 
    
    if run_structure:
        # Text structure analysis
        print("Analyzing text structure...")
        structure_metrics = analyze_text_structure(responses)
        analysis.update(structure_metrics)
    
    if run_semantic:
        # Semantic similarity analysis
        print("Analyzing semantic similarity...")
        try:
            semantic_metrics = calculate_semantic_similarities(responses)
            analysis.update(semantic_metrics)
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            analysis["semantic_error"] = str(e)
    
    if run_entity:
        # Named entity analysis
        print("Analyzing named entities...")
        try:
            # Pass the overlap flag determined above
            entity_metrics = analyze_named_entities(responses, calculate_overlap=run_overlap)
            analysis.update(entity_metrics)
        except Exception as e:
            print(f"Error in entity analysis: {e}")
            analysis["entity_error"] = str(e)
    
    # Print results
    print_analysis_results(analysis)
    
    # Save results to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                # Redirect print output to file
                import io
                from contextlib import redirect_stdout
                
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    print_analysis_results(analysis)
                
                f.write(buffer.getvalue())
                print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to file: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 