#!/usr/bin/env python3
# utils/llm_tester.py

import time
from ai_helper import send_prompt

def test_model(model, prompt, repeats=3, pause_seconds=1):
    """Test a single model with a prompt multiple times.
    
    Args:
        model: The name of the model to test
        prompt: The prompt to send
        repeats: Number of times to repeat the test
        pause_seconds: Seconds to pause between API calls
        
    Returns:
        List of response dictionaries with response text and timing information
    """
    print(f"\n\nTesting model: {model}")
    model_responses = []
    
    for i in range(repeats):
        print(f"  Running test {i+1}/{repeats}...")
        
        try:
            # Send the prompt to the model
            start_time = time.time()
            response = send_prompt(prompt, model=model)
            end_time = time.time()
            
            # Save the response
            model_responses.append({
                "response_text": response,
                "time_taken": end_time - start_time
            })
            
            print(f"  Test {i+1} complete - {len(response.split())} words generated in {end_time - start_time:.2f} seconds")
            
            # Brief pause between API calls
            time.sleep(pause_seconds)
            
        except Exception as e:
            print(f"  Error in test {i+1}: {str(e)}")
            model_responses.append({
                "error": str(e)
            })
    
    return model_responses

def extract_response_texts(responses):
    """Extract just the text from a list of response dictionaries.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        List of response text strings
    """
    return [r.get("response_text", "") for r in responses if "response_text" in r]

def count_successful_responses(responses):
    """Count the number of successful responses.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        Number of successful responses (those without errors)
    """
    return sum(1 for r in responses if "response_text" in r)

def calculate_average_response_time(responses):
    """Calculate the average response time for successful responses.
    
    Args:
        responses: List of response dictionaries
        
    Returns:
        Average response time in seconds
    """
    times = [r.get("time_taken", 0) for r in responses if "time_taken" in r]
    return sum(times) / len(times) if times else 0 