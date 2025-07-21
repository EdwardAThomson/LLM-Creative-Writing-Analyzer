# LLM Creative Writing Analyzer

A tool for testing and analyzing creative writing capabilities of various Large Language Models (LLMs).

## Overview

This project allows you to test multiple LLMs with the same creative writing prompt to analyze and compare their responses. The tool is specifically designed to:

1. Generate creative writing samples from various LLMs based on a set of parameters
2. Send the same prompt multiple times to analyze consistency and variation
3. Calculate similarity metrics between responses
4. Create comprehensive output files with the results
5. Re-analyze existing output files with new metrics as they become available

## Features

- Support for multiple LLM providers (OpenAI GPT models, Google Gemini models, and Anthropic Claude models)
- Configurable number of test repetitions per model
- Advanced similarity analysis between responses:
  - Text-based similarity (exact matches)
  - Semantic similarity (meaning-based comparison)
  - Named entity detection and comparison
  - Name component analysis (detects when name parts like surnames appear across different texts)
  - Text structure analysis (paragraphs, sentences, and words metrics)
- Comprehensive output reports in both text and JSON formats
- Command-line interface with various configuration options
- Graphical user interface for easier configuration and testing
- Standalone reanalysis tool for existing output files

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/EdwardAThomson/LLM-Creative-Writing-Analyzer
   cd llm-creative-tester
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm  # For named entity detection
   ```

3. Set up your API keys:
   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Graphical User Interface

The easiest way to use this tool is through the graphical interface:

```bash
python llm_tester_ui.py
```

This will open a user-friendly interface with the following features:
- Checkbox selection for models to test
- Input fields for test parameters
- A tab to view and edit the parameters file
- Real-time output display
- Buttons to save/load parameters and view results

### Command Line Interface

For automated testing or advanced users, a command-line interface is also available:

```bash
python llm_creative_tester.py
```

This will run the tester with default settings:
- Models: GPT-4o and Gemini 1.5 Pro
- 3 repeats per model
- Parameters from `parameters.txt`
- Target word count: 500 words

### Command Line Options

```bash
python llm_creative_tester.py --models gpt-4o claude-3-sonnet gemini-1.5-pro --repeats 5 --word-count 300
```

Available options:
- `--models`: Space-separated list of models to test
- `--repeats`: Number of times to repeat each test (default: 3)
- `--word-count`: Target word count for generated content (default: 500)
- `--params-file`: File containing story parameters (default: parameters.txt)
- `--output-dir`: Directory to store results (default: results)
- `--pause`: Seconds to pause between API calls (default: 1)

### Parameters File

The parameters file should contain key-value pairs that define the characteristics of the creative writing sample. See the included `parameters.txt` for an example.

## Output

The script generates several output files in the specified output directory:

1. Model-specific text files with detailed responses and metrics, including:
   - Text similarity analysis
   - Vocabulary metrics
   - Semantic similarity analysis (meaning-based comparison)
   - Named entity analysis (detection of characters, locations, etc.)
   - Name component analysis (detection of shared name parts across texts)
2. A JSON file with all raw data
3. A summary text file with key metrics across all models

Example output structure:

```
results/
├── gpt-4o_20230601_120145.txt
├── gemini-1.5-pro_20230601_120145.txt
├── results_20230601_120145.json
└── summary_20230601_120145.txt
```

## Analysis Types

The tool provides several types of similarity measurements:

1. **Text Similarity**: Based on exact text matches using Python's SequenceMatcher.
2. **Semantic Similarity**: Measures similarity in meaning using embeddings (sentence-transformers).
3. **Entity Analysis**: Detects and compares named entities (people, places, organizations).
4. **Name Component Analysis**: A specialized analysis that breaks down person names into components to detect when parts of names (like surnames) appear across different responses, even if the full names differ.
5. **Text Structure Analysis**: Examines the structural characteristics of the generated text, including:
   - Paragraph density (paragraphs per 1000 words)
   - Sentences per paragraph
   - Words per sentence
   - Words per paragraph

The name component analysis is particularly useful for creative writing, as it can detect patterns like the model reusing surnames or first names across different generations, even when the full character names differ.

The text structure analysis helps identify stylistic patterns in how different models organize their writing, which can reveal unique "fingerprints" of different LLMs even when content varies.

### Standalone Analysis

The tool provides a way to re-analyze existing output files without having to regenerate them, which is especially useful when new analysis metrics are added.

### Using the Standalone Analysis Tool

To re-analyze an existing output file:

```bash
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt
```

By default, this will run all available analyses. You can also specify which analyses to run:

```bash
# Run only basic text similarity analysis
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --text-only

# Run text similarity and text structure analysis
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --structure

# Run text similarity and semantic similarity
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --semantic

# Run text similarity and entity analysis
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --entity

# Run all analyses
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --all

# Save analysis results to a file
python -m utils.text_analysis results/gpt-4o_20240601_120145.txt --output results/reanalysis_20240601.txt
```

This allows you to apply new analysis methods to existing results without needing to regenerate the responses. 


## Project Structure

```
.
├── llm_creative_tester.py     # Main script (CLI)
├── llm_tester_ui.py           # Graphical user interface
├── ai_helper.py               # LLM API handling
├── parameters.txt             # Example parameters
├── requirements.txt           # Python dependencies
├── utils/                     # Utility modules
│   ├── __init__.py            # Package initialization
│   ├── llm_tester.py          # LLM testing functions
│   ├── prompt_io.py           # Prompt and I/O handling
│   └── text_analysis.py       # Text analysis functions
└── results/                   # Default output directory
```

## Supported Models

Currently supported models include:
- OpenAI: gpt-4o, o1, o1-mini
- Google: gemini-1.5-pro, gemini-2.0-pro-exp-02-05, gemini-2.5-pro-exp-03-25
- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet,claude-3-7-sonnet 

To add support for additional models, modify the `ai_helper.py` file.

## License

[MIT License](LICENSE)
