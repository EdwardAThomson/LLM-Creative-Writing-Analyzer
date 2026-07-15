# LLM Creative Writing Analyzer

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/EdwardAThomson)

![LLM Creative Writing Analyzer: the same prompt, run many times and measured for repetition and variation](images/llm_creative_writing_analyzer.webp)

A tool for testing and analyzing creative writing capabilities of various Large Language Models (LLMs).

## Overview

This project allows you to test multiple LLMs with the same creative writing prompt to analyze and compare their responses. The tool is specifically designed to:

1. Generate creative writing samples from various LLMs based on a set of parameters
2. Send the same prompt multiple times to analyze consistency and variation
3. Calculate similarity metrics between responses
4. Create comprehensive output files with the results
5. Re-analyze existing output files with new metrics as they become available

## YouTube - Findings Overviews

* 2025: [First benchmark run](https://www.youtube.com/watch?v=viOgoXF4MDQ)
* 2026: [Second benchmark run](https://www.youtube.com/watch?v=d0gkrymMI_w)

## Report & Findings

A written-up analysis lives in [`reports/report.md`](reports/report.md). It began
as a 2025 study of the then-current models (GPT-4o, o1/o3, Gemini 2.0/2.5, Claude
3.5/3.7) and now includes a **Longitudinal Update (2026)** (Section 6) that re-runs
the same 10×1500-word benchmark against the current generation via the local CLI
backends: Gemini 3 Flash/Pro, Claude Opus 4.8 / Sonnet, and Codex (`gpt-5.5`).

Headline finding: the original report's "Elara Phenomenon" — LLMs repeatedly
reaching for a tiny pool of character names — was a near-universal trait in 2025
but has **split sharply by vendor** in 2026:

- **Anthropic (Claude Opus 4.8)** has essentially eliminated it — a fresh cast of
  names almost every run.
- **Google (Gemini 3)** still draws from the same pool (Kaelen/Elara/Vance),
  largely unchanged since Gemini 2.0.
- **OpenAI (`gpt-5.5` via Codex)** shows the strongest name-pull of the cohort,
  having simply *rotated* its favourites (Aster/Mara/Venn) rather than diversifying.

The 2026 run outputs are under `results/gemini_cli_10x/`, `results/claude_cli_10x/`,
and `results/codex_cli_10x/`.

## Features

- Support for multiple LLM providers (OpenAI GPT, Google Gemini, and Anthropic Claude) via either hosted APIs or local agent CLIs (`codex`, `claude`, `gemini`)
- Configurable number of test repetitions per model
- **Configurable analysis steps:** Enable/disable text structure, semantic similarity, named entity, and detailed entity overlap analysis.
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

3. Set up your API keys (only needed for the API backends):
   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```
   `OPENROUTER_API_KEY` is optional: it is only used by the OpenRouter-proxied
   judge models (`openrouter-deepseek`, `openrouter-haiku`, or the
   `openrouter:<upstream-model-id>` passthrough), e.g. as `--judge-model` for
   the Narrative Dynamics benchmark.
   The local CLI backends (`codex-cli`, `claude-cli*`, `gemini-cli-*`) don't use
   these keys — they instead require the corresponding agent CLI (`codex`,
   `claude`, `gemini`) installed and authenticated on your `PATH`.

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
- **Analysis Options:** Checkboxes to selectively enable/disable advanced analysis steps.

### Command Line Interface

For automated testing or advanced users, a command-line interface is also available:

```bash
python llm_creative_tester.py
```

This will run the tester with default settings:
- Models: GPT-5.4 and Gemini 3.1 Pro Preview
- 3 repeats per model
- Parameters from `parameters.txt`
- Target word count: 500 words

### Command Line Options

```bash
python llm_creative_tester.py --models gpt-5.5 claude-opus-4-8 gemini-3.1-pro-preview codex-cli --repeats 5 --word-count 300
```

Available options:
- `--models`: Space-separated list of models to test
- `--repeats`: Number of times to repeat each test (default: 3)
- `--word-count`: Target word count for generated content (default: 500)
- `--params-file`: File containing story parameters (default: parameters.txt)
- `--output-dir`: Directory to store results (default: results)
- `--pause`: Seconds to pause between API calls (default: 1)

#### Analysis Flags (Optional)

By default, all advanced analyses (structure, semantic, entities, overlap) are performed. Use the following flags to **disable** specific analyses:

- `--no-structure`: Disable text structure analysis.
- `--no-semantic`: Disable semantic similarity analysis.
- `--no-entities`: Disable named entity analysis (this also disables entity overlap).
- `--no-entity-overlap`: Disable detailed entity overlap calculation (requires entity analysis to be enabled otherwise).

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
├── gpt-5.4_20260401_120145.txt
├── gemini-3.1-pro-preview_20260401_120145.txt
├── results_20260401_120145.json
└── summary_20260401_120145.txt
```

## Analysis Types

The tool provides several types of similarity measurements. These advanced analyses can be individually enabled or disabled via CLI flags or GUI checkboxes.

1. **Text Similarity**: Based on exact text matches using Python's SequenceMatcher (always performed).
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
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt
```

This command analyzes the specified file. By default, it performs basic text similarity and vocabulary analysis. Use the following flags to **include** specific advanced analyses:

```bash
# Include text structure analysis
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt --structure

# Include semantic similarity analysis
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt --semantic

# Include named entity analysis (including overlap calculation)
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt --entity

# Include all available advanced analyses
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt --all

# Save analysis results to a file
python -m utils.text_analysis results/gpt-5.4_20260401_120145.txt --all --output results/reanalysis_20260401.txt
```

This allows you to apply new analysis methods to existing results without needing to regenerate the responses. 

### Extended Metrics (v2)

The analyses above are the **v1** metric set — kept frozen so the longitudinal
2025/2026 study stays comparable over time. Newer metrics live in an opt-in,
modular library under `utils/metrics/` and are scored **without ever touching the
v1 results**: they write to a parallel *sidecar* file, leaving the original
`results_*.json` byte-identical. The current v2 set:

| Metric | What it measures |
|---|---|
| `phonetic_names` | character-name recurrence by *sound* (closes the "V-surname" finding exact matching misses) |
| `mtld` | length-robust lexical diversity (replaces length-sensitive TTR) |
| `ngram_diversity` | corpus-level distinct-n + Self-BLEU across the N runs |
| `opening_lines` | first-sentence similarity, lexical + semantic — the formulaic-opening tell |
| `burstiness` | sentence-length variation (rhythm); humans vary it, models flatten it |
| `dialogue_ratio` | share of text inside quotes (dialogue vs narration pacing) |
| `intra_text_repetition` | word/phrase overuse *within* a single story |
| `cliche_density` | cliché / stock-phrase + em-dash density (a generic-prose signal) |

Score them over any saved `results_*.json` — no regeneration or API calls needed:

```bash
# List available metrics
python -m utils.metrics --list

# Score one results file with the full v2 set -> writes <file>.metrics.json
python -m utils.metrics results/results_20250326_185105.json --benchmark v2

# Score a single metric or a comma-separated subset
python -m utils.metrics results/<file>.json --metrics mtld,burstiness

# (Re)score an entire directory — every results_*.json gets its own sidecar,
# sharing one model load. Use this whenever a metric or lexicon version changes.
python -m utils.metrics results/ --benchmark v2
```

Each run writes a **self-describing sidecar** (`<file>.metrics.json`) recording the
schema, benchmark version, and exact metric set. Benchmark "versions" are
cumulative, frozen manifests in `benchmarks/vN.yaml` (e.g. `v2` = the eight metrics
above); see [METRICS_ROADMAP.md](METRICS_ROADMAP.md) for the design rationale.

### Narrative Dynamics benchmark (scoring-only, arbitrary text)

A third benchmark measuring **long-range story structure** of a single
arbitrary-length text (a novel, a long generated run): per-unit **tension
trajectory** (0-10 anchored-rubric LLM judge; register, volatility, decile
table, peak, tail/ending mode), **block rhythm** (7-type per-paragraph prose
modes; words-per-mode-segment, interiority exit dynamics, secondary shading,
setting touch), and **thread architecture** (LLM cast extraction plus
deterministic majority-cast Jaccard clustering; runs, hand-offs, convergence,
tension deltas at switches). The rubrics are versioned artifacts ported from the
StoryDaemon masters studies, each with a provenance header carrying the
reliability numbers as measured there and a caveat that they must be re-verified
in this harness before findings are trusted.

There is no generation step, and the analysis pipeline runs at zero LLM spend in
`--dry-run` mode (tests use fakes throughout):

```bash
# Score a text file or a directory of *.txt/*.md (chapter detection by default,
# Gutenberg frontmatter auto-trimmed; --segmentation windows for ~1500-word units)
python -m benchmarks.narrative_dynamics path/to/book.txt
python -m benchmarks.narrative_dynamics corpus/ --make-reference masters_ref.json
python -m benchmarks.narrative_dynamics my_story.txt --reference masters_ref.json

# Long runs are resumable: successful judge calls go to a durable per-document
# cache, so re-running the same command continues where it stopped.
# --max-calls N pauses cleanly after N real judge calls; --no-cache opts out.
python -m benchmarks.narrative_dynamics big_book.txt --max-calls 200

# Cross-corpus aggregation of scored *.nd.json sidecars into tables + anomaly flags
python -m utils.metrics.aggregate_nd1 --input-dir scored/ --outdir tables/
```

Outputs per document: `<stem>.nd.json` + `<stem>.nd.txt` (inputs never touched).
Scoring-only mode also exists for the v2 metrics over raw text:
`python -m utils.metrics --text <file|dir> --benchmark v2`. Full docs:
[benchmarks/narrative_dynamics/README.md](benchmarks/narrative_dynamics/README.md).

### Single-Text benchmark (st1: the vN library over one book, zero LLM calls)

The `st1` series (`benchmarks/st1.yaml`) runs the shared metric library over ONE
arbitrary-length text: the document is segmented into units (chapters, or
~1500-word windows, reusing the narrative-dynamics segmentation layer) and the
units become the "runs". Same library as vN, different unit of account; fully
local (stdlib plus the already-required spaCy/sentence-transformers), no LLM
judge anywhere.

It combines the per-text-valid v2 metrics (MTLD, burstiness, dialogue ratio,
intra-text repetition, cliche density, distinct-n/Self-BLEU across chapters,
phonetic name inventory) with new single-text modules:

| Metric | What it measures |
|---|---|
| `text_structure` | per-unit paragraph/sentence/word profile (stdlib regex) |
| `self_similarity` | adjacent-unit similarity series + longest verbatim match; flags recycled/duplicated units (the defect class where a shipped novel carried ~9,200 verbatim chars across adjacent scenes at 0.668 similarity vs a ~0.02 baseline) |
| `opening_formula` | do the book's own chapters open alike (first-sentence similarity + first-words census) |
| `entity_census` | single-text cast census: cast size, recurring cast vs walk-ons, entity density, name-component inventory |

```bash
python -m utils.metrics --text book.txt --segment chapters --benchmark st1
python -m utils.metrics --text story.txt --segment windows --window-words 1500 --benchmark st1

# Cross-corpus aggregation of scored *.st1.metrics.json sidecars (plus the
# *.extract.json extraction sidecars) into a table + anomaly flags
python -m utils.metrics.aggregate_corpus --input-dir scored/ --extract-dir extracts/ --outdir tables/
```

Writes the usual self-describing sidecar (`book.metrics.json`) with the
segmentation recorded; the input file is never touched.


## Project Structure

```
.
├── llm_creative_tester.py     # Main script (CLI)
├── llm_tester_ui.py           # Graphical user interface
├── ai_helper.py               # LLM backend dispatch (API + CLI)
├── parameters.txt             # Example parameters
├── requirements.txt           # Python dependencies
├── cli_backends/              # Local agent-CLI backends (codex/claude/gemini)
│   ├── __init__.py            # Package initialization
│   ├── agent_cwd.py           # Shared neutral scratch cwd for CLI agents
│   ├── claude_cli_interface.py
│   ├── gemini_cli_interface.py
│   └── codex_interface.py
├── utils/                     # Utility modules
│   ├── __init__.py            # Package initialization
│   ├── llm_tester.py          # LLM testing functions
│   ├── prompt_io.py           # Prompt and I/O handling
│   ├── text_analysis.py       # v1 text analysis (frozen)
│   └── metrics/               # v2+ opt-in metric library (see "Extended Metrics")
│       ├── _base.py           # shared compute() contract + spaCy/embedding helpers
│       ├── _manifests.py      # benchmark version (vN.yaml) resolver
│       ├── _textmode.py       # raw-text input for scoring-only mode (--text)
│       ├── __main__.py        # retroactive scorer CLI (file, directory, or --text)
│       └── <metric>.py        # one module per metric
├── benchmarks/                # frozen metric manifests (v1/v2/st1/nd1.yaml) + benchmark packages
│   └── narrative_dynamics/    # long-range structure benchmark (scoring-only; own README)
├── tests/                     # pytest suite (pure modules + fakes; no LLM calls)
└── results/                   # Output: v1 JSON + v2 .metrics.json sidecars
```

## Supported Models

Currently supported models (defined in `llm_tester_ui.py`) fall into two groups.

**API backends** (require the matching API key in `.env`):
- OpenAI: gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.2
- Google: gemini-3.1-pro-preview, gemini-3.1-flash-preview, gemini-3-pro-preview, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash
- Anthropic: claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5

**Local CLI backends** (no API key — they shell out to an agent CLI installed on your `PATH`, in headless mode, run from a neutral scratch dir so the agent generates text instead of acting on a repo):
- `codex-cli` — GPT-5 family via the [Codex CLI](https://github.com/openai/codex); uses the Codex CLI's configured default model (e.g. `gpt-5.5` in `~/.codex/config.toml`)
- `claude-cli`, `claude-cli-opus`, `claude-cli-sonnet`, `claude-cli-haiku` — Claude via the [Claude Code CLI](https://github.com/anthropics/claude-code) (bare `claude-cli` uses the CLI's configured default model)
- `gemini-cli-pro`, `gemini-cli-flash` — Gemini via the [Gemini CLI](https://github.com/google-gemini/gemini-cli)

> **`codex-cli` on hardened Linux (e.g. Ubuntu 23.10+):** recent Codex sandboxes itself with a bundled bubblewrap that needs to create an unprivileged user namespace, which these distros block by default (`kernel.apparmor_restrict_unprivileged_userns=1`). Rather than weaken that host-wide, `codex_interface.py` detects the restriction and instead runs Codex inside an identity-mapped user namespace via `unshare`, using the setuid `newuidmap`/`newgidmap` helpers. Install them with `sudo apt install uidmap` (you also need `/etc/subuid` + `/etc/subgid` entries for your user, which the package sets up). The other CLI backends don't need this.

### Which backend should I use?

- **Prefer the API backends** (`gpt-*`, `gemini-*`, `claude-*`) for benchmarking runs: they take explicit model IDs and a fixed `temperature`, so results are the most reproducible and comparable across models.
- **Use the CLI backends** (`*-cli`) when you don't want to manage API keys or you're billing against an existing CLI subscription — they shell out to the local `codex`/`claude`/`gemini` tools instead. Note they forward less control (e.g. `max_tokens` isn't passed through) and `claude-cli`/`gemini-cli` use whatever the CLI is configured/authenticated for, so they're better for convenience than for tightly-controlled comparisons.
- **On hardened Linux**, `codex-cli` additionally needs `uidmap` installed (see the note above); the other CLI backends work out of the box once their CLI is on your `PATH`.

OpenAI GPT-5 models support a `reasoning_effort` parameter (none, low, medium, high, xhigh) which controls how much thinking the model does. This defaults to "high" for creative writing tasks.

To add support for additional models, modify the `ai_helper.py` file and update the `AVAILABLE_MODELS` list in `llm_tester_ui.py`. The CLI backends live in the `cli_backends/` package (ported from the StoryDaemon writing ecosystem).

## License

[MIT License](LICENSE)
