# Roadmap — LLM Creative Writing Analyzer

_Status: active · updated 2026-06-19_

A Python tool for empirically benchmarking the creative-writing behavior of LLMs
(OpenAI, Gemini, Claude) — runs repeated identical prompts and analyzes the
outputs for consistency, variation, entity reuse, and structure to surface model
"fingerprints" (name reuse, repetition, laziness). Part of the author's NovelWriter
/ StoryDaemon writing ecosystem.

## Shipped

- [x] Multi-model testing (OpenAI, Gemini, Claude) via both command-line and Tkinter GUI
- [x] Configurable runs (repeat count, target word count, inter-call pause, custom system prompt)
- [x] Text similarity analysis (SequenceMatcher)
- [x] Semantic similarity (sentence-transformers embeddings)
- [x] Named-entity detection + overlap (spaCy)
- [x] Name-component analysis (surname / first-name reuse across generations)
- [x] Text-structure metrics (paragraphs, sentences, words, density)
- [x] Selective analysis flags (`--no-structure`, `--no-semantic`, `--no-entities`, `--no-entity-overlap`)
- [x] Standalone re-analysis of existing output (`python -m utils.text_analysis`)
- [x] Output formats (per-model reports, JSON raw data, summary file)
- [x] Hardening pass (thread-safe Tkinter, lazy imports, command-injection fix, API-key validation)
- [x] Local CLI backends (codex / claude / gemini in headless mode, no API key) ported from StoryDaemon
- [x] Refreshed model list (OpenAI GPT-5.x, Gemini 2.5/3/3.1, Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5)
- [x] Modular v2 metrics library (`utils/metrics/`) — opt-in metric modules, separate sidecar output, cumulative `benchmarks/vN.yaml` manifests; v1 frozen + byte-identical. Retroactive scorer CLI (`python -m utils.metrics <file|dir> --benchmark v2`, directory mode for batch (re)scoring) (design: [METRICS_ROADMAP.md](METRICS_ROADMAP.md))
- [x] v2 metric set (8): phonetic name clustering (closes report §6.6), MTLD, Self-BLEU/distinct-n, opening-line diversity, sentence-length burstiness, dialogue ratio, intra-text repetition, cliché/AI-slop density — scored over the full 2025/2026 corpus (sidecars committed under `results/`)

## Next

- [ ] Keep the available-model list current as new LLMs ship
- [ ] Deeper randomness / statistical analysis of output variation
- [ ] Metrics — remaining diversity/craft: embedding-cloud modality, NER-aware name filtering (formalize the stop-list + hyphen handling), readability spread, concrete/sensory-word density
- [ ] Metrics — prompt-adherence scoring: fraction of the prompt's parameters honored (tense, POV, requested species/tech)
- [ ] Metrics — LLM-as-judge quality axis: rubric scoring + pairwise Elo, genericness via perplexity, judge-bias controls
- [ ] Dev infra — make `utils/__init__.py` lazy: it eagerly imports `utils.text_analysis` (→ `sentence_transformers`), so `import utils.metrics.<name>` pulls heavy deps and fails in a minimal env. This defeats the metrics package's deliberate stdlib-only/lazy design. The first test suite works around it by importing metric modules **by file path** (`tests/conftest.py::load_metric`); once `utils/__init__.py` is lazy, tests can import normally. Note: coverage of the metric modules must be measured as `--cov=utils/metrics` (path form), not `--cov=utils.metrics`, until then.

## Backlog

- [ ] Name-generation bypass experiment (mitigate the name-reuse problem)
- [ ] Integrate findings into the NovelWriter / StoryDaemon generation pipeline
- [ ] Human copy-editing comparison pipeline
