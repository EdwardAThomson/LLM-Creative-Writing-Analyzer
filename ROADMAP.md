# Roadmap — LLM Creative Writing Analyzer

_Status: active · updated 2026-05-31_

A Python tool for empirically benchmarking the creative-writing behavior of LLMs
(OpenAI, Gemini, Claude) — runs repeated identical prompts and analyzes the
outputs for consistency, variation, entity reuse, and structure to surface model
"fingerprints" (name reuse, repetition, laziness). Part of the author's NovelWriter
/ StoryDaemon writing ecosystem.

## Shipped

- [x] Multi-model testing (OpenAI, Gemini, Claude) via both CLI and Tkinter GUI
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

## Next

- [ ] Keep the available-model list current as new LLMs ship
- [ ] Deeper randomness / statistical analysis of output variation

## Backlog

- [ ] Name-generation bypass experiment (mitigate the name-reuse problem)
- [ ] Integrate findings into the NovelWriter / StoryDaemon generation pipeline
- [ ] Human copy-editing comparison pipeline
