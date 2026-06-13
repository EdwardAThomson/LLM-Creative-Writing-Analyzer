# Roadmap — LLM Creative Writing Analyzer

_Status: active · updated 2026-05-31_

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

## Next

- [ ] Keep the available-model list current as new LLMs ship
- [ ] Deeper randomness / statistical analysis of output variation
- [ ] Modular metrics framework — keep v1 frozen; add opt-in v2+ metric modules in `utils/metrics/`, cumulative version manifests (`benchmarks/vN.yaml`), and write v2+ results to a separate sidecar so v1 artifacts stay byte-identical (design: [METRICS_ROADMAP.md](METRICS_ROADMAP.md))
- [ ] Metrics — diversity/fingerprint upgrades: phonetic name clustering (closes report §6.6), length-robust lexical diversity (MTLD), Self-BLEU / distinct-n, embedding-cloud modality, opening-line diversity, NER-aware name filtering
- [ ] Metrics — per-text craft: sentence-length burstiness, readability spread, dialogue ratio, sensory density, intra-text repetition, cliché / "AI-slop" lexicon
- [ ] Metrics — prompt-adherence scoring: fraction of the prompt's parameters honored (tense, POV, requested species/tech)
- [ ] Metrics — LLM-as-judge quality axis: rubric scoring + pairwise Elo, genericness via perplexity, judge-bias controls

## Backlog

- [ ] Name-generation bypass experiment (mitigate the name-reuse problem)
- [ ] Integrate findings into the NovelWriter / StoryDaemon generation pipeline
- [ ] Human copy-editing comparison pipeline
