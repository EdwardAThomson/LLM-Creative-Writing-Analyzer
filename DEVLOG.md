# Dev Log

A chronological narrative of development. Newest entries first.

## 2026-06-19

Capstone day for the v2 metrics effort: wired up batch scoring and turned the
results into reports. Added a directory mode to the metrics CLI so the whole
`results/` corpus can be scored in one pass (reusing a single `ctx` across files
to load spaCy/embeddings once), and stored v2 baseline sidecars for the full
corpus — including a backfill sidecar for the older claude-3-5-sonnet
(2025-03-26) run. On top of those numbers came a stack of docs: a writeup of the
v2 metrics library and the corpus-scoring workflow, a v2 metrics report, a
"2026 newest-models" cohort report, a combined v1+v2 master fingerprint table,
and a "best available model" pick appended to the cohort report.

**Decisions & notes:** v2 stays strictly additive — directory scoring writes
separate `results_*.metrics.json` sidecars and never touches the frozen v1
`analysis` data, preserving the longitudinal series.

## 2026-06-18

Two more v2 metrics landed: `opening_lines` (first-sentence diversity, to catch
models that reach for the same hook every run) and `ngram_diversity` (distinct-n
plus Self-BLEU for measuring repetition across the N repeats).

## 2026-06-16

Added `cliche_density` with a frozen lexicon (v1) for flagging stock phrasing,
plus `dialogue_ratio` and `intra_text_repetition` metrics. Each is its own
module under `utils/metrics/`, following the one-file-per-metric contract.

**Decisions & notes:** the cliche lexicon is versioned (`LEXICON_VERSION`) so
changes are traceable and trigger a corpus re-score rather than silently shifting
historical scores.

## 2026-06-15

Added the sentence-length burstiness metric — variance in sentence length as a
stylistic fingerprint axis, distinguishing uniform "flat" prose from more varied
rhythm.

## 2026-06-14

Added the MTLD metric (lexical diversity that's robust to text length) and
introduced the benchmark version manifests — frozen, cumulative `benchmarks/vN.yaml`
files using `extends:` + `add:`, resolved by `_manifests.py`, so a named
benchmark version pins an exact set of metrics.

## 2026-06-13

Stood up the modular metrics library scaffold (`utils/metrics/`) — the shared
`compute(responses, ctx) -> dict` contract, `_base.py` helpers, and the first
metric, `phonetic_names`. This is the foundation for the v2 metric set, kept
deliberately separate from the untouched v1 analysis in `text_analysis.py`.
