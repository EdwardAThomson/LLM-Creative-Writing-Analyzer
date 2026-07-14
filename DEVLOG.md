# Dev Log

A chronological narrative of development. Newest entries first.

## 2026-07-13

Continued the previous day's push to make the two new benchmarks (`nd1` and
`st1`) usable at corpus scale, splitting the day between extraction/scoring
hygiene and three larger pieces of infrastructure. On the extraction side, a
run of fixes tightened what counts as "story" across real Gutenberg books:
inline-title TOC chapter lists are now screened out generally (fixing Tale of
Two Cities), the trailing printer's colophon is trimmed off the final unit,
roman/digit colon-titled headings are detected (Eddison's Worm Ouroboros), and
a bare "NOTE" is now treated as story rather than editorial apparatus (Dracula's
Harker epilogue). Building on that, scoring gained a general policy to exclude
author apparatus (preface, foreword, footnotes, dedication, etc.) via a closed,
anchored label vocabulary deliberately narrow enough never to fire on story
sections that merely contain words like "Story" or "Narrative" (Woman in
White's epistolary narrator units), with a symmetric `--include-apparatus` opt-
back-in and every exclusion recorded in the sidecar, never silent.

The three infrastructure pieces: a durable judge cache making long `nd1` runs
resumable and boundable (append-only JSONL keyed by judge identity + prompt, so
resume is just re-running the same command, a `--max-calls N` budget stops
cleanly for running giants like War and Peace in windows, and only the last
call can ever be lost on a crash); an OpenRouter backend for the judge, opening
up DeepSeek and any OpenRouter-proxied model via one key and an
`openrouter:<id>` prefix that needs no code changes per model; and an `st1`
cross-corpus aggregator that loads all sidecars in a directory and emits a
per-book table plus an integrity/anomaly flags report for human triage (run
clean over the 26-book masters corpus: zero integrity findings, no duplication).

**Decisions & notes:** The cache keys on `sha256(describe_judge + prompt)`, so a
prompt/rubric change auto-invalidates only affected entries and dry-run vs real
identities never collide; only successfully-parsed calls are cached, so failures
retry on resume. `BudgetExhausted` is deliberately not a `JudgeError` so the
per-unit "holes not failures" handlers don't swallow a budget stop. The
aggregator validates each sidecar against its own recorded exclusions, so it
cannot yet detect a sidecar gone stale against current scoring policy — a
noted follow-up needing scored counts re-derived from live code.
PROLOGUE/EPILOGUE/INTRODUCTION are treated as story for now (usually narrative;
none in the corpus yet).

## 2026-07-12

A big single-day push that added two new benchmark families and then hardened
the machinery that feeds them. First came the Narrative Dynamics benchmark
(`nd1`): a scoring-only, no-generation series that measures the long-range
structure of one arbitrary-length text (tension trajectory, block rhythm,
thread architecture) over ordered units from a new segmentation layer (chapter
detection or ~1500-word windows, with Gutenberg trimming). All LLM traffic sits
behind a single judge seam (real via `ai_helper`, `FakeJudge` for tests,
dry-run for zero-spend rehearsals), and its rubrics were ported from
StoryDaemon as versioned provenance artifacts whose reliability numbers must be
re-verified here. Next, the single-text series (`st1`): a frozen selection over
the shared v2 metric library aimed at ONE segmented text rather than N runs of
one prompt, including within-text adaptations of the cross-run metrics
(chapter-to-chapter `self_similarity`, `opening_formula`, a local `entity_census`).
Its `self_similarity` duplication detector was proven against a planted
duplicated chapter, the exact defect a generated novel had actually shipped.

The rest of the day was spent making segmentation trustworthy on real books.
The heading heuristics were reframed as a one-time PROPOSER that emits canonical
Markdown (via a new `extract` command with a verification sidecar and an
`--expected-units` gate so extraction fleets can be checked mechanically), after
which analysis consumes a trivial, round-trip-identical splitter. Three rounds
of fixes followed, each driven by a growing ledger of wild-shaped real books
(Pride & Prejudice, Middlemarch, War & Peace, The Moonstone, Dracula, Steps):
TOC-density screening, illustration-block and front-matter handling, worded
"Book" ordinals, wrapped headings, orphan-tail stripping, and census hardening
(NFKD folding so names like Kutuzov and Natasha survive, plus a deterministic
capitalized-token census riding alongside spaCy NER to surface its false
negatives). Endpoint: a six-book proposer acceptance run at 10/10, 27/27, 61/61,
88/88, 60/60, and 365/365 units.

**Decisions & notes:** both new series share the existing metric library but the
frozen v1/v2/nd1 manifests stay untouched, with tests guarding manifest
resolution and name collisions, so the longitudinal series is preserved. The
`narrative_dynamics` package deliberately imports only the stdlib until a real
LLM call, so it runs in the minimal test venv (unlike the eagerly-heavy `utils`
package).

## 2026-06-21

Gave the repo its first test suite: a pytest harness covering the five
stdlib-only v2 metric modules (`cliche_density`, `dialogue_ratio`, `mtld`,
`ngram_diversity`, `intra_text_repetition`) — 41 deterministic tests with
human-predictable expected values plus structural-contract checks, hitting 100%
line coverage of each module. No production code was touched. Bootstrapping this
surfaced a design wart and it was logged in ROADMAP.md as a follow-up:
`utils/__init__.py` eagerly imports `text_analysis` (and thus
`sentence_transformers`), which defeats the metrics package's lazy-import design
and breaks normal package import when the heavy dep isn't installed.

**Decisions & notes:** because of that eager import, the tests load each metric
module by file path (`tests/conftest.py::load_metric`) rather than through the
package — a deliberate workaround until the `__init__.py` lazy-import cleanup
lands.

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
