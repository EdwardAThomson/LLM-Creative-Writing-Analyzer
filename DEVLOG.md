# Dev Log

A chronological narrative of development. Newest entries first.

## 2026-07-15

The headline was migration step 3: `ai_helper.py` became a thin compatibility
layer over the shared `llm-backends` package (pinned to the v0.1.1 tag) while
keeping its exact public surface and, crucially, the frozen per-model payload
profile (`MODEL_CONFIG`) in-repo. OpenAI, Anthropic, and OpenRouter now delegate
to the package; Gemini deliberately stays local because the package path lacks
`top_p`/`top_k` and the frozen benchmark payload sends `top_p=1, top_k=40`, so
delegating would have changed requests on the wire. The `*-cli` keys construct
the package's CLI interfaces (the repo's own userns workaround and key-stripping,
returned home), and `cli_backends/` shrank to import-path shims. A nice
side-effect: `AVAILABLE_MODELS` is now literally `get_supported_models()` and
`DEFAULT_MODELS` is import-time validated, turning the old "three places in sync"
rule structural, and the derivation surfaced two registry keys the hand-maintained
UI list had been missing. Because this is a longitudinal benchmark, the whole
point of the step was proving nothing moved on the wire: 47 new tests, with
`test_payload_equality` driving both the pre-migration `ai_helper` (a sha256-guarded
snapshot from git history) and the new path through identical fake clients and
asserting byte-equal request kwargs across the model matrix, plus
`test_cli_backend_equivalence` on argv, stripped child envs, and neutral-cwd
behaviour. Suite went from 353 passed (2 collection errors) to 417 passed; the
migration made `ai_helper` importable SDK-free, restoring 17 OpenRouter tests.

The masters-corpus benchmark also reached a milestone: War & Peace finished its
DeepSeek nd1 run, so all 26 books are now scored on both st1 and nd1. The report
was refreshed to v2 with the five giants folded in, which moved several headline
numbers (peak-position correlation r -0.70 -> -0.64, dialogue max 68% -> 78% for
Monte Cristo, threads now spanning 2-184 with War & Peace as a cast/length
outlier), and `nd1_reference.json` was built: the per-metric distribution across
all 26 DeepSeek-judged masterworks that LLM-generated nd1 scores get compared
against via `--reference`, which is the entire reason the corpus exists. Its
generator refuses to run if the sidecars mix judges, so the reference is always
single-judge. Finally, a hero banner was added to the README and immediately
optimized from a 2.3MB PNG to a 56KB 1200x630 WebP so it isn't a heavy download
as a site thumbnail.

**Decisions & notes:** Non-payload deltas from the package were audited and
accepted as off-the-wire: OpenAI/Anthropic client `max_retries` is now 1 (was the
SDK default 2, transient-retry behaviour only), the package's claude backend also
strips the deprecated `CLAUDE_API_KEY`, and the tempdir prefix was renamed. The
package pin should only be upgraded between scoring campaigns, re-running the
payload test each time.

## 2026-07-14

Harvest day for the masters-corpus effort: the scoring pipeline was hardened
enough to finish the run, the results were collated into a single dataset, and
the human-facing report went from first draft to a reviewed v1.1 in one sitting.
Two operational fixes kept the OpenRouter/DeepSeek judge alive under load:
scoring several books concurrently was tripping OpenRouter's rate limit and
silently dropping 20-26% of block annotations to holes, fixed by giving the SDK
client real backoff room (max_retries=6, 120s timeout); and 402 errors near the
end of a run turned out to be OpenRouter reserving the judge's full 16384-token
max against the account balance when real responses are ~50-800 tokens, so the
default was cut to 4096. A third fix attacked a persistent-failure class: one
Pride & Prejudice batch that DeepSeek malforms on every attempt used to cost
the whole 20-paragraph batch; the block-rhythm parser is now lenient, recovering
every well-formed per-paragraph object and holing only the truly broken slots,
while still raising (and so retrying) when nothing is recoverable.

With scoring stable, `aggregate_nd1.py` landed as the nd1 sibling of the st1
corpus aggregator: per-book table, an integrity/outlier flags report that
surfaces anomalies without deciding anything, and `corpus_dataset.csv`, a
LEFT JOIN of st1 (26 books) with nd1 (21 scored; the five giants get explicit
blanks). That dataset fed the day's main deliverable, the masters corpus
benchmark report under `reports/masters_corpus/`: v1 established the findings
(the tension-by-genre spread, the tension~peak-position correlation at
r=-0.70, block and thread signals, methodology and caveats), then successive
passes added per-book detail, a motivation-first introduction, a complete
metrics guide with authors-by-metrics tables, a genre-bucket face-validity
view, and finally a data-verified review that corrected several superlatives
and overclaims (including reframing the secondary-shading gauge as band
miscalibration, with 11 of 21 books below the floor). Bulky raw artifacts stay
gitignored and regenerable; only the ~84 KB of report and tables are
version-controlled. Also added GitHub Sponsors config and a README badge.

**Decisions & notes:** The lenient parser deliberately accepts a wrong-length
judge response as a best-effort partial rather than retrying it to full; that
trade recovers persistent-fail batches but means a recovered partial is cached
as-is. The max_tokens cut changes reservation only, not spend. Report is
provisional at 21/26 nd1 books; the giants' judged outputs await aggregation.
Operationally, run 2-3 books in parallel, not 4.

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
