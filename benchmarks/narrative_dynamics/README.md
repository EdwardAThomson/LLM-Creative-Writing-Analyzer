# Narrative Dynamics benchmark (scoring-only)

The repo's third benchmark, alongside the frozen v1 pipeline and the v2+ metrics
library. Its object of study is different from both: not "how similar are N runs
of one prompt" but "what long-range story structure does ONE arbitrary-length
text have". It ports StoryDaemon's validated narrative gauges into this analyzer
as first-class metrics: tension trajectory, block/mode rhythm, and thread
architecture.

There is **no generation step**. The benchmark scores user-supplied text (a
novel, a generated long-form run, anything), so it works equally on masters
corpora and on model output, and it never spends tokens unless a real judge
model is selected.

## Running (scoring-only mode)

```bash
# One file (chapter detection by default; Gutenberg frontmatter auto-trimmed)
python -m benchmarks.narrative_dynamics path/to/book.txt

# A directory of *.txt / *.md files
python -m benchmarks.narrative_dynamics path/to/corpus/

# Fixed ~1500-word windows instead of chapters (texts without headings).
# Caveat: windows mode performs no front-matter exclusion; the sidecar notes it.
python -m benchmarks.narrative_dynamics story.txt --segmentation windows

# One-time extraction to canonical Markdown (the chapter heuristics are a
# PROPOSER run here once per book, with a verification report and an
# .extract.json sidecar; --expected-units N exits nonzero on a count mismatch)
python -m benchmarks.narrative_dynamics.extract book.txt --out book.md --expected-units 61

# Analysis then consumes the canonical Markdown: .md inputs auto-select the
# heuristic-free md splitter (top-level "# " headings + provenance header)
python -m benchmarks.narrative_dynamics book.md

# Subset of metrics, or a frozen benchmark manifest (default: nd1)
python -m benchmarks.narrative_dynamics book.txt --metrics tension_trajectory
python -m benchmarks.narrative_dynamics book.txt --benchmark nd1

# Zero-spend pipeline check (placeholder judge; output stamped as such)
python -m benchmarks.narrative_dynamics book.txt --dry-run

# Judge model (any ai_helper key; default claude-haiku-4-5, the study family)
python -m benchmarks.narrative_dynamics book.txt --judge-model claude-haiku-4-5

# Build a reference distribution from a scored corpus, then compare against it
python -m benchmarks.narrative_dynamics masters/ --make-reference masters_ref.json
python -m benchmarks.narrative_dynamics my_story.txt --reference masters_ref.json
```

Per document it writes `<stem>.nd.json` (self-describing results, sidecar style:
inputs are never touched) and `<stem>.nd.txt` (text report). `--out-dir`
redirects both. `--aliases aliases.json` supplies a per-book name-alias map for
thread clustering (`{"mina murray": "mina harker", ...}`).

The v2 metrics library can also score arbitrary text (scoring-only mode for the
*existing* benchmarks): `python -m utils.metrics --text <file|dir> --benchmark v2`.

## Layout (why it is modular)

```
benchmarks/nd1.yaml           frozen manifest for this benchmark series (extends/add,
                              same scheme as vN; new work adds nd2.yaml)
benchmarks/narrative_dynamics/
  __init__.py                 registry + compute_document() + ndN manifest resolver
  __main__.py                 the scoring-only CLI (this file's examples)
  extract.py                  scoring-free extraction CLI: proposer -> canonical
                              Markdown + .extract.json sidecar + verification report
  segmentation.py             PURE: Gutenberg trimmer, chapter detection (the
                              proposer), md ingestion, ~1500-word windows with
                              paragraph snapping, long-unit truncation
  clustering.py               PURE: name normalization + majority-cast Jaccard
                              thread clustering + convergence detection
  judge.py                    the one LLM seam: ai_helper routing, FakeJudge (tests),
                              DryRunJudge (--dry-run), JSON parse + one-retry protocol
  rubrics/                    versioned rubric artifacts with provenance headers
    tension_anchors.py        the 0-10 anchored tension rubric
    block_types.py            the 7-type block rubric + master reference bands
    cast_extraction.py        the cast/POV/strand extraction prompt
  tension_trajectory.py       metric: compute(units, ctx) -> dict
  block_rhythm.py             metric: compute(units, ctx) -> dict
  thread_architecture.py      metric: compute(units, ctx) -> dict
  reference.py                masters-comparison hook: format, loader, builder, compare
  report.py                   text-report rendering (JSON is the result dict itself)
```

Each metric is one module exposing `compute(units, ctx) -> dict` (the
`utils/metrics` contract, with segmentation units instead of run texts), so
metrics are individually toggleable (`--metrics`), individually testable with a
`FakeJudge` in `ctx`, and new metrics are added by dropping in a module and
listing it in `nd2.yaml`. Everything LLM-flavored goes through `ctx["judge"]`;
everything after the judge (clustering, every aggregation) is deterministic pure
code with direct unit tests. The package imports only the stdlib until a real
LLM call happens, so it runs in the minimal test env.

One deliberate coupling: metrics run in canonical order (tension before
threads), and `tension_trajectory` stashes `ctx["unit_tensions"]` so
`thread_architecture` can report tension deltas at thread switches. When tension
is not run, those fields are null and the thread metric still works.

## The metrics

### tension_trajectory

Each unit is scored 0-10 by an LLM judge against the anchored `TENSION_ANCHORS`
rubric. Aggregations: mean register, std, volatility (mean absolute successive
difference), decile table, peak height/position, tail behavior (tail mean, final
value, ending mode: `wind_down` / `moderate` / `climax_hold`), calm and high
shares.

Masters calibration from the source study (149 chapters, four novels): registers
4.3-7.1, volatility 0.9-1.7, no chapter scored 10; endings either descend to 1-3
or hold 8-9 to the last page.

### block_rhythm

Every paragraph gets a primary (and optional secondary) label from the 7-type
block rubric (SETTING, CHARACTER_DESC, LORE, DIALOGUE, ACTION, INTERIORITY,
TRANSITION), in batches of 20 with one re-ask per malformed batch. Aggregations
center on the source study's four *validated* signals (`structural_gauge`):

| Signal | Master band (source harness) |
|---|---|
| words per mode segment | 90-188 (Conrad, the exception: 58.6) |
| interiority self-transition rate | pooled 0.143; above ~0.3 is the generated-prose tell |
| secondary shading rate | 0.26-0.54 |
| setting touch rate (primary or secondary) | 0.05-0.25 |

Plus type distribution, switch rate, interiority exits, and the transition
matrix. The word-normalized segment length is used because the study showed the
paragraph-count form is a paragraph-length artifact.

### thread_architecture

An LLM extracts each unit's POV, principal cast, and one-line strand; everything
downstream is deterministic (ported clustering rules: majority-cast profile,
Jaccard >= 0.3 assignment with most-recently-active tie-break, cast-only
convergence at 50 percent profile coverage of 2+ established threads).
Aggregations: thread counts, switch rate, run/hand-off lengths, convergence
events and first-convergence position, threshold sensitivity (0.2/0.3/0.4), per-
thread tension registers, and tension deltas at thread switches vs same-thread
transitions (the cut-away analysis).

Known edge, inherited and reported honestly: single-POV books whose supporting
cast rotates fragment at theta 0.3 (The Thirty-Nine Steps reads as 5 threads at
0.3, 1 at 0.2), hence `theta_sensitivity` in every result.

## Provenance and the re-verification caveat

The rubrics are versioned artifacts copied from StoryDaemon (source commit
`abb21b7be9ae5c42c710b406d58e906e8d8d1e50`, 2026-07-12), each carrying a
`PROVENANCE` header with the reliability numbers **as measured in the source
harness** (annotator `anthropic/claude-haiku-4.5`):

* tension rubric: within-1 agreement 100 percent (20/20), exact 85 percent,
  MAD 0.15 on a 20-chapter masters re-pass;
* block rubric: self-consistency 96 percent in-context / 80 percent
  decontextualized; cross-model 80 percent (kappa 0.75); known-answer probes
  8/8; LORE is the noisiest label (1/3 cross-model), treat LORE rates with wide
  error bars;
* cast extraction: mean cast Jaccard 0.95, POV match 100 percent.

**Those numbers do not transfer automatically.** They were measured with a
specific judge model, prompt framing, batching, and corpus. Before findings from
THIS harness are trusted, reliability must be re-verified here: a double-pass
agreement check (shuffled order, different temperature) on a stratified sample,
per rubric. Every result JSON carries the provenance block and this caveat.

## Reference comparisons (masters hook)

`--make-reference` aggregates a scored corpus's headline aggregates into an
`nd_reference/1` JSON (values, mean, std, min, max per key; keys listed in
`reference.py:REFERENCE_KEYS`). `--reference` compares a document against one
and adds a `comparison` block (value, reference mean/range, `within_range`) to
the JSON and a MASTERS COMPARISON table to the text report. Out-of-range is a
flag for attention, not a verdict. No reference data ships with the code; the
intended first reference is the 26-work masters corpus, generated when a real
judged run happens.

## Cost of the first masters run (estimate, not executed)

The 26-work Gutenberg corpus is ~4.3M words. At chapter granularity that is
roughly 1,200-1,500 units, so approximately:

* tension_trajectory: ~1,200-1,500 calls (each at most ~4k words of text);
* thread_architecture: ~1,200-1,500 calls (each at most ~8k words);
* block_rhythm: ~70k paragraphs in batches of 20, ~3,500 calls (the dominant cost).

Total ~6,000-6,500 judge calls, on the order of 30-35M input tokens and 2-3M
output tokens; dollar cost depends on the judge model chosen. `--metrics
tension_trajectory,thread_architecture` halves the calls if block rhythm is
deferred; a per-book pilot (one novel, ~250 calls) is the sensible reliability
re-verification step before committing to the full corpus.

## Testing

All tests use fakes (`FakeJudge`) or the dry-run judge; no test performs an LLM
call. Pure functions (segmentation, trimming, clustering, every aggregation)
have direct coverage, and a synthetic multi-chapter fixture runs end to end
through the CLI in `tests/test_nd_end_to_end.py`.

```bash
venv/bin/python -m pytest tests/ -q
```
