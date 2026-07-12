# Metrics Roadmap — Creative-Writing Analysis Upgrades

_Status: partially implemented — v2 ships 8 metrics (Phases 1–2) · updated 2026-06-19 · companion to [ROADMAP.md](ROADMAP.md)_

## Context

The analyzer today measures **cross-run repetition / diversity** — text similarity,
semantic similarity, entity & name reuse, structure, and length — i.e. model
"fingerprints." Three gaps are now visible from the 2026 work:

1. **The sound/shape layer is invisible.** The V-surname signature (report §6.6)
   recurs *phonetically* across vendors even where exact tokens don't repeat
   (Opus 4.8: a distinct V-surname in 8/10 runs, zero token repeats). Exact-string
   matching cannot see it.
2. **No per-text craft quality.** We measure how *varied* the N outputs are, never
   how *good* any single one is.
3. **No prompt-adherence scoring.** The prompt carries ~20 structured parameters
   (genre, POV, tense, species, tech…) and we never check whether the output honors them.

This document plans a series of upgrades to close those gaps, ordered
cheapest-and-highest-leverage first.

## Guiding principles

- **Proxies vs ground truth.** Phases 1–3 are cheap, interpretable *screens*; the
  LLM-judge pass (Phase 4) is the closest thing to ground truth. Treat the judge as
  the anchor and the automated metrics as fast diagnostics that explain *why* a
  piece scored as it did — not as quality measures in themselves.
- **Length-robustness.** Prefer metrics that don't move merely because a text is
  longer (the raw-TTR problem the longitudinal notes keep caveating).
- **NER-awareness.** Every name/entity metric must filter spaCy `PERSON`
  false-positives and split full names into components (report §4.4, §6.4).
- **Re-analyzable.** Each metric must run over saved `results/*.json` (so the
  frozen 2025/2026 corpora can be re-scored without re-generating — you can't
  re-run the 2025 models), not only on fresh generations. See the retroactive
  CLI in *Architecture* below.

---

## Architecture — v1 frozen · flat metric library · cumulative manifests

The design that lets the benchmark grow (v2, v3, …) without ever invalidating an
earlier version. **Separate the two things that get conflated:**

- *What a metric **is*** → a **module**, identified by a stable name, living in one
  flat library (`utils/metrics/`). Organized by metric, **never by version**.
- *Which benchmark **version** includes it* → a **manifest** (`benchmarks/vN.yaml`):
  a frozen, named, cumulative list of metric names. A version is a *selection* over
  the shared library, not a folder of code.

**Why not a directory per batch (`v2/`, `v3/`).** A vN run is cumulative
(v3 = v1 ∪ v2 ∪ v3). With manifests that's a set-union of names; with directories
it's import-churn, and a metric *revised* in a later version forces a copy/fork.
Code keyed by metric identity (stable) beats code keyed by introduction date (a
moving label). Per-version directories only earn their place if a metric's
*implementation* genuinely forks — and metric-level versioning (`phonetic_names`
v1 vs v2 as distinct names) handles even that without batch folders.

```
utils/
  metrics/                 # the LIBRARY — grows forever, one module per metric
    __init__.py            # registry: name -> module + compute(responses, names) seam
    _base.py               # the compute(responses, ctx) -> dict contract
    phonetic_names.py
    mtld.py
    ...
    __main__.py            # retroactive CLI over saved runs
  text_analysis.py         # v1 — FROZEN, untouched

benchmarks/                # the VERSIONS — tiny, frozen, diffable manifests (data, not code)
  v1.yaml                  # names the legacy v1 metrics (base)
  v2.yaml                  # extends: v1  +  [phonetic_names, mtld, burstiness, ...]
  v3.yaml                  # extends: v2  +  [ ... ]
```

A manifest is just `extends: <prior>` + `add: [names]`; resolving `vN` walks the
`extends` chain to the full cumulative name set.

**Hard invariants (these are what protect the longitudinal series):**

1. **v1 is never modified.** Its pipeline (`text_analysis.py`,
   `calculate_advanced_metrics`, the v1 writers) and output keys stay
   byte-identical. v1 metrics are *named* in `v1.yaml` but run by the existing
   pipeline — not rewrapped.
2. **v2+ metrics never merge into the v1 `analysis` dict.** `write_json_results`
   dumps the whole dict, so v2 results go to a **separate sidecar**
   (`results_<ts>.metrics.json`), leaving `results_<ts>.json` unchanged.
3. **Default run = v1 only.** New metrics are opt-in (`--benchmark v2` /
   `--metrics name,...`), so a plain run reproduces v1 forever.
4. **Manifests are frozen + cumulative.** Once `v2.yaml` ships it isn't edited;
   new work adds `v3.yaml` (`extends: v2`).
5. **Output is grouped by version; code is not.** The sidecar records the
   benchmark version, the exact metric set, and a schema stamp, so results are
   self-describing as the library evolves.

**The registry** (`utils/metrics/__init__.py`) maps name → module (explicit map or
`pkgutil` auto-discovery) and exposes one seam — `compute(responses, names) ->
{name: result}` — that both the runner and the retroactive CLI call. Heavy deps
stay lazy-imported inside each module's `compute`, so an unused metric never loads
its dependency (keeps the core/CLI path stdlib-only per CLAUDE.md).

---

Each item below is tagged **What · Why · How · Done-when.**

---

## Phase 1 — Diversity & fingerprint upgrades (cheap; extends the current study)

- [x] **Phonetic name clustering** (`phonetic_names`) — *What:* cluster character names by Double
  Metaphone (or `leading-consonant + syllable-count`) and report repetition of the
  *sound*, not the token. *Why:* directly closes the §6.6 V-surname finding — the
  one effect the current metric provably misses. *How:* `jellyfish`/`metaphone`
  (tiny dep) over the filtered PERSON list. *Done-when:* a "phonetic name overlap"
  figure + per-cluster breakdown appear alongside the existing name-component row.
- [x] **Length-robust lexical diversity (MTLD / MATTR)** (`mtld`) — *What:* replace/augment
  raw TTR with MTLD. *Why:* TTR falls automatically as texts lengthen, forcing the
  current equal-length-pair workaround; MTLD is length-stable so Codex/Opus become
  directly comparable. *How:* `lexical-diversity` pkg or ~30 lines. *Done-when:*
  MTLD reported per run + aggregate; longitudinal doc can drop the length caveat.
- [x] **Self-BLEU + distinct-n** (`ngram_diversity`) — *What:* corpus-level diversity of the N runs
  (Self-BLEU; lower = more distinct) and unique-n-gram ratios (distinct-1/2).
  *Why:* well-understood NLG diversity layer between exact and semantic similarity.
  *How:* `sacrebleu`/NLTK or hand-rolled. *Done-when:* both reported per model.
- [ ] **Embedding-cloud shape & modality** — *What:* beyond mean pairwise cosine,
  report spread (mean distance to centroid) and cluster structure (k-means +
  silhouette). *Why:* detects "model rotates among 2–3 stock plots," which a mean
  can't show. *How:* scikit-learn over embeddings already computed. *Done-when:*
  spread + estimated mode count reported.
- [x] **Opening-line diversity** (`opening_lines`) — *What:* similarity metrics computed on the first
  sentence only. *Why:* identical openings are a notorious tell that whole-text
  similarity dilutes. *How:* reuse existing similarity fns on sentence[0].
- [ ] **NER-aware name metric (formalize)** — *What:* ship the false-positive
  stop-list + component splitting + hyphen handling as a first-class step.
  *Why:* the report repeatedly has to correct the raw metric by hand (Continuum,
  Minds, Kepler; `Okafor-Voss` ≠ `Voss`). *How:* stop-list + `text.split` +
  hyphen-aware tokenization in `text_analysis.py`. *Done-when:* "repeated name
  components" is trustworthy without manual cross-checking.

## Phase 2 — Per-text craft metrics (the missing quality axis; still automated)

- [x] **Sentence-length burstiness** (`burstiness`) — *What:* variance/std of sentence length, not
  just the mean already tracked. *Why:* humans vary rhythm; models flatten it — a
  robust human-vs-AI signal. *How:* free from existing spaCy sentence segmentation.
- [ ] **Readability spread** — *What:* Flesch-Kincaid / Gunning-Fog per run + variance.
  *Why:* cheap register/complexity proxy. *How:* `textstat` (tiny dep).
- [x] **Dialogue-to-narration ratio** (`dialogue_ratio`) — *What:* share of text inside quotation marks.
  *Why:* pacing signal; varies by model. *How:* regex/quote parsing.
- [ ] **Concrete/sensory-word density** — *What:* "show vs tell" proxy. *How:*
  Brysbaert concreteness norms (CSV lookup).
- [x] **Intra-text repetition** (`intra_text_repetition`) — *What:* word/n-gram overuse *within* one story
  (we only measure across runs today). *Why:* genuine quality defect ("word
  obsession"). *How:* per-document n-gram frequency.
- [x] **Cliché / "AI-slop" lexicon hits** (`cliche_density`, lexicon v1) — *What:* frequency of stock markers
  ("a testament to", "in a world where", em-dash density, "couldn't help but").
  *Why:* crude but diagnostic of generic prose. *How:* curated lexicon + counts.

## Phase 3 — Prompt-adherence scoring

- [ ] **Constraint-satisfaction rate** — *What:* fraction of the prompt's ~20
  parameters demonstrably present in the output. *Why:* turns "did it follow
  instructions" into a number; currently unmeasured. *How:* programmatic where
  possible — **tense** (verb POS), **POV** (pronoun distribution → 1st/3rd),
  requested **species/tech** (entity/keyword presence) — and an LLM check for the
  rest. *Done-when:* per-run adherence score + per-parameter pass/fail.

## Narrative Dynamics: the third benchmark (shipped 2026-07, scoring-only)

A separate benchmark series (`benchmarks/nd1.yaml` + the self-contained package
`benchmarks/narrative_dynamics/`), because its object of study differs from the
vN series: not N repeated runs of one prompt but the **long-range structure of a
single arbitrary-length text**. Same architecture instincts as v2 (one module
per metric, frozen cumulative manifests, self-describing output, lazy/spendy
work behind one seam), applied to a new unit of analysis (segmentation units
instead of run texts, `ctx["judge"]` instead of cached NLP models).

- [x] **Segmentation layer** (`segmentation.py`, pure): Gutenberg trimmer,
  chapter-heading detection with windows fallback, ~1500-word paragraph-snapped
  windows, long-unit truncation.
- [x] **`tension_trajectory`**: per-unit 0-10 LLM judge on the anchored
  StoryDaemon tension rubric; register, volatility, deciles, peak, tail/ending
  mode. Rubric ported as a versioned artifact with provenance
  (within-1 reliability 100% as measured in the source harness).
- [x] **`block_rhythm`**: per-paragraph 7-type block annotation; the study's
  four validated signals (words-per-mode-segment, interiority self-transition,
  secondary shading, setting touch) plus distribution/transition matrix.
- [x] **`thread_architecture`**: LLM cast extraction + deterministic
  majority-cast Jaccard clustering (ported rules, pure code, tested); runs,
  convergence, theta sensitivity, tension deltas at switches.
- [x] **Scoring-only mode**: `python -m benchmarks.narrative_dynamics
  <file|dir>` (no generation; `--dry-run` = zero spend), plus `--text` on
  `python -m utils.metrics` so the v2 library scores raw text too.
- [x] **Masters-comparison hook**: `nd_reference/1` format, loader,
  `--make-reference` builder, `--reference` comparison in report + JSON.
  Reference data itself is generated later by a real masters run.
- [ ] **Reliability re-verification in THIS harness**: required before
  trusting findings: double-pass agreement per rubric on a stratified sample
  (the ported reliability numbers were measured in StoryDaemon's harness; see
  each rubric's PROVENANCE header).
- [ ] **First masters run**: 26-work Gutenberg corpus, ~6,000-6,500 judge
  calls (block annotation dominates); pilot one novel (~250 calls) as the
  re-verification step first.

Docs: [benchmarks/narrative_dynamics/README.md](benchmarks/narrative_dynamics/README.md).

## Single-Text series (st1, shipped 2026-07, zero-LLM)

The vN library adapted for ONE arbitrary-length text: `benchmarks/st1.yaml` is a
new manifest series (extends: null, resolved by the same `_manifests.py` loader)
whose "runs" are a single document's segmentation units (chapters or ~1500-word
windows, reusing `benchmarks/narrative_dynamics/segmentation.py`, not
duplicating it). Same library, different unit of account: vN asks "how do N runs
of one prompt vary", st asks "how does one book vary against itself". Everything
is local (stdlib or the repo's existing spaCy/embedding deps); no judge seam at
all, zero LLM calls end to end.

- [x] Per-text v2 metrics reused as-is (per_run = per unit): `mtld`,
  `burstiness`, `dialogue_ratio`, `intra_text_repetition`, `cliche_density`,
  plus `ngram_diversity` (distinct-n/Self-BLEU across chapters) and
  `phonetic_names` (name-sound inventory) whose within-text readings are
  meaningful directly.
- [x] **`text_structure`** (new, stdlib): the structure-equivalent profile
  (words/sentences/paragraphs and ratios); v1's structure numbers live in the
  frozen N-run pipeline and cannot be reused per-text.
- [x] **`self_similarity`** (new, stdlib + optional local embedding): adjacent-
  unit similarity series, max/mean/median, longest verbatim token run, flagged
  outliers. The within-text analog of cross-run text similarity and a
  duplication detector: built against the observed defect class (~9,200
  verbatim chars across adjacent scenes, 0.668 vs ~0.02 baseline), proven in
  tests with a planted duplicated block.
- [x] **`opening_formula`** (new, stdlib): chapter-opening formula detection
  (all-pairs first-sentence similarity + first-words census), the within-text
  analog of `opening_lines`.
- [x] **`entity_census`** (new, spaCy lazy): entity analysis reframed as a
  single-text census (cast size, recurring cast vs walk-ons, densities,
  name-component inventory); cross-run overlap is meaningless for one book.
- [x] Scoring-only invocation through the existing retroactive CLI:
  `python -m utils.metrics --text book.txt --segment chapters --benchmark st1`.

Naming guard: the new module names deliberately avoid the v1 legacy names
(`structure`, `entity_analysis`, ...) so the frozen vN manifests keep resolving
to exactly what they resolved to before (tested).

## Phase 4 — LLM-as-judge (quality ground truth; the anchor)

- [ ] **Rubric scoring** — *What:* a strong model rates each piece on craft
  dimensions (hook, prose, originality, coherence, characterization). *Why:* the
  study's missing quality axis. *How:* via `ai_helper`; structured-output rubric.
- [ ] **Pairwise comparison → Elo / Bradley-Terry** — *What:* head-to-head judgments
  across models aggregated into a ranking. *Why:* "which is better" more reliable
  than absolute scores. *How:* `ai_helper`; randomize order; aggregate offline.
- [ ] **Genericness via perplexity/surprisal** — *What:* score each text under a
  reference LM; low perplexity flags predictable/generic prose. *Why:* objective
  complement to the subjective judge. *How:* a small local LM or API logprobs.
- [ ] **Bias controls (required for any judge metric)** — never let a model judge its
  own output (self-preference bias); randomize order in pairwise (position bias);
  report judge identity + cost alongside scores.

---

## Recommended first batch

The four that are cheap *and* either fix a known weakness or extend a published finding:

1. **Phonetic name clustering** — ✅ shipped (`phonetic_names`). Closes the §6.6 loop.
2. **MTLD** — ✅ shipped (`mtld`). Removes the TTR length-sensitivity the longitudinal doc keeps caveating.
3. **Sentence-length burstiness** — ✅ shipped (`burstiness`). The first real craft metric, ~free from existing parsing.
4. **LLM-judge rubric pass** — ⬜ still open. Adds the quality axis the whole study currently lacks.

**Status (2026-06-19):** v2 ships **8** metrics — all of Phase 1 except embedding-cloud
shape and the NER-aware name formalization, and all of Phase 2 except readability spread
and concrete/sensory density. Phase 3 (adherence) and Phase 4 (LLM-judge) are not started.
Scored over the full 13-model / 140-run corpus; sidecars committed under `results/`.

## Open questions / risks

- **Dependency creep.** Phase 1–2 add small libs (`jellyfish`, `textstat`,
  `scikit-learn`, a concreteness CSV). Keep them optional/lazy-imported so the
  core CLI-backend path stays stdlib-only (per CLAUDE.md).
- **Judge cost & reproducibility.** Phase 4 is metered and non-deterministic;
  pin the judge model + record it, and consider the Batches API for cost.
- **Backend confound persists.** Quality/adherence comparisons inherit the
  CLI-vs-API sampling caveat (report §6.1) until the API-at-fixed-temperature
  re-run (already in the report's "next step") is done.
