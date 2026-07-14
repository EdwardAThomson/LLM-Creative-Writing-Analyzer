# The Masters Corpus: Benchmark Findings (v1, provisional)

*Draft. Scope: 26 public-domain masterworks extracted to canonical Markdown; scored
locally with **st1** (26/26) and with the LLM-judged **nd1** narrative-dynamics
benchmark (21/26 — the 5 largest works are pending a deliberate run and will be
folded in). Single nd1 judge: DeepSeek (`deepseek/deepseek-chat` via OpenRouter),
validated against Claude Haiku 4.5 (see Methodology). Regenerate from
`corpus_dataset.csv` when the giants land.*

> **Data locations.** This report and the collated tables it cites
> (`corpus_dataset.csv`, `{nd1,st1}_corpus_table.{md,csv}`, `*_flags.md`) live
> here in the analyzer repo (version-controlled, ~80 KB). The **bulky raw
> artifacts** — the 26-book extracted Markdown corpus (~24 MB), and the per-book
> st1/nd1 result sidecars + judge caches (~13 MB) — are kept durable-local and
> gitignored under `StoryDaemon/work/corpus/` (regenerable from the corpus + the
> aggregators `utils/metrics/aggregate_corpus.py` and `aggregate_nd1.py`). To
> refresh after the giants land: re-run the aggregators over the sidecars and
> copy the small outputs back here.

---

## 1. What this is

A reference corpus of 26 canonical novels (Austen, Dickens, Conrad, Collins, Wells,
Haggard, Buchan, Sabatini, Dunsany, Eddison, Bronte, Dumas, Eliot, Tolstoy, Stoker,
Childers), each extracted to verified canonical Markdown and scored by two benchmarks:

- **st1** — local, deterministic prose metrics (lexical diversity, cliche/slop
  density, duplication, dialogue ratio, cast census, etc.). No LLM.
- **nd1** — LLM-judged *narrative dynamics*: a per-chapter tension trajectory
  (0-10 anchored rubric), a 7-type block-rhythm annotation of every paragraph, and a
  cast-based thread architecture.

The purpose is a **quantified "what good prose looks like" baseline** — a reference
band that machine-generated writing can later be measured against.

Collated data: `corpus_dataset.csv` (26 rows x 63 columns, st1 joined to nd1),
plus `st1_corpus_table.md` / `nd1_corpus_table.md` and their flag reports.

---

## 2. Headline findings

1. **The masters are a clean quality baseline.** Across all 26 books: **zero
   duplication** flagged (max verbatim overlap 79 characters against a 500-char
   threshold), cliche density **0.00-0.15 per 1k** (median 0.05), slop **0.01-0.41
   per 1k** (median 0.05), MTLD lexical diversity **70-106** (median 91). This is the
   band; LLM prose that duplicates, climbs in cliche/slop, or collapses in diversity
   is measurably *outside* it.

2. **Tension discriminates by genre, cleanly.** Mean dramatic tension spans
   **2.96 to 7.00** across the 21 nd1 books, and the ordering is defensible top to
   bottom — the three Austen domestic novels occupy the three lowest slots; the
   adventure/peril books top out (Section 4).

3. **Peak *position* anti-correlates with tension (r = -0.70).** High-tension books
   front-load their climax; low-tension books delay it. A real structural regularity,
   quantified (Section 5).

4. **Block rhythm separates dialogue-driven from narration-driven prose** (dialogue
   share 26%-68%), and the masters' four structural gauges sit inside the reference
   envelope from the source study (Section 6).

---

## 3. The quality baseline (st1, 26 books)

| metric | min | median | max | note |
|---|---|---|---|---|
| MTLD (lexical diversity) | 70.4 | 91.3 | 105.9 | the masters' diversity band |
| cliche / 1k words | 0.00 | 0.05 | 0.15 | essentially clean |
| slop / 1k words | 0.01 | 0.05 | 0.41 | clean; the 0.41 is Dunsany's ornate register (below) |
| duplication_suspected | — | — | — | **none** across all 26 |
| max verbatim overlap | — | — | 79 chars | vs a 500-char flag threshold |

**Reading it:** a master can be "sloppy" by the slop lexicon and still be a master —
Dunsany's *King of Elfland's Daughter* tops slop at 0.41/1k on words like *gleaming,
myriad, realm, shimmering*. Those are authentic 1924 romantic-fantasy diction that
*also* happen to be words modern LLMs overuse. So `slop_per_1k` reads **register, not
quality**, at these levels — a caveat carried into Limitations.

---

## 4. Tension trajectory: genre discrimination (nd1, 21 books)

Mean dramatic tension, sorted:

| book | mean tension | register |
|---|---|---|
| sabatini-captainblood | 7.00 | swashbuckler |
| eddison-ouroboros | 6.91 | heroic fantasy |
| buchan-thirtyninesteps | 6.90 | chase thriller |
| haggard-she | 6.59 | peril-adventure |
| stoker-dracula | 6.46 | horror |
| wells-warofworlds | 6.33 | invasion sci-fi |
| buchan-greenmantle | 6.27 | spy adventure |
| sabatini-scaramouche | 6.08 | revolution adventure |
| dickens-taleoftwocities | 6.00 | historical drama |
| haggard-ksm | 6.00 | quest adventure |
| wells-timemachine | 5.25 | sci-fi (reflective frame) |
| conrad-lordjim | 5.18 | moral crisis |
| bronte-janeeyre | 5.05 | gothic bildungsroman |
| childers-riddlesands | 4.68 | slow-burn spy |
| collins-moonstone | 4.62 | detective (epistolary) |
| dunsany-elfland | 4.56 | lyrical fantasy |
| conrad-secretagent | 4.54 | ironic political |
| conrad-heartofdarkness | 4.00 | psychological (3 units) |
| austen-pride | 3.51 | domestic romance |
| austen-persuasion | 3.38 | domestic romance |
| austen-emma | 2.96 | domestic comedy |

The **three Austen novels are the three lowest** and cluster tightly (2.96-3.51);
the peril/adventure cluster tops out ~6-7. The judge is reading *dramatic* tension,
not pace or quality — and placing books where a reader would.

---

## 5. Structure: the peak-position regularity (r = -0.70)

Tension **mean** and tension **peak position** (0 = start, 1 = end) correlate at
**-0.70** across the 21 books. Concretely:

- **High-tension books peak early** — Captain Blood 0.08, Ouroboros 0.04,
  Thirty-Nine Steps 0.05, Scaramouche 0.09, Elfland 0.07. These open on the
  inciting violence (an arrest, a duel, a murder) and pay it off across the book.
- **Low-tension books peak late** — Emma 0.84, Persuasion 0.81, Heart of Darkness
  0.83, Secret Agent 0.80, Riddle of the Sands 0.98. The domestic/reflective/ironic
  novels *withhold*, building to a delayed, often quiet, climax (Austen's climaxes are
  letters and proposals; Conrad's are revelations).

So the benchmark captures two orthogonal things: **how hot** a book runs (mean) and
**where the heat sits** (peak position) — and the two are structurally linked.

---

## 6. Block rhythm (7-type prose modes, 21 books)

- **Dialogue share ranges 26%-68%** (median 51%). Dunsany's *Elfland* is the
  narration-heaviest at 26% (its dreamy, incantatory mode); the conversation-driven
  novels (Conrad's *Secret Agent*, Collins) top 55-68%.
- **A structural signature worth noting:** Dracula shows an unusually high
  **TRANSITION** share — its dated journal/letter entries ("3 May", "5 May") create
  constant time-shifts. The metric caught a genuine formal feature of the epistolary
  novel.
- The four validated structural gauges (`words_per_mode_segment`,
  `interiority_self_transition`, `secondary_shading_rate`, `setting_touch_rate`) land
  inside the masters band inherited from the source block-decomposition study.

---

## 7. Thread architecture (cast-based, 21 books)

Threads per book range **2-24** (median 7); convergence events **0-33** (median 2).

- **Convergence leaders are the dense-social and hunt novels:** Pride & Prejudice
  (33), Emma (24), Moonstone (20), She (12), **Dracula (12)**, Tale of Two Cities (10).
  Dracula's convergences mark the vampire-hunters uniting (first convergence ~0.41);
  Austen's mark the social web repeatedly re-braiding at balls, visits, engagements.
- **Caveat:** raw convergence-event and thread counts scale with book *length* (more
  chapters -> more thread switches), so Pride (61 units) and Emma (55) partly top the
  list by size. A length-normalized version is a natural refinement.

---

## 8. Cross-metric notes

- **tension vs peak-position: r = -0.70** (the structural law, Section 5).
- **tension vs dialogue: r = -0.25** — mildly, tenser books lean slightly less on
  dialogue.
- **tension vs cliche: r = +0.23** — weak; the pulpier high-tension adventures carry
  marginally more stock phrasing, but all remain near-zero in absolute terms.
- **MTLD vs tension: r = +0.03** — ~independent. Lexical diversity and dramatic
  tension measure genuinely different things, as they should.

---

## 9. Methodology and validation

- **Extraction-first.** Raw Gutenberg text -> verified canonical Markdown (one-time
  heading-proposer + human-checkable gates), frozen. Analysis consumes the frozen MD,
  never the heuristics. ~8 general extractor fixes were needed and made (TOC screening,
  author-apparatus exclusion, colophon trim, colon-headings, etc.).
- **Judge choice.** DeepSeek was validated against the calibration model (Claude
  Haiku 4.5) via an A/B on Dracula's tension curve: **Pearson r = 0.86**, mean
  absolute difference **0.82** on the 0-10 scale, 27/28 chapters within 2 points, both
  using the full range. DeepSeek was then confirmed sensible across all three metrics
  on two full books before scaling. It is cheap and trustworthy for this rubric.
- **Reliability engineering.** A durable per-call **judge cache** (resume + call
  budget) makes every run checkpointed — proven when it recovered transient
  concurrency holes and a mid-run laptop suspend with zero lost work. A rate-limit
  **backoff fix** (SDK `max_retries=6`) plus running 2-3 books in parallel eliminated
  the concurrency-induced annotation gaps.

---

## 10. Limitations and honest caveats

- **Scope: 21/26 for nd1.** The 5 largest works (Woman in White, Bleak House, Monte
  Cristo, Middlemarch, War & Peace — ~4,000 judge calls between them) are pending a
  deliberate, sleep-prevented run. All findings above are on the 21.
- **Single judge.** nd1 numbers are DeepSeek's. The A/B says DeepSeek tracks Haiku
  well on tension; block/thread were spot-checked, not fully A/B'd.
- **One incomplete book.** `austen-pride` is at 2079/2095 paragraphs (99.24%) — a
  single block batch that *persistently* fails to parse (not a transient hole; retries
  don't fix it). Tension and thread metrics are complete; only ~16 paragraphs of block
  labels are missing. Flagged in `nd1_corpus_flags.md`.
- **Metric caveats to remember when reading:** `slop_per_1k` reads *register* not
  *quality* (Section 3); `opening_formula` (st1) is contaminated by chapter
  titles/epistolary headers/epigraphs and should not be read as prose formulaicity;
  thread convergence/count scale with book length (Section 7); the entity census does
  not coreference names (Van Helsing -> van/helsing) so cast sizes are inflated.

---

## 11. Next

1. **Finish the giants** -> refresh every table and this report (a one-command
   regenerate; scope becomes 26/26).
2. **Build the nd1 masters reference distribution** (`--make-reference`) — the durable
   band that LLM-generated text gets scored against.
3. **Optional refinements** (all logged): length-normalize thread metrics; strip
   non-prose leading lines from `opening_formula`; a lenient/partial-fill block parser
   for the persistent-failure batch class; a DeepSeek-vs-Haiku A/B on block & thread.

*Generated from `corpus_dataset.csv` and the per-benchmark tables under
`work/corpus/scores/`.*
