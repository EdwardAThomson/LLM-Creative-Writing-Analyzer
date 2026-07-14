# The Masters Corpus: Benchmark Findings (v1, provisional)

*Draft. Single nd1 judge: DeepSeek (`deepseek/deepseek-chat` via OpenRouter), validated
against Claude Haiku 4.5 (Section 11). See the Introduction below for motivation and
scope, and Section 12 for the current 21/26 nd1 boundary. Regenerate from
`corpus_dataset.csv` when the giants land.*

---

## 1. Introduction

Judging machine-generated creative writing is hard without a yardstick: a number like
"lexical diversity 91" or "cliché density 0.05 per 1k words" only means something once
you know what range real, acknowledged good prose occupies. This report builds that
yardstick empirically instead of by intuition.

It scores a corpus of 26 canonical, public-domain masterworks (Austen, Dickens, Conrad,
Collins, Wells, Haggard, Buchan, Sabatini, Dunsany, Eddison, Bronte, Dumas, Eliot,
Tolstoy, Stoker, Childers) on the same two benchmarks used elsewhere in this project.
The result is a **reference band**: the range of values skilled, published prose
actually falls in. Machine-generated text can then be measured against real literature,
not against a guess about what "good" should look like.

Two benchmarks, deliberately different in cost and in what they can see:

- **st1** is cheap and deterministic: lexical, stylistic, and structural statistics
  computed with no LLM in the loop (lexical diversity, cliché/slop-word density,
  duplication, punctuation habits, dialogue ratio, cast census, and more). It runs
  cheaply over the whole corpus and acts as a broad, fast screen for gross departures
  from published prose. Does the text repeat itself, drown in stock phrasing, or
  collapse in vocabulary?
- **nd1** is LLM-judged and looks at *narrative dynamics*, the things a word-frequency
  count structurally cannot see: whether a story's dramatic tension actually rises and
  falls, whether the prose alternates between action, dialogue, and interiority in a
  shape that reads like a novel rather than one mode throughout, and whether a
  multi-character cast's plotlines run in parallel and converge the way real narratives
  do. It is more expensive to run (per-paragraph and per-chapter LLM calls), so it
  currently covers 21 of the 26 books; the five longest ("giants") are pending a
  dedicated run (Section 12).

Together the two benchmarks answer complementary questions: st1 flags surface-craft
failures; nd1 checks whether the underlying *story shape* behaves like a story.

### What this is

A reference corpus of 26 canonical novels, each extracted to verified canonical
Markdown and scored by both benchmarks:

- **st1**: local, deterministic prose metrics (lexical diversity, cliche/slop
  density, duplication, dialogue ratio, cast census, etc.). No LLM. **26/26 books.**
- **nd1**: LLM-judged *narrative dynamics*: a per-chapter tension trajectory
  (0-10 anchored rubric), a 7-type block-rhythm annotation of every paragraph, and a
  cast-based thread architecture. **21/26 books**; the 5 largest works are pending a
  deliberate run and will be folded in (Section 12).

Collated data: `corpus_dataset.csv` (26 rows x 63 columns, st1 joined to nd1),
plus `st1_corpus_table.md` / `nd1_corpus_table.md` and their flag reports.

### How to read this report

Section 2 defines every metric and points to its full table. Section 3 shows the
whole corpus at a glance in two curated authors-by-metrics tables. Section 4 states
the four headline findings; each one is then developed in its own section: the
quality baseline in Section 5, dramatic tension and genre in Section 6, the
peak-position regularity in Section 7, prose rhythm in Section 8, and plot-thread
structure in Section 9. Section 10 collects the cross-metric correlations and a
genre-bucket view. Section 11 explains how the pipeline was built and validated,
and Section 12 lists the limitations that should temper any conclusion drawn from
these numbers. Appendix A tables every metric for every book. If you only want the
reference band for scoring LLM output, the short path is Sections 4, 5, and 12.

---

## 2. The metrics

What each metric measures, grouped by benchmark, with a pointer to where it is
discussed and to its full per-book values. **Every metric we compute is listed here
and tabled in [Appendix A](#appendix-a-full-metric-tables)** ("App A" below); the
analysis sections narrate the ones with the most to say.

A note on conventions: tables carry values exactly as they appear in
`corpus_dataset.csv`, so shares are 0-1 decimals there; the prose usually restates
shares as percentages. Both forms refer to the same numbers.

### st1: deterministic, no LLM (26/26 books)

The st1 family answers a surface question: does this text look like competent
published prose at the level of vocabulary, phrasing, and repetition? Everything
here is computed locally and deterministically, so it is cheap to run over hundreds
of thousands of words and completely reproducible. Three of the terms deserve a
plain-language gloss before the table. **MTLD** (measure of textual lexical
diversity) estimates vocabulary variety in a way that neither rewards nor punishes
a text for its length; higher means more varied word choice. **Self-BLEU** measures
how similar a book's chapters are to one another in phrasing; higher means more
internal similarity. **Burstiness** captures whether sentence lengths alternate
irregularly (bursty, which is human-typical) or march along at a steady length.

| Metric | What it measures | See |
|---|---|---|
| Text size | total words, and average words per chapter | App A |
| MTLD | lexical diversity: vocabulary variety, controlled for text length | Sec 3, 5; App A |
| Burstiness | how uneven the sentence lengths are (bursty vs. steady rhythm) | App A |
| N-gram diversity | distinct-trigram ratio and self-BLEU: how much a book reuses its own phrasing | App A |
| Intra-text repetition | unigram / bigram / trigram self-repetition rates | App A |
| Cliché density (per 1k words) | rate of stock-phrase hits against a cliché lexicon | Sec 3, 5; App A |
| Slop density (per 1k words) | rate of hits against a lexicon of words LLMs are known to overuse (*gleaming*, *shimmering*, *tapestry*...) | Sec 3, 5; App A |
| Em-dash rate (per 1k words) | punctuation-tic frequency | Sec 5; App A |
| Dialogue ratio | deterministic share of prose inside quotation marks (a *different* measurement from nd1's LLM-judged dialogue share; see the note under Section 3) | Sec 3; App A |
| Duplication / self-similarity | max pairwise chapter similarity and longest verbatim overlap; flags copy-pasted passages | Sec 3, 5; App A |
| Opening formula | similarity between a book's chapter/section openings (title-contaminated; see caveat) | Sec 12; App A |
| Cast / character count | how many distinct named characters a book has: total, recurring, and mentions per 1k words. NER-based and **not** coreference-merged (Van Helsing → van/helsing), so counts run high; useful as a *relative* ensemble-size signal | Sec 3, App A; Sec 12 caveat |

### nd1: LLM-judged narrative dynamics (21/26 books, giants pending)

The nd1 family answers a deeper question: underneath the sentences, does the story
*behave* like a story? The properties it measures (tension that rises and falls,
alternation between modes of prose, plotlines that separate and reconverge) are
invisible to word counts, so an LLM judge reads the book unit by unit and annotates
it against frozen rubrics. That makes nd1 far more expensive than st1, and it
introduces a judge whose reliability has to be checked rather than assumed; Section
11 covers how that was done.

| Metric | What it measures | See |
|---|---|---|
| Tension trajectory | per-unit 0-10 dramatic-tension rating on an anchored rubric, reported as mean, std, min/max, peak height & position, calm/high share, volatility, and the tail behaviour | Sec 3, 6-7; App A |
| Block rhythm | per-paragraph classification into 7 prose modes (setting, character-desc, lore, dialogue, action, interiority, transition), plus structural gauges: the overall mode switch rate, and the source study's four validated signals (words per mode-segment, interiority self-transition, secondary-shading rate, setting-touch rate) | Sec 3, 8; App A |
| Thread architecture | cast-based narrative-thread segmentation: thread count (total and multi-unit), how long each runs before switching (run length), and how/when threads converge (convergence events + first-convergence position) | Sec 3, 9; App A |

---

## 3. The corpus at a glance (authors x metrics)

A curated slice of the full dataset, covering the metrics most discussed in this
report. It mirrors the models x metrics table format used in the companion cohort
reports (`report_2026_cohort.md`), but with **books** as rows instead of models.
Every value below is read directly from `corpus_dataset.csv`; nothing is recomputed
or estimated. **Every metric we compute (not just this curated slice) is tabled in
[Appendix A](#appendix-a-full-metric-tables)**, grouped by metric family.

The slice is split into two tables rather than one wide one, because the two
benchmarks have different scope (26 books vs. 21) and because a combined ~10-column
table over 26 rows stops being readable. Rows are grouped by author (the multi-book
authors, Austen, Conrad, Buchan, Collins, Dickens, Haggard, Sabatini, and Wells,
read together) so per-author patterns are visible at a glance.

### 3.1 st1: quality baseline, all 26 books

| Author | Book | Words | MTLD | Cliché/1k | Slop/1k | Em-dash/1k | Dialogue ratio | Max verbatim (chars) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Austen | emma | 157.3k | 91.2 | 0.03 | 0.01 | 18.32 | 0.466 | 34 |
| Austen | persuasion | 83.2k | 95.9 | 0.05 | 0.01 | 1.52 | 0.307 | 30 |
| Austen | pride | 126.5k | 102.2 | 0.02 | 0.02 | 3.71 | 0.454 | 32 |
| Bronte | janeeyre | 185.3k | 98.8 | 0.05 | 0.11 | 10.36 | 0.416 | 31 |
| Buchan | greenmantle | 98.7k | 95.9 | 0.00 | 0.03 | 2.82 | 0.328 | 33 |
| Buchan | thirtyninesteps | 40.8k | 91.0 | 0.02 | 0.02 | 2.80 | 0.279 | 29 |
| Childers | riddlesands | 108.4k | 105.8 | 0.05 | 0.12 | 7.51 | 0.325 | 36 |
| Collins | moonstone | 194.6k | 82.0 | 0.03 | 0.03 | 9.39 | 0.307 | 60 |
| Collins | womaninwhite | 244.4k | 83.0 | 0.05 | 0.03 | 9.67 | 0.264 | 79 |
| Conrad | heartofdarkness | 37.9k | 90.4 | 0.02 | 0.05 | 16.12 | 0.948 | 34 |
| Conrad | lordjim | 129.9k | 97.4 | 0.07 | 0.06 | 10.84 | 0.191 | 34 |
| Conrad | secretagent | 90.2k | 100.6 | 0.10 | 0.03 | 5.49 | 0.275 | 31 |
| Dickens | bleakhouse | 353.3k | 81.4 | 0.05 | 0.04 | 6.06 | 0.383 | 45 |
| Dickens | taleoftwocities | 135.5k | 75.7 | 0.04 | 0.05 | 5.17 | 0.350 | 44 |
| Dumas | montecristo | 460.0k | 91.5 | 0.08 | 0.03 | 6.15 | 0.546 | 60 |
| Dunsany | elfland | 69.6k | 70.4 | 0.05 | 0.41 | 0.10 | 0.050 | 40 |
| Eddison | ouroboros | 170.0k | 102.7 | 0.05 | 0.08 | 0.32 | 0.476 | 51 |
| Eliot | middlemarch | 316.2k | 105.9 | 0.10 | 0.05 | 7.64 | 0.357 | 29 |
| Haggard | ksm | 82.2k | 103.4 | 0.02 | 0.07 | 3.93 | 0.290 | 33 |
| Haggard | she | 112.1k | 93.3 | 0.03 | 0.08 | 6.26 | 0.283 | 31 |
| Sabatini | captainblood | 112.9k | 103.8 | 0.15 | 0.08 | 4.24 | 0.316 | 36 |
| Sabatini | scaramouche | 124.3k | 91.2 | 0.14 | 0.06 | 5.74 | 0.422 | 39 |
| Stoker | dracula | 161.0k | 86.0 | 0.06 | 0.06 | 8.83 | 0.360 | 45 |
| Tolstoy | warandpeace | 562.5k | 81.9 | 0.12 | 0.06 | 3.78 | 0.209 | 56 |
| Wells | timemachine | 32.4k | 84.4 | 0.00 | 0.06 | 5.40 | 0.124 | 39 |
| Wells | warofworlds | 59.8k | 86.3 | 0.06 | 0.02 | 4.57 | 0.060 | 34 |

**What to notice in this table.** The MTLD column spans 70 to 106 with both
extremes produced by acknowledged masters, so lexical diversity separates registers,
not good books from bad ones (Section 5). The cliché and slop columns are
essentially zero all the way down: whatever else distinguishes these authors, none
of them leans on stock phrasing. The em-dash column varies by a factor of nearly
200 (0.10 for Dunsany to 18.32 for Austen's *Emma*), a reminder that punctuation
habits are authorial signature rather than quality. And one number that looks like
an error but isn't: *Heart of Darkness*'s dialogue ratio of 0.948 is an artifact of
its frame structure, since nearly the whole novel sits inside Marlow's quoted
monologue and quote-mark detection therefore counts it as dialogue. The nd1 judge's
paragraph-level figure for the same book (39.4%, Table 3.2) is the more sensible
reading, and the pair is a good illustration of why the report keeps the two
dialogue measurements separate.

`duplication_suspected` is `False` for all 26 (omitted as a column since it's
constant); "max verbatim (chars)" is the strongest indication of near-duplication and
tops out at 79 characters against a 500-character flag threshold (Section 5).

### 3.2 nd1: narrative dynamics, 21 scored books (5 giants pending)

| Author | Book | Tension mean | Peak position | Dialogue share | Threads | Convergence events |
|---|---|---:|---:|---:|---:|---:|
| Austen | emma | 2.96 | 0.845 | 65.9% | 4 | 24 |
| Austen | persuasion | 3.38 | 0.812 | 49.9% | 4 | 0 |
| Austen | pride | 3.51 | 0.549 | 64.0% | 5 | 33 |
| Bronte | janeeyre | 5.05 | 0.671 | 68.4% | 5 | 0 |
| Buchan | greenmantle | 6.27 | 0.250 | 46.8% | 4 | 2 |
| Buchan | thirtyninesteps | 6.90 | 0.050 | 38.4% | 7 | 0 |
| Childers | riddlesands | 4.68 | 0.982 | 62.8% | 11 | 7 |
| Collins | moonstone | 4.62 | 0.742 | 52.8% | 20 | 20 |
| Collins | womaninwhite | — | — | — | — | — *(pending)* |
| Conrad | heartofdarkness | 4.00 | 0.833 | 39.4% | 2 | 0 |
| Conrad | lordjim | 5.18 | 0.678 | 46.0% | 14 | 2 |
| Conrad | secretagent | 4.54 | 0.808 | 56.5% | 9 | 0 |
| Dickens | bleakhouse | — | — | — | — | — *(pending)* |
| Dickens | taleoftwocities | 6.00 | 0.611 | 63.3% | 24 | 10 |
| Dumas | montecristo | — | — | — | — | — *(pending)* |
| Dunsany | elfland | 4.56 | 0.074 | 29.6% | 14 | 3 |
| Eddison | ouroboros | 6.91 | 0.045 | 62.1% | 12 | 8 |
| Eliot | middlemarch | — | — | — | — | — *(pending)* |
| Haggard | ksm | 6.00 | 0.548 | 50.8% | 2 | 0 |
| Haggard | she | 6.59 | 0.293 | 48.2% | 5 | 12 |
| Sabatini | captainblood | 7.00 | 0.081 | 55.9% | 11 | 0 |
| Sabatini | scaramouche | 6.08 | 0.097 | 63.7% | 10 | 7 |
| Stoker | dracula | 6.46 | 0.554 | 48.1% | 6 | 12 |
| Tolstoy | warandpeace | — | — | — | — | — *(pending)* |
| Wells | timemachine | 5.25 | 0.719 | 32.8% | 2 | 1 |
| Wells | warofworlds | 6.33 | 0.352 | 25.7% | 8 | 0 |

**What to notice in this table.** The tension column alone recovers the corpus's
genre structure: the three Austen novels sit at 2.96-3.51 while the adventure and
peril books cluster at 6-7 (Section 6). The threads and convergence columns spread
just as widely, from *King Solomon's Mines* and *The Time Machine* (2 threads each,
a single travelling party) to *The Moonstone* (20) and *Tale of Two Cities* (24),
and from eight books with zero convergence events to *Pride & Prejudice*'s 33
(Section 9).

The 5 giants (Woman in White, Bleak House, Monte Cristo, Middlemarch, War & Peace) are
fully scored by st1 (Table 3.1) but have no *aggregated* nd1 data yet; their rows are
shown with `—` throughout the nd1 table rather than omitted, so the pending scope is
visible in place. (Woman in White and Middlemarch have completed judge runs awaiting
the next aggregation; see Section 12.) "Dialogue share" here is nd1's LLM-judged
block-rhythm share, **not** the same metric as st1's deterministic "dialogue ratio"
in Table 3.1. The two use different methods (quote-mark detection vs. paragraph-mode
classification) and can diverge for the same book; the nd1 share is discussed in
Section 8 and the st1 ratio in Section 5.

---

## 4. Headline findings

1. **The masters are a clean quality baseline.** Across all 26 books the surface
   metrics are unambiguous: **zero duplication** flagged (the longest verbatim
   overlap anywhere is 79 characters, against a 500-character threshold), cliche
   density **0.00-0.15 per 1k words** (median 0.05), slop density **0.01-0.41 per
   1k** (median 0.05), and MTLD lexical diversity **70-106** (median 91). This is
   the band that acknowledged good prose occupies, and it is what gives the
   benchmark teeth: an LLM output that duplicates its own passages, climbs into
   whole-number cliche rates, or collapses in vocabulary is not just "different",
   it is measurably outside the range real published fiction lives in (Section 5).

2. **Tension discriminates by genre, cleanly.** Mean dramatic tension spans
   **2.96 to 7.00** across the 21 nd1 books, and the ordering is defensible from
   top to bottom: the three Austen domestic novels occupy the three lowest slots,
   and the adventure and peril books fill the top of the table. The judge was never
   told what kind of book it was reading; it recovered the genre structure from the
   prose alone. That is the strongest evidence in this report that the tension
   metric reads *dramatic register* rather than noise (Section 6).

3. **Peak *position* anti-correlates with tension (r = -0.70).** Books that run hot
   place their single most dramatic chapter early and sustain the heat afterwards;
   books that run calm withhold their climax until near the end. This is a real
   structural regularity across the corpus, now quantified, and it means the two
   numbers should be read jointly: a generated story is best compared against the
   pattern, not against either value in isolation (Section 7).

4. **Block rhythm separates dialogue-driven from narration-driven prose** (dialogue
   share 26%-68%). The four structural gauges are more cautionary: every one of
   them has masters outside the reference envelope inherited from the source study,
   and on one gauge (secondary shading) the majority of the corpus sits below the
   band, so the envelope should be read as calibration context, not a pass/fail
   line (Section 8).

---

## 5. The quality baseline (st1, 26 books)

This is the report's most directly reusable product: the range each surface metric
occupies across all 26 masterworks. A generated text scored with the same tools can
be placed inside or outside these ranges with no judgment calls involved.

| metric | min | median | max | note |
|---|---|---|---|---|
| MTLD (lexical diversity) | 70.4 | 91.3 | 105.9 | the masters' diversity band |
| cliche / 1k words | 0.00 | 0.05 | 0.15 | essentially clean |
| slop / 1k words | 0.01 | 0.05 | 0.41 | clean; the 0.41 is Dunsany's ornate register (below) |
| duplication_suspected | — | — | — | **none** across all 26 |
| max verbatim overlap | — | — | 79 chars | vs a 500-char flag threshold |

**Where the extremes sit:**

- **MTLD.** The floor is Dunsany's *The King of Elfland's Daughter* (70.4) and the
  ceiling is Eliot's *Middlemarch* (105.9). This is not a quality gradient (both
  are masters); it is register. Dunsany's incantatory, deliberately repetitive
  high-fantasy prose narrows its own vocabulary variety, which is consistent with
  the same book also topping the slop lexicon below: repeated ornate diction cuts
  both ways. Eliot's discursive, clause-heavy realism ranges widely. The same
  corpus produces both extremes, which is itself evidence that MTLD and slop
  measure different things.

- **Cliche.** The ceiling is Sabatini's *Captain Blood* (0.15/1k); the floor is
  shared by Buchan's *Greenmantle* and Wells's *Time Machine* (both 0.00/1k). Even
  the ceiling is less damning than it looks. Captain Blood's count is not spread
  across many distinct stock phrases; it is driven almost entirely by one repeated
  rhetorical construction, `not_only...but` (14 of its ~17 phrase hits), a genuine
  period rhetorical tic of Sabatini's prose rather than scattered cliche.
  (Tolstoy's *War and Peace*, a giant outside the nd1 scope but fully scored by
  st1, shows the same construction even more heavily, at 73 hits, for the same
  reason.)

- **Slop.** Dunsany's *Elfland* tops slop at 0.41/1k, driven overwhelmingly by four
  words: **gleaming (9 hits across the book), myriad (7), realm (6), shimmering
  (3)**, counted directly from its per-chapter word-hit ledger. These are authentic
  1924 romantic-fantasy diction that *also* happen to be words modern LLMs overuse.
  So `slop_per_1k` reads **register, not quality**, at these levels, a caveat
  carried into Limitations. At the other extreme, Austen's *Emma* is the cleanest
  book by this metric (0.01/1k) while simultaneously having the *highest* em-dash
  rate in the corpus (18.32/1k, against a corpus median of 5.6). Emma's
  free-indirect style (narration that slips in and out of the heroine's own voice)
  leans on dashes for interruption and irony, not on the slop lexicon's ornate
  vocabulary. Two "LLM tells", measured on the same author, point in opposite
  directions, which is exactly why no single st1 metric should be read alone.

---

## 6. Tension trajectory: genre discrimination (nd1, 21 books)

Mean dramatic tension, sorted from hottest to calmest. **Register tags are
illustrative genre labels assigned by hand for readability: a rough shorthand, not
measured data.** Unlike every other column in this report, they do not come from a
benchmark run.

| Author | Book | Mean tension | Register |
|---|---|---:|---|
| Sabatini | captainblood | 7.00 | swashbuckler |
| Eddison | ouroboros | 6.91 | heroic fantasy |
| Buchan | thirtyninesteps | 6.90 | chase thriller |
| Haggard | she | 6.59 | peril adventure |
| Stoker | dracula | 6.46 | horror |
| Wells | warofworlds | 6.33 | invasion sci-fi |
| Buchan | greenmantle | 6.27 | spy adventure |
| Sabatini | scaramouche | 6.08 | revolution adventure |
| Dickens | taleoftwocities | 6.00 | historical drama |
| Haggard | ksm | 6.00 | quest adventure |
| Wells | timemachine | 5.25 | sci-fi (frame narrative) |
| Conrad | lordjim | 5.18 | moral crisis |
| Bronte | janeeyre | 5.05 | gothic romance |
| Childers | riddlesands | 4.68 | slow-burn spy |
| Collins | moonstone | 4.62 | detective (epistolary) |
| Dunsany | elfland | 4.56 | lyrical fantasy |
| Conrad | secretagent | 4.54 | ironic political |
| Conrad | heartofdarkness | 4.00 | psychological drama |
| Austen | pride | 3.51 | domestic romance |
| Austen | persuasion | 3.38 | domestic romance |
| Austen | emma | 2.96 | domestic comedy |

The **three Austen novels are the three lowest** and cluster tightly (2.96-3.51),
while the peril and adventure cluster tops out around 6-7. The judge is reading
*dramatic* tension, not pace or quality, and it places books where a reader would.

**By cluster:** the top eight books (swashbuckler through revolution adventure,
6.08-7.00) are all physical-peril fiction, and their `calm_share` (the fraction of
units scored 3 or lower) is correspondingly low: Buchan's *Thirty-Nine Steps* never
dips to calm at all (0.00) and Eddison's *Ouroboros* almost never does (0.06). The
bottom cluster (2.96-4.62: the three Austens, Conrad's introspective and ironic
novels, and the quieter Collins and Dunsany books) inverts this, with Emma spending
82% of its units calm, the corpus maximum. The middle band (Wells's *Time Machine*,
Conrad's *Lord Jim*, Bronte's *Jane Eyre*, 5.05-5.25) is where framing devices and
moral crises sit: tense in places, reflective in others.

**Three worked examples from the decile tables** (mean tension by tenth of the
document) show three distinct shapes:

- **Sustained-high (adventure):** Sabatini's *Captain Blood* (mean 7.00, peak 9 at
  position 0.08) barely leaves the "high" register anywhere. Its decile-by-decile
  tension runs **6.7, 5.7, 8.3, 7.7, 6.3, 7.8, 7.0, 5.7, 7.3, 7.3**, so even its
  lowest decile (5.7) is still dramatic. It front-loads its peak (position 0.08)
  and never really lets the temperature drop again.
- **Spike-and-settle (Austen):** *Emma* (mean 2.96, peak 7 at position 0.84) spends
  nine of ten deciles at or under 3.4 (**2.2, 2.8, 3.4, 2.8, 2.4, 2.5, 2.6, 3.0**),
  then jumps to **5.4** in decile 8, its single most dramatic stretch, before
  falling back to **2.7** in the final decile. One late disturbance, quickly
  resolved: the shape of a comedy of manners.
- **Genuine slow burn (Childers):** *The Riddle of the Sands* (mean 4.68, peak 8 at
  position 0.98, the latest peak in the whole 21-book set) does something
  structurally different from Austen's spike-and-settle: it climbs and *stays*
  climbed. Its deciles run **2.3, 3.0, 4.5, 3.3, 4.7, 5.7, 4.7, 6.5, 6.3, 6.3**, a
  real ascent across the entire book with no wind-down, matching its "slow-burn
  spy" tag.

---

## 7. Structure: the peak-position regularity (r = -0.70)

Tension **mean** and tension **peak position** (0 = start, 1 = end) correlate at
**-0.70** across the 21 books. In words: the hotter a book runs overall, the
earlier its single most dramatic chapter tends to arrive.

The earliest-peaking books make the pattern concrete. Eddison's *Ouroboros* (peak
at position 0.045), Buchan's *Thirty-Nine Steps* (0.05), Dunsany's *Elfland*
(0.074), Sabatini's *Captain Blood* (0.081), and Sabatini's *Scaramouche* (0.097)
all open on the inciting violence (an arrest, a duel, a murder) and spend the rest
of the book paying it off. Four of those five are also in the top half of the
tension table: the early spike is not an isolated bang but the opening of a
sustained high register.

The latest-peaking books are the mirror image. Childers' *Riddle of the Sands*
peaks at position 0.982, the latest in the whole set, followed by Austen's *Emma*
(0.845), Conrad's *Heart of Darkness* (0.833), Austen's *Persuasion* (0.812), and
Conrad's *Secret Agent* (0.808). These are the domestic, reflective, and ironic
novels, and they *withhold*: they build toward a delayed and often quiet climax.
Austen's climaxes are letters and proposals; Conrad's are revelations.

**A genuine exception, not swept under the rug:** Dunsany's *Elfland* peaks
third-earliest in the set (0.074), like the high-tension adventure cluster, yet its
mean tension (4.56) sits mid-table, in "lyrical fantasy" territory, not with the
swashbucklers. Its early spike (tension 8) isn't sustained the way Captain Blood's
is. This is exactly what an r of -0.70, rather than -1.0, predicts: a strong
tendency with real exceptions.

So the benchmark captures two orthogonal things, **how hot** a book runs (the mean)
and **where the heat sits** (the peak position), and the two are structurally
linked.

---

## 8. Block rhythm (7-type prose modes, 21 books)

Dialogue share spans **25.7%** (Wells's *War of the Worlds*) to **68.4%** (Bronte's
*Jane Eyre*), median 50.8%. (Correcting the earlier draft here: the
narration-heaviest book is *War of the Worlds*, not Dunsany's *Elfland*, whose own
dialogue share is 29.6%. War of the Worlds' Martian-invasion narrative is almost
entirely one observer's report of action and landscape, with only occasional
exchanges; its ACTION share of 0.40 and SETTING share of 0.151 are both among the
corpus's highest.) At the high end, the domestic and social novels cluster
together: Jane Eyre 68.4%, Emma 65.9%, Pride & Prejudice 64.0%, alongside
Sabatini's *Scaramouche* (63.7%) and Dickens's *Tale of Two Cities* (63.3%). Nearer
the median sit books like Haggard's *King Solomon's Mines* (50.8%, the exact
median), Stoker's *Dracula* (48.1%) and Conrad's *Lord Jim* (46.0%).

**Dracula's TRANSITION signature, deepened:** its TRANSITION share is 11.2% of
paragraphs. That is not just "unusually high" in isolation; it is roughly **22x**
the corpus median (0.5%) and the single largest cross-corpus outlier flagged in the
anomaly report (z = 4.20). Its epistolary form (dated journal entries such as
"3 May" and "5 May", plus letters) means nearly one paragraph in nine is doing
calendar and format bookkeeping rather than scene, character, or dialogue work. The
metric is correctly catching a genuine formal feature of the novel, not noise.

**The four structural gauges, checked against exemplars and against each other:**
the reference band inherited from the source block-decomposition study fits the
masters only partly. Every gauge has books outside it, and on one gauge most of the
corpus sits below the band.

- `words_per_mode_segment` (band 58.6-187.8): Sabatini's *Captain Blood* sits near
  the tight end at 79.6 words per segment, quick scene-cutting that matches its
  swashbuckler pace. Three books exceed the ceiling. Conrad's *Lord Jim* is the
  extreme at **323.6**, 72% above the band ceiling and flagged REVIEW in the
  anomaly report: its long, doubly narrated segments (Marlow at length relaying
  what others told him) run far longer than the reference band expects. Conrad's
  *Heart of Darkness* (280.8, covered by the known 3-unit segmentation limitation)
  and Wells's *Time Machine* (201.0), both long-monologue frame narratives, are
  over it too.
- `interiority_self_transition` (band 0.0-0.28): two masters exceed the ceiling.
  Austen's *Persuasion* is highest at **0.294**, alongside the corpus's highest
  interiority share (18.6%, also flagged): Anne Elliot's interior monologue
  disproportionately gives way to more interior monologue rather than to action or
  dialogue. Buchan's *Thirty-Nine Steps* (**0.283**) is just over the line as well,
  a reminder that a first-person thriller narrator also spends runs of consecutive
  paragraphs inside his own head.
- `secondary_shading_rate` (band 0.26-0.54): this is the gauge where the band
  clearly fails to contain the corpus. **11 of the 21 books fall below the floor**,
  including Jane Eyre (0.155), Moonstone (0.158), Tale of Two Cities (0.136), King
  Solomon's Mines (0.163), both Sabatinis (0.148-0.191), Ouroboros (0.209), and The
  Secret Agent (0.243). The three Austen novels are still the extreme of the
  pattern (Pride & Prejudice lowest at **0.106**, Emma at 0.12, Persuasion at
  0.14), and the free-indirect explanation holds for them: Austen's tight focus on
  a single consciousness gives secondary characters little independent "shading".
  But with a majority of the masters under the floor, the honest conclusion is that
  the source study's band is miscalibrated for this corpus on this gauge, not that
  half the canon is deficient.
- `setting_touch_rate` (band 0.05-0.25): two books sit above the ceiling. Wells's
  *War of the Worlds* (**0.271**) is consistent with an invasion narrative that
  keeps returning to landscape and ruined topography, and Dunsany's *Elfland*
  (**0.264**) dwells on scenery just as insistently. Six books sit *below* the
  floor (the three Austens, Moonstone, and both Sabatinis), with Austen's *Emma*
  the corpus's lowest at **0.009**, barely touching setting at all.

So the "inside the envelope" framing in the v1 draft was optimistic: all four
gauges have masters-corpus exceptions, and one (secondary shading) has the majority
of the corpus outside the band. That is informative on its own. The reference band
was built from a different author sample, and tight free-indirect domestic prose
and long, digressive frame narration are exactly the registers likely to sit
outside it. For scoring LLM text, these gauges are best read as descriptive
coordinates relative to the masters' own distribution (Appendix A), not as
pass/fail bands.

---

## 9. Thread architecture (cast-based, 21 books)

A *thread* here is a run of chapters that keeps company with the same cluster of
characters; a *convergence event* is the moment two previously separate clusters
share a unit. Read together, the two numbers sketch a book's plot architecture:
how many separate strands it runs, and whether they ever braid.

Threads per book range **2-24** (median 7); convergence events **0-33** (median 2).

**Convergence leaders are the dense-social and hunt novels:** Pride & Prejudice
(33), Emma (24), Moonstone (20), She (12), **Dracula (12)**, Tale of Two Cities (10).

Three books, examined directly, show three different convergence shapes:

- **Dracula (Stoker): one-way convergence.** The cast literally consolidates.
  T0 tracks Dracula and Jonathan Harker (units 0-3); T1 tracks Lucy Westenra and
  Mina Murray (units 4-7); T2 brings in Van Helsing alongside Jonathan and Mina
  Harker (units 8-27); T3 is the vampire-hunters' core of Van Helsing, Arthur
  Holmwood, John Seward, and Quincey Morris (12 units); T4 (units 17-23) folds in
  everyone: Van Helsing, Godalming, Jonathan Harker, Mina Harker, Seward, Quincey
  Morris. 12 convergence events, first at position 0.41: separate pairs literally
  merge into a single hunting party partway through the book.
- **Pride & Prejudice (Austen): a re-braiding social web.** The corpus's densest
  convergence structure: 33 events (the corpus max, flagged REVIEW at z = 2.86),
  the first at position 0.17, almost immediately. Its threads don't hand off in
  sequence, they overlap continuously, all Bennet-centered: T2 (33 units, nearly
  the whole book) follows Bennet/Elizabeth/Jane with Elizabeth as POV; T3 (16
  units) is Bingley/Darcy/Elizabeth; T4 (8 units) is Catherine de Bourgh/Charlotte
  Collins/Collins/Elizabeth/Maria Lucas. The social web keeps re-forming at balls,
  visits, and engagements: the opposite shape from Dracula's one-way merge.
- **Lord Jim (Conrad): fragmented, embedded narration.** 14 threads (tied
  third-highest with *Elfland*, behind *Tale of Two Cities*'s 24 and *Moonstone*'s
  20) but only **2** convergence events, the first not until position 0.94, almost
  the end. Its threads are nested frames, not a cast reassembling: T4 (22 units,
  dominant) is Marlow narrating Jim's story; T5 (4 units) is a *different*
  narrator entirely, "the french lieutenant" giving his own account embedded
  inside Marlow's telling; T2, at the start, is the formal inquiry (jim,
  magistrate, assessors); T9/T12/T13, at the end, are the Patusan endgame
  (Cornelius, Doramin, Dain Waris, Brown, Kassim). High thread count with almost
  no convergence is the fingerprint of Conrad's layered, secondhand storytelling,
  structurally the opposite of Austen's tightly reconverging cast.

**Caveat, now quantified:** raw thread and convergence counts do scale with book
length. Convergence events correlate with a book's unit (chapter) count at
**r = +0.77** across the 21 books (weaker but still present against raw word
count, r = +0.53). So Pride & Prejudice (61 units) and Emma (55 units) partly top
the convergence list because they're also two of the longest-segmented books in
the set, not solely because of a denser social structure. A length-normalized
version (e.g. convergence events per unit) is a natural refinement, as the v1 draft
already flagged.

---

## 10. Cross-metric notes

- **tension vs peak-position: r = -0.70.** The structural law of Section 7: the
  hotter a book runs, the earlier its peak arrives.
- **tension vs dialogue: r = -0.25.** A mild tilt: tenser books lean slightly less
  on dialogue, carrying their heat through action and event rather than
  conversation. (Computed against st1's deterministic quote-mark dialogue ratio;
  against nd1's own LLM-judged dialogue share the relationship is weaker still,
  r = -0.17. Same direction either way.)
- **tension vs cliche: r = +0.23.** Weak: the pulpier high-tension adventures carry
  marginally more stock phrasing, but every book remains near zero in absolute
  terms, so this is a tilt inside a clean corpus, not a quality split.
- **MTLD vs tension: r = +0.03.** Effectively independent. Lexical diversity and
  dramatic tension measure genuinely different things, and that is what you want
  from two axes of the same benchmark suite.

Two more, computed the same way from `corpus_dataset.csv` (both n=21, single judge,
exploratory; treat as suggestive, not settled):

- **tension vs INTERIORITY block-rhythm share: r = -0.54.** Moderate, and readable:
  the calmest books also linger longest inside a character's head. Austen's
  *Persuasion* has the corpus's second-lowest mean tension (3.38; only *Emma* is
  lower) *and* its highest interiority share (18.6%). At the other end, Eddison's
  *Ouroboros* (mean 6.91, second only to *Captain Blood*) has the corpus's *lowest*
  interiority share (1.9%): action and dialogue crowd out reflection when the
  stakes are constantly high.
- **convergence events vs unit/chapter count: r = +0.77.** Already discussed in
  Section 9 as the quantified version of the length-normalization caveat; restated
  here because it is the strongest of the "new" correlations found.

### Genre view

The tension-by-genre pattern from Section 6 formalizes cleanly if the 21 nd1 books
are grouped into broad genre buckets. This is a synthesis of metrics already
reported, not a new measurement, and it doubles as a **face-validity check**: does
the benchmark place genres where a reader's intuition would?

| Genre bucket | n | Tension | Peak pos | Dialogue | Calm% | High% | Threads |
|---|---|---|---|---|---|---|---|
| Adventure / thriller | 7 | 6.22 | 0.33 | 52% | 17% | 31% | 7.1 |
| Sci-fi & fantasy | 4 | 5.76 | 0.30 | 38% | 26% | 32% | 9.0 |
| Horror / gothic | 2 | 5.75 | 0.61 | 58% | 24% | 32% | 5.5 |
| Mystery / ironic / psychological | 5 | 4.87 | 0.73 | 52% | 42% | 16% | 13.8 |
| Domestic (Austen) | 3 | 3.28 | 0.74 | 60% | 72% | 1% | 4.3 |

It does, on more than one axis:

- **Tension** descends monotonically from adventure (6.22) to Austen (3.28): the
  "horror/action runs hotter" intuition, quantified.
- **Peak position** separates the *same* way: action and speculative fiction
  front-load the climax (~0.30), while mystery/psychological and domestic novels
  delay it (~0.73). The Section-7 peak-position regularity is, at root, a genre
  regularity.
- **Calm vs high share** is the starkest split: Austen runs 72% calm with ~1%
  high-tension chapters, while the adventure/horror/sci-fi clusters sit near ~31% high.

Three caveats keep this **descriptive, not statistical**:

1. **Small n per bucket** (2-7). Horror is n=2 and gets diluted by Jane Eyre
   (gothic-ish, not really horror).
2. **The buckets are a judgment call.** Jane Eyre could be filed as horror or as
   bildungsroman, Lord Jim as adventure or as psychological; different assignments
   would nudge the small buckets.
3. **The corpus isn't a balanced genre sample.** It's public-domain masters skewed
   toward adventure and Victorian fiction, so this is "how the genres *in this
   corpus* stack up," not a claim about genres in general.

*(Based on the 21 nd1 books in this report. Two of the five giants, Woman in White
and Middlemarch, have completed judge runs since these tables were aggregated; they
and the remaining three fold into the buckets at the next full 26-book
regeneration.)*

---

## 11. Methodology and validation

**Extraction first.** Every book went from raw Project Gutenberg text to a
verified, canonical Markdown file before any scoring happened: a one-time
heading-proposer plus human-checkable gates, after which the Markdown is frozen.
Analysis consumes the frozen Markdown, never the extraction heuristics, so a later
extractor fix cannot silently change already-published scores. Getting 26 books
through this cleanly took roughly eight general extractor fixes (table-of-contents
screening, exclusion of editorial apparatus, colophon trimming, colon-style
headings, and similar).

**Judge choice and validation.** Every nd1 number in this report comes from a
single judge, DeepSeek (`deepseek/deepseek-chat` via OpenRouter). Before trusting
it, it was A/B-tested against the calibration model (Claude Haiku 4.5) on Dracula's
28-chapter tension curve: **Pearson r = 0.86** between the two judges' curves, a
mean absolute difference of **0.82** points on the 0-10 scale, and 27 of 28
chapters within 2 points, with both judges using the full range of the rubric. In
plain terms, the two models tell the same story about where Dracula's tension rises
and falls, and disagree by less than a point on average. DeepSeek was then
sanity-checked across all three metric families on two full books before scaling to
the corpus. It is cheap and, for this rubric, trustworthy.

**Reliability engineering.** LLM-judging a corpus of novels means thousands of API
calls, so the pipeline assumes interruptions rather than hoping to avoid them. A
durable per-call **judge cache** (with resume and a call budget) makes every run
checkpointed; in practice it recovered transient concurrency holes and a mid-run
laptop suspend with zero lost work. A rate-limit **backoff fix** (SDK
`max_retries=6`), combined with running only 2-3 books in parallel, eliminated the
concurrency-induced annotation gaps seen in early runs.

---

## 12. Limitations and honest caveats

- **Scope: 21/26 for nd1.** The 5 largest works (Woman in White, Bleak House, Monte
  Cristo, Middlemarch, War & Peace; roughly 4,000 judge calls between them) are
  absent from every nd1 table in this report. Two of them (Woman in White and
  Middlemarch) have since completed their judge runs but are not yet folded into
  the aggregated tables; the other three are pending a deliberate, sleep-prevented
  run. All nd1 findings above (Sections 6-10) are on the 21; st1 findings (Section
  5, and Table 3.1) already cover all 26.
- **Single judge.** Every nd1 number is DeepSeek's opinion. The A/B in Section 11
  says DeepSeek tracks Haiku well on tension, but block rhythm and thread
  architecture were spot-checked rather than fully A/B'd, so cross-judge agreement
  on those two metric families is assumed, not measured.
- **One incomplete book.** `austen-pride` is at 2079 of 2095 paragraphs annotated
  (99.24%): a single block batch persistently fails to parse, and retries do not
  fix it. Tension and thread metrics are complete; only about 16 paragraphs of
  block labels are missing. Flagged in `nd1_corpus_flags.md`.
- **Metric caveats to remember when reading:**
  - `slop_per_1k` reads *register*, not *quality*, at masters-corpus levels
    (Section 5): authentic period diction and LLM-overused vocabulary overlap.
  - `opening_formula` (st1) is contaminated by chapter titles, epistolary headers,
    and epigraphs, and should not be read as prose formulaicity.
  - Thread convergence and thread counts scale with book length, quantified at
    r = +0.77 against unit count (Section 9); compare books of similar length, or
    wait for the length-normalized variant.
  - The entity census does not coreference names (Van Helsing also counts as "van"
    and "helsing"), so cast sizes are inflated; treat them as relative, not
    absolute.
  - The four block-rhythm structural gauges have real masters-corpus exceptions,
    and one of them has most of the corpus below its band (Section 8), so "inside
    the reference band" is a tendency, not a guarantee, for any single book.
- **Register tags are hand-assigned.** The genre labels in Section 6's table (and
  echoed elsewhere) are illustrative shorthand for readability, not a scored
  metric. Treat them as descriptive color, not data.

---

## 13. Next

1. **Finish the giants**, then refresh every table and this report (a one-command
   regenerate; scope becomes 26/26). Woman in White and Middlemarch are already
   judged and waiting on aggregation; Bleak House, Monte Cristo, and War & Peace
   still need their runs.
2. **Build the nd1 masters reference distribution** (`--make-reference`): the
   durable band that LLM-generated text gets scored against, which is the point of
   the whole exercise.
3. **Optional refinements** (all logged): length-normalize the thread metrics;
   strip non-prose leading lines from `opening_formula`; a lenient partial-fill
   block parser for the persistent-failure batch class; a DeepSeek-vs-Haiku A/B on
   block rhythm and thread architecture.

*Generated from `corpus_dataset.csv` and the per-benchmark tables under
`work/corpus/scores/`.*

---

## Appendix A: Full metric tables

*Every metric computed, straight from `corpus_dataset.csv`. Grouped by author; one table per metric family. st1 covers all 26 books; nd1 covers the 21 scored (the 5 giants are omitted from nd1 tables until their runs finish). See Sections 2 and 5-9 for what each metric means and how to read it; the caveats in Section 12 apply (cast counts are not coreference-merged; opening-formula is title-contaminated).*

### A.1 st1 (deterministic, 26 books)

**Size & lexical**

| author | book | words | words/chapter | MTLD | MTLD unreliable | burstiness |
|---|---|---|---|---|---|---|
| austen | emma | 157321 | 2914.93 | 91.16 | 0 | -0.082 |
|  | persuasion | 83233 | 3470.83 | 95.93 | 0 | -0.107 |
|  | pride | 126462 | 1997.74 | 102.23 | 0 | -0.165 |
| bronte | janeeyre | 185262 | 4904.16 | 98.79 | 0 | -0.121 |
| buchan | greenmantle | 98664 | 4476.82 | 95.89 | 0 | -0.306 |
|  | thirtyninesteps | 40760 | 4089.2 | 90.98 | 0 | -0.296 |
| childers | riddlesands | 108415 | 3876.68 | 105.82 | 0 | -0.162 |
| collins | moonstone | 194601 | 3271.48 | 81.99 | 0 | -0.242 |
|  | womaninwhite | 244423 | 4043.39 | 83.02 | 0 | -0.197 |
| conrad | heartofdarkness | 37904 | 12834 | 90.35 | 0 | -0.1 |
|  | lordjim | 129920 | 2873.64 | 97.37 | 0 | -0.095 |
|  | secretagent | 90202 | 6961.15 | 100.55 | 0 | -0.15 |
| dickens | bleakhouse | 353328 | 5287.45 | 81.42 | 0 | -0.1 |
|  | taleoftwocities | 135539 | 3027.38 | 75.72 | 0 | -0.119 |
| dumas | montecristo | 459951 | 3963.21 | 91.48 | 0 | -0.149 |
| dunsany | elfland | 69632 | 2039.59 | 70.36 | 0 | -0.222 |
| eddison | ouroboros | 169991 | 5085.39 | 102.66 | 0 | -0.162 |
| eliot | middlemarch | 316167 | 3614.93 | 105.86 | 0 | -0.176 |
| haggard | ksm | 82157 | 3911.48 | 103.38 | 0 | -0.189 |
|  | she | 112073 | 3883.17 | 93.28 | 0 | -0.18 |
| sabatini | captainblood | 112892 | 3657.32 | 103.75 | 0 | -0.15 |
|  | scaramouche | 124291 | 3471.25 | 91.17 | 0 | -0.142 |
| stoker | dracula | 161025 | 5753.25 | 86.01 | 0 | -0.216 |
| tolstoy | warandpeace | 562486 | 1572.17 | 81.93 | 0 | -0.156 |
| wells | timemachine | 32361 | 2033.06 | 84.41 | 0 | -0.259 |
|  | warofworlds | 59841 | 2225.56 | 86.28 | 0 | -0.229 |

**Craft & style**

| author | book | cliche/1k | slop/1k | em-dash/1k | dialogue ratio |
|---|---|---|---|---|---|
| austen | emma | 0.03 | 0.01 | 18.32 | 0.466 |
|  | persuasion | 0.05 | 0.01 | 1.52 | 0.307 |
|  | pride | 0.02 | 0.02 | 3.71 | 0.454 |
| bronte | janeeyre | 0.05 | 0.11 | 10.36 | 0.416 |
| buchan | greenmantle | 0 | 0.03 | 2.82 | 0.328 |
|  | thirtyninesteps | 0.02 | 0.02 | 2.8 | 0.279 |
| childers | riddlesands | 0.05 | 0.12 | 7.51 | 0.325 |
| collins | moonstone | 0.03 | 0.03 | 9.39 | 0.307 |
|  | womaninwhite | 0.05 | 0.03 | 9.67 | 0.264 |
| conrad | heartofdarkness | 0.02 | 0.05 | 16.12 | 0.948 |
|  | lordjim | 0.07 | 0.06 | 10.84 | 0.191 |
|  | secretagent | 0.1 | 0.03 | 5.49 | 0.275 |
| dickens | bleakhouse | 0.05 | 0.04 | 6.06 | 0.383 |
|  | taleoftwocities | 0.04 | 0.05 | 5.17 | 0.350 |
| dumas | montecristo | 0.08 | 0.03 | 6.15 | 0.546 |
| dunsany | elfland | 0.05 | 0.41 | 0.1 | 0.050 |
| eddison | ouroboros | 0.05 | 0.08 | 0.32 | 0.476 |
| eliot | middlemarch | 0.1 | 0.05 | 7.64 | 0.357 |
| haggard | ksm | 0.02 | 0.07 | 3.93 | 0.290 |
|  | she | 0.03 | 0.08 | 6.26 | 0.283 |
| sabatini | captainblood | 0.15 | 0.08 | 4.24 | 0.316 |
|  | scaramouche | 0.14 | 0.06 | 5.74 | 0.422 |
| stoker | dracula | 0.06 | 0.06 | 8.83 | 0.360 |
| tolstoy | warandpeace | 0.12 | 0.06 | 3.78 | 0.209 |
| wells | timemachine | 0 | 0.06 | 5.4 | 0.124 |
|  | warofworlds | 0.06 | 0.02 | 4.57 | 0.060 |

**Diversity & self-repetition**

| author | book | distinct-3 | self-BLEU | intra-uni | intra-bi | intra-tri |
|---|---|---|---|---|---|---|
| austen | emma | 0.809 | 0.308 | 0.436 | 0.187 | 0.029 |
|  | persuasion | 0.869 | 0.228 | 0.444 | 0.186 | 0.027 |
|  | pride | 0.839 | 0.283 | 0.352 | 0.140 | 0.017 |
| bronte | janeeyre | 0.872 | 0.225 | 0.402 | 0.170 | 0.020 |
| buchan | greenmantle | 0.865 | 0.243 | 0.436 | 0.205 | 0.027 |
|  | thirtyninesteps | 0.895 | 0.192 | 0.421 | 0.209 | 0.030 |
| childers | riddlesands | 0.888 | 0.211 | 0.368 | 0.167 | 0.019 |
| collins | moonstone | 0.774 | 0.339 | 0.430 | 0.208 | 0.041 |
|  | womaninwhite | 0.776 | 0.338 | 0.443 | 0.214 | 0.038 |
| conrad | heartofdarkness | 0.920 | 0.128 | 0.564 | 0.274 | 0.047 |
|  | lordjim | 0.862 | 0.251 | 0.347 | 0.158 | 0.020 |
|  | secretagent | 0.874 | 0.205 | 0.475 | 0.226 | 0.042 |
| dickens | bleakhouse | 0.781 | 0.323 | 0.494 | 0.236 | 0.048 |
|  | taleoftwocities | 0.859 | 0.229 | 0.415 | 0.188 | 0.037 |
| dumas | montecristo | 0.767 | 0.359 | 0.454 | 0.200 | 0.036 |
| dunsany | elfland | 0.853 | 0.257 | 0.447 | 0.189 | 0.038 |
| eddison | ouroboros | 0.869 | 0.235 | 0.477 | 0.177 | 0.027 |
| eliot | middlemarch | 0.806 | 0.312 | 0.406 | 0.176 | 0.025 |
| haggard | ksm | 0.880 | 0.206 | 0.445 | 0.180 | 0.027 |
|  | she | 0.868 | 0.243 | 0.436 | 0.181 | 0.025 |
| sabatini | captainblood | 0.882 | 0.221 | 0.400 | 0.169 | 0.024 |
|  | scaramouche | 0.859 | 0.252 | 0.391 | 0.183 | 0.033 |
| stoker | dracula | 0.825 | 0.272 | 0.517 | 0.239 | 0.037 |
| tolstoy | warandpeace | 0.744 | 0.379 | 0.375 | 0.160 | 0.033 |
| wells | timemachine | 0.912 | 0.167 | 0.354 | 0.161 | 0.024 |
|  | warofworlds | 0.894 | 0.205 | 0.344 | 0.157 | 0.024 |

**Duplication**

| author | book | dup suspected | max pair sim | max verbatim | flagged pairs |
|---|---|---|---|---|---|
| austen | emma | False | 0.031 | 34 | 0 |
|  | persuasion | False | 0.038 | 30 | 0 |
|  | pride | False | 0.041 | 32 | 0 |
| bronte | janeeyre | False | 0.026 | 31 | 0 |
| buchan | greenmantle | False | 0.026 | 33 | 0 |
|  | thirtyninesteps | False | 0.019 | 29 | 0 |
| childers | riddlesands | False | 0.023 | 36 | 0 |
| collins | moonstone | False | 0.049 | 60 | 0 |
|  | womaninwhite | False | 0.078 | 79 | 0 |
| conrad | heartofdarkness | False | 0.015 | 34 | 0 |
|  | lordjim | False | 0.026 | 34 | 0 |
|  | secretagent | False | 0.017 | 31 | 0 |
| dickens | bleakhouse | False | 0.027 | 45 | 0 |
|  | taleoftwocities | False | 0.036 | 44 | 0 |
| dumas | montecristo | False | 0.040 | 60 | 0 |
| dunsany | elfland | False | 0.034 | 40 | 0 |
| eddison | ouroboros | False | 0.030 | 51 | 0 |
| eliot | middlemarch | False | 0.031 | 29 | 0 |
| haggard | ksm | False | 0.027 | 33 | 0 |
|  | she | False | 0.026 | 31 | 0 |
| sabatini | captainblood | False | 0.025 | 36 | 0 |
|  | scaramouche | False | 0.030 | 39 | 0 |
| stoker | dracula | False | 0.028 | 45 | 0 |
| tolstoy | warandpeace | False | 0.049 | 56 | 0 |
| wells | timemachine | False | 0.031 | 39 | 0 |
|  | warofworlds | False | 0.035 | 34 | 0 |

**Openings & cast**

| author | book | opening sim | opening hi-rate | cast size | recurring cast | mentions/1k |
|---|---|---|---|---|---|---|
| austen | emma | 0.074 | 0.000 | 239 | 72 | 25.76 |
|  | persuasion | 0.084 | 0.000 | 113 | 52 | 27.6 |
|  | pride | 0.092 | 0.000 | 108 | 66 | 25.37 |
| bronte | janeeyre | 0.078 | 0.000 | 316 | 95 | 12.23 |
| buchan | greenmantle | 0.114 | 0.000 | 319 | 72 | 11.76 |
|  | thirtyninesteps | 0.169 | 0.022 | 131 | 33 | 6.75 |
| childers | riddlesands | 0.074 | 0.000 | 208 | 57 | 6.35 |
| collins | moonstone | 0.096 | 0.002 | 173 | 72 | 18.13 |
|  | womaninwhite | 0.085 | 0.000 | 219 | 76 | 14.76 |
| conrad | heartofdarkness | 0.1 | 0.000 | 48 | 6 | 5.01 |
|  | lordjim | 0.072 | 0.000 | 178 | 44 | 8.39 |
|  | secretagent | 0.133 | 0.000 | 101 | 27 | 11.08 |
| dickens | bleakhouse | 0.081 | 0.000 | 471 | 190 | 22.91 |
|  | taleoftwocities | 0.104 | 0.000 | 196 | 63 | 13.59 |
| dumas | montecristo | 0.09 | 0.000 | 696 | 226 | 17.16 |
| dunsany | elfland | 0.121 | 0.000 | 44 | 19 | 6.17 |
| eddison | ouroboros | 0.175 | 0.002 | 476 | 155 | 16.57 |
| eliot | middlemarch | 0.036 | 0.004 | 587 | 176 | 22.04 |
| haggard | ksm | 0.09 | 0.000 | 174 | 63 | 11.11 |
|  | she | 0.09 | 0.000 | 302 | 34 | 7.26 |
| sabatini | captainblood | 0.11 | 0.000 | 237 | 96 | 17.03 |
|  | scaramouche | 0.089 | 0.000 | 192 | 84 | 14.82 |
| stoker | dracula | 0.108 | 0.003 | 270 | 92 | 11.99 |
| tolstoy | warandpeace | 0.089 | 0.000 | 956 | 352 | 18.02 |
| wells | timemachine | 0.123 | 0.000 | 26 | 3 | 1.02 |
|  | warofworlds | 0.146 | 0.017 | 122 | 33 | 4.03 |

### A.2 nd1 (LLM-judged, 21 books; giants pending)

**Tension: level**

| author | book | mean | std | min | max | peak | peak pos |
|---|---|---|---|---|---|---|---|
| austen | emma | 2.96 | 1.210 | 1 | 7 | 7 | 0.845 |
|  | persuasion | 3.38 | 1.070 | 2 | 6 | 6 | 0.812 |
|  | pride | 3.51 | 1.510 | 1 | 8 | 8 | 0.549 |
| bronte | janeeyre | 5.05 | 2.210 | 1 | 9 | 9 | 0.671 |
| buchan | greenmantle | 6.27 | 1.600 | 3 | 8 | 8 | 0.250 |
|  | thirtyninesteps | 6.9 | 1.140 | 4 | 8 | 8 | 0.050 |
| childers | riddlesands | 4.68 | 1.690 | 2 | 8 | 8 | 0.982 |
| collins | moonstone | 4.62 | 1.980 | 1 | 9 | 9 | 0.742 |
| conrad | heartofdarkness | 4 | 2.160 | 2 | 7 | 7 | 0.833 |
|  | lordjim | 5.18 | 2.110 | 2 | 9 | 9 | 0.678 |
|  | secretagent | 4.54 | 2.130 | 2 | 9 | 9 | 0.808 |
| dickens | taleoftwocities | 6 | 2.300 | 2 | 9 | 9 | 0.611 |
| dunsany | elfland | 4.56 | 1.970 | 2 | 8 | 8 | 0.074 |
| eddison | ouroboros | 6.91 | 1.620 | 2 | 9 | 9 | 0.045 |
| haggard | ksm | 6 | 2.490 | 1 | 9 | 9 | 0.548 |
|  | she | 6.59 | 2.210 | 2 | 9 | 9 | 0.293 |
| sabatini | captainblood | 7 | 1.760 | 3 | 9 | 9 | 0.081 |
|  | scaramouche | 6.08 | 2.110 | 2 | 9 | 9 | 0.097 |
| stoker | dracula | 6.46 | 2.110 | 1 | 9 | 9 | 0.554 |
| wells | timemachine | 5.25 | 2.490 | 2 | 9 | 9 | 0.719 |
|  | warofworlds | 6.33 | 2.280 | 2 | 9 | 9 | 0.352 |

**Tension: shape**

| author | book | calm share | high share | volatility | tail mean | final |
|---|---|---|---|---|---|---|
| austen | emma | 0.820 | 0.000 | 0.960 | 2.67 | 2 |
|  | persuasion | 0.670 | 0.000 | 1.090 | 4 | 2 |
|  | pride | 0.660 | 0.020 | 0.950 | 3.83 | 1 |
| bronte | janeeyre | 0.370 | 0.180 | 2.050 | 5.5 | 1 |
| buchan | greenmantle | 0.140 | 0.230 | 1.620 | 7.5 | 7 |
|  | thirtyninesteps | 0.000 | 0.300 | 1.220 | 7 | 7 |
| childers | riddlesands | 0.320 | 0.040 | 0.960 | 6.33 | 8 |
| collins | moonstone | 0.370 | 0.080 | 1.970 | 3.17 | 4 |
| conrad | heartofdarkness | 0.670 | 0.000 | 2.500 | 7 | 7 |
|  | lordjim | 0.360 | 0.160 | 1.590 | 7.75 | 9 |
|  | secretagent | 0.460 | 0.150 | 2.170 | 4 | 4 |
| dickens | taleoftwocities | 0.240 | 0.400 | 1.910 | 8.5 | 9 |
| dunsany | elfland | 0.500 | 0.090 | 1.700 | 6 | 6 |
| eddison | ouroboros | 0.060 | 0.450 | 1.970 | 6 | 2 |
| haggard | ksm | 0.240 | 0.380 | 2.300 | 2.5 | 3 |
|  | she | 0.170 | 0.410 | 2.360 | 8 | 6 |
| sabatini | captainblood | 0.130 | 0.450 | 1.830 | 7.33 | 6 |
|  | scaramouche | 0.220 | 0.330 | 1.830 | 7 | 3 |
| stoker | dracula | 0.110 | 0.460 | 2.190 | 5.33 | 1 |
| wells | timemachine | 0.310 | 0.310 | 2.470 | 2.5 | 3 |
|  | warofworlds | 0.190 | 0.440 | 1.770 | 3.67 | 2 |

**Block rhythm: mode shares**

| author | book | setting | char-desc | lore | dialogue | action | interior | transit |
|---|---|---|---|---|---|---|---|---|
| austen | emma | 0.006 | 0.017 | 0.016 | 0.659 | 0.172 | 0.127 | 0.002 |
|  | persuasion | 0.015 | 0.031 | 0.047 | 0.499 | 0.221 | 0.186 | 0.002 |
|  | pride | 0.007 | 0.017 | 0.013 | 0.640 | 0.225 | 0.096 | 0.002 |
| bronte | janeeyre | 0.032 | 0.024 | 0.010 | 0.684 | 0.156 | 0.093 | 0.001 |
| buchan | greenmantle | 0.062 | 0.033 | 0.013 | 0.468 | 0.310 | 0.111 | 0.003 |
|  | thirtyninesteps | 0.058 | 0.029 | 0.033 | 0.384 | 0.350 | 0.140 | 0.006 |
| childers | riddlesands | 0.048 | 0.012 | 0.038 | 0.628 | 0.187 | 0.083 | 0.006 |
| collins | moonstone | 0.015 | 0.022 | 0.042 | 0.528 | 0.266 | 0.113 | 0.014 |
| conrad | heartofdarkness | 0.141 | 0.091 | 0.015 | 0.394 | 0.242 | 0.096 | 0.020 |
|  | lordjim | 0.084 | 0.040 | 0.067 | 0.460 | 0.226 | 0.120 | 0.003 |
|  | secretagent | 0.031 | 0.065 | 0.013 | 0.565 | 0.209 | 0.118 | 0.000 |
| dickens | taleoftwocities | 0.049 | 0.034 | 0.028 | 0.633 | 0.210 | 0.040 | 0.005 |
| dunsany | elfland | 0.144 | 0.025 | 0.091 | 0.296 | 0.340 | 0.092 | 0.011 |
| eddison | ouroboros | 0.063 | 0.034 | 0.056 | 0.621 | 0.184 | 0.019 | 0.023 |
| haggard | ksm | 0.055 | 0.026 | 0.035 | 0.508 | 0.310 | 0.059 | 0.007 |
|  | she | 0.059 | 0.025 | 0.088 | 0.482 | 0.259 | 0.083 | 0.005 |
| sabatini | captainblood | 0.023 | 0.051 | 0.030 | 0.559 | 0.264 | 0.071 | 0.002 |
|  | scaramouche | 0.016 | 0.037 | 0.018 | 0.637 | 0.210 | 0.081 | 0.001 |
| stoker | dracula | 0.038 | 0.011 | 0.041 | 0.481 | 0.238 | 0.078 | 0.112 |
| wells | timemachine | 0.127 | 0.012 | 0.087 | 0.328 | 0.322 | 0.111 | 0.012 |
|  | warofworlds | 0.151 | 0.015 | 0.074 | 0.257 | 0.400 | 0.090 | 0.013 |

**Block rhythm: structural gauges**

| author | book | switch rate | words/segment | interior self-trans | 2ndary shading | setting touch |
|---|---|---|---|---|---|---|
| austen | emma | 0.398 | 164.6 | 0.256 | 0.116 | 0.009 |
|  | persuasion | 0.512 | 157.3 | 0.294 | 0.136 | 0.026 |
|  | pride | 0.408 | 136.9 | 0.230 | 0.106 | 0.011 |
| bronte | janeeyre | 0.401 | 111.9 | 0.235 | 0.155 | 0.053 |
| buchan | greenmantle | 0.560 | 100.8 | 0.232 | 0.266 | 0.111 |
|  | thirtyninesteps | 0.571 | 98 | 0.283 | 0.273 | 0.118 |
| childers | riddlesands | 0.440 | 126.9 | 0.166 | 0.270 | 0.101 |
| collins | moonstone | 0.542 | 98.3 | 0.173 | 0.158 | 0.026 |
| conrad | heartofdarkness | 0.677 | 280.8 | 0.167 | 0.495 | 0.222 |
|  | lordjim | 0.593 | 323.6 | 0.119 | 0.531 | 0.165 |
|  | secretagent | 0.615 | 90.7 | 0.267 | 0.243 | 0.053 |
| dickens | taleoftwocities | 0.436 | 92.3 | 0.205 | 0.136 | 0.075 |
| dunsany | elfland | 0.596 | 139.5 | 0.083 | 0.436 | 0.264 |
| eddison | ouroboros | 0.409 | 162.6 | 0.119 | 0.209 | 0.100 |
| haggard | ksm | 0.533 | 97.6 | 0.156 | 0.163 | 0.088 |
|  | she | 0.518 | 158.1 | 0.257 | 0.297 | 0.115 |
| sabatini | captainblood | 0.530 | 79.6 | 0.115 | 0.191 | 0.046 |
|  | scaramouche | 0.468 | 83.2 | 0.177 | 0.148 | 0.027 |
| stoker | dracula | 0.543 | 138.1 | 0.153 | 0.281 | 0.079 |
| wells | timemachine | 0.472 | 201 | 0.182 | 0.464 | 0.211 |
|  | warofworlds | 0.578 | 111.4 | 0.169 | 0.369 | 0.271 |

**Thread architecture**

| author | book | threads | threads 2+ | switch rate | run mean | run max | converge | 1st converge pos |
|---|---|---|---|---|---|---|---|---|
| austen | emma | 4 | 3 | 0.481 | 2.04 | 15 | 24 | 0.450 |
|  | persuasion | 4 | 2 | 0.348 | 2.67 | 7 | 0 | — |
|  | pride | 5 | 4 | 0.333 | 2.9 | 9 | 33 | 0.170 |
| bronte | janeeyre | 5 | 5 | 0.216 | 4.22 | 8 | 0 | — |
| buchan | greenmantle | 4 | 3 | 0.286 | 3.14 | 11 | 2 | 0.930 |
|  | thirtyninesteps | 7 | 2 | 0.667 | 1.43 | 3 | 0 | — |
| childers | riddlesands | 11 | 4 | 0.815 | 1.22 | 3 | 7 | 0.620 |
| collins | moonstone | 20 | 7 | 0.661 | 1.5 | 7 | 20 | 0.170 |
| conrad | heartofdarkness | 2 | 1 | 0.500 | 1.5 | 2 | 0 | — |
|  | lordjim | 14 | 6 | 0.477 | 2.05 | 9 | 2 | 0.940 |
|  | secretagent | 9 | 3 | 1.000 | 1 | 1 | 0 | — |
| dickens | taleoftwocities | 24 | 7 | 0.909 | 1.1 | 3 | 10 | 0.570 |
| dunsany | elfland | 14 | 7 | 0.788 | 1.26 | 3 | 3 | 0.460 |
| eddison | ouroboros | 12 | 6 | 0.656 | 1.5 | 7 | 8 | 0.590 |
| haggard | ksm | 2 | 1 | 0.100 | 7 | 12 | 0 | — |
|  | she | 5 | 3 | 0.500 | 1.93 | 5 | 12 | 0.290 |
| sabatini | captainblood | 11 | 5 | 0.567 | 1.72 | 4 | 0 | — |
|  | scaramouche | 10 | 7 | 0.571 | 1.71 | 5 | 7 | 0.540 |
| stoker | dracula | 6 | 5 | 0.444 | 2.15 | 4 | 12 | 0.410 |
| wells | timemachine | 2 | 2 | 0.133 | 5.33 | 12 | 1 | 0.970 |
|  | warofworlds | 8 | 3 | 0.500 | 1.93 | 10 | 0 | — |

---

## Addendum: data locations & regeneration

This report and the collated tables it cites (`corpus_dataset.csv`,
`{nd1,st1}_corpus_table.{md,csv}`, `*_flags.md`) live here in the analyzer repo
(version-controlled, ~80 KB). The **bulky raw artifacts** (the 26-book extracted
Markdown corpus at ~24 MB, and the per-book st1/nd1 result sidecars plus judge
caches at ~13 MB) are kept durable-local and gitignored under
`StoryDaemon/work/corpus/`, regenerable from the corpus plus the aggregators
`utils/metrics/aggregate_corpus.py` and `aggregate_nd1.py`. To refresh after the
giants land: re-run the aggregators over the sidecars and copy the small outputs
back here.
