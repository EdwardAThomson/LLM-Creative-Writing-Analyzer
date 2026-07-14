# The Masters Corpus: Benchmark Findings (v1, provisional)

*Draft. Scope: 26 public-domain masterworks extracted to canonical Markdown; scored
locally with **st1** (26/26) and with the LLM-judged **nd1** narrative-dynamics
benchmark (21/26 — the 5 largest works are pending a deliberate run and will be
folded in). Single nd1 judge: DeepSeek (`deepseek/deepseek-chat` via OpenRouter),
validated against Claude Haiku 4.5 (see Methodology). Regenerate from
`corpus_dataset.csv` when the giants land.*

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
   share 26%-68%), and the masters' four structural gauges mostly, but not
   universally, sit inside the reference envelope from the source study (Section 6).

---

## 3. The quality baseline (st1, 26 books)

| metric | min | median | max | note |
|---|---|---|---|---|
| MTLD (lexical diversity) | 70.4 | 91.3 | 105.9 | the masters' diversity band |
| cliche / 1k words | 0.00 | 0.05 | 0.15 | essentially clean |
| slop / 1k words | 0.01 | 0.05 | 0.41 | clean; the 0.41 is Dunsany's ornate register (below) |
| duplication_suspected | — | — | — | **none** across all 26 |
| max verbatim overlap | — | — | 79 chars | vs a 500-char flag threshold |

**Where the extremes sit:**

- **MTLD.** The floor is Dunsany's *The King of Elfland's Daughter* (70.4) and the
  ceiling is Eliot's *Middlemarch* (105.9). This isn't a quality gradient — both are
  masters — it's register: Dunsany's incantatory, repetitive high-fantasy prose
  narrows its own vocabulary variety (consistent with it also topping the slop
  lexicon below — repeated ornate diction cuts both ways), while Eliot's
  discursive, clause-heavy realism ranges widely. The same corpus produces both
  extremes, which is itself evidence MTLD and slop measure different things.

- **Cliche.** The ceiling is Sabatini's *Captain Blood* (0.15/1k); the floor is
  Buchan's *Greenmantle* (0.00/1k). Captain Blood's count is not spread across many
  distinct stock phrases — it is driven almost entirely by one repeated rhetorical
  construction, `not_only...but` (14 of its ~17 phrase hits), a genuine period
  rhetorical tic of Sabatini's prose rather than scattered cliche. (Tolstoy's *War
  and Peace*, a giant outside the nd1 scope but fully scored by st1, shows the same
  construction even more heavily — 73 hits — for the same reason.)

- **Slop.** Dunsany's *Elfland* tops slop at 0.41/1k, driven overwhelmingly by four
  words: **gleaming (9 hits across the book), myriad (7), realm (6), shimmering
  (3)** — counted directly from its per-chapter word-hit ledger. These are authentic
  1924 romantic-fantasy diction that *also* happen to be words modern LLMs overuse.
  So `slop_per_1k` reads **register, not quality**, at these levels — a caveat
  carried into Limitations. At the other extreme, Austen's *Emma* is the cleanest by
  this metric (0.01/1k) while simultaneously having the *highest* em-dash rate in the
  corpus (18.32/1k, vs. a corpus median of 5.6) — Emma's free-indirect style leans on
  dashes for interruption and irony, not on the slop lexicon's ornate vocabulary.

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

**By cluster:** the top eight books (swashbuckler through spy adventure, 6.08-7.00)
are all physical-peril adventure fiction, and their `calm_share` (fraction of units
scored <=3) is correspondingly low — Buchan's *Thirty-Nine Steps* never dips to calm
at all (0.00) and Eddison's *Ouroboros* almost never does (0.06). The bottom cluster
(Conrad's introspective/ironic novels plus the three Austens, 2.96-4.62) inverts
this: Emma spends 82% of its units calm, the corpus maximum. The middle band
(Wells's *Time Machine*, Conrad's *Lord Jim*, Bronte's *Jane Eyre*, 5.05-5.25) is
where framing devices and moral crises sit — tense in places, reflective in others.

**Three worked examples from the decile tables** (mean tension by tenth of the
document) show three distinct shapes:

- **Sustained-high (adventure):** Sabatini's *Captain Blood* (mean 7.00, peak 9 at
  position 0.08) barely leaves the "high" register anywhere: decile-by-decile
  tension runs **6.7, 5.7, 8.3, 7.7, 6.3, 7.8, 7.0, 5.7, 7.3, 7.3** — even its
  lowest decile (5.7) is still dramatic. It front-loads its peak (position 0.08)
  and never really lets the temperature drop again.
- **Spike-and-settle (Austen):** *Emma* (mean 2.96, peak 7 at position 0.84) spends
  nine of ten deciles at or under 3.4 — **2.2, 2.8, 3.4, 2.8, 2.4, 2.5, 2.6, 3.0** —
  then jumps to **5.4** in decile 8 (its single most dramatic stretch) before
  falling back to **2.7** in the final decile. One late disturbance, quickly
  resolved: the shape of a comedy of manners.
- **Genuine slow burn (Childers):** *The Riddle of the Sands* (mean 4.68, peak 8 at
  position 0.98 — the latest peak in the whole 21-book set) does something
  structurally different from Austen's spike-and-settle: it climbs and *stays*
  climbed. Deciles run **2.3, 3.0, 4.5, 3.3, 4.7, 5.7, 4.7, 6.5, 6.3, 6.3** — a real
  ascent across the entire book with no wind-down, matching its "slow-burn spy" tag.

---

## 5. Structure: the peak-position regularity (r = -0.70)

Tension **mean** and tension **peak position** (0 = start, 1 = end) correlate at
**-0.70** across the 21 books. Concretely, ranked by peak position:

- **Earliest-peaking:** Eddison's *Ouroboros* (0.045), Buchan's *Thirty-Nine Steps*
  (0.05), Dunsany's *Elfland* (0.074), Sabatini's *Captain Blood* (0.081), Sabatini's
  *Scaramouche* (0.097). These open on the inciting violence (an arrest, a duel, a
  murder) and pay it off across the book.
- **Latest-peaking:** Childers' *Riddle of the Sands* (0.982), Austen's *Emma*
  (0.845), Conrad's *Heart of Darkness* (0.833), Austen's *Persuasion* (0.812),
  Conrad's *Secret Agent* (0.808). The domestic/reflective/ironic novels *withhold*,
  building to a delayed, often quiet, climax (Austen's climaxes are letters and
  proposals; Conrad's are revelations).

**A genuine exception, not swept under the rug:** Dunsany's *Elfland* peaks
third-earliest in the set (0.074) — like the high-tension adventure cluster — yet
its mean tension (4.56) sits mid-table, in "lyrical fantasy" territory, not with
the swashbucklers. Its early spike (tension 8) isn't sustained the way Captain
Blood's is. This is exactly what an r of -0.70, rather than -1.0, predicts: a strong
tendency with real exceptions.

So the benchmark captures two orthogonal things: **how hot** a book runs (mean) and
**where the heat sits** (peak position) — and the two are structurally linked.

---

## 6. Block rhythm (7-type prose modes, 21 books)

Dialogue share spans **25.7%** (Wells's *War of the Worlds*) to **68.4%** (Bronte's
*Jane Eyre*), median 50.8%. (Correcting the earlier draft here: the narration-heaviest
book is *War of the Worlds*, not Dunsany's *Elfland* — Elfland's own dialogue share
is 29.6%. War of the Worlds' Martian-invasion narrative is almost entirely one
observer's report of action and landscape — ACTION 0.40 and SETTING 0.151 are both
among the corpus's highest — with only occasional exchanges.) At the high end, the
domestic/social novels cluster together: Jane Eyre 68.4%, Emma 65.9%, Pride &
Prejudice 64.0%, alongside Sabatini's *Scaramouche* (63.7%) and Dickens's *Tale of
Two Cities* (63.3%). Nearer the median sit books like Haggard's *King Solomon's
Mines* (50.8%, the exact median), Stoker's *Dracula* (48.1%) and Conrad's *Lord Jim*
(46.0%).

**Dracula's TRANSITION signature, deepened:** its TRANSITION share is 11.2% of
paragraphs — not just "unusually high" in isolation, but roughly **22x** the
corpus median (0.5%) and the single largest cross-corpus outlier flagged in the
anomaly report (z = 4.20). Its epistolary form (dated journal entries — "3 May",
"5 May" — and letters) means nearly one paragraph in nine is doing calendar/format
bookkeeping rather than scene, character, or dialogue work. The metric is correctly
catching a genuine formal feature of the novel, not noise.

**The four structural gauges, checked against exemplars — and against each other:**
the masters mostly, but not universally, sit inside the reference band inherited
from the source block-decomposition study.

- `words_per_mode_segment` (band 58.6-187.8): Sabatini's *Captain Blood* sits near
  the tight end (79.6 words/segment — quick scene-cutting, matching its swashbuckler
  pace). Conrad's *Lord Jim* is the corpus's clear outlier at **323.6** — 72% above
  the band ceiling, and flagged REVIEW in the anomaly report — its long, doubly
  narrated segments (Marlow at length relaying what others told him) run far
  longer than the reference band expects.
- `interiority_self_transition` (band 0.0-0.28): Austen's *Persuasion* is the one
  master that exceeds the ceiling, at **0.294** (alongside the corpus's highest
  interiority share, 18.6%, also flagged) — Anne Elliot's interior monologue
  disproportionately gives way to more interior monologue rather than to action or
  dialogue.
- `secondary_shading_rate` (band 0.26-0.54): all three Austen novels fall *below*
  the band floor — Pride & Prejudice lowest at **0.106**, Emma at 0.12, Persuasion
  at 0.14. This is a real pattern, not noise: Austen's tight free-indirect focus on
  a single consciousness gives secondary characters comparatively little
  independent "shading" next to the source study's reference authors.
- `setting_touch_rate` (band 0.05-0.25): Wells's *War of the Worlds* sits just above
  the ceiling at **0.271**, consistent with an invasion narrative that keeps
  returning to landscape and ruined topography; Austen's *Emma* sits at the
  opposite extreme, **0.009**, the corpus's lowest — barely touching setting at all.

So the "inside the envelope" framing in the v1 draft was optimistic: three of the
four gauges have masters-corpus exceptions, concentrated in the Austen cluster and
in Conrad's *Lord Jim*. That's informative on its own — the reference band was
built from a different author sample, and tight free-indirect domestic prose and
long, digressive frame narration are exactly the registers likely to sit outside it.

---

## 7. Thread architecture (cast-based, 21 books)

Threads per book range **2-24** (median 7); convergence events **0-33** (median 2).

**Convergence leaders are the dense-social and hunt novels:** Pride & Prejudice
(33), Emma (24), Moonstone (20), She (12), **Dracula (12)**, Tale of Two Cities (10).

Three books, examined directly, show three different convergence shapes:

- **Dracula (Stoker) — one-way convergence.** The cast literally consolidates:
  T0 tracks Dracula and Jonathan Harker (units 0-3); T1 tracks Lucy Westenra and
  Mina Murray (units 4-7); T2 brings in Van Helsing alongside Jonathan and Mina
  Harker (units 8-27); T3 is the vampire-hunters' core — Van Helsing, Arthur
  Holmwood, John Seward, Quincey Morris (12 units); T4 (units 17-23) folds in
  everyone — Van Helsing, Godalming, Jonathan Harker, Mina Harker, Seward, Quincey
  Morris. 12 convergence events, first at position 0.41: separate pairs literally
  merge into a single hunting party partway through the book.
- **Pride & Prejudice (Austen) — a re-braiding social web.** The corpus's densest
  convergence structure: 33 events (the corpus max, flagged REVIEW at z = 2.86),
  the first at position 0.17 — almost immediately. Its threads don't hand off in
  sequence, they overlap continuously, all Bennet-centered: T2 (33 units, nearly
  the whole book) follows Bennet/Elizabeth/Jane with Elizabeth as POV; T3 (16
  units) is Bingley/Darcy/Elizabeth; T4 (8 units) is Catherine de Bourgh/Charlotte
  Collins/Collins/Elizabeth/Maria Lucas. The social web keeps re-forming at balls,
  visits, and engagements — the opposite shape from Dracula's one-way merge.
- **Lord Jim (Conrad) — fragmented, embedded narration.** 14 threads (2nd-highest
  in the set) but only **2** convergence events, the first not until position
  0.94 — almost the end. Its threads are nested frames, not a cast reassembling:
  T4 (22 units, dominant) is Marlow narrating Jim's story; T5 (4 units) is a
  *different* narrator entirely — "the french lieutenant" giving his own account,
  embedded inside Marlow's telling; T2, at the start, is the formal inquiry (jim,
  magistrate, assessors); T9/T12/T13, at the end, are the Patusan endgame
  (Cornelius, Doramin, Dain Waris, Brown, Kassim). High thread count with almost
  no convergence is the fingerprint of Conrad's layered, secondhand storytelling —
  structurally the opposite of Austen's tightly reconverging cast.

**Caveat, now quantified:** raw thread/convergence counts do scale with book
length. Convergence events correlate with a book's unit (chapter) count at
**r = +0.77** across the 21 books (weaker but still present against raw word
count, r = +0.53). So Pride & Prejudice (61 units) and Emma (55 units) partly top
the convergence list because they're also two of the longest-segmented books in
the set, not solely because of a denser social structure. A length-normalized
version (e.g. convergence events per unit) is a natural refinement, as the v1 draft
already flagged.

---

## 8. Cross-metric notes

- **tension vs peak-position: r = -0.70** (the structural law, Section 5).
- **tension vs dialogue: r = -0.25** — mildly, tenser books lean slightly less on
  dialogue.
- **tension vs cliche: r = +0.23** — weak; the pulpier high-tension adventures carry
  marginally more stock phrasing, but all remain near-zero in absolute terms.
- **MTLD vs tension: r = +0.03** — ~independent. Lexical diversity and dramatic
  tension measure genuinely different things, as they should.

Two more, computed the same way from `corpus_dataset.csv` (both n=21, single judge,
exploratory — treat as suggestive, not settled):

- **tension vs INTERIORITY block-rhythm share: r = -0.54** — moderate. The calmest
  books also linger longest inside a character's head: Austen's *Persuasion* has
  both the corpus's lowest mean tension (3.38) *and* its highest interiority share
  (18.6%). The tensest book, Eddison's *Ouroboros* (mean 6.91), has the corpus's
  *lowest* interiority share (1.9%) — action and dialogue crowd out reflection when
  the stakes are constantly high.
- **convergence events vs unit/chapter count: r = +0.77** — already discussed in
  Section 7 as the quantified version of the length-normalization caveat; restated
  here because it's the strongest of the "new" correlations found.

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
  deliberate, sleep-prevented run. All nd1 findings above (Sections 4-8) are on the
  21; st1 findings (Section 3) already cover all 26.
- **Single judge.** nd1 numbers are DeepSeek's. The A/B says DeepSeek tracks Haiku
  well on tension; block/thread were spot-checked, not fully A/B'd.
- **One incomplete book.** `austen-pride` is at 2079/2095 paragraphs (99.24%) — a
  single block batch that *persistently* fails to parse (not a transient hole; retries
  don't fix it). Tension and thread metrics are complete; only ~16 paragraphs of block
  labels are missing. Flagged in `nd1_corpus_flags.md`.
- **Metric caveats to remember when reading:** `slop_per_1k` reads *register* not
  *quality* (Section 3); `opening_formula` (st1) is contaminated by chapter
  titles/epistolary headers/epigraphs and should not be read as prose formulaicity;
  thread convergence/count scale with book length, now quantified at r = +0.77
  against unit count (Section 7); the entity census does not coreference names (Van
  Helsing -> van/helsing) so cast sizes are inflated; the four block-rhythm
  structural gauges have real masters-corpus exceptions (Section 6), so "inside the
  reference band" is a tendency, not a guarantee, for any single book.

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

---

## Addendum: data locations & regeneration

This report and the collated tables it cites (`corpus_dataset.csv`,
`{nd1,st1}_corpus_table.{md,csv}`, `*_flags.md`) live here in the analyzer repo
(version-controlled, ~80 KB). The **bulky raw artifacts** — the 26-book extracted
Markdown corpus (~24 MB), and the per-book st1/nd1 result sidecars + judge caches
(~13 MB) — are kept durable-local and gitignored under `StoryDaemon/work/corpus/`
(regenerable from the corpus + the aggregators `utils/metrics/aggregate_corpus.py`
and `aggregate_nd1.py`). To refresh after the giants land: re-run the aggregators
over the sidecars and copy the small outputs back here.
