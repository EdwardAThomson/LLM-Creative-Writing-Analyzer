# EQ-Bench Creative Writing v3 vs this repo's metric sets

Comparison of EQ-Bench's creative-writing benchmarks (eqbench.com, Sam Paech)
against this repo's metric sets (v1 frozen legacy, v2 library, nd1 narrative
dynamics, st1 single-text). Covers two of their benchmarks: **Creative Writing
v3** (short pieces, leaderboard) and **Longform Writing** (planned 8-chapter
stories). Sources: the leaderboard pages, about.html, and the
`EQ-bench/creative-writing-bench` GitHub repo, as read July 2026.

**What it is:** a quality leaderboard, not a behavioural analyser. 32 prompts x
3 iterations (96 items) generated via OpenRouter at temp 0.7 / min_p 0.1,
~$10/model. Hybrid scoring: a 0-20 rubric score plus an Elo rating from
pairwise matchups (Glicko-2 modified to weight win margins, anchored to
DeepSeek-R1 = 1500). Judge: Claude Sonnet 4.6 (Elo) and Sonnet 4 (rubric);
the Elo judge was upgraded from Sonnet 4 in March 2026, so scores are not
comparable across judge eras. Side metrics: slop score (corpus-derived
GPT-ism lexicon), repetition (summed common word/bigram/trigram frequencies),
vocab complexity (proportion of 3+ syllable words), length.

Legend: ✅ direct counterpart, 🟡 partial or proxy, ❌ not measured here /
there. Direction of the gap noted per row.

## 1. What EQ-Bench measures, mapped to this repo

| EQ-Bench feature | Nearest metric here | Coverage |
|---|---|---|
| Rubric quality score (0-20, LLM judge) | none by design; nd1 rubrics score structure, not overall quality | ❌ deliberate |
| Elo via pairwise matchups (margin-weighted Glicko-2) | none | ❌ deliberate |
| Slop score (corpus-derived over-represented LLM phrases) | `cliche_density` (v2): curated lexicon, frozen `LEXICON_VERSION` | 🟡 theirs is empirically derived, ours is curated |
| Repetition (summed common word/n-gram frequencies) | `mtld`, `burstiness`, `ngram_diversity`, `intra_text_repetition` (v2) | ✅ ours are deeper |
| Vocab complexity (3+ syllable proportion, anti-inflation) | `vocabulary_diversity` (v1) is adjacent, not the same construct | 🟡 |
| Length | trivially present in v1 `structure` | ✅ |
| Longform: per-chapter quality "degradation" sparkline | st1 segments one text into units but has no per-unit quality-decay metric; `tension_trajectory` (nd1) is structural, not quality | 🟡 |
| Longform: per-chapter slop/repetition | v2 metrics apply per run, not per segment; st1 could host per-unit variants | 🟡 |
| Anti-purple-prose / forced-metaphor penalties (incl. `5 x ForcedPoetry^1.7` and a hard-coded single-sentence-paragraph penalty) | none | ❌ |

## 2. What this repo measures that EQ-Bench cannot

| This repo's metric | Why EQ-Bench has no counterpart |
|---|---|
| `semantic_similarity`, `text_similarity` (v1), `ngram_diversity`, `opening_lines`, `phonetic_names` (v2), entity/name reuse (`entity_analysis`) | The cross-run axis. EQ-Bench generates 3 iterations per prompt but never compares them to each other: convergence, name reuse, and opening-line collapse across repeats are invisible to it. Mode collapse is simply not measured. |
| nd1 `tension_trajectory`, `block_rhythm`, `thread_architecture` | No structural measurement at all. Their own docs admit "judges often fail to recognize" structural degradation, patched with the single-sentence-paragraph penalty; that admission validates dedicated per-unit structural metrics. |
| st1 / `--text` scoring-only mode | EQ-Bench is generation-coupled (OpenRouter only); it cannot score an existing corpus, arbitrary text, or a segmented book. |
| Frozen manifests + sidecars (longitudinal integrity) | Their leaderboard mutates under judge upgrades and Elo re-anchoring; it is a snapshot, not a time series. |
| Judge-free v1/v2 mechanical metrics | Their headline numbers depend on a Claude judge ranking a leaderboard containing Claude models. |

## 3. Their acknowledged limitations (their words, roughly)

Uncontrolled: judge self-bias, positivity bias, NSFW aversion ("punish this
severely"), stylistic preference divergence from humans, and slop selection
pressure (labs RL-tune against LLM-judge preferences, so a judge-based
leaderboard partly measures judge-gaming). English-only; "scores and rankings
should only ever be interpreted as a rough guide." Mitigations they do run:
bidirectional A|B judging, 4000-char truncation for pairwise fairness,
explicit anti-verbosity/purple-prose criteria.

Contrast with this repo's judge discipline: nd1 confines all LLM judgement
behind `ctx["judge"]` with versioned rubrics whose reliability must be
re-verified in-harness before findings are trusted, and the mechanical
v1/v2 layers need no judge at all.

## 4. What is worth borrowing (nd2/v3 candidates)

1. **Corpus-derived slop lexicon.** Their one clearly superior artifact.
   Derive a frozen, versioned lexicon from over-represented n-grams in our
   own `results/` corpus (or ingest theirs as a separate versioned lexicon)
   and score like `cliche_density`. Fits the `LEXICON_VERSION` discipline
   exactly.
2. **Per-unit quality-decay curve** (longform "degradation"), done properly:
   per-unit rubric scores over st1/nd segmentation units instead of a
   holistic judge impression plus ad-hoc penalties.
3. **Pairwise comparison as a compression fix.** Their Elo layer exists
   because rubric scores compress at the top of the ability range. If nd1
   rubric scores show compression across models, margin-weighted pairwise
   judging per rubric dimension is the known remedy.
4. **Vocab complexity** (3+ syllable proportion): trivially cheap, zero-dep,
   targets a real failure mode (vocab inflation) that `vocabulary_diversity`
   does not isolate.
5. **Structural-tic detectors** as mechanical penalties: single-sentence-
   paragraph rate is a zero-dep paragraph-shape statistic that slots next to
   `block_rhythm`.

## Takeaways

No competition on the core ground: EQ-Bench answers "who writes better, per
an LLM judge" while this repo answers "what does this model do." The two
overlap only on slop/repetition, where our versions are deeper and theirs
has one better-constructed input (the empirical lexicon). Their concessions
(judges miss structural degradation, judge self-bias uncontrolled, scores
reset on judge upgrades) independently validate this repo's three design
commitments: judge-free mechanical layers, versioned rubrics with in-harness
reliability checks, and frozen manifests for longitudinal comparability.
Same pattern as the StoryScope comparison (storyscope_comparison.md): the
credible third parties keep validating the shortlisted gaps rather than
occupying the core axis.
