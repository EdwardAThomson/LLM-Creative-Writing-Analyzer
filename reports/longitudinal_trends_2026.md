# Longitudinal Trends 2025 → 2026 (analysis notes)

Working notes for the video / Q&A. Goes beyond the name story to ask: **what changed, what stayed the same, and what got *worse*** across each model family. All figures from the two result tables in `report.md` (Sections 3.1 and 6.2); derived ratios are computed from those.

## Important framing (say this if challenged)

- **The cohorts aren't 1:1.** 2025 = GPT-4o/o1, Gemini 2.0/2.5, Claude 3.5/3.7. 2026 = GPT-5.5(Codex), Gemini 3 Flash/Pro, Claude Opus 4.8/Sonnet. I compare *family to family*, not model to model.
- **Backend changed (CLI vs API), so temperature isn't pinned in 2026.** Trends that show up *within* a single family across years are the safe ones to lean on; tiny cross-family gaps are not.
- **Two metrics are length-sensitive** and must be read with per-run word count in mind: *vocabulary diversity* (unique/total words) and raw *entity counts*. Where I claim a real change I've checked it against a near-constant-length pair.

---

## The one-screen summary

| Trend | Direction | Confidence |
| :-- | :-- | :-- |
| Verbatim repetition (text similarity) | **Unchanged** — still ~0 everywhere | High |
| Thematic repetition (semantic similarity) | **Roughly unchanged** — same 0.48–0.61 band | High |
| Name repetition | **Split by vendor** (the headline) | High |
| Length adherence | **Improved** for non-reasoning models; OpenAI still overshoots | High |
| Vocabulary diversity (relative to length) | **Got worse** — dropped across *every* family | Medium-High |
| Name-component diversity (unique/total) | **Up for Anthropic, down for Google** | Medium-High |
| Entity overlap | **Flat-to-slightly-up** — Gemini 3 Pro is the highest of either year | Medium |
| Phonetic name clustering ("V-surnames") | **Cross-vendor & persistent** — present even where exact names don't repeat | High |

The two-sentence version: **the things that were already fine in 2025 (verbatim copying, length for some) stayed fine or improved; the headline name problem fragmented by vendor; and one thing quietly got *worse for everyone* — lexical richness per word.** Underneath all of it sits one regularity nobody escaped: every vendor, in both years, favours the same *sound* for surnames (Vance / Voss / Venn / Vale) — the subject of cross-cutting trend #6.

---

## Per-family deep dive

### OpenAI — GPT-4o / o1 → GPT-5.5 (Codex)

| Metric | 2025 (GPT-4o / o1) | 2026 (GPT-5.5) | Read |
| :-- | :-- | :-- | :-- |
| Top name | Elara 7/10 | Aster 7/10, Mara 7/10 | rotated, not fixed |
| Words/run | 1042 / 2935 | 2481 | still overshoots |
| Semantic sim | 0.596 / 0.508 | **0.476** (lowest ever) | most thematically varied |
| Text sim | 0.028 / 0.015 | **0.0097** (lowest ever) | least verbatim repetition |
| Vocab diversity | 0.298 / 0.269 | 0.241 | down (but longer, so partly length) |

**Story: changed its names, kept its personality.** OpenAI is the most thematically adventurous family in *both* years (o1 lowest semantic sim in 2025, Codex lowest ever in 2026) — and also the most verbose (o1 2935 → Codex 2481, both ~65% over target). So the "reasoning-style overshoot" did **not** go away. And the name fixation is as strong as ever — it just swapped Elara for Aster/Mara/Venn. **Nothing here really got better on the repetition axis; it moved sideways.** New this year: Codex's choppy structure (48.9 paragraphs/1000 words, ~10 words/sentence) — by far the most fragmented prose in the set.

### Google — Gemini 2.0 / 2.5 → Gemini 3 Flash / Pro

| Metric | 2025 (2.0 / 2.5) | 2026 (Flash / Pro) | Read |
| :-- | :-- | :-- | :-- |
| Top name | Kaelen 7/10 | Kaelen 6/10 | **identical signature** |
| Surnames | Vance present | Vance 5–6/10 | persists across 2 gens |
| Semantic sim | 0.612 / 0.556 | 0.550 / 0.608 | non-monotonic; 2.5 was the peak |
| Entity overlap | 0.088 / 0.072 | 0.072 / **0.097** | Pro = highest of either year |
| Name diversity (unique/total components) | 0.272 / **0.293** | 0.158 / 0.169 | **dropped sharply** |
| Vocab diversity | 0.350 / 0.347 | 0.280 / 0.290 | down at ~constant length |

**Story: the most stable family — and not entirely in a good way.** Same favourite names across two generations (Kaelen, Elara, even the surname Vance). But dig past the names-list and Gemini actually **regressed on diversity from 2.5 to 3**: the unique/total name-component ratio fell from 0.293 (the *best* of 2025) to ~0.17, entity overlap rose to 0.097 (the *highest of either year*), and semantic similarity ticked back up from 0.556 to 0.608. **Gemini 2.5 was the diversity high-water mark of the whole project; Gemini 3 traded some of that back for consistency.** This is the cleanest "newer ≠ more varied" example.

### Anthropic — Claude 3.5 / 3.7 → Claude Opus 4.8 / Sonnet

| Metric | 2025 (3.5 / 3.7) | 2026 (Opus / Sonnet) | Read |
| :-- | :-- | :-- | :-- |
| Top name | Elara 7/10 (3.7) | **0 real repeats** (Opus) | the big win |
| Name diversity (unique/total components) | 0.166 / 0.255 | **0.307** / 0.229 | Opus highest of either year |
| Semantic sim | **0.648** / 0.552 | 0.555 / 0.508 | trending more diverse |
| Words/run | 1004 / 1495 | 1479 / 1547 | under-generation fixed |
| Words/sentence | — | 16.3 / 15.3 | longest sentences in 2026 set |
| Vocab diversity | 0.327 / 0.347 | 0.250 / 0.278 | down (Opus partly length) |

**Story: the clearest improvement.** Claude 3.5 was the *most* thematically repetitive model of 2025 (semantic sim 0.648) and reused Elara 7/10. The 2026 pair reversed both: Opus has a fresh cast every run (zero real repeats) and one of the best name-component diversity ratios of the project (0.307), while Sonnet has the lowest semantic similarity of any Claude (0.508). Length under-generation (3.5's 1004 words) is gone. **This is the only family that improved on *both* axes of repetition at once** — though "improved on names" means at the *token* level only; the V-sound survives (cross-cutting trend #6).

**Fable 5 — new top tier, added June 10 2026.** Run identically (same prompt, 10×, 1500 words, via the `claude` CLI). It lands squarely with Opus in the diversity camp and tops two metrics outright: the **lowest entity overlap of either study year** (0.0531, just under 3.7 Sonnet's 0.0539) and the **highest name-component diversity ratio in the whole project** (70 / 202 = **0.347**, edging Opus's 0.307 — note Opus still holds the larger raw name pool and the cleaner zero-repeat record). It keeps a faint token-level habit Opus lacks — `Venn` and `Vale` at 2/10 each, the protagonist "Yara Venn" in two runs — so it sits just short of Opus's clean break. Longest sentences in the 2026 set (17.6 w/sent), near-perfect length (1484 w). Semantic sim 0.5113 (mid-pack, ~Sonnet). Net: Fable extends the Anthropic "improved on both axes" story to a third model — and, via its V-surnames, is also Exhibit B for trend #6.

---

## The cross-cutting trends (true of (nearly) everyone)

These are the ones an audience will ask "wait, is that a real effect?" about — so here's the evidence.

### 1. Verbatim repetition was never the problem, and still isn't — UNCHANGED
Text similarity sat at 0.015–0.037 in 2025 and 0.010–0.026 in 2026. Models almost never repeat exact phrasing across runs. **This was a non-issue then and remains one.** Good myth-buster: "the models aren't copy-pasting themselves — the repetition is deeper than words."

### 2. Thematic repetition is in the same band — ROUGHLY UNCHANGED
Semantic similarity: 0.508–0.648 (2025) vs 0.476–0.608 (2026). The ceiling came down a touch (no 2026 model is as same-y as Claude 3.5's 0.648) and Codex set a new floor, but the **middle of the distribution barely moved**. Honest takeaway: *models did not get dramatically better at telling thematically different stories.*

### 3. Vocabulary diversity dropped across every family — GOT WORSE
This is the strongest "something got worse" finding, so here's the length-controlled proof (TTR falls automatically as texts lengthen, so I compare near-equal-length pairs):

- **Gemini 2.0 (1571 w/run) → Gemini 3 Flash (1607 w/run):** vocab diversity **0.350 → 0.280**. Same length, ~20% relative drop.
- **Claude 3.7 (1495 w/run) → Claude Sonnet (1547 w/run):** vocab diversity **0.347 → 0.278**. Same length, ~20% relative drop.

Two clean, near-constant-length comparisons both show the same ~20% fall. Across the *whole* set, every 2026 model sits at 0.24–0.29 vs 2025's 0.27–0.35. **Newer models reuse words more within a single story** — i.e. lower lexical richness per word. (Caveat for Codex/Opus specifically: they're also longer, so part of their drop is mechanical; the Gemini and Sonnet cases are not.)

> Possible counter-question: "isn't lower TTR just longer outputs?" — Answer: for the two pairs above the length is essentially identical, so no. It's a real narrowing.

### 4. Length adherence improved — GOT BETTER (with one holdout)
2025 spread was 1004–2935 words against a 1500 target (wild). In 2026 the Claude and Gemini models land 1479–1858 — much tighter, Opus/Sonnet the most accurate. **The chronic under-generation of 2025's non-reasoning models (GPT-4o 1042, Claude 3.5 1004) is gone.** The holdout is Codex at 2481 — OpenAI's verbose-reasoning lineage still overshoots, just as o1 did.

### 5. Entity overlap didn't fall — FLAT / SLIGHTLY UP
You'd hope the "specific cast of characters/places" got more varied. It didn't: entity overlap was 0.054–0.088 in 2025 and 0.072–0.097 in 2026, and **Gemini 3 Pro's 0.097 is the highest of either year.** So at the level of *which named things recur*, 2026 is no better and locally a bit worse. (Exception: Fable's 0.053 is a new low — but it's one model, not a trend.)

### 6. The V-sound — a phonetic favouritism that survived — CROSS-VENDOR, UNCHANGED
The one naming regularity spanning **all three vendors in both years** is phonetic, not lexical: character surnames cluster hard on a short V-sound. The home tokens — **Vance** (Google, across 2.0 → 3 Flash → 3 Pro), **Voss** (Anthropic's Sonnet line, 3.7 → 2026), **Venn**/**Vale** (OpenAI Codex *and* Fable, 2026) — differ by vendor, but the *shape* is shared. The killer datapoint is Claude Opus 4.8: with **zero** repeated names, it still hands a character a *distinct* V-surname in **8/10** runs (Vance, Vael, Voss, Veymar, Vendramin, Vahn, Veyra, Vesh). So the token-level "Anthropic broke it" win (Anthropic deep-dive) is real but narrow — **the sound didn't break.** Two consequences:

- **It revives the shared-training-data hypothesis** that the names-list seemed to kill. Exact favourites diverged by vendor (OpenAI's Aster ≠ Google's Kaelen), which looked like post-training divergence — but the *phoneme* is common to everyone, which is exactly what shared web data predicts. The signal just sits one layer down, at the sound rather than the token.
- **Detection can't be string-based.** No single token crosses both the repeat threshold and the vendor boundary, and hyphenation hides it further (Fable's `Okafor-Voss` ≠ `Voss`). Catching this needs clustering by phonetic/orthographic shape. (Full treatment: report §6.6.)

Confidence: **High** for "the cluster exists and is cross-vendor"; **Medium** for the shared-data causal story (it's the most economical explanation, not a proven one).

---

## If someone asks "so did anything actually get *worse*?" — short list

1. **Lexical richness per word fell across the board** (~20% relative at constant length for Gemini & Claude-Sonnet). The clearest regression.
2. **Gemini's internal diversity regressed from 2.5 to 3** — name-component diversity ratio 0.293 → ~0.17, entity overlap up to a project-record 0.097, semantic similarity back up. 2.5 was peak-diversity; 3 walked some of it back.
3. **OpenAI's name fixation didn't improve at all** — it's arguably the strongest in the 2026 cohort (Aster/Mara both 7/10). The favourites changed; the habit didn't.
4. **OpenAI's overshoot persisted** — Codex 2481 words echoes o1's 2935. Verbosity is a durable family trait, not a fixed bug.

## And "what stayed the same?"

1. **Verbatim repetition** — still ~0, still a non-issue.
2. **Thematic repetition band** — still 0.48–0.61; no step-change.
3. **Google's naming signature** — Kaelen / Elara / Vance, intact across two full generations.
4. **OpenAI as the most thematically varied + most verbose family** — true in both years.
5. **The V-sound surname cluster** — Vance / Voss / Venn / Vale, across all three vendors and both years; present even in Opus 4.8's otherwise-fresh casts (a distinct V-surname in 8/10 runs). The deepest "stayed the same" of all.

## The single best framing for the video

> "Three things barely moved, one thing improved, and one thing quietly got worse — and *none* of those is the same across the three companies. The story of 2026 isn't 'models got better,' it's 'models got *different from each other*.'"

---

### Derived-ratio appendix (so the numbers are reproducible)

Name-component diversity = Unique name components ÷ Total name components:

| Model | Unique | Total | Ratio |
| :-- | --: | --: | --: |
| GPT-4o (25) | 24 | 145 | 0.166 |
| o1 (25) | 81 | 305 | 0.266 |
| Gemini 2.0 (25) | 63 | 232 | 0.272 |
| Gemini 2.5 (25) | 68 | 232 | **0.293** |
| Claude 3.5 (25) | 27 | 163 | 0.166 |
| Claude 3.7 (25) | 70 | 275 | 0.255 |
| Gemini 3 Flash (26) | 48 | 304 | 0.158 |
| Gemini 3 Pro (26) | 41 | 243 | 0.169 |
| Claude Opus 4.8 (26) | 71 | 231 | 0.307 |
| Claude Sonnet (26) | 46 | 201 | 0.229 |
| Codex GPT-5.5 (26) | 90 | 515 | 0.175 |
| Claude Fable 5 (26) | 70 | 202 | **0.347** |

(Higher = more distinct name fragments per name used = more naming variety, and unlike raw vocab diversity this is far less sensitive to total length. Fable 5 edges Opus 4.8 for the project lead on this normalized metric — the two Anthropic top-tier models occupy the top two slots; note Opus still has the larger raw name pool and the cleaner zero-repeat record, so "most diverse" depends on which cut you take.)
