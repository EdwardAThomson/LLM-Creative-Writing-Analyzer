# Newest-Generation Models under v2 — 2026 Cohort Report

_Companion to [report.md](report.md) (v1) and [report_v2.md](report_v2.md) (full v2).
Status: draft · 2026-06-19_

## 1. Scope

This report looks only at the **newest model generation** in the corpus, scored
with the [v2 metric set](report_v2.md). These six were generated through local
agent CLIs in headless mode (the "2026 CLI cohort"); each resolves to a current
flagship model:

| Run key | Underlying model |
| :-- | :-- |
| `codex-cli` | OpenAI **GPT-5.5** (Codex CLI default at the time of the June 2026 runs) |
| `claude-cli-opus` | Anthropic **Claude Opus 4.8** |
| `claude-cli-sonnet` | Anthropic **Claude Sonnet 4.6** |
| `claude-cli-fable` | Anthropic **Claude Fable 5** (creative-writing tier) |
| `gemini-cli-pro` | Google **Gemini 3.x** (3.0 or 3.1; the Gemini CLI "pro" default, not pinned in these runs) |
| `gemini-cli-flash` | Google **Gemini 3.x** (3.0 or 3.1; the Gemini CLI "flash" default, not pinned) |

Ten runs per model, same science-fiction-opening prompt as the rest of the study.

### Caveats specific to this cohort

- **CLI sampling defaults.** None of these runs forward `temperature`/`max_tokens`;
  each CLI uses its own defaults. They are therefore **not strictly comparable** to
  the 2025 API runs, and any generational comparison (§5) is directional only.
- **Within-cohort comparability is better but imperfect.** All six share the
  "CLI-default" condition, so comparing them to *each other* is more meaningful than
  comparing to the API cohort — but three different vendors' CLIs still differ in
  ways we don't control.
- **Length-sensitive metrics.** `intra_text_repetition` and `distinct-n` track
  document length; codex-cli's high intra-repetition partly reflects its 2522-word
  outputs (see [report_v2.md](report_v2.md) §2.3).

## 2. Results

Aggregates across 10 runs. **cv** = burstiness CV; **B** = Goh–Barabási rhythm
coefficient; **uni** = intra-text unigram repeat rate; **clich/slop/em** = cliché /
slop-word / em-dash hits per 1000 words; **sBLEU** = Self-BLEU; **d2** = distinct-2;
**opPfx** = largest share of runs sharing an opening prefix; **opSem** = mean
opening-sentence cosine.

| Model | Words | MTLD | cv | B | dlg | uni | clich | slop | em | sBLEU | d2 | opPfx | opSem |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| GPT-5.5 (codex-cli) | 2522 | **162.3** | 0.77 | −0.13 | 0.08 | **0.39** | 0.04 | **0.00** | **0.08** | 0.130 | 0.71 | 3/10 | 0.318 |
| Opus 4.8 (claude-cli) | 1495 | 95.6 | 0.89 | −0.06 | 0.09 | 0.35 | 0.00 | 0.07 | 12.94 | 0.176 | 0.67 | 2/10 | 0.295 |
| Sonnet 4.6 (claude-cli) | 1543 | 125.6 | 0.92 | −0.05 | 0.06 | 0.32 | 0.13 | 0.00 | 13.54 | 0.138 | 0.72 | **1/10** | 0.268 |
| Fable 5 (claude-cli) | 1479 | 109.5 | **0.98** | **−0.01** | 0.08 | 0.34 | 0.00 | 0.46 | **14.06** | 0.149 | 0.70 | 4/10 | 0.313 |
| Gemini pro (gemini-cli) | 1901 | 133.8 | 0.59 | −0.26 | 0.06 | 0.27 | 0.28 | 1.78 | 4.90 | 0.187 | 0.68 | 2/10 | 0.291 |
| Gemini flash (gemini-cli) | 1658 | 93.2 | 0.65 | −0.21 | 0.11 | 0.31 | **0.53** | 1.74 | 7.22 | **0.223** | **0.64** | **7/10** | **0.408** |

## 3. Per-vendor findings

### 3.1 OpenAI GPT-5.5 — the clean stylist

GPT-5.5 (via codex-cli) is conspicuously **free of the stock-prose markers**: cliché
density 0.04/1k, **zero** slop-words, and **near-zero em-dashes** (0.08/1k). It also
produced the longest outputs (2522 words) with the highest CLI-cohort MTLD (162).
Its one elevated number — intra-text unigram repetition (0.39) — is a length
artifact, not a tell (longer texts mechanically repeat more content words). The
profile is restrained, varied prose with no reliance on cliché or punctuation crutches.

### 3.2 Anthropic (Opus 4.8 / Sonnet 4.6 / Fable 5) — the em-dash house style

The defining Anthropic trait is **em-dash density**: 12.9–14.1 per 1000 words across
all three tiers, far above Gemini (4.9–7.2) and GPT-5.5 (0.08). All three are also
clean of cliché (0.00–0.13/1k). They differ by tier on rhythm:

- **Fable 5** has the **most natural sentence rhythm** in the entire study — burstiness
  CV 0.98 and B −0.01, i.e. essentially the variation of a random (human-like)
  process rather than the metronomic cadence every other model shows. Fitting for the
  creative-writing tier. It does lean on sensory slop-words (`beacon` ×4, `gleaming`,
  `shimmering`) and emits "Chapter One" scaffolding.
- **Sonnet 4.6** has the **most varied openings** of any model (1/10 shared prefix,
  lowest opening cosine 0.268) and high rhythm variation (CV 0.92).
- **Opus 4.8** sits lowest on lexical diversity within the cohort (MTLD 95.6) but
  high on rhythm (CV 0.89); it prepends "Book One of …" title scaffolding.

### 3.3 Google Gemini — carries the classic "AI words"

The stereotypical LLM register survives in Gemini, not in OpenAI/Anthropic here:
`gemini-cli-pro`'s top slop-words are **tapestry, shimmering, gleaming, intricate**,
and `gemini-cli-flash` leans hard on **shimmering** (16 occurrences across 10 runs).
The two also have the **flattest rhythm** of the cohort (CV 0.59 / 0.65; most
negative B). **Gemini flash is the most formulaic newest model** on every diversity
axis: highest Self-BLEU (0.223 — most repetition across its runs), lowest distinct-2
(0.64), the strongest opening tell (**7/10** runs open "The sky over…", opening
cosine 0.408), and the cohort's highest cliché density (0.53/1k).

## 4. Cohort summary

- **Cliché/slop** cleanly ranks the vendors: GPT-5.5 ≈ Anthropic (near-zero) <
  Gemini (the AI-word carrier). Gemini flash is the only newest model with a
  meaningful cliché rate.
- **Em-dash** is a vendor signature: Anthropic heavy, Gemini moderate, OpenAI bare.
- **Rhythm** splits Anthropic (varied, CV 0.89–0.98) from Gemini (flat, 0.59–0.65),
  with GPT-5.5 in between (0.77).
- **Cross-run repetition**: Gemini flash most formulaic; the rest moderate and
  similar (Self-BLEU 0.13–0.19).

## 5. Generational note (directional — confounded by API→CLI)

The starkest apparent shift is OpenAI: 2025's **GPT-4o** was the study's *generic-prose
outlier* (cliché 2.83/1k, slop 6.17/1k — both ~4×+ the field; flattest rhythm), while
2026's **GPT-5.5** is the *cleanest* model on exactly those axes (0.04 and 0.00). If
real, that is a large generational change in default prose style. **But** the
comparison crosses the API→CLI boundary, so part of the gap may be sampling/CLI
defaults rather than the model. The Anthropic em-dash habit, by contrast, appears in
*both* the 2025 API runs (Claude 3.7 at 12.2/1k) and the 2026 CLI runs (12.9–14.1) —
its persistence across the path change makes it the most credible durable trait.

## 6. Limitations and next step

All §3–5 statements describe *these CLI-default runs*; they are not controlled
comparisons. The single highest-value follow-up is the **API-at-fixed-temperature
re-run of the newest models** (GPT-5.5, Claude Opus 4.8 / Fable 5, current Gemini
Pro/Flash): scoring those with v2 alongside the 2025 API cohort would remove the
CLI/sampling confound and turn the generational and vendor comparisons here into
properly controlled ones.
