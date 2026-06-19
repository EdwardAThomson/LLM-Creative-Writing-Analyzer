# Combined v1 + v2 Metrics — Master Fingerprint Table

_Reference companion to [report.md](report.md) (v1), [report_v2.md](report_v2.md)
(full v2), and [report_2026_cohort.md](report_2026_cohort.md) (newest models).
Status: draft · 2026-06-19_

One row per model (13 models, 10 runs each), combining both metric layers:

- **v1** (from each `results_*.json`'s `analysis` block — the frozen pipeline):
  cross-run similarity, vocabulary, semantics, entity/name reuse.
- **v2** (from the `results_*.metrics.json` sidecars): the craft + n-gram + opening
  layers.

Regenerate the underlying data with `python -m utils.metrics results/ --benchmark v2`
(v2 sidecars) — the v1 `analysis` blocks are written at generation time.

## Column legend

| Column | Layer | Meaning (direction for "more varied / less generic") |
| :-- | :-- | :-- |
| **TxtSim** | v1 | mean pairwise text similarity across runs (lower = more varied) |
| **VocDiv** | v1 | vocabulary diversity = unique/total words (TTR-like; **length-sensitive**) |
| **SemSim** | v1 | mean pairwise semantic (embedding) similarity (lower = more thematic spread) |
| **EntOvlp** | v1 | mean named-entity overlap across runs (lower = less entity reuse) |
| **Top name** | v1 | most-reused name component (runs/10) — **raw, includes NER false positives** |
| **MTLD** | v2 | length-robust lexical diversity (higher = richer vocab) |
| **cv** | v2 | sentence-length variation / rhythm (higher = more varied) |
| **dlg** | v2 | dialogue fraction (words in quotes / total) |
| **clich** | v2 | cliché-phrase hits per 1000 words (lower = less generic) |
| **em** | v2 | em-dash density per 1000 words |
| **sBLEU** | v2 | Self-BLEU across runs (lower = more cross-run variety) |
| **opPfx** | v2 | largest share of runs sharing an opening prefix (lower = more varied openings) |

## Table

| Model | TxtSim | VocDiv | SemSim | EntOvlp | Top name | MTLD | cv | dlg | clich | em | sBLEU | opPfx |
| :-- | --: | --: | --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: |
| GPT-4o | 0.028 | 0.298 | 0.596 | 0.082 | Elara 7/10 | 110.59 | 0.465 | 0.111 | 2.83 | 4.16 | 0.2179 | 3/10 |
| o1 | 0.015 | 0.269 | 0.508 | 0.08 | metropolis 2/10 | 210.53 | 0.534 | 0.103 | 0.67 | 8.18 | 0.1188 | 1/10 |
| o3 | 0.016 | 0.376 | 0.491 | 0.062 | Kael 3/10 | 327.26 | 0.69 | 0.101 | 0.0 | 13.95 | 0.0536 | 1/10 |
| Gemini 2.0 Pro | 0.023 | 0.349 | 0.612 | 0.088 | Kaelen 7/10 | 202.27 | 0.724 | 0.031 | 0.31 | 0.0 | 0.1315 | 3/10 |
| Gemini 2.5 Pro | 0.019 | 0.347 | 0.556 | 0.072 | Kaelen 5/10 | 218.68 | 0.794 | 0.067 | 0.17 | 0.23 | 0.1129 | 3/10 |
| Claude 3.5 Sonnet | 0.037 | 0.327 | 0.648 | 0.08 | Chen 8/10 | 204.93 | 0.601 | 0.193 | 0.7 | 3.44 | 0.1567 | 8/10 |
| Claude 3.7 Sonnet | 0.018 | 0.347 | 0.552 | 0.054 | Elara 7/10 | 207.0 | 0.672 | 0.257 | 0.49 | 12.21 | 0.0814 | 3/10 |
| _2026 CLI_ | | | | | | | | | | | | |
| GPT-5.5 (codex-cli) | 0.01 | 0.241 | 0.476 | 0.089 | Mara 7/10 | 162.29 | 0.767 | 0.081 | 0.04 | 0.08 | 0.13 | 3/10 |
| Gemini 3.x flash (cli) | 0.026 | 0.28 | 0.55 | 0.072 | Elara 6/10 | 93.17 | 0.649 | 0.114 | 0.53 | 7.22 | 0.2231 | 7/10 |
| Gemini 3.x pro (cli) | 0.019 | 0.29 | 0.608 | 0.097 | Kaelen 6/10 | 133.84 | 0.585 | 0.055 | 0.28 | 4.9 | 0.1871 | 2/10 |
| Opus 4.8 (cli) | 0.023 | 0.25 | 0.555 | 0.075 | Minds 2/10 | 95.58 | 0.887 | 0.085 | 0.0 | 12.94 | 0.1756 | 2/10 |
| Sonnet 4.6 (cli) | 0.02 | 0.278 | 0.508 | 0.072 | Voss 7/10 | 125.62 | 0.92 | 0.056 | 0.13 | 13.54 | 0.1383 | 1/10 |
| Fable 5 (cli) | 0.023 | 0.261 | 0.511 | 0.053 | Continuum 3/10 | 109.49 | 0.977 | 0.075 | 0.0 | 14.06 | 0.1487 | 4/10 |

## How to read it (caveats)

- **CLI vs API confound.** The 2026 CLI rows used each CLI's default sampling, not
  the API cohort's fixed temperature — so the two blocks are **not strictly
  comparable**. See [report_v2.md](report_v2.md) §2.3.
- **`Top name` is unfiltered.** It is the raw v1 name-component count and inherits
  spaCy `PERSON` false positives — `metropolis` (o1), `Minds` (Opus 4.8), and
  `Continuum` (Fable 5) are mislabels, not character names. But it also catches real
  signals: `Voss 7/10` (Sonnet 4.6) is the cross-vendor V-surname of report §6.6, and
  `Elara`/`Kaelen`/`Chen` are genuine model-favoured names. Verify against raw PERSON
  lists before quoting (report §4.4).
- **VocDiv vs MTLD.** VocDiv (v1) falls as texts lengthen; MTLD (v2) is length-robust.
  Where they disagree (e.g. o3: VocDiv 0.376 but MTLD 327 — both high; o1: VocDiv 0.269
  low yet MTLD 210 high, because o1's texts are long) trust MTLD for cross-model vocab
  comparison.
- **TxtSim vs SemSim vs sBLEU.** Three diversity layers: exact (TxtSim, all <0.04),
  n-gram surface (sBLEU), and meaning (SemSim). They can diverge — Claude 3.5 Sonnet
  has the highest SemSim (0.648) yet mid sBLEU, i.e. thematically convergent but not
  phrase-repetitive.
- These remain **automated proxies**, not a quality score; see the craft-proxy
  scorecard in [report_v2.md](report_v2.md) §4.7 and the pending LLM-judge.
