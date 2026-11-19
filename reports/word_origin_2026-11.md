# Word origin across the saved corpus (2026-11-18)

The new library metric `word_origin` (`utils/metrics/word_origin.py`) counts
how much of each run comes from French and Latin rather than Old English. It is
not yet in a frozen manifest (see METRICS_ROADMAP, v3 Track B), so no sidecar
was rewritten. These numbers come from calling the module directly on every
`results_*.json` with at least 150 words per run. Zero LLM calls.

Reference points for `latinate_pct`, which is counted over every word:

- Tolkien 8% and Sanderson 25%, from a five-author table on The Legendarium
  channel (one short passage each).
- Morris, Dunsany and Eddison about 10%, Austen and Ferrier about 19–20%. These
  are whole books, measured with the same table in StoryDaemon.

All runs below answer the same sci-fi space-opera brief (`parameters.txt`).
Sci-fi vocabulary is Latin and Greek, which lifts every model, so compare the
models with each other, not with the novels.

| Model | Run | Runs | Latinate (mean) | Range | Content words only | Words placed | Most used French/Latin content words |
|---|---|---|---|---|---|---|---|
| claude-cli-opus | `claude_cli_10x/results_20260601_195600.json` | 10 | **15.3%** | 12.7–17.1 | 32.3% | 95.8% | human, across, voice, machine, single, because, person, second |
| claude-cli-fable | `results_20260610_134353.json` | 3 | **15.8%** | 14.2–17.4 | 32.1% | 94.6% | because, question, space, human, past, farm, voice, captain |
| claude-cli-fable | `results_20260610_140135.json` | 10 | **16.0%** | 13.0–17.5 | 33.1% | 95.0% | because, voice, machine, lattice, colony, people, human, across |
| claude-cli-fable | `results_20260610_183156.json` | 1 | **19.2%** | 19.2–19.2 | 38.2% | 94.5% | terraces, machine, because, aquifer, human, province, violet, across |
| codex-cli | `codex_cli_10x/results_20260601_233230.json` | 10 | **19.9%** | 17.3–20.9 | 36.9% | 93.5% | machine, city, voice, around, across, blue, alien, machines |
| gemini-cli-flash | `gemini_cli_10x/results_20260601_185808.json` | 10 | **20.1%** | 17.2–23.2 | 43.4% | 91.3% | voice, machine, power, air, consciousness, human, machines, data |
| claude-cli-sonnet | `claude_cli_10x/results_20260601_195600.json` | 10 | **20.8%** | 17.6–23.8 | 41.8% | 93.6% | voice, colony, colonial, across, atmospheric, signal, human, amber |
| gemini-cli-flash | `results_20260601_181933.json` | 3 | **21.1%** | 20.2–22.2 | 43.2% | 91.3% | voice, human, air, machine, blue, city, machines, power |
| gpt-4o | `results_20250326_153045.json` | 10 | **22.5%** | 19.7–26.3 | 47.7% | 92.2% | journey, city, air, voice, power, across, universe, human |
| gemini-cli-pro | `gemini_cli_10x/results_20260601_185808.json` | 10 | **25.6%** | 23.1–28.7 | 50.5% | 88.5% | machine, consciousness, voice, human, alien, air, terrifying, power |
| o3 | `results_20250506_132713.json` | 10 | **25.7%** | 22.6–27.2 | 45.2% | 86.4% | across, voice, human, around, alien, data, planetary, sphere |
| gemini-cli-pro | `results_20260601_181933.json` | 3 | **25.7%** | 23.9–26.7 | 50.1% | 88.6% | human, consciousness, voice, vessel, war, liquid, survival, air |
| claude-3-5-sonnet | `results_20250326_185105.json` | 10 | **26.0%** | 23.0–28.9 | 47.3% | 92.4% | quantum, across, voice, human, grain, massive, space, around |
| o1 | `results_20250327_111016.json` | 10 | **26.6%** | 24.9–28.8 | 50.0% | 89.7% | farmland, alien, across, city, advanced, around, voice, rogue |
| claude-3-7-sonnet | `results_20250326_190128.json` | 10 | **29.6%** | 25.2–33.4 | 54.5% | 91.6% | voice, data, across, human, implant, responds, captain, security |
| gemini-2.5-pro-exp-03-25 | `results_20250326_182702.json` | 10 | **30.9%** | 25.1–35.1 | 53.8% | 87.3% | data, standard, across, city, console, voice, air, alien |
| gemini-2.0-pro-exp-02-05 | `results_20250326_162331.json` | 10 | **31.7%** | 26.1–37.5 | 56.4% | 87.4% | data, standard, voice, across, level, digital, faint, air |

## Reading

- **The models separate clearly.** The 2026 Claude models write the plainest
  English (about 15–16%). Codex, Gemini Flash and Sonnet come next (about
  20–21%), then Gemini Pro (26%), all June 2026 CLI runs. The early-2025 cohort sits at
  23–32%. Their per-run ranges barely overlap, so this is a model fingerprint
  rather than noise.
- **What lifts the high scorers:** genre words (*data, standard, console,
  implant, atmospheric, consciousness*) and abstract ones (*destiny,
  testament, universe, sense*). The plainest models write *voice, face, soil,
  grain* instead.
- **Words placed** is lower for the older Gemini and o-series runs (86–88%):
  more coined sci-fi words that no table knows. The share is computed over
  the words placed, so this lowers precision, not direction.

## Method

The frozen table is `utils/metrics/_data/word_origin.json`, `TABLE_VERSION`
1. It was built in StoryDaemon (`scripts/build_word_origin_table.py`) from
the Etymological Wordnet (Wiktionary, CC BY-SA 3.0) plus about 1,500 hand
labels for its gaps and errors. The labels were checked blind against 300
words the Wordnet resolves, and 96% agreed. Names are left out. Words the
table lacks are tried without their ending, then by a Latin suffix rule.
