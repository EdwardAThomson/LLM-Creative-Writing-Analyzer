# nd1 corpus anomaly report

QA surfacing report only. Nothing here is asserted to be a bug — each item is tagged EXPECTED (matches a pre-declared known limitation) or REVIEW (novel, needs a human look).

## Load status

- Sidecars loaded: 21
- Load failures: 0
- Judge model(s) observed: ai_helper:openrouter:deepseek/deepseek-chat
  - Single judge across the corpus (expected at this stage — nd1 is currently a DeepSeek-only run; no cross-judge comparison possible yet).

## Integrity findings (per book)

Total: 9

| book | metric | kind | tag | detail |
|---|---|---|---|---|
| austen-persuasion | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| austen-pride | paragraphs_annotated | incomplete_paragraph_annotation | REVIEW | paragraphs_annotated(2079) < paragraphs_total(2095); n_unlabeled=16 |
| bronte-janeeyre | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| buchan-thirtyninesteps | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| conrad-heartofdarkness | th_first_convergence_position | null_or_nan | EXPECTED [conrad-heartofdarkness-3units] | th_first_convergence_position is missing/None/NaN (value=None) |
| conrad-secretagent | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| haggard-ksm | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| sabatini-captainblood | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| wells-warofworlds | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |

## Cross-corpus outlier flags (|z| > 2.5)

| book | metric | value | z-score | tag |
|---|---|---|---|---|
| conrad-heartofdarkness | br_character_desc_share | 0.091 | 3.14 | EXPECTED [conrad-heartofdarkness-3units] |
| austen-persuasion | br_interiority_share | 0.186 | 2.57 | REVIEW |
| stoker-dracula | br_transition_share | 0.112 | 4.20 | REVIEW |
| conrad-lordjim | br_words_per_mode_segment | 323.6 | 2.91 | REVIEW |
| austen-persuasion | tension_max | 6 | -2.79 | REVIEW |
| buchan-thirtyninesteps | tension_min | 4 | 2.73 | REVIEW |
| austen-persuasion | tension_peak | 6 | -2.79 | REVIEW |
| austen-pride | th_convergence_events | 33 | 2.86 | REVIEW |
| haggard-ksm | th_run_length_mean | 7 | 3.09 | REVIEW |
| dickens-taleoftwocities | th_threads_total | 24 | 2.63 | REVIEW |

## Hard flags (always reported, regardless of SD)

| book | metric | value | tag |
|---|---|---|---|
| austen-pride | complete | False | REVIEW |

## Known, pre-declared limitations (not bugs)

- **[conrad-heartofdarkness-3units]** (scope: conrad-heartofdarkness; metrics: any metric) — conrad-heartofdarkness is segmented into only 3 units (its 3 long parts), so its per-unit stats (tension std/min/max, thread run lengths, block-rhythm switch rate, etc.) are extreme-but-expected, not bugs.
- **[zero-convergence-null-position]** (scope: any book; metrics: th_first_convergence_position) — th_first_convergence_position is null exactly when n_convergence_events == 0 (verified across the corpus: no thread ever converges, so 'position of first convergence' is undefined) — a well-defined structural null, not missing data.
- **[partial-corpus]** (scope: any book; metrics: none — corpus-level) — this nd1 run is in progress; 5 'giant' books (very long, expensive to judge) are pending and simply absent from this table. Their absence is expected scope, not a load failure.

## Distribution summary (one line per metric)

- **units_scored**: min=3 (conrad-heartofdarkness), median=29, max=61 (austen-pride); mean=31.38, sd=15.55, n=21
- **units_segmented**: min=3 (conrad-heartofdarkness), median=29, max=62 (austen-pride); mean=31.86, sd=15.67, n=21
- **total_words**: min=32361 (wells-timemachine), median=112073, max=194601 (collins-moonstone); mean=1.101e+05, sd=4.742e+04, n=21
- **paragraphs_annotated**: min=198 (conrad-heartofdarkness), median=1724, max=4048 (bronte-janeeyre); mean=1833, sd=1095, n=21
- **paragraphs_total**: min=198 (conrad-heartofdarkness), median=1724, max=4048 (bronte-janeeyre); mean=1834, sd=1096, n=21
- **tension_mean**: min=2.96 (austen-emma), median=5.25, max=7 (sabatini-captainblood); mean=5.346, sd=1.239, n=21
- **tension_std**: min=1.07 (austen-persuasion), median=2.11, max=2.49 (wells-timemachine); mean=1.912, sd=0.4231, n=21
- **tension_min**: min=1 (austen-emma), median=2, max=4 (buchan-thirtyninesteps); mean=1.905, sd=0.7684, n=21
- **tension_max**: min=6 (austen-persuasion), median=9, max=9 (wells-warofworlds); mean=8.429, sd=0.8701, n=21
- **tension_peak**: min=6 (austen-persuasion), median=9, max=9 (wells-warofworlds); mean=8.429, sd=0.8701, n=21
- **tension_peak_position**: min=0.045 (eddison-ouroboros), median=0.554, max=0.982 (childers-riddlesands); mean=0.5045, sd=0.308, n=21
- **tension_calm_share**: min=0 (buchan-thirtyninesteps), median=0.31, max=0.82 (austen-emma); mean=0.3338, sd=0.2247, n=21
- **tension_high_share**: min=0 (austen-emma), median=0.23, max=0.46 (stoker-dracula); mean=0.2324, sd=0.1723, n=21
- **tension_volatility**: min=0.95 (austen-pride), median=1.83, max=2.5 (conrad-heartofdarkness); mean=1.781, sd=0.499, n=21
- **tension_tail_mean**: min=2.5 (haggard-ksm), median=6, max=8.5 (dickens-taleoftwocities); mean=5.504, sd=1.971, n=21
- **tension_tail_final**: min=1 (austen-pride), median=4, max=9 (dickens-taleoftwocities); mean=4.429, sd=2.712, n=21
- **br_setting_share**: min=0.006 (austen-emma), median=0.049, max=0.151 (wells-warofworlds); mean=0.05829, sd=0.04604, n=21
- **br_character_desc_share**: min=0.011 (stoker-dracula), median=0.026, max=0.091 (conrad-heartofdarkness); mean=0.031, sd=0.0191, n=21
- **br_lore_share**: min=0.01 (bronte-janeeyre), median=0.035, max=0.091 (dunsany-elfland); mean=0.04071, sd=0.02683, n=21
- **br_dialogue_share**: min=0.257 (wells-warofworlds), median=0.508, max=0.684 (bronte-janeeyre); mean=0.51, sd=0.1247, n=21
- **br_action_share**: min=0.156 (bronte-janeeyre), median=0.238, max=0.4 (wells-warofworlds); mean=0.2524, sd=0.06472, n=21
- **br_interiority_share**: min=0.019 (eddison-ouroboros), median=0.093, max=0.186 (austen-persuasion); mean=0.09557, sd=0.03521, n=21
- **br_transition_share**: min=0 (conrad-secretagent), median=0.005, max=0.112 (stoker-dracula); mean=0.0119, sd=0.02381, n=21
- **br_switch_rate**: min=0.398 (austen-emma), median=0.53, max=0.677 (conrad-heartofdarkness); mean=0.5143, sd=0.07918, n=21
- **br_words_per_mode_segment**: min=79.6 (sabatini-captainblood), median=126.9, max=323.6 (conrad-lordjim); mean=140.6, sd=62.93, n=21
- **br_interiority_self_transition**: min=0.083 (dunsany-elfland), median=0.177, max=0.294 (austen-persuasion); mean=0.1923, sd=0.05979, n=21
- **br_secondary_shading_rate**: min=0.106 (austen-pride), median=0.243, max=0.531 (conrad-lordjim); mean=0.2592, sd=0.1307, n=21
- **br_setting_touch_rate**: min=0.009 (austen-emma), median=0.088, max=0.271 (wells-warofworlds); mean=0.1034, sd=0.08032, n=21
- **th_threads_total**: min=2 (conrad-heartofdarkness), median=7, max=24 (dickens-taleoftwocities); mean=8.524, sd=5.887, n=21
- **th_threads_2plus**: min=1 (conrad-heartofdarkness), median=4, max=7 (sabatini-scaramouche); mean=4.095, sd=2.022, n=21
- **th_switch_rate**: min=0.1 (haggard-ksm), median=0.5, max=1 (conrad-secretagent); mean=0.5215, sd=0.2402, n=21
- **th_run_length_mean**: min=1 (conrad-secretagent), median=1.93, max=7 (haggard-ksm); mean=2.348, sd=1.505, n=21
- **th_run_length_max**: min=1 (conrad-secretagent), median=7, max=15 (austen-emma); mean=6.667, sd=3.851, n=21
- **th_convergence_events**: min=0 (austen-persuasion), median=2, max=33 (austen-pride); mean=6.714, sd=9.182, n=21
- **th_first_convergence_position**: min=0.17 (austen-pride), median=0.54, max=0.97 (wells-timemachine); mean=0.5469, sd=0.2699, n=13

