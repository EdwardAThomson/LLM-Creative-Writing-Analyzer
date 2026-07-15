# nd1 corpus anomaly report

QA surfacing report only. Nothing here is asserted to be a bug — each item is tagged EXPECTED (matches a pre-declared known limitation) or REVIEW (novel, needs a human look).

## Load status

- Sidecars loaded: 26
- Load failures: 0
- Judge model(s) observed: ai_helper:openrouter:deepseek/deepseek-chat
  - Single judge across the corpus (expected at this stage — nd1 is currently a DeepSeek-only run; no cross-judge comparison possible yet).

## Integrity findings (per book)

Total: 10

| book | metric | kind | tag | detail |
|---|---|---|---|---|
| austen-persuasion | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| austen-pride | paragraphs_annotated | incomplete_paragraph_annotation | REVIEW | paragraphs_annotated(2079) < paragraphs_total(2095); n_unlabeled=16 |
| bronte-janeeyre | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| buchan-thirtyninesteps | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| conrad-heartofdarkness | th_first_convergence_position | null_or_nan | EXPECTED [conrad-heartofdarkness-3units] | th_first_convergence_position is missing/None/NaN (value=None) |
| conrad-secretagent | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| eliot-middlemarch | paragraphs_annotated | incomplete_paragraph_annotation | REVIEW | paragraphs_annotated(4674) < paragraphs_total(4675); n_unlabeled=1 |
| haggard-ksm | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| sabatini-captainblood | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |
| wells-warofworlds | th_first_convergence_position | null_or_nan | EXPECTED [zero-convergence-null-position] | th_first_convergence_position is missing/None/NaN (value=None) |

## Cross-corpus outlier flags (|z| > 2.5)

| book | metric | value | z-score | tag |
|---|---|---|---|---|
| conrad-heartofdarkness | br_character_desc_share | 0.091 | 3.45 | EXPECTED [conrad-heartofdarkness-3units] |
| stoker-dracula | br_transition_share | 0.112 | 4.69 | REVIEW |
| conrad-lordjim | br_words_per_mode_segment | 323.6 | 3.22 | REVIEW |
| dumas-montecristo | paragraphs_annotated | 14380 | 3.42 | REVIEW |
| dumas-montecristo | paragraphs_total | 14380 | 3.42 | REVIEW |
| austen-persuasion | tension_max | 6 | -3.03 | REVIEW |
| buchan-thirtyninesteps | tension_min | 4 | 2.74 | REVIEW |
| austen-persuasion | tension_peak | 6 | -3.03 | REVIEW |
| tolstoy-warandpeace | th_convergence_events | 57 | 3.38 | REVIEW |
| haggard-ksm | th_run_length_mean | 7 | 3.43 | REVIEW |
| tolstoy-warandpeace | th_threads_2plus | 55 | 4.38 | REVIEW |
| tolstoy-warandpeace | th_threads_total | 184 | 4.69 | REVIEW |
| tolstoy-warandpeace | total_words | 562486 | 3.09 | REVIEW |
| tolstoy-warandpeace | units_scored | 365 | 4.56 | REVIEW |
| tolstoy-warandpeace | units_segmented | 366 | 4.55 | REVIEW |

## Hard flags (always reported, regardless of SD)

| book | metric | value | tag |
|---|---|---|---|
| austen-pride | complete | False | REVIEW |
| eliot-middlemarch | complete | False | REVIEW |

## Known, pre-declared limitations (not bugs)

- **[conrad-heartofdarkness-3units]** (scope: conrad-heartofdarkness; metrics: any metric) — conrad-heartofdarkness is segmented into only 3 units (its 3 long parts), so its per-unit stats (tension std/min/max, thread run lengths, block-rhythm switch rate, etc.) are extreme-but-expected, not bugs.
- **[zero-convergence-null-position]** (scope: any book; metrics: th_first_convergence_position) — th_first_convergence_position is null exactly when n_convergence_events == 0 (verified across the corpus: no thread ever converges, so 'position of first convergence' is undefined) — a well-defined structural null, not missing data.
- **[partial-corpus]** (scope: any book; metrics: none — corpus-level) — this nd1 run is in progress; 5 'giant' books (very long, expensive to judge) are pending and simply absent from this table. Their absence is expected scope, not a load failure.

## Distribution summary (one line per metric)

- **units_scored**: min=3 (conrad-heartofdarkness), median=33.5, max=365 (tolstoy-warandpeace); mean=52.19, sd=68.57, n=26
- **units_segmented**: min=3 (conrad-heartofdarkness), median=34.5, max=366 (tolstoy-warandpeace); mean=52.81, sd=68.77, n=26
- **total_words**: min=32361 (wells-timemachine), median=1.254e+05, max=562486 (tolstoy-warandpeace); mean=1.634e+05, sd=1.293e+05, n=26
- **paragraphs_annotated**: min=198 (conrad-heartofdarkness), median=2099, max=14380 (dumas-montecristo); mean=3086, sd=3307, n=26
- **paragraphs_total**: min=198 (conrad-heartofdarkness), median=2107, max=14380 (dumas-montecristo); mean=3086, sd=3306, n=26
- **tension_mean**: min=2.96 (austen-emma), median=5.215, max=7 (sabatini-captainblood); mean=5.252, sd=1.163, n=26
- **tension_std**: min=1.07 (austen-persuasion), median=2.045, max=2.49 (wells-timemachine); mean=1.935, sd=0.3905, n=26
- **tension_min**: min=0 (tolstoy-warandpeace), median=2, max=4 (buchan-thirtyninesteps); mean=1.731, sd=0.8274, n=26
- **tension_max**: min=6 (austen-persuasion), median=9, max=9 (wells-warofworlds); mean=8.462, sd=0.8115, n=26
- **tension_peak**: min=6 (austen-persuasion), median=9, max=9 (wells-warofworlds); mean=8.462, sd=0.8115, n=26
- **tension_peak_position**: min=0.045 (eddison-ouroboros), median=0.5515, max=0.982 (childers-riddlesands); mean=0.473, sd=0.3144, n=26
- **tension_calm_share**: min=0 (buchan-thirtyninesteps), median=0.315, max=0.82 (austen-emma); mean=0.3415, sd=0.2121, n=26
- **tension_high_share**: min=0 (austen-emma), median=0.17, max=0.46 (stoker-dracula); mean=0.2138, sd=0.1664, n=26
- **tension_volatility**: min=0.95 (austen-pride), median=1.825, max=2.5 (conrad-heartofdarkness); mean=1.772, sd=0.4612, n=26
- **tension_tail_mean**: min=2.33 (tolstoy-warandpeace), median=5.75, max=8.5 (dickens-taleoftwocities); mean=5.36, sd=1.967, n=26
- **tension_tail_final**: min=1 (austen-pride), median=3, max=9 (dickens-taleoftwocities); mean=4.115, sd=2.703, n=26
- **br_setting_share**: min=0.006 (austen-emma), median=0.037, max=0.151 (wells-warofworlds); mean=0.05146, sd=0.04373, n=26
- **br_character_desc_share**: min=0.011 (stoker-dracula), median=0.029, max=0.091 (conrad-heartofdarkness); mean=0.03062, sd=0.01749, n=26
- **br_lore_share**: min=0.01 (bronte-janeeyre), median=0.0355, max=0.091 (dunsany-elfland); mean=0.03981, sd=0.02608, n=26
- **br_dialogue_share**: min=0.257 (wells-warofworlds), median=0.518, max=0.777 (dumas-montecristo); mean=0.5305, sd=0.1301, n=26
- **br_action_share**: min=0.126 (eliot-middlemarch), median=0.232, max=0.4 (wells-warofworlds); mean=0.2421, sd=0.06729, n=26
- **br_interiority_share**: min=0.019 (eddison-ouroboros), median=0.0945, max=0.186 (austen-persuasion); mean=0.095, sd=0.03657, n=26
- **br_transition_share**: min=0 (conrad-secretagent), median=0.004, max=0.112 (stoker-dracula); mean=0.0105, sd=0.02162, n=26
- **br_switch_rate**: min=0.281 (dumas-montecristo), median=0.524, max=0.677 (conrad-heartofdarkness); mean=0.5013, sd=0.08887, n=26
- **br_words_per_mode_segment**: min=79.6 (sabatini-captainblood), median=111.7, max=323.6 (conrad-lordjim); mean=135.8, sd=58.34, n=26
- **br_interiority_self_transition**: min=0.083 (dunsany-elfland), median=0.1795, max=0.339 (eliot-middlemarch); mean=0.2003, sd=0.06578, n=26
- **br_secondary_shading_rate**: min=0.106 (austen-pride), median=0.202, max=0.531 (conrad-lordjim); mean=0.2435, sd=0.1227, n=26
- **br_setting_touch_rate**: min=0.009 (austen-emma), median=0.066, max=0.271 (wells-warofworlds); mean=0.09104, sd=0.07655, n=26
- **th_threads_total**: min=2 (conrad-heartofdarkness), median=9.5, max=184 (tolstoy-warandpeace); mean=18.19, sd=35.33, n=26
- **th_threads_2plus**: min=1 (conrad-heartofdarkness), median=5, max=55 (tolstoy-warandpeace); mean=7.615, sd=10.83, n=26
- **th_switch_rate**: min=0.1 (haggard-ksm), median=0.5335, max=1 (conrad-secretagent); mean=0.5606, sd=0.2391, n=26
- **th_run_length_mean**: min=1 (conrad-secretagent), median=1.715, max=7 (haggard-ksm); mean=2.175, sd=1.405, n=26
- **th_run_length_max**: min=1 (conrad-secretagent), median=7, max=15 (austen-emma); mean=6.808, sd=3.589, n=26
- **th_convergence_events**: min=0 (austen-persuasion), median=7, max=57 (tolstoy-warandpeace); mean=10.69, sd=13.7, n=26
- **th_first_convergence_position**: min=0.17 (austen-pride), median=0.455, max=0.97 (wells-timemachine); mean=0.48, sd=0.2616, n=18

