# st1 corpus anomaly report

QA surfacing report only. Nothing here is asserted to be a bug — each item is tagged EXPECTED (matches a pre-declared known limitation) or REVIEW (novel, needs a human look).

## Load status

- Sidecars loaded: 26
- Load failures: 0

## Integrity findings (per book)

None. All books passed the arithmetic/null/range checks.

## Cross-corpus outlier flags (|z| > 2.5)

| book | metric | value | z-score | tag |
|---|---|---|---|---|
| tolstoy-warandpeace | cast_size | 956 | 3.19 | EXPECTED [ner-name-splitting] |
| conrad-heartofdarkness | dialogue_ratio_mean | 0.948 | 3.53 | EXPECTED [conrad-heartofdarkness-3units] |
| austen-emma | em_dash_per_1k | 18.32 | 2.78 | REVIEW |
| conrad-heartofdarkness | intra_bigram | 0.274 | 2.78 | EXPECTED [conrad-heartofdarkness-3units] |
| conrad-heartofdarkness | intra_unigram | 0.564 | 2.57 | EXPECTED [conrad-heartofdarkness-3units] |
| collins-womaninwhite | max_pairwise_sim | 0.078 | 3.65 | REVIEW |
| collins-womaninwhite | max_verbatim_chars | 79 | 3.20 | REVIEW |
| conrad-heartofdarkness | mean_chapter_words | 1.283e+04 | 4.02 | EXPECTED [conrad-heartofdarkness-3units] |
| buchan-thirtyninesteps | opening_high_pair_rate | 0.022 | 3.77 | REVIEW |
| wells-warofworlds | opening_high_pair_rate | 0.017 | 2.83 | REVIEW |
| tolstoy-warandpeace | recurring_cast_size | 352 | 3.43 | EXPECTED [ner-name-splitting] |
| dunsany-elfland | slop_per_1k | 0.41 | 4.55 | REVIEW |
| tolstoy-warandpeace | total_words | 562486 | 3.09 | EXPECTED [size-format-extremes] |
| tolstoy-warandpeace | units_scored | 365 | 4.56 | EXPECTED [size-format-extremes] |
| tolstoy-warandpeace | units_segmented | 366 | 4.55 | EXPECTED [size-format-extremes] |

## Hard flags (always reported, regardless of SD)

| book | metric | value | tag |
|---|---|---|---|
| buchan-thirtyninesteps | opening_high_pair_rate | 0.022 | REVIEW |
| collins-moonstone | opening_high_pair_rate | 0.002 | EXPECTED [collins-moonstone-signpost-calibration] |
| eddison-ouroboros | opening_high_pair_rate | 0.002 | EXPECTED [size-format-extremes] |
| eliot-middlemarch | opening_high_pair_rate | 0.004 | REVIEW |
| stoker-dracula | opening_high_pair_rate | 0.003 | REVIEW |
| wells-warofworlds | opening_high_pair_rate | 0.017 | REVIEW |

## Known, pre-declared limitations (not bugs)

- **[ner-name-splitting]** (scope: any book; metrics: cast_size, person_mentions_per_1k, recurring_cast_size) — NER name-splitting and title conflation inflate/split cast entries ("van"/"helsing", "monte"/"cristo", "prince" as a title, first-name vs surname not coreferenced). cast_size / mentions anomalies driven by naming are known, not a bug.
- **[conrad-heartofdarkness-3units]** (scope: conrad-heartofdarkness; metrics: any metric) — conrad-heartofdarkness has only 3 units (structured in 3 long parts), so its per-chapter structure metrics (mean_chapter_words very high, tiny n) are expected outliers, not bugs.
- **[collins-moonstone-signpost-calibration]** (scope: collins-moonstone; metrics: any metric) — collins-moonstone intentionally drops 9 of 12 short narrator-signpost headings (known calibration, baked in at extraction time); its structure is otherwise fine.
- **[short-chapter-mtld-unreliable]** (scope: any book; metrics: mtld_unreliable_runs) — Short chapters (e.g. War and Peace's ~146-word chapters) can make MTLD "unreliable" for those units — expected, not a failure; just report the count.
- **[size-format-extremes]** (scope: eddison-ouroboros, tolstoy-warandpeace; metrics: any metric) — tolstoy-warandpeace (365 scored units) and eddison-ouroboros (roman-colon headings) are the corpus's size/format extremes, so their unit-count and structure-shaped outliers are expected.

## Distribution summary (one line per metric)

- **units_scored**: min=3 (conrad-heartofdarkness), median=33.5, max=365 (tolstoy-warandpeace); mean=52.19, sd=68.57, n=26
- **units_segmented**: min=3 (conrad-heartofdarkness), median=34.5, max=366 (tolstoy-warandpeace); mean=52.81, sd=68.77, n=26
- **total_words**: min=32361 (wells-timemachine), median=1.254e+05, max=562486 (tolstoy-warandpeace); mean=1.634e+05, sd=1.293e+05, n=26
- **mean_chapter_words**: min=1572 (tolstoy-warandpeace), median=3767, max=1.283e+04 (conrad-heartofdarkness); mean=4048, sd=2184, n=26
- **mtld_mean**: min=70.36 (dunsany-elfland), median=91.33, max=105.9 (eliot-middlemarch); mean=91.99, sd=9.609, n=26
- **mtld_unreliable_runs**: min=0 (austen-emma), median=0, max=0 (wells-warofworlds); mean=0, sd=0, n=26
- **cliche_per_1k**: min=0 (buchan-greenmantle), median=0.05, max=0.15 (sabatini-captainblood); mean=0.05538, sd=0.03932, n=26
- **slop_per_1k**: min=0.01 (austen-emma), median=0.05, max=0.41 (dunsany-elfland); mean=0.06423, sd=0.07601, n=26
- **em_dash_per_1k**: min=0.1 (dunsany-elfland), median=5.615, max=18.32 (austen-emma); mean=6.413, sd=4.283, n=26
- **dialogue_ratio_mean**: min=0.05 (dunsany-elfland), median=0.3205, max=0.948 (conrad-heartofdarkness); mean=0.3379, sd=0.1727, n=26
- **distinct_3_ratio**: min=0.7444 (tolstoy-warandpeace), median=0.8634, max=0.92 (conrad-heartofdarkness); mean=0.8478, sd=0.04771, n=26
- **self_bleu_mean**: min=0.1282 (conrad-heartofdarkness), median=0.2432, max=0.3788 (tolstoy-warandpeace); mean=0.2544, sd=0.06103, n=26
- **intra_unigram**: min=0.344 (wells-warofworlds), median=0.433, max=0.564 (conrad-heartofdarkness); mean=0.4259, sd=0.05369, n=26
- **intra_bigram**: min=0.14 (austen-pride), median=0.1845, max=0.274 (conrad-heartofdarkness); mean=0.19, sd=0.0302, n=26
- **intra_trigram**: min=0.017 (austen-pride), median=0.028, max=0.048 (dickens-bleakhouse); mean=0.03058, sd=0.008566, n=26
- **burstiness_mean**: min=-0.306 (buchan-greenmantle), median=-0.162, max=-0.082 (austen-emma); mean=-0.172, sd=0.06047, n=26
- **max_pairwise_sim**: min=0.015 (conrad-heartofdarkness), median=0.03, max=0.078 (collins-womaninwhite); mean=0.03223, sd=0.01255, n=26
- **max_verbatim_chars**: min=29 (buchan-thirtyninesteps), median=35, max=79 (collins-womaninwhite); mean=40.19, sd=12.15, n=26
- **n_flagged_pairs**: min=0 (austen-emma), median=0, max=0 (wells-warofworlds); mean=0, sd=0, n=26
- **opening_mean_pairwise**: min=0.036 (eliot-middlemarch), median=0.091, max=0.175 (eddison-ouroboros); mean=0.1009, sd=0.0304, n=26
- **opening_high_pair_rate**: min=0 (austen-emma), median=0, max=0.022 (buchan-thirtyninesteps); mean=0.001923, sd=0.005329, n=26
- **cast_size**: min=26 (wells-timemachine), median=202, max=956 (tolstoy-warandpeace); mean=265.5, sd=216.2, n=26
- **recurring_cast_size**: min=3 (wells-timemachine), median=69, max=352 (tolstoy-warandpeace); mean=86.85, sd=77.39, n=26
- **person_mentions_per_1k**: min=1.02 (wells-timemachine), median=12.91, max=27.6 (austen-persuasion); mean=13.73, sd=7.161, n=26

