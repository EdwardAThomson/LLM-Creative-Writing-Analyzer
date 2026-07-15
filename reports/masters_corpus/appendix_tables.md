## Appendix A: Full metric tables

*Every metric computed, straight from `corpus_dataset.csv`. Grouped by author; one table per metric family. Both st1 and nd1 now cover all 26 books. See Sections 2 and 5-9 for what each metric means and how to read it; the caveats in Section 12 apply (cast counts are not coreference-merged; opening-formula is title-contaminated).*

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

### A.2 nd1 (LLM-judged, all 26 books)

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
|  | womaninwhite | 5.31 | 1.980 | 1 | 8 | 8 | 0.057 |
| conrad | heartofdarkness | 4 | 2.160 | 2 | 7 | 7 | 0.833 |
|  | lordjim | 5.18 | 2.110 | 2 | 9 | 9 | 0.678 |
|  | secretagent | 4.54 | 2.130 | 2 | 9 | 9 | 0.808 |
| dickens | bleakhouse | 4.03 | 1.940 | 1 | 9 | 9 | 0.799 |
|  | taleoftwocities | 6 | 2.300 | 2 | 9 | 9 | 0.611 |
| dumas | montecristo | 5.88 | 2.340 | 2 | 9 | 9 | 0.064 |
| dunsany | elfland | 4.56 | 1.970 | 2 | 8 | 8 | 0.074 |
| eddison | ouroboros | 6.91 | 1.620 | 2 | 9 | 9 | 0.045 |
| eliot | middlemarch | 4.62 | 1.780 | 1 | 8 | 8 | 0.608 |
| haggard | ksm | 6 | 2.490 | 1 | 9 | 9 | 0.548 |
|  | she | 6.59 | 2.210 | 2 | 9 | 9 | 0.293 |
| sabatini | captainblood | 7 | 1.760 | 3 | 9 | 9 | 0.081 |
|  | scaramouche | 6.08 | 2.110 | 2 | 9 | 9 | 0.097 |
| stoker | dracula | 6.46 | 2.110 | 1 | 9 | 9 | 0.554 |
| tolstoy | warandpeace | 4.44 | 2.110 | 0 | 9 | 9 | 0.177 |
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
|  | womaninwhite | 0.180 | 0.110 | 1.620 | 6.67 | 3 |
| conrad | heartofdarkness | 0.670 | 0.000 | 2.500 | 7 | 7 |
|  | lordjim | 0.360 | 0.160 | 1.590 | 7.75 | 9 |
|  | secretagent | 0.460 | 0.150 | 2.170 | 4 | 4 |
| dickens | bleakhouse | 0.600 | 0.090 | 1.820 | 2.86 | 1 |
|  | taleoftwocities | 0.240 | 0.400 | 1.910 | 8.5 | 9 |
| dumas | montecristo | 0.260 | 0.350 | 2.160 | 6.58 | 7 |
| dunsany | elfland | 0.500 | 0.090 | 1.700 | 6 | 6 |
| eddison | ouroboros | 0.060 | 0.450 | 1.970 | 6 | 2 |
| eliot | middlemarch | 0.380 | 0.030 | 1.380 | 5.33 | 1 |
| haggard | ksm | 0.240 | 0.380 | 2.300 | 2.5 | 3 |
|  | she | 0.170 | 0.410 | 2.360 | 8 | 6 |
| sabatini | captainblood | 0.130 | 0.450 | 1.830 | 7.33 | 6 |
|  | scaramouche | 0.220 | 0.330 | 1.830 | 7 | 3 |
| stoker | dracula | 0.110 | 0.460 | 2.190 | 5.33 | 1 |
| tolstoy | warandpeace | 0.450 | 0.100 | 1.690 | 2.33 | 2 |
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
|  | womaninwhite | 0.022 | 0.029 | 0.036 | 0.492 | 0.271 | 0.135 | 0.015 |
| conrad | heartofdarkness | 0.141 | 0.091 | 0.015 | 0.394 | 0.242 | 0.096 | 0.020 |
|  | lordjim | 0.084 | 0.040 | 0.067 | 0.460 | 0.226 | 0.120 | 0.003 |
|  | secretagent | 0.031 | 0.065 | 0.013 | 0.565 | 0.209 | 0.118 | 0.000 |
| dickens | bleakhouse | 0.036 | 0.034 | 0.015 | 0.671 | 0.183 | 0.059 | 0.002 |
|  | taleoftwocities | 0.049 | 0.034 | 0.028 | 0.633 | 0.210 | 0.040 | 0.005 |
| dumas | montecristo | 0.016 | 0.014 | 0.016 | 0.777 | 0.148 | 0.029 | 0.001 |
| dunsany | elfland | 0.144 | 0.025 | 0.091 | 0.296 | 0.340 | 0.092 | 0.011 |
| eddison | ouroboros | 0.063 | 0.034 | 0.056 | 0.621 | 0.184 | 0.019 | 0.023 |
| eliot | middlemarch | 0.013 | 0.030 | 0.036 | 0.666 | 0.126 | 0.128 | 0.002 |
| haggard | ksm | 0.055 | 0.026 | 0.035 | 0.508 | 0.310 | 0.059 | 0.007 |
|  | she | 0.059 | 0.025 | 0.088 | 0.482 | 0.259 | 0.083 | 0.005 |
| sabatini | captainblood | 0.023 | 0.051 | 0.030 | 0.559 | 0.264 | 0.071 | 0.002 |
|  | scaramouche | 0.016 | 0.037 | 0.018 | 0.637 | 0.210 | 0.081 | 0.001 |
| stoker | dracula | 0.038 | 0.011 | 0.041 | 0.481 | 0.238 | 0.078 | 0.112 |
| tolstoy | warandpeace | 0.027 | 0.038 | 0.077 | 0.477 | 0.265 | 0.112 | 0.003 |
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
|  | womaninwhite | 0.535 | 109.7 | 0.264 | 0.195 | 0.040 |
| conrad | heartofdarkness | 0.677 | 280.8 | 0.167 | 0.495 | 0.222 |
|  | lordjim | 0.593 | 323.6 | 0.119 | 0.531 | 0.165 |
|  | secretagent | 0.615 | 90.7 | 0.267 | 0.243 | 0.053 |
| dickens | bleakhouse | 0.440 | 109.4 | 0.140 | 0.164 | 0.057 |
|  | taleoftwocities | 0.436 | 92.3 | 0.205 | 0.136 | 0.075 |
| dumas | montecristo | 0.281 | 111.3 | 0.150 | 0.113 | 0.029 |
| dunsany | elfland | 0.596 | 139.5 | 0.083 | 0.436 | 0.264 |
| eddison | ouroboros | 0.409 | 162.6 | 0.119 | 0.209 | 0.100 |
| eliot | middlemarch | 0.403 | 163.2 | 0.339 | 0.182 | 0.021 |
| haggard | ksm | 0.533 | 97.6 | 0.156 | 0.163 | 0.088 |
|  | she | 0.518 | 158.1 | 0.257 | 0.297 | 0.115 |
| sabatini | captainblood | 0.530 | 79.6 | 0.115 | 0.191 | 0.046 |
|  | scaramouche | 0.468 | 83.2 | 0.177 | 0.148 | 0.027 |
| stoker | dracula | 0.543 | 138.1 | 0.153 | 0.281 | 0.079 |
| tolstoy | warandpeace | 0.575 | 84.2 | 0.277 | 0.235 | 0.049 |
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
|  | womaninwhite | 12 | 5 | 0.433 | 2.26 | 10 | 29 | 0.220 |
| conrad | heartofdarkness | 2 | 1 | 0.500 | 1.5 | 2 | 0 | — |
|  | lordjim | 14 | 6 | 0.477 | 2.05 | 9 | 2 | 0.940 |
|  | secretagent | 9 | 3 | 1.000 | 1 | 1 | 0 | — |
| dickens | bleakhouse | 21 | 13 | 0.803 | 1.24 | 4 | 10 | 0.540 |
|  | taleoftwocities | 24 | 7 | 0.909 | 1.1 | 3 | 10 | 0.570 |
| dumas | montecristo | 48 | 23 | 0.759 | 1.31 | 9 | 21 | 0.210 |
| dunsany | elfland | 14 | 7 | 0.788 | 1.26 | 3 | 3 | 0.460 |
| eddison | ouroboros | 12 | 6 | 0.656 | 1.5 | 7 | 8 | 0.590 |
| eliot | middlemarch | 29 | 16 | 0.816 | 1.22 | 6 | 20 | 0.340 |
| haggard | ksm | 2 | 1 | 0.100 | 7 | 12 | 0 | — |
|  | she | 5 | 3 | 0.500 | 1.93 | 5 | 12 | 0.290 |
| sabatini | captainblood | 11 | 5 | 0.567 | 1.72 | 4 | 0 | — |
|  | scaramouche | 10 | 7 | 0.571 | 1.71 | 5 | 7 | 0.540 |
| stoker | dracula | 6 | 5 | 0.444 | 2.15 | 4 | 12 | 0.410 |
| tolstoy | warandpeace | 184 | 55 | 0.813 | 1.23 | 8 | 57 | 0.220 |
| wells | timemachine | 2 | 2 | 0.133 | 5.33 | 12 | 1 | 0.970 |
|  | warofworlds | 8 | 3 | 0.500 | 1.93 | 10 | 0 | — |
