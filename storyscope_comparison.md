# StoryScope core features vs this repo's v2/nd1 metrics

Comparison of the core narrative features from "StoryScope: Investigating
idiosyncrasies in AI fiction" (Russell, Rajendhran, Pham, Iyyer, Wieting;
arXiv:2604.03136, UMD + Google DeepMind) against this repo's metric sets
(v1 frozen legacy, v2 library, nd1 narrative dynamics, st1 single-text).

**Caveat on completeness:** the paper's full 30-feature core list lives in
appendix Tables 13/14, which are absent from every arXiv PDF version (v1-v4
are all 16 pages, ending at the references) and from the released repo
(github.com/jenna-russell/storyscope: `data/taxonomy.json` has all 304
features but no core flags; the released XGBoost models store anonymous
column indices whose names cannot be safely reconstructed because the exact
one-hot encoding and the 8 style-flagged exclusions are unpublished). This
table covers the ~22 core features the paper's main body explicitly names or
quantifies, each matched to its entry in the released taxonomy.

Legend: ✅ direct counterpart, 🟡 partial or proxy, ❌ not measured here.
"Signal" is the direction the paper reports.

## 1. Theme and commentary ("AI over-explains its themes")

| StoryScope core feature (taxonomy ID) | Signal | Nearest metric here | Coverage |
|---|---|---|---|
| Thematic Explicitness and Moralizing (`SIT_MET_303`, scale 1-5) | AI ~20% higher | none | ❌ |
| Narratorial Thematic Commentary (`SIT_MET_501`, binary; 77% AI vs 52% human) | AI | none | ❌ |
| Thematic Unity (`PLT_THM_008`, scale) | AI tighter | none | ❌ |
| Philosophical debate in dialogue (`PER_DIA_003` functions / `SIT_GEN_010`; 59% vs 34%) | AI | `dialogue_ratio` (v2) measures amount, not function | 🟡 |

## 2. Plot, causality, endings ("human authors subvert linearity")

| StoryScope core feature (taxonomy ID) | Signal | Nearest metric here | Coverage |
|---|---|---|---|
| Agency in Resolution (`PLT_CON_007`; protagonist-driven 69% AI vs 46%) | AI | none | ❌ |
| Continuity of main causal chain (`EVT_CAU_002`; AI tighter) | AI | `tension_trajectory` curve shape hints at it, no causal metric | 🟡 |
| Density of Subplots (`PLT_STR_003`; "no subplots" 79% AI vs 57%) | AI | `thread_architecture` (nd1): cast-clustered thread count and switching | 🟡 close |
| Subplot-theme integration (`PLT_THM_009`; 42% human vs 21%) | human | `thread_architecture` counts threads, not integration | 🟡 |
| Ending closure / ambiguity (`PLT_MOR_003`/`PLT_MOR_006`; internal-acceptance endings 47% AI vs 27%) | AI tidy | final-unit tension in `tension_trajectory` only | 🟡 weak |
| Moral Polarity Toward Protagonist (`PLT_MOR_002`; ambivalent 59% human vs 38%) | human | none | ❌ |

## 3. Temporal structure (the strongest human-leaning block)

| StoryScope core feature (taxonomy ID) | Signal | Nearest metric here | Coverage |
|---|---|---|---|
| Degree of Chronological Discontinuity (`TMP_ORD_010`, scale 1-5) | human | none; nd segmentation exists but no ordering metric | ❌ |
| Flashback depth / present-vs-flashback balance (`TMP_ORD_004`, `EVT_TYP_009`) | human | none | ❌ |
| Depth of Recontextualization After Surprise (`REV_SUR_003`, scale 1-5) | human | none | ❌ |
| Global Withholding Intensity / delayed disclosure (`REV_SUS_001`, `REV_DIS_003`) | human | none | ❌ |

## 4. Body and senses ("AI over-writes the body")

| StoryScope core feature (taxonomy ID) | Signal | Nearest metric here | Coverage |
|---|---|---|---|
| Somatic emotion conveyance (`AGENT_EMO_009`/`AGENT_EMO_012`; 81% AI vs 38%) | AI | none (lexicon-friendly, v3 candidate) | ❌ |
| Explicit emotion naming (`AGENT_EMO_001`; 29% human vs 8% AI) | human | none (same module could score both) | ❌ |
| Smell imagery / sensory modalities (`SET_ATM_017`; 82% vs 57%) | AI | none | ❌ |
| Setting as mirror of inner state (`SET_ATM_003`/`SET_ATM_019`) | AI | none | ❌ |

## 5. Engaging the outside world (human-leaning)

| StoryScope core feature (taxonomy ID) | Signal | Nearest metric here | Coverage |
|---|---|---|---|
| Intertextual Strategy Types / Reference Explicitness (`SIT_MET_202`, `SIT_MET_008`; named refs 47% human vs 24%) | human | `entity_analysis` (v1) / `entity_census` (st1) see named entities but don't classify referencing strategy | 🟡 |
| Fourth-Wall Breaking / Permeability (`SIT_MET_002`, `SIT_MET_004`; 67% human vs 39%) | human | none | ❌ |
| Direct reader address (`PER_POV_009`; 28% human vs 7%) | human | none | ❌ |
| Location Variety Scope (`SET_LOC_011`; humans span more) | human | place-entity counts in `entity_census` | 🟡 |
| Dialogue-to-narration proportion (`PER_DIA_001`; humans higher) | human | `dialogue_ratio` (v2), same construct | ✅ |

## Fingerprint features that corroborate nd1 directly

From their 75 fingerprint features (not the core 30), mapping onto nd1:

| StoryScope fingerprint (taxonomy ID) | Their finding | nd1 metric |
|---|---|---|
| Strength of Event Escalation (`EVT_SCH_003`) + Tension Escalation Pattern (`EVT_SCH_013`) | Claude has the flattest escalation of all sources | `tension_trajectory` (slope, peak position, curve shape) ✅ |
| Scene vs Summary Balance (`STY_CPX_006`) | style-dimension pacing | `block_rhythm` (per-paragraph block types) ✅ |
| Post-Climax Denouement Length (`PLT_MOR_007`) | Gemini's extended denouements | tail of `tension_trajectory` + `block_rhythm` 🟡 |
| Dreams as temporal distortion (`TMP_ORD_011`), Gossip as plot mechanism (`SOC_REL_012`), Attitude Toward Literary Tradition (`SIT_MET_009`) | GPT dreams/gossip, Claude reverence | none ❌ |

## The reverse direction: what this repo measures that StoryScope cannot

| This repo's metric | Why StoryScope has no counterpart |
|---|---|
| `semantic_similarity`, `text_similarity` (v1), `ngram_diversity`, `opening_lines`, `phonetic_names` (v2), entity reuse in `entity_analysis` | Cross-run metrics over N repeats of one prompt per model. StoryScope generates one story per model per prompt, so within-model repetition and name reuse are invisible to it. Their population-level "AI clusters / humans are rarer" finding is the between-source cousin of this axis. |
| `mtld`, `burstiness`, `cliche_density`, `intra_text_repetition`, `vocabulary_diversity` | Deliberately ablated on their side (the 39 style features excluded to prove the narrative-only result). Complementary, not redundant. |
| `tension_trajectory` per-unit numeric curve | Their escalation features are whole-story ordinal ratings; nd1 keeps the full trajectory, strictly richer for shape analysis. |

## Takeaways

Overlap tally on the published subset: 1 direct hit (`dialogue_ratio`),
~7 partials (mostly `thread_architecture`, `tension_trajectory`, entity
metrics), ~14 clean gaps. The gaps concentrate in two clusters, a natural
nd2/v3 shortlist:

1. **Temporal structure** (chronological discontinuity, flashback balance,
   recontextualization depth): the paper's strongest human-leaning core
   block, entirely unmeasured here; fits the nd `ctx["judge"]` rubric
   pattern since it needs whole-text reasoning.
2. **Explicitness cluster** (thematic moralizing, narratorial commentary,
   explicit vs somatic emotion): partly rubric-scorable, partly
   lexicon-scorable (emotion labels and body-sensation phrases would work
   like `cliche_density` with a frozen `LEXICON_VERSION`).

Since the full core-30 list is unpublished, any citation should say "the
core features reported in the main text", or email the authors for
Tables 13/14.
