# CLAUDE.md

Guidance for working in this repo. Keep it short and current.

## What this is

A Python tool for benchmarking the creative-writing behaviour of LLMs: it runs
the same prompt N times against one or more models and analyses the outputs for
similarity, entity/name reuse, and structure ("model fingerprints"). Usable via
CLI (`llm_creative_tester.py`) or a Tkinter GUI (`llm_tester_ui.py`).

## Running

```bash
# CLI
python llm_creative_tester.py --models gpt-5.5 claude-opus-4-8 codex-cli --repeats 5 --word-count 300
# GUI
python llm_tester_ui.py
# Re-analyse an existing output file (no API calls)
python -m utils.text_analysis results/<file>.txt --all
# Score saved runs with the v2 metric set -> sidecar (no API calls); dir = whole corpus
python -m utils.metrics results/<file>.json --benchmark v2
python -m utils.metrics results/ --benchmark v2
# Scoring-only mode over arbitrary raw text (no generation step)
python -m utils.metrics --text <file.txt|dir/> --benchmark v2
python -m benchmarks.narrative_dynamics <file.txt|dir/> [--dry-run]
# Single-text series (one book segmented into units; fully local, zero LLM)
python -m utils.metrics --text book.txt --segment chapters --benchmark st1
# Tests (fakes only, no LLM calls; minimal venv/ has just pytest + the
# zero-dependency llm-backends package, installed editable from ../llm-backends)
venv/bin/python -m pytest -q
```

## v2 metrics library (`utils/metrics/`)

Opt-in metrics added *after* the frozen v1 set in `text_analysis.py`. Design lives
in `METRICS_ROADMAP.md`; the rules that protect the longitudinal series:

- **v1 is never touched.** v2 reads `results_*.json` and writes a **separate
  sidecar** `results_*.metrics.json` — the source file stays byte-identical. Don't
  merge v2 output into the v1 `analysis` dict.
- **One module per metric**, filename == metric name, exposing
  `compute(responses, ctx) -> dict` (contract + shared spaCy/embedding helpers in
  `_base.py`). Heavy deps are lazy-imported *inside* `compute`. Underscore-prefixed
  files (`_base`, `_manifests`) are helpers, not metrics (excluded from `available()`).
- **Benchmark versions** are frozen, cumulative manifests `benchmarks/vN.yaml`
  (`extends:` + `add:`), resolved by `_manifests.py`. `v2` = the 8 shipped metrics.
  Once a vN ships it isn't edited; new work adds `vN+1`.
- **Two manifest series share the library:** vN scores N runs of one prompt;
  `st1` scores ONE text whose "runs" are its segmentation units (via `--text
  --segment`, reusing the narrative_dynamics segmentation layer). New library
  module names must never collide with the v1 legacy names (`structure`,
  `entity_analysis`, ...) or they would leak into the frozen vN resolution.
- **`ctx`** is a scratch dict shared across metrics in one run — caches the spaCy
  model (`ctx['_nlp']`) and embedding model (`ctx['_sentence_model']`) so a batch
  loads each once. Directory mode reuses one `ctx` across all files.
- When a metric or a frozen lexicon (e.g. `cliche_density` `LEXICON_VERSION`)
  changes, re-score the corpus with `python -m utils.metrics results/ --benchmark v2`
  to refresh every sidecar.

## Narrative Dynamics benchmark (`benchmarks/narrative_dynamics/`)

The third benchmark: long-range structure (tension trajectory, block rhythm,
thread architecture) of ONE arbitrary-length text, scoring-only, no generation.
Self-contained package + its own frozen manifest series (`benchmarks/nd1.yaml`).
Rules that keep it sane:

- Metric contract mirrors v2: `compute(units, ctx) -> dict`, one module per
  metric; units come from `segmentation.segment` (chapters or ~1500-word
  windows). New metrics ship via `nd2.yaml` (extends: nd1); nd1 is frozen.
- **All LLM traffic goes through `ctx["judge"]`** (`judge.py`): AiHelperJudge
  routes to `ai_helper.send_prompt`; tests inject `FakeJudge`; `--dry-run` uses
  the placeholder judge. The package imports only the stdlib until a real call,
  so it works in the minimal test venv (unlike `utils`, whose `__init__` is
  eagerly heavy).
- The rubrics in `rubrics/` are **versioned provenance artifacts** ported from
  StoryDaemon; their reliability numbers were measured THERE and must be
  re-verified in this harness before findings are trusted. Do not edit rubric
  text in place; a changed rubric is a new version.
- Metric order matters once: tension_trajectory stashes `ctx["unit_tensions"]`
  for thread_architecture's switch-delta analysis (handled by
  `compute_document`; keep it if you touch the runner).

## Backends (the important part)

All model dispatch goes through `ai_helper.send_prompt(prompt, model=...)`.
Since July 2026 `ai_helper.py` is a **compatibility layer over the shared
`llm-backends` package** (pinned in `requirements.txt` to
`llm-backends @ git+https://github.com/EdwardAThomson/llm-backends@v0.1.1`; a
sibling checkout via `pip install -e ../llm-backends` works for dev). The
analyzer's per-model payload profile (model id, system prompt, max_tokens,
temperature presence/absence, reasoning_effort) stays defined HERE, in
`ai_helper.MODEL_CONFIG` — this repo is a longitudinal benchmark, so
`tests/test_payload_equality.py` proves the request kwargs are byte-equal to
the pre-package implementation (frozen snapshot in `tests/_snapshots/`).
Change a default only knowingly, updating that test. Upgrade the package pin
only between scoring campaigns, then re-run the payload test. Two families:

- **API backends** — OpenAI / Anthropic / OpenRouter calls delegate to
  `llm_backends.multi_provider_llm`; Gemini stays local in `ai_helper.py`
  because the package's Gemini path has no `top_p`/`top_k` and the frozen
  payload sends `top_p=1, top_k=40`. Need keys in `.env` (`OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`). The package
  also ships a Venice backend (`venice.ai`, OpenAI-compatible, uncensored
  open-weight models, `VENICE_API_KEY`) — available but not wired into the
  analyzer registry; adding it would be a new-key decision, not a code port.
- **CLI backends** — the `*-cli` keys construct the PACKAGE's interfaces
  (`llm_backends.{codex,claude_cli,gemini_cli}_interface`); the repo's
  `cli_backends/` modules are now import-path shims over them (equivalence
  with the pre-package code: `tests/test_cli_backend_equivalence.py`).
  Model keys: `codex-cli`, `claude-cli{,-opus,-sonnet,-haiku,-fable}`,
  `gemini-cli-pro`, `gemini-cli-flash`. No API key; each runs from a neutral
  empty cwd (`llm_backends.agent_cwd.neutral_cwd`) so the agent generates
  text instead of acting on this repo.

  **Billing gotcha (now enforced by the package):** the CLI interfaces strip
  provider keys from the subprocess env so the CLIs authenticate via their
  subscription/login (`~/.claude`, `~/.codex`), not a metered API key
  inherited from `.env` via `load_dotenv()` — an env-var key outranks the
  CLI's configured login default, so the "no API key" CLI path would
  otherwise silently charge the API. Stripping defaults ON in llm-backends:
  claude strips `ANTHROPIC_API_KEY` (+ deprecated `CLAUDE_API_KEY`), codex
  strips `OPENAI_API_KEY`, gemini strips `GEMINI_API_KEY` + `GOOGLE_API_KEY`.
  Each interface has a `strip_provider_keys=False` opt-out; the analyzer
  never passes it.

When adding/removing a model, edit `ai_helper.MODEL_CONFIG` (and, for API
models, confirm the key resolves in the llm-backends registry —
`tests/test_registry_parity.py` enforces this). The old "three places in
sync" rule is now structural: `AVAILABLE_MODELS` in `llm_tester_ui.py` IS
`ai_helper.get_supported_models()`, and `DEFAULT_MODELS` in
`llm_creative_tester.py` is validated against the registry at import time.

## codex-cli on hardened Linux (gotcha)

Recent Codex sandboxes `codex exec` with a **bundled bubblewrap** that creates an
unprivileged user namespace. Ubuntu 23.10+ blocks that by default
(`kernel.apparmor_restrict_unprivileged_userns=1`), so codex's own sandbox fails
with *"bubblewrap … needs access to create user namespaces."*

Rather than weaken the host, the codex interface (now
`llm_backends/codex_interface.py`; this workaround was developed in this repo
and merged upstream) detects the
restriction (a cheap `unshare --map-root-user` probe) and, when blocked, runs
codex inside an **identity-mapped user namespace** via `unshare`, using the
setuid `newuidmap`/`newgidmap` helpers (which are allowed to write the uid_map),
with codex's own sandbox disabled — the outer namespace is the sandbox. The map
is identity-only so codex still runs as the real user and can read `~/.codex`
(auth survives). On unrestricted hosts / macOS it leaves codex's own read-only
sandbox in place.

**Requirement:** `sudo apt install uidmap` (plus `/etc/subuid` + `/etc/subgid`
entries for your user, which the package creates). Only `codex-cli` needs this;
`claude-cli`/`gemini-cli-*` don't create namespaces.

## Analysis caveats (read before trusting the name metric)

- **NER false positives inflate the name metric.** The entity/name analysis uses
  spaCy `en_core_web_sm`, which mislabels non-names as `PERSON` (e.g. `Minds`,
  `Kepler`, `Metropolis`, `metallic`, `Dirt`, `Frozen`, `ALERT`). These leak into
  "repeated name components", so a small repeated-name count can be *entirely*
  false positives. Verify against the raw per-run PERSON lists (re-run spaCy on
  `results/.../results_*.json` → `responses[i].response_text`) before drawing
  conclusions. Example: claude-opus's only two "repeated names" (`Minds`,
  `Kepler`) are both false positives — its true repeated-character-name count is 0.
- **Full names are split into components** (`text_analysis.py` `text.split()`), so
  surnames count as name-parts too (e.g. Gemini's recurring `Vance`), not just
  first names. "Repeated name components" counts parts appearing in >1 of the N runs.
- **Some API models don't take `temperature` either.** Fable 5 and Opus 4.8
  (like Opus 4.7) removed the sampling params — sending `temperature`/`top_p`/
  `top_k` returns a 400. Their registry entries pass `temperature=None`, so
  `send_prompt_claude` omits it and the model uses its own default. That means
  `claude-fable-5` / `claude-opus-4-8` runs aren't strictly comparable to the
  other API runs (which use temp 0.7) on the sampling axis. `claude-sonnet-4-6`
  and `claude-haiku-4-5` still use temp 0.7.
- **CLI backends don't forward sampling params.** `temperature`/`max_tokens` are
  *not* passed to `codex`/`claude`/`gemini` CLIs — each uses its own defaults.
  So CLI runs aren't strictly comparable to the API runs (which use temp 0.7).
  CLI model aliases resolve to the installed CLI's current default, e.g.
  `claude-cli-opus` → `claude-opus-4-8`, and `codex-cli` → whatever
  `~/.codex/config.toml` pins (`gpt-5.5` at the time of the June 2026 runs).

## Notes

- The CLI backends were ported from the StoryDaemon writing ecosystem; keep them
  roughly in sync if you fix bugs in either place.
- CLI backends use only the stdlib (subprocess/shutil/tempfile/json) — no new
  pip deps.
