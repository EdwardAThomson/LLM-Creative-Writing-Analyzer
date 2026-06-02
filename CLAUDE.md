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
```

## Backends (the important part)

All model dispatch goes through `ai_helper.send_prompt(prompt, model=...)`, which
holds a single `model -> callable` registry. Two families:

- **API backends** — OpenAI / Gemini / Anthropic SDKs. Need keys in `.env`
  (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).
- **CLI backends** (`cli_backends/`) — shell out to locally-installed agent CLIs
  (`codex`, `claude`, `gemini`) in headless mode, **no API key**. Model keys:
  `codex-cli`, `claude-cli{,-opus,-sonnet,-haiku}`, `gemini-cli-pro`,
  `gemini-cli-flash`. Each runs from a neutral empty cwd (`agent_cwd.neutral_cwd`)
  so the agent generates text instead of acting on this repo.

When adding/removing a model, keep three places in sync: the registry in
`ai_helper.py`, `DEFAULT_MODELS` in `llm_creative_tester.py`, and
`AVAILABLE_MODELS` in `llm_tester_ui.py`. (There's a 1:1 check — registry keys
must equal `AVAILABLE_MODELS`.)

## codex-cli on hardened Linux (gotcha)

Recent Codex sandboxes `codex exec` with a **bundled bubblewrap** that creates an
unprivileged user namespace. Ubuntu 23.10+ blocks that by default
(`kernel.apparmor_restrict_unprivileged_userns=1`), so codex's own sandbox fails
with *"bubblewrap … needs access to create user namespaces."*

Rather than weaken the host, `cli_backends/codex_interface.py` detects the
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
