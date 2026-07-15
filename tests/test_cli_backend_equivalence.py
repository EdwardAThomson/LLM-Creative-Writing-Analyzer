"""CLI-backend equivalence: pre-package cli_backends/ vs llm-backends interfaces.

Migration gate for step 3 of the llm-backends adoption: the *-cli model keys
now route through the package's CLI interfaces, so this proves (with a fake
subprocess, no CLI installed, no network) that for codex-cli and claude-cli:

* the command argv is identical (including the codex userns/sandbox branches),
* the billing key-stripping produces the same child environment,
* both run from a neutral (empty, non-repo) cwd.

The OLD implementations are verbatim snapshots of cli_backends/ at commit
22898ef, frozen as the package tests/_snapshots/legacy_cli_backends/ (loaded
here under that name, keeping their relative imports working).

Known, deliberate differences (asserted explicitly, not papered over):
* the package's claude interface strips CLAUDE_API_KEY in addition to
  ANTHROPIC_API_KEY (strictly safer; the old code stripped only the latter);
* the neutral-cwd tempdir prefix changed ("llm-analyzer-agent-" ->
  "llm-backends-agent-"): same behavior, different label;
* the package adds a strip_provider_keys=False opt-out (default ON keeps the
  old behavior exactly).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

_SNAPSHOTS = pathlib.Path(__file__).parent / "_snapshots"
if str(_SNAPSHOTS) not in sys.path:
    sys.path.insert(0, str(_SNAPSHOTS))

import legacy_cli_backends  # noqa: E402  (the frozen pre-package modules)
import legacy_cli_backends.codex_interface as legacy_codex  # noqa: E402
from llm_backends import codex_interface as pkg_codex  # noqa: E402
from llm_backends import ClaudeCliInterface as PkgClaude, CodexInterface as PkgCodex  # noqa: E402

LegacyClaude = legacy_cli_backends.ClaudeCliInterface
LegacyCodex = legacy_cli_backends.CodexInterface

PROMPT = "Write one paragraph about tidal pools."


@pytest.fixture()
def cli_env(monkeypatch):
    """Provider keys present in the parent env, so stripping is observable."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-claude-legacy")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    # Every CLI binary "exists".
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")


def fake_run_factory(record, stdout="CLI STDOUT", probe_fails=False):
    """A subprocess.run stand-in; records real CLI invocations, and lets the
    codex userns probe (`unshare --map-root-user true`) succeed or fail."""
    def fake_run(argv, capture_output=None, text=None, timeout=None, check=None,
                 env=None, cwd=None):
        if argv[0] == "unshare" and "--map-root-user" in argv:
            if probe_fails:
                raise subprocess.CalledProcessError(1, argv)
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)
        record.append({"argv": list(argv), "env": env, "cwd": cwd, "timeout": timeout})
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)
    return fake_run


def _reset_userns_caches(monkeypatch):
    monkeypatch.setattr(legacy_codex, "_userns_prefix_cache", None)
    monkeypatch.setattr(pkg_codex, "_userns_prefix_cache", None)


def _normalize_codex_argv(argv):
    """Replace the per-call tempfile after --output-last-message with a token."""
    argv = list(argv)
    i = argv.index("--output-last-message")
    assert argv[i + 1].endswith(".txt") and "codex-msg-" in argv[i + 1]
    argv[i + 1] = "<MSGFILE>"
    return argv


def _assert_neutral_cwd(cwd, expected_prefix):
    assert cwd is not None and os.path.isdir(cwd)
    assert os.listdir(cwd) == []  # empty scratch dir: the CLI has nothing to act on
    assert expected_prefix in os.path.basename(cwd)
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    assert not pathlib.Path(cwd).resolve().is_relative_to(repo_root)


# --- codex-cli --------------------------------------------------------------------------


def test_codex_userns_probe_prefix_identical(cli_env, monkeypatch):
    """Restricted host: both implementations compute the same identity-mapped
    unshare prefix from the same failed probe."""
    _reset_userns_caches(monkeypatch)
    monkeypatch.setattr("subprocess.run", fake_run_factory([], probe_fails=True))

    old_prefix = legacy_codex._userns_launch_prefix()
    new_prefix = pkg_codex._userns_launch_prefix()

    uid, gid = os.getuid(), os.getgid()
    assert old_prefix == new_prefix == [
        "unshare", "--user",
        f"--map-users={uid}:{uid}:1",
        f"--map-groups={gid}:{gid}:1",
        "--",
    ]


def test_codex_unrestricted_argv_env_cwd_identical(cli_env, monkeypatch):
    """Unrestricted host (probe succeeds): codex's own read-only sandbox flags."""
    _reset_userns_caches(monkeypatch)
    record = []
    monkeypatch.setattr("subprocess.run", fake_run_factory(record))

    old_out = LegacyCodex().generate(PROMPT)
    new_out = PkgCodex().generate(PROMPT)
    assert old_out == new_out == "CLI STDOUT"

    old, new = record
    assert _normalize_codex_argv(old["argv"]) == _normalize_codex_argv(new["argv"]) == [
        "codex", "exec",
        "--sandbox", "read-only", "--ask-for-approval", "never",
        "--skip-git-repo-check",
        "--output-last-message", "<MSGFILE>",
        PROMPT,
    ]
    # Key-stripping: OPENAI_API_KEY removed, everything else inherited untouched.
    for call in (old, new):
        assert "OPENAI_API_KEY" not in call["env"]
        assert call["env"]["ANTHROPIC_API_KEY"] == "sk-anthropic"
    assert old["env"] == new["env"]
    assert old["timeout"] == new["timeout"] == 300

    # Neutral cwd: same behavior, different tempdir label (deliberate rename).
    _assert_neutral_cwd(old["cwd"], "llm-analyzer-agent-")
    _assert_neutral_cwd(new["cwd"], "llm-backends-agent-")


def test_codex_restricted_argv_identical(cli_env, monkeypatch):
    """Restricted host: unshare prefix + codex sandbox disabled, both sides."""
    _reset_userns_caches(monkeypatch)
    record = []
    monkeypatch.setattr("subprocess.run", fake_run_factory(record, probe_fails=True))

    LegacyCodex().generate(PROMPT)
    PkgCodex().generate(PROMPT)

    old, new = record
    uid, gid = os.getuid(), os.getgid()
    assert _normalize_codex_argv(old["argv"]) == _normalize_codex_argv(new["argv"]) == [
        "unshare", "--user",
        f"--map-users={uid}:{uid}:1", f"--map-groups={gid}:{gid}:1", "--",
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--output-last-message", "<MSGFILE>",
        PROMPT,
    ]
    assert old["env"] == new["env"]


# --- claude-cli -------------------------------------------------------------------------


@pytest.mark.parametrize("model,expected_suffix", [
    (None, []),
    ("opus", ["--model", "opus"]),
    ("fable", ["--model", "fable"]),
    ("gpt-4o", []),  # non-Claude default is ignored by the heuristic
])
def test_claude_cli_argv_identical(cli_env, monkeypatch, model, expected_suffix):
    record = []
    monkeypatch.setattr("subprocess.run",
                        fake_run_factory(record, stdout=json.dumps({"result": "hello"})))

    old_out = LegacyClaude(model=model).generate(PROMPT)
    new_out = PkgClaude(model=model).generate(PROMPT)
    assert old_out == new_out == "hello"

    old, new = record
    assert old["argv"] == new["argv"] == (
        ["claude", "-p", PROMPT, "--output-format", "json"] + expected_suffix
    )


def test_claude_cli_key_stripping_equivalent_plus_known_superset(cli_env, monkeypatch):
    record = []
    monkeypatch.setattr("subprocess.run",
                        fake_run_factory(record, stdout=json.dumps({"result": "x"})))

    LegacyClaude().generate(PROMPT)
    PkgClaude().generate(PROMPT)
    old, new = record

    # Both strip the key that caused the June billing incident.
    assert "ANTHROPIC_API_KEY" not in old["env"]
    assert "ANTHROPIC_API_KEY" not in new["env"]
    # KNOWN, DELIBERATE difference: the package also strips the deprecated
    # CLAUDE_API_KEY spelling (strictly safer). The old code left it in.
    assert old["env"].get("CLAUDE_API_KEY") == "sk-claude-legacy"
    assert "CLAUDE_API_KEY" not in new["env"]
    # Apart from that superset strip, the child environments are identical.
    old_env = {k: v for k, v in old["env"].items() if k != "CLAUDE_API_KEY"}
    assert old_env == new["env"]
    # Other providers' keys are NOT stripped by the claude backend.
    assert new["env"]["OPENAI_API_KEY"] == "sk-openai"

    _assert_neutral_cwd(old["cwd"], "llm-analyzer-agent-")
    _assert_neutral_cwd(new["cwd"], "llm-backends-agent-")


def test_package_strip_opt_out_inherits_parent_env(cli_env, monkeypatch):
    """New capability (not equivalence): strip_provider_keys=False passes
    env=None so the child inherits the parent environment untouched."""
    record = []
    monkeypatch.setattr("subprocess.run",
                        fake_run_factory(record, stdout=json.dumps({"result": "x"})))

    PkgClaude(strip_provider_keys=False).generate(PROMPT)
    assert record[0]["env"] is None


def test_ai_helper_cli_keys_route_to_package_interfaces(cli_env, monkeypatch):
    """The *-cli registry keys now construct the PACKAGE interfaces (which is
    where the key-stripping and userns workaround live)."""
    import ai_helper

    record = []
    monkeypatch.setattr("subprocess.run",
                        fake_run_factory(record, stdout=json.dumps({"result": "from-cli"})))

    out = ai_helper.send_prompt(PROMPT, model="claude-cli-fable")
    assert out == "from-cli"
    assert record[0]["argv"] == ["claude", "-p", PROMPT, "--output-format", "json",
                                 "--model", "fable"]
    assert "ANTHROPIC_API_KEY" not in record[0]["env"]
    _assert_neutral_cwd(record[0]["cwd"], "llm-backends-agent-")


def test_cli_backends_shims_reexport_package_classes():
    """cli_backends/ is now a shim layer: same import path, package classes."""
    import cli_backends
    from llm_backends import (
        ClaudeCliInterface, CodexInterface, GeminiCliInterface,
    )

    assert cli_backends.ClaudeCliInterface is ClaudeCliInterface
    assert cli_backends.CodexInterface is CodexInterface
    assert cli_backends.GeminiCliInterface is GeminiCliInterface

    from cli_backends.agent_cwd import neutral_cwd
    from llm_backends.agent_cwd import neutral_cwd as pkg_neutral_cwd
    assert neutral_cwd is pkg_neutral_cwd
