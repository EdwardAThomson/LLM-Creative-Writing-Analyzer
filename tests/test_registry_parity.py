"""Registry parity: ai_helper's frozen model keys vs the llm-backends registry,
and structural derivation of the front-end model lists.

The old CLAUDE.md rule was "keep three places in sync by hand" (ai_helper
registry, llm_creative_tester.DEFAULT_MODELS, llm_tester_ui.AVAILABLE_MODELS).
After the llm-backends adoption the rule is structural:

* AVAILABLE_MODELS = ai_helper.get_supported_models() (no hand copy, checked
  against the source here because llm_tester_ui imports tkinter + heavy utils
  and cannot be imported in the minimal venv);
* DEFAULT_MODELS is validated against the registry at import time;
* every analyzer API key is a PRIMARY key of the llm-backends registry
  (assumption A6), so the package resolves it verbatim; the *-cli keys are
  analyzer-side dispatch (the package exposes CLI backends as classes, not
  registry keys) and must NOT resolve in the package.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import ai_helper
import llm_backends

_REPO = pathlib.Path(__file__).parent.parent

# The analyzer's frozen key list (longitudinal contract). If this changes, it
# must be a deliberate registry change, mirrored in CLAUDE.md.
EXPECTED_KEYS = [
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.2",
    "gemini-3.1-pro-preview", "gemini-3.1-flash-preview",
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
    "openrouter-deepseek", "openrouter-haiku",
    "codex-cli",
    "claude-cli", "claude-cli-opus", "claude-cli-sonnet",
    "claude-cli-haiku", "claude-cli-fable",
    "gemini-cli-pro", "gemini-cli-flash",
]

CLI_KEYS = [k for k in EXPECTED_KEYS if "-cli" in k]
API_KEYS = [k for k in EXPECTED_KEYS if k not in CLI_KEYS]


def test_registry_keys_are_exactly_the_frozen_set_in_order():
    assert ai_helper.get_supported_models() == EXPECTED_KEYS
    assert list(ai_helper.MODEL_CONFIG.keys()) == EXPECTED_KEYS


@pytest.mark.parametrize("key", API_KEYS)
def test_every_api_key_resolves_verbatim_in_llm_backends(key):
    """Assumption A6: the analyzer's hyphenated names are the package
    primaries, so resolution is the identity — no aliasing, no renaming."""
    assert llm_backends.resolve_model(key) == key


@pytest.mark.parametrize("key", CLI_KEYS)
def test_cli_keys_are_analyzer_side_dispatch_not_package_registry(key):
    """Documented split: the *-cli keys are ai_helper registry entries that
    construct the package's CLI interface CLASSES; they are not (and must not
    become) llm-backends registry keys."""
    with pytest.raises(ValueError, match="Unsupported model"):
        llm_backends.resolve_model(key)
    assert key in ai_helper.MODEL_CONFIG


def test_openrouter_prefix_passthrough_matches_package():
    assert llm_backends.resolve_model("openrouter:deepseek/deepseek-chat") == (
        "openrouter:deepseek/deepseek-chat"
    )
    with pytest.raises(ValueError):
        llm_backends.resolve_model("openrouter:")


def _module_ast(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assigned_value(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f"no top-level assignment to {name}")


def test_default_models_is_a_registry_subset():
    """DEFAULT_MODELS (parsed from source: llm_creative_tester imports heavy
    utils and cannot be imported here) must contain only registry keys. The
    module also enforces this at import time."""
    tree = _module_ast(_REPO / "llm_creative_tester.py")
    value = _assigned_value(tree, "DEFAULT_MODELS")
    defaults = ast.literal_eval(value)
    assert defaults, "DEFAULT_MODELS must not be empty"
    unknown = [m for m in defaults if m not in ai_helper.MODEL_CONFIG]
    assert not unknown, f"DEFAULT_MODELS not in registry: {unknown}"
    # And the import-time guard is present.
    src = (_REPO / "llm_creative_tester.py").read_text(encoding="utf-8")
    assert "get_supported_models" in src


def test_available_models_is_derived_not_hand_copied():
    """The 1:1 registry check is structural now: llm_tester_ui assigns
    AVAILABLE_MODELS from get_supported_models(), not a literal list."""
    tree = _module_ast(_REPO / "llm_tester_ui.py")
    value = _assigned_value(tree, "AVAILABLE_MODELS")
    assert isinstance(value, ast.Call), (
        "AVAILABLE_MODELS must be derived from ai_helper.get_supported_models(), "
        "not a hand-maintained literal list"
    )
    assert isinstance(value.func, ast.Name) and value.func.id == "get_supported_models"
