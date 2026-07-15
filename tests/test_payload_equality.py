"""Payload equality: pre-package ai_helper vs the llm-backends-backed ai_helper.

THE migration gate for the llm-backends adoption (StoryDaemon
docs/LLM_BACKENDS_INVENTORY.md section 7.3/7.4 step 3). This repo is a
longitudinal benchmark: the request payload each model key produces is part of
the frozen measurement contract, so the adoption is only valid if the request
kwargs are byte-equal before and after.

Method: the OLD implementation is a verbatim snapshot of ai_helper.py at
commit 22898ef (the last pre-package commit), frozen in
tests/_snapshots/ai_helper_pre_llm_backends.py (sha256
d75602debf8521a78669cb1221e5401e5f7e1bd2f73fd002a1aed7454e0e0f7b). Both the
snapshot and the new ai_helper are driven with the same fake SDK clients,
which record every constructor kwarg and every request kwarg; the recorded
payloads are then compared as canonical JSON bytes.

Coverage (per the migration brief): one OpenAI key (gpt-5.5), claude-fable-5
AND claude-sonnet-4-6 (the omit-sampling-params quirk, both branches), one
Gemini key (gemini-2.5-pro), openrouter-deepseek, and an "openrouter:"-prefixed
model. No network, no SDKs required (fakes are injected via sys.modules /
module attributes), so this runs in the minimal venv.

Known, deliberate NON-payload differences (asserted explicitly below, not
papered over): the package constructs the plain OpenAI and Anthropic clients
with max_retries=1 (SDK-internal retry cap, llm-backends Phase 3 hardening);
the old code left the SDK default (2). This changes retry behavior on
transient errors, never the request payload.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

import ai_helper
from llm_backends import multi_provider_llm as _mpl

_SNAPSHOT = pathlib.Path(__file__).parent / "_snapshots" / "ai_helper_pre_llm_backends.py"

PROMPT = "Write a 300-word scene about a lighthouse keeper."


# --- Fake SDK layer (records constructor kwargs + request kwargs) ---------------------


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="fake reply text")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])


class FakeOpenAI:
    """Stands in for openai.OpenAI; records ctor kwargs and every create() call."""

    instances = []

    def __init__(self, **ctor_kwargs):
        self.ctor_kwargs = ctor_kwargs
        self.completions = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)
        FakeOpenAI.instances.append(self)


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(text="fake reply text")
        return types.SimpleNamespace(content=[block], stop_reason="end_turn")


class FakeAnthropicClient:
    instances = []

    def __init__(self, **ctor_kwargs):
        self.ctor_kwargs = ctor_kwargs
        self.messages = _FakeMessages()
        FakeAnthropicClient.instances.append(self)


def make_fake_anthropic_module():
    mod = types.ModuleType("anthropic")
    mod.Anthropic = FakeAnthropicClient
    return mod


def make_fake_genai_module():
    """A fake google.generativeai; returns (module, recorded_calls)."""
    calls = []

    class GenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GenerativeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config=None, stream=None):
            calls.append({
                "model_name": self.model_name,
                "prompt": prompt,
                "generation_config": dict(generation_config.kwargs),
                "stream": stream,
            })
            return types.SimpleNamespace(text="fake reply text", candidates=[])

    mod = types.ModuleType("google.generativeai")
    mod.configure = lambda api_key=None: calls.append({"configure_api_key": api_key})
    mod.GenerativeModel = GenerativeModel
    mod.types = types.SimpleNamespace(GenerationConfig=GenerationConfig)
    return mod, calls


def canon(payload) -> bytes:
    """Canonical JSON bytes: the byte-equality target for payload comparison."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --- Harness: load legacy snapshot / configure new path with the same fakes -----------


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)


@pytest.fixture()
def legacy(monkeypatch, env):
    """The pre-package ai_helper, loaded from the frozen snapshot with fake SDKs."""
    genai_mod, genai_calls = make_fake_genai_module()

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = FakeOpenAI
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: False
    google_mod = types.ModuleType("google")
    google_mod.generativeai = genai_mod

    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_mod)
    monkeypatch.setitem(sys.modules, "anthropic", make_fake_anthropic_module())

    spec = importlib.util.spec_from_file_location("legacy_ai_helper", _SNAPSHOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._TEST_GENAI_CALLS = genai_calls
    return mod


@pytest.fixture()
def modern(monkeypatch, env):
    """The new (package-backed) ai_helper with the same fakes injected."""
    genai_mod, genai_calls = make_fake_genai_module()

    monkeypatch.setattr(_mpl, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(_mpl, "anthropic", make_fake_anthropic_module())
    monkeypatch.setattr(_mpl, "_openai_client", None)
    monkeypatch.setattr(_mpl, "_openrouter_client", None)
    monkeypatch.setattr(_mpl, "_anthropic_client", None)
    # Gemini stays local in ai_helper (payload freeze, see ai_helper header note).
    monkeypatch.setattr(ai_helper, "genai", genai_mod)
    monkeypatch.setattr(ai_helper, "_gemini_configured", False)
    monkeypatch.setattr(ai_helper, "_TEST_GENAI_CALLS", genai_calls, raising=False)
    return ai_helper


def _openai_call(mod_is_legacy, legacy_mod):
    client = legacy_mod._openai_client if mod_is_legacy else _mpl._openai_client
    assert client is not None and len(client.completions.calls) == 1
    return client.completions.calls[0], client.ctor_kwargs


def _openrouter_call(mod_is_legacy, legacy_mod):
    client = legacy_mod._openrouter_client if mod_is_legacy else _mpl._openrouter_client
    assert client is not None and len(client.completions.calls) == 1
    return client.completions.calls[0], client.ctor_kwargs


def _anthropic_call(mod_is_legacy, legacy_mod):
    client = legacy_mod._anthropic_client if mod_is_legacy else _mpl._anthropic_client
    assert client is not None and len(client.messages.calls) == 1
    return client.messages.calls[0], client.ctor_kwargs


# --- The equality assertions -----------------------------------------------------------


def test_gpt55_payload_byte_equal(legacy, modern):
    old_result = legacy.send_prompt(PROMPT, model="gpt-5.5")
    new_result = modern.send_prompt(PROMPT, model="gpt-5.5")
    old_payload, old_ctor = _openai_call(True, legacy)
    new_payload, new_ctor = _openai_call(False, legacy)

    assert canon(old_payload) == canon(new_payload)
    # Anchor the shared payload to the frozen benchmark profile, so equality
    # can't be satisfied by both sides drifting together.
    assert old_payload == {
        "messages": [
            {"role": "system",
             "content": "You are an expert storyteller focused on character relationships."},
            {"role": "user", "content": PROMPT},
        ],
        "model": "gpt-5.5",
        "max_tokens": 16384,
        "temperature": 0.7,
        "reasoning_effort": "high",
    }
    assert old_result == new_result == "fake reply text"

    # KNOWN, DELIBERATE non-payload difference (llm-backends SDK_MAX_RETRIES=1
    # internal-retry cap). Request payloads above are identical.
    assert old_ctor == {"api_key": "test-openai-key"}
    assert new_ctor == {"api_key": "test-openai-key", "max_retries": 1}


def test_claude_fable5_payload_byte_equal_and_omits_temperature(legacy, modern):
    old_result = legacy.send_prompt(PROMPT, model="claude-fable-5")
    new_result = modern.send_prompt(PROMPT, model="claude-fable-5")
    old_payload, old_ctor = _anthropic_call(True, legacy)
    new_payload, new_ctor = _anthropic_call(False, legacy)

    assert canon(old_payload) == canon(new_payload)
    # The omit-sampling-params quirk: Fable 5 rejects temperature with a 400,
    # so the key must be ABSENT, not None.
    assert "temperature" not in old_payload and "temperature" not in new_payload
    assert old_payload == {
        "model": "claude-fable-5",
        "max_tokens": 8192,
        "system": "You are a skilled creative writer focused on producing original fiction.",
        "messages": [{"role": "user", "content": PROMPT}],
    }
    assert old_result == new_result == "fake reply text"

    # KNOWN, DELIBERATE non-payload difference (internal-retry cap, as above).
    assert old_ctor == {"api_key": "test-anthropic-key"}
    assert new_ctor == {"api_key": "test-anthropic-key", "max_retries": 1}


def test_claude_sonnet46_payload_byte_equal_and_keeps_temperature(legacy, modern):
    legacy.send_prompt(PROMPT, model="claude-sonnet-4-6")
    modern.send_prompt(PROMPT, model="claude-sonnet-4-6")
    old_payload, _ = _anthropic_call(True, legacy)
    new_payload, _ = _anthropic_call(False, legacy)

    assert canon(old_payload) == canon(new_payload)
    assert old_payload["temperature"] == 0.7
    assert old_payload["max_tokens"] == 16384
    assert old_payload["model"] == "claude-sonnet-4-6"
    assert old_payload["system"] == (
        "You are a skilled creative writer focused on producing original fiction."
    )


def test_gemini_payload_byte_equal_including_top_p_top_k(legacy, modern):
    """Gemini stays a LOCAL implementation precisely because llm-backends'
    Gemini function has no top_p/top_k; this proves the frozen payload
    (top_p=1, top_k=40) survived the migration byte-for-byte."""
    old_result = legacy.send_prompt(PROMPT, model="gemini-2.5-pro")
    new_result = modern.send_prompt(PROMPT, model="gemini-2.5-pro")

    old_calls = [c for c in legacy._TEST_GENAI_CALLS if "prompt" in c]
    new_calls = [c for c in modern._TEST_GENAI_CALLS if "prompt" in c]
    assert len(old_calls) == len(new_calls) == 1
    assert canon(old_calls[0]) == canon(new_calls[0])
    assert old_calls[0] == {
        "model_name": "gemini-2.5-pro",
        "prompt": PROMPT,
        "generation_config": {
            "max_output_tokens": 8192, "temperature": 0.7, "top_p": 1, "top_k": 40,
        },
        "stream": False,
    }
    # Both sides configure the SDK with the same key.
    assert {"configure_api_key": "test-gemini-key"} in legacy._TEST_GENAI_CALLS
    assert {"configure_api_key": "test-gemini-key"} in modern._TEST_GENAI_CALLS
    assert old_result == new_result == "fake reply text"


def test_openrouter_deepseek_payload_and_client_byte_equal(legacy, modern):
    old_result = legacy.send_prompt(PROMPT, model="openrouter-deepseek")
    new_result = modern.send_prompt(PROMPT, model="openrouter-deepseek")
    old_payload, old_ctor = _openrouter_call(True, legacy)
    new_payload, new_ctor = _openrouter_call(False, legacy)

    assert canon(old_payload) == canon(new_payload)
    assert old_payload == {
        "messages": [
            {"role": "system",
             "content": "You are a helpful fiction writing assistant. You will create original text only."},
            {"role": "user", "content": PROMPT},
        ],
        "model": "deepseek/deepseek-chat",
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    # The OpenRouter client hardening was ported INTO the package from this
    # repo, so the constructors must match exactly (no known difference here).
    assert canon(old_ctor) == canon(new_ctor)
    assert old_ctor == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-openrouter-key",
        "max_retries": 6,
        "timeout": 120.0,
    }
    assert old_result == new_result == "fake reply text"


def test_openrouter_prefix_payload_byte_equal(legacy, modern):
    model = "openrouter:anthropic/claude-haiku-4.5"
    legacy.send_prompt(PROMPT, model=model)
    modern.send_prompt(PROMPT, model=model)
    old_payload, _ = _openrouter_call(True, legacy)
    new_payload, _ = _openrouter_call(False, legacy)

    assert canon(old_payload) == canon(new_payload)
    assert old_payload["model"] == "anthropic/claude-haiku-4.5"
    assert old_payload["max_tokens"] == 4096
    assert old_payload["temperature"] == 0.7


def test_snapshot_is_the_frozen_pre_migration_source():
    """Guard the OLD side of the comparison: the snapshot must stay byte-identical
    to ai_helper.py @ 22898ef (the last pre-package commit). If this fails,
    someone edited the snapshot and the equality proof no longer means anything."""
    import hashlib

    digest = hashlib.sha256(_SNAPSHOT.read_bytes()).hexdigest()
    assert digest == "d75602debf8521a78669cb1221e5401e5f7e1bd2f73fd002a1aed7454e0e0f7b"


def test_every_registry_key_dispatches_to_the_same_backend_call(legacy, modern, monkeypatch):
    """Key-for-key dispatch parity across the WHOLE registry (all 24 keys,
    including the eight *-cli keys): each key must route to the same backend
    function with the same arguments in the old and new dispatcher, so no
    saved-run model string ever resolves differently."""
    def recorder(store):
        def _rec(name):
            def call(prompt, **kwargs):
                store.append((name, kwargs))
                return "ok"
            return call
        return _rec

    old_calls, new_calls = [], []
    for mod, store in ((legacy, old_calls), (modern, new_calls)):
        rec = recorder(store)
        # The dispatch lambdas late-bind these module globals, so patching the
        # module attributes intercepts every registry entry.
        monkeypatch.setattr(mod, "send_prompt_oai", rec("oai"))
        monkeypatch.setattr(mod, "send_prompt_gemini", rec("gemini"))
        monkeypatch.setattr(mod, "send_prompt_claude", rec("claude"))
        monkeypatch.setattr(mod, "send_prompt_openrouter", rec("openrouter"))
        monkeypatch.setattr(mod, "_send_via_codex_cli", lambda prompt, _s=store: (_s.append(("codex-cli", {})), "ok")[1])
        monkeypatch.setattr(mod, "_send_via_claude_cli", lambda prompt, model=None, _s=store: (_s.append(("claude-cli", {"model": model})), "ok")[1])
        monkeypatch.setattr(mod, "_send_via_gemini_cli", lambda prompt, model=None, _s=store: (_s.append(("gemini-cli", {"model": model})), "ok")[1])

    keys = modern.get_supported_models()
    assert len(keys) == 24
    for key in keys:
        assert legacy.send_prompt(PROMPT, model=key) == "ok"
        assert modern.send_prompt(PROMPT, model=key) == "ok"

    assert len(old_calls) == len(new_calls) == len(keys)
    for key, old, new in zip(keys, old_calls, new_calls):
        assert old == new, f"dispatch mismatch for {key}: {old} != {new}"

    # Rejection contract unchanged too.
    with pytest.raises(ValueError, match="Unsupported model"):
        legacy.send_prompt(PROMPT, model="definitely-not-a-model")
    with pytest.raises(ValueError, match="Unsupported model"):
        modern.send_prompt(PROMPT, model="definitely-not-a-model")
