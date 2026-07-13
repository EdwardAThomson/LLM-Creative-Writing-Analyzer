"""Tests for OpenRouter routing in ai_helper.send_prompt.

These tests verify routing/dispatch logic only — no real OpenRouter (or any
other provider) network calls are made. The OpenAI SDK client construction is
either monkeypatched with a fake, or exercised against a deliberately-unset
env var to confirm the clear error path. Nothing here spends money or touches
the live nd1 pilot / work/corpus/scores/nd1_pilot data.
"""
from __future__ import annotations

import pytest

import ai_helper


@pytest.fixture(autouse=True)
def _reset_openrouter_singleton(monkeypatch):
    """Each test gets a clean lazy-client singleton, regardless of run order."""
    monkeypatch.setattr(ai_helper, "_openrouter_client", None)
    yield
    monkeypatch.setattr(ai_helper, "_openrouter_client", None)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response_text="fake openrouter reply"):
        self.response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletionResponse(self.response_text)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    """Stands in for the real openai.OpenAI client — records constructor args
    and every chat.completions.create() call, makes no network request."""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.completions = _FakeCompletions()
        self.chat = _FakeChat(self.completions)


# --- 1. "openrouter:<upstream-id>" prefix dispatch -----------------------------------


def test_send_prompt_openrouter_prefix_routes_to_openrouter(monkeypatch):
    captured = {}

    def fake_send_prompt_openrouter(prompt, model_name=None, **kwargs):
        captured["prompt"] = prompt
        captured["model_name"] = model_name
        return "ok"

    monkeypatch.setattr(ai_helper, "send_prompt_openrouter", fake_send_prompt_openrouter)

    result = ai_helper.send_prompt("hello world", model="openrouter:deepseek/deepseek-chat")

    assert result == "ok"
    assert captured["prompt"] == "hello world"
    assert captured["model_name"] == "deepseek/deepseek-chat"


def test_send_prompt_openrouter_prefix_supports_anthropic_proxy_ids(monkeypatch):
    captured = {}

    def fake_send_prompt_openrouter(prompt, model_name=None, **kwargs):
        captured["model_name"] = model_name
        return "ok"

    monkeypatch.setattr(ai_helper, "send_prompt_openrouter", fake_send_prompt_openrouter)

    ai_helper.send_prompt("hi", model="openrouter:anthropic/claude-haiku-4.5")

    assert captured["model_name"] == "anthropic/claude-haiku-4.5"


def test_openrouter_prefix_missing_upstream_id_raises():
    with pytest.raises(ValueError, match="openrouter:"):
        ai_helper.send_prompt("hi", model="openrouter:")


def test_openrouter_prefix_checked_before_exact_match_lookup(monkeypatch):
    """The prefix form must never fall through to the "Unsupported model" branch,
    even though "openrouter:deepseek/deepseek-chat" is not (and should never need
    to be) a literal key in model_config."""
    monkeypatch.setattr(
        ai_helper, "send_prompt_openrouter", lambda prompt, model_name=None: "ok"
    )
    # Would raise ValueError("Unsupported model: ...") if the prefix check were
    # missing or placed after the model_config lookup.
    assert ai_helper.send_prompt("hi", model="openrouter:deepseek/deepseek-chat") == "ok"


# --- 2. Convenience keys --------------------------------------------------------------


def test_openrouter_deepseek_convenience_key_maps_to_upstream_id(monkeypatch):
    captured = {}

    def fake_send_prompt_openrouter(prompt, model_name=None, **kwargs):
        captured["model_name"] = model_name
        return "ok"

    monkeypatch.setattr(ai_helper, "send_prompt_openrouter", fake_send_prompt_openrouter)

    ai_helper.send_prompt("hi", model="openrouter-deepseek")

    assert captured["model_name"] == "deepseek/deepseek-chat"


def test_openrouter_haiku_convenience_key_maps_to_upstream_id(monkeypatch):
    captured = {}

    def fake_send_prompt_openrouter(prompt, model_name=None, **kwargs):
        captured["model_name"] = model_name
        return "ok"

    monkeypatch.setattr(ai_helper, "send_prompt_openrouter", fake_send_prompt_openrouter)

    ai_helper.send_prompt("hi", model="openrouter-haiku")

    assert captured["model_name"] == "anthropic/claude-haiku-4.5"


# --- 3. Missing OPENROUTER_API_KEY raises a clear error -------------------------------


def test_get_openrouter_client_raises_when_key_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        ai_helper.get_openrouter_client()


def test_send_prompt_openrouter_raises_when_key_unset_not_swallowed(monkeypatch):
    """The missing-key error must propagate, not be caught and turned into a
    silent None return (unlike the generic provider-error catch further down)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        ai_helper.send_prompt_openrouter("hi", model_name="deepseek/deepseek-chat")


def test_send_prompt_via_prefix_raises_when_key_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        ai_helper.send_prompt("hi", model="openrouter:deepseek/deepseek-chat")


# --- 4. Client points at OpenRouter's base_url / uses OPENROUTER_API_KEY; distinct
#        from the plain OpenAI client ---------------------------------------------


def test_get_openrouter_client_uses_openrouter_base_url_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(ai_helper, "OpenAI", _FakeOpenAIClient)
    # Ensure the plain OpenAI singleton is untouched by this call.
    monkeypatch.setattr(ai_helper, "_openai_client", None)

    client = ai_helper.get_openrouter_client()

    assert isinstance(client, _FakeOpenAIClient)
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.api_key == "test-openrouter-key"
    # The plain OpenAI client singleton must be untouched by the OpenRouter path.
    assert ai_helper._openai_client is None


def test_openrouter_client_is_a_separate_singleton_from_openai_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.setattr(ai_helper, "OpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(ai_helper, "_openai_client", None)
    monkeypatch.setattr(ai_helper, "_anthropic_client", None)

    or_client = ai_helper.get_openrouter_client()
    oai_client = ai_helper.get_openai_client()

    assert or_client is not oai_client
    assert or_client.base_url == "https://openrouter.ai/api/v1"
    assert oai_client.base_url is None  # plain OpenAI client: no base_url override
    assert or_client.api_key == "or-key"
    assert oai_client.api_key == "oai-key"


def test_send_prompt_openrouter_full_flow_uses_fake_client_no_network(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ai_helper, "OpenAI", _FakeOpenAIClient)

    result = ai_helper.send_prompt_openrouter("write a scene", model_name="deepseek/deepseek-chat")

    assert result == "fake openrouter reply"
    client = ai_helper._openrouter_client
    assert isinstance(client, _FakeOpenAIClient)
    assert len(client.completions.calls) == 1
    call_kwargs = client.completions.calls[0]
    assert call_kwargs["model"] == "deepseek/deepseek-chat"
    assert call_kwargs["messages"][-1] == {"role": "user", "content": "write a scene"}


# --- 5. Regression guard: a normal (non-OpenRouter) model is unaffected ---------------


def test_normal_claude_model_unaffected_by_openrouter_prefix_check(monkeypatch):
    captured = {}

    def fake_send_prompt_claude(prompt, model=None, max_tokens=None, temperature=None, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        return "claude reply"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenRouter client should not be constructed for a plain Claude model")

    monkeypatch.setattr(ai_helper, "send_prompt_claude", fake_send_prompt_claude)
    monkeypatch.setattr(ai_helper, "get_openrouter_client", fail_if_called)

    result = ai_helper.send_prompt("hi", model="claude-haiku-4-5")

    assert result == "claude reply"
    assert captured["model"] == "claude-haiku-4-5"


def test_unsupported_model_still_raises_for_non_openrouter_string():
    with pytest.raises(ValueError, match="Unsupported model"):
        ai_helper.send_prompt("hi", model="not-a-real-model")


def test_openrouter_routing_does_not_touch_openai_or_anthropic_clients(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("plain OpenAI/Anthropic client getters must not be used for OpenRouter routing")

    monkeypatch.setattr(ai_helper, "get_openai_client", fail_if_called)
    monkeypatch.setattr(ai_helper, "get_anthropic_client", fail_if_called)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ai_helper, "OpenAI", _FakeOpenAIClient)

    result = ai_helper.send_prompt("hi", model="openrouter:deepseek/deepseek-chat")

    assert result == "fake openrouter reply"


# --- 6. Judge identity: cache keys stay distinct per OpenRouter model -----------------


def test_ai_helper_judge_describe_includes_openrouter_model_id():
    from benchmarks.narrative_dynamics.judge import AiHelperJudge

    judge = AiHelperJudge(model="openrouter:deepseek/deepseek-chat")
    assert judge.describe() == "ai_helper:openrouter:deepseek/deepseek-chat"


def test_ai_helper_judge_describe_distinguishes_deepseek_and_haiku():
    from benchmarks.narrative_dynamics.judge import AiHelperJudge

    deepseek_judge = AiHelperJudge(model="openrouter:deepseek/deepseek-chat")
    haiku_judge = AiHelperJudge(model="openrouter-haiku")

    assert deepseek_judge.describe() != haiku_judge.describe()
    assert "deepseek" in deepseek_judge.describe()
    assert "haiku" in haiku_judge.describe()
