# ai_helper.py
#
# Compatibility layer over the shared llm-backends package
# (https://github.com/EdwardAThomson/llm-backends, pinned in requirements.txt).
#
# This repo is a LONGITUDINAL BENCHMARK: the request payload each model key
# produces is part of the frozen measurement contract. This module therefore
# keeps the analyzer's exact public surface and per-model parameter profile
# (model id, system prompt, max_tokens, temperature presence/absence,
# reasoning_effort) while delegating the actual provider calls to
# llm_backends.multi_provider_llm and the CLI backends to the package's
# hardened interfaces (which carry this repo's own ported key-stripping and
# codex userns workaround).
#
# Payload equality with the pre-package implementation (frozen at commit
# 22898ef, snapshot in tests/_snapshots/ai_helper_pre_llm_backends.py) is
# enforced by tests/test_payload_equality.py. If you change any default here,
# that test MUST be updated knowingly — silent default drift forks the
# longitudinal series.
#
# One deliberate NON-delegation: the Gemini API path stays local. The
# package's send_prompt_gemini_meta does not accept top_p/top_k, and this
# repo's frozen Gemini payload sends top_p=1, top_k=40. Delegating would
# silently change the request. Revisit only if llm-backends grows those
# params (and then re-verify with the payload test).

import os

from llm_backends import multi_provider_llm as _mpl

try:
    from dotenv import load_dotenv
except ImportError:  # minimal test venv: no python-dotenv, env comes from the shell
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    import google.generativeai as genai
except ImportError:  # only needed when a gemini-* API model is actually called
    genai = None


load_dotenv()  # This will load environment variables from the .env file

# Gemini keeps its own lazy-config flag because its payload path stays local
# (see header note). The other providers' client singletons now live in
# llm_backends.multi_provider_llm (shared, one per process).
_gemini_configured = False


def get_openai_client():
    """Return the shared OpenAI client (owned by llm_backends).

    Keeps the analyzer's historical error surface: a missing key raises
    ValueError with the original message (the package alone would raise
    RuntimeError).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return _mpl._get_openai_client()


def get_openrouter_client():
    """Return the shared OpenAI-SDK client pointed at OpenRouter (https://openrouter.ai).

    The client itself (base_url, OPENROUTER_API_KEY, and the hardened
    max_retries=6 / timeout=120s construction — see the evidence comment on
    OPENROUTER_MAX_RETRIES in llm_backends.multi_provider_llm, which was
    ported verbatim from this file) now lives in llm_backends. This wrapper
    keeps the analyzer's ValueError on a missing key.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return _mpl._get_openrouter_client()


def get_gemini_configured():
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        if genai is None:
            raise RuntimeError(
                "google-generativeai package is not installed. "
                "Install it with 'pip install google-generativeai'."
            )
        genai.configure(api_key=api_key)
        _gemini_configured = True


def get_anthropic_client():
    """Return the shared Anthropic client (owned by llm_backends).

    Keeps the analyzer's historical surface: only ANTHROPIC_API_KEY is
    accepted (no CLAUDE_API_KEY fallback here), and a missing key raises
    ValueError with the original message.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return _mpl._get_anthropic_client()


# --- Model registry ---------------------------------------------------------------
# Module-level (previously built inside send_prompt) so llm_creative_tester /
# llm_tester_ui can derive their model lists structurally instead of keeping
# hand-synced copies. Keys are the analyzer's frozen model keys; the API keys
# are also the llm-backends primary keys (assumption A6), which
# tests/test_registry_parity.py verifies. The lambdas late-bind the module
# functions (looked up in globals() at call time), so monkeypatching
# ai_helper.send_prompt_openrouter etc. still intercepts dispatch.
#
# Three families of backend:
#   * API backends   -> send_prompt_oai / _gemini / _claude (need API keys)
#   * OpenRouter      -> convenience keys route to send_prompt_openrouter
#                       (needs OPENROUTER_API_KEY; see the "openrouter:" prefix
#                       in send_prompt for the general passthrough form)
#   * CLI backends   -> *-cli keys route to locally-installed agent CLIs via
#                       llm_backends' hardened interfaces (no API keys)

def _gpt(model_name):
    return lambda prompt: send_prompt_oai(
        prompt=prompt,
        model=model_name,
        max_tokens=16384,
        temperature=0.7,
        reasoning_effort="high",
        role_description="You are an expert storyteller focused on character relationships."
    )


def _gemini(model_name):
    return lambda prompt: send_prompt_gemini(
        prompt=prompt,
        model_name=model_name,
        max_output_tokens=8192,
        temperature=0.7,
        top_p=1,
        top_k=40
    )


def _claude(model_name, max_tokens=16384, temperature=0.7):
    # Fable 5 / Opus 4.8 / 4.7 removed the sampling params: sending
    # temperature/top_p/top_k returns a 400. Pass temperature=None for those
    # so send_prompt_claude omits it (steer via prompt instead).
    return lambda prompt: send_prompt_claude(
        prompt=prompt,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature
    )


MODEL_CONFIG = {
    # --- OpenAI GPT-5 family (API) ---
    "gpt-5.5": _gpt("gpt-5.5"),
    "gpt-5.4": _gpt("gpt-5.4"),
    "gpt-5.4-mini": _gpt("gpt-5.4-mini"),
    "gpt-5.2": _gpt("gpt-5.2"),
    # --- Google Gemini (API) ---
    "gemini-3.1-pro-preview": _gemini("gemini-3.1-pro-preview"),
    "gemini-3.1-flash-preview": _gemini("gemini-3.1-flash-preview"),
    "gemini-3-pro-preview": _gemini("gemini-3-pro-preview"),
    "gemini-3-flash-preview": _gemini("gemini-3-flash-preview"),
    "gemini-2.5-pro": _gemini("gemini-2.5-pro"),
    "gemini-2.5-flash": _gemini("gemini-2.5-flash"),
    # --- Anthropic Claude (API) ---
    # Fable 5 and Opus 4.8 reject sampling params -> temperature=None (omitted).
    "claude-fable-5": _claude("claude-fable-5", max_tokens=8192, temperature=None),
    "claude-opus-4-8": _claude("claude-opus-4-8", temperature=None),
    "claude-sonnet-4-6": _claude("claude-sonnet-4-6"),
    "claude-haiku-4-5": _claude("claude-haiku-4-5", max_tokens=8192),

    # --- OpenRouter convenience keys (API; needs OPENROUTER_API_KEY) ---
    # For any other OpenRouter model, use the "openrouter:<upstream-model-id>"
    # passthrough form handled in send_prompt instead of adding a new key here.
    "openrouter-deepseek": lambda prompt: send_prompt_openrouter(
        prompt, model_name="deepseek/deepseek-chat"
    ),
    "openrouter-haiku": lambda prompt: send_prompt_openrouter(
        prompt, model_name="anthropic/claude-haiku-4.5"
    ),

    # --- Local CLI backends (no API keys; require the CLI on PATH) ---
    "codex-cli": lambda prompt: _send_via_codex_cli(prompt),
    "claude-cli": lambda prompt: _send_via_claude_cli(prompt, model=None),
    "claude-cli-opus": lambda prompt: _send_via_claude_cli(prompt, model="opus"),
    "claude-cli-sonnet": lambda prompt: _send_via_claude_cli(prompt, model="sonnet"),
    "claude-cli-haiku": lambda prompt: _send_via_claude_cli(prompt, model="haiku"),
    "claude-cli-fable": lambda prompt: _send_via_claude_cli(prompt, model="fable"),
    "gemini-cli-pro": lambda prompt: _send_via_gemini_cli(prompt, model="gemini-3-pro-preview"),
    "gemini-cli-flash": lambda prompt: _send_via_gemini_cli(prompt, model="gemini-3-flash-preview"),
}


def get_supported_models():
    """The analyzer's model keys, in registry (dropdown) order.

    llm_tester_ui.AVAILABLE_MODELS and the DEFAULT_MODELS validation in
    llm_creative_tester derive from this, so the old "keep three places in
    sync" rule is now structural.
    """
    return list(MODEL_CONFIG.keys())


def send_prompt(prompt, model="gpt-5.5"):
    # General OpenRouter passthrough: "openrouter:<upstream-model-id>" routes to
    # OpenRouter with the upstream id verbatim (e.g. "openrouter:deepseek/deepseek-chat",
    # "openrouter:anthropic/claude-haiku-4.5"), so any OpenRouter model works without
    # a code change. Checked BEFORE the exact-match MODEL_CONFIG lookup below.
    if model.startswith("openrouter:"):
        upstream_model = model[len("openrouter:"):]
        if not upstream_model:
            raise ValueError(
                f"Unsupported model: {model!r} (missing upstream model id after 'openrouter:')"
            )
        print(f"trying:{model}")
        return send_prompt_openrouter(prompt, model_name=upstream_model)

    # Check if the model is supported
    if model not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model: {model}")

    print(f"trying:{model}")
    # Call the corresponding function by looking up the dictionary
    return MODEL_CONFIG[model](prompt)


# --- Local CLI backends -----------------------------------------------------------
# These shell out to locally-installed agent CLIs (codex / claude / gemini) in
# headless mode, via the llm-backends package interfaces (which contain this
# repo's ported key-stripping and the codex userns workaround; equivalence with
# the pre-package cli_backends/ is enforced by tests/test_cli_backend_equivalence.py).
# Imports are lazy so the API-only path never pays for them and a missing CLI
# only errors when that model is actually selected.

def _send_via_codex_cli(prompt):
    """Generate text via the local `codex` CLI (GPT-5)."""
    from llm_backends import CodexInterface
    return CodexInterface().generate_with_retry(prompt)


def _send_via_claude_cli(prompt, model=None):
    """Generate text via the local `claude` CLI in headless mode."""
    from llm_backends import ClaudeCliInterface
    return ClaudeCliInterface(model=model).generate_with_retry(prompt)


def _send_via_gemini_cli(prompt, model="gemini-3-flash-preview"):
    """Generate text via the local `gemini` CLI."""
    from llm_backends import GeminiCliInterface
    return GeminiCliInterface(model=model).generate_with_retry(prompt)


def send_prompt_oai(prompt, model="gpt-5.4", max_tokens=16384, temperature=0.7,
                reasoning_effort="high",
                role_description="You are a helpful fiction writing assistant. You will create original text only."):
    """Send prompts to OpenAI GPT-5.4 family models (via llm_backends).

    Args:
        prompt: The text prompt to send.
        model: The model ID (e.g., "gpt-5.4", "gpt-5.4-mini").
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness of generations.
        reasoning_effort: Reasoning effort level (none, low, medium, high, xhigh).
        role_description: System prompt that sets the context for the model.
    """
    get_openai_client()  # analyzer error surface: ValueError on a missing key
    content = _mpl.send_prompt_openai(
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        role_description=role_description,
        reasoning_effort=reasoning_effort,
    )

    print("model used: ", model)

    return content


def send_prompt_gemini(prompt, model_name="gemini-3.1-pro-preview", max_output_tokens=8192, temperature=0.7, top_p=1, top_k=40):
    """
    Sends a prompt to the Gemini API and returns the response.

    NOT delegated to llm_backends: the package's Gemini function has no
    top_p/top_k, and this frozen payload sends top_p=1, top_k=40 (see the
    header note).

    Args:
        prompt: The text prompt to send.
        model_name: The name of the Gemini model to use.
        max_output_tokens: The maximum number of tokens to generate.
        temperature: Controls the randomness of the output.
        top_p: Controls the diversity of the output.
        top_k: Controls the diversity of the output (similar to top_p).
    Returns:
        The generated text, or None if there was an error.
    """

    get_gemini_configured()
    model = genai.GenerativeModel(model_name)

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )


    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=False
        )

        print("Used model: ", model)

        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return None


def send_prompt_claude(prompt, model="claude-sonnet-4-6", max_tokens=16384, temperature=0.7,
                     role_description="You are a skilled creative writer focused on producing original fiction."):
    """
    Sends a prompt to Anthropic's Claude API (via llm_backends) and returns the text.

    Args:
        prompt: The text prompt to send.
        model: The Claude model to use (e.g., "claude-fable-5", "claude-opus-4-8").
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness of generations. Pass None to omit it
            entirely — required for Fable 5 / Opus 4.8 / 4.7, which reject the
            sampling params (temperature/top_p/top_k) with a 400 error.
        role_description: System prompt that sets the context for the model.

    Returns:
        The generated text, or None if there was an error.
    """
    try:
        get_anthropic_client()  # analyzer error surface: ValueError on a missing key,
        # swallowed below into a None return, exactly as before.
        text = _mpl.send_prompt_claude(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            role_description=role_description,
        )

        print("Used model: ", model)

        return text

    except Exception as e:
        print(f"Error generating content with Claude: {e}")
        return None


def send_prompt_openrouter(prompt, model_name=None, max_tokens=4096, temperature=0.7,
                           role_description="You are a helpful fiction writing assistant. You will create original text only."):
    # max_tokens default kept at 4096 (lowered from 16384 pre-OpenRouter-adoption):
    # OpenRouter gates requests on the reserved max_tokens (a 402 fires if the
    # balance can't cover the *max* possible output), and the nd1 judge's outputs
    # are tiny. 4096 is ~5x the largest real response, so it never truncates,
    # while not over-reserving credit.
    """
    Sends a prompt to an upstream model via OpenRouter (https://openrouter.ai),
    delegating to llm_backends (whose hardened client was ported from this file).

    Args:
        prompt: The text prompt to send.
        model_name: The upstream OpenRouter model id, e.g. "deepseek/deepseek-chat"
            or "anthropic/claude-haiku-4.5". Required.
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness of generations.
        role_description: System prompt that sets the context for the model.

    Returns:
        The generated text, or None if there was an error.
    """
    if not model_name:
        raise ValueError(
            "model_name must be specified for send_prompt_openrouter "
            "(e.g. 'deepseek/deepseek-chat')"
        )
    # Not wrapped in the try/except below: a missing OPENROUTER_API_KEY should
    # raise clearly (same as get_gemini_configured()'s precedent in this file),
    # not be swallowed into a silent None return.
    get_openrouter_client()

    try:
        text = _mpl.send_prompt_openrouter(
            prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            role_description=role_description,
        )

        print("Used model (openrouter): ", model_name)

        return text

    except Exception as e:
        print(f"Error generating content with OpenRouter: {e}")
        return None
