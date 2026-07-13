# ai_helper.py
# https://developers.openai.com/api/docs/models

from openai import OpenAI
import os
from dotenv import load_dotenv
import google.generativeai as genai
from anthropic import Anthropic


load_dotenv()  # This will load environment variables from the .env file

# Lazy-initialized API clients
_openai_client = None
_anthropic_client = None
_openrouter_client = None
_gemini_configured = False

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def get_openrouter_client():
    """Return a shared OpenAI-SDK client pointed at OpenRouter (https://openrouter.ai).

    OpenRouter is a hosted, OpenAI-compatible router over many upstream models
    (DeepSeek, Anthropic, and others, proxied). Configured from
    OPENROUTER_API_KEY. Kept as its own singleton, distinct from the plain
    OpenAI client above (different base_url + key), so the two backends can
    coexist in one process.
    """
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _openrouter_client

def get_gemini_configured():
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        _gemini_configured = True

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client

def send_prompt(prompt, model="gpt-5.5"):
    # General OpenRouter passthrough: "openrouter:<upstream-model-id>" routes to
    # OpenRouter with the upstream id verbatim (e.g. "openrouter:deepseek/deepseek-chat",
    # "openrouter:anthropic/claude-haiku-4.5"), so any OpenRouter model works without
    # a code change. Checked BEFORE the exact-match model_config lookup below.
    if model.startswith("openrouter:"):
        upstream_model = model[len("openrouter:"):]
        if not upstream_model:
            raise ValueError(
                f"Unsupported model: {model!r} (missing upstream model id after 'openrouter:')"
            )
        print(f"trying:{model}")
        return send_prompt_openrouter(prompt, model_name=upstream_model)

    # Define configurations for each model. Three families of backend:
    #   * API backends   -> send_prompt_oai / _gemini / _claude (need API keys)
    #   * OpenRouter      -> convenience keys below route to send_prompt_openrouter
    #                       (needs OPENROUTER_API_KEY; see the "openrouter:" prefix above
    #                       for the general passthrough form)
    #   * CLI backends   -> *-cli keys route to locally-installed agent CLIs via
    #                       cli_backends/ (no API keys; see _send_via_*_cli below)
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

    model_config = {
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
        # passthrough form handled above instead of adding a new key here.
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

    # Check if the model is supported
    if model not in model_config:
        raise ValueError(f"Unsupported model: {model}")

    print(f"trying:{model}")
    # Call the corresponding function by looking up the dictionary
    return model_config[model](prompt)


# --- Local CLI backends -----------------------------------------------------------
# These shell out to locally-installed agent CLIs (codex / claude / gemini) in
# headless mode. Imports are lazy so the API-only path never pays for them and a
# missing CLI only errors when that model is actually selected.

def _send_via_codex_cli(prompt):
    """Generate text via the local `codex` CLI (GPT-5)."""
    from cli_backends import CodexInterface
    return CodexInterface().generate_with_retry(prompt)


def _send_via_claude_cli(prompt, model=None):
    """Generate text via the local `claude` CLI in headless mode."""
    from cli_backends import ClaudeCliInterface
    return ClaudeCliInterface(model=model).generate_with_retry(prompt)


def _send_via_gemini_cli(prompt, model="gemini-3-flash-preview"):
    """Generate text via the local `gemini` CLI."""
    from cli_backends import GeminiCliInterface
    return GeminiCliInterface(model=model).generate_with_retry(prompt)


def send_prompt_oai(prompt, model="gpt-5.4", max_tokens=16384, temperature=0.7,
                reasoning_effort="high",
                role_description="You are a helpful fiction writing assistant. You will create original text only."):
    """Send prompts to OpenAI GPT-5.4 family models.

    Args:
        prompt: The text prompt to send.
        model: The model ID (e.g., "gpt-5.4", "gpt-5.4-mini").
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness of generations.
        reasoning_effort: Reasoning effort level (none, low, medium, high, xhigh).
        role_description: System prompt that sets the context for the model.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    print("model used: ", model)
    content = response.choices[0].message.content

    return content


def send_prompt_gemini(prompt, model_name="gemini-3.1-pro-preview", max_output_tokens=8192, temperature=0.7, top_p=1, top_k=40):
    """
    Sends a prompt to the Gemini API and returns the response.

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
    Sends a prompt to Anthropic's Claude API and returns the generated text.

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
        anthropic_client = get_anthropic_client()
        create_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=role_description,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = anthropic_client.messages.create(**create_kwargs)

        print("Used model: ", model)

        return response.content[0].text

    except Exception as e:
        print(f"Error generating content with Claude: {e}")
        return None


def send_prompt_openrouter(prompt, model_name=None, max_tokens=16384, temperature=0.7,
                           role_description="You are a helpful fiction writing assistant. You will create original text only."):
    """
    Sends a prompt to an upstream model via OpenRouter (https://openrouter.ai), a
    hosted OpenAI-compatible router over many providers (DeepSeek, Anthropic,
    and others, proxied through OpenRouter's own upstream model ids).

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
    client = get_openrouter_client()

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": role_description},
                {"role": "user", "content": prompt},
            ],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            # Note: unlike a single-provider backend, no provider-specific
            # extra_body is set here. OpenRouter fans out to many different
            # upstream backends, so a hack tuned for one of them would be
            # silently ignored by most others and would be misleading to carry
            # as a default.
        )

        print("Used model (openrouter): ", model_name)

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating content with OpenRouter: {e}")
        return None
