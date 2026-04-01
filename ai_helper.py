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
_gemini_configured = False

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

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

def send_prompt(prompt, model="gpt-5.4"):
    # Define configurations for each model
    model_config = {
        "gpt-5.4": lambda prompt: send_prompt_oai(
            prompt=prompt,
            model="gpt-5.4",
            max_tokens=16384,
            temperature=0.7,
            reasoning_effort="high",
            role_description="You are an expert storyteller focused on character relationships."
        ),
        "gpt-5.4-mini": lambda prompt: send_prompt_oai(
            prompt=prompt,
            model="gpt-5.4-mini",
            max_tokens=16384,
            temperature=0.7,
            reasoning_effort="high",
            role_description="You are an expert storyteller focused on character relationships."
        ),
        "gemini-3.1-pro-preview": lambda prompt: send_prompt_gemini(
            prompt=prompt,
            model_name="gemini-3.1-pro-preview",
            max_output_tokens=8192,
            temperature=0.7,
            top_p=1,
            top_k=40
        ),
        "gemini-3.1-flash-preview": lambda prompt: send_prompt_gemini(
            prompt=prompt,
            model_name="gemini-3.1-flash-preview",
            max_output_tokens=8192,
            temperature=0.7,
            top_p=1,
            top_k=40
        ),
        "claude-opus-4-6": lambda prompt: send_prompt_claude(
            prompt=prompt,
            model="claude-opus-4-6",
            max_tokens=16384,
            temperature=0.7
        ),
        "claude-sonnet-4-6": lambda prompt: send_prompt_claude(
            prompt=prompt,
            model="claude-sonnet-4-6",
            max_tokens=16384,
            temperature=0.7
        ),
        "claude-haiku-4-5": lambda prompt: send_prompt_claude(
            prompt=prompt,
            model="claude-haiku-4-5",
            max_tokens=8192,
            temperature=0.7
        ),
    }

    # Check if the model is supported
    if model not in model_config:
        raise ValueError(f"Unsupported model: {model}")

    print(f"trying:{model}")
    # Call the corresponding function by looking up the dictionary
    return model_config[model](prompt)


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
        model: The Claude model to use (e.g., "claude-opus-4-6").
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness of generations.
        role_description: System prompt that sets the context for the model.

    Returns:
        The generated text, or None if there was an error.
    """
    try:
        anthropic_client = get_anthropic_client()
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=role_description,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("Used model: ", model)

        return response.content[0].text

    except Exception as e:
        print(f"Error generating content with Claude: {e}")
        return None
