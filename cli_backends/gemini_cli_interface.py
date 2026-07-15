"""Shim: the Gemini CLI interface now lives in the llm-backends package.

Carries the GEMINI_API_KEY / GOOGLE_API_KEY subscription-billing strip
(originally developed in this repo, merged upstream in llm-backends stage 2).
"""

from llm_backends.gemini_cli_interface import GeminiCliInterface  # noqa: F401
