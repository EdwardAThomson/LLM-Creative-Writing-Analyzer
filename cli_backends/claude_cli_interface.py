"""Shim: the Claude Code CLI interface now lives in the llm-backends package.

Carries the ANTHROPIC_API_KEY subscription-billing strip (originally developed
in this repo, merged upstream in llm-backends stage 2; the package also strips
the deprecated CLAUDE_API_KEY spelling).
"""

from llm_backends.claude_cli_interface import ClaudeCliInterface  # noqa: F401
