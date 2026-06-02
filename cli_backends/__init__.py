"""Local-CLI LLM backends for the creative-writing analyzer.

These wrappers shell out to locally-installed agent CLIs (`codex`, `claude`,
`gemini`) in headless mode so the analyzer can exercise the same models without
API keys. Each interface verifies its CLI is installed on construction and
raises RuntimeError otherwise. Ported from the StoryDaemon writing ecosystem.
"""

from .claude_cli_interface import ClaudeCliInterface
from .gemini_cli_interface import GeminiCliInterface
from .codex_interface import CodexInterface

__all__ = ["ClaudeCliInterface", "GeminiCliInterface", "CodexInterface"]
