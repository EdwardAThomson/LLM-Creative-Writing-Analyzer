"""Shim: the neutral-cwd helper now lives in the llm-backends package.

Keeps agent CLIs (codex / claude / gemini) running from an empty scratch
directory so they generate text instead of acting on this repo.
"""

from llm_backends.agent_cwd import neutral_cwd  # noqa: F401
