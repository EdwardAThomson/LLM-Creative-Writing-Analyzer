"""Local-CLI LLM backends — compatibility shims over the llm-backends package.

The implementations (including this repo's ported billing key-stripping and
the codex bubblewrap/user-namespace workaround) now live in the shared
llm-backends package (pinned in requirements.txt). These shims keep the old
``cli_backends`` import path working; equivalence with the pre-package
modules (frozen in tests/_snapshots/legacy_cli_backends/) is enforced by
tests/test_cli_backend_equivalence.py. Deleting this package in favor of
direct ``llm_backends`` imports is a later cleanup.
"""

from llm_backends import ClaudeCliInterface, CodexInterface, GeminiCliInterface

__all__ = ["ClaudeCliInterface", "GeminiCliInterface", "CodexInterface"]
