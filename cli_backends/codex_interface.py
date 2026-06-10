"""Wrapper for calling Codex CLI from Python.

Provides a subprocess-based interface to Codex CLI for zero-cost access to GPT-5.
"""
import os
import subprocess
import shutil
import sys
import tempfile
from typing import List, Optional

from .agent_cwd import neutral_cwd

# Safety posture for `codex exec`. The analyzer only needs codex to *generate text*,
# so it needs zero write/exec access — a read-only sandbox that never prompts is
# both non-interactive and safe. Flag names target codex-cli ~0.118; adjust here if
# a future version renames them.
CODEX_SANDBOX = "read-only"
CODEX_APPROVAL = "never"

# Cached result of the user-namespace probe (None = not yet probed).
_userns_prefix_cache: Optional[List[str]] = None


def _userns_launch_prefix() -> List[str]:
    """Return an `unshare` prefix that runs codex inside an identity-mapped user
    namespace, or [] when it isn't needed/possible.

    Recent codex sandboxes `codex exec` with a *bundled* bubblewrap, which must
    create an unprivileged user namespace. Ubuntu 23.10+ (and similar) block that
    via `kernel.apparmor_restrict_unprivileged_userns`, so codex's own sandbox
    fails. When we detect that block *and* the setuid `newuidmap`/`newgidmap`
    helpers are installed, we instead pre-create a mapped namespace ourselves
    (the helpers are allowed to write the uid_map) and run codex inside it with
    its own sandbox disabled — the outer namespace is the sandbox.

    The map is identity-only (uid/gid -> themselves) so codex still runs as the
    real user and can read its own ~/.codex credentials (no API key needed).

    Returns [] on non-Linux, when the helpers are missing, or when unprivileged
    userns mapping is *not* restricted (there codex's own sandbox works and is
    tighter, so we leave it alone).
    """
    global _userns_prefix_cache
    if _userns_prefix_cache is not None:
        return _userns_prefix_cache

    prefix: List[str] = []
    if sys.platform == "linux" and all(
        shutil.which(b) for b in ("unshare", "newuidmap", "newgidmap")
    ):
        try:
            # Helper-less direct map: succeeds iff unprivileged userns mapping is
            # allowed -> codex's bundled bwrap will also work, so don't wrap.
            subprocess.run(
                ["unshare", "--user", "--map-root-user", "true"],
                capture_output=True, timeout=10, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            # Mapping is restricted; route codex through the setuid-helper path.
            uid, gid = os.getuid(), os.getgid()
            prefix = [
                "unshare", "--user",
                f"--map-users={uid}:{uid}:1",
                f"--map-groups={gid}:{gid}:1",
                "--",
            ]

    _userns_prefix_cache = prefix
    return prefix


class CodexInterface:
    """Interface for calling Codex CLI to access GPT-5."""

    def __init__(self, codex_bin: str = "codex", default_timeout: int = 300):
        """Initialize Codex interface.

        Args:
            codex_bin: Path to codex binary (default: 'codex' in PATH)
            default_timeout: Per-call timeout in seconds for `codex exec`.

        Raises:
            RuntimeError: If Codex CLI is not installed or not in PATH
        """
        self.codex_bin = codex_bin
        self.default_timeout = default_timeout
        self._verify_codex_installed()

    def _verify_codex_installed(self):
        """Check if Codex CLI is installed and accessible.

        Raises:
            RuntimeError: If Codex CLI cannot be found
        """
        if not shutil.which(self.codex_bin):
            raise RuntimeError(
                f"Codex CLI not found at '{self.codex_bin}'. "
                "Install with: npm install -g @openai/codex-cli\n"
                "Then authenticate with: codex auth"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        timeout: Optional[int] = None
    ) -> str:
        """Generate text using Codex CLI.

        Args:
            prompt: The prompt to send to Codex
            max_tokens: Maximum tokens to generate (default: 2000)
            timeout: Timeout in seconds (default: self.default_timeout)

        Returns:
            Generated text from GPT-5

        Raises:
            RuntimeError: If Codex CLI returns an error
            subprocess.TimeoutExpired: If generation times out
        """
        eff_timeout = timeout or self.default_timeout
        msg_file = None
        try:
            # codex writes ONLY the final assistant message to this file, so we don't
            # have to dig the answer out of the agent's verbose exec log on stdout.
            fd, msg_file = tempfile.mkstemp(prefix="codex-msg-", suffix=".txt")
            os.close(fd)
            # Non-interactive text generation. Two confinement paths:
            #   * Default: codex's own read-only sandbox + never-prompt.
            #   * Restricted hosts (see _userns_launch_prefix): run codex inside a
            #     pre-mapped user namespace with codex's own (broken) sandbox
            #     disabled — the outer namespace is the sandbox.
            # Either way we run from a neutral cwd so codex stays a text generator,
            # not a repo agent.
            userns_prefix = _userns_launch_prefix()
            if userns_prefix:
                sandbox_args = ['--dangerously-bypass-approvals-and-sandbox']
            else:
                sandbox_args = ['--sandbox', CODEX_SANDBOX, '--ask-for-approval', CODEX_APPROVAL]
            # Strip OPENAI_API_KEY so codex authenticates via its login (~/.codex),
            # not a metered API key. The key from .env is pulled into os.environ by
            # load_dotenv() and would otherwise be inherited here and billed instead
            # of the subscription. See the billing gotcha in CLAUDE.md.
            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            result = subprocess.run(
                [
                    *userns_prefix,
                    self.codex_bin,
                    'exec',
                    *sandbox_args,
                    '--skip-git-repo-check',
                    '--output-last-message', msg_file,
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=eff_timeout,
                check=True,
                env=env,
                cwd=neutral_cwd(),
            )
            try:
                with open(msg_file, 'r', encoding='utf-8') as f:
                    last_message = f.read().strip()
            except OSError:
                last_message = ""
            return last_message or result.stdout.strip()

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            raise RuntimeError(f"Codex CLI error: {error_msg}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Codex CLI timed out after {eff_timeout}s. "
                "Try increasing timeout or simplifying the prompt."
            )
        finally:
            if msg_file:
                try:
                    os.unlink(msg_file)
                except OSError:
                    pass

    def generate_with_retry(
        self,
        prompt: str,
        max_tokens: int = 2000,
        timeout: Optional[int] = None,
        max_retries: int = 3
    ) -> str:
        """Generate text with automatic retry on failure.

        Args:
            prompt: The prompt to send to Codex
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Generated text from GPT-5

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.generate(prompt, max_tokens, timeout)
            except RuntimeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Could add exponential backoff here if needed
                    continue

        raise RuntimeError(
            f"Codex CLI failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )
