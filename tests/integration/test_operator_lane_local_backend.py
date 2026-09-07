"""The live ComputerUse local backend cannot reach a lifecycle verb (#3233).

Not a stub of the sanitizer: the real ``LocalSandboxBackend`` execs the
real CLI entry point (``python -m kestrel_sovereign.cli``), with the
sovereign key present in the *caller's* environment exactly as an agent's
process would have it. The backend's allowlist drops it, the verb finds no
credential in the environment it was invoked with, and refuses before any
handler runs — so this is safe to run on a host with live agents: nothing
is created, started, terminated, restarted, or updated.

The operator path is exercised in-process in ``tests/unit/test_operator_lane.py``
(handlers patched) rather than here: a real ``kestrel terminate`` that passed
the lane on this host would terminate this host.
"""

from __future__ import annotations

import os
import sys

import pytest

from kestrel_sovereign.features.computer_use.backends.local import LocalSandboxBackend
from kestrel_sovereign.security.operator_lane import LIFECYCLE_VERBS

KEY = "stable-sovereign-key-3233"
VERB_ARGV = {
    "create": ["create", "Nobody"],
    "start": ["start"],
    "terminate": ["terminate"],
    "restart": ["restart"],
    "update": ["update", "--dry-run"],
}


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".env").write_text(f"KESTREL_API_KEY={KEY}\n")
    (tmp_path / "kestrel.toml").write_text("")
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", sorted(LIFECYCLE_VERBS))
async def test_agent_shell_cannot_reach_a_lifecycle_verb(project, verb):
    backend = LocalSandboxBackend({"shell_execution_host", "shell_execution_sandboxed"})
    # The agent process holds the key (the host loaded .env at boot); the
    # subprocess it spawns must not.
    caller_env = {**os.environ, "KESTREL_API_KEY": KEY, "KESTREL_HOME": str(project)}
    run = await backend.exec(
        [sys.executable, "-m", "kestrel_sovereign.cli", *VERB_ARGV[verb]],
        cwd=project,
        env=caller_env,
        timeout=120,
    )
    assert run.returncode == 1, (run.stdout, run.stderr)
    assert f"kestrel {verb} refused" in run.stderr, run.stderr
    assert "no sovereign credential in the invoking environment" in run.stderr


@pytest.mark.asyncio
async def test_a_read_verb_still_runs_from_the_agent_shell(project):
    """The lane is about lifecycle mutations, not the CLI: ``kestrel --help``
    from the same sanitized environment is unaffected."""
    backend = LocalSandboxBackend({"shell_execution_host", "shell_execution_sandboxed"})
    run = await backend.exec(
        [sys.executable, "-m", "kestrel_sovereign.cli", "help", "status"],
        cwd=project,
        env={**os.environ, "KESTREL_API_KEY": KEY},
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert "refused" not in run.stderr
