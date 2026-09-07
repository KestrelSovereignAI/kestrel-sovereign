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

Only ``create`` and ``terminate`` are executed here. The safety of this
module has to be structural, and for ``start``, ``restart`` (whose second
leg is ``start``) and ``update`` (whose last leg is ``restart``) it cannot
be: if the lane ever failed open, the real ``cmd_start`` would launch a
detached host from this project's ``multi_agent.toml``. On 2026-09-07 a
review's mutation sweep did exactly that; the stray held :8888 for two
hours and the fleet was down. Those three verbs are covered per verb by
``test_every_lifecycle_verb_is_refused_before_its_handler_runs`` with the
handlers patched. For ``create`` a fail-open provisions an agent directory
under the temporary ``KESTREL_HOME``; for ``terminate`` it finds nothing
on an unused port with no pid file. Neither can outlive the test.
"""

from __future__ import annotations

import os
import sys

import pytest

from kestrel_sovereign.features.computer_use.backends.local import LocalSandboxBackend
from kestrel_sovereign.security.operator_lane import LIFECYCLE_VERBS

KEY = "stable-sovereign-key-3233"
# A literal list, not `sorted(LIFECYCLE_VERBS)`: a test that derives its
# cases from the set under test shrinks with the set, so dropping a verb
# from the lane would drop its case too.
THE_FIVE_VERBS = ["create", "start", "terminate", "restart", "update"]


def test_the_lane_covers_exactly_the_five_verbs():
    assert set(LIFECYCLE_VERBS) == set(THE_FIVE_VERBS)


# A literal list again, for the same reason as THE_FIVE_VERBS. See the module
# docstring for why the other three verbs are never executed for real.
EXECUTED_VERBS = ["create", "terminate"]

VERB_ARGV = {
    "create": ["create", "Nobody"],
    "terminate": ["terminate"],
}


@pytest.fixture
def project(tmp_path):
    """A project whose host port is one nothing listens on.

    Structural, not conditional, safety: `MultiAgentConfig.load` on a
    missing file defaults the host port to 8888 — the live host's. If the
    lane ever failed open, `terminate`/`restart` in this directory would
    reap listeners on that port. With a bound-then-closed ephemeral port
    written here, the worst a fail-open could do is find nothing.
    """
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    unused = probe.getsockname()[1]
    probe.close()
    (tmp_path / ".env").write_text(f"KESTREL_API_KEY={KEY}\n")
    (tmp_path / "kestrel.toml").write_text("")
    (tmp_path / "multi_agent.toml").write_text(f'[host]\nport = {unused}\nbind = "127.0.0.1"\n')
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", EXECUTED_VERBS)
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
