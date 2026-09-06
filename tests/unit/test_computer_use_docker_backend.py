"""What the Docker sandbox backend hands to the compute executor (#3187).

``SandboxBackend.exec`` is documented as "run ``argv`` and return its
result". This backend used to quote the vector into a bash script and
run *that*, which meant the words were read a second time — by a shell,
after the policy had vetted them. ``eval 'printf HACKED'`` therefore ran
``printf`` while the policy had seen only ``eval``, which is not a
program at all.

These tests pin the translation at the seam: an argv vector goes in, an
argv vector comes out, and the script-shaped path is not taken.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from kestrel_sovereign.features.compute.models import ComputeCommand, ExecutionRecord
from kestrel_sovereign.features.computer_use.backends.base import CapabilityBlocked
from kestrel_sovereign.features.computer_use.backends.docker import (
    DockerSandboxBackend,
)


class _RecordingExecutor:
    """Stands in for ``DockerExecutor``, recording which mode was used."""

    def __init__(self, exit_code: Optional[int] = 0) -> None:
        self.commands: list[ComputeCommand] = []
        self.scripts: list[object] = []
        self.working_dirs: list[Optional[str]] = []
        self._exit_code = exit_code

    async def execute_command(
        self,
        command: ComputeCommand,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        self.commands.append(command)
        self.working_dirs.append(working_dir)
        return ExecutionRecord(
            id="exec-1",
            script_id=command.id,
            exit_code=self._exit_code,
            stdout="out",
            stderr="err",
            executor="docker",
        )

    async def execute(self, script, working_dir: Optional[str] = None):
        self.scripts.append(script)
        raise AssertionError("the backend built a script instead of a command")


def _backend(executor: _RecordingExecutor) -> DockerSandboxBackend:
    backend = DockerSandboxBackend(
        granted_capabilities={"shell_execution_sandboxed"},
    )
    backend._executor = executor  # type: ignore[assignment]
    return backend


@pytest.mark.asyncio
async def test_exec_hands_the_vector_to_the_argv_mode_unchanged() -> None:
    """Every element survives, including ones a shell would have read.

    The vector is deliberately full of shell meaning. Under the old
    implementation each of these words was quoted into a script and
    read back by ``sh``; here they must arrive at the executor as the
    same list of strings that went in.
    """
    executor = _RecordingExecutor()
    argv = ["eval", "printf HACKED", ";", "$(id)", "*"]

    result = await (_backend(executor)).exec(argv, cwd=None, env=None, timeout=30)

    assert executor.scripts == []
    assert len(executor.commands) == 1
    assert executor.commands[0].argv == tuple(argv)
    assert result.argv == argv
    assert result.returncode == 0
    assert (result.stdout, result.stderr) == ("out", "err")


@pytest.mark.asyncio
async def test_exec_passes_the_timeout_environment_and_cwd_through() -> None:
    executor = _RecordingExecutor()

    await (_backend(executor)).exec(
        ["printf", "ok"],
        cwd=Path("/tmp/somewhere"),
        env={"TOKEN": "x"},
        timeout=17,
    )

    command = executor.commands[0]
    assert command.timeout_seconds == 17
    assert command.environment == {"TOKEN": "x"}
    assert executor.working_dirs == ["/tmp/somewhere"]


@pytest.mark.asyncio
async def test_exec_reports_a_missing_exit_code_as_a_failure() -> None:
    """A record with no exit code is not a success."""
    executor = _RecordingExecutor(exit_code=None)

    result = await (_backend(executor)).exec(
        ["printf", "ok"], cwd=None, env=None, timeout=5
    )

    assert result.returncode == -1


@pytest.mark.asyncio
async def test_exec_refuses_an_empty_vector() -> None:
    executor = _RecordingExecutor()

    with pytest.raises(ValueError, match="empty argv"):
        await (_backend(executor)).exec([], cwd=None, env=None, timeout=5)

    assert executor.commands == []


def test_the_backend_still_requires_the_sandboxed_grant() -> None:
    """The execution mode changed; the constitutional gate did not."""
    with pytest.raises(CapabilityBlocked):
        DockerSandboxBackend(granted_capabilities=set())
