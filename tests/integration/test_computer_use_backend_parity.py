"""Do the two sandbox backends answer the same question? (#3187)

The differential is between the two real backends, not between a
backend and a model of one: ``LocalSandboxBackend`` execs the vector on
the host and ``DockerSandboxBackend`` runs it in a container. Both are
the production paths.

Before #3187 they disagreed, and the disagreement was the defect. The
Docker backend quoted the vector into a bash script and ran that, so a
word in command position was read as grammar:

    eval 'printf HACKED'   docker rc=0, printed HACKED   local rc=127
    FOO=x printf pwned     docker rc=0, printed pwned    local rc=127
    trap 'printf X' EXIT   docker rc=0, printed X        local rc=127

``BinaryPolicy`` had vetted ``eval``, ``FOO=x`` and ``trap`` — none of
which is a program. Measured on 2026-09-04 before the fix, through
``DockerSandboxBackend.exec``.

Docker cases are opt-in: set ``KESTREL_TEST_DOCKER=1``. They pull
``alpine:3.19`` on first run.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from kestrel_sovereign.features.computer_use.backends.docker import (
    DockerSandboxBackend,
)
from kestrel_sovereign.features.computer_use.backends.local import (
    LocalSandboxBackend,
)


def _docker_is_running() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        ).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.skipif(
        os.environ.get("KESTREL_TEST_DOCKER", "").lower() not in {"1", "true", "yes"},
        reason="Docker backend test; set KESTREL_TEST_DOCKER=1 and run pytest -m docker.",
    ),
    pytest.mark.skipif(not _docker_is_running(), reason="Docker daemon not running"),
]


# Each of these ran a program under the Docker backend that the policy
# never saw, because the first word is grammar rather than a program.
# `chdir` is BusyBox ash's, not bash's — the container's shell was not
# even the shell the guard was written against (codex round 9 on #3129).
GRAMMAR_VECTORS = [
    ["eval", "printf HACKED"],
    ["FOO=x", "printf", "HACKED"],
    ["exec", "printf", "HACKED"],
    ["trap", "printf HACKED", "EXIT"],
    ["chdir", "/tmp"],
]


@pytest.fixture(scope="module")
def backends() -> tuple[DockerSandboxBackend, LocalSandboxBackend]:
    return (
        DockerSandboxBackend(granted_capabilities={"shell_execution_sandboxed"}),
        LocalSandboxBackend({"shell_execution_sandboxed", "shell_execution_host"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", GRAMMAR_VECTORS, ids=lambda a: a[0])
async def test_a_grammar_word_is_not_a_program_on_either_backend(argv, backends):
    docker_backend, local_backend = backends

    in_container = await docker_backend.exec(argv, cwd=None, env=None, timeout=60)
    on_host = await local_backend.exec(argv, cwd=None, env=None, timeout=60)

    assert "HACKED" not in in_container.stdout, in_container.stdout
    assert in_container.returncode != 0
    assert in_container.returncode == on_host.returncode == 127, (
        f"docker={in_container.returncode} local={on_host.returncode}; the "
        f"two backends must agree on whether {argv[0]!r} is a program"
    )


@pytest.mark.asyncio
async def test_a_real_program_still_runs_on_both_backends(backends):
    """The positive control.

    Without it, every assertion above is satisfied by a backend that
    runs nothing at all.
    """
    docker_backend, local_backend = backends
    argv = ["printf", "ok"]

    in_container = await docker_backend.exec(argv, cwd=None, env=None, timeout=60)
    on_host = await local_backend.exec(argv, cwd=None, env=None, timeout=60)

    assert (in_container.returncode, in_container.stdout) == (0, "ok")
    assert (on_host.returncode, on_host.stdout) == (0, "ok")


@pytest.mark.asyncio
async def test_a_shell_the_caller_names_still_runs(backends):
    """Naming ``sh`` is allowed — and is what the policy then vets.

    The fix is not "no shell may ever run"; it is that a shell runs only
    when the caller asked for one by name, where ``BinaryPolicy`` sees
    ``sh`` as ``argv[0]`` and can allow, deny, or seek approval for it.
    ``eval`` hid that decision; ``sh -c`` cannot.
    """
    docker_backend, local_backend = backends
    argv = ["sh", "-c", "printf named"]

    in_container = await docker_backend.exec(argv, cwd=None, env=None, timeout=60)
    on_host = await local_backend.exec(argv, cwd=None, env=None, timeout=60)

    assert (in_container.returncode, in_container.stdout) == (0, "named")
    assert (on_host.returncode, on_host.stdout) == (0, "named")


@pytest.fixture(scope="module")
def entrypoint_image() -> str:
    """An image that would run its own program given half a chance."""
    tag = "kestrel-test-3187-entrypoint:latest"
    build = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input='FROM alpine:3.19\nENTRYPOINT ["/bin/echo", "ENTRYPOINT-RAN"]\n',
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert build.returncode == 0, build.stderr
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=60)


@pytest.mark.asyncio
async def test_the_image_cannot_interpose_its_own_program(entrypoint_image):
    """codex round 1 P1 — position after the image is not enough.

    Words after the image are not the container's argv: Docker appends
    them to the image's ``ENTRYPOINT``. Measured before the fix, this
    exact image turned ``["printf", "HACKED"]`` into
    ``ENTRYPOINT-RAN printf HACKED`` — ``echo`` ran, ``printf`` became
    an argument, and the policy had vetted ``printf``. #3187 one layer
    down.

    Naming the program with ``--entrypoint`` is what makes ``argv[0]``
    the process regardless of what the image wanted to run.
    """
    from kestrel_sovereign.features.compute.executors.docker_executor import (
        DockerExecutor,
    )
    from kestrel_sovereign.features.compute.models import ComputeCommand

    executor = DockerExecutor(command_image=entrypoint_image)
    record = await executor.execute_command(
        ComputeCommand(
            id="entrypoint-interposition",
            name="entrypoint interposition",
            argv=["printf", "VECTOR-RAN"],
            purpose="prove the image cannot interpose its own program",
            timeout_seconds=60,
        )
    )

    assert "ENTRYPOINT-RAN" not in record.stdout, record.stdout
    assert record.stdout == "VECTOR-RAN", record.stdout
    assert record.exit_code == 0


@pytest.mark.asyncio
async def test_an_image_default_command_cannot_run_in_place_of_the_vector(
    entrypoint_image,
):
    """And the image's own ``CMD`` cannot stand in either.

    ``--entrypoint`` clears it, so the container runs the caller's
    program with the caller's arguments and nothing else.
    """
    from kestrel_sovereign.features.compute.executors.docker_executor import (
        DockerExecutor,
    )
    from kestrel_sovereign.features.compute.models import ComputeCommand

    executor = DockerExecutor(command_image=entrypoint_image)
    record = await executor.execute_command(
        ComputeCommand(
            id="single-word-vector",
            name="single word vector",
            argv=["true"],
            purpose="a vector with no arguments must not inherit a CMD",
            timeout_seconds=60,
        )
    )

    assert record.stdout == "", record.stdout
    assert record.exit_code == 0


@pytest.mark.asyncio
async def test_a_vector_element_is_never_read_as_a_docker_option(backends):
    """``docker run`` stops reading options at the image name.

    So an argv whose first element spells a container privilege is a
    request to run a program with that name, and there isn't one.
    """
    docker_backend, _ = backends

    result = await docker_backend.exec(
        ["--privileged"], cwd=None, env=None, timeout=60
    )

    assert result.returncode == 127, result
