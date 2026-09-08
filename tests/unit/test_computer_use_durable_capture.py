"""A long review must leave an artifact, and a clipped one must not read as a verdict (#3243).

The governed shell has no shell: ``shlex`` tokenizes and the argv vector is
executed directly (#3129), and ``bash -lc`` is deliberately absent (#3130).
So the documented merge-gate form::

    cd <worktree> && claude -p --model <opus> "Review..." > review.txt

is unexecutable in both halves — ``&&`` and ``>`` are refused as shell
grammar, and quoting them turns them into literal arguments handed to the
reviewer. The only way back was the tool result, capped at 1 MiB, with the
clip announced on ``CompletedRun`` in a field ``shell`` never forwarded.

On 2026-08-31 the agent ran eight unprompted wakes against PR #3112 and every
one ended at that gate. Its own words: "the log file is the only reason a
verdict exists."

These tests pin the four things the ticket asked for, and every one of them
drives real files and real subprocesses where it can — a stubbed backend
cannot show that a 2 MiB review survives, which is the entire claim.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.computer_use import capture
from kestrel_sovereign.features.computer_use.backends.base import (
    CaptureTarget,
    CompletedRun,
)
from kestrel_sovereign.features.computer_use.backends.local import LocalSandboxBackend
from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from kestrel_sovereign.privacy import PrivacyConfig


# --- fixtures ---------------------------------------------------------------


class FakeApprovalQueue:
    def __init__(self, decision: tuple[bool, str] = (True, "once")):
        self.decision = decision
        self.calls: list[dict] = []

    async def request_approval(self, feature_name, tool_name, tool_args, timeout):
        self.calls.append({"tool": tool_name, "args": tool_args})
        return self.decision


class FakeSecurityFeature:
    def __init__(self, queue):
        self.approval_queue = queue


class FakeAgent:
    def __init__(self, *, queue: FakeApprovalQueue):
        self.privacy_config = PrivacyConfig(computer_access=True)
        self.granted_capabilities = frozenset(
            {
                "filesystem_read",
                "filesystem_write",
                "shell_execution_sandboxed",
                "shell_execution_host",
            }
        )
        self._security = FakeSecurityFeature(queue)
        self.did = "did:test:agent"
        self.features = {"security": self._security}

    def get_feature(self, name):
        return self.features.get(name)


GRANTS = frozenset({"shell_execution_sandboxed", "shell_execution_host"})


def _config(ws: Path, **over: Any) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "backend": "local",
        "allowed_paths": [str(ws)],
        "deny_paths": [str(ws / "secret")],
        "auto_approved_binaries": ["echo", "sh", "python3", "git", "true", "yes", "head"],
        "denied_binaries": ["rm"],
        "auto_approve_read": True,
        "audit_log_path": str(ws / "audit.jsonl"),
        "capture_dir": str(ws / "captures"),
    }
    cfg.update(over)
    return cfg


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "secret").mkdir()
    (tmp_path / "captures").mkdir()
    return tmp_path


@pytest.fixture()
def queue() -> FakeApprovalQueue:
    return FakeApprovalQueue()


async def _feature(ws: Path, queue: FakeApprovalQueue, **over: Any) -> ComputerUseFeature:
    f = ComputerUseFeature(FakeAgent(queue=queue))
    f._cfg = _config(ws, **over)
    await f.initialize()
    return f


def _git_repo(path: Path) -> str:
    """A real repository with one commit. Returns its HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(a, cwd=path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (path / "f.txt").write_text("one")
    run("git", "add", "f.txt")
    run("git", "commit", "-qm", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# 1. The artifact: output that outlives the tool result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_capture_survives_output_far_larger_than_the_inline_cap(tmp_path: Path):
    """The whole claim, at the layer that makes it true.

    The inline path buffers into memory and clips at ``_MAX_OUTPUT_BYTES``.
    A capture hands the child the file descriptor, so the bytes never pass
    through this process — there is no cap to exceed. Driven with a real
    subprocess producing ~2 MiB, because a stubbed backend cannot show it.
    """
    target = CaptureTarget(
        stdout_path=tmp_path / "out.txt", stderr_path=tmp_path / "err.txt"
    )
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", "-c", "print('x' * 2_000_000)"],
        cwd=None,
        env=None,
        timeout=60,
        capture=target,
    )

    assert result.returncode == 0
    assert result.truncated_stdout is False
    written = target.stdout_path.stat().st_size
    assert written > 2_000_000, f"only {written} bytes reached the file"
    assert result.stdout_path == str(target.stdout_path)


@pytest.mark.asyncio
async def test_without_a_capture_the_same_output_is_clipped_and_says_so(tmp_path: Path):
    """Control, and the behaviour the capture exists to escape. If this ever
    stopped clipping, the test above would prove nothing."""
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", "-c", "print('x' * 2_000_000)"],
        cwd=None,
        env=None,
        timeout=60,
    )

    assert result.truncated_stdout is True
    assert len(result.stdout) < 2_000_000
    assert result.stdout_path is None


@pytest.mark.asyncio
async def test_the_capture_path_is_allocated_by_the_runtime(workspace: Path, queue):
    """No parameter names the file. A ``capture_to`` would have been a way to
    write any path by calling it somewhere to put output — reaching the
    filesystem through the shell gate instead of the write gate."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hello", capture_output=True)

    stdout_path = Path(env.data["stdout_path"])
    assert stdout_path.parent == workspace / "captures"
    assert stdout_path.read_text().strip() == "hello"


# ---------------------------------------------------------------------------
# 2. Truncation is never a verdict
# ---------------------------------------------------------------------------


class _StubBackend:
    """Returns exactly the CompletedRun it was given."""

    name = "local"

    def __init__(self, run: CompletedRun):
        self._run = run
        self.calls: list[dict] = []

    async def exec(self, argv, *, cwd, env, timeout, capture=None):
        self.calls.append({"argv": argv, "cwd": cwd, "capture": capture})
        return self._run

    async def shutdown(self):
        pass


def _run(**over: Any) -> CompletedRun:
    base = dict(
        argv=["echo", "hi"],
        returncode=0,
        stdout="...a long review ending in: VERDICT: APPROVE",
        stderr="",
        duration_ms=5,
    )
    base.update(over)
    return CompletedRun(**base)


@pytest.mark.asyncio
async def test_a_clipped_run_is_partial_even_at_rc_zero(workspace: Path, queue):
    """The failure this ticket is named for. A review clipped mid-argument
    still ends in a paragraph that reads like a verdict, and the process that
    produced it exited 0."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(truncated_stdout=True))

    env = await f.shell(command="echo hi")

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["truncated_stdout"] is True
    assert env.data["complete"] is False
    # ToolResult.partial carries the caveat in ``error``.
    assert "INCOMPLETE" in (env.error or "")


@pytest.mark.asyncio
async def test_a_clipped_stderr_also_refuses_success(workspace: Path, queue):
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(truncated_stderr=True))

    env = await f.shell(command="echo hi")

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["truncated_stderr"] is True


@pytest.mark.asyncio
async def test_a_whole_run_is_still_ok(workspace: Path, queue):
    """Control for both of the above: the refusal must key on incompleteness,
    not be a blanket demotion of every shell result."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run())

    env = await f.shell(command="echo hi")

    assert env.status is ToolResultStatus.OK
    assert env.data["complete"] is True
    assert env.data["truncated_stdout"] is False


@pytest.mark.asyncio
async def test_the_truncation_flags_reach_the_caller_at_all(workspace: Path, queue):
    """They were computed by both backends and dropped by ``shell``, so a
    caller could not have honoured them even intending to."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(truncated_stdout=True, truncated_stderr=True))

    env = await f.shell(command="echo hi")

    assert {"truncated_stdout", "truncated_stderr", "complete"} <= set(env.data)


# ---------------------------------------------------------------------------
# 3. The manifest, and the head it reviewed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_manifest_records_the_head_the_command_ran_against(
    workspace: Path, queue
):
    """A verdict is about a specific tree. During the 08-31 run the head moved
    three times in eight hours, and nothing recorded which one was reviewed."""
    repo = workspace / "repo"
    head = _git_repo(repo)
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo reviewing", cwd=str(repo), capture_output=True)

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["git"]["head_before"] == head
    assert body["git"]["head_after"] == head
    assert body["git"]["head_moved"] is False
    assert body["cwd"] == str(repo.resolve())


@pytest.mark.asyncio
async def test_a_head_that_moves_during_the_run_is_recorded_as_moved(
    workspace: Path, queue
):
    """The case the field exists for. The command itself commits, so the tree
    the review started on is not the tree it finished on."""
    repo = workspace / "repo"
    before = _git_repo(repo)
    script = repo / "move.sh"
    script.write_text(
        "cd %s\ngit commit -q --allow-empty -m moved\n" % repo
    )
    f = await _feature(workspace, queue)

    env = await f.shell(command=f"sh {script}", cwd=str(repo), capture_output=True)

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["git"]["head_before"] == before
    assert body["git"]["head_after"] != before
    assert body["git"]["head_moved"] is True


@pytest.mark.asyncio
async def test_a_directory_that_is_not_a_repository_records_no_sha(
    workspace: Path, queue
):
    """Unknown, not guessed. A manifest carrying a WRONG sha would be worse
    than one carrying none, and ``head_moved`` stays null rather than
    claiming nothing moved on the strength of two unknowns."""
    plain = workspace / "plain"
    plain.mkdir()
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", cwd=str(plain), capture_output=True)

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["git"]["head_before"] is None
    assert body["git"]["head_moved"] is None


@pytest.mark.asyncio
async def test_the_manifest_carries_one_field_a_gate_can_read(workspace: Path, queue):
    """``complete`` is the conjunction of every way this run could be less
    than the whole thing, so a gate cannot pass by checking the single
    condition its author happened to remember."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(truncated_stdout=True))

    env = await f.shell(command="echo hi", capture_output=True)
    body = json.loads(Path(env.data["manifest_path"]).read_text())

    assert body["complete"] is False
    assert body["truncated_stdout"] is True


# ---------------------------------------------------------------------------
# 4. A timeout is not a result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timed_out_run_is_never_ok_even_at_rc_zero(workspace: Path, queue):
    """The contradictory triple, verbatim from the 08-31 run: ``rc=0``,
    ``success:true`` and ``timed_out:true`` reported together. Reachable
    whenever the process exits between the timeout firing and the kill —
    "the process exited 0" and "the work finished" are different claims."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(returncode=0, timed_out=True))

    env = await f.shell(command="echo hi")

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["timed_out"] is True
    assert env.data["complete"] is False
    assert "timed out" in (env.error or "")


@pytest.mark.asyncio
async def test_a_real_timeout_reports_itself(workspace: Path, queue):
    """Through the real backend and a real process, not a stub verdict."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="python3 -c 'import time; time.sleep(30)'", timeout=1)

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["timed_out"] is True


@pytest.mark.asyncio
async def test_the_docker_backend_reports_a_timeout_instead_of_raising():
    """It raised ``ExecutionTimeoutError`` out of ``exec`` entirely, so
    ``timed_out`` was structurally always False on that backend and the
    caller saw a generic error rather than the outcome."""
    from kestrel_sovereign.features.compute.executors.base import ExecutionTimeoutError
    from kestrel_sovereign.features.computer_use.backends.docker import (
        DockerSandboxBackend,
    )

    backend = DockerSandboxBackend.__new__(DockerSandboxBackend)

    class _Executor:
        async def execute_command(self, command, working_dir=None):
            raise ExecutionTimeoutError("cmd", 5)

    backend._executor = _Executor()

    result = await backend.exec(["echo", "hi"], cwd=None, env=None, timeout=5)

    assert result.timed_out is True
    assert result.returncode != 0


@pytest.mark.asyncio
async def test_the_docker_backend_recovers_truncation_from_the_marker():
    """That backend never set ``truncated_stdout`` at all: the executor
    signals a clip by appending a marker to the text and keeps no boolean,
    so however much was discarded the flag stayed False."""
    from kestrel_sovereign.features.compute.executors.base import (
        _OUTPUT_TRUNCATED_SUFFIX,
    )
    from kestrel_sovereign.features.computer_use.backends.docker import (
        DockerSandboxBackend,
    )

    backend = DockerSandboxBackend.__new__(DockerSandboxBackend)

    class _Executor:
        async def execute_command(self, command, working_dir=None):
            class _Rec:
                exit_code = 0
                stdout = "a review" + _OUTPUT_TRUNCATED_SUFFIX
                stderr = "clean"

            return _Rec()

    backend._executor = _Executor()

    result = await backend.exec(["echo", "hi"], cwd=None, env=None, timeout=5)

    assert result.truncated_stdout is True
    assert result.truncated_stderr is False
    # The marker is prose, not payload; it must not stay in the text a
    # reader would treat as the review.
    assert result.stdout == "a review"


# ---------------------------------------------------------------------------
# The working directory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_command_runs_in_the_named_directory(workspace: Path, queue):
    """``cd X && Y`` is refused as shell grammar, so without this the other
    half of the documented gate form is unexecutable too."""
    sub = workspace / "sub"
    sub.mkdir()
    f = await _feature(workspace, queue)

    env = await f.shell(command="python3 -c 'import os; print(os.getcwd())'", cwd=str(sub))

    assert env.data["stdout"].strip() == str(sub.resolve())


@pytest.mark.asyncio
async def test_a_denied_directory_is_refused_before_anything_runs(
    workspace: Path, queue
):
    """The working directory is the most consequential path in the call —
    every relative token resolves against it — and it is not an argv token,
    so the existing argument check never sees it."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", cwd=str(workspace / "secret"))

    assert env.status is not ToolResultStatus.OK
    assert "policy:cwd" in (env.error or "")


@pytest.mark.asyncio
async def test_a_traversal_cwd_is_refused(workspace: Path, queue):
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", cwd=str(workspace) + "/../etc")

    assert env.status is not ToolResultStatus.OK
    assert "path_safety" in (env.error or "")


@pytest.mark.asyncio
async def test_relative_argv_tokens_are_vetted_against_the_run_directory(
    workspace: Path, queue
):
    """The argument check resolved relative tokens against the SERVER's cwd,
    which is not where the command runs — so it vetted paths that were not
    the ones being opened. A denied file named relatively must still deny."""
    f = await _feature(workspace, queue)
    (workspace / "secret" / "leak.txt").write_text("keys")

    env = await f.shell(command="echo secret/leak.txt", cwd=str(workspace))

    assert env.status is not ToolResultStatus.OK
    assert "path_policy" in (env.error or "")


# ---------------------------------------------------------------------------
# Preview is not truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_preview_shows_the_tail_where_a_verdict_lives(tmp_path: Path):
    """Head-only is the exact window that made a clipped review look
    finished: a review states its verdict last."""
    body = ("filler\n" * 5000) + "VERDICT: REJECT"
    path = tmp_path / "review.txt"
    path.write_text(body)

    shown = await capture.preview(path, max_chars=400)

    assert "VERDICT: REJECT" in shown
    assert "elided" in shown
    assert len(shown) < len(body)


@pytest.mark.asyncio
async def test_a_short_capture_is_shown_whole(tmp_path: Path):
    path = tmp_path / "review.txt"
    path.write_text("VERDICT: APPROVE")

    assert await capture.preview(path, max_chars=400) == "VERDICT: APPROVE"


@pytest.mark.asyncio
async def test_a_preview_does_not_mark_the_run_incomplete(workspace: Path, queue):
    """A window onto a complete file is not a clipped result, and conflating
    them would make every captured run PARTIAL — which would make the flag
    meaningless exactly where it now has to be trusted."""
    f = await _feature(workspace, queue)

    env = await f.shell(
        command="python3 -c 'print(\"y\" * 200000)'", capture_output=True
    )

    assert env.status is ToolResultStatus.OK
    assert env.data["truncated_stdout"] is False
    assert env.data["complete"] is True
    assert len(env.data["stdout"]) < 200000
    assert Path(env.data["stdout_path"]).stat().st_size > 200000


# ---------------------------------------------------------------------------
# Review round 1. Every finding below was reproduced before it was fixed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_capture_is_owner_only_like_the_manifest_beside_it(tmp_path: Path):
    """Measured under a 022 umask: the streams came out 0644 and their
    directory 0755, next to a 0600 manifest and a 0600 audit log. A durable
    artifact readable by every local account is a worse leak than the
    transient result it replaced, because it persists."""
    import stat

    bundle = capture.allocate(tmp_path / "captures")
    backend = LocalSandboxBackend(GRANTS)
    await backend.exec(
        ["python3", "-c", "print('secret')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )
    await capture.write_manifest(bundle, {"x": 1})

    mode = lambda p: stat.S_IMODE(p.stat().st_mode)
    assert mode(bundle.stdout_path) == 0o600
    assert mode(bundle.stderr_path) == 0o600
    assert mode(bundle.manifest_path) == 0o600
    assert mode(bundle.stdout_path.parent) == 0o700


@pytest.mark.asyncio
async def test_a_preview_does_not_read_the_whole_artifact(tmp_path: Path):
    """The point of writing straight to disk is that output can exceed
    memory; reading it all back to show 4,000 characters puts the ceiling
    back where the capture just removed it, and does it AFTER the subprocess
    succeeded. Measured before the fix: previewing a 50 MB file grew peak RSS
    by 50 MB.

    Asserted by counting bytes read rather than by watching RSS, which is
    noisy: the file is instrumented so any full read is visible."""
    path = tmp_path / "big.txt"
    body = (b"HEAD-MARKER" + b"f" * 8_000_000 + b"TAIL-MARKER")
    path.write_bytes(body)

    read_sizes: list[int] = []
    real_open = open

    def counting_open(p, *a, **k):
        fh = real_open(p, *a, **k)
        if str(p) == str(path):
            real_read = fh.read

            def read(n=-1):
                data = real_read(n)
                read_sizes.append(len(data))
                return data

            fh.read = read
        return fh

    import builtins

    builtins.open = counting_open
    try:
        shown = await capture.preview(path, max_chars=400)
    finally:
        builtins.open = real_open

    assert "HEAD-MARKER" in shown and "TAIL-MARKER" in shown
    assert sum(read_sizes) < 100_000, f"read {sum(read_sizes)} bytes to show 400 chars"


@pytest.mark.asyncio
async def test_a_descendant_still_writing_means_the_capture_is_not_final(
    tmp_path: Path,
):
    """Measured: a forked grandchild appended to the capture 1.5s AFTER
    ``exec`` returned. ``communicate`` has no pipes to drain when the streams
    are files, so it waits only for the direct child, and descendants that
    inherited the descriptors keep writing while the manifest is being
    written calling the run complete."""
    bundle = capture.allocate(tmp_path / "captures")
    script = tmp_path / "fork.py"
    script.write_text(
        "import os, sys, time\n"
        "if os.fork() == 0:\n"
        "    time.sleep(3); print('LATE', flush=True); os._exit(0)\n"
        "print('parent done', flush=True)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", str(script)],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.writers_remaining is True


@pytest.mark.asyncio
async def test_a_plain_command_leaves_no_writers(tmp_path: Path):
    """Control: the flag must mean something. If every run reported survivors
    it would demote every capture to PARTIAL and stop being readable."""
    bundle = capture.allocate(tmp_path / "captures")
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", "-c", "print('done')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.writers_remaining is False


@pytest.mark.asyncio
async def test_surviving_writers_make_the_run_incomplete(workspace: Path, queue):
    """The flag has to reach the verdict, not just the backend — the same
    gap that left ``truncated_stdout`` computed and unforwarded."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(writers_remaining=True))

    env = await f.shell(command="echo hi", capture_output=True)

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["writers_remaining"] is True
    assert env.data["complete"] is False
    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["complete"] is False
    assert body["writers_remaining"] is True


@pytest.mark.asyncio
async def test_a_timeout_kills_the_group_not_just_the_leader(tmp_path: Path):
    """A timed-out command used to leave its children running while the tool
    reported the wait as over — the detachment half of the ticket."""
    bundle = capture.allocate(tmp_path / "captures")
    marker = tmp_path / "still_alive.txt"
    script = tmp_path / "spawn.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    time.sleep(4)\n"
        f"    open({str(marker)!r}, 'w').write('descendant survived')\n"
        "    os._exit(0)\n"
        "time.sleep(30)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", str(script)],
        cwd=None,
        env=None,
        timeout=1,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )
    assert result.timed_out is True

    import asyncio as _a

    await _a.sleep(5)
    assert not marker.exists(), "a descendant outlived the timeout that killed its leader"


@pytest.mark.asyncio
async def test_a_capture_without_a_cwd_still_records_where_it_ran(
    workspace: Path, queue
):
    """A run that named no directory still ran somewhere. The manifest said
    ``cwd: null`` and recorded no revision, so exactly the captures that
    most need provenance — the ones taken casually — could not say which
    checkout they were about."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", capture_output=True)

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["cwd"] is not None
    assert body["cwd"] == str(Path.cwd())
    assert env.data["cwd"] == str(Path.cwd())


@pytest.mark.asyncio
async def test_a_docker_timeout_still_produces_the_files_it_promised():
    """The manifest named bundle paths that were never created, so the
    previews read ``[capture unreadable]`` — a missing artifact reported as
    a broken one."""
    from kestrel_sovereign.features.compute.executors.base import ExecutionTimeoutError
    from kestrel_sovereign.features.computer_use.backends.docker import (
        DockerSandboxBackend,
    )
    import tempfile

    backend = DockerSandboxBackend.__new__(DockerSandboxBackend)

    class _Executor:
        async def execute_command(self, command, working_dir=None):
            raise ExecutionTimeoutError("cmd", 5)

    backend._executor = _Executor()
    bundle = capture.allocate(Path(tempfile.mkdtemp()) / "captures")

    result = await backend.exec(
        ["echo", "hi"],
        cwd=None,
        env=None,
        timeout=5,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.timed_out is True
    assert bundle.stdout_path.exists()
    assert bundle.stderr_path.exists()
    assert result.stdout_path == str(bundle.stdout_path)
    assert "timeout" in bundle.stderr_path.read_text()


# ---------------------------------------------------------------------------
# Review round 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_setsid_descendant_is_still_detected(tmp_path: Path):
    """The hole in the first answer. Group membership was the test, and a
    descendant that calls ``setsid`` leaves the group while keeping the
    capture descriptors — so the probe reported "no writers" about a process
    that was still writing. Inheritance is the property that matters, and the
    sentinel pipe tests it directly."""
    bundle = capture.allocate(tmp_path / "captures")
    script = tmp_path / "daemon.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()            # leave the group the probe watched\n"
        "    time.sleep(3)\n"
        "    print('LATE', flush=True)\n"
        "    os._exit(0)\n"
        "print('parent done', flush=True)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", str(script)],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.writers_remaining is True


@pytest.mark.asyncio
async def test_could_not_check_is_not_no_writers(workspace: Path, queue):
    """Three states, because two would force a guess. ``None`` is what a
    platform without ``pass_fds`` can honestly report, and it must not be
    read as a cleared check — an unverifiable capture is not a clean one."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(writers_remaining=None))

    env = await f.shell(command="echo hi", capture_output=True)

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["complete"] is False
    assert "cannot be detected" in (env.error or "")
    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["complete"] is False


@pytest.mark.asyncio
async def test_an_uncaptured_run_is_not_demoted_by_the_writer_check(
    workspace: Path, queue
):
    """Control. There is no sentinel without a capture, so the answer is
    ``None`` there too — and there is also no file whose finality matters.
    Folding the two together would demote every ordinary shell call."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(writers_remaining=None))

    env = await f.shell(command="echo hi")

    assert env.status is ToolResultStatus.OK
    assert env.data["complete"] is True


@pytest.mark.asyncio
async def test_a_capture_allocation_failure_is_audited(workspace: Path, queue):
    """It happened before the audited boundary, so a call that had passed
    every gate could fail with no computer-use audit row — the trail losing
    exactly the runs that went wrong."""
    f = await _feature(workspace, queue)
    # A capture dir that cannot be created: a regular file sits in its path.
    blocker = workspace / "blocked"
    blocker.write_text("not a directory")
    f._capture_dir = blocker / "captures"

    env = await f.shell(command="echo hi", capture_output=True)

    assert env.status is not ToolResultStatus.OK
    audit = (workspace / "audit.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in audit]
    assert any(r["tool"] == "shell" and r["outcome"] == "error" for r in rows), rows


@pytest.mark.asyncio
async def test_a_cwdless_docker_run_records_no_host_head(workspace: Path, queue):
    """With the docker backend and no cwd the command runs at ``/`` inside a
    container, where this process's repository is not the tree under review.
    Recording the host SHA would put an unrelated revision in the provenance
    the manifest exists to make trustworthy — worse than recording none,
    which is this module's stated rule."""
    f = await _feature(workspace, queue)
    stub = _StubBackend(_run())
    stub.name = "docker"
    f._backend = stub

    env = await f.shell(command="echo hi", capture_output=True)

    body = json.loads(Path(env.data["manifest_path"]).read_text())
    assert body["git"]["head_before"] is None
    assert body["git"]["head_after"] is None


@pytest.mark.asyncio
async def test_a_local_cwdless_run_still_records_its_head(workspace: Path, queue):
    """Control for the pair: the local backend really does run in this
    process's directory, so suppressing the SHA there would lose provenance
    that is genuinely available."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", capture_output=True)
    body = json.loads(Path(env.data["manifest_path"]).read_text())

    expected = await capture.git_head(Path.cwd())
    assert body["git"]["head_before"] == expected


@pytest.mark.asyncio
async def test_the_preview_bound_is_a_character_bound(tmp_path: Path):
    """The byte window is four times the character budget, so a file small
    in bytes could still return four times the advertised characters — a
    15,000-character ASCII capture came back whole under a 4,000 bound, and
    the bound exists to keep the model's context bounded, not merely to
    avoid a large read."""
    path = tmp_path / "review.txt"
    path.write_text("A" * 15_000)

    shown = await capture.preview(path, max_chars=4000)

    assert len(shown) < 4600  # the budget plus the elision notice
    assert "elided" in shown


@pytest.mark.asyncio
async def test_a_capture_under_both_bounds_is_still_whole(tmp_path: Path):
    """Control: the character bound must not start eliding short captures."""
    path = tmp_path / "review.txt"
    path.write_text("VERDICT: APPROVE")

    assert await capture.preview(path, max_chars=4000) == "VERDICT: APPROVE"


def test_the_timeout_path_does_not_reach_a_posix_only_call_on_windows():
    """``os.killpg`` does not exist on Windows and the guard around it did
    not catch ``AttributeError``, so on a platform this package declares
    support for, the probe raised out of every successful local command."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    src = Path(local_mod.__file__).read_text()
    body = src[src.index("def _kill_tree") : src.index("def _writers_remain")]
    assert "is_windows()" in body
    assert "taskkill" in body
    # And the spawn no longer hardcodes the POSIX-only spelling.
    assert "start_new_session=True" not in src
    assert "new_process_group_kwargs()" in src
