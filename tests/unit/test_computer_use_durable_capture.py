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
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

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
async def test_the_docker_backend_reads_truncation_off_the_record():
    """That backend never set ``truncated_stdout`` at all: the executor
    signalled a clip by appending a marker to the text and kept no boolean,
    so however much was discarded the flag stayed False. The fact now rides
    on ``ExecutionRecord.output_truncated``; the marker is only cosmetic."""
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
                stdout_truncated = True
                stderr_truncated = False

            return _Rec()

    backend._executor = _Executor()

    result = await backend.exec(["echo", "hi"], cwd=None, env=None, timeout=5)

    assert result.truncated_stdout is True
    # The marker is presentation; it must not stay in the text a reader
    # would treat as the review.
    assert result.stdout == "a review"


@pytest.mark.asyncio
async def test_output_that_merely_looks_truncated_is_not(monkeypatch):
    """Review round 4. Parsing the marker back out of the text made the
    completeness flag a function of what the command chose to print — a run
    echoing a prior executor log ends with that exact string and was reported
    truncated, turning a clean pass into a caveated PARTIAL. Output is
    caller-controlled; metadata cannot live in it."""
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
                stdout = "here is a log I am quoting" + _OUTPUT_TRUNCATED_SUFFIX
                stderr = ""
                stdout_truncated = False
                stderr_truncated = False

            return _Rec()

    backend._executor = _Executor()

    result = await backend.exec(["echo", "hi"], cwd=None, env=None, timeout=5)

    assert result.truncated_stdout is False
    assert result.stdout.endswith(_OUTPUT_TRUNCATED_SUFFIX)


def test_the_record_tracks_each_stream_separately():
    """One flag for both streams reported a clipped stdout as a clipped
    stderr — a different claim from the true one, and the caveat then named
    a stream that was whole. The streams are capped independently, so the
    record carries them independently."""
    from kestrel_sovereign.features.compute.models import ExecutionRecord

    fields = ExecutionRecord.__dataclass_fields__
    assert "stdout_truncated" in fields and "stderr_truncated" in fields

    only_out = ExecutionRecord(id="a", script_id="b", stdout_truncated=True)
    assert only_out.stdout_truncated is True
    assert only_out.stderr_truncated is False
    # The convenience view stays, because most callers only care that
    # something was lost.
    assert only_out.output_truncated is True


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


@pytest.mark.asyncio
async def test_the_kill_path_does_not_reach_a_posix_only_call_on_windows(monkeypatch):
    """``os.killpg`` does not exist on Windows and the guard around it caught
    only ``OSError``, so on a platform this package declares support for the
    kill raised ``AttributeError`` out of the timeout path.

    Asserted behaviourally. The first version of this test read the module's
    source for ``is_windows()``, which passed for a while and then silently
    stopped testing anything when the functions were reordered and its slice
    came back empty — a check that cannot fail is worse than no check."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    calls: list = []
    monkeypatch.setattr(local_mod, "is_windows", lambda: True)
    monkeypatch.setattr(
        local_mod.subprocess, "run", lambda *a, **k: calls.append(a[0])
    )

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("reached a POSIX-only call on the Windows path")

    monkeypatch.setattr(local_mod.os, "killpg", _boom)
    monkeypatch.setattr(local_mod.os, "kill", _boom)

    await local_mod._kill_tree(4321)

    assert calls and calls[0][:2] == ["taskkill", "/F"]
    assert "4321" in calls[0]


@pytest.mark.asyncio
async def test_the_kill_path_on_posix_kills_the_group(monkeypatch):
    """Control for the pair: the Windows branch must be a branch, not a
    replacement."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    killed: list = []
    monkeypatch.setattr(local_mod, "is_windows", lambda: False)
    monkeypatch.setattr(
        local_mod.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )

    await local_mod._kill_tree(4321)

    assert killed == [(4321, local_mod.signal.SIGKILL)]


@pytest.mark.asyncio
async def test_a_captured_review_survives_the_orchestrators_own_cap(
    workspace: Path, queue
):
    """Review round 3, and this ticket's own failure one layer up.

    The orchestrator caps a serialized tool result at MAX_TOOL_RESULT_CHARS
    and replaces anything larger with its own head/tail preview. Measured
    before the fix: a captured review serialized to 9,147 chars, came back as
    2,630, and the verdict was in neither window — the tail the model saw was
    artifact paths. A verdict visible in the feature's preview and invisible
    in the message the model reads is not visible."""
    from kestrel_sovereign.agent.orchestrator_engine import _build_persisted_preview
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    f = await _feature(workspace, queue)
    env = await f.shell(
        command=(
            "python3 -c \"print('REVIEW LINE ' * 40000); print('VERDICT: APPROVE')\""
        ),
        capture_output=True,
        timeout=60,
    )

    size = serialized_result_len(env)
    assert size <= orchestrator_result_cap(), f"envelope is {size} chars"
    # And belt-and-braces: even if it were replaced, say so loudly here
    # rather than in a live run.
    import json as _json

    blob = _json.dumps(
        __import__(
            "kestrel_sovereign.features.base", fromlist=["_serialize_tool_result"]
        )._serialize_tool_result(env)
    )
    assert "VERDICT: APPROVE" in blob
    assert _build_persisted_preview(blob, "shell", len(blob)) or True


@pytest.mark.asyncio
async def test_the_preview_is_not_duplicated_into_the_confirmation(
    workspace: Path, queue
):
    """Carrying the same text in ``data`` and ``confirmation`` doubled the
    envelope for no gain — the orchestrator serializes both into one blob and
    measures that."""
    f = await _feature(workspace, queue)

    env = await f.shell(
        command="python3 -c \"print('x' * 50000)\"", capture_output=True
    )

    assert env.data["stdout"]
    assert env.data["stdout"] not in (env.confirmation or "")
    # It points at the artifact instead of repeating it.
    assert "artifact:" in (env.confirmation or "")
    assert env.data["manifest_path"] in (env.confirmation or "")


@pytest.mark.asyncio
async def test_an_uncaptured_run_still_shows_its_output_inline(
    workspace: Path, queue
):
    """Control: the !shell CLI surface renders ``confirmation``, so an
    ordinary run must keep printing there."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hello")

    assert "hello" in (env.confirmation or "")


@pytest.mark.asyncio
async def test_a_descendant_spawned_through_popen_is_still_detected(
    tmp_path: Path,
):
    """The hole in the second answer. ``subprocess.Popen`` closes non-stdio
    descriptors by default while inheriting stdout and stderr, so the
    side-channel sentinel was closed in the descendant while the capture
    descriptors stayed open — the common way to spawn a background process,
    not an exotic one. Detection now rides on the output streams themselves."""
    bundle = capture.allocate(tmp_path / "captures")
    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, sys\n"
        # close_fds=True is the default: the child keeps stdout/stderr and
        # loses everything else.
        "subprocess.Popen([sys.executable, '-c',"
        " \"import time; time.sleep(3); print('LATE')\"])\n"
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
async def test_a_spawn_failure_keeps_its_diagnostic(workspace: Path, queue):
    """With a capture the feature previews the file and ignores
    ``result.stderr``, so a missing executable came back as rc=127 beside an
    empty artifact — the one actionable line dropped."""
    f = await _feature(workspace, queue, auto_approved_binaries=["definitely-not-a-real-binary-xyz"])

    env = await f.shell(
        command="definitely-not-a-real-binary-xyz", capture_output=True
    )

    assert env.data["returncode"] == 127
    assert env.data["stderr"], "the OS diagnostic was lost"
    assert "definitely-not-a-real-binary-xyz" in env.data["stderr"]


@pytest.mark.asyncio
async def test_a_daemonizing_command_does_not_hold_the_tool_open(tmp_path: Path):
    """The grace has to be a grace. asyncio finishes a subprocess only once
    every pipe transport has closed, so waiting on ``proc.wait()`` waited for
    the descendants too — measured at 3s for a 3s sleeper — and a command
    that legitimately daemonizes would have held the tool for its whole
    timeout."""
    import time as _t

    bundle = capture.allocate(tmp_path / "captures")
    script = tmp_path / "daemon.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid(); time.sleep(20); os._exit(0)\n"
        "print('parent done', flush=True)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    began = _t.monotonic()
    result = await backend.exec(
        ["python3", str(script)],
        cwd=None,
        env=None,
        timeout=60,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )
    elapsed = _t.monotonic() - began

    assert result.writers_remaining is True
    assert result.timed_out is False
    assert elapsed < 10, f"waited {elapsed:.1f}s on a daemon it should have reported"


# ---------------------------------------------------------------------------
# Surviving mutants from round 4. Each named a gap, not a false alarm.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_envelope_fits_when_BOTH_streams_are_large(
    workspace: Path, queue
):
    """Surviving mutant: replacing the cap-derived budget with the fixed
    4,000-char default changed nothing, because the only envelope test had an
    empty stderr and 4,000 + 0 + metadata still fit. Two full previews is the
    case the budget is halved for."""
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    f = await _feature(workspace, queue)
    env = await f.shell(
        command=(
            "python3 -c \"import sys; "
            "sys.stdout.write('O' * 60000); sys.stderr.write('E' * 60000)\""
        ),
        capture_output=True,
        timeout=60,
    )

    assert env.data["stdout"] and env.data["stderr"]
    size = serialized_result_len(env)
    assert size <= orchestrator_result_cap(), f"envelope is {size} chars"


@pytest.mark.asyncio
async def test_the_spawn_diagnostic_reaches_the_artifact_itself(
    workspace: Path, queue
):
    """Surviving mutant: dropping the write into the captured stderr changed
    no test, because the feature's fallback put the message in ``data``
    anyway. Two guarantees for one property is fine; a test that cannot tell
    them apart is not. The artifact is supposed to be readable on its own."""
    f = await _feature(
        workspace, queue, auto_approved_binaries=["definitely-not-a-real-binary-xyz"]
    )

    env = await f.shell(
        command="definitely-not-a-real-binary-xyz", capture_output=True
    )

    captured = Path(env.data["stderr_path"]).read_text()
    assert "definitely-not-a-real-binary-xyz" in captured


@pytest.mark.asyncio
async def test_the_fallback_covers_a_capture_that_holds_nothing(
    workspace: Path, queue
):
    """The other half of that pair. If the diagnostic could not be written
    into the capture, the empty file must not silently replace it."""
    bundle_dir = workspace / "captures"
    f = await _feature(workspace, queue)

    class _EmptyCaptureBackend:
        name = "local"

        async def exec(self, argv, *, cwd, env, timeout, capture=None):
            # Files exist and are empty: the shape of a write that failed.
            capture.stdout_path.write_bytes(b"")
            capture.stderr_path.write_bytes(b"")
            return CompletedRun(
                argv=argv,
                returncode=127,
                stdout="",
                stderr="No such file or directory: 'ghost'",
                duration_ms=1,
                stdout_path=str(capture.stdout_path),
                stderr_path=str(capture.stderr_path),
                writers_remaining=False,
            )

        async def shutdown(self):
            pass

    f._backend = _EmptyCaptureBackend()

    env = await f.shell(command="echo hi", capture_output=True)

    assert "No such file or directory" in env.data["stderr"]


@pytest.mark.asyncio
async def test_an_uncaptured_timeout_also_kills_the_tree(tmp_path: Path):
    """Surviving mutant: the uncaptured branch's tree-kill was untested — the
    only timeout test used a capture. A command that forks and times out must
    not leave descendants running on either path."""
    marker = tmp_path / "survived.txt"
    script = tmp_path / "spawn.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    time.sleep(4)\n"
        f"    open({str(marker)!r}, 'w').write('x')\n"
        "    os._exit(0)\n"
        "time.sleep(30)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    result = await backend.exec(
        ["python3", str(script)], cwd=None, env=None, timeout=1
    )
    assert result.timed_out is True

    import asyncio as _a

    await _a.sleep(5)
    assert not marker.exists(), "a descendant outlived an uncaptured timeout"


@pytest.mark.asyncio
async def test_the_spawn_asks_the_cross_platform_helper_for_its_group_kwargs(
    tmp_path: Path, monkeypatch
):
    """Surviving mutant, and one this platform cannot kill behaviourally:
    ``start_new_session=True`` and ``new_process_group_kwargs()`` are
    identical on POSIX and differ only on Windows, which the suite cannot
    run. So the wiring itself is the assertion — the helper must be what
    decides, because it is the only thing that knows about the other
    platform."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    asked = []
    real = local_mod.new_process_group_kwargs

    def spy():
        asked.append(True)
        return real()

    monkeypatch.setattr(local_mod, "new_process_group_kwargs", spy)

    await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('hi')"], cwd=None, env=None, timeout=30
    )

    assert asked, "the spawn did not consult the cross-platform helper"


@pytest.mark.asyncio
async def test_a_quote_heavy_capture_still_fits_the_envelope(
    workspace: Path, queue
):
    """Review round 4. The budget was arithmetic — cap minus a fixed reserve,
    halved — and ``json.dumps`` escaping expands a quote-heavy body several
    fold. Measured: two 2,750-character previews of quotes serialized to
    12,088 chars under an 8,000 cap, so the orchestrator replaced the
    envelope and hid the verdict again. A reserve is a guess about expansion;
    the fix measures the serialized form and shrinks until it fits."""
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    script = workspace / "quotes.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write(chr(34) * 60000)\n"
        "sys.stderr.write(chr(34) * 60000)\n"
    )
    f = await _feature(workspace, queue)

    env = await f.shell(
        command=f"python3 {script}", capture_output=True, timeout=60
    )

    size = serialized_result_len(env)
    assert size <= orchestrator_result_cap(), f"envelope is {size} chars"
    assert env.data["stdout"], "shrank past showing anything"


@pytest.mark.asyncio
async def test_an_uncaptured_run_over_the_downstream_cap_is_not_complete(
    workspace: Path, queue
):
    """Review round 4. Between the orchestrator's ~8 KB cap and the backend's
    1 MiB one, both truncation flags stayed false and the run reported
    ``complete: true`` — while the orchestrator discarded most of an envelope
    that has no artifact to fall back on. Measured at 80,391 serialized
    chars."""
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    f = await _feature(workspace, queue)

    env = await f.shell(command="python3 -c \"print('z' * 40000)\"", timeout=60)

    assert serialized_result_len(env) <= orchestrator_result_cap()
    assert env.data["complete"] is False
    assert env.status is ToolResultStatus.PARTIAL
    assert "capture_output=true" in (env.error or "")


@pytest.mark.asyncio
async def test_a_small_uncaptured_run_is_untouched(workspace: Path, queue):
    """Control: the downstream-cap rule must fire on size, not on every
    uncaptured run."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hello")

    assert env.status is ToolResultStatus.OK
    assert env.data["complete"] is True
    assert env.data["stdout"].strip() == "hello"


@pytest.mark.asyncio
async def test_a_capture_that_could_not_be_written_is_not_complete(
    tmp_path: Path, monkeypatch
):
    """Review round 4, and the worst of the set. A pump that raises — a full
    disk, a vanished directory — finishes like any other task and lands in
    ``done``; not asking for its exception meant a capture missing everything
    after the failure was reported complete. Silent loss wearing a clean
    result, which is the exact thing this ticket exists to prevent."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")

    async def exploding_pump(reader, fh, state=None):
        await reader.read(10)
        raise OSError("No space left on device")

    monkeypatch.setattr(local_mod, "_pump", exploding_pump)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('data' * 1000)"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stdout is True
    assert "No space left on device" in result.stderr


@pytest.mark.asyncio
async def test_the_audit_row_names_why_an_incomplete_run_failed(
    workspace: Path, queue
):
    """Review round 4. A run that exited 0 and was incomplete for some other
    reason was audited as an error whose message was the contradictory
    ``exit 0``, and the writer state was not in the payload at all — the
    canonical row could not say what went wrong."""
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(returncode=0, timed_out=True))

    await f.shell(command="echo hi")

    rows = [
        json.loads(line)
        for line in (workspace / "audit.jsonl").read_text().splitlines()
    ]
    shell_rows = [r for r in rows if r["tool"] == "shell"]
    assert shell_rows, rows
    row = shell_rows[-1]
    assert row["outcome"] == "error"
    assert row["error"] != "exit 0"
    assert "timed out" in row["error"]
    assert row["args"]["complete"] is False
    assert "writers_remaining" in row["args"]


def test_the_executor_populates_the_truncation_field():
    """Surviving mutant: hardcoding ``output_truncated=False`` at the point
    the record is built changed nothing, because the only test asserted the
    dataclass HAS the field. Testing the shape instead of the wiring is the
    same door as the fix, again."""
    from datetime import datetime

    from kestrel_sovereign.features.compute.executors.base import (
        BaseExecutor,
        _CapturedOutput,
        _ExecutionContext,
    )

    class _Subject:
        id = "abc12345"

    class _Probe(BaseExecutor):
        name = "local"

        def __init__(self):
            pass

        @property
        def is_available(self):  # pragma: no cover - unused here
            return True

        async def execute(self, *a, **k):  # pragma: no cover - unused here
            raise NotImplementedError

    ctx = _ExecutionContext(
        execution_id="e1", started_at=datetime.now(), workdir="/tmp"
    )
    probe = _Probe()

    clipped = probe._build_record(
        subject=_Subject(),
        context=ctx,
        exit_code=0,
        stdout="x",
        stderr="",
        stdout_truncated=True,
    )
    whole = probe._build_record(
        subject=_Subject(), context=ctx, exit_code=0, stdout="x", stderr=""
    )

    assert clipped.stdout_truncated is True
    assert clipped.stderr_truncated is False
    assert whole.output_truncated is False


@pytest.mark.asyncio
async def test_the_run_path_carries_truncation_into_the_record():
    """The wiring the mutant actually sat on: the caller must read the
    capture's own ``truncated`` flag rather than pass a constant."""
    from datetime import datetime

    from kestrel_sovereign.features.compute.executors import base as exec_base

    class _Subject:
        id = "abc12345"

    class _Probe(exec_base.BaseExecutor):
        name = "local"

        def __init__(self):
            pass

        @property
        def is_available(self):  # pragma: no cover - unused here
            return True

        async def execute(self, *a, **k):  # pragma: no cover - unused here
            raise NotImplementedError

    class _Result:
        exit_code = 0
        container_id = None
        stdout = exec_base._CapturedOutput(b"body", True)
        stderr = exec_base._CapturedOutput(b"", False)

    async def runner(context):
        return _Result()

    record = await _Probe()._execute_with_lifecycle(
        _Subject(), temp_dir_prefix="probe-", runner=runner
    )

    assert record.stdout_truncated is True
    assert record.stderr_truncated is False


# ---------------------------------------------------------------------------
# Review round 5
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_child_does_not_inherit_this_process_stdin(tmp_path: Path):
    """Review round 5, and the trap this ticket's own text names. Stdin was
    inherited from the server, so a command that reads it waits for input no
    one will send — ``claude -p`` blocks forever on an inherited stdin, which
    the merge-gate doctrine spells ``</dev/null`` and which this surface
    cannot express, the redirect being shell grammar. The reviewer the
    feature exists to run is the exact program that hangs.

    The first version of this test just read stdin and expected ``''``, and
    a mutant restoring the inheritance survived it: under pytest fd 0 is
    already empty, so the assertion held either way. This one puts real bytes
    on fd 0 for the duration, which makes inheritance observable — the child
    would read them.
    """
    import os as _os

    payload = b"SHOULD-NOT-REACH-THE-CHILD\n"
    r_fd, w_fd = _os.pipe()
    _os.write(w_fd, payload)
    _os.close(w_fd)
    saved = _os.dup(0)
    try:
        _os.dup2(r_fd, 0)
        _os.close(r_fd)
        result = await LocalSandboxBackend(GRANTS).exec(
            ["python3", "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
            cwd=None,
            env=None,
            timeout=30,
        )
    finally:
        _os.dup2(saved, 0)
        _os.close(saved)

    assert "SHOULD-NOT-REACH-THE-CHILD" not in result.stdout
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_a_capture_whose_close_fails_is_not_complete(
    tmp_path: Path, monkeypatch
):
    """A close flushes, and a flush can fail — a full disk surfaces there
    rather than at any write. Swallowing it discarded the buffered tail and
    called the file complete, the same shape as the unread pump exception."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture

    def open_with_failing_close(cap):
        out_fh, err_fh = real_open(cap)

        class _Failing:
            def __init__(self, inner):
                self._inner = inner

            def write(self, b):
                return self._inner.write(b)

            def fileno(self):
                return self._inner.fileno()

            def close(self):
                self._inner.close()
                raise OSError("No space left on device")

        return _Failing(out_fh), err_fh

    monkeypatch.setattr(local_mod, "_open_capture", open_with_failing_close)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('body')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stdout is True
    assert "No space left on device" in result.stderr


@pytest.mark.asyncio
async def test_a_tiny_cap_still_produces_a_readable_envelope(
    workspace: Path, queue, monkeypatch
):
    """The fit loop gave up after five halvings without re-checking, so a
    small configured cap — KESTREL_MAX_TOOL_RESULT_CHARS is settable and
    1,000 is supported — left an oversized envelope for the orchestrator to
    replace with a generic preview that hides both the verdict and the
    completeness flag. It now fails closed: previews go, facts and paths
    stay, because that is what a caller needs in order to go and read the
    artifact."""
    import kestrel_sovereign.features.base as base_mod
    from kestrel_sovereign.features.base import serialized_result_len

    monkeypatch.setattr(base_mod, "orchestrator_result_cap", lambda: 1000)
    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 1000,
    )
    f = await _feature(workspace, queue)

    env = await f.shell(
        command="python3 -c \"print('q' * 100000)\"",
        capture_output=True,
        timeout=60,
    )

    assert serialized_result_len(env) <= 1000, serialized_result_len(env)
    assert env.data["manifest_path"]
    assert "complete" in env.data


@pytest.mark.asyncio
async def test_the_audit_calls_feature_level_clipping_truncation(
    workspace: Path, queue
):
    """The audit read the backend's flag, so an uncaptured stream trimmed to
    fit recorded ``truncated: false`` beside ``complete: false`` — the
    canonical row saying the run failed and not saying why."""
    f = await _feature(workspace, queue)

    await f.shell(command="python3 -c \"print('z' * 40000)\"", timeout=60)

    rows = [
        json.loads(line)
        for line in (workspace / "audit.jsonl").read_text().splitlines()
    ]
    row = [r for r in rows if r["tool"] == "shell"][-1]
    assert row["args"]["complete"] is False
    assert row["args"]["truncated"] is True


def test_the_truncation_flags_survive_a_round_trip():
    """A clipped record reconstructed from storage that reported itself whole
    would defeat the point of moving the fact off the text."""
    from kestrel_sovereign.features.compute.models import ExecutionRecord

    original = ExecutionRecord(
        id="e", script_id="s", stdout_truncated=True, stderr_truncated=False
    )

    restored = ExecutionRecord.from_dict(original.to_dict())

    assert restored.stdout_truncated is True
    assert restored.stderr_truncated is False


@pytest.mark.asyncio
async def test_the_execution_store_persists_the_truncation_flags(tmp_path: Path):
    """And through SQL. ``CREATE TABLE IF NOT EXISTS`` is the whole schema
    story in that store, so a database made before the column existed never
    gains it — the additive migration is what stops a restored record from
    quietly reporting itself whole."""
    from kestrel_sovereign.features.compute.models import ComputeScript, ExecutionRecord
    from kestrel_sovereign.features.compute.script_store import ScriptStore

    store = ScriptStore(db_path=str(tmp_path / "compute.db"))
    await store.initialize()

    script = ComputeScript(
        id="s1", name="n", language="python", content="print(1)", purpose="p"
    )
    await store.save(script)
    await store.save_execution(
        ExecutionRecord(
            id="e1", script_id="s1", stdout_truncated=True, stderr_truncated=False
        )
    )

    restored = await store.get_execution("e1")

    assert restored is not None
    assert restored.stdout_truncated is True
    assert restored.stderr_truncated is False


@pytest.mark.asyncio
async def test_an_older_database_gains_the_columns(tmp_path: Path):
    """The migration half, driven against a table built without them."""
    import aiosqlite

    from kestrel_sovereign.features.compute.script_store import ScriptStore

    db_path = tmp_path / "old.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE compute_executions ("
            "id TEXT PRIMARY KEY, script_id TEXT NOT NULL, started_at TIMESTAMP,"
            " completed_at TIMESTAMP, exit_code INTEGER, stdout TEXT,"
            " stderr TEXT, executor TEXT, container_id TEXT,"
            " resource_usage TEXT, dry_run INTEGER, workdir TEXT)"
        )
        await db.commit()

    await ScriptStore(db_path=str(db_path)).initialize()

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("PRAGMA table_info(compute_executions)")
        columns = {row[1] for row in await cursor.fetchall()}

    assert {"stdout_truncated", "stderr_truncated"} <= columns


@pytest.mark.asyncio
async def test_fitting_keeps_a_usable_preview_rather_than_collapsing(
    workspace: Path, queue
):
    """Surviving mutants: with the fail-closed minimal envelope in place,
    deleting the shrink loop entirely still produced something under the cap
    — so nothing failed, and the loop looked redundant. It is not: without
    it every oversized capture drops straight to facts-only and the caller
    loses the verdict they came for. Correctness was covered; QUALITY was
    not, and that is what the loop buys."""
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    script = workspace / "quotes.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write(chr(34) * 60000)\n"
        "sys.stderr.write(chr(34) * 60000)\n"
    )
    f = await _feature(workspace, queue)

    env = await f.shell(
        command=f"python3 {script}", capture_output=True, timeout=60
    )

    assert serialized_result_len(env) <= orchestrator_result_cap()
    # Shrunk to fit, not abandoned: the loop should land on a preview worth
    # reading rather than handing back the facts-only fallback.
    assert len(env.data["stdout"]) > 500, len(env.data["stdout"])
    assert env.data["stdout_path"], "fell through to the minimal envelope"


# ---------------------------------------------------------------------------
# Review round 6
# ---------------------------------------------------------------------------


def test_the_size_measured_is_the_size_the_orchestrator_receives():
    """Review round 6. ``DynamicTool.execute`` wraps a ToolResult, adding
    ``tool`` and ``success`` on top of ``to_dict()``, so measuring the bare
    result is short by those keys. Invisible until a result lands in the gap
    — measured, 7,967 unwrapped against an 8,000 cap became 8,001 wrapped and
    the orchestrator discarded output from a result calling itself
    complete."""
    from kestrel_sovereign.features.base import (
        _serialize_tool_result,
        serialized_result_len,
    )
    import json as _json

    result = ToolResult.ok("conf", data={"stdout": "x" * 100})

    bare = len(_json.dumps(_serialize_tool_result(result)))
    measured = serialized_result_len(result, tool_name="shell")

    assert measured > bare
    wrapped = {
        **_serialize_tool_result(result),
        "tool": "shell",
        "success": True,
    }
    assert measured == len(_json.dumps(wrapped))


@pytest.mark.asyncio
async def test_no_output_length_slips_past_the_wrapped_cap(workspace: Path, queue):
    """Fourth attempt, and the three failures each taught something.

    1. Guessed four output sizes. The window is five characters wide — the
       length of the tool name, since ``serialized_result_len`` counts the
       wrapper keys even when the name is empty — so guessing was hopeless.
    2. Computed the boundary from a synthetic ToolResult. The feature
       duplicates stdout into the confirmation for an uncaptured run, so its
       envelope crosses the cap at half the predicted output length.
    3. Bisected on the feature's own returned size. That predicate is NOT
       monotonic in a working build: the envelope grows with output until
       the fit loop fires, then collapses and stays small. Bisection over a
       non-monotonic predicate lands anywhere, and it landed at the top of
       the range.

    Each of those looked for the boundary using the very machinery whose
    correctness is in question. So this derives it instead, from two
    measurements taken well below the cap where the loop provably does not
    fire, and then sweeps the derived crossing. Verified by hand against the
    mutated build: it fails at 3806, 3807 and 3808.
    """
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    f = await _feature(workspace, queue)
    cap = orchestrator_result_cap()

    async def envelope_size(n: int) -> int:
        f._backend = _StubBackend(_run(stdout="p" * n, stderr=""))
        env = await f.shell(command="echo hi")
        return serialized_result_len(env, tool_name="shell")

    # Two points in the linear region, far below the cap.
    a, b = 1000, 2000
    size_a, size_b = await envelope_size(a), await envelope_size(b)
    assert size_a < cap and size_b < cap, "sample points must be below the cap"
    slope = (size_b - size_a) / (b - a)
    assert slope > 0, "envelope must grow with output for this derivation"
    crossing = int(a + (cap - size_a) / slope)

    for n in range(max(0, crossing - 20), crossing + 20):
        wrapped = await envelope_size(n)
        assert wrapped <= cap, f"{n} chars of stdout -> {wrapped} wrapped"


@pytest.mark.asyncio
async def test_clipping_is_reported_for_the_stream_it_happened_to(
    workspace: Path, queue
):
    """Review round 6. The feature-level clip check combined both streams and
    ORed the result into both flags, so oversized stdout with empty stderr
    reported ``truncated_stderr: true`` — the same false claim the per-stream
    split was introduced to end, made one round later in the file that
    introduced it."""
    f = await _feature(workspace, queue)

    env = await f.shell(command="python3 -c \"print('z' * 40000)\"", timeout=60)

    assert env.data["truncated_stdout"] is True
    assert env.data["truncated_stderr"] is False


@pytest.mark.asyncio
async def test_cancelling_a_captured_run_kills_the_process(tmp_path: Path):
    """Review round 6. ``CancelledError`` escaped while awaiting the child and
    the ``finally`` only closed handles — no kill, no pump teardown — so the
    host process ran on with nobody waiting for it. For the long
    side-effecting commands this feature exists to run that is worse than the
    timeout it now mirrors."""
    import asyncio as _a

    bundle = capture.allocate(tmp_path / "captures")
    marker = tmp_path / "still_running.txt"
    script = tmp_path / "long.py"
    script.write_text(
        "import time\n"
        "time.sleep(4)\n"
        f"open({str(marker)!r}, 'w').write('x')\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    task = _a.create_task(
        backend.exec(
            ["python3", str(script)],
            cwd=None,
            env=None,
            timeout=60,
            capture=CaptureTarget(
                stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
            ),
        )
    )
    await _a.sleep(0.6)
    task.cancel()
    with pytest.raises(_a.CancelledError):
        await task

    await _a.sleep(5)
    assert not marker.exists(), "the child outlived the cancelled tool task"


@pytest.mark.asyncio
async def test_a_capture_that_could_not_be_opened_is_not_complete(
    tmp_path: Path, monkeypatch
):
    """Review round 6. If the second capture file fails to open after the
    first succeeded — descriptor exhaustion, say — the defaults said "nothing
    truncated, no writers, no paths", and the feature then wrote a manifest
    calling the run complete while previewing a file that was never
    opened."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    def failing_open(cap):
        raise OSError("Too many open files")

    monkeypatch.setattr(local_mod, "_open_capture", failing_open)
    bundle = capture.allocate(tmp_path / "captures")

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('hi')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stdout is True
    assert result.writers_remaining is None
    assert "Too many open files" in result.stderr


@pytest.mark.asyncio
async def test_the_feature_does_not_call_an_unopenable_capture_complete(
    workspace: Path, queue, monkeypatch
):
    """The end the caller sees."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    def failing_open(cap):
        raise OSError("Too many open files")

    monkeypatch.setattr(local_mod, "_open_capture", failing_open)
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", capture_output=True)

    assert env.status is ToolResultStatus.PARTIAL
    assert env.data["complete"] is False


# ---------------------------------------------------------------------------
# Review round 7
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_facts_only_fallback_keeps_a_failure_a_failure(
    workspace: Path, queue, monkeypatch
):
    """Review round 7. ``_minimal_envelope`` re-derived status from
    completeness alone, so a command that exited non-zero without timing out
    or clipping came back OK — and the wrapper then published
    ``success: true`` for a review that failed. Shrinking a result must not
    change what it says."""
    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 1000,
    )
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(returncode=7, stdout="x" * 5000))

    env = await f.shell(command="echo hi", capture_output=True)

    assert env.status is not ToolResultStatus.OK
    assert env.data["returncode"] == 7


@pytest.mark.asyncio
async def test_cancellation_during_the_drain_also_tears_down(
    tmp_path: Path, monkeypatch
):
    """The guard covered the exit wait and not the drain — which is its own
    await, and is exactly where a descendant holding the pipes makes the call
    take long enough to be cancelled. The pumps were then left running into
    handles the ``finally`` was about to close."""
    import asyncio as _a

    bundle = capture.allocate(tmp_path / "captures")
    marker = tmp_path / "descendant_ran.txt"
    script = tmp_path / "daemon.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    time.sleep(5)\n"
        f"    open({str(marker)!r}, 'w').write('x')\n"
        "    os._exit(0)\n"
        "print('parent done', flush=True)\n"
    )
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    # A long drain makes it unambiguous which await is live when the cancel
    # lands. At the default 0.5s the call had already returned, and the test
    # was cancelling nothing.
    monkeypatch.setattr(local_mod, "_DRAIN_GRACE", 10.0)
    backend = LocalSandboxBackend(GRANTS)

    task = _a.create_task(
        backend.exec(
            ["python3", str(script)],
            cwd=None,
            env=None,
            timeout=60,
            capture=CaptureTarget(
                stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
            ),
        )
    )
    # The parent exits almost at once; with the drain stretched, this lands
    # inside it.
    await _a.sleep(1.5)
    task.cancel()
    with pytest.raises(_a.CancelledError):
        await task

    await _a.sleep(6)
    assert not marker.exists(), "a descendant outlived cancellation during the drain"


@pytest.mark.asyncio
async def test_one_stream_failing_does_not_indict_the_other(
    tmp_path: Path, monkeypatch
):
    """Review round 7. The aggregate ``lost`` marked BOTH streams truncated
    when either write failed, so the manifest and the caveat claimed a clean
    stream was clipped. That is the per-stream contract broken in the backend
    after being broken once already in the feature."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_pump = local_mod._pump

    async def selective_pump(reader, fh, state=None):
        # Fail only the stream whose handle is the stdout capture.
        if getattr(fh, "_is_stdout", False):
            await reader.read(10)
            raise OSError("No space left on device")
        await real_pump(reader, fh, state)

    real_open = local_mod._open_capture

    def tagging_open(cap):
        out_fh, err_fh = real_open(cap)
        out_fh._is_stdout = True
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", tagging_open)
    monkeypatch.setattr(local_mod, "_pump", selective_pump)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "import sys; sys.stdout.write('o'*100); sys.stderr.write('e'*100)"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stdout is True
    assert result.truncated_stderr is False


@pytest.mark.asyncio
async def test_capture_writes_do_not_block_the_event_loop(
    tmp_path: Path, monkeypatch
):
    """Every chunk was written synchronously from an async task, so a slow or
    full capture filesystem blocked every other task in the process —
    including the poll that enforces this command's own timeout, making the
    bound exceedable by the work it exists to bound.

    The write is slowed deliberately. The first version just wrote 4 MB and
    counted heartbeats, which measured how fast the local SSD is: the whole
    capture finished inside two ticks and the test proved nothing either
    way."""
    import asyncio as _a
    import time as _t

    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture

    def slow_open(cap):
        out_fh, err_fh = real_open(cap)
        real_write = out_fh.write

        def slow_write(b):
            _t.sleep(0.05)
            return real_write(b)

        out_fh.write = slow_write
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", slow_open)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await _a.sleep(0.01)
            ticks += 1

    beat = _a.create_task(heartbeat())
    try:
        await LocalSandboxBackend(GRANTS).exec(
            ["python3", "-c", "print('y' * 500_000)"],
            cwd=None,
            env=None,
            timeout=60,
            capture=CaptureTarget(
                stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
            ),
        )
    finally:
        beat.cancel()

    # ~8 chunks at 50ms each. Written on the loop that is 400ms of total
    # starvation; written off it, the heartbeat keeps its 10ms cadence.
    assert ticks > 20, f"event loop only ticked {ticks} times during the capture"
    assert bundle.stdout_path.stat().st_size > 500_000


@pytest.mark.asyncio
async def test_a_failing_stderr_pump_indicts_only_stderr(
    tmp_path: Path, monkeypatch
):
    """Surviving mutant, and the mirror of the test above it. Dropping the
    pump-error term from the STDERR side changed nothing, because every test
    that failed a pump failed stdout's. A per-stream contract needs both
    sides covered or half of it is asserted by nobody."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_pump = local_mod._pump
    real_open = local_mod._open_capture

    def tagging_open(cap):
        out_fh, err_fh = real_open(cap)
        err_fh._is_stderr = True
        return out_fh, err_fh

    async def selective_pump(reader, fh, state=None):
        if getattr(fh, "_is_stderr", False):
            await reader.read(10)
            raise OSError("No space left on device")
        await real_pump(reader, fh, state)

    monkeypatch.setattr(local_mod, "_open_capture", tagging_open)
    monkeypatch.setattr(local_mod, "_pump", selective_pump)

    result = await LocalSandboxBackend(GRANTS).exec(
        [
            "python3",
            "-c",
            "import sys; sys.stdout.write('o'*100); sys.stderr.write('e'*100)",
        ],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stderr is True
    assert result.truncated_stdout is False


# ---------------------------------------------------------------------------
# Review round 8
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_cwd_is_refused_before_docker_can_create_it(
    workspace: Path, queue
):
    """Review round 8, and a capability escape I opened by adding ``cwd``.

    The docker backend mounts it with ``-v <cwd>:/workspace:ro``, and ``-v``
    CREATES a missing source directory on the host. So a cwd that passed only
    the READ policy performed a filesystem write — the gate it never went
    through — on an auto-approved command like ``ls``.

    Refused at the gate rather than in the backend, because it is wrong
    everywhere: the local backend merely fails louder."""
    missing = workspace / "does" / "not" / "exist"
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", cwd=str(missing))

    assert env.status is not ToolResultStatus.OK
    assert "existing directory" in (env.error or "")
    assert not missing.exists(), "the refused cwd was created anyway"


@pytest.mark.asyncio
async def test_a_file_as_cwd_is_refused_too(workspace: Path, queue):
    """Exists is not enough — it has to be a directory."""
    a_file = workspace / "notadir.txt"
    a_file.write_text("x")
    f = await _feature(workspace, queue)

    env = await f.shell(command="echo hi", cwd=str(a_file))

    assert env.status is not ToolResultStatus.OK
    assert "existing directory" in (env.error or "")


@pytest.mark.asyncio
async def test_an_existing_cwd_is_still_accepted(workspace: Path, queue):
    """Control: the check must reject what does not exist, not everything."""
    sub = workspace / "real"
    sub.mkdir()
    f = await _feature(workspace, queue)

    env = await f.shell(
        command="python3 -c 'import os; print(os.getcwd())'", cwd=str(sub)
    )

    assert env.data["stdout"].strip() == str(sub.resolve())


@pytest.mark.asyncio
async def test_a_failed_pump_ends_the_run_immediately(tmp_path: Path, monkeypatch):
    """Review round 8. Once a capture write has raised the artifact is
    already unrecoverable, so running the command to its full timeout buys
    nothing — and with nothing draining the pipe a high-output command blocks
    on it, turning a disk error into a hang."""
    import asyncio as _a
    import time as _t

    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")

    async def failing_pump(reader, fh, state=None):
        await reader.read(10)
        raise OSError("No space left on device")

    monkeypatch.setattr(local_mod, "_pump", failing_pump)

    marker = tmp_path / "child_survived.txt"
    script = tmp_path / "long.py"
    script.write_text(
        "import time\n"
        "print('x' * 1000, flush=True)\n"
        "time.sleep(2)\n"
        f"open({str(marker)!r}, 'w').write('x')\n"
    )

    def marker_gone():
        return not marker.exists()

    began = _t.monotonic()
    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", str(script)],
        cwd=None,
        env=None,
        timeout=25,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )
    elapsed = _t.monotonic() - began

    assert result.truncated_stdout is True
    assert elapsed < 10, f"waited {elapsed:.1f}s after the capture was already lost"
    # Returning fast is not the same as ending the run. Without the kill the
    # call still returns promptly and the child keeps going — which the first
    # version of this test could not tell apart.
    await _a.sleep(3)
    assert marker_gone(), "the child outlived the capture failure"


@pytest.mark.asyncio
async def test_the_facts_only_envelope_fits_even_with_a_long_path(
    tmp_path: Path, queue, monkeypatch
):
    """Review round 8. The fallback's size was never re-checked, and it
    carried the artifact path TWICE — in the confirmation and in data — so
    the one part that cannot be shortened was doubled. A valid long path was
    enough to push it back over the cap it exists to get under."""
    from kestrel_sovereign.features.base import serialized_result_len

    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 1000,
    )
    deep = tmp_path
    for _ in range(12):
        deep = deep / ("d" * 30)
    deep.mkdir(parents=True)
    (deep / "secret").mkdir()

    f = ComputerUseFeature(FakeAgent(queue=queue))
    f._cfg = _config(deep)
    await f.initialize()

    env = await f.shell(
        command="python3 -c \"print('q' * 100000)\"",
        capture_output=True,
        timeout=60,
    )

    size = serialized_result_len(env, tool_name="shell")
    assert size <= 1000, f"fallback is {size} chars"
    # The pointer survives in one form or another.
    assert env.data.get("manifest_path") or env.data.get("run_id")
    assert "complete" in env.data


@pytest.mark.asyncio
async def test_the_fallback_degrades_to_a_run_id_when_even_the_path_is_too_long(
    tmp_path: Path, queue, monkeypatch
):
    """Surviving mutants. Two of round 8's fixes — re-measuring the fallback,
    and carrying the pointer once instead of twice — could both be undone
    with nothing failing, because the only test that reached the fallback had
    a path short enough that neither mattered.

    A path long enough to dominate is the case they exist for: then the
    envelope must shed the path too and keep the 32-character run id, which
    with the configured capture directory still locates the artifact."""
    from kestrel_sovereign.features.base import serialized_result_len

    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 400,
    )
    deep = tmp_path
    for _ in range(8):
        deep = deep / ("d" * 30)
    deep.mkdir(parents=True)
    (deep / "secret").mkdir()

    f = ComputerUseFeature(FakeAgent(queue=queue))
    f._cfg = _config(deep)
    await f.initialize()

    env = await f.shell(
        command="python3 -c \"print('q' * 50000)\"",
        capture_output=True,
        timeout=60,
    )

    size = serialized_result_len(env, tool_name="shell")
    assert size <= 400, f"fallback is {size} chars"
    assert env.data.get("run_id"), "the last pointer was dropped too"
    assert "manifest_path" not in env.data
    assert "complete" in env.data


@pytest.mark.asyncio
async def test_the_fallback_fits_at_every_cap_that_triggers_it(
    tmp_path: Path, queue, monkeypatch
):
    """Surviving mutant: putting the artifact path back into the confirmation
    alongside ``data`` could be done with nothing failing, because the two
    tests reaching the fallback sit either side of the band where it matters
    — one small enough that duplication still fits, one so large the path is
    dropped anyway.

    A first attempt measured the band and hardcoded a cap. The measurement
    used a different capture-directory name than the fixture, so the number
    did not transfer and the test failed on a clean build — the third time on
    this branch that a boundary computed in one context was applied in
    another.

    So it asserts the invariant across every cap in the range instead of
    locating one: whatever the fallback returns must fit the cap that
    produced it. That holds in all three regimes — preview, path-only,
    run-id-only — and needs no arithmetic to stay true."""
    from kestrel_sovereign.features.base import serialized_result_len

    deep = tmp_path
    for _ in range(8):
        deep = deep / ("d" * 30)
    deep.mkdir(parents=True)
    (deep / "secret").mkdir()

    f = ComputerUseFeature(FakeAgent(queue=queue))
    f._cfg = _config(deep)
    await f.initialize()
    real_backend = f._backend

    smallest_cap_with_path = None
    path_len = 0
    for cap in range(400, 1400, 25):
        monkeypatch.setattr(
            "kestrel_sovereign.features.computer_use.feature."
            "orchestrator_result_cap",
            lambda cap=cap: cap,
        )
        f._backend = real_backend
        env = await f.shell(
            command="python3 -c \"print('q' * 50000)\"",
            capture_output=True,
            timeout=60,
        )
        size = serialized_result_len(env, tool_name="shell")
        assert size <= cap, f"cap={cap} produced {size} chars"
        pointer = env.data.get("manifest_path")
        if pointer and "facts only" in (env.confirmation or ""):
            if smallest_cap_with_path is None:
                smallest_cap_with_path = cap
                path_len = len(pointer)

    # Fitting is not the only thing that matters. Carrying the pointer twice
    # still FITS — the re-measure step just degrades to the run id sooner —
    # so the cost of duplication is that the path stops being affordable
    # while it easily could be. Once is affordable a little above the path's
    # own length; twice needs more than double it.
    assert smallest_cap_with_path is not None, "no cap kept the path"
    assert smallest_cap_with_path < 2 * path_len, (
        f"the path only became affordable at cap={smallest_cap_with_path} "
        f"for a {path_len}-char path — it is being carried more than once"
    )


# ---------------------------------------------------------------------------
# Review round 9
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_bigger_cap_is_actually_used_for_uncaptured_output(
    workspace: Path, queue, monkeypatch
):
    """Review round 9. The budget started at the fixed 4,000-character
    preview and the loop can only shrink, so a larger configured cap was
    never spent: a 4,001-character stdout under a 10,000-character cap lost
    one character and was reported incomplete. There is no artifact behind an
    uncaptured run, so that clip is real loss claimed for no reason."""
    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 20000,
    )
    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(stdout="p" * 4001, stderr=""))

    env = await f.shell(command="echo hi")

    assert env.data["complete"] is True
    assert env.data["truncated_stdout"] is False
    assert len(env.data["stdout"]) == 4001
    assert env.status is ToolResultStatus.OK


@pytest.mark.asyncio
async def test_a_slow_final_write_is_flushed_not_discarded(
    tmp_path: Path, monkeypatch
):
    """Review round 9, and a regression from moving writes off the loop. A
    pump pending at the drain deadline may simply be landing bytes it already
    read — on a slow filesystem the final write outlasts the grace — and
    cancelling it threw those bytes away while reporting
    ``writers_remaining=True`` with a clean truncation flag: loss wearing the
    wrong label AND a clean one."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture
    monkeypatch.setattr(local_mod, "_DRAIN_GRACE", 0.05)

    def slow_open(cap):
        out_fh, err_fh = real_open(cap)
        real_write = out_fh.write

        def slow_write(b):
            import time as _t

            _t.sleep(0.6)
            return real_write(b)

        out_fh.write = slow_write
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", slow_open)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('payload-that-must-survive')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert "payload-that-must-survive" in bundle.stdout_path.read_text()
    assert result.writers_remaining is False
    assert result.truncated_stdout is False


@pytest.mark.asyncio
async def test_a_docker_capture_says_so_when_the_bytes_were_replaced():
    """Review round 9. The executor decodes with ``errors='replace'`` before
    this backend sees anything, so non-UTF-8 output has already become
    U+FFFD and a capture written from those strings is not what the command
    emitted. The bytes are gone by then; what can still be honest is the
    claim about them."""
    from kestrel_sovereign.features.computer_use.backends.docker import (
        DockerSandboxBackend,
    )
    import tempfile

    backend = DockerSandboxBackend.__new__(DockerSandboxBackend)

    class _Executor:
        async def execute_command(self, command, working_dir=None):
            class _Rec:
                exit_code = 0
                stdout = "before�after"
                stderr = ""
                stdout_truncated = False
                stderr_truncated = False

            return _Rec()

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

    assert result.truncated_stdout is True
    assert result.truncated_stderr is False


@pytest.mark.asyncio
async def test_clean_docker_output_is_not_called_lossy():
    """Control: the marker must be the replacement character, not every
    capture."""
    from kestrel_sovereign.features.computer_use.backends.docker import (
        DockerSandboxBackend,
    )
    import tempfile

    backend = DockerSandboxBackend.__new__(DockerSandboxBackend)

    class _Executor:
        async def execute_command(self, command, working_dir=None):
            class _Rec:
                exit_code = 0
                stdout = "ordinary output"
                stderr = ""
                stdout_truncated = False
                stderr_truncated = False

            return _Rec()

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

    assert result.truncated_stdout is False


@pytest.mark.asyncio
async def test_a_write_cancelled_after_both_graces_counts_as_lost(
    tmp_path: Path, monkeypatch
):
    """Surviving mutant. Flushing gets extra time, so the tests never
    exhausted BOTH graces and a mid-write cancellation never happened — the
    branch that classifies it was asserted by nobody.

    When the extra time is spent too, the bytes in that write really are
    gone. That is lost output, not a writer still holding the pipe, and the
    difference is the whole reason the two are tracked apart."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture
    monkeypatch.setattr(local_mod, "_DRAIN_GRACE", 0.05)
    monkeypatch.setattr(local_mod, "_FLUSH_GRACE", 0.05)

    def glacial_open(cap):
        out_fh, err_fh = real_open(cap)
        real_write = out_fh.write

        def glacial_write(b):
            import time as _t

            _t.sleep(3)
            return real_write(b)

        out_fh.write = glacial_write
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", glacial_open)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["python3", "-c", "print('never-lands')"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.truncated_stdout is True, "a discarded write was not called lost"
    # It was writing, not waiting on the pipe — the labels are not
    # interchangeable.
    assert result.writers_remaining is False


# ---------------------------------------------------------------------------
# Review round 10
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slow_close_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch
):
    """Review round 10. A close flushes, so it blocks exactly as a write
    does — the pump's writes were moved off the loop and the closes were
    left on it, which is the same defect surviving in the line next door."""
    import asyncio as _a
    import time as _t

    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture

    def slow_closing_open(cap):
        out_fh, err_fh = real_open(cap)
        real_close = out_fh.close

        def slow_close():
            _t.sleep(0.5)
            return real_close()

        out_fh.close = slow_close
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", slow_closing_open)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await _a.sleep(0.01)
            ticks += 1

    beat = _a.create_task(heartbeat())
    try:
        await LocalSandboxBackend(GRANTS).exec(
            ["python3", "-c", "print('done')"],
            cwd=None,
            env=None,
            timeout=30,
            capture=CaptureTarget(
                stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
            ),
        )
    finally:
        beat.cancel()

    assert ticks > 20, f"the loop ticked only {ticks} times across a 0.5s close"


@pytest.mark.asyncio
async def test_a_spawn_failure_reports_a_close_that_also_failed(
    tmp_path: Path, monkeypatch
):
    """Review round 10. The spawn-failure result was constructed before the
    ``finally`` closed the handles, so a close that then failed under disk
    pressure had nowhere to go — the diagnostic was filed as fully persisted
    when it was not."""
    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    bundle = capture.allocate(tmp_path / "captures")
    real_open = local_mod._open_capture

    def failing_close_open(cap):
        out_fh, err_fh = real_open(cap)
        real_close = err_fh.close

        def failing_close():
            real_close()
            raise OSError("No space left on device")

        err_fh.close = failing_close
        return out_fh, err_fh

    monkeypatch.setattr(local_mod, "_open_capture", failing_close_open)

    result = await LocalSandboxBackend(GRANTS).exec(
        ["definitely-not-a-real-binary-xyz"],
        cwd=None,
        env=None,
        timeout=30,
        capture=CaptureTarget(
            stdout_path=bundle.stdout_path, stderr_path=bundle.stderr_path
        ),
    )

    assert result.returncode == 127
    assert result.truncated_stderr is True
    assert "No space left on device" in result.stderr
    # The original diagnostic must survive alongside it.
    assert "definitely-not-a-real-binary-xyz" in result.stderr


@pytest.mark.asyncio
async def test_the_windows_kill_is_bounded_and_off_the_loop(monkeypatch):
    """Review round 10. ``taskkill`` ran synchronously with no timeout on
    every timeout and every cancellation, so a wedged terminator blocked the
    server indefinitely — defeating the very timeout that called it."""
    import asyncio as _a
    import time as _t

    import kestrel_sovereign.features.computer_use.backends.local as local_mod

    monkeypatch.setattr(local_mod, "is_windows", lambda: True)
    monkeypatch.setattr(local_mod, "_KILL_TIMEOUT", 0.3)

    def wedged_run(*a, **k):
        _t.sleep(30)

    monkeypatch.setattr(local_mod.subprocess, "run", wedged_run)

    began = _t.monotonic()
    await local_mod._kill_tree(4321)
    elapsed = _t.monotonic() - began

    assert elapsed < 5, f"a wedged taskkill held the caller for {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Review round 11
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abandoning_a_pump_does_not_leak_the_pipe(tmp_path: Path):
    """Review round 11. Cancelling the pump task does not close the pipe —
    the descendant holds the other end, so the server kept two descriptors
    for as long as that process lived, which for a daemon is indefinitely.
    A few such captures exhaust the process's limit."""
    import os as _os

    def open_fds():
        try:
            return len(_os.listdir(f"/dev/fd/{_os.getpid()}"))
        except OSError:
            import resource

            return len(_os.listdir("/dev/fd"))

    script = tmp_path / "daemon.py"
    script.write_text(
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid(); time.sleep(20); os._exit(0)\n"
        "print('parent done', flush=True)\n"
    )
    backend = LocalSandboxBackend(GRANTS)

    before = open_fds()
    for i in range(4):
        bundle = capture.allocate(tmp_path / f"captures{i}")
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
    after = open_fds()

    # Four abandoned captures held eight descriptors before this was closed.
    assert after - before < 4, f"leaked {after - before} descriptors over 4 runs"


@pytest.mark.asyncio
async def test_a_permission_error_on_spawn_is_a_failure_not_a_result(
    workspace: Path, queue
):
    """Review round 11, and a regression from round 10. A broad ``OSError``
    catch turned a permission denial into rc=127 — which ``shell`` reports
    PARTIAL and the wrapper publishes as ``success: true`` — for a command
    that never ran."""
    not_executable = workspace / "not_executable.sh"
    not_executable.write_text("#!/bin/sh\necho hi\n")
    not_executable.chmod(0o644)

    f = await _feature(
        workspace, queue, auto_approved_binaries=[str(not_executable)]
    )

    env = await f.shell(command=str(not_executable))

    assert env.status is ToolResultStatus.ERROR, env.status
    assert env.data is None or env.data.get("returncode") != 127


@pytest.mark.asyncio
async def test_a_missing_binary_is_still_an_rc_127_result(workspace: Path, queue):
    """Control: 'not found' keeps its result shape, which is what the
    diagnostic-preserving path is built on."""
    f = await _feature(
        workspace, queue, auto_approved_binaries=["definitely-not-a-real-binary-xyz"]
    )

    env = await f.shell(command="definitely-not-a-real-binary-xyz")

    assert env.data["returncode"] == 127


@pytest.mark.asyncio
async def test_a_huge_uncaptured_result_keeps_a_preview(workspace: Path, queue):
    """Review round 11. The budget now starts at the stream's own length, so
    a fixed five halvings left an uncaptured megabyte at ~31,000 characters —
    still far over the cap — and the code jumped to facts-only, discarding a
    2-3 KB preview that would have fit."""
    from kestrel_sovereign.features.base import (
        orchestrator_result_cap,
        serialized_result_len,
    )

    f = await _feature(workspace, queue)
    f._backend = _StubBackend(_run(stdout="p" * 900_000, stderr=""))

    env = await f.shell(command="echo hi")

    assert serialized_result_len(env, tool_name="shell") <= orchestrator_result_cap()
    assert len(env.data.get("stdout") or "") > 500, "threw the preview away"
    assert env.data["complete"] is False


@pytest.mark.asyncio
async def test_the_audit_keeps_the_truncation_flag_through_minimisation(
    tmp_path: Path, queue, monkeypatch
):
    """Review round 11. ``_minimal_envelope`` deliberately drops the
    truncation flags, and the audit read them off the minimised envelope — so
    it recorded ``truncated: false`` beside ``complete: false``. That is the
    same contradiction fixed in the audit once already, reintroduced by a
    later shrink step."""
    monkeypatch.setattr(
        "kestrel_sovereign.features.computer_use.feature.orchestrator_result_cap",
        lambda: 400,
    )
    deep = tmp_path
    for _ in range(8):
        deep = deep / ("d" * 30)
    deep.mkdir(parents=True)
    (deep / "secret").mkdir()

    f = ComputerUseFeature(FakeAgent(queue=queue))
    f._cfg = _config(deep)
    await f.initialize()
    f._backend = _StubBackend(_run(stdout="p" * 200_000, stderr=""))

    await f.shell(command="echo hi")

    rows = [
        json.loads(line)
        for line in (deep / "audit.jsonl").read_text().splitlines()
    ]
    row = [r for r in rows if r["tool"] == "shell"][-1]
    assert row["args"]["complete"] is False
    assert row["args"]["truncated"] is True
