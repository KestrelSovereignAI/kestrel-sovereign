"""A failed Docker launch leaves no staging directory behind (#3117).

The executor stages every container's trash writes in a per-execution
``.staging-<hex>`` directory under the host trash root and promotes them out
after the run. The promote-and-remove used to begin only at the process wait,
so every exit between ``mkdir`` and the spawn (a refused rewrite, a script
write, a docker binary that is not there) leaked the directory: 589 empty ones
on one host, invisible to the trash listing, which hides dot-directories by
design so mid-flight entries are never listed or restored early.

Two halves: the guard now begins the moment the directory exists, and each
run sweeps stale staging directories that older code left behind.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from kestrel_sovereign.features.compute.executors.docker_executor import DockerExecutor
from kestrel_sovereign.features.compute.models import ExecutionRecord
from tests.unit.test_compute_executors import (
    _CompletedProcess,
    _is_docker_command,
    _make_executor,
    _script,
    _SuccessfulProcess,
)


def _staging_dirs(trash_root: Path) -> list[Path]:
    return sorted(p for p in trash_root.iterdir() if p.name.startswith(".staging-"))


@pytest.fixture
def executor_with_trash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    executor = _make_executor(monkeypatch, "docker")
    trash_root = tmp_path / "trash"
    executor._policy.trash_dir = trash_root
    return executor, trash_root


# ---------------------------------------------------------------------------
# The guard covers every exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_docker_binary_at_spawn_leaves_no_staging_dir(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The ordinary case on a machine running unit tests: ``create_subprocess_exec``
    raises FileNotFoundError. The record says failed; the trash root is clean."""
    executor, trash_root = executor_with_trash
    seen_binds: list[str] = []

    async def create_subprocess(*command: object, **_kwargs: object):
        seen_binds.extend(str(c) for c in command if ".staging-" in str(c))
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert isinstance(record, ExecutionRecord)
    assert record.exit_code == -1
    assert seen_binds, "the staging bind was built before the spawn failed"
    assert _staging_dirs(trash_root) == []


@pytest.mark.asyncio
async def test_a_refused_rewrite_leaves_no_staging_dir(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """An exit BEFORE the spawn, the earliest way out after mkdir."""
    executor, trash_root = executor_with_trash

    def refuse(*_args, **_kwargs):
        raise ValueError("script refused by policy")

    monkeypatch.setattr(executor._policy, "rewrite_script", refuse)

    async def must_not_spawn(*_command: object, **_kwargs: object):
        raise AssertionError("no process may be spawned after a refused rewrite")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert record.exit_code == -1
    assert "refused" in record.stderr
    assert _staging_dirs(trash_root) == []


@pytest.mark.asyncio
async def test_a_successful_run_still_promotes_staged_entries_and_removes_the_dir(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The guard moved; the promotion it performs did not change."""
    executor, trash_root = executor_with_trash

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        # The container "deletes" a file: it lands in the staging bind.
        bind = next(str(c) for c in command if ".staging-" in str(c))
        staging = Path(bind.split(":", 1)[0])
        (staging / "rm_deadbeef").mkdir()
        (staging / "rm_deadbeef" / "victim.txt").write_text("v")
        return _SuccessfulProcess(b"ok", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert record.exit_code == 0
    assert _staging_dirs(trash_root) == []
    assert (trash_root / "rm_deadbeef" / "victim.txt").read_text() == "v"


# ---------------------------------------------------------------------------
# Stale staging directories from older code are swept
# ---------------------------------------------------------------------------


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_the_sweep_removes_stale_empties_promotes_stale_entries_and_keeps_live_ones(
    tmp_path: Path,
):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    stale_empty = trash_root / ".staging-aaaaaaaaaaaa"
    stale_empty.mkdir()
    stale_full = trash_root / ".staging-bbbbbbbbbbbb"
    (stale_full / "rm_cafe0001").mkdir(parents=True)
    (stale_full / "rm_cafe0001" / "kept.txt").write_text("k")
    live = trash_root / ".staging-cccccccccccc"
    live.mkdir()
    not_a_dir = trash_root / ".staging-dddddddddddd"
    not_a_dir.write_text("a file with a staging name")
    real_entry = trash_root / "rm_11112222"
    real_entry.mkdir()
    other_hidden = trash_root / ".not-staging"
    other_hidden.mkdir()
    for p in (stale_empty, stale_full, not_a_dir, real_entry, other_hidden):
        _age(p, DockerExecutor.STALE_STAGING_AGE_SECONDS + 60)
    _age(live, DockerExecutor.STALE_STAGING_AGE_SECONDS - 60)

    DockerExecutor._promote_stale_staging_dirs(trash_root)

    assert not stale_empty.exists()
    assert not stale_full.exists()
    assert (trash_root / "rm_cafe0001" / "kept.txt").read_text() == "k"
    assert live.is_dir(), "a directory younger than the bound may be a live bind mount"
    assert not_a_dir.is_file()
    assert real_entry.is_dir()
    assert other_hidden.is_dir(), "only the executor's own .staging- prefix is swept"


def test_the_sweep_tolerates_a_missing_trash_root(tmp_path: Path):
    DockerExecutor._promote_stale_staging_dirs(tmp_path / "absent")


@pytest.mark.asyncio
async def test_each_run_sweeps_before_staging_its_own(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The 589 on the live host go away on the next script run, without a
    manual step; the run's own directory is created after the sweep and
    removed by the guard."""
    executor, trash_root = executor_with_trash
    trash_root.mkdir()
    leaked = trash_root / ".staging-0123456789ab"
    leaked.mkdir()
    _age(leaked, DockerExecutor.STALE_STAGING_AGE_SECONDS + 60)

    async def create_subprocess(*_command: object, **_kwargs: object):
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert not leaked.exists()
    assert _staging_dirs(trash_root) == []


def test_the_stale_bound_exceeds_any_configured_script_timeout(monkeypatch):
    """A live run's directory is at most as old as its timeout; the sweep's
    bound must sit beyond the compute policy's maximum."""
    from kestrel_sovereign.features.compute.models import ComputePolicy

    assert DockerExecutor.STALE_STAGING_AGE_SECONDS > ComputePolicy().max_timeout_seconds
