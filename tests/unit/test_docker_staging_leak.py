"""A failed Docker launch leaves no staging directory behind (#3117).

The executor stages every container's trash writes in a per-execution
``.staging-<hex>`` directory under the host trash root and promotes them out
after the run. The promote-and-remove used to begin only at the process wait,
so every exit between ``mkdir`` and the spawn (a refused rewrite, a script
write, a docker binary that is not there) leaked the directory: 589 empty ones
on one host, invisible to the trash listing, which hides dot-directories by
design so mid-flight entries are never listed or restored early.

Three parts: the guard begins the moment the directory exists; each run
sweeps staging directories whose OWNER is gone (an owner record beside the
directory, outside the bind mount, names the process; a directory with no
record is legacy and is swept once older than the default maximum script
timeout); and nothing in the sweep or the promotion follows a symlink, since
the staging bind is the one writable mount a container gets.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from kestrel_sovereign.features.compute.executors.docker_executor import DockerExecutor
from kestrel_sovereign.features.compute.models import ComputePolicy, ExecutionRecord
from tests.unit.test_compute_executors import (
    _CompletedProcess,
    _is_docker_command,
    _make_executor,
    _script,
    _SuccessfulProcess,
)

OWNER = DockerExecutor.STAGING_OWNER_SUFFIX


def _staging_dirs(trash_root: Path) -> list[Path]:
    return sorted(
        p for p in trash_root.iterdir()
        if p.name.startswith(".staging-") and not p.name.endswith(OWNER)
    )


def _owner_records(trash_root: Path) -> list[Path]:
    return sorted(p for p in trash_root.iterdir() if p.name.endswith(OWNER))


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _dead_pid() -> int:
    """A pid no process has: fork a child that exits at once, reap it."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _write_owner(trash_root: Path, name: str, pid: int) -> Path:
    record = trash_root / f"{name}{OWNER}"
    record.write_text(json.dumps({"pid": pid, "started": time.time()}))
    return record


@pytest.fixture
def executor_with_trash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    executor = _make_executor(monkeypatch, "docker")
    trash_root = tmp_path / "trash"
    executor._policy.trash_dir = trash_root
    return executor, trash_root


def _spawn_failure(monkeypatch: pytest.MonkeyPatch, seen_binds: list[str] | None = None):
    async def create_subprocess(*command: object, **_kwargs: object):
        if seen_binds is not None:
            seen_binds.extend(str(c) for c in command if ".staging-" in str(c))
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)


def _staging_from(command: tuple[object, ...]) -> Path:
    bind = next(str(c) for c in command if ".staging-" in str(c))
    return Path(bind.split(":", 1)[0])


# ---------------------------------------------------------------------------
# The guard covers every exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_docker_binary_at_spawn_leaves_no_staging_dir_or_record(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The ordinary case on a machine running unit tests: ``create_subprocess_exec``
    raises FileNotFoundError. The record says failed; the trash root is clean."""
    executor, trash_root = executor_with_trash
    seen_binds: list[str] = []
    _spawn_failure(monkeypatch, seen_binds)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert isinstance(record, ExecutionRecord)
    assert record.exit_code == -1
    assert seen_binds, "the staging bind was built before the spawn failed"
    assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []


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
    assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []


@pytest.mark.asyncio
async def test_a_successful_run_promotes_staged_entries_and_removes_dir_and_record(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The guard moved; the promotion it performs did not change. While the
    container runs, the owner record names this process and sits beside the
    directory, outside the bind."""
    executor, trash_root = executor_with_trash
    seen: dict[str, object] = {}

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        staging = _staging_from(command)
        record = trash_root / f"{staging.name}{OWNER}"
        seen["record"] = json.loads(record.read_text())
        seen["record_inside_bind"] = (staging / record.name).exists()
        # The container "deletes" a file: it lands in the staging bind.
        (staging / "rm_deadbeef").mkdir()
        (staging / "rm_deadbeef" / "victim.txt").write_text("v")
        return _SuccessfulProcess(b"ok", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert record.exit_code == 0
    assert seen["record"]["pid"] == os.getpid()
    assert seen["record_inside_bind"] is False
    assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []
    assert (trash_root / "rm_deadbeef" / "victim.txt").read_text() == "v"


# ---------------------------------------------------------------------------
# A container cannot turn the sweep against the host
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_symlink_planted_in_the_bind_is_never_promoted_or_followed(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog,
):
    """The review's reproduction: the container's only action is one
    ``os.symlink("../agent_data", staging / ".staging-pwned")``. Before, the
    link was promoted into the root and the next sweep emptied its target."""
    executor, trash_root = executor_with_trash
    victim = tmp_path / "agent_data"
    victim.mkdir()
    (victim / "memory.db").write_text("AGENT MEMORY")
    (victim / "keys").mkdir()
    (victim / "keys" / "signing.pem").write_text("KEY")
    planted = {"done": False}

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        staging = _staging_from(command)
        if not planted["done"]:
            os.symlink("../../agent_data", staging / ".staging-pwned")
            planted["done"] = True
        return _SuccessfulProcess(b"ok", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with caplog.at_level("WARNING"):
        await asyncio.wait_for(executor.execute(_script()), timeout=2)  # plants
    assert not (trash_root / ".staging-pwned").exists()
    assert not (trash_root / ".staging-pwned").is_symlink()
    assert "Refusing to promote symlink" in caplog.text, "promotion removed the link and said so"
    caplog.clear()

    # Even a link that somehow sits in the root is not a sweep candidate: the
    # sweep itself must skip it, not hand it to promotion for refusal there.
    os.symlink(str(victim), trash_root / ".staging-planted")
    _age(victim, DockerExecutor.LEGACY_STAGING_AGE_SECONDS + 3600)
    with caplog.at_level("DEBUG"):
        await asyncio.wait_for(executor.execute(_script()), timeout=2)  # sweeps

    assert sorted(p.name for p in victim.iterdir()) == ["keys", "memory.db"]
    assert (trash_root / ".staging-planted").is_symlink()
    assert "Refusing to promote" not in caplog.text, "the sweep followed the link"


def test_the_promotion_refuses_a_staging_path_that_is_a_symlink(tmp_path: Path, caplog):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("k")
    link = trash_root / ".staging-link"
    os.symlink(str(victim), link)

    with caplog.at_level("WARNING"):
        DockerExecutor._promote_staged_trash(link, trash_root)

    assert (victim / "keep.txt").read_text() == "k"
    assert link.is_symlink()
    assert "Refusing to promote" in caplog.text


# ---------------------------------------------------------------------------
# Stale means "no live owner", not "old"
# ---------------------------------------------------------------------------


def test_a_directory_whose_owner_is_running_is_never_swept_however_old(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    live = trash_root / ".staging-aaaaaaaaaaaa"
    live.mkdir()
    _write_owner(trash_root, live.name, os.getpid())
    _age(live, 30 * 24 * 3600)

    DockerExecutor._promote_stale_staging_dirs(trash_root)

    assert live.is_dir()
    assert (trash_root / f"{live.name}{OWNER}").is_file()


def test_a_directory_whose_owner_is_gone_is_swept_at_once_with_its_record(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    orphan = trash_root / ".staging-bbbbbbbbbbbb"
    (orphan / "rm_cafe0001").mkdir(parents=True)
    (orphan / "rm_cafe0001" / "kept.txt").write_text("k")
    record = _write_owner(trash_root, orphan.name, _dead_pid())

    DockerExecutor._promote_stale_staging_dirs(trash_root)

    assert not orphan.exists() and not record.exists()
    assert (trash_root / "rm_cafe0001" / "kept.txt").read_text() == "k"


def test_a_legacy_directory_with_no_record_is_swept_only_past_the_default_timeout(
    tmp_path: Path,
):
    """The 589 on the live host have no record. A young one may be an
    older process's live bind mount across an upgrade; an old one is not."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    young = trash_root / ".staging-cccccccccccc"
    young.mkdir()
    _age(young, DockerExecutor.LEGACY_STAGING_AGE_SECONDS - 60)
    old = trash_root / ".staging-dddddddddddd"
    old.mkdir()
    _age(old, DockerExecutor.LEGACY_STAGING_AGE_SECONDS + 60)

    DockerExecutor._promote_stale_staging_dirs(trash_root)

    assert young.is_dir() and not old.exists()


def test_the_legacy_floor_is_the_default_maximum_script_timeout():
    """A legacy directory is judged by age because it has nothing better; the
    floor is the policy's shipped maximum, the longest any default run lasts."""
    assert DockerExecutor.LEGACY_STAGING_AGE_SECONDS == ComputePolicy().max_timeout_seconds


def test_an_unreadable_record_counts_as_no_record(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    d = trash_root / ".staging-eeeeeeeeeeee"
    d.mkdir()
    (trash_root / f"{d.name}{OWNER}").write_text("not json")
    _age(d, DockerExecutor.LEGACY_STAGING_AGE_SECONDS - 60)
    DockerExecutor._promote_stale_staging_dirs(trash_root)
    assert d.is_dir(), "young and unreadable: left alone like a legacy directory"
    _age(d, DockerExecutor.LEGACY_STAGING_AGE_SECONDS + 60)
    DockerExecutor._promote_stale_staging_dirs(trash_root)
    assert not d.exists()


def test_the_sweep_touches_only_staging_directories(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    not_a_dir = trash_root / ".staging-ffffffffffff"
    not_a_dir.write_text("a file with a staging name")
    other_hidden = trash_root / ".not-staging"
    other_hidden.mkdir()
    real_entry = trash_root / "rm_11112222"
    real_entry.mkdir()
    stray_record = trash_root / f".staging-000000000000{OWNER}"
    stray_record.write_text("{}")
    for p in (not_a_dir, other_hidden, real_entry, stray_record):
        _age(p, 30 * 24 * 3600)

    DockerExecutor._promote_stale_staging_dirs(trash_root)

    assert not_a_dir.is_file() and other_hidden.is_dir() and real_entry.is_dir()
    assert stray_record.is_file(), "a record is never itself a candidate"


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
    _age(leaked, DockerExecutor.LEGACY_STAGING_AGE_SECONDS + 60)
    _spawn_failure(monkeypatch)

    await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert not leaked.exists()
    assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []


# ---------------------------------------------------------------------------
# Two sweeps on one stale directory
# ---------------------------------------------------------------------------


def test_the_loser_of_a_promotion_race_does_not_cry_stranded(tmp_path: Path, caplog):
    """Four agents share the root; both sweeps pass the check, A promotes
    first, B's renames find nothing. The entries are safe in the root and
    B says so at debug, not as a data-loss warning."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    stale = trash_root / ".staging-abcdefabcdef"
    (stale / "rm_00000001").mkdir(parents=True)
    (stale / "rm_00000001" / "f").write_text("x")
    entries = list(stale.iterdir())

    class _Replay:
        """B's view of the interleaving: it checked the directory and listed
        it BEFORE A promoted, and its renames run after."""

        def __init__(self, path):
            self._path = path

        def __getattr__(self, name):
            return getattr(self._path, name)

        def lstat(self):
            return trash_root.lstat()  # a directory, as B saw it

        def iterdir(self):
            return iter(entries)

    DockerExecutor._promote_staged_trash(stale, trash_root)  # A wins
    assert (trash_root / "rm_00000001" / "f").read_text() == "x"

    with caplog.at_level("DEBUG"):
        DockerExecutor._promote_staged_trash(_Replay(stale), trash_root)  # B loses

    assert "NOT visible" not in caplog.text
    assert (trash_root / "rm_00000001" / "f").read_text() == "x"
