"""A failed Docker launch leaves no staging directory behind (#3117).

The executor stages every container's trash writes in a per-execution
``.staging-<hex>`` directory under the host trash root and promotes them out
after the run. The promote-and-remove used to begin only at the process wait,
so every exit between ``mkdir`` and the spawn (a refused rewrite, a script
write, a docker binary that is not there) leaked the directory: 589 empty ones
on one host, invisible to the trash listing, which hides dot-directories by
design so mid-flight entries are never listed or restored early.

Three parts: the guard begins the moment anything exists (the owner record,
then the directory); each run sweeps staging directories whose OWNER is gone
(an owner record beside the directory, outside the bind mount, names the
process and its container; a directory with no record is legacy and is swept
once older than the configured maximum script timeout; an orphan record is
reaped); and nothing in the sweep or the promotion follows a symlink or trusts
a hidden name, since the staging bind is the one writable mount a container
gets and a promoted hidden entry would pass for a directory or record the
sweep trusts.
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
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def _dead_pid() -> int:
    """A pid no process has: fork a child that exits at once, reap it."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _write_owner(trash_root: Path, name: str, pid: int, container: str | None = None) -> Path:
    record = trash_root / f"{name}{OWNER}"
    record.write_text(json.dumps({"pid": pid, "container": container, "started": time.time()}))
    return record


def _sweep(trash_root: Path, docker_path: str | None = None, **kwargs) -> None:
    """Run one sweep with a bare executor (no docker: no container can be live)."""
    asyncio.run(DockerExecutor(**kwargs)._promote_stale_staging_dirs(trash_root, docker_path))


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
    assert seen["record"]["container"].startswith("kestrel_compute_")
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
    assert "Refusing to promote" in caplog.text, "promotion removed the link and said so"
    caplog.clear()

    # Even a link that somehow sits in the root is not a sweep candidate: the
    # sweep itself must skip it, not hand it to promotion for refusal there.
    # The LINK is aged (utime without following), so only lstat keeps it out.
    os.symlink(str(victim), trash_root / ".staging-planted")
    _age(trash_root / ".staging-planted", DockerExecutor()._legacy_staging_age_seconds + 3600)
    _age(victim, DockerExecutor()._legacy_staging_age_seconds + 3600)
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

    _sweep(trash_root)

    assert live.is_dir()
    assert (trash_root / f"{live.name}{OWNER}").is_file()


def test_a_directory_whose_owner_is_gone_is_swept_at_once_with_its_record(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    orphan = trash_root / ".staging-bbbbbbbbbbbb"
    (orphan / "rm_cafe0001").mkdir(parents=True)
    (orphan / "rm_cafe0001" / "kept.txt").write_text("k")
    # The record names a container too: with no docker binary on the host
    # nothing can be running, so the directory is reaped at once.
    record = _write_owner(trash_root, orphan.name, _dead_pid(), container="kestrel_compute_dead0000")

    _sweep(trash_root)

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
    _age(young, DockerExecutor()._legacy_staging_age_seconds - 60)
    old = trash_root / ".staging-dddddddddddd"
    old.mkdir()
    _age(old, DockerExecutor()._legacy_staging_age_seconds + 60)

    _sweep(trash_root)

    assert young.is_dir() and not old.exists()


def test_the_legacy_floor_defaults_to_the_shipped_maximum_and_follows_the_configured_one(
    tmp_path: Path,
):
    """A legacy directory is judged by age because it has nothing better. The
    floor defaults to the policy's shipped maximum and, when the feature
    passes its configured maximum, follows that: with a 24 h maximum a 2 h old
    record-less directory may be a live run and is left alone."""
    assert DockerExecutor.LEGACY_STAGING_AGE_SECONDS == ComputePolicy().max_timeout_seconds
    grace = DockerExecutor.LEGACY_STAGING_GRACE_SECONDS
    assert grace > 0
    assert DockerExecutor()._legacy_staging_age_seconds == ComputePolicy().max_timeout_seconds + grace
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    two_hours_old = trash_root / ".staging-2h2h2h2h2h2h"
    two_hours_old.mkdir()
    _age(two_hours_old, 2 * 3600)
    _sweep(trash_root, legacy_staging_age_seconds=24 * 3600)
    assert two_hours_old.is_dir()
    _sweep(trash_root)
    assert not two_hours_old.exists()


@pytest.mark.asyncio
async def test_the_feature_passes_its_configured_maximum_to_the_executor(
    monkeypatch, tmp_path: Path,
):
    """The executor's legacy floor is the policy's configured maximum, not
    the shipped default: with KESTREL_COMPUTE_MAX_TIMEOUT at a day, a
    record-less directory two hours old may still be a live run."""
    from unittest.mock import MagicMock

    from kestrel_sovereign.features.compute import destructive_policy, trash_manager
    from kestrel_sovereign.features.compute.feature import ComputeFeature

    monkeypatch.setenv("KESTREL_COMPUTE_MAX_TIMEOUT", "86400")
    # DEFAULT_TRASH_DIR is frozen at import (#3104); the env var alone is inert.
    monkeypatch.setattr(destructive_policy, "DEFAULT_TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(trash_manager, "DEFAULT_TRASH_DIR", tmp_path / "trash")
    monkeypatch.setattr(ComputeFeature, "_docker_available", lambda self: True)
    agent = MagicMock()
    agent.storage_path = str(tmp_path / "agent.db")
    feature = ComputeFeature(agent)
    await feature.initialize()
    assert feature.executors["docker"]._legacy_staging_age_seconds == (
        86400 + DockerExecutor.LEGACY_STAGING_GRACE_SECONDS
    )
    assert (tmp_path / "trash").is_dir(), "the feature used the patched trash root"


def test_an_unreadable_record_counts_as_no_record(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    d = trash_root / ".staging-eeeeeeeeeeee"
    d.mkdir()
    (trash_root / f"{d.name}{OWNER}").write_text("not json")
    _age(d, DockerExecutor()._legacy_staging_age_seconds - 60)
    _sweep(trash_root)
    assert d.is_dir(), "young and unreadable: left alone like a legacy directory"
    _age(d, DockerExecutor()._legacy_staging_age_seconds + 60)
    _sweep(trash_root)
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
    for p in (not_a_dir, other_hidden, real_entry):
        _age(p, 30 * 24 * 3600)

    _sweep(trash_root)

    assert not_a_dir.is_file() and other_hidden.is_dir() and real_entry.is_dir()


def test_an_orphan_record_is_reaped_once_its_owner_is_gone(tmp_path: Path):
    """The record is written before mkdir; a failed mkdir would otherwise leave
    a file no sweep, listing or purge ever touched (22 seen live)."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    dead = _write_owner(trash_root, ".staging-orphan000000", _dead_pid())
    live = _write_owner(trash_root, ".staging-orphan111111", os.getpid())
    garbage = trash_root / f".staging-orphan222222{OWNER}"
    garbage.write_text("not json")
    _sweep(trash_root)
    assert not dead.exists()
    assert live.is_file(), "its owner may be between writing the record and mkdir"
    assert garbage.is_file(), "unreadable and young: left alone"
    _age(garbage, DockerExecutor()._legacy_staging_age_seconds + 60)
    _sweep(trash_root)
    assert not garbage.exists()


@pytest.mark.asyncio
async def test_a_failed_mkdir_leaves_no_record_behind(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    executor, trash_root = executor_with_trash
    real_mkdir = Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self.name.startswith(".staging-"):
            raise OSError(28, "No space left on device")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    record = await asyncio.wait_for(executor.execute(_script()), timeout=2)
    assert record.exit_code == -1
    assert _owner_records(trash_root) == [] and _staging_dirs(trash_root) == []


def test_the_sweep_tolerates_a_missing_trash_root(tmp_path: Path):
    _sweep(tmp_path / "absent")


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
    _age(leaked, DockerExecutor()._legacy_staging_age_seconds + 60)
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


# ---------------------------------------------------------------------------
# A container cannot forge what the sweep trusts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_forged_record_with_an_overflowing_pid_cannot_fail_every_later_run(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The review's P1: a container writes `.staging-poison.owner` with a pid
    that overflows a C int; before, promotion moved it into the root and
    `os.kill` raised OverflowError out of the sweep at the head of every run."""
    executor, trash_root = executor_with_trash
    planted = {"done": False}

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        staging = _staging_from(command)
        if not planted["done"]:
            (staging / ".staging-poison0000").mkdir()
            (staging / f".staging-poison0000{OWNER}").write_text('{"pid": 99999999999999999999}')
            (staging / ".staging-pidone0000").mkdir()
            (staging / f".staging-pidone0000{OWNER}").write_text('{"pid": 1}')
            planted["done"] = True
        return _SuccessfulProcess(b"ok", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    first = await asyncio.wait_for(executor.execute(_script()), timeout=2)
    assert first.exit_code == 0
    assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []
    for _ in range(3):
        record = await asyncio.wait_for(executor.execute(_script()), timeout=2)
        assert record.exit_code == 0, record.stderr


def test_a_record_that_is_not_ours_is_no_record(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    d = trash_root / ".staging-forged000000"
    d.mkdir()
    _age(d, DockerExecutor()._legacy_staging_age_seconds + 60)
    for forged in ('{"pid": 99999999999999999999}', '{"pid": 1.5}', '{"pid": true}', '[1]', '{"pid": -4}', "x" * 5000):
        (trash_root / f"{d.name}{OWNER}").write_text(forged)
        assert DockerExecutor._read_owner_record(trash_root / f"{d.name}{OWNER}") is None
    _sweep(trash_root)
    assert not d.exists(), "an unreadable record on an old directory is legacy: swept"


def test_the_sweep_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    (trash_root / ".staging-boom00000000").mkdir()

    def boom(*_a, **_k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(DockerExecutor, "_read_owner_record", staticmethod(boom))
    with caplog.at_level("WARNING"):
        _sweep(trash_root)
    assert "Sweep of stale staging directories" in caplog.text


@pytest.mark.asyncio
async def test_a_dead_owner_whose_container_docker_still_knows_is_left_alone(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """`docker run --rm` is a foreground client: a killed agent's container
    keeps running with the bind. The sweep asks Docker before reaping."""
    executor, trash_root = executor_with_trash
    trash_root.mkdir()
    live_container = trash_root / ".staging-livecontain"
    live_container.mkdir()
    _write_owner(trash_root, live_container.name, _dead_pid(), container="kestrel_compute_live")
    gone_container = trash_root / ".staging-gonecontain"
    gone_container.mkdir()
    _write_owner(trash_root, gone_container.name, _dead_pid(), container="kestrel_compute_gone")
    inspected: list[str] = []

    class _Exit:
        def __init__(self, code):
            self.returncode = code

        async def wait(self):
            return self.returncode

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "inspect"):
            inspected.append(str(command[2]))
            return _Exit(0 if command[2] == "kestrel_compute_live" else 1)
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert sorted(inspected) == ["kestrel_compute_gone", "kestrel_compute_live"]
    assert live_container.is_dir()
    assert (trash_root / f"{live_container.name}{OWNER}").is_file()
    assert not gone_container.exists()
    assert not (trash_root / f"{gone_container.name}{OWNER}").exists()


# ---------------------------------------------------------------------------
# Round 4: an inconclusive inspect, the listing window of the race, pid reuse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_inconclusive_docker_inspect_leaves_a_dead_owners_directory_alone(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """A wedged daemon or an unspawnable client answers nothing; nothing must
    not reap a bind a container may still be writing to."""
    executor, trash_root = executor_with_trash
    trash_root.mkdir()
    unknown = trash_root / ".staging-unknowncont"
    unknown.mkdir()
    _write_owner(trash_root, unknown.name, _dead_pid(), container="kestrel_compute_unknown")

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "inspect"):
            raise OSError("cannot spawn the docker client")
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    await asyncio.wait_for(executor.execute(_script()), timeout=2)

    assert unknown.is_dir()
    assert (trash_root / f"{unknown.name}{OWNER}").is_file()


def test_the_loser_of_a_promotion_race_is_quiet_when_the_listing_vanishes(tmp_path: Path, caplog):
    """The other window: B's lstat succeeds, A promotes and removes, B's
    listing finds nothing."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    stale = trash_root / ".staging-abcdefabcdef"
    (stale / "rm_00000002").mkdir(parents=True)
    (stale / "rm_00000002" / "f").write_text("x")
    _write_owner(trash_root, stale.name, _dead_pid())

    class _Racing:
        def __init__(self, path):
            self._path = path

        def __getattr__(self, name):
            return getattr(self._path, name)

        def lstat(self):
            st = self._path.lstat()
            DockerExecutor._promote_staged_trash(self._path, trash_root)  # A wins now
            return st

    with caplog.at_level("DEBUG"):
        DockerExecutor._promote_staged_trash(_Racing(stale), trash_root)  # B loses at iterdir

    assert "NOT visible" not in caplog.text
    assert (trash_root / "rm_00000002" / "f").read_text() == "x"
    assert not (trash_root / f"{stale.name}{OWNER}").exists()


def test_a_record_older_than_any_script_may_run_is_reaped_even_if_its_pid_answers(
    tmp_path: Path, caplog,
):
    """After a reboot or a pid wraparound the recorded pid belongs to someone
    else; a pid this user cannot signal counts as alive and would pin the
    directory forever."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    reused = trash_root / ".staging-reusedpid000"
    reused.mkdir()
    record = trash_root / f"{reused.name}{OWNER}"
    record.write_text(json.dumps({
        "pid": 1,  # alive by PermissionError on every host
        "container": "kestrel_compute_old",
        "started": time.time() - DockerExecutor.OWNER_MAX_AGE_SECONDS - 60,
    }))
    fresh = trash_root / ".staging-freshpid000"
    fresh.mkdir()
    _write_owner(trash_root, fresh.name, 1)

    with caplog.at_level("WARNING"):
        _sweep(trash_root)

    assert not reused.exists() and not record.exists()
    assert "treated as reused" in caplog.text
    assert fresh.is_dir(), "a young record naming a live pid is still an owner"


@pytest.mark.asyncio
async def test_an_expired_record_whose_container_docker_still_knows_is_kept(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """Age alone is not proof of a reused pid; Docker's answer is."""
    executor, trash_root = executor_with_trash
    trash_root.mkdir()
    old_live = trash_root / ".staging-oldbutlive00"
    old_live.mkdir()
    (trash_root / f"{old_live.name}{OWNER}").write_text(json.dumps({
        "pid": 1, "container": "kestrel_compute_oldlive",
        "started": time.time() - DockerExecutor.OWNER_MAX_AGE_SECONDS * 3,
    }))

    class _Exit:
        def __init__(self, code):
            self.returncode = code

        async def wait(self):
            return self.returncode

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "inspect"):
            return _Exit(0)
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    await asyncio.wait_for(executor.execute(_script()), timeout=2)
    assert old_live.is_dir()


def test_the_expiry_bound_follows_a_long_configured_maximum(tmp_path: Path):
    """With a ten-day maximum, a record eight days old names a run that may
    still be in flight."""
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    d = trash_root / ".staging-longrun00000"
    d.mkdir()
    (trash_root / f"{d.name}{OWNER}").write_text(json.dumps({
        "pid": os.getpid(), "container": None, "started": time.time() - 8 * 24 * 3600,
    }))
    _sweep(trash_root, legacy_staging_age_seconds=10 * 24 * 3600)
    assert d.is_dir()
    _sweep(trash_root)
    assert not d.exists()


def test_an_orphan_record_older_than_any_script_may_run_is_reaped_too(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    record = trash_root / f".staging-oldorphan000{OWNER}"
    record.write_text(json.dumps({
        "pid": os.getpid(), "started": time.time() - DockerExecutor.OWNER_MAX_AGE_SECONDS - 60,
    }))
    _sweep(trash_root)
    assert not record.exists()


# ---------------------------------------------------------------------------
# Round 5: what a container leaves that the host cannot delete, and a bounded sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hidden_directory_the_host_cannot_remove_is_quarantined_not_left_behind(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """The review's P2: a hidden directory made unwritable defeated rmtree,
    rmdir raised ENOTEMPTY, and the staging directory and its record stayed
    in the root forever. Moving the entry aside needs only the staging
    directory's own permission."""
    executor, trash_root = executor_with_trash
    made: dict = {}

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "rm"):
            return _CompletedProcess()
        staging = _staging_from(command)
        hostile = staging / ".evil"
        (hostile / "inner").mkdir(parents=True)
        (hostile / "inner" / "f").write_text("x")
        os.chmod(hostile / "inner", 0o500)
        os.chmod(hostile, 0o500)
        (staging / "rm_legit0001").mkdir()
        (staging / "rm_legit0001" / "victim.txt").write_text("v")
        made["staging"] = staging
        return _SuccessfulProcess(b"ok", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    try:
        record = await asyncio.wait_for(executor.execute(_script()), timeout=2)
        assert record.exit_code == 0
        assert not made["staging"].exists()
        assert _staging_dirs(trash_root) == [] and _owner_records(trash_root) == []
        assert (trash_root / "rm_legit0001" / "victim.txt").read_text() == "v"
        quarantine = trash_root / DockerExecutor.QUARANTINE_DIR_NAME
        moved = list(quarantine.iterdir())
        assert len(moved) == 1
        # Either the entry moved on its own, or (an unwritable directory will
        # not move) the whole staging directory was moved aside.
        assert moved[0].name == made["staging"].name or moved[0].name.endswith("-.evil")
        assert list(moved[0].rglob("f")), "the container's content is kept, not deleted"
    finally:
        for root, dirs, _files in os.walk(trash_root):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o700)


def test_the_quarantine_is_never_a_sweep_candidate(tmp_path: Path):
    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    quarantine = trash_root / DockerExecutor.QUARANTINE_DIR_NAME
    (quarantine / ".staging-aaaaaaaaaaaa-.staging-forged").mkdir(parents=True)
    _age(quarantine, 30 * 24 * 3600)
    _sweep(trash_root)
    assert (quarantine / ".staging-aaaaaaaaaaaa-.staging-forged").is_dir()


@pytest.mark.asyncio
async def test_one_sweep_spends_a_bounded_time_asking_docker(
    executor_with_trash, monkeypatch: pytest.MonkeyPatch,
):
    """Five dead-owner directories, a daemon that answers slowly, a budget of
    a fraction of a second: not every directory is inspected this run, and
    none of the uninspected ones is reaped."""
    executor, trash_root = executor_with_trash
    trash_root.mkdir()
    dirs = []
    for i in range(5):
        d = trash_root / f".staging-slowdaemon{i:02d}"
        d.mkdir()
        _write_owner(trash_root, d.name, _dead_pid(), container=f"kestrel_compute_slow{i}")
        dirs.append(d)
    monkeypatch.setattr(DockerExecutor, "SWEEP_INSPECT_BUDGET_SECONDS", 0.12)
    inspected: list[str] = []

    class _Exit:
        returncode = 1

        async def wait(self):
            return 1

    async def create_subprocess(*command: object, **_kwargs: object):
        if _is_docker_command(command, "inspect"):
            inspected.append(str(command[2]))
            await asyncio.sleep(0.05)
            return _Exit()
        raise FileNotFoundError(2, "No such file or directory", "/fake/docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    await asyncio.wait_for(executor.execute(_script()), timeout=5)

    assert 0 < len(inspected) < 5
    remaining = [d for d in dirs if d.exists()]
    assert len(remaining) == 5 - len(inspected)
    for d in remaining:
        assert (trash_root / f"{d.name}{OWNER}").is_file()
