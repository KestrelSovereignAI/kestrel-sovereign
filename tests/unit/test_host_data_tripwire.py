"""The session tripwire that keeps the suite off the real host-data root (#3286).

The isolation in ``tests/conftest.py`` is a convention; ``HostDataTripwire`` is
what makes breaking it fail the run. These tests point a tripwire at a
temporary root, never the operator's. The last one runs a nested pytest
session with a fake ``HOME`` to show the whole chain: a test without the
isolation migrates the "production" host database out of ``~/.kestrel``, and
the session fails naming that test.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.shared.host_runtime_isolation import (
    HostDataTripwire,
    real_host_data_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed_host_data(root: Path, *, live: bool) -> Path:
    root.mkdir(mode=0o700)
    database = root / "host-features.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.commit()
    connection.close()
    (root / "phoenix").mkdir()
    (root / "phoenix" / "phoenix.log").write_text("boot\n")
    if live:
        # What a running host leaves beside its database.
        (root / "host-features.db-wal").write_bytes(b"")
        (root / "host-features.db-shm").write_bytes(b"")
    return database


def _touch_later(path: Path, content: str) -> None:
    before = path.stat().st_mtime_ns
    path.write_text(content)
    os.utime(path, ns=(before + 10**9, before + 10**9))


def test_real_roots_name_the_operators_host_data_not_a_test_redirect(
    host_runtime_isolation_root,
):
    """Resolved at conftest import, before any fixture edited the environment."""
    from tests import conftest

    assert conftest._HOST_DATA_TRIPWIRE.roots
    for root in conftest._HOST_DATA_TRIPWIRE.roots:
        assert host_runtime_isolation_root not in root.parents
        assert root.name == "host-data"
    # Inside a test the redirect is in force, so resolving now would guard
    # the wrong directory.
    assert all(
        host_runtime_isolation_root in root.parents
        for root in real_host_data_roots()
    )


def test_quiescent_root_reports_every_created_modified_and_removed_entry(
    tmp_path,
):
    root = tmp_path / "host-data"
    database = _seed_host_data(root, live=False)
    tripwire = HostDataTripwire((root,))
    tripwire.arm()

    _touch_later(root / "phoenix" / "phoenix.log", "boot\nmore\n")
    (root / "stray.lock").write_bytes(b"")
    os.replace(database, tmp_path / "moved.db")

    findings = tripwire.disarm()
    text = "\n".join(findings)

    assert "modified phoenix/phoenix.log" in text
    assert "created stray.lock" in text
    assert "removed host-features.db" in text
    # The audit hook attributes the in-process accesses to this test.
    this_test = "test_quiescent_root_reports_every_created"
    assert any(
        this_test in finding and " os.rename " in finding for finding in findings
    )


def _as_another_process(code: str) -> None:
    """Act as a test's subprocess would: from a process the hook cannot see."""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


_LIVE_WRITER = """
import pathlib, sys
root = pathlib.Path(sys.argv[1])
held = [open(root / name, "ab") for name in sys.argv[2:]]
print("ready", flush=True)
for line in sys.stdin:
    for handle in held:
        handle.write(line.encode())
        handle.flush()
    print("wrote", flush=True)
"""


class _LiveWriter:
    """A stand-in for the running host: holds files open and writes to them."""

    def __init__(self, root: Path, *names: str) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-c", _LIVE_WRITER, str(root), *names],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout.readline().strip() == "ready"

    def write(self, text: str) -> None:
        self._process.stdin.write(text + "\n")
        self._process.stdin.flush()
        assert self._process.stdout.readline().strip() == "wrote"

    def close(self) -> None:
        self._process.stdin.close()
        self._process.wait(timeout=60)


needs_lsof = pytest.mark.skipif(
    shutil.which("lsof") is None,
    reason="identifying a live writer's open files needs lsof",
)


@needs_lsof
def test_live_writer_changes_are_not_findings_but_entry_changes_are(tmp_path):
    """A running host changes its own files; it does not add or drop entries."""
    root = tmp_path / "host-data"
    _seed_host_data(root, live=True)
    writer = _LiveWriter(root, "host-features.db-wal", "phoenix/phoenix.log")
    try:
        tripwire = HostDataTripwire((root,))
        tripwire.arm()
        writer.write("frames")
    finally:
        writer.close()
    _as_another_process(
        f"import pathlib; pathlib.Path({str(root / 'host-features.db-shm')!r})"
        ".unlink()"
    )
    assert tripwire.disarm() == []

    _as_another_process(
        f"import pathlib; pathlib.Path({str(root / 'left-behind')!r}).touch()"
    )
    assert tripwire.disarm() == [f"{root}: created left-behind (file)"]


def test_a_subprocess_changing_a_file_no_live_writer_holds_is_a_finding(
    tmp_path,
):
    """The exemption is per file, not per directory that has a live database.

    Before this, sidecars beside ``host-features.db`` exempted the whole tree,
    so a test's subprocess could rewrite the Phoenix log unseen.
    """
    root = tmp_path / "host-data"
    _seed_host_data(root, live=True)
    tripwire = HostDataTripwire((root,))
    tripwire.arm()

    log = root / "phoenix" / "phoenix.log"
    before = log.stat().st_mtime_ns
    _as_another_process(
        f"import os, pathlib\npath = pathlib.Path({str(log)!r})\n"
        "path.write_text('boot\\nmore\\n')\n"
        f"os.utime(path, ns=({before + 10**9}, {before + 10**9}))\n"
    )

    findings = tripwire.disarm()
    assert [f for f in findings if "phoenix.log" in f] == [
        (
            f"{root}: modified phoenix/phoenix.log (size 5 -> 10, "
            f"mtime_ns {before} -> {before + 10**9})"
        )
    ]


@needs_lsof
def test_a_live_writers_files_are_identified_by_who_holds_them(tmp_path):
    """Files held open at arm are exempt; their unheld neighbours are not."""
    root = tmp_path / "host-data"
    _seed_host_data(root, live=True)
    (root / "phoenix" / "phoenix.pid").write_text("1\n")
    writer = _LiveWriter(root, "phoenix/phoenix.log")
    try:
        tripwire = HostDataTripwire((root,))
        tripwire.arm()
        writer.write("more")
        _as_another_process(
            f"import pathlib; pathlib.Path({str(root / 'phoenix' / 'phoenix.pid')!r})"
            ".write_text('22\\n')"
        )
    finally:
        writer.close()

    findings = tripwire.disarm()
    assert len(findings) == 1, findings
    assert findings[0].startswith(f"{root}: modified phoenix/phoenix.pid (size 2 -> 3")


def test_a_descendant_still_holding_a_root_file_is_a_finding(tmp_path):
    """Even a live writer's file: this session's own process holds it."""
    root = tmp_path / "host-data"
    _seed_host_data(root, live=True)
    tripwire = HostDataTripwire((root,))
    tripwire.arm()

    leaked = _LiveWriter(root, "host-features.db")
    try:
        findings = tripwire.disarm()
    finally:
        leaked.close()

    assert any(
        "still holds host-features.db open" in finding for finding in findings
    ), findings
    assert tripwire.disarm() == []


def test_the_audit_hook_sees_a_write_the_snapshot_cannot(tmp_path):
    """With a live host, content changes are only attributable in-process."""
    root = tmp_path / "host-data"
    database = _seed_host_data(root, live=True)
    tripwire = HostDataTripwire((root,))
    tripwire.arm()

    connection = sqlite3.connect(database)
    connection.close()
    sqlite3.connect(f"file:{database}?mode=ro", uri=True).close()

    def connects() -> list[str]:
        return [f for f in tripwire.disarm() if "sqlite3.connect" in f]

    test = os.environ["PYTEST_CURRENT_TEST"]
    expected = [
        f"{test}: sqlite3.connect {database}",
        f"{test}: sqlite3.connect file:{database}?mode=ro",
    ]
    assert connects() == expected

    # Disarmed: the hook stays installed but records nothing further.
    sqlite3.connect(database).close()
    assert connects() == expected


_NESTED_SESSION = '''
import pytest

from kestrel_sovereign.host_features.storage import prepare_host_database


def test_isolated_lifespan_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent"))
    prepare_host_database()


@pytest.mark.owns_host_paths
def test_without_isolation(tmp_path, monkeypatch):
    # Exactly what an integration test did before #3286: point the agent data
    # root somewhere private and let the host database "follow" it.
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent"))
    monkeypatch.delenv("KESTREL_HOST_DB_PATH", raising=False)
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    prepare_host_database()
'''


@pytest.mark.timeout(120)
def test_a_session_whose_test_skips_isolation_fails_on_the_tripwire(tmp_path):
    """End to end, against a fake ``HOME`` holding a stopped "production" host."""
    fake_home = tmp_path / "home"
    (fake_home / ".kestrel").mkdir(parents=True)
    production = _seed_host_data(fake_home / ".kestrel" / "host-data", live=False)
    production.chmod(0o600)
    session_file = tmp_path / "test_nested_session.py"
    session_file.write_text(_NESTED_SESSION)

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("KESTREL_", "PYTEST_", "COV_CORE_"))
    }
    env["HOME"] = str(fake_home)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            "--rootdir",
            str(REPO_ROOT),
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            str(session_file),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert "2 passed" in output, output
    assert "Host-data tripwire (#3286)" in output, output
    assert "test_without_isolation" in output
    assert "test_isolated_lifespan_resolution" not in output.split(
        "Host-data tripwire"
    )[1]
    assert "removed host-features.db (file)" in output
    # The hazard the tripwire exists for: the migration moved it.
    assert not production.exists()


# ---------------------------------------------------------------------------
# Managed children: the project dotenv file must not undo the redirect
# ---------------------------------------------------------------------------

_DOTENV = "." + "env"


def _conflicting_project(tmp_path: Path) -> tuple[Path, Path]:
    """A project whose dotenv file names an "operator" host-data root."""
    operator = tmp_path / "operator"
    project = tmp_path / "project"
    (project / "agent_data" / "claw").mkdir(parents=True)
    (project / "agent_data" / "claw" / "kestrel_prime.db").touch()
    (project / _DOTENV).write_text(
        f"KESTREL_HOST_DATA_DIR={operator / 'host-data'}\n"
        f"KESTREL_HOME={operator}\n",
        encoding="utf-8",
    )
    return project, operator


def test_a_managed_child_keeps_the_isolated_host_data_root_over_the_dotenv(
    tmp_path, monkeypatch, host_runtime_isolation_root
):
    """``ProcessManager`` pins the child's host database *and* its previous
    default from ``spawned_agent_env``, where the dotenv file wins."""
    from unittest.mock import MagicMock, patch

    from kestrel_sovereign.host_features.storage import (
        HOST_DB_PATH_ENV,
        HOST_DB_PREVIOUS_DEFAULT_ENV,
    )
    from kestrel_sovereign.multi_agent.config import LocalAgentConfig
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager
    from kestrel_sovereign.paths import HOST_DATA_DIR_ENV

    project, operator = _conflicting_project(tmp_path)
    # What the integration tier looks like: no explicit host database, so the
    # child's host database follows the host-data root.
    monkeypatch.delenv(HOST_DB_PATH_ENV, raising=False)
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)
    monkeypatch.delenv("KESTREL_API_KEY", raising=False)

    with patch("subprocess.Popen", return_value=MagicMock(pid=4242)) as popen:
        ProcessManager(project).start_agent(
            "claw", LocalAgentConfig(data_dir=Path("agent_data/claw"), port=8801)
        )

    env = popen.call_args.kwargs["env"]
    for name in (HOST_DATA_DIR_ENV, HOST_DB_PATH_ENV, HOST_DB_PREVIOUS_DEFAULT_ENV):
        resolved = Path(env[name])
        assert host_runtime_isolation_root in resolved.parents, (name, resolved)
        assert operator not in resolved.parents, (name, resolved)


def test_the_pin_reaches_a_binding_imported_before_installation(
    tmp_path, monkeypatch
):
    """``cli`` and ``multi_agent.config`` import the name at module load, so
    patching ``paths`` alone would leave their bindings unpinned."""
    import types

    from kestrel_sovereign import paths
    from tests.shared.host_runtime_isolation import SpawnedEnvPin

    early = types.ModuleType("_early_spawned_env_importer")
    early.spawned_agent_env = paths.spawned_agent_env
    monkeypatch.setitem(sys.modules, early.__name__, early)
    before = early.spawned_agent_env
    project, operator = _conflicting_project(tmp_path)
    pinned_root = tmp_path / "pinned" / "host-data"

    pin = SpawnedEnvPin()
    pin.install()
    try:
        assert early.spawned_agent_env is paths.spawned_agent_env
        assert early.spawned_agent_env is not before
        with monkeypatch.context() as scoped:
            pin.pin(str(pinned_root), scoped)
            env = early.spawned_agent_env(project)
            assert env[paths.HOST_DATA_DIR_ENV] == str(pinned_root)
            assert operator not in Path(env[paths.HOST_DATA_DIR_ENV]).parents
    finally:
        pin.uninstall()
    assert early.spawned_agent_env is before
    assert paths.spawned_agent_env is before


@pytest.mark.owns_host_paths
def test_production_precedence_is_untouched_outside_isolation(tmp_path, monkeypatch):
    """Under the opt-out the wrapper is a pass-through: the dotenv file wins."""
    from kestrel_sovereign.paths import HOST_DATA_DIR_ENV, spawned_agent_env

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(HOST_DATA_DIR_ENV, str(tmp_path / "exported"))
    project, operator = _conflicting_project(tmp_path)

    assert spawned_agent_env(project)[HOST_DATA_DIR_ENV] == str(
        operator / "host-data"
    )
