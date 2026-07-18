"""Tests for the host-supervised Phoenix subprocess + reverse proxy (#2570).

No real Phoenix is needed: the subprocess is stubbed and the reverse proxy is
driven through an ``httpx.MockTransport`` upstream.
"""

import errno
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign import phoenix_supervisor as ps


def _mode(path):
    return path.stat().st_mode & 0o777


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubPopen:
    """Minimal stand-in for ``subprocess.Popen`` — no real process."""

    instances: list = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4321
        self._alive = True
        self.signals: list = []
        _StubPopen.instances.append(self)

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def _running_supervisor(tmp_path, *, root_path="/phoenix", transport=None):
    """A supervisor that reports running, wired to a mock upstream if given."""
    sup = ps.PhoenixSupervisor(
        working_dir=tmp_path / "phoenix",
        port=6006,
        grpc_port=4317,
        root_path=root_path,
    )
    sup.process = _StubPopen(["stub"])
    if transport is not None:
        sup._client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return sup


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_defaults(monkeypatch):
    monkeypatch.delenv("KESTREL_PHOENIX_PORT", raising=False)
    monkeypatch.delenv("KESTREL_PHOENIX_GRPC_PORT", raising=False)
    assert ps.phoenix_port() == 6006
    assert ps.phoenix_grpc_port() == 4317
    # OTLP endpoint = the HTTP port (exporters are otlp-proto-http), NOT gRPC.
    assert ps.phoenix_otlp_endpoint() == "http://127.0.0.1:6006"


def test_port_overrides(monkeypatch):
    monkeypatch.setenv("KESTREL_PHOENIX_PORT", "7007")
    monkeypatch.setenv("KESTREL_PHOENIX_GRPC_PORT", "5555")
    assert ps.phoenix_port() == 7007
    assert ps.phoenix_grpc_port() == 5555
    assert ps.phoenix_otlp_endpoint() == "http://127.0.0.1:7007"


def test_enabled_requires_installed(monkeypatch):
    # Not importable in CI → disabled regardless of the flag.
    monkeypatch.setattr(ps, "phoenix_available", lambda: False)
    monkeypatch.delenv("KESTREL_PHOENIX_ENABLED", raising=False)
    assert ps.phoenix_enabled() is False


def test_enabled_when_installed(monkeypatch):
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    monkeypatch.delenv("KESTREL_PHOENIX_ENABLED", raising=False)
    assert ps.phoenix_enabled() is True


def test_opt_out_flag(monkeypatch):
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    assert ps.phoenix_enabled() is False


def test_supervision_suppressed_under_pytest(monkeypatch):
    # Installed + not opted out, but we ARE under pytest → the lifespan must
    # NOT spawn a real Phoenix subprocess (issue #2570: no real Phoenix in CI).
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    assert ps._running_under_pytest() is True
    assert ps.should_supervise_phoenix() is False


def test_supervision_active_outside_pytest(monkeypatch):
    # Simulate a real host process (not under pytest) with Phoenix installed.
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "_running_under_pytest", lambda: False)
    assert ps.should_supervise_phoenix() is True


def test_supervision_off_when_disabled_outside_pytest(monkeypatch):
    # Not under pytest, but Phoenix not enabled → still no supervision.
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: False)
    monkeypatch.setattr(ps, "_running_under_pytest", lambda: False)
    assert ps.should_supervise_phoenix() is False


def test_default_working_dir_is_outside_source_checkout(tmp_path, monkeypatch):
    source = tmp_path / "source-checkout"
    (source / "kestrel_sovereign").mkdir(parents=True)
    (source / "kestrel_sovereign" / "__init__.py").write_text("")
    fake_home = tmp_path / "operator-home"
    fake_home.mkdir()
    monkeypatch.chdir(source)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("KESTREL_HOME", raising=False)
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)

    assert ps.phoenix_working_dir() == (
        fake_home / ".kestrel" / "host-data" / "phoenix"
    ).resolve()
    assert source not in ps.phoenix_working_dir().parents


def test_working_dir_honours_explicit_home_and_override(tmp_path, monkeypatch):
    explicit_home = tmp_path / "explicit-home"
    monkeypatch.setenv("KESTREL_HOME", str(explicit_home))
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)
    assert ps.phoenix_working_dir() == (
        explicit_home / "host-data" / "phoenix"
    ).resolve()

    override = tmp_path / "dedicated-phoenix-volume"
    monkeypatch.setenv(ps.PHOENIX_WORKING_DIR_ENV, str(override))
    assert ps.phoenix_working_dir() == override.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode contract")
def test_explicit_working_dir_override_does_not_restrict_parent_or_migrate(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "phoenix"
    legacy.mkdir(parents=True)
    (legacy / "phoenix.db").write_text("legacy")
    shared_parent = tmp_path / "shared-volume"
    shared_parent.mkdir(mode=0o755)
    override = shared_parent / "phoenix"
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv(ps.PHOENIX_WORKING_DIR_ENV, str(override))

    supervisor = ps.PhoenixSupervisor()
    supervisor.prepare_storage()

    assert supervisor.working_dir == override.resolve()
    assert _mode(override) == 0o700
    assert _mode(shared_parent) == 0o755
    assert legacy.exists()


# ---------------------------------------------------------------------------
# OTLP autowiring (INV-SOLO)
# ---------------------------------------------------------------------------


def test_autowire_sets_endpoint_when_unset(monkeypatch):
    monkeypatch.delenv("KESTREL_PHOENIX_PORT", raising=False)
    env = {}
    result = ps.autowire_otlp_endpoint(env)
    assert result == "http://127.0.0.1:6006"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:6006"


def test_autowire_respects_operator_value():
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"}
    assert ps.autowire_otlp_endpoint(env) is None
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"


def test_autowire_project_sets_when_unset():
    # Default the fleet-scoped Phoenix project when the operator hasn't pinned one
    # (obs#32) — host + spawned agents group under "kestrel-fleet" not "default".
    env = {}
    assert ps.autowire_otel_project(env) == "kestrel-fleet"
    assert env["KESTREL_OTEL_PROJECT"] == "kestrel-fleet"
    assert ps.DEFAULT_OTEL_PROJECT == "kestrel-fleet"


def test_autowire_project_respects_operator_value():
    env = {"KESTREL_OTEL_PROJECT": "my-project"}
    assert ps.autowire_otel_project(env) is None
    assert env["KESTREL_OTEL_PROJECT"] == "my-project"


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------


def test_build_env_root_path(tmp_path):
    sup = ps.PhoenixSupervisor(
        working_dir=tmp_path, port=6006, grpc_port=4317, root_path="/phoenix"
    )
    env = sup.build_env(base_env={})
    assert env["PHOENIX_HOST"] == "127.0.0.1"
    assert env["PHOENIX_PORT"] == "6006"
    assert env["PHOENIX_GRPC_PORT"] == "4317"
    assert env["PHOENIX_WORKING_DIR"] == str(tmp_path)
    assert env["PHOENIX_HOST_ROOT_PATH"] == "/phoenix"
    assert env["PHOENIX_SQL_DATABASE_URL"].startswith("sqlite:///")


def test_build_env_fallback_drops_root_path(tmp_path):
    sup = ps.PhoenixSupervisor(working_dir=tmp_path, root_path="")
    env = sup.build_env(base_env={"PHOENIX_HOST_ROOT_PATH": "/stale"})
    assert "PHOENIX_HOST_ROOT_PATH" not in env


def test_build_env_disables_external_telemetry(tmp_path):
    # The supervised child must have Phoenix's external telemetry pixels
    # (FullStory + Scarf.sh) turned off so the sovereign backend never phones
    # home and the embedded UI stops emitting ERR_BLOCKED_BY_CLIENT noise.
    # PHOENIX_TELEMETRY_ENABLED parses only the literal "true"/"false".
    sup = ps.PhoenixSupervisor(working_dir=tmp_path, root_path="/phoenix")
    env = sup.build_env(base_env={})
    assert env["PHOENIX_TELEMETRY_ENABLED"] == "false"


def test_build_env_telemetry_operator_override(tmp_path):
    # setdefault: an operator who explicitly wants the pixels can opt back in.
    sup = ps.PhoenixSupervisor(working_dir=tmp_path, root_path="/phoenix")
    env = sup.build_env(base_env={"PHOENIX_TELEMETRY_ENABLED": "true"})
    assert env["PHOENIX_TELEMETRY_ENABLED"] == "true"


# ---------------------------------------------------------------------------
# Subprocess lifecycle (stubbed process)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode contract")
def test_prepare_storage_hardens_existing_tree(tmp_path):
    working_dir = tmp_path / "phx"
    nested = working_dir / "nested"
    nested.mkdir(parents=True)
    files = [
        working_dir / "phoenix.log",
        working_dir / "phoenix.db",
        working_dir / "phoenix.db-wal",
        working_dir / "phoenix.db-shm",
        working_dir / "phoenix.pid",
        nested / "trace-cache",
    ]
    for path in files:
        path.write_text("sensitive")
        path.chmod(0o666)
    working_dir.chmod(0o777)
    nested.chmod(0o777)

    ps.PhoenixSupervisor(working_dir=working_dir).prepare_storage()

    assert _mode(working_dir) == 0o700
    assert _mode(nested) == 0o700
    assert all(_mode(path) == 0o600 for path in files)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode contract")
def test_prepare_storage_migrates_legacy_project_store(tmp_path, monkeypatch):
    home = tmp_path / "kestrel-home"
    legacy = home / "phoenix"
    legacy.mkdir(parents=True)
    for name in (
        "phoenix.log",
        "phoenix.db",
        "phoenix.db-wal",
        "phoenix.db-shm",
    ):
        path = legacy / name
        path.write_text(name)
        path.chmod(0o644)
    legacy.chmod(0o755)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)

    sup = ps.PhoenixSupervisor()
    sup.prepare_storage()

    assert not legacy.exists()
    assert sup.working_dir == (home / "host-data" / "phoenix").resolve()
    assert _mode(sup.working_dir.parent) == 0o700
    assert _mode(sup.working_dir) == 0o700
    assert all(
        _mode(sup.working_dir / name) == 0o600
        for name in ("phoenix.log", "phoenix.db", "phoenix.db-wal", "phoenix.db-shm")
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode contract")
def test_prepare_storage_cross_filesystem_migration_uses_private_staging(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "phoenix"
    legacy.mkdir(parents=True)
    (legacy / "phoenix.db").write_text("history")
    (legacy / "phoenix.db").chmod(0o644)
    legacy.chmod(0o755)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)
    real_replace = ps.os.replace

    def _replace_with_one_cross_device_failure(source, destination):
        if Path(source) == legacy:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source, destination)

    monkeypatch.setattr(ps.os, "replace", _replace_with_one_cross_device_failure)
    sup = ps.PhoenixSupervisor()
    sup.prepare_storage()

    assert not legacy.exists()
    assert (sup.working_dir / "phoenix.db").read_text() == "history"
    assert _mode(sup.working_dir) == 0o700
    assert _mode(sup.working_dir / "phoenix.db") == 0o600
    assert not list(sup.working_dir.parent.glob(".phoenix-migrate-*"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode contract")
def test_prepare_storage_fails_closed_when_legacy_and_destination_both_exist(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "phoenix"
    destination = home / "host-data" / "phoenix"
    legacy.mkdir(parents=True)
    destination.mkdir(parents=True)
    (legacy / "phoenix.db").write_text("legacy")
    (destination / "phoenix.db").write_text("new")
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)

    with pytest.raises(ps.PhoenixStorageError, match="both legacy Phoenix storage"):
        ps.PhoenixSupervisor().prepare_storage()

    # Disclosure is contained before the ambiguous migration is rejected.
    assert _mode(legacy) == 0o700
    assert _mode(legacy / "phoenix.db") == 0o600
    assert _mode(destination) == 0o700
    assert _mode(destination / "phoenix.db") == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PID contract")
def test_prepare_storage_contains_but_does_not_move_live_legacy_store(
    tmp_path, monkeypatch,
):
    home = tmp_path / "kestrel-home"
    legacy = home / "phoenix"
    legacy.mkdir(parents=True)
    (legacy / "phoenix.db").write_text("live")
    (legacy / "phoenix.pid").write_text(str(os.getpid()))
    legacy.chmod(0o755)
    (legacy / "phoenix.db").chmod(0o644)
    (legacy / "phoenix.pid").chmod(0o644)
    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.delenv(ps.PHOENIX_WORKING_DIR_ENV, raising=False)

    with pytest.raises(ps.PhoenixStorageError, match="belongs to live PID"):
        ps.PhoenixSupervisor().prepare_storage()

    assert legacy.exists()
    assert _mode(legacy) == 0o700
    assert _mode(legacy / "phoenix.db") == 0o600
    assert _mode(legacy / "phoenix.pid") == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink contract")
def test_prepare_storage_rejects_symlinked_content(tmp_path):
    working_dir = tmp_path / "phx"
    outside = tmp_path / "outside"
    working_dir.mkdir()
    outside.write_text("do not follow")
    (working_dir / "phoenix.db").symlink_to(outside)

    with pytest.raises(ps.PhoenixStorageError, match="symbolic link"):
        ps.PhoenixSupervisor(working_dir=working_dir).prepare_storage()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hard-link contract")
def test_prepare_storage_rejects_multiply_linked_content(tmp_path):
    working_dir = tmp_path / "phx"
    working_dir.mkdir()
    outside = tmp_path / "outside-copy"
    (working_dir / "phoenix.db").write_text("shared trace data")
    os.link(working_dir / "phoenix.db", outside)

    with pytest.raises(ps.PhoenixStorageError, match="hard links"):
        ps.PhoenixSupervisor(working_dir=working_dir).prepare_storage()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink contract")
def test_prepare_storage_rejects_symlinked_working_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    working_dir = tmp_path / "phx"
    working_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ps.PhoenixStorageError, match="real directory"):
        ps.PhoenixSupervisor(working_dir=working_dir).prepare_storage()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX umask contract")
def test_child_created_database_sidecars_are_private_under_permissive_umask(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: False)
    working_dir = tmp_path / "phx"
    sup = ps.PhoenixSupervisor(working_dir=working_dir, root_path="/phoenix")
    child_code = (
        "import os; from pathlib import Path; "
        "p=Path(os.environ['PHOENIX_WORKING_DIR']); "
        "[(p / n).write_text('sensitive') for n in "
        "('phoenix.db','phoenix.db-wal','phoenix.db-shm')]"
    )
    monkeypatch.setattr(
        sup, "build_command", lambda: [sys.executable, "-c", child_code]
    )

    previous_umask = os.umask(0)
    try:
        assert sup.start(wait_for_health=False) is True
    finally:
        os.umask(previous_umask)
    sup.process.wait(timeout=10)

    assert _mode(working_dir) == 0o700
    for name in (
        "phoenix.log",
        "phoenix.pid",
        "phoenix.db",
        "phoenix.db-wal",
        "phoenix.db-shm",
    ):
        assert _mode(working_dir / name) == 0o600
    sup.stop()


def test_start_disables_tracing_when_storage_cannot_be_secured(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(
        ps,
        "_secure_storage_tree",
        lambda _path: (_ for _ in ()).throw(ps.PhoenixStorageError("denied")),
    )
    spawned = []
    monkeypatch.setattr(ps.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx")
    assert sup.start(wait_for_health=False) is False
    assert spawned == []
    assert sup.process is None


def test_start_reaps_child_when_private_pid_file_cannot_be_written(
    tmp_path, monkeypatch,
):
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: False)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    monkeypatch.setattr(
        ps,
        "_open_private_file",
        lambda *a, **k: (_ for _ in ()).throw(ps.PhoenixStorageError("denied"))
        if Path(a[0]).name == "phoenix.pid"
        else os.open(a[0], a[1], 0o600),
    )

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", root_path="/phoenix")
    assert sup.start(wait_for_health=False) is False

    child = _StubPopen.instances[-1]
    assert child.poll() is not None
    assert sup.process is None


def test_start_and_stop_lifecycle(tmp_path, monkeypatch):
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: True)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    # No Phoenix already serving the port → adopt-or-reap falls through to spawn.
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: False)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    # A pinned root_path was NOT given → resolved from supports_host_root_path.
    sup._root_path_override = None

    assert sup.start(wait_for_health=False) is True
    assert sup.running is True
    assert sup.root_path == "/phoenix"
    # Launched with the phoenix serve argv.
    assert _StubPopen.instances[-1].argv == sup.build_command()
    # PID file tracked on disk.
    assert sup.pid_file.exists()
    assert sup.pid_file.read_text() == "4321"

    sup.stop()
    assert sup.running is False
    assert not sup.pid_file.exists()


def test_start_falls_back_when_root_path_unsupported(tmp_path, monkeypatch):
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: False)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: False)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx")
    sup._root_path_override = None
    assert sup.start(wait_for_health=False) is True
    assert sup.root_path == ""


def test_start_skips_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: False)
    called = {"popen": False}

    def _boom(*a, **k):
        called["popen"] = True

    monkeypatch.setattr(ps.subprocess, "Popen", _boom)

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx")
    assert sup.start(wait_for_health=False) is False
    assert sup.running is False
    assert called["popen"] is False


# ---------------------------------------------------------------------------
# Adopt-or-reap orphans across restarts (#2589)
# ---------------------------------------------------------------------------


def test_adopt_existing_healthy_phoenix(tmp_path, monkeypatch):
    """A hard-killed host leaks a still-serving child; the successor ADOPTS it
    instead of spawning a second child into the held port (#2589)."""
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: True)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    # A healthy Phoenix already holds the port, owned by PID 9999.
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: True)
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pids_listening_on_port", lambda self: [9999]
    )
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pid_alive", staticmethod(lambda pid: bool(pid))
    )

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    sup._root_path_override = None

    assert sup.start(wait_for_health=False) is True
    # Adopted, not spawned — no subprocess launched.
    assert _StubPopen.instances == []
    assert sup._adopted_pid == 9999
    assert sup.process is None
    # Tracked as running via the adopted PID (liveness), reachable via the port.
    assert sup.running is True
    # Pidfile now points at the adopted process so the NEXT successor can reap it.
    assert sup.pid_file.read_text() == "9999"


def test_adopt_reaps_split_brain_zombie(tmp_path, monkeypatch):
    """Port served by the real listener (100) while the pidfile names a different
    live PID (200) — the zombie that lost the race. Adopt 100, reap 200 (#2589)."""
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: True)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: True)
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pids_listening_on_port", lambda self: [100]
    )
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pid_alive", staticmethod(lambda pid: pid in (100, 200))
    )
    reaped: list = []
    monkeypatch.setattr(
        ps.PhoenixSupervisor,
        "_terminate_pid",
        lambda self, pid, **k: reaped.append(pid),
    )

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    sup._root_path_override = None
    sup.working_dir.mkdir(parents=True, exist_ok=True)
    sup.pid_file.write_text("200")

    assert sup.start(wait_for_health=False) is True
    assert sup._adopted_pid == 100
    assert reaped == [200]  # the zombie was reaped
    assert _StubPopen.instances == []  # no second child spawned
    assert sup.pid_file.read_text() == "100"


def test_reap_stale_child_then_spawn(tmp_path, monkeypatch):
    """Port not serving, but the pidfile names a live (hung/bound-failed) child.
    Reap it so a fresh child can bind, then spawn (#2589)."""
    _StubPopen.instances.clear()
    monkeypatch.setattr(ps, "phoenix_enabled", lambda: True)
    monkeypatch.setattr(ps, "supports_host_root_path", lambda: True)
    monkeypatch.setattr(ps.subprocess, "Popen", _StubPopen)
    monkeypatch.setattr(ps.PhoenixSupervisor, "is_healthy", lambda self, **k: False)
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pid_alive", staticmethod(lambda pid: pid == 5150)
    )
    reaped: list = []
    monkeypatch.setattr(
        ps.PhoenixSupervisor,
        "_terminate_pid",
        lambda self, pid, **k: reaped.append(pid),
    )

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    sup._root_path_override = None
    sup.working_dir.mkdir(parents=True, exist_ok=True)
    sup.pid_file.write_text("5150")  # leaked child from a hard-killed host

    assert sup.start(wait_for_health=False) is True
    assert reaped == [5150]  # stale child reaped
    assert len(_StubPopen.instances) == 1  # then a fresh child spawned
    assert sup._adopted_pid is None
    assert sup.pid_file.read_text() == "4321"  # fresh child's PID


def test_running_and_stop_track_adopted_pid(tmp_path, monkeypatch):
    """``running`` reflects an adopted PID with no Popen handle, and ``stop``
    reaps it so a graceful shutdown leaves no orphan (#2589)."""
    monkeypatch.setattr(
        ps.PhoenixSupervisor, "_pid_alive", staticmethod(lambda pid: pid == 777)
    )
    reaped: list = []
    monkeypatch.setattr(
        ps.PhoenixSupervisor,
        "_terminate_pid",
        lambda self, pid, **k: reaped.append(pid),
    )

    sup = ps.PhoenixSupervisor(working_dir=tmp_path / "phx", port=6006)
    sup.working_dir.mkdir(parents=True, exist_ok=True)
    sup._adopted_pid = 777
    sup.pid_file.write_text("777")

    assert sup.process is None
    assert sup.running is True  # tracked purely via the adopted PID

    sup.stop()
    assert reaped == [777]
    assert sup._adopted_pid is None
    assert not sup.pid_file.exists()


# ---------------------------------------------------------------------------
# Reachability = health (#2589)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_reachable_true(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )
    assert await sup.is_reachable() is True
    await sup.aclose()


@pytest.mark.asyncio
async def test_is_reachable_false_when_down(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("phoenix still starting")

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )
    assert await sup.is_reachable() is False
    await sup.aclose()


# ---------------------------------------------------------------------------
# upstream_url
# ---------------------------------------------------------------------------


def test_upstream_url_strips_prefix_even_with_root_path(tmp_path):
    # ASGI root_path is for URL *generation* only — Phoenix's routing matches
    # the unprefixed path, so the proxy always forwards stripped.
    sup = _running_supervisor(tmp_path, root_path="/phoenix")
    assert (
        sup.upstream_url("v1/traces", "a=1")
        == "http://127.0.0.1:6006/v1/traces?a=1"
    )
    assert sup.upstream_url("") == "http://127.0.0.1:6006/"


def test_upstream_url_fallback(tmp_path):
    sup = _running_supervisor(tmp_path, root_path="")
    assert sup.upstream_url("foo") == "http://127.0.0.1:6006/foo"


# ---------------------------------------------------------------------------
# Embed cookie
# ---------------------------------------------------------------------------


def test_embed_cookie_roundtrip():
    secret = "super-secret"
    from starlette.requests import Request

    token = ps.mint_embed_token(secret, identity="jason")
    scope = {
        "type": "http",
        "headers": [(b"cookie", f"{ps.EMBED_COOKIE_NAME}={token}".encode())],
    }
    req = Request(scope)
    assert ps.verify_embed_cookie(req, secret) is True


def test_embed_cookie_rejects_wrong_secret():
    from starlette.requests import Request

    token = ps.mint_embed_token("secret-a")
    scope = {
        "type": "http",
        "headers": [(b"cookie", f"{ps.EMBED_COOKIE_NAME}={token}".encode())],
    }
    req = Request(scope)
    assert ps.verify_embed_cookie(req, "secret-b") is False


def test_embed_cookie_absent():
    from starlette.requests import Request

    req = Request({"type": "http", "headers": []})
    assert ps.verify_embed_cookie(req, "s") is False


# ---------------------------------------------------------------------------
# Reverse proxy unit (direct call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_503_when_disabled():
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/phoenix/",
        "query_string": b"",
        "headers": [],
    }
    req = Request(scope, receive)
    resp = await ps.proxy_to_phoenix(req, None, "")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_proxy_streams_upstream(tmp_path):
    from starlette.requests import Request

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)

        async def body():
            yield b"<html>phoenix</html>"

        return httpx.Response(200, content=body())

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/phoenix/",
        "query_string": b"",
        "headers": [],
    }
    req = Request(scope, receive)
    resp = await ps.proxy_to_phoenix(req, sup, "")

    body = b""
    async for chunk in resp.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()
    assert resp.status_code == 200
    assert b"phoenix" in body
    assert captured["url"] == "http://127.0.0.1:6006/"
    await sup.aclose()


# ---------------------------------------------------------------------------
# Route-level: auth + 503 + embed cookie flow (through the real ASGI stack)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def _client_with_state(phoenix_state, monkeypatch):
    """Return the shared host ``app`` wired for a route test.

    Uses ``monkeypatch`` so every mutation (the no-op lifespan, the app.state
    fields) is restored after the test. ``app`` is a process-wide singleton
    (``from server import app``); permanently swapping its lifespan_context here
    would leak a no-op lifespan into every later ``TestClient(app)`` in the same
    pytest session.
    """
    from server import app

    monkeypatch.setattr(app.router, "lifespan_context", _noop_lifespan)
    monkeypatch.setattr(app.state, "agent", None, raising=False)
    monkeypatch.setattr(app.state, "agent_manager", None, raising=False)
    monkeypatch.setattr(app.state, "startup_error", None, raising=False)
    monkeypatch.setattr(app.state, "phoenix", phoenix_state, raising=False)
    return app


def test_phoenix_route_requires_auth(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        # No auth at all → 401 (never leaks whether Phoenix is up).
        r = client.get("/phoenix/")
    assert r.status_code == 401


def test_phoenix_route_503_when_disabled(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/phoenix/", headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Phoenix not enabled"


def test_mint_endpoint_requires_auth(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    app = _client_with_state(None, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/api/host/phoenix/session")
    assert r.status_code == 401


def test_real_lifespan_never_spawns_phoenix_under_pytest(monkeypatch):
    """End-to-end: the *real* host lifespan must not spawn a Phoenix subprocess
    while running under pytest, even when arize-phoenix looks installed.

    This is the guard that keeps ``pytest -q`` from launching (and then tearing
    down, SIGTERM→SIGKILL) a heavyweight Phoenix per ``TestClient`` startup.
    """
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    # Make Phoenix look installed so only the pytest guard can stop a spawn.
    monkeypatch.setattr(ps, "phoenix_available", lambda: True)
    assert ps.should_supervise_phoenix() is False  # suppressed: we're in pytest

    # Trip on ANY subprocess spawn from the supervisor.
    def _tripwire(*_a, **_k):
        raise AssertionError("Phoenix subprocess spawned under pytest")

    monkeypatch.setattr(ps.subprocess, "Popen", _tripwire)

    from server import app

    # Use the REAL lifespan (do not swap in the no-op) so the guarded startup
    # path actually runs; just neutralise app.state afterwards via monkeypatch.
    with TestClient(app) as client:
        r = client.get("/phoenix/", headers={"X-API-Key": "test-key-123"})
    # Phoenix was never supervised → the proxy reports it disabled.
    assert getattr(app.state, "phoenix", None) is None
    assert r.status_code == 503


def test_real_lifespan_does_not_autowire_otlp_before_storage_preflight(
    monkeypatch,
):
    """A custody failure disables tracing before agents inherit an endpoint."""
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(ps, "should_supervise_phoenix", lambda: True)

    def _deny_custody(_self):
        raise ps.PhoenixStorageError("test custody denial")

    monkeypatch.setattr(ps.PhoenixSupervisor, "prepare_storage", _deny_custody)
    from server import app

    with TestClient(app) as client:
        response = client.get(
            "/phoenix/", headers={"X-API-Key": "test-key-123"}
        )

    assert response.status_code == 503
    assert getattr(app.state, "phoenix", None) is None
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ


def test_mint_503_when_phoenix_unreachable(tmp_path, monkeypatch):
    """Health is reachability, not child liveness (#2589): a tracked supervisor
    whose Phoenix is not answering the port must mint 503, not 200 — this is the
    zombie split-brain the mint used to gate on."""
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("phoenix still starting")

    # Supervisor is tracked and its process reports alive (running is True), but
    # the port is unreachable — the old `not supervisor.running` gate would mint.
    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )
    assert sup.running is True
    app = _client_with_state(sup, monkeypatch)

    with TestClient(app) as client:
        r = client.post(
            "/api/host/phoenix/session", headers={"X-API-Key": "test-key-123"}
        )
    assert r.status_code == 503


def test_mint_then_proxy_with_embed_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "test-key-123")

    def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            yield b"<html>phoenix-ui</html>"

        return httpx.Response(200, content=body())

    sup = _running_supervisor(
        tmp_path, root_path="/phoenix", transport=httpx.MockTransport(handler)
    )
    app = _client_with_state(sup, monkeypatch)

    with TestClient(app) as client:
        # Mint requires standard auth (API key here).
        minted = client.post(
            "/api/host/phoenix/session", headers={"X-API-Key": "test-key-123"}
        )
        assert minted.status_code == 200
        assert ps.EMBED_COOKIE_NAME in minted.cookies

        # Now the browser carries the embed cookie; NO api key header on the
        # iframe request. It must still be authorized (and proxied).
        client.headers.pop("X-API-Key", None)
        r = client.get("/phoenix/")
        assert r.status_code == 200
        assert b"phoenix-ui" in r.content
