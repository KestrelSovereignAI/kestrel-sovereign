"""The test harness can never reach state outside its temporary roots (#3286).

Every resolver that turns a path-shaped variable or configuration into a
storage path passes its answer through ``paths.guard_storage_path``, which,
when ``KESTREL_TEST_STORAGE_ROOTS`` is set, refuses anything outside the listed
roots. ``tests/conftest.py`` lists only the session's temporary roots, so the
operator's state is refused however it was configured. These tests narrow the
allow-list to a synthetic root, so a sibling directory plays the operator, and
pin the harness half against the real session.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.host_features.storage import (
    DERIVED_HOST_DB_PATH_ENV,
    HOST_DB_PATH_ENV,
    HOST_DB_PREVIOUS_DEFAULT_ENV,
    HOST_FEATURE_DB_FILENAME,
    host_database_path,
    legacy_host_database_path,
    prepare_host_database,
    resolve_host_database_launch_context,
)
from kestrel_sovereign.multi_agent.config import MultiAgentConfig

# Imported here, under the session pins: importing ``server`` resolves the
# project directory, which the tests below deliberately unpin.
from kestrel_sovereign.server import resolve_multi_agent_path
from kestrel_sovereign.storage.async_storage import get_default_agent_data_dir
from tests.shared.host_runtime_isolation import (
    ISOLATED_LAYOUT,
    SESSION_ROOT_ENV,
    install_session_guard,
    pin_runtime_paths,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX home and custody contract"
)

Refused = paths.StoragePathOutsideTestRootsError

#: The project environment file a Kestrel home carries.
ENV_FILENAME = "." + "env"


@pytest.fixture
def allowed(tmp_path, monkeypatch, storage_path_refusals):
    """Allow only ``tmp_path/allowed``; the operator lives beside it."""
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv(paths.STORAGE_ROOTS_ENV, str(root))
    for name in paths.RUNTIME_PATH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    paths.reset_cache()
    return root


@pytest.fixture
def operator_home(tmp_path, allowed, monkeypatch):
    """A synthetic operator home, outside the allowed root."""
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _write_env(path: Path, values: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return path


def _write_registry(path: Path, agents: dict[str, dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"[agents.{name}]\n"
            + "".join(f"{key} = {json.dumps(value)}\n" for key, value in fields.items())
            for name, fields in agents.items()
        ),
        encoding="utf-8",
    )
    return path


def test_refusal_names_the_path_and_its_source(
    operator_home, storage_path_refusals
):
    target = operator_home / ".kestrel" / "host-data"

    with pytest.raises(Refused) as refusal:
        paths.guard_storage_path(target, source="probe")

    assert refusal.value.path == target
    assert refusal.value.source == "probe"
    assert [r.source for r in storage_path_refusals.drain()] == ["probe"]


def test_paths_inside_an_allowed_root_resolve_unchanged(allowed):
    inside = allowed / "agent-data" / "host-data"
    assert paths.guard_storage_path(inside, source="probe") == inside
    assert paths.guard_storage_path(allowed, source="probe") == allowed


def test_a_sibling_sharing_the_root_prefix_is_refused(allowed):
    with pytest.raises(Refused):
        paths.guard_storage_path(
            allowed.parent / f"{allowed.name}-other", source="probe"
        )


def test_no_guard_outside_the_harness(operator_home, monkeypatch):
    monkeypatch.delenv(paths.STORAGE_ROOTS_ENV)
    target = operator_home / ".kestrel" / "host-data"
    assert paths.guard_storage_path(target, source="probe") == target


def test_host_data_dir_fallback_is_refused(operator_home):
    with pytest.raises(Refused, match="host data root"):
        paths.host_data_dir()


def test_every_host_database_branch_is_refused(operator_home, monkeypatch):
    operator_db = operator_home / ".kestrel" / "host-data" / HOST_FEATURE_DB_FILENAME

    with pytest.raises(Refused, match="default host database"):
        host_database_path()

    monkeypatch.setenv("KESTREL_DB_PATH", str(operator_home / "agents"))
    with pytest.raises(Refused, match="KESTREL_DB_PATH"):
        host_database_path()

    monkeypatch.delenv("KESTREL_DB_PATH")
    monkeypatch.setenv(HOST_DB_PATH_ENV, str(operator_db))
    with pytest.raises(Refused, match=HOST_DB_PATH_ENV):
        host_database_path()

    monkeypatch.setenv("KESTREL_HOME", str(operator_home / ".kestrel"))
    with pytest.raises(Refused, match="KESTREL_HOME"):
        legacy_host_database_path()


def test_described_runtime_previous_default_is_refused(operator_home, allowed):
    """A launcher describing a child whose previous default is operator state."""
    with pytest.raises(Refused, match="default host database"):
        resolve_host_database_launch_context(
            env={
                "HOME": str(operator_home),
                "KESTREL_DB_PATH": str(allowed / "agent-data"),
            },
            base_dir=allowed,
            project_root=allowed / "project",
        )

    derived = str(allowed / "agent-data" / "host-data" / HOST_FEATURE_DB_FILENAME)
    with pytest.raises(Refused, match=HOST_DB_PREVIOUS_DEFAULT_ENV):
        resolve_host_database_launch_context(
            env={
                "HOME": str(allowed / "child-home"),
                HOST_DB_PATH_ENV: derived,
                DERIVED_HOST_DB_PATH_ENV: derived,
                HOST_DB_PREVIOUS_DEFAULT_ENV: str(
                    operator_home / ".kestrel" / "host-data" / HOST_FEATURE_DB_FILENAME
                ),
            },
            base_dir=allowed,
            project_root=allowed / "project",
        )


def test_agent_data_root_is_refused(operator_home, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_PATH", str(operator_home / "agents"))
    with pytest.raises(Refused, match="KESTREL_DB_PATH"):
        get_default_agent_data_dir()


def test_identity_data_root_is_refused_through_a_package_caller(
    operator_home, monkeypatch
):
    """Identity and key code resolves the legacy ``AGENT_DATA_DIR`` root."""
    from kestrel_sovereign.inception_service import load_kestrel_identity
    from kestrel_sovereign.security.key_storage import SecureKeyStorage

    operator_keys = operator_home / "agents"
    monkeypatch.setenv(paths.LEGACY_AGENT_DATA_DIR_ENV, str(operator_keys))

    with pytest.raises(Refused, match=paths.LEGACY_AGENT_DATA_DIR_ENV):
        SecureKeyStorage()
    with pytest.raises(Refused, match=paths.LEGACY_AGENT_DATA_DIR_ENV):
        load_kestrel_identity("kestrel_0xabc")
    assert not operator_keys.exists()


def test_identity_data_root_default_is_refused_in_the_checkout(
    operator_home, monkeypatch
):
    """Unset, the legacy root is a cwd-relative ``agent_data``."""
    from kestrel_sovereign.storage import get_default_agent_data_dir as identity_root

    monkeypatch.chdir(operator_home)

    with pytest.raises(Refused, match=paths.LEGACY_AGENT_DATA_DIR_ENV):
        identity_root()


def test_phoenix_override_is_refused(operator_home, monkeypatch):
    from kestrel_sovereign import phoenix_supervisor

    monkeypatch.setenv(paths.PHOENIX_WORKING_DIR_ENV, str(operator_home / "phoenix"))
    with pytest.raises(Refused, match=paths.PHOENIX_WORKING_DIR_ENV):
        phoenix_supervisor.phoenix_working_dir()


def _local_mps_adapter():
    from kestrel_sovereign.features.training.adapters.local_mps_adapter import (
        LocalMPSTrainingAdapter,
    )

    return LocalMPSTrainingAdapter


@pytest.mark.parametrize(
    "configure",
    [
        pytest.param(
            lambda env, outside: env.setenv(
                paths.TRAINING_WORKING_DIR_ENV, str(outside)
            ),
            id="LOCAL_MPS_WORKING_DIR",
        ),
        pytest.param(
            lambda env, outside: env.setenv(paths.DATA_DIR_ENV, str(outside)),
            id="KESTREL_DATA_DIR",
        ),
        # Nothing set: the default is ``~/kestrel-training``.
        pytest.param(lambda env, outside: None, id="home-default"),
    ],
)
def test_local_mps_working_dir_is_refused_before_it_is_created(
    operator_home, monkeypatch, configure
):
    outside = operator_home / "training"
    configure(monkeypatch, outside)

    # Refused at whichever resolver meets the outside path first: the
    # variable's read, or the working directory the adapter settles on.
    with pytest.raises(Refused):
        _local_mps_adapter()()

    assert sorted(operator_home.iterdir()) == []


def test_the_harness_pins_the_local_mps_working_dir(host_runtime_isolation_root):
    adapter = _local_mps_adapter()()
    assert adapter.working_dir == host_runtime_isolation_root / "training"
    assert adapter.cache_dir.is_dir()


@pytest.mark.parametrize(
    ("fields", "source"),
    [
        ({"data_dir": "{operator}/agents/Emma"}, "agent data_dir"),
        (
            {"data_dir": "agents/Emma", "identity_export_dir": "{operator}/exports"},
            "agent identity_export_dir",
        ),
    ],
)
def test_a_registry_naming_outside_state_is_refused_by_its_callers(
    operator_home, allowed, monkeypatch, fields, source
):
    """A ``multi_agent.toml`` names agent roots no variable names.

    Both the custody remediation ``kestrel identity harden-exports`` and Doctor
    share, and the host's registry load, resolve its agents through
    ``LocalAgentConfig``, whose resolvers are guarded.
    """
    from kestrel_sovereign.identity.protected_export import (
        effective_identity_export_roots,
    )
    project = allowed / "project"
    monkeypatch.setenv(
        HOST_DB_PATH_ENV, str(allowed / "host" / HOST_FEATURE_DB_FILENAME)
    )
    registry = _write_registry(
        project / "multi_agent.toml",
        {
            "Emma": {
                "port": 8801,
                **{
                    key: value.format(operator=operator_home)
                    for key, value in fields.items()
                },
            }
        },
    )

    with pytest.raises(Refused, match=source):
        effective_identity_export_roots(project, process_env={})
    with pytest.raises(Refused, match=source):
        MultiAgentConfig.from_file(registry)
    assert not (operator_home / "agents").exists()


def test_an_external_registry_naming_an_outside_data_dir_is_refused(
    operator_home, allowed, monkeypatch
):
    """r4: ``KESTREL_MULTI_AGENT_CONFIG`` names a registry outside the project.

    Its relative ``data_dir`` resolves against the server's runtime base, not
    the registry's directory. Either way it lies outside the test's roots, and
    nothing had to know where it pointed for the guard to refuse it.
    """
    runtime_base = allowed / "project"
    runtime_base.mkdir()
    monkeypatch.setenv(
        HOST_DB_PATH_ENV, str(allowed / "host" / HOST_FEATURE_DB_FILENAME)
    )
    registry = _write_registry(
        allowed / "fleet" / "fleet.toml",
        {"Emma": {"port": 8801, "data_dir": "../../operator-home/agents/Emma"}},
    )
    monkeypatch.setenv(paths.MULTI_AGENT_CONFIG_ENV, str(registry))

    selected = resolve_multi_agent_path(os.environ)
    assert selected == registry
    with pytest.raises(Refused, match="agent data_dir"):
        MultiAgentConfig.load(
            str(selected),
            runtime_env=os.environ,
            runtime_base=runtime_base,
        )

    outside = _write_registry(
        operator_home / "fleet.toml", {"Emma": {"port": 8801, "data_dir": "x"}}
    )
    with pytest.raises(Refused, match=paths.MULTI_AGENT_CONFIG_ENV):
        resolve_multi_agent_path({paths.MULTI_AGENT_CONFIG_ENV: str(outside)})


def test_multi_agent_startup_cannot_fall_back_to_a_registry_outside_the_roots(
    operator_home, allowed, monkeypatch
):
    """The pinned registry is missing, so startup falls back to the cwd.

    With ``KESTREL_MULTI_AGENT=1`` the server enters multi-agent mode even when
    the selected registry does not exist, and then asks ``load`` for its
    default: ``multi_agent.toml`` in the current directory. Run from an
    operator checkout, that names the operator's agents; without a registry
    there, it scans the checkout's ``agent_data``. Both are refused before
    anything is read. The call below is the server's own, argument for
    argument.
    """
    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv(
        HOST_DB_PATH_ENV, str(allowed / "host" / HOST_FEATURE_DB_FILENAME)
    )
    pinned = allowed / "multi_agent.toml"
    monkeypatch.setenv(paths.MULTI_AGENT_CONFIG_ENV, str(pinned))
    checkout = operator_home / "checkout"
    _write_registry(
        checkout / "multi_agent.toml",
        {"Emma": {"port": 8801, "data_dir": str(allowed / "agents" / "Emma")}},
    )
    (checkout / "agent_data" / "Emma").mkdir(parents=True)
    monkeypatch.chdir(checkout)

    selected = resolve_multi_agent_path(os.environ)
    assert selected == pinned
    assert not selected.exists()

    def load_as_the_server_does() -> MultiAgentConfig:
        return MultiAgentConfig.load(
            str(selected) if selected.exists() else None,
            auto_discover_fallback=True,
            runtime_env=os.environ,
            runtime_base=Path.cwd(),
        )

    with pytest.raises(Refused, match="current directory"):
        load_as_the_server_does()

    # Refused whether or not a registry is there: the default is refused
    # before it is opened, so the checkout's agent_data is never scanned.
    (checkout / "multi_agent.toml").unlink()
    with pytest.raises(Refused, match="current directory"):
        load_as_the_server_does()

    # A missing registry passed explicitly still falls back to scanning the
    # runtime base's agent_data; that scan root is refused too.
    with pytest.raises(Refused, match="auto-discovery root"):
        MultiAgentConfig.load(
            str(selected),
            auto_discover_fallback=True,
            runtime_env=os.environ,
            runtime_base=Path.cwd(),
        )


def test_a_kestrel_home_env_file_naming_an_outside_agent_root_is_refused(
    operator_home, allowed
):
    """r4: a Kestrel home's env file names a ``KESTREL_DB_PATH`` elsewhere.

    The launcher's env-file precedence hands it to the child; the child's
    resolvers refuse it, in-process and in a real spawned process.
    """
    home = allowed / "kestrel-home"
    outside = operator_home / "agents"
    _write_env(home / ENV_FILENAME, {"KESTREL_DB_PATH": str(outside)})

    child = paths.spawned_agent_env(home)
    assert child["KESTREL_DB_PATH"] == str(outside)
    # A launched child runs in its own project home.
    child[paths.HOME_ENV] = str(home)
    with pytest.raises(Refused, match="KESTREL_DB_PATH"):
        resolve_host_database_launch_context(env=child, base_dir=home)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from kestrel_sovereign.host_features.storage import host_database_path\n"
            "host_database_path()\n",
        ],
        env=child,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=home,
    )
    # Loudly: the child's first resolver (at import, here) refuses and exits.
    assert result.returncode != 0
    assert "StoragePathOutsideTestRootsError: KESTREL_DB_PATH" in result.stderr
    assert not outside.exists()


def test_a_project_env_load_of_an_outside_root_is_refused(operator_home, allowed):
    """``load_project_env`` fills a removed variable; the resolver refuses it."""
    home = allowed / "kestrel-home"
    _write_env(home / ENV_FILENAME, {"KESTREL_DB_PATH": str(operator_home / "agents")})

    paths.load_project_env(home)

    with pytest.raises(Refused, match="KESTREL_DB_PATH"):
        get_default_agent_data_dir()


def test_doctor_mirror_refuses_outside_state(operator_home, allowed):
    """Doctor keeps its own copy of the resolver; the copy is guarded too."""
    from kestrel_sovereign.doctor import _sqlite_hold_database_path

    with pytest.raises(Refused, match="Doctor default host database"):
        _sqlite_hold_database_path({"HOME": str(operator_home)}, allowed)
    with pytest.raises(Refused, match=HOST_DB_PATH_ENV):
        _sqlite_hold_database_path(
            {HOST_DB_PATH_ENV: "~/.kestrel/host.db", "HOME": str(operator_home)},
            allowed,
        )


def test_the_reported_migration_cannot_move_operator_state(
    operator_home, allowed, monkeypatch
):
    """#3286's own scenario: ``KESTREL_DB_PATH`` alone, host stopped.

    The operator's default host database has no sidecars and no Hold evidence,
    exactly the state in which ``prepare_host_database`` used to ``os.replace``
    it into the test's agent-data root.
    """
    operator_db = operator_home / ".kestrel" / "host-data" / HOST_FEATURE_DB_FILENAME
    operator_db.parent.mkdir(parents=True, mode=0o700)
    with sqlite3.connect(operator_db) as connection:
        connection.execute("CREATE TABLE operator_state (value TEXT)")
    operator_db.chmod(0o600)
    before = operator_db.read_bytes()
    monkeypatch.setenv("KESTREL_DB_PATH", str(allowed / "agent-data"))

    with pytest.raises(Refused):
        prepare_host_database()

    assert operator_db.read_bytes() == before
    migrated = allowed / "agent-data" / "host-data" / HOST_FEATURE_DB_FILENAME
    assert not migrated.exists()


def test_a_spawned_child_inherits_the_guard(operator_home, allowed):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in paths.RUNTIME_PATH_ENV_NAMES
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from kestrel_sovereign import paths\n"
            "try:\n"
            "    paths.host_data_dir()\n"
            "except paths.StoragePathOutsideTestRootsError as refusal:\n"
            "    print('refused', refusal.path)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=allowed,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        f"refused {operator_home / '.kestrel' / 'host-data'}"
    )


def test_the_session_pre_pin_leaves_nothing_for_a_dotenv_load_to_fill(
    tmp_path, monkeypatch
):
    """``server.py``'s import-time ``load_dotenv(override=False)`` restores nothing.

    Importing ``server`` during collection loads the operator's checkout env
    file. A registered variable the harness had merely *removed* would be
    filled back in with the operator's path; one that is pinned cannot be.
    """
    from dotenv import load_dotenv

    for name in paths.RUNTIME_PATH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KESTREL_OPERATOR_API_KEY", raising=False)
    (tmp_path / "session").mkdir()
    pin_runtime_paths(tmp_path / "session", monkeypatch.setenv, None)
    pinned = {name: os.environ[name] for name in paths.RUNTIME_PATH_ENV_NAMES}
    assert all(tmp_path / "session" in Path(value).parents for value in pinned.values())

    operator = tmp_path / "operator-home" / ".kestrel"
    env_file = _write_env(
        tmp_path / "checkout" / ENV_FILENAME,
        {
            **{name: str(operator / name) for name in paths.RUNTIME_PATH_ENV_NAMES},
            "KESTREL_OPERATOR_API_KEY": "kept",
        },
    )
    load_dotenv(env_file, override=False)

    assert {name: os.environ[name] for name in paths.RUNTIME_PATH_ENV_NAMES} == pinned
    assert os.environ["KESTREL_OPERATOR_API_KEY"] == "kept"


def test_install_session_guard_pins_every_registered_variable(
    tmp_path, monkeypatch
):
    """The real session entry point pins the whole registry, not a subset."""
    for name in (
        *paths.RUNTIME_PATH_ENV_NAMES,
        SESSION_ROOT_ENV,
        paths.STORAGE_ROOTS_ENV,
    ):
        # Recorded so the teardown undoes the guard's direct writes.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)

    session_root = install_session_guard()
    try:
        assert session_root is not None
        allowed_roots = os.environ[paths.STORAGE_ROOTS_ENV].split(os.pathsep)
        assert allowed_roots[0] == os.path.abspath(tempfile.gettempdir())
        assert set(allowed_roots) <= {allowed_roots[0], "/tmp"}
        for name in paths.RUNTIME_PATH_ENV_NAMES:
            assert session_root in Path(os.environ[name]).parents, name
            paths.guard_storage_path(Path(os.environ[name]), source=name)
    finally:
        if session_root is not None:
            shutil.rmtree(session_root, ignore_errors=True)


def test_isolation_sets_every_host_path_below_its_root(host_runtime_isolation_root):
    for name, relative in ISOLATED_LAYOUT.items():
        if relative is None:
            assert name not in os.environ, name
        else:
            assert Path(os.environ[name]) == host_runtime_isolation_root / relative
    assert Path(os.environ["KESTREL_HOME"]).is_dir()


def test_the_session_allows_its_temporary_roots(tmp_path):
    assert paths.guard_storage_path(tmp_path, source="probe") == tmp_path
    root = Path(os.environ[SESSION_ROOT_ENV])
    assert paths.guard_storage_path(root, source="probe") == root


def test_the_session_refuses_the_real_operator_home(storage_path_refusals):
    """Nothing enumerated the operator's home; the allow-list refuses it."""
    pwd = pytest.importorskip("pwd")  # Windows has no ``pwd``.
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    with pytest.raises(Refused):
        paths.guard_storage_path(real_home / ".kestrel" / "host-data", source="probe")


@pytest.mark.owns_host_paths
def test_without_isolation_the_real_home_is_refused(
    monkeypatch, storage_path_refusals
):
    """What any test would hit if the per-test isolation were removed."""
    pwd = pytest.importorskip("pwd")  # Windows has no ``pwd``.
    monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
    for name in paths.RUNTIME_PATH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(Refused):
        paths.host_data_dir()
    with pytest.raises(Refused):
        prepare_host_database()


def test_a_swallowed_refusal_is_still_recorded(operator_home, storage_path_refusals):
    try:
        paths.host_data_dir()
    except Exception:  # noqa: BLE001 - the swallow is what is under test
        pass

    refusals = storage_path_refusals.drain()
    assert [refusal.source for refusal in refusals] == ["host data root"]
