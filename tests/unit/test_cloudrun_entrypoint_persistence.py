"""Container bootstrap persistence drift guards (#2472)."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_single_agent_entrypoint_never_incepts_durable_identity():
    script = (REPO_ROOT / "docker/cloudrun_entrypoint.sh").read_text()
    durable = script.split("durable_sovereign)", 1)[1].split(
        ";;", 1
    )[0]

    assert "custody_bundle" in durable
    assert "KESTREL_EXPECTED_DID" in durable
    assert "KESTREL_DATABASE_URL" in durable
    assert "KESTREL_HOLD_EVIDENCE_DATABASE_URL" in durable
    assert "unset KESTREL_IDENTITY_BUNDLE" in durable
    assert "create_kestrel_identity" not in durable


def test_multi_agent_entrypoint_refuses_durable_cloudrun():
    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()

    refusal = script.index('if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]')
    inception = script.index("create_kestrel_identity")
    assert refusal < inception
    assert "per agent; refusing local inception" in script
    assert "is_test_instance=True" in script
    assert "is_demo=True" in script


def test_multi_agent_entrypoint_never_bootstraps_host_control_directory():
    """A persistent Hold directory is host state, not an agent candidate."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()

    control_dir = script.index('HOST_CONTROL_DIR="$(dirname -- ')
    agent_loop = script.index('for dir in "$AGENT_DATA_DIR"/*/')
    resolver = script.index(
        'resolved_dir="$(canonicalize_path "$dir")"',
        agent_loop,
    )
    exclusion = script.index(
        'if paths_overlap "$HOST_CONTROL_DIR" "$resolved_dir"; then',
        resolver,
    )
    collision = script.index('[ -f "$dir/kestrel_prime.db" ]', exclusion)
    refusal = script.index("collides with existing agent directory", collision)
    skip = script.index("continue", refusal)
    inception = script.index("create_kestrel_identity", agent_loop)

    assert (
        control_dir
        < agent_loop
        < resolver
        < exclusion
        < collision
        < refusal
        < skip
        < inception
    )


def test_multi_agent_entrypoint_canonicalizes_relative_host_control_directory(
    tmp_path,
):
    """Relative and absolute spellings identify the same excluded directory."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()
    setup = script.split(
        'if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]',
        1,
    )[0]
    setup = setup.replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    agent_data_dir = tmp_path / "agent_data"
    env = os.environ.copy()
    env.update(
        {
            "KESTREL_AGENT_DATA_DIR": str(agent_data_dir),
            "KESTREL_HOST_DB_PATH": "agent_data/host-data/host-features.db",
        }
    )
    probe = setup + r'''
dir="$AGENT_DATA_DIR/host-data/"
case "${HOST_CONTROL_DIR%/}/" in
    "${dir%/}/"*) decision=excluded ;;
    *) decision=admitted ;;
esac
printf '%s\0%s\0%s\0' "$AGENT_DATA_DIR" "$HOST_CONTROL_DIR" "$decision"
'''

    result = subprocess.run(
        ["bash", "-c", probe],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    agent_root, control_root, decision, _ = result.stdout.split(b"\0")

    assert agent_root.decode() == str(agent_data_dir.resolve())
    assert control_root.decode() == str((agent_data_dir / "host-data").resolve())
    assert decision == b"excluded"


def test_multi_agent_entrypoint_marks_only_fallback_host_path_as_derived(
    tmp_path,
):
    """Launcher defaults migrate prior state; operator overrides remain explicit."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()
    setup = script.split(
        'if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]',
        1,
    )[0]
    setup = setup.replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    agent_data_dir = tmp_path / "agent_data"

    def resolved_paths(host_path=None):
        env = os.environ.copy()
        env["KESTREL_AGENT_DATA_DIR"] = str(agent_data_dir)
        if host_path is None:
            env.pop("KESTREL_HOST_DB_PATH", None)
        else:
            env["KESTREL_HOST_DB_PATH"] = str(host_path)
        env["KESTREL_DERIVED_HOST_DB_PATH"] = "stale-inherited-marker"
        probe = setup + r'''
printf '%s\0%s\0' "$KESTREL_HOST_DB_PATH" "${KESTREL_DERIVED_HOST_DB_PATH:-}"
'''
        result = subprocess.run(
            ["bash", "-c", probe],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
        )
        selected, derived, _ = result.stdout.split(b"\0")
        return selected.decode(), derived.decode()

    fallback, fallback_marker = resolved_paths()
    assert fallback == str(
        (agent_data_dir / "host-data" / "host-features.db").resolve()
    )
    assert fallback_marker == fallback

    explicit = tmp_path / "operator-host" / "host.db"
    selected, explicit_marker = resolved_paths(explicit)
    assert selected == str(explicit)
    assert explicit_marker == ""


def test_multi_agent_entrypoint_refuses_existing_agent_at_host_control_root(
    tmp_path,
):
    """An existing config cannot bypass the upgrade collision refusal."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()
    setup = script.split(
        'if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]',
        1,
    )[0]
    agent_loop = script.split(
        "# Bootstrap identity and initialize DB for each agent data dir",
        1,
    )[1].split('echo "Starting Kestrel MultiAgent Host', 1)[0]
    probe = setup + agent_loop
    probe = probe.replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    agent_data_dir = tmp_path / "agent_data"
    host_named_agent = agent_data_dir / "host-data"
    host_named_agent.mkdir(parents=True)
    (host_named_agent / "kestrel_prime.db").touch()
    env = os.environ.copy()
    env.update(
        {
            "KESTREL_AGENT_DATA_DIR": str(agent_data_dir),
            "KESTREL_HOST_DB_PATH": str(host_named_agent / "host-features.db"),
        }
    )

    result = subprocess.run(
        ["bash", "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "collides with existing agent directory" in result.stderr
    assert "set KESTREL_HOST_DB_PATH outside agent_data" in result.stderr


def test_multi_agent_entrypoint_refuses_requested_host_control_collision(
    tmp_path,
):
    """KESTREL_AGENTS cannot request a name reserved by host custody."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()
    probe = script.split("# Generate multi_agent.toml", 1)[0]
    probe = probe.replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    agent_data_dir = tmp_path / "agent_data"
    env = os.environ.copy()
    env.update(
        {
            "KESTREL_AGENT_DATA_DIR": str(agent_data_dir),
            "KESTREL_HOST_DB_PATH": str(
                agent_data_dir / "host-data" / "host-features.db"
            ),
            "KESTREL_MULTI_AGENT_CONFIG": str(tmp_path / "multi_agent.toml"),
            "KESTREL_DEPLOYMENT_PERSISTENCE": "ephemeral_demo",
            "KESTREL_ENV": "development",
            "KESTREL_AGENTS": "kite,host-data",
        }
    )

    result = subprocess.run(
        ["bash", "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "requested agent 'host-data' collides with host control path" in (
        result.stderr
    )


def test_multi_agent_entrypoint_refuses_symlink_alias_to_external_host_custody(
    tmp_path,
):
    """The bootstrap loop compares resolved paths before touching an agent."""

    script = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text()
    setup = script.split(
        'if [ "$PERSISTENCE_MODE" = "durable_sovereign" ]',
        1,
    )[0]
    agent_loop = script.split(
        "# Bootstrap identity and initialize DB for each agent data dir",
        1,
    )[1].split('echo "Starting Kestrel MultiAgent Host', 1)[0]
    probe = setup + agent_loop
    probe = probe.replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    agent_data_dir = tmp_path / "agent_data"
    host_dir = tmp_path / "external-host-data"
    agent_data_dir.mkdir()
    host_dir.mkdir()
    (host_dir / "kestrel_existing.json").write_text("{}", encoding="utf-8")
    (host_dir / "kestrel_prime.db").touch()
    (agent_data_dir / "custody-alias").symlink_to(
        host_dir,
        target_is_directory=True,
    )
    env = os.environ.copy()
    env.update(
        {
            "KESTREL_AGENT_DATA_DIR": str(agent_data_dir),
            "KESTREL_HOST_DB_PATH": str(host_dir / "host-features.db"),
        }
    )

    result = subprocess.run(
        ["bash", "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "collides with existing agent directory" in result.stderr
