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
    exclusion = script.index(
        'case "${HOST_CONTROL_DIR%/}/" in',
        agent_loop,
    )
    subtree = script.index('"${dir%/}/"*) continue ;;', exclusion)
    inception = script.index("create_kestrel_identity", agent_loop)

    assert control_dir < agent_loop < exclusion < subtree < inception


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
