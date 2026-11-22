"""The demo seed carries a pinned signer and only its own initial test state."""

import lzma
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
import pytest_asyncio

from kestrel_sovereign.identity.custody_bundle import (
    IdentityCustodyError,
    create_identity_bundle,
)
from kestrel_sovereign.identity.demo_seed import main, restore_demo_seed
from kestrel_sovereign.identity.identity_package import AgentIdentityPackage
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.identity.signing import sign_package, verify_package_signature
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.security.crypto_suite import MLDSA65Suite

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture
async def demo_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    source = tmp_path / "source"
    identity = await create_kestrel_identity_async(
        str(source), agent_name="Public Demo", is_demo=True, is_test_instance=True,
        did_web_domain="agents.custody.test", did_web_slug="public-demo",
    )
    did = identity.agent_did
    snapshot = source / "kestrel_prime.db"
    bundle = create_identity_bundle(source, expected_did=did).encode()
    database_seed = tmp_path / "initial.db.xz"
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))
    return source, snapshot, did, bundle, database_seed


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="container entrypoint requires Bash")
def test_demo_container_restores_mounted_key_and_sets_unique_tenant(tmp_path, demo_seed):
    _source, _snapshot, did, bundle, database_seed = demo_seed
    bundle_path = tmp_path / "identity.json"
    bundle_path.write_bytes(bundle)
    key_file = tmp_path / "data.key"
    key_file.write_text(os.environ["KESTREL_DATA_KEY"])
    soul = tmp_path / "demo-soul.md"
    soul.write_text("Ready for DEV interoperability")
    startup = (REPO_ROOT / "docker/multi_agent_entrypoint.sh").read_text().split(
        "# Bootstrap identity and initialize DB for each agent data dir", 1
    )[0].replace("/app/.venv/bin/python", shlex.quote(sys.executable))
    startup += '\nprintf "A2A_TENANT=%s\\n" "$KESTREL_A2A_EPHEMERAL_TENANT"\n'
    tenants = set()
    for boot in ("boot-one", "boot-two"):
        data = tmp_path / boot
        registry = data.with_suffix(".toml")
        registry.write_text(
            '[agents.demo]\n'
            f'data_dir = "{data / "demo"}"\n'
            'port = 8801\nautostart = true\n'
        )
        registry.with_suffix(".seeds.toml").write_text(
            f'[demo]\ndid = "{did}"\n'
            f'database = "{database_seed}"\nidentity = "{bundle_path}"\n'
            f'soul = "{soul}"\n'
        )
        env = dict(os.environ)
        del env["KESTREL_DATA_KEY"]
        env.update(
            PYTHONPATH=str(REPO_ROOT), KESTREL_ENV="development",
            KESTREL_AGENT_DATA_DIR=str(data), KESTREL_HOST_DB_PATH=str(data / "host/host.db"),
            KESTREL_MULTI_AGENT_CONFIG=str(registry), KESTREL_AGENTS="demo",
            KESTREL_DB_BACKEND="sqlite", KESTREL_DEPLOYMENT_PERSISTENCE="ephemeral_demo",
            KESTREL_A2A_DISCOVERY_AGENT="demo",
            KESTREL_A2A_PUBLISHED_AGENTS="demo", KESTREL_DEMO_SEED_DATA_KEY_FILE=str(key_file),
        )
        if boot == "boot-one":
            missing_mount = dict(env, KESTREL_DATA_KEY=os.environ["KESTREL_DATA_KEY"])
            del missing_mount["KESTREL_DEMO_SEED_DATA_KEY_FILE"]
            rejected = subprocess.run(
                ["bash", "-c", startup], cwd=tmp_path, env=missing_mount,
                capture_output=True, text=True, check=False, timeout=45,
            )
            assert rejected.returncode != 0
            assert "demo data key mount is required" in rejected.stderr
            assert not (data / "demo").exists()
            unlisted = data / "retired"
            unlisted.mkdir(parents=True)
            (unlisted / "kestrel_prime.db").touch()
            overlapping_env = dict(env, KESTREL_HOST_DB_PATH=str(unlisted / "control" / "host.db"))
            rejected = subprocess.run(
                ["bash", "-c", startup], cwd=tmp_path, env=overlapping_env,
                capture_output=True, text=True, check=False, timeout=45,
            )
            assert rejected.returncode != 0
            assert "collides with existing agent directory" in rejected.stderr
            assert not (data / "demo" / "kestrel_prime.db").exists()
        result = subprocess.run(
            ["bash", "-c", startup], cwd=tmp_path, env=env,
            capture_output=True, text=True, check=False, timeout=45,
        )
        assert result.returncode == 0, result.stderr
        tenant, = (line.removeprefix("A2A_TENANT=") for line in result.stdout.splitlines()
                   if line.startswith("A2A_TENANT="))
        assert tenant.startswith("demo-")
        tenants.add(tenant)
        assert load_agent_identity(None, storage_dir=data / "demo").signing_did == did
        assert (data / "demo" / "SOUL.md").read_bytes() == soul.read_bytes()
    assert len(tenants) == 2


def test_demo_seed_reuses_pinned_keys_and_rejects_unrelated_state(tmp_path, monkeypatch, demo_seed):
    source, snapshot, did, bundle, database_seed = demo_seed
    first = tmp_path / "first"
    first.mkdir()
    (first / "SOUL.md").write_text("Container demo profile")
    monkeypatch.chdir(tmp_path)
    second = Path("second")
    for agent_dir in (first, second):
        restore_demo_seed(agent_dir, database_seed, bundle, expected_did=did)
        restored = load_agent_identity(None, storage_dir=agent_dir)
        assert restored.signing_did == did
        assert restored.new_verification_methods == load_agent_identity(
            None, storage_dir=source
        ).new_verification_methods
        assert (agent_dir / snapshot.name).read_bytes() == snapshot.read_bytes()
    assert (first / "SOUL.md").read_text() == "Container demo profile"

    with pytest.raises(IdentityCustodyError, match="existing agent state"):
        restore_demo_seed(first, database_seed, bundle, expected_did=did)
    with closing(sqlite3.connect(snapshot)) as db:
        with db:
            db.execute("UPDATE graph_nodes SET properties = '{}' WHERE node_id = ?", (did,))
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database_seed.write_bytes(lzma.compress(snapshot.read_bytes()))
    unrelated = tmp_path / "unrelated"
    with pytest.raises(IdentityCustodyError, match="does not bind to the pinned test identity"):
        restore_demo_seed(unrelated, database_seed, bundle, expected_did=did)
    assert not unrelated.exists()


def test_entrypoint_refuses_different_configured_discovery_storage(tmp_path, monkeypatch):
    registry = tmp_path / "multi_agent.toml"
    registry.write_text(
        '[agents.demo]\n'
        f'data_dir = "{tmp_path / "other"}"\n'
        'port = 8801\nautostart = true\n'
    )
    seeds = registry.with_suffix(".seeds.toml")
    seeds.write_text(
        '[demo]\ndid = "did:web:agents.custody.test:demo"\n'
        'database = "db.xz"\nidentity = "bundle"\n'
    )
    monkeypatch.setenv("KESTREL_AGENTS", "demo")
    monkeypatch.setenv("KESTREL_A2A_PUBLISHED_AGENTS", "demo")
    monkeypatch.setattr(sys, "argv", [
        "demo_seed", "--config", str(registry), "--seeds", str(seeds),
        "--agent-data-dir", str(tmp_path / "data"),
        "--host-control-dir", str(tmp_path / "host"),
        "--registration", "demo",
    ])
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    with pytest.raises(IdentityCustodyError, match="require SQLite"):
        main()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    with pytest.raises(IdentityCustodyError, match="configured discovery registration"):
        main()


@pytest.mark.asyncio
async def test_mldsa_codec_change_preserves_signed_identity_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    with monkeypatch.context() as legacy:
        legacy.setattr(MLDSA65Suite, "public_key_multicodec", b"\x87\x24")
        identity = await create_kestrel_identity_async(
            str(tmp_path), agent_name="Legacy Demo", is_demo=True, is_test_instance=True,
            did_web_domain="agents.custody.test", did_web_slug="legacy",
        )
        package = sign_package(AgentIdentityPackage(
            did=identity.agent_did, agent_name="Legacy Demo", created_at="2026-01-01",
            constitution_hash="demo-hash", constitution_text="Demo constitution",
        ), storage_dir=tmp_path)
    verified, message = verify_package_signature(package, storage_dir=tmp_path)
    assert verified, message
