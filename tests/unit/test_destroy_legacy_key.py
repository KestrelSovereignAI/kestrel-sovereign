"""
Tests for ``scripts/quantum_destroy_legacy_key.py``.

The script is paranoid by design — destruction is irreversible.
Tests exercise:
- Default dry-run preserves all files
- --confirm without env var rejects
- --confirm with env var deletes legacy keys, preserves DID doc +
  hybrid keys + succession statement
- Rollback-window gate refuses fresh successions
- Missing hybrid keys gate refuses (don't strand the agent)
- Missing succession refuses
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.inception_service import (
    public_key_to_ethereum_address,
)
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite, SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "quantum_destroy_legacy_key.py"
)


@pytest.fixture
def kestrel_data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)


@pytest.fixture
def post_ceremony_dir(tmp_path, kestrel_data_key, monkeypatch):
    """Set up a complete post-ceremony agent dir with legacy + hybrid
    + succession material on disk. Returns the dir + slug."""
    storage = SecureKeyStorage(storage_dir=tmp_path)
    secp = Secp256k1Suite()
    legacy_kp = secp.generate_keypair()
    address = public_key_to_ethereum_address(legacy_kp.public_key)
    legacy_did = f"did:pkh:eip155:1:{address}"
    key_id = f"kestrel_{address}"
    storage.save_private_key(legacy_kp.private_key, key_id)

    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    pub_hex = legacy_kp.public_key.public_bytes(
        encoding=Encoding.X962, format=PublicFormat.UncompressedPoint,
    ).hex()
    (tmp_path / f"{key_id}.json").write_text(json.dumps({
        "@context": "https://w3id.org/did/v1",
        "id": legacy_did,
        "publicKey": [{
            "id": f"{legacy_did}#keys-1",
            "type": "EcdsaSecp256k1VerificationKey2019",
            "controller": legacy_did,
            "publicKeyHex": pub_hex,
        }],
    }))

    legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_kp.public_key)])
    archival_kp = SLHDSASHA2128sSuite().generate_keypair()
    # Set effective_from to 30 days ago so the rollback window has passed
    eff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_kp,
        predecessor_kid=legacy_vms[0]["id"].rsplit("#", 1)[-1],
        predecessor_verification_methods=legacy_vms,
        new_did_domain="agents.test.example",
        new_did_slug="testbot",
        reason="destroy legacy test",
        effective_from=eff,
        archival_keypair=archival_kp,
    )
    new_kp = result.new_identity.keypair
    storage.save_private_key(new_kp.classical.private_key, "testbot_ed25519")
    storage.save_secret_bytes(new_kp.pq.private_key, "testbot_mldsa65")
    storage.save_secret_bytes(archival_kp.private_key, "testbot_archival_slhdsa")
    storage.save_secret_bytes(archival_kp.public_key, "testbot_archival_slhdsa_pub")
    successions_dir = tmp_path / "successions"
    successions_dir.mkdir()
    (successions_dir / "testbot.json").write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2)
    )
    return tmp_path, key_id, "testbot"


def _run(args: list[str], env_extra: dict | None = None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Dry-run preserves everything
# ---------------------------------------------------------------------------

def test_dry_run_preserves_legacy_key(post_ceremony_dir):
    storage_dir, key_id, _ = post_ceremony_dir
    legacy_enc = storage_dir / f"{key_id}.key.enc"
    assert legacy_enc.exists()
    res = _run([
        "--agent-data-dir", str(storage_dir),
        "--skip-https-check",
    ])
    assert res.returncode == 0, res.stderr
    assert "DRY RUN" in res.stdout
    assert legacy_enc.exists(), "dry run must not delete the legacy key"


# ---------------------------------------------------------------------------
# Confirmation gates
# ---------------------------------------------------------------------------

def test_confirm_without_env_var_rejects(post_ceremony_dir):
    storage_dir, key_id, _ = post_ceremony_dir
    legacy_enc = storage_dir / f"{key_id}.key.enc"
    res = _run([
        "--agent-data-dir", str(storage_dir),
        "--skip-https-check", "--confirm",
    ])
    assert res.returncode == 2
    assert "KESTREL_DESTROY_CONFIRM" in res.stderr
    assert legacy_enc.exists(), "must not delete without env var"


def test_confirm_with_env_var_deletes_legacy_only(post_ceremony_dir):
    storage_dir, key_id, slug = post_ceremony_dir
    legacy_enc = storage_dir / f"{key_id}.key.enc"
    legacy_did_doc = storage_dir / f"{key_id}.json"
    succession = storage_dir / "successions" / f"{slug}.json"
    classical = storage_dir / f"{slug}_ed25519.key.enc"
    pq = storage_dir / f"{slug}_mldsa65.bytes.enc"

    assert all(p.exists() for p in [legacy_enc, legacy_did_doc, succession, classical, pq])

    res = _run(
        ["--agent-data-dir", str(storage_dir), "--skip-https-check", "--confirm"],
        env_extra={
            "KESTREL_DESTROY_CONFIRM": "I-have-verified-the-rollback-window",
        },
    )
    assert res.returncode == 0, res.stderr

    # Legacy private key destroyed
    assert not legacy_enc.exists(), "legacy .key.enc should be deleted"
    # Everything else preserved
    assert legacy_did_doc.exists(), "legacy DID document must be kept"
    assert succession.exists(), "succession statement must be kept"
    assert classical.exists(), "hybrid classical key must be kept"
    assert pq.exists(), "hybrid PQ key must be kept"


# ---------------------------------------------------------------------------
# Rollback window
# ---------------------------------------------------------------------------

def test_rollback_window_blocks_fresh_succession(post_ceremony_dir):
    """Set effective_from to NOW; rollback window of 7 days must reject."""
    storage_dir, key_id, slug = post_ceremony_dir
    succession_path = storage_dir / "successions" / f"{slug}.json"
    statement = json.loads(succession_path.read_text())
    statement["effective_from"] = datetime.now(timezone.utc).isoformat()
    succession_path.write_text(json.dumps(statement))

    legacy_enc = storage_dir / f"{key_id}.key.enc"
    res = _run(
        ["--agent-data-dir", str(storage_dir), "--skip-https-check", "--confirm"],
        env_extra={
            "KESTREL_DESTROY_CONFIRM": "I-have-verified-the-rollback-window",
        },
    )
    assert res.returncode == 1
    assert "rollback window" in res.stderr.lower()
    assert legacy_enc.exists(), "must not delete inside rollback window"


# ---------------------------------------------------------------------------
# Hybrid keys must exist (don't strand the agent)
# ---------------------------------------------------------------------------

def test_missing_hybrid_keys_blocks_destruction(post_ceremony_dir):
    storage_dir, key_id, slug = post_ceremony_dir
    # Delete the hybrid PQ half — agent would be left with no usable
    # signing key if we destroyed the legacy now
    (storage_dir / f"{slug}_mldsa65.bytes.enc").unlink()

    legacy_enc = storage_dir / f"{key_id}.key.enc"
    res = _run(
        ["--agent-data-dir", str(storage_dir), "--skip-https-check", "--confirm"],
        env_extra={
            "KESTREL_DESTROY_CONFIRM": "I-have-verified-the-rollback-window",
        },
    )
    assert res.returncode == 1
    assert "hybrid post-quantum key missing" in res.stderr.lower()
    assert legacy_enc.exists()


# ---------------------------------------------------------------------------
# No succession statement
# ---------------------------------------------------------------------------

def test_missing_succession_blocks_destruction(post_ceremony_dir):
    storage_dir, key_id, slug = post_ceremony_dir
    (storage_dir / "successions" / f"{slug}.json").unlink()

    legacy_enc = storage_dir / f"{key_id}.key.enc"
    res = _run(
        ["--agent-data-dir", str(storage_dir), "--skip-https-check", "--confirm"],
        env_extra={
            "KESTREL_DESTROY_CONFIRM": "I-have-verified-the-rollback-window",
        },
    )
    assert res.returncode == 1
    assert "succession" in res.stderr.lower() or "succession" in res.stdout.lower()
    assert legacy_enc.exists()
