"""Durable Cloud Run identity custody bundle tests (#2472)."""

import json
import os

import pytest

from kestrel_sovereign.identity.custody_bundle import (
    IDENTITY_BUNDLE_ENV,
    IdentityCustodyError,
    create_identity_bundle,
    restore_identity_bundle,
    restore_identity_bundle_from_env,
    write_identity_bundle,
)
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async


@pytest.mark.asyncio
async def test_encrypted_bundle_round_trip_and_fail_closed(tmp_path, monkeypatch):
    """One encrypted bundle restores the exact signing identity, never a clone."""
    monkeypatch.setenv(
        "KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!"
    )
    monkeypatch.setenv("KESTREL_DID_WEB_DOMAIN", "agents.custody.test")
    source = tmp_path / "source"
    credentials = await create_kestrel_identity_async(
        str(source),
        agent_name="Custody Bird",
        did_web_slug="custody-bird",
    )
    expected_did = credentials.agent_did
    source_identity = load_agent_identity(None, storage_dir=source)

    bundle = create_identity_bundle(source, expected_did=expected_did)
    manifest = json.loads(bundle)
    paths = {item["path"] for item in manifest["files"]}
    assert "kestrel_prime.db" not in paths
    assert not any(path.endswith(".pem") for path in paths)
    assert any(path.endswith(".enc") for path in paths)

    restored_dir = tmp_path / "restored"
    assert (
        restore_identity_bundle(bundle, restored_dir, expected_did=expected_did)
        == expected_did
    )
    restored_identity = load_agent_identity(None, storage_dir=restored_dir)
    assert restored_identity.signing_did == source_identity.signing_did
    assert (
        restored_identity.new_verification_methods
        == source_identity.new_verification_methods
    )
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in restored_dir.rglob("*")
        if path.is_file()
    )

    # Exact restore is idempotent, but drift and unexpected local state are
    # never overwritten.
    restore_identity_bundle(bundle, restored_dir, expected_did=expected_did)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "kestrel_prime.db").write_bytes(b"not-authoritative")
    with pytest.raises(IdentityCustodyError, match="unexpected local state"):
        restore_identity_bundle(bundle, occupied, expected_did=expected_did)
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(IdentityCustodyError, match="not a safe directory"):
        restore_identity_bundle(bundle, symlink, expected_did=expected_did)

    tampered = json.loads(bundle)
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(IdentityCustodyError, match="digest mismatch"):
        restore_identity_bundle(
            json.dumps(tampered),
            tmp_path / "tampered",
            expected_did=expected_did,
        )

    unsafe_path = json.loads(bundle)
    unsafe_path["files"][0]["path"] = "../private.key"
    with pytest.raises(IdentityCustodyError, match="unsafe path"):
        restore_identity_bundle(
            json.dumps(unsafe_path),
            tmp_path / "unsafe",
            expected_did=expected_did,
        )

    with pytest.raises(IdentityCustodyError, match="does not match"):
        restore_identity_bundle(
            bundle,
            tmp_path / "wrong-did",
            expected_did="did:web:agents.custody.test:someone-else",
        )
    wrong_signer = json.loads(bundle)
    wrong_signer["signing_did"] = "did:web:agents.custody.test:someone-else"
    with pytest.raises(IdentityCustodyError, match="active signing DID"):
        restore_identity_bundle(
            json.dumps(wrong_signer),
            tmp_path / "wrong-signer",
            expected_did=expected_did,
        )
    with pytest.raises(IdentityCustodyError, match="unavailable"):
        restore_identity_bundle_from_env(
            tmp_path / "missing-env",
            expected_did=expected_did,
            env={},
        )
    assert (
        restore_identity_bundle_from_env(
            tmp_path / "from-env",
            expected_did=expected_did,
            env={IDENTITY_BUNDLE_ENV: bundle},
        )
        == expected_did
    )

    plaintext = source / "kestrel_0xdeadbeef.pem"
    plaintext.write_text("plaintext-private-key", encoding="utf-8")
    with pytest.raises(IdentityCustodyError, match="plaintext private-key"):
        create_identity_bundle(source, expected_did=expected_did)

    output = write_identity_bundle(tmp_path / "custody.json", bundle)
    assert (output.stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        write_identity_bundle(output, bundle)


def test_bundle_writer_does_not_depend_on_process_umask(tmp_path):
    """Custody export remains mode 0600 even under a permissive umask."""
    prior = os.umask(0)
    try:
        output = write_identity_bundle(tmp_path / "bundle.json", "{}")
    finally:
        os.umask(prior)
    assert (output.stat().st_mode & 0o777) == 0o600
