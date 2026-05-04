"""
``kestrel release {sign,verify}`` CLI tests — Wave 5 sub-PR 2 (#920).

Covers the CLI argparse glue + happy paths + error paths:

- argparse wires up "release sign" and "release verify" with required args
- sign happy path: directory of artifacts → signed manifest JSON →
  the multibase pubkey is printed on stderr for the operator
- verify happy path: manifest + same artifacts dir + pinned signer → 0
- verify rejects:
  * missing manifest file
  * malformed manifest JSON
  * wrong trusted signer
  * tampered artifact bytes
  * missing artifact file
- sign rejects empty artifacts directory
- artifact-walk uses POSIX paths even on Windows-style trees
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from kestrel_sovereign.cli_release import (
    add_release_subcommands,
    cmd_release_sign,
    cmd_release_verify,
)
from kestrel_sovereign.security.crypto_suite import SLHDSASHA2128sSuite
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.multikey import public_key_to_multibase


@pytest.fixture
def storage_with_keypair(tmp_path, monkeypatch):
    """Set up a SecureKeyStorage with an SLH-DSA keypair under
    key_id='release-key' (and the .pub side)."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)
    storage_dir = tmp_path / "keys"
    storage_dir.mkdir()
    storage = SecureKeyStorage(storage_dir=storage_dir)
    suite = SLHDSASHA2128sSuite()
    kp = suite.generate_keypair()
    storage.save_secret_bytes(kp.private_key, "release-key")
    storage.save_secret_bytes(kp.public_key, "release-key.pub")
    return storage_dir, kp


@pytest.fixture
def artifacts_dir(tmp_path):
    """Tree with a few files at different depths."""
    root = tmp_path / "release-artifacts"
    root.mkdir()
    (root / "wheel.whl").write_bytes(b"wheel-bytes-go-here")
    (root / "src.tar.gz").write_bytes(b"tarball-bytes")
    (root / "subdir").mkdir()
    (root / "subdir" / "extra.txt").write_bytes(b"more-bytes")
    return root


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    add_release_subcommands(sub)
    return p


def test_release_sign_argparse_required_args():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["release", "sign"])  # missing args


def test_release_verify_argparse_required_args():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["release", "verify"])  # missing args


def test_release_sign_argparse_minimal():
    parser = _build_parser()
    args = parser.parse_args([
        "release", "sign",
        "--artifacts-dir", "/tmp/a",
        "--release-tag", "v1",
        "--key-id", "k",
    ])
    assert args.command == "release"
    assert args.release_command == "sign"
    assert args.artifacts_dir == "/tmp/a"
    assert args.release_tag == "v1"
    assert args.key_id == "k"


# ---------------------------------------------------------------------------
# sign happy path
# ---------------------------------------------------------------------------

def test_sign_writes_manifest_and_returns_zero(
    storage_with_keypair, artifacts_dir, tmp_path, capsys,
):
    storage_dir, kp = storage_with_keypair
    output = tmp_path / "manifest.json"
    args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1.2.3",
        key_id="release-key",
        signer_did="did:web:example.com",
        kid="release-key-1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(args)
    assert rc == 0
    assert output.exists()
    manifest = json.loads(output.read_text())
    assert manifest["release_tag"] == "v1.2.3"
    assert manifest["signer_did"] == "did:web:example.com"
    assert len(manifest["artifacts"]) == 3
    assert {a["path"] for a in manifest["artifacts"]} == {
        "wheel.whl", "src.tar.gz", "subdir/extra.txt",
    }
    # stderr should print the multibase pubkey for the operator
    captured = capsys.readouterr()
    assert "trusted_signer_multibase" in captured.err


def test_sign_rejects_empty_artifacts_dir(storage_with_keypair, tmp_path):
    storage_dir, _ = storage_with_keypair
    empty = tmp_path / "nothing"
    empty.mkdir()
    args = argparse.Namespace(
        artifacts_dir=str(empty),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output="-",
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(args)
    assert rc == 2


def test_sign_handles_missing_artifacts_dir(storage_with_keypair):
    storage_dir, _ = storage_with_keypair
    args = argparse.Namespace(
        artifacts_dir="/nonexistent/path",
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output="-",
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(args)
    assert rc == 2


def test_sign_handles_missing_key(tmp_path, artifacts_dir, monkeypatch):
    """No public-key file under the storage → structured error, not crash."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)
    storage_dir = tmp_path / "keys"
    storage_dir.mkdir()
    args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="never-saved",
        signer_did="",
        kid="k1",
        output="-",
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(args)
    assert rc == 2


# ---------------------------------------------------------------------------
# verify round-trip
# ---------------------------------------------------------------------------

def test_verify_happy_path_after_sign(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    storage_dir, kp = storage_with_keypair
    output = tmp_path / "manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(sign_args) == 0

    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    verify_args = argparse.Namespace(
        manifest=str(output),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(verify_args) == 0


def test_verify_rejects_missing_manifest(tmp_path, artifacts_dir, storage_with_keypair):
    _, kp = storage_with_keypair
    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    args = argparse.Namespace(
        manifest=str(tmp_path / "missing.json"),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(args) == 2


def test_verify_rejects_malformed_manifest(tmp_path, artifacts_dir, storage_with_keypair):
    _, kp = storage_with_keypair
    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    bad = tmp_path / "manifest.json"
    bad.write_text("{not-valid-json")
    args = argparse.Namespace(
        manifest=str(bad),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(args) == 2


def test_verify_rejects_wrong_signer(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    storage_dir, _ = storage_with_keypair
    output = tmp_path / "manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(sign_args) == 0

    other_kp = SLHDSASHA2128sSuite().generate_keypair()
    other_pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), other_kp.public_key)
    args = argparse.Namespace(
        manifest=str(output),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=other_pub_mb,
    )
    assert cmd_release_verify(args) == 3  # signature failure


def test_verify_rejects_tampered_artifact_bytes(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    """After signing, mutate one artifact on disk. The signature still
    verifies (it covered the manifest only) but the artifact-bytes
    check fails."""
    storage_dir, kp = storage_with_keypair
    output = tmp_path / "manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(sign_args) == 0

    # Mutate one artifact
    (artifacts_dir / "wheel.whl").write_bytes(b"TAMPERED-CONTENT")

    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    args = argparse.Namespace(
        manifest=str(output),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(args) == 4  # artifact-bytes failure


def test_verify_rejects_missing_artifact_file(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    """An artifact listed in the manifest but not present on disk."""
    storage_dir, kp = storage_with_keypair
    output = tmp_path / "manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(sign_args) == 0

    # Delete one artifact
    (artifacts_dir / "wheel.whl").unlink()

    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    args = argparse.Namespace(
        manifest=str(output),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(args) == 4


# ---------------------------------------------------------------------------
# Posix paths in manifest
# ---------------------------------------------------------------------------

def test_sign_rejects_mismatched_public_key(tmp_path, artifacts_dir, monkeypatch):
    """Codex P2 round 2: if ``<key_id>.pub`` is stale and doesn't pair
    with the secret, the previous CLI signed anyway and printed a
    multibase that wouldn't verify the manifest. Now self-checks and
    refuses."""
    monkeypatch.setenv("KESTREL_DATA_KEY", "x" * 32)
    storage_dir = tmp_path / "keys"
    storage_dir.mkdir()
    storage = SecureKeyStorage(storage_dir=storage_dir)
    suite = SLHDSASHA2128sSuite()

    # Save secret of one keypair, public of a DIFFERENT keypair → stale
    kp_a = suite.generate_keypair()
    kp_b = suite.generate_keypair()
    storage.save_secret_bytes(kp_a.private_key, "release-key")
    storage.save_secret_bytes(kp_b.public_key, "release-key.pub")  # WRONG pair

    args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output="-",
        storage_dir=str(storage_dir),
    )
    rc = cmd_release_sign(args)
    assert rc == 2


def test_resign_skips_existing_manifest_inside_artifacts_dir(
    storage_with_keypair, artifacts_dir,
):
    """Codex P2 round 1: when --output points inside --artifacts-dir
    and a stale manifest exists from a previous run, it used to be
    hashed into the new manifest (which then verifies as 'manifest
    artifact bytes mismatch' immediately after the rewrite)."""
    storage_dir, kp = storage_with_keypair
    output = artifacts_dir / "manifest.json"  # INSIDE artifacts_dir

    # First sign
    args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(args) == 0
    first = json.loads(output.read_text())
    assert "manifest.json" not in {a["path"] for a in first["artifacts"]}

    # Re-sign — the existing manifest must NOT appear as an artifact
    assert cmd_release_sign(args) == 0
    second = json.loads(output.read_text())
    assert "manifest.json" not in {a["path"] for a in second["artifacts"]}

    # And the re-signed manifest still verifies cleanly
    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)
    verify_args = argparse.Namespace(
        manifest=str(output),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(verify_args) == 0


def test_verify_handles_malformed_manifest_field_types(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    """Codex P2 round 1: a JSON-valid manifest with bad field types
    (e.g. ``artifacts: [1]``, non-int size) used to raise TypeError/
    ValueError out of from_dict. Now wrapped into the exit-2 path."""
    _, kp = storage_with_keypair
    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), kp.public_key)

    # Manifest with non-dict artifact entry
    bad = tmp_path / "bad-manifest.json"
    bad.write_text(json.dumps({
        "format": "kestrel-release-manifest-v1",
        "version": 1,
        "release_tag": "v1",
        "released_at": "2026-05-04T20:00:00+00:00",
        "signer_did": "",
        "artifacts": [1, 2, 3],  # not dicts
        "manifest_id": "0" * 64,
        "signatures": [],
    }))
    args = argparse.Namespace(
        manifest=str(bad),
        artifacts_dir=str(artifacts_dir),
        trusted_signer_multibase=pub_mb,
    )
    assert cmd_release_verify(args) == 2


def test_manifest_uses_posix_paths_for_subdirs(
    storage_with_keypair, artifacts_dir, tmp_path,
):
    """Even on Windows, manifest entries use forward slashes for
    byte-stable hashes across platforms."""
    storage_dir, _ = storage_with_keypair
    output = tmp_path / "manifest.json"
    sign_args = argparse.Namespace(
        artifacts_dir=str(artifacts_dir),
        release_tag="v1",
        key_id="release-key",
        signer_did="",
        kid="k1",
        output=str(output),
        storage_dir=str(storage_dir),
    )
    assert cmd_release_sign(sign_args) == 0
    manifest = json.loads(output.read_text())
    paths = {a["path"] for a in manifest["artifacts"]}
    assert "subdir/extra.txt" in paths
    # No backslashes anywhere in the manifest's recorded paths
    assert all("\\" not in p for p in paths)
