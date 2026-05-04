"""
Tests for ``scripts/release/load_signing_key_from_env.py`` — Wave 5
sub-PR 3 (#920).

The loader is invoked by the release-sign GitHub Action; it pulls
SLH-DSA-SHA2-128s key material from base64-url env vars, validates
the pair, and persists into a SecureKeyStorage directory.

Covers:
- Round-trip: env vars → storage → ``cli_release._load_slh_keypair``
  produces the same Keypair (i.e. exit-0 sign with that key works)
- Wrong-length secret rejected
- Wrong-length public rejected
- Mismatched pair rejected (probe-sign / probe-verify check)
- Empty env vars rejected
- Malformed base64 rejected
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kestrel_sovereign.security.crypto_suite import SLHDSASHA2128sSuite
from kestrel_sovereign.security.key_storage import SecureKeyStorage


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release" / "load_signing_key_from_env.py"


def _run_loader(env: dict, *, storage_dir: Path, key_id: str = "release-key"):
    """Run the loader script as a subprocess. Returns CompletedProcess."""
    proc_env = dict(os.environ)
    proc_env.update(env)
    proc_env.setdefault("KESTREL_DATA_KEY", "x" * 32)
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--storage-dir", str(storage_dir),
         "--key-id", key_id],
        env=proc_env,
        capture_output=True,
        text=True,
    )


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def test_loader_happy_path(tmp_path):
    """Generate a real keypair, encode → load → assert SecureKeyStorage
    holds both halves under the right ids."""
    suite = SLHDSASHA2128sSuite()
    kp = suite.generate_keypair()
    storage_dir = tmp_path / "release-keys"

    env = {
        "KESTREL_RELEASE_SECRET_B64": _b64(kp.private_key),
        "KESTREL_RELEASE_PUBLIC_B64": _b64(kp.public_key),
    }
    res = _run_loader(env, storage_dir=storage_dir)
    assert res.returncode == 0, res.stderr
    storage = SecureKeyStorage(storage_dir=storage_dir)
    assert storage.has_secret_bytes("release-key")
    assert storage.has_secret_bytes("release-key_pub")
    assert storage.load_secret_bytes("release-key") == kp.private_key
    assert storage.load_secret_bytes("release-key_pub") == kp.public_key


def test_loader_rejects_wrong_length_secret(tmp_path):
    suite = SLHDSASHA2128sSuite()
    kp = suite.generate_keypair()
    env = {
        "KESTREL_RELEASE_SECRET_B64": _b64(b"\x00" * 30),  # too short
        "KESTREL_RELEASE_PUBLIC_B64": _b64(kp.public_key),
    }
    res = _run_loader(env, storage_dir=tmp_path / "k")
    assert res.returncode != 0
    assert "secret must be" in res.stderr


def test_loader_rejects_wrong_length_public(tmp_path):
    suite = SLHDSASHA2128sSuite()
    kp = suite.generate_keypair()
    env = {
        "KESTREL_RELEASE_SECRET_B64": _b64(kp.private_key),
        "KESTREL_RELEASE_PUBLIC_B64": _b64(b"\x00" * 16),  # too short
    }
    res = _run_loader(env, storage_dir=tmp_path / "k")
    assert res.returncode != 0
    assert "public must be" in res.stderr


def test_loader_rejects_mismatched_pair(tmp_path):
    """The probe-sign / probe-verify check inside the loader catches
    pairs that don't actually pair, BEFORE the encrypted bundles are
    written. Otherwise the GH Action would silently publish keys that
    don't sign+verify."""
    suite = SLHDSASHA2128sSuite()
    kp_a = suite.generate_keypair()
    kp_b = suite.generate_keypair()
    env = {
        "KESTREL_RELEASE_SECRET_B64": _b64(kp_a.private_key),
        "KESTREL_RELEASE_PUBLIC_B64": _b64(kp_b.public_key),  # WRONG pair
    }
    res = _run_loader(env, storage_dir=tmp_path / "k")
    assert res.returncode != 0
    assert "do not pair" in res.stderr


def test_loader_rejects_empty_secret_env(tmp_path):
    env = {
        "KESTREL_RELEASE_SECRET_B64": "",
        "KESTREL_RELEASE_PUBLIC_B64": _b64(b"\x00" * 32),
    }
    res = _run_loader(env, storage_dir=tmp_path / "k")
    assert res.returncode != 0
    assert "is empty" in res.stderr


def test_loader_rejects_malformed_base64(tmp_path):
    suite = SLHDSASHA2128sSuite()
    kp = suite.generate_keypair()
    env = {
        "KESTREL_RELEASE_SECRET_B64": "!!!not-base64!!!",
        "KESTREL_RELEASE_PUBLIC_B64": _b64(kp.public_key),
    }
    res = _run_loader(env, storage_dir=tmp_path / "k")
    assert res.returncode != 0
    assert "base64 decode failed" in res.stderr
