"""
Load SLH-DSA-SHA2-128s release-signing key material from environment
variables into a SecureKeyStorage directory, then exit.

Wave 5 sub-PR 3 of Quantum Hardening (#921, #920). The GitHub Action
that auto-signs releases stores the SLH-DSA secret + public key as
base64-url GitHub secrets (``KESTREL_RELEASE_SECRET_B64`` and
``KESTREL_RELEASE_PUBLIC_B64``). This script decodes them and writes
the encrypted bundles ``kestrel release sign`` expects.

Why a separate script: keeping the env-var → SecureKeyStorage glue
out of the workflow YAML lets us test it in unit tests, and out of
``cli_release.py`` lets one-shot CI usage not pollute the operator
CLI surface.

Inputs (environment variables)
------------------------------

- ``KESTREL_RELEASE_SECRET_B64``: base64-url-encoded raw SLH-DSA-128s
  secret bytes (64 bytes after decode).
- ``KESTREL_RELEASE_PUBLIC_B64``: base64-url-encoded raw SLH-DSA-128s
  public bytes (32 bytes after decode).
- ``KESTREL_DATA_KEY``: required by SecureKeyStorage to encrypt the
  bundles at rest. The same env var the rest of Kestrel uses.

Usage
-----

::

    python scripts/release/load_signing_key_from_env.py \\
        --storage-dir ./release-keys \\
        --key-id release-key

Outputs (to ``--storage-dir``):

- ``release-key.bytes.enc`` — encrypted secret
- ``release-key_pub.bytes.enc`` — encrypted public sidecar (note
  ``_pub`` not ``.pub`` — see ``cli_release._load_slh_keypair`` for
  the codex-reviewed reasoning)

The release-sign workflow runs this once, then runs
``kestrel release sign --key-id release-key --storage-dir ./release-keys``.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from kestrel_sovereign.security.crypto_suite import SLHDSASHA2128sSuite
from kestrel_sovereign.security.key_storage import SecureKeyStorage


def _b64_decode(s: str, name: str) -> bytes:
    """Tolerant base64-url decode."""
    if not s:
        raise SystemExit(f"error: env var {name} is empty")
    pad = "=" * ((4 - len(s) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(s + pad)
    except Exception as e:
        raise SystemExit(f"error: env var {name} base64 decode failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load SLH-DSA-128s signing keys from env into SecureKeyStorage",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        required=True,
        help="Directory where the encrypted key bundles are written",
    )
    parser.add_argument(
        "--key-id",
        type=str,
        required=True,
        help="Key id (e.g. ``release-key``)",
    )
    parser.add_argument(
        "--secret-env",
        type=str,
        default="KESTREL_RELEASE_SECRET_B64",
        help="Env var holding the base64-url secret bytes",
    )
    parser.add_argument(
        "--public-env",
        type=str,
        default="KESTREL_RELEASE_PUBLIC_B64",
        help="Env var holding the base64-url public bytes",
    )
    args = parser.parse_args()

    secret_b64 = os.environ.get(args.secret_env, "")
    public_b64 = os.environ.get(args.public_env, "")
    secret = _b64_decode(secret_b64, args.secret_env)
    public = _b64_decode(public_b64, args.public_env)

    suite = SLHDSASHA2128sSuite()
    if len(secret) != suite.SECRET_KEY_SIZE:
        raise SystemExit(
            f"error: SLH-DSA-128s secret must be {suite.SECRET_KEY_SIZE} bytes; "
            f"got {len(secret)}"
        )
    if len(public) != suite.PUBLIC_KEY_SIZE:
        raise SystemExit(
            f"error: SLH-DSA-128s public must be {suite.PUBLIC_KEY_SIZE} bytes; "
            f"got {len(public)}"
        )

    # Validate the pair before writing anything: sign a probe and
    # verify against the public side. If they don't pair, the workflow
    # would silently produce an unverifiable release. Fail here.
    probe = b"kestrel-release-key-pair-check"
    sig = suite.sign(probe, secret)
    if not suite.verify(probe, sig, public):
        raise SystemExit(
            "error: KESTREL_RELEASE_SECRET_B64 and KESTREL_RELEASE_PUBLIC_B64 "
            "do not pair. Refusing to publish keys that won't sign+verify."
        )

    storage = SecureKeyStorage(storage_dir=Path(args.storage_dir))
    storage.save_secret_bytes(secret, args.key_id)
    storage.save_secret_bytes(public, f"{args.key_id}_pub")

    print(
        f"loaded SLH-DSA-128s keypair for key_id={args.key_id!r} into "
        f"{args.storage_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
