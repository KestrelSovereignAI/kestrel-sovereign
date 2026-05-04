"""
``kestrel release`` CLI commands — Wave 5 sub-PR 2 of Quantum Hardening (#920).

Two operator commands:

- ``kestrel release sign`` — collect every file under a directory,
  build a release manifest with SHA-256 hashes, sign with SLH-DSA-128s,
  emit JSON to stdout or ``--output``.
- ``kestrel release verify`` — load a manifest, validate the trusted
  signer matches a pinned SLH-DSA public key, validate every artifact
  on disk matches its recorded hash + size.

The CLI is a thin shell over :mod:`kestrel_sovereign.security.release_manifest`.
Key handling: the SLH-DSA secret is loaded via
:meth:`SecureKeyStorage.load_secret_bytes` from the existing key
storage path (Wave 3 sub-PR 4 added the raw-bytes storage API).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S,
    Keypair,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.multikey import public_key_to_multibase
from kestrel_sovereign.security.release_manifest import (
    ReleaseManifest,
    ReleaseManifestError,
    add_artifact_entry_from_path,
    finalize,
    new_manifest,
    sign_manifest,
    signable_payload,
    verify_artifact_path,
    verify_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_release_subcommands(subparsers: "argparse._SubParsersAction") -> None:
    """Register ``kestrel release {sign,verify}`` under the parent
    subparsers.

    Called from :func:`kestrel_sovereign.cli.build_parser`. Kept in
    this module so the release feature is self-contained and an
    operator who only wants sign/verify doesn't have to load
    everything cli.py imports.
    """
    release_p = subparsers.add_parser(
        "release",
        help="Sign or verify release artifacts (Wave 5 of Quantum Hardening)",
    )
    release_sub = release_p.add_subparsers(dest="release_command")

    sign_p = release_sub.add_parser(
        "sign",
        help="Build and sign a release manifest for a directory of artifacts",
    )
    sign_p.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory containing release artifacts (one or more files; "
             "subdirectories are walked recursively)",
    )
    sign_p.add_argument(
        "--release-tag",
        type=str,
        required=True,
        help="Release tag (e.g. v1.2.3)",
    )
    sign_p.add_argument(
        "--key-id",
        type=str,
        required=True,
        help="Key id under SecureKeyStorage to load the SLH-DSA secret "
             "(stored via save_secret_bytes); also used as the public-key "
             "lookup default. Reuses Wave 3's secret-bytes path.",
    )
    sign_p.add_argument(
        "--signer-did",
        type=str,
        default="",
        help="Optional DID identifying the signer (informational; embedded "
             "in the manifest payload but not enforced at verify time)",
    )
    sign_p.add_argument(
        "--kid",
        type=str,
        default="release-key-1",
        help="Signature kid (default: release-key-1)",
    )
    sign_p.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output manifest JSON path (default: stdout)",
    )
    sign_p.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        help="Override SecureKeyStorage directory (default: agent_data/)",
    )

    verify_p = release_sub.add_parser(
        "verify",
        help="Verify a signed release manifest against a pinned trusted signer",
    )
    verify_p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the signed release manifest JSON",
    )
    verify_p.add_argument(
        "--artifacts-dir",
        type=str,
        required=True,
        help="Directory containing the actual artifact bytes to verify "
             "against the manifest's recorded hashes",
    )
    verify_p.add_argument(
        "--trusted-signer-multibase",
        type=str,
        required=True,
        help="Pinned trusted SLH-DSA public key as a multibase z-prefix "
             "string. Operators document this once at release time; "
             "consumers bake it into their verification pipeline.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_artifacts(
    root: Path,
    *,
    skip: List[Path] = None,
) -> Iterator[Tuple[str, Path]]:
    """Walk ``root`` recursively. Yields ``(relative_path, absolute_path)``.

    Paths use forward slashes for cross-platform consistency; the
    manifest's path validator already rejects ``..`` / absolute /
    Windows drive prefixes that could let consumers escape the
    artifact dir.

    ``skip`` lists absolute paths to exclude (e.g. the manifest
    output if it lives inside the artifacts dir).

    Codex P2 round 3: this used to return ``List[Tuple[str, bytes]]``
    which buffered every artifact in memory at once. For real releases
    with multi-GB tarballs that OOMs. Now it yields paths only and
    callers read each file one at a time so peak memory is bounded by
    the largest single artifact.
    """
    if not root.exists():
        raise FileNotFoundError(f"artifacts directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"artifacts path is not a directory: {root}")
    skip_resolved = {p.resolve() for p in (skip or [])}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in skip_resolved:
            continue
        rel = path.relative_to(root)
        rel_str = rel.as_posix()
        yield (rel_str, path)


def _load_slh_keypair(storage: SecureKeyStorage, key_id: str) -> Keypair:
    """Load an SLH-DSA secret from SecureKeyStorage and pair with the
    derived public key.

    The Wave 3 ceremony's archival keypair pattern: the operator stored
    the secret via ``save_secret_bytes(slh_kp.private_key, key_id)``.
    We load the secret and ALSO read the public key from a parallel
    ``<key_id>.pub.bytes.enc`` file (raw 32-byte SLH-DSA public key).

    For now the public key is stored separately because pqcrypto's
    ML-KEM/SLH-DSA APIs don't derive a public from a secret on demand;
    sign-time and verify-time callers persist both. The CLI ``sign``
    command writes the public key alongside when it generates a fresh
    keypair (operator runbook step).
    """
    secret = storage.load_secret_bytes(key_id)
    # Sidecar id uses ``_pub`` (not ``.pub``) because
    # SecureKeyStorage._get_secret_bytes_path strips characters outside
    # ``[A-Za-z0-9-_]``. ``release-key.pub`` would collide with a real
    # ``release-keypub`` id (codex P2 round 5).
    public_id = f"{key_id}_pub"
    if not storage.has_secret_bytes(public_id):
        raise ReleaseManifestError(
            f"public-key file not found for key_id={key_id!r}; "
            f"expected ``{public_id}`` next to the secret. The release "
            f"keypair must have been generated and persisted as a pair "
            f"using SecureKeyStorage.save_secret_bytes(secret, key_id) + "
            f"save_secret_bytes(public, key_id+'_pub')."
        )
    public = storage.load_secret_bytes(public_id)
    return Keypair(suite_id=ALG_SLH_DSA_SHA2_128S, private_key=secret, public_key=public)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_release(args) -> int:
    """Dispatch ``kestrel release {sign,verify}``."""
    handlers = {
        "sign": cmd_release_sign,
        "verify": cmd_release_verify,
    }
    handler = handlers.get(getattr(args, "release_command", None))
    if handler is None:
        print("Usage: kestrel release {sign,verify}", file=sys.stderr)
        return 1
    return handler(args)


def cmd_release_sign(args) -> int:
    """Build + sign a release manifest."""
    storage_dir = Path(args.storage_dir) if args.storage_dir else None
    storage = SecureKeyStorage(storage_dir=storage_dir)
    try:
        keypair = _load_slh_keypair(storage, args.key_id)
    except (FileNotFoundError, ReleaseManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: failed to load signing key {args.key_id!r}: {e}", file=sys.stderr)
        return 2

    artifacts_dir = Path(args.artifacts_dir).resolve()
    # Skip the output path if it lives inside the artifacts dir, so a
    # stale manifest from a previous run doesn't get hashed into the
    # new manifest (codex P2 round 1).
    skip = []
    if args.output and args.output != "-":
        out_path = Path(args.output).resolve()
        try:
            out_path.relative_to(artifacts_dir)
        except ValueError:
            pass  # output is outside the artifacts dir
        else:
            skip.append(out_path)
    try:
        manifest = new_manifest(
            release_tag=args.release_tag,
            signer_did=args.signer_did,
        )
    except ReleaseManifestError as e:
        # Codex P3 round 8: ``--release-tag ''`` (or other operator-
        # invalid inputs that the dataclass validates) used to raise
        # past the CLI boundary. Wrap as a structured exit-2 error.
        print(f"error: {e}", file=sys.stderr)
        return 2
    artifact_count = 0
    try:
        for rel_path, abs_path in _walk_artifacts(artifacts_dir, skip=skip):
            # Streaming hash + size: peak memory is the 1 MiB chunk
            # size set in release_manifest._stream_sha256_and_size,
            # not the size of the file (codex P2 round 4).
            try:
                manifest = add_artifact_entry_from_path(manifest, rel_path, abs_path)
            except ReleaseManifestError as e:
                print(f"error: artifact {rel_path!r}: {e}", file=sys.stderr)
                return 2
            except OSError as e:
                print(f"error: artifact {rel_path!r} read failed: {e}", file=sys.stderr)
                return 2
            artifact_count += 1
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if artifact_count == 0:
        print(
            f"error: no files found under {artifacts_dir}; refusing to sign "
            f"an empty manifest",
            file=sys.stderr,
        )
        return 2

    # If the stored secret bytes are not a valid SLH-DSA secret (e.g.
    # corrupted, wrong alg saved under this id), pqcrypto's sign call
    # raises CryptoSuiteError out of sign_manifest. Catch it as a
    # structured signing-key error (codex P2 round 6).
    try:
        manifest = sign_manifest(manifest, keypair, kid=args.kid)
    except Exception as e:
        # CryptoSuiteError isn't imported here on purpose — any
        # unexpected exception during sign indicates corrupt key
        # material rather than a programming bug we should propagate
        # past the operator.
        print(
            f"error: signing failed (corrupt or wrong-alg key bytes "
            f"under key_id={args.key_id!r}): {e}",
            file=sys.stderr,
        )
        return 2
    manifest = finalize(manifest)

    # Self-verify against the LOADED public key before publishing
    # anything (codex P2 round 2). If <key_id>.pub is stale or
    # doesn't pair with the secret, sign would still succeed with the
    # secret but the operator would publish an unverifiable manifest
    # paired with the wrong multibase. Fail loud here instead.
    suite = SLHDSASHA2128sSuite()
    payload_for_check = signable_payload(manifest)
    sig_hex = manifest.signatures[-1]["sig"]
    if not suite.verify(payload_for_check, bytes.fromhex(sig_hex), keypair.public_key):
        print(
            f"error: signing keypair self-check FAILED — the loaded "
            f"public key {args.key_id}.pub does not pair with the secret "
            f"key {args.key_id}. Refusing to publish a manifest that "
            f"consumers cannot verify with the printed pubkey.",
            file=sys.stderr,
        )
        return 2

    json_str = json.dumps(manifest.to_dict(), indent=2)
    if args.output == "-":
        print(json_str)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")

    # Operator-friendly: print the multibase pubkey on stderr so they
    # can hand it to consumers without parsing the JSON themselves.
    pub_mb = public_key_to_multibase(SLHDSASHA2128sSuite(), keypair.public_key)
    print(
        f"signed {len(manifest.artifacts)} artifact(s) for {args.release_tag}",
        file=sys.stderr,
    )
    print(f"trusted_signer_multibase: {pub_mb}", file=sys.stderr)
    return 0


def cmd_release_verify(args) -> int:
    """Verify a signed release manifest + every artifact's bytes."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ReleaseManifestError(
                f"manifest top-level must be an object; got "
                f"{type(parsed).__name__}"
            )
        manifest = ReleaseManifest.from_dict(parsed)
    except (
        json.JSONDecodeError, KeyError, ReleaseManifestError,
        TypeError, ValueError, AttributeError, OSError,
    ) as e:
        # Cover all the ways untrusted manifest input can fail to
        # parse: malformed JSON, non-object top-level (codex P2 round
        # 6), bad field shapes (round 1), file IO. All collapse to
        # exit code 2.
        print(f"error: malformed manifest: {e}", file=sys.stderr)
        return 2

    result = verify_manifest(
        manifest,
        trusted_signer_multibase=args.trusted_signer_multibase,
    )
    if not result.ok:
        print(f"manifest signature verification FAILED: {result.reason}", file=sys.stderr)
        return 3

    artifacts_dir = Path(args.artifacts_dir).resolve()
    if not artifacts_dir.is_dir():
        print(f"error: artifacts dir not found: {artifacts_dir}", file=sys.stderr)
        return 2

    failed: List[str] = []
    manifested_paths = {entry.path for entry in manifest.artifacts}
    for entry in manifest.artifacts:
        artifact_path = artifacts_dir / entry.path
        if not artifact_path.exists():
            failed.append(f"{entry.path}: file not present in artifacts dir")
            continue
        try:
            # Streaming verify — peak memory bounded by chunk size,
            # not by artifact size (codex P2 round 4).
            ok = verify_artifact_path(manifest, entry.path, artifact_path)
        except OSError as e:
            failed.append(f"{entry.path}: read failed: {e}")
            continue
        if not ok:
            failed.append(f"{entry.path}: hash or size mismatch")

    # Codex P2 round 7: an attacker who can drop a file into the
    # artifacts directory after signing would have it ride along
    # alongside the verified release. Walk the directory and refuse
    # any file that isn't in the manifest. Exempt the manifest file
    # itself if it's colocated (sign-side already excludes it from
    # the artifact walk per round 1's fix).
    manifest_path_resolved = manifest_path.resolve()
    for path in artifacts_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == manifest_path_resolved:
            continue
        rel = path.relative_to(artifacts_dir).as_posix()
        if rel not in manifested_paths:
            failed.append(f"{rel}: present on disk but NOT in signed manifest")

    if failed:
        print(
            f"manifest signature verified, but {len(failed)} artifact(s) failed:",
            file=sys.stderr,
        )
        for line in failed:
            print(f"  - {line}", file=sys.stderr)
        return 4

    print(
        f"OK: manifest signature verified for {manifest.release_tag} "
        f"({len(manifest.artifacts)} artifact(s) hash-checked)"
    )
    return 0


__all__ = [
    "add_release_subcommands",
    "cmd_release",
    "cmd_release_sign",
    "cmd_release_verify",
]
