#!/usr/bin/env python3
"""
Live hybrid-rotation ceremony for Kestrel #1 (a.k.a. Emma).

This is the **real** version of the dry-run rehearsal at
``scripts/quantum_rotation_dry_run.py``. It runs against the
operator's actual ``agent_data/Emma/`` directory, the agent's
real legacy ECDSA keypair, and produces artifacts the operator
will publish to a real HTTPS domain.

Pre-flight requirements
-----------------------

1. ``KESTREL_DATA_KEY`` set in env — the master key used to
   encrypt private-key bundles at rest.
2. ``--did-domain`` reachable over HTTPS, with a static-file host
   ready to serve ``https://<domain>/<slug>/did.json``.
3. Pre-ceremony backup taken (use ``quantum_pre_ceremony_backup.py``)
   and verified at an off-disk location (Google Drive, etc.).
4. Operator has decided whether the agent process should be
   running during the ceremony. The ceremony itself only reads
   the legacy key from disk and writes new keys; it does not
   touch the live agent process. But after the cutoff, the agent
   should be restarted with the new hybrid keys.

What the ceremony does
----------------------

1. Loads Kestrel #1's legacy ECDSA private key from
   ``agent_data/Emma/`` via SecureKeyStorage.
2. Verifies the loaded key matches the public key recorded in
   the existing DID document JSON (sanity check — refuses to run
   if they don't match, since that means the wrong key is loaded).
3. Generates a fresh SLH-DSA-SHA2-128s archival keypair.
4. Calls ``rotation_ceremony.run_rotation_ceremony`` to mint:
   - A new hybrid Ed25519 + ML-DSA-65 keypair
   - A signed succession statement (legacy → hybrid)
   - The new ``did.json`` for the ``did:web`` identity
5. Self-verifies the result (built into ``run_rotation_ceremony``).
6. Persists the new private keys via SecureKeyStorage:
   - ``<slug>_ed25519`` (classical hybrid half)
   - ``<slug>_mldsa65`` (post-quantum hybrid half)
   - ``<slug>_archival_slhdsa`` (SLH-DSA archival)
7. Writes the new ``did.json`` and the signed succession statement
   to a staging output directory ready for the operator to commit
   to the agent-identities repo.
8. Does NOT publish anything, restart the agent, or destroy the
   legacy key. Those are explicitly operator-driven follow-ups.

Usage
-----

::

    export KESTREL_DATA_KEY='<existing master key>'
    uv run python scripts/quantum_kestrel_1_ceremony.py \\
        --did-domain agents.kestrelsovereign.com \\
        --did-slug   kestrel \\
        --effective-from 2026-05-05T17:00:00+00:00

Optionally::

    --agent-data-dir /path/to/agent_data/Emma  (default: agent_data/Emma)
    --legacy-eth-address 0xB4E7F05F...           (default: derived from existing DID JSON)
    --output-dir /tmp/kestrel-1-ceremony-<timestamp>  (default; auto-timestamped)
    --skip-confirm  (skip the interactive y/N gate; still requires explicit env var)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat,
)

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.security.crypto_suite import (
    Keypair,
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage


PROJECT_ROOT = Path("/Volumes/data2/projects/kestrel-sovereign")
DEFAULT_AGENT_DATA = PROJECT_ROOT / "agent_data" / "Emma"
GO_AHEAD_ENV = "KESTREL_CEREMONY_CONFIRM"


def _step(title: str) -> None:
    print(f"\n=== {title} ===")


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _info(msg: str) -> None:
    print(f"       {msg}")


def _err(msg: str) -> None:
    print(f"  ERR  {msg}", file=sys.stderr)


def _public_key_to_uncompressed_hex(public_key) -> str:
    raw = public_key.public_bytes(
        encoding=Encoding.X962,
        format=PublicFormat.UncompressedPoint,
    )
    return raw.hex()


def _load_existing_did_doc(agent_data: Path, eth_address: str) -> dict:
    did_doc_path = agent_data / f"kestrel_{eth_address}.json"
    if not did_doc_path.exists():
        raise SystemExit(
            f"error: DID document not found at {did_doc_path}. "
            f"Pass --legacy-eth-address explicitly if the address differs."
        )
    return json.loads(did_doc_path.read_text())


def _derive_eth_address_from_filenames(agent_data: Path) -> str:
    """Find the agent's Ethereum address by inspecting filenames in the
    data dir. Each Kestrel agent persists ``kestrel_<eth_address>.json``
    + ``kestrel_<eth_address>.key.enc`` siblings. If both exist with
    matching addresses, that's the agent's identity.
    """
    candidates = sorted(agent_data.glob("kestrel_0x*.json"))
    if not candidates:
        raise SystemExit(
            f"error: no kestrel_0x*.json file found in {agent_data}; "
            f"pass --legacy-eth-address explicitly."
        )
    if len(candidates) > 1:
        raise SystemExit(
            f"error: multiple kestrel_0x*.json files in {agent_data}; "
            f"pass --legacy-eth-address explicitly to disambiguate: "
            f"{[c.name for c in candidates]}"
        )
    return candidates[0].stem.removeprefix("kestrel_")


def _confirm_go_ahead() -> None:
    """Refuses to run unless the operator has explicitly set
    ``KESTREL_CEREMONY_CONFIRM=I-have-backups-and-want-to-rotate``.

    This is a deliberate footgun-removal: the ceremony writes new keys
    and mints a succession statement that, once distributed, cannot
    be retracted. Anyone running this script must be running it on
    purpose.
    """
    expected = "I-have-backups-and-want-to-rotate"
    actual = os.environ.get(GO_AHEAD_ENV, "")
    if actual != expected:
        _err(
            f"refusing to run without explicit confirmation. To proceed, set:\n"
            f"  export {GO_AHEAD_ENV}='{expected}'\n"
            f"and re-run. This is a one-time gate to prevent accidental "
            f"ceremony runs."
        )
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    parser.add_argument(
        "--did-domain", required=True,
        help="HTTPS-served domain for the new did:web identity, "
             "e.g. agents.kestrelsovereign.com",
    )
    parser.add_argument(
        "--did-slug", required=True,
        help="Path segment after the domain, e.g. 'kestrel'. The new "
             "DID will be did:web:<domain>:<slug>.",
    )
    parser.add_argument(
        "--effective-from", required=True,
        help="ISO 8601 UTC cutoff timestamp, e.g. "
             "2026-05-05T17:00:00+00:00. Must be in the future.",
    )
    parser.add_argument(
        "--agent-data-dir",
        type=Path,
        default=DEFAULT_AGENT_DATA,
        help=f"Path to Kestrel #1's agent data dir. Default: {DEFAULT_AGENT_DATA}",
    )
    parser.add_argument(
        "--legacy-eth-address",
        default=None,
        help="The agent's Ethereum address (0x-prefixed). Defaults to "
             "auto-detected from kestrel_0x*.json in the data dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the new did.json + succession statement. "
             "Defaults to /tmp/kestrel-1-ceremony-<timestamp>/.",
    )
    parser.add_argument(
        "--reason",
        default="Quantum Hardening Wave 3 migration (epic #921)",
        help="Free-text reason recorded in the succession statement.",
    )
    args = parser.parse_args()

    print("Kestrel #1 / Emma — hybrid-rotation ceremony (LIVE)")
    print(f"Started:        {datetime.now(timezone.utc).isoformat()}")

    # Pre-flight: env vars and confirmation
    if "KESTREL_DATA_KEY" not in os.environ:
        _err("KESTREL_DATA_KEY is required to decrypt the legacy private key.")
        return 2
    _confirm_go_ahead()

    # Pre-flight: timestamp validity
    try:
        cutoff = datetime.fromisoformat(args.effective_from)
    except ValueError as e:
        _err(f"--effective-from is not a valid ISO 8601 timestamp: {e}")
        return 2
    if cutoff.tzinfo is None:
        _err("--effective-from must be timezone-aware (e.g. ...+00:00 or ...Z)")
        return 2
    now = datetime.now(timezone.utc)
    if cutoff <= now:
        _err(f"--effective-from {args.effective_from} is not in the future "
             f"(now is {now.isoformat()}). Pick a later timestamp.")
        return 2
    minutes_ahead = (cutoff - now).total_seconds() / 60
    _info(f"cutoff is {minutes_ahead:.1f} minutes from now")

    # Pre-flight: data dir + legacy address
    agent_data = args.agent_data_dir.resolve()
    if not agent_data.is_dir():
        _err(f"agent data dir does not exist: {agent_data}")
        return 2
    eth_address = args.legacy_eth_address or _derive_eth_address_from_filenames(agent_data)
    _info(f"agent data dir: {agent_data}")
    _info(f"legacy address: {eth_address}")

    legacy_did = f"did:pkh:eip155:1:{eth_address}"

    storage = SecureKeyStorage(storage_dir=agent_data)

    # ------------------------------------------------------------------
    # Step 1: load and verify legacy keypair
    # ------------------------------------------------------------------
    _step("Step 1: load legacy ECDSA keypair from SecureKeyStorage")
    try:
        legacy_priv = storage.load_private_key(f"kestrel_{eth_address}")
    except Exception as e:
        _err(f"could not load legacy private key: {e}")
        return 1
    legacy_pub = legacy_priv.public_key()
    if not isinstance(legacy_pub, ec.EllipticCurvePublicKey):
        _err(f"loaded key is not an ECDSA key: {type(legacy_pub).__name__}")
        return 1
    if not isinstance(legacy_pub.curve, ec.SECP256K1):
        _err(f"loaded key is not on SECP256K1: {legacy_pub.curve.name}")
        return 1
    pub_hex = _public_key_to_uncompressed_hex(legacy_pub)
    _ok(f"loaded legacy ECDSA key (uncompressed pub: {pub_hex[:24]}...)")

    # Cross-check: existing DID document records this same key
    did_doc = _load_existing_did_doc(agent_data, eth_address)
    recorded_pub_hex = (
        did_doc.get("publicKey", [{}])[0].get("publicKeyHex")
        or did_doc.get("verificationMethod", [{}])[0].get("publicKeyHex")
    )
    if not recorded_pub_hex:
        _err("existing DID document has no publicKeyHex to cross-check")
        return 1
    if recorded_pub_hex.lower() != pub_hex.lower():
        _err(
            "loaded private key's public side does NOT match the public key "
            "in the existing DID document. The wrong key was loaded; refusing "
            "to sign a succession statement."
        )
        return 1
    _ok("loaded key cross-checked against existing DID document — match")

    secp = Secp256k1Suite()
    legacy_kp = Keypair(
        suite_id=secp.alg_id,
        private_key=legacy_priv,
        public_key=legacy_pub,
    )
    legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_pub)])
    legacy_kid = legacy_vms[0]["id"].rsplit("#", 1)[-1]

    # ------------------------------------------------------------------
    # Step 2: SLH-DSA archival keypair
    # ------------------------------------------------------------------
    _step("Step 2: mint SLH-DSA-SHA2-128s archival keypair")
    slh = SLHDSASHA2128sSuite()
    archival_kp = slh.generate_keypair()
    _ok("archival keypair generated (FIPS 205, hash-based)")
    _info(f"public {len(archival_kp.public_key)} bytes, "
          f"private {len(archival_kp.private_key)} bytes")

    # ------------------------------------------------------------------
    # Step 3: run the ceremony
    # ------------------------------------------------------------------
    _step("Step 3: run rotation_ceremony.run_rotation_ceremony")
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_kp,
        predecessor_kid=legacy_kid,
        predecessor_verification_methods=legacy_vms,
        new_did_domain=args.did_domain,
        new_did_slug=args.did_slug,
        reason=args.reason,
        effective_from=args.effective_from,
        archival_keypair=archival_kp,
    )
    new_did = result.new_identity.did
    _ok(f"new DID: {new_did}")
    _ok("ceremony self-verified before returning")

    # ------------------------------------------------------------------
    # Step 4: persist the new keys
    # ------------------------------------------------------------------
    # Match the persistence shape used by the dry-run + the runbook:
    # - Ed25519 is a cryptography object → save_private_key (PEM-wrapped,
    #   .key.enc). The agent's startup code uses load_private_key for
    #   classical halves, which expects PEM. Storing as raw bytes via
    #   save_secret_bytes would write to .bytes.enc and the load path
    #   would not find it. Claude-CLI review P1 catch.
    # - ML-DSA-65 + SLH-DSA are raw pqcrypto bytes → save_secret_bytes
    _step("Step 4: persist new keys via SecureKeyStorage")
    new_kp = result.new_identity.keypair
    storage.save_private_key(new_kp.classical.private_key, f"{args.did_slug}_ed25519")
    storage.save_secret_bytes(new_kp.pq.private_key, f"{args.did_slug}_mldsa65")
    storage.save_secret_bytes(archival_kp.private_key, f"{args.did_slug}_archival_slhdsa")
    storage.save_secret_bytes(archival_kp.public_key, f"{args.did_slug}_archival_slhdsa_pub")
    _ok(f"{args.did_slug}_ed25519.key.enc          (classical hybrid half, PEM)")
    _ok(f"{args.did_slug}_mldsa65.bytes.enc        (post-quantum hybrid half, raw)")
    _ok(f"{args.did_slug}_archival_slhdsa.bytes.enc(SLH-DSA archival, raw)")

    # ------------------------------------------------------------------
    # Step 5: write the new DID document + succession statement
    # ------------------------------------------------------------------
    _step("Step 5: write artifacts to output directory")
    if args.output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        args.output_dir = Path(f"/tmp/kestrel-1-ceremony-{ts}")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        _err(f"output dir {args.output_dir} already exists; refusing to clobber")
        return 1
    # Tighten umask before any writes — the succession statement contains
    # public keys (no secrets) but we keep dir access owner-only anyway.
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, mode=0o700)

    did_doc_path = args.output_dir / "did.json"
    did_doc_path.write_text(
        json.dumps(result.new_identity.did_document, indent=2, sort_keys=True),
    )
    _ok(f"did.json -> {did_doc_path} ({did_doc_path.stat().st_size} bytes)")

    succession_path = args.output_dir / "succession-statement.json"
    succession_path.write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2, sort_keys=True),
    )
    _ok(f"succession-statement.json -> {succession_path} "
        f"({succession_path.stat().st_size} bytes)")

    # Also archive the succession statement INSIDE the agent data dir
    # so the agent itself can find its own chain at startup.
    # Claude-CLI review P1: mkdir(exist_ok=True) doesn't tighten perms
    # on a pre-existing dir; explicitly chmod 0o700 after to handle
    # both fresh and existing cases.
    agent_succession_dir = agent_data / "successions"
    agent_succession_dir.mkdir(mode=0o700, exist_ok=True)
    agent_succession_dir.chmod(0o700)
    agent_succession_path = agent_succession_dir / f"{args.did_slug}.json"
    agent_succession_path.write_text(
        json.dumps(result.succession_statement.to_dict(), indent=2, sort_keys=True),
    )
    _ok(f"agent-side archive -> {agent_succession_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ceremony complete.")
    print("=" * 60)
    print()
    print(f"  Legacy DID:     {legacy_did}")
    print(f"  New DID:        {new_did}")
    print(f"  Cutoff:         {args.effective_from}  ({minutes_ahead:.0f} min from now)")
    print()
    print("Operator next steps (DO THESE BEFORE THE CUTOFF):")
    print(f"  1. Commit {did_doc_path.name} to KestrelSovereignAI/agent-identities")
    print(f"     at path {args.did_slug}/did.json:")
    print(f"       cd /path/to/agent-identities clone")
    print(f"       mkdir -p {args.did_slug}")
    print(f"       cp {did_doc_path} {args.did_slug}/did.json")
    print(f"       git add {args.did_slug}/did.json")
    print(f'       git commit -m "publish initial did.json for {new_did}"')
    print(f"       git push origin main")
    print(f"  2. Verify HTTPS resolution:")
    print(f"       curl -sI https://{args.did_domain}/{args.did_slug}/did.json")
    print(f"  3. Stop Emma if running, point her at the new keys, restart.")
    print()
    print("DO NOT destroy the legacy private key yet. Wait for a 7-day")
    print("rollback window with the new identity producing artifacts that")
    print("verify under the chain walker before secure-deleting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
