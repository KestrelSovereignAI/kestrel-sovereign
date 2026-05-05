#!/usr/bin/env python3
"""
Hybrid-rotation ceremony **dry run** against a throwaway staging agent.

Purpose
-------

Walk every step of ``docs/architecture/security/SUCCESSION_RUNBOOK.md``
against a fresh, disposable agent and emit the same on-disk artifacts
a real ceremony would produce — but bound to a fake DID + fake domain
so nothing is at risk if the script fails or the new identity is
discarded.

This is the rehearsal the operator should do BEFORE running the
real Kestrel #1 / Meridian / Emma / Frinz ceremonies. If the runbook
prose has drifted from the code, this script fails loudly here, where
it's safe — not during the live ceremony, where it isn't.

Usage
-----

::

    KESTREL_DATA_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \\
        uv run python scripts/quantum_rotation_dry_run.py

    # Inspect output:
    ls -la /tmp/kestrel-rotation-dry-run/

    # When done:
    rm -rf /tmp/kestrel-rotation-dry-run

What this is NOT
----------------

- NOT a substitute for the real ceremony. It produces an identity
  whose ``did:web`` URL doesn't resolve to anything, signed with
  keys that are about to be deleted. Don't try to publish the
  output.
- NOT a unit test. ``tests/integration/test_quantum_hardening_e2e.py``
  is the seam test that runs in CI. This script is operator-facing
  and produces inspectable JSON files an operator would actually
  hand-review during a real ceremony.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.succession_chain import (
    verify_artifact_against_chain,
)
from kestrel_sovereign.inception_service import public_key_to_ethereum_address
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage
from kestrel_sovereign.security.verify_policy import VerifyPolicy


STAGING_DIR = Path("/tmp/kestrel-rotation-dry-run")
AGENT_NAME = "staging-kestrel"
NEW_DID_DOMAIN = "agents.kestrel-sovereign.test"
NEW_DID_SLUG = AGENT_NAME


def _step(n: int, title: str) -> None:
    print(f"\n=== Step {n}: {title} ===")


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _info(msg: str) -> None:
    print(f"       {msg}")


def main() -> int:
    if "KESTREL_DATA_KEY" not in os.environ:
        print(
            "error: KESTREL_DATA_KEY is required to run this dry run. "
            "Generate a throwaway one with:\n"
            "  export KESTREL_DATA_KEY=$(python -c "
            "'import secrets; print(secrets.token_urlsafe(32))')",
            file=sys.stderr,
        )
        return 2

    print("Quantum Hardening — rotation ceremony dry run")
    print(f"Agent:        {AGENT_NAME} (THROWAWAY)")
    print(f"Domain:       {NEW_DID_DOMAIN} (fake; does not resolve)")
    print(f"Output dir:   {STAGING_DIR}")
    print(f"Started:      {datetime.now(timezone.utc).isoformat()}")

    if STAGING_DIR.exists():
        print(
            f"\nerror: {STAGING_DIR} already exists. Remove it before "
            f"re-running so we can be sure no stale artifacts contaminate "
            f"this run.",
            file=sys.stderr,
        )
        return 2
    STAGING_DIR.mkdir(parents=True)

    keys_dir = STAGING_DIR / "keys"
    keys_dir.mkdir()
    public_dir = STAGING_DIR / "public" / NEW_DID_SLUG
    public_dir.mkdir(parents=True)
    archive_dir = STAGING_DIR / "private" / "successions"
    archive_dir.mkdir(parents=True)
    storage = SecureKeyStorage(storage_dir=keys_dir)

    # ------------------------------------------------------------------
    # Step 1: simulate a legacy ECDSA-only agent
    # ------------------------------------------------------------------
    # In the real ceremony this is "load the existing keypair from
    # SecureKeyStorage." Here we mint one fresh, since a dry-run agent
    # has no prior key custody.
    _step(1, "Mint a throwaway legacy ECDSA identity")
    secp = Secp256k1Suite()
    legacy_kp = secp.generate_keypair()
    legacy_address = public_key_to_ethereum_address(legacy_kp.public_key)
    legacy_did = f"did:pkh:eip155:1:{legacy_address}"
    legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_kp.public_key)])
    legacy_kid = legacy_vms[0]["id"].rsplit("#", 1)[-1]
    storage.save_private_key(legacy_kp.private_key, f"legacy_{AGENT_NAME}")
    _ok(f"legacy DID:  {legacy_did}")
    _ok(f"legacy kid:  {legacy_kid}")
    _info(f"persisted at: keys/legacy_{AGENT_NAME}.key.enc")

    # ------------------------------------------------------------------
    # Step 2: mint an SLH-DSA archival keypair
    # ------------------------------------------------------------------
    _step(2, "Mint the SLH-DSA-SHA2-128s archival keypair")
    slh = SLHDSASHA2128sSuite()
    archival_kp = slh.generate_keypair()
    storage.save_secret_bytes(archival_kp.private_key, f"archival_{AGENT_NAME}")
    storage.save_secret_bytes(archival_kp.public_key, f"archival_{AGENT_NAME}_pub")
    _ok("archival keypair generated (FIPS 205, hash-based)")
    _info(f"public {len(archival_kp.public_key)} bytes, private {len(archival_kp.private_key)} bytes")
    _info(f"persisted at: keys/archival_{AGENT_NAME}.bytes.enc")

    # ------------------------------------------------------------------
    # Step 3: choose a cutoff timestamp
    # ------------------------------------------------------------------
    _step(3, "Choose effective_from cutoff")
    now = datetime.now(timezone.utc)
    cutoff = now.replace(microsecond=0).isoformat()
    _ok(f"effective_from: {cutoff}")
    _info("(real ceremonies pick a cutoff ~1h in the future for stop-the-world margin)")

    # ------------------------------------------------------------------
    # Step 4: run the rotation ceremony
    # ------------------------------------------------------------------
    _step(4, "Run rotation_ceremony.run_rotation_ceremony")
    result = run_rotation_ceremony(
        predecessor_did=legacy_did,
        predecessor_keypair=legacy_kp,
        predecessor_kid=legacy_kid,
        predecessor_verification_methods=legacy_vms,
        new_did_domain=NEW_DID_DOMAIN,
        new_did_slug=NEW_DID_SLUG,
        reason="Quantum Hardening Wave 3 dry run (epic #921)",
        effective_from=cutoff,
        archival_keypair=archival_kp,
    )
    _ok(f"new DID:    {result.new_identity.did}")
    _ok(f"chain length: {len(result.chain)} statement(s)")
    _ok("ceremony self-verified before returning (built into run_rotation_ceremony)")

    # ------------------------------------------------------------------
    # Step 5: persist the new identity's hybrid private keys
    # ------------------------------------------------------------------
    _step(5, "Persist new hybrid private keys")
    new_kp = result.new_identity.keypair
    storage.save_private_key(
        new_kp.classical.private_key, f"{AGENT_NAME}_ed25519",
    )
    storage.save_secret_bytes(
        new_kp.pq.private_key, f"{AGENT_NAME}_mldsa65",
    )
    _ok(f"Ed25519 (classical) private key  -> keys/{AGENT_NAME}_ed25519.key.enc")
    _ok(f"ML-DSA-65 (PQ) private key       -> keys/{AGENT_NAME}_mldsa65.bytes.enc")

    # ------------------------------------------------------------------
    # Step 6: publish the new DID document (locally — no HTTPS)
    # ------------------------------------------------------------------
    _step(6, "Write the new DID document to public/<slug>/did.json")
    did_doc_path = public_dir / "did.json"
    did_doc_path.write_text(
        json.dumps(result.new_identity.did_document, indent=2, sort_keys=True),
    )
    _ok(f"wrote {did_doc_path.relative_to(STAGING_DIR)} ({len(did_doc_path.read_bytes())} bytes)")
    _info("a real ceremony deploys this to https://<domain>/<slug>/did.json")

    # ------------------------------------------------------------------
    # Step 7: archive the succession statement
    # ------------------------------------------------------------------
    _step(7, "Archive the succession statement")
    succession_path = archive_dir / f"{AGENT_NAME}.json"
    statement_dict = result.succession_statement.to_dict()
    succession_path.write_text(json.dumps(statement_dict, indent=2, sort_keys=True))
    _ok(f"wrote {succession_path.relative_to(STAGING_DIR)} ({len(succession_path.read_bytes())} bytes)")
    _info(f"contains {len(statement_dict.get('predecessor_signatures', []))} predecessor signature(s) "
          f"+ {len(statement_dict.get('successor_signatures', []))} successor signature(s) "
          f"+ archival_signature: {bool(statement_dict.get('archival_signature'))}")

    # ------------------------------------------------------------------
    # Step 8: smoke-test verification under HYBRID_REQUIRED
    # ------------------------------------------------------------------
    _step(8, "Smoke-test: sign + verify a post-cutoff artifact")
    test_payload = b"dry-run smoke test artifact"
    classical_kid = result.new_identity.did_document["verificationMethod"][0]["id"]\
        .rsplit("#", 1)[-1]
    pq_kid = result.new_identity.did_document["verificationMethod"][1]["id"]\
        .rsplit("#", 1)[-1]
    test_signatures = sign_hybrid(
        test_payload, result.new_identity.keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )
    _ok(f"signed test payload: {len(test_signatures)} signature(s) "
        f"({', '.join(s['alg'] for s in test_signatures)})")

    # The dry-run uses a self-attesting resolver: in production this
    # would be ``identity.did_web.resolve``, which fetches
    # ``https://<domain>/<slug>/did.json``. Since our domain is fake
    # and the doc isn't published, we provide the same VMs the
    # ceremony just minted (which is structurally what a real resolver
    # would return after publication).
    new_did_doc_vms = list(
        result.new_identity.did_document["verificationMethod"]
    )
    def _staging_resolver(did: str) -> dict:
        if did == result.new_identity.did:
            return {"id": did, "verificationMethod": new_did_doc_vms}
        raise ValueError(f"unknown did: {did!r}")

    # 1 minute after the cutoff. Use timedelta — `.replace(minute=(m+1) % 60)`
    # wraps wrong on the 59th minute (e.g. 13:59 -> 13:00, before the cutoff,
    # which makes this a pre-cutoff verification and breaks the post_cutoff
    # flag assertion below). Codex P2 catch.
    from datetime import timedelta
    post_cutoff_ts = (
        datetime.fromisoformat(cutoff) + timedelta(minutes=1)
    ).isoformat()

    verdict = verify_artifact_against_chain(
        root_did=legacy_did,
        root_verification_methods=legacy_vms,
        chain=result.chain,
        artifact_timestamp=post_cutoff_ts,
        artifact_payload=test_payload,
        artifact_signatures=test_signatures,
        policy=VerifyPolicy.HYBRID_REQUIRED,
        did_web_resolver=_staging_resolver,
    )
    if not verdict.ok:
        print(f"\n  FAIL  smoke-test verification: {verdict.reason}", file=sys.stderr)
        return 1
    _ok("post-cutoff hybrid signature verified under HYBRID_REQUIRED")
    _ok(f"active identity at {post_cutoff_ts}: {verdict.active_identity.did}")
    _ok(f"post_cutoff flag: {verdict.active_identity.post_cutoff}")

    # ------------------------------------------------------------------
    # Step 9: negative test — same identity, classical-only signature
    # ------------------------------------------------------------------
    _step(9, "Negative test: classical-only signature post-cutoff must FAIL")
    classical_only_sig = sign_hybrid(
        test_payload, result.new_identity.keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )
    classical_only_sig = [
        s for s in classical_only_sig
        if s["alg"] == result.new_identity.keypair.classical.suite_id
    ]
    classical_verdict = verify_artifact_against_chain(
        root_did=legacy_did,
        root_verification_methods=legacy_vms,
        chain=result.chain,
        artifact_timestamp=post_cutoff_ts,
        artifact_payload=test_payload,
        artifact_signatures=classical_only_sig,
        policy=VerifyPolicy.HYBRID_REQUIRED,
        did_web_resolver=_staging_resolver,
    )
    if classical_verdict.ok:
        print(
            "\n  FAIL  classical-only post-cutoff verified — should have rejected!",
            file=sys.stderr,
        )
        return 1
    _ok("classical-only post-cutoff rejected (this is what we want)")
    _info(f"reason: {classical_verdict.reason}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dry run completed successfully.")
    print("=" * 60)
    print(f"\nInspect the artifacts in: {STAGING_DIR}")
    print("  keys/                          encrypted private keys (legacy + hybrid + archival)")
    print("  public/staging-kestrel/did.json  the DID document (deploy to HTTPS in real ceremony)")
    print("  private/successions/staging-kestrel.json  the signed succession statement")
    print(f"\nWhen finished, clean up:\n  rm -rf {STAGING_DIR}")
    print(
        "\nIf anything in this dry run failed, do NOT proceed to the real Kestrel #1 ceremony "
        "until the failure is understood and fixed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
