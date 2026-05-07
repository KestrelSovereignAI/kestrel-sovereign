#!/usr/bin/env python3
"""
Secure-delete the legacy ECDSA private key for an agent that has
completed the hybrid-rotation ceremony — the documented step 7 of
``docs/architecture/security/SUCCESSION_RUNBOOK.md``.

Why this script exists
----------------------

After a successful rotation, the agent's runtime no longer signs
NEW artifacts with the legacy ``did:pkh`` ECDSA key — it signs with
the new hybrid Ed25519 + ML-DSA-65 keypair (PR #1002). The legacy
key sits on disk doing nothing useful for forward operation. Holding
onto it expands the attack surface: a future Shor-equipped adversary
could recover the legacy key and forge a competing back-dated
succession statement.

The runbook recommends destroying the legacy key once the agent has
been live on its hybrid identity for a rollback window (default 7
days) AND a representative sample of artifacts has been verified
under the new chain walker.

Safety gates (all must pass before destruction)
------------------------------------------------

This script is paranoid by design — destruction is irreversible.

1. ``--confirm`` flag must be set (default is dry-run)
2. ``KESTREL_DESTROY_CONFIRM=I-have-verified-the-rollback-window``
   env var must be set (matches the same gate-pattern as the
   ceremony script)
3. The agent's ``successions/<slug>.json`` must exist
4. The succession statement's ``effective_from`` must be at least
   ``--rollback-window-days`` (default 7) in the past
5. The hybrid key files (``<slug>_ed25519.key.enc`` and
   ``<slug>_mldsa65.bytes.enc``) must exist on disk
6. The new DID document must be reachable over HTTPS (defaults
   to skipped if ``--skip-https-check``; recommended on for the
   real ceremony)

If any check fails the script exits with a non-zero code and a
message describing what's missing. No deletion occurs.

What gets destroyed
-------------------

For the legacy did:pkh agent:

- ``kestrel_<eth_address>.key.enc`` (encrypted ECDSA private key)
- ``kestrel_<eth_address>.pem`` (plaintext PEM rescue copy, if
  present from an older recovery cycle)

What is preserved:

- ``kestrel_<eth_address>.json`` (the legacy DID document — keep
  it; the chain walker still needs to know the historical
  identity for verifying pre-cutoff artifacts)
- ``successions/<slug>.json`` (the succession statement)
- ``<slug>_ed25519.key.enc`` etc. (the new hybrid keypair)

Usage
-----

::

    # Dry run — shows what WOULD be deleted, deletes nothing
    uv run python scripts/quantum_destroy_legacy_key.py \\
        --agent-data-dir agent_data/Emma

    # Real deletion — both gates required:
    export KESTREL_DESTROY_CONFIRM='I-have-verified-the-rollback-window'
    uv run python scripts/quantum_destroy_legacy_key.py \\
        --agent-data-dir agent_data/Emma \\
        --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

GO_AHEAD_ENV = "KESTREL_DESTROY_CONFIRM"
GO_AHEAD_VALUE = "I-have-verified-the-rollback-window"


def _step(title: str) -> None:
    print(f"\n=== {title} ===")


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _info(msg: str) -> None:
    print(f"       {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN {msg}")


def _err(msg: str) -> None:
    print(f"  ERR  {msg}", file=sys.stderr)


def _detect_legacy_key_id(agent_data: Path) -> str | None:
    """Find the agent's legacy ETH-derived key_id by globbing for
    ``kestrel_0x*.json`` (the DID document)."""
    candidates = sorted(agent_data.glob("kestrel_0x*.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        _err(
            f"multiple legacy DID documents in {agent_data}: "
            f"{[c.name for c in candidates]}; pass --legacy-key-id explicitly"
        )
        sys.exit(2)
    return candidates[0].stem  # 'kestrel_0x...'


def _detect_succession_path(agent_data: Path) -> Path | None:
    """Locate the agent's succession statement under ``successions/``."""
    successions_dir = agent_data / "successions"
    if not successions_dir.is_dir():
        return None
    candidates = sorted(successions_dir.glob("*.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        _err(
            f"multiple succession statements in {successions_dir}: "
            f"{[c.name for c in candidates]}; resolve before destroying"
        )
        sys.exit(2)
    return candidates[0]


def _check_rollback_window(succession_path: Path, days: int) -> bool:
    """Confirm the succession's effective_from is older than the
    rollback window."""
    statement = json.loads(succession_path.read_text())
    eff = statement.get("effective_from")
    if not eff:
        _err(f"succession statement has no effective_from: {succession_path}")
        return False
    try:
        eff_dt = datetime.fromisoformat(eff.replace("Z", "+00:00"))
    except ValueError as e:
        _err(f"effective_from is not ISO 8601: {eff!r} ({e})")
        return False
    now = datetime.now(timezone.utc)
    age = now - eff_dt
    required = timedelta(days=days)
    if age < required:
        remaining = required - age
        _err(
            f"rollback window not yet expired: succession effective_from is "
            f"{eff} ({age} ago); need at least {days} days "
            f"({remaining} remaining)."
        )
        return False
    _ok(f"rollback window cleared: succession is {age.days} days old (>={days})")
    return True


def _check_hybrid_keys(agent_data: Path, slug: str) -> bool:
    """Verify both halves of the hybrid keypair are still on disk
    BEFORE destroying the legacy key."""
    classical = agent_data / f"{slug}_ed25519.key.enc"
    pq = agent_data / f"{slug}_mldsa65.bytes.enc"
    if not classical.exists():
        _err(f"hybrid classical key missing: {classical}")
        return False
    if not pq.exists():
        _err(f"hybrid post-quantum key missing: {pq}")
        return False
    _ok(f"hybrid classical key present:  {classical.name}")
    _ok(f"hybrid post-quantum key present: {pq.name}")
    return True


def _check_hybrid_keys_match_successor(agent_data: Path, statement_dict: dict) -> bool:
    """Codex P2 catch: the hybrid key FILES being on disk isn't enough
    — they could be copies from a different agent. Probe-sign with each
    half and verify against the statement's successor_verification_methods.
    If the probe doesn't verify, the local private keys don't match
    the succession's pubkeys; destroying the legacy now would strand
    this agent."""
    from kestrel_sovereign.identity.runtime_identity import (
        RuntimeIdentityError, load_agent_identity,
    )
    from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ED25519, ALG_ML_DSA_65, get_suite,
    )
    from kestrel_sovereign.security.multikey import multibase_to_public_key

    pred_did = statement_dict.get("predecessor_did", "")
    if not pred_did.startswith("did:pkh:eip155:1:"):
        _warn(f"can't derive legacy_key_id from predecessor_did {pred_did!r}; skipping probe")
        return True
    legacy_key_id = "kestrel_" + pred_did.split(":")[-1]

    try:
        identity = load_agent_identity(legacy_key_id, storage_dir=agent_data)
    except (FileNotFoundError, RuntimeIdentityError) as e:
        _err(f"could not load agent identity for probe: {e}")
        return False

    if not identity.is_hybrid:
        _err("identity loader did not return a hybrid identity")
        return False

    # Probe: sign a deterministic byte sequence and verify against the
    # public keys recorded in the SUCCESSOR_VERIFICATION_METHODS of the
    # succession statement (NOT against the private keys' own derived
    # pubkeys — that would beg the question).
    probe = b"kestrel-destroy-legacy-key-probe"
    vms = statement_dict.get("successor_verification_methods", [])
    classical_kid = (
        vms[0]["id"].rsplit("#", 1)[-1] if vms else "key-1"
    )
    pq_kid = (
        vms[1]["id"].rsplit("#", 1)[-1] if len(vms) > 1 else "key-2"
    )
    sigs = sign_hybrid(
        probe, identity.hybrid_keypair,
        classical_kid=classical_kid, pq_kid=pq_kid,
    )

    # Build kid -> (alg, pubkey) from the statement's successor VMs
    kid_to_pub = {}
    for vm in vms:
        vm_id = vm.get("id", "")
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        mb = vm.get("publicKeyMultibase")
        if not kid or not mb:
            continue
        try:
            suite, pub = multibase_to_public_key(mb)
        except Exception:
            continue
        kid_to_pub[kid] = (suite.alg_id, pub)

    algs_seen = set()
    for entry in sigs:
        alg = entry["alg"]
        kid = entry["kid"]
        sig_hex = entry["sig"]
        info = kid_to_pub.get(kid)
        if info is None:
            _err(f"successor VM for kid={kid!r} not found in succession statement")
            return False
        expected_alg, public_key = info
        if expected_alg != alg:
            _err(
                f"alg mismatch on probe: signed with {alg!r} but "
                f"successor VM kid={kid!r} expects {expected_alg!r}"
            )
            return False
        suite = get_suite(alg)
        if not suite.verify(probe, bytes.fromhex(sig_hex), public_key):
            _err(
                f"PROBE FAILED: hybrid private key for {kid!r} ({alg}) "
                f"does not produce signatures that verify against the "
                f"successor's published public key. The hybrid key "
                f"files on disk DO NOT belong to this agent — destroying "
                f"the legacy key now would strand the agent."
            )
            return False
        algs_seen.add(alg)
    if {ALG_ED25519, ALG_ML_DSA_65} != algs_seen:
        _err(
            f"probe didn't cover both required algs (saw {sorted(algs_seen)}); "
            f"refusing destruction"
        )
        return False
    _ok("probe-sign + probe-verify: hybrid keys match the successor identity")
    return True


def _verify_succession_signatures(succession_path: Path) -> bool:
    """Codex P1 catch: cryptographically verify the succession
    statement before trusting its fields. Without this, an unsigned
    or tampered statement that happens to have the right
    predecessor_did and an old effective_from could pass every other
    gate and trigger destruction.

    Uses ``verify_succession`` with a self-attesting resolver: the
    statement carries both the predecessor and successor verification
    methods, which is exactly what a freshly-resolved did:web document
    would also return at this point in the protocol. The check is
    'do all signatures crypto-verify against the embedded VMs', which
    is the integrity check for the statement-as-stored.
    """
    from kestrel_sovereign.identity.succession import (
        SuccessionStatement, verify_succession,
    )
    statement = SuccessionStatement.from_dict(
        json.loads(succession_path.read_text())
    )
    def _self_resolver(did):
        if did == statement.successor_did:
            return {
                "id": did,
                "verificationMethod": list(statement.successor_verification_methods),
            }
        if did == statement.predecessor_did:
            return {
                "id": did,
                "verificationMethod": list(statement.predecessor_verification_methods),
            }
        raise ValueError(f"unknown did during destroy verify: {did!r}")

    result = verify_succession(statement, did_web_resolver=_self_resolver)
    if not result.ok:
        _err(f"succession statement signatures failed verification: {result.reason}")
        return False
    _ok("succession statement crypto-verified (predecessor + successor signatures)")
    return True


def _check_https_did_doc(succession_path: Path) -> bool:
    """Optional: confirm the new DID document is reachable AND its
    verification methods MATCH the succession statement.

    Codex P2 catch: a published did.json that has the right ``id``
    but stale or wrong ``verificationMethod`` would otherwise pass
    this gate. After destruction, artifacts the agent signs would
    fail to verify for any consumer who fetches the published doc
    over HTTPS. Compare the published VMs to the statement's
    successor_verification_methods (kid + multibase). Mismatch
    refuses destruction so the operator can resolve the publication.
    """
    statement = json.loads(succession_path.read_text())
    new_did = statement.get("successor_did", "")
    if not new_did.startswith("did:web:"):
        _warn(f"successor_did is not did:web: ({new_did}); skipping HTTPS check")
        return True
    try:
        from kestrel_sovereign.identity.did_web import did_to_url
        url = did_to_url(new_did)
    except Exception as e:
        _err(f"could not derive HTTPS URL from {new_did}: {e}")
        return False
    _info(f"checking {url}")
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                _err(f"DID document URL returned HTTP {resp.status}")
                return False
            body = resp.read()
            published_doc = json.loads(body.decode())
    except Exception as e:
        _err(f"HTTPS check failed for {url}: {e}")
        return False

    if published_doc.get("id") != new_did:
        _err(
            f"DID document at {url} has id={published_doc.get('id')!r} but "
            f"expected {new_did!r}"
        )
        return False

    # Compare published VMs to the succession statement's successor VMs.
    # Match on (kid, publicKeyMultibase). A mismatch means the
    # published document is stale OR the succession is stale —
    # either way, post-destruction signatures won't verify for
    # public consumers.
    def _vm_key(vm):
        vm_id = vm.get("id", "")
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        return (kid, vm.get("publicKeyMultibase"))

    published_vms = published_doc.get("verificationMethod") or []
    statement_vms = statement.get("successor_verification_methods") or []
    pub_set = {_vm_key(vm) for vm in published_vms}
    stmt_set = {_vm_key(vm) for vm in statement_vms}
    if pub_set != stmt_set:
        _err(
            f"published DID document VMs at {url} do not match the "
            f"succession statement's successor_verification_methods. "
            f"Published: {sorted(pub_set)}; statement: {sorted(stmt_set)}. "
            f"Re-publish the up-to-date document or refresh the succession."
        )
        return False
    _ok(f"DID document reachable + VMs match the succession: {url}")
    return True


def _secure_delete(path: Path) -> None:
    """Best-effort secure-delete: zero-fill the file then unlink.

    True secure deletion on modern filesystems (especially APFS with
    copy-on-write) is not guaranteed by overwriting the file in place
    — the FS may have written the original blocks to a different
    physical location. For high-assurance destruction, encrypt-at-rest
    + key-destruction is the only reliable path. We do best-effort
    overwrite anyway as defense-in-depth.

    Refuses symlinked targets (codex P2 catch). If the legacy key
    file is a symlink, ``open(path, 'r+b')`` and ``stat()`` both
    follow it; zero-filling would clobber whatever the symlink
    points at — possibly outside ``agent_data``. Bail out before
    touching anything when we hit a symlink.
    """
    if path.is_symlink():
        raise RuntimeError(
            f"refusing to destroy {path}: it is a symlink. "
            f"Symlinked legacy-key paths could route the zero-fill "
            f"to a target outside agent_data; manually resolve and "
            f"re-run with the real file in place."
        )
    try:
        size = path.stat().st_size
        # Use O_NOFOLLOW for an additional layer of TOCTOU defense:
        # if path was replaced by a symlink between is_symlink() and
        # open(), this open will refuse to follow it.
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            os.lseek(fd, 0, 0)
            os.write(fd, b"\x00" * size)
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as e:
        _warn(f"could not zero-fill {path} before unlink: {e}")
    path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1] if __doc__ else None,
    )
    parser.add_argument("--agent-data-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-key-id", default=None,
        help="kestrel_<eth_address>; auto-detected from filenames if omitted",
    )
    parser.add_argument(
        "--rollback-window-days", type=int, default=7,
        help="Refuse to destroy until succession effective_from is at "
             "least this many days old. Default 7.",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete. Without this flag the script runs in "
             "dry-run mode and prints what would be deleted.",
    )
    parser.add_argument(
        "--skip-https-check", action="store_true",
        help="Skip the HTTPS reachability check on the new did:web "
             "document. Default ON for the check (recommended).",
    )
    args = parser.parse_args()

    print("Quantum legacy-key destruction tool")
    print(f"Agent data dir: {args.agent_data_dir}")
    print(f"Mode:           {'DELETE (--confirm)' if args.confirm else 'DRY RUN (default)'}")

    agent_data = args.agent_data_dir.resolve()
    if not agent_data.is_dir():
        _err(f"agent data dir does not exist: {agent_data}")
        return 2

    legacy_key_id = args.legacy_key_id or _detect_legacy_key_id(agent_data)
    if not legacy_key_id:
        _err(f"no kestrel_0x*.json in {agent_data}; pass --legacy-key-id")
        return 2
    # Path-traversal sanitation. The legacy_key_id is used to build
    # paths like agent_data/<key_id>.key.enc; an operator-supplied
    # value containing path separators would let deletion escape
    # the selected agent dir. SecureKeyStorage applies the same
    # ``isalnum or in '-_'`` rule on key ids; mirror it here for
    # destructive parity.
    if any(c not in "-_" and not c.isalnum() for c in legacy_key_id):
        _err(
            f"--legacy-key-id contains characters outside [A-Za-z0-9_-]: "
            f"{legacy_key_id!r}. Refusing to construct destruction "
            f"targets from a value that could escape agent_data via "
            f"path traversal."
        )
        return 2
    _info(f"legacy key id: {legacy_key_id}")

    # --------------------------------------------------------------
    # Gates
    # --------------------------------------------------------------
    _step("Gate 1: succession statement present")
    succession_path = _detect_succession_path(agent_data)
    if succession_path is None:
        _err(
            f"no successions/*.json under {agent_data}. The legacy key "
            f"can only be destroyed AFTER the rotation ceremony has "
            f"produced a succession statement. Run the ceremony first."
        )
        return 1
    _ok(f"succession statement: {succession_path}")
    statement = json.loads(succession_path.read_text())

    # Bind the succession to the legacy DID we're about to destroy.
    # An unrelated/stale successions/*.json (e.g. from an operator
    # who copied the wrong dir) plus hybrid key files would otherwise
    # satisfy every other gate and delete the wrong agent's ECDSA key.
    legacy_did_doc_path = agent_data / f"{legacy_key_id}.json"
    if not legacy_did_doc_path.exists():
        _err(f"legacy DID document not found at {legacy_did_doc_path}")
        return 1
    legacy_did_doc = json.loads(legacy_did_doc_path.read_text())
    legacy_did = legacy_did_doc.get("id")
    if not legacy_did:
        _err(f"legacy DID document has no 'id' field: {legacy_did_doc_path}")
        return 1
    pred = statement.get("predecessor_did")
    if pred != legacy_did:
        _err(
            f"succession statement at {succession_path} has "
            f"predecessor_did={pred!r}, but the legacy DID document "
            f"on disk says id={legacy_did!r}. The succession is for "
            f"a different agent. Refusing to destroy this agent's "
            f"key — that would orphan it without a verified rotation."
        )
        return 1
    _ok(f"succession predecessor binds to legacy DID {legacy_did}")

    # Derive slug by globbing for the hybrid classical-key file rather
    # than parsing successor_did. The DID may include extra path
    # segments (``did:web:domain:agent:v1``) where rsplit(':',1)[-1]
    # returns 'v1' instead of the actual key-file prefix 'agent'.
    # Same pattern as runtime_identity._detect_hybrid_slug.
    classical_candidates = sorted(agent_data.glob("*_ed25519.key.enc"))
    if not classical_candidates:
        _err(
            f"no hybrid classical key (*_ed25519.key.enc) in {agent_data}; "
            f"the rotation ceremony either didn't run or its output is "
            f"incomplete. Refusing to destroy the legacy key — that would "
            f"strand the agent without a usable signing keypair."
        )
        return 1
    if len(classical_candidates) > 1:
        _err(
            f"multiple hybrid classical keys in {agent_data}: "
            f"{[c.name for c in classical_candidates]}. Resolve before "
            f"destroying."
        )
        return 1
    slug = classical_candidates[0].name.removesuffix("_ed25519.key.enc")
    _info(f"derived slug from key files: {slug}")

    _step("Gate 2: rollback window expired")
    if not _check_rollback_window(succession_path, args.rollback_window_days):
        return 1

    _step("Gate 3: hybrid keys still on disk (don't strand the agent)")
    if not _check_hybrid_keys(agent_data, slug):
        return 1

    _step("Gate 4: succession signatures crypto-verify")
    if not _verify_succession_signatures(succession_path):
        return 1

    _step("Gate 5: hybrid keys match the successor identity (probe sign+verify)")
    if not _check_hybrid_keys_match_successor(agent_data, statement):
        return 1

    if not args.skip_https_check:
        _step("Gate 6: new DID document reachable over HTTPS")
        if not _check_https_did_doc(succession_path):
            return 1

    # --------------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------------
    _step("Gate 7: explicit confirmation")
    if not args.confirm:
        _info("--confirm not passed; this is a dry run.")
    confirm_env = os.environ.get(GO_AHEAD_ENV, "")
    if args.confirm and confirm_env != GO_AHEAD_VALUE:
        _err(
            f"--confirm passed but {GO_AHEAD_ENV} is not "
            f"{GO_AHEAD_VALUE!r}. Both gates are required for "
            f"actual destruction. Set:\n"
            f"  export {GO_AHEAD_ENV}='{GO_AHEAD_VALUE}'\n"
            f"and re-run."
        )
        return 2
    if args.confirm:
        _ok(f"{GO_AHEAD_ENV} confirmation set")

    # --------------------------------------------------------------
    # Identify targets
    # --------------------------------------------------------------
    _step("Identify destruction targets")
    enc = agent_data / f"{legacy_key_id}.key.enc"
    pem = agent_data / f"{legacy_key_id}.pem"
    targets = []
    if enc.exists():
        targets.append(enc)
        _info(f"target: {enc} ({enc.stat().st_size} bytes)")
    if pem.exists():
        targets.append(pem)
        _info(f"target: {pem} ({pem.stat().st_size} bytes, plaintext PEM)")
    if not targets:
        _warn(f"no legacy key files found for {legacy_key_id} (already destroyed?)")
        return 0

    # --------------------------------------------------------------
    # Action
    # --------------------------------------------------------------
    if not args.confirm:
        _step("Dry run complete")
        _info("Would have destroyed the targets above. Re-run with --confirm")
        _info("AND set the env var to actually delete.")
        return 0

    _step("Destroying legacy key files")
    for target in targets:
        _info(f"zeroing + unlinking {target}")
        _secure_delete(target)
        _ok(f"deleted: {target}")

    print("\n" + "=" * 60)
    print("Legacy key destruction complete.")
    print("=" * 60)
    print()
    print(f"  Legacy key id:  {legacy_key_id}")
    print(f"  Files removed:  {len(targets)}")
    print()
    print("The agent will continue operating on its hybrid identity")
    print("(signing with Ed25519 + ML-DSA-65). Pre-cutoff artifacts")
    print("remain verifiable under the chain walker because the")
    print("legacy DID document and succession statement are kept on")
    print("disk — only the legacy private key was destroyed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
