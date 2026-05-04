# Hybrid-Identity Rotation Runbook

> **Wave 3 of Quantum Hardening (#921, #918)** — operational procedure for migrating a legacy `did:pkh` Kestrel agent to a hybrid `did:web` identity (Ed25519 + ML-DSA-65) via a signed succession statement.

## Why this runbook exists

Pre-Wave-3 agents (Kestrel #1, Emma, Meridian, current Frinz tenants) carry classical-only ECDSA secp256k1 identities under `did:pkh:eip155:1:0x…`. Wave 2 shipped the new hybrid `did:web` path (#917), but only for new agents. This runbook covers migrating each existing agent: minting their new identity, building a cryptographically signed bridge, and gradually retiring the legacy key.

## Why this is deployment-timing critical

The succession statement carries a `effective_from` timestamp. A future verifier compares artifact timestamps to this cutoff: anything dated *after* the cutoff must carry at least one post-quantum signature, otherwise it is rejected (`post_cutoff_classical_allowed=False`).

**The window to migrate is "before HNDL becomes practical."** If we delay until a Shor-equipped adversary has already recovered a predecessor's ECDSA key, the adversary can produce a competing back-dated succession statement and fork the chain. NIST projects 2030+ for cryptographically-relevant quantum computers; current best practice is to migrate the long-tail of legacy agents in 2026–2027.

## Pre-flight checks

Run these before touching keys:

1. **Confirm the new identity's domain is reachable over HTTPS**. `did:web` resolution requires the DID document to be published at `https://<domain>/<slug>/did.json` — without that, no verifier can fetch it.
2. **Backup the legacy private key**, encrypted, to two independent locations. Until the new identity is live and has signed at least one artifact post-cutoff, the legacy key is the only thing that can re-anchor the agent.
3. **Mint an SLH-DSA-SHA2-128s archival keypair**. Strongly recommended — gives the succession statement conservative-tier durability even if both Wave 2 hybrid suites are later broken. The runbook below assumes you've done this.
4. **Plan the cutoff timestamp**. Pick a UTC time at least an hour in the future to give yourself a stop-the-world window for last-minute corrections, but not so far ahead that the agent operates in a "dual identity" state for long. 1 hour ahead is typical.

## The ceremony

```python
from kestrel_sovereign.identity.rotation_ceremony import run_rotation_ceremony
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.crypto_suite import (
    Secp256k1Suite,
    SLHDSASHA2128sSuite,
)
from kestrel_sovereign.security.key_storage import SecureKeyStorage

# 1) Load the legacy keypair from secure storage
storage = SecureKeyStorage(storage_dir=AGENT_DATA_DIR)
legacy_priv = storage.load_private_key(f"kestrel_{ETH_ADDRESS}")
legacy_pub = legacy_priv.public_key()
legacy_kp = ...  # wrap as Keypair(suite_id="ecdsa-secp256k1-sha256", ...)

# 2) Build legacy verification methods
secp = Secp256k1Suite()
legacy_did = "did:pkh:eip155:1:" + ETH_ADDRESS
legacy_vms = build_verification_methods(legacy_did, [(secp, legacy_pub)])
legacy_kid = legacy_vms[0]["id"].rsplit("#", 1)[-1]

# 3) Mint the SLH-DSA archival keypair (one-time, store separately)
slh = SLHDSASHA2128sSuite()
archival_kp = slh.generate_keypair()
storage.save_private_key(archival_kp.private_key, f"archival_slhdsa_{AGENT_NAME}")

# 4) Run the ceremony
result = run_rotation_ceremony(
    predecessor_did=legacy_did,
    predecessor_keypair=legacy_kp,
    predecessor_kid=legacy_kid,
    predecessor_verification_methods=legacy_vms,
    new_did_domain="kestrel-sovereign.example",
    new_did_slug=AGENT_SLUG,                   # e.g. "kestrel" / "emma" / "meridian"
    reason="Quantum Hardening Wave 3 migration (epic #921)",
    effective_from="2026-06-01T17:00:00+00:00",   # planned cutoff
    archival_keypair=archival_kp,
)

# 5) Persist the new identity's private keys
new_kp = result.new_identity.keypair
storage.save_private_key(new_kp.classical.private_key, f"{AGENT_SLUG}_ed25519")
storage.save_pq_secret(new_kp.pq.private_key, f"{AGENT_SLUG}_mldsa65")

# 6) Publish the new DID document
import json
with open(f"public/{AGENT_SLUG}/did.json", "w") as f:
    json.dump(result.new_identity.did_document, f, indent=2)
# (deploy via your usual web-publish path so HTTPS GET resolves it)

# 7) Archive the succession statement
with open(f"private/successions/{AGENT_SLUG}.json", "w") as f:
    json.dump(result.succession_statement.to_dict(), f, indent=2)
```

## Verification (do this NOW, before the cutoff)

```python
from kestrel_sovereign.identity.succession_chain import verify_artifact_against_chain
from kestrel_sovereign.security.verify_policy import VerifyPolicy

# Sign a test artifact with the new hybrid keypair
from kestrel_sovereign.identity.hybrid_keypair import sign_hybrid
test_payload = b"post-rotation smoke test"
test_signatures = sign_hybrid(test_payload, result.new_identity.keypair)

# Verify under the chain at a post-cutoff timestamp
verdict = verify_artifact_against_chain(
    root_did=legacy_did,
    root_verification_methods=legacy_vms,
    chain=result.chain,
    artifact_timestamp="2026-06-01T18:00:00+00:00",  # AFTER effective_from
    artifact_payload=test_payload,
    artifact_signatures=test_signatures,
    policy=VerifyPolicy.HYBRID_REQUIRED,
)
assert verdict.ok, verdict.reason
```

If this fails, something in steps 1–5 is wrong. **Do not proceed to the cutoff** until smoke-test verification is green.

## Cutoff window: what changes at `effective_from`

| Action                                          | Pre-cutoff | Post-cutoff |
| :---                                            | :---:       | :---:        |
| New agent signs identity-package with hybrid    | OK          | OK           |
| New agent signs constitution checkpoint hybrid  | OK          | OK           |
| Legacy key signs anything                       | OK          | **REJECTED** |
| Verifier walks chain to find new agent's keys   | n/a         | YES          |
| Classical-only signature from new agent         | OK          | **REJECTED** |

The `post_cutoff_classical_allowed=False` rule in `verify_policy.py` is what enforces the rejection on the verifier side. Every consumer of signed artifacts (constitution audit, identity-package import, spawn-mandate verify, capsule import) calls into the chain walker, which threads the cutoff through automatically.

## Post-cutoff: legacy key destruction

Once the new identity has been live for the rollback window (e.g. 7 days) and you've verified a representative sample of artifacts under the chain walker, secure-delete the legacy private key:

```python
from kestrel_sovereign.security.key_storage import secure_delete

secure_delete(AGENT_DATA_DIR / f"kestrel_{ETH_ADDRESS}.key.enc")
```

**Do not auto-delete from the ceremony itself.** Operators should take this destructive step with eyes open, after manual verification.

## Rollback: what if something is wrong

If the new identity's DID document doesn't resolve, or smoke-test verification fails:

1. **Before cutoff**: throw away the new identity bundle. The succession statement isn't published yet, so no consumer is affected. Restart the ceremony.
2. **After cutoff**: the chain walker is now rejecting classical-only artifacts from the legacy key. You CAN still sign new things with the legacy key, but they won't be accepted by chain-aware verifiers. The remediation path is to issue a *second* succession statement (legacy → new-take-2), with a later `effective_from`, signed by the still-live legacy key. This appends a second link to the chain. Document the corrupted middle link in audit logs.

## Order of operations for the four current agents

Recommended migration order (smallest blast radius first):

1. **Kestrel #1** — operator (UncleSaurus) is the only consumer; rollback is cheap if anything goes wrong
2. **Meridian** — newer agent, simpler artifact history
3. **Emma** — has more migration history but no current dependents
4. **Frinz tenants** — highest blast radius (multiple users); migrate after the protocol has been proven on the three above

Stage each migration at least 24 hours apart so any verifier-side issue surfaces before the next one fires.

## Related modules

- [`kestrel_sovereign/identity/rotation_ceremony.py`](../../../kestrel_sovereign/identity/rotation_ceremony.py) — the ceremony orchestrator
- [`kestrel_sovereign/identity/succession.py`](../../../kestrel_sovereign/identity/succession.py) — `SuccessionStatement` data structure + per-statement signer/verifier
- [`kestrel_sovereign/identity/succession_chain.py`](../../../kestrel_sovereign/identity/succession_chain.py) — chain walker with temporal-cutoff hook
- [`kestrel_sovereign/security/verify_policy.py`](../../../kestrel_sovereign/security/verify_policy.py) — `post_cutoff_classical_allowed` enforcement point
- [`tests/unit/test_rotation_ceremony.py`](../../../tests/unit/test_rotation_ceremony.py) — exercises the full flow with mock keys
