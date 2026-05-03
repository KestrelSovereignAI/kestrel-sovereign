# Post-Quantum Cryptography Migration — PRD v2

**Status:** Active. Supersedes the 2024 aspirational draft (preserved in git history).
**Tracking:** Epic [`#921`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/921). Wave issues `#913–#920`.
**Companion docs:** [`CRYPTO_INVENTORY.md`](CRYPTO_INVENTORY.md), [`PQ_THREAT_MODEL.md`](PQ_THREAT_MODEL.md), [`SERIALIZATION_COMPATIBILITY.md`](SERIALIZATION_COMPATIBILITY.md).

## 1. Purpose

Migrate Kestrel cryptography to be secure against both classical and quantum adversaries, **without orphaning existing agents** and without trusting unaged lattice assumptions any more than necessary.

The promise of *lifelong* sovereignty extends the threat horizon to decades. A cryptographically-relevant quantum computer (CRQC) is a 2035–2045 problem that we cannot afford to ignore in 2026.

## 2. Why the 2024 PRD needed rewriting

The original draft was a one-page sketch. Reviewers and inventory work surfaced eight substantive errors:

1. "Burn the boats" with no continuity story orphaned every existing agent.
2. Pure post-quantum (no hybrid) contradicted CNSA 2.0 / IETF transition guidance.
3. The DID method change (`did:pkh:eip155:1` is structurally welded to Ethereum and secp256k1) was not addressed.
4. ML-KEM "for at-rest data" misframed the threat — local AES-GCM with locally-derived keys is not HNDL-vulnerable. The real KEM target is **recipient-wrapped** export and sharing. Note: agent CAR export already exists today (`SovereignStorageAdapter.export_agent`, [`storage/sovereign_adapter.py:252+`](../../../kestrel_sovereign/storage/sovereign_adapter.py#L252)) — encrypted shards/assets/keyring packed into a CARv1 archive, uploaded to IPFS/Filecoin. What's missing is wrapping the keyring to a **recipient's** public key; today the keyring is encrypted to the agent's own `KESTREL_DATA_KEY`-derived key, so "export" today is really "self-archive." Wave 4 introduces recipient KEM wrapping. Until then, every `export_agent` upload is HNDL-relevant if the CAR ever leaves the originating agent's custody.
5. Spawn-mandate signing ([`spawn/mandate.py:77-89`](../../../kestrel_sovereign/spawn/mandate.py#L77-L89)) and the public-HMAC tamper-tag ([`features/compute/script_signer.py:170`](../../../kestrel_sovereign/features/compute/script_signer.py#L170)) were missed.
6. ML-DSA + SLH-DSA required everywhere was overkill — SLH-DSA is for irrevocable, long-horizon checkpoints, not routine signing.
7. Verifier-rejects-all-unverified-sigs was too strict for migration; legacy import and offline recovery need policy modes.
8. liboqs-python was named without evaluation, and FIPS validation was implicitly gating abstraction work.

This PRD-v2 corrects all eight.

## 3. Design principles

**P1. Hybrid-first.** Classical + PQ in parallel during transition. Pure-PQ deferred indefinitely; lattice assumptions need a decade more cryptanalysis seasoning. If a structural lattice break appears, the classical half buys time for re-migration.

**P2. Continuity over clean break.** Existing agents migrate via succession statements. Kestrel #1, Emma, Meridian carry their full history forward. The sovereignty promise demands no orphaned identities.

**P3. Versioned containers everywhere.** Every signed/encrypted artifact carries `{v, alg, ...}`. The serialization matrix in [`SERIALIZATION_COMPATIBILITY.md`](SERIALIZATION_COMPATIBILITY.md) is the foundation; every wave fills in another row.

**P4. Policy-based verification.** `LEGACY_ALLOWED | HYBRID_REQUIRED | PQ_REQUIRED` per context, default-tightening over releases (schedule in §8). Not a global flag day.

**P5. HNDL where it actually applies.** Local AES-GCM with locally-derived keys is fine. The KEM target is **exports and shared capsules**, not the local DB.

**P6. Wallet keys are non-authoritative for identity.** Filecoin/EVM keys are chain-bound (must remain secp256k1) and explicitly cannot sign any spawn mandate, constitution anchor, or identity package. Their inevitable post-quantum break must not break Kestrel identity.

**P7. Abstraction before library choice.** `CryptoSuite` + KAT vectors land in Wave 1. Production-default selection of a specific PQ library is gated on FIPS validation maturity, **separately** from abstraction work.

**P8. SLH-DSA is reserved.** Not an everywhere-required dual-sign. Used only for: succession ceremonies (Wave 3), long-horizon constitution checkpoints (Wave 3), release signing (Wave 5). Slow signing/verification is acceptable for these once-in-a-while events; security-assumption diversity is the point.

## 4. Algorithm choices

| Role | Primary | Backup / long-horizon | Rationale |
|---|---|---|---|
| Routine signature (identity, mandate, audit) | Ed25519 + ML-DSA-65 (composite, separate verification methods) | — | Hybrid; classical Ed25519 has decades of cryptanalysis; ML-DSA-65 (NIST Cat-3) is balanced for sig size (~3.3 KB) vs strength |
| Succession / checkpoint / release | SLH-DSA-SHA2-128s (countersignature) | — | Hash-based, stateless; conservative assumptions diverging from lattices; slow (~ms signing) but acceptable for irrevocable events |
| Key encapsulation (export, sharing) | X25519 + ML-KEM-768 (hybrid combiner) | — | Pinned combiner construction (Chempat or X-Wing — Wave 4 selects exactly one and documents) |
| Symmetric AEAD | AES-256-GCM | — | Grover halves to 128-bit effective; acceptable. Replaces all Fernet sites in Wave 0C. |
| KDF | HKDF-SHA256 (for AEAD), PBKDF2-HMAC-SHA256 with ≥600k iterations (password-derived contexts) | Argon2id (separate hygiene track) | HKDF for in-system key derivation; PBKDF2 retained for `KESTREL_DATA_KEY` until Argon2id track lands |
| Hash | SHA-256, BLAKE2b-256 | SHA-512 in some long-horizon anchor contexts | Existing usage retained; long-horizon anchors may dual-hash |

## 5. DID method

`did:pkh:eip155:1` cannot host PQ keys (the DID *is* `keccak(secp256k1_pubkey)[-20:]`). Two viable alternatives evaluated:

- **`did:web` with full DID document carrying multiple `verificationMethod` entries (W3C Multikey).** Path-based DIDs resolve per the W3C did:web method spec: `did:web:agents.kestrel.sh:<agent>` → `https://agents.kestrel.sh/<agent>/did.json` (path components after the host map directly to URL path; `/.well-known/did.json` is the resolution path only for *bare-domain* DIDs like `did:web:agents.kestrel.sh`). Standards-aligned, supports multi-key naturally, key rotation is just a DID document update.
- **`did:key` with a composite multicodec.** Cleaner self-contained URI but requires W3C-blessed multicodec for hybrid composites; that work is in flight but not landed.

**Decision: `did:web`.** Path of least resistance to ship; revisit `did:key` once W3C lands a hybrid multicodec. Configuration flag `KESTREL_IDENTITY_METHOD` to allow opt-in to legacy `did:pkh` only during Wave 2 transition (deprecation warning attached).

## 6. Continuity model

Every existing agent gets exactly one **succession ceremony** during Wave 3:

```
KeyRotation {
  old_did:        did:pkh:eip155:1:0x...
  new_did:        did:web:agents.kestrel.sh:<agent-name>
  old_pub:        secp256k1 uncompressed point
  new_pubs:       [Ed25519 multibase, ML-DSA-65 multibase, SLH-DSA multibase]
  effective_from: <ISO timestamp>
  reason:         "post-quantum migration"
  signatures: [
    { alg: "ecdsa-secp256k1-sha256", kid: <old kid>, sig: ... },
    { alg: "ed25519",                kid: <new ed25519 kid>, sig: ... },
    { alg: "ml-dsa-65",              kid: <new mldsa kid>, sig: ... },
    { alg: "slh-dsa-sha2-128s",      kid: <new slhdsa kid>, sig: ... }
  ]
}
```

Verifier walks the chain: trust the current sig OR trust a valid succession path back to a previously-trusted key. Cycle detection mandatory (no key may appear twice in a chain). Kestrel #1, Emma, Meridian are the canonical first three migrations; each gets a manual review.

## 7. Verify-policy modes

Defined as an enum in `security/verify_policy.py` (Wave 1):

- `LEGACY_ALLOWED` — accept v1 single-classical signature.
- `HYBRID_REQUIRED` — require v2 with at least one classical + one PQ signature.
- `PQ_REQUIRED` — reject anything without a PQ signature.

Per-context defaults, **not** a global flag:

| Context | Default policy at landing | Tightens to |
|---|---|---|
| Archival import (read-only) | `LEGACY_ALLOWED` | Stays |
| Live identity assertion | `LEGACY_ALLOWED` (Wave 0–1) | `HYBRID_REQUIRED` once Wave 2 lands |
| New identity-package issuance | `HYBRID_REQUIRED` once Wave 2 lands | Stays |
| Spawn-mandate verification | `LEGACY_ALLOWED` | `HYBRID_REQUIRED` once parent has rotated (Wave 3) |
| Constitution audit (routine) | Inherits agent's identity policy | — |
| Constitution checkpoint (rotation events) | `PQ_REQUIRED` once Wave 3 lands | — |

The policy modes apply *within* a key's effectivity window. Temporal-validity rules in [`SERIALIZATION_COMPATIBILITY.md`](SERIALIZATION_COMPATIBILITY.md#temporal-validity-rules) govern how succession chains map an artifact's timestamp to the key authoritative at that moment. In particular: `HYBRID_REQUIRED` against a legacy-only in-window key softens to "verify the classical signature on the legacy key" — historical artifacts signed before a parent's rotation remain verifiable without a re-signing pass. This is the resolution of the otherwise contradictory pair "tighten to `HYBRID_REQUIRED` after rotation" and "do not re-sign historical mandates."

## 8. Default-tightening release schedule

Tentative; finalize in PRD-v3 once Wave 2 ships and benchmarks land:

| Release | Default change |
|---|---|
| `v0.X` (Wave 0 lands) | Hygiene only; classical-only still default. |
| `v0.Y` (Wave 1 lands) | All artifacts written as v2 containers; verifier per-context defaults still permissive. |
| `v0.Z` (Wave 2 lands) | New agents default to `did:web` + hybrid keys. `KESTREL_IDENTITY_METHOD=did:pkh` requires explicit opt-in with deprecation warning. |
| `v1.0` (Wave 3 lands) | Live identity assertion default flips to `HYBRID_REQUIRED`. Legacy agents that haven't run succession ceremony get a startup warning. |
| `v1.1` (Wave 4 lands) | All export/sharing surfaces use hybrid KEM by default. |
| `v1.2` (Wave 5 lands) | Releases signed with SLH-DSA; verification tooling published. |

## 9. Library bake-off

Decision criteria: FIPS validation status, API stability, key/sig sizes on disk, signing/verification latency on M-series and Cloud Run x86, **prebuilt wheels available (no compile-on-deploy)**, supply-chain provenance.

Candidates:

- **`pqcrypto`** — CFFI bindings to PQClean-derived C implementations of ML-KEM/ML-DSA/SLH-DSA. PyPI ships prebuilt wheels for common platforms; meets the "no compile-on-deploy" bar. Performance acceptable for low-frequency operations; verify before committing for hot paths.
- **`oqs-python` / `liboqs-python`** — broadest algorithm coverage, depends on liboqs C library at the system level. Requires either prebuilt wheels with vendored liboqs or an image that ships liboqs; liboqs itself is research-grade and tracks NIST drafts faster than FIPS validation.
- **Upstream `cryptography>=45.x`** — track only; PQ support is staged but not yet shipping in stable.

**Approach:** Wave 1 ships `CryptoSuite` with `Secp256k1Suite` implemented and ML-DSA/ML-KEM/SLH-DSA suites stubbed. Wave 2 implements 2–3 PQ suites behind the same interface, runs NIST CAVP KAT vectors against all of them in CI, and selects the production default based on the criteria above. The abstraction means the choice is reversible.

## 10. Wave plan

Detailed in epic [`#921`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/921):

| Wave | Issue | One-line |
|---|---|---|
| 0A | [`#913`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/913) | Inventory + threat model + serialization matrix + this PRD |
| 0B | [`#914`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/914) | Kill the public-HMAC tamper-tag (fail-closed) |
| 0C | [`#915`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/915) | Fernet → versioned AES-256-GCM, system-wide |
| 1 | [`#916`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/916) | `CryptoSuite` + `KeypairFactory` + identity-package v2 |
| 2 | [`#917`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/917) | Hybrid identity for new agents on `did:web` |
| 3 | [`#918`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/918) | Succession statements for Kestrel #1, Emma, Meridian |
| 4 | [`#919`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/919) | Hybrid X25519 + ML-KEM-768 for CAR exports / capsule sharing |
| 5 | [`#920`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/920) | SLH-DSA release signing + transport tracking |

Dependency graph:

```
0A ──┬─→ 0C ──┐
     └─→  1  ─┼─→ 2 ──→ 3
              └─→ 4
0B (independent, ship first)
5 depends on 0A; benefits from 3
```

## 11. Non-goals

- Pure-PQ identity (deferred indefinitely; hybrid is the destination).
- Burning legacy agent identities (replaced by succession).
- Migrating wallet/Filecoin/EVM keys to PQ (chain-bound, not possible).
- Replacing local at-rest AES-GCM with KEM (no benefit, real cost).
- Implementing hybrid TLS ourselves (wait for upstream).
- HSM / hardware-key custody (separate hardening track).
- Replacing PBKDF2 with Argon2id (separate hygiene issue).
- Side-channel-resistant PQ implementations (separate hardware-custody track).

## 12. Open decisions tracked here

| # | Decision | Owner | Target wave |
|---|---|---|---|
| D1 | Hybrid combiner construction (Chempat vs X-Wing vs concat-KDF) | TBD | Wave 4 |
| D2 | Production-default PQ library after bake-off | TBD | Wave 2 |
| D3 | DID hosting for `did:web` (KestrelSovereignAI domain vs self-hosted vs federation) | TBD | Wave 2 |
| D4 | Succession ceremony custody — does the legacy secp256k1 key get destroyed after rotation, or held in cold storage for chain-walking? | TBD | Wave 3 |
| D5 | Argon2id migration for `KESTREL_DATA_KEY` (separate epic, but this PRD references it) | TBD | Out of scope |
| D6 | Constitution checkpoint cadence (every audit? every N audits? every rotation?) | TBD | Wave 3 |

These decisions are intentionally deferred until the wave where they bind. Each will land in a follow-up addendum to this PRD when resolved.
