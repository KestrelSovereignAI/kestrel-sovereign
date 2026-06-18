---
type: Architecture Spec
title: Serialization Compatibility Matrix
description: '**Status:** Wave 0A deliverable. Catalogs every signed/encrypted artifact
  format in the codebase, whether it carries a version/algorithm tag, what the v2
  design must look like,...'
resource: /docs/architecture/security/SERIALIZATION_COMPATIBILITY.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Serialization Compatibility Matrix

**Status:** Wave 0A deliverable. Catalogs every signed/encrypted artifact format in the codebase, whether it carries a version/algorithm tag, what the v2 design must look like, and how to migrate v1 → v2 without orphaning data.

The foundational bug this exposes: **most artifacts today have no `{v, alg}` tag**. A future verifier or decryptor cannot tell what produced the bytes. Every later wave depends on fixing this before introducing new algorithms.

## Artifact-by-artifact

### 1. Identity package signature

| Field | v1 (today) | v2 (Wave 1) |
|---|---|---|
| Producer | [`identity/signing.py:104-110`](../../../kestrel_sovereign/identity/signing.py#L104-L110) | Same module, suite-based |
| Container | `package.signature: str` (hex DER) + `package.content_hash: str` (hex SHA-256) | `package.signatures: [{alg, kid, sig}]` array + `package.verificationMethods: [{type, id, controller, publicKeyMultibase}]` (W3C Multikey) + `package.version: 2` |
| Version tag? | **No** | `version: 2` top-level |
| Algorithm tag? | **No** (implicit secp256k1+SHA256) | Per-signature `alg` (e.g., `"ecdsa-secp256k1-sha256"`, `"ed25519"`, `"ml-dsa-65"`) |
| Migration | Lazy-on-read: v1 packages parsed into a synthetic v2 with one signature entry tagged `ecdsa-secp256k1-sha256`. New packages always v2. |

### 2. Spawn-mandate signature

| Field | v1 (today) | v2 (Wave 2 co-migrates) |
|---|---|---|
| Producer | [`spawn/mandate.py:77-89`](../../../kestrel_sovereign/spawn/mandate.py#L77-L89) | Same, via `CryptoSuite` |
| Container | `mandate.parent_signature: str` (hex DER) | `mandate.signatures: [{alg, kid, sig}]` + `mandate.version: 2` |
| Version tag? | **No** | `version: 2` |
| Algorithm tag? | **No** | Per-signature `alg` |
| Migration | v1 mandate accepted under `LEGACY_ALLOWED` policy; new mandates always v2 hybrid. Existing parent-child relationships are **not re-signed** at parent rotation — instead, verifiers apply temporal-validity windows (see *Temporal validity rules* below) so that pre-rotation mandates remain verifiable against the recorded legacy key, while post-rotation mandates must use the new hybrid key. |

### 3. Signed compute script

| Field | v1 (today) | v2 (post Wave 0B + Wave 2) |
|---|---|---|
| Producer | [`features/compute/script_signer.py:153-174`](../../../kestrel_sovereign/features/compute/script_signer.py#L153-L174) | Same module |
| Container | `script.signature: str` with prefix `ecdsa:` or `hmac:` (forgeable) | `script.signature: {alg, kid, sig, version: 2}` — single signature, no array (mandates are the multi-sig surface) |
| Version tag? | **Implicit via prefix** (insufficient) | Explicit `version: 2` |
| Algorithm tag? | Prefix only, no agility | Per-suite `alg` field |
| Migration | Wave 0B: any `hmac:`-prefixed signature is **untrustworthy** and must be rejected. v1 `ecdsa:`-prefixed scripts kept-readable transitionally; new scripts produced as v2. |

### 4. Constitution anchor

| Field | v1 (today) | v2 (Wave 2 co-migrates) |
|---|---|---|
| Producer | [`agent/constitution.py:71`](../../../kestrel_sovereign/agent/constitution.py#L71) (anchor stored on agent graph node), [`inception_service.py:262-267`](../../../kestrel_sovereign/inception_service.py#L262-L267) (initial anchor) | Same |
| Container | `agent_node.properties.constitution_hash: str` (SHA-256 hex) — **no signature**, no algorithm tag | Versioned anchor block: `{v: 2, hash: {alg: "sha256", value}, signatures: [{alg, kid, sig}, ...]}` — anchor itself becomes a hybrid-signed artifact (matches identity-package signature-array shape so policy rules apply uniformly) |
| Version tag? | **No** | Yes |
| Algorithm tag? | **No** (implicit SHA-256) | Yes |
| Migration | v1 anchor accepted as legacy on read; on next constitution rotation or audit anchor write, upgrade to v2. Wave 3 succession ceremony writes new v2 anchor signed by new hybrid key + SLH-DSA countersign. |

### 5. CAR keyring (per-shard data keys)

| Field | v1 (today) | v2 (Wave 4) |
|---|---|---|
| Producer | [`storage/sovereign_adapter.py:233-246`](../../../kestrel_sovereign/storage/sovereign_adapter.py#L233-L246) | Same module |
| Container | `nonce(12) ‖ aesgcm_ct` — bare bytes, no version, no alg, key derived from `KESTREL_KEYRING_V2` constant | `{v: 2, alg: "X25519+MLKEM768", recipient_kid, ephemeral_x25519_pub, mlkem_ct, wrapped_dk, aad}` for export wraps; local-only keyring keeps a versioned-AES-GCM container |
| Version tag? | **No** (no header byte at all) | Yes |
| Algorithm tag? | **No** | Yes |
| Migration | Lazy-on-read: bare-bytes layout decoded as v1; new keyrings always written with `v=2` framed container. Eager re-wrap via key-rotation infra. |

### 6. Fernet ciphertext (17 sites)

| Field | v1 (today) | v2 (Wave 0C) |
|---|---|---|
| Producer | `Fernet(key).encrypt(...)` across 17 modules | Single `AEADContainer` module |
| Container | Fernet token (`gAAAAA…` base64), version byte = `0x80` (Fernet's own), no Kestrel-level versioning | `AEADContainer`: `{v: 2, alg: "AES-256-GCM", kdf: "HKDF-SHA256", salt, nonce, ct, aad}` serialized as compact base64 or CBOR |
| Version tag? | Implicit via Fernet prefix; not extensible | Explicit Kestrel-level `v` |
| Algorithm tag? | Implicit Fernet (AES-128-CBC + HMAC-SHA256) | Explicit `alg` |
| Migration | Lazy-on-read: container detects `gAAAAA…` prefix → decrypts via Fernet → re-encrypts as v2 on next write. Eager batch via `key_rotation.py --upgrade-aead`. |

### 7. JWT (OAuth)

| Field | v1 (today) | v2 (Wave 5 audit only) |
|---|---|---|
| Producer | [`endpoints/auth_oauth.py:182`](../../../endpoints/auth_oauth.py#L182) | Same |
| Container | Standard JWT, `alg: "HS256"`, secret from env | Same; symmetric, Grover-safe at ≥256-bit secret |
| Version tag? | JOSE `alg` header | Already version-tagged via JOSE |
| Algorithm tag? | Yes (`alg`) | Yes; verifier already pins `algorithms=["HS256"]` |
| Migration | Audit secret length; rotate if below 256-bit. No format change required. |

### 8. Private-key keystore on disk

| Field | v1 (today) | v2 (no immediate change) |
|---|---|---|
| Producer | [`security/key_storage.py`](../../../kestrel_sovereign/security/key_storage.py) | Same; storage layer extends to hold PQ key types in same container |
| Container | JSON file: `{salt, nonce, ciphertext, key_type, ...}` (already versioned-ish) | Add explicit `format_version: 2` and `key_alg` field; existing files migrate on next save |
| Version tag? | Partial (key_type is a tag but not format-versioned) | Explicit `format_version` |
| Algorithm tag? | Partial | Explicit `key_alg` (e.g., `"secp256k1"`, `"ed25519+ml-dsa-65"`) |

## Temporal validity rules

Succession statements (Wave 3) define key effectivity windows: a key is *authoritative* only for artifacts whose timestamp falls within the window between when it was issued and when it was succeeded. This is what makes the verify-policy work without forcing a re-signing pass on every historical artifact.

For a mandate, identity package, or constitution anchor signed by some `kid`:

1. Resolve the agent's succession chain: `[K₀ → K₁ → K₂ → ... → Kₙ]` where `K₀` is the original key, `Kₙ` is the current.
2. Find the window for `kid`: starts at the agent's inception (for `K₀`) or the rotation `effective_from` of the succession statement that introduced this key, ends at the rotation `effective_from` of the succession statement that retired it (or `+∞` if it's the current key).
3. The artifact's timestamp must fall within `kid`'s window. Outside the window → reject regardless of policy mode.
4. Within the window: apply the verify-policy mode active for this *context*. `LEGACY_ALLOWED` permits a single classical sig; `HYBRID_REQUIRED` requires both classical and PQ sigs **provided the key in question carried both** (a pre-rotation `K₀` only ever signed classically, so `HYBRID_REQUIRED` against an in-window `K₀` artifact must accept the single-classical sig — i.e., temporal validity *softens* HYBRID_REQUIRED for legacy keys, and PRD-v2 §7 must be read with this in mind).

**Concrete example.** Emma rotates on `2026-06-15` from `K_emma_legacy` (secp256k1) to `K_emma_hybrid` (Ed25519+ML-DSA-65). After rotation:

- A spawn mandate Emma signed on `2026-04-10` against her legacy key: verify against `K_emma_legacy` per the succession chain. Pre-rotation timestamp = in `K_emma_legacy`'s window. `HYBRID_REQUIRED` accepts (legacy-key softening). ✅
- A spawn mandate purportedly signed by `K_emma_legacy` on `2026-07-01`: timestamp out-of-window. Reject. ✅ (this is the forgery case after rotation)
- A spawn mandate signed by `K_emma_hybrid` on `2026-07-01`: in-window for `K_emma_hybrid`. `HYBRID_REQUIRED` requires both Ed25519 and ML-DSA-65 sigs to verify. ✅

This temporal-validity model resolves the apparent contradiction between PRD-v2 §7 ("verification tightens to `HYBRID_REQUIRED` once parent has rotated") and historical artifacts being immutable: the policy tightens for *new* artifacts the rotated key cannot have signed, while artifacts from before the rotation remain verifiable through their original key as recorded in the succession chain. No re-signing pass required.

Cycle detection in the succession chain walker is mandatory (no `kid` may appear twice in a chain).

## Cross-cutting design rules

1. **Every new artifact carries `{v, alg}` from the start.** No more implicit-by-position formats.
2. **Readers tolerate v1; writers emit v2.** No big-bang re-encryption; lazy-on-read upgrade is the default.
3. **`alg` strings are strict identifiers**, registered in a single module (`security/alg_registry.py` to be created in Wave 1). Verifier rejects unknown `alg` values per the active verify-policy.
4. **`kid` fields use stable, agent-scoped identifiers**, derived from the verification method's `id` field in the DID document. Not the public key bytes themselves.
5. **Multibase encoding for any public-key-bytes-on-disk** (W3C Multikey) — not raw hex, not raw base64. This makes future suite additions painless.
6. **AAD binding is mandatory** for AEAD ciphertexts that have a meaningful context (e.g., conversation-row encryption gets `aad = agent_id ‖ row_id ‖ "conversation"`), so a ciphertext cannot be silently moved between contexts.

## Migration policy levers

The verify-policy enum (Wave 1) gates how strict each context is:

- `LEGACY_ALLOWED` — accept v1 single-classical signature. Used for: archival import, offline-recovery, reading historical data during the migration window.
- `HYBRID_REQUIRED` — require v2 with at least one classical + one PQ signature. Default for live identity assertion and new package issuance after Wave 2.
- `PQ_REQUIRED` — reject anything without a PQ signature. Reserved for narrow long-horizon contexts (e.g., constitution checkpoints after Wave 3).

The default-tightening schedule lives in PRD-v2.

## What's left untagged after this work

Two surfaces remain implicit-by-position even in v2:

- **Constitution file content itself** (the markdown). Not signed; only its SHA-256 is anchored. This is intentional — the file lives in the repo, governance lives on chain-of-trust over the anchor.
- **Wallet keys.** Chain-bound, never produced or consumed by Kestrel-level signing logic. Out of scope.

Everything else gets a version and an algorithm tag.
