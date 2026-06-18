---
type: Architecture Spec
title: Cryptographic Inventory
description: '**Status:** Wave 0A deliverable. Authoritative inventory of every cryptographic
  primitive in use, every call-site producing or consuming a signed/MAC''d/ciphertext
  artifact, and...'
resource: /docs/architecture/security/CRYPTO_INVENTORY.md
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

# Cryptographic Inventory

**Status:** Wave 0A deliverable. Authoritative inventory of every cryptographic primitive in use, every call-site producing or consuming a signed/MAC'd/ciphertext artifact, and the on-disk format each produces.

This document feeds [`PQ_THREAT_MODEL.md`](PQ_THREAT_MODEL.md), [`SERIALIZATION_COMPATIBILITY.md`](SERIALIZATION_COMPATIBILITY.md), and [`POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md`](POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md) (PRD-v2). Updated when primitives are added/removed.

## Asymmetric primitives

### secp256k1 ECDSA — agent identity, lineage, governance

The single classical signature primitive in use across all sovereign-identity surfaces.

| Surface | File:Line | Sign | Verify | Output format |
|---|---|---|---|---|
| Agent keypair generation | [`inception_service.py:54-58`](../../../kestrel_sovereign/inception_service.py#L54-L58) | `ec.generate_private_key(SECP256K1)` | — | EllipticCurvePrivateKey |
| Public-key → Ethereum address | [`inception_service.py:70-87`](../../../kestrel_sovereign/inception_service.py#L70-L87) | Keccak-256(uncompressed pubkey) → last 20 bytes + EIP-55 checksum | — | `0x{40-hex}` |
| DID document creation | [`inception_service.py:110-129`](../../../kestrel_sovereign/inception_service.py#L110-L129) | — | — | `did:pkh:eip155:1:{address}` + `EcdsaSecp256k1VerificationKey2019` |
| Identity-package signing | [`identity/signing.py:104-110`](../../../kestrel_sovereign/identity/signing.py#L104-L110) | `private_key.sign(content_hash.encode('utf-8'), ec.ECDSA(SHA256))` (signs the **hex-encoded** SHA-256 digest as ASCII bytes, not the raw 32-byte digest) | [`identity/signing.py:160-164`](../../../kestrel_sovereign/identity/signing.py#L160-L164), [`identity/signing.py:222-232`](../../../kestrel_sovereign/identity/signing.py#L222-L232) (DID-doc fallback) | DER signature, hex-encoded into `package.signature` |
| Spawn-mandate signing | [`spawn/mandate.py:77-89`](../../../kestrel_sovereign/spawn/mandate.py#L77-L89) | `parent_private_key.sign(payload, ec.ECDSA(SHA256))` | [`spawn/mandate.py:92+`](../../../kestrel_sovereign/spawn/mandate.py#L92) | DER signature, hex into `mandate.parent_signature` |
| Script signing (primary path) | [`features/compute/script_signer.py:153-163`](../../../kestrel_sovereign/features/compute/script_signer.py#L153-L163) | `_private_key.sign(content_hash_bytes, ec.ECDSA(SHA256))` (signs `sha256(content_hash_string).digest()` — i.e., the script's hex-content-hash *re-hashed* into raw 32 bytes, see `script_signer.py:151`) | [`features/compute/script_signer.py:198+`](../../../kestrel_sovereign/features/compute/script_signer.py#L198) | `ecdsa:{base64}` |
| Wallet — Filecoin/EVM | [`features/wallet/filecoin_keys.py`](../../../kestrel_sovereign/features/wallet/filecoin_keys.py) | secp256k1 (chain-bound) | n/a (chain-verified) | wallet address |

### Ed25519

Available as the classical half of the Wave 2 hybrid-identity composite via `Ed25519Suite` in [`security/crypto_suite.py`](../../../kestrel_sovereign/security/crypto_suite.py). No production signing surfaces in-tree currently consume it; reserved for the Wave 2 hybrid signer.

### "HMAC-as-signature" — public-keyed tamper tag (BROKEN, Wave 0B kill target)

| File:Line | Issue |
|---|---|
| [`features/compute/script_signer.py:165-174`](../../../kestrel_sovereign/features/compute/script_signer.py#L165-L174) | HMAC-SHA256 with key = `(self.agent_did or "kestrel-unsigned").encode()`. **The HMAC key is the agent's PUBLIC DID.** Anyone reading the script can forge the `hmac:` tag. Verify path at [`script_signer.py:198+`](../../../kestrel_sovereign/features/compute/script_signer.py#L198) accepts it. |

### `MigrationCertificate.signature` — present but unwired (future surface)

| File:Line | Status |
|---|---|
| [`identity/continuity_verifier.py:137`](../../../kestrel_sovereign/identity/continuity_verifier.py#L137) | Schema field `signature: str = ""` exists; no signing call-site found. If this gets wired up, it joins the long-lived signed-artifact set and must adopt the v2 signature container from Wave 1. Tracked here so Wave 1 covers it. |

### Post-quantum primitives

None in use. ML-DSA, ML-KEM, SLH-DSA, hybrid combiners — all unimplemented.

## Symmetric primitives

### AES-256-GCM — at-rest encryption (modern path)

| Surface | File:Line | Key derivation |
|---|---|---|
| Private-key keystore | [`security/key_storage.py:31-37`, constants `48-53`](../../../kestrel_sovereign/security/key_storage.py#L48-L53) | PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte salt, 12-byte nonce, 32-byte key |
| User API key storage *(Frinz)* | `frinz/security/user_key_storage.py` (Frinz repo) | Same pattern. Relocated out of `kestrel_sovereign/security/` in 2026-05 (kestrel-sovereign#1156, frinz#156); was Frinz product code in foundation. |
| Platform API key storage *(Frinz)* | `frinz/security/platform_key_storage.py` (Frinz repo) | Same pattern. Relocated alongside `user_key_storage.py`. |
| CAR keyring (per-shard data keys) | [`storage/sovereign_adapter.py:233-246`](../../../kestrel_sovereign/storage/sovereign_adapter.py#L233-L246) | `derive_key(b"KESTREL_KEYRING_V2")` from `KESTREL_DATA_KEY` master. 12-byte nonce, AES-GCM with no AAD. |

### Fernet (AES-128-CBC + HMAC-SHA256) — legacy path (Wave 0C kill target)

**16 files** with direct `Fernet` / `cryptography.fernet` usage. Below modern bar today (AES-128 < AES-256); Grover halves to ~64-bit effective. Authoritative list from `grep -lE '(Fernet|fernet|cryptography\.fernet)' --include='*.py' --include='*.sh' -r .` (excluding `.venv`, `__pycache__`, `worktrees`, and `tests/` — see "Test files" note below).

#### Production code (13 files)

| Layer | File |
|---|---|
| SDK encryption module | [`kestrel_sdk/security/encryption.py`](../../../kestrel_sdk/security/encryption.py) |
| Sovereign re-export | [`kestrel_sovereign/security/encryption.py`](../../../kestrel_sovereign/security/encryption.py) |
| Storage encryption helper | [`kestrel_sovereign/storage/encryption.py`](../../../kestrel_sovereign/storage/encryption.py) |
| Storage init | [`kestrel_sovereign/storage/__init__.py`](../../../kestrel_sovereign/storage/__init__.py) |
| Conversation store | [`kestrel_sovereign/storage/async_conversation_store.py`](../../../kestrel_sovereign/storage/async_conversation_store.py) |
| File store | [`kestrel_sovereign/storage/async_file_store.py`](../../../kestrel_sovereign/storage/async_file_store.py) |
| Lighthouse provider | [`kestrel_sovereign/storage/providers/lighthouse_provider.py`](../../../kestrel_sovereign/storage/providers/lighthouse_provider.py) |
| Filebase provider | [`kestrel_sovereign/storage/providers/filebase_provider.py`](../../../kestrel_sovereign/storage/providers/filebase_provider.py) |
| Filecoin adapter | [`kestrel_sovereign/filecoin_adapter.py`](../../../kestrel_sovereign/filecoin_adapter.py) |
| Key rotation tooling | [`kestrel_sovereign/security/key_rotation.py`](../../../kestrel_sovereign/security/key_rotation.py) |
| Conversations endpoint | [`endpoints/conversations.py`](../../../endpoints/conversations.py) |
| Reflection — memory check | `kestrel-feature-reflection` optional package |
| Reflection — arms check | `kestrel-feature-reflection` optional package |

#### Tooling and examples (3 files)

| Layer | File | Notes |
|---|---|---|
| Demo script | [`examples/demo_sovereignty.py`](../../../examples/demo_sovereignty.py) | Migrates with the rest |
| Agent key rotation script | [`scripts/rotate_agent_key.py`](../../../scripts/rotate_agent_key.py) | Will need to handle both v1 Fernet and v2 AEAD during transition |
| Backup-restore test harness | [`tests/integration/test_backup_restore.py`](../../../tests/integration/test_backup_restore.py) | Drives the full export/import round-trip via `SovereignStorageAdapter` (AES-256-GCM); replaced the bash + Fernet harness in epic #1050 tier 2.3 |

#### Test files (downstream consumers, not direct migration targets)

Six unit test files reference `Fernet` to exercise the production crypto: [`tests/unit/test_agent_encryption.py`](../../../tests/unit/test_agent_encryption.py), [`tests/unit/test_commands_conversations_endpoint_contracts.py`](../../../tests/unit/test_commands_conversations_endpoint_contracts.py), [`tests/unit/test_decryption_failure.py`](../../../tests/unit/test_decryption_failure.py), [`tests/unit/test_encryption.py`](../../../tests/unit/test_encryption.py), [`tests/unit/test_key_rotation.py`](../../../tests/unit/test_key_rotation.py), [`tests/unit/test_post_response_pipeline.py`](../../../tests/unit/test_post_response_pipeline.py). These get **extended** in Wave 0C to cover both v1 read and v2 write — they are not migrated away from in the same sense as production code.

### HMAC-SHA256 — JWT signing and webhook auth (legitimate uses)

| Surface | File:Line | Algorithm | Notes |
|---|---|---|---|
| OAuth JWT issuance | [`endpoints/auth_oauth.py:182`](../../../endpoints/auth_oauth.py#L182) | `jwt.encode(payload, secret, algorithm="HS256")` | Symmetric; Grover-safe at ≥256-bit secret |
| OAuth JWT verification | [`endpoints/auth_oauth.py:185-190`](../../../endpoints/auth_oauth.py#L185-L190) | `jwt.decode(token, secret, algorithms=["HS256"])` | Pin algorithms list (already correct) |
| Webhook signature validation | [`features/webhooks/auth.py:124-130`](../../../kestrel_sovereign/features/webhooks/auth.py#L124-L130) | `hmac.new(secret, body, sha256).hexdigest()` then `hmac.compare_digest` | Symmetric; PQ-safe with ≥256-bit secret per provider. Wave 5 audits secret length. |

## Hash primitives

### SHA-256 — content addressing, signing inputs, identifier hashing

| Use | File:Line |
|---|---|
| Identity-package content hash (signing input) | [`identity/signing.py:99-101`](../../../kestrel_sovereign/identity/signing.py#L99-L101) |
| Constitution file integrity hash | [`agent/constitution.py:91`](../../../kestrel_sovereign/agent/constitution.py#L91) |
| Constitution anchor stored on graph | [`agent/constitution.py:71`](../../../kestrel_sovereign/agent/constitution.py#L71) |
| Constitution US-content hash for inception | [`inception_service.py:262-267`](../../../kestrel_sovereign/inception_service.py#L262-L267) |
| Script content hash (signing input) | [`features/compute/script_signer.py:151`](../../../kestrel_sovereign/features/compute/script_signer.py#L151) |
| API-key identifier hash *(Frinz)* | `frinz/security/user_key_storage.py` (Frinz repo) |

### Keccak-256 — Ethereum address derivation only

| Use | File:Line |
|---|---|
| Public-key → 20-byte Ethereum address | [`inception_service.py:81-83`](../../../kestrel_sovereign/inception_service.py#L81-L83) |
| EIP-55 checksum | [`inception_service.py:90-107`](../../../kestrel_sovereign/inception_service.py#L90-L107) |

### BLAKE2b — Filecoin / IPFS content addressing

| Use | File:Line |
|---|---|
| Filecoin address derivation | [`features/wallet/filecoin_keys.py:66-78`](../../../kestrel_sovereign/features/wallet/filecoin_keys.py#L66-L78) |
| CAR-file CID computation | [`storage/car_builder.py`](../../../kestrel_sovereign/storage/car_builder.py) |

## Library footprint

| Library | Pinned version | Used for |
|---|---|---|
| `cryptography` | `>=45.0.5` | EC keys, AES-GCM, PBKDF2, Fernet, serialization |
| `pycryptodome` | `3.23.0` | Keccak-256 only |
| `web3` | `>=7.0.0` | Wallet/blockchain (secp256k1 via eth-keys) |
| `PyJWT` | (transitive) | OAuth JWT |
| `cbor2` | (optional) | dag-cbor in CAR builder |

No PQ libraries today. Wave 1 abstraction lets us evaluate `pqcrypto`, `oqs-python`/`liboqs-python`, or upstream `cryptography>=45.x` (track) behind a `CryptoSuite` interface without picking a winner up front.

## DID method footprint

| DID method | Where produced | Where resolved |
|---|---|---|
| `did:pkh:eip155:1:{address}` | [`inception_service.py:112`](../../../kestrel_sovereign/inception_service.py#L112) | DID document loaded from local storage at `{key_id}.json` per [`identity/signing.py:204`](../../../kestrel_sovereign/identity/signing.py#L204); fallback verification via [`_verify_with_did_document`](../../../kestrel_sovereign/identity/signing.py#L177-L241) |

The DID format is **structurally welded to secp256k1** — the DID *is* `keccak(secp256k1_pubkey)[-20:]` with EIP-55 checksum. There is no extension point for additional verification methods or PQ keys. Wave 2 must change DID method.

## Wallet keys (out of scope, called out for clarity)

[`features/wallet/filecoin_keys.py`](../../../kestrel_sovereign/features/wallet/filecoin_keys.py) and any EVM signing must remain secp256k1 — they are **chain-bound**: Filecoin and Ethereum mandate that curve and any PQ-resistant chain replacement is decades away. Per the threat model, wallet keys are explicitly **non-authoritative for agent identity continuity**: a wallet is a payment instrument the agent holds, not its selfhood. This must be enforced in code (no spawn mandate, no constitution anchor, no identity package can sign with a wallet key) so that the inevitable post-quantum break of these chains' identity layer does not break Kestrel's identity layer.

## Hardware-backed custody

None. No YubiKey, TPM, secure-enclave, or KMS integration. All keys live in encrypted files under `KESTREL_DATA_KEY`. Adding hardware custody is a separate hardening track and explicitly out of scope for this epic.

## Summary by HNDL relevance

| Category | HNDL-relevant? | Why |
|---|---|---|
| All ECDSA signatures (identity, mandate, constitution, script) | **No** (signatures don't decrypt anything) — but **forge-vulnerable post-quantum** | Shor breaks signing, allowing future forgery of historical artifacts. Threat is identity/lineage forgery, not data exposure. |
| Local AES-256-GCM (keystore, API keys) | **No** | Symmetric, locally-derived from `KESTREL_DATA_KEY`. Grover halves to 128-bit effective — still acceptable. |
| Local AES-128 Fernet (17 sites) | **Mild** | Grover → ~64-bit effective. Already below modern bar; replace independent of PQ. |
| CAR keyring (AES-GCM, locally derived) | **No** as currently used | But Wave 4 will introduce KEM-wrapping for export/sharing — *that* surface is HNDL-relevant. |
| HMAC-SHA256 for JWT | **No** | Symmetric, secret-keyed; size ≥256 bits = Grover-safe. |
| HMAC "signature" with public DID as key | **N/A — already broken classically** | Wave 0B fix. |
| Wallet secp256k1 keys | Out of scope | Chain-bound, explicitly non-authoritative for agent identity. |

The post-quantum exposure that genuinely matters is **forgery of long-lived signed artifacts** (identity packages, spawn-mandate chains, constitution anchors) and **export-time KEM wrapping that doesn't yet exist**. Local at-rest data is fine.
