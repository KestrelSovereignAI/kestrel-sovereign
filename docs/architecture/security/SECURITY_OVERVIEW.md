---
type: Architecture Spec
title: Kestrel Security Overview
description: A plain-language tour of Kestrel's cryptographic posture as of the Quantum
  Hardening epic ([`#921`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/921)).
  Written...
resource: /docs/architecture/security/SECURITY_OVERVIEW.md
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

# Kestrel Security Overview

> A plain-language tour of Kestrel's cryptographic posture as of the Quantum Hardening epic ([`#921`](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/921)). Written for sovereign operators, evaluators, and future-you. No prior crypto background required.

If you want the deep technical version, see the linked documents at the end. This page tells you **what** we do, **why**, and **what's your job vs. the framework's**.

> **Status (May 2026):** All four production agents (Kestrel #1 / Emma, Meridian, Nellie, Claw) are running on hybrid identities. The runtime detects each agent's `successions/<slug>.json` at startup and signs new artifacts with both Ed25519 and ML-DSA-65 by default. New artifacts use the v2 `signatures` array; pre-rotation artifacts continue to verify under the chain walker. See **"Operator playbook"** below for the rollout procedure.

## TL;DR

- Every Kestrel agent has a cryptographic identity (a DID — pronounced "did," like the past tense of "do") that signs everything important. Today, **new agents sign with two algorithms at once**: a classical one (Ed25519) and a post-quantum one (ML-DSA-65). To forge a signature, an attacker must break **both**.
- Existing agents that were minted before the post-quantum migration carry a single classical key. They migrate to the new dual-key form via a one-time **rotation ceremony** that produces a **succession statement** — a signed bridge linking the old identity to the new one.
- Encrypted data shipped between agents (capsules, exports) is wrapped with **two key-agreement schemes at once** — classical (X25519) and post-quantum (ML-KEM-768). Same logic: an attacker must break both.
- Release artifacts (the wheels you `pip install`) are signed with **SLH-DSA**, a hash-based post-quantum signature whose security only depends on the strength of SHA-256.
- All of this is to keep the **lifelong sovereignty** promise: artifacts produced today must still be trustworthy in 2045 — including against attackers who have a quantum computer by then.

## The promise we're trying to keep

Kestrel claims **lifelong sovereignty** — that an agent's identity, conversation history, and constitutional anchors will still be theirs, verifiable, and intact decades from now. That's a hard promise. Most software is happy to be obsolete in 5 years; we're signing up for 30+.

The biggest threat to that promise is **cryptography aging**. Algorithms we trust today have known weaknesses against future computers — specifically, against **quantum computers** running well-known algorithms (Shor's algorithm and Grover's algorithm). Nobody has a useful one yet. NIST estimates 2035–2045 for the first cryptographically-relevant quantum computer. But the threat is already here in a subtle way: an adversary today can capture our encrypted traffic, store it, and decrypt it later when their quantum computer arrives. This is called **HNDL** — Harvest Now, Decrypt Later.

The Quantum Hardening epic (closed in May 2026) was our migration to defenses that survive that horizon.

## The threat model in 90 seconds

There are two quantum capabilities that matter:

**Shor's algorithm** breaks the math behind almost all classical asymmetric crypto — RSA, ECDSA, Diffie-Hellman, etc. If an attacker has a working Shor implementation, they can take any public key you've published and recover the private key. **Every signature you ever produced becomes forgeable.** Every encrypted message anyone sent to your old public key becomes decryptable.

**Grover's algorithm** halves the effective strength of brute-force attacks on symmetric crypto (AES, SHA-256). It doesn't break them — it just means AES-128 acts like AES-64 (too weak), so we use AES-256 (which acts like AES-128, still strong).

Practical effects on Kestrel **if we did nothing**:

- An attacker captures any published DID document → forges signatures from that agent forever
- An attacker captures any encrypted conversation export → decrypts it whenever they get a quantum computer
- An attacker captures a release wheel → swaps in malicious code with a forged signature, every future user installs the malware

Every defense in the rest of this document exists to break one of those chains.

## What we shipped, layer by layer

### 1. Identity (Wave 2)

**Old agents** had a single ECDSA key on the secp256k1 curve. Their DID looked like `did:pkh:eip155:1:0x997B7...`. This is fine until quantum.

**New agents** have a **hybrid keypair** — two private keys, two public keys. The classical half is Ed25519 (a modern, fast classical signature). The post-quantum half is ML-DSA-65 (NIST FIPS 204, lattice-based). Their DID looks like `did:web:agents.kestrel-sovereign.test:meridian` and resolves to a JSON document published over HTTPS that lists both public keys.

When a hybrid agent signs something, they sign **twice** — once with each key. A verifier under the `HYBRID_REQUIRED` policy refuses anything missing either signature. The bet: an attacker must break BOTH algorithms to forge — and breaking ML-DSA isn't currently known and breaking Ed25519 needs a quantum computer.

> **Files:**  [`identity/inception_did_web.py`](../../../kestrel_sovereign/identity/inception_did_web.py), [`identity/hybrid_keypair.py`](../../../kestrel_sovereign/identity/hybrid_keypair.py), [`security/crypto_suite.py`](../../../kestrel_sovereign/security/crypto_suite.py).

### 2. Succession — bridging old agents into hybrid (Wave 3)

The four existing agents (Kestrel #1, Meridian, Emma, Frinz tenants) need to migrate **without losing their history**. Burning the old DID and minting a fresh one would mean every artifact they ever signed becomes orphaned — no chain back to who they are.

Instead, each agent runs a **rotation ceremony** once, which produces a signed **succession statement**: a small JSON document that says "I, the holder of legacy DID X, hereby designate hybrid DID Y as my successor, effective at this UTC timestamp." It's signed by:

- the legacy key (proves the old agent authorized the bridge)
- both halves of the new hybrid key (proves the new identity accepted the role)
- an SLH-DSA "archival" signature (the most conservative post-quantum algorithm we have — explained below)

Verifiers walking an artifact's signature use a **chain walker** that, given the artifact's timestamp, finds which identity was active at that moment. After the cutoff, the verifier rejects classical-only signatures from the legacy key. The legacy key still works, but only artifacts dated **before** the cutoff trust it on its own.

> **Files:** [`identity/succession.py`](../../../kestrel_sovereign/identity/succession.py), [`identity/succession_chain.py`](../../../kestrel_sovereign/identity/succession_chain.py), [`identity/rotation_ceremony.py`](../../../kestrel_sovereign/identity/rotation_ceremony.py). **Operator runbook:** [`SUCCESSION_RUNBOOK.md`](SUCCESSION_RUNBOOK.md).

### 3. Sealed capsules — encrypted shipments (Wave 4)

When one agent sends another agent something private — an identity package, a conversation export, a backup tarball — they wrap it in a **sealed capsule**. A capsule is encrypted to a specific recipient's hybrid public keys. Only that recipient (with both private halves) can open it.

The encryption uses **two key-agreement schemes simultaneously** — X25519 (classical, very fast) and ML-KEM-768 (post-quantum, NIST FIPS 203). Both produce a shared secret; they're combined via HKDF into a single AES-256-GCM key. Same hybrid logic as identity: an attacker capturing a capsule today and trying to decrypt it later must break both schemes.

A capsule is a single self-describing JSON file. The recipient's expected public keys are embedded in it — if you tamper with them, the AES authentication fails (they're cryptographically bound to the encryption key via the transcript). You can ship a capsule through any channel — email, gist, IPFS — without leaking the contents.

> **Files:** [`security/sealed_capsule.py`](../../../kestrel_sovereign/security/sealed_capsule.py), [`security/hybrid_kem.py`](../../../kestrel_sovereign/security/hybrid_kem.py), [`security/kem_suite.py`](../../../kestrel_sovereign/security/kem_suite.py).

### 4. Release signing (Wave 5)

When we publish a Kestrel release (a new wheel + tarball on a `v*` tag), a GitHub Action generates a **release manifest** — a JSON file listing every artifact's SHA-256 hash and size — and signs it with **SLH-DSA-SHA2-128s** (NIST FIPS 205).

SLH-DSA is **not hybrid**. It's pure post-quantum. We chose it on purpose: SLH-DSA is **hash-based**, which means its security argument only depends on SHA-256 being strong (no novel lattice math). Of all the post-quantum schemes NIST standardized, it has the strongest historical pedigree. Signature size is large (~7.8 KB) so it's bad for high-throughput signing — but a release ships once a release, so size is irrelevant.

The verify side is `kestrel release verify --manifest ... --trusted-signer-multibase z...`. As long as you have the trusted-signer pubkey baked into your install procedure, you can independently confirm the wheel you downloaded is exactly what we shipped — no GitHub-trust required.

> **Files:** [`security/release_manifest.py`](../../../kestrel_sovereign/security/release_manifest.py), [`cli_release.py`](../../../kestrel_sovereign/cli_release.py), [`.github/workflows/release-sign.yml`](../../../.github/workflows/release-sign.yml).

### 5. At-rest data — AES-256-GCM (Wave 0C)

Every encrypted file Kestrel writes locally — encrypted private keys, encrypted conversation rows, encrypted backups — uses **AES-256-GCM**. We migrated from `cryptography.fernet` (Wave 0C, late 2025) because Fernet's format isn't versioned and we wanted explicit forward-compat for future suite changes. The token format is `KSAv2:<base64>` so a glance tells you what you're looking at.

AES-256-GCM is **not hybrid** because it doesn't need to be. Local at-rest encryption isn't HNDL-vulnerable — an attacker harvesting the encrypted file would also need to harvest the master key, and if they have your master key they win regardless of algorithm. The threat model for at-rest data is "what if someone copies the disk?" — and AES-256 against Grover gives you ~128-bit effective security, which is fine.

Old Fernet tokens still decrypt automatically (we kept read-compat) so you don't have to migrate any data on disk.

> **Files:** [`kestrel_sdk/security/aead.py`](../../../kestrel_sdk/security/aead.py), [`security/key_storage.py`](../../../kestrel_sovereign/security/key_storage.py).

## Why hybrid is the steady state, not a transition

This is the question I keep getting: *isn't hybrid just a bridge until we're sure post-quantum is solid? When do we drop the classical half?*

**Answer:** We don't, for the foreseeable future. Here's why.

Post-quantum algorithms (ML-DSA, ML-KEM) are based on **lattice problems**. Lattice math is younger than elliptic-curve math by roughly 30 years. The standardized parameters look strong today, but cryptography history is full of "we thought this was strong, then a clever attack appeared" stories — RSA-512, MD5, SHA-1. NIST is confident enough in the lattice schemes to standardize them, but most serious cryptographers (Cloudflare, Google, Signal, AWS, the NSA's CNSA 2.0 guidance) deploy them **alongside** classical, not in place of.

The bet is asymmetric:

- If lattice schemes hold up: hybrid costs us a small extra signature/key on every artifact. Negligible.
- If a lattice break appears in 2031: every pure-PQ artifact from 2026 becomes forgeable. Every hybrid artifact from 2026 is still safe — the classical half holds against a non-quantum attacker, which is what 2031 has.
- If a quantum computer arrives in 2040: every classical artifact becomes forgeable. Every hybrid artifact from 2026 is still safe — the PQ half holds against the quantum attacker.

The only world where pure-PQ wins over hybrid is one where we've had 30+ years of lattice cryptanalysis without a break **and** quantum computers exist **and** we want smaller signatures. That's a 2050s decision, not a 2026 decision.

The framework does include the escape hatch: `VerifyPolicy.PQ_REQUIRED` already exists in code (see [`security/verify_policy.py`](../../../kestrel_sovereign/security/verify_policy.py)). Whenever the cryptography community signals "lattice has weathered enough scrutiny," we can flip a config and start producing PQ-only artifacts. No code change needed.

**Special case: release signing is already pure-PQ.** Because release signing uses SLH-DSA (hash-based, not lattice-based), the lattice-break concern doesn't apply. Hash-based signatures rely only on SHA-256. We didn't ship hybrid for release signing because the conservative post-quantum option already exists for that surface.

So the picture is:

| Surface              | Today's posture          | Why                                                |
| :------------------- | :----------------------- | :------------------------------------------------- |
| Live identity        | Hybrid (Ed25519 + ML-DSA-65) | Lattice schemes need more cryptanalysis time       |
| Sealed capsules      | Hybrid (X25519 + ML-KEM-768) | Same reasoning + HNDL                              |
| Release signing      | Pure PQ (SLH-DSA)        | Hash-based, no lattice assumption needed           |
| At-rest local data   | AES-256-GCM              | Not HNDL-vulnerable; Grover doesn't break AES-256  |

## What's the operator's job vs. the framework's

**The framework gives you:**
- A way to mint new hybrid identities
- A rotation ceremony for migrating old agents
- Verification of every signed artifact under whatever policy you choose
- Encrypted-shipment tooling (sealed capsules)
- A signed release pipeline that runs on every `v*` tag

**You, the operator, are responsible for:**
1. **Setting `KESTREL_DATA_KEY`** before running anything. This is the master key that encrypts every key bundle on disk. Without it, the framework can't read or write encrypted bundles. Pick something with at least 32 bytes of entropy and back it up — losing it locks you out of every encrypted secret.
2. **Running the rotation ceremony** for any pre-Wave-3 agents you operate (Kestrel #1, Meridian, Emma, Frinz tenants). The runbook is at [`SUCCESSION_RUNBOOK.md`](SUCCESSION_RUNBOOK.md). The dry-run script at [`scripts/quantum_rotation_dry_run.py`](../../../scripts/quantum_rotation_dry_run.py) lets you rehearse on a throwaway agent first.
3. **Choosing where to host `did:web` documents.** The new hybrid identities resolve via HTTPS. Pick a domain you control, deploy `did.json` files under `/<slug>/did.json`, and confirm the URL is reachable before flipping the cutoff.
4. **Provisioning the release-signing GitHub secrets** (`KESTREL_RELEASE_SECRET_B64`, `KESTREL_RELEASE_PUBLIC_B64`, `KESTREL_DATA_KEY`) if you publish releases. Without those, the auto-sign workflow won't run on your tag pushes.
5. **Backing up keys.** Encrypted key bundles + the master key should be in two independent locations. The framework does not make backups for you.
6. **Eventually destroying legacy keys** (after the cutoff, after a rollback window). The runbook has the procedure; the framework deliberately does NOT auto-delete because key destruction is a one-way door.

## Operator playbook — full hybrid rollout

The end-to-end migration for an existing legacy agent runs like this. Each step is a separate concern and they can be days apart.

### 1. Pre-flight: encrypted backup

`scripts/quantum_pre_ceremony_backup.py` produces a verifiable encrypted tarball (AES-256-GCM, PBKDF2-HMAC-SHA256, 600k iterations) of the agent's data dir. Self-verifies by decrypting+extracting+hashing every file before declaring success. Copy the encrypted output off-disk (Google Drive, S3, etc.) and vault the passphrase separately.

### 2. Ceremony

`scripts/quantum_kestrel_1_ceremony.py` (despite the name, parameterized for any agent) runs the rotation: loads the legacy ECDSA private key, mints a hybrid Ed25519 + ML-DSA-65 keypair plus an SLH-DSA archival keypair, signs the succession statement, persists everything via `SecureKeyStorage`, and writes a fresh `did.json` ready to publish. Five gates including a key-cross-check against the existing DID document.

### 3. Publish the new `did.json`

Drop the produced `did.json` into the `agent-identities` repo (or wherever you serve `https://<domain>/<slug>/did.json`). Confirm `curl -sI` returns HTTP 200 with `Content-Type: application/json`.

### 4. Restart the agent runtime

`KestrelAgent.__init__` detects the succession statement on disk, loads the hybrid keypair, and the agent starts producing hybrid signatures by default. Verify via:

```bash
curl -H "X-API-Key: $KEY" http://localhost:8888/api/agents/<name>/api/identity | python3 -m json.tool
```

You should see `is_hybrid: true` and a `signing_did` of `did:web:<domain>:<slug>`. The UI's identity tab shows a green `[HYBRID]` badge with the new DID.

### 5. Wait at least 7 days

The legacy ECDSA private key still on disk is the rollback escape hatch — if anything about the new identity is broken, you can re-rotate or fall back. Don't destroy the legacy key until you have evidence the new identity is working in production: artifacts signed under it verify under the chain walker, the published `did.json` resolves cleanly, etc.

### 6. Destroy the legacy private key

`scripts/quantum_destroy_legacy_key.py` secure-deletes the legacy private key after seven gates pass:

1. `--confirm` flag set (default is dry-run)
2. `KESTREL_DESTROY_CONFIRM=I-have-verified-the-rollback-window` env var set
3. Succession statement on disk
4. Succession `effective_from` is at least `--rollback-window-days` (default 7) in the past
5. Succession statement crypto-verifies (predecessor + successor signatures)
6. Hybrid keys on disk + probe-sign verifies they actually match the published successor identity
7. Published `did.json` reachable AND its verification methods match the succession statement

The runtime tolerates the missing legacy private after destruction: `runtime_identity.load_agent_identity` derives the legacy public key from the on-disk DID document for the chain walker; signing call sites use the hybrid keypair.

## When something looks wrong

| Symptom                                                       | First place to look                                                                        |
| :------------------------------------------------------------ | :----------------------------------------------------------------------------------------- |
| `MasterKeyNotConfiguredError`                                 | `KESTREL_DATA_KEY` not set in the env                                                      |
| `did:web` resolution fails                                    | Is `https://<domain>/<slug>/did.json` reachable? Try in a browser                          |
| Verification fails: "post-cutoff artifact has no PQ signature" | Artifact was signed by a classical-only key after the agent's cutoff. Re-sign with hybrid. |
| Verification fails: "chain anchor mismatch"                   | Wrong `root_did` passed to the chain walker                                                |
| `kestrel release verify` rejects an artifact                  | Artifact is corrupted, the manifest was tampered with, or the trusted-signer pubkey is wrong |
| Sealed capsule won't open                                     | Wrong recipient keypair, OR the capsule is corrupted (any tamper fails AEAD authentication) |

For everything else, the unit and integration tests are the spec — they encode every guarantee the code makes. See [`tests/integration/test_quantum_hardening_e2e.py`](../../../tests/integration/test_quantum_hardening_e2e.py) for the cross-wave seam tests.

## Documents in this directory

Plain-language entrypoints:

- **`SECURITY_OVERVIEW.md`** (this file) — start here

Operator-facing procedures:

- [`SUCCESSION_RUNBOOK.md`](SUCCESSION_RUNBOOK.md) — how to run a rotation ceremony for a real agent
- [`KEY_MANAGEMENT.md`](KEY_MANAGEMENT.md) — how key storage works on disk, how rotation interacts with it
- [`KEY_ROTATION.md`](KEY_ROTATION.md) — narrower deep-dive on rotation mechanics

Technical depth:

- [`POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md`](POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md) — the PRD that drove the Quantum Hardening epic, with wave plan and design decisions
- [`PQ_THREAT_MODEL.md`](PQ_THREAT_MODEL.md) — the formal threat model for the post-quantum migration
- [`CRYPTO_INVENTORY.md`](CRYPTO_INVENTORY.md) — every cryptographic primitive in Kestrel, what it's used for, what its quantum status is
- [`SERIALIZATION_COMPATIBILITY.md`](SERIALIZATION_COMPATIBILITY.md) — every signed/encrypted artifact format, with version invariants
- [`CRYPTOGRAPHIC_ANCHORING.md`](CRYPTOGRAPHIC_ANCHORING.md) — how the constitution and identity are anchored cryptographically
- [`CONSTITUTION_EMBEDDING.md`](CONSTITUTION_EMBEDDING.md) — how constitutional protections are bound to agent identity
- [`INTEGRITY_AUDIT_SYSTEM.md`](INTEGRITY_AUDIT_SYSTEM.md) — periodic integrity checks
- [`ANTI_CORRUPTION_ANALYSIS.md`](ANTI_CORRUPTION_ANALYSIS.md) — how we detect tampering with stored data

Privacy (related but not crypto-specific):

- [`PRIVACY_AGENT.md`](PRIVACY_AGENT.md), [`PRIVACY_MODES.md`](PRIVACY_MODES.md)

## Glossary

- **AEAD** — Authenticated Encryption with Associated Data. Encryption that also detects tampering. We use AES-256-GCM as our AEAD.
- **DID** — Decentralized Identifier. A URL-like string that identifies an agent without depending on a central registry. We use `did:pkh` for legacy and `did:web` for hybrid.
- **HNDL** — Harvest Now, Decrypt Later. The threat where an attacker captures encrypted traffic today and decrypts it once they have a quantum computer.
- **Hybrid (crypto)** — Using two algorithms simultaneously so an attacker must break both to win. Not the same as "transitional."
- **KEM** — Key Encapsulation Mechanism. A way to securely send a fresh symmetric key to a recipient over an insecure channel. ML-KEM-768 is the post-quantum one we use.
- **Lattice cryptography** — A family of post-quantum schemes (ML-KEM, ML-DSA) based on hard math problems involving high-dimensional grids. Younger than ECC.
- **Multikey / multibase** — W3C standards for encoding public keys in a self-describing way. The `z...` strings you see in DID documents.
- **Rotation ceremony** — The one-time procedure to migrate a legacy agent to a hybrid identity. Produces a succession statement.
- **Sealed capsule** — Our format for encrypted-to-recipient shipments. Contains a hybrid-KEM-wrapped AEAD ciphertext.
- **Succession statement** — The signed bridge between an agent's legacy and hybrid DIDs. Contains both signatures + an archival signature.
- **CRQC** — Cryptographically-Relevant Quantum Computer. The hypothetical quantum machine that can break classical asymmetric crypto.
