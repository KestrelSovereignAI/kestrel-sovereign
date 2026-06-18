---
type: Architecture Spec
title: Post-Quantum Threat Model
description: '**Status:** Wave 0A deliverable, threats now mitigated by waves 1–5
  (May 2026). The threats below remain relevant — a threat model doesn''t expire when
  defenses ship — but each i...'
resource: /docs/architecture/security/PQ_THREAT_MODEL.md
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

# Post-Quantum Threat Model

**Status:** Wave 0A deliverable, threats now mitigated by waves 1–5 (May 2026). The threats below remain relevant — a threat model doesn't expire when defenses ship — but each is now addressed by a specific control. See [`SECURITY_OVERVIEW.md`](SECURITY_OVERVIEW.md) for the operator-facing summary or [`POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md`](POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md) for the wave-by-wave map.

**Companion to:** [`SECURITY_OVERVIEW.md`](SECURITY_OVERVIEW.md), [`CRYPTOGRAPHIC_INVENTORY`](CRYPTO_INVENTORY.md), [`POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md`](POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md).

## What we are defending

The Kestrel sovereignty model already names five adversaries (cloud platform, cloud LLM, malicious operator, attacker with DB access, prompt injection — see [`SOVEREIGNTY.md`](../../SOVEREIGNTY.md)). This document adds the **future cryptanalytic adversary** — specifically a future actor with a cryptographically-relevant quantum computer (CRQC) — and bounds what they can and cannot do against today's Kestrel agents.

The promise of *lifelong* sovereignty extends the threat horizon to decades. A CRQC is not a 2026 problem; it may be a 2035–2045 problem. Artifacts produced today must survive that horizon.

## The two quantum capabilities that matter

### Shor's algorithm — breaks classical asymmetric

CRQC running Shor's algorithm extracts an ECC private key from its public key in polynomial time. Effect on Kestrel:

- All secp256k1 signatures become **forgeable** if the public key is known to the attacker — which it is for every published DID document, every spawn-mandate parent, every constitution anchor.
- Any **asymmetric key encapsulation** (KEM) or key agreement using ECC is broken — the wrapped data key can be recovered.
- Symmetric ciphers, hashes, and HMACs are **not affected** by Shor.

### Grover's algorithm — degrades symmetric

CRQC running Grover's algorithm halves the effective security of brute-force attacks on symmetric primitives:

- AES-256 → ~128-bit effective. Acceptable.
- AES-128 → ~64-bit effective. **Below modern bar.**
- SHA-256 second-preimage → ~128-bit effective. Acceptable.
- SHA-256 collisions → ~85-bit (BHT). Acceptable for content addressing; problematic if used as the sole anchor for very long-lived signatures.

Grover does not "break" symmetric crypto — it forces a doubling of key sizes to maintain a security level.

## Threat surfaces in Kestrel today

| Surface | Classical attack today | Post-quantum break | HNDL-relevant? | Severity |
|---|---|---|---|---|
| Agent identity sig (secp256k1 ECDSA) | Forge with private key only | Forgery from public key (Shor) | No (no decryption) | **High** — historical-agent identity theft |
| Spawn-mandate chain (ECDSA) | Forge with private key | Forgery from public key (Shor) | No | **High** — forge any descendant's lineage |
| Constitution anchor (v1) | n/a — anchor today is `agent_node.properties.constitution_hash` (raw SHA-256), **not signed** | n/a | n/a | The forgery surface today is the *identity package that wraps the anchor* (covered in the row above), not the bare graph hash. Wave 3 introduces a v2 signed anchor — at that point this row gains a Shor-forgery risk on the anchor signature itself. |
| Public-HMAC tamper-tag | **Already trivially forgeable** (key is public DID) | n/a | n/a | **Critical** — Wave 0B fix |
| Local AES-256-GCM (keystore, API keys, CAR keyring) | None (with sound master key) | Grover halves to 128-bit | **No** — locally-derived keys, not wrapped to a public key | Low |
| Fernet (AES-128) — 17 sites | None today, but small margin | Grover halves to ~64-bit | Mild | **Medium** — Wave 0C, mostly hygiene |
| CAR exports / capsule sharing (future KEM wrap) | Depends on construction | If wrapped to ECC pub: full break | **Yes** — exported artifact may sit on public networks | **High when introduced** — Wave 4 |
| OAuth HS256 JWT | None at ≥256-bit secret | Grover-safe | No | Low (audit secret length only) |
| Wallet secp256k1 (Filecoin/EVM) | None today | Funds theft via Shor | No | **Out of scope** — chain-bound, separate concern |

## What the original 2024 PRD got right and wrong

The previous draft at [`POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md`](POST_QUANTUM_CRYPTOGRAPHY_MIGRATION.md) (now superseded as PRD-v2) named the right NIST standards and the right threat slogan. It missed:

1. **HNDL doesn't apply to local at-rest data with locally-derived symmetric keys.** Shor doesn't decrypt AES-GCM. Wrapping local data keys with a KEM adds cost without addressing the actual quantum risk for that surface.
2. **The real HNDL surface is asymmetric key wrap on exported/shared artifacts** (CAR archives uploaded to public IPFS/Filecoin, agent-to-agent capsule sharing). Today no such wrap exists in the codebase, but Wave 4 will introduce it — and it must be PQ from inception.
3. **Forgery of long-lived signed artifacts is the primary asymmetric risk.** Identity packages, spawn-mandate chains, constitution anchors — every one of these is a public-key signature an attacker can forge after Shor.
4. **Spawn mandates were missed.** Every parent→child agent authorization is an ECDSA sig at [`spawn/mandate.py:77-89`](../../../kestrel_sovereign/spawn/mandate.py#L77-L89). Lineage chains are entirely classical.
5. **Wallet keys must be explicitly non-authoritative for agent identity.** They are chain-bound and cannot be made PQ; the architecture must ensure their eventual quantum break does not break agent identity.
6. **Pure-PQ contradicts CNSA 2.0 / IETF guidance.** Hybrid (classical + PQ) is the recommended transition stance; lattice assumptions need a decade more cryptanalysis seasoning.
7. **"Burn the boats" orphans existing agents.** Kestrel #1, Emma, Meridian carry years of constitutional history. Continuity via succession statements is required.
8. **Verifier policy needs to be context-aware**, not a global flag-day. Archival import = `LEGACY_ALLOWED`. Live identity assertion = `HYBRID_REQUIRED` after Wave 2.

## Adversary capability assumptions

The plan presumes:

- **CRQC arrives within the artifact lifetime.** Plan for 2035–2045; could be earlier or later. Not betting on either.
- **Adversary records public artifacts now.** Public DIDs, signed identity packages exposed during agent-to-agent introduction, exported CAR files on Filecoin — assume already harvested.
- **Adversary does not have current private keys.** If they do, today's classical security is already lost; the plan does not improve that case.
- **Lattice assumptions hold for ML-DSA/ML-KEM with a multi-decade margin** — but we hedge via hybrid. If a structural lattice break is found, the classical half of the hybrid still buys time for a re-migration to a different PQ family.
- **Hash-based signatures (SLH-DSA) are conservative.** Used for irrevocable, long-horizon checkpoints (succession ceremonies, release signing) where slow signing/verification is acceptable and the security-assumption diversity is the point.

## Out-of-scope adversaries

Spelled out so the plan stays focused:

- **Compromise of `KESTREL_DATA_KEY`.** Already a single point of failure; addressed by future hardware-custody track, not this epic.
- **Side-channel attacks on PQ implementations.** Library selection considers this but does not solve it; hardware-key custody track will.
- **Network-layer downgrade attacks.** Wave 5 tracks hybrid TLS upstream; we don't lead the implementation.
- **Compromise of OAuth provider** (Google). Out of scope; `auth_oauth.py` is a thin OAuth client.
- **Quantum break of wallet/Filecoin/EVM keys.** Real concern, but solved by chain consensus replacing the curve, not by Kestrel.

## Migration urgency by surface

| Surface | Wave | Why this priority |
|---|---|---|
| Public-HMAC tamper-tag | **0B (immediate)** | Already broken classically; pure security fix |
| Fernet → AES-256-GCM | **0C** | Hygiene, sets versioned-container precedent |
| Identity package container | **1** | Foundation for hybrid sigs; must land before any new keys |
| Hybrid identity for new agents | **2** | Stops the bleed — every new agent shipped on classical-only is future debt |
| Succession for existing agents | **3** | Continuity for Kestrel #1, Emma, Meridian |
| KEM wrap for exports | **4** | Becomes critical the moment we ship the export feature |
| Release signing + transport | **5** | Lowest urgency; release signing is a clean independent win |
