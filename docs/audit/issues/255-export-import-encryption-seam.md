## Parent

Part of #255.

## Problem

Export/import, encryption, key rotation, storage receipts, identity, and sovereignty features each have proof, but the cross-feature seam can still silently downgrade encrypted state or lose receipt metadata.

## Goal

Prove export/import flows preserve encryption truth, key identity, storage receipts, and sovereignty metadata across the supported paths.

## Required scenarios

- encrypted conversation/export preview with failed decrypt remains visibly encrypted
- identity export/import preserves DID, SOUL, and receipt metadata
- sovereignty export/import preserves encryption flags and storage provenance
- key rotation or missing key material fails closed with legible errors

## Invariants

- ciphertext is never silently presented as verified plaintext
- failed decrypt does not mutate persisted metadata
- imported records retain enough provenance to audit where they came from
- storage receipts survive round trip or fail explicitly

## Proof expectations

- adversarial unit tests for decrypt/key-failure branches
- integration round trip for identity/sovereignty export-import
- update `docs/audit/SEAM_CAMPAIGNS.md` when proven
