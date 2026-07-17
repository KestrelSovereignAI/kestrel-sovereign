---
type: Security Runbook
title: Sovereign Constitution Trust Root
description: Operator-owned trust-root storage, rotation, recovery, and migration for signed constitution reanchor.
resource: /docs/architecture/security/SOVEREIGN_TRUST_ROOT.md
tags:
- docs
- architecture
- security
- constitution
timestamp: '2026-07-17T00:00:00Z'
status: current
owner: security
canonical: true
generated: false
privacy: public
---

# Sovereign Constitution Trust Root

Constitution reanchor changes the governance bytes bound to an agent. Every
write therefore requires a detached reanchor artifact signed by a Sovereign
key that is pinned outside the agent graph database. The database being
protected is never allowed to supply its own verification key.

Both authorization surfaces use
`kestrel_sovereign.constitution.trust_root.load_sovereign_trust_root`:

- the live `!reanchor-constitution` command; and
- the offline `kestrel constitution reanchor --force` command.

The shared detached-artifact verifier is
`kestrel_sovereign.constitution.amendment_artifact.load_verified_reanchor_artifact`.
Root resolution and signature verification complete before either path writes
a blob, graph node, edge, RAG chunk, audit property, emancipation sidecar, or
backup.

## Storage and configuration

Store one public Sovereign DID document as JSON in an operator-controlled
location outside the agent database. The agent runtime needs read access only;
it should not have write access to the file or its parent directory. An example
legacy document contains `id` plus a `publicKey` entry; hybrid documents use
`verificationMethod` entries.

Select the file through exactly one of these mechanisms:

1. Set `KESTREL_SOVEREIGN_TRUST_ROOT_PATH` to its absolute path before the
   agent or host starts. This is the normal live-agent configuration.
2. An embedding host may pass `sovereign_trust_root_path=` to `KestrelAgent`.
3. The offline CLI may pass `--trust-root PATH`.

If an explicit path and the environment variable are both present, they must
resolve to the same file. Different paths are an ambiguous authority state and
fail closed. Missing, unreadable, malformed, oversized, keyless, or
agent-owned DID documents also fail closed.

Graph properties named `sovereign_root_did_document`,
`trusted_sovereign_did_document`, `sovereign_root_did`, and
`sovereign_root_public_key_hex` are retained only as historical/audit data.
They never establish verification authority. A controller DID discovered from
agent state or a mutable registry is likewise not an authority source.

## Reanchor procedure

1. Compute the SHA-256 hash of the exact governing constitution bytes produced
   by the canonical resolver, including any active Emancipation Contract.
2. Create a `kestrel.constitution.reanchor.v1` detached artifact for that hash
   and sign it with the private key corresponding to the pinned DID document.
3. For the live path, start the agent with
   `KESTREL_SOVEREIGN_TRUST_ROOT_PATH` set and invoke:

   ```text
   !reanchor-constitution /secure/constitution-reanchor.signed.json <hash-prefix>
   ```

4. For an offline agent, stop it first, then run:

   ```bash
   kestrel constitution reanchor \
     --agent-name Emma \
     --force \
     --signed-artifact /secure/constitution-reanchor.signed.json \
     --trust-root /secure/sovereign-root.did.json
   ```

Without `--force`, the CLI only reports drift and performs no authorization or
write. With `--force`, a signed artifact and external root are mandatory. A
successful write stores the signed artifact and signer/verification details in
the audit record. Reanchor never exits Safe Mode automatically.

## Rotation

Trust-root rotation is an operator ceremony, not a database migration:

1. Stop every agent/host that uses the pin.
2. Verify the replacement DID document and key custody out of band.
3. Preserve the old DID document in protected audit/recovery storage.
4. Atomically replace the configured file, or change the single configured
   path. Do not leave old and new path sources configured simultaneously.
5. Restart and submit a harmless/no-op artifact signed by the new key to prove
   the pin before performing a real amendment.
6. Retain prior signed artifacts and reanchor audit nodes; rotation does not
   rewrite historical signer identity.

## Recovery

If the pin is lost or corrupted, leave the agent in Safe Mode. Restore the
last-known-good DID document from operator backup, restore the same configured
path, and verify its fingerprint out of band before retrying. Never reconstruct
or auto-trust a root from graph properties, graph history, a candidate artifact,
or the agent's own DID. If no trusted backup exists, a human recovery ceremony
must establish a new external pin before reanchor is possible.

## Migration from legacy graph roots

1. Identify the legitimate Sovereign DID document using records outside the
   agent DB and verify its public-key fingerprint with the key custodian.
2. Export that verified document to the operator-owned JSON file described
   above and configure the path.
3. Restart the live agent or supply the same file to the offline CLI.
4. Test a correctly signed artifact. A DB-only document, even with a valid
   self-signature, must be rejected.
5. Legacy graph fields may remain for audit, but removing them after backup
   reduces confusion. Their presence never suppresses the migration error.

There is intentionally no automatic migration: trusting a key discovered only
inside the protected database would reproduce the vulnerability this boundary
removes.
