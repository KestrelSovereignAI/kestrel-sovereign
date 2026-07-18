---
type: Security Architecture
title: Identity Export Custody
description: Private publication and metadata-only remediation of local continuity packages
status: active
---

# Identity export custody

A local identity export contains portable continuity state: DID, constitution,
memories, personality, relationships, skills, saved items, and calibration
material. It is a plaintext custody artifact even when its internal signature
protects integrity. A signature does not provide confidentiality.

## Publication contract

The ordinary `export_identity` local tier, its IPFS/Filecoin downgrade fallback,
and `identity.signing.sign_and_export` share one protected writer. On supported
POSIX hosts it:

1. Opens every existing path component without following symbolic links.
2. Creates or validates an operator-owned export directory and sets it to
   `0700`.
3. Writes the complete payload into a new sibling regular file, forces mode
   `0600`, and fsyncs the file.
4. Atomically links the complete inode into the collision-resistant final name.
   An existing final entry is never followed or overwritten.
5. Fsyncs the directory and removes staging state. A failed write or publish is
   reported as failure and does not leave a partial final package.

The tool still returns the generated `identity_*.json` import path. A tier
downgrade returns the same restorable local path and remains a partial result
because the requested remote tier was unavailable.

The returned path is absolute and bound to the active agent. Runtime placement
uses the following precedence:

1. the agent's `identity_export_dir` from `multi_agent.toml`;
2. the process manager's internal `KESTREL_IDENTITY_EXPORT_DIR` child binding;
3. the intentional standalone `KESTREL_DATA_DIR` override;
4. the active agent data root (`storage_path` / `KESTREL_DB_PATH`); and
5. the historical `agent_data` fallback when no runtime binding exists.

A relative `identity_export_dir` is resolved below that agent's `data_dir`, so
the same relative value on two agents still produces two distinct custody
roots. Process-managed children receive their resolved root in
`KESTREL_IDENTITY_EXPORT_DIR`; a host-level `KESTREL_DATA_DIR` is not copied
into every child as a shared export destination or repurposed for unrelated
storage consumers.

`sign_and_export(..., replace_existing=True)` is the only replacement seam.
Replacement must be explicit and is accepted only when the destination:

- is named `identity_*.json`;
- is a non-link regular file owned by the current operator; and
- is below `KESTREL_DATA_DIR`, `AGENT_DATA_DIR`, or the supplied identity key
  storage root.

The new payload is fsynced privately before an atomic replacement. The previous
inode is retained privately until the new directory entry is durable, allowing
rollback if publication fails. The default remains no-clobber.

If the host cannot provide descriptor-relative, no-follow POSIX operations,
local plaintext export fails closed instead of weakening the custody contract.

## Legacy exports

Older releases commonly created `agent_data` as `0755` and
`identity_*.json` as `0644`. Kestrel does not silently open, parse, move, or
delete those packages during migration.

Run:

```bash
uv run kestrel doctor
```

Doctor performs a metadata-only scan of direct `identity_*.json` entries under
configured data roots. It reports unsafe root/file modes, links, non-regular
entries, and ownership problems as a warning. It reports counts and roots, not
package contents.

Doctor and `identity harden-exports` resolve the exact same root list. They read
identity-placement values from the target project's `.env`, then overlay the
live process environment so an exported `KESTREL_IDENTITY_EXPORT_DIR`,
`KESTREL_DATA_DIR`, or legacy `AGENT_DATA_DIR` wins. Relative values in either
source resolve against the target project directory. Other environment values,
including credentials, are not copied into this metadata-only resolver.

To remediate eligible entries:

```bash
uv run kestrel identity harden-exports
```

The command changes only filesystem metadata. It sets an operator-owned export
root to `0700` and owner-matching, non-link regular `identity_*.json` children to
`0600`. It refuses links, foreign-owned files, non-regular entries, and anything
outside configured data roots. A refusal produces a non-zero exit so an
operator can investigate without Kestrel reading or disclosing the artifact.

Existing exports remain importable after permission hardening; package bytes and
the returned import pathname do not change.

## Import and verification intake

`!identity import` and `!identity verify` share one bounded package loader. A
local source must be an operator-owned regular file whose group/other write bits
are clear; legacy read-only modes such as `0644` remain accepted. Kestrel opens
the final path entry with no-follow and nonblocking flags, validates the opened
descriptor, and reads at most 64 MiB on a worker thread. Links, directories,
FIFOs, sockets, devices, ownership mismatches, writable files, path-swap races,
invalid UTF-8, and oversized packages fail before parsing or mutation.

CID sources use the same 64 MiB decoded-package ceiling. The Filecoin/IPFS
adapter bounds local-cache and streamed network input as well as decompression
and decrypted output, preventing a compressed response from bypassing the
limit. Expected intake failures return source metadata and a sanitized reason;
they do not log or echo package bytes.

## Fresh-target signature trust

New signed exports carry an `identity_trust` bundle containing public
verification methods and the ordered, signed succession statements needed to
identify the current signer. It contains no private keys. The bundle is part of
the v2 content hash, so changing a link or public key invalidates the package
signature as well as the link's own predecessor/successor signatures.

Verification still needs a trust bootstrap. Kestrel applies these rules:

1. A `did:pkh` or `did:key` root is self-certifying. The verifier proves that
   every root Multikey derives the DID, then walks the signed chain offline.
2. A root `did:web` is not self-certifying. A fresh receiver must supply an
   `IdentityTrustPolicy` whose `trusted_root_verification_methods` exactly pin
   the root keys. Repeating only the DID is insufficient.
3. Arbitrary network resolution and package-declared `did:web` methods are
   never trust anchors. A package cannot make its own keys authoritative.
4. Succession revocations come from the receiver's
   `revoked_succession_ids`. They are intentionally not carried as an
   authoritative package field because a compromised exporter could omit its
   own revocation.

Self-certification answers “which key controls this DID,” not “is this the DID
the operator intended to restore.” Receivers that expect a specific agent
should set `trusted_root_did` even for `did:pkh` / `did:key`; a mismatch then
fails before import. For a chain-less compromised root, exclude it by pinning
the expected root/allowlist. `revoked_succession_ids` applies only to actual
succession statements and is not a root-DID denylist.

For the ordinary legacy-to-hybrid rotation, the package DID remains the
self-certifying legacy `did:pkh` root while the active Ed25519 + ML-DSA-65
successor signs the package. A package copied from source directory A therefore
verifies and imports in an empty target directory B without copying source
private custody or using the network. Verification output reports
`trust_root=<did>` and the exact `chain_evidence=<statement ids>` used.

Born-hybrid `did:web` agents have no self-certifying predecessor. Their packages
remain portable, but the root key pin must travel through a separate trusted
operator channel. Applications may pass `IdentityTrustPolicy` directly to
`verify_package_signature` / `IdentityImporter.import_package`; the identity
feature tools accept the equivalent `identity_trust_policy` object. Archival
SLH-DSA enforcement is optional receiver policy and requires its public-key pin.

Packages written before this format remain compatible with same-host local-key
or local-DID-document verification. They cannot acquire a safe portable trust
chain retroactively; re-export them from the source host after upgrading when a
fresh-target disaster-recovery artifact is required.
