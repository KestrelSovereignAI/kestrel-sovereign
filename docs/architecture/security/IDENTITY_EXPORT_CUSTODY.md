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
