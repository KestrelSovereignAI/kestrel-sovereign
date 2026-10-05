---
type: Security Runbook
title: Sovereign Constitution Trust Root
description: Operator-owned trust-root storage, rotation, recovery, and migration for signed constitution reanchor and Sovereign-signed governing-constitution source descriptors.
resource: /docs/architecture/security/SOVEREIGN_TRUST_ROOT.md
tags:
- docs
- architecture
- security
- constitution
timestamp: '2026-10-02T00:00:00Z'
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

The same root also verifies governing-constitution **source descriptors**
(#2553), which choose the source whose bytes govern an agent; see
[Custom governing constitution sources](#custom-governing-constitution-sources).

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

### Hosted PostgreSQL agents without a local anchor

An embedding host that stores agent identities in PostgreSQL, with no local
`kestrel_prime.db`, must not invent a local anchor to use the offline CLI. Its
operator can call `reanchor_constitution` with `agent_dir=None`,
`hosted_agent_did=` set to the exact DID from the host's authoritative tenant
registry, `runtime_backend="postgres"`, and an explicit `runtime_dsn=`. The host
must independently verify the selected DID belongs to the intended tenant;
the API cannot infer that relationship from a caller-supplied display name.

Stop hosted agent writers first, take a PostgreSQL snapshot, and run a
`force=False` preview for each DID. A forced call still requires the same
signed artifact and operator-owned trust root as the CLI. The write is
transactional and DID-scoped, but **does not take a file backup**. Verify the
result and subsequent integrity audit before restarting hosted agents. A
missing or unowned agent node is a refusal, not a reason to create or retarget
one. Never infer a DID from the first row in the shared graph table.

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

## Custom governing constitution sources

By default the packaged `KESTREL_CONSTITUTION.md` (`config.CONSTITUTION_PATH`)
governs every agent: inception anchors it, and the periodic integrity audit,
explicit verification, reanchor, and `kestrel doctor` recompute from it. A
Sovereign may instead govern an agent by another constitution through a
**source descriptor**: a detached JSON artifact, signed by the pinned trust
root, that names the source and pins its content.

### Descriptor format

`kestrel_sovereign.constitution.source_descriptor` defines
`kestrel.constitution.source.v1`. Every field below is covered by the
signature; any other top-level field is refused rather than ignored.

| Field | Meaning |
|---|---|
| `artifact_type` | `kestrel.constitution.source.v1` |
| `version` | `1` (an integer; JSON `true` is refused) |
| `signer` | The pinned trust root's DID |
| `subject` | `constitution_source` |
| `source_kind` | `package` or `external` |
| `source_path` | Absolute path of an `external` source; `null` for `package` |
| `content_sha256` | SHA-256 of the source's raw bytes, before Amendment VIII rendering |
| `created_at`, `reason` | Provenance |

It carries exactly one of `signature` (legacy secp256k1, verified against the
root's `publicKey`) or `signatures` (hybrid, `HYBRID_REQUIRED` against the
root's `verificationMethod`), checked by the same verifier as reanchor
artifacts. `artifact_type` is part of the signed bytes, so a reanchor
artifact's signature can never pass as a descriptor's.

An `external` descriptor selects another file. A `package` descriptor keeps
the packaged source but pins its digest, so a package upgrade or edit must be
re-signed before the agent accepts it.

### Where authority comes from

Which source governs is decided by operator configuration and the trust root
only. Nothing in the agent's graph database selects a source. The resolver is
`kestrel_sovereign.constitution.resolver.resolve_governing_source`, and
inception, the audit, both reanchor writers, and doctor all call it.

Configure the descriptor through exactly one of:

1. `KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH` (process-wide; in a
   multi-agent host this applies to **every** co-hosted agent);
2. `constitution_source_descriptor = "/abs/path"` on one agent in
   `multi_agent.toml` (an absolute path is required);
3. `constitution_source_descriptor_path=` on `KestrelAgent` or
   `create_kestrel_identity_async`.

An explicit path and the environment variable may both be set only when they
resolve to the same file, the same rule as the trust root. With no descriptor
configured the package governs and no trust root is needed. Nothing changes
for an agent that never opts in.

Every launch path gives an agent the same answer:

- The in-process fleet host passes the per-agent descriptor to `KestrelAgent`.
  Its own environment is the launcher's (`paths.spawned_agent_env`), where the
  project `.env` outranks an exported value.
- A managed subprocess (`ProcessManager.start_agent`) cannot receive
  constructor arguments and starts with `multi_agent.toml` loading disabled.
  The launcher resolves the per-agent descriptor against the launch
  environment under the conflict rule above, then sets the child's
  `KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH` to the one resolved file, or to
  an explicit blank when none is configured. A conflicting or missing
  descriptor refuses the launch.
- `kestrel shell`'s in-process fallback passes the agent's descriptor too.
  It refuses to start when the shell exports
  `KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH` or
  `KESTREL_SOVEREIGN_TRUST_ROOT_PATH` with a value the project `.env`
  overrides, as offline reanchor does, rather than audit a source the
  launched agent does not.
- `kestrel doctor` and `kestrel constitution reanchor` resolve from the
  per-agent setting and the launcher's environment.

### Fail-closed semantics

A configured descriptor never falls back to the package. Each of these is an
integrity failure (Safe Mode at audit; refusal at inception and reanchor; a
doctor `fail`):

- the descriptor file is missing, unreadable, oversized, or not JSON;
- the configuration is ambiguous;
- the trust root is missing, ambiguous, malformed, or names the agent's own
  DID;
- the descriptor is unsigned, carries both signature forms or an unsigned
  field, names another signer, or fails its signature;
- the selected source is missing, unreadable, or empty, or its raw bytes no
  longer hash to `content_sha256`.

The hash comparison still applies. An agent whose anchored hash does not
match the selected source, rendered with its anchored Amendment VIII
contract, fails its audit. Moving an agent between sources therefore always
needs a signed reanchor.

### What the database records

Inception under a descriptor writes `constitution_source_receipt` on the
agent node. Both reanchor writers add `source_kind` and, when a descriptor
was used, `source_content_sha256`, `source_descriptor_path`,
`source_descriptor_sha256`, and `source_descriptor_signer` to the
`constitution_reanchor` receipt. These are audit evidence only. No code path
reads them to choose a source. A database writer who flips a recorded kind
from `package` to `external`, or plants a DID document in the legacy root
properties, changes nothing the resolver reads, and package-drift enforcement
still holds.

### Creating a descriptor

Sign on the operator host that holds the Sovereign key, never inside an
agent:

```python
import hashlib, json
from pathlib import Path
from kestrel_sovereign.constitution.source_descriptor import (
    build_legacy_signed_source_descriptor,  # or build_hybrid_signed_source_descriptor
)

source = Path("/secure/constitutions/acme.md")
descriptor = build_legacy_signed_source_descriptor(
    signer_did="did:pkh:eip155:1:0x…",          # the pinned root's DID
    source_kind="external",
    source_path=str(source),
    content_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    private_key=sovereign_private_key,
    reason="Acme governing constitution v3",
)
Path("/secure/constitution-source.signed.json").write_text(json.dumps(descriptor))
```

Keep the descriptor and the source file read-only to the agent runtime, like
the trust root.

### Procedures

- **New agent:** configure the descriptor and the trust root, then incept.
  `constitution_path`, if passed at all, must be the descriptor's source.
- **Existing agent:** configure the descriptor, then reanchor with an artifact
  signed over the SHA-256 of the selected source **as rendered for this
  agent**. For a dormant agent that is the source's own digest. Live:
  `!reanchor-constitution <artifact>`. Offline: `kestrel constitution
  reanchor --agent-name X --force --signed-artifact …`.

  The offline command anchors the source the agent will audit. It reads the
  agent's `constitution_source_descriptor` and the project `.env`, not
  whatever the shell exports, and it refuses rather than choosing when:

  - the shell exports `KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH` or
    `KESTREL_SOVEREIGN_TRUST_ROOT_PATH` with a value the project `.env`
    overrides (an empty value in `.env` counts as a value);
  - `--source-descriptor` names a file other than the agent's configured
    descriptor. The flag only confirms that selection;
  - `--trust-root` is given for a descriptor-governed agent whose `.env`
    names no trust root. The agent could not verify its own descriptor and
    would enter Safe Mode whatever was anchored.
- **Editing an external source:** sign a new descriptor for the new bytes,
  then reanchor as above. Until both happen the agent is in Safe Mode, which
  is the intended result of an unsigned edit.
- **Returning to the package:** remove the descriptor configuration, then
  reanchor to the package hash.

### Limitations

- The descriptor selects the governing bytes that are anchored and audited.
  The computer-use capability gate still reads Amendment IX grants from the
  packaged text and from the anchored per-agent overlay. An external source
  cannot grant capabilities through that gate.
- Replaying a different, still-valid descriptor is an operator-configuration
  change, in the same trust domain as replacing the trust-root file. The
  database cannot do it.

### Security properties

- **Path and content are both signed.** `source_kind`, `source_path`, and
  `content_sha256` are all in the signed bytes. Changing the kind, pointing an
  `external` descriptor at another file, or editing the pinned digest fails the
  signature.
- **Replay and downgrade.** A descriptor carries no expiry or counter, so an
  older descriptor the Sovereign once signed still verifies. Using one needs
  write access to operator configuration, and it changes which bytes govern
  the agent. Those bytes then no longer match the anchored hash, so the agent
  Safe-Modes until it is reanchored with a Sovereign-signed artifact for that
  hash; an older artifact for the same hash also verifies. Rotate the trust
  root to retire every descriptor and artifact it signed.
- **No check-then-use gap.** The descriptor is parsed from the same bytes
  whose signature was checked. The governing source is read once. Those bytes
  are hashed against `content_sha256`, rendered, then anchored or compared;
  they are never re-read from the path.
- **Malformed input is refused, not raised.** A non-string `source_kind`, a
  non-list `signatures`, or a signature entry whose `alg`, `kid`, or `sig` is
  not a string makes the descriptor untrusted. It never escapes doctor's or
  reanchor's untrusted-descriptor handling as a `TypeError`.

## Migration from `constitution_path` overrides

Releases before #2463 accepted any `constitution_path` at inception and
reanchor. #2463 refused non-packaged paths because the audit always
recomputed from the package. #2553 restores custom sources through
descriptors.

1. **Callers that pass `constitution_path=`** to `create_kestrel_identity` /
   `create_kestrel_identity_async`: sign a descriptor for the file and pass
   `constitution_source_descriptor_path=` (or set the environment variable),
   with a trust root configured. Passing the path alone is still refused as
   non-authoritative.
2. **Agents incepted from a custom file before #2463**, which Safe-Mode at
   every audit: sign a descriptor whose `content_sha256` is the digest of the
   exact file they were incepted from. Configure it for that agent only,
   restart, run `!verify-constitution`, then `!safe-mode exit`. No database
   write or reanchor is needed: the anchored hash already matches. If the
   file has since changed, restore the original bytes or follow
   "Editing an external source".
3. **`kestrel constitution reanchor --constitution-path PATH`**: configure
   `constitution_source_descriptor` on the agent (or the environment variable
   in the project `.env`), then reanchor. Without `--constitution-path` the
   command anchors the governing source. With it, the path must be that
   source.
4. **Ambiguous legacy rows** (an agent anchored to bytes that no configured
   source reproduces) stay in Safe Mode and read-only. They are never
   migrated automatically, and graph properties never establish a source.

## Release notes (#2553)

- New: Sovereign-signed governing-constitution source descriptors
  (`kestrel.constitution.source.v1`), verified against the existing
  operator-pinned trust root.
- New configuration: `KESTREL_CONSTITUTION_SOURCE_DESCRIPTOR_PATH`, the
  per-agent `constitution_source_descriptor` in `multi_agent.toml`,
  `constitution_source_descriptor_path=` on `KestrelAgent` and inception,
  `sovereign_trust_root_path=` on inception, and `--source-descriptor` on
  `kestrel constitution reanchor` (a confirmation of the agent's configured
  descriptor, never a replacement).
- Managed subprocess launches carry the per-agent descriptor into the child,
  and refuse to start on a conflicting or missing one.
- `kestrel constitution reanchor` resolves the descriptor and trust root from
  the project `.env` the agent launches with, and refuses an exported value
  that disagrees with it.
- Changed: `kestrel constitution reanchor` without `--constitution-path` now
  anchors the resolved governing source rather than always the package.
- Changed: `kestrel doctor` checks drift against each agent's resolved
  source. An untrusted descriptor is reported as a failure.
- Fixed (#3451): `kestrel doctor` fails, rather than skipping the check,
  when a source a descriptor selected is missing, unreadable, or empty.
  An unreadable packaged constitution no longer hides the checks of
  agents governed by an external source. `kestrel shell` refuses a
  descriptor or trust-root export that the project `.env` overrides.
- Unchanged: agents with no descriptor configured resolve, audit, and
  reanchor exactly as before, and need no trust root to boot.
