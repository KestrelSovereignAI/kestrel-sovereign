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

### Native publication and recovery custody

The native runtime and offline signed writers reserve their complete graph
write set in canonical order before writing file blobs or ownership rows.
Inside that owning transaction they compare the exact governing pointer,
rights, receipt/history and edge witness against the facts inspected before
authorization. A concurrent change requires a fresh inspection and signed
repair attempt; unrelated identity metadata is preserved, not overwritten.
Historical plaintext must hash to its addressed digest before it can supply
legacy emancipation-rights evidence. Readable corrupt bytes are unreadable
evidence, not permission to treat an active contract as dormant.
Native edge deletion and avatar publication participate in the same graph-first
order; they must not acquire edge/file ownership before reserving graph rows.
Signed file publication retains both the physical blob and exact tenant owner
through commit. Restoring a missing owner never copies another tenant's shared
blob metadata into the new ownership record.
Both the constitution and its exact detached signed artifact use that same
byte-validation and ownership-custody path. An absent locking-read result is
not custody, even if a later ordinary read can see a replacement.

A signed same-hash repair is still a repair. It restores missing verified
public constitution content or this agent's file ownership, records the new
signer and retains the previous reanchor receipt as history. Existing content
must exactly match the signed bytes; corrupted content is refused. Because
the governing hash did not change, an existing genesis receipt is preserved
and the content audit is not repeated merely for key rotation.
This also applies when signed repair restores a lost operative pointer from
unambiguous validated historical evidence: unchanged bytes preserve the exact
completed receipt, including rejection. Pointer loss is not a new constitution.
The signed writer retains a physical blob lock while validating those bytes,
including validating the actual winner of a concurrent absent-row insertion.
Restoring ownership inserts only a missing witness and preserves existing
per-agent filenames and provenance metadata.

Genesis content-audit results are published against their captured governing
and runtime-state witness. The native publisher merges only the new receipt
into a freshly locked identity and publishes its new runtime CAS token after
commit. If the witness changed, cognition reports a blocked genesis audit
without overwriting the newer governance, rejection or identity metadata.
Conversation notices are delivered only after the authoritative receipt commits;
volatile privacy buffers cannot retain a success notice from SQL rollback.
An ordinary SQL notice-delivery failure is reported separately and cannot
replace the committed audit outcome or cause that hash to be audited again.
The locking reads must actually return each captured existing edge and the
current edge's own tenant witness; a later ordinary reread is not a substitute.
Required-target attestations also retain the actual target node and this
agent's ownership witness, not just an edge pointing to that target.
Initial bootstrap refuses any prior governance evidence rather than expanding
its lock set and pruning history without a signed repair.

Safe Mode exit verifies again inside its owning transaction. PostgreSQL holds
the identity/ownership, governing edges/ownership and constitution file/owner
rows through the exit commit; SQLite retains its native writer custody. Both
backends require actual physical identity and governing-edge ownership rows:
serialization does not substitute for existence. Exit decrypts and hashes the
locked tenant-owned native blob; an ISOLATED session cache is not durable
integrity evidence. Periodic integrity verification, governing-text retrieval
and genesis input read the bound native store as well, not that cache. A passed
genesis publication re-attests native bytes inside its commit owner after the
auditor await; concurrent blob corruption cannot publish a fresh pass.
Successful ordinary integrity-audit publication likewise repeats verification
under native custody before resetting the durable audit deadline. Direct and
streaming requests recheck hash-bound genesis readiness after acquiring their
turn boundary, so queued requests cannot inherit admission for an older hash.
Admission also rereads the native runtime revision/generation and restrictions;
graph, governing-file and runtime-row custody retain a consistent snapshot in
one bounded native transaction, without holding locks over provider work.
another replica's committed Safe Mode cannot be hidden by a local flag. A
task-local witness pairs the admitted turn with its passed governing hash and
receipt. Later governing-text retrieval refuses a changed pointer or receipt
instead of supplying a newly repaired but unaudited constitution to cognition.
Commit-time attestation refusal propagates the final integrity verdict to
explicit diagnostics and periodic observers; the earlier positive diagnostic
is not reported as successful publication or mislabeled as a storage outage.
New-identity bootstrap uses the same exact-native-byte
publisher as signed repair, only after resolver verification and the durable
single-use bootstrap fence, so volatile privacy storage cannot consume authority
without publishing the constitution. Birth-record replication reserves its full
graph and payload-blob set before file-owner writes, while still publishing
files before nodes for tenant admission. Source and destination bytes are
verified, including existing physical conflict winners. Same-named file custody
cannot claim another tenant's private graph properties; public shared-content
metadata must pass canonical admission. A missing runtime identity cannot be
restored from a frozen birth record once its constitutional lifetime was
consumed. Signed recovery must preserve its existing audit history.

Native inception checks deterministic identity/lifetime custody before minting,
then checks again after audit/provider awaits. Keys remain in a private staging
directory until final native admission, so even `force` cannot replace active
keys on a refused birth. A partial filesystem publication restores the original
active files; an uncertain database commit retains the published keys for
recovery rather than deleting potentially committed identity material.

Genesis receipt history has one shared publication/runtime limit of 128 entries.
A repair requiring a 129th entry, or encountering already-overflowed or malformed
history, refuses atomically and preserves all existing evidence. Neither runtime
nor offline repair truncates receipts to make room. Repeating signed repairs is
not permission to discard earlier content-audit verdicts.
SQLite turn admission and final successful-audit/Safe-Mode-exit attestation reserve
the writer slot before their first governance read, avoiding stale deferred WAL
snapshots after unrelated writers commit. No provider work runs in that span.
Surviving runtime events also veto unsigned replay if the current state row is
lost. Such identities are not pending first-boot targets; offline tools retain
the runtime target and return a structured refusal, not an uncaught exception.
Existing-identity restore likewise refuses to treat surviving lifetime history
as legacy migration: it creates a restricted new generation requiring explicit
authorized recovery, never an unrestricted automatic audit. Recovery also
requires the native feature registry's independent repair proof:
lost runtime state cannot prove that the prior restriction was constitutional
rather than a quarantined feature lifecycle. Recreating a lifetime checks its
surviving history inside the native allocation transaction, not only before it.
The populated-runtime upgrade assigns nonempty immutable `legacy:` generations
without resetting restrictions, timestamps, counters or pending markers. Those
generations permit ordinary audited turn admission but cannot authorize a new
automatic first anchor; an old pending bit is not new-identity authority.
Frozen birth replay likewise refuses a migrated pending lifetime rather than
restoring it as a newly created identity.
Doctrine metadata
writers use native compare-and-swap to preserve concurrent completed receipts.
Bootstrap status, description, rename and overlay metadata writers merge only
their own fields into a freshly locked native identity. An old identity read
cannot replace newer governance, and a failed write does not mutate its read
object or report a live rename. Runtime genesis audit, turn admission and
commit-time exit reconcile matching current and historical terminal receipts
before granting authority or calling an auditor. A missing or pending current
receipt cannot reroll a historical rejection; contradictory terminals refuse
without choosing a favorable result. History validation is bounded.
Completed same-content genesis receipts survive an incorrect operative pointer
as well as a missing one. Conflicting completed evidence is refused rather than
selecting a favorable result. Supported legacy timestamp/risk/hash receipts are
normalized through the same validator as runtime migration; malformed matching
current or historical receipts are refused, not silently skipped and rerolled.
Verified signed repair publishes governance in
the native control-plane transaction in volatile privacy modes too; ordinary
feature-facing privacy restrictions remain unchanged.
Inception uses the same exact-byte native publisher, and a post-commit SQL
conversation-notice failure does not undo creation or skip remaining completion
work. Under its owning graph and lifetime custody, inception refuses an existing
physical identity or any prior constitutional lifetime, including a consumed
lifetime whose root was deleted. Deterministic DID reuse requires authorized
recovery, not new key minting over old governance. Pre-commit publication failures remove only that attempt's newly minted
identity artifacts and close/remove its internally created database; an external
database remains caller-owned, and committed identity artifacts are preserved.
Cancellation receives the same rollback cleanup, including while the genesis
auditor awaits before publication. Inception refuses caller-owned transactions
before minting. Once its publication body has completed, lost commit delivery is
uncertain even if the identity or its ownership rows later disappear: minted
keys are retained, never deleted based on a tenant-filtered absence. If its
outcome cannot be read, identity keys are retained rather than erased.
Native SQLite advertises its optional savepoint extension explicitly. Standard
SDK adapters retain the argument-free top-level transaction contract; adapters
that only join nested scopes refuse isolated avatar publication before writes.
Avatar publication retains and verifies the actual decrypted blob and exact
tenant owner before publishing a graph reference or identity avatar pointer.
The input digest alone is not evidence that a conflict winner contains the
image; corruption rolls back isolated publication even when a caller catches
the refusal and commits its surrounding transaction.
Physical lock ordering uses PostgreSQL's canonical `C`
collation; target equality does not depend on database locale.
An existing genesis receipt must validate as a literal pass, not merely be
terminal. Failure leaves the durable restriction and exit-event history intact.
After a changed-hash repair the public genesis audit/readiness path can complete
the new hash's pending audit while Safe Mode remains active. Explicit authorized
exit follows that real audit; repeating a same-hash repair does not reroll it.
Physical edge locks follow ownership-then-edge order, matching native deletion.
Exit locks only its reserved current governing target, not unrelated stale-edge
ownership held by cleanup. Other publishers reserve the complete captured set
and refuse set changes without acquiring new endpoints out of order.

All native SQL entry points, including script execution and SQLite diagnostic
reads, refuse SQL transaction-control commands while an owning transaction is
open. Callers must use the adapter's transaction context rather than issuing
`BEGIN`, `COMMIT`, `ROLLBACK`, savepoint commands or `PREPARE` through SQL.
SQLite scripts in an owned transaction execute statement-by-statement without
the driver's implicit pre-script commit; trigger bodies and PostgreSQL
dollar-quoted and SQL-standard `BEGIN ATOMIC` routine bodies, and
newline-continued escape strings retain their database
semantics. This lexical refusal
protects the commit boundary; it is not a general SQL authorization parser.
If a conflict policy or trigger implicitly rolls back SQLite's native owner,
the adapter poisons the scope: subsequent work and successful completion are
refused rather than escaping into autocommit. Explicit nested savepoint scopes
make avatar publication independently rollback-safe even if an outer caller
catches rejection and commits unrelated work. Default nested scopes remain
joined for compatibility.

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

## Genesis content audit and publisher authority

### Runtime audit deadlines on PostgreSQL

Core 0.53.24 preserves the absolute UTC instant of runtime-state and transition
timestamps through PostgreSQL's timezone-aware binding. Older versions could
shift them by the operator process's local timezone. No blanket offset
backfill is safe: historical writers may have used different timezones.
On restart, a future last-successful-audit time now requires a fresh native
full integrity verification before cognition, just like an expired deadline.
Only successful verification persists the actual current audit time; failure
retains Safe Mode and does not invent a successful historical timestamp.
The future-deadline verifier first persists the existing audit-required
interaction marker. Restart honors that marker even after the future time
has elapsed, including when saving the later failure/Safe Mode result failed.
If the marker cannot be saved, verification/readiness remain fail-closed.
This does not repeat the genesis content evaluation or alter its receipt,
the governing constitution bytes, or signed reanchor authority.

Source authentication and the genesis content audit answer different questions.
The operator-controlled resolver and detached signatures establish which exact
bytes may govern an agent and authorize their adoption. The content audit
evaluates those bytes for safety, privacy, sovereignty, clarity, and unauthorized
changes. It cannot verify runtime controls merely from claims in the document.

The platform-authority contract (#3423) permits publisher revisions, including
Book I and the governance frame, followed by an existing agent's explicit signed
adoption. This is not permission for a hosted agent or a lower Book to rewrite
the base. Nor does a publisher signature make harmful provisions safe, or make
adoption blanket consent to data disclosure or training. The contract does not
promise that every future publisher release preserves an immutable safety floor.

Genesis audit specification 2 makes this distinction explicit and retains risk-3
rejection. Newly completed pass and failure receipts record `audit_spec_version`
and `audit_prompt_sha256`, binding the actual submitted prompt as well as the
existing constitution digest. These are audit evidence, not authority to adopt a
source. Legacy receipts retain their meaning; installing a new specification
does not invalidate a prior failure, convert it to a pass, or rerun it. An
operator's reviewed reevaluation must preserve the old failure evidence and
use the normal signed recovery and fail-closed native acceptance gates.

## Deploys that change the governing constitution

A deploy can change the governing constitution, for example a release that
amends the packaged text. Every agent stays anchored to the old hash, so
restarting onto that code would boot each of them into Safe Mode until the
reanchor procedure above runs (#3517). Every deploy path therefore runs one
check first, `kestrel_sovereign.constitution_adoption.check_constitution_adoption`.
It is `kestrel doctor`'s drift check: it reads each agent's anchored hash
from the agent database, read-only, and computes the governing hash through
the canonical resolver, as the startup integrity audit does. It does not need
the host to be up.

The code that renders a constitution is part of what a deploy replaces, and a
process that has imported `kestrel_sovereign` keeps that code after an install
replaces it on disk. So once code is installed, it is judged by
`check_installed_constitution_adoption`, which runs the same check in a fresh
interpreter launched the way the host is: same Python, the project as working
directory, and the launcher's environment. Before an install, only the
deploying process's own resolver exists to judge the incoming bytes with.

| Path | What is judged | On a mismatch |
|---|---|---|
| `kestrel update [name]` | Before any step runs: the revision `git pull --ff-only` will land on, after a fetch. With `--no-pull`, or with no source checkout, the code on disk. Whenever a pull, install or feature step runs, every local agent is judged, because the package is shared; a name narrows only the restart. Its restart step then judges what was installed, in a fresh interpreter, for the same agents. | The update is refused with exit status 5. Before any step, nothing is pulled, installed or restarted. At the restart step the new code is installed but no agent is stopped. With `--no-restart` the first check refuses only while a blocking agent is running, since its periodic integrity audit reads the constitution from disk; otherwise it warns. |
| `kestrel restart [name]` | The installed code, in a fresh interpreter, for the named agent or every local agent. | Refused with exit status 5 before anything is terminated. |
| Restart coordinator | A `restart_only` request: the installed code, in a fresh interpreter, before the request is claimed. An `update_then_restart` request: the fetched revision, before the update checks it out, and then the code the update installed, in a fresh interpreter, before the restart. | The request ends in the terminal `refused` status. Its `status_reason` names the agents and both hashes, and says when the update was already installed. |

The refusal names each agent, its anchored hash, the governing hash, and the
adoption steps below. Two findings refuse:

- the anchored hash differs from the governing hash;
- the governing constitution cannot be produced at all: an untrusted
  descriptor, a blank source, or new package bytes that no longer match a
  `package` descriptor's pinned digest.

The audit puts the agent in Safe Mode in both cases. An agent whose anchor
cannot be read (an encrypted database, no agent node) is reported as not
verified. It does not block the restart.

A deploy replaces the package. A descriptor-selected external source is
operator configuration, so it is read from disk.

### Adopting a new constitution

`--allow-constitution-safe-mode` on `kestrel update` or `kestrel restart`
restarts anyway. Use it only when you are about to run the ceremony. The
coordinator has no override, so an agent cannot restart the fleet into Safe
Mode. `kestrel start` is not gated. Either order works:

- **Offline.** `kestrel terminate` first: a running agent's periodic
  integrity audit reads the constitution from disk, and the Safe Mode it
  enters persists across restarts. Then `kestrel update --no-restart`
  installs the new code. Reanchor each agent with `kestrel constitution
  reanchor --force` and an artifact signed for its new hash, then
  `kestrel start`. The agents boot already anchored.
- **Live.** `kestrel restart --allow-constitution-safe-mode`, or the same
  flag on `kestrel update`. Then on each agent run
  `!reanchor-constitution <artifact> <hash-prefix>` and `!safe-mode exit`.

`kestrel update` hashes the incoming bytes with the resolver it was started
with. Its restart step checks again, in a fresh interpreter, against what was
installed. That second check catches a revision that changes how the resolver
or the Amendment VIII rendering produces the governing bytes, a feature step
that upgraded the core package, and an upstream that moved between the gate's
fetch and the pull. It refuses before anything restarts, leaving the new code
installed and the running agents on the old code. Adopt the constitution
before they restart, and before their next periodic integrity audit if the
installed text itself changed.

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
