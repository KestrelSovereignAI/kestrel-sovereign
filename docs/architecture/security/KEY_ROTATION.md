---
type: Architecture Spec
title: Key Rotation Mechanism
description: Internal maintenance service for resumable rotation of registered encrypted stores, with explicit coverage and host integration limits.
resource: /docs/architecture/security/KEY_ROTATION.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-24T00:00:00Z'
status: experimental
owner: architecture
canonical: false
generated: false
privacy: public
---

# Key Rotation Mechanism

`KeyRotationService` is an internal maintenance service. Its Python API is
implemented; the module's command-line entry point only parses arguments and
prints a stub message. There is no working rotation CLI, agent command, or
host endpoint. This document describes the service contract, not a deployed
operator workflow.

## Maintenance ownership and keys

The host must provide one service owner per database, stop concurrent legacy
encrypted-data writers, and retain both original keys until the journal is
`COMPLETED`. The service has a process-local admission lock and durable journal
checks; it does not provide a distributed maintenance lease or online writer
cutover. Rotation changes rows incrementally, so readers cannot assume every
row uses the same key while maintenance is in progress.

Use generated Fernet-format master keys. Salted passphrase derivation currently
differs between runtime encryption and this service; such deployments cannot
reliably rotate through this API. This existing limitation must be resolved
before enabling an operator workflow for passphrase keys; tracked in
[#3726](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3726).

The service does not change `KESTREL_DATA_KEY`, secrets files, runtime cipher
caches, or running agents. The host must verify completion and perform its own
key configuration/cipher reload before restoring normal access. Retain the old
key for any backups or other encrypted stores outside this registry.

## Covered data

The encrypted-table registry in
[`key_rotation.py`](../../../kestrel_sovereign/security/key_rotation.py) covers:

- `conversation_history.content` and `rendered_content`, with per-agent key
  derivation when `agent_id` is present;
- encrypted `files.content` rows; legacy plaintext files are skipped;
- wrapped roots in `artifact_keys`, `tool_effect_keys`,
  `visual_computer_keys` and `mcp_task_keys`, when their tables exist in the
  same database. An existing envelope table must contain valid key rows;
- version-2 `mcp_input_scopes.wrapped_key` roots, authenticated to their
  original agent scope during opening, rotation and resumed verification.

`ArtifactsFeature` uses this rotation boundary for its encrypted saved-output
repository in the agent's database. Stop the host before rotating its key and
restart it with the new key after verification. Rewrapping an envelope root
preserves artifact addresses and content keys. It is not recovery from a
compromised envelope root. Graph properties, external backups and unregistered
stores are outside this service's coverage. In particular, encrypted channel
messages and agent identity resources are not registered; coverage and safe
cutover are tracked in [#3728](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3728).

## MCP input-envelope migration

Before rotating, initialize every original agent's input store under its
original data key. Input schema revision 2 records a wrapping version per scope.
Initialization opens a version-1 root with the original agent-purpose key and
rewraps the same random root under the global master with scope/version AEAD
context. It overwrites the old envelope, preserving checkpoint ciphertext,
HMAC identities, admission counters and response claims. A version-2 root never
tries the old purpose key. Missing or invalid original keys cannot be replaced.

Both start and resume validate the complete registry before journal changes or
work launch. An absent version column, unmigrated scope or unsupported wrapping
version refuses rotation with a migration-required diagnostic. The service does
not initialize agents or change their schemas itself. Migration requires the
original agent identity and key; the scope hash is not a substitute for either.

## Journal and failure behavior

`initialize()` creates `key_rotations` and `rotation_progress`. A rotation
journal records key fingerprints, status, counters and a failure message identifying the record. Each ciphertext update, exact write verification and progress
receipt commit together. The entire database rotation is not one transaction.

Both counting and rotation select conversation/file rows using the runtime
`metadata.enc` flag. Ciphertext-like plaintext is left unchanged; flagged
ciphertext with a damaged or missing prefix still fails closed. A NULL optional
`rendered_content` has no ciphertext to rotate. Same-key AEAD upgrades verify
and rewrite all flagged rows, including those already in AEAD format.

Every selected encrypted row must succeed. A decrypt, malformed mandatory
root, or write-verification failure stops the run and saves `FAILED`; the
service does not silently skip failed encrypted records and report completion.
Cancellation leaves an `IN_PROGRESS` journal. `shutdown()` cancels and joins
owned tasks before storage closes. Completed tasks remain owned until observed.
Drain surfaces original native control, commit and retirement failures after
joining every accepted owner; a FAILED journal is not a replacement for that
evidence. Only a plain cancellation authenticated as this drain's own shutdown
request is suppressed. Independent source cancellation and observer withdrawal
remain observable.

Resume the same failed or interrupted journal using its original new-key file
while the original old key remains configured. The service checks both key
fingerprints, rebuilds the processed count from committed progress receipts and
refreshes the total from the current complete registry, including input roots
that older journals did not count. Already-rotated roots are verified with the
same scope context under the new key before their progress receipt is accepted.
Correct the recorded failure before retrying; supplying an unrelated key does
not repair it.

Both start and resume use the same admission guard. Any running task blocks
another admission. An unresolved journal also blocks another rotation: an
`IN_PROGRESS` journal, or a `FAILED` journal with durable progress. Only that
exact journal may resume. A historical zero-progress attempt cannot take over
a partially applied transition.

Older databases may already contain inconsistent history, such as a partial
failed journal followed by a later completed rotation. The service refuses
that state; there is no automatic journal abandonment or rollback API. Diagnose
and repair it with the original keys and backups before attempting maintenance.

## Python API example

This example assumes initialized storage, exclusive maintenance ownership,
quiesced writers and the original key already configured. It does not perform
the host's key cutover or restart.

```python
from kestrel_sovereign.security.key_rotation import (
    KeyRotationService,
    RotationStatus,
)

async def rotate_key(storage, new_key_file):
    service = KeyRotationService(storage)
    await service.initialize()
    try:
        rotation_id = await service.start_rotation(new_key_file=new_key_file)
        await service.drain_rotations()
        record = await service.get_status(rotation_id)
        if record is None or record.status is not RotationStatus.COMPLETED:
            raise RuntimeError(f"Rotation {rotation_id} needs recovery: {record}")
        return rotation_id
    finally:
        await service.shutdown()
```

Recovery uses `await service.resume_rotation(rotation_id, key_file)` followed
by `drain_rotations()` and a status check under the same maintenance ownership.
`upgrade_to_aead()` uses the rotation machinery with the current key to upgrade
legacy Fernet ciphertext to AEAD; it has the same ownership and key-derivation
constraints.

## Verification and remaining integration

The recovery integration suite exercises SQLite and native PostgreSQL failure,
resume, key identity, silent-write rejection and competing-journal admission.
It does not prove host lifecycle integration, distributed maintenance ownership,
backup recovery or an operator-facing command. Those must be implemented and
verified before exposing this service as a supported maintenance workflow.
