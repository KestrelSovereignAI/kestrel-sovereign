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
- `artifact_keys.wrapped_key`, when the artifact envelope table exists in the
  same database. An existing envelope table must contain valid key rows.

Artifact support here prepares the rotation boundary for the saved-output
repository; it does not enable an artifact service. Rewrapping an envelope root
preserves artifact addresses and content keys. It is not recovery from a
compromised envelope root. Graph properties, external backups and unregistered
stores are outside this service's coverage. In particular, encrypted channel
messages and agent identity resources are not registered; coverage and safe
cutover are tracked in [#3728](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3728).

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
owned tasks before storage closes.

Resume the same failed or interrupted journal using its original new-key file
while the original old key remains configured. The service checks both key
fingerprints and rebuilds the processed count from committed progress receipts.
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
