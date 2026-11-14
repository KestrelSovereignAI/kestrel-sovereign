---
type: Architecture Spec
title: Native artifact producer source successor
description: Explicit finite signed-run output admission through the canonical artifact authority; bounded native producer source contract.
resource: /docs/architecture/NATIVE_ARTIFACT_PRODUCER_SOURCE_CANDIDATE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-10-27T00:00:00Z'
status: implementation-candidate
owner: architecture
canonical: false
generated: false
privacy: public
---

# Native artifact producer: bounded source contract

This bounded implementation originated as a Core 0.57.0 source successor and is
now present in merged Core source. Merged source or release metadata alone does
not establish the native producer capability on a consuming host. Verify the
actual immutable publication, consumer normal-index floor/lock, installed native
producer API1 and host discovery identity API1, host registration and required
acceptance independently. Whole-source review, CI and required release approval
remain separate delivery gates.

The original operator service operation API remains version 1;
the additional discovery identity advertisement is separately versioned.

The ArtifactsFeature owns finite native output admissions and uses its existing
ArtifactAuthority, encrypted ArtifactService and PermissionStore audit. A plan,
audit row, service discovery identity, native output intent or tool permission
does not grant publication. The runtime user must explicitly approve one exact
accepted signed prepared run, selected output slots, plan hash and finite
limits. The genuine native Resume receipt must then activate that admission.

Only the original first-party SOLO policy mapping is supported for this new
approval. Custom and Claws policies remain denied here until they provide an
explicit retained owner contract; their existing native controls are unchanged.
Scheduler/registration windows and unattended HOST relation attachment are not
implemented. Preparation is inert and the native engine remains the executor.

Each publication obtains a fresh genuine HOST service operation and its exact
registration identity, original signed native stage Work, and an independent
per-operation artifact privacy/authority hold. No foreground caller is retained
or serialized. Exact feature, host registration, privacy epoch, configured key,
activation and finite policy are checked through the actual canonical mutation
and native acknowledgment. The producer facade only creates text artifacts in
the admitted original slot; it cannot edit, delete, attach or browse content.

Expiry narrows the original stage timer. Explicit revocation, supersession and
feature retirement notify the same original native cutoff/Stop owner in its
captured context. Observed key/privacy changes deny current operations and
retire the record; this does not promise an immediate key/privacy notification
hook while a handler waits. No new Task, executor, lease or control ledger is
created for revocation. Cleanup reports pending until the original native
effect, heartbeat, monitor and outer accepted children have actually drained.
Unknown native/cleanup debt remains pending or unknown. A task completion flag,
empty facade map or closed source frame is not a cleanup acknowledgment.

Canonical and native acknowledgments are separate. Unknown canonical commits
retain the original intent/publication key and quota charge. One current query
of that same key may establish factual canonical bytes/ref; it cannot fabricate
the native acknowledgment, replay a stage or attach a relation. Missing/error
or revoked queries remain unknown without private ref delivery. Fresh explicit
foreground reconciliation retains independent run.attach and artifact policy
through relation COMMIT and final private response delivery. Status GETs do not
reconcile, activate or mutate either store.

## Lifetime proposal bound

The [admission owner](../../kestrel_sovereign/features/artifacts/producer_admission.py)
stores at most **128 proposal records for the lifetime of its owning
`ArtifactsFeature` instance**. This is not a count of currently active approvals
or publications. All stored states count: `proposed`, `approval_pending`,
`approved`, `denied`, `expired`, `revoked` and `retired`. Denial, expiry,
supersession, revocation and `retire()` change state without removing a record
or making room for another proposal.

After 128 records have been stored, further insertions fail closed, even if no
approval is currently active. At the bound check, the in-process admission
methods raise `ArtifactAccessDenied("native output proposal bound reached")`.
The [HTTP proposal route](../../kestrel_sovereign/features/artifacts/producer_http.py)
maps that denial to HTTP 403 with `{"error":{"code":"artifact_access_denied"}}`;
it does not deliver the exception message. That generic code alone cannot
distinguish capacity exhaustion from other authorization denials. Earlier policy
or native-source checks may refuse a proposal before the bound is checked.
Repeated attempts do not replenish capacity. The
[feature lifecycle](../../kestrel_sovereign/features/artifacts/feature.py)
creates the admission owner once; disabling and re-enabling the feature retires
the same owner's records and does not replenish proposal capacity.

Core exposes no remaining-capacity count. A failed or unknown proposal response
may already have consumed a stored record, so counting successful responses is
not a reliable capacity measure. Plan long-lived instances around this finite
lifetime bound and the ambiguity of a generic denial. Any newly constructed
`ArtifactsFeature` instance starts with an empty live admission owner, including
when a host unloads and reloads an agent without restarting the process. The
replacement does not preserve or renew the old owner's live approvals, replay
old admissions, or prove physical native/publication cleanup. Do not remove
approval or tombstone records, renew authority or replay a run to bypass the
bound. Any future retirement, pruning or renewal design needs its own
original-custody proof.

Admissions are not restored or renewed when the feature instance is replaced,
including a process restart or agent reload. Missing live admission refuses new
effects; parking additionally requires the physically joined exact original
native owner and its still-valid claim/control CAS. An expired, replaced or
foreign native claim is never revived to force PAUSED.

Current proofs and their limits, failed runs, semantic mutations and exact
resource ownership are recorded outside the checkout. The source is not a
claim that the public Core floor or background artifact readiness is shipped.
