---
type: Architecture Spec
title: Hosted feature recovery without business startup
description: Permanent attended maintenance boot policy with exact feature inventory and native lifecycle verification.
resource: /docs/architecture/MAINTENANCE_FEATURE_RECOVERY.md
status: proposed
privacy: public
owner: architecture
canonical: false
generated: false
tags:
- lifecycle
- recovery
- features
---

# Hosted feature recovery without business startup

An attended host may need to reconstruct an existing agent's full feature
registry before Core can verify and exit a durable
`feature_lifecycle_uncertain` restriction. A constitution-only object or an
empty registry is not that proof. Repeating a constitutional audit does not
repair feature-lifecycle uncertainty.

Before calling the real `initialize()`, the host claims the existing readiness
boundary with `agent.defer_agent_readiness_to_host(maintenance=True)`.
`agent.maintenance_boot` then remains true for that object. There is no
release/downgrade API. Normal callers retain the existing no-argument behavior.

For repair, also pass `expected_features={...}` in that same preboot call.
Core freezes this complete inventory (plus mandatory features) and refuses
native lifecycle verification if any feature is missing, unexpected or
natively inactive. This uses the original contribution registry's object
identity, not a feature's product-facing `enabled` flag: a provider-free
VisualIdentity feature may have no image provider while remaining genuinely
activated. An object left loaded after native deactivation cannot certify
repair, even if the remaining registry is internally consistent.
A maintenance boot without this explicit inventory can inspect state
but cannot certify repair. Caller mutation of the supplied set cannot change
the bound inventory, and a second claim cannot replace it.

The real boot phases still construct features and their contribution registry
and may reach genuine `BootPhaseState.READY`. This is deliberately different
from publishing runtime workers or declaring application health. Maintenance:

- Keeps Core's original resident-publication event closed, including after a
  successful native SafeMode exit or idempotent `initialize()`.
- Defers both the initial ready-hook pass and late feature ready notifications.
- Does not run inline A2A wake recovery/startup consumers, session-start hooks,
  sync startup or the salvage janitor.
- Prepares the real Delivery queue without reclaiming old in-flight rows or
  starting its worker.
- Registers Scheduler contributions without seeding/removing durable schedules
  or arming polling. Retirement joins original owners/publications without
  creating a new optional stopped-status write.
- Refuses public streaming/nonstreaming turn admission independently of the
  durable SafeMode and Hold latches.

Participating external features must also hold their **inline** startup work.
In particular Workflows must construct its store/runner/wait contribution but
not recover runs, restore waits, reconcile triggers or start recovery from
`post_all_features_loaded` or `on_agent_ready`. A resident task barrier by
itself is too late: claiming work can happen before that task is created.

## Host responsibilities

This policy is not an authorization grant, a provider sandbox, a replacement
execution-custody object, or a guarantee about arbitrary third-party feature
code. Do not use it to publish a serving agent or execute tools manually.
The host must:

1. Retain the original live operator lease, exact companion/effective-owner
   generation and native commit validation for the entire boot/exit/retirement.
2. Use the real configured feature composition and verify its intended nonempty
   inventory, contribution wiring and enabled state. Optional load failures
   must not silently reduce the recovery proof.
3. Supply genuinely provider-free business resource policy and isolated runtime
   configuration; audit every selected feature's startup and shutdown paths.
   Disabling network egress alone does not prevent claims/retries.
4. Keep the candidate private and out of host serving caches and fleet polling.
5. Call the native lifecycle verifier and authorized `exit_safe_mode` without
   setting READY, registry, safety, audit or repair flags. Native constitutional
   attestation and durable exit CAS remain mandatory. An already passed receipt
   for the current governing bytes does not require a replacement model audit.
6. Physically join every original startup/worker/shutdown owner before closing
   operational/advisory pools, including on cancellation or native uncertainty.
   A timeout or cancellation request is not a join.

After successful recovery, retire the maintenance generation completely. The
normal hosted factory must create and accept a fresh serving generation under
its own original authority. Recovery evidence cannot manufacture that admission.

The component/unit and native SQLite integration tests demonstrate the Core
policy and real restart/exit plumbing. They do not establish PostgreSQL fleet
custody, the complete Frinz feature inventory, external vendor behavior or
production deployment acceptance; those remain separate delivery gates.
