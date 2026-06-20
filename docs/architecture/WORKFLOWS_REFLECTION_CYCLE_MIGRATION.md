---
type: Architecture Spec
title: Reflection Cycle Workflow Migration
description: 'Parent: #1131. Sub-issue: #1145.'
resource: /docs/architecture/WORKFLOWS_REFLECTION_CYCLE_MIGRATION.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Reflection Cycle Workflow Migration

Parent: #1131. Sub-issue: #1145.

`cron.reflect` is owned by the optional `kestrel-feature-reflection`
package, but the core workflows feature now exposes the migration
contract in
`kestrel_sovereign.features.workflows.reflection_cycle`.

## Workflow Definition

The pilot workflow is named `reflection_cycle` and keeps the existing
cadence, `0 */4 * * *`, with default params:

```json
{"scope": "all", "depth": "normal"}
```

Stages map the current imperative flow:

| Stage | SourceRegistration | Gate | Compensation |
| --- | --- | --- | --- |
| `gather_observations` | `reflection.gather_observations` | `signal_status_ok` | `noop_idempotent` |
| `analyze_observations` | `reflection.analyze_observations` | `signal_status_ok` | `noop_idempotent` |
| `propose_improvement` | `reflection.propose_improvement` | `signal_status_ok` | `noop_idempotent` |
| `constitutional_review` | `reflection.constitutional_review` | `constitution_echo_verified` | `reflection.compensate_constitutional_review` |
| `council_review` | `reflection.council_review` | `council_approve` | `reflection.compensate_council_review` |
| `apply_or_defer` | `reflection.apply_or_defer` | `signal_status_ok` | `reflection.compensate_apply_or_defer` |

The optional reflection package must register those sources before
cutover. `reflection.constitutional_review` must be a COGNITION source
with `require_constitution_echo=True`.

## Shadow Validation

Before replacing the `cron.reflect` artifact handler, run old and new
paths side-by-side:

1. Keep the existing `reflect` scheduled task enabled.
2. Define and sign `reflection_cycle_spec_payload()` with
   `workflow_define`.
3. For each legacy `reflect` execution, run
   `workflow_run("reflection_cycle", {"scope": ..., "depth": ..., "shadow_legacy_result": ...})`.
4. Record `reflection_shadow_report(...)` with the legacy output,
   workflow output, and workflow run metadata.
5. Cut over only after shadow reports match for the operator's chosen
   soak window.

After cutover, `cron.reflect` should become a cron trigger that starts
`workflow_run("reflection_cycle", params)` instead of calling the legacy
artifact handler directly. Keep the old handler available behind a
rollback switch until the workflow has completed at least one retention
window without mismatch.

## Other ARTIFACT Sources

Use the same migration pattern for ARTIFACT sources that become
workflow-shaped:

1. Name the workflow after the existing task.
2. Split imperative steps into explicit Stage definitions.
3. Add `constitution_echo_verified`, `council_approve`, or other gates
   where they already exist implicitly.
4. Declare compensation for every writeful or irreversible stage.
5. Shadow-run before changing the scheduler's handler.
