# Signals Workflows Talon Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

- Signal docs: `docs/architecture/SIGNAL_DISPATCHER.md`, `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- Workflow docs: `docs/architecture/WORKFLOWS_*`
- Talon/core feature docs: `README.md`, `KESTREL_FEATURES.md`, `kestrel_sovereign/data/feature_registry.toml`
- Code evidence: `kestrel_sovereign/signals/`, `kestrel_sovereign/signals/sources/`, `kestrel_sovereign/features/talon/`, `kestrel_sovereign/features/workflows/`

## Canonical Doc Recommendation

Use `docs/architecture/SIGNAL_SOURCES_GUIDE.md` plus `kestrel_sovereign/signals/dispatcher.py` docstring as the canonical signal model. Keep `SIGNAL_DISPATCHER.md` as architecture background, but refresh its inventory and status. Treat `WORKFLOWS_*` as design/proposal docs unless the missing workflows source is restored or confirmed extracted. Talon docs should say: standalone engine is `kestrel-talon`; in-core Kestrel only exposes a coordinator/control surface.

## Stale Or Conflicting Claims

- `docs/architecture/SIGNAL_SOURCES_GUIDE.md` says "all four built-in sources"; code has heartbeat, cron, Stripe, channel messages, `a2a.task_complete`, `a2a.task_submitted`, and `a2a.question_answered`.
- `docs/architecture/SIGNAL_DISPATCHER.md` omits `channel.message`, `a2a.task_submitted`, and `a2a.question_answered`; `memory_consolidate` is marked "likely" though code classifies it as ARTIFACT.
- `docs/architecture/WORKFLOWS_STAGE_TO_SIGNAL_MAPPING.md` references `kestrel_sovereign/features/workflows/models.py`, `schema.py`, `store.py`, `signing.py`; those source files are absent.
- `docs/architecture/WORKFLOWS_REFLECTION_CYCLE_MIGRATION.md` says core workflows exposes `kestrel_sovereign.features.workflows.reflection_cycle`; no such source exists.
- `docs/generated/FEATURES_developer.md` lists `workflows | WorkflowsFeature`; no workflow source files are present.
- `kestrel_sovereign/features/talon/coordinator.py` still says Agent Mesh Protocol preferred in one place, while later comments say mesh is gone and A2A replaced it.
- `kestrel_sovereign/data/feature_registry.toml` describes peers as "mesh networking"; generated docs repeat mesh wording.
- `kestrel_sovereign/data/feature_registry.toml` says Talon package is `kestrel-feature-talon` and `core = true`; `pyproject.toml` says `kestrel-talon` is out-of-tree and installed independently.

## Code/Package Evidence

- Signal registrations at startup: `kestrel_sovereign/kestrel_agent.py`, channel registration in `kestrel_sovereign/features/channels/feature.py`, scheduler cron registration in `kestrel_sovereign/features/scheduler/feature.py`.
- Current source inventory: `kestrel_sovereign/signals/sources/scheduler.py`, `heartbeat.py`, `wallet.py`, `channels.py`, `a2a.py`, `a2a_task_submitted.py`, `a2a_question_answered.py`.
- Dispatcher hook/signal boundary is explicit and current: `kestrel_sovereign/signals/dispatcher.py`.
- Talon runtime separation from chat LLM routing: `kestrel_sovereign/features/talon/runtime.py`.
- Talon policy/preference split and tool surface: `kestrel_sovereign/features/talon/runtime.py` and `kestrel_sovereign/features/talon/coordinator.py`.
- No workflow implementation source under `kestrel_sovereign/features/workflows/`; directory contains only `__pycache__`.

## Docs To Update

- `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- `docs/architecture/SIGNAL_DISPATCHER.md`
- `docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`
- `docs/architecture/WORKFLOWS_STAGE_TO_SIGNAL_MAPPING.md`
- `docs/architecture/WORKFLOWS_REFLECTION_CYCLE_MIGRATION.md`
- `kestrel_sovereign/data/feature_registry.toml`
- `KESTREL_FEATURES.md`
- Talon coordinator docstring in `kestrel_sovereign/features/talon/coordinator.py`

## Docs To Archive Or Mark Historical

- Mark `docs/architecture/WORKFLOWS_*` as proposal / not currently implemented in this repo unless workflows source is restored.
- Mark `docs/development/SCOUT_PLAN_CODE.md` historical/proposal if it remains a design sketch.
- Replace or mark mesh-era generated/user-facing language as historical after canonical inventory is corrected.

## Generated Docs To Regenerate

- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`

## Open Questions

- Was `kestrel_sovereign/features/workflows` intentionally extracted to an external package, accidentally deleted, or only planned?
- Is the Talon feature registry supposed to point to `kestrel-talon`, `kestrel-feature-talon`, or the in-core `TalonCoordinatorFeature`?
- Should `channel.message` be documented as a first-class current wake source alongside heartbeat/cron/A2A/Stripe?
- Should remaining "mesh" terms be globally replaced with A2A, or preserved only in historical docs?

## Suggested First PR Slice

Refresh signal/Talon truth first: update the signal source inventory in `SIGNAL_SOURCES_GUIDE.md` and `SIGNAL_DISPATCHER.md`, fix Talon coordinator wording from mesh to A2A, correct the peers/Talon registry descriptions, then regenerate the three `docs/generated/FEATURES_*` files. Keep workflow status changes as a second PR because they depend on the open implementation/extraction question.

