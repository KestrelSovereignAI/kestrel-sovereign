# Lane Brief: Signals Workflows Talon

Goal: reconcile wake-source docs, workflow ownership, and Talon standalone vs in-agent control-surface boundaries.

Start with:

- `docs/architecture/SIGNAL_DISPATCHER.md`
- `docs/architecture/SIGNAL_SOURCES_GUIDE.md`
- `docs/architecture/WORKFLOWS_FEATURE_DESIGN.md`
- `docs/architecture/WORKFLOWS_DEVELOPER_GUIDE.md`
- `docs/architecture/WORKFLOWS_STAGE_TO_SIGNAL_MAPPING.md`
- `docs/architecture/WORKFLOWS_REFLECTION_CYCLE_MIGRATION.md`
- `README.md`
- `KESTREL_FEATURES.md`
- `kestrel_sovereign/signals/sources/`
- `kestrel_sovereign/features/talon/`

Check for:

- workflow docs that still imply in-core ownership after extraction
- missing wake-source inventory
- confusion between hooks and signals
- Talon docs that conflate standalone `kestrel-talon` with Kestrel chat routing
- missing distinction between Talon runtime preferences and `[talon.policy]`
- A2A vs retired mesh terminology

Report to: `reports/signals_workflows_talon_report.md`

