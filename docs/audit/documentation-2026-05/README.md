# May 2026 Documentation Audit Workspace

This directory holds the working artifacts for the May 2026 documentation audit. It exists so the audit can reuse the older inventory/issue-batch pattern without scattering one-off review files through `docs/`.

Canonical ledger:

- [`../DOCUMENTATION_AUDIT_5_2026.md`](../DOCUMENTATION_AUDIT_5_2026.md)

Working artifacts:

- [`SHARED_CONTEXT.md`](SHARED_CONTEXT.md) - common context for all review lanes.
- [`LANES.md`](LANES.md) - lane assignments and expected report format.
- [`REPORT_INDEX.md`](REPORT_INDEX.md) - collected subagent reports and consolidated first slices.
- [`ARCHIVE_DECISIONS.md`](ARCHIVE_DECISIONS.md) - cleanup and supersession decisions made during this audit.
- [`lanes/`](lanes/) - focused prompts/briefs for each review lane.
- [`reports/`](reports/) - lane reports produced by agents or reviewers.

Rules:

- Review by system/feature lane, not by isolated document.
- Each lane report must cite file paths and concrete stale claims.
- Do not edit generated docs directly. Fix the canonical source, then regenerate.
- When an old audit/index/review artifact is no longer active, either move it under `docs/archive/` or add a clear superseded pointer.
- Keep public docs, internal planning, generated artifacts, and historical archives visibly separate.
