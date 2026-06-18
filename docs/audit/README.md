# Kestrel Whole-of-Vision Audit

This directory is the canonical source for the GitHub issue bodies used to track
the feature-by-feature audit, inspection, and red-team program derived from
`KESTREL_FEATURES.md`.

Operating rules:

- Test first: write or tighten failing proof before implementation changes.
- Fix root causes only: no spackle, compatibility shims, or duplicate logic.
- One source of truth: when behavior diverges, consolidate to a single canonical path.
- Whole-of-vision: audit feature seams and cross-domain interactions, not just isolated modules.
- Lower layers first: unit, integration, e2e, adversarial, load, and real-resource tests in that order.

Issue bodies live in `docs/audit/issues/`.
Worktree strategy lives in `docs/audit/worktree-plan.md`.

Current documentation audit:

- [`DOCUMENTATION_AUDIT_5_2026.md`](DOCUMENTATION_AUDIT_5_2026.md) - May 2026 audit ledger for package extraction, context, memory, storage, LLM, signals, Talon, cloud, generated docs, and public-doc hygiene.
- [`documentation-2026-05/`](documentation-2026-05/) - working audit workspace with shared context, lane briefs, subagent reports, stale-artifact review, and execution order.
- [`OKF_MIGRATION_PLAN.md`](OKF_MIGRATION_PLAN.md) - plan and implementation ledger for converting Kestrel docs to OKF and wiring documentation freshness into Talon/Flight/Eye workflows.
- [`index.md`](index.md) / [`log.md`](log.md) - generated OKF concept index and timestamp log for this audit tree.

OKF checks:

```bash
uv run python scripts/docs_okf.py validate
uv run python scripts/docs_okf.py validate --all docs/audit/documentation-2026-05
uv run python scripts/docs_okf.py index --check
uv run python scripts/docs_okf.py log --check
```
