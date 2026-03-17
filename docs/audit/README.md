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
