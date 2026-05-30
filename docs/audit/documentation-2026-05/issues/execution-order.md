# Documentation Audit Execution Order - May 2026

Parent ledger: `docs/audit/DOCUMENTATION_AUDIT_5_2026.md`

## Recommended Order

1. Truth sources: `KESTREL_FEATURES.md`, `feature_registry.toml`, package boundary docs.
2. Main entry points: `README.md`, `docs/README.md`, `docs/architecture/README.md`.
3. Volatile architecture: context, memory/retrieval/storage, LLM routing, signals/workflows/Talon.
4. Operator and user docs: deployment, user guides, demo scripts, generated feature docs.
5. Diagrams, archive cleanup, public-release hygiene, validation checks.

## Why This Order

- Generated docs should not be regenerated until their canonical inputs are correct.
- Main entry points should not repeat unstable details; they should link to corrected sources.
- Architecture docs need one ownership story before user docs can be made honest.
- Archive cleanup is safest after lane reviewers identify which artifacts are still active.

