---
type: Review Lane
title: Package Boundaries
description: Review prompt for the Package Boundaries lane of the May 2026 documentation
  audit.
resource: /docs/audit/documentation-2026-05/lanes/package_boundaries.md
tags:
- audit
- documentation
- may-2026
- review-lane
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Lane Brief: Package Boundaries

Goal: determine the current source of truth for what ships in `kestrel-sovereign`, what is an external feature package, what is a provider package, and what is a standalone tool.

Start with:

- `README.md`
- `KESTREL_FEATURES.md`
- `pyproject.toml`
- `kestrel_sovereign/data/feature_registry.toml`
- `docs/guides/BUILDING_FEATURES.md`
- `docs/architecture/core/MODULAR_RUNTIME.md`

Check for:

- stale claims about `pip install kestrel-sovereign`
- inconsistent `core = true` usage
- extracted features still listed as core
- provider packages described as feature packages
- feature packages described as provider packages
- standalone tools such as `kestrel-talon` described as in-core agent features
- generated docs that inherit stale claims

Report to: `reports/package_boundaries_report.md`

