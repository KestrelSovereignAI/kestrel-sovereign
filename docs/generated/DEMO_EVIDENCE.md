---
type: Generated Reference
title: Kestrel Demo Evidence Index
description: Generated inventory of executable demos, kestrel-flight scripts, and
  kestrel-eye review configs.
resource: /docs/generated/DEMO_EVIDENCE.md
tags:
- demos
- generated-docs
- kestrel-flight
- kestrel-eye
timestamp: '2026-06-18T00:00:00Z'
status: generated
generated: true
canonical: false
source: /demos/
generator: scripts/generate_demo_evidence_docs.py
---

# Kestrel Demo Evidence Index

This generated reference links human demo docs to executable `kestrel-flight` demos and `kestrel-eye` visual review configs.

Regenerate with:

```bash
uv run python scripts/generate_demo_evidence_docs.py
```

| Demo | Script | Eye config | Expected screenshots | Test command | Narration |
|---|---|---|---:|---|---|
| `demo-isolation` | `/demos/demo-isolation/demo.cjs` | `/demos/demo-isolation/eye.toml` | 4 | `kestrel demo run demo-isolation` | `/demos/demo-isolation/narration.md` |
| `ephemeral-purge` | `/demos/ephemeral-purge/demo.cjs` | `/demos/ephemeral-purge/eye.toml` | 6 | `kestrel demo run ephemeral-purge` | `/demos/ephemeral-purge/narration.md` |
| `feature-store` | `/demos/feature-store/demo.cjs` | `/demos/feature-store/eye.toml` | 6 | `kestrel demo run feature-store` | `/demos/feature-store/narration.md` |
| `metrics` | `/demos/metrics/demo.cjs` | `/demos/metrics/eye.toml` | 9 | `kestrel demo run metrics` | `/demos/metrics/narration.md` |
| `privacy-modes` | `/demos/privacy-modes/demo.cjs` | `/demos/privacy-modes/eye.toml` | 7 | `kestrel demo run privacy-modes` | `/demos/privacy-modes/narration.md` |
| `spawn` | `/demos/spawn/demo.cjs` | `/demos/spawn/eye.toml` | 10 | `cd demos/spawn && npx playwright test --config=config.cjs` | `/demos/spawn/narration.md` |
| `tasks` | `/demos/tasks/demo.cjs` | `/demos/tasks/eye.toml` | 7 | `kestrel demo run tasks` | `/demos/tasks/narration.md` |
| `technical` | `/demos/technical/demo.cjs` | `/demos/technical/eye.toml` | 27 | `npx playwright test -c demos/technical/config.cjs` | `none` |
| `voice` | `/demos/voice/demo.cjs` | `/demos/voice/eye.toml` | 8 | `kestrel demo run voice` | `/demos/voice/narration.md` |

## Talon Gate

Use a demo-specific gate when a PR changes a documented UI workflow:

```talon-verify
kestrel demo run technical
kestrel-eye review --config demos/technical/eye.toml
```

Keep generated screenshots and video in `demo-output/` or CI artifacts unless a PR intentionally updates stable documentation evidence.
