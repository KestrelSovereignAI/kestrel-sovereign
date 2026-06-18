---
type: Issue Body
title: 273 Progress Update
description: Audience-doc generation pipeline has been structurally verified against
  the new canonical inventory.
resource: /docs/audit/issues/273-progress-update.md
tags:
- docs
- audit
- issue-body
timestamp: '2026-06-18T00:00:00Z'
status: snapshot
owner: documentation
canonical: false
generated: false
privacy: public
---

# 273 Progress Update

Audience-doc generation pipeline has been structurally verified against the new canonical inventory.

Completed:

- `KESTREL_FEATURES.md` rewritten as the canonical source
- legacy catalog archived
- generator prompt updated so investor output no longer hardcodes stale metrics
- guardrail tests added
- dry-run validated for all audiences

Verification:

- `uv run pytest tests/unit/test_generate_feature_docs.py tests/unit/test_feature_doc_canonicality.py -v`
- `uv run python scripts/generate_feature_docs.py --all --dry-run`

Current blocker:

- actual generated output files were not produced because `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are not available in the current environment

When keys are available, this issue should move to:

1. generate all audience docs
2. review the outputs for drift or ugly prompt artifacts
3. tighten prompts if needed
4. decide whether to check generated docs into version control
