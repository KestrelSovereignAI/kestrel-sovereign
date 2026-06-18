---
type: Review Lane
title: Llm Routing
description: Review prompt for the Llm Routing lane of the May 2026 documentation
  audit.
resource: /docs/audit/documentation-2026-05/lanes/llm_routing.md
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


# Lane Brief: LLM Routing

Goal: reconcile docs for LLM configuration, model preference, provider capability contracts, provider packages, streaming, honesty markers, and transport behavior.

Start with:

- `docs/architecture/LLM_SERVICE_ARCHITECTURE.md`
- `docs/architecture/LLM_PROVIDER_CAPABILITIES.md`
- `docs/architecture/llm/PROVIDER_PLUGINS.md`
- `docs/architecture/llm/HONESTY_LAYER.md`
- `README.md`
- `pyproject.toml`
- `kestrel_sovereign/llm/`

Check for:

- references that imply `llm_config.toml` is still read
- duplicate model preference paths
- unclear `kestrel-llms` vs in-repo provider ownership
- provider capability docs that disagree with SDK contracts
- stale streaming/reasoning marker claims
- missing Codex/Claude/OpenAI transport caveats

Report to: `reports/llm_routing_report.md`

