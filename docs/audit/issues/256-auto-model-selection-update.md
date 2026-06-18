---
type: Issue Body
title: 256 Auto Model Selection Update
description: Reworked the default model-selection path so shipped config no longer
  depends on pinned provider model IDs.
resource: /docs/audit/issues/256-auto-model-selection-update.md
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

# 256 Auto Model Selection Update

Reworked the default model-selection path so shipped config no longer depends on pinned provider model IDs.

What changed:
- `llm_config.toml`
  - primary providers now use `model = "auto"` instead of version-pinned model IDs
  - added `selection_hints` so config still expresses intent at the model-family level
  - this makes the config the single source of truth for startup selection policy without requiring constant version churn
- `kestrel_sovereign/llm/model_discovery.py`
  - auto-resolution no longer picks the first discovered chat model by accident
  - selection order is now:
    1. config `selection_hints`
    2. featured discovered models
    3. ranked non-hidden discovered models
  - fallback ranking prefers tool-capable, non-preview models and uses newer timestamps when available
- `kestrel_sovereign/llm/provider_registry.py`
  - anthropic/google/vertex initializers now tolerate omitted models and normalize to `auto`
  - this aligns them with the other providers instead of requiring a pinned `model=` value

New proof:
- `tests/unit/test_auto_model_selection_contracts.py`
  - selection hints beat discovery order
  - featured beats plain fallback when no hints exist
  - preview models are avoided when a stable alternative exists
  - shipped `llm_config.toml` uses `auto` for the primary providers

Verification:
- `uv run pytest tests/unit/test_auto_model_selection_contracts.py tests/unit/test_llm_service.py -v`
- result: `40 passed`

Expanded focused verification:
- `uv run pytest tests/unit/test_auto_model_selection_contracts.py tests/unit/test_llm_service.py tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py -v`
- result: `102 passed`

Architectural outcome:
- Before discovery, config says which providers participate and what family-level model policy is desired.
- On startup, disk cache resolves concrete model IDs immediately when available.
- After discovery, the concrete model IDs are refreshed from provider APIs and the cache is updated.
- This removes the need to keep editing the shipped config for every new Sonnet/GPT/Gemini release while preserving a clear config-owned policy.
