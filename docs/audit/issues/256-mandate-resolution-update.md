Mandate/model-routing seam update is now in place.

What changed:
- removed the dedicated `feedback_audit_model` config from the active mandate path
- deleted the stale package-local `kestrel_sovereign/model_mandate.toml` so the root/unified config is the only live source
- made `model_mandate.toml` discovery-backed for cheap subquery selection via:
  - `cheap_model = "auto"`
  - `cheap_model_hints = [...]`
- normalized mandate selectors so provider names like `anthropic` resolve to the provider's current discovered model instead of leaking through as fake model IDs
- kept `get_audit_response()` as a runtime capability, but it now uses the normal provider-routing path instead of a special audit-model setting
- updated the active security docs/examples to describe the new routing model

New proof:
- `tests/unit/test_mandate_resolution_contracts.py`
- updated `tests/unit/test_llm_service.py`
- updated `tests/unit/test_context_management.py`

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py tests/unit/test_auto_model_selection_contracts.py tests/unit/test_mandate_resolution_contracts.py tests/unit/test_llm_service.py -v`
- result: `106 passed`

Why this matters:
- mandate/config now expresses intent and policy, not pinned release IDs
- audit behavior no longer depends on a dead special-model concept
- runtime selector resolution is more coherent across preferred, mandated, cheap, and default paths
