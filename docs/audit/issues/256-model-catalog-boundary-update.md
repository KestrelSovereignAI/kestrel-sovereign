Resolved the model-catalog boundary and removed another duplicate config source.

What changed:
- made root `model_catalog.toml` the single live manual-override catalog
- deleted stale package-local `kestrel_sovereign/model_catalog.toml`
- updated `kestrel_sovereign/llm/model_catalog.py` to load the root catalog by default
- corrected `kestrel_sovereign/llm/model_selection.py` so it resolves concrete provider defaults from:
  - config intent in `llm_config.toml`
  - cached discovered models in `model_discovery_cache.json`
  - legacy `[featured]` only as a fallback for backward compatibility
- updated example files to reflect the intended split:
  - `llm_config.toml.example` uses `model = "auto"` + `selection_hints`
  - `kestrel.toml.example` now describes `llm.catalog` as manual overrides only

New proof:
- `tests/unit/test_example_model_config_contracts.py`
- updated `tests/unit/test_model_selection_contracts.py`
- existing `tests/unit/test_model_catalog.py` now runs against the root manual-override catalog

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py tests/unit/test_auto_model_selection_contracts.py tests/unit/test_mandate_resolution_contracts.py tests/unit/test_model_selection_contracts.py tests/unit/test_council_resolution_contracts.py tests/unit/test_council_costing_contracts.py tests/unit/test_example_model_config_contracts.py tests/unit/test_model_catalog.py tests/unit/test_llm_service.py -v`
- result: `157 passed`

Architectural outcome:
- config files now express provider intent and policy
- `model_discovery_cache.json` is the concrete startup cache
- `model_catalog.toml` is manual overrides only
- the repo no longer carries two conflicting model-catalog TOMLs
