Removed hidden fallback models from the RunPod deployment path.

What changed:
- `kestrel_sovereign/features/runpod/ollama.py`
  - `start_ollama_pod()` no longer silently falls back to `qwen2.5:7b`
  - it now trusts the configured profile default model
- `kestrel_sovereign/features/runpod/core.py`
  - `resume_stopped_pod()` no longer silently falls back to `"flux"`
  - it now raises when the profile lacks `default_model`
- updated example/docstring references in:
  - `kestrel_sovereign/features/runpod/manager.py`
  - `kestrel_sovereign/features/runpod/__init__.py`

New proof:
- `tests/unit/test_runpod_model_contracts.py`
  - resume requires a configured profile default model
  - Ollama startup uses the profile default without hidden fallback
  - resume path does not invent a new model override

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py tests/unit/test_auto_model_selection_contracts.py tests/unit/test_mandate_resolution_contracts.py tests/unit/test_model_selection_contracts.py tests/unit/test_council_resolution_contracts.py tests/unit/test_council_costing_contracts.py tests/unit/test_example_model_config_contracts.py tests/unit/test_model_catalog.py tests/unit/test_runpod_model_contracts.py tests/unit/test_llm_service.py tests/integration/test_runpod_feature.py -v`
- result: `162 passed`

Why this matters:
- deployment profile config is now the single source of truth for those workload-bound model defaults
- the system fails clearly when a profile is incomplete instead of silently drifting to an unrelated baked-in model
