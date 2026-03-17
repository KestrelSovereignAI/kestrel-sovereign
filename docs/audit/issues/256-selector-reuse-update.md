Shared config-driven model selection is now reused beyond the core LLM runtime.

What changed:
- added `kestrel_sovereign/llm/model_selection.py` as a shared pre-discovery selector utility
- `scripts/generate_feature_docs.py` now resolves its default Anthropic/OpenAI models from:
  - `llm_config.toml` intent
  - `model_catalog.toml` featured-cache fallback
  instead of hardcoded release IDs
- council auto-selection is now real:
  - `kestrel_sovereign/features/council/deliberation.py` resolves `model = "auto"` before adapter init
  - council transcripts/token usage now record the resolved concrete model rather than the string `"auto"`
- collapsed another duplicate config seam:
  - `kestrel_sovereign/features/council/feature.py` now points to repo-root `council_config.toml`
  - deleted stale package-local `kestrel_sovereign/council_config.toml`
- converted active council configs to `model = "auto"`:
  - `council_config.toml`
  - `kestrel.toml.example`

New proof:
- `tests/unit/test_model_selection_contracts.py`
- `tests/unit/test_council_resolution_contracts.py`
- updated `tests/unit/test_generate_feature_docs.py`

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py tests/unit/test_auto_model_selection_contracts.py tests/unit/test_mandate_resolution_contracts.py tests/unit/test_model_selection_contracts.py tests/unit/test_council_resolution_contracts.py tests/unit/test_llm_service.py -v`
- result: `112 passed`

Why this matters:
- the same config/cache policy now governs core runtime, mandate routing, feature-doc generation, and council startup
- we removed another pair of split-brain config files
- active configs are moving away from pinned release IDs toward provider intent plus cached/discovered concrete models
