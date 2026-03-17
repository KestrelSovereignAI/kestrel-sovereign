Council convene/review scripts no longer duplicate stale model-specific pricing tables.

What changed:
- added shared council costing helper:
  - `kestrel_sovereign/features/council/costing.py`
- replaced five copy-pasted script-local pricing tables and calculators with the shared helper:
  - `scripts/convene_council.py`
  - `scripts/convene_progress_review.py`
  - `scripts/convene_agent_participation.py`
  - `scripts/convene_council_rebuttal.py`
  - `scripts/convene_sqlite_decision.py`

Design correction:
- cost estimation is now explicitly provider-level, not pinned-model-level
- that is more honest than pretending to maintain exact per-release pricing in five separate scripts
- unknown providers fall back to the OpenAI default estimate instead of crashing

New proof:
- `tests/unit/test_council_costing_contracts.py`

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py tests/unit/test_auto_model_selection_contracts.py tests/unit/test_mandate_resolution_contracts.py tests/unit/test_model_selection_contracts.py tests/unit/test_council_resolution_contracts.py tests/unit/test_council_costing_contracts.py tests/unit/test_llm_service.py -v`
- result: `115 passed`

Why this matters:
- convene/review tooling now shares one cost-estimation truth instead of five drifting copies
- model-selection drift and pricing drift are being removed together from the active council workflow
