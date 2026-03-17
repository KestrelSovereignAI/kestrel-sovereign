Deeper runtime seam audit uncovered and fixed two real cross-cutting issues.

1. Privacy-mode model restore bug
- `POST /agent/privacy-mode` was only saving the explicit model mandate before forcing a local model.
- If the user was on default cloud routing with no explicit mandate, switching to a local-only privacy mode and then back to a cloud-allowed mode restored `None/None` instead of the active cloud route.
- Fixed in `endpoints/agent.py` by saving the resolved active provider/model when no explicit mandate is set.

Proof:
- `tests/unit/test_privacy_mode_model_restore_contracts.py`
  - default cloud route is restored after `normal -> isolated -> normal`
  - explicit cloud preference is restored after `normal -> isolated -> normal`

2. Built-in command inventory drift
- `/api/commands` maintained its own stale built-in command list that had drifted from `CommandHandler`.
- This omitted real agent-level commands such as `!sleep`, `!consolidate`, `!compress`, `!continue`, `!reload-context`, and `!heartbeat`.
- Fixed by making `kestrel_sovereign/command_handler.py` the single source of truth via `BUILTIN_COMMAND_SPECS`, and deriving the endpoint inventory from that list.

Proof:
- `tests/unit/test_commands_conversations_endpoint_contracts.py`
  - `/api/commands` built-ins match `CommandHandler` specs exactly
  - verifies presence of previously missing built-ins

Verification:
- `uv run pytest tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py -v`
- result: `8 passed`

Expanded focused proof bundle:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py tests/unit/test_privacy_mode_model_restore_contracts.py -v`
- result: `62 passed`

This is the kind of whole-of-vision seam drift the audit is meant to catch: the individual endpoints looked plausible in isolation, but the shared invariants were not actually holding.
