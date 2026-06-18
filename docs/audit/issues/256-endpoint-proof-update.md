---
type: Issue Body
title: 256 Endpoint Proof Update
description: Expanded direct contract proof across the remaining public endpoint families.
resource: /docs/audit/issues/256-endpoint-proof-update.md
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

# 256 Endpoint Proof Update

Expanded direct contract proof across the remaining public endpoint families.

New focused suites:
- `tests/unit/test_models_configuration_endpoint_contracts.py`
- `tests/unit/test_commands_conversations_endpoint_contracts.py`
- `tests/unit/test_security_endpoint_contracts.py`
- `tests/unit/test_secondary_endpoint_contracts.py`

This closes the main route-family gaps left after the earlier auth, sovereignty, identity/constitution, and canonical-doc work.

Covered endpoint families now include:
- auth decision table and route classes
- commands
- conversations and transcripts
- database explorer
- files
- observability events and summary
- saved-items CRUD/search/schema/tag flows
- sovereignty export/import/files/stats
- identity and constitution
- models, keys, wallet, IPFS, model selection
- security permissions, approvals, audit, session reset
- OpenAI-compatible endpoints

Verification:
- `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py -v`
- Result: `55 passed`

The umbrella audit issue should remain open because broader feature-level and adversarial/system seam proof still remains, but the public route contract slice is now materially stronger and much more explicit.
