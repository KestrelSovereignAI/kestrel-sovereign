---
type: Issue Body
title: 256 Feature Discovery And Regeneration Update
description: Fixed a real feature-discovery drift bug and regenerated the derived
  audience docs against the corrected canonical inventory.
resource: /docs/audit/issues/256-feature-discovery-and-regeneration-update.md
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

# 256 Feature Discovery And Regeneration Update

Fixed a real feature-discovery drift bug and regenerated the derived audience docs against the corrected canonical inventory.

What changed:
- `kestrel_sovereign/features/__init__.py`
  - `discover_feature_modules()` now filters candidate modules to only those that actually export a discoverable `Feature` subclass.
  - This removed support packages that were being counted as features even though they are not loadable feature modules.
- `KESTREL_FEATURES.md`
  - corrected the audited feature snapshot from `41/36` to `36/36`
  - clarified that support packages under `kestrel_sovereign/features/` are not discoverable features unless they export a `Feature` subclass
- `docs/audit/FEATURE_MODULE_MATRIX.md`
  - reconciled the module list to the corrected loader behavior
- `tests/unit/test_feature_inventory_contracts.py`
  - added proof that every discoverable module exports a feature class
  - added proof that `discover_features()` matches the unique class inventory
  - added proof that disabling features by class name filters exact classes
- `tests/unit/test_canonical_inventory_sync.py`
  - now derives the canonical feature-module inventory from the actual discovery function instead of filesystem heuristics

Verification:
- `uv run pytest tests/unit/test_feature_inventory_contracts.py tests/unit/test_canonical_inventory_sync.py -v`
- result: `10 passed`
- Expanded focused proof bundle:
  - `uv run pytest tests/unit/test_auth_decision_table.py tests/unit/test_endpoint_contract_suite.py tests/unit/test_secondary_endpoint_contracts.py tests/unit/test_identity_constitution_endpoint_contracts.py tests/unit/test_sovereignty_endpoint_contracts.py tests/unit/test_models_configuration_endpoint_contracts.py tests/unit/test_commands_conversations_endpoint_contracts.py tests/unit/test_security_endpoint_contracts.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py tests/unit/test_privacy_preset_consistency.py tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_inventory_contracts.py -v`
  - result: `59 passed`

Audience-doc generation:
- Regenerated successfully with `uv run python -u scripts/generate_feature_docs.py --audience developer`
- Regenerated successfully with `uv run python -u scripts/generate_feature_docs.py --audience user`
- Regenerated successfully with `uv run python -u scripts/generate_feature_docs.py --audience investor`

This removes a category error from the canonical inventory itself: support packages are no longer masquerading as discoverable features.
