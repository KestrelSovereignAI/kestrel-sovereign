# Feature Module Matrix

Discovery rule: [`kestrel_sovereign/features/__init__.py`](../../kestrel_sovereign/features/__init__.py) scans single-file modules, package `__init__.py`, and package `feature.py`.

## Discoverable modules

- `audit_anchor`
- `bootstrap`
- `bridge`
- `channels`
- `code_edit`
- `compute`
- `consent`
- `constitution`
- `context`
- `council`
- `delivery`
- `deploy`
- `gcp_compute`
- `github`
- `heartbeat`
- `identity`
- `keys`
- `mcp`
- `memory`
- `memory_agency`
- `model`
- `peers`
- `reflection`
- `runpod`
- `save`
- `scheduler`
- `security`
- `sovereignty`
- `state_of_mind`
- `tasks`
- `vastai`
- `visual_identity`
- `wallet`
- `web_search`
- `webhooks`
- `wellness`

## Reconciliation note

The old catalog described `28` feature plugins. The corrected discovery surface is `36` modules with `36` exported `Feature` subclasses; support packages that do not export a `Feature` subclass should not be counted as discoverable features.

Direct proof status for these modules now lives in [`FEATURE_PROOF_MATRIX.md`](./FEATURE_PROOF_MATRIX.md).
