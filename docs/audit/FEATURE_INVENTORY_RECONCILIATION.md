# Feature Inventory Reconciliation

The legacy catalog used fixed plugin counts that no longer matched the implementation.

Current reality:

- feature discovery is code-defined
- the discoverable module surface is larger than the old catalog
- several user-facing modules were absent from the old public description, including `audit_anchor`, `channels`, `consent`, `heartbeat`, `memory_agency`, `scheduler`, `webhooks`, and `wellness`

The canonical fix is to let `KESTREL_FEATURES.md` describe the live inventory rather than preserving stale counts.
