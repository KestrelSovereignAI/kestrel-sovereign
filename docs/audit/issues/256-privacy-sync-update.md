Another proof slice is now in place under the umbrella audit.

Completed:

- added privacy preset consistency proof in `tests/unit/test_privacy_preset_consistency.py`
- aligned `kestrel_sovereign/command_handler.py` privacy descriptions with the canonical preset meanings
- added canonical inventory sync proof in `tests/unit/test_canonical_inventory_sync.py`

The new sync tests now verify that `KESTREL_FEATURES.md`:

- matches the live discoverable feature-module count and exported `Feature` subclass count
- mentions every router file under `endpoints/`
- mentions every live router route and app-level route
- links only to existing paths

These tests immediately caught real drift:

- `DELETE /api/memories/{node_id}` was live but missing from the canonical inventory
- stale path references pointed at non-existent `kestrel_sovereign/privacy_agent.py`, `kestrel_sovereign/storage.py`, and `kestrel_sovereign/async_storage.py`

All of those have been corrected in the canonical source.

Verification:

- `uv run pytest tests/unit/test_privacy_preset_consistency.py tests/unit/test_privacy_agent.py tests/unit/test_privacy_wrapper.py -v`
- result: `50 passed`
- `uv run pytest tests/unit/test_canonical_inventory_sync.py tests/unit/test_feature_doc_canonicality.py tests/unit/test_generate_feature_docs.py -v`
- result: `11 passed`

Derived docs were regenerated again after the canonical source updates.
