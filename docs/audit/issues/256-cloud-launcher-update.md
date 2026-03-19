Cloud launcher invariants are tighter now.

Completed in this slice:
- removed `TARGET_MODEL=None` env injection from the RunPod and Vast.ai feature wrappers
- sanitized cloud launcher env payloads so `None` values are dropped instead of serialized into container env
- preserved absent model identity as `null`/`None` for GCP and Vast.ai session serialization instead of fabricating empty strings
- hardened the GCP startup-script path so nullable env overrides do not crash script generation

Proof:
- added `tests/unit/test_cloud_launcher_contracts.py`
- focused verification passed: `32 passed, 6 skipped`

This closes another whole-of-vision seam: launch-time config now preserves the difference between “no resolved model” and “empty model string,” and the cloud backends no longer smuggle unset values into runtime state.
