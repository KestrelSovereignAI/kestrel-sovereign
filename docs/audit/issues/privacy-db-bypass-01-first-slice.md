## Problem

Nellie's live log is repeatedly emitting:

`Privacy bypass: direct access to .db property`

The immediate source is the heartbeat path resolving `self.agent.storage.db` during feature initialization. That makes the privacy wrapper complain even during normal healthy operation, which turns a real smell into constant background noise.

## Goal

Remove the noisy direct `.db` access from the first live path affecting Nellie, starting with `HeartbeatFeature`.

## Scope

- fix `HeartbeatFeature` to resolve database access without calling the deprecated privacy-wrapper `.db` property
- prefer an existing raw-storage or privacy-aware access path if one already exists
- keep runtime behavior unchanged apart from removing the warning spam
- add or update focused tests around heartbeat initialization if needed

## Acceptance criteria

- Nellie's normal heartbeat loop no longer emits repeated privacy-bypass warnings
- heartbeat history and table initialization still work
- no new privacy bypass is introduced to replace the old one

## References

- `kestrel_sovereign/features/heartbeat/feature.py`
- `kestrel_sovereign/storage/privacy_wrapper.py`
- `tests/unit/test_heartbeat_feature.py`
