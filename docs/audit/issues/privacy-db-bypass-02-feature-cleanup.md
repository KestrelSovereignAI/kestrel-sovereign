---
type: Issue Body
title: Privacy Db Bypass 02 Feature Cleanup
description: 'The `.db` bypass warning in Nellie''s log exposed a broader architectural
  smell: multiple sovereign features and services reach through `PrivacyEnforcingStorage.db`
  directly even...'
resource: /docs/audit/issues/privacy-db-bypass-02-feature-cleanup.md
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

# Privacy Db Bypass 02 Feature Cleanup

## Problem

The `.db` bypass warning in Nellie's log exposed a broader architectural smell: multiple sovereign features and services reach through `PrivacyEnforcingStorage.db` directly even though the wrapper marks that as deprecated and privacy-bypassing.

Examples include:

- `features/heartbeat/feature.py`
- `features/bridge/feature.py`
- `features/scheduler/feature.py`
- `features/delivery/feature.py`
- `features/webhooks/feature.py`
- `features/wellness/feature.py`
- `features/reflection/feature.py`
- `features/identity/feature.py`
- `features/consent/feature.py`
- `features/keys/feature.py`
- `services/key_resolution.py`

## Goal

Reduce or eliminate direct `storage.db` access in sovereign feature code so privacy-wrapper boundaries become real instead of advisory.

## Scope

- inventory current direct `.db` call sites that go through wrapped storage
- group them into:
  - safe raw-storage usage
  - missing privacy-aware helper methods
  - feature-local anti-patterns
- add the missing privacy-aware accessors or resolve through raw storage intentionally where appropriate
- migrate the highest-value live paths first

## Acceptance criteria

- the main live-path warning sources are removed
- direct wrapper `.db` access is reduced and intentionally documented where it remains
- future features have a clearer pattern for DB access that does not rely on deprecated bypasses

## References

- `kestrel_sovereign/storage/privacy_wrapper.py`
- `kestrel_sovereign/features/heartbeat/feature.py`
- `kestrel_sovereign/features/bridge/feature.py`
- `kestrel_sovereign/features/scheduler/feature.py`
- `kestrel_sovereign/features/delivery/feature.py`
- `kestrel_sovereign/features/webhooks/feature.py`
- `kestrel_sovereign/features/wellness/feature.py`
