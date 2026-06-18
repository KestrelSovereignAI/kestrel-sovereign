---
type: Issue Body
title: Spawn Lifecycle Guard
description: Recent spawn lifecycle wiring made `delegate_task()` and `terminate_child()`
  trust any truthy `manager._lifecycle` value. In practice that means partially wired
  managers or loos...
resource: /docs/audit/issues/spawn-lifecycle-guard.md
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

# Spawn Lifecycle Guard

## Problem

Recent spawn lifecycle wiring made `delegate_task()` and `terminate_child()` trust any truthy `manager._lifecycle` value. In practice that means partially wired managers or loose test doubles can be treated as the real lifecycle manager, and the feature will try to `await` methods on a non-lifecycle object.

That creates a regression in delegated task execution and termination paths even when child execution itself is healthy.

## Reproduction

- construct a spawn feature with a manager that exposes `_lifecycle` but not a real `SpawnedAgentLifecycle`
- delegate a task to a child agent
- the async task tries to `await lifecycle.report_result(...)` on the wrong object and records a failed child result instead of the real task output

## Fix

- add a narrow helper that only accepts actual `SpawnedAgentLifecycle` instances
- use that helper for delegated task result reporting and explicit termination
- update unit tests to exercise real lifecycle wiring and cover the non-lifecycle `_lifecycle` regression

## Validation

- `uv run pytest tests/unit/test_spawn_feature.py tests/unit/test_spawn_lifecycle.py -q`
