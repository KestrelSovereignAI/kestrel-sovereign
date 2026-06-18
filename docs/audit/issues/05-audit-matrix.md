---
type: Issue Body
title: 05 Audit Matrix
description: Without a canonical matrix, feature claims remain narrative rather than
  auditable. The test suite may be large and still leave blind spots because it is
  not tied directly to the...
resource: /docs/audit/issues/05-audit-matrix.md
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

# 05 Audit Matrix

## Problem

Without a canonical matrix, feature claims remain narrative rather than auditable. The test suite may be large and still leave blind spots because it is not tied directly to the product catalog.

## Goal

Build the audit matrix that maps every claim in `KESTREL_FEATURES.md` to code ownership, invariants, proof requirements, and current coverage status.

## Scope

- enumerate every feature claim by section
- identify source-of-truth code and public surfaces
- map existing tests to each claim
- mark proof gaps and risk level
- identify cross-feature seams requiring dedicated red-team cases

## Deliverable

A machine-readable and human-readable matrix checked into the repo and used as the canonical reference for all audit tickets.

## Exit Criteria

- all 14 feature catalog sections are represented
- all 28 plugins and all documented API groups are represented
- all uncovered or weakly covered claims are linked to follow-up issues
