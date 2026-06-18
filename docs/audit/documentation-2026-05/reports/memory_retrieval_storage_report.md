---
type: Audit Report
title: Memory Retrieval Storage Report
description: 'Lane report from the May 2026 documentation audit: Memory Retrieval
  Storage Report.'
resource: /docs/audit/documentation-2026-05/reports/memory_retrieval_storage_report.md
tags:
- audit
- documentation
- may-2026
- report
timestamp: 2026-05-30 00:00:00+00:00
status: snapshot
owner: documentation-audit
canonical: false
generated: false
privacy: public
---


# Memory Retrieval Storage Lane Report

Source: subagent lane review, read-only, 2026-05-30.

## Scope Reviewed

Memory/retrieval/storage lane only. Reviewed the lane docs, primary memory/storage/sovereignty docs, feature inventory, generated docs, and current code under `kestrel_sovereign/storage/`, `kestrel_sovereign/features/memory/`, `kestrel_sovereign/features/save/`, and related endpoints.

## Canonical Doc Recommendation

Use `docs/architecture/MEMORY_SYSTEM.md` as the canonical cognitive memory doc, but update it first. It is closest to current architecture, but its scoring weights and some ownership/status language have drifted.

Use `docs/architecture/storage/STORAGE_ARCHITECTURE.md` as the backend/storage-tier canonical doc only after a refresh for `AsyncStorage`, `AsyncDatabase`, SQLAlchemy vector search, pgvector, and current env vars.

Use `docs/architecture/storage/SOVEREIGNTY_V2_TECHNICAL.md` as historical/technical background unless updated to V3 CAR behavior.

## Stale Or Conflicting Claims

- `docs/architecture/MEMORY_SYSTEM.md` claims 5-signal memory scoring with semantic `0.30`, emotional `0.25`, importance `0.20`, recency `0.15`, access `0.10`; code now uses 6 signals including certainty: `0.25/0.20/0.20/0.15/0.10/0.10`.
- `docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md` internally conflicts. It says `search_memory` decrypts client-side, then later says encrypted search does DB-level search and needs `full_history_search`; `full_history_search` is no longer the current path.
- `docs/SOVEREIGNTY.md` says all graph/file/database memory writes into one SQLite file and "no export API required." Current code supports SQLite and PostgreSQL, has REST export/import endpoints, and saved-items vector search uses SQLAlchemy/pgvector on Postgres.
- `docs/architecture/storage/STORAGE_ARCHITECTURE.md` uses older examples (`DATABASE_URL`, `storage/database.py`, `postgres_adapter.py`, Alembic planned) that do not match current `AsyncStorage`/`AsyncDatabase` paths or `KESTREL_DATABASE_URL`.
- `docs/architecture/storage/SOVEREIGNTY_IMPLEMENTATION.md` describes removed/old classes like `SovereigntyReceipt`, `AgentSnapshot`, and "entire state" export including graph/files. Current adapter is V3 CAR with `RootManifest`, conversation shards, optional asset collectors/restorers, and import audit logging.
- `docs/architecture/storage/SOVEREIGNTY_V2_TECHNICAL.md` describes uploading separate shard/keyring/manifest CIDs and manifest version 2; current code packs encrypted shards/assets/keyring/root manifest into one CAR and uses `MANIFEST_VERSION = "3.0"`.
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md` user flow still says chat commands return a root CID/key and manual rebuilds SQLite. Needs alignment with API/UI export envelopes, V3 CAR, partial/error statuses, and asset restore caveats.
- `docs/architecture/README.md` marks all memory/storage sovereignty docs Active, even where several are historical or stale.

## Code/Package Evidence

- Memory facade and router: `kestrel_sovereign/storage/memory_system.py` initializes `MemoryRetriever`, `MemoryConsolidator`, and `SchemaRouter`.
- Actual weighted retrieval: `kestrel_sovereign/storage/memory_retriever.py` uses six weights and skips user messages during emotional recall.
- Encryption-aware conversation search: `kestrel_sovereign/storage/async_conversation_store.py` decrypts rows client-side in `search_history`, handles per-agent key migration, and preserves canonical/rendered content split.
- Privacy mode source of truth: `kestrel_sovereign/privacy.py` defines `ephemeral`, `isolated`, `anonymous`, `normal`, `public` presets.
- Privacy enforcement: `kestrel_sovereign/storage/privacy_wrapper.py` blocks EPHEMERAL writes, stores ISOLATED in session-local buffers, anonymizes ANONYMOUS, and blocks backups where policy disallows them.
- Saved-item semantic search: `kestrel_sovereign/storage/saved_items_store.py` computes embeddings, dual-writes `embedding`/`embedding_vec`, uses vector backend when available, falls back to legacy in-Python or text search.
- Vector backends: `kestrel_sovereign/storage/vector/factory.py`, `kestrel_sovereign/storage/vector/pg.py`, `kestrel_sovereign/storage/vector/python.py`.
- pgvector migration/backfill: `kestrel_sovereign/storage/sqla/migrations.py` adds/backfills `saved_items.embedding_vec`, uses `CREATE EXTENSION vector`, and HNSW index on Postgres.
- Backend support: `kestrel_sovereign/storage/async_storage.py`, `kestrel_sovereign/storage/async_database.py`, `kestrel_sovereign/server.py`.
- Direct deps for current vector path: `pyproject.toml` includes `sqlalchemy[asyncio]`, `pgvector`, and `numpy`.
- Sovereignty V3 CAR/export/import/assets: `kestrel_sovereign/storage/sovereign_adapter.py`.
- REST saved items and sovereignty endpoints: `kestrel_sovereign/endpoints/saved_items.py`, `kestrel_sovereign/endpoints/sovereignty.py`.

## Docs To Update

- `docs/architecture/MEMORY_SYSTEM.md`
- `docs/architecture/MEMORY_OWNERSHIP.md`
- `docs/architecture/storage/STORAGE_ARCHITECTURE.md`
- `docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md`
- `docs/architecture/storage/DECENTRALIZED_STORAGE.md`
- `docs/architecture/storage/SOVEREIGNTY_V2_TECHNICAL.md`
- `docs/SOVEREIGNTY.md`
- `docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md`
- `KESTREL_FEATURES.md`, especially saved-items semantic search, privacy/storage wording, and sovereignty export/import wording.

## Docs To Archive Or Mark Historical

- Mark `docs/architecture/storage/SOVEREIGNTY_IMPLEMENTATION.md` historical unless it is rewritten around V3 CAR.
- Mark parts of `docs/architecture/storage/SOVEREIGNTY_V2_TECHNICAL.md` historical, or rename/update to V3.
- Consider folding `docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md` into `docs/architecture/MEMORY_SYSTEM.md` to avoid duplicated drift.

## Generated Docs To Regenerate

After updating `KESTREL_FEATURES.md`, regenerate:

- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_investor.md`

## Open Questions

- Is PostgreSQL plus pgvector now officially production ready for sovereign memory, or still an opt-in backend with SQLite as the default local path?
- Should saved items remain documented as core `kestrel-sovereign`, or is there a planned `kestrel-feature-save` / storage package extraction?
- Should sovereignty export docs promise full graph/files export, or only conversations plus downstream assets supplied by `AssetCollector`?
- Are `!export-sovereignty` / `!import-sovereignty` still user-facing canonical commands, or should docs lead with API/UI flows?
- Should external-ref asset restoration be called fully supported now that `_fetch_external_ref_bytes` exists, or still caveated because restorer wiring is downstream-dependent?

## Suggested First PR Slice

Start small: update `docs/architecture/MEMORY_SYSTEM.md`, `docs/architecture/storage/HUMAN_MEMORY_SYSTEM.md`, and `docs/architecture/README.md` to fix the memory scoring weights, remove `full_history_search` guidance, document encrypted client-side search, and mark duplicate/stale memory docs appropriately.

