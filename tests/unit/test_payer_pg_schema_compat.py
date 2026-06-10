"""Postgres-compatibility regression tests for the payer key tables.

These guard two PG-only failure classes that SQLite-backed unit tests cannot
catch (both were hit when running #1646/#1647 against a real Postgres):

1. A ';' inside a schema comment splits CORE_SCHEMA mid-statement (``_init_schema``
   does ``schema.split(';')``), producing a comment-only fragment that SQLite
   no-ops but asyncpg rejects (``'NoneType'.decode``).
2. ``INSERT OR REPLACE`` → PG ``ON CONFLICT`` needs the *real* conflict target.
   For tables whose UNIQUE/PK is not the first column, the converter must look
   it up in ``known_pks`` or it emits ``ON CONFLICT (<first col>)`` which has no
   matching constraint.
"""
from __future__ import annotations

import re

from kestrel_sovereign.storage.async_database import CORE_SCHEMA
from kestrel_sovereign.storage.db.placeholder import sqlite_to_postgres


def _comment_only_fragments(schema: str) -> list[str]:
    bad = []
    for stmt in schema.split(";"):
        s = stmt.strip()
        if not s:
            continue
        lines = [ln for ln in s.splitlines() if ln.strip()]
        if lines and all(ln.strip().startswith("--") for ln in lines):
            bad.append(s[:80])
    return bad


def test_core_schema_has_no_comment_only_fragments():
    """No schema comment may contain a ';' — it would split a CREATE statement
    into a comment-only fragment that breaks _init_schema on Postgres."""
    bad = _comment_only_fragments(CORE_SCHEMA)
    assert bad == [], (
        "CORE_SCHEMA splits (on ';') into comment-only fragment(s) — a ';' in a "
        f"schema comment will break _init_schema on Postgres: {bad}"
    )


def _on_conflict_target(query: str) -> str:
    converted, _ = sqlite_to_postgres(query)
    m = re.search(r"ON CONFLICT \(([^)]*)\) DO UPDATE", converted)
    assert m, f"expected ON CONFLICT (...) DO UPDATE in: {converted}"
    return m.group(1)


def test_sponsor_beneficiaries_conflict_targets_agent_did():
    q = (
        "INSERT OR REPLACE INTO sponsor_beneficiaries "
        "(sponsor_did, agent_did, is_active, enrolled_at) "
        "VALUES (?, ?, 1, CURRENT_TIMESTAMP)"
    )
    # PK is agent_did (not the first column) — re-enroll must conflict on it.
    assert _on_conflict_target(q) == "agent_did"


def test_user_master_keys_conflict_targets_master_did_provider():
    q = (
        "INSERT OR REPLACE INTO user_master_service_keys "
        "(id, master_did, provider_id, encrypted_key, key_hash, is_active, created_at) "
        "VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)"
    )
    # `id` is a fresh UUID; the real UNIQUE is (master_did, provider_id).
    assert _on_conflict_target(q) == "master_did, provider_id"


def test_sponsor_master_keys_conflict_targets_master_did_provider():
    q = (
        "INSERT OR REPLACE INTO sponsor_master_service_keys "
        "(id, master_did, provider_id, encrypted_key, key_hash, is_active, created_at) "
        "VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)"
    )
    assert _on_conflict_target(q) == "master_did, provider_id"
