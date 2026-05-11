"""Unit tests for SignalLogStore — table creation, redaction, retention.

Most of the store's write path is exercised through the dispatcher tests;
this file focuses on store-only concerns: schema migration, retention
purge, and the TRUSTED + opt-in raw-storage path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SignalResult,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sovereign.signals import SignalLogStore
from kestrel_sovereign.storage.db import SQLiteBackend


@pytest.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "signals.db"))
    await backend.connect()
    s = SignalLogStore(backend)
    await s.initialize()
    yield s
    await backend.close()


def _registration(
    name: str = "test",
    *,
    trust: Trust = Trust.TRUSTED,
    store_raw_trusted: bool = False,
    retention_days: int = 7,
) -> SourceRegistration:
    async def _h(p):
        return None

    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=_h,
        trust=trust,
        log_redaction=RedactionPolicy(
            summarize=lambda p: f"keys={sorted(p.keys())}",
            store_raw_trusted=store_raw_trusted,
        ),
        retention_days=retention_days,
    )


def _signal(source: str = "test", payload=None) -> Signal:
    return Signal(
        source=source,
        kind="tick",
        mode=SignalMode.ACTION,
        payload=payload if payload is not None else {"a": 1, "b": 2},
        target_agent="agent-1",
    )


def _ok_result(sig: Signal) -> SignalResult:
    return SignalResult(
        signal_id=sig.id,
        status=Status.OK,
        mode=sig.mode,
        duration_ms=42,
        action_result={"ok": True},
    )


@pytest.mark.asyncio
async def test_initialize_creates_table_and_indexes(store):
    """Idempotent — running twice must not error."""
    await store.initialize()
    exists = await store.backend.table_exists("signal_log")
    assert exists


@pytest.mark.asyncio
async def test_append_writes_redacted_summary(store):
    sig = _signal()
    await store.append(sig, _registration(), _ok_result(sig))
    rows = await store.backend.fetch_all(
        "SELECT payload_redacted, payload_raw FROM signal_log"
    )
    assert len(rows) == 1
    assert rows[0][0] == "keys=['a', 'b']"
    assert rows[0][1] is None  # store_raw_trusted=False by default


@pytest.mark.asyncio
async def test_trusted_opt_in_stores_raw(store):
    sig = _signal()
    reg = _registration(trust=Trust.TRUSTED, store_raw_trusted=True)
    await store.append(sig, reg, _ok_result(sig))
    rows = await store.backend.fetch_all("SELECT payload_raw FROM signal_log")
    assert rows[0][0] is not None
    assert "1" in rows[0][0]  # values present


@pytest.mark.asyncio
async def test_untrusted_never_stores_raw_even_with_opt_in(store):
    """Defense in depth: store_raw_trusted is ignored for UNTRUSTED sources.
    The dispatcher won't invoke an UNTRUSTED non-ACTION without a sanitizer,
    but the store still refuses raw UNTRUSTED storage."""
    sig = _signal()
    reg = _registration(trust=Trust.UNTRUSTED, store_raw_trusted=True)
    await store.append(sig, reg, _ok_result(sig))
    rows = await store.backend.fetch_all("SELECT payload_raw FROM signal_log")
    assert rows[0][0] is None


@pytest.mark.asyncio
async def test_failed_redaction_does_not_block_logging(store):
    """If the redaction policy raises, the row still lands with a placeholder
    summary — log entry is preserved for debugging the broken policy."""
    async def _h(p):
        return None

    def bad_summarize(p):
        raise RuntimeError("redaction broke")

    reg = SourceRegistration(
        name="bad_redact",
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=_h,
        log_redaction=RedactionPolicy(summarize=bad_summarize),
    )
    sig = _signal()
    await store.append(sig, reg, _ok_result(sig))
    rows = await store.backend.fetch_all(
        "SELECT payload_redacted FROM signal_log"
    )
    assert rows[0][0].startswith("<redaction failed:")


@pytest.mark.asyncio
async def test_constitution_columns_default_to_null(store):
    """kestrel-sovereign#1137 chunk 1C — when a caller doesn't pass the
    new constitutional-injection kwargs (i.e. legacy ACTION/ARTIFACT
    sources, or the dispatcher hasn't wired them yet), the columns are
    NULL. Pinning this so the migration is genuinely additive."""
    sig = _signal()
    reg = _registration()
    await store.append(sig, reg, _ok_result(sig))

    rows = await store.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, echo_canary_status, "
        "injected_clauses_json, dropped_clauses_json, prompt_template_hash "
        "FROM signal_log"
    )
    assert rows[0] == (None, None, None, None, None, None)


@pytest.mark.asyncio
async def test_prompt_template_hash_persists_supplied_value(store):
    sig = _signal()
    reg = _registration()
    await store.append(sig, reg, _ok_result(sig), prompt_template_hash="abc123")
    rows = await store.backend.fetch_all(
        "SELECT prompt_template_hash FROM signal_log"
    )
    assert rows[0][0] == "abc123"


@pytest.mark.asyncio
async def test_constitution_columns_persist_supplied_values(store):
    """The dispatcher (chunk 1G) will pass these for COGNITION dispatches
    that go through constitutional injection. Pin the round-trip."""
    import json as _json

    sig = _signal()
    reg = _registration()
    await store.append(
        sig,
        reg,
        _ok_result(sig),
        constitution_hash="con_abc123",
        doctrine_bundle_hash="bun_def456",
        echo_canary_status="verified",
        injected_clauses=["KESTREL_CONSTITUTION", "TORTOISE_DOCTRINE", "AGENTS.md"],
        dropped_clauses=["docs/research/long-supplement.md"],
    )
    rows = await store.backend.fetch_all(
        "SELECT constitution_hash, doctrine_bundle_hash, echo_canary_status, "
        "injected_clauses_json, dropped_clauses_json FROM signal_log"
    )
    row = rows[0]
    assert row[0] == "con_abc123"
    assert row[1] == "bun_def456"
    assert row[2] == "verified"
    assert _json.loads(row[3]) == [
        "KESTREL_CONSTITUTION",
        "TORTOISE_DOCTRINE",
        "AGENTS.md",
    ]
    assert _json.loads(row[4]) == ["docs/research/long-supplement.md"]


@pytest.mark.asyncio
async def test_empty_clause_list_distinct_from_null(store):
    """Empty list serializes to '[]' (system-prompt path ran but
    contributed nothing trackable); None stays NULL (no system-prompt
    path at all). Pinning this distinction so an auditor can tell the
    difference."""
    import json as _json

    sig1 = _signal("with_empty_lists")
    reg = _registration()
    await store.append(
        sig1,
        reg,
        _ok_result(sig1),
        injected_clauses=[],
        dropped_clauses=[],
    )

    sig2 = _signal("with_none_lists")
    await store.append(sig2, reg, _ok_result(sig2))

    rows = await store.backend.fetch_all(
        "SELECT source, injected_clauses_json, dropped_clauses_json "
        "FROM signal_log ORDER BY source"
    )
    by_source = {r[0]: (r[1], r[2]) for r in rows}
    assert by_source["with_empty_lists"] == ("[]", "[]")
    assert by_source["with_none_lists"] == (None, None)
    # Round-trip JSON parsing
    assert _json.loads(by_source["with_empty_lists"][0]) == []


@pytest.mark.asyncio
async def test_constitution_hash_index_exists(store):
    """The partial index `idx_signal_log_constitution_hash` enables the
    auditor query 'all dispatches under constitution X' without a full
    table scan. Verify by querying SQLite's sqlite_master catalog."""
    rows = await store.backend.fetch_all(
        "SELECT name, sql FROM sqlite_master WHERE type='index' "
        "AND name='idx_signal_log_constitution_hash'"
    )
    assert len(rows) == 1
    assert "constitution_hash IS NOT NULL" in (rows[0][1] or "")


@pytest.mark.asyncio
async def test_initialize_is_idempotent_on_new_columns(store):
    """Running initialize() twice doesn't error (additive ALTERs are
    silently skipped on already-applied)."""
    await store.initialize()
    await store.initialize()
    # If we got here without raising, idempotency holds.


@pytest.mark.asyncio
async def test_purge_expired_deletes_only_old_rows(store):
    sig_old = _signal("old")
    sig_new = _signal("new")
    reg_short = _registration("old", retention_days=1)
    reg_long = _registration("new", retention_days=30)

    await store.append(sig_old, reg_short, _ok_result(sig_old))
    await store.append(sig_new, reg_long, _ok_result(sig_new))

    # Sweep with a clock 7 days in the future — short-retention row expired,
    # long-retention row not yet.
    future = datetime.now(timezone.utc) + timedelta(days=7)
    deleted = await store.purge_expired(now=future)
    assert deleted == 1

    rows = await store.backend.fetch_all("SELECT source FROM signal_log")
    assert [r[0] for r in rows] == ["new"]
