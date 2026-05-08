"""ToolResult contract tests for IdentityFeature (#1085).

The integration tests in tests/integration/test_identity_export_import.py
exercise the underlying ``kestrel_sovereign.identity`` helpers, not
the ``IdentityFeature`` ``@tool`` methods. These unit tests pin the
ToolResult shape and the honesty edges introduced by the migration:

  - migration_history: db-down vs empty vs malformed-rows distinction
  - assess_substrate: UNKNOWN substrate surfaces as PARTIAL
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.identity.feature import IdentityFeature


def _make_feature(db=None):
    agent = MagicMock()
    agent.agent_id = "did:test:identity-agent"
    agent.did = agent.agent_id
    agent.storage = MagicMock()
    agent.storage._raw_storage = MagicMock()
    if db is not None:
        agent.storage._raw_storage.db = db
        agent.storage.db = db
    else:
        agent.storage._raw_storage.db = None
        agent.storage.db = None
    feat = IdentityFeature(agent)
    feat.disabled_skills = frozenset()
    return feat


# ---------------------------------------------------------------------------
# migration_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_history_db_unavailable_returns_error():
    """Round 0 honesty: db-unavailable was conflated with 'no
    migrations' pre-fix. Now db-down → ERROR distinct from empty."""
    feat = _make_feature(db=None)
    result = await feat.migration_history()
    assert result.status is ToolResultStatus.ERROR
    assert "database not available" in result.error.lower()


@pytest.mark.asyncio
async def test_migration_history_empty_returns_ok():
    """Empty history (legitimately no migrations) → OK with explanatory text."""
    db = MagicMock()
    db.fetchall = AsyncMock(return_value=[])
    feat = _make_feature(db=db)
    result = await feat.migration_history()
    assert result.status is ToolResultStatus.OK
    assert result.data["total_migrations"] == 0
    assert "no migrations recorded" in result.confirmation.lower()


@pytest.mark.asyncio
async def test_migration_history_malformed_row_partial():
    """Round 1 codex finding: malformed row props (e.g. stats=null,
    non-string timestamp, non-object JSON, raw bytes) must not raise
    out of the @tool — they are skipped and surface in PARTIAL.error."""
    db = MagicMock()
    db.fetchall = AsyncMock(return_value=[
        # Valid row
        ("mig-aaaa-1111", '{"timestamp": "2026-05-08T10:00:00", "source_substrate": "claude", "target_substrate": "gpt", "stats": {"episodes_imported": 5}}'),
        # null stats — would raise dict(None) pre-fix
        ("mig-bbbb-2222", '{"timestamp": "2026-05-08T11:00:00", "stats": null}'),
        # numeric timestamp — would raise on slicing pre-fix
        ("mig-cccc-3333", '{"timestamp": 1234567890, "stats": {}}'),
        # JSON parses but isn't an object
        ("mig-dddd-4444", '"this is a string, not an object"'),
        # Non-JSON garbage
        ("mig-eeee-5555", 'not-json-at-all'),
    ])
    feat = _make_feature(db=db)

    result = await feat.migration_history()

    assert result.status is ToolResultStatus.PARTIAL
    # Valid + null-stats + numeric-ts all yield records (defensive coercion);
    # non-object JSON and non-JSON garbage produce parse_errors
    assert result.data["total_migrations"] >= 3
    assert len(result.data["parse_errors"]) >= 2
    # The two unreadable rows are named in the error
    assert "mig-dddd" in result.error or "mig-eeee" in result.error
    # The valid record is fully populated
    valid = next(r for r in result.data["records"] if r["id"].startswith("mig-aaaa"))
    assert valid["from"] == "claude"
    assert valid["to"] == "gpt"
    assert valid["stats"]["episodes_imported"] == 5
    # The null-stats row coerced to empty dict, didn't crash
    null_stats = next(r for r in result.data["records"] if r["id"].startswith("mig-bbbb"))
    assert null_stats["stats"] == {}
    # The numeric-timestamp row stringified, didn't crash
    numeric_ts = next(r for r in result.data["records"] if r["id"].startswith("mig-cccc"))
    assert numeric_ts["timestamp"] == "1234567890"


# ---------------------------------------------------------------------------
# assess_substrate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assess_substrate_unknown_is_partial(monkeypatch):
    """Substrate UNKNOWN surfaces as PARTIAL with a 'best-effort' caveat."""
    from kestrel_sovereign import config as config_mod

    def fake_load_config():
        return {"llm": {"default_provider": "some_random_provider", "default_model": "??"}}

    monkeypatch.setattr(config_mod, "load_config", fake_load_config)

    feat = _make_feature()
    result = await feat.assess_substrate()
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["substrate_type"] == "unknown"
    assert "best-effort" in result.error.lower() or "untrusted" in result.error.lower()


@pytest.mark.asyncio
async def test_assess_substrate_anthropic_is_ok(monkeypatch):
    """Known substrate → OK, no PARTIAL caveat."""
    from kestrel_sovereign import config as config_mod

    def fake_load_config():
        return {"llm": {"default_provider": "anthropic", "default_model": "claude-opus-4"}}

    monkeypatch.setattr(config_mod, "load_config", fake_load_config)

    feat = _make_feature()
    result = await feat.assess_substrate()
    assert result.status is ToolResultStatus.OK
    # SubstrateType uses colon-separated values
    assert "anthropic" in result.data["substrate_type"].lower()
    assert result.data["substrate_type"] != "unknown"
