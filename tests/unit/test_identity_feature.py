"""ToolResult contract tests for IdentityFeature (#1085).

The integration tests in tests/integration/test_identity_export_import.py
exercise the underlying ``kestrel_sovereign.identity`` helpers, not
the ``IdentityFeature`` ``@tool`` methods. These unit tests pin the
ToolResult shape and the honesty edges introduced by the migration:

  - migration_history: db-down vs empty vs malformed-rows distinction
  - assess_substrate: UNKNOWN substrate surfaces as PARTIAL
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
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


def test_unique_export_filename_does_not_collide():
    """Round 4 codex finding: per-second timestamp granularity used
    to collide on rapid concurrent exports. The new helper appends
    microsecond + uuid hex to make collisions vanishingly unlikely."""
    from kestrel_sovereign.features.identity.feature import _unique_export_filename
    seen = set()
    for _ in range(2000):
        name = _unique_export_filename()
        assert name not in seen, f"filename collision: {name}"
        assert name.startswith("identity_")
        assert name.endswith(".json")
        seen.add(name)


@pytest.mark.asyncio
async def test_import_identity_forwards_allow_unsigned(monkeypatch, tmp_path):
    """#2112 F185: the import tool exposes allow_unsigned and forwards it, so an
    unsigned export (which the tool's own remediation advice tells the user to
    `!identity import`) is actually importable — previously hardcoded False with
    no way to override."""
    from unittest.mock import AsyncMock, MagicMock as MM

    pkg_path = tmp_path / "unsigned_pkg.json"
    pkg_path.write_text('{"fake": "package"}')

    fake_pkg = MM()
    fake_pkg.constitution_text = None
    fake_pkg.content_hash = None
    fake_pkg.verify_constitution = lambda: True
    fake_pkg.verify_content_hash = lambda: True
    fake_pkg.did = "did:test:unsigned"
    fake_pkg.get_summary.return_value = {
        "did": "did:test:unsigned", "agent_name": "Unsigned Agent",
        "created_at": "2026-05-08T00:00:00",
        "export_timestamp": "2026-05-08T00:00:00",
        "source_substrate": "anthropic_claude", "package_version": "1",
        "episodes_count": 0, "saved_items_count": 0, "relationships_count": 0,
        "skills_count": 0, "migrations_count": 0,
    }

    import kestrel_sovereign.identity as identity_mod
    monkeypatch.setattr(
        identity_mod, "AgentIdentityPackage",
        MM(from_json=MM(return_value=fake_pkg)),
    )

    captured = {}
    fake_result = MM(success=True, errors=[], warnings=[], stats={},
                     imported_counts={})

    async def _fake_import_package(package, **kwargs):
        captured.update(kwargs)
        return fake_result

    fake_importer = MM(import_package=AsyncMock(side_effect=_fake_import_package))
    importer_factory = MM(return_value=fake_importer)
    monkeypatch.setattr(
        identity_mod, "IdentityImporter", importer_factory,
    )
    # resolve_feature_database must return a truthy db.
    import kestrel_sovereign.features.identity.feature as feat_mod
    monkeypatch.setattr(feat_mod, "resolve_feature_database", lambda agent: MM())

    feat = _make_feature()
    agent_dir = tmp_path / "runtime-agent"
    feat.agent.storage_path = str(agent_dir / "kestrel_prime.db")
    await feat.import_identity(str(pkg_path), allow_unsigned=True)
    assert captured.get("allow_unsigned") is True, captured
    assert importer_factory.call_args.kwargs["storage_dir"] == agent_dir

    captured.clear()
    await feat.import_identity(str(pkg_path))  # default
    assert captured.get("allow_unsigned") is False, captured


@pytest.mark.asyncio
async def test_verify_identity_unsigned_is_partial(monkeypatch, tmp_path):
    """Round 3 codex finding: an UNSIGNED package isn't a verify
    failure but is unimportable under the default
    allow_unsigned=False path. Returning OK lets the LLM say
    'verified' for a package that can't actually be imported.
    Now PARTIAL with the importability caveat."""
    from unittest.mock import MagicMock as MM

    # Write a minimal package JSON to a temp file so verify can read it
    pkg_path = tmp_path / "unsigned_pkg.json"
    pkg_path.write_text('{"fake": "package"}')

    fake_pkg = MM()
    fake_pkg.signature = None
    fake_pkg.signatures = []
    fake_pkg.constitution_text = None
    fake_pkg.content_hash = None
    fake_pkg.verify_constitution = lambda: True
    fake_pkg.verify_content_hash = lambda: True
    fake_pkg.get_summary.return_value = {
        "did": "did:test:unsigned",
        "agent_name": "Unsigned Agent",
        "created_at": "2026-05-08T00:00:00",
        "export_timestamp": "2026-05-08T00:00:00",
        "source_substrate": "anthropic_claude",
        "package_version": "1",
        "episodes_count": 0,
        "saved_items_count": 0,
        "relationships_count": 0,
        "skills_count": 0,
        "migrations_count": 0,
    }

    import kestrel_sovereign.identity as identity_mod
    monkeypatch.setattr(
        identity_mod, "AgentIdentityPackage",
        MM(from_json=MM(return_value=fake_pkg)),
    )

    feat = _make_feature()
    result = await feat.verify_identity(str(pkg_path))

    assert result.status is ToolResultStatus.PARTIAL
    assert "unsigned" in result.error.lower()
    assert "allow_unsigned" in result.error.lower()
    assert result.data["signature_status"] == "UNSIGNED"


@pytest.mark.asyncio
async def test_export_identity_tier_downgrade_is_partial(monkeypatch, tmp_path):
    """Round 2 codex finding: when storage_tier=ipfs but the
    FilecoinAdapter downgrades to LOCAL_ONLY (IPFS unavailable),
    pre-fix returned OK with content_hash in restore instructions —
    but `!identity import <hash>` doesn't match Qm/bafy prefix and
    can't find it. Now surfaces as PARTIAL with the local-only
    framing so the LLM cannot claim 'stored to ipfs'."""
    from unittest.mock import MagicMock as MM
    from kestrel_sovereign.identity import SubstrateType
    from kestrel_sovereign.filecoin_adapter import StorageTier

    feat = _make_feature(db=MM())
    feat.agent.agent_id = "did:test:export-agent"

    # Mock IdentityExporter and sign_package
    fake_pkg = MM()
    fake_pkg.did = "did:test:export-agent"
    fake_pkg.to_json.return_value = '{"fake": "json"}'
    fake_pkg.get_summary.return_value = {
        "did": "did:test:export-agent",
        "agent_name": "Test Agent",
        "created_at": "2026-05-08T00:00:00",
        "episodes_count": 1,
        "saved_items_count": 0,
        "relationships_count": 0,
        "skills_count": 0,
        "is_signed": False,
    }
    monkeypatch.setattr(
        "kestrel_sovereign.identity.IdentityExporter",
        lambda **kwargs: MM(export=AsyncMock(return_value=fake_pkg)),
    )

    # Mock FilecoinAdapter.store_content to return a downgrade
    fake_result = MM()
    fake_result.tier = StorageTier.LOCAL_ONLY  # downgraded!
    fake_result.ipfs_cid = None
    fake_result.cid = None
    fake_result.content_hash = "abc123def456"
    fake_result.encryption_key_hash = "keyhash123"  # F187: encrypted upload

    fake_adapter = MM()
    fake_adapter.store_content = MM(return_value=fake_result)
    monkeypatch.setattr(
        "kestrel_sovereign.filecoin_adapter.FilecoinAdapter",
        lambda *a, **kw: fake_adapter,
    )

    # F187: non-local export requires an encryption key to be configured
    # (else it fails cleanly rather than uploading plaintext).
    from cryptography.fernet import Fernet
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    export_root = tmp_path / "exports"
    monkeypatch.setenv("KESTREL_DATA_DIR", str(export_root))
    previous_umask = os.umask(0)
    try:
        result = await feat.export_identity(storage_tier="ipfs", sign=False)
    finally:
        os.umask(previous_umask)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["tier_downgraded"] is True
    assert result.data["requested_storage_tier"] == "ipfs"
    assert result.data["actual_storage_tier"] == "local_only"
    # Confirmation must NOT promise IPFS
    assert "stored to ipfs" not in result.confirmation.lower()
    assert "stored to tier=ipfs" not in result.confirmation.lower()
    # Round 3 codex: when IPFS is unavailable, export should write a
    # JSON file the user can actually feed to !identity import — so
    # the data dict has a fallback_file_path AND the file exists.
    assert result.data["fallback_file_path"] is not None
    fallback_path = Path(result.data["fallback_file_path"])
    assert fallback_path.exists()
    assert stat.S_IMODE(export_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(fallback_path.stat().st_mode) == 0o600
    # Confirmation now points at the importable file
    assert "use `!identity import" in result.confirmation.lower()
    # Error half still names the failure mode + restore path
    assert "ipfs" in result.error.lower()


# ---------------------------------------------------------------------------
# export_identity: storage_tier validation (#1946)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_identity_unknown_tier_is_failed():
    """An unknown/typo'd storage_tier must FAIL loudly, not silently
    fall through to LOCAL_ONLY. Pre-fix, ``tier_map.get(..., LOCAL_ONLY)``
    meant 'ipfsss' produced a local-only export while the agent believed
    it went to IPFS."""
    feat = _make_feature(db=MagicMock())
    result = await feat.export_identity(storage_tier="ipfsss", sign=False)

    assert result.status is ToolResultStatus.ERROR
    # Lists the valid tiers and does NOT claim a local export happened.
    assert "local" in result.error and "ipfs" in result.error and "filecoin" in result.error
    assert "ipfsss" in result.error
    # No data implying an export occurred.
    assert not (result.data or {}).get("file_path")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tier", [False, 0, [], {}])
async def test_export_identity_falsy_nonstring_tier_is_failed(bad_tier):
    """A falsy NON-STRING tier (false/0/[]/{}) is a wrong type, not an
    omission — it must be rejected, not coerced to the 'local' default."""
    feat = _make_feature(db=MagicMock())
    result = await feat.export_identity(storage_tier=bad_tier, sign=False)
    assert result.status is ToolResultStatus.ERROR
    assert "local" in result.error and "ipfs" in result.error


def _mock_local_export(monkeypatch, tmp_path):
    """Wire up IdentityExporter + sign so a local export reaches disk."""
    fake_pkg = MagicMock()
    fake_pkg.did = "did:test:export-agent"
    fake_pkg.to_json.return_value = '{"fake": "json"}'
    fake_pkg.get_summary.return_value = {
        "did": "did:test:export-agent",
        "agent_name": "Test Agent",
        "created_at": "2026-05-08T00:00:00",
        "episodes_count": 1,
        "saved_items_count": 0,
        "relationships_count": 0,
        "skills_count": 0,
        "is_signed": False,
    }
    monkeypatch.setattr(
        "kestrel_sovereign.identity.IdentityExporter",
        lambda **kwargs: MagicMock(export=AsyncMock(return_value=fake_pkg)),
    )
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path / "exports"))
    return fake_pkg


@pytest.mark.asyncio
async def test_export_identity_valid_tier_resolves(monkeypatch, tmp_path):
    """A valid tier ('local') passes validation and produces an export."""
    feat = _make_feature(db=MagicMock())
    feat.agent.agent_id = "did:test:export-agent"
    _mock_local_export(monkeypatch, tmp_path)

    previous_umask = os.umask(0)
    try:
        result = await feat.export_identity(storage_tier="local", sign=False)
    finally:
        os.umask(previous_umask)
    assert result.status is ToolResultStatus.OK
    assert result.data["storage_tier"] == "local"
    export_path = Path(result.data["file_path"])
    assert export_path.exists()
    assert stat.S_IMODE(export_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(export_path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_export_identity_omitted_tier_keeps_local_default(monkeypatch, tmp_path):
    """Omitting storage_tier keeps the documented 'local' default."""
    feat = _make_feature(db=MagicMock())
    feat.agent.agent_id = "did:test:export-agent"
    _mock_local_export(monkeypatch, tmp_path)

    result = await feat.export_identity(sign=False)
    assert result.status is ToolResultStatus.OK
    assert result.data["storage_tier"] == "local"


@pytest.mark.asyncio
async def test_export_identity_refuses_existing_link_destination(monkeypatch, tmp_path):
    """Generated local exports never follow or replace a pre-existing link."""

    feat = _make_feature(db=MagicMock())
    feat.agent.agent_id = "did:test:export-agent"
    _mock_local_export(monkeypatch, tmp_path)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    (export_root / "identity_fixed.json").symlink_to(outside)
    monkeypatch.setattr(
        "kestrel_sovereign.features.identity.feature._unique_export_filename",
        lambda: "identity_fixed.json",
    )

    result = await feat.export_identity(storage_tier="local", sign=False)

    assert result.status is ToolResultStatus.ERROR
    assert outside.read_text(encoding="utf-8") == "outside"
    assert (export_root / "identity_fixed.json").is_symlink()
    assert list(export_root.glob(".identity-export-*")) == []


@pytest.mark.asyncio
async def test_concurrent_feature_exports_have_distinct_private_files(monkeypatch, tmp_path):
    feat = _make_feature(db=MagicMock())
    feat.agent.agent_id = "did:test:export-agent"
    _mock_local_export(monkeypatch, tmp_path)

    results = await asyncio.gather(
        *(feat.export_identity(storage_tier="local", sign=False) for _ in range(24))
    )

    assert all(result.status is ToolResultStatus.OK for result in results)
    paths = [Path(result.data["file_path"]) for result in results]
    assert len(set(paths)) == len(paths)
    assert all(path.exists() for path in paths)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)
    assert stat.S_IMODE(paths[0].parent.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_feature_export_signs_with_runtime_agent_key_directory(monkeypatch, tmp_path):
    """The live feature must not look for signing keys under the process CWD."""

    feat = _make_feature(db=MagicMock())
    feat.agent.agent_id = "did:test:export-agent"
    agent_dir = tmp_path / "agent"
    feat.agent.storage_path = str(agent_dir / "kestrel_prime.db")
    fake_package = _mock_local_export(monkeypatch, tmp_path)
    observed = []

    def record_signing_directory(package, storage_dir):
        observed.append(storage_dir)
        return package

    monkeypatch.setattr(
        "kestrel_sovereign.identity.sign_package",
        record_signing_directory,
    )

    result = await feat.export_identity(storage_tier="local", sign=True)

    assert result.status is ToolResultStatus.OK
    assert observed == [agent_dir]
    assert fake_package.to_json.called


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
