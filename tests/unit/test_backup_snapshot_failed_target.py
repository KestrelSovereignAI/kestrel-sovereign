"""A scheduled backup names which kind of target failed (#3189).

For eighty consecutive runs the only record of Emma's four-hourly backup was
``RuntimeError: scheduled tool backup_snapshot failed``: two targets, one of
them (GCS) current the whole time, the other (Lighthouse) timing out for
eighteen days, collapsed into one unlabelled failure at the signal boundary.

The handler already had the per-target map. What crosses now:

* the ``reason_code`` says whether every target failed, several did, or
  exactly one did and of which kind (``BACKUP_TARGET_FAILED_LIGHTHOUSE``);
* the local artifact carries an ``outcome`` (``ok``/``partial``/``failed``),
  and each target's ``kind`` and ``error``.

A target's ``kind`` is the bounded part of its identity. Its ``name`` is a URL
carrying the bucket and prefix, which may not cross into ``signal_log.error``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kestrel_sdk.signals import Signal, SignalMode, Status

from kestrel_sovereign.features.scheduler.feature import (
    BACKUP_SNAPSHOT_REASON_CODES,
    SchedulerFeature,
    backup_target_failure_code,
)
from kestrel_sovereign.signals.sources.scheduler import (
    _bounded_token,
    build_cron_registrations,
    cron_source_name,
)
from kestrel_sovereign.storage.sync.gcs_target import GCSTarget
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.s3_target import S3Target
from kestrel_sovereign.storage.sync.service import SyncService
from kestrel_sovereign.storage.sync.sovereign_ipfs_target import SovereignIPFSTarget
from kestrel_sovereign.storage.sync.targets import (
    SYNC_TARGET_KINDS,
    SyncResult,
    SyncTarget,
)

from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.storage.db import SQLiteBackend

from tests.unit.test_signals_scheduler_source import _FakeAgent

SHIPPED_TARGETS = (GCSTarget, S3Target, LighthouseTarget, SovereignIPFSTarget)


@pytest.fixture
async def dispatcher_components(tmp_path):
    """The real dispatcher over a fresh signal log, as the sibling module builds it."""
    backend = SQLiteBackend(str(tmp_path / "backup_dispatch.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(
        agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store,
    )
    yield (agent, registry, dispatcher, backend)
    await backend.close()


def _result(success: bool, kind: str = "", error: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(success=success, bytes_synced=0, kind=kind, error=error)


async def _dispatch(components, results):
    agent, registry, dispatcher, _ = components
    agent._sync_service = SimpleNamespace(
        snapshot_if_changed=AsyncMock(return_value=results)
    )
    feature = SchedulerFeature(agent)
    feature._agent_id = agent.did

    async def unused_lookup(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        reason_codes_lookup=feature._declared_reason_codes,
        tool_lookup=unused_lookup,
        builtin_handlers={"backup_snapshot": feature._handle_backup_snapshot},
    ):
        registry.register(registration)

    return await dispatcher.dispatch_signal(Signal(
        source=cron_source_name("backup_snapshot"),
        kind="run",
        mode=SignalMode.ACTION,
        payload={},
        target_agent=agent.did,
    ))


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_the_kinds_census_equals_what_the_shipped_targets_declare():
    """``SYNC_TARGET_KINDS`` is a claim about the target classes; keep it true."""
    assert {cls.kind for cls in SHIPPED_TARGETS} == SYNC_TARGET_KINDS
    assert all(cls.kind for cls in SHIPPED_TARGETS)
    assert SyncTarget.kind == ""


def test_every_backup_code_is_declared_and_passes_the_token_fence():
    declared = SchedulerFeature.tool_reason_codes["backup_snapshot"]
    assert declared == BACKUP_SNAPSHOT_REASON_CODES
    for kind in SYNC_TARGET_KINDS:
        assert backup_target_failure_code(kind) in declared
    assert {"BACKUP_ALL_TARGETS_FAILED", "BACKUP_TARGETS_FAILED", "BACKUP_TARGET_FAILED"} <= declared
    for code in declared:
        assert _bounded_token(code) == code, code


def test_an_undeclared_kind_yields_the_kind_less_code():
    assert backup_target_failure_code("") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("  ") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("lighthouse") == "BACKUP_TARGET_FAILED_LIGHTHOUSE"


# ---------------------------------------------------------------------------
# The dispatch: what signal_log.error says
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failed_target_is_named_by_kind_in_the_dispatch_error(
    dispatcher_components,
):
    """Emma's shape: GCS current, Lighthouse timing out."""
    result = await _dispatch(dispatcher_components, {
        "gs://bucket/prefix/agent": _result(True, "gcs"),
        "lighthouse://agent": _result(False, "lighthouse", "ReadTimeout: "),
    })
    assert result.status == Status.FAILED
    assert result.error.endswith(
        "scheduled tool backup_snapshot failed (BACKUP_TARGET_FAILED_LIGHTHOUSE)"
    )
    # The URL-shaped target name stays out of the bounded error.
    assert "bucket" not in result.error and "gs://" not in result.error


@pytest.mark.asyncio
async def test_every_target_failing_says_so(dispatcher_components):
    result = await _dispatch(dispatcher_components, {
        "gs://bucket/prefix/agent": _result(False, "gcs", "Forbidden"),
        "lighthouse://agent": _result(False, "lighthouse", "ReadTimeout: "),
    })
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_ALL_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_several_but_not_all_targets_failing_says_so(dispatcher_components):
    result = await _dispatch(dispatcher_components, {
        "gs://bucket/prefix/agent": _result(True, "gcs"),
        "s3://bucket/prefix": _result(False, "s3", "AccessDenied"),
        "lighthouse://agent": _result(False, "lighthouse", "ReadTimeout: "),
    })
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_a_failed_target_of_undeclared_kind_still_reports_as_a_failed_target(
    dispatcher_components,
):
    result = await _dispatch(dispatcher_components, {
        "gs://bucket/prefix/agent": _result(True, "gcs"),
        "custom://somewhere": _result(False, "", "boom"),
    })
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED)")


@pytest.mark.asyncio
async def test_a_result_without_a_kind_attribute_is_treated_as_undeclared(
    dispatcher_components,
):
    """Older doubles (and results a third-party service builds) carry no kind."""
    result = await _dispatch(dispatcher_components, {
        "a": SimpleNamespace(success=True, bytes_synced=1),
        "b": SimpleNamespace(success=False, bytes_synced=0),
    })
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED)")


@pytest.mark.asyncio
async def test_a_fully_successful_pass_is_ok_with_the_outcome_in_the_artifact(
    dispatcher_components,
):
    result = await _dispatch(dispatcher_components, {
        "gs://bucket/prefix/agent": _result(True, "gcs"),
        "lighthouse://agent": _result(True, "lighthouse"),
    })
    assert result.status == Status.OK
    payload = json.loads(result.action_result)
    assert payload["success"] is True
    assert payload["outcome"] == "ok"
    assert "reason_code" not in payload and "error" not in payload
    assert payload["targets"]["lighthouse://agent"]["kind"] == "lighthouse"


# ---------------------------------------------------------------------------
# The local artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_artifact_distinguishes_partial_from_failed_and_names_kinds():
    agent = SimpleNamespace(
        did="did:test:backup",
        _sync_service=SimpleNamespace(snapshot_if_changed=AsyncMock(return_value={
            "gs://bucket/prefix/agent": _result(True, "gcs"),
            "lighthouse://agent": _result(False, "lighthouse", "ReadTimeout: "),
        })),
    )
    feature = SchedulerFeature(agent)
    payload = json.loads(await feature._handle_backup_snapshot({}))
    assert payload["success"] is False
    assert payload["outcome"] == "partial"
    assert payload["error"] == "backup_snapshot_failed: lighthouse"
    assert payload["reason_code"] == "BACKUP_TARGET_FAILED_LIGHTHOUSE"
    assert payload["targets"]["lighthouse://agent"] == {
        "success": False, "bytes": 0, "kind": "lighthouse", "error": "ReadTimeout: ",
    }

    agent._sync_service = SimpleNamespace(snapshot_if_changed=AsyncMock(return_value={
        "gs://bucket/prefix/agent": _result(False, "gcs", "Forbidden"),
        "lighthouse://agent": _result(False, "lighthouse", "ReadTimeout: "),
    }))
    payload = json.loads(await feature._handle_backup_snapshot({}))
    assert payload["outcome"] == "failed"
    assert payload["error"] == "backup_snapshot_failed: gcs, lighthouse"
    assert payload["reason_code"] == "BACKUP_ALL_TARGETS_FAILED"


# ---------------------------------------------------------------------------
# The sync service stamps the kind on what it hands back
# ---------------------------------------------------------------------------


class _KindedTarget(SyncTarget):
    kind = "lighthouse"

    def __init__(self, name: str, *, fail: bool = False, raise_: bool = False):
        self._name, self._fail, self._raise = name, fail, raise_

    @property
    def name(self) -> str:
        return self._name

    async def sync_snapshot(self, db_path):
        if self._raise:
            raise TimeoutError()
        return SyncResult(
            success=not self._fail,
            target_name=self._name,
            bytes_synced=0,
            frames_synced=0,
            timestamp=datetime.now(timezone.utc),
            error="" if self._fail else None,
        )

    async def sync_wal(self, wal_path, position):  # pragma: no cover - unused
        raise NotImplementedError

    async def get_latest_position(self):
        return None

    async def restore_latest(self, dest_path):  # pragma: no cover - unused
        raise NotImplementedError

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_force_snapshot_stamps_the_targets_kind_on_returned_and_raised_results(
    tmp_path,
):
    db = tmp_path / "agent.db"
    db.write_bytes(b"")
    service = SyncService(db_path=str(db), state_file=str(tmp_path / "sync.state"))
    service.add_target(_KindedTarget("lighthouse://ok"))
    service.add_target(_KindedTarget("lighthouse://failed", fail=True))
    service.add_target(_KindedTarget("lighthouse://raised", raise_=True))

    results = await service.force_snapshot()

    assert {name: r.kind for name, r in results.items()} == {
        "lighthouse://ok": "lighthouse",
        "lighthouse://failed": "lighthouse",
        "lighthouse://raised": "lighthouse",
    }
    # A raised exception is recorded by type: ``str(TimeoutError())`` is "".
    assert results["lighthouse://raised"].error == "TimeoutError: "
    assert results["lighthouse://raised"].success is False


# ---------------------------------------------------------------------------
# The Lighthouse target names the exception type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lighthouse_target_records_the_exception_type_when_its_message_is_empty(
    tmp_path, monkeypatch, caplog,
):
    """880 log lines read ``Failed to sync to Lighthouse: `` because an httpx
    timeout stringifies to nothing. The type is the diagnosis."""
    import sqlite3

    import httpx

    from kestrel_sovereign.storage.providers import lighthouse_rest

    db = tmp_path / "agent.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()

    class _TimingOutClient:
        def __init__(self, *args, **kwargs):
            pass

        async def upload_car(self, **kwargs):
            raise httpx.ReadTimeout("")

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _TimingOutClient)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)

    with caplog.at_level("ERROR"):
        result = await target.sync_snapshot(db)

    assert result.success is False
    assert result.error == "ReadTimeout: "
    assert "Failed to sync to Lighthouse: ReadTimeout: " in caplog.text
