"""A scheduled backup names which kind of target failed (#3189).

For eighty consecutive runs the only record of Emma's four-hourly backup was
``RuntimeError: scheduled tool backup_snapshot failed``: two targets, one of
them (GCS) current the whole time, the other (Lighthouse) timing out for
eighteen days, collapsed into one unlabelled failure at the signal boundary.

The handler already had the per-target map. Two things now survive the
boundary on failure, which raises and discards the JSON payload:

* the ``reason_code`` says whether every target failed, several did, or
  exactly one did and of which kind (``BACKUP_TARGET_FAILED_LIGHTHOUSE``);
  it crosses into ``signal_log.error``;
* the ``error`` string, logged at the local diagnostic boundary, names each
  failed target by kind with the error its target recorded (type and
  message).

A target's ``kind`` is the bounded part of its identity. Its ``name`` is a URL
carrying the bucket and prefix, which may not cross into ``signal_log.error``.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from kestrel_sdk.signals import Signal, SignalMode, Status

import kestrel_sovereign.storage.sync as sync_package
from kestrel_sovereign.features.scheduler.feature import (
    BACKUP_SNAPSHOT_REASON_CODES,
    SchedulerFeature,
    backup_target_failure_code,
)
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.scheduler import (
    _bounded_token,
    build_cron_registrations,
    cron_source_name,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.sync import gcs_target, sovereign_ipfs_target
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
from tests.unit.test_signals_scheduler_source import _FakeAgent


def _result(name: str, success: bool, kind: str = "", error: str | None = None) -> SyncResult:
    """A real result, as ``SyncService.force_snapshot`` hands them back."""
    return SyncResult(
        success=success,
        target_name=name,
        bytes_synced=0,
        frames_synced=0,
        timestamp=datetime.now(UTC),
        error=error,
        kind=kind,
    )


def _results(*specs: tuple) -> dict[str, SyncResult]:
    return {spec[0]: _result(*spec) for spec in specs}


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


def _all_subclasses(cls):
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def test_the_kinds_census_equals_what_every_shipped_target_declares():
    """``SYNC_TARGET_KINDS`` is a claim about the target classes in the sync
    package. Every module there is imported and every ``SyncTarget`` subclass
    it defines must declare a kind that the census lists, so a new shipped
    target cannot land without the reason code that names it."""
    for module in pkgutil.iter_modules(sync_package.__path__):
        importlib.import_module(f"{sync_package.__name__}.{module.name}")
    shipped = {
        cls for cls in _all_subclasses(SyncTarget)
        if cls.__module__.startswith(f"{sync_package.__name__}.")
    }
    assert len(shipped) >= 4
    assert {GCSTarget, S3Target, LighthouseTarget, SovereignIPFSTarget} <= shipped
    assert all(cls.kind for cls in shipped), [c for c in shipped if not c.kind]
    assert {cls.kind for cls in shipped} == SYNC_TARGET_KINDS
    assert SyncTarget.kind == ""


def test_every_backup_code_is_declared_and_passes_the_token_fence():
    declared = SchedulerFeature.tool_reason_codes["backup_snapshot"]
    assert declared == BACKUP_SNAPSHOT_REASON_CODES
    for kind in SYNC_TARGET_KINDS:
        assert backup_target_failure_code(kind) in declared
    assert {"BACKUP_ALL_TARGETS_FAILED", "BACKUP_TARGETS_FAILED", "BACKUP_TARGET_FAILED"} <= declared
    for code in declared:
        assert _bounded_token(code) == code, code


def test_an_undeclared_or_unbounded_kind_yields_the_kind_less_code():
    assert backup_target_failure_code("") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("  ") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code(None) == "BACKUP_TARGET_FAILED"  # type: ignore[arg-type]
    assert backup_target_failure_code(42) == "BACKUP_TARGET_FAILED"  # type: ignore[arg-type]
    assert backup_target_failure_code("lighthouse") == "BACKUP_TARGET_FAILED_LIGHTHOUSE"


# ---------------------------------------------------------------------------
# The dispatch: what signal_log.error says
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failed_target_is_named_by_kind_in_the_dispatch_error(
    dispatcher_components,
):
    """Emma's shape: GCS current, Lighthouse timing out."""
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", True, "gcs"),
        ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith(
        "scheduled tool backup_snapshot failed (BACKUP_TARGET_FAILED_LIGHTHOUSE)"
    )
    # The URL-shaped target name and the recorded error stay out of the bounded error.
    assert "bucket" not in result.error and "gs://" not in result.error
    assert "ReadTimeout" not in result.error


@pytest.mark.asyncio
async def test_every_target_failing_says_so(dispatcher_components):
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", False, "gcs", "Forbidden: 403"),
        ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_ALL_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_several_but_not_all_targets_failing_says_so(dispatcher_components):
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", True, "gcs"),
        ("s3://bucket/prefix", False, "s3", "ClientError: AccessDenied"),
        ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_a_failed_target_of_undeclared_kind_still_reports_as_a_failed_target(
    dispatcher_components,
):
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", True, "gcs"),
        ("custom://somewhere", False, "", "boom"),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED)")


@pytest.mark.asyncio
async def test_the_local_log_names_each_failed_kind_with_its_recorded_error(
    dispatcher_components, caplog,
):
    """The payload is discarded on failure; its ``error`` string is what the
    scheduler source logs at the local diagnostic boundary."""
    with caplog.at_level("ERROR"):
        result = await _dispatch(dispatcher_components, _results(
            ("gs://bucket/prefix/agent", True, "gcs"),
            ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
        ))
    assert result.status == Status.FAILED
    assert "backup_snapshot_failed: lighthouse (ReadTimeout: )" in caplog.text


@pytest.mark.asyncio
async def test_a_fully_successful_pass_is_ok_and_its_artifact_carries_each_kind(
    dispatcher_components,
):
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", True, "gcs"),
        ("lighthouse://agent", True, "lighthouse"),
    ))
    assert result.status == Status.OK
    payload = json.loads(result.action_result)
    assert payload["success"] is True
    assert "reason_code" not in payload and "error" not in payload
    assert payload["targets"]["lighthouse://agent"] == {
        "success": True, "bytes": 0, "kind": "lighthouse",
    }


# ---------------------------------------------------------------------------
# The handler's failure string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_error_string_names_kinds_and_recorded_errors_sorted():
    agent = SimpleNamespace(
        did="did:test:backup",
        _sync_service=SimpleNamespace(snapshot_if_changed=AsyncMock(return_value=_results(
            ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
            ("gs://bucket/prefix/agent", False, "gcs", "Forbidden: 403"),
            ("custom://x", False, "", None),
        ))),
    )
    feature = SchedulerFeature(agent)
    payload = json.loads(await feature._handle_backup_snapshot({}))
    assert payload["success"] is False
    assert payload["reason_code"] == "BACKUP_ALL_TARGETS_FAILED"
    assert payload["error"] == (
        "backup_snapshot_failed: gcs (Forbidden: 403), lighthouse (ReadTimeout: ), undeclared"
    )
    assert "outcome" not in payload
    assert set(payload["targets"]["gs://bucket/prefix/agent"]) == {"success", "bytes", "kind"}


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
            timestamp=datetime.now(UTC),
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
async def test_force_snapshot_stamps_the_targets_kind_on_a_copy_of_each_result(
    tmp_path, caplog,
):
    db = tmp_path / "agent.db"
    db.write_bytes(b"")
    service = SyncService(db_path=str(db), state_file=str(tmp_path / "sync.state"))
    ok = _KindedTarget("lighthouse://ok")
    service.add_target(ok)
    service.add_target(_KindedTarget("lighthouse://failed", fail=True))
    service.add_target(_KindedTarget("lighthouse://raised", raise_=True))
    returned: list[SyncResult] = []
    original = ok.sync_snapshot

    async def spy(db_path):
        result = await original(db_path)
        returned.append(result)
        return result

    ok.sync_snapshot = spy  # type: ignore[method-assign]

    with caplog.at_level("ERROR"):
        results = await service.force_snapshot()

    assert {name: r.kind for name, r in results.items()} == {
        "lighthouse://ok": "lighthouse",
        "lighthouse://failed": "lighthouse",
        "lighthouse://raised": "lighthouse",
    }
    # The target's own object is left as it returned it; the stored one is a copy.
    assert returned[0].kind == "" and results["lighthouse://ok"] is not returned[0]
    # A raised exception is recorded and logged by type: ``str(TimeoutError())`` is "".
    assert results["lighthouse://raised"].error == "TimeoutError: "
    assert results["lighthouse://raised"].success is False
    assert "Snapshot failed for lighthouse://raised: TimeoutError: " in caplog.text


# ---------------------------------------------------------------------------
# Every shipped target names the exception type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lighthouse_target_records_the_exception_type_when_its_message_is_empty(
    tmp_path, monkeypatch, caplog,
):
    """880 log lines read ``Failed to sync to Lighthouse: `` because an httpx
    timeout stringifies to nothing. The type is the diagnosis."""
    import sqlite3

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


def _timing_out(*_args, **_kwargs):
    raise httpx.ReadTimeout("")


async def _timing_out_async(*_args, **_kwargs):
    raise httpx.ReadTimeout("")


def _gcs(tmp, monkeypatch):
    monkeypatch.setattr(gcs_target, "_create_consistent_snapshot", _timing_out)
    return GCSTarget("bucket", agent_id="a", state_dir=tmp)


def _s3(tmp, monkeypatch):
    target = S3Target("bucket")
    monkeypatch.setattr(target, "_get_client", _timing_out_async)  # first call in its try
    return target


def _ipfs(tmp, monkeypatch):
    monkeypatch.setattr(sovereign_ipfs_target, "_create_consistent_snapshot", _timing_out)
    return SovereignIPFSTarget("http://ipfs.invalid", "a", state_dir=tmp)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("build", "label"),
    [(_gcs, "GCS"), (_s3, "S3"), (_ipfs, "sovereign IPFS")],
)
async def test_every_sibling_target_records_the_exception_type_too(
    build, label, tmp_path, monkeypatch, caplog,
):
    """The same empty-message timeout at the GCS, S3 and IPFS doors."""
    target = build(tmp_path, monkeypatch)

    with caplog.at_level("ERROR"):
        result = await target.sync_snapshot(tmp_path / "agent.db")

    assert result.success is False
    assert result.error == "ReadTimeout: "
    assert f"Failed to sync snapshot to {label}: ReadTimeout: " in caplog.text
