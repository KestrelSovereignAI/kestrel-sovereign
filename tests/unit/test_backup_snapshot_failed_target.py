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


def _result(
    name: str, success: bool, kind: str = "", error: str | None = None, *,
    attempted: bool = True, metadata: dict | None = None,
) -> SyncResult:
    """A real result, as ``SyncService.force_snapshot`` hands them back."""
    return SyncResult(
        success=success,
        target_name=name,
        bytes_synced=0,
        frames_synced=0,
        timestamp=datetime.now(UTC),
        error=error,
        metadata=metadata,
        kind=kind,
        attempted=attempted,
    )


def _skipped(name: str, kind: str) -> SyncResult:
    """A target the policy denied: never called, as ``_record_policy_skip`` records it."""
    return _result(name, True, kind, attempted=False,
                   metadata={"skipped": True, "policy_denied": True, "reason": "tier"})


def _current(name: str, kind: str) -> SyncResult:
    """A target that found its content already uploaded: called, and a success.
    The targets mark this with ``metadata["skipped"]`` too."""
    return _result(name, True, kind, metadata={"skipped": True, "cid": "QmCurrent"})


def _results(*specs) -> dict[str, SyncResult]:
    made = [spec if isinstance(spec, SyncResult) else _result(*spec) for spec in specs]
    return {r.target_name: r for r in made}


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
    for module in pkgutil.walk_packages(sync_package.__path__, f"{sync_package.__name__}."):
        importlib.import_module(module.name)
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


def test_a_kind_outside_the_census_yields_the_declared_kind_less_code():
    """Declaring a kind the census does not know must not be worse than
    declaring none: the per-kind code would fail the membership door and
    drop to the bare failure this ticket removes."""
    assert backup_target_failure_code("") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("  ") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code(None) == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code(42) == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("azure") == "BACKUP_TARGET_FAILED"
    assert backup_target_failure_code("lighthouse") == "BACKUP_TARGET_FAILED_LIGHTHOUSE"
    assert backup_target_failure_code(" Lighthouse ") == "BACKUP_TARGET_FAILED_LIGHTHOUSE"
    for kind in SYNC_TARGET_KINDS:
        assert backup_target_failure_code(kind) in BACKUP_SNAPSHOT_REASON_CODES


@pytest.mark.asyncio
async def test_a_failed_target_of_a_kind_the_census_does_not_know_still_crosses(
    dispatcher_components,
):
    result = await _dispatch(dispatcher_components, _results(
        ("gs://bucket/prefix/agent", True, "gcs"),
        ("azure://container", False, "azure", "HttpResponseError: 503"),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED)")


@pytest.mark.asyncio
async def test_a_policy_denied_target_counts_on_neither_side(dispatcher_components):
    """A destination the policy denied wrote nothing: with every attempted
    target failing there is no snapshot anywhere, so the code says ALL."""
    result = await _dispatch(dispatcher_components, _results(
        _skipped("lighthouse://agent", "lighthouse"),
        ("gs://bucket/prefix/agent", False, "gcs", "Forbidden: 403"),
        ("s3://bucket/prefix", False, "s3", "ClientError: AccessDenied"),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_ALL_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_the_only_attempted_target_failing_means_no_snapshot_anywhere(
    dispatcher_components,
):
    """Skipped plus one failed: the one attempted target failed, so ALL."""
    result = await _dispatch(dispatcher_components, _results(
        _skipped("lighthouse://agent", "lighthouse"),
        ("gs://bucket/prefix/agent", False, "gcs", "Forbidden: 403"),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_ALL_TARGETS_FAILED)")


@pytest.mark.asyncio
async def test_a_policy_denied_target_does_not_make_one_failure_several(dispatcher_components):
    """Skipped, one succeeded, one failed: a snapshot exists on the succeeded
    target, and exactly one attempted target failed, named by kind."""
    result = await _dispatch(dispatcher_components, _results(
        _skipped("lighthouse://agent", "lighthouse"),
        ("s3://bucket/prefix", True, "s3"),
        ("gs://bucket/prefix/agent", False, "gcs", "Forbidden: 403"),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED_GCS)")


@pytest.mark.asyncio
async def test_a_pass_where_nothing_was_attempted_is_not_a_failure(dispatcher_components):
    """Every target policy-denied, or the unchanged-DB marker: nothing failed."""
    result = await _dispatch(dispatcher_components, _results(
        _skipped("lighthouse://agent", "lighthouse"),
    ))
    assert result.status == Status.OK
    payload = json.loads(result.action_result)
    assert payload["success"] is True
    assert payload["targets"]["lighthouse://agent"]["attempted"] is False


@pytest.mark.asyncio
async def test_a_target_whose_content_was_already_current_counts_as_a_success(
    dispatcher_components,
):
    """The ticket's own idle-DB shape: GCS found its content current (the
    targets mark that with metadata["skipped"] too) and Lighthouse timed out.
    GCS was called and holds a snapshot: one failed target, named by kind,
    never 'no snapshot anywhere'."""
    result = await _dispatch(dispatcher_components, _results(
        _current("gs://bucket/prefix/agent", "gcs"),
        ("lighthouse://agent", False, "lighthouse", "ReadTimeout: "),
    ))
    assert result.status == Status.FAILED
    assert result.error.endswith("failed (BACKUP_TARGET_FAILED_LIGHTHOUSE)")


@pytest.mark.asyncio
async def test_the_artifact_marks_a_current_target_as_attempted(dispatcher_components):
    result = await _dispatch(dispatcher_components, _results(
        _current("gs://bucket/prefix/agent", "gcs"),
        _skipped("lighthouse://agent", "lighthouse"),
    ))
    assert result.status == Status.OK
    payload = json.loads(result.action_result)
    assert payload["targets"]["gs://bucket/prefix/agent"]["attempted"] is True
    assert payload["targets"]["lighthouse://agent"]["attempted"] is False


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
        "success": True, "bytes": 0, "kind": "lighthouse", "attempted": True,
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
    assert set(payload["targets"]["gs://bucket/prefix/agent"]) == {"success", "bytes", "kind", "attempted"}


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


@pytest.mark.asyncio
async def test_the_service_marks_only_denied_targets_and_the_unchanged_pass_as_not_attempted(
    tmp_path,
):
    """The two real producers of ``attempted=False``, and a called target
    that stays attempted."""
    import sqlite3

    from kestrel_sovereign.storage.sync.service import (
        RemoteTierPolicyContext,
        RemoteTierPolicyDecision,
    )

    db = tmp_path / "agent.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    service = SyncService(db_path=str(db), state_file=str(tmp_path / "sync.state"))
    called = _KindedTarget("lighthouse://called")
    service.add_target(called)
    await service.start()
    results = await service.snapshot_if_changed()  # first pass: a real call
    assert results["lighthouse://called"].attempted is True

    # An unchanged DB on the next change-aware pass: the placeholder, not a call.
    again = await service.snapshot_if_changed()
    assert set(again) == {"__unchanged__"}
    assert again["__unchanged__"].attempted is False and again["__unchanged__"].success is True

    # A policy-denied remote target is recorded without a call, with its kind.
    service._record_policy_skip("gs://denied", "tier not allowed", kind="gcs")
    denied = service._policy_skips["gs://denied"]
    assert denied.attempted is False and denied.success is True
    assert denied.kind == "gcs"
    assert denied.metadata == {"skipped": True, "policy_denied": True, "reason": "tier not allowed"}
    assert RemoteTierPolicyContext is not None and RemoteTierPolicyDecision is not None
    await service.stop()


@pytest.mark.asyncio
async def test_the_lighthouse_restore_gives_the_download_the_manifests_size(
    tmp_path, monkeypatch,
):
    from kestrel_sovereign.storage.providers import lighthouse_rest

    seen: dict = {}

    class _Recording:
        def __init__(self, *args, **kwargs):
            pass

        async def download(self, cid, timeout=None, *, expected_bytes=None):
            seen["cid"], seen["expected_bytes"] = cid, expected_bytes
            return b"not a real snapshot"

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _Recording)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)
    target._save_local_manifest({
        "agent_id": "agent", "snapshot_cid": "QmSnap", "snapshot_size": 1_209_462_784,
        "content_hash": "x", "uploaded_at": "2026-08-31T12:01:32+00:00",
    })
    monkeypatch.setattr(target, "_resolve_latest_cid", AsyncMock(return_value="QmSnap"))

    await target.restore_snapshot(tmp_path / "restored.db")

    assert seen["cid"] == "QmSnap"
    assert seen["expected_bytes"] == 1_209_462_784


@pytest.mark.asyncio
async def test_the_lighthouse_restore_records_the_exception_type_too(tmp_path, monkeypatch, caplog):
    from kestrel_sovereign.storage.providers import lighthouse_rest

    class _TimingOut:
        def __init__(self, *args, **kwargs):
            pass

        async def download(self, cid, timeout=None, *, expected_bytes=None):
            raise httpx.ReadTimeout("")

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _TimingOut)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)
    monkeypatch.setattr(target, "_resolve_latest_snapshot", AsyncMock(return_value=("QmSnap", None)))

    with caplog.at_level("ERROR"):
        result = await target.restore_snapshot(tmp_path / "restored.db")

    assert result is not None and result.success is False
    assert result.error == "ReadTimeout: "
    assert "Failed to restore from Lighthouse: ReadTimeout: " in caplog.text


@pytest.mark.asyncio
async def test_a_cold_start_restore_gets_the_size_from_the_uploads_api(tmp_path, monkeypatch):
    """No local manifest: the uploads API names the latest snapshot and its
    size, so the cold-start download gets the payload-sized budget too."""
    from kestrel_sovereign.storage.providers import lighthouse_rest

    seen: dict = {}

    class _Api:
        def __init__(self, *args, **kwargs):
            pass

        async def get_uploads(self):
            return {"fileList": [{
                "cid": "QmCold", "fileName": "kestrel_state__agent__20260831_120132.car",
                "fileSizeInBytes": "1209462784", "createdAt": "2026-08-31T12:01:32Z",
            }]}

        async def download(self, cid, timeout=None, *, expected_bytes=None):
            seen["cid"], seen["expected_bytes"] = cid, expected_bytes
            return b"not a real snapshot"

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _Api)
    monkeypatch.delenv("LIGHTHOUSE_STATE_CID", raising=False)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)
    monkeypatch.setattr(target, "_is_structured_snapshot_upload", lambda u: u.get("cid") == "QmCold")

    await target.restore_snapshot(tmp_path / "restored.db")

    assert seen["cid"] == "QmCold"
    assert seen["expected_bytes"] == 1_209_462_784


@pytest.mark.asyncio
async def test_an_explicit_state_cid_carries_no_size(tmp_path, monkeypatch):
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)
    monkeypatch.setenv("LIGHTHOUSE_STATE_CID", "QmEnv")
    assert await target._resolve_latest_snapshot() == ("QmEnv", None)
    assert await target._resolve_latest_cid() == "QmEnv"


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


@pytest.mark.asyncio
async def test_lighthouse_manifest_upload_warning_names_the_exception_type(
    tmp_path, monkeypatch, caplog,
):
    """The manifest upload is one function below the snapshot upload and
    swallows the same empty-message timeout into a warning."""
    import sqlite3

    from kestrel_sovereign.storage.providers import lighthouse_rest

    db = tmp_path / "agent.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()

    class _ManifestTimesOut:
        def __init__(self, *args, **kwargs):
            pass

        async def upload_car(self, **kwargs):
            return {"Hash": "QmSnap", "Size": "10"}

        async def upload(self, *args, **kwargs):
            raise httpx.ReadTimeout("")

        async def close(self):
            pass

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", _ManifestTimesOut)
    target = LighthouseTarget(api_key="k", agent_id="agent", state_dir=tmp_path)

    with caplog.at_level("WARNING"):
        result = await target.sync_snapshot(db)

    assert result.success is True
    assert "Failed to upload manifest (snapshot is safe): ReadTimeout: " in caplog.text


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
