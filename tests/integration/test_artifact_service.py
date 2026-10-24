"""Authorized operations over real encrypted SQLite/PostgreSQL repositories."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from kestrel_sdk.artifacts import (
    ArtifactCommit,
    ArtifactLink,
    ArtifactManifest,
    ArtifactService,
)
from kestrel_sdk.operator import OperatorAuthorizationError, OperatorContext
from kestrel_sdk.resources import ResourceNotFound, ResourceOutcomeUnknown

from kestrel_sovereign._async_rwlock import AsyncReaderWriterLock
from kestrel_sovereign.artifacts.service import (
    ArtifactAccessDenied,
    ArtifactAction,
    ArtifactAuthority,
    AuthorizedArtifactService,
)
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from tests.integration.test_artifact_repository import SCOPE, repository as _repository_fixture

repository = _repository_fixture


class PolicyHolder:
    def __init__(self):
        self.gate = DurablePersistenceGate()
        self.set_mode(PrivacyMode.NORMAL)

    def set_mode(self, mode):
        self.privacy_mode = mode
        self.privacy_config = privacy_mode_to_config(mode)

    def _get_durable_persistence_gate(self):
        return self.gate


class Access:
    def __init__(self, store):
        self.now = datetime.now(UTC)
        self.authority = ArtifactAuthority(clock=lambda: self.now)
        self.service = AuthorizedArtifactService(store, self.authority)
        self.holder = PolicyHolder()
        self.authority_gate = AsyncReaderWriterLock()
        self.granted = True
        self.link_ids = {"conversation"}
        self.checked_links = []

    def context(self, **changes):
        context = OperatorContext(
            tenant_id=SCOPE.tenant_id, principal_id=SCOPE.principal_id,
            acting_agent_id=SCOPE.agent_id,
            granted_actions=frozenset(action.value for action in ArtifactAction),
            granted_capabilities=frozenset({"artifacts"}),
            permitted_boundary_ids=frozenset({"local"}), correlation_id="request",
            issued_at=self.now - timedelta(seconds=1), expires_at=self.now + timedelta(minutes=5),
        )
        return replace(context, **changes)

    def check(self, context, action, link):
        if not self.granted:
            raise OperatorAuthorizationError("sensitive grant diagnostic")
        if link is not None:
            self.checked_links.append(link)
            if link.context_id not in self.link_ids:
                raise OperatorAuthorizationError("context access refused")

    def bind(self, context):
        return self.authority.bind_authenticated(context, privacy_holder=self.holder,
            check_current=self.check, authority_gate=self.authority_gate)


@pytest_asyncio.fixture
async def access(repository):
    store, _ = repository
    yield Access(store)


async def staged(access, context, *, key="save", content=b"private text", ref=None):
    file = await access.service.stage(context, content, publication_key=key,
                                     path="notes.md", media_type="text/markdown")
    return ArtifactCommit(idempotency_key=key, title="Private notes",
                          artifact_id=ref.artifact_id if ref else None,
                          expected_revision_id=ref.revision_id if ref else None,
                          manifest=ArtifactManifest(files=(file,), entrypoint=file.path))


@pytest.mark.asyncio
async def test_service_protocol_and_all_producer_neutral_operations(access):
    service = access.service
    assert isinstance(service, ArtifactService)
    context = access.context()
    with access.bind(context):
        request = await staged(access, context)
        first = await service.commit(context, request)
        assert first.scope == SCOPE
        assert await service.publication_status(context, "save") == first
        assert await service.read(context, first.ref, "notes.md") == b"private text"
        second = await service.commit(context, await staged(access, context, key="edit", ref=first.ref))
        assert (await service.get(context, second.ref)).parent_revision_id == first.ref.revision_id
        assert len((await service.list(context)).items) == 1
        assert len((await service.history(context, first.ref.artifact_id)).items) == 2
        link = ArtifactLink(ref=second.ref, context_kind="conversation", context_id="conversation")
        await service.link(context, link)
        await service.unlink(context, link)
        assert access.checked_links and all(item == link for item in access.checked_links)
        assert await service.read(context, second.ref, "notes.md") == b"private text"
        await service.delete(context, second.ref)
        with pytest.raises(ResourceNotFound):
            await service.get(context, first.ref)
        with pytest.raises(ResourceNotFound):
            await service.publication_status(context, "edit")


@pytest.mark.asyncio
async def test_equal_or_expired_lifetime_context_is_not_a_credential(access):
    context = access.context()
    with pytest.raises(ArtifactAccessDenied):
        await access.service.list(context)
    with access.bind(context):
        assert replace(context) == context
        with pytest.raises(ArtifactAccessDenied):
            await access.service.list(replace(context))
        assert (await access.service.list(context)).items == ()
    with pytest.raises(ArtifactAccessDenied):
        await access.service.list(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("granted_actions", frozenset()), ("granted_actions", frozenset({"artifacts.admin"})),
    ("granted_capabilities", frozenset()), ("permitted_boundary_ids", frozenset({"elsewhere"})),
])
async def test_exact_grants_and_storage_boundary_required(access, field, value):
    context = access.context(**{field: value})
    with access.bind(context), pytest.raises(ArtifactAccessDenied):
        await access.service.list(context)
    assert (await access.service.repository.db.fetchone("SELECT COUNT(*) FROM artifact_owners"))[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("tenant_id", "other"), ("principal_id", "other"),
                                        ("acting_agent_id", None), ("acting_agent_id", "other")])
async def test_exact_authenticated_scope_never_inherits_other_namespace(access, field, value):
    original = access.context()
    with access.bind(original):
        receipt = await access.service.commit(original, await staged(access, original))
    other = access.context(**{field: value})
    with access.bind(other):
        assert (await access.service.list(other)).items == ()
        assert await access.service.publication_status(other, "save") is None
        with pytest.raises(ResourceNotFound):
            await access.service.read(other, receipt.ref, "notes.md")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.ISOLATED, PrivacyMode.EPHEMERAL,
                                 PrivacyMode.ANONYMOUS, PrivacyMode.DEIDENTIFIED])
async def test_modes_without_raw_content_authority_refuse_reads_and_writes(access, mode):
    context = access.context()
    with access.bind(context):
        receipt = await access.service.commit(context, await staged(access, context))
        access.holder.set_mode(mode)
        with pytest.raises(ArtifactAccessDenied):
            await access.service.read(context, receipt.ref, "notes.md")
        with pytest.raises(ArtifactAccessDenied):
            await staged(access, context, key="refused")
        await access.service.delete(context, receipt.ref)
    assert (await access.service.repository.db.fetchone("SELECT COUNT(*) FROM artifact_revisions"))[0] == 0


@pytest.mark.asyncio
async def test_revocation_inside_commit_rolls_back_and_preserves_edit(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    execute = store.db.execute

    async def revoke_after_write(sql, *args, **kwargs):
        result = await execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO artifact_heads"):
            access.granted = False
        return result

    with access.bind(context):
        request = await staged(access, context)
        monkeypatch.setattr(store.db, "execute", revoke_after_write)
        with pytest.raises(ArtifactAccessDenied, match="authority is unavailable") as caught:
            await access.service.commit(context, request)
        assert "sensitive" not in str(caught.value)
        access.granted = True
        monkeypatch.setattr(store.db, "execute", execute)
        assert await access.service.publication_status(context, "save") is None
        assert (await access.service.list(context)).items == ()
        receipt = await access.service.commit(context, request)
        assert await access.service.read(context, receipt.ref, "notes.md") == b"private text"


@pytest.mark.asyncio
async def test_expiry_inside_stage_rolls_back_bytes(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    execute = store.db.execute

    async def expire_after_staging(sql, *args, **kwargs):
        result = await execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO artifact_staging"):
            access.now = context.expires_at
        return result

    with access.bind(context):
        monkeypatch.setattr(store.db, "execute", expire_after_staging)
        with pytest.raises(ArtifactAccessDenied):
            await staged(access, context)
    for table in ("artifact_blobs", "artifact_staging"):
        assert (await store.db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0


@pytest.mark.asyncio
async def test_commit_ack_after_expiry_requires_reconciliation(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    transaction = store.db.transaction

    @asynccontextmanager
    async def expire_after_commit(*args, **kwargs):
        async with transaction(*args, **kwargs):
            yield
        access.now = context.expires_at

    with access.bind(context):
        request = await staged(access, context)
        monkeypatch.setattr(store.db, "transaction", expire_after_commit)
        with pytest.raises(ResourceOutcomeUnknown, match="reconcile"):
            await access.service.commit(context, request)
    monkeypatch.setattr(store.db, "transaction", transaction)
    fresh = access.context()
    with access.bind(fresh):
        receipt = await access.service.publication_status(fresh, "save")
        assert receipt is not None
        assert await access.service.commit(fresh, request) == receipt
        assert len((await access.service.history(fresh, receipt.ref.artifact_id)).items) == 1


@pytest.mark.asyncio
async def test_read_revoked_after_fetch_withholds_content(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    blob = store._blob

    async def revoke_after_blob(*args):
        result = await blob(*args)
        access.granted = False
        return result

    with access.bind(context):
        receipt = await access.service.commit(context, await staged(access, context))
        monkeypatch.setattr(store, "_blob", revoke_after_blob)
        with pytest.raises(ArtifactAccessDenied):
            await access.service.read(context, receipt.ref, "notes.md")


@pytest.mark.asyncio
async def test_host_privacy_transition_waits_for_inflight_read(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    blob = store._blob
    entered, release, transitioned = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow_blob(*args):
        result = await blob(*args)
        entered.set()
        await release.wait()
        return result

    async def transition():
        async with access.holder.gate.exclusive():
            access.holder.set_mode(PrivacyMode.ISOLATED)
            transitioned.set()

    with access.bind(context):
        receipt = await access.service.commit(context, await staged(access, context))
        monkeypatch.setattr(store, "_blob", slow_blob)
        read_task = asyncio.create_task(access.service.read(context, receipt.ref, "notes.md"))
        await asyncio.wait_for(entered.wait(), 3)
        transition_task = asyncio.create_task(transition())
        try:
            await asyncio.sleep(0)
            assert not transitioned.is_set()
            release.set()
            assert await asyncio.wait_for(read_task, 3) == b"private text"
            await asyncio.wait_for(transition_task, 3)
            with pytest.raises(ArtifactAccessDenied):
                await access.service.list(context)
        finally:
            release.set()
            for task in (read_task, transition_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(read_task, transition_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_links_require_current_target_authority(access):
    context = access.context()
    with access.bind(context):
        receipt = await access.service.commit(context, await staged(access, context))
        link = ArtifactLink(ref=receipt.ref, context_kind="conversation", context_id="foreign")
        with pytest.raises(ArtifactAccessDenied):
            await access.service.link(context, link)
    assert (await access.service.repository.db.fetchone("SELECT COUNT(*) FROM artifact_links"))[0] == 0


@pytest.mark.asyncio
async def test_revocation_serializes_with_commit_acknowledgement(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    transaction = store.db.transaction
    entered, release, revoked = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def delayed_commit(*args, **kwargs):
        async with transaction(*args, **kwargs):
            yield
            entered.set()
            await release.wait()

    async def revoke():
        async with access.authority_gate:
            access.granted = False
            revoked.set()

    with access.bind(context):
        request = await staged(access, context)
        monkeypatch.setattr(store.db, "transaction", delayed_commit)
        commit_task = asyncio.create_task(access.service.commit(context, request))
        await asyncio.wait_for(entered.wait(), 3)
        revoke_task = asyncio.create_task(revoke())
        try:
            await asyncio.sleep(0)
            assert not revoked.is_set()
            release.set()
            receipt = await asyncio.wait_for(commit_task, 3)
            assert receipt.idempotency_key == "save"
            await asyncio.wait_for(revoke_task, 3)
            with pytest.raises(ArtifactAccessDenied):
                await access.service.publication_status(context, "save")
        finally:
            release.set()
            for task in (commit_task, revoke_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(commit_task, revoke_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [PermissionError, RuntimeError])
async def test_any_authority_lookup_failure_after_commit_is_unknown(access, monkeypatch, error_type):
    context = access.context()
    store = access.service.repository
    transaction = store.db.transaction
    failed = False
    check = access.check
    def lookup(*args):
        if failed:
            raise error_type("private grant infrastructure error")
        return check(*args)
    access.check = lookup

    @asynccontextmanager
    async def fail_after_commit(*args, **kwargs):
        nonlocal failed
        async with transaction(*args, **kwargs):
            yield
        failed = True

    with access.bind(context):
        request = await staged(access, context)
        monkeypatch.setattr(store.db, "transaction", fail_after_commit)
        with pytest.raises(ResourceOutcomeUnknown):
            await access.service.commit(context, request)
        monkeypatch.setattr(store.db, "transaction", transaction)
        failed = False
        receipt = await access.service.publication_status(context, request.idempotency_key)
        assert receipt is not None
        assert await access.service.commit(context, request) == receipt


@pytest.mark.asyncio
async def test_plain_permission_failure_before_commit_rolls_back_as_denial(access, monkeypatch):
    context = access.context()
    store = access.service.repository
    execute = store.db.execute
    failed = False
    check = access.check
    def lookup(*args):
        if failed:
            raise PermissionError("private lookup failure")
        return check(*args)
    access.check = lookup
    async def execute_then_fail(sql, *args, **kwargs):
        nonlocal failed
        result = await execute(sql, *args, **kwargs)
        if sql.startswith("INSERT INTO artifact_heads"):
            failed = True
        return result
    with access.bind(context):
        request = await staged(access, context)
        monkeypatch.setattr(store.db, "execute", execute_then_fail)
        with pytest.raises(ArtifactAccessDenied):
            await access.service.commit(context, request)
    assert (await store.db.fetchone("SELECT COUNT(*) FROM artifact_revisions"))[0] == 0


@pytest.mark.asyncio
async def test_head_requires_live_authenticated_read_authority(access, monkeypatch):
    context = access.context()
    with access.bind(context):
        receipt = await access.service.commit(context, await staged(access, context))
        assert (await access.service.head(context, receipt.ref.artifact_id)).ref == receipt.ref
    with pytest.raises(ArtifactAccessDenied):
        await access.service.head(context, receipt.ref.artifact_id)
    denied = access.context(granted_actions=frozenset({"artifacts.write"}))
    with access.bind(denied), pytest.raises(ArtifactAccessDenied):
        await access.service.head(denied, receipt.ref.artifact_id)
    store = access.service.repository
    fetchone = store.db.fetchone
    async def revoke_during_head(sql, *args, **kwargs):
        result = await fetchone(sql, *args, **kwargs)
        if "FROM artifact_heads" in sql:
            access.granted = False
        return result
    with access.bind(context):
        monkeypatch.setattr(store.db, "fetchone", revoke_during_head)
        with pytest.raises(ArtifactAccessDenied):
            await access.service.head(context, receipt.ref.artifact_id)
