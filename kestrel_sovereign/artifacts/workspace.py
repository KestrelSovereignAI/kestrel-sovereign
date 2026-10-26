"""Bridge governed SDK workspace transfers to the one durable artifact service.

No filesystem paths, second content store or substitute owner resolver live
here. Both exact original contexts remain independently authorized. Cleanup
borrows only the already selected session/transport and cannot read bytes.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256

from pydantic import SecretBytes
from kestrel_sdk.artifacts import ArtifactCommit, ArtifactProvenance
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, ResourceUnsupported, ResourceError
from kestrel_sdk.workspaces import WorkspaceExportRequest, WorkspaceImportRequest, WorkspaceTransfer, WorkspaceTransferProvider

from kestrel_sovereign._async_ownership import create_owned_task, await_owned_task, raise_owned_outcome
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate
from .repository import ArtifactIntegrityError
from .service import ArtifactAccessDenied, ArtifactAction


@dataclass(frozen=True)
class WorkspaceContexts:
    artifacts: object
    visual: object

    @property
    def expires_at(self):
        return min(self.artifacts.expires_at, self.visual.expires_at)


@dataclass
class _Lease:
    contexts: WorkspaceContexts
    session: object
    transfer: WorkspaceTransfer
    action: ArtifactAction
    check: object
    task: object
    active: bool = True
    effect: bool = False


class ArtifactWorkspaceBroker:
    """One lexical bridge; SDK transport and canonical receipts own recovery."""

    def __init__(self, service, runtime, privacy_holder):
        self.service, self.runtime, self.privacy_holder = service, runtime, privacy_holder

    @asynccontextmanager
    async def operation(self, contexts, session_id, action):
        if action not in {ArtifactAction.READ, ArtifactAction.WRITE}:
            raise ValueError('invalid workspace operation')
        if self.service._scope(contexts.artifacts) != self.runtime.scope(contexts.visual):
            raise ArtifactAccessDenied('workspace owners do not match')
        # Deterministic order: original privacy -> artifacts -> visual ->
        # session snapshot -> database. Both authorities check their own grant.
        async with require_durable_persistence_gate(self.privacy_holder).shared():
            async with self.service.authority.operation(contexts.artifacts, ArtifactAction.TRANSFER, self.service.repository.storage_id) as transfer_check:
                async with self.service.authority.operation(contexts.artifacts, action, self.service.repository.storage_id) as artifact_check, self.service.authority.operation(contexts.artifacts, ArtifactAction.READ, self.service.repository.storage_id) as read_check:
                    async with self.runtime.session_operation(contexts.visual, session_id) as (session, driver, visual_check):
                        if 'files' not in session.capabilities or not isinstance(driver, WorkspaceTransferProvider):
                            raise ResourceUnsupported('workspace transfer is unavailable')
                        transfer = driver.workspace_transfer()
                        if not isinstance(transfer, WorkspaceTransfer):
                            raise ResourceUnsupported('workspace extension is unavailable')
                        lease = None
                        def check():
                            if lease is not None and not lease.active:
                                raise ArtifactAccessDenied('workspace operation has ended')
                            transfer_check(); artifact_check(); read_check(); visual_check()
                        lease = _Lease(contexts, session, transfer, action, check, asyncio.current_task())
                        try:
                            async with self.service.restrict_current(contexts.artifacts, check):
                                check()
                                yield lease
                                self._ack(lease)
                        finally:
                            lease.active = False

    def _request(self, lease, request):
        if lease.task is not asyncio.current_task():
            raise ArtifactAccessDenied('workspace operation is not owned by this task')
        lease.check()
        session = lease.session
        if (request.session_id, request.lease_id, request.control_epoch) != (session.session_id, session.lease_id, session.control_epoch):
            raise ResourceConflict('workspace request differs from the admitted session')
        if (request.expires_at <= self.runtime.authority.clock() or request.expires_at > session.expires_at
                or request.limits.max_files > 32 or request.limits.max_file_bytes > 128*1024
                or request.limits.max_total_bytes > 2*1024*1024):
            raise ValueError('workspace request exceeds the host transfer policy')

    async def _call(self, lease, function, *args, request=None, mutation=False):
        lease.check()
        deadline = min(lease.contexts.expires_at, lease.session.expires_at,
                       request.expires_at if request is not None else lease.session.expires_at)
        remaining = (deadline - self.runtime.authority.clock()).total_seconds()
        if remaining <= 0:
            raise ArtifactAccessDenied('workspace operation expired')
        try:
            async with asyncio.timeout(remaining):
                if mutation:
                    lease.effect = True
                result = await function(*args)
        except TimeoutError:
            raise ResourceOutcomeUnknown('workspace outcome requires original-ID reconciliation') from None
        try:
            lease.check()
        except (OperatorAuthorizationError, ResourceConflict):
            if mutation:
                raise ResourceOutcomeUnknown('workspace acknowledgment authority expired') from None
            raise
        return result

    @staticmethod
    def _ack(lease):
        try:
            lease.check()
        except (OperatorAuthorizationError, ResourceConflict):
            if lease.effect:
                raise ResourceOutcomeUnknown('workspace result requires original-ID reconciliation') from None
            raise

    async def _cleanup(self, lease, method, transfer_id):
        async def retire():
            try:
                # No new context, current-data grant, or session selection is
                # acquired after revocation. SDK cleanup accepts retained owner.
                async with asyncio.timeout(5):
                    status = await method(lease.session, transfer_id)
                return status.state in {'released', 'aborted', 'complete'}
            except (ResourceError, OSError, ValueError, TimeoutError):
                return False
        cleanup = create_owned_task(retire(), name='artifact-workspace-retirement')
        outcome = await await_owned_task(cleanup)
        return raise_owned_outcome(outcome, operation='workspace retirement')

    @staticmethod
    def _commit(request, manifest, title, publication_key, artifact_id, expected_revision_id):
        return ArtifactCommit(idempotency_key=publication_key, artifact_id=artifact_id,
            expected_revision_id=expected_revision_id, title=title, manifest=manifest,
            provenance=ArtifactProvenance(session_id=request.session_id, run_id=request.digest(), tool_name='artifact_publish_workspace'))

    async def publish(self, lease, request, *, title, publication_key, artifact_id=None, expected_revision_id=None):
        request = WorkspaceExportRequest.model_validate(request)
        self._request(lease, request)
        if lease.action is not ArtifactAction.WRITE:
            raise ArtifactAccessDenied('workspace publication is not granted')
        context = lease.contexts.artifacts
        retiring = False
        try:
            receipt = await self.service.publication_status(context, publication_key)
            if receipt is not None:
                revision = await self.service.get(context, receipt.ref)
                commit = self._commit(request, revision.manifest, title, publication_key, artifact_id, expected_revision_id)
                if commit.digest() != receipt.request_sha256:
                    raise ResourceConflict('publication key accepted a different workspace request')
                retiring = True
            else:
                try:
                    snapshot = await self._call(lease, lease.transfer.prepare_export, lease.session, request, request=request, mutation=True)
                except (ResourceOutcomeUnknown, asyncio.CancelledError):
                    retiring = True  # The original prepare can be accepted.
                    raise
                retiring = True
                if (snapshot.request != request or snapshot.manifest.entrypoint != request.entrypoint
                        or {file.path for file in snapshot.manifest.files} != set(request.paths)):
                    raise ArtifactIntegrityError('workspace snapshot does not match')
                request.limits.check(snapshot.manifest)
                for file in snapshot.manifest.files:
                    content = (await self._call(lease, lease.transfer.read_export, lease.session, snapshot.snapshot_id, file.path, request=request)).get_secret_value()
                    if len(content) != file.size_bytes or sha256(content).hexdigest() != file.sha256:
                        raise ArtifactIntegrityError('workspace snapshot bytes do not match')
                    try:
                        staged = await self.service.stage(context, content, publication_key=publication_key, path=file.path, media_type=file.media_type)
                    except ResourceConflict:
                        # A concurrent identical commit can win after status.
                        # Only canonical CAS/digest reconciliation decides.
                        break
                    if staged != file:
                        raise ArtifactIntegrityError('workspace staging does not match')
                commit = self._commit(request, snapshot.manifest, title, publication_key, artifact_id, expected_revision_id)
                lease.effect = True
                receipt = await self.service.commit(context, commit)
                if (receipt.request_sha256 != commit.digest() or receipt.manifest_sha256 != commit.manifest.digest()
                        or receipt.scope != self.service._scope(context) or receipt.storage_id != self.service.repository.storage_id):
                    raise ResourceOutcomeUnknown('workspace publication receipt does not match')
                revision = await self.service.get(context, receipt.ref)
        finally:
            cleanup_confirmed = await self._cleanup(lease, lease.transfer.release_export, request.transfer_id) if retiring else None
        self._ack(lease)
        return {'state':'saved', 'receipt':receipt.model_dump(mode='json'), 'revision':revision.model_dump(mode='json'),
                'transfer_id':request.transfer_id, 'cleanup_confirmed':cleanup_confirmed}

    async def materialize(self, lease, request):
        request = WorkspaceImportRequest.model_validate(request)
        self._request(lease, request)
        if lease.action is not ArtifactAction.READ:
            raise ArtifactAccessDenied('workspace source read is not granted')
        revision = await self.service.get(lease.contexts.artifacts, request.source)
        if revision.manifest != request.manifest:
            raise ResourceConflict('workspace source manifest is not the pinned revision')
        attempted = False
        try:
            try:
                status = await self._call(lease, lease.transfer.prepare_import, lease.session, request, request=request, mutation=True)
            except (ResourceOutcomeUnknown, asyncio.CancelledError):
                attempted = True
                raise
            attempted = True
            self._status(request, status, 'import')
            if status.state == 'complete':
                receipt = status.materialization
            else:
                for file in revision.manifest.files:
                    content = await self.service.read(lease.contexts.artifacts, request.source, file.path)
                    if len(content) != file.size_bytes or sha256(content).hexdigest() != file.sha256:
                        raise ArtifactIntegrityError('pinned artifact bytes do not match')
                    status = await self._call(lease, lease.transfer.write_import_file, lease.session,
                        request.transfer_id, file, SecretBytes(content), request=request, mutation=True)
                    self._status(request, status, 'import')
                receipt = await self._call(lease, lease.transfer.commit_import, lease.session, request.transfer_id, request=request, mutation=True)
            if (receipt.transfer_id, receipt.session_id, receipt.lease_id, receipt.control_epoch, receipt.workspace_id,
                    receipt.source, receipt.manifest_sha256, receipt.request_sha256) != (
                    request.transfer_id, request.session_id, request.lease_id, request.control_epoch, request.workspace_id,
                    request.source, request.manifest.digest(), request.digest()):
                raise ResourceOutcomeUnknown('workspace materialization receipt does not match')
            self._ack(lease)
            return {'state':'complete', 'materialization':receipt.model_dump(mode='json'), 'revision':revision.model_dump(mode='json')}
        except BaseException:
            # Unknown commit can already be complete; SDK abort returns that
            # original receipt and never uninstalls/replays a completed import.
            if attempted:
                await self._cleanup(lease, lease.transfer.abort_import, request.transfer_id)
            raise

    @staticmethod
    def _status(request, status, direction):
        if (status.transfer_id, status.session_id, status.request_sha256, status.direction) != (
                request.transfer_id, request.session_id, request.digest(), direction):
            raise ResourceOutcomeUnknown('workspace status does not match the original request')

    async def status(self, lease, transfer_id):
        if lease.task is not asyncio.current_task():
            raise ArtifactAccessDenied('workspace operation is not owned by this task')
        value = await self._call(lease, lease.transfer.status, lease.session, transfer_id)
        if value is None:
            return {'state':'unknown', 'transfer_id':transfer_id}
        if value.transfer_id != transfer_id or value.session_id != lease.session.session_id:
            raise ResourceOutcomeUnknown('workspace status does not match')
        if value.materialization is not None:
            await self.service.get(lease.contexts.artifacts, value.materialization.source)
        lease.check()
        return value.model_dump(mode='json')
