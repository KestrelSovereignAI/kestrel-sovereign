"""Original-caller host and producer adapters for the canonical workspace broker."""
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict
from kestrel_sdk.features.base import tool
from kestrel_sdk.resources import ResourceError, ResourceUnsupported, ResourceOutcomeUnknown
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.tools import ToolCategory, ToolResult
from kestrel_sdk.workspaces import WorkspaceExportRequest, WorkspaceImportRequest

from kestrel_sovereign.artifacts.http import _body, _json, _delivery_error
from kestrel_sovereign.artifacts.service import ArtifactAccessDenied, ArtifactAction
from kestrel_sovereign.artifacts.workspace import ArtifactWorkspaceBroker, WorkspaceContexts
from kestrel_sovereign.auth import current_caller_context
from kestrel_sovereign.endpoints.agent_helpers import get_caller
from kestrel_sovereign.security.host_authority import HostAuthorityError
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from .producers import _failure


class WorkspacePublication(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    request: WorkspaceExportRequest
    title: str
    publication_key: str
    artifact_id: str | None = None
    expected_revision_id: str | None = None


@dataclass
class _Binding:
    contexts: WorkspaceContexts
    broker: ArtifactWorkspaceBroker

    @property
    def expires_at(self):
        return self.contexts.expires_at


def _result(value):
    revision = value.pop('revision', None)
    parts = [{'type':'artifact', 'data':{'ref':revision['ref'], 'title':revision['title']}}] if revision else []
    return ToolResult.ok('Workspace artifact transfer confirmed.', data=value, parts=parts)


def _workspace_failure(error, *, publication_key=None, transfer_id=None):
    if isinstance(error, ResourceOutcomeUnknown):
        return ToolResult.partial('Workspace transfer outcome remains unknown.',
            'Check artifact_workspace_status with the original session/transfer ID; reconcile publication using artifact_publication_status and its original key.',
            data={'state':'unknown','transfer_id':transfer_id,'publication_key':publication_key})
    if isinstance(error, OperatorAuthorizationError):
        return ToolResult.failed('Workspace transfer authority is unavailable for the original caller.', data={'state':'denied'})
    if isinstance(error, ResourceUnsupported):
        return ToolResult.failed('This governed computer has no supported workspace transport.', data={'state':'unavailable'})
    return _failure(error)


class ArtifactWorkspaceTools:
    @asynccontextmanager
    async def _workspace_binding(self, request=None):
        caller = get_caller(request) if request is not None else current_caller_context()
        visual = self.agent.features.get('VisualComputerFeature')
        if visual is None or visual.agent is not self.agent or visual.runtime is None:
            raise ArtifactAccessDenied('governed workspace capability is unavailable')
        artifact_context = self._resolve_context(request) if request is not None else self._resolve_tool_context()
        # Both existing feature resolvers receive the SAME original caller.
        # Neither context receives the other feature's actions/capability.
        async with artifact_context as artifacts:
            async with visual._bind_context(caller, request=request) as computer:
                contexts = WorkspaceContexts(artifacts, computer)
                yield _Binding(contexts, ArtifactWorkspaceBroker(self._service, visual.runtime, self.agent))

    @tool('artifact_publish_workspace',
          'Publish selected regular files from an exact paused governed computer session as a durable artifact. request_json is the SDK WorkspaceExportRequest; publication_json contains title, publication_key and optional CAS target. Retain all original IDs after an unknown outcome.',
          category=ToolCategory.UTILITY)
    async def artifact_publish_workspace(self, request_json: str, publication_json: str) -> ToolResult:
        value = None
        try:
            if len(request_json.encode()) + len(publication_json.encode()) > 256*1024:
                raise ValueError('workspace request exceeds bounds')
            import json
            publication = json.loads(publication_json)
            if not isinstance(publication, dict):
                raise ValueError('invalid workspace publication')
            value = WorkspacePublication.model_validate_json(json.dumps({**publication, 'request':json.loads(request_json)}))
            async with self._workspace_binding() as binding:
                async with binding.broker.operation(binding.contexts, value.request.session_id, ArtifactAction.WRITE) as lease:
                    return _result(await binding.broker.publish(lease, value.request, **value.model_dump(exclude={'request'})))
        except (ResourceError, HostAuthorityError, ValueError) as error:
            return _workspace_failure(error, publication_key=value.publication_key if value else None, transfer_id=value.request.transfer_id if value else None)

    @tool('artifact_materialize_workspace',
          'Install an exact pinned artifact revision into a fresh paused computer workspace. request_json is the SDK WorkspaceImportRequest including the pinned source manifest; an unknown outcome requires same-ID status reconciliation.',
          category=ToolCategory.UTILITY)
    async def artifact_materialize_workspace(self, request_json: str) -> ToolResult:
        value = None
        try:
            if len(request_json.encode()) > 256*1024:
                raise ValueError('workspace request exceeds bounds')
            value = WorkspaceImportRequest.model_validate_json(request_json)
            async with self._workspace_binding() as binding:
                async with binding.broker.operation(binding.contexts, value.session_id, ArtifactAction.READ) as lease:
                    return _result(await binding.broker.materialize(lease, value))
        except (ResourceError, HostAuthorityError, ValueError) as error:
            return _workspace_failure(error, transfer_id=value.transfer_id if value else None)

    @tool('artifact_workspace_status', 'Reconcile the original session/transfer ID. Missing metadata remains unknown, never proof an accepted transfer failed.', category=ToolCategory.UTILITY)
    async def artifact_workspace_status(self, session_id: str, transfer_id: str) -> ToolResult:
        try:
            async with self._workspace_binding() as binding:
                async with binding.broker.operation(binding.contexts, session_id, ArtifactAction.READ) as lease:
                    value = await binding.broker.status(lease, transfer_id)
                    if value['state'] == 'unknown':
                        return ToolResult.partial('Workspace transfer outcome remains unknown.', 'Retain the original transfer ID and request.', data=value)
                    return ToolResult.ok('Workspace transfer status.', data=value)
        except (ResourceError, HostAuthorityError, ValueError) as error:
            return _workspace_failure(error)


def create_workspace_router(feature):
    router = APIRouter(prefix='/workspaces', tags=['artifacts'])
    def respond(request, action, build, *, prepare=None, session_id=None):
        admitted = {}
        @asynccontextmanager
        async def operation(binding, value):
            sid = session_id if session_id is not None else (value.request.session_id if isinstance(value, WorkspacePublication) else value.session_id)
            async with binding.broker.operation(binding.contexts, sid, action) as lease:
                admitted['lease'] = lease
                yield lease.check
        async def body(binding, value=None):
            return _json(await build(binding.broker, admitted['lease'], value))
        return AuthorizedCapabilityResponse(request, authority=feature._service.authority,
            resolve_context=feature._workspace_binding, operation=operation, build=body,
            prepare=prepare, access_denied=ArtifactAccessDenied,
            error_response=lambda error, prepared: _delivery_error(error, prepared, request))

    @router.post('/publish')
    async def publish(request: Request):
        async def build(broker, lease, value):
            return await broker.publish(lease, value.request, **value.model_dump(exclude={'request'}))
        return respond(request, ArtifactAction.WRITE, build, prepare=lambda:_body(request, 256*1024, WorkspacePublication))

    @router.post('/materialize')
    async def materialize(request: Request):
        async def build(broker, lease, value):
            return await broker.materialize(lease, value)
        return respond(request, ArtifactAction.READ, build, prepare=lambda:_body(request, 256*1024, WorkspaceImportRequest))

    @router.get('/sessions/{session_id}/transfers/{transfer_id}')
    async def status(request: Request, session_id: str, transfer_id: str):
        async def build(broker, lease, _):
            return await broker.status(lease, transfer_id)
        return respond(request, ArtifactAction.READ, build, session_id=session_id)
    return router
