"""Private native Task outcomes for the original conversation or App View."""

from secrets import compare_digest

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown

from kestrel_sovereign.api_errors import api_error_response, execution_control_api_error
from kestrel_sovereign.auth import caller_identity_digest, current_caller_context
from kestrel_sovereign.mcp_input import _Action as InputAction
from kestrel_sovereign.mcp_task_store import RECOVERABLE_OBSERVATION_BLOCKS
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin, _is_digest
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.host_authority import HostAuthorityError
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from kestrel_sovereign.security.request_body import read_json_model


class _TaskOperation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    operation_id: str = Field(min_length=1, max_length=256)


class _TaskInput(_TaskOperation):
    input_responses: dict[str, dict] = Field(min_length=1)


def _error(error, _prepared):
    control = execution_control_api_error(error)
    if control is not None:
        return api_error_response(status_code=control.status_code, code=control.code,
            message=control.message, details=control.details)
    status, code = 500, 'mcp_task_result_service_error'
    if isinstance(error, (HostCapabilityAccessDenied, HostAuthorityError, CSRFError)):
        status, code = 403, 'mcp_task_result_access_denied'
    elif isinstance(error, ResourceOutcomeUnknown):
        status, code = 503, 'mcp_task_mutation_outcome_unknown'
    elif isinstance(error, ResourceConflict):
        status, code = 404, 'mcp_task_result_unavailable'
    elif isinstance(error, ValueError):
        status, code = 400, 'invalid_mcp_task_result_request'
    return JSONResponse({'error': {'code': code}}, status_code=status)


class MCPTaskResults:
    def __init__(self, host):
        self.host = host

    @staticmethod
    def router_for(resolve_host):
        router = APIRouter(prefix='/api/mcp/task-results', tags=['mcp-task-results'])

        def results():
            owner = resolve_host()
            if owner is None:
                raise HTTPException(status_code=404, detail='mcp_task_result_unavailable')
            return owner.results

        @router.get('/{handle}')
        async def chat_result(handle: str, request: Request):
            return results().chat_response(request, handle)

        @router.post('/{handle}/responses')
        async def respond(handle: str, request: Request):
            return results().chat_mutation(request, handle, 'update', _TaskInput)

        @router.post('/{handle}/cancel')
        async def cancel(handle: str, request: Request):
            return results().chat_mutation(request, handle, 'cancel', _TaskOperation)

        return router

    def chat_response(self, request, handle):
        inputs = self.host.input_host

        async def build(_context):
            reader = inputs._require_reader()
            value = await self._read(handle, reader.context.principal_id, reader.session_id,
                view_id=None, reference=self.request_reference(request), require_current=reader.check)
            return JSONResponse(value)

        return self._chat_response(request, InputAction.READ, build)

    def chat_mutation(self, request, handle, kind, model):
        inputs = self.host.input_host

        async def build(_context, value):
            reader = inputs._require_reader()
            record = await self._select(handle, reader.context.principal_id, reader.session_id,
                view_id=None, reference=self.request_reference(request), require_current=reader.check)
            result = await self._mutate(record, reader.session_id, kind, value)
            reader.check()
            return JSONResponse(result)

        return self._chat_response(request, InputAction.RESPOND, build,
            prepare=lambda: read_json_model(request, model, max_bytes=inputs.policy.max_request_bytes))

    def _chat_response(self, request, action, build, *, prepare=None):
        inputs = self.host.input_host
        return AuthorizedCapabilityResponse(request, authority=inputs.authority,
            resolve_context=inputs._resolve_context,
            operation=lambda context, _value: inputs._operation(context, action),
            build=build, error_response=_error, timeout_seconds=inputs.policy.operation_seconds,
            transport_operation=inputs._transport_operation, prepare=prepare)

    def _runtime(self):
        runtime = self.host.feature._task_runtime
        if runtime is None or runtime.host is not self.host:
            raise ResourceConflict('Original MCP Task runtime is unavailable')
        return runtime

    @staticmethod
    def request_reference(request):
        values = request.headers.getlist('x-kestrel-task-result')
        if len(values) != 1 or not _is_digest(values[0]):
            raise HostCapabilityAccessDenied('Original MCP Task result reference is required')
        return values[0]

    def reference(self, record, delivery):
        return self.host.store._digest('mcp-task-result-reference-v1', [
            record.handle, record.submission_digest, delivery.kind, delivery.session_id, delivery.view_token])

    def _app_destination(self, apps, view_id):
        if apps.agent is not self.host.agent or apps.feature is not self.host.feature:
            raise HostCapabilityAccessDenied('Original MCP Task App host is required')
        reader = apps._reader.get()
        physical = apps._physical(view_id)

        def check():
            if apps._physical(view_id) is not physical:
                raise HostCapabilityAccessDenied('Original MCP Task App View changed')

        return reader, check

    def require_app_delivery(self, delivery):
        for apps in getattr(self.host.agent, '_mcp_apps_hosts', ()):
            if (apps.feature is self.host.feature and apps.current_call_session_id() == delivery.session_id
                    and apps._reader.get().physical is apps._views.get(delivery.view_token)):
                _, check = self._app_destination(apps, delivery.view_token)
                return check
        raise HostCapabilityAccessDenied('Original MCP Task App delivery is required')

    async def app_result(self, apps, handle, view_id, reference):
        reader, check = self._app_destination(apps, view_id)
        return await self._read(handle, caller_identity_digest(reader.caller), reader.session_id,
            view_id=view_id, reference=reference, require_current=check)

    async def app_mutation(self, apps, handle, view_id, reference, kind, value):
        reader, check = self._app_destination(apps, view_id)
        record = await self._select(handle, caller_identity_digest(reader.caller), reader.session_id,
            view_id=view_id, reference=reference, require_current=check)
        result = await self._mutate(record, reader.session_id, kind, value)
        check()
        return result

    async def _mutate(self, record, session_id, kind, value):
        runtime = self._runtime()
        arguments = {'inputResponses': value.input_responses} if kind == 'update' else {}
        return await runtime.invoke_mutation(runtime.action_tool(record), record.handle,
            session_id=session_id, operation_id=value.operation_id, kind=kind, arguments=arguments)

    async def _select(self, handle, principal_id, session_id, *, view_id, reference, require_current):
        host = self.host
        require_current()
        host.require_current()
        record = await host.store.read_handle(handle)
        if record is None or record.remote_task_id is None:
            raise ResourceConflict('Original remote MCP Task is unavailable')
        origin = record.binding.origin
        if (type(origin) is not MCPTaskInteractiveOrigin or origin.principal_digest != principal_id
                or origin.session_id != session_id):
            raise HostCapabilityAccessDenied('Original MCP Task operator and conversation are required')
        delivery = (await host.store.open_submission(record)).delivery
        host.require_current()
        require_current()
        if (not compare_digest(reference, self.reference(record, delivery))
                or delivery.session_id != session_id or delivery.view_token != view_id
                or delivery.kind != ('chat' if view_id is None else 'app')):
            raise HostCapabilityAccessDenied('Original MCP Task delivery is required')
        return record

    async def _read(self, handle, principal_id, session_id, *, view_id, reference, require_current):
        record = await self._select(handle, principal_id, session_id, view_id=view_id,
            reference=reference, require_current=require_current)
        local = record.local
        block = local['observation_block']
        if block is not None:
            status = 'suspended' if block['reason'] in RECOVERABLE_OBSERVATION_BLOCKS else 'unknown'
            return {'status': status}
        observation = record.observation
        status = (observation if observation is not None else record.envelope['native'])['status']
        if local['withheld'] is not None:
            return {'status': status, 'result_withheld': True}
        if record.phase == 'SETTLED' and status == 'completed':
            return {'status': status, 'result': observation['result']}
        projection = {'status': status}
        if observation is not None and status == 'input_required':
            projection['forms'] = self._runtime().describe_inputs(record)
        caller = current_caller_context()
        if (status in {'working', 'input_required'} and caller is not None and caller.is_sovereign
                and not record.cancel_requested):
            projection['can_cancel'] = True
        return projection
