"""Closed CURRENT inspect and foreground erase; no portable or resident grant.

A native 204 is retained privately as a control-plane fact. This candidate does
not map it to physical worker destruction or final billing, and never returns a
released SDK lease. No provider/worker is registered by this module.
"""

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    execution_terminal_error,
    require_execution_work,
)

from .native_response import _CurrentRunpodNativeResponse
from .runpod_acquisition import _parts
from .runpod_quote_storage import _RunpodInspection, _RunpodReleaseFact

RUNPOD_ENVIRONMENT_LIFECYCLE_API_VERSION = 1
_ISSUER = object()


class _LifecycleProgram:
    """One fixed canonical request, not a caller executable."""

    def __init__(self, stage):
        self.stage = stage

    def __call__(self, native):
        stage = self.stage
        stage._enter_worker(native, self)
        terminal = None
        try:
            stage.check_parse_current()
            if stage._method == "GET":
                result = stage._control.get_pod(stage._outer._pod_id)
                stage.check_parse_current()
                if result.id != stage._outer._pod_id:
                    raise ResourceConflict("Original Pod identity differs")
            else:
                result = stage._control.delete_pod(stage._outer._pod_id)
                stage.check_parse_current()
                if result is not None:
                    raise ResourceOutcomeUnknown("Original termination result differs")
            stage._native_result = result
            return result
        except BaseException as error:
            terminal = error
            raise
        finally:
            try:
                stage._control.close()
                stage._client_closed = True
            except BaseException as cleanup:  # noqa: BLE001 - retain cancellation and both physical close outcomes.
                raise execution_terminal_error(terminal, cleanup, retain_originals=True)


class _OriginalRunpodNativeStage(_CurrentRunpodNativeResponse):
    """One exact original current outer owner's GET or DELETE native run."""

    def __init__(self, outer, method, *, _issuer):
        if (_issuer is not _ISSUER or type(outer) not in
                (_OriginalRunpodInspect, _OriginalRunpodRelease)
                or type(self) is not _OriginalRunpodNativeStage
                or method not in ("GET", "DELETE")
                or (method == "DELETE" and type(outer) is not _OriginalRunpodRelease)):
            raise ExecutionAuthorityError("Exact original lifecycle stage required")
        self._outer, self._host, self._method = outer, outer._host, method
        self._original_stage = (outer, method, outer._pod_id)
        self._mutation = outer._mutation
        self._owner_task, self._loop = outer._owner_task, outer._loop
        self._state = "preparing"
        self._control = self._client = self._configuration = self._program = None
        self._native_request = self._native_parts = None
        self._thread = self._worker = self._native_result = None
        self._worker_entered = self._worker_exited = self._client_closed = False
        self._request = self._response = None
        self._exhausted = self._response_closed = False
        self._raw_bytes, self._sockets = 0, set()
        self._binding = self

    def _check_original(self):
        self._outer._check_original()
        if (self._outer._stage is not self or self._original_stage !=
                (self._outer, self._method, self._outer._pod_id)):
            raise ExecutionAuthorityError("Original lifecycle stage retired")

    def _require_origin(self):
        self._outer._require_origin()
        self._check_original()

    def _bind_original_vendor_lifecycle(self, control):
        from kestrel_sovereign.execution_sync_http import _configuration

        self._require_origin()
        canonical = self._host._canonical
        if (self._state != "preparing" or self._control is not None
                or type(control) is not canonical.client_type
                or type(control.transport) is not canonical.transport_type
                or control.transport._cpu_environment_lifecycle is not True):
            raise ExecutionAuthorityError("Original fresh lifecycle profile required")
        self._control, self._client = control, control.transport._client
        self._configuration = _configuration(self._client)
        control.transport._prepared_configuration()
        resource = "pods/" + self._host._modules["clients"]._segment(self._outer._pod_id)
        # Same actual builder/kwargs as canonical _request; never concatenate an origin.
        self._native_request = self._client.build_request(
            self._method, resource, params=None, json=None
        )
        self._native_parts = _parts(self._native_request)

    def require_vendor_request(self, transport, method, path, params, json_body,
                               body_bytes, statuses, ambiguous):
        self.check_parse_current()
        expected = "/pods/" + self._host._modules["clients"]._segment(self._outer._pod_id)
        if (transport is not self._control.transport or method != self._method
                or path != expected or params is not None or json_body is not None
                or body_bytes is not None or ambiguous
                or tuple(statuses) != ((200,) if method == "GET" else (204,))):
            raise ExecutionAuthorityError("Original fixed lifecycle request differs")

    def _require_native_invocation(self, client, operation):
        self._require_origin()
        if (self._state != "prepared" or client is not self._client
                or operation is not self._program or type(operation) is not _LifecycleProgram
                or operation.stage is not self):
            raise ExecutionAuthorityError("Original lifecycle native invocation differs")
        self._mutation.check_current()
        self._control.transport._prepared_configuration()

    async def run_original(self):
        from kestrel_sovereign.execution_sync_http import (
            await_native_sync_http,
            bind_native_sync_http_execution_admission,
        )

        self._require_origin()
        clients = self._host._modules["clients"]
        control = clients.RunpodControlPlaneClient(
            api_key=self._host._api_key.get_secret_value(), base_url=self._host._base_url,
            prepared_cpu_account=self._host._account,
            prepared_cpu_ca_data=self._host._ca_data.get_secret_value(),
            cpu_environment_lifecycle=True, max_safe_retries=0,
            connect_timeout=2, read_timeout=4,
        )
        try:
            control.bind_original_cpu_lifecycle(self)
        except BaseException as original:
            try:
                control.close()
                self._client_closed = True
            except BaseException as cleanup:  # noqa: BLE001 - retain cancellation and both physical close outcomes.
                raise execution_terminal_error(original, cleanup, retain_originals=True)
            raise
        self._program, self._state = _LifecycleProgram(self), "prepared"
        terminal = None
        try:
            with bind_native_sync_http_execution_admission(self._client, self._invocation_current):
                result = await await_native_sync_http(
                    self._client, self._program, _original_lifecycle=self
                )
        except BaseException as original:
            terminal = original
            raise
        finally:
            # A pre-entry refusal is not evidence of a physically joined worker.
            # Preserve any original constructor/admission ambiguity unchanged.
            self._worker_exited = self._worker_entered
            if not self._worker_entered:
                try:
                    control.close()
                    self._client_closed = True
                except BaseException as cleanup:  # noqa: BLE001 - retain cancellation and both physical close outcomes.
                    raise execution_terminal_error(terminal, cleanup, retain_originals=True)
        self._require_origin()
        if (not self._client_closed or not self._exhausted or not self._response_closed
                or self._thread.sockets or self._thread.requests
                or result is not self._native_result
                or self._response.status_code != (200 if self._method == "GET" else 204)
                or (self._method == "DELETE" and (self._raw_bytes != 0
                    or self._response.headers.get("content-length", "0") != "0"
                    or "transfer-encoding" in self._response.headers))):
            raise ResourceOutcomeUnknown("Original lifecycle native result did not settle")
        self._state = "joined"
        return result


class _LifecycleOperation:
    def __init__(self, host, context, record, check, *, _issuer):
        from .runpod_environment import RunpodEnvironmentHost

        if (_issuer is not _ISSUER or type(host) is not RunpodEnvironmentHost
                or type(self) not in (_OriginalRunpodInspect, _OriginalRunpodRelease)):
            raise ExecutionAuthorityError("Exact original lifecycle issuer required")
        if (record.lease is None or type(record.lease.provider_resource_id) is not str
                or not 0 < len(record.lease.provider_resource_id) <= 512
                or record.remote_environment is None or record.phase in {"released", "retired"}):
            raise ResourceOutcomeUnknown("Original known Pod custody required")
        self._host, self._context, self._record, self._check = host, context, record, check
        self._original_request, self._original_lease = record.request, record.lease
        self._pod_id = record.lease.provider_resource_id
        for fact in (record.remote_environment.inspection, record.remote_environment.release):
            if fact is not None and fact.provider_resource_id != self._pod_id:
                raise ResourceConflict("Original lifecycle fact target differs")
        self._owner_task, self._loop = asyncio.current_task(), asyncio.get_running_loop()
        self._active, self._used, self._kind = True, False, None
        self._mutation = self._ack = self._stage = self._projection = self._lease = None
        self._finished = self._loop.create_future()
        scope = record.request.scope
        self._key = (scope.tenant_id, scope.principal_id, scope.agent_id,
            record.session_id, None, None, None, record.request.request_id,
            "environment-inspect" if type(self) is _OriginalRunpodInspect else "environment-release")

    def _check_original(self):
        runtime = self._host._runtime
        if (not self._active or runtime.egress._calls.get(self._key) is not self
                or self._record.request != self._original_request
                or self._record.lease.lease_id != self._original_lease.lease_id
                or self._record.lease.provider_resource_id != self._pod_id
                or self._record.lease.expires_at != self._original_lease.expires_at):
            raise ExecutionAuthorityError("Original lifecycle activation retired")
        self._check()
        require_execution_work(runtime.repository.db.backend)
        if type(self) is _OriginalRunpodInspect:
            runtime._request(self._context, self._record.request)
        else:
            runtime._destination(self._record.request.location)
            if self._record.request.location not in runtime.policy.permitted_locations:
                raise ExecutionAuthorityError("Current release destination unavailable")
        if (self._host._provider is None
                or runtime.providers.get(self._record.request.provider_id) is not self._host._provider):
            raise ExecutionAuthorityError("Original lifecycle provider retired")
        self._host._require_configuration()

    def _require_origin(self):
        if asyncio.current_task() is not self._owner_task or asyncio.get_running_loop() is not self._loop:
            raise ExecutionAuthorityError("Original lifecycle task differs")
        self._check_original()

    def revoke(self):
        self._active = False
        if self._stage is not None and self._stage._thread is not None:
            self._stage._thread.cancel(None)

    def _quoted_plain(self, mutation):
        self._require_origin()
        if mutation is not self._mutation or self._projection is None or self._lease is None:
            raise ExecutionAuthorityError("Original captured lifecycle facts absent")
        plain = json.loads(mutation._plain)
        if plain["remote_environment"] != self._record.remote_environment.private_value():
            raise ResourceConflict("Original lifecycle projection changed")
        plain.update(remote_environment=self._projection.private_value(),
            lease=self._lease.model_dump(mode="json"), revision=mutation._revision + 1)
        if type(self) is _OriginalRunpodRelease:
            plain.update(phase="fenced", frame=None)
        return plain

    async def _commit(self):
        self._ack = await self._mutation.commit_quote()
        # ACK and exact committed ciphertext are consumed by this SAME operation.
        # No generic stale save or separate projection store.

    async def _reload_ack(self):
        self._record = await self._host._runtime.repository._accepted_runpod_lifecycle(
            self._record.request.scope, self, self._lease
        )

    async def _stage_run(self, method):
        self._stage = _OriginalRunpodNativeStage(self, method, _issuer=_ISSUER)
        return await self._stage.run_original()


class _OriginalRunpodInspect(_LifecycleOperation):
    async def run_original(self):
        self._require_origin()
        if self._used:
            raise ExecutionAuthorityError("Original inspect already used")
        self._used = True
        try:
            plan = self._host._saved_quote_plan(self)
            async with self._host._runtime.repository._current_runpod_quote(self) as mutation:
                self._mutation = mutation
                pod = await self._stage_run("GET")
                self._require_origin()
                observation = self._host._modules["cpu_realized"].inspect_cpu_readiness(pod, plan, expected_pod_id=self._pod_id, at=datetime.now(UTC))
                # RUNNING API facts do not admit a worker or supply driver credentials.
                self._lease = self._record.lease.model_copy(update={"state": "pending"})
                self._projection = self._record.remote_environment.model_copy(update={
                    "inspection": _RunpodInspection(provider_resource_id=self._pod_id,
                        observed_at=observation.observed_at, state=observation.phase),
                })
                await self._commit()
            await self._reload_ack()
            return self._lease
        finally:
            self._finished.set_result(None)


class _OriginalRunpodRelease(_LifecycleOperation):
    async def run_original(self):
        self._require_origin()
        if self._used:
            raise ExecutionAuthorityError("Original release already used")
        self._used = True
        try:
            self._host._saved_quote_plan(self)  # Historical original facts, never spend freshness.
            original = self._record.remote_environment.release
            if original is not None and original.state == "acknowledged":
                raise ResourceOutcomeUnknown("Original termination acknowledged; physical release unverified")
            # Intent ACK precedes either native stage; loss never authorizes DELETE.
            async with self._host._runtime.repository._current_runpod_quote(self) as mutation:
                self._mutation, self._kind = mutation, "release-intent"
                fact = original or _RunpodReleaseFact(intent_id=uuid4().hex,
                    provider_resource_id=self._pod_id, requested_at=datetime.now(UTC), state="intent")
                self._lease = self._record.lease.model_copy(update={"state": "unknown"})
                self._projection = self._record.remote_environment.model_copy(update={"release": fact})
                await self._commit()
            await self._reload_ack()
            async with self._host._runtime.repository._current_runpod_quote(self) as mutation:
                self._mutation, self._kind = mutation, "release-acknowledgement"
                await self._stage_run("GET")  # Exact owned ID, no readiness/price/spend predicate.
                await self._stage_run("DELETE")
                self._require_origin()
                self._projection = self._record.remote_environment.model_copy(update={
                    "release": fact.model_copy(update={"state": "acknowledged",
                        "acknowledged_at": datetime.now(UTC), "native_status": 204}),
                })
                await self._commit()
            await self._reload_ack()
            return self._lease
        finally:
            self._finished.set_result(None)
