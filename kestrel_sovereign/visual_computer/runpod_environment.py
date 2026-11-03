"""Closed CURRENT Runpod quote binding; no provider registration or erase grant."""

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from kestrel_sdk.computer_use import EnvironmentCapabilities
from kestrel_sdk.resources import ResourceConflict, ResourceUnsupported
from pydantic import SecretBytes, SecretStr

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    require_execution_work,
)

from .native_response import _CurrentRunpodNativeResponse
from .runpod_acquisition import _canonical_runpod, _parts
from .runpod_quote_storage import (
    _bytes,
    _caps,
    _RunpodControlPlaneBinding,
    _RunpodEnvironmentEnvelope,
)

RUNPOD_ENVIRONMENT_HOST_API_VERSION = 1
_ISSUER = object()


class RunpodEnvironmentHost:
    """Trusted factory configuration, never authority without a live activation."""

    api_version = RUNPOD_ENVIRONMENT_HOST_API_VERSION
    lifecycle_api_version = 1

    def __init__(
        self,
        runtime,
        *,
        profile,
        noncompute,
        capabilities,
        account,
        api_key,
        base_url,
        ca_data,
        credential_selector,
    ):
        from .runtime import VisualComputerRuntime

        if (
            type(self) is not RunpodEnvironmentHost
            or type(runtime) is not VisualComputerRuntime
        ):
            raise ResourceUnsupported("Exact canonical runtime host required")
        canonical = _canonical_runpod()
        modules = {
            module.__name__.rsplit(".", 1)[-1]: module
            for module, _ in canonical.modules
        }
        if "cpu_environment_codec" not in modules:
            raise ResourceUnsupported("Canonical private Runpod quote codec required")
        cpu = modules["cpu_environment"]
        if (
            type(profile) is not cpu.CpuEnvironmentProfile
            or type(noncompute) is not cpu.CpuNonComputeCostPolicy
            or type(capabilities) is not EnvironmentCapabilities
            or capabilities.provider_id != profile.provider_id
            or capabilities.location != "remote"
            or capabilities.cpu_only is not True
            or capabilities.profile_persistence is not False
            or type(api_key) is not SecretStr
            or type(ca_data) is not SecretBytes
        ):
            raise ResourceUnsupported(
                "Explicit original Runpod profile and independent capabilities required"
            )
        self._runtime, self._canonical, self._modules = runtime, canonical, modules
        self._profile, self._noncompute, self._capabilities = (
            profile,
            noncompute,
            capabilities,
        )
        self._account, self._api_key, self._base_url = account, api_key, base_url
        self._ca_data, self._credential_selector = ca_data, credential_selector
        self._provider = None
        self._original_configuration = (
            profile.model_dump_json(),
            noncompute.model_dump_json(),
            _bytes(_caps(capabilities)),
            account,
            api_key.get_secret_value(),
            base_url,
            ca_data.get_secret_value(),
            credential_selector,
        )

    def _require_configuration(self):
        self._canonical.require_original()
        actual = (
            self._profile.model_dump_json(),
            self._noncompute.model_dump_json(),
            _bytes(_caps(self._capabilities)),
            self._account,
            self._api_key.get_secret_value(),
            self._base_url,
            self._ca_data.get_secret_value(),
            self._credential_selector,
        )
        if actual != self._original_configuration:
            raise ExecutionAuthorityError("Original Runpod host configuration changed")

    def _bind_provider(self, provider):
        if self._provider is not None:
            raise ResourceConflict("Original Runpod host provider already bound")
        self._provider = provider

    @asynccontextmanager
    async def _activate_quote(self, context, record):
        from .runtime import VisualAction

        if self._runtime.scope(context) != record.request.scope:
            raise ExecutionAuthorityError("Original quote scope differs")
        async with self._runtime.operation(context, VisualAction.ADMIT) as check:
            self._runtime._request(context, record.request)
            if (
                self._provider is None
                or self._runtime.providers.get(record.request.provider_id)
                is not self._provider
            ):
                raise ExecutionAuthorityError(
                    "Original Runpod provider binding unavailable"
                )
            operation = _OriginalRunpodQuote(
                self, context, record, check, _issuer=_ISSUER
            )
            async with self._runtime.egress._activate_environment_quote(operation):
                yield operation

    @asynccontextmanager
    async def quote_operation(self, request):
        # Lookup uses the exact original Task and host issuer, never DTO/IDs.
        task = asyncio.current_task()
        calls = [
            call
            for call in self._runtime.egress._calls.values()
            if type(call) is _OriginalRunpodQuote
            and call._host is self
            and call._owner_task is task
        ]
        if len(calls) != 1:
            raise ExecutionAuthorityError("No original current quote operation")
        call = calls[0]
        call._require_origin()
        if request != call._record.request:
            raise ResourceConflict("Original quote request differs")
        yield call
        call._check_original()

    @asynccontextmanager
    async def _activate_lifecycle(self, context, record, *, release):
        from .runtime import VisualAction
        from .runpod_lifecycle import _ISSUER, _OriginalRunpodInspect, _OriginalRunpodRelease

        if self._runtime.scope(context) != record.request.scope:
            raise ExecutionAuthorityError("Original lifecycle scope differs")
        action = VisualAction.RELEASE if release else VisualAction.ADMIT
        async with self._runtime.operation(context, action) as check:
            if release:
                key = self._runtime._key(record.request.scope, record.session_id)
                if key not in self._runtime._fences:
                    self._runtime._fence(record.request.scope, record.session_id)
                # Drain prior calls before registering this priority operation.
                await self._runtime.egress.revoke_session(record.request.scope, record.session_id)
            else:
                self._runtime._request(context, record.request)
            kind = _OriginalRunpodRelease if release else _OriginalRunpodInspect
            call = kind(self, context, record, check, _issuer=_ISSUER)
            async with self._runtime.egress._activate_environment_quote(call):
                yield call

    def _activate_inspect(self, context, record):
        return self._activate_lifecycle(context, record, release=False)

    def _activate_release(self, context, record):
        return self._activate_lifecycle(context, record, release=True)

    @asynccontextmanager
    async def _lifecycle_operation(self, scope, lease, *, release):
        from .runpod_lifecycle import _OriginalRunpodInspect, _OriginalRunpodRelease

        kind = _OriginalRunpodRelease if release else _OriginalRunpodInspect
        calls = [call for call in self._runtime.egress._calls.values()
                 if type(call) is kind and call._host is self
                 and call._owner_task is asyncio.current_task()]
        if len(calls) != 1:
            raise ExecutionAuthorityError("No original current lifecycle operation")
        call = calls[0]
        call._require_origin()
        if scope != call._record.request.scope or lease != call._record.lease:
            raise ResourceConflict("Original lifecycle resource facts differ")
        yield call
        call._check_original()

    def inspect_operation(self, scope, lease):
        return self._lifecycle_operation(scope, lease, release=False)

    def release_operation(self, scope, lease):
        return self._lifecycle_operation(scope, lease, release=True)

    def _saved_quote_plan(self, call):
        # Private restoration is factual only, on the same original current
        # activation. It neither reprices nor restores cloud I/O permission.
        call._require_origin()
        record = call._record
        projection = record.remote_environment
        if projection is None or record.quote is None:
            raise ResourceUnsupported("Original private Runpod quote facts absent")
        projection.require_lifecycle_reservation()
        codec = self._modules["cpu_environment_codec"]
        plan = codec.deserialize_cpu_environment_plan(
            projection.plan_bytes,
            expected_storage_sha256=projection.plan_storage_sha256,
        )
        binding = projection.binding
        fingerprint = hmac.new(
            self._runtime.repository._key,
            b"runpod-current-key-v1\0" + self._api_key.get_secret_value().encode(),
            hashlib.sha256,
        ).hexdigest()
        if (
            plan.digest() != projection.plan_digest
            or plan.request != record.request
            or plan.profile != self._profile
            or plan.noncompute != self._noncompute
            or projection.capabilities != self._capabilities
            or binding.account != self._account
            or binding.credential_selector != self._credential_selector
            or binding.base_url
            != self._modules["clients"]._validate_v2_base_url(self._base_url)
            or not hmac.compare_digest(binding.credential_fingerprint, fingerprint)
            or binding.tls_sha256
            != hashlib.sha256(self._ca_data.get_secret_value()).hexdigest()
            or binding.source_sha256
            != hashlib.sha256(_bytes(dict(self._canonical.source_sha256))).hexdigest()
            or projection.quote_binding_sha256
            != hashlib.sha256(
                codec._json_bytes(record.quote.model_dump(mode="python"))
            ).hexdigest()
        ):
            raise ResourceUnsupported("Original private Runpod quote binding differs")
        codec.require_cpu_environment_quote(
            plan, record.quote, capabilities=self._capabilities, at=plan.planned_at
        )
        call._require_origin()
        return plan


class _QuoteProgram:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, native):
        owner = self.owner
        owner._enter_worker(native, self)
        terminal = None
        try:
            owner.check_parse_current()
            cpu = owner._host._modules["cpu_environment"]
            planner = cpu.CpuEnvironmentPlanner(owner._control)
            plan = planner.plan(
                owner._record.request, owner._host._profile, owner._host._noncompute
            )
            owner.check_parse_current()
            owner._native_plan = plan
            return plan
        except BaseException as error:
            terminal = error
            raise
        finally:
            try:
                owner._control.close()
                owner._client_closed = True
            except BaseException as cleanup:
                if terminal is not None:
                    from kestrel_sovereign.execution_custody import (
                        execution_terminal_error,
                    )

                    raise execution_terminal_error(
                        terminal, cleanup, retain_originals=True
                    )
                raise


class _OriginalRunpodQuote(_CurrentRunpodNativeResponse):
    """Exact original task/host/key owner; no public constructor admission."""

    def __init__(self, host, context, record, check, *, _issuer):
        if (
            type(self) is not _OriginalRunpodQuote
            or type(host) is not RunpodEnvironmentHost
            or _issuer is not _ISSUER
        ):
            raise ExecutionAuthorityError("Exact original quote issuer required")
        self._host, self._context, self._record, self._check = (
            host,
            context,
            record,
            check,
        )
        self._owner_task, self._loop = (
            asyncio.current_task(),
            asyncio.get_running_loop(),
        )
        self._active, self._state, self._used = True, "issued", False
        self._mutation = self._control = self._client = self._program = None
        self._native_request = self._configuration = self._projection = self._quote = (
            self._ack
        ) = None
        self._thread = self._worker = self._native_plan = None
        self._worker_entered = self._worker_exited = self._client_closed = False
        self._request = self._response = None
        self._exhausted = self._response_closed = False
        self._raw_bytes, self._sockets = 0, set()
        self._binding = self
        self._finished = self._loop.create_future()
        scope = record.request.scope
        self._key = (
            scope.tenant_id,
            scope.principal_id,
            scope.agent_id,
            record.session_id,
            None,
            None,
            None,
            record.request.request_id,
            "environment-quote",
        )

    def _check_original(self):
        if (
            type(self) is not _OriginalRunpodQuote
            or not self._active
            or self._host._runtime.egress._calls.get(self._key) is not self
        ):
            raise ExecutionAuthorityError("Original quote activation retired")
        self._check()
        runtime = self._host._runtime
        require_execution_work(runtime.repository.db.backend)
        runtime._request(self._context, self._record.request)
        if (
            self._host._provider is None
            or runtime.providers.get(self._record.request.provider_id)
            is not self._host._provider
        ):
            raise ExecutionAuthorityError("Original Runpod provider binding retired")
        self._host._require_configuration()

    def _require_origin(self):
        if (
            asyncio.current_task() is not self._owner_task
            or asyncio.get_running_loop() is not self._loop
        ):
            raise ExecutionAuthorityError("Original quote task differs")
        self._check_original()

    def revoke(self):
        self._active = False
        if self._thread is not None:
            self._thread.cancel(None)

    def _bind_original_vendor_quote(self, control):
        self._require_origin()
        canonical = self._host._canonical
        canonical.require_original()
        if (
            self._state != "preparing"
            or self._control is not None
            or type(control) is not canonical.client_type
            or type(control.transport) is not canonical.transport_type
            or getattr(control.transport, "_cpu_environment_quote", None) is not True
        ):
            raise ExecutionAuthorityError("Original fresh quote profile required")
        self._control, self._client = control, control.transport._client
        from kestrel_sovereign.execution_sync_http import _configuration

        self._configuration = _configuration(self._client)
        control.transport._prepared_configuration()
        product = self._host._modules["models"].ComputeProduct.POD.value
        self._native_request = self._client.build_request(
            "GET",
            "catalog/cpus",
            params={
                "include": "AVAILABILITY",
                "product": product,
                "vcpuCount": self._host._profile.vcpu_count,
            },
        )
        self._native_parts = _parts(self._native_request)

    def _require_native_invocation(self, client, operation):
        self._require_origin()
        if (
            self._state != "prepared"
            or client is not self._client
            or operation is not self._program
            or type(operation) is not _QuoteProgram
            or operation.owner is not self
        ):
            raise ExecutionAuthorityError("Original quote native invocation differs")
        self._mutation.check_current()
        self._host._canonical.require_original()
        self._control.transport._prepared_configuration()






    def _quoted_plain(self, mutation):
        self._require_origin()
        if (
            mutation is not self._mutation
            or self._state != "quoted"
            or self._projection is None
            or self._quote is None
            or self._ack is not None
        ):
            raise ExecutionAuthorityError("Original captured quote facts absent")
        plain = json.loads(mutation._plain)
        if (
            plain["phase"] != "admitted"
            or plain.get("quote") is not None
            or plain.get("remote_environment") is not None
        ):
            raise ResourceConflict("Original quote row is not unquoted")
        plain.update(
            quote=self._quote.model_dump(mode="json"),
            phase="quoted",
            revision=mutation._revision + 1,
            remote_environment=self._projection.private_value(),
        )
        return plain

    def restore_original(self):
        self._require_origin()
        if self._used:
            raise ExecutionAuthorityError("Original quote invocation already used")
        self._used = True
        try:
            return self._host._saved_quote_plan(self)
        finally:
            if not self._finished.done():
                self._finished.set_result(None)

    async def run_original(self):
        from kestrel_sovereign.execution_sync_http import (
            await_native_sync_http,
            bind_native_sync_http_execution_admission,
        )

        self._require_origin()
        if self._used:
            raise ExecutionAuthorityError("Original quote invocation already used")
        self._used = True
        try:
            async with self._host._runtime.repository._current_runpod_quote(
                self
            ) as mutation:
                self._mutation, self._state = mutation, "preparing"
                clients = self._host._modules["clients"]
                if (
                    "bind_original_cpu_quote"
                    not in clients.RunpodControlPlaneClient.__dict__
                ):
                    raise ResourceUnsupported(
                        "Canonical native Runpod quote profile unavailable"
                    )
                control = clients.RunpodControlPlaneClient(
                    api_key=self._host._api_key.get_secret_value(),
                    base_url=self._host._base_url,
                    prepared_cpu_account=self._host._account,
                    prepared_cpu_ca_data=self._host._ca_data.get_secret_value(),
                    cpu_environment_quote=True,
                    max_safe_retries=0,
                    connect_timeout=2,
                    read_timeout=4,
                )
                try:
                    control.bind_original_cpu_quote(self)
                except BaseException as original:
                    # Binding is metadata-only, but its pristine native client
                    # is still an owned resource and must retire under the
                    # original key/backing holds before this scope exits.
                    try:
                        control.close()
                        self._client_closed = True
                    except BaseException as cleanup:  # noqa: BLE001 - preserve both original physical outcomes.
                        from kestrel_sovereign.execution_custody import (
                            execution_terminal_error,
                        )

                        raise execution_terminal_error(
                            original, cleanup, retain_originals=True
                        )
                    raise
                self._program, self._state = _QuoteProgram(self), "prepared"
                terminal = None
                try:
                    with bind_native_sync_http_execution_admission(
                        self._client, self._invocation_current
                    ):
                        plan = await await_native_sync_http(
                            self._client, self._program, _original_quote=self
                        )
                except BaseException as original:
                    terminal = original
                    raise
                finally:
                    self._worker_exited = self._worker_entered
                    if not self._worker_entered:
                        try:
                            control.close()
                            self._client_closed = True
                        except BaseException as cleanup:  # noqa: BLE001 - preserve both original physical outcomes.
                            from kestrel_sovereign.execution_custody import execution_terminal_error
                            raise execution_terminal_error(terminal, cleanup, retain_originals=True)
                self._require_origin()
                if (
                    not self._client_closed
                    or not self._exhausted
                    or not self._response_closed
                    or self._thread.sockets
                    or self._thread.requests
                    or plan is not self._native_plan
                ):
                    raise ExecutionAuthorityError(
                        "Original quote native resources did not join"
                    )
                canonical = self._host._canonical
                canonical.require_original()
                codec = self._host._modules["cpu_environment_codec"]
                plan_bytes = codec.serialize_cpu_environment_plan(plan)
                caps = self._host._capabilities
                quote = codec.project_cpu_environment_quote(
                    plan, capabilities=caps, at=datetime.now(UTC)
                )
                source = hashlib.sha256(
                    _bytes(dict(canonical.source_sha256))
                ).hexdigest()
                key = self._host._runtime.repository._key
                binding = _RunpodControlPlaneBinding(
                    account=self._host._account,
                    base_url=str(control.transport.base_url),
                    credential_selector=self._host._credential_selector,
                    credential_fingerprint=hmac.new(
                        key,
                        b"runpod-current-key-v1\0"
                        + self._host._api_key.get_secret_value().encode(),
                        hashlib.sha256,
                    ).hexdigest(),
                    tls_sha256=hashlib.sha256(
                        self._host._ca_data.get_secret_value()
                    ).hexdigest(),
                    source_sha256=source,
                )
                self._projection = _RunpodEnvironmentEnvelope(
                    plan_bytes=plan_bytes,
                    plan_storage_sha256=hashlib.sha256(
                        plan_bytes.get_secret_value()
                    ).hexdigest(),
                    plan_digest=plan.digest(),
                    capabilities=caps,
                    capabilities_sha256=hashlib.sha256(_bytes(_caps(caps))).hexdigest(),
                    quote_binding_sha256=hashlib.sha256(
                        codec._json_bytes(quote.model_dump(mode="python"))
                    ).hexdigest(),
                    binding=binding,
                )
                self._projection.require_lifecycle_reservation()
                self._quote, self._state = quote, "quoted"
                self._ack = await mutation.commit_quote()
                self._require_origin()
                return quote
        finally:
            if not self._finished.done():
                self._finished.set_result(None)
