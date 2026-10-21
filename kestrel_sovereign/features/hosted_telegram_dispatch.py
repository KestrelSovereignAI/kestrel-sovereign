"""Host-authorized Telegram activation and send on one admitted child.

The existing private ingress owner also owns this complete dispatch. Route
authority belongs to the embedding host, not to process-local activation.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    current_execution_custody,
)


@dataclass(frozen=True, slots=True)
class HostedTelegramBinding:
    """Nonsecret binding selected under the host's current route authority."""

    tenant_id: str
    agent_id: str
    bot_id: str
    chat_id: str
    activation_generation: str


@dataclass(slots=True)
class HostedTelegramOutbound:
    """One host authority interval, never a persistent readiness assertion.

    A host context may supply an already-confirmed message ID to avoid replay.
    Otherwise Core records its detached actual RPC response before context
    exit, allowing the SAME owner to persist confirmation before unlocking.
    """

    binding: HostedTelegramBinding
    prior_message_id: str | None = None
    response: Mapping[str, Any] | None = field(default=None, init=False, repr=False)


_active_dispatch: ContextVar[Any] = ContextVar("hosted_telegram_dispatch", default=None)


def is_hosted_send(proxy: Any, name: Any) -> bool:
    return (
        type(name) is str
        and name == "telegram_send"
        and proxy._is_telegram_runtime()
        and proxy._hosted_telegram_startup_attested
    )


def start_dispatch(proxy: Any, name: Any, arguments: Any):
    """Snapshot and admit ONE original owner before publishing its task."""
    from kestrel_sovereign.features import isolated_runtime as runtime
    from kestrel_sovereign.features.scheduler.runner import (
        capture_scheduler_execution_scope,
    )

    # This is private owner metadata, not a service ingress RPC. Tool names
    # use underscores; canonical ingress labels intentionally use hyphens.
    request = runtime._prepare_host_ingress_request("telegram-hosted-outbound", arguments)
    slot = runtime._HostIngressOutcomeSlot()
    caller = asyncio.current_task()
    admission = runtime._HostIngressAdmission(
        proxy._terminal_lifecycle_generation,
        current_execution_custody(proxy.agent),
        proxy=proxy,
        withdrawal_stops_work=True,
        withdrawal_task=caller,
        withdrawal_cancel_count=caller.cancelling() if caller is not None else 0,
        dispatch_deadline=asyncio.get_running_loop().time() + proxy._hosted_telegram_outbound_timeout,
        # Capture trusted provenance BEFORE allocation/publication can refuse
        # the original worker. No refusal may become scheduler success data.
        scheduled_tool=capture_scheduler_execution_scope() is not None,
    )
    proxy._host_ingress_admissions.add(admission)

    async def execute():
        async def dispatch(_label, payload):
            return await proxy._call_isolated_tool_owned("telegram_send", payload)

        token = _active_dispatch.set(admission)
        try:
            await proxy._run_host_ingress(
                request, slot, admission,
                _rpc_call=dispatch,
            )
        finally:
            _active_dispatch.reset(token)

    if request is None:
        proxy._host_ingress_admissions.discard(admission)
    else:
        try:
            create_owned_task(
                execute(), name=f"hosted-telegram-send:{proxy.name}",
                retain_originals=True, admission=admission.admission,
            )
        except BaseException as error:  # noqa: BLE001 - preserve original task publication refusal
            admission.opening = error
    return proxy, admission, slot


async def wait_dispatch(owner):
    """Observe cancellation without abandoning the original authority owner.

    Deadline expiry seals the proxy and starts its EXISTING terminal cleanup.
    A still-running operation remains rooted by its original ingress admission;
    it retains its own host context until accepted work really settles.
    """
    from kestrel_sovereign.features import isolated_runtime as runtime

    proxy, admission, slot = owner
    task = admission.admission.task
    if task is None:
        outcome = runtime._finish_host_ingress_admission(admission, slot.take())
        return outcome, None

    def expire():
        if admission.dispatch_expired:
            return proxy._host_ingress_public_control or admission.core_outcome
        admission.dispatch_expired = True
        error = (
            ExecutionCommitOutcomeError("unknown") if admission.provider_rpc_started
            else ExecutionAuthorityError("hosted Telegram dispatch deadline expired")
        )
        admission.native_error = error
        admission.core_outcome = runtime._HostIngressOutcome(
            runtime._HOST_INGRESS_CONTROL,
            commit_outcome="unknown" if admission.provider_rpc_started else None,
        )
        # Public projection is monotonic even while the original child stop
        # and authority interval are still settling in the retained owner.
        outcome = runtime._retain_host_ingress_public_outcome(proxy, admission.core_outcome)
        proxy._retain_execution_control(error)
        proxy._latch_terminal_lifecycle()
        try:
            # The original canonical proxy closer, never a second stop task.
            proxy._admit_terminal_cleanup(retry_retirement=False)
        except BaseException as opening:  # noqa: BLE001 - retain original canonical closer refusal
            admission.opening = opening
        if not admission.provider_rpc_started:
            task.cancel()

        def harvest(_task):
            runtime._consume_host_ingress_operation(_task)
            runtime._finish_host_ingress_admission(admission, slot.take())

        try:
            task.add_done_callback(harvest)
        except BaseException as opening:  # noqa: BLE001 - original remains rooted by its retained admission
            if admission.opening is None:
                admission.opening = opening
        return outcome

    admission.dispatch_expiry = expire
    outcome, cancellation = await runtime._wait_for_host_ingress_operation(
        task, slot, admission, deadline_seconds=max(0.0, admission.dispatch_deadline - asyncio.get_running_loop().time()),
        on_deadline=expire,
    )
    outcome = runtime._finish_host_ingress_admission(admission, outcome)
    if admission.cancellation is not None and admission.provider_rpc_started:
        outcome = runtime._HostIngressOutcome(
            runtime._HOST_INGRESS_CONTROL, commit_outcome="unknown",
        )
        admission.native_error = ExecutionCommitOutcomeError("unknown")
        admission.core_outcome = outcome
        outcome = runtime._retain_host_ingress_public_outcome(proxy, outcome)
        proxy._retain_execution_control(admission.native_error)
        proxy._latch_terminal_lifecycle()
        cancellation = None
    return outcome, cancellation


def _binding_payload(proxy, selected, destination):
    from kestrel_sovereign.features.isolated_runtime import canonical_telegram_bot_id

    binding = selected.binding
    if type(binding) is not HostedTelegramBinding:
        raise TypeError("invalid hosted Telegram binding")
    for value in (binding.tenant_id, binding.agent_id, binding.bot_id,
                  binding.chat_id, binding.activation_generation):
        if type(value) is not str or not value or len(value) > 512:
            raise ValueError("invalid hosted Telegram binding identity")
    if (binding.agent_id != proxy.agent.did
        or canonical_telegram_bot_id(binding.bot_id) != binding.bot_id
        or binding.chat_id != destination):
        raise ValueError("hosted Telegram binding differs from dispatch")
    return {
        "scope": {"agent_id": binding.agent_id, "tenant_id": binding.tenant_id},
        "bot_id": binding.bot_id, "chat_id": binding.chat_id,
        "activation_generation": binding.activation_generation,
    }


async def dispatch_admitted(proxy, arguments, context):
    """Prepare and send inside the SAME existing traffic admission and owner."""
    from kestrel_sovereign.features import isolated_runtime as runtime

    admission = _active_dispatch.get()
    result, failure = None, None
    activation_started = activation_settled = False
    try:
        if (admission is None or admission.proxy is not proxy
            or admission.admission.task is not asyncio.current_task()):
            raise RuntimeError("hosted Telegram requires its original dispatch owner")
        admission.require_work(proxy.agent)
        if (type(arguments) is not dict or set(arguments) != {"to", "message"}
            or any(type(value) is not str for value in arguments.values())):
            raise TypeError("invalid hosted Telegram send arguments")
        factory = proxy._hosted_telegram_outbound_authority
        if not callable(factory):
            raise TypeError("hosted Telegram outbound requires host authority")
        client = proxy._client
        admission.client = client
        admission.physical_client_retired = asyncio.get_running_loop().create_future()
        authority = factory(proxy, MappingProxyType(dict(arguments)))
        async with authority as selected:
            try:
                admission.require_work(proxy.agent)
                if type(selected) is not HostedTelegramOutbound:
                    raise TypeError("invalid hosted Telegram outbound authority")
                payload = _binding_payload(proxy, selected, arguments["to"])
                if selected.prior_message_id is not None:
                    if type(selected.prior_message_id) is not str or not selected.prior_message_id:
                        raise ValueError("invalid prior confirmed Telegram message identity")
                    result = {"status": "ok", "to": arguments["to"],
                              "confirmation": "Previously confirmed hosted Telegram send",
                              "message_id": selected.prior_message_id}
                else:
                    activation_started = True
                    activated = await proxy._call_exact_client_host_ingress(
                        client, "telegram-hosted-activate", payload,
                    )
                    activation_settled = True
                    if not (type(activated) is dict and activated.get("status") == "ok"
                            and type(activated.get("http_status")) is int
                            and activated["http_status"] == 200
                            and activated.get("transport") == "hosted"):
                        raise RuntimeError("hosted Telegram activation was refused")
                    admission.require_work(proxy.agent)
                    if proxy._client is not client:
                        raise RuntimeError("hosted Telegram child changed during dispatch")
                    if context is not None:
                        from kestrel_sovereign.features.scheduler.runner import (
                            get_current_scheduler_execution,
                        )

                        # Host authority entry and activation suspend. The
                        # original occurrence may have been revoked meanwhile.
                        execution = get_current_scheduler_execution()
                        if execution is None or execution.id != context.invocation_id:
                            raise runtime.SchedulerExecutionContextUnavailable("scheduled hosted dispatch authority changed")
                    admission.provider_rpc_started = True
                    response = (
                        await client.call_tool("telegram_send", dict(arguments)) if context is None
                        else await client.call_tool("telegram_send", dict(arguments), context=context)
                    )
                    result = runtime._snapshot_host_ingress_payload(response)
                    if type(result) is not dict:
                        raise ValueError("invalid hosted Telegram send response")
                    selected.response = MappingProxyType(runtime._snapshot_host_ingress_payload(result))
            except BaseException as error:
                failure = runtime._redacted_host_ingress_error(error)
                if ((activation_started and not activation_settled)
                    or (admission.provider_rpc_started and selected.response is None)):
                    # An RPC failure may precede completion of a supervisor's
                    # ORIGINAL physical stop. Keep host authority until its
                    # existing completion callback confirms exact retirement.
                    proxy._latch_terminal_lifecycle()
                    try:
                        proxy._admit_terminal_cleanup(retry_retirement=False)
                    except BaseException as opening:  # noqa: BLE001 - retain original canonical closer refusal
                        admission.opening = opening
                    retirement = await await_owned_task(admission.physical_client_retired)
                    if retirement.error is not None and admission.opening is None:
                        admission.opening = retirement.error
                raise
    except BaseException as error:  # noqa: BLE001 - detach foreign callback traceback, preserve native control
        final = runtime._redacted_host_ingress_error(error)
        if failure is None or final.status == runtime._HOST_INGRESS_CONTROL:
            failure = final
    if failure is not None:
        result = None
        if admission is not None and admission.provider_rpc_started:
            failure = runtime._HostIngressOutcome(runtime._HOST_INGRESS_CONTROL, commit_outcome="unknown")
            # This classification comes from Core's actual provider-dispatch
            # boundary, NOT control-shaped attributes on a foreign exception.
            # Carry it through the existing native outcome lane rather than
            # asking the untrusted RPC-error redactor to authenticate it.
            admission.native_error = ExecutionCommitOutcomeError("unknown")
            admission.core_outcome = failure
    # Drop host/child/request-bearing references before detached projection.
    proxy = arguments = context = client = authority = factory = selected = payload = response = activated = admission = None
    if failure is not None:
        return runtime._deliver_host_ingress_outcome(failure, None)
    return result


def validate_outbound_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 300:
        raise ValueError("hosted Telegram outbound timeout must be finite in (0,300]")
    return float(value)
