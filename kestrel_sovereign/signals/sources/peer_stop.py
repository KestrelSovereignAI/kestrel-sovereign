"""Authenticated peer cooperative Stop as a typed ACTION signal (#3169).

Peer Stop deliberately has no direct cancellation door.  Both the signed wire
path (``POST /api/agent/peer/stop``) and the host-attested same-process path
(``AgentManager.stop_host_attested_local_peer``) call
:func:`dispatch_peer_stop`, which builds an ``a2a.peer_stop`` signal and awaits
``SignalDispatcher.dispatch_signal``.  Causation-cycle detection, depth TTL,
durable source-event deduplication, and per-source rate limiting therefore
decide every peer Stop before the handler below can cancel anything.

Identities are routing principals, never payload:

* the **actor** is ``Signal.caller`` — the verified envelope sender (wire) or
  the manager-attested sender (local);
* the **target** is ``Signal.target_agent`` — the recipient's own DID.

The payload carries only the signed intent (scope, work target, reason,
cascade, correlation id).  Its schema forbids any other key.

Every peer Stop leaves a durable Stop receipt whose ``actor_id`` is the
verified sender.  A dispatcher refusal (cycle/depth, rate limit, validation)
is recorded as a ``refused`` receipt naming the refusal, not dropped silently.

A peer never cascades (#3143): following signed descendants is the
sovereign's authority.  A ``cascade: true`` intent is refused by validation
with a receipt, never silently downgraded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from kestrel_sdk.signals import (
    AttentionPolicy,
    CausationFrame,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SignalResult,
    Status,
    Trust,
    Urgency,
    Visibility,
)

from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign.a2a.envelope_signing import A2A_AUDIENCE_METADATA_KEY
from kestrel_sovereign.signals.context import get_current_signal
from kestrel_sovereign.signals.in_flight_control import (
    InFlightControlActionRegistration,
)
from kestrel_sovereign.stop import (
    CancellationAuthority,
    StopCleanupRegistry,
    StopDisposition,
    StopOutcome,
    StopReceiptConflict,
    StopRequest,
    StopScope,
    UnavailableStopReceiptStore,
)
from kestrel_sovereign.stop.receipt import StopReceipt
from kestrel_sovereign.stop.runtime_target import (
    build_runtime_stop_target,
    resolve_runtime_stop_identity,
)

SOURCE_NAME = "a2a.peer_stop"
PEER_STOP_A2A_VERB = "peer_stop"
PEER_STOP_RATE_LIMIT_PER_MINUTE = 4
PEER_STOP_RATE_LIMIT_PER_HOUR = 20
PEER_STOP_RATE_LIMIT_BURST = 4
PEER_STOP_RETENTION_DAYS = 14
# Same bound as the operator ``POST /api/agent/stop`` door.
MAX_PEER_STOP_REASON_CHARS = 1024
# Deliveries of one peer Stop whose response was lost.  Each attempt re-signs
# the same intent and correlation id with a fresh replay nonce; the recipient's
# durable source event answers a duplicate with the original receipt.
PEER_STOP_DELIVERY_ATTEMPTS = 3

_OPERATION_ID_DOMAIN = b"kestrel:a2a.peer_stop:operation:v1\x00"
_OPERATION_ID_PREFIX = "peer-stop:"
_INTENT_KEYS = frozenset({"scope", "target", "reason", "cascade", "correlation_id"})
_REFUSED_STATUSES = frozenset(
    {Status.DROPPED_CYCLE, Status.DROPPED_RATE_LIMIT, Status.DROPPED_VALIDATION}
)
_RECEIPT_STORE_ATTRIBUTE = "_stop_receipt_store"
_CLEANUP_REGISTRY_ATTRIBUTE = "_stop_cleanup_registry"


class PeerStopIntentError(ValueError):
    """The peer Stop intent is not grammatical; no signal can be built."""


# ---------------------------------------------------------------------------
# Intent grammar and policy
# ---------------------------------------------------------------------------


def parse_peer_stop_intent(payload: object) -> dict[str, Any]:
    """Validate the intent's *shape* only and return its canonical form.

    Grammar failures cannot name a Stop request, so they are rejected before a
    signal exists.  Policy (host/tool_call scope, cascade) is a separate,
    receipted refusal decided by the dispatcher — see
    :func:`peer_stop_policy_refusal`.
    """

    if not isinstance(payload, Mapping):
        raise PeerStopIntentError("peer Stop payload must be an object")
    if set(payload) != _INTENT_KEYS:
        raise PeerStopIntentError(
            "peer Stop payload contains missing or forbidden fields"
        )
    try:
        scope = StopScope(payload["scope"])
    except (TypeError, ValueError) as error:
        raise PeerStopIntentError("peer Stop scope is invalid") from error

    target = payload["target"]
    if scope in {StopScope.AGENT, StopScope.HOST}:
        if target is not None:
            raise PeerStopIntentError(
                f"peer {scope.value} Stop target comes from signal routing"
            )
    else:
        try:
            validate_invocation_id(target)
        except ValueError as error:
            raise PeerStopIntentError(
                f"peer {scope.value} Stop requires a valid work target"
            ) from error

    reason = payload["reason"]
    if reason is not None:
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > MAX_PEER_STOP_REASON_CHARS
        ):
            raise PeerStopIntentError(
                "peer Stop reason must be non-empty and at most "
                f"{MAX_PEER_STOP_REASON_CHARS} characters"
            )
        try:
            reason.encode("utf-8")
        except UnicodeEncodeError as error:
            raise PeerStopIntentError("peer Stop reason must be valid UTF-8") from error

    cascade = payload["cascade"]
    if not isinstance(cascade, bool):
        raise PeerStopIntentError("peer Stop cascade must be boolean")
    correlation_id = payload["correlation_id"]
    try:
        validate_invocation_id(correlation_id)
    except ValueError as error:
        raise PeerStopIntentError("peer Stop correlation_id is invalid") from error
    return {
        "scope": scope.value,
        "target": target,
        "reason": reason,
        "cascade": cascade,
        "correlation_id": correlation_id,
    }


def peer_stop_policy_refusal(intent: Mapping[str, Any]) -> str | None:
    """Name why a grammatical intent is outside peer authority, or ``None``."""

    scope = StopScope(intent["scope"])
    if scope is StopScope.HOST:
        return "Peer Stop cannot target host scope"
    if scope is StopScope.TOOL_CALL:
        return (
            "Peer Stop cannot target tool_call scope until the agent exposes "
            "a live tool-call cancellation address"
        )
    if intent["cascade"]:
        return (
            "Peer Stop cannot cascade; following signed descendants is "
            "reserved to the sovereign"
        )
    return None


def _peer_stop_schema(payload: dict) -> dict:
    """The registered dispatcher schema: grammar, then peer policy."""

    intent = parse_peer_stop_intent(payload)
    refusal = peer_stop_policy_refusal(intent)
    if refusal is not None:
        raise ValueError(refusal)
    return intent


def encode_peer_stop_intent(
    *,
    scope: StopScope,
    target: str | None,
    reason: str | None,
    cascade: bool,
    correlation_id: str,
) -> str:
    """Return the canonical signed message text for a peer Stop intent."""

    intent = parse_peer_stop_intent(
        {
            "scope": scope.value if isinstance(scope, StopScope) else scope,
            "target": target,
            "reason": reason,
            "cascade": cascade,
            "correlation_id": correlation_id,
        }
    )
    return json.dumps(intent, sort_keys=True, separators=(",", ":"))


def decode_peer_stop_intent(message: object) -> dict[str, Any]:
    """Parse one signed peer Stop message; only canonical encoding is valid."""

    if not isinstance(message, str):
        raise PeerStopIntentError("peer Stop message must be text")
    try:
        raw = json.loads(message)
    except (TypeError, ValueError) as error:
        raise PeerStopIntentError("peer Stop message is not valid JSON") from error
    intent = parse_peer_stop_intent(raw)
    if message != json.dumps(intent, sort_keys=True, separators=(",", ":")):
        raise PeerStopIntentError("peer Stop message must use canonical encoding")
    return intent


def decode_peer_stop_action_envelope(
    envelope: object,
) -> tuple[dict[str, Any], str, str, dict]:
    """Validate the task-shaped peer Stop action envelope.

    The hybrid A2A signature binds ``id``, ``sessionId``, the exact message
    text, the ``a2a_verb``, the audience, and the causation chain.  This parser
    keeps peer Stop on that reviewed authentication format without treating it
    as a persisted A2A task.
    """

    if not isinstance(envelope, Mapping):
        raise PeerStopIntentError("peer Stop envelope must be an object")
    if envelope.get("artifacts") not in (None, []):
        raise PeerStopIntentError("peer Stop cannot carry artifacts")
    correlation_id = envelope.get("id")
    session_id = envelope.get("sessionId")
    try:
        validate_invocation_id(correlation_id)
        validate_invocation_id(session_id)
    except ValueError as error:
        raise PeerStopIntentError(
            "peer Stop envelope requires valid id and sessionId"
        ) from error

    message = envelope.get("message")
    if not isinstance(message, Mapping):
        raise PeerStopIntentError("peer Stop message must be an object")
    parts = message.get("parts")
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], Mapping)
        or parts[0].get("type") != "text"
        or not isinstance(parts[0].get("text"), str)
    ):
        raise PeerStopIntentError("peer Stop requires exactly one text message part")
    intent = decode_peer_stop_intent(parts[0]["text"])
    if intent["correlation_id"] != correlation_id:
        raise PeerStopIntentError("peer Stop correlation does not match envelope id")

    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise PeerStopIntentError("peer Stop metadata must be an object")
    if metadata.get("a2a_verb") != PEER_STOP_A2A_VERB:
        raise PeerStopIntentError(
            f"peer Stop envelope must bind a2a_verb={PEER_STOP_A2A_VERB}"
        )
    peer_stop_audience(metadata)
    return intent, correlation_id, session_id, metadata


def peer_stop_audience(metadata: Mapping[str, Any]) -> str:
    """Return the required signed audience; it is checked, never trusted."""

    audience = metadata.get(A2A_AUDIENCE_METADATA_KEY)
    if not isinstance(audience, str) or not audience.strip():
        raise PeerStopIntentError("peer Stop envelope requires a signed audience")
    return audience


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def peer_stop_target_identity(agent: object) -> str:
    """The recipient's own stable Stop address, or fail closed."""

    for attribute in ("did", "agent_id"):
        candidate = getattr(agent, attribute, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise ValueError("peer Stop requires a stable target identity")


def peer_stop_operation_id(actor_id: str, correlation_id: str) -> str:
    """Actor-scoped, restart-stable idempotency key for one peer Stop.

    It is both the durable ``source_event_id`` and the receipt correlation id,
    so two peers choosing the same correlation id never collide and a replay
    of one peer's request resolves to that peer's original receipt.
    """

    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("peer Stop actor must be authenticated")
    validate_invocation_id(correlation_id)
    material = json.dumps(
        [actor_id, correlation_id], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(_OPERATION_ID_DOMAIN + material).hexdigest()
    return f"{_OPERATION_ID_PREFIX}{digest}"


def peer_stop_request(
    *,
    target_agent_id: str,
    actor_id: str,
    intent: Mapping[str, Any],
    trace_id: str | None = None,
    span_id: str | None = None,
) -> StopRequest:
    """The one Stop request a peer intent names, for effects and receipts."""

    scope = StopScope(intent["scope"])
    target: str | None
    if scope is StopScope.HOST:
        target = None
    elif scope is StopScope.AGENT:
        target = target_agent_id
    else:
        target = intent["target"]
    return StopRequest(
        scope=scope,
        actor_id=actor_id,
        target=target,
        target_agent_id=None if scope is StopScope.HOST else target_agent_id,
        reason=intent["reason"],
        cascade=intent["cascade"],
        correlation_id=peer_stop_operation_id(actor_id, intent["correlation_id"]),
        target_is_turn_id=scope is StopScope.TURN,
        trace_id=trace_id,
        span_id=span_id,
    )


def build_peer_stop_signal(
    *,
    agent: object,
    actor_id: str,
    intent: Mapping[str, Any],
    causation_chain: Sequence[CausationFrame] = (),
) -> Signal:
    """Build the envelope from already-authenticated routing principals."""

    payload = parse_peer_stop_intent(intent)
    return Signal(
        source=SOURCE_NAME,
        kind=payload["scope"],
        mode=SignalMode.ACTION,
        payload=payload,
        target_agent=peer_stop_target_identity(agent),
        caller=actor_id,
        visibility=Visibility.INTERNAL,
        urgency=Urgency.HIGH,
        causation_chain=list(causation_chain),
    )


# ---------------------------------------------------------------------------
# Host services
# ---------------------------------------------------------------------------


def attach_stop_evidence(
    agent: object,
    *,
    receipt_store: object,
    cleanup_registry: StopCleanupRegistry,
) -> None:
    """Give an agent's peer Stop handler the host's Stop evidence services."""

    if not isinstance(cleanup_registry, StopCleanupRegistry):
        raise TypeError("cleanup_registry must be a StopCleanupRegistry")
    agent.__dict__[_RECEIPT_STORE_ATTRIBUTE] = receipt_store
    agent.__dict__[_CLEANUP_REGISTRY_ATTRIBUTE] = cleanup_registry


def _stop_evidence(agent: object) -> tuple[object, StopCleanupRegistry]:
    namespace = getattr(agent, "__dict__", {})
    receipt_store = namespace.get(_RECEIPT_STORE_ATTRIBUTE)
    if receipt_store is None:
        receipt_store = UnavailableStopReceiptStore(
            "Stop receipt storage is not attached to this agent"
        )
    registry = namespace.get(_CLEANUP_REGISTRY_ATTRIBUTE)
    if not isinstance(registry, StopCleanupRegistry):
        # Without attached evidence no effect can run (receipt preflight
        # refuses first), so a private registry has no tail to drain.
        registry = StopCleanupRegistry()
    return receipt_store, registry


def peer_stop_breaker_refusal(agent: object, request: StopRequest) -> str | None:
    """Decide whether this peer Stop is honored; ``None`` honors it.

    This is the single seam the fleet circuit breaker (#3170) owns: peer
    Stops past its threshold cease to be honored and a human is notified,
    because repeated Stop would otherwise synthesize Hold.  Until #3170
    lands every peer Stop that survived dispatcher policy is honored.
    """

    return None


# ---------------------------------------------------------------------------
# Registration and handler
# ---------------------------------------------------------------------------


def build_peer_stop_registration(
    agent: object,
) -> InFlightControlActionRegistration:
    """Register peer Stop as a rate-limited, non-looping in-flight ACTION."""

    async def handle_peer_stop(payload: dict) -> list[dict[str, Any]]:
        signal = get_current_signal()
        target_agent_id = peer_stop_target_identity(agent)
        if signal is None or signal.source != SOURCE_NAME:
            raise ValueError("peer Stop requires its dispatcher signal principal")
        if signal.mode is not SignalMode.ACTION:
            raise ValueError("peer Stop requires an ACTION signal")
        if signal.target_agent != target_agent_id:
            raise ValueError("peer Stop signal target no longer matches this agent")
        actor_id = signal.caller
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("peer Stop signal has no authenticated caller principal")

        intent = _peer_stop_schema(payload)
        scope = StopScope(intent["scope"])
        turn_id = intent["target"] if scope is StopScope.TURN else None
        _canonical_turn, trace_id, span_id = resolve_runtime_stop_identity(
            agent, explicit_turn_id=turn_id
        )
        request = peer_stop_request(
            target_agent_id=target_agent_id,
            actor_id=actor_id,
            intent=intent,
            trace_id=trace_id,
            span_id=span_id,
        )
        receipt_store, cleanup_registry = _stop_evidence(agent)
        breaker_refusal = peer_stop_breaker_refusal(agent, request)
        if breaker_refusal is not None:
            outcomes = await _persist_single_outcome(
                receipt_store,
                request,
                target_agent_id=target_agent_id,
                disposition=StopDisposition.REFUSED,
                detail=breaker_refusal,
            )
            return [outcome.to_dict() for outcome in outcomes]

        distributed_registry = getattr(agent, "__dict__", {}).get(
            "_distributed_invocation_registry"
        )

        def inventory():
            return (
                build_runtime_stop_target(
                    agent,
                    agent_id=target_agent_id,
                    explicit_turn_id=turn_id,
                    distributed_registry=distributed_registry,
                    resolve_turn_addresses=scope is StopScope.TURN,
                ),
            )

        authority = CancellationAuthority(
            inventory,
            cleanup_registry=cleanup_registry,
            receipt_store=receipt_store,
        )
        outcomes = await authority.stop(request)
        return [outcome.to_dict() for outcome in outcomes]

    return InFlightControlActionRegistration(
        name=SOURCE_NAME,
        schema=_peer_stop_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handle_peer_stop,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(
            per_minute=PEER_STOP_RATE_LIMIT_PER_MINUTE,
            per_hour=PEER_STOP_RATE_LIMIT_PER_HOUR,
            burst=PEER_STOP_RATE_LIMIT_BURST,
        ),
        attention_policy=AttentionPolicy(),
        resources=frozenset(),
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_peer_stop_redaction,
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
        retention_days=PEER_STOP_RETENTION_DAYS,
    )


# ---------------------------------------------------------------------------
# The one door function
# ---------------------------------------------------------------------------


async def dispatch_peer_stop(
    agent: object,
    *,
    actor_id: str,
    intent: Mapping[str, Any],
    causation_chain: Sequence[CausationFrame] = (),
) -> dict[str, Any]:
    """Dispatch one authenticated peer Stop and return its receipted result.

    Callers must already have authenticated ``actor_id`` and checked the
    signed audience against this recipient.  This function never cancels
    anything itself: effects happen only inside the dispatcher-invoked
    handler, and dispatcher refusals are recorded as ``refused`` receipts.
    """

    dispatcher = getattr(agent, "dispatcher", None)
    if dispatcher is None or not callable(getattr(dispatcher, "dispatch_signal", None)):
        raise RuntimeError("Peer Stop signal dispatcher is unavailable")
    parsed = parse_peer_stop_intent(intent)
    signal = build_peer_stop_signal(
        agent=agent,
        actor_id=actor_id,
        intent=parsed,
        causation_chain=causation_chain,
    )
    request = peer_stop_request(
        target_agent_id=signal.target_agent,
        actor_id=actor_id,
        intent=parsed,
    )
    result = await dispatcher.dispatch_signal(
        signal,
        source_event_id=request.correlation_id,
    )
    outcomes = await _outcomes_for_result(agent, result, request, parsed)
    return {
        "correlation_id": parsed["correlation_id"],
        "stop_correlation_id": request.correlation_id,
        "signal_receipt": {
            "signal_id": result.signal_id,
            "status": result.status.value,
            "detail": _dispatcher_outcome_detail(result.status, parsed),
        },
        "stop_outcomes": [outcome.to_dict() for outcome in outcomes],
    }


async def _outcomes_for_result(
    agent: object,
    result: SignalResult,
    request: StopRequest,
    intent: Mapping[str, Any],
) -> tuple[StopOutcome, ...]:
    """Map every dispatcher terminal state to honest, receipted outcomes."""

    target_agent_id = request.target_agent_id or peer_stop_target_identity(agent)
    receipt_store, _registry = _stop_evidence(agent)
    status = result.status

    if status is Status.OK:
        raw = result.action_result
        if not isinstance(raw, list) or not raw:
            return await _persist_single_outcome(
                receipt_store,
                request,
                target_agent_id=target_agent_id,
                disposition=StopDisposition.UNREACHABLE,
                detail="Peer Stop handler returned no typed outcomes",
            )
        outcomes = tuple(StopOutcome.from_dict(item) for item in raw)
        if any(
            outcome.agent_id != target_agent_id
            or outcome.correlation_id != request.correlation_id
            or outcome.scope is not request.scope
            or outcome.requested_target != request.target
            for outcome in outcomes
        ):
            raise ValueError("peer Stop handler returned mismatched principals")
        return outcomes

    if status is Status.COALESCED:
        # A replay of an accepted source event. Answer with the original
        # receipt; never execute the Stop a second time.
        try:
            replay = await receipt_store.load(request)
        except StopReceiptConflict:
            return (
                _outcome(
                    request,
                    target_agent_id,
                    StopDisposition.REFUSED,
                    "Peer Stop correlation id was reused for a different request",
                ),
            )
        except Exception:  # noqa: BLE001 - durable evidence boundary
            return (
                _outcome(
                    request,
                    target_agent_id,
                    StopDisposition.UNREACHABLE,
                    "Stop receipt storage is unavailable; the original "
                    "outcome could not be read",
                ),
            )
        if isinstance(replay, StopReceipt):
            return replay.outcomes
        return (
            _outcome(
                request,
                target_agent_id,
                StopDisposition.UNREACHABLE,
                "A duplicate of this peer Stop is still in progress; its "
                "outcome is not yet recorded",
            ),
        )

    disposition = (
        StopDisposition.REFUSED
        if status in _REFUSED_STATUSES
        else StopDisposition.UNREACHABLE
    )
    return await _persist_single_outcome(
        receipt_store,
        request,
        target_agent_id=target_agent_id,
        disposition=disposition,
        detail=_dispatcher_outcome_detail(status, intent),
    )


def _outcome(
    request: StopRequest,
    target_agent_id: str,
    disposition: StopDisposition,
    detail: str | None,
) -> StopOutcome:
    return StopOutcome(
        scope=request.scope,
        requested_target=request.target,
        resolved_target=target_agent_id,
        agent_id=target_agent_id,
        disposition=disposition,
        correlation_id=request.correlation_id,
        detail=detail,
    )


async def _persist_single_outcome(
    receipt_store: Any,
    request: StopRequest,
    *,
    target_agent_id: str,
    disposition: StopDisposition,
    detail: str | None,
) -> tuple[StopOutcome, ...]:
    """Record one non-effect outcome; an existing receipt wins as a replay."""

    outcome = _outcome(request, target_agent_id, disposition, detail)
    try:
        receipt = await receipt_store.persist(request, (outcome,))
    except StopReceiptConflict:
        return (
            _outcome(
                request,
                target_agent_id,
                StopDisposition.REFUSED,
                "Peer Stop correlation id conflicts with durable Stop evidence",
            ),
        )
    except Exception:  # noqa: BLE001 - durable evidence boundary
        return (
            _outcome(
                request,
                target_agent_id,
                disposition,
                f"{detail}; its Stop receipt could not be persisted"
                if detail
                else "Peer Stop receipt could not be persisted",
            ),
        )
    if not isinstance(receipt, StopReceipt):
        raise TypeError("Stop receipt storage returned invalid evidence")
    return receipt.outcomes


def _dispatcher_outcome_detail(
    status: Status,
    intent: Mapping[str, Any],
) -> str | None:
    """Name a dispatcher outcome without exposing dispatcher internals."""

    if status is Status.OK:
        return None
    if status is Status.COALESCED:
        return "Duplicate peer Stop; answered from the original request's receipt"
    if status is Status.DROPPED_CYCLE:
        return "Peer Stop was refused by causation cycle or depth policy"
    if status is Status.DROPPED_RATE_LIMIT:
        return "Peer Stop was refused by peer rate limits"
    if status is Status.DROPPED_VALIDATION:
        policy = peer_stop_policy_refusal(intent)
        if policy is not None:
            return f"Peer Stop was refused by signal validation: {policy}"
        return "Peer Stop was refused by signal validation"
    if status is Status.FAILED:
        return "Peer Stop signal failed before cancellation could be confirmed"
    return f"Peer Stop signal ended as {status.value}"


def _peer_stop_redaction(payload: dict) -> str:
    return (
        f"scope={payload.get('scope', '?')} "
        f"target_present={payload.get('target') is not None} "
        f"cascade={payload.get('cascade', False)!r} "
        f"correlation_present={isinstance(payload.get('correlation_id'), str)}"
    )


__all__ = [
    "MAX_PEER_STOP_REASON_CHARS",
    "PEER_STOP_A2A_VERB",
    "PEER_STOP_DELIVERY_ATTEMPTS",
    "PEER_STOP_RATE_LIMIT_BURST",
    "PEER_STOP_RATE_LIMIT_PER_HOUR",
    "PEER_STOP_RATE_LIMIT_PER_MINUTE",
    "PeerStopIntentError",
    "SOURCE_NAME",
    "attach_stop_evidence",
    "build_peer_stop_registration",
    "build_peer_stop_signal",
    "decode_peer_stop_action_envelope",
    "decode_peer_stop_intent",
    "dispatch_peer_stop",
    "encode_peer_stop_intent",
    "parse_peer_stop_intent",
    "peer_stop_audience",
    "peer_stop_breaker_refusal",
    "peer_stop_operation_id",
    "peer_stop_policy_refusal",
    "peer_stop_request",
    "peer_stop_target_identity",
]
