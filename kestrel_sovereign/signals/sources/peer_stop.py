"""Authenticated peer cooperative Stop as a typed ACTION signal.

Peer Stop deliberately has no direct cancellation door.  Both the signed wire
path and the host-attested same-process path construct this envelope and await
``SignalDispatcher.dispatch_signal`` so causation-cycle, depth, durable replay,
and per-source rate-limit decisions remain load-bearing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
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
from kestrel_sovereign.signals.durable_payload_policy import (
    AlwaysElidedActionSourceRegistration,
)
from kestrel_sovereign.stop import StopDisposition, StopOutcome, StopRequest, StopScope
from kestrel_sovereign.stop.agent_target import (
    agent_stop_identity,
    build_agent_cancellation_authority,
)

SOURCE_NAME = "a2a.peer_stop"
PEER_STOP_RATE_LIMIT_PER_MINUTE = 4
PEER_STOP_RATE_LIMIT_PER_HOUR = 20
PEER_STOP_RATE_LIMIT_BURST = 4
PEER_STOP_COALESCING_WINDOW = timedelta(minutes=5)
PEER_STOP_RETENTION_DAYS = 14
_PEER_STOP_BINDING_KEY_CONTEXT = b"kestrel:a2a.peer_stop:binding-key:v1\x00"
_PEER_STOP_SOURCE_EVENT_CONTEXT = b"kestrel:a2a.peer_stop:source-event:v1\x00"
_DURABLE_SOVEREIGN_PERSISTENCE = "durable_sovereign"
MAX_PEER_STOP_REASON_CHARS = 4096

_INTENT_KEYS = frozenset(
    {"scope", "target", "reason", "cascade", "correlation_id"}
)
_REFUSED_STATUSES = frozenset(
    {Status.DROPPED_CYCLE, Status.DROPPED_RATE_LIMIT, Status.DROPPED_VALIDATION}
)


def _peer_stop_schema(payload: dict) -> dict:
    """Validate only signed intent; identities live on the Signal envelope."""

    if not isinstance(payload, dict):
        raise ValueError("peer Stop payload must be an object")
    if set(payload) != _INTENT_KEYS:
        raise ValueError("peer Stop payload contains missing or forbidden fields")
    try:
        scope = StopScope(payload["scope"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("peer Stop scope is invalid") from error
    if scope is StopScope.HOST:
        raise ValueError("peer Stop cannot target host scope")
    if scope is StopScope.TOOL_CALL:
        raise ValueError(
            "peer Stop cannot target tool_call scope until the agent exposes "
            "a live tool-call cancellation address"
        )

    target = payload["target"]
    if scope is StopScope.AGENT:
        if target is not None:
            raise ValueError("peer agent Stop target comes from signal routing")
    else:
        try:
            validate_invocation_id(target)
        except ValueError as error:
            raise ValueError(
                f"peer {scope.value} Stop requires a valid work target"
            ) from error

    reason = payload["reason"]
    if reason is not None and (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > MAX_PEER_STOP_REASON_CHARS
    ):
        raise ValueError(
            "peer Stop reason must be non-empty and at most "
            f"{MAX_PEER_STOP_REASON_CHARS} characters"
        )
    if reason is not None:
        try:
            reason.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("peer Stop reason must be valid UTF-8") from error
    cascade = payload["cascade"]
    if not isinstance(cascade, bool):
        raise ValueError("peer Stop cascade must be boolean")
    correlation_id = payload["correlation_id"]
    try:
        validate_invocation_id(correlation_id)
    except ValueError as error:
        raise ValueError("peer Stop correlation_id is invalid") from error
    return {
        "scope": scope.value,
        "target": target,
        "reason": reason,
        "cascade": cascade,
        "correlation_id": correlation_id,
    }


def encode_peer_stop_intent(
    *,
    scope: StopScope,
    target: str | None,
    reason: str | None,
    cascade: bool,
    correlation_id: str,
) -> str:
    """Return the canonical signed message for a peer Stop intent."""

    payload = _peer_stop_schema(
        {
            "scope": scope.value if isinstance(scope, StopScope) else scope,
            "target": target,
            "reason": reason,
            "cascade": cascade,
            "correlation_id": correlation_id,
        }
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_peer_stop_intent(message: str) -> dict:
    """Parse and canonicalize one signed peer Stop message."""

    if not isinstance(message, str):
        raise ValueError("peer Stop message must be text")
    try:
        raw = json.loads(message)
    except (TypeError, ValueError) as error:
        raise ValueError("peer Stop message is not valid JSON") from error
    payload = _peer_stop_schema(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if message != canonical:
        raise ValueError("peer Stop message must use canonical encoding")
    return payload


def decode_peer_stop_action_envelope(
    envelope: Mapping[str, Any],
) -> tuple[dict, str, str, dict]:
    """Validate the task-shaped, signature-compatible peer action envelope.

    The existing hybrid A2A signature contract binds ``id``, ``sessionId``,
    the exact message parts, the action verb, and the causation chain.  This
    parser keeps peer Stop on that reviewed authentication format without
    treating it as a persisted A2A task.
    """

    if not isinstance(envelope, Mapping):
        raise ValueError("peer Stop envelope must be an object")
    if envelope.get("artifacts") not in (None, []):
        raise ValueError("peer Stop cannot carry artifacts")
    correlation_id = envelope.get("id")
    session_id = envelope.get("sessionId")
    try:
        validate_invocation_id(correlation_id)
        validate_invocation_id(session_id)
    except ValueError as error:
        raise ValueError("peer Stop envelope requires valid id and sessionId") from error

    message = envelope.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("peer Stop message must be an object")
    parts = message.get("parts")
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], Mapping)
        or parts[0].get("type") != "text"
        or not isinstance(parts[0].get("text"), str)
    ):
        raise ValueError("peer Stop requires exactly one text message part")
    message_text = parts[0]["text"]
    intent = decode_peer_stop_intent(message_text)
    if intent["correlation_id"] != correlation_id:
        raise ValueError("peer Stop correlation does not match envelope id")

    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("peer Stop metadata must be an object")
    if metadata.get("a2a_verb") != "peer_stop":
        raise ValueError("peer Stop envelope must bind a2a_verb=peer_stop")
    peer_stop_audience(metadata)
    return intent, correlation_id, session_id, metadata


def peer_stop_audience(metadata: Mapping[str, Any]) -> str:
    """Return the required action audience without treating it as authority."""

    audience = metadata.get(A2A_AUDIENCE_METADATA_KEY)
    if not isinstance(audience, str) or not audience.strip():
        raise ValueError("peer Stop envelope requires a signed audience")
    try:
        audience.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("peer Stop audience is not valid UTF-8") from error
    return audience


def peer_stop_source_event_id(actor_id: str, correlation_id: str) -> str:
    """Restart-stably bind replay identity to actor and signed request id.

    Durable deployments already pin ``KESTREL_DATA_KEY`` as custody material.
    A purpose-separated MAC derived from that key keeps peer identities opaque
    to a database reader without coupling idempotency to the independently
    rotatable A2A transport credential.  Keyless local development retains its
    existing project-persisted transport-key fallback; a runtime that declares
    durable-sovereign persistence must never take that fallback.
    """

    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ValueError("peer Stop actor must be authenticated")
    try:
        validate_invocation_id(correlation_id)
    except ValueError as error:
        raise ValueError("peer Stop correlation_id is invalid") from error
    material = json.dumps(
        [actor_id, correlation_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    from kestrel_sovereign.security.encryption import (
        MasterKeyNotConfiguredError,
        get_master_key_bytes,
    )

    missing_key_error: Exception | None = None
    try:
        master_key = get_master_key_bytes()
    except MasterKeyNotConfiguredError as error:
        master_key = None
        missing_key_error = error
    if not isinstance(master_key, (bytes, bytearray)) or not master_key:
        import os

        persistence_mode = os.environ.get(
            "KESTREL_DEPLOYMENT_PERSISTENCE", ""
        ).strip().lower()
        if persistence_mode == _DURABLE_SOVEREIGN_PERSISTENCE:
            raise RuntimeError(
                "Durable peer Stop replay binding requires KESTREL_DATA_KEY"
            ) from missing_key_error
        from kestrel_sovereign.a2a.transport_auth import ensure_a2a_transport_key

        binding_key = ensure_a2a_transport_key().encode("utf-8")
    else:
        binding_key = hmac.new(
            master_key,
            _PEER_STOP_BINDING_KEY_CONTEXT,
            hashlib.sha256,
        ).digest()
    return hmac.new(
        binding_key,
        _PEER_STOP_SOURCE_EVENT_CONTEXT + material,
        hashlib.sha256,
    ).hexdigest()


def build_peer_stop_signal(
    *,
    agent: Any,
    actor_id: str,
    intent: Mapping[str, Any],
    causation_chain: Sequence[CausationFrame] = (),
) -> Signal:
    """Build an envelope from already-authenticated routing principals."""

    target_agent_id = agent_stop_identity(agent)
    payload = _peer_stop_schema(dict(intent))
    event_id = peer_stop_source_event_id(actor_id, payload["correlation_id"])
    return Signal(
        source=SOURCE_NAME,
        kind=payload["scope"],
        mode=SignalMode.ACTION,
        payload=payload,
        target_agent=target_agent_id,
        caller=actor_id,
        visibility=Visibility.INTERNAL,
        urgency=Urgency.HIGH,
        dedupe_key=event_id,
        causation_chain=list(causation_chain),
    )


def build_peer_stop_registration(
    agent: Any,
) -> AlwaysElidedActionSourceRegistration:
    """Register peer Stop as a rate-limited, non-looping ACTION."""

    async def handle_peer_stop(payload: dict) -> list[dict[str, Any]]:
        signal = get_current_signal()
        target_agent_id = agent_stop_identity(agent)
        if signal is None or signal.source != SOURCE_NAME:
            raise ValueError("peer Stop requires its dispatcher signal principal")
        if signal.mode is not SignalMode.ACTION:
            raise ValueError("peer Stop requires an ACTION signal")
        if signal.target_agent != target_agent_id:
            raise ValueError("peer Stop signal target no longer matches this agent")
        actor_id = signal.caller
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("peer Stop signal has no authenticated caller principal")

        normalized = _peer_stop_schema(payload)
        scope = StopScope(normalized["scope"])
        request = StopRequest(
            scope=scope,
            actor_id=actor_id,
            target=(
                target_agent_id
                if scope is StopScope.AGENT
                else normalized["target"]
            ),
            target_agent_id=(
                target_agent_id
                if scope in {StopScope.TURN, StopScope.TOOL_CALL}
                else None
            ),
            reason=normalized["reason"],
            cascade=normalized["cascade"],
            correlation_id=normalized["correlation_id"],
            target_is_turn_id=scope is StopScope.TURN,
        )
        outcomes = await build_agent_cancellation_authority(agent).stop(request)
        return [outcome.to_dict() for outcome in outcomes]

    return AlwaysElidedActionSourceRegistration(
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
        coalescing_window=PEER_STOP_COALESCING_WINDOW,
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


def signal_result_to_peer_stop_response(
    result: SignalResult,
    *,
    target_agent_id: str,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    """Map every dispatcher terminal state to an honest typed Stop result."""

    payload = _peer_stop_schema(dict(intent))
    status = result.status
    signal_receipt = {
        "signal_id": result.signal_id,
        "status": status.value,
        "detail": _public_dispatcher_outcome_detail(status),
    }
    if status is Status.OK:
        raw_outcomes = result.action_result
        if not isinstance(raw_outcomes, list):
            status = Status.FAILED
            signal_receipt["status"] = status.value
            signal_receipt["detail"] = "peer Stop handler returned no typed outcomes"
        else:
            outcomes = [StopOutcome.from_dict(item) for item in raw_outcomes]
            expected_scope = StopScope(payload["scope"])
            expected_requested_target = (
                target_agent_id
                if expected_scope is StopScope.AGENT
                else payload["target"]
            )
            if len(outcomes) != 1 or any(
                outcome.agent_id != target_agent_id
                or outcome.correlation_id != payload["correlation_id"]
                or outcome.scope is not expected_scope
                or outcome.requested_target != expected_requested_target
                for outcome in outcomes
            ):
                raise ValueError("peer Stop handler returned mismatched principals")
            public_outcomes = [
                StopOutcome(
                    scope=outcome.scope,
                    requested_target=outcome.requested_target,
                    # A turn Stop resolves its public address to a private
                    # request-generation key.  The peer-facing receipt names
                    # only the authenticated target agent, never that key.
                    resolved_target=target_agent_id,
                    agent_id=outcome.agent_id,
                    disposition=outcome.disposition,
                    correlation_id=outcome.correlation_id,
                    detail=outcome.detail,
                )
                for outcome in outcomes
            ]
            return {
                "signal_receipt": signal_receipt,
                "stop_outcomes": [
                    outcome.to_dict() for outcome in public_outcomes
                ],
            }

    if status in _REFUSED_STATUSES:
        disposition = StopDisposition.REFUSED
    elif status is Status.COALESCED:
        # Coalescing proves only that this delivery was suppressed.  The
        # dispatcher does not durably retain the original handler outcome, so
        # claiming cancellation completed would be false when the first
        # delivery was rate-limited or failed before invoking the handler.
        disposition = StopDisposition.UNREACHABLE
    else:
        disposition = StopDisposition.UNREACHABLE
    scope = StopScope(payload["scope"])
    requested_target = (
        target_agent_id if scope is StopScope.AGENT else payload["target"]
    )
    outcome = StopOutcome(
        scope=scope,
        requested_target=requested_target,
        resolved_target=target_agent_id,
        agent_id=target_agent_id,
        disposition=disposition,
        correlation_id=payload["correlation_id"],
        detail=_public_dispatcher_outcome_detail(status),
    )
    return {
        "signal_receipt": signal_receipt,
        "stop_outcomes": [outcome.to_dict()],
    }


def _public_dispatcher_outcome_detail(status: Status) -> str | None:
    """Describe a terminal state without exposing dispatcher internals."""

    if status is Status.OK:
        return None
    if status is Status.COALESCED:
        return (
            "Duplicate peer Stop was suppressed; the original outcome is "
            "unavailable and completion is not claimed"
        )
    if status is Status.DROPPED_CYCLE:
        return "Peer Stop was refused by causation cycle or depth policy"
    if status is Status.DROPPED_RATE_LIMIT:
        return "Peer Stop was refused by peer rate limits"
    if status is Status.DROPPED_VALIDATION:
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
    "PEER_STOP_COALESCING_WINDOW",
    "PEER_STOP_RATE_LIMIT_PER_HOUR",
    "PEER_STOP_RATE_LIMIT_PER_MINUTE",
    "PEER_STOP_RATE_LIMIT_BURST",
    "SOURCE_NAME",
    "build_peer_stop_registration",
    "build_peer_stop_signal",
    "decode_peer_stop_intent",
    "decode_peer_stop_action_envelope",
    "encode_peer_stop_intent",
    "peer_stop_audience",
    "peer_stop_source_event_id",
    "signal_result_to_peer_stop_response",
]
