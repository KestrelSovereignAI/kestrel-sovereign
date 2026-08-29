"""Shared helpers for endpoint modules."""

from typing import Mapping

from fastapi import HTTPException, Request

from kestrel_sovereign.agent.invocation import (
    InvocationProvenance,
    invocation_id_response_header,
    request_provenance,
    resolve_transport_invocation_id,
)
from kestrel_sovereign.api_errors import ApiHTTPException


def get_caller(request: Request):
    """Return the CallerContext attached by the auth middleware, or None.

    Used by endpoints that hand off to ``agent.process_input`` /
    ``agent.process_input_streaming`` so that governance-command
    authorization (e.g. ``!safe-mode exit``, ``!reanchor-constitution``)
    is evaluated consistently regardless of which endpoint the caller
    reached.  The middleware that populates ``request.state.caller``
    lives in :mod:`server`.

    Returns ``None`` only when the middleware was bypassed — e.g. for
    webhooks or internal calls — which the command handler itself
    rejects as non-sovereign.
    """
    return getattr(request.state, "caller", None)


def resolve_request_invocation_id(
    request: Request,
    body: Mapping[str, object] | object | None,
) -> str:
    """Resolve the shared retry id for every HTTP turn-producing route."""
    try:
        return resolve_transport_invocation_id(
            body,
            request.headers.get("X-Request-ID"),
        )
    except ValueError as error:
        raise ApiHTTPException(
            status_code=400,
            code="invalid_request_id",
            message=(
                "request_id must be a non-empty valid Unicode string no "
                f"longer than 256 characters: {error}"
            ),
        ) from error


def request_invocation_provenance(
    request: Request,
    *,
    source_locator: str,
    fallback_actor: str | None = None,
) -> InvocationProvenance:
    """Bind authenticated actor and transport provenance for one HTTP turn.

    Payload fields such as a bridge ``sender_id`` are intentionally excluded:
    they are gateway content, not an authenticated authority claim. A missing
    authenticated identity remains ``None`` unless an endpoint supplies a
    fixed, trusted service principal for its own authentication lane.
    """
    caller = get_caller(request)
    actor = getattr(caller, "identity", None)
    if not isinstance(actor, str) or not actor:
        actor = fallback_actor
    return request_provenance(
        actor=actor,
        source_kind="http_request",
        source_locator=source_locator,
    )


async def prime_durable_stop_fence(
    request: Request,
    agent: object,
    invocation_id: str,
) -> bool:
    """Reinstall a durable exact-turn Stop before lifecycle registration.

    The in-memory reservation closes the live race between Stop and request
    registration. This lookup closes the longer restart/delivery-delay window:
    an acknowledged durable Stop remains authoritative even after that short
    reservation ages out. Stores without this optional query are retained for
    compatibility in tests and embedded deployments that do not expose Stop.
    """

    store = getattr(request.app.state, "stop_receipt_store", None)
    lookup = getattr(store, "has_acknowledged_turn_stop", None)
    if not callable(lookup):
        return False
    agent_id = getattr(agent, "agent_id", None)
    if not isinstance(agent_id, str) or not agent_id.strip():
        agent_id = "local-agent"
    try:
        stopped = await lookup(agent_id, invocation_id)
    except Exception as error:
        raise ApiHTTPException(
            status_code=503,
            code="stop_evidence_unavailable",
            message=(
                "Durable Stop evidence could not be checked before execution."
            ),
        ) from error
    if not isinstance(stopped, bool):
        raise ApiHTTPException(
            status_code=503,
            code="stop_evidence_unavailable",
            message="Durable Stop evidence returned an invalid result.",
        )
    if not stopped:
        return False
    reserve = getattr(type(agent), "reserve_request_cancellation", None)
    if not callable(reserve):
        raise ApiHTTPException(
            status_code=503,
            code="stop_fence_unavailable",
            message="The stopped request cannot be fenced safely.",
        )
    reserve(agent, invocation_id)
    return True


def stopped_invocation_http_error(invocation_id: str) -> ApiHTTPException:
    """Translate cooperative turn cancellation at a synchronous HTTP door."""

    return ApiHTTPException(
        status_code=409,
        code="request_stopped",
        message="Request stopped during execution.",
        headers={"X-Request-ID": invocation_id_response_header(invocation_id)},
    )


def get_agent(request: Request):
    """Get the active KestrelAgent for this request.

    In multi-agent mode, the routing middleware sets request.state.agent
    based on the /api/agents/{name}/... path prefix.
    In single-agent mode, falls back to app.state.agent.

    Raises:
        HTTPException(503) if no agent is available.
    """
    # Multi-agent: middleware already resolved the agent
    agent = getattr(request.state, "agent", None)
    if agent is not None:
        return agent

    # Single-agent fallback
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        return agent

    raise HTTPException(status_code=503, detail="Agent not initialized.")


def privacy_hides_persisted(storage) -> bool:
    """Whether persisted rows are outside the agent's current visible state."""
    privacy_config = getattr(storage, "privacy_config", None)
    return privacy_config is not None and (
        privacy_config.is_ephemeral() or privacy_config.uses_temp_storage()
    )
