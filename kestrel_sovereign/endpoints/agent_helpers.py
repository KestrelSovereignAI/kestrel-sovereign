"""Shared helpers for endpoint modules."""

from typing import Mapping

from fastapi import HTTPException, Request

from kestrel_sovereign.agent.invocation import (
    InvocationProvenance,
    invocation_id_response_header,
    request_provenance,
    resolve_transport_invocation_id,
    validate_invocation_id,
)
from kestrel_sovereign.api_errors import ApiHTTPException


def require_sovereign_host_lifecycle(request: Request):
    """Admit only the sovereign-key principal to host lifecycle mutations.

    A FastAPI dependency rather than a call inside a handler, and that
    placement is load-bearing: it runs before the handler body, so the
    refusal itself carries no state about what it refused. A check placed
    after a registry lookup would answer 404 for an unknown package and
    403 for a known one, making the refusal a probe.

    That is a property of the refusal, not a confidentiality guarantee
    about the surface. `GET /api/features` deliberately returns the whole
    catalogue with per-package status to any caller, so a non-sovereign
    caller can already read what is installed and never needs a probe
    pair. Do not cite this as though it hid anything.

    Lives here, next to :func:`get_caller`, because it is the host's
    authority predicate and not one endpoint module's private helper.
    #3214 was what that privacy cost: `POST /api/features/{name}/install`
    documented "requires a sovereign agent — governed agents cannot
    install packages" and enforced nothing, while the predicate that
    would have said so sat in a sibling module guarding
    `POST /api/agents`.
    """

    caller = get_caller(request)
    if getattr(caller, "is_sovereign", False) is not True:
        raise HTTPException(
            status_code=403,
            detail="Sovereign authority is required.",
        )
    return caller


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
        raise _invalid_request_id(error) from error


def validate_request_invocation_id(value: object) -> str:
    """Validate a present literal body/query request identity."""

    try:
        return validate_invocation_id(value)
    except ValueError as error:
        raise _invalid_request_id(error) from error


def _invalid_request_id(error: ValueError) -> ApiHTTPException:
    return ApiHTTPException(
        status_code=400,
        code="invalid_request_id",
        message=(
            "request_id must be a non-empty valid Unicode string no "
            f"longer than 256 characters: {error}"
        ),
    )


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
