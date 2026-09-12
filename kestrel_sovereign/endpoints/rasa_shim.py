"""
Rasa-compatible webhook shim for Kestrel AI.

Allows any Rasa-protocol client to send SMS messages to Kestrel without
changing any client-side code. Just point the Rasa REST endpoint at the
Kestrel host and this endpoint handles the protocol translation.

Rasa webhook protocol:
  POST /webhooks/rest/webhook
  Body: {"sender": "<sender_id>", "message": "<sms text>"}
  Response: [{"recipient_id": "<sender_id>", "text": "<response>"}]

Kestrel maps:
  sender → session_id (conversation continuity per sender)
  message → user_input
  response → text in Rasa response array
"""
import asyncio
import logging
import os
import secrets
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from kestrel_sovereign.endpoints.agent_helpers import (
    get_agent,
    prime_durable_stop_fence,
    request_invocation_provenance,
    resolve_request_invocation_id,
    self_fenced_invocation_http_error,
    stopped_invocation_http_error,
)
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    InvocationSelfFencedError,
    invocation_id_response_header,
)
from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.hold import HoldTurnRefusal
from kestrel_sovereign.rate_limit import limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Cap concurrent agent processing to prevent DB/LLM contention under load.
# One semaphore PER routed agent (kestrel-sovereign#3220): before the routed
# agent was bound only the host default could ever be invoked here, so ten
# permits belonged to one agent by construction; a single shared gate would
# now let agent A's slow turns stall agent B with no 429 and no timeout.
# The trade is stated: there is no host-wide aggregate cap any more — an
# N-agent fleet can hold 10·N concurrent turns here. N is the operator's
# opt-in list and arrival stays bounded at 30/minute per (source, agent).
_AGENT_CONCURRENCY = 10
_agent_semaphores: dict[str, asyncio.Semaphore] = {}


def _agent_semaphore_for(routed_name: Optional[str]) -> asyncio.Semaphore:
    """The concurrency gate for one routed agent (``"default"`` unprefixed)."""
    key = routed_name or "default"
    semaphore = _agent_semaphores.get(key)
    if semaphore is None:
        semaphore = _agent_semaphores[key] = asyncio.Semaphore(_AGENT_CONCURRENCY)
    return semaphore

router = APIRouter(prefix="/webhooks/rest", tags=["rasa-shim"])


def _routed_agent_name(request: Request) -> Optional[str]:
    """The routing name of the agent an agent-prefixed request named, or
    ``None`` for the unprefixed form.

    The routing middleware pins ``request.state.agent`` but not the name it
    matched; the AgentManager maps the agent back to its routing key
    (``None`` for a fenced spawn route).
    """
    routed = getattr(request.state, "agent", None)
    if routed is None:
        return None
    manager = getattr(request.app.state, "agent_manager", None)
    if manager is None:
        return None
    name = manager.get_agent_name(getattr(routed, "did", None))
    return name if isinstance(name, str) and name else None


def _rasa_rate_limit_key(request: Request) -> str:
    """Rate-limit bucket: remote address × routed agent (kestrel-sovereign#3220).

    The middleware strips the agent prefix before SlowAPI sees the path, so
    without this every agent on the host shared one 30/minute bucket per
    source and one gateway forwarding for the fleet had agent A's traffic
    starve agent B.
    """
    return f"{get_remote_address(request)}|{_routed_agent_name(request) or 'default'}"


def _rasa_enabled_agents() -> frozenset[str]:
    """Routing names the sovereign has enabled the Rasa channel for, casefolded.

    ``KESTREL_RASA_WEBHOOK_AGENTS`` is a comma-separated list. It governs
    ONLY the agent-prefixed alias; the unprefixed form reaches the host
    default under the token's pre-existing authority. Compared casefolded,
    the way ``AgentManager.get_agent`` resolves routing names (no two
    agents can differ by case only).
    """
    raw = os.environ.get("KESTREL_RASA_WEBHOOK_AGENTS", "")
    return frozenset(
        name.strip().casefold() for name in raw.split(",") if name.strip()
    )


def _verify_routed_agent_enabled(request: Request, routed_name: Optional[str]) -> None:
    """Refuse the agent-prefixed alias for an agent not enabled for Rasa.

    One host-wide token authenticates this endpoint. Before #3220 that token
    could only ever reach the host default; binding the routed agent would
    have let the same token drive a paid ``process_input`` turn on EVERY
    agent on the host by changing one path segment. So the prefixed form is
    an explicit, per-agent, sovereign-configured opt-in and FAILS CLOSED:
    unset or unlisted answers with the routing middleware's
    ``agent_not_found`` envelope and nobody is invoked. (A token holder can
    still tell an unlisted agent from an absent one — the refusal echoes the
    canonical routing name and counts against the rate bucket — so this is
    a fail-closed opt-in whose refusal is name-shaped, not a secrecy
    property.) The refusal is host-logged with the variable to set, since a
    dead endpoint whose only explanation is a docstring is not an operator
    surface.

    ``routed_name`` is ``None`` for two different facts: the unprefixed
    form, and a prefixed request whose agent the registry can no longer
    name (a spawn route fenced between routing and this check). Only the
    first is admitted; the second fails closed too.
    """
    routed = getattr(request.state, "agent", None)
    if routed is None:
        return
    if routed_name is None:
        logger.warning(
            "[rasa-shim] refused prefixed alias: the routed agent is no longer "
            "resolvable in the routing registry (fenced?)"
        )
        raise ApiHTTPException(
            status_code=404, code="agent_not_found", message="Agent not found"
        )
    if routed_name.casefold() not in _rasa_enabled_agents():
        logger.warning(
            "[rasa-shim] refused prefixed alias for agent '%s': not listed in "
            "KESTREL_RASA_WEBHOOK_AGENTS",
            routed_name,
        )
        raise ApiHTTPException(
            status_code=404,
            code="agent_not_found",
            message=f"Agent '{routed_name}' not found",
        )


def _verify_webhook_token(request: Request) -> None:
    """Authenticate the Rasa webhook (#1729).

    ``/webhooks/*`` is exempt from the host API-key middleware (webhooks
    self-authenticate), so this endpoint — which drives a full, paid
    ``process_input`` turn — MUST authenticate itself. It requires a shared
    secret in ``KESTREL_RASA_WEBHOOK_TOKEN``, presented as ``Authorization:
    Bearer <token>`` or the ``X-Webhook-Token`` header. FAILS CLOSED: if the
    token isn't configured, the endpoint is disabled (no anonymous LLM access).
    """
    expected = os.environ.get("KESTREL_RASA_WEBHOOK_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Rasa webhook disabled: set KESTREL_RASA_WEBHOOK_TOKEN to enable.",
        )
    presented = request.headers.get("X-Webhook-Token", "")
    if not presented:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            presented = auth[len("Bearer "):]
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing webhook token.")

# SMS context prefix prepended to every patient message so Kestrel understands
# the channel without constitution changes.
_SMS_CONTEXT = (
    "[CONTEXT: This is an inbound SMS from a home health patient. "
    "Respond briefly and warmly — 1-3 sentences max. "
    "If the patient appears to be logging a vital sign, acknowledge it encouragingly. "
    "If it is a question, answer it clearly. Never use markdown.]\n\n"
    "Patient says: "
)


class RasaWebhookRequest(BaseModel):
    sender: str
    message: str
    request_id: str | None = None
    invocation_id: str | None = None


class RasaWebhookResponse(BaseModel):
    recipient_id: str
    text: str


@router.post("/webhook", response_model=list[RasaWebhookResponse])
@limiter.limit("30/minute", key_func=_rasa_rate_limit_key)
async def rasa_webhook(
    request: Request,
    payload: RasaWebhookRequest,
    http_response: Response,
):
    """
    Rasa REST-channel compatible webhook.

    Accepts the standard Rasa webhook payload and routes the message through
    the Kestrel agent, returning a Rasa-format response list.

    The patient GUID (sender) is used as the Kestrel session_id so that each
    patient has their own persistent conversation context.

    Authenticated via a shared webhook token and rate-limited (#1729) — this is
    an anonymous-path endpoint that drives a full paid LLM turn.

    The agent is the request-routed one (kestrel-sovereign#3220): on a
    multi-agent host the routing middleware pins ``request.state.agent`` for
    ``/api/agents/{name}/webhooks/rest/webhook``, and this handler used to
    ignore it and read ``app.state.agent`` instead. Every multi-agent boot
    sets ``app.state.agent`` to ``None`` (the two topologies are mutually
    exclusive), so the real pre-fix symptom was a 503 on EVERY prefixed
    request: agent B was unreachable, not misrouted. ``get_agent`` prefers
    the routed agent and falls back to the single-agent default — the
    precedence also holds over the synthetic "manager plus default" state,
    as defence in depth. The unprefixed form keeps its behaviour with one
    visible change: the no-agent case now answers the canonical
    ``503 "Agent not initialized."`` (it said "Kestrel agent not
    initialized." before).

    The prefixed alias is additionally an explicit per-agent opt-in
    (``KESTREL_RASA_WEBHOOK_AGENTS``), because one host-wide token must not
    become paid turns on every agent by path segment. Both per-agent
    bounds are keyed on the routed agent: the rate-limit bucket (remote
    address AND routed agent) and the concurrency semaphore, so one gateway
    forwarding for the fleet cannot let agent A starve agent B on either.
    Order: token, then opt-in, then agent resolution, then payload
    validation.
    """
    _verify_webhook_token(request)
    routed_name = _routed_agent_name(request)
    _verify_routed_agent_enabled(request, routed_name)

    agent = get_agent(request)

    sender = payload.sender.strip()
    message = payload.message.strip()

    if not sender:
        raise HTTPException(status_code=400, detail="'sender' is required.")
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required.")

    # Prepend SMS context so the agent can calibrate its response length and tone
    enriched_input = f"{_SMS_CONTEXT}{message}"
    request_id = resolve_request_invocation_id(request, payload)
    invocation_provenance = request_invocation_provenance(
        request,
        source_locator="POST:/webhooks/rest/webhook",
        # The endpoint's shared-secret gate authenticates a gateway service,
        # not the untrusted payload ``sender``. Record that service principal
        # rather than falsely attributing a patient delivery to the agent.
        fallback_actor="rasa_webhook",
    )

    try:
        await prime_durable_stop_fence(request, agent, request_id)
        async with _agent_semaphore_for(routed_name):
            # Semaphore admission can wait beyond the short in-memory Stop
            # reservation TTL. Re-read the durable exact-turn authority at the
            # execution boundary so an acknowledged queued Stop cannot age out
            # and then start work.
            await prime_durable_stop_fence(request, agent, request_id)
            response_text = await agent.process_input(
                user_input=enriched_input,
                session_id=f"sms:{sender}",  # namespace prevents collision with UI sessions
                include_memories=False,  # HIPAA: prevent cross-patient memory leakage
                invocation_id=request_id,
                invocation_provenance=invocation_provenance,
            )
        logger.info(f"[rasa-shim] sender={sender} msg_len={len(message)} resp_len={len(response_text)}")
        http_response.headers["X-Request-ID"] = invocation_id_response_header(request_id)
        return [RasaWebhookResponse(recipient_id=sender, text=response_text)]

    except InvocationSelfFencedError as error:
        raise self_fenced_invocation_http_error(request_id) from error
    except InvocationCancelledError as error:
        raise stopped_invocation_http_error(request_id) from error
    except HoldTurnRefusal as exc:
        raise exc.as_http_exception() from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[rasa-shim] Error processing message from {sender}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kestrel agent failed to process the message.")
