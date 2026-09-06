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
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from kestrel_sovereign.endpoints.agent_helpers import (
    get_agent,
    request_invocation_provenance,
    resolve_request_invocation_id,
    stopped_invocation_http_error,
)
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    invocation_id_response_header,
)
from kestrel_sovereign.rate_limit import limiter

logger = logging.getLogger(__name__)

# Cap concurrent agent processing to prevent DB/LLM contention under load
_agent_semaphore = asyncio.Semaphore(10)

router = APIRouter(prefix="/webhooks/rest", tags=["rasa-shim"])


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
@limiter.limit("30/minute")
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
    ignore it and run ``app.state.agent`` — the host default — so a message
    explicitly addressed to agent B executed on agent A, or 503'd when no
    default existed. ``get_agent`` prefers the routed agent and falls back to
    the single-agent default, so the standalone unprefixed form is unchanged.
    """
    _verify_webhook_token(request)

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
        async with _agent_semaphore:
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

    except InvocationCancelledError as error:
        raise stopped_invocation_http_error(request_id) from error
    except Exception as exc:
        logger.error(f"[rasa-shim] Error processing message from {sender}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kestrel agent failed to process the message.")
