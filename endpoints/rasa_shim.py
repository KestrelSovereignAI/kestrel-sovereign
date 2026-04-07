"""
Rasa-compatible webhook shim for Kestrel AI.

Allows RemoteCares (and any Rasa-protocol client) to send SMS messages to Kestrel
without changing any client-side code. Just point RasaAI:Endpoint at the Kestrel
host and this endpoint handles the protocol translation.

Rasa webhook protocol:
  POST /webhooks/rest/webhook
  Body: {"sender": "<patientGuid>", "message": "<sms text>"}
  Response: [{"recipient_id": "<patientGuid>", "text": "<response>"}]

Kestrel maps:
  sender → session_id (conversation continuity per patient)
  message → user_input
  response → text in Rasa response array

Reference: jaslogic1/RemoteCares#42
"""
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cap concurrent agent processing to prevent DB/LLM contention under load
_agent_semaphore = asyncio.Semaphore(10)

router = APIRouter(prefix="/webhooks/rest", tags=["rasa-shim"])

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


class RasaWebhookResponse(BaseModel):
    recipient_id: str
    text: str


@router.post("/webhook", response_model=list[RasaWebhookResponse])
async def rasa_webhook(payload: RasaWebhookRequest, request: Request):
    """
    Rasa REST-channel compatible webhook.

    Accepts the standard Rasa webhook payload and routes the message through
    the Kestrel agent, returning a Rasa-format response list.

    The patient GUID (sender) is used as the Kestrel session_id so that each
    patient has their own persistent conversation context.
    """
    if not hasattr(request.app.state, "agent") or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Kestrel agent not initialized.")

    sender = payload.sender.strip()
    message = payload.message.strip()

    if not sender:
        raise HTTPException(status_code=400, detail="'sender' is required.")
    if not message:
        raise HTTPException(status_code=400, detail="'message' is required.")

    # Prepend SMS context so the agent can calibrate its response length and tone
    enriched_input = f"{_SMS_CONTEXT}{message}"

    try:
        agent = request.app.state.agent
        async with _agent_semaphore:
            response_text = await agent.process_input(
                user_input=enriched_input,
                session_id=f"sms:{sender}",  # namespace prevents collision with UI sessions
                model_override="anthropic/claude-sonnet-4-20250514",
                include_memories=False,  # HIPAA: prevent cross-patient memory leakage
            )
        logger.info(f"[rasa-shim] sender={sender} msg_len={len(message)} resp_len={len(response_text)}")
        return [RasaWebhookResponse(recipient_id=sender, text=response_text)]

    except Exception as exc:
        logger.error(f"[rasa-shim] Error processing message from {sender}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Kestrel agent failed to process the message.")
