"""
Bridge API Router.

Provides FastAPI endpoints for external gateway integration.
These endpoints are separate from the existing /agent/invoke -- they add
gateway context (channel, sender, session mapping) on top of the agent's
core capabilities.

Endpoints:
    POST /api/bridge/invoke       -- synchronous invocation
    POST /api/bridge/stream       -- streaming invocation via SSE
    GET  /api/bridge/capabilities -- list available features/tools
    GET  /api/bridge/health       -- bridge-specific health check
    POST /api/bridge/session      -- create or resume a bridge session

Usage:
    The router is NOT registered in server.py directly. Instead, call
    ``get_router()`` to obtain the APIRouter and include it in the app.

    Example (in server.py or a plugin):
        from kestrel_sovereign.features.bridge.router import get_router
        app.include_router(get_router())
"""

import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.endpoints.agent_helpers import get_agent, get_caller

from .protocol import (
    BridgeCapabilitiesResponse,
    BridgeCapability,
    BridgeRequest,
    BridgeResponse,
    BridgeSession,
    ChannelType,
)

logger = logging.getLogger(__name__)


def _get_bridge_feature(request: Request):
    """
    Resolve the BridgeFeature from the running agent.

    Raises HTTPException 503 if the agent or bridge feature is not available.
    """
    agent = get_agent(request)

    features = getattr(agent, "features", {})
    bridge = features.get("BridgeFeature")
    if not bridge:
        raise HTTPException(
            status_code=503, detail="Bridge feature not available."
        )

    return agent, bridge


def get_router() -> APIRouter:
    """
    Build and return the bridge APIRouter.

    This factory function creates the router with all bridge endpoints.
    Call it once and include the result in the FastAPI app.
    """
    router = APIRouter(prefix="/api/bridge", tags=["bridge"])

    # ------------------------------------------------------------------
    # POST /api/bridge/invoke -- synchronous invocation
    # ------------------------------------------------------------------

    @router.post("/invoke", response_model=BridgeResponse)
    @limiter.limit("120/minute")
    async def bridge_invoke(request: Request, body: BridgeRequest):
        """
        Synchronous bridge invocation.

        The gateway sends a message and receives the full response once
        the agent has finished processing. Use /api/bridge/stream for
        real-time streaming.
        """
        agent, bridge = _get_bridge_feature(request)
        start_ms = time.monotonic()

        # Resolve or create a session
        session = await bridge.get_or_create_session(
            gateway_session_id=body.session_id,
            channel_type=body.channel_type,
            sender_id=body.sender_id,
        )

        # Log inbound request
        await bridge.log_invocation(
            session_id=session.id,
            direction="inbound",
            content_preview=body.message,
        )

        # Build context note from gateway context
        context_note = _build_context_note(body)

        # Route through the agent's process_input
        try:
            user_input = body.message
            if context_note:
                user_input = f"{user_input}\n\n[Bridge context: {context_note}]"

            response_text = await agent.process_input(
                user_input,
                model_override=body.model_override,
                session_id=session.id,
                caller=get_caller(request),
            )
        except Exception as e:
            logger.error(f"Bridge invoke error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Agent processing error.")

        elapsed_ms = int((time.monotonic() - start_ms) * 1000)

        # Log outbound response
        await bridge.log_invocation(
            session_id=session.id,
            direction="outbound",
            content_preview=response_text,
            duration_ms=elapsed_ms,
        )

        return BridgeResponse(
            message=response_text,
            session_id=session.id,
            metadata={
                "channel_type": body.channel_type.value,
                "duration_ms": elapsed_ms,
                "gateway_session_id": body.session_id,
            },
        )

    # ------------------------------------------------------------------
    # POST /api/bridge/stream -- streaming invocation via SSE
    # ------------------------------------------------------------------

    @router.post("/stream")
    @limiter.limit("120/minute")
    async def bridge_stream(request: Request, body: BridgeRequest):
        """
        Streaming bridge invocation via Server-Sent Events.

        Returns text chunks as they are generated. The final event
        includes metadata (duration, session ID).
        """
        agent, bridge = _get_bridge_feature(request)
        start_ms = time.monotonic()

        # Resolve or create a session
        session = await bridge.get_or_create_session(
            gateway_session_id=body.session_id,
            channel_type=body.channel_type,
            sender_id=body.sender_id,
        )

        # Log inbound request
        await bridge.log_invocation(
            session_id=session.id,
            direction="inbound",
            content_preview=body.message,
        )

        # Check streaming support
        if not hasattr(agent, "process_input_streaming"):
            raise HTTPException(
                status_code=501,
                detail="Streaming not supported. Use /api/bridge/invoke instead.",
            )

        # Build context note from gateway context
        context_note = _build_context_note(body)
        user_input = body.message
        if context_note:
            user_input = f"{user_input}\n\n[Bridge context: {context_note}]"

        async def event_generator():
            full_response = []
            try:
                # Wave 5E: bridge consumers (Slack/Discord/email/etc.)
                # don't speak the chat-protocol revise sentinel —
                # strip it before serializing each chunk into the
                # bridge SSE event payload.
                from kestrel_sovereign.agent.streaming import strip_revise_sentinels
                async for chunk in agent.process_input_streaming(
                    user_input,
                    model_override=body.model_override,
                    session_id=session.id,
                    caller=get_caller(request),
                ):
                    chunk = strip_revise_sentinels(chunk)
                    if not chunk:
                        continue
                    full_response.append(chunk)
                    event_data = json.dumps({"type": "chunk", "content": chunk})
                    yield f"data: {event_data}\n\n"

                # Send completion event with metadata
                elapsed_ms = int((time.monotonic() - start_ms) * 1000)
                complete_data = json.dumps({
                    "type": "done",
                    "session_id": session.id,
                    "duration_ms": elapsed_ms,
                    "channel_type": body.channel_type.value,
                })
                yield f"data: {complete_data}\n\n"

                # Log outbound response
                response_text = "".join(full_response)
                await bridge.log_invocation(
                    session_id=session.id,
                    direction="outbound",
                    content_preview=response_text,
                    duration_ms=elapsed_ms,
                )
            except Exception as e:
                logger.error(f"Bridge stream error: {e}", exc_info=True)
                error_data = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # GET /api/bridge/capabilities -- discovery
    # ------------------------------------------------------------------

    @router.get("/capabilities", response_model=BridgeCapabilitiesResponse)
    @limiter.limit("60/minute")
    async def bridge_capabilities(request: Request):
        """
        List available agent features and tools for gateway discovery.

        Gateways call this to understand what the agent can do, so they
        can build context menus, command palettes, etc.
        """
        agent, bridge = _get_bridge_feature(request)

        raw_capabilities = bridge.get_capabilities()
        capabilities = [
            BridgeCapability(
                name=c["name"],
                description=c["description"],
                category=c["category"],
                command_prefix=c.get("command_prefix"),
                parameters=c.get("parameters", []),
            )
            for c in raw_capabilities
        ]

        feature_names = list(getattr(agent, "features", {}).keys())

        return BridgeCapabilitiesResponse(
            agent_id=getattr(agent, "agent_id", "unknown"),
            features=feature_names,
            capabilities=capabilities,
        )

    # ------------------------------------------------------------------
    # GET /api/bridge/health -- bridge-specific health
    # ------------------------------------------------------------------

    @router.get("/health")
    async def bridge_health(request: Request):
        """
        Bridge-specific health check.

        Returns bridge status information. This endpoint does NOT require
        auth (suitable for gateway health probes).
        """
        try:
            agent = get_agent(request)
        except HTTPException:
            return {"status": "unavailable", "bridge": False, "agent": False}

        features = getattr(agent, "features", {})
        bridge = features.get("BridgeFeature")
        if not bridge:
            return {"status": "unavailable", "bridge": False, "agent": True}

        # bridge_status returns a ToolResult since #1061 wave 25.
        # The legacy dict the health endpoint quoted lives under .data.
        envelope = await bridge.bridge_status()
        status = envelope.data or {}
        return {
            "status": "ok",
            "bridge": True,
            "agent": True,
            "uptime_seconds": status.get("uptime_seconds", 0),
            "active_sessions": status.get("active_sessions_memory", 0),
            "database_available": status.get("database_available", False),
        }

    # ------------------------------------------------------------------
    # POST /api/bridge/session -- create or resume a session
    # ------------------------------------------------------------------

    @router.post("/session")
    @limiter.limit("60/minute")
    async def bridge_session(request: Request):
        """
        Create a new bridge session or resume an existing one.

        Body:
            session_id (optional): Gateway session ID to resume
            channel_type (optional): Channel type (default: "api")
            sender_id (optional): Sender identifier

        Returns the bridge session details including the internal session
        ID to use for subsequent invoke/stream calls.
        """
        agent, bridge = _get_bridge_feature(request)

        try:
            data = await request.json()
        except Exception:
            data = {}

        gateway_session_id = data.get("session_id")
        channel_type_str = data.get("channel_type", "api")
        sender_id = data.get("sender_id")

        try:
            channel_type = ChannelType(channel_type_str)
        except ValueError:
            channel_type = ChannelType.API

        session = await bridge.get_or_create_session(
            gateway_session_id=gateway_session_id,
            channel_type=channel_type,
            sender_id=sender_id,
        )

        return {
            "session_id": session.id,
            "gateway_session_id": session.gateway_session_id,
            "channel_type": session.channel_type.value if isinstance(session.channel_type, ChannelType) else session.channel_type,
            "sender_id": session.sender_id,
            "created_at": session.created_at.isoformat(),
            "last_activity_at": session.last_activity_at.isoformat(),
        }

    return router


def _build_context_note(body: BridgeRequest) -> str:
    """
    Build a concise context note from the gateway-provided context.

    This is appended to the user message so the agent is aware of the
    gateway context without requiring special handling.
    """
    if not body.context:
        return ""

    parts = []
    if "url" in body.context:
        parts.append(f"URL: {body.context['url']}")
    if "selected_text" in body.context:
        text = body.context["selected_text"]
        if len(text) > 500:
            text = text[:497] + "..."
        parts.append(f"Selected: {text}")
    if "page_title" in body.context:
        parts.append(f"Page: {body.context['page_title']}")
    if "channel_name" in body.context:
        parts.append(f"Channel: {body.context['channel_name']}")

    # Include any remaining keys as-is (but not the ones we already handled)
    handled_keys = {"url", "selected_text", "page_title", "channel_name"}
    for key, value in body.context.items():
        if key not in handled_keys:
            parts.append(f"{key}: {value}")

    return "; ".join(parts)
