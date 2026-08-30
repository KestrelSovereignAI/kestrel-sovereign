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

import asyncio
from functools import lru_cache
import json
import logging
import time
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.endpoints.agent_helpers import (
    get_agent,
    get_caller,
    request_invocation_provenance,
    resolve_request_invocation_id,
    stopped_invocation_http_error,
)
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    invocation_id_response_header,
)
from kestrel_sovereign.agent.request_lifecycle import (
    RequestCompletionDisposition,
)
from kestrel_sovereign._async_ownership import OwnedAsyncIterator

from .protocol import (
    BridgeCapabilitiesResponse,
    BridgeCapability,
    BridgeRequest,
    BridgeResponse,
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


@lru_cache(maxsize=1)
def get_router() -> APIRouter:
    """
    Build and return the process-local bridge APIRouter.

    SlowAPI indexes decorated routes by ``module.function``. Rebuilding this
    router re-registers identical limits under those keys, multiplying the
    cost of every request until legitimate traffic receives a false 429. The
    handlers are request-scoped and hold no agent state, so one cached router
    is the correct lifecycle and remains safe for multi-agent mounting.
    """
    router = APIRouter(prefix="/api/bridge", tags=["bridge"])

    # ------------------------------------------------------------------
    # POST /api/bridge/invoke -- synchronous invocation
    # ------------------------------------------------------------------

    @router.post("/invoke", response_model=BridgeResponse)
    @limiter.limit("120/minute")
    async def bridge_invoke(
        request: Request,
        body: BridgeRequest,
        http_response: Response,
    ):
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
        request_id = resolve_request_invocation_id(request, body)
        invocation_provenance = request_invocation_provenance(
            request,
            source_locator="POST:/api/bridge/invoke",
        )

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
                invocation_id=request_id,
                invocation_provenance=invocation_provenance,
            )
        except InvocationCancelledError as error:
            raise stopped_invocation_http_error(request_id) from error
        except Exception:
            # Exception text and tracebacks can contain bridge message/context
            # content.  The client receives only the fixed HTTP detail below;
            # logs retain the event category and no request-derived material.
            logger.error("Bridge invoke failed")
            raise HTTPException(status_code=500, detail="Agent processing error.")

        elapsed_ms = int((time.monotonic() - start_ms) * 1000)

        # Log outbound response
        await bridge.log_invocation(
            session_id=session.id,
            direction="outbound",
            content_preview=response_text,
            duration_ms=elapsed_ms,
        )

        http_response.headers["X-Request-ID"] = invocation_id_response_header(request_id)
        return BridgeResponse(
            message=response_text,
            session_id=session.id,
            metadata={
                "channel_type": body.channel_type.value,
                "duration_ms": elapsed_ms,
                "gateway_session_id": body.session_id,
                "request_id": request_id,
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
        request_id = resolve_request_invocation_id(request, body)
        invocation_provenance = request_invocation_provenance(
            request,
            source_locator="POST:/api/bridge/stream",
        )

        async def event_generator():
            full_response = []
            request_lifecycle_registered = False
            agent_stream = None

            def stopped_event() -> str:
                stopped_data = json.dumps(
                    {
                        "type": "stopped",
                        "request_id": request_id,
                    }
                )
                return f"data: {stopped_data}\n\n"

            try:
                if hasattr(agent, "register_active_request"):
                    agent.register_active_request(request_id)
                else:
                    agent._current_request_id = request_id
                request_lifecycle_registered = True
                request_cancelled = getattr(agent, "is_request_cancelled", None)
                if (
                    callable(request_cancelled)
                    and request_cancelled(request_id) is True
                ):
                    yield stopped_event()
                    return
                # Wave 5E: bridge consumers (Slack/Discord/email/etc.)
                # don't speak the chat-protocol revise sentinel —
                # strip it before serializing each chunk into the
                # bridge SSE event payload.
                from kestrel_sovereign.agent.streaming import strip_revise_sentinels
                agent_stream = OwnedAsyncIterator(
                    lambda: agent.process_input_streaming(
                        user_input,
                        model_override=body.model_override,
                        session_id=session.id,
                        caller=get_caller(request),
                        request_id=request_id,
                        invocation_provenance=invocation_provenance,
                    ),
                    operation="bridge agent stream cleanup",
                    cleanup_requested=lambda: agent.is_request_cancelled(
                        request_id
                    ),
                )
                async for chunk in agent_stream:
                    if (
                        callable(request_cancelled)
                        and request_cancelled(request_id) is True
                    ):
                        yield stopped_event()
                        return
                    chunk = strip_revise_sentinels(chunk)
                    if not chunk:
                        continue
                    full_response.append(chunk)
                    event_data = json.dumps({"type": "chunk", "content": chunk})
                    yield f"data: {event_data}\n\n"

                # Command streaming delegates to an isolated process_input
                # child. Cooperative Stop unwinds that child as clean iterator
                # exhaustion so a persistent gateway task survives. Re-read
                # the exact request marker before publishing success: normal
                # EOF and stopped command EOF are intentionally distinct here.
                if (
                    callable(request_cancelled)
                    and request_cancelled(request_id) is True
                ):
                    yield stopped_event()
                    return

                # Send completion event with metadata
                elapsed_ms = int((time.monotonic() - start_ms) * 1000)
                complete_data = json.dumps({
                    "type": "done",
                    "session_id": session.id,
                    "duration_ms": elapsed_ms,
                    "channel_type": body.channel_type.value,
                    "request_id": request_id,
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
                # The SSE client gets only the stable safe payload built by
                # the same shared boundary /api/agent/stream uses.  Logging
                # also remains content-safe for gateway-provided input.
                # Reflecting ``str(e)`` verbatim leaked a
                # BRIDGE_STRICT_WITHHELD_PROSE_MARKER (Terra): an adapter that
                # raises after yielding partial prose wraps that late exception,
                # so its text can carry withheld response content under a strict
                # buffered audit. Never emit the underlying/message/provider text.
                logger.error("Bridge stream failed")
                from kestrel_sovereign.llm.streaming_errors import (
                    bridge_sse_error_event,
                )
                yield bridge_sse_error_event(e)
            finally:
                agent_stream_cleanup_failed = False
                try:
                    if agent_stream is not None:
                        await agent_stream.aclose()
                except BaseException:
                    agent_stream_cleanup_failed = (
                        agent_stream is not None
                        and agent_stream.cleanup_error is not None
                    )
                    raise
                else:
                    agent_stream_cleanup_failed = (
                        agent_stream is not None
                        and agent_stream.cleanup_error is not None
                    )
                finally:
                    # Bridge streams use the same counted lifecycle contract as
                    # /api/agent/stream. Duplicate retry ids deliberately share
                    # a cancellation key; each generator releases only its own
                    # registration after nested stream cleanup is terminal.
                    if request_lifecycle_registered:
                        if agent_stream_cleanup_failed:
                            agent._cleanup_cancelled_request(
                                request_id,
                                disposition=(
                                    RequestCompletionDisposition.ABANDONED
                                ),
                            )
                        else:
                            agent._cleanup_cancelled_request(request_id)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": invocation_id_response_header(request_id),
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
