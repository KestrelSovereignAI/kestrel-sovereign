"""Agent invoke and streaming endpoints."""
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import json
import logging
import re
import time

from kestrel_sovereign.streams.tap import AgentStreamTap

from kestrel_sovereign.kestrel_config.constants import (
    MAX_SSE_CONNECTIONS_PER_CLIENT,
    SSE_PING_INTERVAL_SECONDS,
)
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
from kestrel_sovereign.endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

# SSE connection tracking: maps (client_ip, agent_id) -> active connection count
# In multi-agent mode each agent gets its own connection pool per client.
_sse_connections: dict[tuple[str, str], int] = defaultdict(int)
_sse_lock = asyncio.Lock()

# #871 — every Kestrel HTTP route lives under /api/* now. The deprecated
# /agent/* prefix is kept working by a thin path-rewrite middleware in
# server.py for one release.
router = APIRouter(prefix="/api/agent", tags=["agent"])

# Regex strips invalid JSON escape sequences (e.g. \! from zsh shells)
_INVALID_JSON_ESCAPE = re.compile(rb'\\([^"\\/bfnrtu])')


async def _parse_json_body(request: Request) -> dict:
    """Parse JSON body, recovering from common shell-escaping issues."""
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError) as orig_err:
        raw = await request.body()
        cleaned = _INVALID_JSON_ESCAPE.sub(lambda m: m.group(1), raw)
        if cleaned != raw:
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {orig_err}")


async def _parse_optional_json_body(request: Request) -> dict:
    """Parse JSON when present, returning an empty dict for an empty body."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as orig_err:
        cleaned = _INVALID_JSON_ESCAPE.sub(lambda m: m.group(1), raw)
        if cleaned != raw:
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {orig_err}")


@router.post("/invoke")
@limiter.limit("60/minute")
async def invoke_agent(request: Request):
    """
    Main endpoint to interact with the Kestrel Agent.
    It takes user input and returns the agent's response.
    Optionally accepts:
      - 'model' parameter to override the default model
      - 'session_id' to load context from a specific conversation session
    """
    try:
        data = await _parse_json_body(request)
        user_input = data.get("input")
        model_override = data.get("model")
        provider_override = data.get("provider")
        session_id = data.get("session_id")

        if user_input is None:
            raise HTTPException(status_code=400, detail="Input not provided.")

        # Combine provider and model for proper routing
        if provider_override and model_override:
            model_override = f"{provider_override}/{model_override}"

        agent = get_agent(request)
        caller = getattr(request.state, "caller", None)

        # Pre-resolve the effective session_id so it can be returned to
        # the client. Without this, the frontend pane never learns the
        # implicit UUID derived inside add_conversation and stays
        # anchored on `null`, causing later auto-load + context-status
        # paths to lose continuity. Reviewer flagged at chat.js:520.
        try:
            effective_session_id = await agent.storage.resolve_session_id(session_id)
        except Exception:
            effective_session_id = session_id  # fall back; never block the request

        response = await agent.process_input(
            user_input,
            model_override=model_override,
            session_id=effective_session_id,
            caller=caller,
        )
        return {"response": response, "session_id": effective_session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invoking agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@router.post("/stream")
@limiter.limit("60/minute")
async def stream_agent_response(request: Request):
    """
    Streaming endpoint for chat responses.
    Returns text chunks as they are generated.
    Optionally accepts 'session_id' to load context from a specific conversation.
    """
    import uuid
    
    try:
        data = await _parse_json_body(request)
        user_input = data.get("input")
        model_override = data.get("model")
        provider_override = data.get("provider")
        session_id = data.get("session_id")
        audit_before_streaming = data.get("audit_before_streaming", False)

        if user_input is None:
            raise HTTPException(status_code=400, detail="Input not provided.")

        agent = get_agent(request)
        caller = getattr(request.state, "caller", None)

        # Combine provider and model into provider/model format for routing.
        # The streaming path uses "/" in model_override to identify the provider
        # and filter to only that provider (avoids trying all providers).
        if provider_override and model_override:
            model_override = f"{provider_override}/{model_override}"

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())
        if hasattr(agent, "register_active_request"):
            agent.register_active_request(request_id)
        else:
            agent._current_request_id = request_id

        # Register the stream tap so TTS consumers can subscribe
        stream_tap = AgentStreamTap.get_instance()
        stream_tap.register(request_id)

        # Pre-resolve the effective session_id and surface it via a
        # response header. Resolved BEFORE StreamingResponse is created
        # because headers are immutable once the body starts streaming.
        # The frontend pane uses this to learn its durable conversation
        # id on first send (replacing the prior pane.sessionId=null
        # heuristic that left auto-load + context-status fragile).
        try:
            effective_session_id = await agent.storage.resolve_session_id(session_id)
        except Exception:
            effective_session_id = session_id  # fall back; never block the stream

        async def generate():
            try:
                from kestrel_sovereign.agent.streaming import strip_revise_sentinels
                async for chunk in agent.process_input_streaming(
                    user_input,
                    model_override=model_override,
                    session_id=effective_session_id,
                    audit_before_streaming=audit_before_streaming,
                    caller=caller,
                    request_id=request_id,
                ):
                    # Check if request was cancelled
                    if agent.is_request_cancelled(request_id):
                        yield "\n\n---\n⏹️ **Request stopped**\n\nType `!continue` to resume from where I left off, or start a new message."
                        break
                    # Wave 5E: strip the in-band revise sentinel before
                    # publishing to TTS subscribers — voice/TTS speaks
                    # raw chunks aloud, so leaking ``\\x1eKESTREL:REVISE...``
                    # into the audio path is a regression. The chat
                    # client receives the sentinel-bearing chunk on
                    # the yield below and strips it client-side.
                    tts_chunk = strip_revise_sentinels(chunk)
                    if tts_chunk:
                        await stream_tap.publish(request_id, tts_chunk)
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                # Surface the real error to the user instead of a generic
                # "something went wrong". Especially important for mandate
                # failures (LLMStreamingError) where the user needs to see
                # WHICH route broke and why so they can fix it (pick a
                # different model, refresh OAuth, etc.) — NOT silently get
                # an answer from a fallback model.
                from kestrel_sovereign.llm.streaming import LLMStreamingError
                if isinstance(e, LLMStreamingError):
                    route = e.provider or "unknown route"
                    yield (
                        f"\n\n---\n⚠️ **Model route `{route}` failed.**\n\n"
                        f"Error: `{e.underlying or e}`\n\n"
                        "No fallback response was generated — you selected this route "
                        "explicitly. To recover, pick a different model/route from the "
                        "dropdown, or fix the underlying issue (auth token, quota, etc.)."
                    )
                else:
                    yield f"\n\n---\n⚠️ **Error generating response:** `{e}`"
            finally:
                # Signal stream completion for TTS consumers
                await stream_tap.finish(request_id)
                # Cleanup request tracking
                agent._cleanup_cancelled_request(request_id)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        }
        if effective_session_id:
            headers["X-Session-Id"] = effective_session_id
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting up stream.")


@router.post("/stop")
async def stop_agent_request(request: Request):
    """
    Stop the current agent request/streaming.
    Used by the stop button in the UI.
    """
    try:
        data = await _parse_optional_json_body(request)
        request_id = data.get("request_id") or request.query_params.get("request_id")
        agent = get_agent(request)
        cancelled = agent.cancel_current_request(request_id=request_id)
        return {
            "success": True,
            "cancelled": cancelled,
            "request_id": request_id,
            "message": "Request cancelled" if cancelled else "No active request to cancel"
        }
    except Exception as e:
        logger.error(f"Error stopping agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error stopping agent.")


@router.get("/info")
async def get_agent_info(request: Request):
    """Get agent information including DID and privacy mode."""
    try:
        agent = get_agent(request)
        return {
            "agent_id": agent.agent_id,
            "privacy_mode": agent.privacy_mode.value if hasattr(agent.privacy_mode, 'value') else str(agent.privacy_mode),
            "features": list(agent.features.keys()) if hasattr(agent, 'features') else [],
            "audit_enabled": getattr(agent, 'audit_enabled', True),
        }
    except Exception as e:
        logger.error(f"Error getting agent info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agent info.")


@router.get("/privacy-mode")
async def get_privacy_mode(request: Request):
    """Get current privacy mode."""
    try:
        agent = get_agent(request)
        mode = agent.privacy_mode
        return {
            "privacy_mode": mode.value if hasattr(mode, 'value') else str(mode),
            "allows_cloud_llm": agent.privacy_agent.privacy_config.allows_cloud_llm() if hasattr(agent, 'privacy_agent') else True,
            "allows_storage": agent.privacy_agent.privacy_config.allows_persistent_storage() if hasattr(agent, 'privacy_agent') else True,
        }
    except Exception as e:
        logger.error(f"Error getting privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving privacy mode.")


@router.post(
    "/privacy-mode",
    dependencies=[Depends(enforce_destructive_op)],
)
async def set_privacy_mode(request: Request):
    """Set privacy mode.

    Gated by the demo-isolation rail (#766 / #867).  A privacy-mode flip is
    destructive in practice — flipping a live agent into EPHEMERAL means the
    next exit triggers the leak-purge, which can hard-DELETE rows the agent
    didn't author during the session if the leak-purge isn't scoped (see
    #867 for the wipe that prompted this gate).  On a live agent the rail
    therefore requires the ``X-Kestrel-Allow-Destructive`` header so a
    stray script can't change a live agent's privacy contract by accident.
    """
    try:
        from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config

        data = await request.json()
        mode_str = data.get("mode", "").upper()

        try:
            new_mode = PrivacyMode[mode_str]
        except KeyError:
            valid_modes = [m.name for m in PrivacyMode]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid privacy mode '{mode_str}'. Valid modes: {valid_modes}"
            )

        agent = get_agent(request)
        transition = None
        if getattr(type(agent), "set_privacy_mode_with_effects", None):
            transition = await agent.set_privacy_mode_with_effects(new_mode)
        else:
            await agent.set_privacy_mode(new_mode)

        # If switching to a local-only mode, auto-switch model to a local provider
        # If switching back to cloud-allowed mode, restore the previous model
        config = privacy_mode_to_config(new_mode)
        model_switched = getattr(transition, "model_switched", None)
        if transition is None and hasattr(agent, 'llm_service') and agent.llm_service:
            llm = agent.llm_service
            if not config.allows_cloud_llm():
                # Save the resolved active cloud selection before overriding to local,
                # so we can restore it when privacy allows cloud again.
                current_pref = llm.get_model_preference() or {}
                current_vendor = current_pref.get("vendor")
                current_model = current_pref.get("model")
                current_route = current_pref.get("route")
                if not current_model and getattr(llm, "providers", None):
                    first = llm.providers[0]
                    current_vendor = first.get("vendor")
                    current_model = first.get("model")
                    current_route = first.get("route")
                if current_model and not (
                    next((p for p in llm.providers if p.get("vendor") == current_vendor and p.get("is_local")), None)
                ):
                    llm._pre_ephemeral_preference = {
                        "vendor": current_vendor,
                        "model": current_model,
                        "route": current_route,
                    }

                local_routes = [p for p in llm.providers if p.get("is_local")]
                # Prefer ollama over llama_cpp — ollama is more universally available
                local_route = next(
                    (p for p in local_routes if p.get("vendor") == "ollama"),
                    local_routes[0] if local_routes else None,
                )
                if local_route:
                    llm.set_model_preference(
                        local_route["model"], local_route.get("vendor"), local_route.get("route")
                    )
                    model_switched = {
                        "vendor": local_route.get("vendor"),
                        "route": local_route.get("route"),
                        "model": local_route["model"],
                    }
            elif config.allows_cloud_llm():
                # Restore previous cloud preference if we saved one
                saved = getattr(llm, '_pre_ephemeral_preference', None)
                if saved:
                    llm.set_model_preference(
                        saved.get("model", ""),
                        saved.get("vendor"),
                        saved.get("route"),
                    )
                    model_switched = saved
                    llm._pre_ephemeral_preference = None

        # Auto-switch voice providers if VoiceFeature is active
        voice_switched = getattr(transition, "voice_switched", None)
        biometric_warning = getattr(transition, "biometric_warning", None)
        features = getattr(agent, "features", {})
        vf = features.get("VoiceFeature") if features else None
        if transition is None and vf and hasattr(vf, "on_privacy_mode_changed"):
            try:
                voice_switched = await vf.on_privacy_mode_changed()
            except Exception as ve:
                logger.warning("Voice auto-switch failed: %s", ve)

            # Biometric warning when switching TO a mode that allows cloud voice
            if config.allows_cloud_llm() and hasattr(vf, "biometric_warning"):
                # Only warn if there are cloud voice providers configured
                vc = getattr(vf, "_voice_config", None)
                if vc and (vc.tts_provider or vc.stt_provider):
                    biometric_warning = vf.biometric_warning()

        return {
            "success": True,
            "mode": new_mode.value,
            "message": f"Privacy mode set to {new_mode.value}",
            "allows_cloud_llm": config.allows_cloud_llm(),
            "model_switched": model_switched,
            "voice_switched": voice_switched,
            "biometric_warning": biometric_warning,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting privacy mode.")


@router.get("/notifications")
async def get_notifications(request: Request):
    """
    Get and clear pending task completion notifications.

    This is a polling endpoint - call it periodically to check for
    notifications about completed background tasks.
    """
    try:
        agent = get_agent(request)
        notifications = agent.get_pending_notifications()
        return {
            "notifications": notifications,
            "count": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving notifications.")


@router.get("/notifications/sse")
@limiter.limit("30/minute")
async def notifications_sse(request: Request):
    """
    Server-Sent Events endpoint for real-time task notifications.

    Clients connect and receive events as background tasks complete.
    Events are formatted as:
        event: task_notification
        data: {"message": "...", "type": "completed|failed|canceled"}

    Also sends periodic keepalive pings every 15 seconds.

    Connection limits: max MAX_SSE_CONNECTIONS_PER_CLIENT concurrent connections
    per client IP to prevent resource exhaustion.
    """
    import json

    # Validate agent is available before starting SSE stream
    agent = get_agent(request)

    # Enforce per-client, per-agent SSE connection limit
    client_ip = request.client.host if request.client else "unknown"
    agent_id = getattr(agent, 'agent_id', 'default')
    conn_key = (client_ip, agent_id)
    async with _sse_lock:
        if _sse_connections[conn_key] >= MAX_SSE_CONNECTIONS_PER_CLIENT:
            raise HTTPException(
                status_code=429,
                detail=f"Too many SSE connections (limit: {MAX_SSE_CONNECTIONS_PER_CLIENT})"
            )
        _sse_connections[conn_key] += 1

    async def event_generator():
        """Generate SSE events for task notifications and agent event bus."""
        agent = get_agent(request)

        # Forward events from agent.emit_event (e.g. approval_request) to this
        # stream. Without this listener, SecurityFeature._emit_approval_request
        # fires into an empty _event_listeners list and approval popups never
        # reach the browser. See #748.
        event_queue: asyncio.Queue = asyncio.Queue()

        async def _forward(event_type: str, data):
            await event_queue.put((event_type, data))

        listener_registered = False
        if hasattr(agent, "add_event_listener"):
            agent.add_event_listener(_forward)
            listener_registered = True

        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"

            # Replay any events buffered while no listener was connected —
            # e.g. the restart `completed` status emitted from
            # feature.initialize() during host startup, before this
            # reconnect landed (#1551). Drained once; registering the
            # listener above first means any event emitted from here on
            # goes to the queue instead, so nothing is dropped or doubled.
            if hasattr(agent, "get_pending_events"):
                for ev_type, ev_data in agent.get_pending_events():
                    yield f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n"

            ping_interval = SSE_PING_INTERVAL_SECONDS
            # Wave 5C: revising events are time-sensitive — the chat
            # UI uses them to retract pre-tool prose BEFORE post-tool
            # synthesis chunks land on the parallel /api/agent/stream
            # channel. The previous polling interval of 500ms could
            # cause SSE delivery to race the chat-stream chunks. We
            # block on event_queue.get() with a short timeout instead
            # so emit_event-sourced events deliver sub-frame; the
            # timeout still drives the legacy task-notification poll
            # and the keepalive ping. Codex P2 of #1084.
            task_poll_interval = 0.5
            last_ping = time.monotonic()
            last_task_poll = 0.0

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected")
                    break

                # Block waiting for an event, but bound the wait so
                # task notifications + ping still fire on schedule.
                try:
                    event_type, data = await asyncio.wait_for(
                        event_queue.get(), timeout=0.1,
                    )
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    # Drain any other events queued behind the first.
                    while True:
                        try:
                            event_type, data = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    pass

                current_time = time.monotonic()

                # Task-completion notifications come from a polling
                # source (agent.get_pending_notifications), not the
                # event queue. Keep them on the slower interval.
                if current_time - last_task_poll >= task_poll_interval:
                    notifications = agent.get_pending_notifications()
                    for notification in notifications:
                        if notification.startswith("✅"):
                            notif_type = "completed"
                        elif notification.startswith("❌"):
                            notif_type = "failed"
                        elif notification.startswith("⚠️"):
                            notif_type = "canceled"
                        else:
                            notif_type = "info"

                        event_data = json.dumps({
                            "message": notification,
                            "type": notif_type,
                        })
                        yield f"event: task_notification\ndata: {event_data}\n\n"
                    last_task_poll = current_time

                # Send keepalive ping
                if current_time - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'time': current_time})}\n\n"
                    last_ping = current_time

        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'Internal server error'})}\n\n"
        finally:
            if listener_registered:
                agent.remove_event_listener(_forward)
            async with _sse_lock:
                _sse_connections[conn_key] -= 1
                if _sse_connections[conn_key] <= 0:
                    del _sse_connections[conn_key]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/context-status")
async def get_context_status(
    request: Request,
    session_id: Optional[str] = Query(None, description="Session ID to get context for"),
    full: bool = Query(
        False,
        description=(
            "When True, run the full breakdown including RAG retrieval. "
            "The frequent footer poll passes False (cheap path); the "
            "breakdown popup (#1310) passes True on open."
        ),
    ),
):
    """Honest whole-window context status + per-section breakdown.

    The pill in the chat footer (chat.js) reads ``utilization_percent``
    and renders the ● N msgs · X% indicator. The popup (#1310) reads
    ``breakdown`` for the layered taxonomy. Both come from a single
    source of truth: ``ContextBuilder.measure_context_breakdown``
    (introduced by #1308). The ``breakdown`` field is the entire
    measurement dict, with Emma's canonical sections (system, tools,
    history, episodes, memories, rag, dynamic_context_overhead) plus
    the elastic-budget snapshot from #1309.

    Two modes:

    - ``full=False`` (default): cheap path for the frequent footer
      poll. RAG retrieval is skipped; the section is flagged
      ``skipped=true``. Memories are also off unless the agent supplies
      a side-effect-free retriever.
    - ``full=True``: invoked once when the popup opens. RAG is
      retrieved live; the popup labels it as ``estimated``.
    """
    try:
        agent = get_agent(request)

        from kestrel_sovereign.agent.token_counter import get_token_counter
        from kestrel_sovereign.agent.token_budget import RESPONSE_RESERVE
        from kestrel_sovereign.kestrel_config.constants import MAX_CONVERSATION_HISTORY_LIMIT

        # 1. Get CURRENT model (respects mandate/preference system)
        current_model = agent.get_current_model()

        # 2. Create token counter for current model (gets correct context limit)
        counter = get_token_counter(current_model)
        context_limit = counter.get_context_limit()

        # 2b. No active session → return an idle shape.  Previously passing
        # session_id=None into get_conversation_history leaked the agent's
        # cross-session aggregate count and falsely rolled utilization to
        # 100%, which surfaced in the chat footer as "472 msgs · 100%
        # Compact" on an empty pane.  See #713.  "Context window status"
        # is only meaningful for an active conversation; with none, there's
        # nothing to report.
        if not session_id:
            return {
                "model": current_model,
                "message_count": 0,
                "total_tokens": 0,
                "context_limit": context_limit,
                "response_reserve": RESPONSE_RESERVE,
                "total_budget": context_limit - RESPONSE_RESERVE,
                "utilization_percent": 0.0,
                "compaction_recommended": False,
                "status": "idle",
                "warnings": [],
                "breakdown": None,
                "route_cap": None,
                "silently_pruned_path_active": False,
            }

        # 3. Get conversation history for the specified session
        history = await agent.storage.get_conversation_history(
            limit=MAX_CONVERSATION_HISTORY_LIMIT, session_id=session_id
        )
        message_count = len(history)

        # 4. Run the canonical per-section measurement (A / #1308).
        # ``measure_context_breakdown`` is the single source of truth
        # for what the LLM call would actually see — popup and pill
        # cannot drift from production accounting (Emma's "popup must
        # reflect what the model sees" invariant from PR #1306).
        ctx_builder = getattr(agent, 'context_builder', None)
        if ctx_builder is None:
            from kestrel_sovereign.agent.context_builder import ContextBuilder
            ctx_builder = ContextBuilder(storage=agent.storage)

        # Constitution and state-of-mind for the measurement match the
        # production call path. Best-effort fetch — failure here only
        # affects measurement accuracy, not the pill rendering.
        constitution_text = ""
        get_const = getattr(agent, "get_constitution", None)
        if callable(get_const):
            try:
                got = get_const()
                constitution_text = await got if hasattr(got, "__await__") else got
                constitution_text = constitution_text or ""
            except Exception as e:
                logger.debug(f"constitution fetch failed for breakdown: {e}")

        state_of_mind = None
        llm_service = getattr(agent, "llm_service", None)
        if llm_service is not None and hasattr(llm_service, "get_state_of_mind"):
            try:
                state_of_mind = llm_service.get_state_of_mind()
            except Exception as e:
                logger.debug(f"state_of_mind fetch failed for breakdown: {e}")

        # Tool schemas the agent would send. Best-effort — surfaces the
        # previously-invisible slice; if the registry isn't reachable,
        # tool tokens stay at 0 and the popup labels them not-counted.
        tool_schemas: Optional[List[Dict[str, Any]]] = None
        registry = getattr(agent, "tool_registry", None)
        if registry is not None and hasattr(registry, "_build_all_tools"):
            try:
                tool_schemas = list(registry._build_all_tools())
            except Exception as e:
                logger.debug(f"tool schema fetch failed for breakdown: {e}")

        # When the popup runs the full breakdown (RAG included), use
        # the most recent user turn as the query so the RAG figure
        # approximates what the next LLM turn would see (codex round
        # 1 P2 caught the previous empty-query path overstating
        # accuracy). When no user turn is available, label the row
        # so the popup does not pretend the figure is representative.
        rag_query = ""
        rag_query_label: Optional[str] = None
        if full:
            try:
                from kestrel_sovereign.agent.context_builder import (
                    extract_raw_user_content,
                )
                for row in reversed(history):
                    if (row.get("role") or "").lower() == "user":
                        rag_query = extract_raw_user_content(
                            row.get("content", "") or ""
                        )
                        break
            except Exception as e:
                logger.debug(f"last-user-query lookup failed for breakdown: {e}")
            if not rag_query:
                rag_query_label = (
                    "estimated against latest stored chunks — no recent user "
                    "turn available for query-specific retrieval"
                )

        breakdown = await ctx_builder.measure_context_breakdown(
            query=rag_query,
            history=history,
            constitution=constitution_text,
            include_briefing=True,
            message_count=message_count,
            tools=tool_schemas,
            state_of_mind=state_of_mind,
            include_rag=full,
            memory_retriever=None,
        )

        # Drop the internal artifacts blob — it's the assembled bytes,
        # only useful to ``build_full_context``; the popup doesn't need
        # the bodies, only the per-section figures.
        breakdown.pop("_artifacts", None)

        # Attach the "no current query" annotation when RAG was run
        # without one; the popup renders it under the RAG row so the
        # operator can tell estimated-with-query from estimated-without.
        if full and rag_query_label and "sections" in breakdown:
            rag_section = breakdown["sections"].get("rag")
            if isinstance(rag_section, dict):
                rag_section["query_used_label"] = rag_query_label

        # C / #1311: attach salvage-state counts so the popup can
        # render the layered taxonomy (pointer-only / pending-fold /
        # folded / failed-fold) and surface back-pressure warnings.
        # Best-effort — failure to load counts must not break the
        # endpoint, just degrade the popup's salvage row to zeros.
        try:
            from kestrel_sovereign.agent.salvage import (
                DEFAULT_PENDING_WARN_THRESHOLD,
                get_salvage_state_counts,
            )
            conv_store_for_counts = (
                getattr(agent.conversation_manager, "_get_conversation_store", lambda: None)()
                if hasattr(agent, "conversation_manager")
                else None
            )
            if conv_store_for_counts is not None and "sections" in breakdown:
                salvage_counts = await get_salvage_state_counts(
                    conv_store_for_counts, session_id=session_id
                )
                hist_section = breakdown["sections"].get("history")
                if isinstance(hist_section, dict):
                    hist_section["salvages"] = salvage_counts
                    hist_section["salvages"]["warn_threshold"] = (
                        DEFAULT_PENDING_WARN_THRESHOLD
                    )
        except Exception as e:
            logger.debug(f"salvage counts fetch failed for breakdown: {e}")

        # 5. Pill % = honest whole-window utilization (the design's
        # core correctness fix: previously the pill reported history
        # slice utilization, which was misleading whenever other
        # sections dominated). Greenfield — no compat constraint
        # (Emma's 2026-05-20 review: "make the number correct").
        utilization_percent = float(breakdown["utilization_percent"])
        total_measured = int(breakdown["total_measured"])
        total_budget = int(breakdown["total_budget"])

        # 6. Status + warnings keyed off the whole-window figure.
        warnings: List[str] = []
        if utilization_percent < 50:
            status_str = "healthy"
        elif utilization_percent < 70:
            status_str = "normal"
        elif utilization_percent < 85:
            status_str = "warning"
            warnings.append(
                f"Context window {utilization_percent:.0f}% full - "
                "consider !compact to save older turns into a durable summary"
            )
        else:
            status_str = "critical"
            warnings.append(
                f"Context window {utilization_percent:.0f}% full - "
                "compaction strongly recommended"
            )

        # 7. Auto-detect the legacy silent-prune path (Emma's
        # 2026-05-20 hardening, design doc §"D auto-detect invariant").
        # When C / #1311's feature flag is enabled in production, the
        # prune path emits sync salvage records and this flag flips
        # to False — which is the release-gate signal for epic #1307
        # (Emma 2026-05-21: gate keys off this flag, not off ticket
        # closure). When the flag is disabled the legacy silent-prune
        # remains active and the popup unconditionally surfaces the
        # warning.
        try:
            from kestrel_sovereign.agent.salvage import (
                is_durable_salvage_enabled,
            )
            silently_pruned_path_active = not is_durable_salvage_enabled()
        except Exception:
            silently_pruned_path_active = True

        # #1503: route per-turn cap visibility. Some subscription tiers
        # (notably ChatGPT-Plus on ``openai:plan``) enforce a per-turn
        # payload cap well below the model's full context window. Pure
        # whole-window utilization is misleading on those routes — a
        # session at 3 % on a 256K model can still bust a 32768-token
        # route cap. Surface the cap so the UI can show binding
        # headroom before the turn fires (catches the over-cap failure
        # mode handled reactively by #1395 / #1410).
        route_cap_block: Optional[Dict[str, Any]] = None
        try:
            from kestrel_sovereign.llm.model_catalog import get_catalog_service
            catalog = get_catalog_service()
            cap_tokens = catalog.get_route_context_cap(current_model)
            if isinstance(cap_tokens, int) and cap_tokens > 0:
                projected = max(0, int(total_measured))
                cap_util_percent = (
                    (projected / cap_tokens) * 100.0 if cap_tokens else 0.0
                )
                # Resolve the route key the cap applied to so the UI
                # can show its name. Use the catalog's matched-route
                # helper, which spans ALL precedence layers (env var,
                # discovered, file) — the previous local loop only
                # scanned the file layer, so env-only / discovered-only
                # deployments showed ``route: null`` and the knob hint
                # vanished (codex round 2 P3 on the dynamic-cap PR).
                route_id: Optional[str] = None
                helper = getattr(catalog, "get_matched_route_cap_key", None)
                if callable(helper):
                    try:
                        candidate = helper(current_model)
                        # Validate the return is a real string — a mocked
                        # catalog (MagicMock) auto-returns a child mock
                        # that's truthy but not a route name, so guard
                        # explicitly rather than smuggling the mock
                        # through into the response shape.
                        if isinstance(candidate, str) and candidate:
                            route_id = candidate
                    except Exception:
                        route_id = None
                if route_id is None:
                    # Fall back to the legacy file-layer scan when the
                    # helper is absent or returned no real match. This
                    # also catches mocked-catalog test paths cleanly.
                    for known_route in getattr(catalog, "_route_context_caps", {}):
                        if (
                            current_model.lower() == known_route.lower()
                            or current_model.lower().startswith(
                                known_route.lower() + "/"
                            )
                        ) and (
                            route_id is None or len(known_route) > len(route_id)
                        ):
                            route_id = known_route
                # Operator knob hint per route (best-effort). ``openai:plan``
                # uses ``KESTREL_OPENAI_PLAN_CONTEXT_CAP`` (#1395 wiring);
                # other routes leave the knob hint empty.
                knob = (
                    "KESTREL_OPENAI_PLAN_CONTEXT_CAP"
                    if route_id == "openai:plan"
                    else None
                )
                # IMPORTANT: ``TokenCounter.get_context_limit()`` already
                # returns the route cap on capped routes (see
                # ``agent/token_counter.py:241-246``), so ``context_limit``
                # above is typically equal to ``cap_tokens`` — the
                # existing whole-window pill is therefore already
                # measuring against the route cap (modulo the response
                # reserve). The route_cap block exists so the UI can
                # NAME that cap, show the actionable knob, and report
                # raw headroom — not to provide a separate percentage
                # that would be redundant with the existing pill
                # (codex round 2 P2 on #1503).
                route_cap_block = {
                    "route": route_id,
                    "cap_tokens": cap_tokens,
                    "projected_turn_payload": projected,
                    "utilization_percent": round(cap_util_percent, 1),
                    "headroom_tokens": max(0, cap_tokens - projected),
                    "knob": knob,
                    # On the cheap footer poll (``full=False``) the
                    # breakdown was measured without RAG, so the
                    # projection is a FLOOR — the real turn payload may
                    # be higher. The popup (``full=True``) runs RAG and
                    # the projection is accurate. The UI uses this flag
                    # to label the pill / popup honestly (codex round 1
                    # P2 on #1503).
                    "includes_rag": bool(full),
                }
        except Exception as e:
            # Catalog probe must never break the endpoint — degrade to
            # "no route cap surface" rather than 500ing the footer poll.
            logger.debug(f"route_cap probe failed for breakdown: {e}")

        return {
            "model": current_model,
            "message_count": message_count,
            "total_tokens": total_measured,  # honest whole-window total
            "context_limit": context_limit,
            "response_reserve": breakdown["response_reserve"],
            "total_budget": total_budget,
            "utilization_percent": utilization_percent,
            "compaction_recommended": utilization_percent >= 70,
            "status": status_str,
            "warnings": warnings,
            # Layered breakdown the popup renders. Sections include
            # system (with subsections), tools, history (with
            # messages_kept_after_pruning + raw_tokens), episodes,
            # memories, rag, and dynamic_context_overhead.
            "breakdown": breakdown,
            # Route-level per-turn cap (#1503). ``None`` when the active
            # route declares no cap or the catalog probe failed.
            "route_cap": route_cap_block,
            # While C has not shipped, this stays True per the
            # auto-detection invariant. When C lands and the prune
            # path emits sync salvage records, flip this to False.
            "silently_pruned_path_active": silently_pruned_path_active,
        }
    except Exception as e:
        logger.error(f"Error getting context status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving context status.")


@router.get("/reflection/status")
async def reflection_status(request: Request):
    """Get reflection and self-improvement status.

    Returns scheduled reflection tasks, recent execution history,
    and health trend from training cycles.
    """
    agent = get_agent(request)

    result = {
        "reflection_hook_active": getattr(agent, "reflection_hook", None) is not None,
        "scheduled_tasks": [],
        "recent_executions": [],
    }

    # Get scheduled reflection tasks. SchedulerFeature.schedule_list now
    # returns a ToolResult envelope (#1061 wave 8); the .data dict still
    # carries the legacy {"tasks": [...]} shape.
    scheduler = agent.features.get("SchedulerFeature") if hasattr(agent, "features") else None
    if scheduler:
        try:
            envelope = await scheduler.schedule_list()
            scheduled = (envelope.data or {}).get("tasks", []) if envelope.data else []
            result["scheduled_tasks"] = [
                t for t in scheduled
                if t["task_name"] in ("reflect", "training_cycle")
            ]
        except Exception as e:
            logger.warning(f"Failed to get scheduled tasks: {e}")

    # Get recent reflection execution history from task_execution_log
    db = None
    if hasattr(agent, "_raw_storage") and hasattr(agent._raw_storage, "db"):
        db = agent._raw_storage.db
    if db:
        try:
            agent_id = getattr(agent, "agent_id", "") or getattr(agent, "did", "")
            rows = await db.fetchall(
                """
                SELECT tel.task_id, st.task_name, tel.status, tel.duration_ms, tel.executed_at,
                       SUBSTR(tel.result_text, 1, 500) as result_preview
                FROM task_execution_log tel
                JOIN scheduled_tasks st ON tel.task_id = st.id
                WHERE tel.agent_id = ? AND st.task_name IN ('reflect', 'training_cycle')
                ORDER BY tel.executed_at DESC LIMIT 10
                """,
                (agent_id,),
            )
            result["recent_executions"] = [
                {
                    "task_id": r[0], "task_name": r[1], "status": r[2],
                    "duration_ms": r[3], "executed_at": r[4], "result_preview": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to get execution history: {e}")

    return result


@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status: working, completed, failed, submitted, canceled"),
    limit: int = Query(50, le=100, description="Max results")
):
    """
    List background A2A tasks.

    Returns tasks managed by the agent's TaskManager.
    """
    agent = get_agent(request)

    # Check if agent has a task_manager
    if not hasattr(agent, 'task_manager') or not agent.task_manager:
        return {"tasks": [], "total": 0, "message": "TaskManager not available"}

    try:
        from kestrel_sovereign.a2a.types import TaskState

        # Parse status filter
        task_state = None
        if status:
            try:
                task_state = TaskState(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: working, completed, failed, submitted, canceled"
                )

        # Get tasks from task store
        tasks = await agent.task_manager.task_store.list_tasks(limit=limit)

        # Filter by status if provided
        if task_state:
            tasks = [t for t in tasks if t.status.state == task_state]

        # Convert to response format
        task_list = []
        for task in tasks:
            # Extract message text
            message_text = None
            if task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if hasattr(part, 'text'):
                        message_text = part.text
                        break

            # Extract metadata
            metadata = task.metadata or {}

            task_list.append({
                "id": task.id,
                "status": task.status.state.value,
                "message": message_text,
                "agent_id": metadata.get("agent_id"),
                "skill": metadata.get("skill"),
                "artifacts_count": len(task.artifacts) if task.artifacts else 0,
            })

        return {
            "tasks": task_list,
            "total": len(task_list)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing tasks.")


@router.post("/tasks/send")
@limiter.limit("120/minute")
async def send_task(request: Request):
    """
    Receive an A2A task creation request from another agent.

    Inbound shape (matches A2A ``TaskSendParams``):

        {
          "id": "<uuid>",                # caller-assigned task id
          "sessionId": "<uuid>",
          "message": {
            "role": "user",
            "parts": [{"type":"text", "text": "..."}]
          },
          "metadata": {                  # optional
            "sender": "<agent name or did>",
            "skill": "<workflow.* skill id>",
            ...
          },
          "artifacts": [                 # optional — send-side handoff
            {                            # payload (docs, refs, evidence)
              "name": "plan",
              "parts": [{"type":"text", "text": "..."}],
              "index": 0,
              "lastChunk": true
            }
          ]
        }

    The endpoint calls ``task_manager.create_task`` which persists the
    task AND fires the ``on_task_submitted`` callback. That callback
    builds a ``a2a.task_submitted`` Signal and enqueues it via the
    dispatcher so this agent wakes up and acts on the new task. Without
    this endpoint, agents had no wire-level way to submit a task —
    only the local agent's own code could call create_task — which made
    inter-agent A2A submission impossible to surface from a tool.

    TODO (v2, separate epic): cryptographic sender verification. Today
    we accept ``metadata["sender"]`` as a plain string claim — v1 trust
    model is same-host shared-API-key boundary, where all callers are
    inside the kestrel multi_agent host. For federation / cross-
    environment agents (different orgs, different trust tiers), this
    endpoint needs:
      * Signed envelope: ``{sender_did, signature, body, timestamp}``
        validated against the sender's DID public key.
      * Reuse the SLH-DSA infrastructure from #921 (quantum hardening
        already provisioned the keypair format + verification path).
      * Identity-injection middleware: after verification, the cognition
        turn fires with a system-context note ("Message from agent X,
        verified DID Y") so the LLM applies the right governance tier.
    See follow-up epic for the full peer-attribution layer.
    """
    agent = get_agent(request)
    body = await _parse_json_body(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(
            status_code=503,
            detail="TaskManager not available — agent cannot accept A2A tasks",
        )

    from kestrel_sovereign.a2a.types import (
        Artifact,
        Message,
        TextPart,
        TaskSendParams,
    )
    try:
        # Parse body into TaskSendParams. Sender-side already validated
        # the shape, but we re-validate here because this is the only
        # untrusted-input boundary.
        message_data = body.get("message") or {}
        parts_data = message_data.get("parts") or []
        parts = []
        for p in parts_data:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(TextPart(text=str(p.get("text", ""))))
        if not parts:
            raise HTTPException(
                status_code=400,
                detail="task message must contain at least one text part",
            )
        message = Message(
            role=str(message_data.get("role", "user")),
            parts=parts,
        )
        params = TaskSendParams(
            id=str(body.get("id") or ""),
            sessionId=str(body.get("sessionId") or ""),
            message=message,
            metadata=body.get("metadata") or {},
        )
        # Send-side artifacts/references: a sender may attach durable
        # handoff payload (planning docs, evidence bundles, saved-memory
        # references, logs, diffs) at task-creation time. This is the
        # send-side mirror of the responder-side attach flow — the
        # artifacts land on the task at SUBMITTED so the recipient can
        # retrieve them before doing any work. Validate here because
        # this is the untrusted-input boundary.
        raw_artifacts = body.get("artifacts") or []
        if not isinstance(raw_artifacts, list):
            raise HTTPException(
                status_code=400,
                detail="task 'artifacts' must be a list of artifact objects",
            )
        sender_artifacts = []
        for a in raw_artifacts:
            if not isinstance(a, dict):
                raise HTTPException(
                    status_code=400,
                    detail="each artifact must be an object",
                )
            sender_artifacts.append(Artifact.model_validate(a))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid TaskSendParams: {e}",
        )

    if not params.id or not params.sessionId:
        raise HTTPException(
            status_code=400,
            detail="TaskSendParams.id and TaskSendParams.sessionId are required",
        )

    # ``agent_name`` here is the local (recipient) agent's identifier —
    # the same value `create_task` logs as ``agent_name`` for the
    # observability row. Use the agent's DID for stable identity.
    local_name = (
        getattr(agent, "did", None)
        or getattr(agent, "_agent_name", None)
        or "unknown"
    )

    try:
        task = await agent.task_manager.create_task(
            params=params, agent_name=local_name,
            artifacts=sender_artifacts or None,
        )
    except Exception as e:
        logger.error(
            "Failed to create A2A task from peer submission: %s",
            e, exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to create task")

    # Return the canonical A2A Task envelope (model_dump produces the
    # standard JSON-RPC-friendly shape).
    return task.model_dump()


@router.get("/tasks/{task_id}")
async def get_task(request: Request, task_id: str):
    """
    Fetch a single background A2A task with its artifacts.

    Used by the Tasks panel to render "Load artifacts" for a given task.
    """
    agent = get_agent(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(status_code=404, detail="TaskManager not available")

    try:
        task = await agent.task_manager.task_store.get(task_id)
    except Exception as e:
        logger.error(f"Error loading task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading task.")

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    message_text = None
    if task.status.message and task.status.message.parts:
        for part in task.status.message.parts:
            if hasattr(part, "text"):
                message_text = part.text
                break

    artifacts_payload = []
    for artifact in (task.artifacts or []):
        if hasattr(artifact, "model_dump"):
            artifacts_payload.append(artifact.model_dump())
        else:
            artifacts_payload.append(artifact)

    return {
        "id": task.id,
        "status": task.status.state.value,
        "message": message_text,
        "artifacts": artifacts_payload,
        "metadata": task.metadata or {},
    }


@router.get("/tasks/{task_id}/subscribe")
@limiter.limit("30/minute")
async def subscribe_task(request: Request, task_id: str):
    """
    SSE stream of status updates for a single A2A task.

    Subscribers receive:
        event: status
        data: {"id": "...", "status": {...}, "final": true|false}

    Plus periodic ``event: keepalive`` pings so HTTP intermediaries
    (reverse proxies, Castle towers, NAT idle-close) don't close the
    long-lived connection between updates. The stream closes after the
    first event whose ``final == true`` is delivered.

    This endpoint exists so peer agents that just POST'd a question via
    ``/tasks/send`` can wait for the answer with a push subscription
    instead of polling ``GET /tasks/{id}`` on an adaptive backoff
    (#1444). The sender's ``PeersFeature._post_a2a_task`` opens this
    stream in a background-tracked coroutine and turns the terminal
    event into a local ``a2a.question_answered`` cognition signal.

    Auth: same handshake as ``POST /tasks/send`` — the agent-routing
    middleware applies its API-key check before this handler runs.

    Connection limits: ``MAX_SSE_CONNECTIONS_PER_CLIENT`` per
    (client_ip, agent_id) pair, same posture as ``/notifications/sse``.

    Final-state snapshot on connect: ``TaskManager.subscribe`` already
    yields a "status" event with the current task state immediately
    after subscription so a late subscriber doesn't miss a terminal
    that already fired. We forward that snapshot as the first SSE
    frame.
    """
    agent = get_agent(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(
            status_code=404, detail="TaskManager not available",
        )

    # 404 on unknown task_id rather than holding an SSE connection
    # open against a non-existent subscription target — the sender
    # would otherwise idle forever waiting for terminal events that
    # can never fire.
    task = await agent.task_manager.task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found",
        )

    client_ip = request.client.host if request.client else "unknown"
    agent_id = getattr(agent, "agent_id", "default")
    conn_key = (client_ip, agent_id)
    async with _sse_lock:
        if _sse_connections[conn_key] >= MAX_SSE_CONNECTIONS_PER_CLIENT:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many SSE connections "
                    f"(limit: {MAX_SSE_CONNECTIONS_PER_CLIENT})"
                ),
            )
        _sse_connections[conn_key] += 1

    async def event_generator():
        import json
        last_ping = time.monotonic()
        ping_interval = 10.0  # 10s heartbeat for intermediaries
        try:
            # ``TaskManager.subscribe`` already yields the current state
            # first, then streams updates, then breaks on the first
            # final event. It also yields its own ``keepalive`` events
            # on its internal timeout — we forward those as SSE
            # comments-or-pings so the connection stays warm even when
            # the task sits in SUBMITTED for a while.
            async for ev in agent.task_manager.subscribe(task_id):
                if await request.is_disconnected():
                    logger.debug(
                        "task subscribe client disconnected (task=%s)",
                        task_id[:8],
                    )
                    break
                ev_name = ev.get("event") or "status"
                ev_data = ev.get("data") or ""
                yield f"event: {ev_name}\ndata: {ev_data}\n\n"
                last_ping = time.monotonic()
                if ev.get("final"):
                    # Terminal event delivered; subscribe()'s loop
                    # already breaks, but yielding a small comment line
                    # signals end-of-stream to the SSE client cleanly.
                    yield ": end-of-stream\n\n"
                    break
                # Top up the keepalive cadence if a long stretch of
                # quiet just ended.
                now = time.monotonic()
                if now - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'t': now})}\n\n"
                    last_ping = now
        except asyncio.CancelledError:
            logger.debug(
                "task subscribe cancelled (task=%s)", task_id[:8],
            )
        except Exception as e:
            logger.error(
                "task subscribe error (task=%s): %s",
                task_id[:8], e, exc_info=True,
            )
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "Internal server error"})
                + "\n\n"
            )
        finally:
            async with _sse_lock:
                _sse_connections[conn_key] -= 1
                if _sse_connections[conn_key] <= 0:
                    del _sse_connections[conn_key]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Heartbeat Endpoints ---
#
# In the OpenClaw / kestrel-claw tradition, a "heartbeat" is a scheduled LLM
# turn that reads HEARTBEAT.md and replies HEARTBEAT_OK or an alert.  That
# surface is owned by HeartbeatRunner (kestrel_sovereign/heartbeat.py) and
# these endpoints route to it.  Liveness / readiness probes (structured
# subsystem checks, no LLM) live under /agent/health/* below.


@router.get("/heartbeat/status")
async def heartbeat_status(request: Request):
    """Get heartbeat system status and recent history."""
    agent = get_agent(request)
    runner = getattr(agent, 'heartbeat_runner', None)
    if not runner:
        return {"enabled": False, "message": "Heartbeat not configured"}

    return runner.get_status()


@router.post("/heartbeat/trigger")
async def heartbeat_trigger(request: Request):
    """Manually trigger a heartbeat check."""
    agent = get_agent(request)
    runner = getattr(agent, 'heartbeat_runner', None)
    if not runner:
        raise HTTPException(status_code=404, detail="Heartbeat not configured")

    try:
        result = await runner.run_once()
        return result.to_dict()
    except Exception as e:
        logger.error(f"Heartbeat trigger error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error triggering heartbeat.")


# --- Health (liveness probe) Endpoints ---


def _get_health_feature(agent):
    """Resolve the HealthFeature instance from an agent, or None."""
    features = getattr(agent, 'features', None) or {}
    if isinstance(features, dict):
        candidates = features.values()
    else:
        candidates = features
    for feat in candidates:
        if feat.__class__.__name__ == "HealthFeature":
            return feat
    return None


@router.get("/health/status")
async def agent_health_status(request: Request):
    """Return HealthFeature status (feature state, interval, last result).

    Separate from :func:`heartbeat_status` — heartbeat is an LLM-driven
    self-check while ``/agent/health/*`` is the structured liveness probe.
    """
    agent = get_agent(request)
    feature = _get_health_feature(agent)
    if not feature:
        return {"enabled": False, "message": "HealthFeature not available on this agent"}
    return feature.get_status()


@router.post("/health/trigger")
async def agent_health_trigger(request: Request):
    """Run a single liveness check synchronously and return the result."""
    agent = get_agent(request)
    feature = _get_health_feature(agent)
    if not feature:
        raise HTTPException(
            status_code=404,
            detail="HealthFeature not available on this agent",
        )
    try:
        return await feature.run_once()
    except Exception as e:
        logger.error(f"Health trigger error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error running liveness check.")


# =========================================================================
# Agent Mesh Protocol — RETIRED (#1367 phase 5).
#
# The legacy POST /agent/mesh and GET /agent/mesh/inbox endpoints have
# been removed. All inter-agent messaging now goes through the A2A
# task path (POST /api/agent/tasks/send + GET /api/agent/tasks/{id}),
# which provides persistence, lifecycle states, and signal-driven
# inbound wake (a2a.task_submitted). Falconer workflow events that
# used to be MeshMessage(type=ASSIGN|REVIEW_NEEDED|...) are now A2A
# tasks with metadata["skill"]="workflow.assign" (etc.).
# =========================================================================
