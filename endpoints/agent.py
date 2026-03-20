"""Agent invoke and streaming endpoints."""
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import json
import logging
import re
import time

from kestrel_sovereign.kestrel_config.constants import (
    MAX_SSE_CONNECTIONS_PER_CLIENT,
    SSE_PING_INTERVAL_SECONDS,
)
from kestrel_sovereign.rate_limit import limiter
from endpoints.agent_helpers import get_agent

logger = logging.getLogger(__name__)

# SSE connection tracking: maps (client_ip, agent_id) -> active connection count
# In multi-agent mode each agent gets its own connection pool per client.
_sse_connections: dict[tuple[str, str], int] = defaultdict(int)
_sse_lock = asyncio.Lock()

router = APIRouter(prefix="/agent", tags=["agent"])

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
        response = await agent.process_input(
            user_input,
            model_override=model_override,
            session_id=session_id
        )
        return {"response": response}
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

        async def generate():
            try:
                async for chunk in agent.process_input_streaming(
                    user_input,
                    model_override=model_override,
                    session_id=session_id,
                    audit_before_streaming=audit_before_streaming
                ):
                    # Check if request was cancelled
                    if agent.is_request_cancelled(request_id):
                        yield "\n\n---\n⏹️ **Request stopped**\n\nType `!continue` to resume from where I left off, or start a new message."
                        break
                    yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                yield "\n\nAn error occurred while generating the response."
            finally:
                # Cleanup request tracking
                agent._cleanup_cancelled_request(request_id)

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            }
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


@router.post("/privacy-mode")
async def set_privacy_mode(request: Request):
    """Set privacy mode."""
    try:
        from kestrel_sovereign.privacy import PrivacyMode

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
        await agent.set_privacy_mode(new_mode)

        # If switching to a local-only mode, auto-switch model to a local provider
        # If switching back to cloud-allowed mode, restore the previous model
        config = new_mode.to_config()
        model_switched = None
        if hasattr(agent, 'llm_service') and agent.llm_service:
            llm = agent.llm_service
            if not config.allows_cloud_llm():
                local_names = llm._get_local_provider_names()
                # Save the resolved active cloud route before overriding to local.
                # A user may be running on the default provider/model without an
                # explicit mandate preference, and that effective route still needs
                # to be restored when leaving local-only privacy modes.
                current_pref = llm.get_model_preference() or {}
                current_provider = current_pref.get("provider")
                current_model = current_pref.get("model")
                if not current_model and getattr(llm, "providers", None):
                    first_provider = llm.providers[0]
                    current_provider = first_provider.get("name")
                    current_model = first_provider.get("model")
                if current_model and current_provider not in (local_names or []):
                    llm._pre_ephemeral_preference = {
                        "provider": current_provider,
                        "model": current_model,
                    }

                local_providers = [p for p in llm.providers if p["name"] in local_names]
                # Prefer ollama over llama_cpp — ollama is more universally available
                local_provider = next(
                    (p for p in local_providers if p["name"] == "ollama"),
                    local_providers[0] if local_providers else None,
                )
                if local_provider:
                    llm.set_model_preference(local_provider["model"], local_provider["name"])
                    model_switched = {"provider": local_provider["name"], "model": local_provider["model"]}
            elif config.allows_cloud_llm():
                # Restore previous cloud preference if we saved one
                saved = getattr(llm, '_pre_ephemeral_preference', None)
                if saved:
                    llm.set_model_preference(saved.get("model", ""), saved.get("provider", ""))
                    model_switched = saved
                    llm._pre_ephemeral_preference = None

        return {
            "success": True,
            "mode": new_mode.value,
            "message": f"Privacy mode set to {new_mode.value}",
            "allows_cloud_llm": config.allows_cloud_llm(),
            "model_switched": model_switched,
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
        """Generate SSE events for task notifications."""
        try:
            agent = get_agent(request)

            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"

            ping_interval = SSE_PING_INTERVAL_SECONDS
            poll_interval = 0.5  # Check for notifications every 500ms
            last_ping = time.monotonic()

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected")
                    break

                # Check for pending notifications
                notifications = agent.get_pending_notifications()
                for notification in notifications:
                    # Determine notification type from emoji prefix
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
                        "type": notif_type
                    })
                    yield f"event: task_notification\ndata: {event_data}\n\n"

                # Send keepalive ping
                current_time = time.monotonic()
                if current_time - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'time': current_time})}\n\n"
                    last_ping = current_time

                # Brief sleep before next poll
                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'Internal server error'})}\n\n"
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
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/context-status")
async def get_context_status(
    request: Request,
    session_id: Optional[str] = Query(None, description="Session ID to get context for")
):
    """
    Get current context window status including token usage and budget.

    Args:
        session_id: Optional session ID to get context for specific session

    Returns:
        - model: Current active model (respects mandate/preference system)
        - message_count: Total messages in conversation history
        - total_tokens: Actual token count of conversation history
        - context_limit: Model's context window limit
        - total_budget: Available budget after response reserve
        - utilization_percent: Overall context utilization percentage
        - compression_recommended: Whether compression is recommended
        - status: healthy/normal/warning/critical
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

        # 3. Get conversation history for the specified session and count actual tokens
        history = await agent.storage.get_conversation_history(limit=MAX_CONVERSATION_HISTORY_LIMIT, session_id=session_id)
        message_count = len(history)

        total_tokens = sum(
            counter.count(msg.get("content", ""))
            for msg in history
        )

        # 4. Calculate utilization
        response_reserve = RESPONSE_RESERVE
        total_budget = context_limit - response_reserve
        utilization = (total_tokens / total_budget * 100) if total_budget > 0 else 0

        # 5. Determine status and warnings
        warnings = []
        if utilization < 50:
            status_str = "healthy"
        elif utilization < 70:
            status_str = "normal"
        elif utilization < 85:
            status_str = "warning"
            warnings.append(f"Context {utilization:.0f}% full - consider using !compress")
        else:
            status_str = "critical"
            warnings.append(f"Context {utilization:.0f}% full - compression strongly recommended")

        return {
            "model": current_model,
            "message_count": message_count,
            "total_tokens": total_tokens,
            "context_limit": context_limit,
            "response_reserve": response_reserve,
            "total_budget": total_budget,
            "utilization_percent": round(utilization, 1),
            "compression_recommended": utilization >= 70,
            "status": status_str,
            "warnings": warnings
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

    # Get scheduled reflection tasks
    scheduler = agent.features.get("SchedulerFeature") if hasattr(agent, "features") else None
    if scheduler:
        try:
            tasks = await scheduler.schedule_list()
            result["scheduled_tasks"] = [
                t for t in tasks.get("tasks", [])
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


# --- Heartbeat Endpoints ---


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
