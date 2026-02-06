"""Agent invoke and streaming endpoints."""
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import logging

from kestrel_sovereign.kestrel_config.constants import SSE_PING_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/invoke")
async def invoke_agent(request: Request):
    """
    Main endpoint to interact with the Kestrel Agent.
    It takes user input and returns the agent's response.
    Optionally accepts:
      - 'model' parameter to override the default model
      - 'session_id' to load context from a specific conversation session
    """
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        data = await request.json()
        user_input = data.get("input")
        model_override = data.get("model")
        session_id = data.get("session_id")

        if user_input is None:
            raise HTTPException(status_code=400, detail="Input not provided.")

        agent = request.app.state.agent
        response = await agent.process_input(
            user_input,
            model_override=model_override,
            session_id=session_id
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error invoking agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@router.post("/stream")
async def stream_agent_response(request: Request):
    """
    Streaming endpoint for chat responses.
    Returns text chunks as they are generated.
    Optionally accepts 'session_id' to load context from a specific conversation.
    """
    import uuid
    
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        data = await request.json()
        user_input = data.get("input")
        model_override = data.get("model")
        session_id = data.get("session_id")
        audit_before_streaming = data.get("audit_before_streaming", False)

        if user_input is None:
            raise HTTPException(status_code=400, detail="Input not provided.")

        agent = request.app.state.agent
        
        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())
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
                yield f"\n\nError: {str(e)}"
            finally:
                # Cleanup request tracking
                agent._cleanup_cancelled_request(request_id)

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    except Exception as e:
        logger.error(f"Error setting up stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting up stream.")


@router.post("/stop")
async def stop_agent_request(request: Request):
    """
    Stop the current agent request/streaming.
    Used by the stop button in the UI.
    """
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        cancelled = agent.cancel_current_request()
        return {
            "success": True,
            "cancelled": cancelled,
            "message": "Request cancelled" if cancelled else "No active request to cancel"
        }
    except Exception as e:
        logger.error(f"Error stopping agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error stopping agent.")


@router.get("/info")
async def get_agent_info(request: Request):
    """Get agent information including DID and privacy mode."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

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

        agent = request.app.state.agent
        agent.set_privacy_mode(new_mode)

        return {
            "success": True,
            "mode": new_mode.value,
            "message": f"Privacy mode set to {new_mode.value}"
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        notifications = agent.get_pending_notifications()
        return {
            "notifications": notifications,
            "count": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving notifications.")


@router.get("/notifications/sse")
async def notifications_sse(request: Request):
    """
    Server-Sent Events endpoint for real-time task notifications.

    Clients connect and receive events as background tasks complete.
    Events are formatted as:
        event: task_notification
        data: {"message": "...", "type": "completed|failed|canceled"}

    Also sends periodic keepalive pings every 15 seconds.
    """
    import asyncio
    import json

    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    async def event_generator():
        """Generate SSE events for task notifications."""
        agent = request.app.state.agent

        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"

        ping_interval = SSE_PING_INTERVAL_SECONDS
        poll_interval = 0.5  # Check for notifications every 500ms
        last_ping = asyncio.get_event_loop().time()

        try:
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
                current_time = asyncio.get_event_loop().time()
                if current_time - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'time': current_time})}\n\n"
                    last_ping = current_time

                # Brief sleep before next poll
                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent

        from kestrel_sovereign.agent.token_counter import get_token_counter
        from kestrel_sovereign.agent.token_budget import RESPONSE_RESERVE

        # 1. Get CURRENT model (respects mandate/preference system)
        current_model = agent.get_current_model()

        # 2. Create token counter for current model (gets correct context limit)
        counter = get_token_counter(current_model)
        context_limit = counter.get_context_limit()

        # 3. Get conversation history for the specified session and count actual tokens
        history = await agent.storage.get_conversation_history(limit=10000, session_id=session_id)
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    agent = request.app.state.agent

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
