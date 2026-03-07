"""Shared helpers for endpoint modules."""

from fastapi import HTTPException, Request


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
