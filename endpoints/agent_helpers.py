"""Shared helpers for endpoint modules."""

from typing import Optional

from fastapi import HTTPException, Request


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
