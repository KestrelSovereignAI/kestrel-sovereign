"""Unit tests for the rail-gate on POST /agent/privacy-mode (#867-A).

The 2026-04-26 wipe slipped through because ``POST /agent/privacy-mode``
wasn't behind the demo-isolation rail.  A demo flipped Meridian to
EPHEMERAL via a UI click that didn't carry ``X-Kestrel-Allow-Destructive``,
and the rail had no opinion because the route wasn't gated.

These tests only exercise the dependency wiring — the rail itself has
its own coverage in :mod:`tests.unit.test_demo_isolation`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from kestrel_sovereign.security.demo_isolation import (
    ALLOW_DESTRUCTIVE_HEADER,
    enforce_destructive_op,
)


def _request(*, headers=None, agent=None, server_demo_mode=False):
    """Build a minimal FastAPI Request stand-in for the rail."""
    app = MagicMock()
    app.state.demo_mode = server_demo_mode
    app.state.agent = agent
    req = MagicMock(spec=Request)
    req.app = app
    req.state = MagicMock()
    req.state.agent = None
    req.url.path = "/agent/privacy-mode"
    req.method = "POST"
    req.headers = headers or {}
    req.client.host = "127.0.0.1"
    return req


def _agent(*, is_demo: bool):
    a = MagicMock()
    a.is_demo = is_demo
    a.did = "did:test:agent"
    a.features = {}
    return a


@pytest.mark.asyncio
async def test_rail_refuses_privacy_mode_flip_on_live_agent_without_header():
    """A live agent must not be flipped to a different privacy mode by a
    request that didn't carry the destructive opt-in header."""
    req = _request(agent=_agent(is_demo=False), headers={})
    with pytest.raises(HTTPException) as exc:
        await enforce_destructive_op(req)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_rail_allows_privacy_mode_flip_on_live_agent_with_header():
    """The production UI carries the header — the rail records the reason
    and lets the call through."""
    req = _request(
        agent=_agent(is_demo=False),
        headers={ALLOW_DESTRUCTIVE_HEADER: "user-initiated-mode-change"},
    )
    # No exception → rail allows
    await enforce_destructive_op(req)


@pytest.mark.asyncio
async def test_rail_allows_privacy_mode_flip_on_demo_agent_without_header():
    """Demo-scoped agents are safe to mess with — the rail allows
    destructive ops on them by design.  The bug being fixed (#867) was
    that this *exact* path also let live agents through; the rail still
    allows the demo path so demo runs continue to work."""
    req = _request(agent=_agent(is_demo=True), headers={})
    await enforce_destructive_op(req)


def test_endpoint_module_declares_rail_dependency():
    """Belt-and-braces — even if a refactor renames the dependency, this
    test catches a regression where ``POST /privacy-mode`` ships without
    the rail attached.  The route definition lives in ``endpoints.agent``;
    we just confirm the dependency is present in its kwargs."""
    from endpoints import agent as agent_endpoints

    routes = [
        r for r in agent_endpoints.router.routes
        if getattr(r, "path", None) == "/agent/privacy-mode"
        and "POST" in getattr(r, "methods", set())
    ]
    assert routes, "POST /privacy-mode route not found"
    deps = list(routes[0].dependencies or [])
    dep_callables = [getattr(d, "dependency", None) for d in deps]
    assert enforce_destructive_op in dep_callables, (
        "POST /privacy-mode must declare Depends(enforce_destructive_op) "
        "to satisfy #867-A"
    )
