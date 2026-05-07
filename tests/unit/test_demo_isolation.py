"""Unit tests for the server-side demo-mode isolation rail (#766).

The 2026-04-24 incident wiped three live agents because a Playwright
demo harness pointed at the live server. The convention layer
(`kestrel demo run`) is discipline; this rail is enforcement. Every test
below corresponds to an acceptance-criterion bullet on the ticket.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, HTTPException, Request

from kestrel_sovereign.security.demo_isolation import (
    ALLOW_DESTRUCTIVE_HEADER,
    classify_server_mode,
    enforce_destructive_op,
)


# --- classify_server_mode ----------------------------------------------------


def _agent(*, is_demo: bool):
    a = MagicMock()
    a.is_demo = is_demo
    a.did = "did:test:agent"
    return a


def test_empty_multi_agent_is_live_mode():
    """No agents loaded → server is live (rail engaged).

    Rationale: an empty multi_agent shouldn't license destructive ops to
    fire freely — the operator may still spin up live agents later.
    """
    assert classify_server_mode({}) is False


def test_all_demo_agents_classifies_as_demo():
    agents = {"a": _agent(is_demo=True), "b": _agent(is_demo=True)}
    assert classify_server_mode(agents) is True


def test_one_live_agent_keeps_server_live():
    """A single live agent in the multi_agent flips the server to live mode.

    The rail must protect that one live agent; demo agents in the same
    multi_agent still pass the rail because they're demo-scoped at the
    target level.
    """
    agents = {"demo": _agent(is_demo=True), "live": _agent(is_demo=False)}
    assert classify_server_mode(agents) is False


def test_truthy_non_bool_is_demo_does_not_flip_classification():
    """Defense against MagicMock auto-truthy attributes."""
    sneaky = MagicMock()
    # A bare MagicMock has truthy attributes by default — we must not
    # treat that as a demo agent.
    sneaky.did = "did:test:sneaky"
    # No explicit is_demo set; getattr returns MagicMock (truthy)
    assert classify_server_mode({"sneaky": sneaky}) is False


# --- enforce_destructive_op ---------------------------------------------------


def _fake_request(
    *,
    server_demo_mode: bool,
    agent_is_demo: bool | None,
    header: str | None = None,
    permission_store=None,
) -> Request:
    """Build a minimal Request stand-in with the bits the rail reads.

    Skips the full ASGI scope dance — the rail uses request.app.state,
    request.state, request.headers, request.client, request.url, and
    request.method, all of which we can set directly on a SimpleNamespace.
    """
    app_state = SimpleNamespace(demo_mode=server_demo_mode, agent=None)
    app = SimpleNamespace(state=app_state)

    if agent_is_demo is None:
        agent = None
    else:
        agent = SimpleNamespace(
            did=f"did:test:{'demo' if agent_is_demo else 'live'}",
            is_demo=agent_is_demo,
            features={"Security": SimpleNamespace(
                permission_store=permission_store
            )} if permission_store else {},
        )

    state = SimpleNamespace(agent=agent)
    headers = {}
    if header is not None:
        headers[ALLOW_DESTRUCTIVE_HEADER] = header
    request = SimpleNamespace(
        app=app,
        state=state,
        headers=headers,
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/conversations/sess-1"),
        method="DELETE",
    )
    return request  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_live_server_live_target_no_header_is_refused_with_audit():
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=False,
        permission_store=permission_store,
    )

    with pytest.raises(HTTPException) as exc:
        await enforce_destructive_op(request)

    assert exc.value.status_code == 403
    assert "X-Kestrel-Allow-Destructive" in exc.value.detail

    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["feature_name"] == "demo_isolation"
    assert kwargs["decision"] == "refused"
    # Audit summary must contain the endpoint and refusal reason
    assert "refused-no-destructive-header" in kwargs["args_summary"]
    assert "/api/conversations/sess-1" in kwargs["args_summary"]


@pytest.mark.asyncio
async def test_live_server_live_target_with_header_is_allowed_and_audited():
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=False,
        header="user-initiated GDPR purge",
        permission_store=permission_store,
    )

    # Should not raise
    await enforce_destructive_op(request)

    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["decision"] == "allowed"
    assert "allowed-with-header" in kwargs["args_summary"]
    assert "user-initiated GDPR purge" in kwargs["args_summary"]


@pytest.mark.asyncio
async def test_live_server_demo_target_passes_silently():
    """Demos hitting demo agents through the live server are fine — no
    audit noise, no refusal. This is the path Playwright takes against
    a demo agent the operator manually mounted on the live server.
    """
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=True,
        permission_store=permission_store,
    )

    await enforce_destructive_op(request)
    audit.assert_not_awaited()


@pytest.mark.asyncio
async def test_demo_server_demo_target_passes_silently():
    """The blessed path: ``kestrel demo run`` starts a demo server,
    the demo runner exercises a demo agent. Zero rail friction.
    """
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=True,
        agent_is_demo=True,
        permission_store=permission_store,
    )

    await enforce_destructive_op(request)
    audit.assert_not_awaited()


@pytest.mark.asyncio
async def test_demo_server_live_target_is_refused_loudly():
    """The reverse rail: a demo server that somehow has a live agent
    mounted refuses. This catches the misconfig where KESTREL_DB_PATH
    points at a directory containing both live and demo agents, or the
    multi_agent.toml is wrong.
    """
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=True,
        agent_is_demo=False,
        # Even with the bypass header, a demo server must not destroy
        # a live agent — that's the entire point of the reverse rail.
        header="any-reason",
        permission_store=permission_store,
    )

    with pytest.raises(HTTPException) as exc:
        await enforce_destructive_op(request)

    assert exc.value.status_code == 403
    assert "demo mode" in exc.value.detail
    audit.assert_awaited_once()
    kwargs = audit.await_args.kwargs
    assert kwargs["decision"] == "refused"
    assert "refused-demo-server-vs-live-agent" in kwargs["args_summary"]


@pytest.mark.asyncio
async def test_audit_summary_redacts_secret_headers():
    """Audit log must never carry API keys, bearer tokens, or session
    cookies — these blobs may be exported and inspected later.
    """
    audit = AsyncMock()
    permission_store = SimpleNamespace(log_decision=audit)
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=False,
        permission_store=permission_store,
    )
    request.headers["X-API-Key"] = "secret-api-key"
    request.headers["Authorization"] = "Bearer secret-jwt"
    request.headers["Cookie"] = "session=secret-cookie"
    request.headers["X-Other-Header"] = "kept"

    with pytest.raises(HTTPException):
        await enforce_destructive_op(request)

    summary = audit.await_args.kwargs["args_summary"]
    assert "secret-api-key" not in summary
    assert "secret-jwt" not in summary
    assert "secret-cookie" not in summary
    assert "kept" in summary  # non-sensitive headers passed through


@pytest.mark.asyncio
async def test_missing_audit_store_does_not_break_refusal():
    """If the SecurityFeature isn't loaded (early startup, slim test),
    the rail still refuses — it just logs a warning instead of writing
    to security_audit_log. A destructive op must never silently succeed
    because the audit store is unavailable.
    """
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=False,
        # No permission_store wired
    )

    with pytest.raises(HTTPException) as exc:
        await enforce_destructive_op(request)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_no_agent_in_request_treated_as_live_target():
    """If routing didn't resolve an agent, the rail must default to
    the strict path — refuse without the header. Anything else opens a
    bypass for endpoints that forget to require an agent.
    """
    request = _fake_request(
        server_demo_mode=False,
        agent_is_demo=None,
    )

    with pytest.raises(HTTPException):
        await enforce_destructive_op(request)
