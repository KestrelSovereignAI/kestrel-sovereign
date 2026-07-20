"""HTTPException contract tests for agent-dependent endpoints (#2495).

``get_agent(request)`` intentionally raises ``HTTPException(503, "Agent not
initialized.")`` when neither request-scoped nor app-scoped agent exists.
Handlers that wrap that call in ``try ... except Exception`` without first
re-raising ``HTTPException`` launder the 503 (and any handler-authored 4xx)
into route-specific 500s, so clients cannot distinguish "no agent/routing
context" from an internal server failure (#2489 demonstrated this live for
``GET /api/identity``).

Three layers of coverage:

1. A parameterized real-ASGI/TestClient contract: every affected route, with
   an authenticated request and no bound agent, must return the 503 verbatim.
2. Representative routes proving handler-authored 4xx HTTPExceptions pass
   through unchanged while unexpected runtime failures still become the
   sanitized 500.
3. An AST regression test preventing future ``get_agent`` + broad-except
   laundering anywhere under ``kestrel_sovereign/endpoints/``.
"""

import ast
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _api_headers():
    return {"X-API-Key": "test-key"}


# ---------------------------------------------------------------------------
# 1. Zero-agent contract: every affected route preserves the 503
# ---------------------------------------------------------------------------

# Every route the #2495 audit found laundering get_agent's 503 into a
# route-specific 500. ``GET /api/commands`` is deliberately absent: its
# documented contract returns built-in commands when no agent resolves.
AGENT_DEPENDENT_ROUTES = [
    ("POST", "/api/agent/stop"),
    ("GET", "/api/agent/info"),
    ("GET", "/api/agent/privacy-mode"),
    ("GET", "/api/agent/notifications"),
    ("GET", "/api/sessions"),
    ("GET", "/api/conversations"),
    ("POST", "/api/conversations/new"),
    ("GET", "/api/trash"),
    ("GET", "/api/db/tables"),
    ("GET", "/api/memories"),
    ("GET", "/api/agents"),
    ("GET", "/api/identity"),
    ("GET", "/api/constitution"),
    ("GET", "/api/ipfs/status"),
    ("GET", "/api/wallet"),
    ("GET", "/api/model/current"),
    ("GET", "/v1/models"),
    ("GET", "/api/storage/stats"),
    ("GET", "/api/sovereignty/exports"),
]


@pytest.mark.parametrize("method,path", AGENT_DEPENDENT_ROUTES)
def test_agent_dependent_route_returns_503_when_no_agent_bound(method, path):
    """An authenticated request with no bound agent must surface get_agent's
    HTTPException(503, "Agent not initialized.") verbatim — not a laundered
    route-specific 500."""
    app, original = _prepare_app(agent=None)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.request(method, path, headers=_api_headers())
        assert response.status_code == 503, (
            f"{method} {path}: expected 503 from get_agent, "
            f"got {response.status_code}: {response.text}"
        )
        assert "Agent not initialized" in response.json().get("detail", ""), (
            f"{method} {path}: {response.json()}"
        )
    finally:
        _restore_app(app, original)


def test_commands_discovery_stays_gracefully_degraded_without_agent():
    """The allowlisted exception: command discovery's documented contract is
    to return built-ins when no agent resolves, not to 503."""
    app, original = _prepare_app(agent=None)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/commands", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] == len(payload["commands"]) > 0
    finally:
        _restore_app(app, original)


# ---------------------------------------------------------------------------
# 2. Representative routes: handler-authored 4xx preserved, runtime
#    failures still sanitized to the intended 500
# ---------------------------------------------------------------------------


def test_sessions_route_preserves_handler_authored_4xx():
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(
        side_effect=HTTPException(status_code=422, detail="Bad session filter.")
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/sessions", headers=_api_headers())
        assert response.status_code == 422
        assert response.json()["detail"] == "Bad session filter."
    finally:
        _restore_app(app, original)


def test_memories_route_preserves_handler_authored_4xx():
    storage = MagicMock()
    storage.get_nodes_by_type = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="Memory access denied.")
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/memories?node_type=memory", headers=_api_headers()
                )
        assert response.status_code == 403
        assert response.json()["detail"] == "Memory access denied."
    finally:
        _restore_app(app, original)


def test_sessions_route_sanitizes_unexpected_runtime_failure_to_500():
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(
        side_effect=ValueError("secret internal state leaked")
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/sessions", headers=_api_headers())
        assert response.status_code == 500
        assert response.json()["detail"] == "Error retrieving sessions."
        assert "secret internal state" not in response.text
    finally:
        _restore_app(app, original)


def test_memories_route_sanitizes_unexpected_runtime_failure_to_500():
    storage = MagicMock()
    storage.get_nodes_by_type = AsyncMock(
        side_effect=RuntimeError("db exploded at /private/path")
    )
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/memories?node_type=memory", headers=_api_headers()
                )
        assert response.status_code == 500
        assert response.json()["detail"] == "Error retrieving memories."
        assert "/private/path" not in response.text
    finally:
        _restore_app(app, original)


# ---------------------------------------------------------------------------
# 3. AST regression: no future get_agent + broad-except laundering
# ---------------------------------------------------------------------------

ENDPOINTS_DIR = (
    Path(__file__).resolve().parents[2] / "kestrel_sovereign" / "endpoints"
)

# (module filename, function name) pairs whose broad except around get_agent
# is an intentional, documented graceful-degradation contract. Add entries
# here only with an explicit docstring/comment in the handler explaining the
# degraded response — everything else must re-raise HTTPException first.
GRACEFUL_DEGRADATION_ALLOWLIST = {
    ("commands.py", "get_commands"),
}


def _calls_get_agent(stmts) -> bool:
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None
                )
                if name == "get_agent":
                    return True
    return False


def _handler_type_names(handler_type):
    """Flatten an except clause's type expression into a list of names.

    ``None`` (a bare ``except:``) yields ``[None]``.
    """
    if handler_type is None:
        return [None]
    if isinstance(handler_type, ast.Name):
        return [handler_type.id]
    if isinstance(handler_type, ast.Attribute):
        return [handler_type.attr]
    if isinstance(handler_type, ast.Tuple):
        names = []
        for element in handler_type.elts:
            names.extend(_handler_type_names(element))
        return names
    return []


def _laundering_violations():
    violations = []
    for path in sorted(ENDPOINTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Try):
                    continue
                if not _calls_get_agent(node.body):
                    continue
                # Walk handlers in match order: an HTTPException clause seen
                # before any broad clause preserves the contract; a broad
                # clause seen first swallows the intentional status.
                laundered = False
                for handler in node.handlers:
                    names = _handler_type_names(handler.type)
                    if "HTTPException" in names:
                        break
                    if any(
                        name in (None, "Exception", "BaseException")
                        for name in names
                    ):
                        laundered = True
                        break
                if not laundered:
                    continue
                if (path.name, func.name) in GRACEFUL_DEGRADATION_ALLOWLIST:
                    continue
                violations.append(
                    f"{path.name}:{node.lineno} in {func.name}()"
                )
    return violations


def test_no_get_agent_broad_except_laundering_in_endpoints():
    """Any try block that calls get_agent and catches Exception (or bare
    except) without an HTTPException clause first rewrites the intentional
    503 — and any handler-authored 4xx — into a laundered 500. New handlers
    must re-raise HTTPException before generic translation, or be explicitly
    allowlisted as a documented graceful-degradation contract."""
    violations = _laundering_violations()
    assert not violations, (
        "get_agent HTTPException laundering detected (add `except "
        "HTTPException: raise` before the broad except, or allowlist a "
        "documented graceful-degradation route):\n  " + "\n  ".join(violations)
    )


def test_graceful_degradation_allowlist_matches_reality():
    """Every allowlist entry must still exist and still have the laundering
    shape — otherwise the entry is stale and should be removed."""
    remaining = set()
    for path in sorted(ENDPOINTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (path.name, func.name) not in GRACEFUL_DEGRADATION_ALLOWLIST:
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Try) and _calls_get_agent(node.body):
                    remaining.add((path.name, func.name))
    assert remaining == GRACEFUL_DEGRADATION_ALLOWLIST, (
        f"Stale allowlist entries: {GRACEFUL_DEGRADATION_ALLOWLIST - remaining}"
    )
