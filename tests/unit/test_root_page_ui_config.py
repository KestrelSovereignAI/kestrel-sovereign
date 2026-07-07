"""GET / seeds ``window.KESTREL_UI_CONFIG`` per serving topology (#2048).

Three topologies serve the console page:

* standalone ``server:app`` — a single agent is resolvable at render, so the
  page is seeded with its ``featureCapabilities`` map (#2041);
* multi-agent ``server:app`` — no agent is resolvable, so the page is seeded
  with ``multiAgentHost: true`` and app.js skips the un-prefixed boot fetches
  of /api/ui/capabilities and /api/ui/contributions that are known to 503
  ("Agent not initialized") until selectAgent() pins routing;
* subprocess ``host.py`` — unconditionally multi-agent, same
  ``multiAgentHost`` seed.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _prepare_server_app(agent, agent_manager):
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
    app.state.agent_manager = agent_manager
    return app, original


def _restore_server_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _get_root(app):
    with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
        with TestClient(app) as client:
            return client.get("/", headers={"X-API-Key": "test-key"})


class TestServerReadRoot:
    def test_multi_agent_host_mode_seeds_multi_agent_host_flag(self):
        app, original = _prepare_server_app(agent=None, agent_manager=MagicMock())
        try:
            response = _get_root(app)
        finally:
            _restore_server_app(app, original)

        assert response.status_code == 200
        assert '"multiAgentHost": true' in response.text
        assert "featureCapabilities" not in response.text

    def test_standalone_mode_seeds_feature_capabilities_not_flag(self):
        with patch(
            "kestrel_sovereign.ui_capabilities.compute_feature_capabilities",
            return_value={"voice": True},
        ):
            app, original = _prepare_server_app(
                agent=MagicMock(), agent_manager=None
            )
            try:
                response = _get_root(app)
            finally:
                _restore_server_app(app, original)

        assert response.status_code == 200
        assert "featureCapabilities" in response.text
        assert "multiAgentHost" not in response.text

    def test_no_agent_and_no_manager_seeds_nothing(self):
        # e.g. a standalone boot whose agent failed to initialize: keep the
        # page un-seeded so app.js falls back to fetching (and surfaces) the
        # real error instead of silently skipping.
        app, original = _prepare_server_app(agent=None, agent_manager=None)
        try:
            response = _get_root(app)
        finally:
            _restore_server_app(app, original)

        assert response.status_code == 200
        assert "multiAgentHost" not in response.text
        assert "featureCapabilities" not in response.text


class TestHostServeIndex:
    def test_host_page_always_seeds_multi_agent_host_flag(self):
        from kestrel_sovereign import host

        with TestClient(host.app) as client:
            response = client.get("/")

        assert response.status_code == 200
        assert '"multiAgentHost": true' in response.text
