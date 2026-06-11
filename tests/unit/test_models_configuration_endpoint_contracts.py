"""Focused contract tests for model/configuration endpoints."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


class _FakeAiohttpResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class _FakeAiohttpSession:
    def __init__(self, posts=None, heads=None):
        self._posts = posts or {}
        self._heads = heads or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url):
        response = self._posts.get(url)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise AssertionError(f"Unexpected POST url: {url}")
        return response

    def head(self, url):
        response = self._heads.get(url)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise AssertionError(f"Unexpected HEAD url: {url}")
        return response


def test_ipfs_status_reports_local_node_gateways_and_filecoin_adapter():
    from kestrel_sovereign.endpoints import models as model_endpoints

    local_api = f"{model_endpoints.get_ipfs_api_url()}/api/v0"
    test_cid = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"

    local_posts = {
        f"{local_api}/id": _FakeAiohttpResponse(
            payload={"ID": "peer-123", "AgentVersion": "kubo/1.0.0"}
        ),
        f"{local_api}/version": _FakeAiohttpResponse(payload={"Version": "1.2.3"}),
        f"{local_api}/pin/ls?type=recursive": _FakeAiohttpResponse(
            payload={"Keys": {"bafy1": {"Type": "recursive"}}}
        ),
    }
    gateway_heads = {
        f"https://ipfs.io/ipfs/{test_cid}": _FakeAiohttpResponse(status=200),
        f"https://dweb.link/ipfs/{test_cid}": _FakeAiohttpResponse(status=503),
        f"https://cloudflare-ipfs.com/ipfs/{test_cid}": _FakeAiohttpResponse(status=200),
    }

    client_sessions = [
        _FakeAiohttpSession(posts=local_posts),
        _FakeAiohttpSession(heads=gateway_heads),
    ]
    filecoin_adapter = SimpleNamespace(cache_dir="/tmp/filecoin-cache")
    sovereign_adapter = SimpleNamespace(filecoin_adapter=filecoin_adapter)
    storage = MagicMock(sovereign_adapter=sovereign_adapter)
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.endpoints.models.aiohttp.ClientSession", side_effect=client_sessions):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get("/api/ipfs/status", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload["local_node"]["available"] is True
        assert payload["local_node"]["peer_id"] == "peer-123"
        assert payload["local_node"]["agent_version"] == "kubo/1.0.0"
        assert payload["local_node"]["version"] == "1.2.3"
        assert payload["pinned_content"] == [{"cid": "bafy1", "type": "recursive"}]
        assert len(payload["gateways"]) == 3
        assert payload["gateways"][0]["name"] == "ipfs.io"
        assert payload["gateways"][0]["available"] is True
        assert payload["gateways"][1]["name"] == "dweb.link"
        assert payload["gateways"][1]["available"] is False
        assert payload["filecoin_adapter"]["configured"] is True
        assert payload["filecoin_adapter"]["cache_dir"] == "/tmp/filecoin-cache"
    finally:
        _restore_app(app, original)


def test_wallet_endpoint_prefers_wallet_agent_and_falls_back_to_identity_balance():
    wallet_agent = SimpleNamespace(balance=7, audit_reserve=3)
    storage = MagicMock()
    storage.get_node = AsyncMock(
        return_value=SimpleNamespace(properties={"initialBalance": 11})
    )

    with_wallet = MagicMock(wallet_agent=wallet_agent, storage=storage, agent_id="did:agent")
    fallback_agent = MagicMock(wallet_agent=None, storage=storage, agent_id="did:agent")

    for agent, expected_total in [(with_wallet, 10), (fallback_agent, 11)]:
        app, original = _prepare_app(agent)
        try:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get("/api/wallet", headers=_api_headers())
            assert response.status_code == 200
            payload = response.json()
            assert payload["currency"] == "FIL"
            assert payload["total"] == expected_total
        finally:
            _restore_app(app, original)


def test_keys_endpoints_use_storage_contract_without_exposing_secrets():
    stored_at = datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc)
    existing_key = SimpleNamespace(
        id="key-1",
        provider_id="openai",
        is_active=True,
        quota_limit=1000,
        quota_used=120,
        created_at=stored_at,
    )
    storage = MagicMock(db=MagicMock())
    agent = MagicMock(storage=storage, agent_id="did:agent")
    service_key_storage = MagicMock()
    service_key_storage.list_keys = AsyncMock(return_value=[existing_key])
    service_key_storage.store_key = AsyncMock(return_value="key-2")

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.security.service_key_storage.ServiceKeyStorage", return_value=service_key_storage):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    list_response = client.get("/api/keys", headers=_api_headers())
                    duplicate_response = client.post(
                        "/api/keys",
                        headers=_api_headers(),
                        json={"provider": "openai", "api_key": "sk-duplicate"},
                    )
                    create_response = client.post(
                        "/api/keys",
                        headers=_api_headers(),
                        json={"provider": "anthropic", "api_key": "sk-new", "quota_limit": 50},
                    )
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed["count"] == 1
        assert listed["keys"][0]["provider"] == "openai"
        assert "api_key" not in listed["keys"][0]
        assert duplicate_response.status_code == 409
        assert create_response.status_code == 200
        assert create_response.json()["key_id"] == "key-2"
        service_key_storage.store_key.assert_awaited_once_with(
            provider_id="anthropic",
            api_key="sk-new",
            quota_limit=50,
        )
    finally:
        _restore_app(app, original)


def test_keys_endpoints_refuse_privacy_hidden_persistent_storage():
    from kestrel_sovereign.privacy import PrivacyConfig

    storage = MagicMock(db=MagicMock())
    agent = MagicMock(storage=storage, agent_id="did:agent")
    agent.privacy_config = PrivacyConfig(storage="none", llm_location="local")

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.security.service_key_storage.ServiceKeyStorage") as storage_cls:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    list_response = client.get("/api/keys", headers=_api_headers())
                    create_response = client.post(
                        "/api/keys",
                        headers=_api_headers(),
                        json={"provider": "openai", "api_key": "sk-nope"},
                    )

        assert list_response.status_code == 403
        assert create_response.status_code == 403
        assert "privacy mode" in list_response.json()["detail"]
        storage_cls.assert_not_called()
    finally:
        _restore_app(app, original)


def test_key_read_endpoints_keep_empty_shape_when_storage_absent():
    storage = SimpleNamespace()
    agent = MagicMock(storage=storage, agent_id="did:agent")

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.security.service_key_storage.ServiceKeyStorage") as storage_cls:
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    keys_response = client.get("/api/keys", headers=_api_headers())
                    usage_response = client.get(
                        "/api/keys/openai/usage",
                        headers=_api_headers(),
                    )

        assert keys_response.status_code == 200
        assert keys_response.json() == {"keys": [], "count": 0}
        assert usage_response.status_code == 200
        assert usage_response.json() == {
            "provider": "openai",
            "usage": [],
            "count": 0,
            "days": 30,
        }
        storage_cls.assert_not_called()
    finally:
        _restore_app(app, original)


def test_key_write_endpoint_still_errors_when_storage_absent():
    storage = SimpleNamespace()
    agent = MagicMock(storage=storage, agent_id="did:agent")

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                delete_response = client.delete(
                    "/api/keys/openai",
                    headers=_api_headers(),
                )

        assert delete_response.status_code == 503
        assert delete_response.json()["detail"] == "Storage not available"
    finally:
        _restore_app(app, original)


def test_key_update_delete_and_usage_endpoints_preserve_provider_contracts():
    key = SimpleNamespace(
        id="key-1",
        provider_id="openai",
        is_active=True,
        quota_limit=1000,
        quota_used=120,
        created_at=None,
    )
    usage = SimpleNamespace(
        id="usage-1",
        operation="chat.completion",
        units_consumed=42,
        cost_estimate_usd=0.12,
        recorded_at=datetime(2026, 3, 17, 13, 0, tzinfo=timezone.utc),
    )
    db = MagicMock()
    db.execute = AsyncMock()
    storage = MagicMock(db=db)
    agent = MagicMock(storage=storage, agent_id="did:agent")
    service_key_storage = MagicMock()
    service_key_storage.list_keys = AsyncMock(return_value=[key])
    service_key_storage.get_usage = AsyncMock(return_value=[usage])

    app, original = _prepare_app(agent)
    try:
        with patch("kestrel_sovereign.security.service_key_storage.ServiceKeyStorage", return_value=service_key_storage):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    update_response = client.patch(
                        "/api/keys/openai",
                        headers=_api_headers(),
                        json={"quota_limit": 250, "is_active": False},
                    )
                    usage_response = client.get(
                        "/api/keys/openai/usage?days=14",
                        headers=_api_headers(),
                    )
                    delete_response = client.delete("/api/keys/openai", headers=_api_headers())
        assert update_response.status_code == 200
        db.execute.assert_any_await(
            "UPDATE agent_service_keys SET quota_limit = ?, is_active = ? WHERE id = ?",
            (250, 0, "key-1"),
        )
        assert usage_response.status_code == 200
        usage_payload = usage_response.json()
        assert usage_payload["provider"] == "openai"
        assert usage_payload["days"] == 14
        assert usage_payload["usage"][0]["operation"] == "chat.completion"
        assert delete_response.status_code == 200
        db.execute.assert_any_await(
            "DELETE FROM agent_service_keys WHERE id = ?",
            ("key-1",),
        )
    finally:
        _restore_app(app, original)


def test_models_endpoint_groups_results_and_rejects_invalid_category():
    model_one = MagicMock()
    model_one.provider = "openai"
    model_one.is_featured = True
    model_one.to_dict.return_value = {"id": "gpt-5-mini", "provider": "openai"}
    model_two = MagicMock()
    model_two.provider = "anthropic"
    model_two.is_featured = False
    model_two.to_dict.return_value = {"id": "claude-sonnet", "provider": "anthropic"}

    llm_service = MagicMock()
    llm_service.discover_all_models = AsyncMock(return_value=[model_one, model_two])
    llm_service.get_active_model_id = MagicMock(return_value="openai/gpt-5-mini")
    agent = MagicMock(llm_service=llm_service)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                ok_response = client.get(
                    "/api/models?providers=openai,anthropic&featured_only=true",
                    headers=_api_headers(),
                )
                bad_response = client.get(
                    "/api/models?category=forbidden",
                    headers=_api_headers(),
                )
        assert ok_response.status_code == 200
        payload = ok_response.json()
        assert payload["count"] == 2
        assert payload["default"] == "openai/gpt-5-mini"
        assert set(payload["by_vendor"]) == {"openai", "anthropic"}
        assert payload["featured"] == [{"id": "gpt-5-mini", "provider": "openai"}]
        llm_service.discover_all_models.assert_awaited_once()
        assert bad_response.status_code == 400
        assert "Invalid category" in bad_response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_current_and_set_model_endpoints_share_runtime_preference_contract():
    # Simulate a realistic set→get roundtrip: the endpoint now re-reads the
    # persisted mandate after set_model_preference (so auto-resolution or
    # normalization inside the service is reflected in the response).
    _state = {"vendor": "openai", "model": "gpt-5-mini", "route": None}

    def _set(model, vendor=None, route=None):
        _state["vendor"] = vendor
        _state["model"] = model
        _state["route"] = route

    llm_service = MagicMock()
    llm_service.get_model_preference = MagicMock(side_effect=lambda: dict(_state))
    llm_service.providers = [
        {"name": "anthropic:api", "vendor": "anthropic", "route": "api", "model": "claude-sonnet"}
    ]
    llm_service.set_model_preference = MagicMock(side_effect=_set)
    agent = MagicMock(llm_service=llm_service)

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                current_response = client.get("/api/model/current", headers=_api_headers())
                set_response = client.post(
                    "/api/model/set",
                    headers=_api_headers(),
                    json={"model": "anthropic/claude-opus"},
                )
                set_composite_response = client.post(
                    "/api/model/set",
                    headers=_api_headers(),
                    json={"model": "anthropic:plan/claude-opus"},
                )
                missing_response = client.post(
                    "/api/model/set",
                    headers=_api_headers(),
                    json={},
                )
        assert current_response.status_code == 200
        assert current_response.json() == {
            "model": "openai/gpt-5-mini",
            "vendor": "openai",
            "route": None,
            "model_name": "gpt-5-mini",
        }
        assert set_response.status_code == 200
        assert set_response.json()["full_model"] == "anthropic/claude-opus"
        assert set_composite_response.status_code == 200
        assert set_composite_response.json()["full_model"] == "anthropic:plan/claude-opus"
        # Vendor-only and composite-route forms both call set_model_preference
        # with the parsed vendor/route/model triple.
        llm_service.set_model_preference.assert_any_call("claude-opus", "anthropic", None)
        llm_service.set_model_preference.assert_any_call("claude-opus", "anthropic", "plan")
        assert missing_response.status_code == 400
        assert missing_response.json()["detail"] == "'model' field is required."
    finally:
        _restore_app(app, original)


# ---------------------------------------------------------------------------
# Three-tier key panel endpoints (resources.js) — see #735
# ---------------------------------------------------------------------------


def _local_sqlite_agent(has_key: bool = False):
    """Agent fixture simulating a local-sovereign SQLite deployment.

    No Postgres pool is available — BYOK/platform storage is not configured.
    ``has_key`` toggles whether ServiceKeyStorage reports the agent having
    a provisioned key for the queried provider.
    """
    db = MagicMock()
    db.backend = MagicMock()
    db.backend_type = "sqlite"  # No asyncpg pool on this backend
    storage = MagicMock(db=db)
    agent = MagicMock(storage=storage, agent_id="did:agent")
    service_key_storage = MagicMock()
    service_key_storage.has_key = AsyncMock(return_value=has_key)
    return agent, service_key_storage


def test_available_sources_reports_only_agent_tier_on_sqlite_deployments():
    """Regression for #735.  Local-sovereign Kestrel has no pool, so user/
    platform tiers must report False — and the badge shows "Agent Key" only
    when the local ServiceKeyStorage actually has that provider."""
    agent, service_key_storage = _local_sqlite_agent(has_key=True)

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.security.service_key_storage.ServiceKeyStorage",
            return_value=service_key_storage,
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get(
                        "/api/keys/available-sources?provider=openrouter",
                        headers=_api_headers(),
                    )
        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "openrouter"
        assert payload["sources"] == {"agent": True, "user": False, "platform": False}
        assert payload["platform_margin"] is None
        service_key_storage.has_key.assert_awaited_once_with(provider_id="openrouter")
    finally:
        _restore_app(app, original)


def test_available_sources_returns_all_false_when_agent_has_no_key():
    agent, service_key_storage = _local_sqlite_agent(has_key=False)

    app, original = _prepare_app(agent)
    try:
        with patch(
            "kestrel_sovereign.security.service_key_storage.ServiceKeyStorage",
            return_value=service_key_storage,
        ):
            with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
                with TestClient(app) as client:
                    response = client.get(
                        "/api/keys/available-sources?provider=openai",
                        headers=_api_headers(),
                    )
        assert response.status_code == 200
        payload = response.json()
        assert payload["sources"] == {"agent": False, "user": False, "platform": False}
    finally:
        _restore_app(app, original)


def test_list_user_keys_returns_empty_on_local_sqlite_deployment():
    """The Resources panel must see an empty list (not a 405) so the
    "No personal keys added" empty state renders."""
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/keys/user", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"keys": [], "count": 0, "available": False}
    finally:
        _restore_app(app, original)


def test_add_user_key_returns_503_on_local_deployments_with_clear_message():
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/keys/user",
                    headers=_api_headers(),
                    json={"provider": "openrouter", "api_key": "sk-x", "passphrase": "longenough"},
                )
        assert response.status_code == 503
        assert "Agent Keys" in response.json()["detail"]
    finally:
        _restore_app(app, original)


def test_verify_user_passphrase_returns_not_available_on_local_deployments():
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/keys/user/verify",
                    headers=_api_headers(),
                    json={"passphrase": "anything"},
                )
        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is False
        assert payload["available"] is False
    finally:
        _restore_app(app, original)


def test_delete_user_key_returns_503_on_local_deployments():
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.delete(
                    "/api/keys/user/openrouter",
                    headers=_api_headers(),
                )
        assert response.status_code == 503
        assert "platform" in response.json()["detail"].lower()
    finally:
        _restore_app(app, original)


def test_platform_access_returns_empty_on_local_deployments():
    """No vending-machine providers on SQLite; the panel shows the graceful
    empty state instead of a 405."""
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/keys/platform", headers=_api_headers())
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"providers": [], "available": False}
    finally:
        _restore_app(app, original)


def test_available_sources_requires_provider_query_param():
    """Missing provider should 422, not 500."""
    agent, _ = _local_sqlite_agent()

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get(
                    "/api/keys/available-sources",
                    headers=_api_headers(),
                )
        assert response.status_code == 422
    finally:
        _restore_app(app, original)
