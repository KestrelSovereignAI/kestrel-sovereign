"""OpenRouter registration from a management key + public key-limit SDK (#2243).

The registry can register the OpenRouter route from only
``OPENROUTER_MANAGEMENT_API_KEY`` (no static ``OPENROUTER_API_KEY``) by minting
an ephemeral bootstrap child key via the async ``finalize_providers()`` hook.
When a static key IS present, behavior is byte-identical to before. The public
SDK helper updates a key's limit by hash.

All OpenRouter keys-API traffic is mocked — no network.
"""

import httpx
import pytest

from kestrel_sovereign.llm.provider_registry import (
    ProviderRegistry,
    _reset_bootstrap_openrouter_key_cache,
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("GET", "http://test"), response=self
            )

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Records requests; returns canned OpenRouter keys-API responses."""

    instances: list["_FakeAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_closed = False
        self.requests: list[tuple] = []
        _FakeAsyncClient.instances.append(self)

    async def post(self, url, json=None):
        self.requests.append(("POST", url, json))
        return _FakeResponse(
            200, {"key": "sk-or-v1-bootstrap-xyz", "data": {"hash": "boot-hash"}}
        )

    async def get(self, url):
        self.requests.append(("GET", url, None))
        return _FakeResponse(
            200,
            {
                "data": {
                    "limit": 250.0,
                    "limit_remaining": 250.0,
                    "usage": 0.0,
                    "usage_monthly": 0.0,
                    "is_free_tier": False,
                    "rate_limit": {},
                }
            },
        )

    async def patch(self, url, json=None):
        self.requests.append(("PATCH", url, json))
        return _FakeResponse(200, {})

    async def delete(self, url):
        self.requests.append(("DELETE", url, None))
        return _FakeResponse(200, {})

    async def aclose(self):
        self.is_closed = True


@pytest.fixture(autouse=True)
def _mock_openrouter_http(monkeypatch):
    """Stub the provisioning HTTP client (scoped) + reset caches.

    Patching the service's own ``_get_client`` keeps the fake confined to
    OpenRouter provisioning — replacing ``httpx.AsyncClient`` globally would
    leak into unrelated clients (e.g. google-genai) built during registry init.
    """
    from kestrel_sovereign.features.llm_keys.openrouter_provisioning import (
        OpenRouterProvisioningService,
    )

    _FakeAsyncClient.instances = []
    _reset_bootstrap_openrouter_key_cache()

    async def _fake_get_client(self):
        if self._client is None or self._client.is_closed:
            self._client = _FakeAsyncClient()
        return self._client

    monkeypatch.setattr(
        OpenRouterProvisioningService, "_get_client", _fake_get_client
    )
    yield
    _reset_bootstrap_openrouter_key_cache()


def _openrouter_config():
    return {
        "route_priority": ["openrouter:api"],
        "vendors": {
            "openrouter": {
                "is_cloud": True,
                "routes": {
                    "api": {
                        "adapter": "OpenRouterAdapter",
                        "model": "auto",
                        "api_key_env": "OPENROUTER_API_KEY",
                    }
                },
            }
        },
    }


def _or_provider(providers):
    return next(p for p in providers if p.vendor == "openrouter")


# ------------------------------------------------------------------ path (a)

def test_static_key_registers_unchanged_no_mint(monkeypatch):
    """OPENROUTER_API_KEY set → route uses it directly, no bootstrap mint."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-static-123")
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)

    registry = ProviderRegistry(config=_openrouter_config())
    providers = registry.initialize_providers()

    client = _or_provider(providers).client
    assert client.api_key == "sk-or-static-123"
    # No deferral, no mint traffic.
    assert registry._deferred_openrouter_routes == []
    assert _FakeAsyncClient.instances == []


# ------------------------------------------------------------------ path (b)

@pytest.mark.asyncio
async def test_management_key_only_registers_via_bootstrap(monkeypatch):
    """Only OPENROUTER_MANAGEMENT_API_KEY → route registers after finalize."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "sk-or-mgmt-abc")

    registry = ProviderRegistry(config=_openrouter_config())
    # Sync init defers the route rather than raising.
    providers = registry.initialize_providers()
    assert not any(p.vendor == "openrouter" for p in providers)
    assert len(registry._deferred_openrouter_routes) == 1

    # Async finalize mints a bootstrap child key and registers the route.
    finalized = await registry.finalize_providers()
    client = _or_provider(finalized).client
    assert client.api_key == "sk-or-v1-bootstrap-xyz"
    assert registry._deferred_openrouter_routes == []

    # The adapter itself is authenticated with the bootstrap key, not just
    # the injected OpenAI client. list_models()/_get_client read
    # ``adapter.api_key`` directly (ignoring the framework client), so without
    # this the catalog would come back empty and model="auto" would fail.
    assert _or_provider(finalized).adapter.api_key == "sk-or-v1-bootstrap-xyz"

    # Exactly one mint POST /keys happened.
    posts = [r for c in _FakeAsyncClient.instances for r in c.requests if r[0] == "POST"]
    assert posts and posts[0][1] == "/keys"
    # Generous default limit, ephemeral (monthly reset).
    assert posts[0][2]["limit"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_finalize_is_idempotent(monkeypatch):
    """A second finalize does not double-register or re-mint."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_API_KEY", "sk-or-mgmt-abc")

    registry = ProviderRegistry(config=_openrouter_config())
    registry.initialize_providers()
    await registry.finalize_providers()
    first_posts = sum(
        1 for c in _FakeAsyncClient.instances for r in c.requests if r[0] == "POST"
    )
    await registry.finalize_providers()
    second_posts = sum(
        1 for c in _FakeAsyncClient.instances for r in c.requests if r[0] == "POST"
    )
    assert first_posts == second_posts == 1
    assert len([p for p in registry.providers if p.vendor == "openrouter"]) == 1


# ------------------------------------------------------------------ path (c)

def test_neither_key_raises_clear_value_error(monkeypatch):
    """Neither key set → same clear ValueError as before (fail closed)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MANAGEMENT_API_KEY", raising=False)

    cfg = _openrouter_config()
    registry = ProviderRegistry(config=cfg)
    route_cfg = cfg["vendors"]["openrouter"]["routes"]["api"]
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        registry._build_route(
            "openrouter", "api", cfg["vendors"]["openrouter"], route_cfg
        )
    # Fail closed: the route is not registered and not deferred for a mint.
    assert registry._deferred_openrouter_routes == []


# ------------------------------------------------------------- public SDK

@pytest.mark.asyncio
async def test_update_provider_key_limit_patches_and_returns_usage(monkeypatch):
    """update_provider_key_limit PATCHes /keys/{hash} then returns usage."""
    from kestrel_sovereign.features.llm_keys import update_provider_key_limit

    usage = await update_provider_key_limit(
        management_key="sk-or-mgmt-abc",
        key_hash="boot-hash",
        limit_usd=250.0,
        reset="monthly",
    )

    reqs = [r for c in _FakeAsyncClient.instances for r in c.requests]
    patches = [r for r in reqs if r[0] == "PATCH"]
    assert patches and patches[0][1] == "/keys/boot-hash"
    assert patches[0][2]["limit"] == pytest.approx(250.0)
    assert patches[0][2]["limit_reset"] == "monthly"
    # get_key_usage GET follows the PATCH.
    assert any(r[0] == "GET" and r[1] == "/keys/boot-hash" for r in reqs)
    assert usage.limit_usd == pytest.approx(250.0)


@pytest.mark.asyncio
async def test_mint_managed_openrouter_key_public_helper(monkeypatch):
    """mint_managed_openrouter_key mints via an explicit management key."""
    from kestrel_sovereign.features.llm_keys import mint_managed_openrouter_key

    info = await mint_managed_openrouter_key(
        management_key="sk-or-mgmt-abc",
        name="frinz-user-42",
        limit_usd=5.0,
    )
    assert info.key == "sk-or-v1-bootstrap-xyz"
    assert info.key_hash == "boot-hash"
    posts = [r for c in _FakeAsyncClient.instances for r in c.requests if r[0] == "POST"]
    assert posts and posts[0][2]["limit"] == pytest.approx(5.0)
