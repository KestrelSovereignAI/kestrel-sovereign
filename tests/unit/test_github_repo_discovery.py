import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.endpoints import github as github_endpoints


@pytest.fixture(autouse=True)
def clear_cache():
    github_endpoints.clear_repo_cache()
    yield
    github_endpoints.clear_repo_cache()


@pytest.mark.asyncio
async def test_discovers_org_repos_with_configured_includes_and_excludes(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(
        github_endpoints,
        "load_section",
        lambda section: {
            "orgs": ["KestrelSovereignAI"],
            "include_repos": ["jaslogic1/RemoteCares"],
            "exclude_repos": ["KestrelSovereignAI/private-scratch"],
        },
        raising=False,
    )

    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert request.headers["Authorization"] == "token ghp_test"
        assert request.url.params["type"] == "all"
        return httpx.Response(
            200,
            json=[
                {"full_name": "KestrelSovereignAI/kestrel-sovereign"},
                {"full_name": "KestrelSovereignAI/private-scratch"},
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repos = await github_endpoints.discover_accessible_repos(client=client)

    assert repos == [
        "jaslogic1/RemoteCares",
        "KestrelSovereignAI/kestrel-sovereign",
    ]
    expected_url = (
        "https://api.github.com/orgs/KestrelSovereignAI/repos"
        "?type=all&per_page=100&page=1"
    )
    assert seen_urls == [
        expected_url,
    ]


@pytest.mark.asyncio
async def test_include_private_false_requests_public_org_repos(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(
        github_endpoints,
        "load_section",
        lambda section: {"orgs": ["KestrelSovereignAI"]},
        raising=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["type"] == "public"
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repos = await github_endpoints.discover_accessible_repos(
            include_private=False,
            client=client,
        )

    assert repos == []


@pytest.mark.asyncio
async def test_repo_discovery_cache_avoids_repeat_github_calls(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(
        github_endpoints,
        "load_section",
        lambda section: {"orgs": ["KestrelSovereignAI"]},
        raising=False,
    )

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json=[{"full_name": "KestrelSovereignAI/kestrel-sovereign"}],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await github_endpoints.discover_accessible_repos(client=client)
        second = await github_endpoints.discover_accessible_repos(client=client)

    assert first == ["KestrelSovereignAI/kestrel-sovereign"]
    assert second == first
    assert call_count == 1


@pytest.mark.asyncio
async def test_github_repos_endpoint_returns_plain_slug_list(monkeypatch):
    app = FastAPI()
    app.include_router(github_endpoints.router)

    async def fake_discover(*, org=None, include_private=True, client=None):
        assert org == "KestrelSovereignAI"
        assert include_private is True
        return ["KestrelSovereignAI/kestrel-sovereign"]

    monkeypatch.setattr(github_endpoints, "discover_accessible_repos", fake_discover)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/github/repos",
            params={"org": "KestrelSovereignAI", "include_private": "true"},
        )

    assert response.status_code == 200
    assert response.json() == ["KestrelSovereignAI/kestrel-sovereign"]


def _scoped_proxy_app(monkeypatch, *, handler, config):
    """Build a FastAPI app whose proxy uses a mock-backed shared client."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(
        github_endpoints,
        "load_section",
        lambda section: config,
        raising=False,
    )
    app = FastAPI()
    app.include_router(github_endpoints.router)
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


_ORG_REPOS = [
    {"full_name": "KestrelSovereignAI/kestrel-sovereign"},
    {"full_name": "KestrelSovereignAI/private-scratch"},
]


@pytest.mark.asyncio
async def test_proxy_forwards_allowed_repo_path(monkeypatch):
    forwarded = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/orgs/" in url:
            return httpx.Response(200, json=_ORG_REPOS)
        forwarded.append(url)
        assert request.headers["Authorization"] == "token ghp_test"
        return httpx.Response(200, json=[{"number": 1, "title": "hello"}])

    app = _scoped_proxy_app(
        monkeypatch,
        handler=handler,
        config={"orgs": ["KestrelSovereignAI"], "exclude_repos": ["KestrelSovereignAI/private-scratch"]},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/github/repos/KestrelSovereignAI/kestrel-sovereign/issues",
            params={"state": "open"},
        )

    assert response.status_code == 200
    assert response.json() == [{"number": 1, "title": "hello"}]
    assert forwarded == [
        "https://api.github.com/repos/KestrelSovereignAI/kestrel-sovereign/issues?state=open"
    ]
    await app.state.http_client.aclose()


@pytest.mark.asyncio
async def test_proxy_rejects_excluded_repo(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "/orgs/" in str(request.url):
            return httpx.Response(200, json=_ORG_REPOS)
        raise AssertionError("excluded repo must not be proxied to GitHub")

    app = _scoped_proxy_app(
        monkeypatch,
        handler=handler,
        config={"orgs": ["KestrelSovereignAI"], "exclude_repos": ["KestrelSovereignAI/private-scratch"]},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/github/repos/KestrelSovereignAI/private-scratch/issues"
        )

    assert response.status_code == 403
    assert "outside the configured GitHub scope" in response.json()["error"]
    await app.state.http_client.aclose()


@pytest.mark.asyncio
async def test_proxy_rejects_repo_not_in_scope(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "/orgs/" in str(request.url):
            return httpx.Response(200, json=_ORG_REPOS)
        raise AssertionError("out-of-scope repo must not be proxied to GitHub")

    app = _scoped_proxy_app(
        monkeypatch,
        handler=handler,
        config={"orgs": ["KestrelSovereignAI"]},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/github/repos/someoneelse/secret/issues")

    assert response.status_code == 403
    assert "outside the configured GitHub scope" in response.json()["error"]
    await app.state.http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "users/octocat/repos",
        "orgs/KestrelSovereignAI/repos",
        "search/issues",
        "user",
        "repos/KestrelSovereignAI",
        # Dot-segment traversal: httpx would normalize this to /user.
        "repos/KestrelSovereignAI/kestrel-sovereign/../../../user",
        "repos/KestrelSovereignAI/kestrel-sovereign/%2e%2e/%2e%2e/user",
    ],
)
async def test_proxy_rejects_non_repo_scoped_paths(monkeypatch, path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("non-repo path must never reach GitHub")

    app = _scoped_proxy_app(
        monkeypatch,
        handler=handler,
        config={"orgs": ["KestrelSovereignAI"]},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(f"/api/github/{path}")

    assert response.status_code == 403
    assert "repos/{owner}/{repo}" in response.json()["error"]
    await app.state.http_client.aclose()


@pytest.mark.asyncio
async def test_proxy_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setattr(
        github_endpoints, "_github_token", lambda: None, raising=False
    )
    app = FastAPI()
    app.include_router(github_endpoints.router)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/github/repos/KestrelSovereignAI/kestrel-sovereign/issues"
        )

    assert response.status_code == 503
