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
