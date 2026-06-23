"""GitHub API proxy and repository discovery endpoints."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from kestrel_sovereign.config import load_section

router = APIRouter(tags=["github"])

_CACHE_TTL_SECONDS = 300
_repo_cache: dict[tuple[Any, ...], tuple[float, list[str]]] = {}


def clear_repo_cache() -> None:
    """Clear the in-process GitHub repo discovery cache."""
    _repo_cache.clear()


def _github_token() -> str | None:
    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_PAT")
    )
    if token:
        return token.strip().strip('"').strip("'")

    for env_path in (Path.cwd() / ".env",):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"}:
                return value.strip().strip('"').strip("'")
    return None


def _list_config(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _github_config() -> dict[str, Any]:
    config = load_section("github")
    return config if isinstance(config, dict) else {}


def _repo_slug(repo: dict[str, Any]) -> str | None:
    full_name = repo.get("full_name")
    if isinstance(full_name, str) and "/" in full_name:
        return full_name

    owner = repo.get("owner")
    owner_login = owner.get("login") if isinstance(owner, dict) else None
    name = repo.get("name")
    if isinstance(owner_login, str) and isinstance(name, str):
        return f"{owner_login}/{name}"
    return None


def _cache_key(
    *,
    token: str,
    orgs: list[str],
    include_private: bool,
    include_repos: list[str],
    exclude_repos: list[str],
) -> tuple[Any, ...]:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return (
        token_hash,
        tuple(sorted(orgs)),
        include_private,
        tuple(sorted(include_repos)),
        tuple(sorted(exclude_repos)),
    )


async def _get_json(client: httpx.AsyncClient, url: str, token: str) -> Any:
    response = await client.get(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "kestrel-host",
        },
        timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
    )
    response.raise_for_status()
    return response.json()


async def _list_org_repos(
    client: httpx.AsyncClient,
    *,
    org: str,
    token: str,
    include_private: bool,
) -> list[str]:
    repo_type = "all" if include_private else "public"
    page = 1
    repos: list[str] = []
    while True:
        data = await _get_json(
            client,
            f"https://api.github.com/orgs/{org}/repos?type={repo_type}&per_page=100&page={page}",
            token,
        )
        if not isinstance(data, list):
            raise HTTPException(
                status_code=502,
                detail="GitHub returned an unexpected repository list",
            )
        if not data:
            break
        for repo in data:
            if isinstance(repo, dict):
                slug = _repo_slug(repo)
                if slug:
                    repos.append(slug)
        if len(data) < 100:
            break
        page += 1
    return repos


async def discover_accessible_repos(
    *,
    org: str | None = None,
    include_private: bool = True,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Discover GitHub repositories visible to the configured server token."""
    token = _github_token()
    if not token:
        raise HTTPException(status_code=503, detail="No GITHUB_TOKEN configured")

    config = _github_config()
    orgs = [org] if org else _list_config(config.get("orgs"))
    if not orgs:
        orgs = ["KestrelSovereignAI"]

    include_repos = _list_config(config.get("include_repos"))
    exclude_repos = set(_list_config(config.get("exclude_repos")))
    key = _cache_key(
        token=token,
        orgs=orgs,
        include_private=include_private,
        include_repos=include_repos,
        exclude_repos=sorted(exclude_repos),
    )

    now = time.monotonic()
    cached = _repo_cache.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return list(cached[1])

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        discovered: list[str] = []
        for org_name in orgs:
            discovered.extend(
                await _list_org_repos(
                    client,
                    org=org_name,
                    token=token,
                    include_private=include_private,
                )
            )

        repos = sorted(
            {
                repo
                for repo in [*discovered, *include_repos]
                if repo and repo not in exclude_repos
            },
            key=str.lower,
        )
        _repo_cache[key] = (now, repos)
        return list(repos)
    finally:
        if owns_client:
            await client.aclose()


@router.get("/api/github/repos")
async def github_repos(
    org: str | None = Query(default=None),
    include_private: bool = Query(default=True),
):
    """Return repo slugs visible to the server-side GitHub token."""
    return await discover_accessible_repos(org=org, include_private=include_private)


@router.get("/api/github/{path:path}")
async def github_proxy(path: str, request: Request):
    """Proxy GitHub API requests using the server-side GitHub token."""
    token = _github_token()
    if not token:
        return JSONResponse({"error": "No GITHUB_TOKEN configured"}, status_code=503)

    gh_url = f"https://api.github.com/{path}"
    if request.url.query:
        gh_url += f"?{request.url.query}"

    client = getattr(request.app.state, "http_client", None)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        response = await client.get(
            gh_url,
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "kestrel-host",
            },
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
        )
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    finally:
        if owns_client:
            await client.aclose()
