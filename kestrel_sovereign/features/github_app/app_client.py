"""
GitHub App authentication and API client.

Uses JWT + installation tokens for authenticated API access.
Handles both REST (issues) and GraphQL (discussions) endpoints.
"""

import base64
import logging
import os
import time
from typing import Optional

import httpx
import jwt

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
JWT_EXPIRY_SECONDS = 600  # 10 minutes (GitHub max)
TOKEN_REFRESH_BUFFER = 60  # Refresh token 1 minute before expiry


class GitHubAppClient:
    """GitHub App API client with JWT and installation token auth."""

    def __init__(self):
        self._app_id = os.environ.get("GITHUB_APP_ID", "")
        self._private_key = self._load_private_key()
        self._webhook_secret = os.environ.get("GITHUB_APP_WEBHOOK_SECRET", "")

        # Cached installation tokens: {installation_id: (token, expiry_timestamp)}
        self._tokens: dict[int, tuple[str, float]] = {}

    @staticmethod
    def _load_private_key() -> str:
        """Load the App private key from env var (raw PEM or base64-encoded)."""
        raw = os.environ.get("GITHUB_APP_PRIVATE_KEY", "")
        if not raw:
            return ""
        # If base64-encoded, decode it
        if not raw.startswith("-----"):
            try:
                return base64.b64decode(raw).decode("utf-8")
            except Exception:
                pass
        return raw

    @property
    def is_configured(self) -> bool:
        return bool(self._app_id and self._private_key)

    @property
    def webhook_secret(self) -> str:
        return self._webhook_secret

    def _generate_jwt(self) -> str:
        """Generate a JWT for GitHub App authentication."""
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Allow for clock drift
            "exp": now + JWT_EXPIRY_SECONDS,
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def _get_installation_token(self, installation_id: int) -> str:
        """Get or refresh an installation access token."""
        cached = self._tokens.get(installation_id)
        if cached:
            token, expiry = cached
            if time.time() < expiry - TOKEN_REFRESH_BUFFER:
                return token

        app_jwt = self._generate_jwt()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        token = data["token"]
        # GitHub tokens expire in 1 hour
        expiry = time.time() + 3600
        self._tokens[installation_id] = (token, expiry)
        logger.info("Refreshed installation token for installation %d", installation_id)
        return token

    async def create_issue_comment(
        self, installation_id: int, repo: str, issue_number: int, body: str
    ) -> dict:
        """Post a comment on a GitHub issue."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/comments",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"body": body},
            )
            resp.raise_for_status()
            return resp.json()

    async def create_discussion_comment(
        self, installation_id: int, discussion_node_id: str, body: str
    ) -> dict:
        """Post a comment on a GitHub Discussion via GraphQL."""
        token = await self._get_installation_token(installation_id)
        mutation = """
        mutation($discussionId: ID!, $body: String!) {
            addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
                comment { id url }
            }
        }
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                GITHUB_GRAPHQL_URL,
                headers={
                    "Authorization": f"bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                json={
                    "query": mutation,
                    "variables": {
                        "discussionId": discussion_node_id,
                        "body": body,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def add_reaction(
        self, installation_id: int, repo: str, issue_number: int, reaction: str = "eyes"
    ) -> None:
        """Add a reaction to an issue to signal processing."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/reactions",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"content": reaction},
            )

    async def get_file_content(
        self, installation_id: int, repo: str, path: str, ref: str = "main"
    ) -> Optional[str]:
        """Read a file from the repo."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
                params={"ref": ref},
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.raw+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if resp.status_code == 200:
                return resp.text
            return None

    async def search_code(
        self, installation_id: int, repo: str, query: str, max_results: int = 10
    ) -> list[dict]:
        """Search code in the repo. Returns list of {path, text_matches}."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GITHUB_API_BASE}/search/code",
                params={"q": f"{query} repo:{repo}", "per_page": max_results},
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.text-match+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("items", [])
            return [
                {
                    "path": item["path"],
                    "text_matches": [
                        m.get("fragment", "") for m in item.get("text_matches", [])
                    ],
                }
                for item in items
            ]

    async def get_directory_listing(
        self, installation_id: int, repo: str, path: str = "", ref: str = "main"
    ) -> list[str]:
        """List files in a directory."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
                params={"ref": ref},
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if resp.status_code != 200:
                return []
            items = resp.json()
            if not isinstance(items, list):
                return []
            return [item["path"] for item in items]

    async def get_repo_tree(
        self, installation_id: int, repo: str, ref: str = "main"
    ) -> list[str]:
        """Get the full file tree of the repo (recursive)."""
        token = await self._get_installation_token(installation_id)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{ref}",
                params={"recursive": "1"},
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if resp.status_code != 200:
                return []
            tree = resp.json().get("tree", [])
            return [item["path"] for item in tree if item["type"] == "blob"]
