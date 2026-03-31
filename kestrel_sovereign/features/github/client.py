"""GitHub API client for repository access."""
import base64
import logging
import os
from typing import Optional
from urllib.parse import quote

import httpx

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_DEFAULT
from .models import FileContent, FileType, RepoFile, SearchResult

logger = logging.getLogger(__name__)


class GitHubClientError(Exception):
    """Error from GitHub API."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class GitHubClient:
    """Client for GitHub REST API."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        key_resolver: Optional["KeyResolutionService"] = None,
    ):
        """Initialize with optional token.

        If no token is provided, the client will be created in a limited mode
        where all API calls return an error explaining that GITHUB_PAT is required.
        This allows the application to start without GITHUB_PAT configured.

        Args:
            token: GitHub Personal Access Token
            key_resolver: Optional KeyResolutionService for dynamic key resolution
        """
        self._key_resolver = key_resolver
        self.token = token or os.getenv("GITHUB_PAT") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self._configured = bool(self.token)

        if self._configured:
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        else:
            self.headers = {}
            logger.warning("GitHub client initialized without token - GitHub features will be unavailable")

        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_configured(self):
        """Ensure client is configured, using key resolver if available."""
        if self._configured:
            return

        # Try key resolver
        if self._key_resolver:
            try:
                token = await self._key_resolver.resolve_key("github", require=False)
                if token:
                    self.token = token
                    self._configured = True
                    self.headers = {
                        "Accept": "application/vnd.github.v3+json",
                        "Authorization": f"Bearer {token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    }
                    # Reset client to pick up new headers
                    if self._client and not self._client.is_closed:
                        await self._client.aclose()
                    self._client = None
                    logger.info("GitHub client configured via key resolver")
                    return
            except Exception as e:
                logger.warning(f"Key resolver failed for GitHub: {e}")

    def _check_configured(self):
        """Check if client is properly configured with a token."""
        if not self._configured:
            raise GitHubClientError(
                "GitHub feature not available: No GITHUB_PAT, GITHUB_TOKEN, or GH_TOKEN environment variable set. "
                "Contact your administrator to enable GitHub integration.",
                status_code=503
            )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        await self._ensure_configured()
        self._check_configured()
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=HTTP_TIMEOUT_DEFAULT,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _parse_repo(self, repo: str) -> tuple[str, str]:
        """Parse owner/repo string."""
        if "/" not in repo:
            raise GitHubClientError(f"Invalid repo format: {repo}. Expected 'owner/repo'.")
        parts = repo.split("/", 1)
        return parts[0], parts[1]
    
    async def get_file_content(
        self,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> FileContent:
        """Get content of a file from repository.
        
        Args:
            repo: Repository in 'owner/repo' format
            path: Path to file within repository
            ref: Branch, tag, or commit SHA
            
        Returns:
            FileContent with decoded content
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()
        
        # URL encode the path
        encoded_path = quote(path, safe="")
        url = f"/repos/{owner}/{repo_name}/contents/{encoded_path}"
        
        response = await client.get(url, params={"ref": ref})
        
        if response.status_code == 404:
            raise GitHubClientError(f"File not found: {path} in {repo}", 404)
        elif response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)
        
        data = response.json()
        
        if data.get("type") != "file":
            raise GitHubClientError(f"Path is not a file: {path}")
        
        # Decode base64 content
        content_b64 = data.get("content", "")
        try:
            content = base64.b64decode(content_b64).decode("utf-8")
        except Exception as e:
            raise GitHubClientError(f"Failed to decode file content: {e}")
        
        return FileContent(
            path=path,
            content=content,
            sha=data.get("sha", ""),
            size=data.get("size", 0),
            encoding="utf-8",
            repo=repo,
            ref=ref,
        )
    
    async def list_directory(
        self,
        repo: str,
        path: str = "",
        ref: str = "main",
    ) -> list[RepoFile]:
        """List contents of a directory.
        
        Args:
            repo: Repository in 'owner/repo' format
            path: Path to directory (empty for root)
            ref: Branch, tag, or commit SHA
            
        Returns:
            List of files and directories
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()
        
        # URL encode the path
        encoded_path = quote(path, safe="") if path else ""
        url = f"/repos/{owner}/{repo_name}/contents/{encoded_path}"
        
        response = await client.get(url, params={"ref": ref})
        
        if response.status_code == 404:
            raise GitHubClientError(f"Path not found: {path} in {repo}", 404)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)
        
        data = response.json()
        
        # Handle single file case
        if isinstance(data, dict):
            return [self._parse_repo_file(data)]
        
        # Parse directory listing
        return [self._parse_repo_file(item) for item in data]
    
    def _parse_repo_file(self, data: dict) -> RepoFile:
        """Parse API response into RepoFile."""
        file_type = FileType.FILE
        type_str = data.get("type", "file")
        if type_str == "dir":
            file_type = FileType.DIR
        elif type_str == "symlink":
            file_type = FileType.SYMLINK
        elif type_str == "submodule":
            file_type = FileType.SUBMODULE
        
        return RepoFile(
            path=data.get("path", ""),
            name=data.get("name", ""),
            type=file_type,
            size=data.get("size", 0),
            sha=data.get("sha", ""),
            download_url=data.get("download_url"),
        )
    
    async def search_code(
        self,
        query: str,
        repo: Optional[str] = None,
        path: Optional[str] = None,
        extension: Optional[str] = None,
        max_results: int = 30,
    ) -> list[SearchResult]:
        """Search for code in repositories.
        
        Args:
            query: Search query
            repo: Limit to specific repo (owner/repo format)
            path: Limit to path prefix
            extension: Limit to file extension
            max_results: Maximum results to return
            
        Returns:
            List of search results
        """
        client = await self._get_client()
        
        # Build search query
        q_parts = [query]
        if repo:
            q_parts.append(f"repo:{repo}")
        if path:
            q_parts.append(f"path:{path}")
        if extension:
            q_parts.append(f"extension:{extension}")
        
        q = " ".join(q_parts)
        
        response = await client.get(
            "/search/code",
            params={
                "q": q,
                "per_page": min(max_results, 100),
            },
            headers={
                **self.headers,
                "Accept": "application/vnd.github.text-match+json",
            },
        )
        
        if response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code == 422:
            raise GitHubClientError(f"Invalid search query: {query}", 422)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)
        
        data = response.json()
        items = data.get("items", [])
        
        results = []
        for item in items[:max_results]:
            repo_info = item.get("repository", {})
            results.append(SearchResult(
                path=item.get("path", ""),
                repo=repo_info.get("full_name", ""),
                name=item.get("name", ""),
                sha=item.get("sha", ""),
                score=item.get("score", 0.0),
                html_url=item.get("html_url", ""),
                text_matches=item.get("text_matches", []),
            ))
        
        return results
    
    async def get_repo_info(self, repo: str) -> dict:
        """Get repository metadata.
        
        Args:
            repo: Repository in 'owner/repo' format
            
        Returns:
            Repository metadata dict
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()
        
        response = await client.get(f"/repos/{owner}/{repo_name}")
        
        if response.status_code == 404:
            raise GitHubClientError(f"Repository not found: {repo}", 404)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)
        
        return response.json()
    
    async def get_tree(
        self,
        repo: str,
        ref: str = "main",
        recursive: bool = True,
    ) -> list[RepoFile]:
        """Get full repository tree.
        
        More efficient than recursive list_directory calls.
        
        Args:
            repo: Repository in 'owner/repo' format
            ref: Branch, tag, or commit SHA
            recursive: Whether to get full tree recursively
            
        Returns:
            List of all files in repository
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()
        
        url = f"/repos/{owner}/{repo_name}/git/trees/{ref}"
        params = {"recursive": "1"} if recursive else {}
        
        response = await client.get(url, params=params)
        
        if response.status_code == 404:
            raise GitHubClientError(f"Tree not found: {ref} in {repo}", 404)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)
        
        data = response.json()
        tree = data.get("tree", [])
        
        results = []
        for item in tree:
            file_type = FileType.FILE if item.get("type") == "blob" else FileType.DIR
            if item.get("type") == "commit":
                file_type = FileType.SUBMODULE
            
            results.append(RepoFile(
                path=item.get("path", ""),
                name=item.get("path", "").split("/")[-1],
                type=file_type,
                size=item.get("size", 0),
                sha=item.get("sha", ""),
            ))

        return results

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
    ) -> dict:
        """Create a GitHub issue.

        Args:
            repo: Repository in 'owner/repo' format
            title: Issue title
            body: Issue body (markdown supported)
            labels: Optional list of label names
            assignees: Optional list of GitHub usernames to assign

        Returns:
            Created issue data including 'html_url', 'number', 'id'

        Raises:
            GitHubClientError: If issue creation fails
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()

        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees

        response = await client.post(
            f"/repos/{owner}/{repo_name}/issues",
            json=data,
        )

        if response.status_code == 201:
            result = response.json()
            logger.info(f"Created issue #{result.get('number')} in {repo}: {title}")
            return result
        elif response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code == 404:
            raise GitHubClientError(f"Repository not found or no permission: {repo}", 404)
        elif response.status_code == 410:
            raise GitHubClientError("Issues are disabled for this repository", 410)
        elif response.status_code == 422:
            raise GitHubClientError(f"Validation failed: {response.text}", 422)
        else:
            raise GitHubClientError(f"Issue creation failed: {response.text}", response.status_code)

    async def list_issues(
        self,
        repo: str,
        state: str = "open",
        labels: Optional[list[str]] = None,
        per_page: int = 30,
    ) -> list[dict]:
        """List issues in a repository.

        Args:
            repo: Repository in 'owner/repo' format
            state: Issue state filter ('open', 'closed', 'all')
            labels: Optional list of label names to filter by
            per_page: Number of results per page (max 100)

        Returns:
            List of issue dicts (excludes pull requests)

        Raises:
            GitHubClientError: If request fails
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()

        params = {
            "state": state,
            "per_page": min(per_page, 100),
        }
        if labels:
            params["labels"] = ",".join(labels)

        response = await client.get(
            f"/repos/{owner}/{repo_name}/issues",
            params=params,
        )

        if response.status_code == 404:
            raise GitHubClientError(f"Repository not found: {repo}", 404)
        elif response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)

        # GitHub's issues endpoint also returns PRs; filter them out
        items = response.json()
        return [i for i in items if "pull_request" not in i]

    async def get_issue(
        self,
        repo: str,
        issue_number: int,
    ) -> dict:
        """Get a specific issue by number.

        Args:
            repo: Repository in 'owner/repo' format
            issue_number: Issue number

        Returns:
            Issue data dict

        Raises:
            GitHubClientError: If request fails
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()

        response = await client.get(
            f"/repos/{owner}/{repo_name}/issues/{issue_number}",
        )

        if response.status_code == 404:
            raise GitHubClientError(f"Issue #{issue_number} not found in {repo}", 404)
        elif response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)

        return response.json()

    async def get_issue_comments(
        self,
        repo: str,
        issue_number: int,
        per_page: int = 30,
    ) -> list[dict]:
        """Get comments on an issue.

        Args:
            repo: Repository in 'owner/repo' format
            issue_number: Issue number
            per_page: Number of results per page (max 100)

        Returns:
            List of comment dicts

        Raises:
            GitHubClientError: If request fails
        """
        owner, repo_name = self._parse_repo(repo)
        client = await self._get_client()

        response = await client.get(
            f"/repos/{owner}/{repo_name}/issues/{issue_number}/comments",
            params={"per_page": min(per_page, 100)},
        )

        if response.status_code == 404:
            raise GitHubClientError(f"Issue #{issue_number} not found in {repo}", 404)
        elif response.status_code == 403:
            raise GitHubClientError("Rate limited or access denied", 403)
        elif response.status_code != 200:
            raise GitHubClientError(f"GitHub API error: {response.text}", response.status_code)

        return response.json()
