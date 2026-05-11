"""CLI feature: registered feature-owned command-line adapters."""

from __future__ import annotations

from typing import Any

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool

from .adapters import CliAdapterError, GitCliAdapter, GitHubCliAdapter
from .terminal import CliRisk, TerminalExecutionService


class CliFeature(Feature):
    """Expose feature-owned CLI adapters without opening arbitrary shell access."""

    tool_name = "cli"

    def __init__(self, agent=None):
        super().__init__(agent)
        self.terminal = TerminalExecutionService()
        self.adapters = {
            "github": GitHubCliAdapter(self.terminal),
            "git": GitCliAdapter(self.terminal),
        }

    @property
    def tool_description(self) -> str:
        return (
            "Run registered, feature-owned CLI workflows using installed tools "
            "and the user's existing CLI authentication. This is not arbitrary shell access."
        )

    async def initialize(self) -> None:
        return None

    @tool(
        name="cli_status",
        description="Show platform metadata and registered CLI adapter tool availability.",
        category=ToolCategory.SYSTEM,
        command_prefix="!cli-status",
    )
    async def cli_status(self) -> ToolResult:
        reports = {}
        for adapter_id, adapter in self.adapters.items():
            report = await adapter.check_availability()
            reports[adapter_id] = {
                "available": report.available,
                "tools": report.tools,
                "commands": [
                    {
                        "id": command.command_id,
                        "description": command.description,
                        "risk": command.risk.value,
                        "tools": list(command.tools),
                    }
                    for command in adapter.commands
                ],
            }
        return ToolResult.ok(
            "CLI adapter status collected.",
            data={
                "platform": self.terminal.platform_metadata(),
                "adapters": reports,
            },
        )

    @tool(
        name="github_cli_auth_status",
        description="Check GitHub CLI authentication state through `gh auth status`.",
        category=ToolCategory.SYSTEM,
        command_prefix="!gh-auth-status",
    )
    async def github_cli_auth_status(self, hostname: str = "github.com") -> ToolResult:
        result = await self.adapters["github"].auth_status(hostname=hostname)
        data = _command_data(result)
        status = "authenticated" if result.ok else "not_authenticated"
        if result.ok:
            return ToolResult.ok(
                "GitHub CLI authentication is available.",
                data={**data, "status": status},
        )
        return ToolResult.partial(
            "GitHub CLI authentication check did not pass.",
            result.redacted_stderr or "gh auth status returned non-zero",
            data={**data, "status": status},
        )

    @tool(
        name="github_pr_view",
        description=(
            "Inspect GitHub PR metadata, changed files, commits, head ref, "
            "and checks via `gh pr view`."
        ),
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-pr-view",
    )
    async def github_pr_view(self, repo: str, number: int) -> ToolResult:
        try:
            payload = await self.adapters["github"].get_pull_request(repo=repo, number=number)
        except CliAdapterError as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read PR #{number} from {repo}.",
            data=payload,
        )

    @tool(
        name="github_pr_diff",
        description="Read a GitHub PR unified diff via `gh pr diff`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-pr-diff",
    )
    async def github_pr_diff(self, repo: str, number: int) -> ToolResult:
        try:
            diff = await self.adapters["github"].get_pull_request_diff(repo=repo, number=number)
        except CliAdapterError as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read diff for PR #{number} from {repo}.",
            data={
                "repo": repo,
                "number": number,
                "diff": diff,
            },
        )

    @tool(
        name="github_pr_files",
        description="List changed files for a GitHub PR via `gh pr view`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-pr-files",
    )
    async def github_pr_files(self, repo: str, number: int) -> ToolResult:
        try:
            files = await self.adapters["github"].list_pull_request_files(
                repo=repo,
                number=number,
            )
        except CliAdapterError as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {len(files)} changed file(s) for PR #{number} from {repo}.",
            data={
                "repo": repo,
                "number": number,
                "files": files,
            },
        )

    @tool(
        name="github_pr_checks",
        description="Read check/status rollup for a GitHub PR via `gh pr view`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-pr-checks",
    )
    async def github_pr_checks(self, repo: str, number: int) -> ToolResult:
        try:
            checks = await self.adapters["github"].get_pull_request_checks(
                repo=repo,
                number=number,
            )
        except CliAdapterError as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {len(checks)} check(s) for PR #{number} from {repo}.",
            data={
                "repo": repo,
                "number": number,
                "checks": checks,
            },
        )

    @tool(
        name="github_read_file_at_ref",
        description="Read a GitHub repository file at a branch, tag, or commit ref via `gh api`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-read-file",
    )
    async def github_read_file_at_ref(self, repo: str, path: str, ref: str) -> ToolResult:
        try:
            payload = await self.adapters["github"].read_file_at_ref(repo=repo, path=path, ref=ref)
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {path} at {ref} from {repo}.",
            data=payload,
        )

    @tool(
        name="github_read_file_at_pr_head",
        description="Read a GitHub repository file at a pull request head commit.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-read-pr-file",
    )
    async def github_read_file_at_pr_head(
        self,
        repo: str,
        number: int,
        path: str,
    ) -> ToolResult:
        try:
            payload = await self.adapters["github"].read_file_at_pull_request_head(
                repo=repo,
                number=number,
                path=path,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {path} from PR #{number} head in {repo}.",
            data=payload,
        )

    @tool(
        name="github_pr_review_context",
        description="Build GitHub PR review context with metadata, files, checks, and diff.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!gh-pr-context",
    )
    async def github_pr_review_context(
        self,
        repo: str,
        number: int,
        include_file_contents: bool = False,
        max_files: int = 10,
        max_file_bytes: int = 20_000,
    ) -> ToolResult:
        try:
            context = await self.adapters["github"].get_pull_request_review_context(
                repo=repo,
                number=number,
                include_file_contents=include_file_contents,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Built review context for PR #{number} from {repo}.",
            data=context,
        )

    @tool(
        name="git_status",
        description="Read local repository status via `git status --short --branch`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-status",
    )
    async def git_status(self, repo_path: str = ".") -> ToolResult:
        try:
            payload = await self.adapters["git"].status(repo_path=repo_path)
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git status for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_diff",
        description="Read local repository diff via `git diff`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-diff",
    )
    async def git_diff(
        self,
        ref: str = "",
        path: str = "",
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].diff(
                repo_path=repo_path,
                ref=ref,
                path=path,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git diff for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_log",
        description="Read recent local repository commits via `git log`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-log",
    )
    async def git_log(self, max_count: int = 20, repo_path: str = ".") -> ToolResult:
        try:
            payload = await self.adapters["git"].log(
                repo_path=repo_path,
                max_count=max_count,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git log for {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_show_file",
        description="Read a local repository file from a git ref via `git show`.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-show-file",
    )
    async def git_show_file(
        self,
        ref: str,
        path: str,
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].show_file(
                repo_path=repo_path,
                ref=ref,
                path=path,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read {path} at {ref} from {payload['repo_path']}.",
            data=payload,
        )

    @tool(
        name="git_merge_base",
        description="Read the merge-base for two local git refs.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!git-merge-base",
    )
    async def git_merge_base(
        self,
        left_ref: str,
        right_ref: str,
        repo_path: str = ".",
    ) -> ToolResult:
        try:
            payload = await self.adapters["git"].merge_base(
                repo_path=repo_path,
                left_ref=left_ref,
                right_ref=right_ref,
            )
        except (CliAdapterError, ValueError) as exc:
            return ToolResult.failed(error=str(exc))
        return ToolResult.ok(
            f"Read git merge-base for {payload['repo_path']}.",
            data=payload,
        )


def _command_data(result: Any) -> dict[str, Any]:
    return {
        "argv": result.argv,
        "risk": CliRisk.READ_ONLY.value,
        "returncode": result.returncode,
        "stdout": result.redacted_stdout,
        "stderr": result.redacted_stderr,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
    }
