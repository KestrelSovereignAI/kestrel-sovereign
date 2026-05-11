"""Feature-owned CLI adapters."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .terminal import (
    CliCommandDefinition,
    CliRisk,
    CliToolDeclaration,
    TerminalCommandRequest,
    TerminalCommandResult,
    TerminalExecutionService,
    decode_github_content_response,
    redact_secrets,
)


_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@~^+-]{0,255}$")
_GIT_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SYSTEMROOT",
    "WINDIR",
)


class CliAdapterError(RuntimeError):
    """Raised when a registered CLI adapter command fails."""


@dataclass
class CliAvailabilityReport:
    adapter_id: str
    tools: dict[str, dict[str, Any]]

    @property
    def available(self) -> bool:
        return all(
            tool["available"] for tool in self.tools.values() if tool["required"]
        )


class FeatureCliAdapter:
    """Base contract for feature-specific CLI adapters."""

    adapter_id = "base"
    tools: tuple[CliToolDeclaration, ...] = ()
    commands: tuple[CliCommandDefinition, ...] = ()

    def __init__(self, terminal: TerminalExecutionService):
        self.terminal = terminal

    async def check_availability(self) -> CliAvailabilityReport:
        statuses: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            availability = await self.terminal.which(tool.name)
            statuses[tool.name] = {
                "required": tool.required,
                "available": availability.available,
                "path": availability.path,
                "version": availability.version,
            }
        return CliAvailabilityReport(adapter_id=self.adapter_id, tools=statuses)


class GitHubCliAdapter(FeatureCliAdapter):
    """GitHub PR introspection through the authenticated `gh` CLI."""

    adapter_id = "github"
    tools = (
        CliToolDeclaration("gh", required=True),
    )
    commands = (
        CliCommandDefinition(
            "github.auth_status",
            "Check GitHub CLI authentication state without reading credential files.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.pr_view",
            "Read pull request metadata, files, commits, and status rollup.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.issue_view",
            "Read issue metadata, labels, assignees, and state.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.issue_comments",
            "Read issue comments.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.pr_diff",
            "Read a pull request unified diff.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.read_file_at_ref",
            "Read a repository file at a branch, tag, or commit ref.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.pr_files",
            "List changed files for a pull request.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.pr_checks",
            "Read check/status rollup for a pull request.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.read_file_at_pr_head",
            "Read a repository file at a pull request head commit.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
        CliCommandDefinition(
            "github.pr_review_context",
            "Build a compact review context from PR metadata, files, checks, and diff.",
            CliRisk.READ_ONLY,
            ("gh",),
        ),
    )

    async def auth_status(self, *, hostname: str = "github.com") -> TerminalCommandResult:
        return await self.terminal.run(
            TerminalCommandRequest(
                argv=["gh", "auth", "status", "--hostname", hostname],
                timeout=20,
                risk=CliRisk.READ_ONLY,
                command_id="github.auth_status",
            )
        )

    async def get_pull_request(self, *, repo: str, number: int | str) -> dict[str, Any]:
        repo = _validate_repo(repo)
        number = _validate_pr_number(number)
        fields = [
            "additions",
            "author",
            "baseRefName",
            "body",
            "changedFiles",
            "commits",
            "deletions",
            "files",
            "headRefName",
            "headRefOid",
            "isDraft",
            "mergeable",
            "number",
            "state",
            "statusCheckRollup",
            "title",
            "url",
        ]
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=[
                    "gh",
                    "pr",
                    "view",
                    "--repo",
                    repo,
                    str(number),
                    "--json",
                    ",".join(fields),
                ],
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id="github.pr_view",
            )
        )
        return redact_json(_json_or_raise(result))

    async def get_issue(self, *, repo: str, number: int | str) -> dict[str, Any]:
        repo = _validate_repo(repo)
        number = _validate_issue_number(number)
        fields = [
            "assignees",
            "author",
            "body",
            "closed",
            "closedAt",
            "comments",
            "createdAt",
            "labels",
            "milestone",
            "number",
            "state",
            "title",
            "updatedAt",
            "url",
        ]
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=[
                    "gh",
                    "issue",
                    "view",
                    "--repo",
                    repo,
                    str(number),
                    "--json",
                    ",".join(fields),
                ],
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id="github.issue_view",
            )
        )
        return redact_json(_json_or_raise(result))

    async def list_issue_comments(
        self, *, repo: str, number: int | str
    ) -> list[dict[str, Any]]:
        issue = await self.get_issue(repo=repo, number=number)
        comments = issue.get("comments", [])
        if not isinstance(comments, list):
            raise CliAdapterError("issue comments payload was not a list")
        return comments

    async def get_pull_request_diff(self, *, repo: str, number: int | str) -> str:
        repo = _validate_repo(repo)
        number = _validate_pr_number(number)
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=["gh", "pr", "diff", "--repo", repo, str(number)],
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id="github.pr_diff",
            )
        )
        if not result.ok:
            raise CliAdapterError(
                result.redacted_stderr or f"gh pr diff exited {result.returncode}"
            )
        if result.truncated_stdout or result.truncated_stderr:
            raise CliAdapterError("gh pr diff output exceeded the capture limit")
        return result.redacted_stdout

    async def read_file_at_ref(self, *, repo: str, path: str, ref: str) -> dict[str, Any]:
        owner_repo = _validate_repo(repo)
        safe_path = _validate_repo_path(path)
        safe_ref = _validate_ref(ref)
        encoded_path = "/".join(quote(part, safe="") for part in safe_path.split("/"))
        endpoint = (
            f"repos/{owner_repo}/contents/{encoded_path}"
            f"?ref={quote(safe_ref, safe='')}"
        )
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=["gh", "api", endpoint],
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id="github.read_file_at_ref",
            )
        )
        payload = _json_or_raise(result)
        return {
            "repo": repo,
            "path": safe_path,
            "ref": safe_ref,
            "sha": payload.get("sha"),
            "size": payload.get("size"),
            "content": redact_secrets(decode_github_content_response(payload)),
        }

    async def list_pull_request_files(
        self, *, repo: str, number: int | str
    ) -> list[dict[str, Any]]:
        payload = await self.get_pull_request(repo=repo, number=number)
        files = payload.get("files", [])
        if not isinstance(files, list):
            raise CliAdapterError("pull request files payload was not a list")
        return files

    async def get_pull_request_checks(
        self, *, repo: str, number: int | str
    ) -> list[dict[str, Any]]:
        payload = await self.get_pull_request(repo=repo, number=number)
        checks = payload.get("statusCheckRollup", [])
        if not isinstance(checks, list):
            raise CliAdapterError("pull request checks payload was not a list")
        return checks

    async def read_file_at_pull_request_head(
        self,
        *,
        repo: str,
        number: int | str,
        path: str,
    ) -> dict[str, Any]:
        payload = await self.get_pull_request(repo=repo, number=number)
        head_ref = payload.get("headRefOid")
        if not isinstance(head_ref, str) or not head_ref:
            raise CliAdapterError("pull request head ref is unavailable")
        return await self.read_file_at_ref(repo=repo, path=path, ref=head_ref)

    async def get_pull_request_review_context(
        self,
        *,
        repo: str,
        number: int | str,
        include_file_contents: bool | str = False,
        max_files: int | str = 10,
        max_file_bytes: int | str = 20_000,
    ) -> dict[str, Any]:
        include_file_contents = _validate_bool(
            include_file_contents,
            "include_file_contents",
        )
        max_files = _validate_non_negative_int(max_files, "max_files")
        max_file_bytes = _validate_positive_int(max_file_bytes, "max_file_bytes")
        pr = await self.get_pull_request(repo=repo, number=number)
        diff = await self.get_pull_request_diff(repo=repo, number=number)

        file_contents: list[dict[str, Any]] = []
        if include_file_contents:
            head_ref = pr.get("headRefOid")
            if not isinstance(head_ref, str) or not head_ref:
                raise CliAdapterError("pull request head ref is unavailable")
            for file_info in _reviewable_files(pr.get("files", []), limit=max_files):
                content = await self.read_file_at_ref(
                    repo=repo,
                    path=file_info["path"],
                    ref=head_ref,
                )
                text = content["content"]
                encoded = text.encode("utf-8")
                truncated = len(encoded) > max_file_bytes
                if truncated:
                    text = encoded[:max_file_bytes].decode("utf-8", errors="replace")
                file_contents.append(
                    {
                        **content,
                        "content": text,
                        "truncated": truncated,
                    }
                )

        return {
            "repo": _validate_repo(repo),
            "number": _validate_pr_number(number),
            "pull_request": pr,
            "files": pr.get("files", []),
            "checks": pr.get("statusCheckRollup", []),
            "diff": diff,
            "file_contents": file_contents,
        }


class GitCliAdapter(FeatureCliAdapter):
    """Read-only local repository inspection through `git`."""

    adapter_id = "git"
    tools = (
        CliToolDeclaration("git", required=True),
    )
    commands = (
        CliCommandDefinition(
            "git.status",
            "Read local repository status using `git status --short --branch`.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.diff",
            "Read local repository diff output.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.log",
            "Read recent local repository commit history.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.show_file",
            "Read a file from a local repository ref.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
        CliCommandDefinition(
            "git.merge_base",
            "Read the merge-base for two local refs.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
    )

    async def status(self, *, repo_path: str = ".") -> dict[str, Any]:
        result = await self._run_text(
            repo_path=repo_path,
            argv=["status", "--short", "--branch"],
            command_id="git.status",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "status": result,
        }

    async def diff(
        self,
        *,
        repo_path: str = ".",
        ref: str = "",
        path: str = "",
    ) -> dict[str, Any]:
        argv = ["diff", "--no-ext-diff", "--no-textconv"]
        safe_ref = ""
        if ref:
            safe_ref = _validate_git_ref(ref, name="ref")
            argv.append(safe_ref)
        safe_path = ""
        if path:
            safe_path = _validate_git_pathspec(path)
            argv.extend(["--", safe_path])

        result = await self._run_text(
            repo_path=repo_path,
            argv=argv,
            command_id="git.diff",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "ref": safe_ref,
            "path": safe_path,
            "diff": result,
        }

    async def log(
        self,
        *,
        repo_path: str = ".",
        max_count: int | str = 20,
    ) -> dict[str, Any]:
        count = _validate_positive_int(max_count, "max_count")
        result = await self._run_text(
            repo_path=repo_path,
            argv=[
                "log",
                "--oneline",
                "--decorate",
                "--max-count",
                str(min(count, 100)),
            ],
            command_id="git.log",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "max_count": min(count, 100),
            "log": result,
        }

    async def show_file(
        self,
        *,
        repo_path: str = ".",
        ref: str = "HEAD",
        path: str,
    ) -> dict[str, Any]:
        safe_ref = _validate_git_ref(ref, name="ref")
        safe_path = _validate_git_pathspec(path)
        result = await self._run_text(
            repo_path=repo_path,
            argv=["show", f"{safe_ref}:{safe_path}"],
            command_id="git.show_file",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "ref": safe_ref,
            "path": safe_path,
            "content": result,
        }

    async def merge_base(
        self,
        *,
        repo_path: str = ".",
        left_ref: str,
        right_ref: str,
    ) -> dict[str, Any]:
        safe_left = _validate_git_ref(left_ref, name="left_ref")
        safe_right = _validate_git_ref(right_ref, name="right_ref")
        result = await self._run_text(
            repo_path=repo_path,
            argv=["merge-base", safe_left, safe_right],
            command_id="git.merge_base",
        )
        return {
            "repo_path": str(_validate_local_repo_path(repo_path)),
            "left_ref": safe_left,
            "right_ref": safe_right,
            "merge_base": result.strip(),
        }

    async def _run_text(
        self,
        *,
        repo_path: str,
        argv: list[str],
        command_id: str,
    ) -> str:
        safe_repo_path = _validate_local_repo_path(repo_path)
        result = await self.terminal.run(
            TerminalCommandRequest(
                argv=["git", "--no-optional-locks", "-C", str(safe_repo_path), *argv],
                env=_git_env(),
                timeout=60,
                risk=CliRisk.READ_ONLY,
                command_id=command_id,
            )
        )
        if not result.ok:
            raise CliAdapterError(
                result.redacted_stderr or f"git command exited {result.returncode}"
            )
        if result.truncated_stdout or result.truncated_stderr:
            raise CliAdapterError("git command output exceeded the capture limit")
        return result.redacted_stdout


def _json_or_raise(result: TerminalCommandResult) -> dict[str, Any]:
    if not result.ok:
        raise CliAdapterError(
            result.redacted_stderr or f"command exited {result.returncode}"
        )
    if result.truncated_stdout or result.truncated_stderr:
        raise CliAdapterError("command output exceeded the capture limit")
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CliAdapterError(f"command did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CliAdapterError("command JSON payload was not an object")
    return parsed


def _validate_repo(repo: str) -> str:
    if not isinstance(repo, str):
        raise CliAdapterError("repo must be in owner/name form")
    parts = repo.split("/")
    if len(parts) != 2:
        raise CliAdapterError("repo must be in owner/name form")
    owner, name = parts
    if not _OWNER_RE.fullmatch(owner):
        raise CliAdapterError("repo owner is invalid")
    if not _REPO_NAME_RE.fullmatch(name) or name in (".", ".."):
        raise CliAdapterError("repo name is invalid")
    return repo


def _validate_pr_number(number: int | str) -> int:
    if isinstance(number, bool):
        raise CliAdapterError("pull request number must be a positive integer")
    try:
        parsed = int(number)
    except (TypeError, ValueError) as exc:
        raise CliAdapterError("pull request number must be a positive integer") from exc
    if parsed < 1 or str(number).strip() != str(parsed):
        raise CliAdapterError("pull request number must be a positive integer")
    return parsed


def _validate_issue_number(number: int | str) -> int:
    if isinstance(number, bool):
        raise CliAdapterError("issue number must be a positive integer")
    try:
        parsed = int(number)
    except (TypeError, ValueError) as exc:
        raise CliAdapterError("issue number must be a positive integer") from exc
    if parsed < 1 or str(number).strip() != str(parsed):
        raise CliAdapterError("issue number must be a positive integer")
    return parsed


def _validate_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise CliAdapterError(f"{name} must be a boolean")


def _validate_positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CliAdapterError(f"{name} must be a positive integer") from exc
    if parsed < 1 or str(value).strip() != str(parsed):
        raise CliAdapterError(f"{name} must be a positive integer")
    return parsed


def _validate_non_negative_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CliAdapterError(f"{name} must be a non-negative integer") from exc
    if parsed < 0 or str(value).strip() != str(parsed):
        raise CliAdapterError(f"{name} must be a non-negative integer")
    return parsed


def _validate_repo_path(path: str) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
        raise CliAdapterError("path must be a relative repository path")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CliAdapterError("path must not contain empty, '.', or '..' segments")
    return path


def _validate_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip() or "\x00" in ref:
        raise CliAdapterError("ref must be a non-empty branch, tag, or commit")
    return ref.strip()


def _validate_local_repo_path(repo_path: str) -> Path:
    if not isinstance(repo_path, str) or not repo_path.strip() or "\x00" in repo_path:
        raise CliAdapterError("repo_path must be a local directory")
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise CliAdapterError("repo_path must be an existing local directory")
    allowed_roots = _allowed_git_repo_roots()
    if not any(_path_is_relative_to(path, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise CliAdapterError(f"repo_path must be under an allowed root: {roots}")
    return path


def _allowed_git_repo_roots() -> tuple[Path, ...]:
    roots = [Path.cwd().resolve()]
    configured = os.environ.get("KESTREL_CLI_ALLOWED_REPO_ROOTS", "")
    for raw_root in configured.split(os.pathsep):
        if raw_root.strip():
            roots.append(Path(raw_root).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_env() -> dict[str, str]:
    env = {key: value for key in _GIT_ENV_KEYS if (value := os.environ.get(key))}
    env.update(
        {
            "GIT_EXTERNAL_DIFF": "",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return env


def _validate_git_ref(ref: str, *, name: str) -> str:
    if not isinstance(ref, str) or not ref.strip() or "\x00" in ref:
        raise CliAdapterError(f"{name} must be a non-empty git ref")
    safe_ref = ref.strip()
    if safe_ref.startswith("-") or not _GIT_REF_RE.fullmatch(safe_ref):
        raise CliAdapterError(f"{name} is not a safe git ref")
    if ".." in safe_ref:
        raise CliAdapterError(f"{name} must not contain revision ranges")
    return safe_ref


def _validate_git_pathspec(path: str) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
        raise CliAdapterError("path must be a relative repository path")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise CliAdapterError("path must not contain empty, '.', or '..' segments")
    if path.startswith("-"):
        raise CliAdapterError("path must not look like a command option")
    return path


def redact_json(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [redact_json(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_json(item) for key, item in value.items()}
    return value


def _reviewable_files(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CliAdapterError("pull request files payload was not a list")

    files: list[dict[str, Any]] = []
    for item in value:
        if len(files) >= limit:
            break
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        status = str(item.get("status") or item.get("changeType") or "").lower()
        if status in {"removed", "deleted"}:
            continue
        files.append({"path": path})
    return files
