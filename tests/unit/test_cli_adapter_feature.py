"""Tests for feature-owned CLI adapter support."""

from __future__ import annotations

import base64
import json
import sys
from unittest.mock import Mock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features import discover_feature_modules
from kestrel_sovereign.features.cli.adapters import GitCliAdapter, GitHubCliAdapter
from kestrel_sovereign.features.cli.feature import CliFeature, _approval_argv_summary
from kestrel_sovereign.features.cli.terminal import (
    CliRisk,
    TerminalCommandRequest,
    TerminalCommandResult,
    TerminalExecutionService,
    ToolAvailability,
    redact_secrets,
)


class FakeTerminal:
    def __init__(self, results: list[TerminalCommandResult] | None = None):
        self.results = list(results or [])
        self.requests: list[TerminalCommandRequest] = []

    def platform_metadata(self):
        return {"system": "TestOS", "machine": "test64", "platform": "TestOS-test64"}

    async def which(self, command: str):
        return ToolAvailability(
            name=command,
            path=f"/usr/bin/{command}",
            available=True,
            version=f"{command} 1.0",
        )

    async def run(self, request: TerminalCommandRequest):
        self.requests.append(request)
        if not self.results:
            raise AssertionError("FakeTerminal has no queued result")
        return self.results.pop(0)


class FakeApprovalQueue:
    def __init__(self, approved: bool = True):
        self.approved = approved
        self.requests: list[dict] = []

    async def request_approval(self, **kwargs):
        self.requests.append(kwargs)
        return self.approved, "once"


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> TerminalCommandResult:
    return TerminalCommandResult(
        argv=["gh"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=5,
    )


def _allow_repo_root(monkeypatch: pytest.MonkeyPatch, path) -> None:
    monkeypatch.setenv("KESTREL_CLI_ALLOWED_REPO_ROOTS", str(path))


@pytest.mark.asyncio
async def test_github_pr_view_uses_registered_read_only_gh_command():
    payload = {
        "number": 1161,
        "headRefOid": "abc123",
        "body": "token: ghp_abcdefghijklmnopqrstuvwxyz123456",
        "files": [],
    }
    terminal = FakeTerminal([_result(stdout=json.dumps(payload))])
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.get_pull_request(
        repo="KestrelSovereignAI/kestrel-sovereign",
        number=1161,
    )

    assert parsed["number"] == payload["number"]
    assert parsed["headRefOid"] == payload["headRefOid"]
    assert parsed["body"] == "token: [REDACTED]"
    request = terminal.requests[0]
    assert request.risk is CliRisk.READ_ONLY
    assert request.command_id == "github.pr_view"
    assert request.argv[:4] == ["gh", "pr", "view", "--repo"]
    assert "--json" in request.argv
    assert "headRefOid" in request.argv[-1]
    assert "statusCheckRollup" in request.argv[-1]


@pytest.mark.asyncio
async def test_github_issue_view_uses_registered_read_only_gh_command():
    payload = {
        "number": 885,
        "title": "Hardcoded session id",
        "state": "CLOSED",
        "body": "token: ghp_abcdefghijklmnopqrstuvwxyz123456",
        "comments": [{"body": "fixed"}],
    }
    terminal = FakeTerminal([_result(stdout=json.dumps(payload))])
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.get_issue(
        repo="KestrelSovereignAI/kestrel-sovereign",
        number=885,
    )

    assert parsed["number"] == payload["number"]
    assert parsed["state"] == "CLOSED"
    assert parsed["body"] == "token: [REDACTED]"
    request = terminal.requests[0]
    assert request.risk is CliRisk.READ_ONLY
    assert request.command_id == "github.issue_view"
    assert request.argv[:4] == ["gh", "issue", "view", "--repo"]
    assert "--json" in request.argv
    assert "comments" in request.argv[-1]
    assert "closedAt" in request.argv[-1]


@pytest.mark.asyncio
async def test_terminal_execution_blocks_non_read_only_without_approval_callback():
    terminal = TerminalExecutionService()

    result = await terminal.run(
        TerminalCommandRequest(
            argv=[sys.executable, "-c", "print('should not run')"],
            risk=CliRisk.LOCAL_MUTATION,
            command_id="test.local_mutation",
        )
    )

    assert result.returncode == 126
    assert "requires approval" in result.stderr
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_terminal_execution_blocks_non_read_only_when_approval_denies():
    seen: list[TerminalCommandRequest] = []

    async def deny(request: TerminalCommandRequest) -> bool:
        seen.append(request)
        return False

    terminal = TerminalExecutionService(approval_callback=deny)

    result = await terminal.run(
        TerminalCommandRequest(
            argv=[sys.executable, "-c", "print('should not run')"],
            risk=CliRisk.REMOTE_MUTATION,
            command_id="test.remote_mutation",
        )
    )

    assert seen[0].command_id == "test.remote_mutation"
    assert result.returncode == 126
    assert "denied by approval gate" in result.stderr
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_terminal_execution_runs_non_read_only_after_approval():
    seen: list[TerminalCommandRequest] = []

    async def approve(request: TerminalCommandRequest) -> bool:
        seen.append(request)
        return True

    terminal = TerminalExecutionService(approval_callback=approve)

    result = await terminal.run(
        TerminalCommandRequest(
            argv=[sys.executable, "-c", "print('approved')"],
            risk=CliRisk.LOCAL_MUTATION,
            command_id="test.local_mutation",
        )
    )

    assert seen[0].risk is CliRisk.LOCAL_MUTATION
    assert result.ok is True
    assert result.stdout.strip() == "approved"


@pytest.mark.asyncio
async def test_cli_feature_approval_callback_uses_security_queue():
    queue = FakeApprovalQueue(approved=True)
    security = Mock(approval_queue=queue)
    agent = Mock()
    agent.get_feature.return_value = security
    feature = CliFeature(agent)

    approved = await feature._approve_cli_command(
        TerminalCommandRequest(
            argv=["gh", "pr", "merge", "1", "--repo", "owner/repo"],
            risk=CliRisk.REMOTE_MUTATION,
            command_id="github.pr_merge",
            env={"GITHUB_TOKEN": "secret", "PATH": "/usr/bin"},
        )
    )

    assert approved is True
    request = queue.requests[0]
    assert request["feature_name"] == "cli"
    assert request["tool_name"] == "github.pr_merge"
    assert request["timeout"] == 300
    assert request["tool_args"]["risk"] == "remote_mutation"
    assert request["tool_args"]["env_keys"] == ["GITHUB_TOKEN", "PATH"]
    assert request["tool_args"]["argc"] == 6


@pytest.mark.asyncio
async def test_cli_feature_approval_callback_fails_closed_without_security_queue():
    agent = Mock()
    agent.get_feature.return_value = None
    agent.features = {}
    feature = CliFeature(agent)

    approved = await feature._approve_cli_command(
        TerminalCommandRequest(
            argv=["gh", "pr", "merge", "1"],
            risk=CliRisk.REMOTE_MUTATION,
            command_id="github.pr_merge",
        )
    )

    assert approved is False


def test_cli_approval_argv_summary_redacts_values_and_sensitive_flags():
    summary = _approval_argv_summary(
        [
            "deployctl",
            "--repo",
            "owner/repo",
            "--password",
            "hunter2",
            "--token=abc123",
            "--message=ship it",
            "positional-secret",
            "-p",
            "short-secret",
            "-phunter2",
            "-p=hunter2",
            "--api_key",
            "sk-proj-secretsecretsecretsecret",
        ]
    )

    assert summary == [
        "deployctl",
        "--repo",
        "[ARG]",
        "--password",
        "[REDACTED]",
        "--token=[REDACTED]",
        "--message=[ARG]",
        "[ARG]",
        "-p",
        "[ARG]",
        "-p[ARG]",
        "-p=[ARG]",
        "--api_key",
        "[REDACTED]",
    ]
    assert "hunter2" not in summary
    assert "hunter2" not in "".join(summary)
    assert "abc123" not in "".join(summary)
    assert "positional-secret" not in summary


@pytest.mark.asyncio
async def test_git_status_uses_registered_read_only_git_command(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    terminal = FakeTerminal([_result(stdout="## main...origin/main\n M app.py\n")])
    adapter = GitCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.status(repo_path=str(tmp_path))

    assert parsed["status"].startswith("## main")
    request = terminal.requests[0]
    assert request.risk is CliRisk.READ_ONLY
    assert request.command_id == "git.status"
    assert request.argv == [
        "git",
        "--no-optional-locks",
        "-C",
        str(tmp_path.resolve()),
        "status",
        "--short",
        "--branch",
    ]
    assert request.env["GIT_OPTIONAL_LOCKS"] == "0"
    assert request.env["GIT_EXTERNAL_DIFF"] == ""


@pytest.mark.asyncio
async def test_git_diff_validates_ref_and_path_before_argv(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    terminal = FakeTerminal([_result(stdout="diff --git a/app.py b/app.py\n")])
    adapter = GitCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.diff(
        repo_path=str(tmp_path),
        ref="HEAD~1",
        path="src/app.py",
    )

    assert parsed["diff"].startswith("diff --git")
    assert terminal.requests[0].argv == [
        "git",
        "--no-optional-locks",
        "-C",
        str(tmp_path.resolve()),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD~1",
        "--",
        "src/app.py",
    ]


@pytest.mark.asyncio
async def test_git_log_caps_max_count(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    terminal = FakeTerminal([_result(stdout="abc1234 subject\n")])
    adapter = GitCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.log(repo_path=str(tmp_path), max_count="500")

    assert parsed["max_count"] == 100
    assert terminal.requests[0].argv[-2:] == ["--max-count", "100"]


@pytest.mark.asyncio
async def test_git_show_file_and_merge_base_build_safe_argv(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    terminal = FakeTerminal(
        [_result(stdout="file content\n"), _result(stdout="abc123\n")]
    )
    adapter = GitCliAdapter(terminal)  # type: ignore[arg-type]

    file_payload = await adapter.show_file(
        repo_path=str(tmp_path),
        ref="HEAD",
        path="docs/readme.md",
    )
    merge_payload = await adapter.merge_base(
        repo_path=str(tmp_path),
        left_ref="main",
        right_ref="feature/test",
    )

    assert file_payload["content"] == "file content\n"
    assert merge_payload["merge_base"] == "abc123"
    assert terminal.requests[0].argv == [
        "git",
        "--no-optional-locks",
        "-C",
        str(tmp_path.resolve()),
        "show",
        "HEAD:docs/readme.md",
    ]
    assert terminal.requests[1].argv == [
        "git",
        "--no-optional-locks",
        "-C",
        str(tmp_path.resolve()),
        "merge-base",
        "main",
        "feature/test",
    ]


@pytest.mark.asyncio
async def test_git_adapter_rejects_unsafe_refs_paths_and_missing_repo(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    adapter = GitCliAdapter(FakeTerminal())  # type: ignore[arg-type]

    with pytest.raises(Exception, match="revision ranges"):
        await adapter.diff(repo_path=str(tmp_path), ref="main..feature")

    with pytest.raises(Exception, match="safe git ref"):
        await adapter.diff(repo_path=str(tmp_path), ref="--cached")

    with pytest.raises(Exception, match="path"):
        await adapter.show_file(repo_path=str(tmp_path), ref="HEAD", path="../secret")

    with pytest.raises(Exception, match="existing local directory"):
        await adapter.status(repo_path=str(tmp_path / "missing"))


@pytest.mark.asyncio
async def test_git_adapter_rejects_repos_outside_allowed_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()
    _allow_repo_root(monkeypatch, allowed)
    adapter = GitCliAdapter(FakeTerminal())  # type: ignore[arg-type]

    with pytest.raises(Exception, match="allowed root"):
        await adapter.status(repo_path=str(other))


@pytest.mark.asyncio
async def test_git_adapter_redacts_failures_and_rejects_truncation(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    nonzero = GitCliAdapter(
        FakeTerminal(
            [_result(stderr="token: ghp_abcdefghijklmnopqrstuvwxyz123456", returncode=1)]
        )
    )  # type: ignore[arg-type]
    truncated = GitCliAdapter(
        FakeTerminal(
            [
                TerminalCommandResult(
                    argv=["git"],
                    returncode=0,
                    stdout="partial diff",
                    stderr="",
                    duration_ms=1,
                    truncated_stdout=True,
                )
            ]
        )
    )  # type: ignore[arg-type]

    with pytest.raises(Exception) as nonzero_exc:
        await nonzero.status(repo_path=str(tmp_path))
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in str(nonzero_exc.value)

    with pytest.raises(Exception, match="capture limit"):
        await truncated.diff(repo_path=str(tmp_path))


@pytest.mark.asyncio
async def test_github_pr_files_and_checks_project_pr_payload_lists():
    payload = {
        "number": 1161,
        "headRefOid": "abc123",
        "files": [{"path": "app.py", "additions": 3, "deletions": 1}],
        "statusCheckRollup": [{"name": "unit-tests", "conclusion": "SUCCESS"}],
    }
    terminal = FakeTerminal(
        [_result(stdout=json.dumps(payload)), _result(stdout=json.dumps(payload))]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    files = await adapter.list_pull_request_files(repo="owner/repo", number=1161)
    checks = await adapter.get_pull_request_checks(repo="owner/repo", number=1161)

    assert files == payload["files"]
    assert checks == payload["statusCheckRollup"]
    assert terminal.requests[0].command_id == "github.pr_view"
    assert terminal.requests[1].command_id == "github.pr_view"


@pytest.mark.asyncio
async def test_github_issue_comments_project_issue_payload_list():
    payload = {
        "number": 885,
        "comments": [
            {"author": {"login": "octo"}, "body": "linked to PR #900"},
        ],
    }
    terminal = FakeTerminal([_result(stdout=json.dumps(payload))])
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    comments = await adapter.list_issue_comments(repo="owner/repo", number=885)

    assert comments == payload["comments"]
    assert terminal.requests[0].command_id == "github.issue_view"


@pytest.mark.asyncio
async def test_github_read_file_at_ref_decodes_contents_response():
    encoded = base64.b64encode(b"hello sk-ant-api03-secretsecretsecret from branch").decode(
        "ascii"
    )
    terminal = FakeTerminal(
        [
            _result(
                stdout=json.dumps(
                    {
                        "sha": "abc123",
                        "size": 17,
                        "encoding": "base64",
                        "content": encoded,
                    }
                )
            )
        ]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.read_file_at_ref(
        repo="owner/repo",
        path="dir/file name.txt",
        ref="feature/branch",
    )

    assert parsed["content"] == "hello [REDACTED] from branch"
    assert parsed["sha"] == "abc123"
    assert terminal.requests[0].argv == [
        "gh",
        "api",
        "repos/owner/repo/contents/dir/file%20name.txt?ref=feature%2Fbranch",
    ]


@pytest.mark.asyncio
async def test_github_read_file_at_pr_head_uses_head_oid_ref():
    pr_payload = {
        "number": 42,
        "headRefOid": "headsha",
        "files": [],
        "statusCheckRollup": [],
    }
    encoded = base64.b64encode(b"head content").decode("ascii")
    file_payload = {
        "sha": "filesha",
        "size": 12,
        "encoding": "base64",
        "content": encoded,
    }
    terminal = FakeTerminal([
        _result(stdout=json.dumps(pr_payload)),
        _result(stdout=json.dumps(file_payload)),
    ])
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    parsed = await adapter.read_file_at_pull_request_head(
        repo="owner/repo",
        number=42,
        path="src/app.py",
    )

    assert parsed["content"] == "head content"
    assert parsed["ref"] == "headsha"
    assert terminal.requests[1].argv == [
        "gh",
        "api",
        "repos/owner/repo/contents/src/app.py?ref=headsha",
    ]


@pytest.mark.asyncio
async def test_github_pr_review_context_can_include_bounded_file_contents():
    pr_payload = {
        "number": 7,
        "headRefOid": "headsha",
        "files": [
            {"path": "a.py", "status": "modified"},
            {"path": "b.py", "status": "removed"},
            {"path": "c.py", "status": "added"},
        ],
        "statusCheckRollup": [{"name": "unit-tests", "conclusion": "SUCCESS"}],
    }
    file_payload = {
        "sha": "filesha",
        "size": 100,
        "encoding": "base64",
        "content": base64.b64encode(b"0123456789abcdef").decode("ascii"),
    }
    terminal = FakeTerminal(
        [
            _result(stdout=json.dumps(pr_payload)),
            _result(stdout="diff --git a/a.py b/a.py\n"),
            _result(stdout=json.dumps(file_payload)),
        ]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    context = await adapter.get_pull_request_review_context(
        repo="owner/repo",
        number=7,
        include_file_contents=True,
        max_files=1,
        max_file_bytes=8,
    )

    assert context["pull_request"]["number"] == 7
    assert context["checks"] == pr_payload["statusCheckRollup"]
    assert context["diff"].startswith("diff --git")
    assert len(context["file_contents"]) == 1
    assert context["file_contents"][0]["path"] == "a.py"
    assert context["file_contents"][0]["content"] == "01234567"
    assert context["file_contents"][0]["truncated"] is True


@pytest.mark.asyncio
async def test_github_pr_review_context_coerces_string_boolean_false():
    pr_payload = {
        "number": 7,
        "headRefOid": "headsha",
        "files": [{"path": "a.py", "status": "modified"}],
        "statusCheckRollup": [],
    }
    terminal = FakeTerminal(
        [
            _result(stdout=json.dumps(pr_payload)),
            _result(stdout="diff --git a/a.py b/a.py\n"),
        ]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    context = await adapter.get_pull_request_review_context(
        repo="owner/repo",
        number=7,
        include_file_contents="false",  # type: ignore[arg-type]
    )

    assert context["file_contents"] == []
    assert [request.command_id for request in terminal.requests] == [
        "github.pr_view",
        "github.pr_diff",
    ]


@pytest.mark.asyncio
async def test_github_read_file_at_ref_rejects_endpoint_escape_inputs():
    adapter = GitHubCliAdapter(FakeTerminal())  # type: ignore[arg-type]

    with pytest.raises(Exception, match="owner/name"):
        await adapter.read_file_at_ref(
            repo="owner/repo/issues",
            path="README.md",
            ref="main",
        )

    with pytest.raises(Exception, match="path"):
        await adapter.read_file_at_ref(repo="owner/repo", path="../../user", ref="main")

    with pytest.raises(Exception, match="repo owner"):
        await adapter.read_file_at_ref(repo="../repo", path="README.md", ref="main")

    with pytest.raises(Exception, match="repo name"):
        await adapter.read_file_at_ref(repo="owner/..", path="README.md", ref="main")


@pytest.mark.asyncio
async def test_github_pr_number_rejects_option_like_values():
    adapter = GitHubCliAdapter(FakeTerminal())  # type: ignore[arg-type]

    with pytest.raises(Exception, match="positive integer"):
        await adapter.get_pull_request(repo="owner/repo", number="--web")

    with pytest.raises(Exception, match="positive integer"):
        await adapter.get_issue(repo="owner/repo", number="--web")


@pytest.mark.asyncio
async def test_github_adapter_reports_nonzero_invalid_json_and_truncation():
    nonzero = GitHubCliAdapter(
        FakeTerminal(
            [_result(stderr="token: ghp_abcdefghijklmnopqrstuvwxyz123456", returncode=1)]
        )
    )  # type: ignore[arg-type]
    invalid_json = GitHubCliAdapter(
        FakeTerminal([_result(stdout="not json")])
    )  # type: ignore[arg-type]
    truncated = GitHubCliAdapter(
        FakeTerminal(
            [
                TerminalCommandResult(
                    argv=["gh"],
                    returncode=0,
                    stdout="partial diff",
                    stderr="",
                    duration_ms=1,
                    truncated_stdout=True,
                )
            ]
        )
    )  # type: ignore[arg-type]

    with pytest.raises(Exception) as nonzero_exc:
        await nonzero.get_pull_request(repo="owner/repo", number=1)
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in str(nonzero_exc.value)

    with pytest.raises(Exception, match="valid JSON"):
        await invalid_json.get_pull_request(repo="owner/repo", number=1)

    with pytest.raises(Exception, match="capture limit"):
        await truncated.get_pull_request_diff(repo="owner/repo", number=1)


@pytest.mark.asyncio
async def test_github_read_file_at_ref_rejects_non_base64_contents():
    terminal = FakeTerminal(
        [_result(stdout=json.dumps({"encoding": "utf-8", "content": "hello"}))]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="base64"):
        await adapter.read_file_at_ref(repo="owner/repo", path="README.md", ref="main")


@pytest.mark.asyncio
async def test_cli_feature_exposes_status_and_github_pr_tool():
    payload = {
        "number": 42,
        "headRefOid": "abc123",
        "files": [{"path": "tests/test_x.py"}],
    }
    feature = CliFeature(Mock())
    terminal = FakeTerminal([_result(stdout=json.dumps(payload))])
    feature.terminal = terminal  # type: ignore[assignment]
    feature.adapters = {"github": GitHubCliAdapter(terminal)}  # type: ignore[arg-type]

    status = await feature.cli_status()
    pr = await feature.github_pr_view(repo="owner/repo", number=42)

    assert status.status is ToolResultStatus.OK
    assert status.data["adapters"]["github"]["available"] is True
    assert status.data["adapters"]["github"]["commands"][0]["risk"] == "read_only"
    assert pr.status is ToolResultStatus.OK
    assert pr.data["headRefOid"] == "abc123"


@pytest.mark.asyncio
async def test_cli_feature_exposes_github_issue_tools():
    issue_payload = {
        "number": 885,
        "title": "Hardcoded session id",
        "state": "CLOSED",
        "comments": [{"body": "fixed by broader restructure"}],
    }
    feature = CliFeature(Mock())
    terminal = FakeTerminal(
        [
            _result(stdout=json.dumps(issue_payload)),
            _result(stdout=json.dumps(issue_payload)),
        ]
    )
    feature.terminal = terminal  # type: ignore[assignment]
    feature.adapters = {"github": GitHubCliAdapter(terminal)}  # type: ignore[arg-type]

    issue = await feature.github_issue_view(repo="owner/repo", number=885)
    comments = await feature.github_issue_comments(repo="owner/repo", number=885)

    assert issue.status is ToolResultStatus.OK
    assert issue.data["title"] == "Hardcoded session id"
    assert comments.status is ToolResultStatus.OK
    assert comments.data["comments"] == issue_payload["comments"]


@pytest.mark.asyncio
async def test_cli_feature_exposes_git_helper_tools(tmp_path, monkeypatch):
    _allow_repo_root(monkeypatch, tmp_path)
    feature = CliFeature(Mock())
    terminal = FakeTerminal(
        [
            _result(stdout="## main\n"),
            _result(stdout="diff --git a/app.py b/app.py\n"),
            _result(stdout="abc123 subject\n"),
            _result(stdout="file content\n"),
            _result(stdout="abc123\n"),
        ]
    )
    feature.terminal = terminal  # type: ignore[assignment]
    feature.adapters = {"git": GitCliAdapter(terminal)}  # type: ignore[arg-type]

    status = await feature.git_status(repo_path=str(tmp_path))
    diff = await feature.git_diff(repo_path=str(tmp_path))
    log = await feature.git_log(repo_path=str(tmp_path), max_count=3)
    file_payload = await feature.git_show_file(
        repo_path=str(tmp_path),
        ref="HEAD",
        path="README.md",
    )
    merge_base = await feature.git_merge_base(
        repo_path=str(tmp_path),
        left_ref="main",
        right_ref="feature/test",
    )

    assert status.status is ToolResultStatus.OK
    assert diff.status is ToolResultStatus.OK
    assert log.status is ToolResultStatus.OK
    assert file_payload.status is ToolResultStatus.OK
    assert merge_base.status is ToolResultStatus.OK
    assert status.data["status"] == "## main\n"
    assert file_payload.data["content"] == "file content\n"
    assert merge_base.data["merge_base"] == "abc123"


@pytest.mark.asyncio
async def test_cli_feature_exposes_pr_review_helper_tools():
    pr_payload = {
        "number": 42,
        "headRefOid": "abc123",
        "files": [{"path": "tests/test_x.py"}],
        "statusCheckRollup": [{"name": "unit-tests", "conclusion": "SUCCESS"}],
    }
    feature = CliFeature(Mock())
    terminal = FakeTerminal(
        [
            _result(stdout=json.dumps(pr_payload)),
            _result(stdout=json.dumps(pr_payload)),
            _result(stdout=json.dumps(pr_payload)),
            _result(stdout="diff --git a/tests/test_x.py b/tests/test_x.py\n"),
        ]
    )
    feature.terminal = terminal  # type: ignore[assignment]
    feature.adapters = {"github": GitHubCliAdapter(terminal)}  # type: ignore[arg-type]

    files = await feature.github_pr_files(repo="owner/repo", number=42)
    checks = await feature.github_pr_checks(repo="owner/repo", number=42)
    context = await feature.github_pr_review_context(repo="owner/repo", number=42)

    assert files.status is ToolResultStatus.OK
    assert files.data["files"] == pr_payload["files"]
    assert checks.status is ToolResultStatus.OK
    assert checks.data["checks"] == pr_payload["statusCheckRollup"]
    assert context.status is ToolResultStatus.OK
    assert context.data["diff"].startswith("diff --git")


@pytest.mark.asyncio
async def test_cli_feature_redacts_auth_status_partial_error():
    feature = CliFeature(Mock())
    terminal = FakeTerminal(
        [_result(stderr="token: ghp_abcdefghijklmnopqrstuvwxyz123456", returncode=1)]
    )
    feature.terminal = terminal  # type: ignore[assignment]
    feature.adapters = {"github": GitHubCliAdapter(terminal)}  # type: ignore[arg-type]

    result = await feature.github_cli_auth_status()

    assert result.status is ToolResultStatus.PARTIAL
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in result.error
    assert "[REDACTED]" in result.error


@pytest.mark.asyncio
async def test_github_pr_diff_redacts_hyphenated_provider_tokens():
    terminal = FakeTerminal(
        [_result(stdout="+OPENAI_API_KEY=sk-proj-secretsecretsecretsecret\n")]
    )
    adapter = GitHubCliAdapter(terminal)  # type: ignore[arg-type]

    diff = await adapter.get_pull_request_diff(repo="owner/repo", number=1)

    assert "sk-proj-secretsecretsecretsecret" not in diff
    assert "[REDACTED]" in diff


def test_cli_command_prefixes_parse_positional_args():
    feature = CliFeature(Mock())
    tools = {tool.schema.name: tool for tool in feature.get_tools()}

    assert tools["github_pr_view"].parse_command_args("!gh-pr-view owner/repo 42") == {
        "repo": "owner/repo",
        "number": "42",
    }
    assert tools["github_pr_diff"].parse_command_args("!gh-pr-diff owner/repo 42") == {
        "repo": "owner/repo",
        "number": "42",
    }
    assert tools["github_issue_view"].parse_command_args(
        "!gh-issue-view owner/repo 885"
    ) == {
        "repo": "owner/repo",
        "number": "885",
    }
    assert tools["github_issue_comments"].parse_command_args(
        "!gh-issue-comments owner/repo 885"
    ) == {
        "repo": "owner/repo",
        "number": "885",
    }
    assert tools["github_read_file_at_ref"].parse_command_args(
        "!gh-read-file owner/repo README.md main"
    ) == {
        "repo": "owner/repo",
        "path": "README.md",
        "ref": "main",
    }
    assert tools["github_pr_files"].parse_command_args("!gh-pr-files owner/repo 42") == {
        "repo": "owner/repo",
        "number": "42",
    }
    assert tools["github_pr_checks"].parse_command_args("!gh-pr-checks owner/repo 42") == {
        "repo": "owner/repo",
        "number": "42",
    }
    assert tools["github_read_file_at_pr_head"].parse_command_args(
        "!gh-read-pr-file owner/repo 42 README.md"
    ) == {
        "repo": "owner/repo",
        "number": "42",
        "path": "README.md",
    }
    assert tools["github_pr_review_context"].parse_command_args(
        "!gh-pr-context owner/repo 42 true 3 1000"
    ) == {
        "repo": "owner/repo",
        "number": "42",
        "include_file_contents": "true",
        "max_files": "3",
        "max_file_bytes": "1000",
    }
    assert tools["git_status"].parse_command_args("!git-status") == {
        "repo_path": ".",
    }
    assert tools["git_diff"].parse_command_args("!git-diff HEAD src/app.py") == {
        "ref": "HEAD",
        "path": "src/app.py",
        "repo_path": ".",
    }
    assert tools["git_log"].parse_command_args("!git-log 5") == {
        "max_count": "5",
        "repo_path": ".",
    }
    assert tools["git_show_file"].parse_command_args(
        "!git-show-file HEAD README.md"
    ) == {
        "ref": "HEAD",
        "path": "README.md",
        "repo_path": ".",
    }
    assert tools["git_merge_base"].parse_command_args(
        "!git-merge-base main feature/test"
    ) == {
        "left_ref": "main",
        "right_ref": "feature/test",
        "repo_path": ".",
    }


def test_cli_status_lists_pr_review_helper_commands():
    feature = CliFeature(Mock())
    github_commands = {
        command.command_id for command in feature.adapters["github"].commands
    }

    assert "github.issue_view" in github_commands
    assert "github.issue_comments" in github_commands
    assert "github.pr_files" in github_commands
    assert "github.pr_checks" in github_commands
    assert "github.read_file_at_pr_head" in github_commands
    assert "github.pr_review_context" in github_commands


def test_cli_status_lists_git_helper_commands():
    feature = CliFeature(Mock())
    git_commands = {command.command_id for command in feature.adapters["git"].commands}

    assert "git.status" in git_commands
    assert "git.diff" in git_commands
    assert "git.log" in git_commands
    assert "git.show_file" in git_commands
    assert "git.merge_base" in git_commands


def test_cli_feature_is_discoverable():
    assert "kestrel_sovereign.features.cli.feature" in discover_feature_modules()


def test_redact_secrets_masks_common_token_forms():
    redacted = redact_secrets(
        "Token: gho_abcdefghijklmnopqrstuvwxyz123456\n"
        "api_key = sk_test_abcdefghijklmnopqrstuvwxyz123456\n"
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456\n"
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
    )

    assert "gho_abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "sk_test_abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_terminal_execution_enforces_output_capture_limit():
    terminal = TerminalExecutionService(max_output_bytes=16)

    result = await terminal.run(
        TerminalCommandRequest(
            argv=[sys.executable, "-c", "print('x' * 100000)"],
            timeout=10,
            risk=CliRisk.READ_ONLY,
            command_id="test.large_output",
        )
    )

    assert result.truncated_stdout is True
    assert len(result.stdout.encode("utf-8")) <= 16
