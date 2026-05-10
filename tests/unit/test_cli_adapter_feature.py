"""Tests for feature-owned CLI adapter support."""

from __future__ import annotations

import base64
import json
import sys
from unittest.mock import Mock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features import discover_feature_modules
from kestrel_sovereign.features.cli.adapters import GitHubCliAdapter
from kestrel_sovereign.features.cli.feature import CliFeature
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


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> TerminalCommandResult:
    return TerminalCommandResult(
        argv=["gh"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=5,
    )


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


def test_cli_status_lists_pr_review_helper_commands():
    feature = CliFeature(Mock())
    github_commands = {
        command.command_id for command in feature.adapters["github"].commands
    }

    assert "github.pr_files" in github_commands
    assert "github.pr_checks" in github_commands
    assert "github.read_file_at_pr_head" in github_commands
    assert "github.pr_review_context" in github_commands


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
