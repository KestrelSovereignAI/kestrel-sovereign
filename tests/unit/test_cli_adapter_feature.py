"""Tests for feature-owned CLI adapter support.

The core ``cli`` feature exposes only local, read-only ``git`` inspection plus
the substrate that gates non-read-only commands. GitHub access is *not* a core
CLI workflow: it lives in the optional ``kestrel-feature-github`` package
(httpx against ``api.github.com``), so there is intentionally no ``gh`` adapter
here. The git adapter is covered both as argv/validation unit tests *and* as
real end-to-end runs against a real git repository, so a runtime regression
(executable missing, argv wrong, output mishandled) cannot pass green.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import Mock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features import discover_feature_modules
from kestrel_sovereign.features.cli.adapters import GitCliAdapter
from kestrel_sovereign.features.cli.feature import CliFeature, _approval_argv_summary
from kestrel_sovereign.features.cli.terminal import (
    CliRisk,
    TerminalCommandRequest,
    TerminalCommandResult,
    TerminalExecutionService,
    redact_secrets,
)


class FakeTerminal:
    def __init__(self, results: list[TerminalCommandResult] | None = None):
        self.results = list(results or [])
        self.requests: list[TerminalCommandRequest] = []

    def platform_metadata(self):
        return {"system": "TestOS", "machine": "test64", "platform": "TestOS-test64"}

    async def which(self, command: str):
        from kestrel_sovereign.features.cli.terminal import ToolAvailability

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
        argv=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=5,
    )


def _allow_repo_root(monkeypatch: pytest.MonkeyPatch, path) -> None:
    monkeypatch.setenv("KESTREL_CLI_ALLOWED_REPO_ROOTS", str(path))


# --- git fixture for real end-to-end runs ------------------------------------

_GIT_FIXTURE_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(repo, *args: str) -> str:
    """Run a real git command for *test scaffolding* (not the code under test)."""
    import os

    env = {**os.environ, **_GIT_FIXTURE_ENV}
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


@pytest.fixture
def git_repo(tmp_path):
    """A real git repository with two branches and a small history."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("print('one')\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "first commit")
    (repo / "app.py").write_text("print('one')\nprint('two')\n")
    (repo / "README.md").write_text("# Demo\n")
    _git(repo, "add", "app.py", "README.md")
    _git(repo, "commit", "-m", "second commit")
    _git(repo, "branch", "feature/test")
    return repo


# --- substrate: approval gating ----------------------------------------------


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
            argv=["deployctl", "deploy", "--token", "secret", "--env", "prod"],
            risk=CliRisk.REMOTE_MUTATION,
            command_id="deploy.push",
            env={"API_TOKEN": "secret", "PATH": "/usr/bin"},
        )
    )

    assert approved is True
    request = queue.requests[0]
    assert request["feature_name"] == "cli"
    assert request["tool_name"] == "deploy.push"
    assert request["timeout"] == 300
    assert request["tool_args"]["risk"] == "remote_mutation"
    assert request["tool_args"]["env_keys"] == ["API_TOKEN", "PATH"]
    assert request["tool_args"]["argc"] == 6


@pytest.mark.asyncio
async def test_cli_feature_approval_callback_fails_closed_without_security_queue():
    agent = Mock()
    agent.get_feature.return_value = None
    agent.features = {}
    feature = CliFeature(agent)

    approved = await feature._approve_cli_command(
        TerminalCommandRequest(
            argv=["deployctl", "deploy"],
            risk=CliRisk.REMOTE_MUTATION,
            command_id="deploy.push",
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


# --- git adapter: argv construction and input validation ---------------------


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


# --- git adapter: REAL end-to-end runs against a real repository -------------


@pytest.mark.asyncio
async def test_git_adapter_runs_against_real_repo(git_repo, monkeypatch):
    """Drive the real TerminalExecutionService + real `git` against a real repo.

    This is the coverage that a fully-mocked terminal cannot give: it proves the
    executable resolves, the argv runs, and output is parsed for actual git.
    """
    _allow_repo_root(monkeypatch, git_repo)
    adapter = GitCliAdapter(TerminalExecutionService())

    availability = await adapter.check_availability()
    assert availability.available is True

    status = await adapter.status(repo_path=str(git_repo))
    assert status["status"].startswith("## main")

    log = await adapter.log(repo_path=str(git_repo), max_count=10)
    assert "second commit" in log["log"]
    assert "first commit" in log["log"]

    diff = await adapter.diff(repo_path=str(git_repo), ref="HEAD~1", path="app.py")
    assert "print('two')" in diff["diff"]

    show = await adapter.show_file(repo_path=str(git_repo), ref="HEAD", path="README.md")
    assert show["content"] == "# Demo\n"

    merge = await adapter.merge_base(
        repo_path=str(git_repo),
        left_ref="main",
        right_ref="feature/test",
    )
    assert len(merge["merge_base"]) == 40  # a real commit sha


@pytest.mark.asyncio
async def test_git_feature_tools_run_against_real_repo(git_repo, monkeypatch):
    """End-to-end through the CliFeature @tool surface with a real git repo."""
    _allow_repo_root(monkeypatch, git_repo)
    feature = CliFeature(Mock())

    status = await feature.git_status(repo_path=str(git_repo))
    log = await feature.git_log(repo_path=str(git_repo), max_count=5)
    diff = await feature.git_diff(repo_path=str(git_repo), ref="HEAD~1", path="app.py")
    show = await feature.git_show_file(
        repo_path=str(git_repo), ref="HEAD", path="README.md"
    )

    assert status.status is ToolResultStatus.OK
    assert status.data["status"].startswith("## main")
    assert log.status is ToolResultStatus.OK
    assert "second commit" in log.data["log"]
    assert diff.status is ToolResultStatus.OK
    assert "print('two')" in diff.data["diff"]
    assert show.status is ToolResultStatus.OK
    assert show.data["content"] == "# Demo\n"


@pytest.mark.asyncio
async def test_git_feature_tool_surfaces_real_error(tmp_path, monkeypatch):
    """A real git failure (not a git repo) surfaces as a failed ToolResult."""
    _allow_repo_root(monkeypatch, tmp_path)
    feature = CliFeature(Mock())

    result = await feature.git_status(repo_path=str(tmp_path))

    assert result.status is ToolResultStatus.ERROR
    assert result.error


# --- git adapter: tool wiring through FakeTerminal (argv-shape contract) ------


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


# --- regression guard: core carries no GitHub / gh tooling -------------------


def test_cli_feature_carries_no_github_or_gh_tooling():
    """Core must not ship GitHub access via `gh`. It lives in kestrel-feature-github."""
    feature = CliFeature(Mock())

    assert set(feature.adapters) == {"git"}
    for adapter in feature.adapters.values():
        for declaration in adapter.tools:
            assert declaration.name != "gh"
        for command in adapter.commands:
            assert not command.command_id.startswith("github")

    tool_names = {tool.schema.name for tool in feature.get_tools()}
    assert not any("github" in name or name.startswith("gh_") for name in tool_names)
    assert "git_status" in tool_names


def test_cli_command_prefixes_parse_positional_args():
    feature = CliFeature(Mock())
    tools = {tool.schema.name: tool for tool in feature.get_tools()}

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
