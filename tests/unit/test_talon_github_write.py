"""Tests for the Talon GitHub-write bounded job (#2581).

Two layers:

* pure-core request builders + allowlist in
  ``kestrel_sovereign.features.talon.github_write`` (no I/O), and
* the ``talon_github_write`` coordinator tool + its in-process REST executor,
  exercised with the HTTP call mocked so no network is touched.
"""

import io
import urllib.error
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.talon import github_write as gw
from kestrel_sovereign.features.talon.coordinator import TalonCoordinatorFeature


# --------------------------------------------------------------------------
# Pure-core: request builders
# --------------------------------------------------------------------------

REPO = "KestrelSovereignAI/kestrel-sovereign"


class TestBuildRequests:
    def test_close_issue_default_reason(self):
        (req,) = gw.build_github_write_requests("close_issue", REPO, 1927)
        assert req.method == "PATCH"
        assert req.url == f"https://api.github.com/repos/{REPO}/issues/1927"
        assert req.payload == {"state": "closed", "state_reason": "completed"}
        assert req.success_statuses == (200,)

    def test_close_issue_not_planned(self):
        (req,) = gw.build_github_write_requests(
            "close_issue", REPO, 1927, state_reason="not_planned"
        )
        assert req.payload["state_reason"] == "not_planned"

    def test_close_issue_rejects_bad_reason(self):
        with pytest.raises(gw.GithubWriteError):
            gw.build_github_write_requests(
                "close_issue", REPO, 1927, state_reason="bogus"
            )

    def test_reopen_issue(self):
        (req,) = gw.build_github_write_requests("reopen_issue", REPO, 1927)
        assert req.method == "PATCH"
        assert req.payload == {"state": "open", "state_reason": "reopened"}

    def test_comment_success_status_is_201(self):
        (req,) = gw.build_github_write_requests(
            "comment", REPO, 1927, body="work complete"
        )
        assert req.method == "POST"
        assert req.url.endswith("/issues/1927/comments")
        assert req.payload == {"body": "work complete"}
        assert req.success_statuses == (201,)

    def test_comment_rejects_empty_body(self):
        with pytest.raises(gw.GithubWriteError):
            gw.build_github_write_requests("comment", REPO, 1927, body="   ")

    def test_add_labels(self):
        (req,) = gw.build_github_write_requests(
            "add_labels", REPO, 1927, labels="bug, enhancement"
        )
        assert req.method == "POST"
        assert req.url.endswith("/issues/1927/labels")
        assert req.payload == {"labels": ["bug", "enhancement"]}

    def test_add_labels_requires_at_least_one(self):
        with pytest.raises(gw.GithubWriteError):
            gw.build_github_write_requests("add_labels", REPO, 1927, labels=" , ")

    def test_remove_labels_one_request_per_label_urlencoded(self):
        reqs = gw.build_github_write_requests(
            "remove_labels", REPO, 1927, labels="needs review,done"
        )
        assert len(reqs) == 2
        assert all(r.method == "DELETE" for r in reqs)
        # spaces are URL-encoded in the path segment
        assert reqs[0].url.endswith("/issues/1927/labels/needs%20review")
        assert reqs[1].url.endswith("/issues/1927/labels/done")
        # 404 (already absent) is an idempotent success
        assert reqs[0].success_statuses == (200, 404)

    def test_update_issue_title_and_body(self):
        (req,) = gw.build_github_write_requests(
            "update_issue", REPO, 1927, title="New title", body="New body"
        )
        assert req.method == "PATCH"
        assert req.payload == {"title": "New title", "body": "New body"}

    def test_update_issue_requires_something(self):
        with pytest.raises(gw.GithubWriteError):
            gw.build_github_write_requests("update_issue", REPO, 1927)

    def test_unknown_operation_lists_valid_ops(self):
        with pytest.raises(gw.GithubWriteError) as excinfo:
            gw.build_github_write_requests("delete_repo", REPO, 1927)
        assert "close_issue" in str(excinfo.value)


class TestParseAndNormalize:
    @pytest.mark.parametrize("value", [123, "123", "#123", " #123 "])
    def test_parse_issue_number_accepts(self, value):
        assert gw.parse_issue_number(value) == 123

    @pytest.mark.parametrize("value", [0, -1, "abc", "", True])
    def test_parse_issue_number_rejects(self, value):
        with pytest.raises(gw.GithubWriteError):
            gw.parse_issue_number(value)

    def test_normalize_labels_dedupes_case_insensitively(self):
        assert gw.normalize_labels("Bug, bug , Enhancement") == [
            "Bug",
            "Enhancement",
        ]

    def test_normalize_labels_accepts_list(self):
        assert gw.normalize_labels(["a", " b ", ""]) == ["a", "b"]


class TestWriteAllowlist:
    def test_self_repo_allowed(self, monkeypatch):
        monkeypatch.delenv("GITHUB_FLEET_REPOS", raising=False)
        monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
        assert gw.resolve_write_repo(REPO) == REPO

    def test_case_insensitive_returns_canonical(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
        assert gw.resolve_write_repo(REPO.lower()) == REPO

    def test_fleet_repo_allowed(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
        monkeypatch.setenv("GITHUB_FLEET_REPOS", "acme/widgets, acme/gadgets")
        assert gw.resolve_write_repo("acme/gadgets") == "acme/gadgets"

    def test_external_repo_denied(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
        monkeypatch.delenv("GITHUB_FLEET_REPOS", raising=False)
        with pytest.raises(gw.GithubWriteError):
            gw.resolve_write_repo("someone/else")

    def test_malformed_repo_denied(self, monkeypatch):
        monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
        with pytest.raises(gw.GithubWriteError):
            gw.resolve_write_repo("not-a-repo")


class TestExtractErrorMessage:
    def test_pulls_message(self):
        assert gw.extract_error_message('{"message": "Not Found"}') == "Not Found"

    def test_message_with_errors(self):
        out = gw.extract_error_message(
            '{"message": "Validation Failed", "errors": [{"code": "x"}]}'
        )
        assert "Validation Failed" in out

    def test_non_json_body(self):
        assert gw.extract_error_message("boom") == "boom"

    def test_empty(self):
        assert gw.extract_error_message("") == ""


# --------------------------------------------------------------------------
# Coordinator tool: talon_github_write
# --------------------------------------------------------------------------


def _make_agent():
    agent = MagicMock()
    agent.agent_name = "kestrel"
    agent._features = []
    # No SecurityFeature -> outcome audit is skipped (best-effort), keeping
    # the tool assertions focused on the write path.
    agent.get_feature = MagicMock(return_value=None)
    agent.features = None
    return agent


@pytest.fixture
def feature(monkeypatch, tmp_path):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.setenv("GITHUB_SELF_REPO", REPO)
    monkeypatch.delenv("GITHUB_FLEET_REPOS", raising=False)
    # A real GH_TOKEN/GITHUB_PAT often leaks in from the workspace shell and
    # (GH_TOKEN first) would shadow the test token — pin them deterministically.
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    return TalonCoordinatorFeature(_make_agent())


class TestTalonGithubWriteTool:
    @pytest.mark.asyncio
    async def test_close_issue_success(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            mock_write.return_value = {"ok": True, "status": 200, "response": {}}
            result = await feature.talon_github_write(
                operation="close_issue", issue=1927, repo="self"
            )
        assert result.status is ToolResultStatus.OK
        assert result.data["success"] is True
        assert result.data["repo"] == REPO
        assert result.data["issue"] == 1927
        # the executor received the built request + the token (never a shell)
        request_arg, token_arg = mock_write.await_args[0][:2]
        assert request_arg.method == "PATCH"
        assert request_arg.payload == {
            "state": "closed",
            "state_reason": "completed",
        }
        assert token_arg == "ghp_test_token"

    @pytest.mark.asyncio
    async def test_comment_success(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            mock_write.return_value = {"ok": True, "status": 201, "response": {}}
            result = await feature.talon_github_write(
                operation="comment",
                issue=1927,
                repo=REPO,
                body="Work complete — closing.",
            )
        assert result.status is ToolResultStatus.OK
        assert mock_write.await_args[0][0].url.endswith("/comments")

    @pytest.mark.asyncio
    async def test_remove_labels_issues_one_call_per_label(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            mock_write.return_value = {"ok": True, "status": 200, "response": {}}
            result = await feature.talon_github_write(
                operation="remove_labels",
                issue=1927,
                repo=REPO,
                labels="agent-claimed, in progress",
            )
        assert result.status is ToolResultStatus.OK
        assert mock_write.await_count == 2

    @pytest.mark.asyncio
    async def test_not_allowlisted_repo_rejected(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            result = await feature.talon_github_write(
                operation="close_issue", issue=1927, repo="attacker/repo"
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["reason_code"] == "INVALID_REQUEST"
        # never made an HTTP call for a denied target
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_operation_rejected(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            result = await feature.talon_github_write(
                operation="nuke", issue=1927, repo=REPO
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["reason_code"] == "INVALID_REQUEST"
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_token_reports_no_token(self, feature, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_PAT", raising=False)
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            result = await feature.talon_github_write(
                operation="close_issue", issue=1927, repo=REPO
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["reason_code"] == "NO_TOKEN"
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_github_api_error_surfaced(self, feature):
        with patch.object(
            feature, "_github_api_write", new_callable=AsyncMock
        ) as mock_write:
            mock_write.return_value = {
                "ok": False,
                "status": 403,
                "error": "Resource not accessible by integration",
            }
            result = await feature.talon_github_write(
                operation="close_issue", issue=1927, repo=REPO
            )
        assert result.status is ToolResultStatus.ERROR
        assert result.data["reason_code"] == "GITHUB_API_ERROR"
        assert "403" in result.error


# --------------------------------------------------------------------------
# In-process REST executor: _github_api_write_sync
# --------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status, body=b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_request(method="PATCH", success=(200,), payload=None):
    return gw.GithubWriteRequest(
        method=method,
        url="https://api.github.com/repos/x/y/issues/1",
        payload=payload if payload is not None else {"state": "closed"},
        summary="close x/y#1",
        success_statuses=success,
    )


class TestGithubApiWriteSync:
    def test_success_200(self):
        with patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            return_value=_FakeResp(200, b'{"state": "closed"}'),
        ):
            out = TalonCoordinatorFeature._github_api_write_sync(
                _make_request(), "ghp_x", 30
            )
        assert out["ok"] is True
        assert out["status"] == 200

    def test_comment_success_201(self):
        req = _make_request(method="POST", success=(201,), payload={"body": "hi"})
        with patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            return_value=_FakeResp(201, b'{"id": 1}'),
        ):
            out = TalonCoordinatorFeature._github_api_write_sync(req, "ghp_x", 30)
        assert out["ok"] is True

    def test_remove_label_404_is_idempotent_success(self):
        req = _make_request(method="DELETE", success=(200, 404), payload=None)
        err = urllib.error.HTTPError(
            req.url, 404, "Not Found", {}, io.BytesIO(b'{"message": "Label does not exist"}')
        )
        with patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            side_effect=err,
        ):
            out = TalonCoordinatorFeature._github_api_write_sync(req, "ghp_x", 30)
        assert out["ok"] is True
        assert out["status"] == 404

    def test_http_error_surfaces_github_message(self):
        err = urllib.error.HTTPError(
            "https://api.github.com/repos/x/y/issues/1",
            422,
            "Unprocessable",
            {},
            io.BytesIO(b'{"message": "Validation Failed"}'),
        )
        with patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            side_effect=err,
        ):
            out = TalonCoordinatorFeature._github_api_write_sync(
                _make_request(), "ghp_x", 30
            )
        assert out["ok"] is False
        assert out["status"] == 422
        assert "Validation Failed" in out["error"]

    def test_network_error_fails_closed(self):
        with patch(
            "kestrel_sovereign.features.talon.coordinator.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            out = TalonCoordinatorFeature._github_api_write_sync(
                _make_request(), "ghp_x", 30
            )
        assert out["ok"] is False
        assert out["status"] is None
