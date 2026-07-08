"""Prerequisite classification for Morning Signal live GitHub scanning (#2271).

An empty ``scan_repos`` must produce a configuration remediation, not a
misleading "Set GITHUB_TOKEN" message, since the token may be present and valid.
"""

import urllib.error
from unittest.mock import patch

import pytest

from kestrel_sovereign.features.strategic_memory.github_integration import (
    GITHUB_SIGNAL_NO_SCAN_REPOS,
    GITHUB_SIGNAL_NO_TOKEN,
    GITHUB_SIGNAL_READY,
    GitHubAuthError,
    fetch_github_signal,
    github_api_get,
    github_signal_prerequisite,
)
from kestrel_sovereign.features.strategic_memory.morning_signal import (
    generate_morning_signal,
)

_GH_MOD = "kestrel_sovereign.features.strategic_memory.github_integration"


def _data(repos):
    return {"morning_signal_config": {"scan_repos": repos}}


# --- github_signal_prerequisite -------------------------------------------


def test_prerequisite_empty_scan_repos_even_with_token():
    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_valid"):
        assert github_signal_prerequisite(_data([])) == GITHUB_SIGNAL_NO_SCAN_REPOS


def test_prerequisite_missing_token_with_repos():
    with patch(f"{_GH_MOD}.get_github_token", return_value=None):
        assert (
            github_signal_prerequisite(_data(["owner/repo"])) == GITHUB_SIGNAL_NO_TOKEN
        )


def test_prerequisite_ready_with_repos_and_token():
    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_valid"):
        assert (
            github_signal_prerequisite(_data(["owner/repo"])) == GITHUB_SIGNAL_READY
        )


# --- fetch_github_signal ---------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_for_empty_scan_repos_without_checking_token():
    # Token must NOT be consulted when there are no repos to scan.
    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_valid") as tok:
        result = await fetch_github_signal(_data([]))
    assert result == {}
    tok.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_returns_empty_for_missing_token():
    with patch(f"{_GH_MOD}.get_github_token", return_value=None):
        result = await fetch_github_signal(_data(["owner/repo"]))
    assert result == {}


def _http_error(code):
    return urllib.error.HTTPError(
        url="https://api.github.com/x",
        code=code,
        msg="denied",
        hdrs=None,
        fp=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403])
async def test_github_api_get_raises_on_auth_when_opted_in(code):
    def _boom():
        raise _http_error(code)

    with patch(f"{_GH_MOD}.urllib.request.urlopen", side_effect=lambda *a, **k: _boom()):
        with pytest.raises(GitHubAuthError):
            await github_api_get("/repos/owner/repo/issues", "ghp_bad", raise_on_auth=True)


@pytest.mark.asyncio
async def test_github_api_get_swallows_auth_by_default():
    def _boom():
        raise _http_error(401)

    with patch(f"{_GH_MOD}.urllib.request.urlopen", side_effect=lambda *a, **k: _boom()):
        result = await github_api_get("/repos/owner/repo/issues", "ghp_bad")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_raises_auth_error_for_invalid_token_with_repos():
    def _boom():
        raise _http_error(401)

    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_invalid"), patch(
        f"{_GH_MOD}.urllib.request.urlopen", side_effect=lambda *a, **k: _boom()
    ):
        with pytest.raises(GitHubAuthError):
            await fetch_github_signal(_data(["owner/repo"]))


# --- generate_morning_signal remediation -----------------------------------


@pytest.mark.asyncio
async def test_signal_reports_scan_repos_remediation_when_token_present():
    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_valid"):
        report = await generate_morning_signal(_data([]))
    assert "scan_repos" in report
    assert "Set GITHUB_TOKEN" not in report


@pytest.mark.asyncio
async def test_signal_reports_token_remediation_when_token_missing():
    with patch(f"{_GH_MOD}.get_github_token", return_value=None):
        report = await generate_morning_signal(_data(["owner/repo"]))
    assert "Set GITHUB_TOKEN" in report
    assert "scan_repos" not in report


@pytest.mark.asyncio
async def test_signal_reports_token_remediation_when_token_invalid():
    # Configured repo + present-but-rejected token must NOT be reported as
    # live data; it maps to the token remediation (#2271).
    def _boom():
        raise _http_error(403)

    with patch(f"{_GH_MOD}.get_github_token", return_value="ghp_invalid"), patch(
        f"{_GH_MOD}.urllib.request.urlopen", side_effect=lambda *a, **k: _boom()
    ):
        report = await generate_morning_signal(_data(["owner/repo"]))
    assert "Set GITHUB_TOKEN" in report
    assert "Live data from GitHub" not in report
