"""Integration: create_issue via kestrel-feature-github (#1579).

Closes Emma's gap from the orchestration session: ``gh issue create``
through ``ComputerUseFeature.shell`` always hits ``BinaryPolicy``
approval and (per #1575) the codex-native shell path. The supported
production route is the ``create_issue`` @tool exposed by
``kestrel-feature-github``, which runs through the normal feature
pipeline (PRE_TOOL_USE / SecurityHook / approval queue) over httpx
without touching ``BinaryPolicy``.

This test exercises the end-to-end path against a stub GitHub HTTP
endpoint — no real GitHub API call. It SKIPS gracefully when the
sibling package isn't installed so kestrel-sovereign's CI doesn't
hard-depend on an extracted feature.

Two defenses against accidentally hitting real GitHub:

1. ``httpx.AsyncClient`` is patched at the
   ``kestrel_feature_github.client`` module level so any client the
   feature constructs (including the lazy ``_get_client`` path)
   resolves to the mock — not just the pre-set ``_client`` attr.
2. ``GITHUB_TOKEN`` and ``GH_TOKEN`` are cleared from the env so
   even if the patch were to leak, the request would 401 instead of
   silently succeeding under a real user token.

Manual operator step to actually enable the feature for Emma:

  1. Install the package into Emma's venv:
       ``uv pip install -e /Volumes/data2/projects/kestrel-feature-github``
  2. Add to ``agent_data/emma/kestrel.toml``::
       [features.github]
       enabled = true
  3. Restart Emma (multi_agent host re-loads features on agent
     restart). Verify with::
       kestrel ask Emma "list your enabled features"
     and look for ``github``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Graceful skip when the sibling package isn't installed (CI without
# the editable install, prod containers that don't ship the feature).
pytest.importorskip(
    "kestrel_feature_github",
    reason="kestrel-feature-github not installed; install editable "
    "from /Volumes/data2/projects/kestrel-feature-github to run "
    "this integration test.",
)

from kestrel_sdk.tools.result import ToolResultStatus  # noqa: E402
from kestrel_feature_github.feature import GitHubFeature  # noqa: E402


@pytest.fixture
def stub_agent():
    """Minimal agent surface GitHubFeature reads at init / call time."""
    return SimpleNamespace(_agent_name="emma", did="did:test:emma")


@pytest.fixture(autouse=True)
def scrub_github_tokens(monkeypatch):
    """Defense in depth: even if the httpx patch is somehow bypassed,
    the request must NOT carry a real user token. A test that calls
    real GitHub previously created issue #1586 on KestrelSovereignAI's
    repo before this guard was added — never again."""
    # GitHubClient.__init__ resolves a token via GITHUB_PAT first,
    # then GITHUB_TOKEN, then GH_TOKEN. Clear ALL three so even a
    # leaky httpx patch can't authenticate against real GitHub.
    monkeypatch.delenv("GITHUB_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)


def _mock_httpx_response(status_code: int, json_payload: dict):
    response = MagicMock(status_code=status_code)
    response.json.return_value = json_payload
    response.raise_for_status.return_value = None
    response.text = ""
    return response


def _async_client_returning(post_resp):
    """Build an AsyncMock httpx.AsyncClient that:
    - reports is_closed=False so _get_client doesn't tear it down
    - returns ``post_resp`` from .post(...)
    - usable as an async context manager (some callers wrap)."""
    client = AsyncMock()
    client.is_closed = False
    client.post.return_value = post_resp
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.aclose = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_create_issue_returns_ok_against_stub_github(stub_agent):
    """End-to-end through ``GitHubFeature.create_github_issue`` (the
    @tool method) and ``GitHubClient.create_issue`` (the httpx layer)
    against a stub HTTP response that mimics GitHub's 201 Created."""
    feature = GitHubFeature(stub_agent)
    await feature.initialize()

    stub_issue = {
        "number": 4242,
        "id": 9999999,
        "html_url": (
            "https://github.com/KestrelSovereignAI/kestrel-sovereign/"
            "issues/4242"
        ),
        "title": "Test issue from integration test",
    }

    mock_client = _async_client_returning(
        _mock_httpx_response(201, stub_issue)
    )

    # Patch httpx.AsyncClient at the kestrel_feature_github.client
    # module level so the lazy ``_get_client`` path constructs OUR
    # mock — not a real httpx.AsyncClient. Patching only the
    # pre-cached ``_client`` attr is insufficient: _get_client
    # re-creates the client if ``is_closed`` is truthy, and a bare
    # MagicMock attr returns truthy by default → the real httpx
    # client would be created and a real GitHub API call would fire.
    with patch(
        "kestrel_feature_github.client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        # Bypass the token-config gate; the test patches the wire.
        feature.client._configured = True
        result = await feature.create_github_issue(
            repo="KestrelSovereignAI/kestrel-sovereign",
            title="Test issue from integration test",
            body="Body text",
        )

    assert result.status is ToolResultStatus.OK, result.error
    assert result.data["number"] == 4242
    assert "issues/4242" in result.data["url"]
    # The POST hit the expected GitHub endpoint shape with the
    # expected payload (title + body).
    call = mock_client.post.call_args
    assert call.args[0].endswith(
        "/repos/KestrelSovereignAI/kestrel-sovereign/issues"
    )
    posted_body = call.kwargs["json"]
    assert posted_body["title"] == "Test issue from integration test"
    assert posted_body["body"] == "Body text"


@pytest.mark.asyncio
async def test_create_issue_bypasses_shell_subprocess(stub_agent):
    """Regression guard: the feature path uses httpx, not shell. Any
    subprocess.run / Popen call would mean the path regressed to
    shelling out (the very pattern this ticket exists to avoid).
    A Kestrel BinaryPolicy DENY on ``gh`` therefore cannot block
    this path — it's not on the call graph at all."""
    feature = GitHubFeature(stub_agent)
    await feature.initialize()

    mock_client = _async_client_returning(
        _mock_httpx_response(201, {"number": 1, "html_url": "u"})
    )

    with patch(
        "kestrel_feature_github.client.httpx.AsyncClient",
        return_value=mock_client,
    ), patch("subprocess.run") as fake_run, \
         patch("subprocess.Popen") as fake_popen:
        feature.client._configured = True
        await feature.create_github_issue(
            repo="o/r", title="x", body="y",
        )

    assert fake_run.call_count == 0, (
        "GitHub create_issue must not shell out — feature path uses httpx"
    )
    assert fake_popen.call_count == 0
    assert mock_client.post.call_count == 1
