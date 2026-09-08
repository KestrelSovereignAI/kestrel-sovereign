"""An under-scoped token must not read as a CI gate that is merely slow.

This provider's contract is that every observed state is either terminal or
provably still-progressing (#2939). A credential that can read a PR but not
its check runs is neither: nothing is progressing, and nothing changes until
a human edits the token's permissions.

Measured 2026-09-07 against a real fine-grained PAT:

    /pulls/20                  200
    /commits/<sha>/check-runs  403      (gh's token: 200)
    /commits/<sha>/status      403

Reported as ordinary PENDING that is indistinguishable from a queued check,
and an agent waited 920s on it.
"""

import pytest

from kestrel_sovereign.features.scheduler import ci_wait_provider as mod
from kestrel_sovereign.features.scheduler.ci_wait_provider import CIWaitable, Outcome as _OUTCOME
from kestrel_sovereign.signals.sources.github_pr_watch import (
    PRWatchAuthError,
    PRWatchNetworkError,
)

HANDLE = "KestrelSovereignAI/kestrel-feature-talon#20"


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.features.strategic_memory.github_integration"
        ".get_github_token",
        lambda: "ghp_fake",
    )


def _provider(fetch):
    from unittest.mock import MagicMock

    p = CIWaitable(MagicMock())
    p._fetch = fetch
    return p


@pytest.mark.asyncio
async def test_underscoped_token_is_flagged_as_a_permission_block():
    async def fetch(repo, number, token):
        raise mod._UnderscopedToken("check-runs returned 403")

    st = await _provider(fetch).poll(HANDLE)
    assert st.data.get("blocked") == "permission"
    assert st.data.get("actionable") is True
    assert "BLIND" in st.summary


@pytest.mark.asyncio
async def test_a_transient_auth_blip_is_still_plain_auth():
    """Control: the distinction must be real, not a blanket relabelling. A
    genuine blip has to keep its old classification so the watch keeps
    waiting through it."""
    async def fetch(repo, number, token):
        raise PRWatchAuthError("GitHub returned 401 for the PR read")

    st = await _provider(fetch).poll(HANDLE)
    assert st.data.get("blocked") == "auth"
    assert st.data.get("actionable") is not True
    assert "BLIND" not in st.summary


@pytest.mark.asyncio
async def test_a_network_blip_is_still_network():
    async def fetch(repo, number, token):
        raise PRWatchNetworkError("connection reset")

    st = await _provider(fetch).poll(HANDLE)
    assert st.data.get("blocked") == "network"


@pytest.mark.asyncio
async def test_neither_case_is_terminal():
    """Widening the token must let the watch complete, so neither may settle
    terminally — the provider must never fabricate a merge or a failure from
    a credential problem."""
    for exc in (mod._UnderscopedToken("x"), PRWatchAuthError("y")):
        async def fetch(repo, number, token, _e=exc):
            raise _e

        st = await _provider(fetch).poll(HANDLE)
        assert st.outcome is _OUTCOME.PENDING


@pytest.mark.asyncio
async def test_the_message_names_the_missing_permission():
    """An operator-facing block has to say what to change, or it is just a
    different way of being stuck."""
    async def fetch(repo, number, token):
        raise mod._UnderscopedToken(
            "the PR read succeeded but check-runs returned an authorization "
            "error. The token is valid and is missing the Checks / "
            "Commit-statuses read permission; this will not resolve on its own."
        )

    st = await _provider(fetch).poll(HANDLE)
    assert "Checks" in st.summary
    assert "will not resolve on its own" in st.summary


# ---------------------------------------------------------------------------
# The detection itself, not just poll()'s handling of it
# ---------------------------------------------------------------------------
# Every test above stubs _fetch, so none of them exercise the classification
# that decides an under-scoped token from a blip. These drive the real _fetch
# with the HTTP layer mocked to the exact shape measured on 2026-09-07.

@pytest.mark.asyncio
async def test_fetch_classifies_pr_ok_then_every_check_endpoint_403_as_underscoped(
    monkeypatch,
):
    """The real discrimination: PR read succeeds, every check read is refused.

    Note what this now takes. A refused Checks API alone is no longer blind —
    the provider falls back to the Actions API — so reaching
    ``_UnderscopedToken`` requires check-runs, actions-runs AND status to all
    refuse, which is what this fake does for every non-``/pulls/`` URL.
    """
    async def fake_get(url, *, token, timeout, ref):
        if "/pulls/" in url:
            return {"head": {"sha": "deadbeef"}}
        # ``status_code`` is load-bearing: only a 403 is an endpoint refusing
        # a valid token, and only that becomes _UnderscopedToken. A 401 would
        # mean the credential itself is finished and stays plain auth.
        raise PRWatchAuthError(f"GitHub returned 403 for {ref}", status_code=403)

    monkeypatch.setattr(
        "kestrel_sovereign.signals.sources.github_pr_watch._github_get", fake_get
    )

    from unittest.mock import MagicMock

    with pytest.raises(mod._UnderscopedToken):
        await CIWaitable(MagicMock())._fetch("o/r", 20, "ghp_fake")


@pytest.mark.asyncio
async def test_fetch_leaves_a_failing_pr_read_as_plain_auth(monkeypatch):
    """Control: when the PR read ITSELF is refused, the token may simply be
    bad or expired — that is the transient class, and must not be relabelled
    as a permission gap."""
    async def fake_get(url, *, token, timeout, ref):
        raise PRWatchAuthError(f"GitHub returned 401 for {ref}", status_code=401)

    monkeypatch.setattr(
        "kestrel_sovereign.signals.sources.github_pr_watch._github_get", fake_get
    )

    from unittest.mock import MagicMock

    with pytest.raises(PRWatchAuthError):
        await CIWaitable(MagicMock())._fetch("o/r", 20, "ghp_fake")
