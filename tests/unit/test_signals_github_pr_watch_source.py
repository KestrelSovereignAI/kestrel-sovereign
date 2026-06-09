"""Tests for the github.pr_activity signal source and the change-detection
core that backs the github_pr_watch cron task (#1618).

Covers the registration/schema/builder contract plus the pure
change-detection logic: a fingerprint change in a watched, triggered
category emits; a no-op poll (same fingerprint) and a first observation
do not.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import SignalMode, Trust
from kestrel_sovereign.signals.sources import github_pr_watch
from kestrel_sovereign.signals.sources.github_pr_watch import (
    DEFAULT_TRIGGERS,
    SOURCE_NAME,
    WatchDecision,
    build_github_pr_activity_registration,
    build_signal_for_pr_change,
    changed_categories,
    compute_fingerprint,
    evaluate_pr_watch,
    fetch_pr_state,
    normalize_pr_state,
    summarize_checks,
)


# A representative open-PR payload.
def _pr(**overrides):
    base = {
        "state": "open",
        "merged": False,
        "comments": 2,
        "review_comments": 1,
        "updated_at": "2026-06-09T16:00:00Z",
        "head": {"sha": "abc123"},
        "checks_status": "success",
        "mergeable_state": "clean",
        "html_url": "https://github.com/owner/name/pull/1614",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Registration / schema / builder
# ---------------------------------------------------------------------------


def test_source_name_is_canonical():
    assert SOURCE_NAME == "github.pr_activity"


def test_registration_shape():
    reg = build_github_pr_activity_registration()
    assert reg.name == SOURCE_NAME
    assert reg.default_mode is SignalMode.COGNITION
    assert SignalMode.COGNITION in reg.allowed_modes
    assert reg.trust is Trust.TRUSTED
    assert reg.allow_self_loops is False
    assert reg.prompt_template.is_file(), (
        f"prompt template missing at {reg.prompt_template}"
    )


def test_schema_requires_repo_and_number():
    reg = build_github_pr_activity_registration()
    with pytest.raises(ValueError, match="repo"):
        reg.schema({"number": "1614"})
    with pytest.raises(ValueError, match="number"):
        reg.schema({"repo": "owner/name"})


def test_schema_rejects_non_dict():
    reg = build_github_pr_activity_registration()
    with pytest.raises(ValueError, match="must be a dict"):
        reg.schema("nope")  # type: ignore[arg-type]


def test_schema_injects_template_defaults():
    reg = build_github_pr_activity_registration()
    payload = reg.schema({"repo": "owner/name", "number": "1614"})
    for key in (
        "state", "merged", "comments", "review_comments",
        "checks_status", "changed", "html_url", "updated_at",
    ):
        assert key in payload


def test_build_signal_envelope():
    decision = evaluate_pr_watch(
        _pr(comments=5),
        last_fingerprint="stale",
        last_normalized=normalize_pr_state(_pr(comments=2)),
    )
    sig = build_signal_for_pr_change(
        repo="owner/name",
        number=1614,
        decision=decision,
        target_agent="did:test:agent",
        html_url="https://github.com/owner/name/pull/1614",
    )
    assert sig.source == SOURCE_NAME
    assert sig.mode is SignalMode.COGNITION
    assert sig.target_agent == "did:test:agent"
    assert sig.payload["repo"] == "owner/name"
    assert sig.payload["number"] == "1614"
    assert "comments" in sig.payload["changed"]
    # dedupe key pins one wake per observed fingerprint.
    assert sig.dedupe_key.startswith("owner/name#1614:")


# ---------------------------------------------------------------------------
# Pure change-detection core
# ---------------------------------------------------------------------------


def test_normalize_extracts_head_sha_from_pr_and_issue_shapes():
    assert normalize_pr_state(_pr())["head_sha"] == "abc123"
    # Issue payloads have no nested head; fall back to head_sha.
    assert normalize_pr_state({"head_sha": "xyz"})["head_sha"] == "xyz"


def test_fingerprint_is_stable_for_same_state():
    a = compute_fingerprint(normalize_pr_state(_pr()))
    b = compute_fingerprint(normalize_pr_state(_pr()))
    assert a == b


def test_fingerprint_changes_when_comments_change():
    a = compute_fingerprint(normalize_pr_state(_pr(comments=2)))
    b = compute_fingerprint(normalize_pr_state(_pr(comments=3)))
    assert a != b


def test_changed_categories_maps_fields():
    prev = normalize_pr_state(_pr())
    curr = normalize_pr_state(_pr(state="closed", merged=True))
    cats = changed_categories(prev, curr)
    assert "state" in cats
    assert "merge" in cats


def test_first_observation_does_not_signal():
    decision = evaluate_pr_watch(_pr(), last_fingerprint=None)
    assert decision.should_signal is False
    assert decision.reason == "first_observation"
    assert decision.fingerprint  # baseline persisted by caller


def test_no_op_poll_does_not_signal():
    prev = normalize_pr_state(_pr())
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(), last_fingerprint=fp, last_normalized=prev,
    )
    assert decision.should_signal is False
    assert decision.reason == "no_change"


def test_new_comment_signals():
    prev = normalize_pr_state(_pr(comments=2))
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(comments=3), last_fingerprint=fp, last_normalized=prev,
    )
    assert decision.should_signal is True
    assert "comments" in decision.matched


def test_check_status_change_signals():
    prev = normalize_pr_state(_pr(checks_status="pending"))
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(checks_status="failure"), last_fingerprint=fp, last_normalized=prev,
    )
    assert decision.should_signal is True
    assert "checks" in decision.matched


def test_state_and_merge_change_signals():
    prev = normalize_pr_state(_pr(state="open", merged=False))
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(state="closed", merged=True),
        last_fingerprint=fp,
        last_normalized=prev,
    )
    assert decision.should_signal is True
    assert {"state", "merge"} & decision.matched


def test_bare_updated_at_bump_does_not_signal_under_defaults():
    """A change only to updated_at (category 'update') is not in the
    default triggers, so it must not wake the agent."""
    prev = normalize_pr_state(_pr(updated_at="2026-06-09T16:00:00Z"))
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(updated_at="2026-06-09T17:00:00Z"),
        last_fingerprint=fp,
        last_normalized=prev,
    )
    assert decision.should_signal is False
    assert decision.reason == "change_not_in_triggers"
    assert "update" in decision.changed


def test_any_trigger_wakes_on_updated_at_bump():
    prev = normalize_pr_state(_pr(updated_at="2026-06-09T16:00:00Z"))
    fp = compute_fingerprint(prev)
    decision = evaluate_pr_watch(
        _pr(updated_at="2026-06-09T17:00:00Z"),
        last_fingerprint=fp,
        last_normalized=prev,
        triggers=["any"],
    )
    assert decision.should_signal is True


def test_default_triggers_exclude_update():
    assert "update" not in DEFAULT_TRIGGERS
    assert "state" in DEFAULT_TRIGGERS


# ---------------------------------------------------------------------------
# Real check/status fingerprinting (#1618 follow-up)
#
# Standard GitHub pull/issue payloads carry NO aggregate ``checks_status``
# field. The watcher must fetch the head commit's real check runs + commit
# statuses and fingerprint those, so a CI transition wakes the agent and a
# no-op poll does not. These tests use realistic payloads that omit
# ``checks_status`` entirely.
# ---------------------------------------------------------------------------


# A realistic PR payload exactly as GitHub returns it — note: NO
# ``checks_status`` key anywhere.
def _real_pr(**overrides):
    base = {
        "state": "open",
        "merged": False,
        "comments": 2,
        "review_comments": 1,
        "updated_at": "2026-06-09T16:00:00Z",
        "head": {"sha": "abc123"},
        "mergeable_state": "clean",
        "html_url": "https://github.com/owner/name/pull/1614",
    }
    base.update(overrides)
    return base


def test_realistic_payload_has_no_checks_status_key():
    # Guard: the fixture must not leak a synthetic field.
    assert "checks_status" not in _real_pr()


def test_summarize_checks_empty_when_nothing():
    assert summarize_checks(None, None) == ""
    assert summarize_checks({"check_runs": []}, {"state": "", "statuses": []}) == ""


def test_summarize_checks_captures_check_run_conclusions():
    runs = {"check_runs": [
        {"name": "build", "status": "completed", "conclusion": "success"},
        {"name": "test", "status": "completed", "conclusion": "failure"},
    ]}
    combined = {"state": "failure", "statuses": []}
    summary = summarize_checks(runs, combined)
    assert "build=completed/success" in summary
    assert "test=completed/failure" in summary
    assert "combined=failure" in summary


def test_summarize_checks_is_order_independent():
    a = summarize_checks(
        {"check_runs": [
            {"name": "build", "status": "completed", "conclusion": "success"},
            {"name": "test", "status": "completed", "conclusion": "success"},
        ]},
        {"state": "success"},
    )
    b = summarize_checks(
        {"check_runs": [
            {"name": "test", "status": "completed", "conclusion": "success"},
            {"name": "build", "status": "completed", "conclusion": "success"},
        ]},
        {"state": "success"},
    )
    assert a == b


def test_summarize_checks_transition_changes_summary():
    running = summarize_checks(
        {"check_runs": [
            {"name": "ci", "status": "in_progress", "conclusion": None},
        ]},
        {"state": "pending"},
    )
    done = summarize_checks(
        {"check_runs": [
            {"name": "ci", "status": "completed", "conclusion": "failure"},
        ]},
        {"state": "failure"},
    )
    assert running != done


def test_summarize_checks_includes_legacy_commit_statuses():
    summary = summarize_checks(
        {"check_runs": []},
        {"state": "success", "statuses": [
            {"context": "ci/jenkins", "state": "success"},
        ]},
    )
    assert "status:ci/jenkins=success" in summary


@pytest.mark.asyncio
async def test_fetch_pr_state_derives_checks_from_real_apis(monkeypatch):
    """A realistic PR payload (no checks_status) gains a derived
    checks_status fetched from the Checks + Statuses APIs."""
    calls = []

    async def fake_get(url, *, token, timeout, ref):
        calls.append(url)
        if url.endswith("/pulls/1614"):
            return _real_pr()
        if url.endswith("/commits/abc123/check-runs"):
            return {"check_runs": [
                {"name": "build", "status": "completed", "conclusion": "success"},
            ]}
        if url.endswith("/commits/abc123/status"):
            return {"state": "success", "statuses": []}
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)

    state = await fetch_pr_state("owner/name", 1614, token="t", kind="pr")
    # Head-commit check/status endpoints were both queried.
    assert any("check-runs" in u for u in calls)
    assert any(u.endswith("/commits/abc123/status") for u in calls)
    # The derived summary reflects the real check run, not a synthetic field.
    assert "build=completed/success" in state["checks_status"]


@pytest.mark.asyncio
async def test_fetch_pr_state_ci_transition_changes_fingerprint(monkeypatch):
    """CI completing/failing on the same head SHA must change the
    fingerprint even though the pull payload itself is unchanged."""
    check_runs_box = {"value": {"check_runs": [
        {"name": "ci", "status": "in_progress", "conclusion": None},
    ]}}
    status_box = {"value": {"state": "pending", "statuses": []}}

    async def fake_get(url, *, token, timeout, ref):
        if url.endswith("/pulls/1614"):
            return _real_pr()
        if url.endswith("/check-runs"):
            return check_runs_box["value"]
        if url.endswith("/status"):
            return status_box["value"]
        raise AssertionError(url)

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)

    first = await fetch_pr_state("owner/name", 1614, token="t", kind="pr")
    fp1 = compute_fingerprint(normalize_pr_state(first))

    # CI finishes (failure) on the same commit — no pull-payload change.
    check_runs_box["value"] = {"check_runs": [
        {"name": "ci", "status": "completed", "conclusion": "failure"},
    ]}
    status_box["value"] = {"state": "failure", "statuses": []}

    second = await fetch_pr_state("owner/name", 1614, token="t", kind="pr")
    fp2 = compute_fingerprint(normalize_pr_state(second))

    assert fp1 != fp2
    decision = evaluate_pr_watch(
        second,
        last_fingerprint=fp1,
        last_normalized=normalize_pr_state(first),
    )
    assert decision.should_signal is True
    assert "checks" in decision.matched


@pytest.mark.asyncio
async def test_fetch_pr_state_issue_skips_checks(monkeypatch):
    """An issue has no head SHA, so no check/status calls are made and
    checks_status stays empty."""
    calls = []

    async def fake_get(url, *, token, timeout, ref):
        calls.append(url)
        if url.endswith("/issues/1614"):
            return {
                "state": "open",
                "comments": 3,
                "updated_at": "2026-06-09T16:00:00Z",
            }
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(github_pr_watch, "_github_get", fake_get)

    state = await fetch_pr_state("owner/name", 1614, token="t", kind="issue")
    assert not any("check-runs" in u or u.endswith("/status") for u in calls)
    assert normalize_pr_state(state)["checks_status"] == ""
