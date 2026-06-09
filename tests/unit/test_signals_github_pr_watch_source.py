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
from kestrel_sovereign.signals.sources.github_pr_watch import (
    DEFAULT_TRIGGERS,
    SOURCE_NAME,
    WatchDecision,
    build_github_pr_activity_registration,
    build_signal_for_pr_change,
    changed_categories,
    compute_fingerprint,
    evaluate_pr_watch,
    normalize_pr_state,
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
