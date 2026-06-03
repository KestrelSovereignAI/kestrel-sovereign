"""Tests for the talon.job_complete signal source (#1510).

Covers the registration, schema validation, and signal-builder
contract used by ``TalonCoordinatorFeature.talon_monitor``. The
end-to-end emit-on-transition behaviour lives in
``tests/unit/test_talon_env_and_health.py``; this file pins the
shape of the source itself so unrelated refactors don't silently
break the cognition wake path.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import Signal, SignalMode, Trust
from kestrel_sovereign.signals.sources.talon import (
    SOURCE_NAME,
    build_signal_for_completed_job,
    build_talon_job_complete_registration,
)


def test_source_name_is_canonical():
    assert SOURCE_NAME == "talon.job_complete"


def test_registration_shape():
    reg = build_talon_job_complete_registration()
    assert reg.name == SOURCE_NAME
    assert SignalMode.COGNITION in reg.allowed_modes
    assert reg.default_mode is SignalMode.COGNITION
    assert reg.trust is Trust.TRUSTED
    # Cycle protection: a local-only signal, can't self-loop.
    assert reg.allow_self_loops is False
    # Prompt template must exist on disk — without it the dispatcher
    # has nothing to render at cognition time.
    assert reg.prompt_template.is_file(), (
        f"prompt template missing at {reg.prompt_template}"
    )


def test_schema_requires_job_id_and_status():
    reg = build_talon_job_complete_registration()
    with pytest.raises(ValueError, match="job_id"):
        reg.schema({"status": "complete"})
    with pytest.raises(ValueError, match="status"):
        reg.schema({"job_id": "abc"})


def test_schema_rejects_non_dict():
    reg = build_talon_job_complete_registration()
    with pytest.raises(ValueError, match="must be a dict"):
        reg.schema("not-a-dict")  # type: ignore[arg-type]


def test_schema_injects_defaults_for_prompt_template_placeholders():
    """The prompt template references repo/issue/label/log_path/etc.
    Callers that build a payload missing those keys must still render
    cleanly — the schema injects empty-string defaults so the
    dispatcher's ``_render_prompt`` never KeyErrors on legacy
    payloads.
    """
    reg = build_talon_job_complete_registration()
    payload = reg.schema({"job_id": "abc", "status": "complete"})
    for key in (
        "repo", "issue", "label", "returncode",
        "log_path", "log_tail", "started_at", "completed_at",
    ):
        assert key in payload, f"schema should default-fill {key}"


def test_builder_emits_cognition_signal_with_dedupe_key():
    info = {
        "status": "complete",
        "repo": "x/y", "issue": 7,
        "label": "claim:x/y#7",
        "returncode": 0,
        "log_path": "/tmp/x.log",
        "started_at": "2026-06-03T00:00:00+00:00",
        "completed_at": "2026-06-03T00:01:00+00:00",
    }
    sig = build_signal_for_completed_job(
        "abc123", info, target_agent="did:test:agent",
        log_tail="last line\n",
    )
    assert isinstance(sig, Signal)
    assert sig.mode is SignalMode.COGNITION
    assert sig.source == SOURCE_NAME
    assert sig.target_agent == "did:test:agent"
    # dedupe_key is ``job_id:status`` — collapses double-fires of
    # the SAME terminal state across adjacent polls, but lets a
    # status correction (e.g. finished_unknown → failed once the
    # sidecar lands) emit a fresh wake.
    assert sig.dedupe_key == "abc123:complete"
    assert sig.payload["job_id"] == "abc123"
    assert sig.payload["status"] == "complete"
    assert sig.payload["log_tail"] == "last line\n"


def test_builder_coerces_non_string_payload_fields():
    """``issue`` is often an int. The signal payload must coerce to
    str so the prompt template's {payload[issue]} interpolation
    never raises on a non-string.
    """
    info = {"status": "failed", "repo": "x/y", "issue": 42, "returncode": 9}
    sig = build_signal_for_completed_job(
        "j", info, target_agent="did:test:agent",
    )
    assert sig.payload["issue"] == "42"
    assert sig.payload["returncode"] == "9"


def test_dedupe_key_changes_with_status_correction():
    """A status correction (e.g. ``finished_unknown`` → ``failed``
    once the sidecar lands a poll later) must produce a different
    dedupe_key so the dispatcher's coalescing window does not
    swallow the corrected terminal wake.
    """
    info_a = {"status": "finished_unknown"}
    info_b = {"status": "failed", "returncode": 17}
    a = build_signal_for_completed_job(
        "same-job", info_a, target_agent="t",
    )
    b = build_signal_for_completed_job(
        "same-job", info_b, target_agent="t",
    )
    assert a.dedupe_key != b.dedupe_key
