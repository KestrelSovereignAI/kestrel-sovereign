"""Tests for the talon.job_complete signal source (#1510).

Covers the registration + schema validation of the source. The talon
wake is now driven by the generic wait reconciler via
:class:`TalonWaitable` (Wave 2 of #1860) — the bespoke
``build_signal_for_completed_job`` builder and the ``talon_monitor`` cron
it served are retired, so the builder tests moved to the reconciler's
generic signal-construction coverage in ``test_wait_reconciler.py``. This
file pins the shape of the source itself so unrelated refactors don't
silently break the cognition wake path; end-to-end emit-on-transition is
covered by ``test_wait_reconciler.py``.
"""

from __future__ import annotations

import pytest

from kestrel_sdk.signals import SignalMode, Trust
from kestrel_sovereign.signals.sources.talon import (
    SOURCE_NAME,
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
