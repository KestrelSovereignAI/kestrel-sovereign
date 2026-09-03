"""The bounded failure reason a scheduled tool hands back (#3184).

A scheduled tool that fails raises ``RuntimeError: scheduled tool <name>
failed``, deliberately WITHOUT the tool's error prose — the dispatcher persists
exception text outside the bounded result-summary channel, so free-form text
must not cross that boundary.

The cost of stripping everything was measured: ``signal_dispatch`` failed daily
for five consecutive days and every ``signal_log`` row read exactly
``RuntimeError: scheduled tool signal_dispatch failed``. Nothing said why, and
nothing distinguished day one from day five.

A ``reason_code`` is not prose. It is a short controlled token the tool itself
chose, so it can cross the boundary while the rationale above is preserved.
"""

from __future__ import annotations

import json

import pytest
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome

from kestrel_sovereign.signals.sources.scheduler import _failure_reason_code


def test_reason_code_is_read_from_a_toolresult():
    result = ToolResult.failed("prose that must not travel", data={
        "reason_code": "DISPATCH_CAPABILITY_UNAVAILABLE",
    })
    assert _failure_reason_code(result) == "DISPATCH_CAPABILITY_UNAVAILABLE"


def test_reason_code_is_read_from_a_dict_result():
    assert _failure_reason_code(
        {"data": {"reason_code": "WORKFLOW_RUN_REJECTED"}}
    ) == "WORKFLOW_RUN_REJECTED"
    # Some callers hand back a flat dict rather than a nested payload.
    assert _failure_reason_code({"reason_code": "WORKFLOW_RUN_REJECTED"}) == (
        "WORKFLOW_RUN_REJECTED"
    )


def test_absent_reason_code_yields_empty_so_the_message_is_unchanged():
    assert _failure_reason_code(ToolResult.failed("no code here")) == ""
    assert _failure_reason_code({"error": "boom"}) == ""
    assert _failure_reason_code(None) == ""


@pytest.mark.parametrize("value", [
    "a code with spaces",           # prose, not a token
    "line\nbreak",                  # multiline text
    "x" * 65,                       # longer than the bound
    "reason: something happened",   # punctuation-bearing prose
    "",
    "   ",
])
def test_prose_shaped_values_never_cross_the_boundary(value):
    """The whole point is that free-form text cannot ride out on this field."""
    assert _failure_reason_code({"data": {"reason_code": value}}) == ""


def test_a_non_string_reason_code_is_ignored():
    assert _failure_reason_code({"data": {"reason_code": {"nested": "obj"}}}) == ""
    assert _failure_reason_code({"data": {"reason_code": 42}}) == ""


def test_underscores_and_digits_are_token_shaped():
    assert _failure_reason_code({"data": {"reason_code": "ERR_2249_SKIPPED"}}) == (
        "ERR_2249_SKIPPED"
    )


# --------------------------------------------------------------------------
# The raise itself — the helper being right is not the same as the message
# carrying it. Testing only the helper would cover a different door than the fix.
# --------------------------------------------------------------------------

from kestrel_sovereign.signals.sources.scheduler import (
    _require_successful_task_result,
)


def test_the_raised_message_names_the_reason_code():
    failed = ToolResult.failed(
        "Dispatch workflow 'fleet_coding_pipeline' did not start: "
        "workflow definition not found: fleet_coding_pipeline",
        data={"reason_code": "WORKFLOW_RUN_REJECTED", "dispatched": False},
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("signal_dispatch", failed)

    message = str(excinfo.value)
    assert "signal_dispatch" in message
    assert "WORKFLOW_RUN_REJECTED" in message, (
        f"the bounded reason did not reach the exception: {message!r}"
    )


def test_the_raised_message_never_carries_the_error_prose():
    """The redaction rationale must survive this change."""
    failed = ToolResult.failed(
        "workflow definition not found: fleet_coding_pipeline",
        data={"reason_code": "WORKFLOW_RUN_REJECTED"},
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("signal_dispatch", failed)

    message = str(excinfo.value)
    assert "definition not found" not in message
    assert "fleet_coding_pipeline" not in message


def test_without_a_reason_code_the_message_is_exactly_as_before():
    failed = ToolResult.failed("something went wrong", data={})
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("signal_dispatch", failed)
    assert str(excinfo.value) == "scheduled tool signal_dispatch failed"


def test_the_json_envelope_door_carries_the_reason_code_too():
    # Built-in handlers return a JSON object STRING; the code must be read
    # from the decoded envelope, not from the raw string.
    envelope = json.dumps(
        {"success": False, "error": "boom", "data": {"reason_code": "TRASH_SWEEP_BLOCKED"}}
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result(
            "trash_retention", envelope, decode_json_envelope=True
        )
    assert str(excinfo.value) == "scheduled tool trash_retention failed (TRASH_SWEEP_BLOCKED)"


def test_a_failed_outcome_names_its_reason_code():
    outcome = ScheduledTaskOutcome(
        status="failed",
        result_text=json.dumps({"error": "sleep_failed"}),
        reason_code="SLEEP_FAILED",
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome)
    assert str(excinfo.value) == "scheduled task sleep returned failed (SLEEP_FAILED)"


def test_a_failed_outcome_without_a_code_reads_exactly_as_before():
    outcome = ScheduledTaskOutcome(status="failed", result_text="{}")
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome)
    assert str(excinfo.value) == "scheduled task sleep returned failed"


def test_a_failed_outcome_prose_reason_never_crosses_the_boundary():
    outcome = ScheduledTaskOutcome(
        status="failed",
        result_text="{}",
        reason_code="the semantic sweep hit: /Users/someone/private",
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome)
    assert str(excinfo.value) == "scheduled task sleep returned failed"
