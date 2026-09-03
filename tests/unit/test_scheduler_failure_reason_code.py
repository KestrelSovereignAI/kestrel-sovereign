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
import pathlib

import pytest
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
from kestrel_sovereign.agent.sleep import SLEEP_FAILURE_REASONS
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature

#: What the scheduler feature declares for its own sleep built-in.
SLEEP_DECLARED = SchedulerFeature.tool_reason_codes["sleep"]

from kestrel_sovereign.signals.sources.scheduler import _failure_reason_code

#: What the strategic-memory feature declares for ``signal_dispatch``.
from kestrel_sovereign.features.strategic_memory.feature import (
    SIGNAL_DISPATCH_REASON_CODES,
)


def _code(result, declared=SIGNAL_DISPATCH_REASON_CODES, task_name="signal_dispatch"):
    return _failure_reason_code(result, task_name=task_name, declared=declared)


def test_reason_code_is_read_from_a_toolresult():
    result = ToolResult.failed("prose that must not travel", data={
        "reason_code": "DISPATCH_CAPABILITY_UNAVAILABLE",
    })
    assert _code(result) == "DISPATCH_CAPABILITY_UNAVAILABLE"


def test_reason_code_is_read_from_a_dict_result():
    assert _code({"data": {"reason_code": "WORKFLOW_RUN_REJECTED"}}) == (
        "WORKFLOW_RUN_REJECTED"
    )
    # Some callers hand back a flat dict rather than a nested payload.
    assert _code({"reason_code": "WORKFLOW_RUN_REJECTED"}) == "WORKFLOW_RUN_REJECTED"


def test_absent_reason_code_yields_empty_so_the_message_is_unchanged(caplog):
    with caplog.at_level("WARNING"):
        assert _code(ToolResult.failed("no code here")) == ""
        assert _code({"error": "boom"}) == ""
        assert _code(None) == ""
        assert _code({"data": {"reason_code": ""}}) == ""
        assert _code({"data": {"reason_code": "   "}}) == ""
    # Not setting a code is the normal case, not a producer defect.
    assert "has not declared" not in caplog.text


@pytest.mark.parametrize("value", [
    "a code with spaces",                  # prose, not a token
    "line\nbreak",                         # multiline text
    "x" * 65,                              # longer than the bound
    "reason: something happened",          # punctuation-bearing prose
    "claim_denied_acme_repo",              # an identifier wearing a token's shape
    "CLAIM_DENIED_ACME_PRIVATE_PAYROLL",   # the same identifier in constant spelling
    "Claim_Denied",                        # mixed case
    "ERR_2249_SKIPPED",                    # a perfectly good token nobody declared
])
def test_undeclared_values_never_cross_the_boundary(value):
    """The door is a vocabulary, not a shape: a value the owning feature did
    not declare is dropped however constant-like it looks. A shape rule let
    ``CLAIM_DENIED_<REPO>`` ride a private repo name into signal_log.error."""
    assert _code({"data": {"reason_code": value}}) == ""


def test_an_undeclared_value_is_logged_by_tool_name_never_by_value(caplog):
    with caplog.at_level("WARNING"):
        assert _code({"data": {"reason_code": "CLAIM_DENIED_ACME_PRIVATE_PAYROLL"}}) == ""
    assert "signal_dispatch" in caplog.text
    assert "has not declared" in caplog.text
    assert "ACME" not in caplog.text


def test_a_non_string_reason_code_is_ignored():
    assert _code({"data": {"reason_code": {"nested": "obj"}}}) == ""
    assert _code({"data": {"reason_code": 42}}) == ""


def test_a_declared_code_is_admitted_whatever_its_spelling():
    # Membership is the rule; the producer's house style (lowercase, digits)
    # is its own business. Shape would have dropped the first two.
    declared = frozenset({"catalog_workload_unavailable", "err_2249_skipped", "ERR_2249_SKIPPED"})
    assert _code({"data": {"reason_code": "catalog_workload_unavailable"}}, declared) == (
        "catalog_workload_unavailable"
    )
    assert _code({"data": {"reason_code": "ERR_2249_SKIPPED"}}, declared) == "ERR_2249_SKIPPED"


def test_a_declared_code_that_is_prose_still_cannot_cross():
    # Both fences hold: declaring prose does not unlock it.
    declared = frozenset({"the sweep hit /Users/someone/private"})
    assert _code({"data": {"reason_code": "the sweep hit /Users/someone/private"}}, declared) == ""


def test_nothing_declared_means_nothing_crosses():
    assert _code({"data": {"reason_code": "WORKFLOW_RUN_REJECTED"}}, frozenset()) == ""


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
        _require_successful_task_result(
            "signal_dispatch", failed, declared_reason_codes=SIGNAL_DISPATCH_REASON_CODES
        )

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
        _require_successful_task_result(
            "signal_dispatch", failed, declared_reason_codes=SIGNAL_DISPATCH_REASON_CODES
        )

    message = str(excinfo.value)
    assert "definition not found" not in message
    assert "fleet_coding_pipeline" not in message


def test_without_a_reason_code_the_message_is_exactly_as_before():
    failed = ToolResult.failed("something went wrong", data={})
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result(
            "signal_dispatch", failed, declared_reason_codes=SIGNAL_DISPATCH_REASON_CODES
        )
    assert str(excinfo.value) == "scheduled tool signal_dispatch failed"


def test_the_raised_message_drops_a_code_the_owner_did_not_declare():
    failed = ToolResult.failed(
        "claim denied", data={"reason_code": "CLAIM_DENIED_ACME_PRIVATE_PAYROLL"},
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result(
            "signal_dispatch", failed, declared_reason_codes=SIGNAL_DISPATCH_REASON_CODES
        )
    assert str(excinfo.value) == "scheduled tool signal_dispatch failed"


def test_the_json_envelope_door_carries_the_reason_code_too():
    # Built-in handlers return a JSON object STRING; the code must be read
    # from the decoded envelope, not from the raw string.
    envelope = json.dumps(
        {"success": False, "error": "boom", "data": {"reason_code": "TRASH_SWEEP_BLOCKED"}}
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result(
            "trash_retention",
            envelope,
            declared_reason_codes=frozenset({"TRASH_SWEEP_BLOCKED"}),
            decode_json_envelope=True,
        )
    assert str(excinfo.value) == "scheduled tool trash_retention failed (TRASH_SWEEP_BLOCKED)"


def test_a_failed_outcome_names_its_reason_code():
    outcome = ScheduledTaskOutcome(
        status="failed",
        result_text=json.dumps({"error": "sleep_failed"}),
        reason_code="SLEEP_FAILED",
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome, declared_reason_codes=SLEEP_DECLARED)
    assert str(excinfo.value) == "scheduled task sleep returned failed (SLEEP_FAILED)"


def test_a_failed_outcome_without_a_code_reads_exactly_as_before():
    outcome = ScheduledTaskOutcome(status="failed", result_text="{}")
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome, declared_reason_codes=SLEEP_DECLARED)
    assert str(excinfo.value) == "scheduled task sleep returned failed"


def test_a_failed_outcome_prose_reason_never_crosses_the_boundary():
    outcome = ScheduledTaskOutcome(
        status="failed",
        result_text="{}",
        reason_code="the semantic sweep hit: /Users/someone/private",
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome, declared_reason_codes=SLEEP_DECLARED)
    assert str(excinfo.value) == "scheduled task sleep returned failed"


def test_a_failed_outcome_carries_the_producers_lowercase_code():
    # The code is in the producer's closed vocabulary, so the outcome door
    # admits it (lowercase and all); shape alone would not have been enough.
    outcome = ScheduledTaskOutcome(
        status="failed", result_text="{}",
        reason_code="semantic_artifact_expiry_sweep_failed",
    )
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome, declared_reason_codes=SLEEP_DECLARED)
    assert str(excinfo.value) == (
        "scheduled task sleep returned failed (semantic_artifact_expiry_sweep_failed)"
    )


def test_the_outcome_door_bounds_by_declaration_not_shape(caplog):
    # An identifier wearing a token's shape is dropped at this door exactly
    # as the tool door drops it: one rule, two doors — and the drop is
    # logged by task name so the producer learns to declare it.
    outcome = ScheduledTaskOutcome(
        status="failed", result_text="{}", reason_code="CLAIM_DENIED_ACME_REPO",
    )
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("sleep", outcome, declared_reason_codes=SLEEP_DECLARED)
    assert str(excinfo.value) == "scheduled task sleep returned failed"
    assert "sleep" in caplog.text and "has not declared" in caplog.text
    assert "ACME" not in caplog.text


def test_the_outcome_door_honours_a_features_own_declaration():
    # The reviewer's scenario: a feature's built-in returns a failed outcome
    # with a code it declared — the same declaration mechanism the tool door
    # uses, not a hardcoded set in the core signals module.
    outcome = ScheduledTaskOutcome(status="failed", result_text="{}", reason_code="T_BUSY")
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("T", outcome, declared_reason_codes=frozenset({"T_BUSY"}))
    assert str(excinfo.value) == "scheduled task T returned failed (T_BUSY)"
    with pytest.raises(RuntimeError) as excinfo:
        _require_successful_task_result("T", outcome, declared_reason_codes=frozenset())
    assert str(excinfo.value) == "scheduled task T returned failed"


# --------------------------------------------------------------------------
# The wiring — the door is only as good as what the registrations hand it.
# --------------------------------------------------------------------------

from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.signals.sources.scheduler import build_cron_registrations


def _registration(regs, task_name):
    (reg,) = [r for r in regs if r.name == f"cron.{task_name}"]
    return reg


@pytest.mark.asyncio
async def test_a_tool_handler_resolves_the_owners_vocabulary_at_failure_time():
    asked = []
    vocabulary = {"signal_dispatch": frozenset()}

    async def lookup(task_name, payload):
        return ToolResult.failed("prose", data={"reason_code": "WORKFLOW_RUN_REJECTED"})

    def reason_codes(task_name):
        asked.append(task_name)
        return vocabulary[task_name]

    reg = _registration(
        build_cron_registrations(tool_lookup=lookup, reason_codes_lookup=reason_codes),
        "signal_dispatch",
    )
    with pytest.raises(RuntimeError) as excinfo:
        await reg.handler({})
    assert str(excinfo.value) == "scheduled tool signal_dispatch failed"
    assert asked == ["signal_dispatch"]

    # Declared later (features load after the scheduler registers) — the
    # next failure sees it, because the lookup runs per failure.
    vocabulary["signal_dispatch"] = frozenset({"WORKFLOW_RUN_REJECTED"})
    with pytest.raises(RuntimeError) as excinfo:
        await reg.handler({})
    assert str(excinfo.value) == "scheduled tool signal_dispatch failed (WORKFLOW_RUN_REJECTED)"


@pytest.mark.asyncio
async def test_a_builtin_handler_is_bounded_by_the_same_lookup():
    async def builtin(payload):
        return json.dumps(
            {"success": False, "error": "boom", "data": {"reason_code": "TRASH_SWEEP_BLOCKED"}}
        )

    async def lookup(task_name, payload):
        raise AssertionError("builtins bypass tool lookup")

    for declared, expected in (
        (frozenset(), "scheduled tool trash_retention failed"),
        (frozenset({"TRASH_SWEEP_BLOCKED"}), "scheduled tool trash_retention failed (TRASH_SWEEP_BLOCKED)"),
    ):
        reg = _registration(
            build_cron_registrations(
                tool_lookup=lookup,
                reason_codes_lookup=lambda task_name, d=declared: d,
                builtin_handlers={"trash_retention": builtin},
            ),
            "trash_retention",
        )
        with pytest.raises(RuntimeError) as excinfo:
            await reg.handler({})
        assert str(excinfo.value) == expected


class _Tool:
    def __init__(self, name):
        self.name = name


class _Feature:
    def __init__(self, tools, declared=None, enabled=True):
        self._tools = [_Tool(n) for n in tools]
        self.enabled = enabled
        if declared is not None:
            self.tool_reason_codes = declared

    def get_tools(self):
        return list(self._tools)


class _Agent:
    def __init__(self, features):
        self.features = features


def test_the_scheduler_feature_resolves_a_declaration_only_from_the_tools_owner():
    owner = _Feature(["signal_dispatch"], {"signal_dispatch": {"A"}})
    bystander = _Feature(["other_tool"], {"signal_dispatch": {"B"}})
    feature = SchedulerFeature(_Agent({"Bystander": bystander, "Owner": owner}))
    # A feature cannot widen the vocabulary of a tool it does not own.
    assert feature._declared_reason_codes("signal_dispatch") == frozenset({"A"})
    assert feature._declared_reason_codes("other_tool") == frozenset()
    assert feature._declared_reason_codes("unknown") == frozenset()


def test_a_disabled_owners_declaration_does_not_apply():
    owner = _Feature(["signal_dispatch"], {"signal_dispatch": {"A"}}, enabled=False)
    feature = SchedulerFeature(_Agent({"Owner": owner}))
    assert feature._declared_reason_codes("signal_dispatch") == frozenset()


def test_the_scheduler_features_own_declaration_covers_its_builtins(monkeypatch):
    feature = SchedulerFeature(_Agent({}))
    monkeypatch.setattr(
        feature, "tool_reason_codes", {"trash_retention": {"TRASH_SWEEP_BLOCKED"}},
    )
    assert feature._declared_reason_codes("trash_retention") == frozenset({"TRASH_SWEEP_BLOCKED"})
    assert feature._declared_reason_codes("backup_snapshot") == frozenset()


def test_a_malformed_declaration_declares_nothing():
    owner = _Feature(["t"], {"t": "NOT_A_SET"})
    feature = SchedulerFeature(_Agent({"Owner": owner}))
    assert feature._declared_reason_codes("t") == frozenset()
    owner = _Feature(["t"], "not a mapping")
    feature = SchedulerFeature(_Agent({"Owner": owner}))
    assert feature._declared_reason_codes("t") == frozenset()


def test_strategic_memory_declares_every_code_signal_dispatch_returns():
    """The declaration and the producer must not drift: every literal
    ``"reason_code": "..."`` in the feature source is in the declared set."""
    import inspect
    import re
    from kestrel_sovereign.features.strategic_memory import feature as sm

    source = pathlib.Path(inspect.getsourcefile(sm)).read_text()
    used = set(re.findall(r'"reason_code":\s*"([^"]+)"', source))
    assert used, "the scan found no producer literals — regex out of date?"
    assert used <= sm.StrategicMemoryFeature.tool_reason_codes["signal_dispatch"], (
        used - sm.StrategicMemoryFeature.tool_reason_codes["signal_dispatch"]
    )
    assert sm.StrategicMemoryFeature.tool_reason_codes["signal_dispatch"] == (
        SIGNAL_DISPATCH_REASON_CODES
    )


@pytest.mark.asyncio
async def test_a_builtin_outcome_is_bounded_by_the_same_lookup():
    async def builtin(payload):
        return ScheduledTaskOutcome(status="failed", result_text="{}", reason_code="SLEEP_FAILED")

    async def lookup(task_name, payload):
        raise AssertionError("builtins bypass tool lookup")

    for declared, expected in (
        (frozenset(), "scheduled task sleep returned failed"),
        (SLEEP_DECLARED, "scheduled task sleep returned failed (SLEEP_FAILED)"),
    ):
        reg = _registration(
            build_cron_registrations(
                tool_lookup=lookup,
                reason_codes_lookup=lambda task_name, d=declared: d,
                builtin_handlers={"sleep": builtin},
            ),
            "sleep",
        )
        with pytest.raises(RuntimeError) as excinfo:
            await reg.handler({})
        assert str(excinfo.value) == expected


def test_the_scheduler_feature_declares_the_sleep_vocabulary_it_produces():
    # Every code _handle_sleep can set: the raised-cycle code and the report's
    # own failure vocabulary. Declared by the owner, resolved by name.
    assert SLEEP_DECLARED == frozenset({"SLEEP_FAILED"}) | SLEEP_FAILURE_REASONS
    feature = SchedulerFeature(_Agent({}))
    assert feature._declared_reason_codes("sleep") == SLEEP_DECLARED
