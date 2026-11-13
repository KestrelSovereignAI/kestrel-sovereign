"""One tool invocation's outcome type, characterized against main (#3977).

``kestrel_sovereign.tools.invocation_outcome`` is the contract a host executor
will consume; no path uses it yet. These tests pin two things:

* the type: exactly five immutable variants, an exhaustive projection checked
  by ``assert_never``, and ``outcome_unknown`` as a fact on every variant;
* its characterization: for what each Phase A path (#3921, #3922, #3924) and
  the #3946 cases produce on main, the classifier's variant, and a projection
  that reproduces what that path hands the model, raising the same object
  where the path raises. Control errors are never stringified, deferred
  results keep their private publication, and ``outcome_unknown`` survives
  every variant.

#3923 (atomic dynamic registration) and #3925 (isolated schema projection)
change what a model is offered, not what an invocation produces, so they have
no outcome to classify.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import pickle
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from uuid import uuid4

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SdkFeature
from kestrel_sdk.hooks.base import HookOutput
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown, observed_outcome_unknown
from kestrel_sdk.tools import DeferredToolResult, ToolCategory, ToolResult

from kestrel_sovereign.a2a.stores.unified.observability_store import infer_tool_result_status
from kestrel_sovereign.agent.parts import part_collector
from kestrel_sovereign.agent.orchestrator_engine import (
    _AuditedDeferredResult,
    _AuditedUnconfirmedToolResult,
)
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    execution_evidence_carriers,
    is_execution_control_error,
)
from kestrel_sovereign.features.base import (
    Feature as CoreFeature,
    _bind_flattened_outcome_unknown,
    _serialize_tool_result,
    report_flattened_outcome_unknown,
    tool,
)
from kestrel_sovereign.hooks.decision_gate import BlockedDecision, evaluate_blocking_decision
from kestrel_sovereign.tools.execution_outcome import (
    TOOL_OUTCOME_UNCONFIRMED,
    TOOL_PRECONDITION_CHANGED,
    _PreconditionFailedToolResult,
    _UnconfirmedToolResult,
)
from kestrel_sovereign.tools.invocation_outcome import (
    Cancelled,
    ControlDeny,
    Deferred,
    Error,
    Success,
    ToolInvocationOutcome,
    classify_tool_invocation,
    project_public,
)
from tests.unit import test_tool_argument_binding_parity as parity
from tests.unit import test_tool_outcome_audit_agreement as audit
from tests.unit.test_deferred_tool_delivery import boundary  # noqa: F401 - the imported fixture

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (Success, ControlDeny, Cancelled, Deferred, Error)


def _deferred(**envelope_fields) -> tuple[DeferredToolResult, list]:
    published: list = []

    def publish():
        published.append("published")
        return [{"type": "private_view", "data": {"reference": "opaque"}}]

    envelope = {**ToolResult.ok("Complete").to_dict(), **envelope_fields}
    return DeferredToolResult(envelope, publish=publish), published


def _denial() -> BlockedDecision:
    decided = evaluate_blocking_decision(HookOutput.deny("host denied"))
    assert decided is not None
    return decided


def _sample(variant) -> ToolInvocationOutcome:
    """One instance of ``variant``, as the classifier builds it."""
    if variant is Success:
        return classify_tool_invocation(returned={"success": True, "result": 1})
    if variant is ControlDeny:
        return classify_tool_invocation(raised=ExecutionAuthorityError("retired"))
    if variant is Cancelled:
        return classify_tool_invocation(raised=asyncio.CancelledError())
    if variant is Deferred:
        return classify_tool_invocation(returned=_deferred()[0])
    return classify_tool_invocation(raised=LookupError("lookup failed"))


def _projection(outcome):
    """``("returned", value)`` or ``("raised", the exact object)``."""
    try:
        return "returned", project_public(outcome)
    except BaseException as raised:  # noqa: BLE001 - the projection IS the raise
        return "raised", raised


# --- the type ----------------------------------------------------------------


def test_the_outcome_is_exactly_the_five_variants():
    assert get_args(ToolInvocationOutcome) == VARIANTS


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.__name__)
def test_every_variant_is_immutable(variant):
    outcome = _sample(variant)
    assert type(outcome) is variant
    field = dataclasses.fields(outcome)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(outcome, field, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(outcome, "outcome_unknown", True)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.__name__)
def test_every_variant_has_a_public_projection(variant):
    outcome = _sample(variant)
    kind, value = _projection(outcome)
    method_kind, method_value = "returned", None
    try:
        method_value = outcome.project_public()
    except BaseException as raised:  # noqa: BLE001
        method_kind, method_value = "raised", raised
    assert (kind, value) == (method_kind, method_value)
    expected = {
        Success: ("returned", {"success": True, "result": 1}),
        Deferred: ("returned", ToolResult.ok("Complete").to_dict()),
    }
    if variant in expected:
        assert (kind, value) == expected[variant]
    elif variant is ControlDeny:
        assert kind == "raised" and value is outcome.error
    elif variant is Cancelled:
        assert kind == "raised" and value is outcome.cause
    else:
        assert (kind, value) == ("returned", {"success": False, "error": "lookup failed"})


def test_the_projection_refuses_anything_outside_the_type():
    with pytest.raises(AssertionError):
        project_public(object())  # type: ignore[arg-type]


def _pyright(target: Path, tmp_path) -> tuple[int, list]:
    config = tmp_path / "pyrightconfig.json"
    config.write_text(json.dumps({
        "extraPaths": [str(ROOT)],
        "typeCheckingMode": "basic",
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
    }))
    completed = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson", "--project", str(config),
         "--pythonpath", sys.executable, str(target)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.stderr == "", completed.stderr
    report = json.loads(completed.stdout)
    assert report["version"] == "1.1.403"
    return completed.returncode, report["generalDiagnostics"]


def test_the_projection_match_is_checked_exhaustive(tmp_path):
    module = ROOT / "kestrel_sovereign" / "tools" / "invocation_outcome.py"
    assert "assert_never(outcome)" in module.read_text()
    returncode, diagnostics = _pyright(module, tmp_path)
    assert (returncode, diagnostics) == (0, [])


def test_a_match_missing_a_variant_fails_the_static_check(tmp_path):
    fixture = ROOT / "tests" / "static" / "invocation_outcome_missing_variant.py"
    returncode, diagnostics = _pyright(fixture, tmp_path)
    assert returncode == 1
    [diagnostic] = diagnostics
    assert diagnostic["rule"] == "reportArgumentType"
    assert diagnostic["message"].startswith('Argument of type "Error" cannot be assigned')
    lines = fixture.read_text().splitlines()
    assert lines[diagnostic["range"]["start"]["line"]].strip() == "assert_never(outcome)"


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.__name__)
def test_outcome_unknown_is_a_fact_on_every_variant(variant):
    outcome = _sample(variant)
    assert outcome.outcome_unknown is False
    flagged = dataclasses.replace(outcome, outcome_unknown=True)
    assert type(flagged) is variant and flagged.outcome_unknown is True
    first = dataclasses.fields(outcome)[0].name
    assert getattr(flagged, first) is getattr(outcome, first)
    # The flag is for the host's retry decision; what the model sees is unchanged.
    before, after = _projection(outcome), _projection(flagged)
    assert before[0] == after[0]
    assert before[1] == after[1] if before[0] == "returned" else after[1] is before[1]
    with pytest.raises(TypeError, match="outcome_unknown must be a bool"):
        dataclasses.replace(outcome, outcome_unknown=1)


@pytest.mark.parametrize("make", [
    lambda: Success(_deferred()[0]),
    lambda: Deferred({"status": "ok", "confirmation": "x"}),
    lambda: ControlDeny(ValueError("ordinary")),
    lambda: ControlDeny(asyncio.CancelledError()),
    lambda: Cancelled(ValueError("ordinary")),
    lambda: Cancelled("cancelled"),
    lambda: Error(ExecutionAuthorityError("retired")),
    lambda: Error(asyncio.CancelledError()),
    lambda: ControlDeny("Permission denied: host denied"),
    lambda: Error("lookup failed"),
], ids=["success-deferred", "deferred-dict", "deny-ordinary", "deny-cancel",
        "cancel-ordinary", "cancel-string", "error-control", "error-cancel",
        "deny-string", "error-string"])
def test_a_variant_refuses_another_variants_payload(make):
    with pytest.raises(TypeError):
        make()


# --- the classifier ----------------------------------------------------------


@pytest.mark.parametrize("supplied", [
    {}, {"returned": 1, "raised": ValueError()}, {"raised": ValueError(), "refused": _denial()},
    {"returned": None, "refused": _denial()},
], ids=["none", "returned+raised", "raised+refused", "returned+refused"])
def test_the_classifier_takes_exactly_one_input(supplied):
    with pytest.raises(TypeError, match="exactly one"):
        classify_tool_invocation(**supplied)


def test_a_returned_none_is_a_success():
    assert classify_tool_invocation(returned=None) == Success(None)


def test_a_tool_cannot_return_a_host_refusal():
    decision = _denial()
    outcome = classify_tool_invocation(returned=decision)
    assert type(outcome) is Success and outcome.result is decision
    with pytest.raises(TypeError):
        classify_tool_invocation(refused={"success": False})  # type: ignore[arg-type]


def _caused_by(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


CONTROL_CARRIERS = {
    "authority": lambda: ExecutionAuthorityError("admission retired"),
    "commit-outcome": lambda: ExecutionCommitOutcomeError("unknown"),
    "explicit-cause": lambda: _caused_by(RuntimeError("wrapper"), ExecutionAuthorityError("retired")),
    "group-member": lambda: ExceptionGroup("joined", [ValueError("x"), ExecutionAuthorityError("retired")]),
    "unknown-caused-by-control": lambda: _caused_by(
        ResourceOutcomeUnknown("uncertain"), ExecutionAuthorityError("retired")),
    "retirement-debt": lambda: _with_retirement_debt(OSError("close failed")),
}


def _with_retirement_debt(error: BaseException) -> BaseException:
    error._kestrel_native_retirement_debts = ((object(), object()),)
    return error


@pytest.mark.parametrize("name", list(CONTROL_CARRIERS))
def test_control_errors_are_a_control_deny_never_an_error(name):
    error = CONTROL_CARRIERS[name]()
    assert is_execution_control_error(error)
    outcome = classify_tool_invocation(raised=error)
    assert type(outcome) is ControlDeny
    assert outcome.error is error and outcome.error_class == type(error).__name__
    assert outcome.outcome_unknown is bool(observed_outcome_unknown(error))
    assert _projection(outcome) == ("raised", error)
    with pytest.raises(TypeError):
        Error(error)  # type: ignore[arg-type]


CANCELLATION_CARRIERS = {
    "cancelled": lambda: asyncio.CancelledError(),
    "keyboard-interrupt": lambda: KeyboardInterrupt(),
    "system-exit": lambda: SystemExit(3),
    "group": lambda: BaseExceptionGroup("joined", [asyncio.CancelledError(), ValueError("x")]),
    "cancelled-carrying-control": lambda: _with_retirement_debt(asyncio.CancelledError()),
}


@pytest.mark.parametrize("name", list(CANCELLATION_CARRIERS))
def test_cancellation_carriers_are_cancelled_and_propagate_unchanged(name):
    cause = CANCELLATION_CARRIERS[name]()
    outcome = classify_tool_invocation(raised=cause)
    assert type(outcome) is Cancelled and outcome.cause is cause
    assert _projection(outcome) == ("raised", cause)


def test_an_ordinary_error_projects_the_envelope_every_wrapper_builds_on():
    error = LookupError("lookup failed")
    outcome = classify_tool_invocation(raised=error)
    assert outcome == Error(error) and outcome.error_class == "LookupError"
    assert project_public(outcome) == {"success": False, "error": "lookup failed"}


@pytest.mark.parametrize("make", [
    lambda: ResourceOutcomeUnknown("uncertain"),
    lambda: _caused_by(RuntimeError("wrapper"), ResourceOutcomeUnknown("uncertain")),
    lambda: ExceptionGroup("joined", [ValueError("x"), ResourceOutcomeUnknown("uncertain")]),
], ids=["itself", "explicit-cause", "group-member"])
def test_an_error_that_requires_reconciliation_is_never_a_result(make):
    error = make()
    outcome = classify_tool_invocation(raised=error)
    assert type(outcome) is Error and outcome.outcome_unknown is True
    assert _projection(outcome) == ("raised", error)


def test_implicit_handling_context_is_not_outcome_evidence():
    try:
        try:
            raise ResourceOutcomeUnknown("uncertain")
        except ResourceOutcomeUnknown:
            raise LookupError("handled elsewhere")
    except LookupError as error:
        assert isinstance(error.__context__, ResourceOutcomeUnknown)
        outcome = classify_tool_invocation(raised=error)
    assert type(outcome) is Error and outcome.outcome_unknown is False
    assert project_public(outcome) == {"success": False, "error": "handled elsewhere"}


async def test_a_task_group_member_counts_as_an_unknown_outcome():
    async def uncertain():
        raise ResourceOutcomeUnknown("uncertain")

    async def failed():
        raise ValueError("other")

    with pytest.raises(ExceptionGroup) as joined:
        async with asyncio.TaskGroup() as group:
            group.create_task(uncertain())
            group.create_task(failed())
    outcome = classify_tool_invocation(raised=joined.value)
    assert type(outcome) is Error and outcome.outcome_unknown is True


def test_a_resource_conflict_is_an_ordinary_error():
    outcome = classify_tool_invocation(raised=ResourceConflict("precondition changed"))
    assert type(outcome) is Error and outcome.outcome_unknown is False


@pytest.mark.parametrize("returned", [
    ToolResult.ok("Saved").to_dict(),
    {**ToolResult.failed("store unavailable").to_dict(), "success": False},
    {"success": False, "error": "store unavailable"},
    {**ToolResult.partial("Saved", "index degraded").to_dict(), "success": True},
    "plain text", 0, [1, 2],
], ids=["ok", "failed", "failed-dict", "partial", "text", "zero", "list"])
def test_the_variant_never_depends_on_the_returned_status(returned):
    outcome = classify_tool_invocation(returned=returned)
    assert outcome == Success(returned)
    assert project_public(outcome) is returned


@pytest.mark.parametrize("marker,unknown", [
    (lambda: _UnconfirmedToolResult({"success": False, "error": TOOL_OUTCOME_UNCONFIRMED}), True),
    (lambda: _PreconditionFailedToolResult({"success": False, "error": TOOL_PRECONDITION_CHANGED}), False),
    (lambda: _AuditedDeferredResult({"status": "ok", "confirmation": "Saved"},
                                    host=object(), tool_name="t", session_id="s"), False),
    (lambda: _AuditedUnconfirmedToolResult({"success": False, "error": TOOL_OUTCOME_UNCONFIRMED},
                                           host=object(), tool_name="t", session_id="s"), True),
], ids=["unconfirmed", "precondition", "audited-receipt", "audited-unconfirmed"])
def test_host_private_facts_never_reach_public_serialization(marker, unknown):
    result = marker()
    outcome = classify_tool_invocation(returned=result)
    assert type(outcome) is Success and outcome.result is result
    assert outcome.outcome_unknown is unknown
    public = project_public(outcome)
    assert type(public) is dict and public == dict(result)
    for transformed in (copy.copy(outcome), copy.deepcopy(outcome),
                        pickle.loads(pickle.dumps(outcome))):
        assert type(transformed) is Success
        assert type(transformed.result) is dict and transformed.result == dict(result)
        assert transformed.outcome_unknown is unknown


def test_a_blocking_decision_is_a_control_deny_with_its_error_class():
    deny = _denial()
    ask = evaluate_blocking_decision(HookOutput.ask("approval-1", "needs a human"))
    stop = evaluate_blocking_decision(HookOutput(continue_execution=False, stop_reason="halt"))
    for decision, error_class in ((deny, "PermissionDenied"), (ask, "ApprovalRequired"),
                                  (stop, "PermissionDenied")):
        assert decision is not None
        outcome = classify_tool_invocation(refused=decision)
        assert type(outcome) is ControlDeny and outcome.error is decision
        assert outcome.error_class == error_class
        public = project_public(outcome)
        assert public == decision.envelope


def test_observed_unknown_outcomes_survive_every_variant():
    """A path that flattened an unknown outcome reports it, whatever it ended in."""
    flattened: list = []
    with _bind_flattened_outcome_unknown(flattened):
        report_flattened_outcome_unknown(
            _caused_by(RuntimeError("wrapper"), ResourceOutcomeUnknown("uncertain")))
    assert len(flattened) == 1
    inputs = {
        Success: {"returned": ToolResult.ok("Saved").to_dict()},
        ControlDeny: {"refused": _denial()},
        Cancelled: {"raised": asyncio.CancelledError()},
        Deferred: {"returned": _deferred()[0]},
        Error: {"raised": LookupError("lookup failed")},
    }
    for variant, produced in inputs.items():
        outcome = classify_tool_invocation(**produced, observed=flattened)
        assert type(outcome) is variant and outcome.outcome_unknown is True
        assert classify_tool_invocation(**produced, observed=[LookupError("ordinary")]).outcome_unknown is False
    raised = classify_tool_invocation(raised=ExecutionAuthorityError("retired"), observed=flattened)
    assert type(raised) is ControlDeny and raised.outcome_unknown is True


def test_a_deferred_result_keeps_its_private_publication():
    result, published = _deferred()
    outcome = classify_tool_invocation(returned=result)
    assert type(outcome) is Deferred and outcome.result is result
    public = project_public(outcome)
    assert type(public) is dict and public == dict(result)
    assert copy.copy(outcome) is outcome
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(outcome)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(outcome)
    with pytest.raises(TypeError):
        _serialize_tool_result(outcome.result)
    # Classification and projection consumed nothing: the factory publishes once.
    assert published == []
    finalized = outcome.result.finalize(outcome.result, allowed=True)
    assert published == ["published"] and not finalized.withheld
    with pytest.raises(RuntimeError):
        outcome.result.finalize(outcome.result, allowed=True)


# --- characterization: #3921 across every dispatch path ----------------------


@pytest_asyncio.fixture
async def probe_path(tmp_path, native_tool_effect_agent):
    made: list = []

    def make(path: str, kind: str):
        made.append(parity.PATHS[path](kind, tmp_path, native_tool_effect_agent))
        return made[-1]

    yield make
    for path in made:
        await path.close()


@pytest.mark.parametrize("kind", sorted(parity.KINDS))
@pytest.mark.parametrize("path", sorted(parity.PATHS))
async def test_a_returned_call_projects_what_the_path_returned(probe_path, path, kind):
    probe = probe_path(path, kind)
    returned = await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")
    outcome = classify_tool_invocation(returned=returned)
    assert type(outcome) is Success and outcome.outcome_unknown is False
    # The named path returns its dispatch receipt; the host keeps that object.
    assert outcome.result is returned
    public = project_public(outcome)
    assert public == returned
    assert type(public) is dict


@pytest.mark.parametrize("kind", sorted(parity.KINDS))
@pytest.mark.parametrize("path", sorted(parity.PATHS))
async def test_a_refused_argument_is_a_returned_failure_not_an_error(probe_path, path, kind):
    probe = probe_path(path, kind)
    returned = await probe.call("probe", {"limit": "abc"}, "!probe abc")
    outcome = classify_tool_invocation(returned=returned)
    assert type(outcome) is Success and outcome.outcome_unknown is False
    assert project_public(outcome) == returned
    assert probe.feature.calls == []


@pytest.mark.parametrize("kind", sorted(parity.KINDS))
@pytest.mark.parametrize("path", sorted(parity.PATHS))
async def test_a_raised_control_error_is_never_stringified(probe_path, path, kind):
    probe = probe_path(path, kind)
    denial = probe.error = ExecutionAuthorityError("original admission retired")
    with pytest.raises(ExecutionAuthorityError) as raised:
        await probe.call("probe_raise", {}, "!probe-raise")
    assert raised.value is denial
    outcome = classify_tool_invocation(raised=raised.value)
    assert type(outcome) is ControlDeny and outcome.error is denial
    assert outcome.error_class == "ExecutionAuthorityError"
    assert _projection(outcome) == ("raised", denial)


@pytest.mark.parametrize("kind", sorted(parity.KINDS))
@pytest.mark.parametrize("path", sorted(parity.PATHS))
async def test_an_ordinary_raise_is_an_error_whose_outcome_is_unknown(probe_path, path, kind):
    probe = probe_path(path, kind)
    original = probe.error = LookupError("lookup failed")
    with pytest.raises(ResourceOutcomeUnknown) as raised:
        await probe.call("probe_raise", {}, "!probe-raise")
    outcome = classify_tool_invocation(raised=raised.value)
    assert type(outcome) is Error and outcome.error is raised.value
    assert outcome.outcome_unknown is True
    assert _projection(outcome) == ("raised", raised.value)
    assert any(carrier is original for carrier in execution_evidence_carriers(raised.value))


# --- characterization: #3922 deferred results --------------------------------


@pytest.mark.parametrize("base", [CoreFeature, SdkFeature], ids=["core", "sdk"])
async def test_a_wrapped_deferred_result_classifies_with_its_factory(base):
    """What each ``@tool`` wrapper returns for a deferred result (#3922)."""
    published: list = []

    class Source(base):
        tool_name = "deferring_feature"
        tool_description = "Deferred wrapper characterization"

        async def initialize(self):
            pass

        @tool(name="deferred_result", description="Defers", category=ToolCategory.UTILITY)
        async def deferred_result(self) -> ToolResult:
            return DeferredToolResult(ToolResult.ok("Complete").to_dict(),
                                      publish=lambda: published.append("once") or [])

    [wrapped_tool] = Source(SimpleNamespace(did="did:test:outcome")).get_tools()
    wrapped = await wrapped_tool.execute()
    outcome = classify_tool_invocation(returned=wrapped)
    assert type(outcome) is Deferred and outcome.result is wrapped
    assert project_public(outcome) == dict(wrapped)
    assert wrapped["tool"] == "deferred_result"
    assert published == []
    assert not wrapped.finalize(wrapped, allowed=True).withheld
    assert published == ["once"]


@pytest.mark.parametrize("path", ["chat", "named", "bound"])
async def test_classifying_mid_completion_leaves_publication_to_the_path(
    boundary,  # noqa: F811 - the imported fixture
    path,
):
    """Classify the deferred result where the path's own completion sees it."""
    peer = boundary
    seen: list = []
    keep = peer.policy.observe_result

    def observe(result):
        keep(result)
        outcome = classify_tool_invocation(returned=result)
        seen.append((outcome, project_public(outcome), result))

    peer.policy.observe_result = observe
    peer.policy.release.set()
    with part_collector():
        returned = await peer.invoke(path)

    [(outcome, public, result)] = seen
    assert type(outcome) is Deferred and outcome.result is result
    assert type(public) is dict and "private_view" not in json.dumps(public)
    # The path published exactly once, after its own completion.
    assert peer.tool.executions == [{}] and peer.tool.publications == ["published-once"]
    assert [part["type"] for part in returned["parts"]] == ["original_part", "private_view"]
    completed = classify_tool_invocation(returned=returned)
    assert type(completed) is Success and project_public(completed) == returned


# --- characterization: #3924 returned envelopes and refusals -----------------


@pytest_asyncio.fixture
async def audit_host(native_tool_effect_agent):
    return await native_tool_effect_agent(did="did:test:outcome-audit")


RETURNED_CASES = [case for case, (produce, _) in audit.OUTCOMES.items() if produce is not audit.DENIED]


@pytest.mark.parametrize("case", RETURNED_CASES)
async def test_a_direct_tools_returned_envelope_keeps_its_status(audit_host, case):
    agent = audit_host
    produce, expected = audit.OUTCOMES[case]
    probe = audit._Tool(produce)
    agent._direct_tools[probe.name] = probe
    agent._tool_to_feature[probe.name] = "FactFeature"

    returned = await audit._dispatch(agent, probe.name)

    outcome = classify_tool_invocation(returned=returned)
    assert type(outcome) is Success and outcome.outcome_unknown is False
    assert project_public(outcome) == returned
    assert infer_tool_result_status(project_public(outcome)) == expected


@pytest.mark.parametrize("path", ["direct", "bound", "named"])
async def test_a_permission_denial_projects_the_envelope_the_path_returns(audit_host, path):
    agent = audit_host
    probe = audit._Tool(lambda: {"never": "run"}, name="record_fact" if path == "direct" else "view.update")
    agent.hooks_manager.register(audit._Deny())
    if path == "direct":
        agent._direct_tools[probe.name] = probe
        agent._tool_to_feature[probe.name] = "FactFeature"
        returned = await audit._dispatch(agent, probe.name)
    else:
        bindings = []
        agent.register_dynamic_tools(
            "mcp:views", [probe], on_registered=lambda _tool, binding: bindings.append(binding))
        if path == "bound":
            returned = await bindings[0].invoke(
                {"count": 3}, session_id="mcp-app:view-1", source="mcp-app", operation_id=uuid4().hex)
        else:
            returned = await agent.execute_named_tool(
                bindings[0].name, {"count": 3}, session_id="sess-outcome", source="outcome-test",
                operation_id=uuid4().hex)
    assert probe.calls == 0

    decision = evaluate_blocking_decision(await audit._Deny().execute(None))
    assert decision is not None
    outcome = classify_tool_invocation(refused=decision)
    assert type(outcome) is ControlDeny and outcome.error_class == "PermissionDenied"
    assert project_public(outcome) == returned


async def test_a_raising_direct_tool_is_an_unknown_error(audit_host):
    agent = audit_host
    probe = audit._Tool(audit.RAISES)
    agent._direct_tools[probe.name] = probe
    agent._tool_to_feature[probe.name] = "FactFeature"
    with pytest.raises(ResourceOutcomeUnknown) as raised:
        await audit._dispatch(agent, probe.name)
    outcome = classify_tool_invocation(raised=raised.value)
    assert type(outcome) is Error and outcome.outcome_unknown is True
    assert _projection(outcome) == ("raised", raised.value)
