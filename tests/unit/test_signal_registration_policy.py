"""Explicit signal-source registration policy tests (#2522 discussion).

The signal-duplication audit found that feature/boot paths variously failed
fast, prechecked by name and accepted any existing contract, caught/logged and
continued, or left partially-registered source sets — so a same-name source
with a *different* trust/mode/redaction/handler/ownership was silently treated
as equivalent. ``SourceRegistry`` now makes the intent an explicit
:class:`RegistrationPolicy`:

* ``MANDATORY`` / ``IDEMPOTENT`` — a non-equivalent clash is a hard error; an
  equivalent re-registration is a no-op success; a batch is atomic.
* ``OPTIONAL`` — never raises; a mismatch or invalid source is *reported* (a
  structured :class:`RegistrationOutcome`) and the existing registration kept.

These tests pin the mismatch detection and the partial-registration rollback
the issue asks for.
"""

import logging
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    ResourceLock,
    SignalMode,
    SourceRegistration,
    Trust,
)
from kestrel_sovereign.signals import (
    RegistrationError,
    RegistrationOutcome,
    RegistrationPolicy,
    RegistrationState,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.fleet_coding_pipeline import (
    FLEET_CODING_APPROVAL,
    register_fleet_coding_pipeline_sources,
)
from kestrel_sovereign.signals.sources.talon_pipeline import (
    SOURCE_NAME as TALON_PIPELINE_SOURCE,
    register_talon_pipeline_source,
)
from kestrel_sovereign.signals.sources.workflow_rescue import (
    FLEET_STALLED_SWEEP,
    SOURCE_NAMES as RESCUE_SOURCE_NAMES,
    register_workflow_rescue_sources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redaction(summarize=lambda p: "<redacted>") -> RedactionPolicy:
    return RedactionPolicy(summarize=summarize)


async def _handler(payload):
    return None


async def _other_handler(payload):
    return None


def _action_reg(
    name: str = "src",
    *,
    trust: Trust = Trust.TRUSTED,
    handler=_handler,
    resources=frozenset(),
    redaction=None,
) -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=handler,
        trust=trust,
        resources=resources,
        log_redaction=redaction or _redaction(),
    )


# ---------------------------------------------------------------------------
# register_with_policy — the three outcomes
# ---------------------------------------------------------------------------


def test_new_source_is_registered():
    reg = SourceRegistry()
    outcome = reg.register_with_policy(_action_reg("a"), RegistrationPolicy.MANDATORY)
    assert outcome.state is RegistrationState.REGISTERED
    assert outcome.ok
    assert "a" in reg


def test_equivalent_reregistration_is_a_noop_success():
    reg = SourceRegistry()
    reg.register_with_policy(_action_reg("a"), RegistrationPolicy.MANDATORY)
    outcome = reg.register_with_policy(_action_reg("a"), RegistrationPolicy.MANDATORY)
    assert outcome.state is RegistrationState.ALREADY_EQUIVALENT
    assert outcome.ok
    assert len(reg) == 1


def test_mandatory_mismatch_raises():
    reg = SourceRegistry()
    reg.register_with_policy(
        _action_reg("a", trust=Trust.TRUSTED), RegistrationPolicy.MANDATORY
    )
    # Same name, DIFFERENT trust — must not be silently equated.
    with pytest.raises(RegistrationError, match="DIFFERENT contract"):
        reg.register_with_policy(
            _action_reg("a", trust=Trust.UNTRUSTED, handler=_handler),
            RegistrationPolicy.MANDATORY,
        )
    # The original registration is untouched.
    assert reg.get("a").trust is Trust.TRUSTED


def test_idempotent_mismatch_raises_but_equivalent_is_ok():
    reg = SourceRegistry()
    reg.register_with_policy(_action_reg("a"), RegistrationPolicy.IDEMPOTENT)
    # Equivalent re-run: fine.
    assert reg.register_with_policy(
        _action_reg("a"), RegistrationPolicy.IDEMPOTENT
    ).ok
    # Different handler identity: mismatch.
    with pytest.raises(RegistrationError, match="DIFFERENT contract"):
        reg.register_with_policy(
            _action_reg("a", handler=_other_handler),
            RegistrationPolicy.IDEMPOTENT,
        )


def test_optional_mismatch_is_reported_not_raised():
    reg = SourceRegistry()
    reg.register_with_policy(
        _action_reg("a", trust=Trust.TRUSTED), RegistrationPolicy.OPTIONAL
    )
    outcome = reg.register_with_policy(
        _action_reg("a", handler=_other_handler),
        RegistrationPolicy.OPTIONAL,
    )
    assert outcome.state is RegistrationState.MISMATCH
    assert not outcome.ok
    assert outcome.detail  # a human-readable reason is reported
    # The existing registration is KEPT; the incoming one is dropped.
    assert reg.get("a").handler is _handler


def test_optional_invalid_source_is_reported_not_raised():
    reg = SourceRegistry()
    # Empty name fails validation.
    bad = _action_reg("")
    outcome = reg.register_with_policy(bad, RegistrationPolicy.OPTIONAL)
    assert outcome.state is RegistrationState.INVALID
    assert not outcome.ok
    assert len(reg) == 0


def test_mandatory_invalid_source_raises():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError):
        reg.register_with_policy(_action_reg(""), RegistrationPolicy.MANDATORY)


# ---------------------------------------------------------------------------
# contract_signature / contract_equivalent — the load-bearing axes
# ---------------------------------------------------------------------------


def test_equivalent_when_all_contract_axes_match():
    a = _action_reg("s")
    b = _action_reg("s")
    assert SourceRegistry.contract_equivalent(a, b)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda: _action_reg("s", trust=Trust.UNTRUSTED), id="trust"),
        pytest.param(lambda: _action_reg("s", handler=_other_handler), id="handler"),
        pytest.param(
            lambda: _action_reg("s", resources=frozenset({ResourceLock.WALLET})),
            id="resources",
        ),
    ],
)
def test_mismatch_detected_on_each_axis(mutate):
    base = _action_reg("s", trust=Trust.TRUSTED)
    assert not SourceRegistry.contract_equivalent(base, mutate())


def test_redaction_same_summarizer_is_equivalent():
    # The signature compares the redaction summarizer's *qualified name* and its
    # flags (not the closure's object identity), so two summarizers declared at
    # the same source location — as a feature rebuilding an identical
    # registration on re-init produces — stay equivalent and are not spuriously
    # flagged. (Both lambdas below share one qualname.)
    a = _action_reg("s", redaction=_redaction(lambda p: "x"))
    b = _action_reg("s", redaction=_redaction(lambda p: "y"))
    assert SourceRegistry.contract_equivalent(a, b)


def test_redaction_flag_change_is_a_mismatch():
    # A changed RedactionPolicy *flag* is behavior-affecting and must be caught
    # as a mismatch, not silently equated (#2522 P1). Comparing only the policy
    # class (the old signature) missed this.
    a = _action_reg("s", redaction=RedactionPolicy(summarize=lambda p: "x"))
    b = _action_reg(
        "s",
        redaction=RedactionPolicy(summarize=lambda p: "x", store_raw_trusted=True),
    )
    assert not SourceRegistry.contract_equivalent(a, b)


def _capturing_handler(owner):
    async def handler(payload):
        return owner

    return handler


def test_handler_bound_to_different_owner_is_a_mismatch():
    # Two handlers built from the same factory but capturing DIFFERENT owners
    # share a qualname; the old signature (qualname-only) equated them, so an
    # OPTIONAL re-registration kept the stale owner's handler. The fingerprint
    # now folds in the captured owner's identity (#2522 P1).
    owner_a, owner_b = object(), object()
    a = _action_reg("s", handler=_capturing_handler(owner_a))
    b = _action_reg("s", handler=_capturing_handler(owner_b))
    assert not SourceRegistry.contract_equivalent(a, b)
    # A genuine same-owner rebuild stays equivalent (idempotent re-init).
    a2 = _action_reg("s", handler=_capturing_handler(owner_a))
    assert SourceRegistry.contract_equivalent(a, a2)


def test_optional_reregistration_with_new_owner_is_reported_not_accepted():
    # End-to-end: under OPTIONAL a re-registration whose handler captures a NEW
    # owner is reported as a MISMATCH (existing kept) rather than silently
    # accepted as ALREADY_EQUIVALENT — the concrete defect the audit named.
    reg = SourceRegistry()
    owner_a, owner_b = object(), object()
    first = reg.register_with_policy(
        _action_reg("s", handler=_capturing_handler(owner_a)),
        RegistrationPolicy.OPTIONAL,
    )
    assert first.state is RegistrationState.REGISTERED
    second = reg.register_with_policy(
        _action_reg("s", handler=_capturing_handler(owner_b)),
        RegistrationPolicy.OPTIONAL,
    )
    assert second.state is RegistrationState.MISMATCH


# ---------------------------------------------------------------------------
# register_batch — atomic partial-registration rollback
# ---------------------------------------------------------------------------


def test_batch_registers_all_when_clean():
    reg = SourceRegistry()
    outcomes = reg.register_batch(
        [_action_reg("a"), _action_reg("b"), _action_reg("c")],
        RegistrationPolicy.MANDATORY,
    )
    assert [o.state for o in outcomes] == [RegistrationState.REGISTERED] * 3
    assert len(reg) == 3


def test_mandatory_batch_is_atomic_on_midway_mismatch():
    reg = SourceRegistry()
    # Pre-seed 'b' with a DIFFERENT contract so the batch's 'b' mismatches.
    reg.register(_action_reg("b", handler=_other_handler))
    assert len(reg) == 1

    batch = [_action_reg("a"), _action_reg("b"), _action_reg("c")]
    with pytest.raises(RegistrationError, match="DIFFERENT contract"):
        reg.register_batch(batch, RegistrationPolicy.MANDATORY)

    # Atomic rollback: 'a' (added earlier in this batch) is removed, and 'c'
    # (after the failure) never got added. Only the pre-existing 'b' remains,
    # untouched. No partial source set survives.
    assert "a" not in reg
    assert "c" not in reg
    assert len(reg) == 1
    assert reg.get("b").handler is _other_handler


def test_optional_batch_reports_each_independently():
    reg = SourceRegistry()
    reg.register(_action_reg("b", handler=_other_handler))
    outcomes = reg.register_batch(
        [_action_reg("a"), _action_reg("b"), _action_reg("")],
        RegistrationPolicy.OPTIONAL,
    )
    states = {o.name or "<empty>": o.state for o in outcomes}
    assert states["a"] is RegistrationState.REGISTERED
    assert states["b"] is RegistrationState.MISMATCH
    assert states["<empty>"] is RegistrationState.INVALID
    # 'a' survives; the pre-existing 'b' is kept; the empty one never lands.
    assert "a" in reg
    assert len(reg) == 2


# ---------------------------------------------------------------------------
# unregister — the deliberate inverse used by boot rollback
# ---------------------------------------------------------------------------


def test_unregister_removes_and_reports_presence():
    reg = SourceRegistry()
    reg.register(_action_reg("a"))
    assert reg.unregister("a") is True
    assert "a" not in reg
    # Idempotent: unregistering a missing name is a benign False.
    assert reg.unregister("a") is False


def test_registration_outcome_ok_property():
    assert RegistrationOutcome("x", RegistrationState.REGISTERED).ok
    assert RegistrationOutcome("x", RegistrationState.ALREADY_EQUIVALENT).ok
    assert not RegistrationOutcome("x", RegistrationState.MISMATCH).ok
    assert not RegistrationOutcome("x", RegistrationState.INVALID).ok


# ---------------------------------------------------------------------------
# The feature-family helpers now register under the explicit OPTIONAL policy
# (#2522). These pin the defect fix at the call sites the issue named: a
# same-name source with a DIFFERENT contract is *reported*, not silently
# skipped-and-equated as the old precheck-by-name loops did — while an
# equivalent re-registration stays a benign no-op and nothing ever raises.
# ---------------------------------------------------------------------------


def test_workflow_rescue_helper_reports_mismatch_not_silent_skip(caplog):
    reg = SourceRegistry()
    # A host pre-registered fleet_stalled_sweep with a DIFFERENT contract.
    reg.register(_action_reg(FLEET_STALLED_SWEEP, trust=Trust.UNTRUSTED))
    with caplog.at_level(logging.ERROR):
        newly = register_workflow_rescue_sources(reg)
    # The mismatch is NOT silently equated: excluded from the newly-registered
    # set, the host's version KEPT, and the clash reported. The other sources
    # still register (one bad source does not abort the rest).
    assert FLEET_STALLED_SWEEP not in newly
    assert set(newly) == set(RESCUE_SOURCE_NAMES) - {FLEET_STALLED_SWEEP}
    assert reg.get(FLEET_STALLED_SWEEP).trust is Trust.UNTRUSTED
    assert "DIFFERENT contract" in caplog.text


def test_fleet_coding_helper_reports_mismatch_not_silent_skip(caplog):
    reg = SourceRegistry()
    reg.register(_action_reg(FLEET_CODING_APPROVAL, trust=Trust.UNTRUSTED))
    with caplog.at_level(logging.ERROR):
        newly = register_fleet_coding_pipeline_sources(reg)
    assert FLEET_CODING_APPROVAL not in newly
    assert reg.get(FLEET_CODING_APPROVAL).trust is Trust.UNTRUSTED
    assert "DIFFERENT contract" in caplog.text


def test_talon_pipeline_helper_reports_mismatch_not_silent_skip(caplog):
    reg = SourceRegistry()
    reg.register(_action_reg(TALON_PIPELINE_SOURCE, trust=Trust.UNTRUSTED))
    with caplog.at_level(logging.ERROR):
        result = register_talon_pipeline_source(reg, SimpleNamespace())
    # A non-equivalent existing source: not newly registered, host version kept,
    # the clash reported — never raised.
    assert result is False
    assert reg.get(TALON_PIPELINE_SOURCE).trust is Trust.UNTRUSTED
    assert "DIFFERENT contract" in caplog.text


def test_feature_helpers_are_idempotent_when_equivalent():
    reg = SourceRegistry()
    # First registration lands the whole rescue set.
    assert set(register_workflow_rescue_sources(reg)) == set(RESCUE_SOURCE_NAMES)
    # A second, contract-equivalent call registers nothing new and never raises.
    assert register_workflow_rescue_sources(reg) == []
