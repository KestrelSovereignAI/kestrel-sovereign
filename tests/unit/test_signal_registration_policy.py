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


def _rebuilt_redaction() -> RedactionPolicy:
    """A fresh RedactionPolicy whose summarizer is rebuilt from one ``def``.

    Simulates a feature's ``build_*_registration()`` reconstructing an
    identical policy on re-init: the nested summarizer is a *distinct object*
    every call but shares one compiled code object, so the fingerprint stays
    stable across rebuilds.
    """
    return RedactionPolicy(summarize=lambda p: "<redacted>")


def test_redaction_rebuilt_summarizer_is_equivalent():
    # The signature compares the summarizer's module + compiled code + captured
    # values (NOT its object identity), so a feature that rebuilds an identical
    # registration on re-init — same summarizer ``def``, fresh object — stays
    # equivalent and is not spuriously flagged (#2522 P1).
    a = _action_reg("s", redaction=_rebuilt_redaction())
    b = _action_reg("s", redaction=_rebuilt_redaction())
    assert SourceRegistry.contract_equivalent(a, b)


def test_redaction_distinct_summarizer_body_is_a_mismatch():
    # Two summarizers with DIFFERENT bodies compile to distinct code objects
    # and behave differently, so they are a genuine contract mismatch. The old
    # qualname-only fingerprint equated them (both are ``<lambda>``) — the
    # false-equivalence #2522 P1 targets.
    a = _action_reg("s", redaction=RedactionPolicy(summarize=lambda p: "x"))
    b = _action_reg("s", redaction=RedactionPolicy(summarize=lambda p: "y"))
    assert not SourceRegistry.contract_equivalent(a, b)


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


def _default_bound_handler(reply):
    # A PLAIN function (no closure cell): ``reply`` is baked into ``__defaults__``
    # at def time, and the body reads the *parameter*, so nothing is captured as
    # a free variable. Two handlers from this factory therefore share a qualname,
    # module, AND compiled code object, differing ONLY in their default-bound
    # ``reply`` — the exact shape the old qualname-only fingerprint equated.
    async def handler(payload, reply=reply):
        return reply

    return handler


def test_default_bound_handlers_differ_in_contract_signature():
    # Unit-level guard on the fingerprint itself: distinct default-bound
    # behavior is a mismatch, an identical default-bound rebuild is not (#2522
    # P1 — plain functions were reduced to __qualname__, ignoring __defaults__).
    a = _action_reg("s", handler=_default_bound_handler("A"))
    b = _action_reg("s", handler=_default_bound_handler("B"))
    a2 = _action_reg("s", handler=_default_bound_handler("A"))
    assert not SourceRegistry.contract_equivalent(a, b)
    assert SourceRegistry.contract_equivalent(a, a2)


def test_optional_reregistration_with_different_default_bound_handler_is_reported():
    # End-to-end through register_with_policy (the load-bearing path, not just
    # contract_equivalent): two handlers built from the same factory with
    # DIFFERENT default-bound behavior must be reported as a MISMATCH so OPTIONAL
    # keeps the existing handler instead of silently RETAINING the wrong one — the
    # concrete #2522 P1 defect the audit named for plain functions.
    reg = SourceRegistry()
    first = reg.register_with_policy(
        _action_reg("s", handler=_default_bound_handler("A")),
        RegistrationPolicy.OPTIONAL,
    )
    assert first.state is RegistrationState.REGISTERED
    second = reg.register_with_policy(
        _action_reg("s", handler=_default_bound_handler("B")),
        RegistrationPolicy.OPTIONAL,
    )
    assert second.state is RegistrationState.MISMATCH
    # The existing "A"-bound handler is KEPT (its default is unchanged).
    assert reg.get("s").handler.__defaults__ == ("A",)
    # A genuine same-default rebuild is still a benign no-op.
    third = reg.register_with_policy(
        _action_reg("s", handler=_default_bound_handler("A")),
        RegistrationPolicy.OPTIONAL,
    )
    assert third.state is RegistrationState.ALREADY_EQUIVALENT


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
# The provider-neutral helper remains tolerant for embedders that register
# richer source implementations before core.  Core boot itself registers the
# default batch as MANDATORY and therefore fails atomically on a mismatch.
# ---------------------------------------------------------------------------


def test_workflow_rescue_helper_reports_mismatch_not_silent_skip(caplog):
    reg = SourceRegistry()
    reg.register(_action_reg(FLEET_STALLED_SWEEP, trust=Trust.UNTRUSTED))
    with caplog.at_level(logging.ERROR):
        newly = register_workflow_rescue_sources(reg)
    assert FLEET_STALLED_SWEEP not in newly
    assert set(newly) == set(RESCUE_SOURCE_NAMES) - {FLEET_STALLED_SWEEP}
    assert reg.get(FLEET_STALLED_SWEEP).trust is Trust.UNTRUSTED
    assert "DIFFERENT contract" in caplog.text


def test_workflow_rescue_helper_is_idempotent_when_equivalent():
    reg = SourceRegistry()
    assert set(register_workflow_rescue_sources(reg)) == set(RESCUE_SOURCE_NAMES)
    assert register_workflow_rescue_sources(reg) == []


# ---------------------------------------------------------------------------
# Ownership claims (issue #3053)
# ---------------------------------------------------------------------------


def _owner_test_source(name="own.test"):
    from kestrel_sdk.signals import RedactionPolicy, SignalMode, SourceRegistration, Trust

    async def handle(payload):
        return payload

    return SourceRegistration(
        name=name, schema=dict, default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}), handler=handle,
        trust=Trust.TRUSTED, log_redaction=RedactionPolicy(summarize=lambda p: ""),
    )


def test_an_ownerless_re_registration_does_not_pin_a_feature_owned_source():
    """"No owner supplied" is not "the host owns this".

    The imperative path registers ownerless and claims a moment later, so a
    repeated `initialize()` would staple a permanent host claim onto a
    feature's own source — releasing every real owner would then never remove
    it and the disabled feature's handler stayed registered forever.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    feature = object()
    source = _owner_test_source()

    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL, owner=feature)
    # A second, ownerless idempotent registration — the repeated initialize().
    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL)

    assert registry.owners_of(source.name) == (feature,)
    assert registry.release(source.name, feature) is True
    assert registry.get(source.name) is None


def test_a_failed_owner_scoped_batch_unwinds_its_claims():
    """Atomic means the CLAIMS unwind too, not just the newly-added sources.

    The case that matters is a claim on an INCUMBENT: an equivalent
    registration claims a source the batch did not create, so the
    newly-added rollback cannot reach it. A failed owner left holding it
    would keep that source alive forever.
    """
    import dataclasses

    import pytest as _pytest
    from kestrel_sdk.signals import Trust

    from kestrel_sovereign.signals import (
        RegistrationError,
        RegistrationPolicy,
        SourceRegistry,
    )

    registry = SourceRegistry()
    owner = object()

    # An incumbent the batch will ride (equivalent), and a clash that raises.
    shared = _owner_test_source("batch.shared")
    registry.register(shared)
    clash_incumbent = _owner_test_source("batch.clash")
    registry.register(clash_incumbent)
    clashing = dataclasses.replace(clash_incumbent, trust=Trust.UNTRUSTED)

    with _pytest.raises(RegistrationError):
        registry.register_batch(
            [shared, clashing], RegistrationPolicy.MANDATORY, owner=owner
        )

    # The incumbent survives and the failed owner does NOT hold it.
    assert registry.get("batch.shared") is shared
    assert owner not in registry.owners_of("batch.shared")
    assert registry.get("batch.clash") is clash_incumbent


def test_a_failed_batch_keeps_a_claim_the_owner_already_held():
    """Rollback unwinds what the BATCH acquired, not what predated it.

    A feature that registered a source imperatively and also declares it holds
    a claim before the batch runs. Releasing that on a later failure would
    delete a source the feature is still running on.
    """
    import dataclasses

    import pytest as _pytest
    from kestrel_sdk.signals import Trust

    from kestrel_sovereign.signals import (
        RegistrationError,
        RegistrationPolicy,
        SourceRegistry,
    )

    registry = SourceRegistry()
    owner = object()

    shared = _owner_test_source("prior.shared")
    registry.register_with_policy(shared, RegistrationPolicy.OPTIONAL, owner=owner)
    assert registry.owners_of("prior.shared") == (owner,)   # premise: held BEFORE

    clash_incumbent = _owner_test_source("prior.clash")
    registry.register(clash_incumbent)
    clashing = dataclasses.replace(clash_incumbent, trust=Trust.UNTRUSTED)

    with _pytest.raises(RegistrationError):
        registry.register_batch(
            [shared, clashing], RegistrationPolicy.MANDATORY, owner=owner
        )

    # The pre-existing claim survives, and so does the source it protects.
    assert registry.owners_of("prior.shared") == (owner,)
    assert registry.get("prior.shared") is shared


def test_unregister_does_not_retain_the_owner_object():
    """The registry must not pin a feature instance after its source is gone."""
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    owner = object()
    source = _owner_test_source("leak.test")
    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL, owner=owner)
    assert registry.owners_of("leak.test") == (owner,)

    registry.unregister("leak.test")

    assert registry.owners_of("leak.test") == ()
    assert id(owner) not in registry._claim_owners


def test_core_requiring_a_source_a_feature_created_first_is_a_holder():
    """Phase ordering: a feature can create a source core registers later.

    Heartbeat is the live case — a feature contributes the equivalent
    registration in phase 4, core registers it MANDATORY in phase 6. Recording
    no claim for core meant disabling that feature deleted a source
    `HeartbeatRunner` was still dispatching on.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    feature = object()
    source = _owner_test_source("phase.ordered")

    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL, owner=feature)
    # Core, later, ownerless and MANDATORY: it REQUIRES this source.
    registry.register_with_policy(source, RegistrationPolicy.MANDATORY)

    registry.release_all(feature)

    # Core still holds it, so it survives the feature going away.
    assert registry.get("phase.ordered") is source


def test_an_optional_ownerless_re_registration_still_does_not_pin():
    """The other half of the same rule — OPTIONAL is "nice to have".

    Every imperative feature site registers OPTIONAL and ownerless, then claims
    a moment later. Treating that as a host claim strands the feature's
    handlers forever.
    """
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    feature = object()
    source = _owner_test_source("optional.retry")

    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL, owner=feature)
    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL)

    assert registry.owners_of("optional.retry") == (feature,)
    registry.release_all(feature)
    assert registry.get("optional.retry") is None


def test_a_failed_ownerless_batch_unwinds_its_host_claim():
    """Host claims unwind with a failed batch too.

    An ownerless MANDATORY batch meeting an equivalent feature-owned source
    takes a host claim. Leaving it behind after a failure meant the feature
    could never release its own source again.
    """
    import dataclasses

    import pytest as _pytest
    from kestrel_sdk.signals import Trust

    from kestrel_sovereign.signals import (
        RegistrationError,
        RegistrationPolicy,
        SourceRegistry,
    )

    registry = SourceRegistry()
    feature = object()
    shared = _owner_test_source("hostclaim.shared")
    registry.register_with_policy(shared, RegistrationPolicy.OPTIONAL, owner=feature)

    clash_incumbent = _owner_test_source("hostclaim.clash")
    registry.register(clash_incumbent)
    clashing = dataclasses.replace(clash_incumbent, trust=Trust.UNTRUSTED)

    with _pytest.raises(RegistrationError):
        registry.register_batch(
            [shared, clashing], RegistrationPolicy.MANDATORY,
        )

    # The host claim the failed batch took is gone, so the feature is once
    # again the last holder and can release its own source.
    registry.release_all(feature)
    assert registry.get("hostclaim.shared") is None


def test_activation_rollback_keeps_a_claim_the_feature_already_held():
    """A pre-existing imperative claim survives a failed activation."""
    from kestrel_sovereign.signals import RegistrationPolicy, SourceRegistry

    registry = SourceRegistry()
    feature = object()
    source = _owner_test_source("preexisting.claim")
    registry.register_with_policy(source, RegistrationPolicy.OPTIONAL, owner=feature)

    # An activation that acquires nothing new, then rolls back.
    with registry.claims_acquired(feature) as acquired:
        registry.register_batch(
            [source], RegistrationPolicy.MANDATORY, owner=feature
        )
    assert acquired == []          # nothing NEW was taken
    registry.release_acquired(acquired, feature)

    # The claim that predated the activation is intact.
    assert registry.owners_of("preexisting.claim") == (feature,)
    assert registry.get("preexisting.claim") is source
