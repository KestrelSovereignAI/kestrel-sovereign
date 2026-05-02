"""Unit tests for signal models (SDK contract) and source registry validation.

Verifies the v1 invariants from SIGNAL_DISPATCHER.md:
- CausationFrame is immutable
- Mode-specific contracts are required at registration
- COGNITION sources MUST NOT declare CONVERSATION
- UNTRUSTED + non-ACTION mode requires sanitizer
- log_redaction is required (no defaults)
"""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from kestrel_sdk.signals import (
    AttentionPolicy,
    CausationFrame,
    RateLimit,
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
    Urgency,
    Visibility,
)
from kestrel_sovereign.signals import RegistrationError, SourceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redaction(summarize=lambda p: "<redacted>") -> RedactionPolicy:
    return RedactionPolicy(summarize=summarize)


async def _no_handler(payload):
    return None


async def _no_artifact(signal):
    return None


# ---------------------------------------------------------------------------
# CausationFrame
# ---------------------------------------------------------------------------


def test_causation_frame_is_frozen():
    frame = CausationFrame(
        agent_id="agent-1",
        source="test",
        signal_id="sig-1",
        turn_id=None,
        depth=1,
        emitted_at=datetime.now(timezone.utc),
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        frame.depth = 2  # type: ignore


# ---------------------------------------------------------------------------
# Signal envelope defaults
# ---------------------------------------------------------------------------


def test_signal_defaults_are_safe():
    """Sane defaults: visibility=INTERNAL, urgency=NORMAL, empty chain.
    Visibility default of INTERNAL forces sources to opt into UI rendering
    (Concern #4 / open question 6)."""
    sig = Signal(
        source="test",
        kind="tick",
        mode=SignalMode.ACTION,
        payload={},
        target_agent="agent-1",
    )
    assert sig.visibility == Visibility.INTERNAL
    assert sig.urgency == Urgency.NORMAL
    assert sig.session_id is None
    assert sig.causation_chain == []
    assert sig.id.startswith("sig_")
    assert sig.arrived_at.tzinfo is not None


def test_urgency_at_or_above():
    assert Urgency.HIGH.at_or_above(Urgency.NORMAL)
    assert Urgency.HIGH.at_or_above(Urgency.HIGH)
    assert not Urgency.LOW.at_or_above(Urgency.NORMAL)
    assert Urgency.NORMAL.at_or_above(Urgency.LOW)


# ---------------------------------------------------------------------------
# SourceRegistration validation
# ---------------------------------------------------------------------------


def _valid_action_reg(name: str = "test") -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=_no_handler,
        log_redaction=_redaction(),
    )


def test_register_valid_action_source():
    reg = SourceRegistry()
    reg.register(_valid_action_reg())
    assert "test" in reg
    assert len(reg) == 1


def test_register_rejects_duplicate_name():
    reg = SourceRegistry()
    reg.register(_valid_action_reg())
    with pytest.raises(RegistrationError, match="already registered"):
        reg.register(_valid_action_reg())


def test_register_rejects_empty_name():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="non-empty name"):
        reg.register(
            SourceRegistration(
                name="",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_default_mode_not_in_allowed():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="default_mode"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.COGNITION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_action_without_handler():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="ACTION but provides no handler"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=None,
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_artifact_without_handler():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="ARTIFACT but provides no artifact_handler"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.ARTIFACT,
                allowed_modes=frozenset({SignalMode.ARTIFACT}),
                artifact_handler=None,
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_cognition_without_template():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="COGNITION but provides no prompt_template"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.COGNITION,
                allowed_modes=frozenset({SignalMode.COGNITION}),
                prompt_template=None,
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_untrusted_cognition_without_sanitizer():
    """SIGNAL_DISPATCHER.md §"The Source Registry": UNTRUSTED + non-ACTION
    modes require sanitizer. Stripe webhook is the canonical case."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="UNTRUSTED with non-ACTION modes"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.COGNITION,
                allowed_modes=frozenset({SignalMode.COGNITION}),
                prompt_template=Path("/tmp/fake.md"),
                trust=Trust.UNTRUSTED,
                sanitizer=None,
                log_redaction=_redaction(),
            )
        )


def test_register_accepts_untrusted_action_without_sanitizer():
    """ACTION mode does not need a sanitizer because payload doesn't enter
    a prompt. Sanitizer requirement only applies to non-ACTION modes."""
    reg = SourceRegistry()
    reg.register(
        SourceRegistration(
            name="ok",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=_no_handler,
            trust=Trust.UNTRUSTED,
            sanitizer=None,
            log_redaction=_redaction(),
        )
    )


def test_register_rejects_cognition_declaring_conversation():
    """The load-bearing v3.1 invariant: CONVERSATION is owned solely by the
    turn lifecycle. COGNITION sources MUST NOT pre-acquire it."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="CONVERSATION"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.COGNITION,
                allowed_modes=frozenset({SignalMode.COGNITION}),
                prompt_template=Path("/tmp/fake.md"),
                resources=frozenset({ResourceLock.CONVERSATION}),
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_action_declaring_conversation():
    """ACTION sources also cannot declare CONVERSATION — there is no scenario
    where a non-cognition handler should hold the turn lock."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="CONVERSATION"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                resources=frozenset({ResourceLock.CONVERSATION}),
                log_redaction=_redaction(),
            )
        )


def test_register_rejects_missing_redaction_policy():
    """No defaults for redaction — too important to default."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="log_redaction"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                log_redaction=None,
            )
        )


def test_register_rejects_negative_retention():
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="retention_days"):
        reg.register(
            SourceRegistration(
                name="bad",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                log_redaction=_redaction(),
                retention_days=-1,
            )
        )


def test_register_accepts_multi_mode_source():
    """Cron source with multiple allowed modes (e.g. some tasks ACTION, some
    ARTIFACT) registers as a single source with both handlers."""
    reg = SourceRegistry()
    reg.register(
        SourceRegistration(
            name="cron",
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION, SignalMode.ARTIFACT}),
            handler=_no_handler,
            artifact_handler=_no_artifact,
            log_redaction=_redaction(),
        )
    )


def test_registry_iter_and_require():
    reg = SourceRegistry()
    reg.register(_valid_action_reg("a"))
    reg.register(_valid_action_reg("b"))
    names = sorted(r.name for r in reg)
    assert names == ["a", "b"]
    assert reg.require("a").name == "a"
    with pytest.raises(RegistrationError, match="Unknown source"):
        reg.require("missing")
