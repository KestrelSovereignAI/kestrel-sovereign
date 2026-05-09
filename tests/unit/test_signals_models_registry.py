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


# ---------------------------------------------------------------------------
# Constitutional injection — kestrel-sovereign#1137 chunk 1D
# ---------------------------------------------------------------------------


def _valid_cognition_reg(
    name: str = "cog",
    *,
    prompt_template_format: str = "claude_code",
    require_constitution_echo: bool = False,
    constitution_injection: str = "none",
    system_prompt_budget_bytes=None,
    sanitizer=None,
    trust: Trust = Trust.TRUSTED,
) -> SourceRegistration:
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=Path("/tmp/fake.md"),
        trust=trust,
        sanitizer=sanitizer,
        log_redaction=_redaction(),
        prompt_template_format=prompt_template_format,
        require_constitution_echo=require_constitution_echo,
        constitution_injection=constitution_injection,
        system_prompt_budget_bytes=system_prompt_budget_bytes,
    )


def test_register_rejects_codex_format_without_echo():
    """Reviewer formats may not opt out of echo verification —
    the entire reason `codex` exists as a format is to verify."""
    reg = SourceRegistry()
    with pytest.raises(
        RegistrationError, match="prompt_template_format='codex'.*require_constitution_echo=True"
    ):
        reg.register(
            _valid_cognition_reg(
                "codex_no_echo",
                prompt_template_format="codex",
                require_constitution_echo=False,
            )
        )


def test_register_rejects_local_format_without_echo():
    reg = SourceRegistry()
    with pytest.raises(
        RegistrationError, match="prompt_template_format='local'.*require_constitution_echo=True"
    ):
        reg.register(
            _valid_cognition_reg(
                "local_no_echo",
                prompt_template_format="local",
                require_constitution_echo=False,
            )
        )


def test_register_accepts_codex_format_with_echo():
    """Reviewer formats with echo=True is the one valid configuration."""
    reg = SourceRegistry()
    reg.register(
        _valid_cognition_reg(
            "codex_review",
            prompt_template_format="codex",
            require_constitution_echo=True,
            constitution_injection="full",
        )
    )
    assert reg.get("codex_review").prompt_template_format == "codex"


def test_register_accepts_local_format_with_echo():
    reg = SourceRegistry()
    reg.register(
        _valid_cognition_reg(
            "local_review",
            prompt_template_format="local",
            require_constitution_echo=True,
            constitution_injection="full",
        )
    )
    assert reg.get("local_review").prompt_template_format == "local"


def test_register_accepts_claude_code_default():
    """Legacy in-agent path: claude_code + echo=False is the default
    and must continue to register without warnings or errors."""
    reg = SourceRegistry()
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # surface any spurious warning
        reg.register(_valid_cognition_reg("legacy"))
    assert reg.get("legacy").require_constitution_echo is False


def test_register_accepts_bare_format_without_echo():
    """`bare` is caller-responsibility — the echo flag is unconstrained."""
    reg = SourceRegistry()
    reg.register(
        _valid_cognition_reg(
            "bare_caller",
            prompt_template_format="bare",
            require_constitution_echo=False,
        )
    )
    reg.register(
        _valid_cognition_reg(
            "bare_caller_echo",
            prompt_template_format="bare",
            require_constitution_echo=True,
        )
    )


def test_register_rejects_zero_budget():
    reg = SourceRegistry()
    with pytest.raises(
        RegistrationError, match="system_prompt_budget_bytes.*> 0"
    ):
        reg.register(
            _valid_cognition_reg(
                "zero_budget",
                system_prompt_budget_bytes=0,
            )
        )


def test_register_rejects_negative_budget():
    reg = SourceRegistry()
    with pytest.raises(
        RegistrationError, match="system_prompt_budget_bytes.*> 0"
    ):
        reg.register(
            _valid_cognition_reg(
                "neg_budget",
                system_prompt_budget_bytes=-1,
            )
        )


def test_register_accepts_positive_budget():
    reg = SourceRegistry()
    reg.register(
        _valid_cognition_reg(
            "with_budget",
            system_prompt_budget_bytes=8192,
        )
    )
    assert reg.get("with_budget").system_prompt_budget_bytes == 8192


def test_register_accepts_none_budget():
    """`None` is the operator-default sentinel — must remain accepted."""
    reg = SourceRegistry()
    reg.register(_valid_cognition_reg("default_budget"))
    assert reg.get("default_budget").system_prompt_budget_bytes is None


def test_claude_code_echo_warns_when_no_rationale_in_docstring():
    """claude_code + echo=True is unusual; if the handler's module
    docstring lacks rationale the operator gets a registration-time
    warning. The registration still succeeds."""
    import warnings as _w

    reg = SourceRegistry()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        # `_no_handler` lives in this test module; this module's
        # docstring is the one at the top of this file and does NOT
        # mention require_constitution_echo / phantom tool.
        reg.register(
            SourceRegistration(
                name="claude_echo_undocumented",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_no_handler,
                log_redaction=_redaction(),
                prompt_template_format="claude_code",
                require_constitution_echo=True,
            )
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "require_constitution_echo=True" in str(w.message)
        and "claude_code" in str(w.message)
        for w in user_warnings
    ), f"Expected UserWarning, got: {[str(w.message) for w in caught]}"
    assert "claude_echo_undocumented" in reg


def test_claude_code_echo_no_warning_when_docstring_documents_rationale():
    """If the handler's module docstring documents the choice (mentions
    one of the rationale phrases) the warning is suppressed."""
    import types
    import warnings as _w

    documented_module = types.ModuleType("kestrel_sovereign_test_documented_handler")
    documented_module.__doc__ = (
        "Test handler module that opts INTO require_constitution_echo "
        "because this source dispatches to a high-trust workflow."
    )

    async def documented_handler(payload):
        return None

    documented_handler.__module__ = documented_module.__name__
    import sys

    sys.modules[documented_module.__name__] = documented_module
    try:
        reg = SourceRegistry()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            reg.register(
                SourceRegistration(
                    name="claude_echo_documented",
                    schema=dict,
                    default_mode=SignalMode.ACTION,
                    allowed_modes=frozenset({SignalMode.ACTION}),
                    handler=documented_handler,
                    log_redaction=_redaction(),
                    prompt_template_format="claude_code",
                    require_constitution_echo=True,
                )
            )
        user_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "require_constitution_echo" in str(w.message)
        ]
        assert user_warnings == [], (
            f"Expected no echo-rationale warning, got: "
            f"{[str(w.message) for w in user_warnings]}"
        )
    finally:
        sys.modules.pop(documented_module.__name__, None)


def test_claude_code_echo_no_warning_when_callable_is_unintrospectable():
    """Lambdas / dynamically-generated callables can't be placed in a
    module reliably; the validator should not fire spurious warnings
    in that case (the author has nothing to act on)."""
    import warnings as _w

    handler = eval("lambda payload: None")  # lambda with no source module
    handler.__module__ = "<dynamic>"

    reg = SourceRegistry()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        reg.register(
            SourceRegistration(
                name="claude_echo_dynamic",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=handler,
                log_redaction=_redaction(),
                prompt_template_format="claude_code",
                require_constitution_echo=True,
            )
        )
    echo_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "require_constitution_echo" in str(w.message)
    ]
    assert echo_warnings == [], (
        f"Did not expect a warning for unintrospectable callable: "
        f"{[str(w.message) for w in echo_warnings]}"
    )


def test_codex_format_does_not_warn_when_documented():
    """`codex` with echo=True is the required configuration; no warning
    fires regardless of docstring contents."""
    import warnings as _w

    reg = SourceRegistry()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        reg.register(
            _valid_cognition_reg(
                "codex_review_2",
                prompt_template_format="codex",
                require_constitution_echo=True,
            )
        )
    echo_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "require_constitution_echo" in str(w.message)
    ]
    assert echo_warnings == []


def test_register_rejects_unknown_prompt_template_format():
    """Codex round-1 P2: SDK `Literal` annotations are not runtime-
    enforced. A typo like 'codxe' must fail at registration, not slip
    through to the dispatcher as an unknown-format error after the
    fact."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="prompt_template_format='codxe'"):
        reg.register(
            _valid_cognition_reg(
                "typo_format",
                prompt_template_format="codxe",
                require_constitution_echo=True,
            )
        )


def test_register_rejects_unknown_constitution_injection_value():
    """Same concern for `constitution_injection` — 'ful' must fail
    at registration."""
    reg = SourceRegistry()
    with pytest.raises(
        RegistrationError, match="constitution_injection='ful'"
    ):
        reg.register(
            _valid_cognition_reg(
                "typo_injection",
                constitution_injection="ful",
            )
        )


def test_typo_format_does_not_bypass_echo_requirement():
    """Specifically pin the round-1 attack vector: a typo in a value
    that LOOKS like a reviewer format must NOT slip past the echo
    requirement just because the literal-string match in
    `_ECHO_REQUIRED_FORMATS` fails. The allowlist check fires first
    and raises before the echo rule even runs."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="not a recognized format"):
        reg.register(
            _valid_cognition_reg(
                "near_codex",
                prompt_template_format="cdex",
                require_constitution_echo=False,
            )
        )


def test_constitution_injection_validation_runs_after_other_invariants():
    """The new validator must not mask earlier errors — a registration
    that fails an existing invariant (e.g. missing handler) must still
    surface the existing error, not the new one."""
    reg = SourceRegistry()
    with pytest.raises(RegistrationError, match="ACTION but provides no handler"):
        reg.register(
            SourceRegistration(
                name="masked",
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=None,
                log_redaction=_redaction(),
                prompt_template_format="codex",
                require_constitution_echo=False,  # would also fail 1D
            )
        )
