"""Unit tests for the constitutional echo-canary primitive.

Covers derivation determinism, format-specific verification, and
instruction-text construction. The dispatcher integration (chunk 1G)
glues these together; here we lock down the contract each piece
exposes.

kestrel-sovereign#1137 chunk 1F.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.signals.constitution_canary import (
    CANARY_HEX_LENGTH,
    CODEX_CANARY_FIELD,
    LOCAL_CANARY_FIELD,
    PHANTOM_RECEIPT_ARG_NAME,
    PHANTOM_RECEIPT_TOOL_NAME,
    CanaryStatus,
    build_canary_instruction,
    derive_canary,
    is_well_formed_canary,
    verify_in_json_response,
    verify_in_structured_response,
    verify_in_tool_calls,
)


# ---------------------------------------------------------------------------
# derive_canary
# ---------------------------------------------------------------------------


def test_derive_canary_is_deterministic():
    """Same inputs → same output. Required for verification across
    process boundaries (the dispatcher derives at injection time, the
    verifier derives at receipt time with the same triple)."""
    a = derive_canary(
        signal_id="sig_abc",
        constitution_hash="con_def",
        engine_nonce="nonce_xyz",
    )
    b = derive_canary(
        signal_id="sig_abc",
        constitution_hash="con_def",
        engine_nonce="nonce_xyz",
    )
    assert a == b


def test_derive_canary_returns_lowercase_hex_of_correct_length():
    canary = derive_canary(
        signal_id="sig_1",
        constitution_hash="con_2",
        engine_nonce="nonce_3",
    )
    assert len(canary) == CANARY_HEX_LENGTH
    assert all(c in "0123456789abcdef" for c in canary)


def test_derive_canary_changes_when_any_input_changes():
    base = derive_canary(
        signal_id="sig_1", constitution_hash="con_1", engine_nonce="n_1"
    )
    differs_signal = derive_canary(
        signal_id="sig_2", constitution_hash="con_1", engine_nonce="n_1"
    )
    differs_hash = derive_canary(
        signal_id="sig_1", constitution_hash="con_2", engine_nonce="n_1"
    )
    differs_nonce = derive_canary(
        signal_id="sig_1", constitution_hash="con_1", engine_nonce="n_2"
    )
    assert {base, differs_signal, differs_hash, differs_nonce}.__len__() == 4


def test_derive_canary_separator_prevents_field_collision():
    """Without the unit-separator between fields,
    (signal='ab', hash='cd') and (signal='abc', hash='d') would hash
    identically. The 0x1f separator prevents that. Pin it explicitly."""
    a = derive_canary(
        signal_id="ab", constitution_hash="cd", engine_nonce="n"
    )
    b = derive_canary(
        signal_id="abc", constitution_hash="d", engine_nonce="n"
    )
    assert a != b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_id": "", "constitution_hash": "h", "engine_nonce": "n"},
        {"signal_id": "s", "constitution_hash": "", "engine_nonce": "n"},
        {"signal_id": "s", "constitution_hash": "h", "engine_nonce": ""},
    ],
)
def test_derive_canary_rejects_empty_inputs(kwargs):
    with pytest.raises(ValueError, match="non-empty"):
        derive_canary(**kwargs)


# ---------------------------------------------------------------------------
# is_well_formed_canary
# ---------------------------------------------------------------------------


def test_is_well_formed_canary_accepts_correct_shape():
    assert is_well_formed_canary("0123456789abcdef")


def test_is_well_formed_canary_rejects_wrong_length():
    assert not is_well_formed_canary("abc")
    assert not is_well_formed_canary("0" * (CANARY_HEX_LENGTH + 1))


def test_is_well_formed_canary_rejects_uppercase():
    """Output is lowercase hex; verifier must not accept variants the
    derivation never produces."""
    assert not is_well_formed_canary("ABCDEF0123456789")


def test_is_well_formed_canary_rejects_non_string():
    assert not is_well_formed_canary(None)
    assert not is_well_formed_canary(0x0123456789ABCDEF)


# ---------------------------------------------------------------------------
# verify_in_tool_calls (claude_code phantom-tool path)
# ---------------------------------------------------------------------------


def _canary() -> str:
    return derive_canary(
        signal_id="sig_test",
        constitution_hash="con_test",
        engine_nonce="nonce_test",
    )


def test_verify_in_tool_calls_verified_on_match_with_dict_args():
    canary = _canary()
    calls = [
        {
            "name": PHANTOM_RECEIPT_TOOL_NAME,
            "arguments": {PHANTOM_RECEIPT_ARG_NAME: canary},
        }
    ]
    assert verify_in_tool_calls(calls, canary) is CanaryStatus.VERIFIED


def test_verify_in_tool_calls_verified_with_json_string_args():
    """Some adapters emit `arguments` as a raw JSON string; both
    shapes are tolerated."""
    import json

    canary = _canary()
    calls = [
        {
            "name": PHANTOM_RECEIPT_TOOL_NAME,
            "arguments": json.dumps({PHANTOM_RECEIPT_ARG_NAME: canary}),
        }
    ]
    assert verify_in_tool_calls(calls, canary) is CanaryStatus.VERIFIED


def test_verify_in_tool_calls_missing_when_no_phantom_call():
    canary = _canary()
    calls = [
        {"name": "some_other_tool", "arguments": {"x": canary}},
    ]
    assert verify_in_tool_calls(calls, canary) is CanaryStatus.MISSING


def test_verify_in_tool_calls_missing_when_canary_mismatched():
    canary = _canary()
    calls = [
        {
            "name": PHANTOM_RECEIPT_TOOL_NAME,
            "arguments": {PHANTOM_RECEIPT_ARG_NAME: "ffffffffffffffff"},
        }
    ]
    assert verify_in_tool_calls(calls, canary) is CanaryStatus.MISSING


def test_verify_in_tool_calls_missing_on_unparseable_args():
    canary = _canary()
    calls = [
        {
            "name": PHANTOM_RECEIPT_TOOL_NAME,
            "arguments": "not valid json {",
        }
    ]
    assert verify_in_tool_calls(calls, canary) is CanaryStatus.MISSING


def test_verify_in_tool_calls_handles_empty_iterable():
    canary = _canary()
    assert verify_in_tool_calls([], canary) is CanaryStatus.MISSING


def test_verify_in_tool_calls_rejects_malformed_canary_input():
    """If the dispatcher accidentally passes a malformed canary (e.g.
    truncated, uppercase) we must not silently accept whatever the
    model echoed — return MISSING."""
    calls = [
        {
            "name": PHANTOM_RECEIPT_TOOL_NAME,
            "arguments": {PHANTOM_RECEIPT_ARG_NAME: "not-hex"},
        }
    ]
    assert verify_in_tool_calls(calls, "not-hex") is CanaryStatus.MISSING


# ---------------------------------------------------------------------------
# verify_in_structured_response (codex path)
# ---------------------------------------------------------------------------


def test_verify_in_structured_response_verified_on_default_field():
    canary = _canary()
    response = {CODEX_CANARY_FIELD: canary, "verdict": "OK"}
    assert (
        verify_in_structured_response(response, canary)
        is CanaryStatus.VERIFIED
    )


def test_verify_in_structured_response_supports_custom_field():
    canary = _canary()
    response = {"custom_canary": canary}
    assert (
        verify_in_structured_response(
            response, canary, field="custom_canary"
        )
        is CanaryStatus.VERIFIED
    )


def test_verify_in_structured_response_missing_when_field_absent():
    canary = _canary()
    response = {"verdict": "OK"}
    assert (
        verify_in_structured_response(response, canary)
        is CanaryStatus.MISSING
    )


def test_verify_in_structured_response_missing_when_value_wrong_type():
    canary = _canary()
    response = {CODEX_CANARY_FIELD: 12345}  # int instead of str
    assert (
        verify_in_structured_response(response, canary)
        is CanaryStatus.MISSING
    )


def test_verify_in_structured_response_missing_when_response_not_mapping():
    canary = _canary()
    assert (
        verify_in_structured_response("a string", canary)  # type: ignore[arg-type]
        is CanaryStatus.MISSING
    )


# ---------------------------------------------------------------------------
# verify_in_json_response (local-model path)
# ---------------------------------------------------------------------------


def test_verify_in_json_response_strict_json():
    import json

    canary = _canary()
    text = json.dumps({LOCAL_CANARY_FIELD: canary, "verdict": "OK"})
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.VERIFIED
    )


def test_verify_in_json_response_extracts_from_prose_wrapper():
    """Local models often wrap structured output. The verifier must
    extract the first balanced `{...}` block and parse that."""
    canary = _canary()
    text = (
        "Sure, here is the JSON response:\n\n"
        f'{{"{LOCAL_CANARY_FIELD}": "{canary}", "ok": true}}\n\n'
        "Hope that helps!"
    )
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.VERIFIED
    )


def test_verify_in_json_response_extracts_from_markdown_fence():
    canary = _canary()
    text = (
        "```json\n"
        f'{{"{LOCAL_CANARY_FIELD}": "{canary}"}}\n'
        "```"
    )
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.VERIFIED
    )


def test_verify_in_json_response_missing_when_no_json():
    canary = _canary()
    assert (
        verify_in_json_response("just plain prose", canary)
        is CanaryStatus.MISSING
    )


def test_verify_in_json_response_missing_when_field_absent():
    canary = _canary()
    text = '{"verdict": "OK"}'
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.MISSING
    )


def test_verify_in_json_response_missing_on_canary_mismatch():
    canary = _canary()
    text = f'{{"{LOCAL_CANARY_FIELD}": "ffffffffffffffff"}}'
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.MISSING
    )


def test_verify_in_json_response_handles_string_with_braces_in_quotes():
    """Brace-counting must respect string boundaries — `{"k":"}"}`
    has only one balanced pair, not zero."""
    canary = _canary()
    text = f'{{"note": "}}", "{LOCAL_CANARY_FIELD}": "{canary}"}}'
    assert (
        verify_in_json_response(text, canary) is CanaryStatus.VERIFIED
    )


# ---------------------------------------------------------------------------
# build_canary_instruction
# ---------------------------------------------------------------------------


def test_build_instruction_for_claude_code_mentions_phantom_tool():
    canary = _canary()
    text = build_canary_instruction(canary, "claude_code")
    assert text is not None
    assert PHANTOM_RECEIPT_TOOL_NAME in text
    assert PHANTOM_RECEIPT_ARG_NAME in text
    assert canary in text
    # Operational-metadata directive: do not mention canary in chat.
    assert "do" in text.lower() and "not" in text.lower()


def test_build_instruction_for_codex_references_field_name():
    canary = _canary()
    text = build_canary_instruction(canary, "codex")
    assert text is not None
    assert CODEX_CANARY_FIELD in text
    assert canary in text


def test_build_instruction_for_local_requires_json():
    canary = _canary()
    text = build_canary_instruction(canary, "local")
    assert text is not None
    assert LOCAL_CANARY_FIELD in text
    assert "JSON" in text or "json" in text


def test_build_instruction_for_bare_returns_none():
    """`bare` is caller-responsibility — no instruction injected."""
    canary = _canary()
    assert build_canary_instruction(canary, "bare") is None


def test_build_instruction_rejects_malformed_canary():
    """Defensive: don't embed garbage into the system prompt."""
    with pytest.raises(ValueError, match="invalid canary"):
        build_canary_instruction("not-hex", "claude_code")


def test_build_instruction_rejects_unknown_format():
    canary = _canary()
    with pytest.raises(ValueError, match="unknown prompt_template_format"):
        build_canary_instruction(canary, "weird_format")


# ---------------------------------------------------------------------------
# CanaryStatus enum semantics
# ---------------------------------------------------------------------------


def test_canary_status_string_values_match_signal_log_contract():
    """The string values are persisted into `signal_log.echo_canary_status`.
    Pin them so an auditor's `WHERE echo_canary_status = 'verified'`
    query keeps working."""
    assert CanaryStatus.VERIFIED.value == "verified"
    assert CanaryStatus.MISSING.value == "missing"
    assert CanaryStatus.NOT_REQUIRED.value == "not_required"
