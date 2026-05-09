"""Constitutional echo-canary primitive.

The canary is a per-COGNITION-dispatch receipt token. It proves that the
constitution-bearing system prompt actually reached the model and was
incorporated, defending against:

- silent system-prompt truncation by the runtime
- a model swapped for one that didn't see the system prompt
- a downstream reviewer (`codex`, `local`) that passes on prompts
  without honoring them

The canary is content-agnostic — it doesn't rely on the model "saying
the right thing about the constitution"; it's just a structural receipt.
The dispatcher injects the canary into the system prompt with a
format-specific instruction, then verifies the canary came back via a
format-specific channel that does NOT pollute user-visible chat output.

Receipt mechanisms (per `docs/architecture/CONSTITUTION_INJECTION.md`
§3 "Echo-canary verification"):

| Format | Receipt channel |
|--------|-----------------|
| `claude_code` | Phantom `_constitution_receipt(canary)` tool call |
| `codex`       | Structured-response field (`constitution_canary`) |
| `local`       | JSON response with `_canary` field |
| `bare`        | Caller-responsibility — primitives provided here  |

This module is the pure-function core. The dispatcher (chunk 1G) wires
canary derivation, system-prompt injection, ephemeral tool registration,
and `signal_log.echo_canary_status` writes.

kestrel-sovereign#1137 chunk 1F.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import re
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)


# Hex prefix length of the sha256 digest used as the canary value.
# 16 hex chars = 64 bits of entropy — ample for collision resistance
# at per-invocation scale (we expect ~1 canary per COGNITION dispatch,
# i.e. << 2^32 over a system's lifetime), and small enough that it
# barely impacts prompt budget. (Open question 2 in the design;
# resolved to 16.)
CANARY_HEX_LENGTH = 16

# Canonical phantom-tool name. The dispatcher registers an ephemeral
# tool by this name for the duration of a single COGNITION turn when
# `prompt_template_format == "claude_code"` and
# `require_constitution_echo=True`. Per-turn lifecycle is chunk 1G's
# responsibility; this constant is the contract both sides share.
PHANTOM_RECEIPT_TOOL_NAME = "_constitution_receipt"

# Argument name the phantom tool expects.
PHANTOM_RECEIPT_ARG_NAME = "canary"

# Codex/local structured-response keys.
CODEX_CANARY_FIELD = "constitution_canary"
LOCAL_CANARY_FIELD = "_canary"

# Hex-only canary so we don't accept arbitrary content-shaped strings.
# Match exactly CANARY_HEX_LENGTH lowercase hex chars; the dispatcher
# always emits lowercase via `hashlib.sha256().hexdigest()`.
_CANARY_RE = re.compile(rf"^[0-9a-f]{{{CANARY_HEX_LENGTH}}}$")


class CanaryStatus(str, enum.Enum):
    """Outcome of canary verification.

    Persisted as the literal enum value into
    `signal_log.echo_canary_status` (string column). Auditors filter
    on `WHERE echo_canary_status = 'missing'` to find dispatches
    whose constitution receipt failed.
    """

    VERIFIED = "verified"
    MISSING = "missing"
    NOT_REQUIRED = "not_required"


def derive_canary(
    *, signal_id: str, constitution_hash: str, engine_nonce: str
) -> str:
    """Compute the per-invocation canary token.

    `canary = sha256(signal_id || constitution_hash || engine_nonce)[:16]`

    The three inputs together ensure the canary is unique per dispatch
    even when the signal_id collides (cross-replay), the constitution
    is reanchored mid-flight, or the engine-supplied nonce exposes
    that the dispatcher (not the caller) generated this canary.

    Args:
        signal_id: The dispatch's signal envelope id (`sig_...`).
        constitution_hash: The agent's anchored constitution hash at
            dispatch time. Including it means a reanchor between
            derivation and verification flips the canary, which is
            the desired behavior (drift caught at verify time).
        engine_nonce: A per-dispatcher random nonce that prevents an
            adversary who controls just `signal_id` (e.g. by
            harvesting causation chains) from precomputing canaries.

    Returns:
        16 lowercase hex characters.
    """
    if not signal_id or not constitution_hash or not engine_nonce:
        raise ValueError(
            "derive_canary requires non-empty signal_id, "
            "constitution_hash, and engine_nonce."
        )
    digest = hashlib.sha256(
        signal_id.encode("utf-8")
        + b"\x1f"  # unit-separator — keeps individual fields demarcated
        + constitution_hash.encode("utf-8")
        + b"\x1f"
        + engine_nonce.encode("utf-8")
    ).hexdigest()
    return digest[:CANARY_HEX_LENGTH]


def is_well_formed_canary(value: Any) -> bool:
    """Cheap structural check — returns True for a string of exactly
    CANARY_HEX_LENGTH lowercase hex characters."""
    return isinstance(value, str) and bool(_CANARY_RE.match(value))


# ---------------------------------------------------------------------------
# Format-specific verification primitives
# ---------------------------------------------------------------------------


def verify_in_tool_calls(
    tool_calls: Iterable[Mapping[str, Any]], expected_canary: str
) -> CanaryStatus:
    """Scan a turn's captured tool-call telemetry for the phantom
    `_constitution_receipt` invocation.

    `tool_calls` is a sequence of dict-like records following the
    sovereign tool-use telemetry shape — each record has at minimum
    `{"name": str, "arguments": Mapping[str, Any] | str}`. Both shapes
    of `arguments` (parsed dict OR raw JSON string) are tolerated to
    match what different LLM adapters emit.

    Match criterion: at least one tool call where `name ==
    PHANTOM_RECEIPT_TOOL_NAME` AND its arguments contain
    `canary == expected_canary` (string-equal). Returns VERIFIED on
    match, MISSING otherwise.

    This function is intentionally permissive about extra fields —
    the dispatcher captures the same shape from many adapters and we
    don't want a schema mismatch to silently downgrade to MISSING.
    """
    if not is_well_formed_canary(expected_canary):
        return CanaryStatus.MISSING

    for call in tool_calls:
        if call.get("name") != PHANTOM_RECEIPT_TOOL_NAME:
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(args, Mapping):
            continue
        received = args.get(PHANTOM_RECEIPT_ARG_NAME)
        if isinstance(received, str) and received == expected_canary:
            return CanaryStatus.VERIFIED
    return CanaryStatus.MISSING


def verify_in_structured_response(
    response: Mapping[str, Any],
    expected_canary: str,
    *,
    field: str = CODEX_CANARY_FIELD,
) -> CanaryStatus:
    """Verify a `codex`-format reviewer's structured-response payload.

    Codex CLI emits structured output with caller-defined fields; the
    dispatcher reserves `constitution_canary` (overridable). VERIFIED
    when the field equals `expected_canary` exactly; MISSING when the
    field is absent, of the wrong type, or doesn't match.

    `response` is the parsed structured response (already JSON-decoded
    by the caller). For raw JSON text, see `verify_in_json_response`.
    """
    if not is_well_formed_canary(expected_canary):
        return CanaryStatus.MISSING
    if not isinstance(response, Mapping):
        return CanaryStatus.MISSING
    received = response.get(field)
    if isinstance(received, str) and received == expected_canary:
        return CanaryStatus.VERIFIED
    return CanaryStatus.MISSING


def verify_in_json_response(
    response_text: str,
    expected_canary: str,
    *,
    field: str = LOCAL_CANARY_FIELD,
) -> CanaryStatus:
    """Verify a `local`-format reviewer's response.

    Local reviewers (llama.cpp, vllm) get a prompt that REQUIRES a
    JSON body with a `_canary` field. The response_text is the raw
    string the reviewer produced. We try strict JSON parse first,
    then fall back to extracting the first balanced `{...}` block —
    local models sometimes wrap structured output in prose or
    markdown fences and we don't want to fail VERIFIED on cosmetic
    surrounding text.

    Returns VERIFIED on match, MISSING otherwise (including unparsable
    response, missing field, type mismatch).
    """
    if not is_well_formed_canary(expected_canary):
        return CanaryStatus.MISSING
    if not isinstance(response_text, str):
        return CanaryStatus.MISSING

    parsed = _parse_json_or_extract(response_text)
    if not isinstance(parsed, Mapping):
        return CanaryStatus.MISSING

    received = parsed.get(field)
    if isinstance(received, str) and received == expected_canary:
        return CanaryStatus.VERIFIED
    return CanaryStatus.MISSING


def _parse_json_or_extract(text: str) -> Optional[Any]:
    """Best-effort JSON parser tolerant of surrounding prose / fences.

    Tries `json.loads(text)` first. On failure, scans for the first
    `{` and tries to parse from there to a balanced `}`. Returns
    None if nothing parsable is found. Used by the local-model
    canary verifier where output discipline varies.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Find the first balanced top-level {...} block. This is naive
    # (string-aware but not nesting-aware beyond brace counting); good
    # enough for canary extraction where the payload is small and
    # we control the schema instruction.
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        c = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
    return None


# ---------------------------------------------------------------------------
# System-prompt instruction text
# ---------------------------------------------------------------------------


def build_canary_instruction(
    canary: str, prompt_template_format: str
) -> Optional[str]:
    """Build the format-appropriate instruction for receiving the canary.

    The dispatcher embeds this string in the system prompt (as its own
    clause via the assembler) so the model knows to acknowledge the
    constitution via the right channel.

    Returns None for `bare` (caller-responsibility) and for the legacy
    `claude_code` opt-out path (echo not required → no instruction).
    """
    if not is_well_formed_canary(canary):
        raise ValueError(
            f"build_canary_instruction: invalid canary value {canary!r}; "
            f"expected {CANARY_HEX_LENGTH} lowercase hex chars."
        )

    if prompt_template_format == "claude_code":
        return (
            "--- CONSTITUTION RECEIPT (REQUIRED) ---\n"
            "Before responding, call the "
            f"`{PHANTOM_RECEIPT_TOOL_NAME}` tool with this exact "
            "argument to confirm you received the constitutional "
            "context:\n\n"
            f"  {PHANTOM_RECEIPT_ARG_NAME}: {canary}\n\n"
            "The tool returns immediately and does not affect "
            "user-visible output. Then proceed with your normal "
            "response. This receipt is operational metadata; do "
            "NOT mention the canary in your text reply.\n"
            "--- END CONSTITUTION RECEIPT ---"
        )

    if prompt_template_format == "codex":
        return (
            "constitution_canary: your structured response MUST "
            f"include the field `{CODEX_CANARY_FIELD}` with the "
            f"exact value {canary!r}. This is a non-negotiable "
            "receipt that the constitutional context reached you."
        )

    if prompt_template_format == "local":
        return (
            "Reply with a JSON object containing a "
            f"`{LOCAL_CANARY_FIELD}` field set to the exact string "
            f"{canary!r}. Surrounding prose is tolerated but the "
            "JSON object must be present and parsable."
        )

    if prompt_template_format == "bare":
        # Caller is responsible for whatever instruction discipline
        # they want; the canary is exposed and verification is theirs.
        return None

    raise ValueError(
        f"build_canary_instruction: unknown prompt_template_format "
        f"{prompt_template_format!r}."
    )
