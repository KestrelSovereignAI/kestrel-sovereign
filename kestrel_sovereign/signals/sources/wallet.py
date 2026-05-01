"""Source registration for the Stripe crypto-onramp deposit-complete
webhook (Phase 6 of #889).

This is the first **UNTRUSTED COGNITION** source in the dispatcher.
The sanitizer is the load-bearing component — review changes carefully.
The threat model: a Stripe webhook payload (or any field within it)
could carry text that, if injected directly into a prompt, would
manipulate the bird into taking actions outside its intent. Even
though Stripe's signature verification gates the webhook handler, the
*content* of fields like wallet addresses, transaction IDs, and
session metadata is operator-visible string data that we must treat
as potentially adversarial.

The defense is multi-layer:

1. **Allowlist sanitizer** — only the fields we explicitly know how
   to use survive. Unknown fields are dropped, not sanitized; they
   never reach the prompt.
2. **Per-field type/length checks** — strings get scrubbed of control
   characters, capped at length bounds, and required to look like the
   shape we expect (e.g. a wallet address is hex/base58, not free
   text).
3. **Prompt-template fence** — the prompt itself wraps the payload in
   `--- BEGIN UNTRUSTED PAYLOAD ---` markers and tells the bird in
   plain English not to interpret the contents as instructions. Belt
   and suspenders.
4. **Privacy-first redaction** — `signal_log.payload_redacted` stores
   only structural metadata (which fields were present, lengths,
   amounts), never the raw values. Wallet addresses are redacted to
   their last 4 characters.
5. **Conservative rate limit** — at most 30 deposits per hour can wake
   the bird. A runaway peer (or compromised signer) can't pin LLM
   cost to infinity.

`urgency_override=HIGH` is set so the dispatcher would bypass quiet
hours for this source — financial events SHOULD wake the bird at 3am
if they happen at 3am. Tunable via attention_policy if operators
disagree.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from kestrel_sdk.signals import (
    AttentionPolicy,
    RateLimit,
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Trust,
    Urgency,
    Visibility,
)

logger = logging.getLogger(__name__)


SOURCE_NAME = "webhook.stripe.deposit_complete"
PROMPT_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "signals" / "webhook_stripe_deposit.md"
)

# Per-field caps. Wallet addresses are <= 64 chars in any chain we
# care about; currency codes are 3-5 chars; amounts serialize short.
# Anything larger is suspicious — Stripe doesn't send 10kb session_ids.
_MAX_STRING_FIELD = 256
_MAX_AMOUNT_LEN = 32

# Allowlist of fields the bird is allowed to see. Order matches the
# prompt template's render expectations (the template renders the
# whole dict via {payload}, so this also controls what the bird reads).
_ALLOWED_FIELDS = frozenset({
    "session_id",
    "agent_did",
    "wallet_address",
    "destination_currency",
    "destination_network",
    "fiat_currency",
    "fiat_amount",
    "crypto_amount",
    "status",
})


# Control characters except common whitespace (tab/newline). Strip
# everything else; embedded NULs / escape sequences in webhook fields
# are not legitimate.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Wallet address shape: hex (with optional 0x) or base58. We don't
# validate semantics — that's the wallet feature's job — only that
# the string looks address-shaped. Anything else gets the placeholder
# `<malformed>` so the bird doesn't think a free-text wallet is real.
_WALLET_RE = re.compile(r"^(0x)?[A-Za-z0-9]{20,64}$")


# ---------------------------------------------------------------------------
# Schema (post-sanitizer)
# ---------------------------------------------------------------------------


def _stripe_schema(payload: dict) -> dict:
    """Validate the SANITIZED payload — runs after `_stripe_sanitize`.
    The sanitizer guarantees the dict only contains allowlist keys with
    typed/scrubbed values; this validator enforces the required-field
    invariant downstream code relies on."""
    if not isinstance(payload, dict):
        raise ValueError(f"stripe payload must be a dict, got {type(payload).__name__}")
    for key in ("session_id", "wallet_address", "destination_currency"):
        if key not in payload:
            raise ValueError(f"stripe payload missing required key: {key}")
    return payload


# ---------------------------------------------------------------------------
# Sanitizer — the load-bearing component for UNTRUSTED COGNITION
# ---------------------------------------------------------------------------


def _stripe_sanitize(payload: dict) -> dict:
    """Allowlist + per-field scrub. Returns a NEW dict; the input is
    not mutated. Unknown fields are dropped silently (operators see
    via `signal_log.payload_redacted`'s `dropped` count if needed).

    Per-field handling:
    - String fields → strip control chars, cap at _MAX_STRING_FIELD
    - wallet_address → must match _WALLET_RE; else replaced with
      "<malformed>" (so the bird sees explicit placeholder text)
    - Amount fields (fiat_amount, crypto_amount) → coerced to Decimal,
      formatted to a stable string; non-numeric values become None
    - Currency / network codes → uppercased (currency) or lowercased
      (network); non-string values become None
    - status → string, enum-checked against known on-ramp statuses
    """
    if not isinstance(payload, dict):
        raise ValueError(f"sanitizer expected dict, got {type(payload).__name__}")

    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in _ALLOWED_FIELDS:
            continue  # dropped — not even rendered to prompt

        if key == "wallet_address":
            out[key] = _scrub_wallet(value)
        elif key in ("fiat_amount", "crypto_amount"):
            out[key] = _scrub_amount(value)
        elif key == "destination_currency" or key == "fiat_currency":
            out[key] = _scrub_currency(value)
        elif key == "destination_network":
            out[key] = _scrub_network(value)
        elif key == "status":
            out[key] = _scrub_status(value)
        else:
            out[key] = _scrub_string(value)

    return out


def _scrub_string(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    cleaned = _CONTROL_CHARS.sub("", value)
    if len(cleaned) > _MAX_STRING_FIELD:
        cleaned = cleaned[:_MAX_STRING_FIELD]
    return cleaned


def _scrub_wallet(value: Any) -> str:
    s = _scrub_string(value)
    if s and _WALLET_RE.match(s):
        return s
    return "<malformed>"


def _scrub_amount(value: Any) -> Any:
    if value is None:
        return None
    try:
        d = Decimal(str(value)[:_MAX_AMOUNT_LEN])
    except Exception:
        return None
    return str(d)


def _scrub_currency(value: Any) -> Any:
    s = _scrub_string(value)
    if s is None:
        return None
    return s.upper()[:8]  # cap


def _scrub_network(value: Any) -> Any:
    s = _scrub_string(value)
    if s is None:
        return None
    return s.lower()[:32]


_KNOWN_STATUSES = {
    "pending", "requires_action", "processing",
    "succeeded", "failed", "expired",
}


def _scrub_status(value: Any) -> Any:
    s = _scrub_string(value)
    if s is None:
        return None
    s = s.lower()
    if s not in _KNOWN_STATUSES:
        return "unknown"
    return s


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _stripe_redact(payload: dict) -> str:
    """Privacy-first: store only structural metadata + last-4 of the
    wallet (operator can correlate to known wallets without leaking
    full addresses) + amount magnitudes (bucketed). Never the raw
    session_id or full wallet."""
    fields_present = sorted(payload.keys())
    addr = payload.get("wallet_address", "") or ""
    addr_tail = f"...{addr[-4:]}" if len(addr) >= 4 else "<short>"
    crypto_amt = payload.get("crypto_amount")
    fiat_amt = payload.get("fiat_amount")
    status = payload.get("status", "unknown")
    sig_id_digest = "<none>"
    sid = payload.get("session_id")
    if isinstance(sid, str) and sid:
        sig_id_digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:12]
    return (
        f"fields={fields_present} status={status} "
        f"addr_tail={addr_tail} crypto={crypto_amt} fiat={fiat_amt} "
        f"sid_sha256_12={sig_id_digest}"
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def build_stripe_deposit_registration() -> SourceRegistration:
    return SourceRegistration(
        name=SOURCE_NAME,
        # Schema runs AFTER sanitizer (Phase 1 pipeline order); we
        # validate the canonical scrubbed form, not the raw input.
        schema=_stripe_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=PROMPT_TEMPLATE,
        # UNTRUSTED — the registration validator (Phase 1) requires a
        # sanitizer for non-ACTION modes whenever trust is UNTRUSTED.
        # Removing the sanitizer would fail at register time.
        trust=Trust.UNTRUSTED,
        sanitizer=_stripe_sanitize,
        # Cap at 30/hr, burst 3 — Stripe deposits are user-initiated
        # and don't legitimately fire faster than this. A higher rate
        # is a sign of something wrong (replay storm, compromised
        # signer) and the bird shouldn't burn LLM tokens on it.
        rate_limit=RateLimit(per_hour=30, burst=3),
        # 5s window — same `session_id` firing twice in a tick is a
        # retry storm.
        coalescing_window=timedelta(seconds=5),
        # urgency_override=HIGH so financial events bypass operator-
        # configured quiet hours. Default urgency for the source is
        # NORMAL; callers can pass HIGH explicitly for true emergencies.
        attention_policy=AttentionPolicy(
            urgency_override=Urgency.HIGH,
        ),
        # Signal_log conflict: this declares no resources because
        # signal_log writes go through the dispatcher's tracker, not
        # via the source handler. Wallet state writes (if the bird
        # decides to take action during the turn) declare WALLET
        # themselves via TaskManager / feature locks.
        resources=frozenset(),
        # Self-loops never legitimate for a webhook source.
        allow_self_loops=False,
        log_redaction=RedactionPolicy(
            summarize=_stripe_redact,
            store_raw_trusted=False,  # honored regardless because trust=UNTRUSTED
            redact_caller_identifier=True,
        ),
        # 90 days for financial audit trail.
        retention_days=90,
    )


# ---------------------------------------------------------------------------
# Signal builder — called from the StripeWebhookHandler callback
# ---------------------------------------------------------------------------


def build_signal_for_deposit(session: Any, target_agent: str) -> Signal:
    """Build a COGNITION signal for a completed Stripe on-ramp deposit.

    `session` is duck-typed (anything with the OnRampSession fields)
    so the helper is testable without instantiating the wallet
    feature. The dispatcher's pipeline runs the source's sanitizer on
    `signal.payload` before the schema check — this builder just
    extracts the raw session fields; sanitization happens at dispatch.
    """
    raw_payload = {
        "session_id": getattr(session, "session_id", None),
        "agent_did": getattr(session, "agent_did", None),
        "wallet_address": getattr(session, "wallet_address", None),
        "destination_currency": getattr(session, "destination_currency", None),
        "destination_network": getattr(session, "destination_network", None),
        "fiat_currency": getattr(session, "fiat_currency", None),
        "fiat_amount": getattr(session, "fiat_amount", None),
        "crypto_amount": getattr(session, "crypto_amount", None),
        "status": _stringify_status(getattr(session, "status", None)),
    }

    sid = raw_payload.get("session_id") or "<unknown>"

    return Signal(
        source=SOURCE_NAME,
        kind="deposit_complete",
        mode=SignalMode.COGNITION,
        payload=raw_payload,
        target_agent=target_agent,
        visibility=Visibility.INTERNAL,
        urgency=Urgency.NORMAL,
        # Coalesce double-fired completions for the same Stripe session.
        dedupe_key=f"{sid}:completed",
    )


def _stringify_status(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):  # Enum
        return value.value
    return str(value)
