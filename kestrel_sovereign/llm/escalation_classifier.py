"""Classify a tool-escalation failure outcome honestly (#1540 / #1563).

The bug this module fixes: when a tool invocation fails under sandbox
or approval routing — e.g. Codex's shell sandbox surfaces a
``CreateProcess { message: "Rejected(\"rejected by user\")" }`` error
to the LLM — the LLM (or a downstream narrative-formatter) was
echoing the wording verbatim and reporting "the user rejected
escalation". The security audit later showed NO pending approval,
NO denial record; the user never decided anything. The Codex CLI's
``rejected by user`` substring is its INTERNAL diagnostic for an
auto-mode sandbox refusal, not Kestrel-attributable provenance.

This module is the documented source of truth for converting a raw
tool-error string + an optional security-audit cross-reference into
a typed ``EscalationOutcome``. Callers (Codex adapter, response
audit, tool-result formatter) MUST route through here before
narrating an escalation failure to the user. The narrative helper
``format_escalation_outcome`` guarantees the wording never blames
the user without audit evidence.

Hooked taxonomy mirrors ``features/talon/verification.py``'s
``VerificationState``: same five states, same provenance contract.
Sharing the contract (but not importing across feature boundaries)
keeps both call-sites — Talon's reviewer-side test-evidence path
and the LLM's general tool-result path — in lockstep about what
counts as a real user denial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class EscalationOutcome(str, Enum):
    """Typed lifecycle outcome for a failed tool escalation.

    String-valued so it serializes cleanly through tool results,
    audit rows, and signal payloads without a custom encoder.
    """

    USER_DENIED = "user_denied"
    """A human explicitly pressed deny via the approval prompt
    AND the security audit recorded ``user_denied`` (or one of the
    web-UI per-scope variants ``once`` / ``session`` / ``always``
    that only a human can select). This is the only state that
    may be narrated as "the user rejected/denied …"."""

    POLICY_BLOCKED = "policy_blocked"
    """Operator/auto policy refused, OR the approval request timed
    out / was cancelled, OR no user ever decided. This is NOT a
    user denial — it is a Kestrel-policy / approval-plumbing
    refusal that must NEVER be reported as such."""

    SANDBOX_BLOCKED = "sandbox_blocked"
    """Execution environment (Codex shell sandbox, container,
    seccomp, etc.) refused to run the command before any approval
    decision. The Codex ``Rejected("rejected by user")`` substring
    lands here when no Kestrel audit row corroborates a user
    decision — the ``"by user"`` wording is Codex's internal
    sandbox diagnostic, not evidence of a Kestrel-attributable
    user action."""

    TOOLING_ERROR = "tooling_error"
    """The command could not run for a tooling reason: binary
    missing, RPC timeout, CreateProcess failure unrelated to
    sandboxing, exception during invocation."""

    UNCONFIRMED = "unconfirmed"
    """We have no evidence at all about what happened. The
    fall-back when neither the raw error nor the audit positively
    attributes the outcome. Surfaces in the narrative as
    "could not be confirmed"."""


# Recent-decision audit row shape. Callers pass a list of dicts (or
# any Mapping) sourced from ``PermissionStore.get_audit_log`` so
# this module does not depend on a specific store class.
RecentDecision = Mapping[str, Any]


@dataclass(frozen=True)
class EscalationDecision:
    """Result of classifying an escalation failure."""

    outcome: EscalationOutcome
    """Typed lifecycle outcome — the source of truth for narration."""

    reason: str
    """One-line human-readable rationale tying the outcome to the
    evidence the classifier used. Safe to include in a chat
    narration; the wording never says "user denied" unless the
    outcome is USER_DENIED."""

    evidence_source: str
    """Where the classifier got its evidence. One of:
       - ``audit``            audit row positively attributed
       - ``raw_error``        raw error string matched a known pattern
       - ``default``          fallback when nothing else matched
    """

    raw_error: str = ""
    """The raw tool error string passed in, retained for debugging
    so a future call-site can trace why a particular classification
    landed."""


# Wire-level patterns Codex / sandbox / OS layers emit. Order matters:
# more specific patterns first. The ``USER_DENIED`` claim from the
# raw error is INTENTIONALLY treated as advisory only — the canonical
# user-denial signal is the security audit, never the raw string.
_TOOLING_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"binary not found", re.IGNORECASE),
    re.compile(r"executable not found", re.IGNORECASE),
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"no such file or directory.+(executable|binary)", re.IGNORECASE),
    re.compile(r"createprocess.+enoent", re.IGNORECASE),
    re.compile(r"timed? ?out", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"rpc error", re.IGNORECASE),
)

# Sandbox-refusal patterns. The Codex ``Rejected("rejected by user")``
# wording maps here whenever the audit does NOT corroborate it; that
# is the entire reason this classifier exists.
_SANDBOX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sandbox\s+(refused|blocked|denied)", re.IGNORECASE),
    re.compile(r"permission denied.+(seccomp|capability)", re.IGNORECASE),
    re.compile(r"rejected.+user", re.IGNORECASE),  # Codex CLI string
    re.compile(r"createprocess.+rejected", re.IGNORECASE),
    re.compile(r"capability.+denied", re.IGNORECASE),
    re.compile(r"escalation.+denied.+sandbox", re.IGNORECASE),
)

# Policy / approval-plumbing patterns. These mean Kestrel's own layer
# refused or never resolved — distinct from sandbox.
_POLICY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"approval.*timed?\s*out", re.IGNORECASE),
    re.compile(r"approval.+(cancel|cancell)ed", re.IGNORECASE),
    re.compile(r"auto[_ -]?denied", re.IGNORECASE),
    re.compile(r"policy.+(blocked|denied|refused)", re.IGNORECASE),
    re.compile(r"blocked.+policy", re.IGNORECASE),
    re.compile(r"approval.+no.+decision", re.IGNORECASE),
)


# Audit decision values that constitute a real user denial. Mirrors
# the contract in ``features/talon/verification.classify_denial`` — keep
# both lists in sync if the security feature ever extends them.
_USER_DENIAL_DECISIONS: frozenset[str] = frozenset({
    "user_denied",
})
_USER_DENIAL_USER_CHOICES: frozenset[str] = frozenset({
    "user_denied", "once", "session", "always",
})


def _audit_attributes_user_denial(
    audit: Iterable[RecentDecision],
    *,
    tool_name: str = "",
    feature_name: str = "",
) -> Optional[RecentDecision]:
    """Return the first audit row that explicitly attributes a user
    denial, or ``None``. When ``tool_name`` / ``feature_name`` are
    set, only rows matching them count — so an unrelated deny for a
    different tool cannot be misattributed to the current attempt.
    """
    for row in audit:
        decision = str(row.get("decision", "")).strip().lower()
        user_choice = str(row.get("user_choice", "")).strip().lower()
        if (
            decision not in _USER_DENIAL_DECISIONS
            and user_choice not in _USER_DENIAL_USER_CHOICES
        ):
            continue
        if tool_name and str(row.get("tool", "")) != tool_name:
            continue
        if feature_name and str(row.get("feature", "")) != feature_name:
            continue
        return row
    return None


def classify_escalation_failure(
    raw_error: str,
    *,
    recent_decisions: Optional[Iterable[RecentDecision]] = None,
    tool_name: str = "",
    feature_name: str = "",
) -> EscalationDecision:
    """Classify a failed tool escalation honestly.

    Resolution order:

    1. **Security audit** — if a recent audit row positively
       attributes the failure to a real user denial (``user_denied``
       decision, or a per-scope ``once`` / ``session`` / ``always``
       user_choice on a deny), the outcome is ``USER_DENIED``. This
       is the ONLY path that produces that classification.

    2. **Raw error patterns** — if the audit cannot confirm a user
       denial, parse the raw error string against the known
       sandbox / tooling / policy regexes. The Codex
       ``Rejected("rejected by user")`` substring lands in
       ``SANDBOX_BLOCKED`` here because no audit row backs the
       wording — exactly the #1540 acceptance criterion.

    3. **Default** — neither audit nor patterns gave a positive
       attribution. Return ``UNCONFIRMED`` so the narrator says
       "could not be confirmed" instead of guessing.

    ``recent_decisions`` is typically the last ~50 rows from
    ``PermissionStore.get_audit_log``; the caller passes whatever
    window is timely (the security feature ages out stale rows).
    """
    raw = (raw_error or "").strip()

    # 1. Audit first. The audit is the source of truth.
    if recent_decisions is not None:
        match = _audit_attributes_user_denial(
            recent_decisions,
            tool_name=tool_name,
            feature_name=feature_name,
        )
        if match is not None:
            chosen = str(
                match.get("user_choice", "") or match.get("decision", "")
            )
            return EscalationDecision(
                outcome=EscalationOutcome.USER_DENIED,
                reason=(
                    f"security audit recorded an explicit user denial "
                    f"({chosen!r}) for this tool"
                ),
                evidence_source="audit",
                raw_error=raw,
            )

    # 2. Raw error pattern matching. The audit had nothing positive,
    # so we cannot promote any wording from the raw string into a
    # user-denial claim — even if the string literally says
    # ``rejected by user``.
    if raw:
        for pattern in _SANDBOX_PATTERNS:
            if pattern.search(raw):
                return EscalationDecision(
                    outcome=EscalationOutcome.SANDBOX_BLOCKED,
                    reason=(
                        "sandbox / approval plumbing refused the "
                        "command before any user decision was recorded"
                    ),
                    evidence_source="raw_error",
                    raw_error=raw,
                )
        for pattern in _POLICY_PATTERNS:
            if pattern.search(raw):
                return EscalationDecision(
                    outcome=EscalationOutcome.POLICY_BLOCKED,
                    reason=(
                        "operator/auto policy refused the command, or "
                        "no user ever decided — not a user denial"
                    ),
                    evidence_source="raw_error",
                    raw_error=raw,
                )
        for pattern in _TOOLING_ERROR_PATTERNS:
            if pattern.search(raw):
                return EscalationDecision(
                    outcome=EscalationOutcome.TOOLING_ERROR,
                    reason=(
                        "the command could not run for a tooling "
                        "reason (binary, timeout, or invocation error)"
                    ),
                    evidence_source="raw_error",
                    raw_error=raw,
                )

    # 3. Default — say so plainly.
    return EscalationDecision(
        outcome=EscalationOutcome.UNCONFIRMED,
        reason=(
            "the outcome could not be confirmed from either the tool "
            "error or the security audit"
        ),
        evidence_source="default",
        raw_error=raw,
    )


def format_escalation_outcome(
    decision: EscalationDecision, *, command: str = "",
) -> str:
    """Render an ``EscalationDecision`` into LLM-safe narrative text.

    The contract: the returned string NEVER says "user rejected" /
    "rejected by user" / "user denied" unless ``decision.outcome ==
    USER_DENIED`` (audit-backed). For every other outcome, the
    wording explicitly attributes the failure to the sandbox /
    policy / tooling / unconfirmed source so the assistant cannot
    later be quoted blaming the user.

    Use this helper everywhere the LLM (or an audit/response
    formatter) renders an escalation failure into prose — that is
    the entire #1540 acceptance criterion ("not blame the user
    unless audit-backed").
    """
    cmd = (command or "").strip()
    cmd_suffix = f" for `{cmd}`" if cmd else ""
    if decision.outcome == EscalationOutcome.USER_DENIED:
        return (
            f"The user explicitly denied the escalation{cmd_suffix} "
            f"at the approval prompt; reason: {decision.reason}."
        )
    if decision.outcome == EscalationOutcome.SANDBOX_BLOCKED:
        return (
            f"The escalation{cmd_suffix} was blocked by the "
            f"sandbox / approval plumbing — not by an explicit user "
            f"denial. {decision.reason.capitalize()}."
        )
    if decision.outcome == EscalationOutcome.POLICY_BLOCKED:
        return (
            f"The escalation{cmd_suffix} was blocked by policy — "
            f"not by an explicit user denial. "
            f"{decision.reason.capitalize()}."
        )
    if decision.outcome == EscalationOutcome.TOOLING_ERROR:
        return (
            f"The escalation{cmd_suffix} could not run due to a "
            f"tooling error — not by an explicit user denial. "
            f"{decision.reason.capitalize()}."
        )
    return (
        f"The escalation{cmd_suffix} outcome could not be confirmed. "
        f"{decision.reason.capitalize()}. "
        f"Do not attribute this to a user denial without audit "
        f"evidence."
    )
