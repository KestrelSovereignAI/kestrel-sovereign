"""
Deterministic narration check for streaming agents (#1042 layer 3).

The streaming agent may emit pre-tool prose to the client BEFORE the
underlying tool actually runs. When the tool then fails or returns no
positive confirmation, the user has already seen confident
past-tense success language. This module provides a deterministic
post-turn check that compares:

* the pre-tool prose snapshot (text streamed before the first
  ``ToolCallStarted`` marker — captured upstream in
  ``agent/streaming.py``), against
* the actual tool result envelopes observed during the turn.

When confident past-tense language appears in the pre-tool prose
without a successful confirmation in any tool result, the check
returns an elevated risk + reasoning string. ``ResponseAuditHook``
folds that into its existing risk score so audit warnings/blocks
fire on a deterministic signal rather than an LLM-judgment-only path.

Why deterministic
-----------------
The earlier audit path was a separate LLM call (``get_audit_response``),
which is itself probabilistic. The narration check is a pure-Python
regex + envelope-shape match: same inputs always yield the same
verdict. That makes it suitable for compliance-gated deployments
where "I told you the agent doesn't lie about tool success" needs a
falsifiable claim.

Limits
------
This is a heuristic. False positives are possible:

* The agent writes "Saved your previous turn — now adding to it"
  before calling the new tool, where "Saved" refers to past state.
* A multi-tool turn where the success verb refers to one tool that
  succeeded but a different tool failed (e.g. "Saved the draft" with
  ``save_draft`` returning ok but ``notify_team`` returning error).
  The check currently flags any failure in the result list against
  any past-tense verb; correlating verbs to specific tool names is
  future work.

Hook authors who care about precision should compose this with their
own checks. The module ships with the audit hook in WARN mode by
default to keep operational impact low.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Past-tense success verbs the streaming agent has been observed to
# emit ("saved", "stored", "sent", etc.). The trailing context allows
# matching "Saved!" / "Saved:" / "Saved your..." / "Done." but does
# NOT match "I'll save" or "saving that now" — those are honest
# present-progressive hedges (per the TOOL_HONESTY_SYSTEM_PROMPT
# doctrine, kestrel-sovereign #1043). Must be early in the prose
# (within the first ~200 chars, anchored loosely so leading
# acknowledgements like "Sure! Saved..." still match).
_PAST_TENSE_SUCCESS_VERBS = (
    r"saved|stored|captured|recorded|created|added|inserted|"
    r"deleted|removed|updated|edited|changed|modified|"
    r"sent|delivered|posted|published|"
    r"completed|finished|done|got it stored|got it saved|"
    r"scheduled|booked"
)

# Anchored at start of a sentence-ish boundary, optionally preceded by
# a short acknowledgement / punctuation. The verb itself is followed
# by punctuation, end-of-sentence, or " your"/" the"/" that"/" it"
# (the typical "Saved your favorite color" pattern).
_PAST_TENSE_SUCCESS_RE = re.compile(
    rf"(?:^|[\.!\n]\s*|\b(?:sure|okay|got it)[!,]?\s+)"
    rf"(?P<verb>{_PAST_TENSE_SUCCESS_VERBS})"
    rf"(?=[\.\!\:\,\s]|\Z)",
    re.IGNORECASE,
)

# Cap how far into the pre-tool prose we look — past-tense claims
# after several paragraphs are increasingly likely to be referring to
# past state, not the about-to-run tool.
_PRE_TOOL_SCAN_LIMIT_CHARS = 400


@dataclass(frozen=True)
class NarrationVerdict:
    """Result of the narration check.

    * ``risk_boost`` — non-negative integer added to the audit hook's
      existing risk score. ``0`` means no violation; ``2`` means the
      check is confident enough to elevate the response toward DENY/
      modify in strict mode.
    * ``reasoning`` — human-readable explanation referencing the
      offending verb and (when applicable) which tool returned a
      non-success status. Empty when ``risk_boost == 0``.
    * ``offending_verb`` / ``offending_tool`` — the specific tokens
      that drove the verdict, surfaced separately for telemetry +
      structured logging. ``None`` when no violation was found.
    """

    risk_boost: int
    reasoning: str
    offending_verb: Optional[str] = None
    offending_tool: Optional[str] = None


def summarize_tool_result_for_audit(result: Any) -> Any:
    """Slim a tool's return envelope down to only the fields the
    narration check (and a typical POST_RESPONSE audit hook) needs.

    The motivation is privacy + size: the orchestrator's full
    ``_serialize_tool_result`` envelope can contain memory contents,
    search hits, file contents, or other sensitive payload data.
    Passing that to every registered POST_RESPONSE hook (including
    third-party plugins) is over-broad. Codex review of
    kestrel-sovereign #1076 caught this: the LLM message gets
    truncated to ``MAX_TOOL_RESULT_CHARS`` but the audit-hook payload
    bypassed that gate.

    Returns a small dict with only the audit-relevant envelope shape:

    * ``status`` — when the result conforms to the ToolResult
      envelope (#1042 layer 4).
    * ``success`` — when the result is a legacy ``{success: bool}``
      envelope.
    * ``error`` — capped at 500 chars to prevent log-bomb amplification.

    Non-dict results (raw strings, numbers, lists, ``None``, etc.)
    are coerced to an opaque ``{"status": "unknown"}`` envelope so the
    raw value never reaches a POST_RESPONSE hook. Codex re-review of
    #1076 caught this: tools that return primitive strings (file
    contents, search snippets, raw model output) would otherwise pass
    through verbatim. ``analyze_narration`` treats unknown / non-dict
    as failure either way, so audit semantics are preserved.
    """
    if not isinstance(result, dict):
        return {"status": "unknown"}
    summary: Dict[str, Any] = {}
    if "status" in result:
        summary["status"] = result["status"]
    if "success" in result:
        summary["success"] = result["success"]
    err = result.get("error")
    if err:
        summary["error"] = err[:500] if isinstance(err, str) else err
    return summary


def _result_indicates_failure(result: Any) -> bool:
    """Return True if a tool's result envelope encodes a failure or
    a no-positive-confirmation outcome.

    Handles three observed shapes:
    * ToolResult envelope (``{status, ...}``): failure when status
      is ``error`` or ``partial`` (#1042 layer 4).
    * Legacy ``{success: bool, ...}``: failure when ``success is False``.
    * Anything else: treat as failure if an ``error`` key is present
      with a truthy value.

    A None or non-dict result is treated as failure — the agent
    cannot honestly claim success against a result it didn't
    receive.
    """
    if not isinstance(result, dict):
        return True
    status = result.get("status")
    if isinstance(status, str):
        if status == "ok":
            return False
        # ``partial`` and ``error`` both block past-tense success
        # claims — the agent cannot honestly say "Saved" if the
        # operation was partial (some sub-step failed) or errored.
        if status in ("error", "partial"):
            return True
        # Unknown status string — treat conservatively as failure.
        return True
    if "success" in result:
        # Use ``is not True`` rather than ``is False`` so legacy
        # envelopes with success=None / success="false" / numeric 0
        # are treated as failure-for-audit-purposes. Codex review of
        # #1076 caught this: ``is False`` only matches literal False
        # and would silently let stringly-typed legacy envelopes pass.
        return result["success"] is not True
    if result.get("error"):
        return True
    # No status, no success flag, no error — ambiguous. The honesty
    # doctrine says: don't claim success without explicit
    # confirmation. Treat ambiguous as failure-for-audit-purposes.
    return True


def analyze_narration(
    pre_tool_prose: Optional[str],
    tool_results: Optional[List[Dict[str, Any]]],
) -> NarrationVerdict:
    """Check whether the pre-tool prose makes a past-tense success
    claim that the tool results don't support.

    Args:
        pre_tool_prose: Text the agent streamed before the first
            ``ToolCallStarted`` marker. ``None`` or empty when no
            pre-tool text was streamed — in which case there's
            nothing to audit.
        tool_results: Tool result envelopes observed in the turn,
            shape ``[{tool_call_id, name, result}]``. ``None`` /
            empty when the turn fired no tools — in which case the
            pre-tool/post-tool distinction doesn't apply (the streamed
            text IS the answer, not speculation about a tool action).

    Returns:
        ``NarrationVerdict`` with ``risk_boost == 0`` when no
        violation is found, ``2`` otherwise.
    """
    if not pre_tool_prose:
        return NarrationVerdict(risk_boost=0, reasoning="")
    if not tool_results:
        # Tools didn't fire (or hook caller didn't supply results).
        # Without observed tool results, this check has no factual
        # basis for declaring the prose a lie.
        return NarrationVerdict(risk_boost=0, reasoning="")

    head = pre_tool_prose[:_PRE_TOOL_SCAN_LIMIT_CHARS]
    match = _PAST_TENSE_SUCCESS_RE.search(head)
    if not match:
        return NarrationVerdict(risk_boost=0, reasoning="")

    failures = [tr for tr in tool_results if _result_indicates_failure(tr.get("result"))]
    if not failures:
        # Past-tense claim was made AND every tool result confirms
        # success. The narration is consistent with the observed
        # action — no audit elevation.
        return NarrationVerdict(risk_boost=0, reasoning="")

    # At least one tool returned non-success while the agent already
    # claimed past-tense completion. Surface the first offender.
    offender = failures[0]
    offender_name = offender.get("name", "<unknown>")
    verb = match.group("verb").lower()
    return NarrationVerdict(
        risk_boost=2,
        reasoning=(
            f"Pre-tool prose claimed past-tense success ({verb!r}) but "
            f"tool {offender_name!r} returned a non-success result. "
            f"This pattern violates the constitutional honesty layer "
            f"(#1042 layer 3): the agent narrated completion before "
            f"observing the actual tool outcome."
        ),
        offending_verb=verb,
        offending_tool=offender_name,
    )


# ---------------------------------------------------------------------------
# Escalation-attribution check (#1563 wire-up of the #1540 classifier)
# ---------------------------------------------------------------------------


# Phrases the agent must not use unless an audit row actually attributes
# the failure to a user denial. Codex's raw ``Rejected("rejected by
# user")`` substring lands here even when no Kestrel approval row backs
# it — that's the exact #1563 reproduction.
_FORBIDDEN_USER_DENIAL_RE = re.compile(
    r"\b(rejected by (?:the )?user"
    r"|denied by (?:the )?user"
    r"|declined by (?:the )?user"
    r"|refused by (?:the )?user"
    r"|user (?:explicitly )?(?:denied|rejected|declined|refused)"
    r"|escalation (?:rejected|denied|declined|refused) by (?:the )?user"
    r"|you denied the escalation"
    r"|user denial)\b",
    re.IGNORECASE,
)


def check_escalation_attribution(
    response_text: Optional[str],
    tool_results: Optional[List[Dict[str, Any]]],
    *,
    recent_decisions: Optional[List[Dict[str, Any]]] = None,
    tool_name: str = "",
    feature_name: str = "",
) -> NarrationVerdict:
    """Catch dishonest escalation-attribution wording in the response.

    The #1563 root case: the LLM, having observed a Codex sandbox
    rejection like ``CreateProcess { message: "Rejected(\"rejected by
    user\")" }``, narrates "the user rejected escalation" — when the
    security audit has no denial row at all and the user never
    decided anything. The Codex string's ``"by user"`` substring is
    the sandbox's INTERNAL diagnostic, not Kestrel-attributable
    provenance.

    This check runs alongside ``analyze_narration``. It scans the
    response for forbidden user-denial wording; if found, it routes
    the most-recent failing tool result through
    :func:`classify_escalation_failure` and flags as a narration
    violation when the classifier's outcome is anything other than
    ``USER_DENIED``.

    Returns:
        ``NarrationVerdict`` with ``risk_boost == 0`` when the
        response makes no user-denial claim, OR makes one that the
        audit confirms. Returns ``risk_boost == 2`` when the agent
        narrated user denial but the classifier disagreed —
        equivalent severity to the past-tense-success violation
        above so the existing audit threshold logic catches it.
    """
    if not response_text:
        return NarrationVerdict(risk_boost=0, reasoning="")
    match = _FORBIDDEN_USER_DENIAL_RE.search(response_text)
    if not match:
        return NarrationVerdict(risk_boost=0, reasoning="")

    # The response narrated user denial. Verify against the audit.
    # Pull the MOST-RECENT failing tool result's raw error string —
    # that's what the classifier operates on. We iterate from the
    # END of ``tool_results`` so a turn with multiple escalations
    # picks the one the response most likely refers to (codex P1
    # round 1 — front-iteration would let an earlier audit-backed
    # denial corroborate a later unbacked sandbox refusal). When no
    # tool failures are visible, fall back to empty + ``unknown``;
    # the classifier then defaults to UNCONFIRMED, still NOT a user
    # denial, still trips the violation.
    raw_error = ""
    selected_tool_name = tool_name
    selected_feature_name = feature_name
    if tool_results:
        for tr in reversed(tool_results):
            result = tr.get("result")
            if not _result_indicates_failure(result):
                continue
            if isinstance(result, dict):
                err = result.get("error") or result.get("message") or ""
                if isinstance(err, str):
                    raw_error = err
            # If the caller didn't pin a tool/feature scope, derive
            # them from the selected failing result so the classifier
            # only matches audit rows ACTUALLY about this tool —
            # otherwise any unrelated user_denied row in the last 50
            # would falsely corroborate the narration (codex P1
            # round 1).
            if not selected_tool_name:
                selected_tool_name = str(tr.get("name") or "")
            if not selected_feature_name:
                selected_feature_name = str(tr.get("feature") or "")
            break

    # Lazy import to avoid a circular dep between this honesty-layer
    # module and the llm package.
    from kestrel_sovereign.llm.escalation_classifier import (
        EscalationOutcome,
        classify_escalation_failure,
    )
    # codex P2 round 2: when we could not derive a tool scope from
    # the response's failing results AND the caller did not pin one,
    # the audit-row lookup must be DISABLED — otherwise any unrelated
    # ``user_denied`` row in the last 50 silently corroborates this
    # narration. Force the classifier through its raw-error /
    # default path by passing an empty audit set.
    scoped_audit: list = (
        list(recent_decisions or []) if selected_tool_name else []
    )
    decision = classify_escalation_failure(
        raw_error,
        recent_decisions=scoped_audit,
        tool_name=selected_tool_name,
        feature_name=selected_feature_name,
    )
    if decision.outcome is EscalationOutcome.USER_DENIED:
        # Audit confirms the wording. Narration is honest.
        return NarrationVerdict(risk_boost=0, reasoning="")

    offending_phrase = match.group(0)
    return NarrationVerdict(
        risk_boost=2,
        reasoning=(
            f"Response narrated user denial ({offending_phrase!r}) but "
            f"the security audit does not corroborate it: "
            f"classifier returned {decision.outcome.value!r} "
            f"({decision.reason}). This violates the #1540 honesty "
            f"contract — only an audit-backed USER_DENIED outcome may "
            f"be narrated as a user denial."
        ),
        offending_verb=offending_phrase.lower(),
        offending_tool="",
    )
