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
This is a heuristic. False positives are possible (e.g. the agent
writes "Saved your previous turn — now adding to it" before calling
the new tool, where "Saved" refers to past state). Hook authors who
care about precision should compose this with their own checks. The
module ships with the audit hook in WARN mode by default to keep
operational impact low.
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
        return result["success"] is False
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
