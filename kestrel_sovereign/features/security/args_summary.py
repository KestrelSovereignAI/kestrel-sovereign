"""Shared masking + summarization for security audit ``args_summary`` values.

Both :class:`SecurityHook` (the PreToolUse path) and :class:`ApprovalQueue`
(the decision-persistence path) write tool arguments into
``security_audit_log``. Those rows sit in plaintext SQLite and are returned
verbatim by ``/api/security/audit``, so a raw secret in the args (e.g.
``{"api_key": "sk-live-..."}``) would leak at rest and over the wire.

This module is the single source of truth for masking sensitive values and
truncating the summary. Keeping one helper means the two write paths cannot
drift — previously the queue had its own *unmasked* copy (F252).
"""

import json
import re
from typing import Any, Optional

#: Substrings (matched case-insensitively) that mark a key as sensitive.
#: A sensitive name in KEY POSITION — `"...key...":` — rather than anywhere in
#: the text. The value side is data and may legitimately contain these words.
#: :func:`mask_sensitive` is the one masker every row gets — parseable rows
#: directly, truncated rows after :func:`repair_unparseable_summary` closes
#: what the cut left open — in both the searchable projection
#: (permissions.fold_stored_summary) and the displayed one (remask_summary),
#: so the two cannot disagree about what is masked.
SENSITIVE_KEY_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "key",
    "api_key",
    "private_key",
    "credit_card",
    "ssn",
    "social_security",
)

#: What ``summarize_args`` appends when it cuts a row; stripped before repair.
_TRUNCATION_MARK = "..."
#: A JSON key position (``"...":``) — the one shape that can carry a secret.
_KEY_POSITION = re.compile(r'"\s*:')
#: Trailing characters repair may drop to reach a parseable cut point: a cut
#: inside a ``\\uXXXX`` escape needs up to six, a bare literal (``tru``) four.
_MAX_REPAIR_TRIM = 8


def _close_open_structures(text: str) -> Optional[str]:
    """Append what a cut left open so ``text`` parses, or None.

    Tracks strings (with escapes) and container nesting in one pass. An
    unterminated VALUE string is closed; an unterminated KEY string is
    dropped — there is no value to mask or show (a complete key the cut
    separated from its value reaches this same branch once the caller trims
    its closing quote). A trailing ``,`` is dropped and a trailing ``:`` gets
    ``null``; then every open container is closed in reverse. None when the
    text is not JSON-shaped or its brackets do not nest.
    """
    if text[:1] not in "{[":
        return None
    closers: list[str] = []
    in_string = False
    escape = False
    string_start = -1
    string_is_key = False
    previous = ""
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                previous = '"'
            continue
        if ch.isspace():
            continue
        if ch == '"':
            in_string = True
            string_start = i
            string_is_key = bool(closers) and closers[-1] == "}" and previous in "{,"
            continue
        if ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not closers or closers[-1] != ch:
                return None
            closers.pop()
        previous = ch
    if in_string:
        if string_is_key:
            text = text[:string_start]
        else:
            if escape:
                text = text[:-1]
            text += '"'
    tail = text.rstrip()
    if tail.endswith(","):
        tail = tail[:-1]
    elif tail.endswith(":"):
        tail += " null"
    return tail + "".join(reversed(closers))


def complete_truncated_json(text: Any) -> Any:
    """Parse JSON text that a truncation cut, or return None.

    Tries the text as it is, then with its open string and containers
    closed, dropping up to ``_MAX_REPAIR_TRIM`` trailing characters so a cut
    that landed inside an escape sequence, a number or a bare literal still
    reaches a parseable point. Only a dict or list counts: the shape is what
    :func:`mask_sensitive` walks.
    """
    if not isinstance(text, str):
        return None
    body = text.rstrip()
    if body.endswith(_TRUNCATION_MARK):
        body = body[: -len(_TRUNCATION_MARK)]
    for trim in range(_MAX_REPAIR_TRIM + 1):
        candidate = body[: len(body) - trim] if trim else body
        if not candidate:
            return None
        closed = _close_open_structures(candidate)
        if closed is None:
            return None
        try:
            parsed = json.loads(closed)
        except ValueError:
            continue
        return parsed if isinstance(parsed, (dict, list)) else None
    return None


def repair_unparseable_summary(text: str) -> Optional[tuple[str, Any]]:
    """Split a stored summary that will not parse into its prose prefix and
    the JSON it embeds, repaired and MASKED.

    ``summarize_args`` output is JSON cut mid-structure; ``tool_audit`` writes
    ``"<reason> | args=<json>"``. Either way the JSON is repaired by closing
    what the cut left open and masked STRUCTURALLY by :func:`mask_sensitive`
    — the one masker every parseable row already gets — so there is a single
    masking rule and no text scanner to get a value shape wrong (rounds
    13–16 each found one: a container, an escaped string, prose).

    Returns ``(text, None)`` for prose that embeds no JSON at all (nothing
    with a key position, so nothing to mask), ``(prefix, masked)`` when the
    JSON repaired, and None when JSON was found but cannot be repaired — that
    row cannot be masked, so it must not be shown or searched.
    """
    starts = sorted({m.start() for m in re.finditer(r"[{\[]", text)})[:8]
    if not starts:
        return text, None
    for start in starts:
        prefix = text[:start]
        if _KEY_POSITION.search(prefix):
            # Text before this candidate carries a key position — an earlier
            # JSON region that did not repair, or JSON-shaped prose. It would
            # be handed back RAW as the prefix, shown and searchable; the row
            # is withheld instead (round 17 review).
            return None
        parsed = complete_truncated_json(text[start:])
        if parsed is not None:
            return prefix, mask_sensitive(parsed)
    return None


#: Placeholder written in place of a sensitive value.
MASK = "***MASKED***"


def mask_sensitive(data: Any) -> Any:
    """Recursively mask values whose key looks sensitive.

    Dicts and lists are walked recursively; a key matching any
    :data:`SENSITIVE_KEY_SUBSTRINGS` substring has its value replaced with
    :data:`MASK` regardless of the value's type (so a nested secret dict is
    masked wholesale rather than descended into). A string value that IS
    JSON — a JSON-encoded payload carried as a string
    (``{"payload": "{\\"api_key\\": ...}"}``) has no dict key for this walk
    to see — is parsed (repaired first if a cut left it open), masked the
    same way and re-serialized, and only when masking changed something.
    Every other string is prose and passes through byte-for-byte: this runs
    on the WRITE path for every tool call, and a text scanner applied to an
    issue body that merely quoted ``"api_key":`` — or that began with a
    markdown ``[link]`` — ate the rest of the body from the audit row,
    permanently (rounds 15 and 16). Whether a string is JSON is decided by
    parsing it, never by its first character.
    """
    if isinstance(data, str):
        nested = complete_truncated_json(data.lstrip())
        if nested is None:
            return data
        masked = mask_sensitive(nested)
        return data if masked == nested else json.dumps(masked, default=str)
    if isinstance(data, dict):
        result: dict = {}
        for key, value in data.items():
            if any(s in str(key).lower() for s in SENSITIVE_KEY_SUBSTRINGS):
                result[key] = MASK
            else:
                result[key] = mask_sensitive(value)
        return result
    if isinstance(data, list):
        return [mask_sensitive(item) for item in data]
    return data


def remask_summary(summary: Optional[str]) -> Optional[str]:
    """Re-apply masking to an ALREADY-PERSISTED summary, at read time (#3107).

    Masking substitutes; it never removes. There is deliberately no cap here:
    the writer's cap is the only cap (core writers 500 characters, Talon's
    outcome rows 1,000, ``tool_audit`` unbounded), and a read-side cap of 500
    cut a matched Talon outcome down to a row that showed neither the phrase
    the match was made on nor the ``issue_number`` the caller asked for.

    ``summarize_args`` masks on the way in, which makes the stored value safe
    only for rows written by a code path that had that masking. It says nothing
    about rows written by an older version — ``ApprovalQueue`` kept its own
    UNMASKED copy until F252 — or by any writer a reader has not inspected.

    A read-back tool cannot verify the provenance of 86,000 historical rows, so
    it should not depend on it. Masking again here moves the guarantee from the
    write path (many, historical, unknowable) to the read path (one, current,
    ours), which is the only place it can be made to hold for every row.

    Cheap for the common case: a row already masked re-masks to itself.

    A summary too truncated to parse is repaired — the cut's open string and
    containers closed — and masked field-by-field like any other row, by
    :func:`repair_unparseable_summary`, the same helper the searchable
    projection applies (``fold_stored_summary``), so display and match are
    the same text; the rest of the row is shown.
    ``summarize_args`` truncates at 500 chars mid-structure, so on the live
    corpus ~30% of rows with arguments are unparseable, the long-issue-body
    filings that motivated this tool among them; withholding all of them
    degraded the read-back to what ``security_audit`` already gave. A value
    cannot appear without its key, and a repaired row is masked by key
    position exactly as a parseable one is — field by field, one masker.
    """
    if not summary:
        return summary
    try:
        parsed = json.loads(summary)
    except (ValueError, TypeError):
        # Same rule as the searchable projection: repair the cut JSON, mask
        # it structurally, keep the rest. A row whose JSON cannot be repaired
        # cannot be masked, and is withheld rather than shown raw.
        repaired = repair_unparseable_summary(summary)
        if repaired is None:
            return "(summary truncated past repair; not shown)"
        prefix, masked = repaired
        if masked is None:
            return summary
        return prefix + json.dumps(masked, default=str)
    if not isinstance(parsed, (dict, list)):
        return summary
    try:
        return json.dumps(mask_sensitive(parsed), default=str)
    except (TypeError, ValueError):
        return "(summary could not be re-masked; not shown)"


def summarize_args(args: Optional[dict], max_length: int = 500) -> Optional[str]:
    """Return a masked, truncated JSON summary of ``args`` for the audit log.

    Returns ``None`` for empty/falsey args. Sensitive values are masked
    *before* serialization, so even the failure/truncation paths never emit a
    raw secret. The result is capped at ``max_length`` characters.
    """
    if not args:
        return None
    try:
        masked = mask_sensitive(args)
        summary = json.dumps(masked, default=str)
    except (TypeError, ValueError):
        return "(args could not be summarized)"
    if len(summary) > max_length:
        return summary[: max_length - 3] + "..."
    return summary
