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
#: Structure the repair could not parse but that may carry a value: a bracket
#: with a quote or a colon INSIDE it (before any closing bracket). A JSON key
#: position is one such shape; a single-quoted dict or a bare array of
#: strings is another, and a guard keyed on the double-quoted key alone let
#: those through raw (round 18 review). Prose that merely brackets a word
#: (``[wallet] | args=``, ``argument 'files[0]' exceeds``) closes its bracket
#: before any quote or colon and stays — scanning past the closing bracket
#: withheld intact guardrail rows from both read paths (round 24 review).
_UNREPAIRED_STRUCTURE = re.compile(r"[{\[][^{\[\]}]*[\"':]")
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


def repair_json_text(text: Any, *, max_trim: int = _MAX_REPAIR_TRIM) -> tuple[Any, bool]:
    """Parse JSON text that a truncation cut: ``(parsed, altered)``.

    ``altered`` is True when the parse needed the marker stripped, characters
    trimmed or a structure closed — that is, when the result is a
    reconstruction rather than the text as written, which the display marks
    (round 24 review: an intact ``<reason> | args={...}`` row reaches this
    path only because of its prose prefix and must not be marked).

    Tries the text as it is, then with its open string and containers
    closed, dropping up to ``max_trim`` trailing characters so a cut that
    landed inside an escape sequence, a number or a bare literal still
    reaches a parseable point. Only a dict or list counts: the shape is what
    :func:`mask_sensitive` walks. ``max_trim`` is the READ path's slack for a
    real cut; the write path, where nothing has been cut, passes 0.
    """
    if not isinstance(text, str):
        return None, False
    body = text.rstrip()
    marked = body.endswith(_TRUNCATION_MARK)
    if marked:
        body = body[: -len(_TRUNCATION_MARK)]
    for trim in range(max_trim + 1):
        candidate = body[: len(body) - trim] if trim else body
        if not candidate:
            return None, False
        closed = _close_open_structures(candidate)
        if closed is None:
            return None, False
        try:
            parsed = json.loads(closed)
        except (ValueError, RecursionError):
            # RecursionError is a RuntimeError, not a ValueError: a deeply
            # nested value escaped every guard, denied the call on the write
            # path and poisoned the query on the read path (round 23 review).
            continue
        if not isinstance(parsed, (dict, list)):
            return None, False
        return parsed, marked or trim > 0 or closed != candidate
    return None, False


def repair_unparseable_summary(text: str) -> tuple[str, Any, bool]:
    """Split a stored summary that will not parse into its prose prefix and
    the JSON it embeds, repaired and MASKED.

    ``summarize_args`` output is JSON cut mid-structure; ``tool_audit`` writes
    ``"<reason> | args=<json>"``. Either way the JSON is repaired by closing
    what the cut left open and masked STRUCTURALLY by :func:`mask_sensitive`
    — the one masker every parseable row already gets — so there is a single
    masking rule and no text scanner to get a value shape wrong (rounds
    13–16 each found one: a container, an escaped string, prose).

    Returns ``(text, None, False)`` for prose that embeds no JSON at all
    (nothing with a key position, so nothing to mask) and ``(prefix, masked,
    altered)`` when JSON parsed — ``altered`` says whether it had to be
    reconstructed. Structure the repair could not parse and that carries a
    value is never handed back: it is replaced by the ``(prefix withheld …)``
    placeholder, in front of a repaired region or as the whole row, so the
    row stays shown and searchable minus exactly that text. Nothing returns
    None (round 31 review: the old withhold-the-row branch was dead).
    """
    starts = sorted({m.start() for m in re.finditer(r"[{\[]", text)})[:8]
    if not starts:
        return text, None, False
    for start in starts:
        prefix = text[:start]
        if _UNREPAIRED_STRUCTURE.search(prefix):
            # Text before this candidate carries structure the repair did not
            # parse — an earlier region that did not repair, in any quoting.
            # It must not be handed back RAW, but the ROW is not discarded
            # for it: tool_audit interpolates the caller's own tool name and
            # argument keys into the reason, so a bracket the model chose
            # was blanking its own refusal record from both read paths
            # (round 27 review). The prefix is withheld; the masked args and
            # the fact of the refusal survive.
            prefix = "(prefix withheld: unmaskable structure) "
        parsed, altered = repair_json_text(text[start:])
        if parsed is not None:
            nested_repairs: list = []
            masked = mask_sensitive(parsed, repair_slack=_MAX_REPAIR_TRIM, reconstructed=nested_repairs)
            return prefix, masked, altered or bool(nested_repairs)
    # No region repaired. A bracket is not JSON: a prose-only row such as
    # "Tool 'files[0]' is not in the known tool allowlist" (tool_audit with
    # no args) has nothing to mask and stays; only structure the repair
    # could not parse AND that carries a value is withheld — by the same
    # rule the prefix uses (round 29 review).
    if _UNREPAIRED_STRUCTURE.search(text):
        return "(prefix withheld: unmaskable structure) ", None, False
    return text, None, False


#: Placeholder written in place of a sensitive value.
MASK = "***MASKED***"


def mask_sensitive(
    data: Any, *, repair_slack: int = 0, reconstructed: Optional[list] = None
) -> Any:
    """Recursively mask values whose key looks sensitive.

    Dicts and lists are walked recursively; a key matching any
    :data:`SENSITIVE_KEY_SUBSTRINGS` substring has its value replaced with
    :data:`MASK` regardless of the value's type (so a nested secret dict is
    masked wholesale rather than descended into). A string value that IS
    JSON — a JSON-encoded payload carried as a string
    (``{"payload": "{\\"api_key\\": ...}"}``) has no dict key for this walk
    to see — is parsed, masked the same way and re-serialized, and only when
    masking changed something; text after the JSON (``{...} yes``) is kept.
    ``repair_slack`` is the READ path's allowance for a cut nested payload
    (``repair_unparseable_summary`` passes it); on the WRITE path nothing has
    been cut, so it stays 0 and a prose tail is never trimmed away (round 19
    review). Every other string is prose and passes through byte-for-byte:
    a text scanner applied to an issue body that merely quoted
    ``"api_key":`` — or that began with a markdown ``[link]`` — ate the rest
    of the body from the audit row, permanently (rounds 15 and 16). Whether
    a string is JSON is decided by parsing it, never by its first character.
    """
    if isinstance(data, str):
        stripped = data.lstrip()
        if stripped[:1] not in "{[":
            return data
        altered = False
        try:
            nested, end = _JSON_PREFIX.raw_decode(stripped)
        except (ValueError, RecursionError):
            nested, altered = repair_json_text(stripped, max_trim=repair_slack) if repair_slack else (None, False)
            end = len(stripped)
            if nested is not None and altered and reconstructed is not None:
                # A nested payload the READ path reconstructed: the display
                # must mark it exactly as it marks a top-level reconstruction
                # (round 25 review) — the caller's flag carries the fact out.
                reconstructed.append(True)
        if not isinstance(nested, (dict, list)):
            return data
        try:
            masked = mask_sensitive(nested, repair_slack=repair_slack, reconstructed=reconstructed)
            if masked == nested:
                # Nothing to mask: the value stays as written (a cut one is
                # still marked, through the flag set above).
                return data
            replaced = json.dumps(masked, default=str)
        except RecursionError:
            # The decoder accepts a payload nested deeper than this walk (or
            # the encoder) can follow. It cannot be masked, so it is not
            # shown; raising here denied the tool call on the write path
            # and poisoned the query on the read path (round 23 review).
            return "(payload nested past the limit; not shown)"
        return data[: len(data) - len(stripped)] + replaced + stripped[end:]
    if isinstance(data, dict):
        result: dict = {}
        for key, value in data.items():
            if any(s in str(key).lower() for s in SENSITIVE_KEY_SUBSTRINGS):
                result[key] = MASK
            else:
                result[key] = mask_sensitive(value, repair_slack=repair_slack, reconstructed=reconstructed)
        return result
    if isinstance(data, list):
        return [mask_sensitive(item, repair_slack=repair_slack, reconstructed=reconstructed) for item in data]
    return data


#: Parses a JSON value at the START of a string and reports where it ends,
#: so prose after it is kept rather than trimmed away.
_JSON_PREFIX = json.JSONDecoder()


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
    if not isinstance(summary, str):
        # The column has TEXT affinity but stores bytes as bytes (and an int
        # as an int): the fold path already refuses these, and this door —
        # on BOTH store read paths, the operator endpoint included — handed
        # them to a string regex and raised for every caller until the row
        # aged out (round 22 review). One rule at both doors.
        return None if summary is None else "(summary not text; not shown)"
    if not summary:
        return summary
    try:
        return _remask_text(summary)
    except RecursionError:
        # A row nested past the interpreter's limit cannot be walked, so it
        # cannot be masked; withheld, never shown raw or raised.
        return "(summary could not be re-masked; not shown)"


def _remask_text(summary: str) -> str:
    try:
        parsed = json.loads(summary)
    except (ValueError, TypeError):
        # Same rule as the searchable projection: repair the cut JSON, mask
        # it structurally, keep the rest. A row whose JSON cannot be repaired
        # cannot be masked, and is withheld rather than shown raw. A repaired
        # row is shown WITH the truncation marker: the repair closed what the
        # cut left open and rendered a cut field as null, and a reader must
        # not take that reconstruction for the record as written (round 23
        # review).
        prefix, masked, altered = repair_unparseable_summary(summary)
        if masked is None:
            return prefix  # the text as written, or the withheld placeholder
        shown = prefix + json.dumps(masked, default=str)
        # Only a RECONSTRUCTED row is marked: an intact tool_audit row reaches
        # this path for its prose prefix alone, and marking it made a plain
        # record indistinguishable from a cut one (round 24 review).
        return shown + _TRUNCATION_MARK if altered else shown
    if isinstance(parsed, str):
        # The whole row is a JSON string — the payload the walker masks one
        # level down, carried at the top (round 21 review).
        nested_repairs: list = []
        masked = mask_sensitive(parsed, repair_slack=_MAX_REPAIR_TRIM, reconstructed=nested_repairs)
        if masked == parsed:
            # Nothing masked — but a reconstructed payload is still marked,
            # as the dict/list branch marks it (round 27 review).
            return summary + (_TRUNCATION_MARK if nested_repairs else "")
        return json.dumps(masked) + (_TRUNCATION_MARK if nested_repairs else "")
    if not isinstance(parsed, (dict, list)):
        return summary
    try:
        # READ path: a nested JSON-encoded payload can be cut inside a row
        # whose outer JSON parses, so the string branch gets the repair slack
        # here exactly as it does under repair_unparseable_summary — and a
        # nested payload it reconstructed is marked exactly as a top-level
        # reconstruction is (round 25 review).
        nested_repairs = []
        shown = json.dumps(mask_sensitive(parsed, repair_slack=_MAX_REPAIR_TRIM, reconstructed=nested_repairs), default=str)
        return shown + _TRUNCATION_MARK if nested_repairs else shown
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
    except (TypeError, ValueError, RecursionError):
        # RecursionError included: a body nested past the interpreter's
        # limit raised through the hook, which failed closed — the call was
        # wrongly DENIED and no audit row was written (round 23 review).
        return "(args could not be summarized)"
    if len(summary) > max_length:
        return summary[: max_length - 3] + "..."
    return summary
