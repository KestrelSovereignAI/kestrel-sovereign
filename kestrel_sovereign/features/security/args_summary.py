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
#: Shared by the searchable projection (permissions.fold_stored_summary) and
#: the displayed one (remask_summary) so the two cannot disagree about which
#: unparseable rows are safe.
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

SENSITIVE_JSON_KEY = re.compile(
    r'"[^"]*(?:%s)[^"]*"\s*:' % "|".join(
        re.escape(sub) for sub in SENSITIVE_KEY_SUBSTRINGS
    ),
    re.IGNORECASE,
)

#: Placeholder written in place of a sensitive value.
MASK = "***MASKED***"


def mask_sensitive(data: Any) -> Any:
    """Recursively mask values whose key looks sensitive.

    Dicts and lists are walked recursively; a key matching any
    :data:`SENSITIVE_KEY_SUBSTRINGS` substring has its value replaced with
    :data:`MASK` regardless of the value's type (so a nested secret dict is
    masked wholesale rather than descended into). Scalars pass through
    unchanged.
    """
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


def remask_summary(summary: Optional[str], max_length: int = 500) -> Optional[str]:
    """Re-apply masking to an ALREADY-PERSISTED summary, at read time (#3107).

    ``summarize_args`` masks on the way in, which makes the stored value safe
    only for rows written by a code path that had that masking. It says nothing
    about rows written by an older version — ``ApprovalQueue`` kept its own
    UNMASKED copy until F252 — or by any writer a reader has not inspected.

    A read-back tool cannot verify the provenance of 86,000 historical rows, so
    it should not depend on it. Masking again here moves the guarantee from the
    write path (many, historical, unknowable) to the read path (one, current,
    ours), which is the only place it can be made to hold for every row.

    Cheap for the common case: a row already masked re-masks to itself.

    A summary too truncated to parse cannot be re-masked field-by-field. It is
    shown raw only when it names no sensitive key in key position — the same
    rule the searchable projection applies (``fold_stored_summary``) — and is
    withheld otherwise. ``summarize_args`` truncates at 500 chars mid-structure,
    so on the live corpus ~30% of rows with arguments are unparseable, the
    long-issue-body filings that motivated this tool among them; withholding
    all of them degraded the read-back to what ``security_audit`` already gave.
    A value cannot appear without its key, so a key-position check on the
    truncated text is exactly as strong as field-by-field masking would be.
    """
    if not summary:
        return summary
    try:
        parsed = json.loads(summary)
    except (ValueError, TypeError):
        if SENSITIVE_JSON_KEY.search(summary):
            return "(summary truncated and names a sensitive key; not shown)"
        return summary[:max_length]
    if not isinstance(parsed, (dict, list)):
        return summary[:max_length]
    try:
        return json.dumps(mask_sensitive(parsed), default=str)[:max_length]
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
