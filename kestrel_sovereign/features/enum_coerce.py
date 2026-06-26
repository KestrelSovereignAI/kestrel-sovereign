"""Shared helpers for agent-facing ``@tool`` parameters validated against a
fixed set of values (priorities, statuses, modes, scopes, tiers, …).

Why this module exists (issue #1923): several tools accept a parameter that is
checked against a constant set, but the tool schema never tells the model what
the valid values are. The LLM then guesses a natural synonym (``medium`` for a
priority whose canonical middle value is ``normal``; ``completed`` for ``done``;
``in-progress`` for ``in_progress``), the validator hard-rejects it, and the
agent either wastes a round-trip or gives up. The fix is two-fold and lives
here so it is not re-implemented per feature:

1. :func:`normalize_choice` — case-fold a value and map a known synonym onto its
   canonical enum value *before* validation. Unknown values pass through
   (lower-cased) so genuine typos still hit the validator and get a helpful,
   value-listing error.
2. :func:`coerce_enum` — normalize **and** validate in one call, returning a
   standardized ``"<field> must be one of: a, b, c (got 'x')"`` error that
   always lists the valid values.

The canonical vocabulary differs by domain — todo priority's middle value is
``normal`` while ``strategic.severity``'s is ``medium``; ``restart`` statuses use
American ``canceled`` while ``todo`` uses British ``cancelled`` — so alias maps
stay **local to each feature**. Only the mechanism is centralized here. The
named alias maps below cover the synonyms that recur across features; import and
extend them, or pass a feature-specific dict.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

__all__ = [
    "normalize_choice",
    "coerce_enum",
    "STATUS_DONE_CANCELLED_ALIASES",
    "LOW_NORMAL_HIGH_URGENT_ALIASES",
]


def normalize_choice(value, aliases: Optional[Mapping[str, str]] = None):
    """Lower-case ``value`` and map a known synonym onto its canonical form.

    Unknown values pass through (stripped + lower-cased) so the caller's
    validator still rejects genuine typos with a helpful message. Non-strings
    (e.g. ``None``) pass through untouched so optional params keep working.
    """
    if not isinstance(value, str):
        return value
    key = value.strip().lower()
    if aliases is None:
        return key
    return aliases.get(key, key)


def coerce_enum(
    value,
    valid: Iterable[str],
    *,
    field: str,
    aliases: Optional[Mapping[str, str]] = None,
    default: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize ``value`` then validate it against ``valid``.

    Returns ``(normalized_value, None)`` on success, or ``(None, error)`` where
    ``error`` lists the valid values. ``None``/blank input falls back to
    ``default`` when one is given (and the default is returned without
    validation, since defaults are caller-controlled, not model-supplied).
    """
    valid_list = list(valid)
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default, None
    normalized = normalize_choice(value, aliases)
    if normalized in valid_list:
        return normalized, None
    shown = ", ".join(valid_list)
    return None, f"{field} must be one of: {shown} (got {value!r})"


# --- Reusable alias maps -------------------------------------------------
# Synonyms LLMs reliably reach for, grouped by the canonical vocabulary they
# target. Feature-specific maps should start from one of these (``{**MAP, ...}``)
# rather than redefining the common cases.

# For status fields whose canonical values are {..., done, cancelled} (British
# spelling). Maps the completion/cancellation synonyms models default to.
STATUS_DONE_CANCELLED_ALIASES: dict = {
    "complete": "done",
    "completed": "done",
    "finished": "done",
    "resolved": "done",
    "canceled": "cancelled",  # American -> British canonical
    "cancel": "cancelled",
    "abandoned": "cancelled",
    "dropped": "cancelled",
}

# For priority/urgency fields whose canonical scale is low/normal/high/urgent
# (middle value is ``normal``, not the universal ``medium``).
LOW_NORMAL_HIGH_URGENT_ALIASES: dict = {
    "medium": "normal",
    "med": "normal",
    "moderate": "normal",
    "critical": "urgent",
    "p0": "urgent",
    "p1": "high",
    "p2": "normal",
    "p3": "low",
}
