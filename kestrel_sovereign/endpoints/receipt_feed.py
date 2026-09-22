"""Shared wire vocabulary for the two sovereign receipt-history doors (#3159).

Stop receipts and Hold receipts answer different questions — one is an event,
the other is the history of a state — but an observability consumer pages
through both the same way, so the cursor grammar, the page bounds, the time
window parsing, and the schema version live here once rather than twice.

Two rules this module exists to keep honest:

* **Order is a property of the read model.** Both feeds are ordered by the
  database clock's ``(occurred_at, receipt_id)``, which is total over immutable
  rows, so late or out-of-order telemetry converges: any fetch order yields the
  same set in the same order.
* **A blinded identity is not an identity.** Stop stores caller-supplied
  request ids and turn handles as unkeyed digests. Rendering one as though it
  were a DID would invent an identity the row never recorded, so
  :func:`disclosable_identity` reports it absent instead.

Every filter arrives as raw text and is validated by the functions here, INSIDE
the handler and only after its sovereign gate has passed. Declaring the bounds
as ``Query(ge=..., min_length=...)`` annotations instead looks equivalent and
is not: FastAPI resolves a path operation's own parameters *before* it calls
the function, so an unauthorized request carrying ``limit=201`` was answered
422 — telling a caller the gate rejects that its request was well-formed enough
to be considered. Authority is not a validation result, and it goes first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from kestrel_sovereign.api_errors import ApiHTTPException

# Bumped when a payload's SHAPE changes. Consumers pin it; a feed that adds a
# field additively keeps the version, a feed that removes or re-means one does
# not.
RECEIPT_FEED_SCHEMA_VERSION = 1

DEFAULT_RECEIPT_PAGE_SIZE = 50
# A hard cap, not a default: one request must never be able to ask a sovereign
# host to materialize its whole receipt history.
MAX_RECEIPT_PAGE_SIZE = 200

MAX_RECEIPT_CURSOR_LENGTH = 512
# Neither half can contain it: an ``occurred_at`` is an ISO-8601 timestamp and
# a ``receipt_id`` is a UUID, so the split is unambiguous.
_CURSOR_SEPARATOR = "|"

_OPAQUE_DIGEST_PREFIX = "sha256:"
_OPAQUE_DIGEST_LENGTH = len(_OPAQUE_DIGEST_PREFIX) + 64


def disclosable_identity(value: object) -> Optional[str]:
    """Return a recorded identity, or ``None`` when only a digest was stored.

    Stop blinds the caller-supplied address it was asked to cancel. That digest
    is a durable lookup key, not an agent identity, and a reader that renders
    it as one is reading an identity the receipt never recorded.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    if len(value) == _OPAQUE_DIGEST_LENGTH and value.startswith(
        _OPAQUE_DIGEST_PREFIX
    ):
        suffix = value[len(_OPAQUE_DIGEST_PREFIX):]
        if all(character in "0123456789abcdef" for character in suffix):
            return None
    return value


def parse_time_bound(value: Optional[str], field: str) -> Optional[datetime]:
    """Parse one ISO-8601 window bound into an aware UTC instant."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ApiHTTPException(
            status_code=400,
            code="receipt_window_invalid",
            message=f"{field} must be an ISO-8601 timestamp.",
        )
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_window_invalid",
            message=f"{field} must be an ISO-8601 timestamp.",
        ) from error
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def resolve_page_size(limit: Optional[int]) -> int:
    """Clamp a caller's page size to the hard cap, refusing nonsense."""

    if limit is None:
        return DEFAULT_RECEIPT_PAGE_SIZE
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_page_invalid",
            message="limit must be a positive integer.",
        )
    return min(limit, MAX_RECEIPT_PAGE_SIZE)


def encode_cursor(key: Optional[tuple[str, str]]) -> Optional[str]:
    """Render one keyset position as the opaque token a client echoes back."""

    if key is None:
        return None
    occurred_at, receipt_id = key
    return f"{occurred_at}{_CURSOR_SEPARATOR}{receipt_id}"


def decode_cursor(value: Optional[str]) -> Optional[tuple[str, str]]:
    """Read back a cursor this module produced, refusing anything else."""

    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_RECEIPT_CURSOR_LENGTH
    ):
        raise ApiHTTPException(
            status_code=400,
            code="receipt_cursor_invalid",
            message="cursor is not a cursor this feed issued.",
        )
    occurred_at, separator, receipt_id = value.partition(_CURSOR_SEPARATOR)
    if not separator or not occurred_at.strip() or not receipt_id.strip():
        raise ApiHTTPException(
            status_code=400,
            code="receipt_cursor_invalid",
            message="cursor is not a cursor this feed issued.",
        )
    return occurred_at, receipt_id


__all__ = [
    "DEFAULT_RECEIPT_PAGE_SIZE",
    "MAX_RECEIPT_CURSOR_LENGTH",
    "MAX_RECEIPT_PAGE_SIZE",
    "RECEIPT_FEED_SCHEMA_VERSION",
    "decode_cursor",
    "disclosable_identity",
    "encode_cursor",
    "parse_time_bound",
    "resolve_page_size",
]
