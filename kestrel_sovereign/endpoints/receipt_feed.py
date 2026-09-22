"""Shared wire vocabulary for the two sovereign receipt-history doors (#3159).

Stop receipts and Hold receipts answer different questions — one is an event,
the other is the history of a state — but an observability consumer pages
through both the same way, so the cursor grammar, the page bounds, the time
window parsing, and the schema version live here once rather than twice.

Two rules this module exists to keep honest:

* **Order is a property of the read model.** Both feeds page on ``feed_seq``,
  a dense integer each store allocates under the one lock every writer holds
  until commit, so it is total over immutable rows AND increasing in commit
  order. Late or out-of-order telemetry therefore converges: any fetch order
  yields the same set in the same order, and a row that commits after a page
  was served can never land behind the cursor that page issued. ``occurred_at``
  is the database clock's reading, for display only — two transactions can
  read the clock in one order and commit in the other.
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
from kestrel_sovereign.storage.database_clock import LATEST_TIMESTAMP_BOUND

# Bumped when a payload's SHAPE changes. Consumers pin it; a feed that adds a
# field additively keeps the version, a feed that removes or re-means one does
# not.
RECEIPT_FEED_SCHEMA_VERSION = 1

DEFAULT_RECEIPT_PAGE_SIZE = 50
# A hard cap, not a default: one request must never be able to ask a sovereign
# host to materialize its whole receipt history.
MAX_RECEIPT_PAGE_SIZE = 200

# A cursor is the decimal ``feed_seq`` of the last row served, bounded by the
# largest signed 64-bit integer either backend can compare it against.
MAX_RECEIPT_CURSOR_LENGTH = 19
_MAX_FEED_SEQUENCE = 2**63 - 1
MAX_TIME_BOUND_LENGTH = 64

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
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_TIME_BOUND_LENGTH
    ):
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
        # Normalizing an offset can itself leave the calendar: year 9999 at
        # -01:00 is already past ``datetime.max`` in UTC.
        instant = (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )
    except (ValueError, OverflowError) as error:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_window_invalid",
            message=f"{field} must be an ISO-8601 timestamp.",
        ) from error
    # A bound the coarsest receipt store cannot render at its precision is a
    # caller error, and is refused here rather than escaping a store as an
    # ``OverflowError`` and a sanitized 500.
    if instant > LATEST_TIMESTAMP_BOUND:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_window_invalid",
            message=(
                f"{field} must not be later than "
                f"{LATEST_TIMESTAMP_BOUND.isoformat()}."
            ),
        )
    return instant


def parse_time_window(
    since: Optional[str], until: Optional[str]
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse the ``[since, until)`` window, refusing an empty or inverted one."""

    window_start = parse_time_bound(since, "since")
    window_end = parse_time_bound(until, "until")
    if (
        window_start is not None
        and window_end is not None
        and window_end <= window_start
    ):
        raise ApiHTTPException(
            status_code=400,
            code="receipt_window_invalid",
            message="until must be later than since.",
        )
    return window_start, window_end


def resolve_page_size(limit: Optional[str]) -> int:
    """Read a caller's raw page size, refusing anything past the hard cap.

    Refused rather than clamped: a client asking for more than one page may
    hold is told so, instead of silently receiving fewer rows than it believes
    it asked for.
    """

    if limit is None:
        return DEFAULT_RECEIPT_PAGE_SIZE
    text = limit.strip() if isinstance(limit, str) else ""
    if not text.isascii() or not text.isdigit():
        raise ApiHTTPException(
            status_code=400,
            code="receipt_page_invalid",
            message="limit must be a positive integer.",
        )
    # Measure before converting: ``int()`` refuses strings past the
    # interpreter's digit limit with a ValueError, so an oversized but
    # well-formed number would otherwise escape as a 500. Leading zeros carry
    # no magnitude and are dropped first so ``050`` still means fifty.
    digits = text.lstrip("0")
    if len(digits) > len(str(MAX_RECEIPT_PAGE_SIZE)):
        size = MAX_RECEIPT_PAGE_SIZE + 1
    else:
        size = int(digits or "0")
    if size < 1 or size > MAX_RECEIPT_PAGE_SIZE:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_page_invalid",
            message=(
                f"limit must be between 1 and {MAX_RECEIPT_PAGE_SIZE}."
            ),
        )
    return size


def bounded_filter_text(
    value: Optional[str],
    field: str,
    *,
    max_length: int,
) -> Optional[str]:
    """Validate one raw identity filter after the gate, verbatim.

    The value is compared exactly against stored identities, so it is not
    normalized — only refused when it cannot name anything.
    """

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ApiHTTPException(
            status_code=400,
            code="receipt_filter_invalid",
            message=(
                f"{field} must be non-empty text no longer than "
                f"{max_length} characters."
            ),
        )
    return value


def encode_cursor(key: Optional[int]) -> Optional[str]:
    """Render one feed position as the opaque token a client echoes back."""

    if key is None:
        return None
    return str(key)


def decode_cursor(value: Optional[str]) -> Optional[int]:
    """Read back a cursor this module produced, refusing anything else."""

    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
        or len(value) > MAX_RECEIPT_CURSOR_LENGTH
        or not 1 <= int(value) <= _MAX_FEED_SEQUENCE
    ):
        raise ApiHTTPException(
            status_code=400,
            code="receipt_cursor_invalid",
            message="cursor is not a cursor this feed issued.",
        )
    return int(value)


__all__ = [
    "DEFAULT_RECEIPT_PAGE_SIZE",
    "MAX_RECEIPT_CURSOR_LENGTH",
    "MAX_RECEIPT_PAGE_SIZE",
    "RECEIPT_FEED_SCHEMA_VERSION",
    "bounded_filter_text",
    "decode_cursor",
    "disclosable_identity",
    "encode_cursor",
    "parse_time_bound",
    "parse_time_window",
    "resolve_page_size",
]
