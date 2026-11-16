"""Rate-limit response headers, copied off a provider response (#4114).

Nothing in Kestrel knew how much of a subscription plan's allowance was left
until the provider refused a call (#4097). The provider reports it, if at all,
in response headers, and Kestrel kept none of them. This module decides what
of a response's headers is kept. It records; nothing reads what it records.

What is kept, per ruling rv-4097-1 as amended by rv-4097-1a:

* A header's value is kept only if its name, compared without case, contains
  ``ratelimit``, ``rate-limit`` or ``rate_limit``, or is ``retry-after`` or
  ``retry-after-ms``. No other header name is assumed: which rate-limit
  headers a provider sends is not established.
* A matched value is kept only if it is at most 64 characters and is a
  number, an RFC 3339 date-time or a duration. Any other matched value is
  replaced by a descriptor of its shape, so an identifier in a header that
  happens to match is never written to a log or a durable row.
* At most 32 matched headers per response.
* Every other response header keeps its name only. Request headers are never
  read.

The copy is plain strings: nothing here holds the response, and nothing here
awaits.
"""

from __future__ import annotations

import json
import re
import string
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

#: The key the copied headers travel under, on the ``llm.usage`` line and in
#: the durable ``llm_calls`` row's metadata. Absent when nothing matched.
RATE_LIMIT_HEADERS_KEY = "rate_limit_headers"

#: The attribute a response carries its copy on, from adapter to service.
_RESPONSE_ATTR = "_kestrel_rate_limit_headers"

#: At most this many matched headers are kept from one response.
MAX_MATCHED_HEADERS = 32

#: A matched value longer than this is never kept, whatever its shape.
MAX_KEPT_VALUE_CHARS = 64

#: The two header names matched exactly. Every other match is by fragment.
_EXACT_NAMES = frozenset({"retry-after", "retry-after-ms"})
_NAME_FRAGMENTS = ("ratelimit", "rate-limit", "rate_limit")

_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_NUMBER_RE = re.compile(_NUMBER)
_DURATION_RE = re.compile(rf"(?:{_NUMBER}(?:ms|s|m|h))+")
_DATE_TIME_RE = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.[0-9]+)?(?:[Zz]|[+-]([0-9]{2}):([0-9]{2}))"
)
_ASCII_SPACE = frozenset(" \t\n\r\v\f")
_ASCII_PUNCTUATION = frozenset(string.punctuation)

#: A copied value: the value as received, or the descriptor of one withheld.
HeaderValue = Union[str, Dict[str, Any]]


def is_rate_limit_header(name: str) -> bool:
    """Whether ``name`` is a header whose value may be kept."""
    folded = name.lower()
    return folded in _EXACT_NAMES or any(
        fragment in folded for fragment in _NAME_FRAGMENTS
    )


def _is_rfc3339_date_time(value: str) -> bool:
    match = _DATE_TIME_RE.fullmatch(value)
    if match is None:
        return False
    year, month, day, hour, minute, second = (int(part) for part in match.groups()[:6])
    offset_hour, offset_minute = match.groups()[6:]
    if second > 60 or (offset_hour is not None and (
        int(offset_hour) > 23 or int(offset_minute) > 59
    )):
        return False
    try:
        # RFC 3339 admits a leap second (60), which datetime does not.
        datetime(year, month, day, hour, minute, min(second, 59))
    except ValueError:
        return False
    return True


def has_kept_shape(value: str) -> bool:
    """Whether a matched header's value is short and one of the kept shapes:
    a number, an RFC 3339 date-time, or a duration in ``ms``, ``s``, ``m`` or
    ``h``. The shapes are a guess; none is established for any provider."""
    if len(value) > MAX_KEPT_VALUE_CHARS:
        return False
    return bool(
        _NUMBER_RE.fullmatch(value)
        or _DURATION_RE.fullmatch(value)
        or _is_rfc3339_date_time(value)
    )


def withheld_value_descriptor(value: str) -> Dict[str, Any]:
    """What is recorded in place of a matched value that is not kept: its
    length, which character classes it contains, and the distinct ASCII
    punctuation in it. Nothing else of the value."""
    classes = {
        "digit": any("0" <= char <= "9" for char in value),
        "lower": any("a" <= char <= "z" for char in value),
        "upper": any("A" <= char <= "Z" for char in value),
        "space": any(char in _ASCII_SPACE for char in value),
        "non-ascii": any(ord(char) > 127 for char in value),
    }
    return {
        "withheld": True,
        "length": len(value),
        "classes": [name for name, present in classes.items() if present],
        "punctuation": "".join(sorted({c for c in value if c in _ASCII_PUNCTUATION})),
    }


def _recorded_value(value: str) -> HeaderValue:
    return value if has_kept_shape(value) else withheld_value_descriptor(value)


def observation_time() -> str:
    """Now, in UTC, as ISO 8601 with its offset."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResponseHeaders:
    """What one response's headers left behind: plain strings only.

    ``names`` is every response header's name, lowercased and sorted.
    ``matched`` holds the matched headers in the order received, each with its
    value as received or the descriptor of a withheld one. ``omitted`` counts
    the matched headers past :data:`MAX_MATCHED_HEADERS`.
    """

    observed_at: str
    names: Tuple[str, ...]
    matched: Tuple[Tuple[str, HeaderValue], ...]
    omitted: int = 0

    def record(self) -> Optional[Dict[str, Any]]:
        """The JSON shape recorded, or ``None`` when nothing matched."""
        if not self.matched:
            return None
        recorded: Dict[str, Any] = {
            "observed_at": self.observed_at,
            "headers": [[name, _copy_value(value)] for name, value in self.matched],
        }
        if self.omitted:
            recorded["omitted_headers"] = self.omitted
        return recorded


def _copy_value(value: HeaderValue) -> HeaderValue:
    return dict(value, classes=list(value["classes"])) if isinstance(value, dict) else value


def _header_pairs(headers: Any) -> List[Tuple[str, str]]:
    multi_items = getattr(headers, "multi_items", None)
    pairs: Iterable[Any] = multi_items() if callable(multi_items) else headers.items()
    copied: List[Tuple[str, str]] = []
    for name, value in pairs:
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError(
                f"response header items must be strings, not "
                f"{type(name).__name__}/{type(value).__name__}"
            )
        copied.append((name.lower(), value))
    return copied


def read_response_headers(response: Any) -> Optional[ResponseHeaders]:
    """Copy what may be kept of ``response``'s headers, or ``None`` when
    there is no response or it has no headers.

    The time of observation is taken here, when the headers are copied, not
    when a record of them is later written. The result holds no reference to
    ``response``.
    """
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    pairs = _header_pairs(headers)
    observed_at = observation_time()
    matched: List[Tuple[str, HeaderValue]] = []
    omitted = 0
    for name, value in pairs:
        if not is_rate_limit_header(name):
            continue
        if len(matched) < MAX_MATCHED_HEADERS:
            matched.append((name, _recorded_value(value)))
        else:
            omitted += 1
    return ResponseHeaders(
        observed_at=observed_at,
        names=tuple(sorted({name for name, _ in pairs})),
        matched=tuple(matched),
        omitted=omitted,
    )


class HeaderNameLedger:
    """The set of response header names each route last reported, for one
    process, so the names-only line is written when a route's set is first
    seen and again only when it changes."""

    def __init__(self) -> None:
        self._names: Dict[Optional[str], Tuple[str, ...]] = {}
        self._lock = threading.Lock()

    def changed(self, route: Optional[str], names: Tuple[str, ...]) -> bool:
        """Remember ``names`` for ``route``; whether they differ from the last."""
        if not names:
            return False
        with self._lock:
            if self._names.get(route) == names:
                return False
            self._names[route] = names
            return True


#: The process's ledger. Per process by design: the names-only line is
#: written once per route per process, not once per agent.
RESPONSE_HEADER_NAMES = HeaderNameLedger()


def names_line(route: Optional[str], names: Tuple[str, ...]) -> str:
    """The body of the names-only line."""
    return json.dumps({"route": route, "names": list(names)})


def attempt_line(
    route: Optional[str], captured: ResponseHeaders, status: Optional[int],
) -> Optional[str]:
    """The body of the line recording a failed open's matched headers, or
    ``None`` when nothing matched. It carries no part of the error."""
    recorded = captured.record()
    if recorded is None:
        return None
    line: Dict[str, Any] = {"route": route, "observed_at": recorded.pop("observed_at")}
    if status is not None:
        line["status"] = status
    line.update(recorded)
    return json.dumps(line)


def attach_rate_limit_headers(response: Any, captured: Optional[ResponseHeaders]) -> Any:
    """Carry ``captured`` on ``response`` to the service, and return it. A
    copy with nothing matched is not carried."""
    if captured is not None and captured.matched:
        setattr(response, _RESPONSE_ATTR, captured)
    return response


def response_rate_limit_headers(response: Any) -> Optional[Dict[str, Any]]:
    """The record of the headers an adapter attached to ``response``, if any."""
    captured = getattr(response, _RESPONSE_ATTR, None)
    return captured.record() if isinstance(captured, ResponseHeaders) else None


__all__ = [
    "HeaderNameLedger",
    "MAX_KEPT_VALUE_CHARS",
    "MAX_MATCHED_HEADERS",
    "RATE_LIMIT_HEADERS_KEY",
    "RESPONSE_HEADER_NAMES",
    "ResponseHeaders",
    "attach_rate_limit_headers",
    "attempt_line",
    "has_kept_shape",
    "is_rate_limit_header",
    "names_line",
    "observation_time",
    "read_response_headers",
    "response_rate_limit_headers",
    "withheld_value_descriptor",
]
