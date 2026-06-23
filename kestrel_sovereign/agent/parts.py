"""In-band typed-part emission for first-class console components (#1914).

A *part* is a typed, structured component the agent stream can render as its
own first-class message bubble in the sovereign console — a todo card, a
citation, an A2A artifact, a structured-data card — rather than as markdown
prose or a tool-activity card. Parts ride the SAME in-band sentinel pipeline as
TOOL/THINK/REVISE (:mod:`kestrel_sovereign.agent.streaming`), inheriting its
ordering against surrounding prose, its persistence, and its replay for free.

Features emit a part imperatively with :func:`emit_part`. The value is buffered
on a per-turn collector (a context var) that the orchestrator drains and yields
as a ``\\x1eKESTREL:PART:{json}\\x1e`` sentinel into the live stream at a
well-defined point (right after the tool that produced it). The streaming layer
then strips the sentinel from the persisted prose and records the part —
position-stamped — in the assistant message's ``parts`` metadata so a
conversation reload re-renders every component bubble, not just the live turn.

Security: ``data`` is host/agent-influenceable, so the wire format is the only
trust boundary this module owns. ``emit_part`` rejects non-printable/control
chars in the type, rejects non-JSON-serializable data, and enforces a hard size
cap so an oversized or malformed part can never blow the SSE frame or break
sentinel framing (``json.dumps`` escapes the ``\\x1e`` record separator). The
console renderers own the rest of sanitization for the values they paint.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
from typing import Any, List, Optional

# Wire format mirrors the TOOL/THINK/REVISE sentinels: an ASCII Record
# Separator (0x1e) bookends a ``KESTREL:PART:`` namespaced JSON payload. The RS
# byte never appears in normal model prose and ``json.dumps`` escapes it inside
# the payload, so the framing is unambiguous.
PART_SENTINEL_PREFIX = "\x1eKESTREL:PART:"
PART_SENTINEL_SUFFIX = "\x1e"

# Hard cap on a single serialized PART payload (the JSON between the bookends).
# Oversized parts are dropped with a warning rather than streamed — an unbounded
# part could blow the SSE frame / persisted row and is almost always a bug or an
# injection attempt.
MAX_PART_PAYLOAD_BYTES = 16 * 1024

# A part type is an opaque renderer key; keep it short, printable, and free of
# the control bytes that frame sentinels.
_MAX_TYPE_LEN = 64

logger = logging.getLogger(__name__)

# Per-turn buffer of emitted parts. ``None`` (the default) means "no active
# streaming turn" — an ``emit_part`` outside a turn is a no-op, not an error, so
# a tool that emits a part still works when invoked off the streaming path.
_part_collector: contextvars.ContextVar[Optional[List[dict]]] = contextvars.ContextVar(
    "kestrel_part_collector", default=None,
)


@contextlib.contextmanager
def part_collector():
    """Bind a fresh per-turn part buffer for the duration of one agent turn.

    The streaming layer wraps each turn in this so ``emit_part`` calls made by
    tools during the turn accumulate on a turn-scoped list. Resets to the prior
    token on exit so nested/concurrent turns don't bleed parts into each other.
    """
    token = _part_collector.set([])
    try:
        yield
    finally:
        _part_collector.reset(token)


def _sanitize_type(part_type: Any) -> Optional[str]:
    """Return a clean renderer key, or ``None`` if ``part_type`` is unusable."""
    if not isinstance(part_type, str):
        return None
    t = part_type.strip()
    if not t or len(t) > _MAX_TYPE_LEN:
        return None
    # Reject control chars (incl. the 0x1e record separator that frames
    # sentinels) so a type can never smuggle wire bytes into the stream.
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in t):
        return None
    return t


def _serialized_size_ok(part: dict) -> Optional[str]:
    """Serialize ``part`` and enforce the size cap. Returns the JSON payload on
    success, ``None`` on failure (non-serializable, oversized, or non-finite).

    ``allow_nan=False`` rejects ``NaN``/``Infinity``: Python's default emits the
    non-standard tokens ``NaN``/``Infinity`` which the browser's ``JSON.parse``
    throws on — that would make the live component silently vanish despite
    ``emit_part`` reporting success. Reject it here instead.
    """
    try:
        payload = json.dumps(
            part, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    if len(payload.encode("utf-8")) > MAX_PART_PAYLOAD_BYTES:
        return None
    return payload


def emit_part(part_type: str, data: Any, part_id: Optional[str] = None) -> bool:
    """Emit a typed part to be rendered as its own console bubble.

    Buffers ``{type, data, id}`` on the active turn's collector; the
    orchestrator drains it after the current tool and yields a PART sentinel
    into the live stream. Returns ``True`` if the part was accepted, ``False``
    if it was rejected (invalid type / non-serializable / oversized) or there is
    no active turn collector (called outside a streaming turn — a no-op).
    """
    collector = _part_collector.get()
    if collector is None:
        logger.debug("emit_part(%r) called with no active turn collector; ignored", part_type)
        return False
    clean_type = _sanitize_type(part_type)
    if not clean_type:
        logger.warning("emit_part: rejected invalid part type %r", part_type)
        return False
    part: dict = {"type": clean_type, "data": data}
    if part_id is not None:
        part["id"] = str(part_id)[:128]
    if _serialized_size_ok(part) is None:
        logger.warning(
            "emit_part: dropped non-serializable or oversized part (type=%s)", clean_type
        )
        return False
    collector.append(part)
    return True


def drain_parts() -> List[dict]:
    """Return and clear the parts buffered since the last drain.

    Called by the orchestrator after a tool batch so each part is yielded into
    the stream exactly once, in emission order, at the point its tool ran.
    """
    collector = _part_collector.get()
    if not collector:
        return []
    drained = list(collector)
    collector.clear()
    return drained


def build_part_sentinel(part: dict) -> Optional[str]:
    """Serialize a part dict to its in-band PART sentinel wire form.

    The single wire-format chokepoint: re-checks serializability + the size cap
    (already enforced in :func:`emit_part`) and returns ``None`` rather than
    emitting a malformed or oversized sentinel.
    """
    payload = _serialized_size_ok(part)
    if payload is None:
        return None
    return f"{PART_SENTINEL_PREFIX}{payload}{PART_SENTINEL_SUFFIX}"
