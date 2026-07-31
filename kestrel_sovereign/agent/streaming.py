"""Streaming response handling for Kestrel Agent."""
import asyncio
import inspect
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, Any, Optional

from kestrel_sdk.hooks.base import HookEvent, HookInput
from kestrel_sovereign.hooks.decision_gate import evaluate_blocking_decision
from kestrel_sovereign.hooks.manager import _hook_is_enforcing
from kestrel_sdk.llm import ToolCallStarted
from kestrel_sovereign.agent.parts import (
    PART_SENTINEL_PREFIX,
    PART_SENTINEL_SUFFIX,
    build_part_sentinel,
    drain_parts,
    part_collector,
)
from kestrel_sovereign.agent.operator_signals import inject_operator_turn
from kestrel_sovereign.agent.invocation import (
    bind_async_generator_invocation,
    current_invocation_id,
)
from kestrel_sovereign.agent.context_manager import CONTEXT_HISTORY_LIMIT
from kestrel_sovereign.llm.adapter import LLMResponse, ThinkingDelta
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
)
from kestrel_sovereign.telemetry import (
    KESTREL_AGENT_NAME,
    OI_SPAN_KIND,
    OI_SPAN_KIND_CHAIN,
    start_span,
    end_span,
)


def resolve_turn_invocation_context(
    llm_service,
    invocation_context: Optional[LLMInvocationContext],
    session_id: Optional[str],
) -> LLMInvocationContext:
    """Resolve the correlation identity for one agent turn's LLM call (#2614).

    ``invocation_context`` is the modern immutable per-request path: when the
    caller supplies it, its fields win. When it is ``None`` the legacy
    ``LLMService.set_observability_context`` ambient state is the fallback, so
    consumers that have not migrated keep working unchanged. Either way an
    explicit ``session_id`` fills an empty session slot so stateful adapters and
    per-session metering stay correlated with the turn.

    The service's own ``_resolve_invocation_context`` is the canonical resolver
    (explicit → ambient → per-session fallback); delegate to it when present so
    the agent and service never disagree on precedence. A minimal service double
    without that resolver falls back to the explicit context merged with the
    session id.
    """
    resolver = getattr(llm_service, "_resolve_invocation_context", None)
    if resolver is not None:
        return resolver(invocation_context, session_id=session_id)
    resolved = (
        invocation_context
        if invocation_context is not None
        else LLMInvocationContext()
    )
    if session_id and not resolved.session_id:
        resolved = replace(resolved, session_id=session_id)
    return resolved


# Wave 5E in-band revising sentinel — kestrel-sovereign #1086.
#
# The chat client uses this sentinel as the AUTHORITATIVE boundary
# between pre-tool prose and post-tool synthesis on the
# /api/agent/stream text/plain channel. It's strictly ordered with
# the chunks themselves, so unlike the parallel SSE `revising` event
# it can't race the post-tool chunks.
#
# Wire format: ``\x1eKESTREL:REVISE:<json>\x1e``
#   * ``\x1e`` (Record Separator, ASCII 30) bookends — chosen because
#     LLMs don't emit it, it's UTF-8-safe, and HTTP middleware
#     doesn't strip it.
#   * Fixed prefix ``KESTREL:REVISE:`` — namespace + intent.
#   * JSON payload: ``{"index": int, "tool_call_id": str|None,
#     "tool_name": str|None}``. Same fields as the SSE event minus
#     request_id/session_id (the in-band sentinel is implicitly
#     scoped to the stream that carries it).
#
# The Wave 5C SSE `revising` event is still emitted for non-chat
# consumers (audit dashboards, accessibility tools); chat clients
# treat the two signals as idempotent — whichever arrives first sets
# ``pane.pendingRevise``; the other is a no-op.
REVISE_SENTINEL_PREFIX = "\x1eKESTREL:REVISE:"
REVISE_SENTINEL_SUFFIX = "\x1e"

# #1662 eager-vision read-path bounds (independent of the upload endpoint's
# per-file/count caps, since a content-addressed read can name any stored
# hash). Keep peak memory for one turn's pasted images bounded.
_MAX_EAGER_IMAGES = 6
_MAX_EAGER_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MB (a touch over the 10 MB upload cap)


def _looks_like_image(data: bytes) -> bool:
    """Magic-number check that ``data`` is a real PNG/JPEG/GIF/WEBP.

    The client's declared ``kind``/``mime`` are not trusted: a tampered ref
    could mark a stored PDF/text hash as an inline image, and the image
    pipeline defaults unknown bytes to JPEG, so a non-image would be shipped to
    a vision provider as a bogus image. Sniff the actual bytes instead — only
    these four signatures are accepted for eager vision.
    """
    return (
        data.startswith(b"\x89PNG")
        or data.startswith(b"\xff\xd8\xff")
        or data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )
THINKING_SENTINEL_PREFIX = "\x1eKESTREL:THINK:"
THINKING_SENTINEL_SUFFIX = "\x1e"
# #1659: tool activity is the last signal class still emitted as
# regex-scraped emoji text. It now rides the same in-band sentinel
# channel as REVISE/THINK. Payload:
#   {"phase":"start"|"done"|"error","name":str,"index":int|None,
#    "ms":int|None,"detail":str|None}
# Like REVISE/THINK it is NEVER persisted in the assistant text — the
# structured ``tool_events`` metadata (now position-stamped) is the
# single source of truth for rendering tool cards, live and on reload.
TOOL_SENTINEL_PREFIX = "\x1eKESTREL:TOOL:"
TOOL_SENTINEL_SUFFIX = "\x1e"

# #1914: typed component parts (todo cards, citations, A2A artifacts, …) ride
# the same in-band channel. Payload: {"type":str,"data":any,"id":str?}. Like
# TOOL/THINK/REVISE the sentinel is NEVER persisted in assistant text — the
# position-stamped ``parts`` metadata is the single source of truth for
# re-rendering each component bubble, live and on reload. Constants imported
# from :mod:`kestrel_sovereign.agent.parts` (the emit/collect side).
#
# All sentinel classes share the \x1e suffix and are mutually exclusive in the
# stream; this tuple drives every strip/extract loop.
_ALL_SENTINEL_PREFIXES = (
    REVISE_SENTINEL_PREFIX,
    THINKING_SENTINEL_PREFIX,
    TOOL_SENTINEL_PREFIX,
    PART_SENTINEL_PREFIX,
)
_SENTINEL_SUFFIX = REVISE_SENTINEL_SUFFIX  # "\x1e" — identical for all


def _build_tool_sentinel(
    phase: str,
    name: str,
    index: Optional[int] = None,
    ms: Optional[int] = None,
    detail: Optional[str] = None,
) -> str:
    """Construct an in-band tool-activity sentinel.

    ``phase`` is "start" | "done" | "error". ``detail`` carries the
    shell-command preview (start) or error text (error); ``ms`` the
    dispatch duration (done).
    """
    payload = json.dumps({
        "phase": phase,
        "name": name,
        "index": index,
        "ms": ms,
        "detail": detail,
    }, separators=(",", ":"))
    return f"{TOOL_SENTINEL_PREFIX}{payload}{TOOL_SENTINEL_SUFFIX}"


def is_only_sentinels(text: str) -> bool:
    """True when ``text`` is one or more complete sentinels and nothing
    else (no visible prose). Used to keep a tool/think sentinel chunk
    from materializing a pending revise paragraph boundary — sentinels
    are wire bytes, not the visible text the weld waits for.
    """
    clean, _, _ = _parse_stream_sentinels(text)
    return text != "" and clean == ""


def _parse_stream_sentinels(text: str, base_offset: int = 0):
    """Strip all in-band sentinels from ``text`` and extract structured tool
    parts AND typed component parts, in a single pass.

    Returns ``(clean_text, tool_parts, parts)``:

    * ``tool_parts`` — the TOOL sentinel payloads (#1659), each augmented with
      ``pos``, the character offset into ``clean_text`` (plus ``base_offset``)
      at which the tool sentinel sat. ``pos`` lets the renderer place a tool
      card back at the right point between prose segments, live and on reload.
    * ``parts`` — the PART sentinel payloads (#1914), each ``{type,data,id?}``
      augmented with the same ``pos`` so a typed component bubble re-renders in
      document order on reload.

    REVISE sentinels weld a ``\\n\\n`` paragraph boundary at their point
    (mirroring the chat client's ``stripStreamSentinels`` — #1547); THINK, TOOL
    and PART sentinels are stripped without welding (they are not prose
    boundaries). Operates on a complete string, so the cross-packet edge cases
    the client handles don't arise here.
    """
    if not any(p in text for p in _ALL_SENTINEL_PREFIXES):
        return text, [], []
    result = ""
    tool_parts: list = []
    parts: list = []
    # #1914: a shared monotonic counter over TOOL and PART sentinels records
    # their WIRE order. ``pos`` can't disambiguate a tool and a part at the same
    # clean-text offset (the common ``tool done → PART → next tool`` with no
    # prose between); ``seq`` lets the reload renderer interleave them exactly as
    # they streamed. (Per-call, so only items from the same parse are compared —
    # same-offset collisions only ever occur within one stream half.)
    seq = 0
    i = 0
    n = len(text)
    weld_pending = False  # a removed revise sentinel awaits its next visible char
    while i < n:
        nxt_idx = -1
        nxt_prefix = ""
        for p in _ALL_SENTINEL_PREFIXES:
            idx = text.find(p, i)
            if idx >= 0 and (nxt_idx < 0 or idx < nxt_idx):
                nxt_idx, nxt_prefix = idx, p
        seg_end = nxt_idx if nxt_idx >= 0 else n
        seg = text[i:seg_end]
        if seg:
            if weld_pending and seg.strip():
                if result and not result[-1].isspace() and not seg[0].isspace():
                    result += "\n\n"
                weld_pending = False
            result += seg
        if nxt_idx < 0:
            break
        close = text.find(_SENTINEL_SUFFIX, nxt_idx + len(nxt_prefix))
        if close < 0:
            # Unterminated sentinel at the tail — drop the remainder,
            # matching the client's partial-buffer behavior.
            break
        if nxt_prefix == TOOL_SENTINEL_PREFIX:
            payload = text[nxt_idx + len(nxt_prefix):close]
            try:
                evt = json.loads(payload)
                if isinstance(evt, dict):
                    evt["pos"] = base_offset + len(result)
                    evt["seq"] = seq
                    seq += 1
                    tool_parts.append(evt)
            except (ValueError, TypeError):
                pass
        elif nxt_prefix == PART_SENTINEL_PREFIX:
            payload = text[nxt_idx + len(nxt_prefix):close]
            try:
                part = json.loads(payload)
                # A part must carry a string ``type`` (the renderer key); drop
                # anything malformed so a bad PART payload can't corrupt prose
                # or persist a junk component.
                if isinstance(part, dict) and isinstance(part.get("type"), str):
                    part["pos"] = base_offset + len(result)
                    part["seq"] = seq
                    seq += 1
                    parts.append(part)
            except (ValueError, TypeError):
                pass
        elif nxt_prefix == REVISE_SENTINEL_PREFIX:
            weld_pending = True
        i = close + len(_SENTINEL_SUFFIX)
    return result, tool_parts, parts


def _tool_parts_to_events(parts: list) -> list:
    """Convert parsed TOOL sentinel parts (phase/name/ms/detail/pos) into the
    persisted ``tool_events`` metadata shape (type/tool/ms/error/pos). Used on
    paths where the sentinels are the ONLY structured signal (e.g. codex-native
    tools that don't populate executed_tool_calls) so reload can reconstruct
    the cards instead of losing them after the sentinels are stripped.
    """
    events: list = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        phase = p.get("phase")
        name = p.get("name", "")
        pos = p.get("pos")
        # #1914: carry the shared wire-order ``seq`` so the reload merge can
        # interleave a tool card and a same-position part in stream order.
        seq = p.get("seq")
        if phase == "start":
            ev = {"type": "start", "tool": name, "pos": pos}
        elif phase == "done":
            ev = {"type": "complete", "tool": name, "pos": pos}
            if p.get("ms") is not None:
                ev["ms"] = p["ms"]
        elif phase == "error":
            ev = {
                "type": "error", "tool": name,
                "error": p.get("detail") or "", "pos": pos,
            }
        else:
            continue
        if isinstance(seq, int):
            ev["seq"] = seq
        events.append(ev)
    return events


def _utf16_offset(text: str, cp_offset: int) -> int:
    """Convert a Python code-point offset into the UTF-16 code-unit offset that
    JS ``String.slice`` uses (#1914). A part after a non-BMP char (e.g. an emoji
    🐢 — one code point, two UTF-16 units) would otherwise be stored an index too
    short, splitting the surrogate pair and misplacing the bubble on reload.
    """
    return len(text[:cp_offset].encode("utf-16-le")) // 2


def _rebase_events_for_parts(events: list, base: int, text: str) -> list:
    """Rebase ``tool_events`` onto the persisted post-tool ``text`` (subtract the
    retracted pre-half length ``base``) and convert each ``pos`` to a UTF-16
    offset — so tool cards share the SAME origin AND unit as the component parts
    persisted alongside them (#1914). The reload renderer merges both by ``pos``,
    so a mixed unit/origin would mis-order or split the bubbles (e.g. after an
    emoji). Applied ONLY when a turn carries parts, so the tool-card-only persist
    path is unchanged. Returns a new list; non-dict / posless events pass through.
    """
    out = []
    for ev in events or []:
        if isinstance(ev, dict) and isinstance(ev.get("pos"), int):
            ev = {**ev, "pos": _utf16_offset(text, max(0, ev["pos"] - base))}
        out.append(ev)
    return out


def _finalize_component_parts(
    parts: list, text: str, *, keep_hook_parts: bool = True,
) -> list:
    """Finalize component parts for persistence (#1914):

    1. Convert each ``pos`` from a code-point offset into the persisted ``text``
       to the UTF-16 offset the browser slices with on reload.
    2. Append any parts a POST_RESPONSE hook emitted after streaming — they're
       still buffered on the collector, so drain them here and position them at
       the end of ``text`` (they carry no in-stream offset). Persist-only: the
       live stream is already closed, so these surface on reload.

    The collector is ALWAYS drained (clearing the per-turn buffer), but the
    drained hook-emitted parts are DISCARDED when ``keep_hook_parts`` is False.
    A strict audit DENY passes ``keep_hook_parts=False`` (#2674): a POST_RESPONSE
    hook that runs *before* the denying audit hook can stash the (about-to-be-
    denied) ``response_text`` into an emitted part, so persisting the drained
    parts would leak exactly the text the audit blocked — it would re-render on
    reload beside the block message. Clearing the ``parts`` argument to ``[]`` is
    not enough on its own: the leak rides the collector, not that list.
    """
    result = [{**p, "pos": _utf16_offset(text, p.get("pos", 0))} for p in parts]
    hook_parts = drain_parts()
    if hook_parts and keep_hook_parts:
        end = _utf16_offset(text, len(text))
        result.extend({**p, "pos": end} for p in hook_parts)
    return result


def _flush_part_list(pending: list, full_response: list):
    """Emit the accumulated ``pending`` component parts (#1914) as PART
    sentinels — yield each for the live stream AND append to ``full_response``
    so the inline / no-tool persist path records it, position-stamped, like the
    orchestrator path captures its drained sentinels from the post-tool chunks.

    The inline loop accumulates parts (drained at each resume) and flushes them
    here right before the NEXT visible text — but AFTER any tool ``done``/``error``
    sentinel items — so a component lands after its producing tool's card and
    before the post-tool answer prose, in both live order and persisted ``seq``.
    Clears ``pending`` once emitted.
    """
    for part in pending:
        sentinel = build_part_sentinel(part)
        if sentinel:
            full_response.append(sentinel)
            yield sentinel
    pending.clear()


_TOOL_EVENT_PHASE = {"start": "start", "complete": "done", "error": "error"}


def _stamp_tool_event_positions(tool_events: list, parts: list) -> list:
    """Assign each ``tool_events`` entry a ``pos`` (clean-text offset) by
    matching it to the TOOL sentinel part that produced it.

    The by-ref ``tool_events`` (built server-side) and the in-band sentinel
    ``parts`` (which carry the positions) can arrive in different orders — the
    orchestrator emits all start sentinels up front, then the batch appends
    start/complete interleaved — so we match each event to the next unconsumed
    part with the same (phase, name) rather than relying on positional order.
    A startless terminal event (e.g. a follow-up-timeout ``error`` for ``llm``)
    therefore keeps its OWN position instead of inheriting the prior tool's.
    Events with no matching part fall back to the most recent stamped pos.
    Mutates + returns the list.
    """
    consumed = [False] * len(parts)
    last_pos = 0
    last_seq = 0
    for ev in tool_events:
        if not isinstance(ev, dict):
            continue
        want_phase = _TOOL_EVENT_PHASE.get(ev.get("type"))
        name = ev.get("tool", "")
        matched = None
        for i, p in enumerate(parts):
            if consumed[i]:
                continue
            if p.get("phase") == want_phase and p.get("name", "") == name:
                matched = i
                break
        if matched is not None:
            consumed[matched] = True
            pos = parts[matched].get("pos")
            if isinstance(pos, int):
                last_pos = pos
            # #1914: carry the wire-order ``seq`` too so reload can interleave a
            # tool card and a same-position part in the order they streamed.
            mseq = parts[matched].get("seq")
            if isinstance(mseq, int):
                last_seq = mseq
        ev["pos"] = last_pos
        ev["seq"] = last_seq
    return tool_events


def _build_revise_sentinel(marker: ToolCallStarted) -> str:
    """Construct the in-band revise sentinel for a ToolCallStarted marker."""
    payload = json.dumps({
        "index": marker.index,
        "tool_call_id": marker.id,
        "tool_name": marker.name,
    }, separators=(",", ":"))
    return f"{REVISE_SENTINEL_PREFIX}{payload}{REVISE_SENTINEL_SUFFIX}"


def _build_thinking_sentinel(delta: ThinkingDelta) -> str:
    """Construct an in-band thinking sentinel for chat thought bubbles."""
    payload = json.dumps({
        "content": delta.content,
        "provider": delta.provider,
    }, separators=(",", ":"))
    return f"{THINKING_SENTINEL_PREFIX}{payload}{THINKING_SENTINEL_SUFFIX}"


def strip_revise_sentinels(chunk: str) -> str:
    """Strip all complete in-band stream-control sentinels from a chunk.

    Server-side helper for non-chat consumers of
    ``process_input_streaming`` — voice/TTS, bridge stream, anything
    that re-publishes chunks downstream. The chat /stream endpoint
    passes chunks through verbatim because the chat *client* strips
    them, but other consumers would otherwise display or speak
    ``\\x1eKESTREL:REVISE:...`` literally. Codex P1 of #1089 caught
    the TTS-reading-aloud regression.

    Handles complete sentinels only — split sentinels (across chunks)
    leak the partial in this function. Server-side consumers that
    care about the split-chunk case should buffer at their own layer.
    In practice the server emits the sentinel as a single Python
    yield, so single-chunk delivery is the overwhelming common case.
    """
    prefixes = _ALL_SENTINEL_PREFIXES
    if not any(prefix in chunk for prefix in prefixes):
        return chunk
    out = []
    i = 0
    while i < len(chunk):
        matches = [
            (idx, prefix)
            for prefix in prefixes
            for idx in [chunk.find(prefix, i)]
            if idx >= 0
        ]
        if not matches:
            out.append(chunk[i:])
            break
        prefix_idx, prefix = min(matches, key=lambda pair: pair[0])
        out.append(chunk[i:prefix_idx])
        close_idx = chunk.find(
            REVISE_SENTINEL_SUFFIX,
            prefix_idx + len(prefix),
        )
        if close_idx < 0:
            # Split sentinel — no close in this chunk. Drop everything
            # from the prefix on; the sentinel-close half lands in
            # the next chunk and looks like leading wire metadata
            # there. Single-chunk-yield from the server makes this
            # path effectively unreachable.
            break
        i = close_idx + len(REVISE_SENTINEL_SUFFIX)
    return "".join(out)


def _strip_and_weld_revise_sentinels(text: str) -> str:
    """Strip in-band stream sentinels and weld a paragraph boundary at
    each revise point — the server-side mirror of the chat client's
    ``stripStreamSentinels`` (kestrel-sovereign #1547).

    Why this exists: the post-tool half of a streamed turn can carry
    embedded revise sentinels (the orchestrator emits one before each
    follow-up tool call). Persisting the raw join would (a) leak
    ``\\x1eKESTREL:REVISE:...`` wire bytes into the assistant row and
    (b) glue the prose on either side of a removed sentinel into
    "...part one.🔧 part two". The browser strips the same markers and
    welds a break where they sat between two non-whitespace chars; this
    function reproduces that exactly so the persisted turn equals what
    the user actually saw.

    Operates on a complete string (not a packet stream), so the
    cross-packet edge cases the client handles don't arise here. The
    pre-tool half needs no welding of its own: its boundary against the
    post-tool half was already recorded as a ``\\n\\n`` in
    ``full_response`` at the first ``ToolCallStarted`` marker, mirroring
    the client's weld at the first revise sentinel.

    Thin wrapper over :func:`_parse_stream_sentinels` (#1659) — callers
    that only want the cleaned text keep this name; the tool-part
    extraction is ignored here.
    """
    clean, _tool_parts, _parts = _parse_stream_sentinels(text)
    return clean


# Sentinel for "this hook exposes no mutable ``mode``" in a turn-start snapshot,
# so a hook without a ``mode`` attribute is captured (for pinned execution) yet
# never has a spurious ``mode`` attribute created on it by the pin (#2674).
_NO_MODE = object()


class PostResponseHookSnapshotError(RuntimeError):
    """The turn-start POST_RESPONSE hook snapshot could not be read (#2674).

    Both the streaming and non-streaming turn paths capture this snapshot BEFORE
    any provider/output work and derive the fail-closed ``buffer_audit`` /
    completion-enforcement decisions from it. If the underlying
    ``get_enabled_hooks`` read fails, returning an EMPTY snapshot would silently
    set ``buffer_audit=False`` and stream/persist raw output with NO audit — a
    fail-OPEN that masquerades as the legitimate "no audit configured" state.

    Raising instead fails CLOSED: because every entry path snapshots before the
    LLM is invoked and before any byte is yielded or persisted, propagating this
    error aborts the turn with no raw response data released. The genuine "agent
    has no hooks_manager" case still returns ``[]`` (a real absence of audit, not
    an infrastructure failure) — only a raising ``get_enabled_hooks`` reaches
    here.
    """


def _snapshot_post_response_hooks(hooks_manager) -> list:
    """Snapshot the enabled POST_RESPONSE hook set — with each hook's turn-start
    ``mode`` AND turn-start enforcement state — at turn start (#2674).

    Returns ``[(hook, mode_or_sentinel, enforcing), ...]`` for EVERY hook enabled
    when the turn begins (not only those exposing a mutable ``mode``). Two
    decisions read this one snapshot so they can never diverge within a turn:

    * **Buffering.** ``buffer_audit`` (whether to WITHHOLD streamed bytes) is
      derived from the captured ``enforcing`` flag — :func:`_hook_is_enforcing`
      evaluated HERE, at TRUE turn start (before ``USER_PROMPT_SUBMIT``), NOT
      re-read from the live hook later. That closes the finding-1 race: a
      ``USER_PROMPT`` hook that flips a strict audit to warn AFTER this snapshot
      (but before ``buffer_audit`` is computed) would otherwise make a live
      ``_hook_is_enforcing`` read return False and stream raw bytes, while the
      completion audit — pinned to the captured strict ``mode`` — still DENIES,
      the raw-streamed / block-persisted fail-open split. Capturing enforcement
      at the same instant as the mode keeps the withhold decision and the pinned
      completion verdict in lockstep for this turn (the flip takes effect NEXT
      turn), with no temporary shared hook mutation. It is also the predicate the
      hook manager fails closed on, so the withhold decision and the manager's
      timeout/crash → DENY behavior stay aligned.
    * **Enforcement.** ``_fire_post_response_hook`` runs EXACTLY this captured
      hook set through ``HooksManager.execute_hooks_snapshot`` — bypassing the
      live registry — so a hook the turn's tools *register*, *enable*, or
      *disable* mid-turn cannot change what audits this turn. Concretely:
      ``skip → audit_enable("strict")`` registers a brand-new strict hook, and
      running the live registry at completion would DENY a turn that already
      streamed raw (``buffer_audit`` was False) — the fail-open split. And
      ``strict → audit_disable`` drops the hook from the live enabled list, so
      running the live registry would release a *buffered* turn with no audit at
      all. Freezing the set to turn start closes both; the transition takes
      effect on the NEXT turn.

    Modes are pinned during the fire by :func:`_pinned_hook_modes`, which also
    defers an in-place ``warn → strict`` flip on an already-registered hook to
    the next turn. Hooks without a ``mode`` carry the ``_NO_MODE`` sentinel and
    are executed but never mode-pinned. The captured ``enforcing`` flag is a
    static per-hook boolean taken at this instant; because a generic hook's
    ``fail_closed`` / ``awaits_user_input`` do not depend on ``mode`` it matches
    a later live read, while a mode-derived ``fail_closed`` (the response audit
    hook) is frozen to its turn-start value here.

    **Fail closed on infrastructure failure (#2674).** A missing ``hooks_manager``
    is a legitimate "no audit configured" state → empty snapshot. But if the
    registry read itself (``get_enabled_hooks``) RAISES, the snapshot is unknown,
    not empty: swallowing it into ``[]`` would set ``buffer_audit=False`` and
    release/persist raw output with no audit — a fail-OPEN. Both callers snapshot
    before any provider/output work, so this raises
    :class:`PostResponseHookSnapshotError` to abort the turn before the LLM runs,
    guaranteeing no raw response byte is yielded or persisted.
    """
    if not hooks_manager:
        return []
    try:
        enabled = hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE)
    except Exception as e:
        # #2674 finding 3: FAIL CLOSED — do NOT swallow into an empty snapshot.
        # An empty snapshot disables buffering (``buffer_audit`` False) and the
        # completion audit, releasing/persisting raw output while masquerading as
        # "no audit". Both entry paths capture this snapshot before invoking any
        # LLM/provider and before yielding/persisting response data, so raising
        # here aborts the turn with no raw output released.
        raise PostResponseHookSnapshotError(
            "failed to read enabled POST_RESPONSE hooks at turn start; failing "
            "closed so no unaudited response is released or persisted"
        ) from e
    # Capture ``mode`` (for pinned execution) AND enforcement (for the buffering
    # decision) at the SAME instant — true turn start — so a later USER_PROMPT
    # mode flip can never make the withhold decision disagree with the pinned
    # completion verdict (#2674 finding 1).
    return [
        (h, getattr(h, "mode", _NO_MODE), _hook_is_enforcing(h)) for h in enabled
    ]


@contextmanager
def _pinned_hook_modes(snapshot: list):
    """Pin each snapshotted POST_RESPONSE hook to its turn-start ``mode`` for the
    duration of the audit fire, then restore the live mode (#2674).

    Restoring the *live* (possibly mid-turn-mutated) mode on exit — not the
    pinned one — is what makes a warn→strict switch apply to the NEXT turn: this
    turn's audit runs at the turn-start mode, subsequent turns see the new mode.
    The turn lifecycle lock serializes turns per agent, so nothing else mutates
    ``mode`` across the single ``await`` this wraps. Hooks captured with the
    ``_NO_MODE`` sentinel (no mutable ``mode``) are skipped so the pin never
    fabricates a ``mode`` attribute on them.
    """
    live: list = []
    for hook, start_mode, *_rest in snapshot:
        if start_mode is _NO_MODE:
            continue
        live.append((hook, getattr(hook, "mode", _NO_MODE)))
        try:
            hook.mode = start_mode
        except Exception:
            pass
    try:
        yield
    finally:
        for hook, live_mode in live:
            if live_mode is _NO_MODE:
                continue
            try:
                hook.mode = live_mode
            except Exception:
                pass


class _PostResponseText(str):
    """The reviewed assistant text carrying an EXPLICIT audit verdict (#2674).

    Subclasses ``str`` so every existing caller that persists, concatenates, or
    compares the returned text keeps working unchanged — but the strict
    streaming path reads the explicit ``denied`` / ``modified`` flags instead of
    re-deriving the verdict from ``text != original``. That inference is unsafe:
    a fail-closed DENY whose block message happens to equal the original prose
    would be misread as an ALLOW and leak the raw (unaudited) buffer — sentinels,
    typed parts and all. ``denied`` is the load-bearing flag; ``modified`` is
    true for both DENY and a MODIFY rewrite (any hook-changed text).

    A plain ``str`` returned by a test double (or any legacy caller) reports
    ``denied=False`` / ``modified=False`` / ``enforcing=False`` via ``getattr``
    fallback — i.e. it is treated as an unaudited ALLOW passthrough, which is
    exactly what such a double (or a pure-local command result) means.

    ``enforcing`` records whether the POST_RESPONSE audit that produced this
    verdict ran under a fail-closed (strict) hook, derived from the SAME
    turn-start ``hook_snapshot`` / live hook set ``_fire_post_response_hook``
    actually executed. The streaming command wrapper reads it to decide whether
    to withhold this turn's typed parts, instead of taking a fresh registry
    snapshot that could diverge from the one the audit ran under (#2674).
    """

    denied: bool = False
    modified: bool = False
    enforcing: bool = False

    def __new__(
        cls, text: str, *,
        denied: bool = False, modified: bool = False, enforcing: bool = False,
    ):
        obj = super().__new__(cls, text)
        obj.denied = denied
        obj.modified = modified
        obj.enforcing = enforcing
        return obj


# #2674 finding 2: the deterministic block a strict (buffered) turn releases
# when its post-tool continuation LLM call times out. It carries NO route name,
# duration, raw error, tool argument, or model-generated text — a fixed, safe
# body so the fail-closed audit path can never resolve to an empty 200. Held as
# a module constant so the endpoint contract and the regression tests share one
# string.
STRICT_AUDIT_CONTINUATION_TIMEOUT_BLOCK = (
    "⚠️ The response could not be completed: the model did not finish "
    "after a tool step within the allowed time. No partial or unreviewed "
    "content was released. Please try again."
)


def _strict_audit_release(final_text: str) -> list:
    """Post-verdict release for a strict (enforcing) buffered turn (#2674).

    Only the reviewed ``final_text`` reaches the live client — the exact text
    the POST_RESPONSE hook examined and returned: the block message on DENY, the
    rewrite on MODIFY, the approved prose on ALLOW. The withheld raw buffer
    (pre-tool prose, thinking, tool / revise / part sentinels) is DISCARDED and
    never replayed, because:

    * it contains material the audit never reviewed (the hook sees only the
      final post-tool text, not pre-tool prose / thinking / structured parts), so
      replaying it would emit unaudited bytes on an ALLOW; and
    * pre-tool prose is retracted from the persisted turn anyway, so releasing it
      would surface text that never survives to history reload.

    Structured parts do NOT survive a strict buffered turn — on ALLOW as much as
    on DENY. The buffered-persist path drops the whole metadata envelope (parts,
    pre_tool_reasoning, tool_events, tool_results) under an enforcing audit, so
    the invariant holds on every verdict: persisted == reloaded == released
    reviewed prose, nothing more. The audit never reviewed those side channels,
    so releasing OR persisting them would surface unaudited bytes. An empty
    ``final_text`` releases nothing (e.g. an approved-but-empty turn).
    """
    return [str(final_text)] if final_text else []


def _post_response_verdict(
    hook_output, original_text: str, *, enforcing: bool,
) -> "_PostResponseText":
    """Translate a POST_RESPONSE ``HookOutput`` into an explicit verdict (#2674).

    Returns a ``_PostResponseText`` carrying BLOCK vs ALLOW/MODIFY directly, not
    inferred from string equality — a block whose message equals the original
    prose would otherwise be misread as ALLOW and leak the raw buffer.

    A POST_RESPONSE hook blocks on BOTH ``DENY`` and ``ASK`` (and a bare
    ``continue_execution=False``). An approval hook exposes ``awaits_user_input``
    → the turn was BUFFERED, and ``ASK`` means "not approved yet"; releasing the
    raw buffer on ASK would leak unapproved content. Route through the shared
    ``evaluate_blocking_decision`` fail-closed gate so ASK can never fall through
    to the ALLOW passthrough.

    Module-level (not a method) so the MagicMock-based streaming test harnesses,
    which bind only a curated set of mixin methods onto a bare mock agent, call
    the REAL verdict logic instead of an auto-returned child mock.
    """
    blocked = evaluate_blocking_decision(
        hook_output,
        deny_reason_default="blocked by response audit",
        ask_reason_default="response withheld pending approval",
    )
    if blocked is not None:
        logging.warning(
            "POST_RESPONSE hook blocked (streaming, "
            f"{blocked.decision.value}): {blocked.reason}"
        )
        # #2674 finding 2: this block message is the ONLY text a strict
        # (enforcing / buffered) turn releases AND persists. ``blocked.reason``
        # carries UNTRUSTED diagnostics (the audit LLM's reasoning may quote raw
        # response content; a fail-closed crash carries exception text), so
        # reflecting it would leak exactly what the buffer withheld. Release a
        # CONSTANT, decision-typed message; the full reason is in the operator
        # log above (a different trust boundary).
        safe_reason = (
            "response withheld pending approval"
            if blocked.is_ask
            else "response withheld by audit policy"
        )
        return _PostResponseText(
            f"[Response blocked by audit: {safe_reason}]",
            denied=True,
            modified=True,
            enforcing=enforcing,
        )
    if hook_output.updated_input and "response_text" in hook_output.updated_input:
        return _PostResponseText(
            hook_output.updated_input["response_text"],
            modified=True,
            enforcing=enforcing,
        )
    return _PostResponseText(original_text, enforcing=enforcing)


class StreamingMixin:
    """Mixin class providing streaming response methods."""

    @staticmethod
    def _lazy_attachment_hint(attachments) -> str:
        """A compact note listing this turn's LAZY (non-inline) attachments so
        the agent knows their ids for `read_attachment`. Empty when there are
        none. Eager (inline) images are folded as vision and not listed.
        """
        if not attachments:
            return ""
        lazy = [
            a for a in attachments
            if isinstance(a, dict) and a.get("hash") and not a.get("inline")
        ]
        if not lazy:
            return ""
        lines = [
            f"- {a.get('name') or 'attachment'} (id: {a['hash']})"
            for a in lazy
        ]
        return (
            "\n\n[Attachments available to read with the read_attachment tool:\n"
            + "\n".join(lines)
            + "\nCall read_attachment with an id above to read that file.]"
        )

    async def _resolve_eager_images(self, attachments) -> list:
        """Resolve this turn's *inline* image attachments (#1662 eager vision)
        to raw bytes for the model.

        ``inline`` marks an image pasted or dropped straight into the composer
        — the user's intent is "see this now", so it rides this turn as vision
        input. Non-inline attachments are lazy references the agent reads on
        demand via a tool (PR C), and documents are always lazy; both are
        skipped here. The ref is already persisted on the user turn, so an
        attachment whose bytes are missing/unreadable is skipped without losing
        the history record.

        Resolution is bounded independently of the upload endpoint. The
        content-addressed read can name *any* stored hash (up to the 50 MB
        store cap), not just files that came through the 10 MB upload gate, so
        a turn could otherwise pull many large blobs into memory at once. We
        cap both the number of eager images and each image's size, and log
        loudly when something is dropped.
        """
        if not attachments:
            return []
        storage = getattr(self, "storage", None)
        if storage is None or not hasattr(storage, "retrieve_file"):
            return []
        images: list = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            if att.get("kind") != "image" or not att.get("inline"):
                continue
            content_hash = att.get("hash")
            if not isinstance(content_hash, str):
                continue
            if len(images) >= _MAX_EAGER_IMAGES:
                logging.warning(
                    "Eager vision: more than %d inline images this turn; "
                    "extras ignored.", _MAX_EAGER_IMAGES,
                )
                break
            try:
                data = await storage.retrieve_file(content_hash)
            except Exception as exc:
                logging.warning(
                    "Eager vision: could not load attachment %s: %s",
                    content_hash[:12], exc,
                )
                continue
            if not data:
                continue
            if len(data) > _MAX_EAGER_IMAGE_BYTES:
                logging.warning(
                    "Eager vision: attachment %s is %d bytes (> %d cap); "
                    "skipped.", content_hash[:12], len(data),
                    _MAX_EAGER_IMAGE_BYTES,
                )
                continue
            if not _looks_like_image(data):
                # Client lied about kind/mime, or the file isn't really an
                # image — don't ship non-image bytes to a vision provider.
                logging.warning(
                    "Eager vision: attachment %s is not a recognized image "
                    "(PNG/JPEG/GIF/WEBP); skipped.", content_hash[:12],
                )
                continue
            images.append(data)
        return images

    @bind_async_generator_invocation("request_id")
    async def process_input_streaming(
        self,
        user_input: str,
        model_override: str = None,
        audit_before_streaming: bool = False,
        session_id: str = None,
        caller=None,
        request_id: Optional[str] = None,
        attachments: Optional[list] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
        invocation_provenance=None,
    ):
        """
        Streaming version of process_input. Yields text chunks as generated.

        ``invocation_context`` is the modern identity-passing path (immutable,
        per-request). Prefer it in new consumers; the legacy
        ``LLMService.set_observability_context`` state remains as a fallback.
        See :func:`resolve_turn_invocation_context` for the precedence rule.

        Uses single-call streaming with tool detection pattern:
        1. Build context and feature tools (same as process_input)
        2. Single streaming LLM call that yields text AND detects tools
        3. If tool calls detected at end, execute them and continue streaming
        4. Text chunks streamed to user in real-time

        Strict-audit latency tradeoff (#2674): response audit is optional and
        OFF by default. When it is enabled in an *enforcing* (fail-closed /
        strict) mode, the fail-closed guarantee requires that no user-visible
        byte reach the client before the POST_RESPONSE verdict exists. This path
        therefore WITHHOLDS the entire visible turn while the turn is assembled
        for the audit, then releases ONLY the reviewed text once the verdict
        lands — the approved/rewritten prose on ALLOW/MODIFY, or just the block
        message on DENY (``_strict_audit_release``). The withheld raw stream
        (pre-tool prose, thinking, tool/revise/part sentinels) is never
        replayed: the audit never reviewed it, and pre-tool prose is retracted
        from the persisted turn anyway. A strict buffered turn ALSO drops its
        whole unaudited metadata envelope on persist — parts, pre_tool_reasoning,
        tool_events and tool_results — on ALLOW as much as on DENY, and suppresses
        the out-of-band ``revising`` SSE event while sanitizing the STOP hook's
        tool_calls/tool_results (#2674). So the live stream, the persisted row,
        the reloaded turn and every hook side channel all carry the reviewed prose
        ONLY — nothing the audit did not see. That trades real-time token delivery
        (and live tool/part cards, which no longer survive a buffered turn) for
        the integrity gate. With audit disabled, or in advisory/warn mode,
        streaming stays fully incremental and this method's behavior is unchanged.

        Args:
            user_input: The user's message
            model_override: Optional model to use
            audit_before_streaming: Deprecated, ignored. Kept for API compat.
            session_id: Optional session ID to load conversation context from a specific session
            caller: Optional CallerContext with auth identity and role.
            request_id: The endpoint-assigned id for this stream. Used as
                the routing key on emitted ``revising`` events so the
                frontend can match the event to the correct in-flight
                pane when multiple streams are active concurrently.
                Falls back to ``self._current_request_id`` (the legacy
                agent-global) when omitted, but callers should prefer
                explicit values — the global is overwritten by the next
                ``register_active_request`` call and races between
                overlapping streams can otherwise misroute events.
            invocation_provenance: Endpoint-owned authenticated actor and
                transport metadata bound task-locally for governed tools.
        """
        # Match process_input's retryable pre-initialization behavior. Without
        # storage there is no durable genesis receipt to inspect yet.
        if getattr(self, "storage", None) is None:
            raise RuntimeError(
                "agent not fully initialized: context_manager and storage "
                "unavailable; deferring turn for retry until initialize() completes"
            )

        # Same pre-cognition lifecycle boundary as process_input. Run before
        # periodic checks, turn locks, context construction, and streaming LLM
        # setup so two simultaneous first turns cannot outrun the audit.
        genesis_block = await self._genesis_audit_cognition_block(user_input)
        if genesis_block is not None:
            yield genesis_block
            return

        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # The streaming endpoint is the primary chat path and must enforce the
        # same constitutional boundary as process_input. Commands delegate
        # below so Sovereign recovery remains available; ordinary cognition is
        # refused before a turn lock, context build, or LLM stream can start.
        safe_mode = getattr(self, "_safe_mode", False) is True
        audit_pending = (
            getattr(self, "_constitution_audit_pending", False) is True
        )
        if (safe_mode or audit_pending) and not user_input.startswith("!"):
            restriction = (
                "a required startup integrity audit"
                if audit_pending
                else "an integrity failure"
            )
            yield (
                "🚨 SAFE MODE ACTIVE\n\n"
                f"The agent cannot process queries due to {restriction}.\n"
                "Use !safe-mode to check status or !verify-constitution to "
                "re-verify.\n\n"
                "Normal operation will resume once integrity is restored."
            )
            return

        # Commands are not streamable - delegate to non-streaming handler.
        # #1894: the delegate runs OUTSIDE the per-turn ``part_collector()``
        # bound on the normal streaming path below, so a ``!todo add/update/
        # complete`` command (todo tools advertise ``command_prefix``) would
        # mutate the graph but ``emit_part`` would see no active collector and
        # no-op — the typed todo bubble would never reach the stream. Bind a
        # collector around the delegation and flush any parts it emitted as
        # PART sentinels after the command result, mirroring the normal path.
        if user_input.startswith("!"):
            # #2674 finding 1: a command that produces an LLM response — notably
            # ``!continue`` (process_input rewrites it into a continuation
            # prompt) and any command whose handler returns None and falls
            # through to a normal LLM turn — is audited by process_input's
            # POST_RESPONSE fire, now fail-closed + turn-start-snapshot-pinned
            # (see kestrel_agent._process_input_traced_locked). So ``result`` is
            # already the reviewed text (or the block message on
            # DENY/ASK/failure/timeout). A *pure local* command response returns
            # before that fire and is intentionally NOT audited (established
            # contract) — its ``result`` is a plain ``str``, not a
            # ``_PostResponseText``.
            #
            # But typed parts a tool emitted during the turn ride the per-turn
            # collector, NOT ``response_text``, so the audit never reviewed them.
            # Releasing them would leak unreviewed data past the fail-closed gate
            # exactly like findings 2/3 on the streaming path. Mirror that here:
            # under an enforcing audit, or on a blocked verdict, drop the drained
            # parts (the client receives only the reviewed/blocked text). A pure
            # local command (plain-``str`` result) is never an audited LLM turn,
            # so its parts — e.g. a ``!todo add`` card — always flow, even with a
            # strict audit enabled.
            #
            # #2674: the enforcing verdict is read off ``result`` itself — the
            # ``_PostResponseText`` stamped by ``_fire_post_response_hook`` from
            # the SAME turn-start snapshot the audit ran under — NOT a fresh
            # registry snapshot taken here. A second snapshot could diverge from
            # ``process_input``'s turn-start capture if a hook is
            # registered/enabled/disabled/mode-flipped between the two reads,
            # which would let a strict-audited turn's unaudited parts escape while
            # this wrapper believed the turn was non-enforcing. A plain ``str``
            # (pure-local command) reports ``enforcing=False`` via ``getattr``, so
            # its parts always flow.
            with part_collector():
                # Preserve the pre-existing positional call shape
                # (test_streaming_audit asserts it) — invocation_context
                # rides as a trailing kwarg. Codex round-1 P1 backwards-compat.
                result = await self.process_input(
                    user_input,
                    model_override,
                    session_id=session_id,
                    caller=caller,
                    invocation_context=invocation_context,
                    invocation_id=current_invocation_id(),
                )
                yield result
                release_parts = not getattr(result, "denied", False) and not (
                    getattr(result, "enforcing", False)
                )
                drained = drain_parts()
                if release_parts:
                    for part in drained:
                        sentinel = build_part_sentinel(part)
                        if sentinel:
                            yield sentinel
            return

        # Start OTEL span for streaming request lifecycle
        _otel_span = start_span("agent.process_input_streaming", {
            OI_SPAN_KIND: OI_SPAN_KIND_CHAIN,
            # Defensive read — the StreamingMixin can be hosted by a minimal /
            # duck-typed object with no ``agent_name`` (#2699). ``start_span``
            # drops the None so telemetry stamping never crashes the turn.
            KESTREL_AGENT_NAME: getattr(self, "agent_name", None),
            "agent.did": self.did,
            "agent.session_id": session_id or "",
            "agent.input_length": len(user_input),
            "agent.streaming": True,
        })

        try:
            # Lock order — CONVERSATION (via _turn_lifecycle) BEFORE the privacy
            # transition lock — is the deadlock-freedom invariant. The in-turn
            # `!privacy` path runs inside process_input, which already holds
            # CONVERSATION and then acquires the transition lock; acquiring in the
            # same order here means no two callers can take this pair in opposite
            # directions (the AB-BA wedge this replaces, where streaming took the
            # transition lock first and then blocked on CONVERSATION).
            async with self._turn_lifecycle():
                transition_lock = self._get_privacy_transition_lock()
                async with transition_lock:
                    # #1914: bind a per-turn part buffer so tools/features can
                    # ``emit_part`` typed component bubbles; the orchestrator
                    # drains it into PART sentinels at the point each tool ran.
                    with part_collector():
                        async for chunk in self._process_input_streaming_traced_locked(
                            user_input, model_override, session_id, _otel_span,
                            request_id=request_id, attachments=attachments,
                            invocation_context=invocation_context,
                        ):
                            yield chunk
        except Exception as exc:
            end_span(_otel_span, error=exc)
            raise
        else:
            end_span(_otel_span)
            return

    async def _process_input_streaming_traced_locked(
        self, user_input, model_override, session_id, _otel_span,
        request_id: Optional[str] = None, attachments: Optional[list] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
    ):
        """Inner streaming logic wrapped in an OTEL span.

        Caller MUST hold the turn lifecycle (CONVERSATION lock). The
        wrapper `process_input_streaming` enters the lifecycle once;
        delegating here from a held-lifecycle context avoids the
        non-reentrant-lock self-deadlock that the dispatcher (Phase 1)
        would otherwise hit when COGNITION signals route through this
        path."""
        # #1662: record THIS turn's session so tools that must scope to the
        # active conversation (read_attachment) have an authoritative value —
        # the tool-call `session_id` arg is model-controlled and usually omitted,
        # and the inline executor binds session only into hook context, not tool
        # args. The turn-lifecycle lock serializes turns per agent, so this is
        # safe per-turn.
        self._active_session_id = session_id
        # Prompt injection detection (log-only, does not block)
        check_prompt_injection(user_input)

        # #2674 finding 2: snapshot the enabled POST_RESPONSE hook set at TRUE
        # turn start — BEFORE USER_PROMPT_SUBMIT fires — so the fail-closed
        # buffering / enforcement decision reflects the audit that was pinned when
        # the turn began. Capturing it AFTER USER_PROMPT_SUBMIT would let a
        # USER_PROMPT hook that unregisters / disables the strict audit turn the
        # gate OFF for the very turn it was present at start of, streaming raw
        # bytes with no audit — a fail-OPEN that violates the documented contract
        # (a strict hook present at turn start remains pinned; registration,
        # removal and warn↔strict flips are NEXT-turn transitions). The turn-
        # lifecycle lock already serializes turns, so nothing else mutates the
        # registry between here and the completion audit. ``None`` hooks_manager
        # stays a legitimate "no audit configured" empty snapshot; a raising
        # registry read fails CLOSED (PostResponseHookSnapshotError) before any
        # byte is streamed. The command / ``!continue`` fall-through returns above
        # and re-enters ``process_input``, which snapshots the same way BEFORE its
        # own USER_PROMPT_SUBMIT — one snapshot per path, never a double capture.
        audit_hook_snapshot = _snapshot_post_response_hooks(
            getattr(self, "hooks_manager", None)
        )

        # Fire USER_PROMPT_SUBMIT hook
        if self.hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
                user_message=user_input,
            )
            hook_output = await self.hooks_manager.execute_hooks(
                HookEvent.USER_PROMPT_SUBMIT, hook_input
            )
            # DENY and ASK both block the prompt (F038): an ASK on
            # USER_PROMPT_SUBMIT gates the turn behind approval, so it
            # must not fall through and run.
            blocked = evaluate_blocking_decision(hook_output)
            if blocked is not None:
                yield f"[Input rejected: {blocked.reason}]"
                return
            # The manager applies updated_input to hook_input.tool_input;
            # check if hooks modified the user_message via that path.
            if hook_input.tool_input and "user_message" in hook_input.tool_input:
                user_input = hook_input.tool_input["user_message"]

        # #1844 Stage 2: Kestrel-owned compaction for openai:plan. Compact +
        # reset the codex thread BEFORE fetching history / building context so
        # the freshly-seeded thread carries the compacted view rather than
        # letting codex auto-compact opaquely server-side. Gated on the codex
        # adapter's own recorded occupancy (not route prediction). Best-effort.
        #
        # The hook lives on KestrelAgent; StreamingMixin is reusable, so guard
        # it as OPTIONAL — a standalone-mixin host (or a partial test host) that
        # lacks the real coroutine helper is a clean no-op rather than an
        # ``await MagicMock()`` TypeError (codex review r9).
        _compact_hook = getattr(self, "_maybe_compact_codex_thread", None)
        if inspect.iscoroutinefunction(_compact_hook):
            await _compact_hook(session_id)

        # Build full context using context_manager (same as process_input)
        force_local_only = not self.privacy_agent.privacy_config.allows_cloud_llm()
        privacy_mode = self.privacy_agent.privacy_mode.name

        # Get session-filtered conversation history
        history = await self.privacy_agent.get_conversation_history(
            limit=CONTEXT_HISTORY_LIMIT,
            session_id=session_id,
        )

        # Get constitution the same way as process_input
        constitution = await self._get_governing_constitution()

        # FAIL CLOSED (#2463 review): `_get_governing_constitution()` returns
        # error-sentinel strings ("Error: ...") when the anchored constitution
        # cannot be retrieved/anchored. Those are non-empty strings but are NOT
        # a constitution body — proceeding would issue the model call with the
        # error text standing in for the governing constitution. Refuse the turn
        # instead, exactly as the signal dispatcher does for full injection.
        if isinstance(constitution, str) and constitution.lstrip().startswith("Error:"):
            logging.error(
                "STREAMING refused: _get_governing_constitution returned an "
                "error sentinel (%s); not issuing the model call without a "
                "governing constitution.",
                constitution,
            )
            yield (
                "I cannot continue this turn safely: my governing constitution "
                "could not be loaded. Please verify system integrity."
            )
            return

        # Pass the session-filtered history so context_manager uses it
        # Fetch reflection guidance for prompt injection
        reflection_guidance = None
        reflection_feature = self.features.get("ReflectionFeature") if hasattr(self, 'features') else None
        if reflection_feature:
            try:
                reflection_guidance = await reflection_feature.get_active_guidance()
            except Exception:
                pass  # Non-critical — log already handled in feature

        # Snapshot tool schemas before planning so dry/live accounting and the
        # eventual provider payload use identical registry bytes.
        feature_tools = self._build_all_tools()
        context_result = await self.context_manager.build_context(
            query=user_input,
            constitution=constitution,
            include_briefing=not getattr(self, "_session_briefed", False),
            include_memories=True,
            include_rag=True,
            privacy_mode=privacy_mode,
            conversation_history=history,
            reflection_guidance=reflection_guidance,
            tools=feature_tools,
        )

        # B / #1309 + C / #1311 degraded-mode fail-closed (streaming
        # path). When ``build_context`` returns ``degraded_mode=True``
        # (mandatory governance floor doesn't fit, or durable-salvage
        # write failed under C's feature flag), the LLM call MUST NOT
        # proceed. Yield a refusal chunk so the stream consumer sees
        # the failure and stops, instead of silently proceeding to
        # ``stream_with_tool_detection`` against an empty context.
        # Explicit ``is True`` so MagicMock-returning test fixtures
        # don't trip the gate inadvertently.
        if getattr(context_result, "degraded_mode", False) is True:
            warn_text = " | ".join(context_result.warnings or ["context build degraded"])
            logging.error(
                "DEGRADED MODE on streaming build_context — refusing to "
                "issue the model call. Warnings: %s",
                warn_text,
            )
            yield (
                "I cannot continue this turn safely: the context window "
                f"is in a degraded state. Details: {warn_text}"
            )
            return

        # Keep streaming and non-streaming context inputs identical: the
        # session briefing is admitted once, after a non-degraded plan, before
        # the provider call starts.
        self._session_briefed = True

        # Build user prompt. `context` carries the per-turn retrieved content
        # (memories + RAG) — kept OUT of the system message so the system prefix
        # is stable across turns and prompt caches can hit (see issue #703).
        wrapped_user = wrap_user_input(user_input)
        prompt = self.user_prompt_template.format(
            context=context_result.dynamic_user_context,
            query=wrapped_user
        )

        # Canonical/transport split (#1402): persist raw wrapped user turn
        # as ``content`` and the rendered prompt (memories + RAG baked in)
        # as ``rendered_content``. History-load replays ``rendered_content``
        # verbatim so Anthropic's cache_control marker at messages[-2]
        # still compounds across turns, while every other consumer
        # (search, audit, UI, memory ingestion) sees clean user speech.
        # #1662: persist attachment refs on the user turn so the composer's
        # images/docs survive reload (and a later turn can resolve them). The
        # bytes live in the encrypted file store; only the refs ride here.
        _user_meta = {"sent_form": True}
        if attachments:
            _user_meta["attachments"] = attachments
        await self.privacy_agent.add_conversation(
            "user", wrapped_user,
            metadata=_user_meta,
            session_id=session_id,
            rendered_content=prompt,
        )
        # Apply post-build assembly via the agent helper so this path
        # uses the same route-aware budget gate as the non-streaming
        # path (#1399). Without sharing the helper, route-capped
        # subscriptions hit the same codex-stdout-close 500 from the
        # streaming endpoint that the non-streaming endpoint just got
        # protected against.
        system_prompt = self._assemble_post_build_system_prompt(
            context_result.system_prompt, context_result,
            user_prompt=prompt,
        )

        if self.extension:
            try:
                prefix = self.extension.get_system_prompt_prefix()
                if prefix:
                    system_prompt = f"{prefix}\n{system_prompt}"
            except Exception:
                pass

        # Determine model
        effective_model = model_override
        if not effective_model:
            effective_model = await self.check_solvency()

        # ``feature_tools`` was captured before context planning.

        # Log tool availability
        tool_names = [t['function']['name'] for t in feature_tools] if feature_tools else []
        await self.observability_store.log_metric(
            agent_name=self.did,
            metric_name="feature_tools_built_streaming",
            metric_value=len(feature_tools),
            metadata={
                "tool_names": tool_names,
                "model": effective_model,
            }
        )

        # Build full messages array for multi-turn conversation
        # Format: [system, ...history, user]
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_result.messages)  # Add conversation history
        # #1662: tell the agent which LAZY attachments it can read this turn, so
        # it knows the ids to pass to `read_attachment`. Live-only — appended to
        # the LLM-bound user content, NOT the persisted prompt (keeps sent-form
        # cache stability and the raw-user extraction clean). Inline (eager)
        # image attachments are folded as vision downstream and aren't listed.
        lazy_hint = self._lazy_attachment_hint(attachments)
        messages.append({"role": "user", "content": prompt + lazy_hint})

        # #2530: the operator notice is collected and injected LAST, right
        # before the provider call — see the injection site below for why the
        # turn setup between here and there must not sit inside that window.

        # Single streaming call with tool detection
        llm_start = time.time()
        llm_event_id = await self.observability_store.log_tool_call(
            agent_name=self.did,
            tool_name="llm_stream_with_tool_detection",
            metadata={
                "model": effective_model,
                "tools_count": len(feature_tools) if feature_tools else 0,
                "force_local_only": force_local_only,
                "history_messages": len(context_result.messages),
            }
        )

        # Accumulate text and watch for tool response at end
        full_response = []
        tool_response = None
        # Snapshot of streamed text at the moment the FIRST
        # ToolCallStarted marker arrives — this is the boundary between
        # "what the agent said it was about to do" and "what the agent
        # actually saw the tool return" for the narration check
        # (#1042 layer 3). ``None`` when no marker fired this turn;
        # empty string when the marker fired before any text arrived.
        pre_tool_prose_snapshot: Optional[str] = None
        # #1547: mirrors the chat client's `pendingReviseBoundary`. A
        # ToolCallStarted marker arms a paragraph boundary that is only
        # MATERIALIZED (as `\n\n`) when the next visible text actually
        # arrives — either a later text chunk in this loop (inline
        # execution) or the post-tool synthesis at the join below. Lazy,
        # not eager, so a turn that ends right after the marker (e.g. the
        # user cancels before tools run) never persists a dangling `\n\n`
        # the client never rendered.
        pending_visible_boundary = False
        # #1914: parts emitted by inline-executed tools accumulate here and flush
        # right before the next VISIBLE text (after any tool done/error
        # sentinels), so each component lands after its producing tool's card.
        pending_parts: list = []

        # #1662 eager vision: resolve this turn's inline (pasted/dropped) image
        # attachments to bytes so the LLM service can fold them into the user
        # message in the resolved provider's native format. Lazy refs and
        # documents are skipped here — the agent reads those on demand.
        eager_images = (
            await self._resolve_eager_images(attachments) if attachments else None
        )
        cancel_token = (
            (lambda: self.is_request_cancelled(request_id))
            if request_id else None
        )

        # #2614: resolve the per-request correlation identity for THIS turn.
        # Explicit ``invocation_context`` wins; otherwise fall back to the
        # legacy ``set_observability_context`` ambient state, with the explicit
        # ``session_id`` filling an empty session slot. Same helper as the
        # non-streaming path so both agree on precedence.
        resolved_context = resolve_turn_invocation_context(
            self.llm_service, invocation_context, session_id
        )

        # #2674 finding 2: ``audit_hook_snapshot`` was captured at the TOP of
        # this method — BEFORE USER_PROMPT_SUBMIT — so a USER_PROMPT hook that
        # registers / removes / mode-flips the audit takes effect NEXT turn, not
        # this one. ONE snapshot drives two decisions so they can never diverge
        # mid-turn: the buffering decision below and the completion-time
        # enforcement in ``_fire_post_response_hook`` (which runs exactly this set
        # via ``execute_hooks_snapshot``, ignoring any hook a tool registers/
        # enables/disables mid-turn). See the function's docstring for the
        # fail-open split this closes.
        #
        # #2674: strict response audit is fail-closed, so when an enforcing
        # POST_RESPONSE hook is active we must NOT stream any user-visible byte
        # before its verdict exists. WITHHOLD every visible chunk (text +
        # sentinels + parts) — the assembled turn is still accumulated into
        # ``full_response`` / ``tool_response_chunks`` for the audit + persist —
        # and release ONLY the reviewed text (via ``_strict_audit_release``)
        # after ``_fire_post_response_hook`` returns. When no enforcing hook is
        # active this is False and streaming stays fully incremental.
        #
        # #2674 finding 1: read the enforcement flag CAPTURED in the snapshot at
        # true turn start — NOT a fresh ``_hook_is_enforcing(h)`` on the live
        # hook. USER_PROMPT_SUBMIT has already fired by this point, so a
        # USER_PROMPT hook that flipped a strict audit to warn would make a live
        # read return False and stream raw bytes, while the completion audit —
        # pinned to the captured strict mode — still DENIES (the raw-streamed /
        # block-persisted fail-open split). The captured flag and the pinned
        # completion verdict both reflect the turn-start mode, so they stay in
        # lockstep with no temporary shared hook mutation; the flip applies NEXT
        # turn.
        buffer_audit = any(
            enforcing for _h, _mode, enforcing in audit_hook_snapshot
        )
        # #2674 finding 3: when the turn is buffered for an enforcing audit, the
        # assistant prose is withheld from the user pending the verdict — so the
        # provider call's raw prompt/response must NOT land in durable llm_calls
        # telemetry or the OTel LLM span before (or, on DENY, ever after) it.
        # Carry the content-redaction flag on the FROZEN per-turn context (never
        # global state, so concurrent turns are unaffected); it also covers the
        # follow-up tool-synthesis calls that reuse ``resolved_context``.
        # ``isinstance`` guard: a partial test host / double may supply a
        # non-dataclass context (MagicMock) — never crash the turn to set a
        # telemetry flag.
        if buffer_audit and isinstance(resolved_context, LLMInvocationContext):
            resolved_context = replace(resolved_context, redact_content=True)

        # #2530: collect and inject the operator notice LAST, immediately
        # before the provider call it rides on. Collecting DRAINS the
        # producer's pending auto-mode queue and advances its budget/governance
        # dedupe state, so every ``await`` between that drain and the delivery
        # boundary is a window in which a raised exception loses the notice
        # permanently. The non-streaming path closes its window by wrapping the
        # provider call (kestrel_agent.py). Here the turn setup that used to
        # sit in between — tool-call logging, eager image resolution,
        # invocation-context resolution — reads none of this, so it runs FIRST
        # and the window is closed by construction rather than merely guarded.
        # Keep it that way: anything added below this line and above the
        # iteration must be settle-safe.
        inline_tool_executor = self._make_inline_tool_executor(session_id)
        operator_turn = await inject_operator_turn(
            self, messages, context_result, session_id, effective_model, force_local_only
        )

        logging.debug(f"[CONTEXT-STREAM] Sending {len(messages)} messages to LLM")

        # #2530: ``watch_stream`` settles the injected operator notice from
        # this stream's own lifecycle — delivered on the first chunk (the
        # provider accepted the request, so the model provably saw an inline
        # notice), failed/cancelled if the stream dies or closes before one
        # arrives. A stop-button cancel AFTER the first chunk deliberately
        # stays delivered: the notice was carried, and requeuing it would
        # re-report the same fact on the next turn.
        #
        # Constructing the stream is guarded even though nothing above it
        # awaits today. The invariant is "an injected notice always reaches a
        # terminal state", and an invariant that holds only because of the
        # current statement order is one edit away from being false;
        # ``watch_stream`` can only settle from INSIDE the iteration, so the
        # construction needs its own guard.
        try:
            operator_stream = operator_turn.watch_stream(
                self.llm_service.stream_with_tool_detection(
                    messages=messages,
                    tools=feature_tools if feature_tools else None,
                    force_local_only=force_local_only,
                    model_override=effective_model,
                    system_prompt=system_prompt,
                    session_id=session_id,
                    tool_executor=inline_tool_executor,
                    images=eager_images or None,
                    keep_trailing_system=operator_turn.keep_trailing_system,
                    cancel_token=cancel_token,
                    invocation_context=resolved_context,
                )
            )
        except BaseException as exc:
            await operator_turn.settle_interrupted(exc)
            raise

        async for item in operator_stream:
            # #1256: Honor stop-button cancellation INSIDE the agent
            # loop, not just at the HTTP response layer. Before this
            # check existed, /api/agent/stop only set a flag that the
            # endpoint generator inspected — the upstream LLM stream
            # kept pulling tokens (and any tool-using turn kept
            # dispatching tools with side effects) right through to
            # completion. Breaking out here closes the underlying SDK
            # stream and leaves ``tool_response is None`` so the
            # has_tool_calls branch is skipped; partial text already in
            # ``full_response`` flows through to the no-tools persist
            # path below and the cancellation marker lands in metadata.
            if request_id and self.is_request_cancelled(request_id):
                # #2530: close the stream explicitly rather than dropping the
                # reference and waiting on the async-generator finalizer. The
                # provider stream holds the cancel token and the upstream
                # connection, and ``operator_stream`` is a named local that
                # stays alive to the end of this turn, so "stop" must release
                # it here. This does NOT un-deliver the notice — it already
                # settled ``delivered`` on the first chunk and the first
                # terminal settle wins.
                await operator_stream.aclose()
                break
            # #1914: accumulate any part an inline tool emitted while the adapter
            # was resumed to produce THIS item. They flush right before the next
            # VISIBLE text below — AFTER this item if it's a tool sentinel — so a
            # component lands after its producing tool's card, not before it.
            pending_parts.extend(drain_parts())
            if isinstance(item, str):
                # #1914: a component emitted by the tool that produced THIS
                # visible text flushes here — after any tool done/error sentinel
                # items already appended to full_response, before this answer
                # prose — so it lands after its producing tool's card. A
                # sentinel-only item (the tool's done marker) is NOT visible, so
                # the parts wait for the real text that follows it.
                if pending_parts and item.strip() and not is_only_sentinels(item):
                    for _ps in _flush_part_list(pending_parts, full_response):
                        if not buffer_audit:
                            yield _ps
                # #1547: materialize a pending revise boundary lazily —
                # only when real post-marker text lands, and only when it
                # would otherwise weld two non-whitespace chars. Mirrors
                # the client weld so the persisted turn equals the render.
                # #1659: a tool sentinel is wire bytes, not the visible
                # text the weld waits for — never let it materialize (or
                # clear) the boundary, and don't weld a \n\n in front of
                # it (it'd survive the persist-time sentinel strip).
                if (
                    pending_visible_boundary
                    and item.strip()
                    and not is_only_sentinels(item)
                ):
                    acc = "".join(full_response)
                    if acc and not acc[-1].isspace() and not item[:1].isspace():
                        full_response.append("\n\n")
                    pending_visible_boundary = False
                # Text chunk - yield immediately for real-time streaming, or
                # withhold it under an enforcing (strict) POST_RESPONSE audit so
                # no unaudited byte reaches the client before the verdict (#2674).
                # The text is still accumulated into ``full_response`` for the
                # audit + persist regardless.
                full_response.append(item)
                if not buffer_audit:
                    yield item
            elif isinstance(item, ThinkingDelta):
                if not buffer_audit:
                    yield _build_thinking_sentinel(item)
            elif isinstance(item, ToolCallStarted):
                # Honesty-layer signal (#1042 layer 2 / #1045): the LLM
                # has just begun emitting a tool call. Any pre-tool prose
                # the user is currently watching is about to be obsolete
                # — the agent will substitute the tool's actual result.
                # Two idempotent retraction signals tell the client to
                # clear the in-flight bubble: a Wave 5C "revising" SSE
                # event (carries the request_id for pane routing and the
                # marker's `index` for ordering) and a Wave 5E in-band
                # sentinel on the chat stream itself (strictly ordered with
                # the chunks, so the client can never race the post-tool
                # synthesis; chat clients treat both as idempotent).
                #
                # #2674 finding 1: suppress BOTH under an enforcing
                # (buffered) audit. The strict client never received the
                # pre-tool prose — every visible byte was withheld pending
                # the verdict — so there is nothing to retract. And both
                # signals carry response-derived tool id/name the audit has
                # NOT yet approved; the SSE event in particular is delivered
                # (or buffered) on the PARALLEL notifications channel
                # (``/api/agent/notifications/sse``), OUTSIDE the buffered
                # chat stream, so firing it here would leak unaudited —
                # possibly-to-be-denied — tool metadata past the fail-closed
                # gate. Emit both only when streaming live (warn / no-audit),
                # preserving the established behavior there exactly. The
                # sentinel is likewise NOT appended to ``full_response`` (the
                # persisted assistant turn must not contain wire-protocol
                # bytes). The pre-tool accumulation below runs regardless of
                # buffering — the audit + persist still need it.
                if not buffer_audit:
                    await self._emit_revising_event(
                        item, session_id=session_id, request_id=request_id,
                    )
                    yield _build_revise_sentinel(item)
                # Snapshot pre-tool prose at the FIRST marker only —
                # subsequent markers arrive between tool calls of the
                # same LLM turn and don't introduce new pre-tool text
                # the user hadn't already seen. The narration check
                # downstream wants the user-visible prose that
                # PRECEDED any tool call, not the inter-tool prose.
                if pre_tool_prose_snapshot is None:
                    pre_tool_prose_snapshot = "".join(full_response)
                # #1547: arm — do NOT eagerly write — a paragraph boundary
                # at this revise point. It materializes when the next
                # visible text arrives (lazy weld above, or the pre/post
                # seam at the join below), so a turn that stops right here
                # never persists a trailing `\n\n` the client never drew.
                pending_visible_boundary = True
            elif isinstance(item, LLMResponse):
                # Tool calls detected at end of stream
                tool_response = item

        # #1914: flush any still-pending parts (a turn that ended on its tool
        # with no trailing visible text never hit the in-loop flush) plus parts
        # the FINAL resume produced. They append at the end of full_response —
        # after the tool's done sentinel — preserving tool-before-part order.
        pending_parts.extend(drain_parts())
        for _ps in _flush_part_list(pending_parts, full_response):
            if not buffer_audit:
                yield _ps

        # Log LLM response
        llm_duration = int((time.time() - llm_start) * 1000)
        has_tool_calls = tool_response is not None and tool_response.has_tool_calls
        # Inline-executing adapters (codex app-server today; potentially
        # MCP/voice realtime tomorrow) execute tools INSIDE the LLM call
        # and report what they did via an ``executed_tool_calls`` runtime
        # attribute on LLMResponse. Surface that to the same persisted
        # ``tool_events`` metadata channel the chat UI subscribes to.
        inline_executed = (
            getattr(tool_response, "executed_tool_calls", None)
            if tool_response is not None else None
        )

        logging.info(f"[AGENTIC-STREAM] LLM stream complete: has_tool_calls={has_tool_calls}, text_chunks={len(full_response)}")

        await self.observability_store.log_tool_response(
            event_id=llm_event_id,
            success=True,
            duration_ms=llm_duration,
        )

        # #1256: If the user hit Stop after the LLM finished but before
        # we dispatched tools, skip the tool batch entirely. The tools
        # are about to mutate state (send messages, write files, hit
        # external APIs) — running them after the user said "stop" is
        # the surprising-side-effect bug. Persist whatever pre-tool
        # prose the LLM already yielded so the user can see what the
        # agent had been about to do.
        if has_tool_calls and request_id and self.is_request_cancelled(request_id):
            if buffer_audit:
                # #2674: under an enforcing (strict) audit this partial pre-tool
                # prose was WITHHELD from the client and — because we cancel
                # before dispatch, never reaching ``_fire_post_response_hook`` —
                # was never audited. Persisting it would leak unreviewed content
                # back on history reload (and via ContextBuilder into the next
                # turn), defeating the fail-closed gate. Record the cancellation
                # with NO assistant content; ``_persist_assistant_turn_safely``
                # still stamps the ``cancelled`` marker.
                cancelled_text = ""
            else:
                # #1659: strip any codex inline tool sentinels from the
                # partial pre-tool text before persisting.
                cancelled_text, _, _ = _parse_stream_sentinels("".join(full_response))
            await self._persist_assistant_turn_safely(
                cancelled_text, metadata=None, session_id=session_id,
                request_id=request_id,
                response=tool_response,
            )
            return

        # Handle tool calls if present (A2A pattern)
        if has_tool_calls:
            logging.info(f"[AGENTIC-STREAM] Tool calls: {[tc.name for tc in tool_response.tool_calls]}")
            for tc in tool_response.tool_calls:
                await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"llm_tool_call:{tc.name}",
                    metadata={"arguments": tc.arguments}
                )

            # Stream tokens as they arrive from LLM after tool execution.
            # ``tool_results`` accumulates each call's serialized return
            # envelope as ``{tool_call_id, name, result}``. The orchestrator
            # writes into it from ``_dispatch_tool_call`` so the post-
            # response hook can run the narration check (#1042 layer 3)
            # over the same envelopes the LLM saw.
            tool_response_chunks = []
            tool_events = []
            tool_results: list = []
            # #2674 finding 2: per-turn channel for a strict continuation-timeout
            # signal. In advisory mode a follow-up-LLM timeout renders as an ❌
            # tool card; under an enforcing (buffered) audit that sentinel is
            # stripped, so the orchestrator sets ``timed_out`` here instead and we
            # substitute a deterministic safe block below.
            strict_timeout_state: Dict[str, Any] = {}
            async for chunk in self._handle_orchestrator_response_streaming(
                response=tool_response,
                feature_tools=feature_tools,
                system_prompt=system_prompt,
                force_local_only=force_local_only,
                effective_model=effective_model,
                user_message=user_input,
                tool_events=tool_events,
                tool_results=tool_results,
                session_id=session_id,
                request_id=request_id,
                images=eager_images or None,
                invocation_context=resolved_context,
                buffer_audit=buffer_audit,
                strict_timeout_state=strict_timeout_state,
            ):
                if isinstance(chunk, ThinkingDelta):
                    if not buffer_audit:
                        yield _build_thinking_sentinel(chunk)
                    continue
                tool_response_chunks.append(chunk)
                if not buffer_audit:
                    yield chunk
                # #1256: belt-and-suspenders cancel check at the outer
                # boundary. The inner orchestrator generator also checks
                # ``is_request_cancelled`` at iteration boundaries; this
                # outer break stops accumulating tokens if a cancel
                # arrives between the inner check and the next yield.
                if request_id and self.is_request_cancelled(request_id):
                    break
            # #2674 finding 2: a strict (buffered) POST-TOOL continuation stopped
            # before its reviewed release withheld every byte and never audited
            # the partial synthesis. Discard the whole withheld buffer (prose,
            # parts, tool_events, tool_results) and persist an EMPTY cancelled row
            # — matching the strict cancel-before-dispatch path — instead of
            # persisting the unfinished synthesis (which would resurface on reload
            # and replay into the next turn's context). Nothing is released, so the
            # endpoint surfaces its standard stop notice exactly once. Return
            # before the audit fire / STOP hook / memory pipeline, none of which
            # should run over discarded content. Advisory turns already streamed
            # their partial, so this is gated on ``buffer_audit``. Inlined (not a
            # helper method) so the MagicMock-based streaming harnesses run the
            # REAL check via the bound ``is_request_cancelled`` /
            # ``_persist_assistant_turn_safely`` instead of an auto-truthy mock.
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # #2674 finding 2: a strict (buffered) continuation that TIMED OUT
            # withheld every byte and yielded no reviewable text. Discard the
            # withheld continuation prose entirely — any partial buffered text or
            # typed parts are protected, unfinished content the audit should not
            # release — and replace the REVIEWABLE text with a single
            # deterministic safe block. It then flows through the SAME buffered
            # audit → persist → release path as any other turn, so persisted ==
            # reloaded == released == delivered, the client gets a non-empty safe
            # body (never a silent empty 200), and no protected marker, tool
            # argument, or raw error can escape. ``tool_events`` / ``tool_results``
            # are left intact: the buffered persist drops the whole metadata
            # envelope (``meta = None`` below) so they never reach the client or
            # history, while the audit narration check and the STOP-hook payload
            # keep seeing the real tool activity that actually ran.
            if buffer_audit and strict_timeout_state.get('timed_out'):
                tool_response_chunks = [STRICT_AUDIT_CONTINUATION_TIMEOUT_BLOCK]
            # Persist only the post-tool synthesis in ``content``. Any
            # pre-tool prose was already retracted from the SSE client by
            # the ToolCallStarted/revising protocol, so storing it in
            # content makes the lie reappear on history reload. Keep that
            # prose in metadata instead; ContextBuilder replays it to the
            # next-turn LLM context so #877 self-recall is preserved.
            # #1547/#1659: strip + weld in-band sentinels from BOTH halves
            # and recover each tool sentinel's clean-text position. The pre
            # half can carry codex inline tool sentinels (emitted on the LLM
            # stream itself); the post half carries the orchestrator's
            # per-batch start sentinels plus any follow-up revise sentinels.
            pre_tool_text, pre_parts, pre_components = _parse_stream_sentinels(
                "".join(full_response)
            )
            post_tool_text, post_parts, post_components = _parse_stream_sentinels(
                "".join(tool_response_chunks)
            )
            # Materialize the pre/post seam boundary armed by the first
            # ToolCallStarted marker — but ONLY when the marker fired
            # (`pending_visible_boundary`) and the seam would weld two
            # non-whitespace chars. This mirrors the client, which welds
            # at the in-band revise sentinel; when no marker fired the
            # client glues, so we glue too (plain concat).
            if (
                pending_visible_boundary
                and pre_tool_text
                and post_tool_text
                and not pre_tool_text[-1].isspace()
                and not post_tool_text[:1].isspace()
            ):
                seam = "\n\n"
            else:
                seam = ""
            context_replay_text = pre_tool_text + seam + post_tool_text
            pending_visible_boundary = False
            # Position-stamp tool_events so the renderer can place each card
            # back between the right prose segments (live + on reload). Pre
            # positions already index the final text (pre half is at the
            # front); post positions shift past the clean pre half + seam.
            post_base = len(pre_tool_text) + len(seam)
            combined_parts = (
                list(pre_parts)
                + [{**p, "pos": post_base + p["pos"]} for p in post_parts]
            )
            _stamp_tool_event_positions(tool_events, combined_parts)
            # #1914: typed component parts are positioned relative to the
            # PERSISTED content (``tool_final_text`` — the post-tool synthesis),
            # NOT the pre+post combined text, so a reload places each component
            # bubble correctly against the prose it sits in. The collector is
            # drained by the orchestrator AFTER the tool batch, so parts only
            # ever ride the post-tool stream; ``post_components`` already index
            # the post half. (Any pre-half PART bytes index retracted pre-tool
            # prose that isn't persisted in content, so they're not carried.)
            component_parts = list(post_components)
            # Narration check (#1042 layer 3): the hook receives the
            # pre-tool prose snapshot taken at the first ToolCallStarted
            # marker boundary, plus the tool calls + result envelopes.
            # If the marker never fired (model emitted tool_use without
            # streaming markers), fall back to the pre-tool half of
            # full_response — that's still the text the user saw before
            # the tool ran.
            pre_tool_for_audit = (
                _parse_stream_sentinels(pre_tool_prose_snapshot)[0]
                if pre_tool_prose_snapshot is not None
                else pre_tool_text
            )
            tool_calls_payload = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_response.tool_calls
            ]
            tool_final_text = await self._fire_post_response_hook(
                post_tool_text, session_id,
                pre_tool_prose=pre_tool_for_audit,
                tool_calls=tool_calls_payload,
                tool_results=tool_results,
                hook_snapshot=audit_hook_snapshot,
            )
            # #2674 P0-2: cancellation can arrive WHILE the audit provider call is
            # in flight (``get_audit_response`` hangs, user hits stop, the audit
            # then returns ALLOW). The pre-dispatch/pre-audit cancel checks above
            # ran BEFORE this await, so they cannot see it. Under ``buffer_audit``
            # nothing streamed live and the strict release below is suppressed on
            # cancel — but the persist would still store the reviewed-yet-withheld
            # prose (resurfacing on reload AND replaying into the next turn's
            # context) with a ``cancelled`` marker, the exact split this gate
            # exists to prevent. Recheck AFTER the await: discard the reviewed
            # text / parts / tool data and persist exactly ONE empty cancelled row
            # (matching the strict cancel-before-dispatch path), then return
            # before metadata / persist / release / memory / STOP. Inlined (not a
            # helper) so the MagicMock streaming harnesses run the REAL bound
            # ``is_request_cancelled`` / ``_persist_assistant_turn_safely``.
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # #2674: read the EXPLICIT audit verdict, not string equality.
            audit_denied = getattr(tool_final_text, "denied", False)
            # #1914 (codex P1): a POST_RESPONSE hook can rewrite or BLOCK the
            # assistant text (e.g. an audit denial → "[Response blocked ...]").
            # The component parts were positioned against the ORIGINAL post-tool
            # prose; persisting them would re-render the structured bubbles next
            # to replaced/blocked text on reload — leaking exactly what the hook
            # removed. Drop them whenever the hook changed OR denied the text
            # (``audit_denied`` covers the equality-collision case where a block
            # message happens to equal the original prose — #2674).
            if audit_denied or tool_final_text != post_tool_text:
                component_parts = []
            # #1914: UTF-16-offset the positions for the browser + fold in any
            # parts a POST_RESPONSE hook emitted. #2674: on a DENY, DISCARD the
            # hook-emitted parts too — a prior POST_RESPONSE hook can smuggle the
            # denied text into a part, and clearing ``component_parts`` above does
            # not touch the collector those parts still ride.
            component_parts = _finalize_component_parts(
                component_parts, tool_final_text, keep_hook_parts=not audit_denied,
            )
            # #1914: when this turn carries parts, rebase the tool cards onto the
            # same post-tool UTF-16 origin so the reload merge orders both
            # streams consistently (``post_base`` is the retracted pre half).
            if component_parts:
                tool_events = _rebase_events_for_parts(
                    tool_events, post_base, tool_final_text,
                )
            # tool_results carry the structured call/result envelopes;
            # persisting them alongside tool_events means a future
            # conversation-history reader can reconstruct *which tools
            # produced which output*, not just *that some tool ran*.
            # Without this the actual result content was lost after the
            # turn — same gap the other-agent diagnosis flagged for the
            # codex inline_executed branch, but it applied identically
            # here (every adapter that goes through the multi-iteration
            # tool loop). Close it in both branches uniformly.
            meta: Optional[Dict[str, Any]] = None
            if tool_events or tool_results or pre_tool_text or component_parts:
                meta = {}
                # #2674: on a fail-closed DENY the original assistant prose was
                # REJECTED, so it must not persist — ContextBuilder reinjects
                # ``pre_tool_reasoning.content`` (and its ``context_replay``) into
                # the next turn's LLM context, which would smuggle the denied
                # text forward despite the blocked assistant content. Drop it on
                # DENY; keep only the structural (non-content-bearing) events.
                if pre_tool_text and not audit_denied:
                    meta['pre_tool_reasoning'] = {
                        'content': pre_tool_text,
                        'seam': seam,
                        'context_replay': context_replay_text,
                    }
                if tool_events:
                    meta['tool_events'] = tool_events
                if tool_results:
                    meta['tool_results'] = tool_results
                if component_parts:
                    meta['parts'] = component_parts
                if not meta:
                    meta = None
            # #2674 findings 2 & 3: under an enforcing (buffered) audit the client
            # received ONLY the reviewed release (``_strict_audit_release``) —
            # nothing streamed live. Every other channel here is UNREVIEWED by the
            # audit, which examined only the sentinel-stripped ``post_tool_text``:
            # typed ``parts`` carry structured tool data, ``pre_tool_reasoning``
            # replays into the next turn's LLM context AND renders on reload,
            # ``tool_events`` carry tool exception strings the frontend renders,
            # ``tool_results`` carry result payloads. Persisting any of them would
            # leak material the audit never saw (ALLOW) or explicitly rejected
            # (DENY/ASK), and make the reloaded turn richer than what was
            # released. Drop all of it so persisted == reloaded == released
            # reviewed text. Warn / no-audit turns (``buffer_audit`` False) keep
            # their metadata exactly as before.
            if buffer_audit:
                meta = None
            await self._persist_assistant_turn_safely(
                tool_final_text, metadata=meta, session_id=session_id,
                request_id=request_id,
                response=tool_response,
            )
            # #2674: strict-audit release. Nothing visible was streamed live this
            # turn; now that the POST_RESPONSE verdict exists, release ONLY the
            # reviewed text (block message on DENY, reviewed prose on
            # ALLOW/MODIFY). The withheld raw buffer is never replayed — see
            # ``_strict_audit_release``. Suppressed on cancellation — a stopped,
            # unaudited partial must not surface. No-op when buffering was off.
            if buffer_audit and not (
                request_id and self.is_request_cancelled(request_id)
            ):
                for _c in _strict_audit_release(tool_final_text):
                    yield _c
            final_assistant_text = tool_final_text
            stop_tool_results: Optional[list] = tool_results
            # Derive stop_tool_calls from the accumulated tool_results
            # envelopes — multi-iteration tool flows (model calls A,
            # sees result, calls B) require this so tool_calls and
            # tool_results stay aligned across all iterations.
            # Cancellation edge: when the request was stopped after the
            # initial LLM emitted tool_calls but before any result was
            # appended, `tool_results` is empty but the LLM *did* emit
            # tools. Fall back to the initial ``tool_calls_payload`` in
            # that case so a STOP subscriber can distinguish
            # "cancelled mid-tool" from "no tools at all" (codex review).
            if tool_results:
                stop_tool_calls: Optional[list] = [
                    {
                        "id": env.get("tool_call_id"),
                        "name": env.get("name"),
                        "arguments": env.get("arguments"),
                    }
                    for env in tool_results
                ]
            elif tool_calls_payload:
                stop_tool_calls = list(tool_calls_payload)
            else:
                stop_tool_calls = None
        elif inline_executed:
            # #2674 finding 2: a strict (buffered) inline-executed turn stopped
            # before its reviewed release withheld every byte and never audited
            # the synthesis. Discard the withheld buffer and persist an EMPTY
            # cancelled row (same as the post-tool and no-tool paths) rather than
            # the unfinished, unreviewed content. Gated on ``buffer_audit`` so
            # advisory turns — which already streamed incrementally — are
            # unchanged.
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # Inline-executed branch: the adapter ran tools mid-call
            # (codex app-server's item/tool/call RPC). No
            # ``_dispatch_tool_call`` ran, so synthesize the same
            # ``tool_events`` / ``tool_results`` envelopes the
            # orchestrator-dispatched path would have produced and
            # persist via the metadata path the chat UI already reads.
            #
            # Also persist into ``a2a_tool_dispatches`` via
            # ``_log_tool_dispatch`` so structured observability sees
            # the dominant streaming-path execution today (gpt-5.5 plan
            # / codex app-server). Without this, the inline branch
            # silently bypassed the structured log and runtime
            # tool-call queries returned stale data from before the
            # inline-execution migration.
            from kestrel_sovereign.agent.orchestrator_engine import (
                OrchestratorEngineMixin,
                _extract_inline_error_fields,
            )
            from kestrel_sovereign.features.base import _serialize_tool_result
            from kestrel_sovereign.security.narration_check import (
                summarize_tool_result_for_audit,
            )
            synth_tool_events = []
            tool_results: list = []
            features_by_tool_name = (
                self._visible_features_by_tool_name()
                if hasattr(self, "_visible_features_by_tool_name") else {}
            )
            for e in inline_executed:
                synth_tool_events.append({"type": "start", "tool": e["name"]})
                synth_tool_events.append({"type": "complete", "tool": e["name"], "ms": 0})
                serialized = _serialize_tool_result(e["result"])
                tool_results.append({
                    "tool_call_id": e["id"],
                    "name": e["name"],
                    "arguments": e["arguments"],
                    "result": summarize_tool_result_for_audit(serialized),
                })
                # Structured dispatch log (codex app-server inline path).
                # Latency=0 by design — accurate per-call latency needs
                # adapter-level instrumentation, see codex_adapter's
                # ``_make_inline_tool_executor``.
                args = e["arguments"]
                adapter = "inline:" + OrchestratorEngineMixin._tool_call_adapter(
                    self, e["name"], features_by_tool_name,
                )
                err_msg, err_class = _extract_inline_error_fields(e["result"])
                await OrchestratorEngineMixin._log_tool_dispatch(
                    self,
                    tool_name=e["name"],
                    adapter=adapter,
                    args=args if isinstance(args, dict) else {},
                    result=e["result"],
                    session_id=session_id,
                    dispatch_start=time.time(),
                    error_class=err_class,
                    error_message=err_msg,
                )
            # #1659: the codex tool sentinels rode the LLM stream itself, so
            # they're embedded in full_response. Strip them out of the
            # persisted text and recover their positions to stamp onto the
            # synthesized events (inline_executed order == start-sentinel
            # order on the wire).
            context_replay_text, inline_parts, inline_components = _parse_stream_sentinels(
                "".join(full_response)
            )
            _stamp_tool_event_positions(synth_tool_events, inline_parts)
            tool_calls_payload = [
                {"id": e["id"], "name": e["name"], "arguments": e["arguments"]}
                for e in inline_executed
            ]
            pre_tool_text = _parse_stream_sentinels(
                pre_tool_prose_snapshot or ""
            )[0]
            post_tool_text = context_replay_text
            seam = ""
            # Bytes trimmed off the front of context_replay to get the persisted
            # post-tool content — used to rebase component-part positions onto
            # it (#1914), so a reload places each bubble against the prose it
            # actually sits in rather than the retracted pre-tool half.
            inline_pre_len = 0
            if pre_tool_text and context_replay_text.startswith(pre_tool_text):
                post_tool_text = context_replay_text[len(pre_tool_text):]
                inline_pre_len = len(pre_tool_text)
                if post_tool_text.startswith("\n\n"):
                    seam = "\n\n"
                    post_tool_text = post_tool_text[2:]
                    inline_pre_len += 2
            # #1914: rebase inline component parts onto the persisted post-tool
            # content; drop any landing in the retracted pre-tool region.
            inline_post_components = [
                {**p, "pos": p["pos"] - inline_pre_len}
                for p in inline_components
                if p.get("pos", 0) >= inline_pre_len
            ]
            final_text = await self._fire_post_response_hook(
                post_tool_text, session_id,
                pre_tool_prose=pre_tool_text,
                tool_calls=tool_calls_payload,
                tool_results=tool_results,
                hook_snapshot=audit_hook_snapshot,
            )
            # #2674 P0-2: cancellation that lands WHILE the audit await was in
            # flight is invisible to the pre-audit check above. Recheck here and
            # discard the withheld inline synthesis — persist exactly one empty
            # cancelled row and return before metadata / persist / release /
            # memory / STOP (see the has_tool_calls branch for the full rationale).
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # #2674: read the EXPLICIT audit verdict, not string equality.
            inline_denied = getattr(final_text, "denied", False)
            # #1914 (codex P1): drop component parts when a POST_RESPONSE hook
            # rewrote/blocked the text — their positions no longer match the
            # persisted content and would leak blocked structured data on reload.
            # ``inline_denied`` also covers the equality-collision case (#2674).
            if inline_denied or final_text != post_tool_text:
                inline_post_components = []
            # #1914: UTF-16-offset + fold in hook-emitted parts. #2674: on a DENY,
            # discard hook-emitted parts too (they can carry the denied text).
            inline_post_components = _finalize_component_parts(
                inline_post_components, final_text, keep_hook_parts=not inline_denied,
            )
            # #1914: when parts ride along, rebase the tool cards onto the same
            # post-tool UTF-16 origin so the reload merge stays consistent.
            if inline_post_components:
                synth_tool_events = _rebase_events_for_parts(
                    synth_tool_events, inline_pre_len, final_text,
                )
            # See parallel comment in the has_tool_calls branch above:
            # tool_results in the persisted metadata preserves the
            # actual call envelopes (id, name, arguments, summarized
            # result) so a future loader can see what produced what,
            # not just *that* something ran.
            inline_meta: Optional[Dict[str, Any]] = {
                # #2674: drop the rejected pre-tool prose on a DENY — see the
                # has_tool_calls branch; ContextBuilder would otherwise replay
                # the denied content into the next turn's context.
                **({
                    "pre_tool_reasoning": {
                        "content": pre_tool_text,
                        "seam": seam,
                        "context_replay": context_replay_text,
                    }
                } if (pre_tool_text and not inline_denied) else {}),
                "tool_events": synth_tool_events,
                "tool_results": tool_results,
                # #1914: component parts emitted during the inline turn,
                # rebased onto the persisted post-tool content.
                **({"parts": inline_post_components} if inline_post_components else {}),
            }
            # #2674 findings 2 & 3: under an enforcing (buffered) audit persist
            # ONLY the reviewed release — drop every unreviewed side channel
            # (parts, pre_tool_reasoning, tool_events incl. tool exception
            # strings, tool_results). Same invariant as the has_tool_calls
            # branch: persisted == reloaded == released reviewed text.
            if buffer_audit:
                inline_meta = None
            await self._persist_assistant_turn_safely(
                final_text,
                metadata=inline_meta,
                session_id=session_id, request_id=request_id,
                response=tool_response,
            )
            # #2674: strict-audit release (inline-executed path) — only the
            # reviewed text; the withheld raw buffer is never replayed.
            if buffer_audit and not (
                request_id and self.is_request_cancelled(request_id)
            ):
                for _c in _strict_audit_release(final_text):
                    yield _c
            final_assistant_text = final_text
            stop_tool_results = tool_results
            stop_tool_calls = tool_calls_payload
        else:
            # #2674 finding 2: a strict (buffered) NO-TOOL turn stopped mid-stream
            # withheld every byte and never audited the partial synthesis. Under
            # ``buffer_audit`` nothing streamed live, so a cancellation here means
            # the buffered partial was never released; persisting it as content
            # would resurface it on reload and replay into the next turn's context.
            # Discard it and persist an EMPTY cancelled row — matching the strict
            # cancel-before-dispatch path — before the audit fire / STOP hook /
            # memory pipeline. Advisory turns already streamed their partial, so
            # this is gated on ``buffer_audit`` and their behavior is unchanged.
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # No tool calls - text was already streamed above. Pre-tool
            # prose / tool_calls / tool_results all stay None: a hook
            # writing the narration check should treat ``tool_results
            # is None`` as "this turn didn't call any tools, no
            # narration to verify". (#1659: strip in-band sentinels from the
            # persisted text.) Codex-native tools (commandExecution, fileChange,
            # webSearch) that don't populate executed_tool_calls land here with
            # their TOOL sentinels embedded — build tool_events from them so the
            # cards survive reload rather than vanishing once stripped.
            final_text, tool_parts, no_tool_components = _parse_stream_sentinels(
                "".join(full_response)
            )
            _no_tool_pre_hook_text = final_text
            final_text = await self._fire_post_response_hook(
                final_text, session_id,
                hook_snapshot=audit_hook_snapshot,
            )
            # #2674 P0-2: cancellation that lands WHILE the audit await was in
            # flight is invisible to the pre-audit check above. Recheck here and
            # discard the withheld reviewed prose — persist exactly one empty
            # cancelled row and return before metadata / persist / release /
            # memory / STOP (see the has_tool_calls branch for the full rationale).
            if buffer_audit and request_id and self.is_request_cancelled(
                request_id
            ):
                await self._persist_assistant_turn_safely(
                    "", metadata=None, session_id=session_id,
                    request_id=request_id, response=tool_response,
                )
                return
            # #2674: read the EXPLICIT audit verdict, not string equality.
            no_tool_denied = getattr(final_text, "denied", False)
            # #1914 (codex P1): drop parts when a POST_RESPONSE hook rewrote or
            # blocked the text — their positions reference the original prose, so
            # persisting them would leak blocked structured data on reload.
            # ``no_tool_denied`` also covers the equality-collision case (#2674).
            if no_tool_denied or final_text != _no_tool_pre_hook_text:
                no_tool_components = []
            # #1914: UTF-16-offset + fold in any parts the POST_RESPONSE hook
            # emitted (the collector is still active while the hook runs). #2674:
            # on a DENY, discard hook-emitted parts too (they can carry the denied
            # text — clearing ``no_tool_components`` above doesn't drain the
            # collector).
            no_tool_components = _finalize_component_parts(
                no_tool_components, final_text, keep_hook_parts=not no_tool_denied,
            )
            _no_tool_events = _tool_parts_to_events(tool_parts)
            # #1914: with parts present, convert the tool cards to the same
            # UTF-16 origin (base 0 here — no pre-tool retraction) so the reload
            # merge orders both consistently.
            if no_tool_components:
                _no_tool_events = _rebase_events_for_parts(_no_tool_events, 0, final_text)
            # #1914: a turn can emit component parts without dispatching tools
            # (e.g. a feature emitting a card from a hook). Persist them here so
            # those bubbles survive reload too. ``final_text`` is the clean
            # persisted content, so the parts' ``pos`` index it directly.
            _no_tool_meta: Optional[Dict[str, Any]] = None
            if _no_tool_events or no_tool_components:
                _no_tool_meta = {}
                if _no_tool_events:
                    _no_tool_meta["tool_events"] = _no_tool_events
                if no_tool_components:
                    _no_tool_meta["parts"] = no_tool_components
            # #2674 findings 2 & 3: under an enforcing (buffered) audit persist
            # ONLY the reviewed release — drop the unreviewed side channels
            # (codex-native ``tool_events`` and typed ``parts``). Same invariant
            # as the tool branches: persisted == reloaded == released text.
            if buffer_audit:
                _no_tool_meta = None
            await self._persist_assistant_turn_safely(
                final_text,
                metadata=_no_tool_meta,
                session_id=session_id,
                request_id=request_id,
                response=tool_response,
            )
            # #2674: strict-audit release (no-tool path) — only the reviewed
            # text; the withheld raw buffer is never replayed.
            if buffer_audit and not (
                request_id and self.is_request_cancelled(request_id)
            ):
                for _c in _strict_audit_release(final_text):
                    yield _c
            final_assistant_text = final_text
            stop_tool_results = None
            stop_tool_calls = None

        # Post-response memory pipeline — parity with the non-streaming path
        # (kestrel_agent._post_response_pipeline). Routed through the SINGLE
        # privacy gate so EPHEMERAL/ISOLATED (and deidentified) turns never
        # derive durable state (temporal patterns, concept-graph nodes,
        # emotional metadata, embeddings) from raw user input. The guard is
        # also the gate's call site for the streaming path: when it returns
        # True the pipeline is never entered, so no background memory work is
        # scheduled. Mirrors the non-streaming fire at kestrel_agent.py.
        post_pipeline = getattr(self, "_post_response_pipeline", None)
        if callable(post_pipeline) and not self._privacy_blocks_background_memory():
            await post_pipeline(user_input, final_assistant_text, session_id)

        # Fire STOP hook (streaming response cycle complete).
        #
        # STOP HookInput carries the turn's user message, the final visible
        # assistant text, and (when applicable) tool_calls + tool_results
        # so per-turn subscribers — e.g. the kestrel-feature-reflection
        # `on_stop` handler for #1238 — don't have to round-trip through
        # storage to reconstruct the turn that just completed. Without
        # this, every subscriber would issue its own query against
        # `conversation_history` / `a2a_tool_dispatches` at hook-fire
        # time, multiplying read load and risking races with the
        # persistence write above. SDK HookInput already declares
        # `user_message`, `response_text`, `tool_calls`, `tool_results`
        # fields (used by POST_RESPONSE) — we're just populating them
        # for STOP too.
        hooks_manager = getattr(self, "hooks_manager", None)
        if hooks_manager:
            # #2674 finding 2: a STOP hook is an ARBITRARY subscriber that may
            # persist or emit its HookInput (e.g. the reflection ``on_stop``
            # handler). Under an enforcing (buffered) audit it must receive ONLY
            # the reviewed final text. ``tool_calls`` / ``tool_results`` carry
            # raw, unaudited tool envelopes — ids, arguments, full result
            # payloads (incl. tool exception strings) — that the POST_RESPONSE
            # audit never saw, and on a DENY explicitly rejected. Forwarding them
            # would leak the same unreviewed side channel the persist path now
            # drops (findings 2/3), just through the hook instead of storage.
            # Null BOTH on every verdict (ALLOW / MODIFY / ASK / DENY) and every
            # branch (tool / inline / no-tool): ``buffer_audit`` is the enforcing
            # flag, independent of the verdict. ``final_assistant_text`` is
            # already the reviewed release (block message on DENY, rewrite on
            # MODIFY, approved prose on ALLOW), so ``response_text`` stays safe.
            # Warn / no-audit turns (``buffer_audit`` False) keep the full STOP
            # payload exactly as before.
            stop_hook_tool_calls = None if buffer_audit else stop_tool_calls
            stop_hook_tool_results = None if buffer_audit else stop_tool_results
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
                user_message=user_input,
                response_text=final_assistant_text,
                tool_calls=stop_hook_tool_calls,
                tool_results=stop_hook_tool_results,
            )
            await hooks_manager.execute_hooks_parallel(
                HookEvent.STOP, hook_input
            )

    async def _emit_revising_event(
        self,
        marker: ToolCallStarted,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Emit a ``revising`` event when a ToolCallStarted marker arrives.

        Routed through ``self.emit_event`` so the existing
        ``/api/agent/notifications/sse`` channel forwards it to the
        browser as ``event: revising / data: {...}``. Frontend (Wave 5C)
        listens on that channel and clears the in-flight optimistic
        message bubble for the matching ``request_id``.

        ``request_id`` is the routing key. Callers should always pass
        the endpoint-assigned id for the stream that produced the
        marker — the legacy ``self._current_request_id`` fallback is
        agent-global and is overwritten by the next concurrent stream's
        ``register_active_request`` call, so two overlapping streams
        could otherwise misroute events to the wrong pane.

        Failures here must never break the chat stream — the user is
        already receiving text through the parallel channel. ``emit_event``
        itself swallows per-listener errors; this wrapper additionally
        guards the case where the host class doesn't expose ``emit_event``
        at all (e.g. tests mocking a bare agent).
        """
        emit_event = getattr(self, "emit_event", None)
        if not callable(emit_event):
            return
        effective_request_id = request_id
        if effective_request_id is None:
            effective_request_id = getattr(self, "_current_request_id", None)
        try:
            await emit_event("revising", {
                "type": "revising",
                "request_id": effective_request_id,
                "session_id": session_id,
                "index": marker.index,
                "tool_call_id": marker.id,
                "tool_name": marker.name,
            })
        except Exception:
            logging.warning(
                "Failed to emit revising event for index=%s",
                getattr(marker, "index", None), exc_info=True,
            )

    async def _persist_assistant_turn_safely(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        response: Optional[LLMResponse] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Persist the assistant turn under cancellation-safe handling.

        FastAPI's ``StreamingResponse`` cancels the wrapping async
        generator on client disconnect — browser nav, agent switch,
        network blip. Without protection, a disconnect after the last
        chunk is yielded but BEFORE this insert completes would lose
        the assistant turn entirely: the user already saw the response,
        but the next turn's history loader can't find it. Surfaced by
        Meridian's "I don't see my own quantum response" report.

        ``asyncio.shield`` keeps the persist task alive even when the
        outer generator is cancelled. The exception handler logs and
        emits a metric so production can detect this happening at all.
        We never re-raise — the request is already over from the
        client's perspective; raising here would only mask the original
        cancellation.

        When ``request_id`` is supplied and the request has been
        cancelled by the user (stop button), stamp ``cancelled: True``
        into the persisted metadata. The next turn's history loader and
        any UI surface that re-renders the turn can then distinguish a
        user-aborted partial from a normally-completed turn. The check
        runs here rather than in the caller so the marker lands even on
        the disconnect-before-persist path that the shield protects.
        """
        if request_id and self.is_request_cancelled(request_id):
            metadata = {**(metadata or {}), "cancelled": True}
        identity = {}
        get_identity = getattr(self, "_conversation_response_identity", None)
        if (
            callable(get_identity)
            and getattr(type(get_identity), "__module__", "") != "unittest.mock"
        ):
            identity = get_identity(response, use_last_identity=True)
        kwargs = {
            "metadata": metadata,
            "session_id": session_id,
        }
        resolved_model = model or identity.get("model")
        resolved_provider = provider or identity.get("provider")
        if resolved_model is not None:
            kwargs["model"] = resolved_model
        if resolved_provider is not None:
            kwargs["provider"] = resolved_provider
        try:
            await asyncio.shield(
                self.privacy_agent.add_conversation(
                    "assistant",
                    text,
                    **kwargs,
                )
            )
        except asyncio.CancelledError:
            # Outer task cancelled mid-persist. shield() means the
            # add_conversation coroutine keeps running to completion in
            # the background — we just don't get to await it. Re-raise
            # so the cancellation propagates correctly.
            raise
        except Exception as exc:
            logging.error(
                "Failed to persist assistant turn (session_id=%s): %s",
                session_id, exc, exc_info=True,
            )
            try:
                await self.observability_store.log_metric(
                    agent_name=self.did,
                    metric_name="assistant_turn_persist_failed",
                    metric_value=1.0,
                    metadata={
                        "session_id": session_id or "",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc)[:500],
                    },
                )
            except Exception:
                # Telemetry failures must never propagate from a
                # post-response persist path. If observability is also
                # broken, the logged ERROR above is the last line of
                # defense.
                pass

    async def _fire_post_response_hook(
        self,
        response_text: str,
        session_id: str = None,
        *,
        pre_tool_prose: Optional[str] = None,
        tool_calls: Optional[list] = None,
        tool_results: Optional[list] = None,
        hook_snapshot: Optional[list] = None,
    ) -> str:
        """Fire POST_RESPONSE hooks on completed response text.

        Returns a ``_PostResponseText`` — a ``str`` subclass carrying the
        reviewed text plus an EXPLICIT verdict (``denied`` / ``modified``): the
        block message on DENY, the rewritten text on MODIFY, otherwise the input
        unchanged. Callers treat it as a plain string for persistence; the
        strict streaming path reads ``.denied`` to decide the release/metadata
        rules rather than inferring the verdict from string equality (#2674).

        The caller decides what reaches the live client:

        * When an **enforcing** (fail-closed) POST_RESPONSE hook is active, the
          streaming loop withheld every visible byte until this verdict (#2674),
          so ONLY the returned text is released — a DENY exposes none of the
          original assistant text.
        * When only advisory/warn hooks are active, the text was already
          streamed incrementally, so DENY only changes what is stored/logged and
          MODIFY only annotates the persisted copy.

        Args:
            response_text: Final assembled assistant text (post-tool
                synthesis when tools fired; otherwise the only text).
            session_id: Active session id.
            pre_tool_prose: Text the agent streamed before the first
                ``ToolCallStarted`` marker arrived. ResponseAuditHook
                narration check (#1042 layer 3) compares this against
                the actual ``tool_results`` to catch confident-lie
                patterns ("Saved!" before the tool returned). ``None``
                or empty string means no pre-tool prose was streamed.
            tool_calls: Tool calls issued in this turn, as JSON-shaped
                dicts ``{id, name, arguments}``.
            tool_results: Tool result envelopes observed in this turn,
                as ``{tool_call_id, name, result}`` dicts. ``result``
                is whatever ``_serialize_tool_result`` produced —
                usually a ``ToolResult.to_dict()`` envelope (#1042
                layer 4) or a legacy ``{success, ...}`` dict.
            hook_snapshot: ``[(hook, mode, enforcing), ...]`` — the enabled
                POST_RESPONSE hook set captured at turn start (#2674).
                When provided (the streaming path always passes it, even
                empty), ONLY this set audits the turn, run via
                ``execute_hooks_snapshot`` and pinned to its turn-start
                modes: a hook a tool *registers* / *enables* / *disables*
                mid-turn cannot change what enforces on a turn whose
                buffering decision was already fixed from this same
                snapshot (the transition takes effect next turn).
                ``None`` (non-streaming / legacy callers) runs the live
                enabled set and pins nothing.

        Returns:
            Possibly modified response_text.
        """
        hooks_manager = getattr(self, "hooks_manager", None)
        if not hooks_manager:
            return _PostResponseText(response_text)

        # #2674: a provided snapshot (even empty) freezes the audited hook set to
        # turn start; ``None`` falls back to the live enabled set for
        # non-streaming / legacy callers. An empty snapshot means no hook was
        # enabled at turn start — nothing enforces, regardless of what the tools
        # registered/enabled since.
        pinned = hook_snapshot is not None
        # #2674 P0-1: when pinned, the enforcement flag CAPTURED at turn start —
        # ``entry[2]`` — is AUTHORITATIVE for this turn: it drove the buffering
        # (withhold) decision, so it must also drive the returned verdict, the
        # gate/observer partition, AND the manager's crash/timeout fail-closed
        # resolution. A live ``_hook_is_enforcing(h)`` re-read here could disagree
        # (a generic hook whose ``fail_closed`` flips to False mid-turn, or an
        # approval hook that drops ``awaits_user_input`` — neither is mode-pinned)
        # and produce the raw-streamed / block-persisted fail-open split. Key the
        # override by ``id(hook)`` (each hook is held alive by the snapshot for
        # the whole fire, so identity is stable). Live callers (``None`` snapshot)
        # keep reading the current flag.
        captured_enforcing: Optional[dict] = None
        if pinned:
            snapshot_hooks = [entry[0] for entry in hook_snapshot]
            captured_enforcing = {
                id(entry[0]): bool(entry[2]) for entry in hook_snapshot
            }
            if not snapshot_hooks:
                return _PostResponseText(response_text)
        elif not hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE):
            return _PostResponseText(response_text)

        def _is_enforcing(hook) -> bool:
            """Turn-authoritative fail-closed verdict for ``hook``: the captured
            snapshot value when pinned, else the live read (#2674 P0-1)."""
            if captured_enforcing is not None:
                override = captured_enforcing.get(id(hook))
                if override is not None:
                    return override
            return _hook_is_enforcing(hook)

        hook_input = HookInput(
            session_id=session_id or "",
            hook_event_name=HookEvent.POST_RESPONSE.value,
            response_text=response_text,
            pre_tool_prose=pre_tool_prose,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        # #2674: pin each snapshotted hook to its turn-start mode while the audit
        # runs (defers an in-place warn→strict flip to the next turn), and run
        # EXACTLY the snapshot set via ``execute_hooks_snapshot`` so the live
        # registry — which a mid-turn register/enable/disable would have mutated
        # — is not consulted. The buffering decision (``buffer_audit``) came from
        # this same snapshot, so the two stay in lockstep.
        with _pinned_hook_modes(hook_snapshot or []):
            # #2674: derive the enforcing verdict from the EXACT hook set this
            # audit executes, at its pinned (turn-start) modes — the snapshot set
            # when pinned, else the live enabled set. The streaming command
            # wrapper reads THIS flag rather than re-snapshotting the registry.
            if pinned:
                audited_hooks = list(snapshot_hooks)
            else:
                audited_hooks = list(
                    hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE)
                )
            # #2674 P0-1: derive from the CAPTURED (turn-start) flag when pinned,
            # so this verdict's ``enforcing`` — read by the streaming command
            # wrapper to decide whether to withhold typed parts — stays in
            # lockstep with the buffering decision that came from the same
            # snapshot. A live read could diverge under a mid-turn flip.
            enforcing = any(_is_enforcing(h) for h in audited_hooks)

            # #2674 finding 2: the principled enforcing pipeline. When an
            # enforcing (fail-closed) gate hook is present, ONLY the gate hook(s)
            # may see the raw unreviewed response — an advisory hook that ran
            # first (lower priority number) could otherwise persist/emit the raw
            # text before a later DENY, an irreversible side effect no post-hoc
            # part-drop can undo; and an advisory hook that ran last (higher
            # priority number) could rewrite the gate-approved prose into an
            # UNAUDITED release. Partition on ``_hook_is_enforcing`` (preserving
            # priority order within each partition) and run the gate first, then
            # the observers over the REVIEWED text only.
            # #2674 P0-1: partition on the CAPTURED enforcement (via
            # ``_is_enforcing``), NOT a fresh live read — a generic gate hook that
            # dropped ``fail_closed`` mid-turn would otherwise be misfiled as an
            # observer, run over the raw response (leaking it past the freeze),
            # and lose fail-closed handling on the very turn its capture buffered.
            gate_hooks = [h for h in audited_hooks if _is_enforcing(h)]
            observer_hooks = [
                h for h in audited_hooks if not _is_enforcing(h)
            ]

            if not gate_hooks:
                # No enforcing hook: preserve the normal single priority-ordered
                # chain with full MODIFY propagation (nothing is buffered, so an
                # advisory MODIFY only annotates the already-streamed copy).
                if pinned:
                    hook_output = await hooks_manager.execute_hooks_snapshot(
                        HookEvent.POST_RESPONSE, hook_input, audited_hooks,
                        enforcement_overrides=captured_enforcing,
                    )
                else:
                    hook_output = await hooks_manager.execute_hooks(
                        HookEvent.POST_RESPONSE, hook_input,
                    )
                return _post_response_verdict(
                    hook_output, response_text, enforcing=enforcing,
                )

            # PHASE 1 — enforcement. Only the gate hook(s) receive the raw
            # unreviewed response; multiple gate hooks chain in priority order
            # (MODIFY propagates between them, DENY/ASK short-circuits). The
            # captured overrides make a gate hook's crash/timeout fail closed off
            # its turn-start verdict even if its live ``fail_closed`` flipped.
            gate_output = await hooks_manager.execute_hooks_snapshot(
                HookEvent.POST_RESPONSE, hook_input, gate_hooks,
                enforcement_overrides=captured_enforcing,
            )
            gate_verdict = _post_response_verdict(
                gate_output, response_text, enforcing=enforcing,
            )

            # PHASE 2 — observation. Advisory / observer hooks receive the
            # REVIEWED release (the reviewed prose on ALLOW/MODIFY, or the
            # sanitized block message on DENY/ASK) — NEVER the raw response — and
            # cannot mutate that release (``execute_post_response_observers``
            # freezes ``response_text``). They may still record/emit and warn,
            # and a stricter observer DENY/ASK can re-block an otherwise-approved
            # release (always fail-safe). Run them even on a gate block so an
            # external recorder observes the sanitized outcome.
            #
            # #2674 finding 1: the reviewed release is ``str(gate_verdict)``, but
            # ``pre_tool_prose`` / ``tool_calls`` / ``tool_results`` are the RAW,
            # unaudited side channels the gate never released — the pre-tool prose
            # (retracted from the persisted turn), the model-generated tool
            # arguments, and the full tool result payloads. An observer is an
            # ARBITRARY advisory consumer (an external recorder, a webhook) that
            # may persist or forward its HookInput; handing it those raw fields
            # would leak past the fail-closed gate exactly what findings 2/4 drop
            # from the live stream, the persisted row and the STOP hook. This
            # observer phase runs ONLY when a gate hook is present (enforcing), and
            # under an enforcing audit there is NO reviewed equivalent of the raw
            # side channels — the audit reviewed prose only — so NULL all three.
            # Advisory-only turns never reach here (they take the ``not gate_hooks``
            # branch above with the full, already-streamed input), so their
            # behavior is unchanged.
            if observer_hooks:
                observer_input = HookInput(
                    session_id=session_id or "",
                    hook_event_name=HookEvent.POST_RESPONSE.value,
                    response_text=str(gate_verdict),
                    pre_tool_prose=None,
                    tool_calls=None,
                    tool_results=None,
                )
                observer_output = (
                    await hooks_manager.execute_post_response_observers(
                        HookEvent.POST_RESPONSE, observer_input, observer_hooks,
                        enforcement_overrides=captured_enforcing,
                    )
                )
                if not gate_verdict.denied:
                    observer_block = evaluate_blocking_decision(
                        observer_output,
                        deny_reason_default="blocked by response audit",
                        ask_reason_default="response withheld pending approval",
                    )
                    if observer_block is not None:
                        logging.warning(
                            "POST_RESPONSE observer re-blocked an approved "
                            "release (%s): %s",
                            observer_block.decision.value, observer_block.reason,
                        )
                        return _post_response_verdict(
                            observer_output, str(gate_verdict),
                            enforcing=enforcing,
                        )
            return gate_verdict
