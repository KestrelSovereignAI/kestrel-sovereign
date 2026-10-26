"""The whole text of tool results the orchestrator clipped (#3740).

A serialized tool result longer than ``MAX_TOOL_RESULT_CHARS`` reaches the
model as a preview: its first and last characters and a count of those
between. Before #3740 the middle was discarded, and no tool could bring it
back, so an agent could not read a long file or a subagent's long answer by
any path it had. The orchestrator now keeps the whole serialized text here,
names it in the preview by a handle, and ``read_tool_output``
(:class:`~kestrel_sovereign.features.tool_output.feature.ToolOutputFeature`)
reads it in ranged chunks.

What is kept, and for how long:

* The exact text the preview was cut from, so a chunk's offsets index the
  same characters the preview's "first N chars" count does.
* In process memory only. Nothing here is written to storage, which keeps it
  inside every privacy mode's storage contract (EPHEMERAL is "pure
  in-memory"); a restart loses it.
* Bounded per agent by an entry count and a total character budget
  (``KESTREL_TOOL_OUTPUT_RETAIN_ENTRIES`` / ``KESTREL_TOOL_OUTPUT_RETAIN_CHARS``).
  The least recently used entry is evicted first. An output larger than the
  whole budget is not kept, and :meth:`ToolOutputStore.retain` says why so the
  preview can say so too.
* Readable only from the session that produced it, and only under the privacy
  policy it was produced under. A later turn of the same conversation may read
  it; another conversation may not, and neither may the same conversation once
  a privacy transition has applied, which could otherwise carry output produced
  under a stricter mode into a looser one. That holds across a round trip too:
  output produced in EPHEMERAL is not readable after EPHEMERAL -> NORMAL ->
  EPHEMERAL, because leaving EPHEMERAL closed the session it belonged to. The
  policy is the agent's privacy-policy epoch, which every applied transition
  advances, and the transition also clears the store so nothing it can no
  longer serve stays in memory. Output produced outside any chat session is
  readable only where no session is bound either.

Two things a reader needs besides the text, both kept here:

* Authority. A spawn mandate's ``allowed_tools`` ceiling may predate
  ``read_tool_output``, and is signed by the parent, so it cannot simply be
  widened. Reading a retained output continues the call that produced it, so
  :func:`continued_tool_resolver` names that call's tool, and the mandate hook
  permits the read wherever it permits that tool.
* Turn budget. A long output takes many chunks, and each is a tool-loop
  iteration. :class:`ToolOutputPaging` tells the orchestrator which iterations
  only paged retained output forward, so they do not spend the turn's
  ``KESTREL_MAX_TOOL_ITERATIONS``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

TOOL_OUTPUT_HANDLE_PREFIX = "tool_output:"
#: The one tool that reads a retained output back.
READ_TOOL_OUTPUT = "read_tool_output"

#: Characters of retained output kept per agent, across all its entries.
DEFAULT_RETAIN_CHARS = 4_000_000
#: Retained outputs kept per agent.
DEFAULT_RETAIN_ENTRIES = 64

RETAIN_CHARS_ENV = "KESTREL_TOOL_OUTPUT_RETAIN_CHARS"
RETAIN_ENTRIES_ENV = "KESTREL_TOOL_OUTPUT_RETAIN_ENTRIES"

_TOKEN_BYTES = 12
_HANDLE_RE = re.compile(
    rf"^{re.escape(TOOL_OUTPUT_HANDLE_PREFIX)}[0-9a-f]{{{_TOKEN_BYTES * 2}}}$"
)


class ToolOutputUnavailable(LookupError):
    """A handle that cannot be read; the message is written for the agent."""


@dataclass(frozen=True)
class ToolOutputScope:
    """Where a retained output was produced, and so who may read it back.

    ``privacy_policy`` names the privacy policy in force: the agent's policy
    epoch (``<mode>:<instance>:<epoch>``), which every applied privacy
    transition advances, or the bare mode for an embedding with no transition
    machinery. ``privacy_mode`` is that policy's mode, for messages.
    """

    session_id: Optional[str]
    privacy_mode: Optional[str]
    privacy_policy: Optional[str]


@dataclass(frozen=True)
class RetainedToolOutput:
    """One clipped tool result, kept whole."""

    handle: str
    tool_name: str
    text: str
    scope: ToolOutputScope


@dataclass(frozen=True)
class Retention:
    """What :meth:`ToolOutputStore.retain` did: a handle, or why there is none."""

    handle: Optional[str]
    reason: Optional[str] = None


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


class ToolOutputStore:
    """A bounded, in-memory map from handle to a clipped tool result."""

    def __init__(self, *, max_chars: int, max_entries: int) -> None:
        if max_chars < 1 or max_entries < 1:
            raise ValueError("max_chars and max_entries must both be positive")
        self.max_chars = max_chars
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, RetainedToolOutput]" = OrderedDict()
        self._chars = 0

    @classmethod
    def from_environment(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "ToolOutputStore":
        """Build a store sized by the environment; a malformed size fails."""
        env = os.environ if env is None else env
        return cls(
            max_chars=_positive_int(env, RETAIN_CHARS_ENV, DEFAULT_RETAIN_CHARS),
            max_entries=_positive_int(
                env, RETAIN_ENTRIES_ENV, DEFAULT_RETAIN_ENTRIES
            ),
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def retained_chars(self) -> int:
        return self._chars

    def retain(
        self,
        text: str,
        *,
        tool_name: str,
        scope: ToolOutputScope,
    ) -> Retention:
        """Keep ``text`` and return its handle, evicting the oldest to fit."""
        if len(text) > self.max_chars:
            return Retention(
                None,
                f"it is larger than the {self.max_chars:,}-character "
                f"tool-output retention budget",
            )
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self._chars + len(text) > self.max_chars
        ):
            self._drop(next(iter(self._entries)))
        handle = TOOL_OUTPUT_HANDLE_PREFIX + secrets.token_hex(_TOKEN_BYTES)
        self._entries[handle] = RetainedToolOutput(
            handle=handle,
            tool_name=tool_name,
            text=text,
            scope=scope,
        )
        self._chars += len(text)
        return Retention(handle)

    def read(self, handle: Any, *, scope: ToolOutputScope) -> RetainedToolOutput:
        """The entry ``handle`` names, if a caller in ``scope`` may read it.

        Raises :class:`ToolOutputUnavailable` with an agent-facing reason
        otherwise. An entry read under a privacy policy other than its own is
        dropped: the policy it was produced under never comes back, so it can
        never become readable again.
        """
        if not isinstance(handle, str) or not _HANDLE_RE.match(handle.strip()):
            raise ToolOutputUnavailable(
                f"{handle!r} is not a tool-output handle; pass the "
                f"'{TOOL_OUTPUT_HANDLE_PREFIX}…' handle a clipped result named"
            )
        handle = handle.strip()
        entry = self._entries.get(handle)
        if entry is None:
            raise ToolOutputUnavailable(
                f"{handle} is not retained: it was evicted to make room for "
                f"newer tool output, a privacy-mode change released it, or "
                f"the agent restarted since it was produced. Run the tool "
                f"again to get its output back"
            )
        if entry.scope.privacy_policy != scope.privacy_policy:
            self._drop(handle)
            produced_in = entry.scope.privacy_mode or "unknown"
            now_in = scope.privacy_mode or "unknown"
            if produced_in != now_in:
                change = f"the agent is now in {now_in}"
            else:
                change = (
                    f"the privacy mode has changed since, even though it is "
                    f"{now_in} again"
                )
            raise ToolOutputUnavailable(
                f"{handle} was produced in privacy mode {produced_in} and "
                f"{change}, so it can no longer be read"
            )
        if entry.scope.session_id != scope.session_id:
            raise ToolOutputUnavailable(
                f"{handle} was produced in another conversation and can be "
                f"read only there"
            )
        self._entries.move_to_end(handle)
        return entry

    def producer_of(self, handle: Any) -> Optional[str]:
        """The tool whose output ``handle`` names, or ``None`` if none is kept.

        Authorization only: it reads no text, moves nothing in the eviction
        order and checks no scope, which :meth:`read` still does.
        """
        if not isinstance(handle, str):
            return None
        entry = self._entries.get(handle.strip())
        return None if entry is None else entry.tool_name

    def clear(self) -> int:
        """Forget every retained output; return how many there were.

        A privacy transition calls this when it applies: nothing retained
        before it can be read after it.
        """
        dropped = len(self._entries)
        self._entries.clear()
        self._chars = 0
        return dropped

    def _drop(self, handle: str) -> None:
        entry = self._entries.pop(handle)
        self._chars -= len(entry.text)


def agent_tool_output_store(agent: Any) -> Optional[ToolOutputStore]:
    """The agent's store, or ``None`` for an agent that keeps none."""
    store = getattr(agent, "tool_output_store", None)
    return store if isinstance(store, ToolOutputStore) else None


def tool_output_scope(agent: Any) -> ToolOutputScope:
    """The scope the calling task works in.

    Both ends use this one answer: the orchestrator when it keeps a clipped
    result, and ``read_tool_output`` when it reads one back. The session is
    the turn lifecycle's own (``get_turn_bound_session_id``), not the
    agent-global active session, which a task outside the live turn would read
    as a stranger's. The privacy policy is the agent's policy epoch, the same
    name the signal dispatcher records in ingress trust receipts (#3484); an
    agent without one falls back to its mode, so the check is never vacuous
    where a mode exists.
    """
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import (
        _resolve_privacy_policy_epoch,
    )

    session_id = None
    accessor = getattr(agent, "get_turn_bound_session_id", None)
    if callable(accessor):
        bound = accessor()
        session_id = bound if isinstance(bound, str) else None
    mode = getattr(agent, "privacy_mode", None)
    privacy_mode = mode.value if isinstance(mode, PrivacyMode) else None
    policy = _resolve_privacy_policy_epoch(agent)
    return ToolOutputScope(
        session_id=session_id,
        privacy_mode=privacy_mode,
        privacy_policy=policy if policy is not None else privacy_mode,
    )


def continued_tool_resolver(
    agent: Any,
) -> Callable[[str, Mapping[str, Any]], Optional[str]]:
    """Name the tool whose call a tool call continues, for the mandate hook.

    ``read_tool_output`` continues the call that produced the output its handle
    names: it reads characters that call already returned, which the
    orchestrator only held back from the preview. Every other call continues
    nothing. The spawn-mandate hook permits a continuation wherever it permits
    the tool it continues, so a child whose signed ``allowed_tools`` ceiling
    predates ``read_tool_output`` can still read the whole of a file it was
    allowed to read, and nothing else (#3740).
    """

    def continued_tool(tool_name: str, tool_input: Mapping[str, Any]) -> Optional[str]:
        if tool_name != READ_TOOL_OUTPUT or not isinstance(tool_input, Mapping):
            return None
        store = agent_tool_output_store(agent)
        return None if store is None else store.producer_of(tool_input.get("handle"))

    return continued_tool


class ToolOutputPaging:
    """Which iterations of one turn's tool loop only paged retained output.

    The orchestrator stops a turn after ``max_iterations`` tool-loop passes, so
    a model that keeps calling tools without getting anywhere ends. Paging a
    retained output is not that: each chunk is new text, and the output is
    finite. But a long output takes many chunks: a 50,000-character file of
    non-ASCII text serializes to about 300,000 characters, some 45 chunks under
    the default 8,000-character cap. Counted, they ended the turn partway
    through the file (#3740).

    An iteration is credited, and does not count against the budget, when
    every tool call in its batch was a successful ``read_tool_output`` that
    returned a whole chunk from where this turn last stopped reading that
    handle:

    * it starts at the end of this turn's previous read of the handle (a
      turn's first read of a handle may start anywhere);
    * it returned at least one character; and
    * it was cut only by the result cap or the end of the output, not by a
      ``length`` the caller passed.

    Credited reads of one handle are therefore consecutive whole chunks that
    never revisit a character, and their number is bounded by that output's
    size. The budget still counts an iteration that re-reads, skips ahead,
    reads a few characters at a time, or calls any other tool.
    """

    def __init__(self) -> None:
        self.credited = 0
        self._frontier: dict[str, int] = {}

    def budget_iteration(self, iteration: int) -> int:
        """How many of the first ``iteration`` passes count against the budget."""
        return iteration - self.credited

    def settle(
        self,
        tool_calls: Optional[Iterable[Any]],
        tool_messages: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Credit the iteration if its whole batch paged forward; say whether.

        ``tool_messages`` are the messages the batch appended, which carry
        each call's result exactly as the model reads it.
        """
        calls = list(tool_calls or ())
        if not calls:
            return False
        contents = {
            message.get("tool_call_id"): message.get("content")
            for message in tool_messages
            if isinstance(message, Mapping) and message.get("role") == "tool"
        }
        advanced: dict[str, int] = {}
        for call in calls:
            if getattr(call, "name", None) != READ_TOOL_OUTPUT:
                return False
            chunk = _read_chunk(contents.get(getattr(call, "id", None)))
            if chunk is None:
                return False
            handle, start, end, total = chunk
            frontier = advanced.get(handle, self._frontier.get(handle))
            if frontier is not None and start != frontier:
                return False
            if end <= start or not _whole_chunk(call, start, end, total):
                return False
            advanced[handle] = end
        self._frontier.update(advanced)
        self.credited += 1
        return True


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_chunk(content: Any) -> Optional[tuple[str, int, int, int]]:
    """``(handle, start, end, total)`` of a successful chunk's result, else None."""
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    handle = data.get("handle")
    start, length, total = data.get("offset"), data.get("length"), data.get("total_chars")
    if not isinstance(handle, str) or not all(_is_count(v) for v in (start, length, total)):
        return None
    return handle, start, start + length, total


def _whole_chunk(call: Any, start: int, end: int, total: int) -> bool:
    """True unless the chunk stopped where the caller's ``length`` told it to."""
    if end >= total:
        return True
    arguments = getattr(call, "arguments", None)
    requested = arguments.get("length") if isinstance(arguments, Mapping) else None
    if requested is None:
        return True
    try:
        wanted = int(requested)
    except (TypeError, ValueError):
        return False
    return end - start < wanted
