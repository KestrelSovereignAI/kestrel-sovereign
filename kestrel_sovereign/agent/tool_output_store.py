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
"""

from __future__ import annotations

import os
import re
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Optional

TOOL_OUTPUT_HANDLE_PREFIX = "tool_output:"

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
