"""What the tool allowlist held when it refused a call (#2935).

#2929 collected nine-plus reports of a tool that answered one call and was
refused as "not in the known tool allowlist" on a later call in the SAME turn.
A static gap cannot do that; a name set that shrinks mid-turn can. #2934
derived the allowlist from the live registry and audited every refusal, but
recorded nothing about the set itself, so a bounce still left only the model's
account of what happened.

This module keeps the evidence:

* :class:`KnownToolNames` is the allowlist as a frozenset that also carries the
  registry state it was derived from;
* a per-turn ledger records which names the guardrails let through, so a
  refusal can say whether the same name was accepted earlier in the turn, and
  LRU eviction can avoid unmounting a tool the agent's live turn is using.
  Both ways a turn's model reaches a tool feed it: the orchestrator loops'
  ``_dispatch_tool_call`` and ``execute_named_tool``, which the codex inline
  executor calls. A call a running tool makes by name is recorded at that
  tool's own position and not counted again (:func:`begin_named_tool_call`);
* :class:`AllowlistEvidence` is attached to every ``_dispatch_tool_call``
  refusal. It holds names, counts and digests only, never tool arguments.

Every helper takes the agent and reads it defensively: the orchestrator's unit
tests drive these paths on ``MagicMock`` agents, where an unset attribute is a
``MagicMock`` rather than an ``AttributeError``.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Iterator, Optional

#: Turns whose ledgers are kept. The CONVERSATION lock serializes turns per
#: agent, so the live turn is always the newest; the rest is slack for
#: out-of-turn work that interleaves with it.
MAX_TRACKED_TURNS = 8

#: Hex characters of the SHA-256 kept as a name-set digest. Enough to tell
#: two sets apart in one agent's history; short enough to read in a log line.
DIGEST_HEX_CHARS = 16

_LEDGERS_ATTR = "_tool_turn_ledgers"
_GENERATION_ATTR = "_tool_registry_generation"

# Feature names and turn ids are interpolated into the audit row, whose read
# path withholds any bracketed structure it cannot parse. Keep them to plain
# identifiers.
_UNSAFE_TOKEN_CHARS = re.compile(r"[^A-Za-z0-9_.:-]")


def tool_name_set_digest(names: Iterable[str]) -> str:
    """Return a stable digest of a tool-name set.

    Order-independent and process-independent (no ``hash()``), so the digest
    of the set at an earlier acceptance can be compared with the one at a
    later refusal, and across log lines from different processes. Names are
    compared as strings: a digest is computed on a refusal path, which must
    not raise on a malformed registry entry.
    """
    canonical = json.dumps(
        sorted({str(name) for name in names}), separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:DIGEST_HEX_CHARS]


def registry_generation(agent: Any) -> int:
    """Return how many times the agent's tool registry has been changed."""
    value = getattr(agent, _GENERATION_ATTR, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def bump_registry_generation(agent: Any) -> int:
    """Record one change to the tool registry and return the new generation.

    Called by every mutation of the mounted direct-tool registry (mount,
    unmount, eviction, the constitution receipt tool) and by the agent's own
    feature load, enable and teardown. A refusal whose set carries a later
    generation than the acceptance before it was checked against a changed
    registry. The feature endpoints also set a loaded feature's ``enabled``
    flag directly in places, which this does not see; the feature counts on
    :class:`KnownToolNames` do.
    """
    generation = registry_generation(agent) + 1
    setattr(agent, _GENERATION_ATTR, generation)
    return generation


class KnownToolNames(frozenset):
    """The guardrail allowlist, carrying the registry state it came from.

    A frozenset, so every membership check against it is unchanged. The
    attributes exist so a refusal can name the registry that produced the set,
    which may be older than the registry at the moment of the refusal: the
    loops derive the set once per iteration, and a batch can explore a feature
    before a later call in the same batch is checked.
    """

    __slots__ = (
        "registry_generation",
        "features_loaded",
        "features_enabled",
        "unreadable_features",
        "_digest",
    )

    def __new__(
        cls,
        names: Iterable[str] = (),
        *,
        registry_generation: int,
        features_loaded: int,
        features_enabled: int,
        unreadable_features: Iterable[str] = (),
    ) -> "KnownToolNames":
        self = super().__new__(cls, names)
        self.registry_generation = registry_generation
        self.features_loaded = features_loaded
        self.features_enabled = features_enabled
        self.unreadable_features = tuple(unreadable_features)
        self._digest: Optional[str] = None
        return self

    @property
    def digest(self) -> str:
        """:func:`tool_name_set_digest` of this set, computed once."""
        if self._digest is None:
            self._digest = tool_name_set_digest(self)
        return self._digest

    def widened(self, names: Iterable[str]) -> "KnownToolNames":
        """This set plus ``names``, carrying the same provenance."""
        return KnownToolNames(
            frozenset(self) | frozenset(names),
            registry_generation=self.registry_generation,
            features_loaded=self.features_loaded,
            features_enabled=self.features_enabled,
            unreadable_features=self.unreadable_features,
        )

    def __reduce__(self):
        # frozenset's own reduce rebuilds through ``cls(names)``, which the
        # keyword-only provenance makes fail; copy and pickle must keep both.
        return (
            _rebuild_known_tool_names,
            (
                tuple(self),
                self.registry_generation,
                self.features_loaded,
                self.features_enabled,
                self.unreadable_features,
            ),
        )


def _rebuild_known_tool_names(
    names, registry_generation, features_loaded, features_enabled, unreadable
) -> KnownToolNames:
    return KnownToolNames(
        names,
        registry_generation=registry_generation,
        features_loaded=features_loaded,
        features_enabled=features_enabled,
        unreadable_features=unreadable,
    )


def _digest_of(names: Collection[str]) -> str:
    if isinstance(names, KnownToolNames):
        return names.digest
    return tool_name_set_digest(names)


@dataclass(frozen=True)
class AcceptedToolCall:
    """One call in a turn that the guardrails let a name through."""

    call_index: int
    set_size: int
    set_digest: str
    registry_generation: Optional[int]


@dataclass
class TurnToolLedger:
    """Tool calls seen in one turn, and every acceptance per name.

    Every acceptance is kept, not only the last: calls in a parallel batch
    reach their acceptance in completion order, so a later call can be
    recorded before an earlier one is refused.
    """

    calls: int = 0
    accepted: dict[str, list[AcceptedToolCall]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallContext:
    """Where one tool call sits in its turn, and the allowlist it was checked
    against. ``turn_id`` and ``call_index`` are ``None`` outside a turn."""

    turn_id: Optional[str]
    call_index: Optional[int]
    known_tools: Collection[str]


@dataclass(frozen=True)
class AllowlistEvidence:
    """What the allowlist held when a call was refused.

    Names, counts and digests only: nothing here is taken from the call's
    arguments, so it can be logged and stored without masking.
    """

    in_allowlist: bool
    set_size: int
    set_digest: str
    registry_generation: Optional[int]
    registry_generation_now: int
    features_enabled: Optional[int]
    features_loaded: Optional[int]
    unreadable_features: tuple[str, ...]
    turn_id: Optional[str]
    call_index: Optional[int]
    accepted_earlier: Optional[AcceptedToolCall]

    @property
    def accepted_earlier_this_turn(self) -> bool:
        return self.accepted_earlier is not None

    def describe(self) -> str:
        """Render the evidence as ``key=value`` prose for a log or audit row.

        Deliberately free of brackets and quotes: the audit read path
        (``remask_summary``) withholds any bracketed structure in front of the
        ``args=`` JSON that it cannot parse.
        """
        parts = [
            "allowlist",
            f"in_set={_yes_no(self.in_allowlist)}",
            f"size={self.set_size}",
            f"sha256_{DIGEST_HEX_CHARS}={self.set_digest}",
            f"registry_generation={_or_unknown(self.registry_generation)}",
            f"registry_generation_now={self.registry_generation_now}",
            "features_enabled="
            f"{_or_unknown(self.features_enabled)}/{_or_unknown(self.features_loaded)}",
            "unreadable_features="
            + (",".join(self.unreadable_features) or "none"),
            "turn="
            + (safe_evidence_token(self.turn_id) if self.turn_id else "none"),
            f"call={_or_unknown(self.call_index)}",
            f"accepted_earlier_this_turn={_yes_no(self.accepted_earlier_this_turn)}",
        ]
        accepted = self.accepted_earlier
        if accepted is not None:
            parts.extend([
                f"accepted_at_call={accepted.call_index}",
                f"accepted_size={accepted.set_size}",
                f"accepted_sha256_{DIGEST_HEX_CHARS}={accepted.set_digest}",
                "accepted_registry_generation="
                f"{_or_unknown(accepted.registry_generation)}",
            ])
        return " ".join(parts)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _or_unknown(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)


def safe_evidence_token(value: Any) -> str:
    """A name reduced to characters the audit row keeps verbatim."""
    return _UNSAFE_TOKEN_CHARS.sub("_", str(value))


def current_turn_id(agent: Any) -> Optional[str]:
    """The turn the calling task belongs to, or ``None`` outside one."""
    getter = getattr(agent, "_get_current_turn_id", None)
    if not callable(getter):
        return None
    value = getter()
    return value if isinstance(value, str) and value.strip() else None


def _ledgers(agent: Any) -> "OrderedDict[str, TurnToolLedger]":
    ledgers = getattr(agent, _LEDGERS_ATTR, None)
    if not isinstance(ledgers, OrderedDict):
        ledgers = OrderedDict()
        setattr(agent, _LEDGERS_ATTR, ledgers)
    return ledgers


def _existing_ledger(agent: Any, turn_id: Optional[str]) -> Optional[TurnToolLedger]:
    if turn_id is None:
        return None
    ledgers = getattr(agent, _LEDGERS_ATTR, None)
    if not isinstance(ledgers, OrderedDict):
        return None
    return ledgers.get(turn_id)


@dataclass
class _ToolCallInFlight:
    """Marks a task as running one counted tool call of one agent."""

    agent_id: int
    context: ToolCallContext
    open: bool = True


_TOOL_CALL_IN_FLIGHT: contextvars.ContextVar[Optional[_ToolCallInFlight]] = (
    contextvars.ContextVar("kestrel_tool_call_in_flight", default=None)
)


@contextmanager
def tool_call_scope(agent: Any, context: ToolCallContext) -> Iterator[None]:
    """Run the body as the dispatch of the tool call ``context`` positions.

    A tool that dispatches another tool by name from inside the scope (a
    workflow runner, ``signal_dispatch``) makes a nested call, which
    :func:`begin_named_tool_call` places at this call's position.
    """
    marker = _ToolCallInFlight(id(agent), context)
    token = _TOOL_CALL_IN_FLIGHT.set(marker)
    try:
        yield
    finally:
        # A task created inside the scope keeps this marker after the scope
        # ends; closing it stops that task's later calls reading as nested.
        marker.open = False
        _TOOL_CALL_IN_FLIGHT.reset(token)


def enclosing_tool_call(agent: Any) -> Optional[ToolCallContext]:
    """The counted call of ``agent`` the calling task runs inside, if any.

    Only a call still in flight encloses: a long-lived task born inside one
    (the codex app-server's reader loop can be) carries its closed marker
    forever, and another agent's call is not this agent's.
    """
    marker = _TOOL_CALL_IN_FLIGHT.get()
    if marker is None or not marker.open or marker.agent_id != id(agent):
        return None
    return marker.context


def begin_named_tool_call(
    agent: Any, known_tools: Collection[str]
) -> ToolCallContext:
    """Position one ``execute_named_tool`` call in the current turn.

    A call from the turn itself (the codex inline executor's) is counted like
    a ``_dispatch_tool_call``. A call a running tool makes by name is that
    tool's work: it takes the enclosing call's position, so the turn's count
    stays the model's own calls, while what it is accepted for is still
    recorded and still held against eviction.
    """
    outer = enclosing_tool_call(agent)
    if outer is None:
        return begin_tool_call(agent, known_tools)
    return ToolCallContext(outer.turn_id, outer.call_index, known_tools)


def begin_tool_call(agent: Any, known_tools: Collection[str]) -> ToolCallContext:
    """Count one tool call into the current turn and return its position."""
    turn_id = current_turn_id(agent)
    if turn_id is None:
        return ToolCallContext(None, None, known_tools)
    ledgers = _ledgers(agent)
    ledger = ledgers.get(turn_id)
    if ledger is None:
        ledger = ledgers[turn_id] = TurnToolLedger()
        while len(ledgers) > MAX_TRACKED_TURNS:
            ledgers.popitem(last=False)
    else:
        ledgers.move_to_end(turn_id)
    ledger.calls += 1
    return ToolCallContext(turn_id, ledger.calls, known_tools)


def record_tool_acceptance(
    agent: Any, context: Optional[ToolCallContext], tool_name: str
) -> None:
    """Record that the guardrails let ``tool_name`` through for this call."""
    if context is None or context.call_index is None:
        return
    ledger = _existing_ledger(agent, context.turn_id)
    if ledger is None:
        return
    known = context.known_tools
    ledger.accepted.setdefault(tool_name, []).append(AcceptedToolCall(
        call_index=context.call_index,
        set_size=len(known),
        set_digest=_digest_of(known),
        registry_generation=(
            known.registry_generation if isinstance(known, KnownToolNames) else None
        ),
    ))


def names_held_by_live_turn(agent: Any) -> frozenset[str]:
    """Names the guardrails let through so far in the agent's live turn.

    Read from the agent's live-turn mirror rather than the calling task's
    turn id: eviction can run from a task outside the turn (an MCP server
    remounting in the background), and a task detached from a finished turn
    still carries that turn's id.
    """
    live = getattr(agent, "_live_turn_id", None)
    if not isinstance(live, str) or not live.strip():
        return frozenset()
    ledger = _existing_ledger(agent, live)
    return frozenset(ledger.accepted) if ledger is not None else frozenset()


def allowlist_evidence(
    agent: Any, context: Optional[ToolCallContext], tool_name: str
) -> Optional[AllowlistEvidence]:
    """Describe the allowlist ``context`` was checked against, for a refusal.

    ``None`` when the caller has no context to describe.
    """
    if context is None:
        return None
    known = context.known_tools
    # An empty allowlist is falsy, so provenance is tested by type, not truth.
    provenance = known if isinstance(known, KnownToolNames) else None
    has_provenance = provenance is not None
    accepted = None
    ledger = _existing_ledger(agent, context.turn_id)
    if ledger is not None and context.call_index is not None:
        # The acceptance closest before this call: it brackets the change.
        earlier = [
            candidate for candidate in ledger.accepted.get(tool_name, ())
            if candidate.call_index < context.call_index
        ]
        if earlier:
            accepted = max(earlier, key=lambda candidate: candidate.call_index)
    return AllowlistEvidence(
        in_allowlist=tool_name in known,
        set_size=len(known),
        set_digest=_digest_of(known),
        registry_generation=(
            provenance.registry_generation if has_provenance else None
        ),
        registry_generation_now=registry_generation(agent),
        features_enabled=provenance.features_enabled if has_provenance else None,
        features_loaded=provenance.features_loaded if has_provenance else None,
        unreadable_features=(
            provenance.unreadable_features if has_provenance else ()
        ),
        turn_id=context.turn_id,
        call_index=context.call_index,
        accepted_earlier=accepted,
    )
