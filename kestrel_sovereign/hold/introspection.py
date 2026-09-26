"""Self-scoped, read-only Hold introspection for an agent (#3166).

A held agent cannot ask anything: Hold refuses every turn at turn start. So
an agent that can run this is almost always *not* held right now, and a
current-latch-only answer would nearly always read ``not_held`` — true, and no
help to an agent explaining why it was silent. The answer is therefore the
current effective state AND the agent's own recent Hold history, paired into
episodes ("held from Tuesday to Friday by the sovereign for <reason>").

Three rules carry the authority boundary:

* **The subject is bound by the trusted runtime.** It is the DID the turn-start
  latch is scoped to (``_hold_scoped_agent_did``, the enforcement guard), read
  from the agent the runtime bound the store to. No caller payload names it,
  so no caller can inspect another agent.
* **Read-only.** This module reads the latch snapshot enforcement reads and
  the store's receipt history. It has no path to a mutation.
* **Role comes from provenance.** An actor's role is ``self`` when the actor is
  the subject, otherwise the authority its receipt recorded. It is never
  guessed from the shape of the actor string, and the raw actor identity is
  never returned. An ancestor's mandate latch (#3168) therefore reads as role
  ``mandate``; its holder's DID is withheld like every other actor's.

A failed read is reported as a typed ``unknown`` with its cause type, never as
``not_held``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .enforcement import (
    _hold_scoped_agent_did,
    bound_hold_store,
    get_effective_hold_state,
)
from .state import (
    HOST_HOLD_TARGET,
    EffectiveHoldState,
    HoldAction,
    HoldAuthority,
    HoldCorruptStateError,
    HoldDisposition,
    HoldFeedEntry,
    HoldReceipt,
    HoldScope,
    HoldState,
    HoldStateError,
)

logger = logging.getLogger(__name__)

# The most receipts one history page carries, merged across the host and
# agent scopes. A caller whose result channel is smaller renders fewer from the
# same read (``SelfHoldSnapshot.render``); the rest stay reachable by cursor.
SELF_HOLD_HISTORY_LIMIT = 25

# The explicit disclosure contract for what a held subject may learn about
# the governance acts performed on it. Returned with every answer so a reader
# never has to infer what was withheld.
SELF_HOLD_REDACTION_POLICY: dict[str, str] = {
    "subject_identity": "bound_by_runtime",
    "actor_identity": "role_only",
    "target_identity": "omitted",
    "receipt_identity": "omitted",
    "operation_identity": "omitted",
    "reason": "visible_to_subject",
    "timestamps": "visible_to_subject",
    "history_cursor": "feed_position_only",
}

SELF_ROLE = "self"

# Continuation grammar: ``<host>:<agent>:<mandate>``, each position either
# ``new`` (that scope has served nothing yet, so it starts at its newest
# receipt), ``end``
# (that scope is exhausted), or the decimal ``feed_seq`` of the last receipt
# served from it. Bounded like the host receipt feed's cursor.
_CURSOR_NEW = "new"
_CURSOR_END = "end"
_CURSOR_MAX_DIGITS = 19


class SelfHoldStateUnavailable(HoldStateError):
    """No Hold store is bound to this agent's runtime."""


class InvalidHoldHistoryCursor(ValueError):
    """A history continuation that this module did not issue."""


def actor_role(
    *, actor_id: str, authority: HoldAuthority, subject_did: str
) -> str:
    """Name an actor by provenance: ``self``, else its recorded authority."""

    if actor_id == subject_did:
        return SELF_ROLE
    return authority.value


# A scope's position: ``None`` = from its newest receipt, ``_END`` = exhausted,
# an int = strictly below that ``feed_seq``.
_END = object()
_SCOPES = (HoldScope.HOST, HoldScope.AGENT, HoldScope.MANDATE)


def _parse_position(token: str) -> Any:
    if token == _CURSOR_NEW:
        return None
    if token == _CURSOR_END:
        return _END
    if (
        not token
        or len(token) > _CURSOR_MAX_DIGITS
        or not token.isascii()
        or not token.isdigit()
        or int(token) < 1
    ):
        raise InvalidHoldHistoryCursor("Hold history cursor is not valid")
    return int(token)


def parse_history_cursor(cursor: str | None) -> dict[HoldScope, Any]:
    """Read a continuation issued by :meth:`SelfHoldSnapshot.render`.

    A cursor carries only feed positions. It never names a subject: the
    subject is re-bound by the runtime on every call, so a forged cursor can
    at most move within the caller's own history.
    """

    if cursor is None:
        return {scope: None for scope in _SCOPES}
    if not isinstance(cursor, str):
        raise InvalidHoldHistoryCursor("Hold history cursor is not valid")
    parts = cursor.strip().split(":")
    if len(parts) != len(_SCOPES):
        raise InvalidHoldHistoryCursor("Hold history cursor is not valid")
    return {scope: _parse_position(part) for scope, part in zip(_SCOPES, parts)}


def _encode_position(position: Any) -> str:
    if position is None:
        return _CURSOR_NEW
    if position is _END:
        return _CURSOR_END
    return str(position)


def _cause(error: BaseException) -> dict[str, str]:
    cause: dict[str, str] = {"cause_type": type(error).__name__}
    underlying = error.__cause__ or error.__context__
    if underlying is not None:
        cause["underlying_cause_type"] = type(underlying).__name__
    return cause


def _receipt_view(receipt: HoldReceipt, *, subject_did: str) -> dict[str, Any]:
    return {
        "scope": receipt.scope.value,
        "action": receipt.action.value,
        "disposition": receipt.disposition.value,
        "reason": receipt.reason,
        "actor_role": actor_role(
            actor_id=receipt.actor_id,
            authority=receipt.authority,
            subject_did=subject_did,
        ),
        "occurred_at": receipt.occurred_at,
    }


async def _latch_view(
    store: Any, latch: HoldState | None, *, subject_did: str
) -> dict[str, Any] | None:
    if latch is None:
        return None
    # The latch row carries no authority of its own; its authority receipt
    # does, and the store proves that receipt against the latch.
    receipt = await store.get_receipt_by_id(latch.hold_receipt_id)
    if receipt is None or receipt.receipt_id != latch.hold_receipt_id:
        raise HoldCorruptStateError(
            "active hold latch references a missing authority receipt"
        )
    return {
        "scope": latch.scope.value,
        "reason": latch.reason,
        "actor_role": actor_role(
            actor_id=latch.actor_id,
            authority=receipt.authority,
            subject_did=subject_did,
        ),
        "set_at": latch.set_at,
        "revision": latch.revision,
    }


def _expected_subject(scope: HoldScope, subject_did: str) -> str:
    return HOST_HOLD_TARGET if scope is HoldScope.HOST else subject_did


def _assert_subject_receipt(
    receipt: HoldReceipt, *, scope: HoldScope, subject_did: str
) -> None:
    if receipt.scope is not scope or receipt.subject_id != _expected_subject(
        scope, subject_did
    ):
        # The store filtered on exactly this key; a row outside it is not
        # this subject's history and must never be rendered as though it
        # were.
        raise HoldCorruptStateError(
            "Hold history returned a receipt outside the subject's scope"
        )


@dataclass(frozen=True)
class _ScopeRead:
    position: Any
    entries: tuple[HoldFeedEntry, ...]
    has_more: bool


async def _read_scope(
    store: Any,
    *,
    scope: HoldScope,
    position: Any,
    subject_did: str,
    limit: int,
) -> _ScopeRead:
    if position is _END:
        return _ScopeRead(position=position, entries=(), has_more=False)
    key: dict[str, str | None]
    if scope is HoldScope.MANDATE:
        # Every ancestor's mandate latch on this subject, whoever holds it.
        key = {"subject_id": subject_did}
    else:
        key = {"target_id": subject_did if scope is HoldScope.AGENT else None}
    page = await store.list_receipts(
        scope=scope,
        **key,
        after=position,
        limit=limit,
        newest_first=True,
    )
    for entry in page.entries:
        _assert_subject_receipt(entry.receipt, scope=scope, subject_did=subject_did)
    return _ScopeRead(
        position=position, entries=page.entries, has_more=page.next_key is not None
    )


async def _episode_starts(
    store: Any,
    reads: dict[HoldScope, _ScopeRead],
    *,
    subject_did: str,
) -> dict[str, HoldReceipt]:
    """Every hold an applied receipt in the read ended, by receipt id.

    A hold that began on an older page than the receipt that ended it is read
    by id, so an episode is complete on the page that holds its end.
    """

    starts: dict[str, HoldReceipt] = {}
    for read in reads.values():
        for entry in read.entries:
            starts[entry.receipt.receipt_id] = entry.receipt
    for scope, read in reads.items():
        for entry in read.entries:
            prior = _ended_hold_id(entry.receipt)
            if prior is None or prior in starts:
                continue
            start = await store.get_receipt_by_id(prior)
            if (
                start is None
                or start.receipt_id != prior
                or start.action is not HoldAction.HOLD
                or start.disposition is not HoldDisposition.APPLIED
            ):
                raise HoldCorruptStateError(
                    "Hold receipt ends a hold that is not an applied hold"
                )
            _assert_subject_receipt(start, scope=scope, subject_did=subject_did)
            starts[prior] = start
    return starts


def _ended_hold_id(receipt: HoldReceipt) -> str | None:
    if (
        receipt.disposition is HoldDisposition.APPLIED
        and receipt.prior_hold_receipt_id
    ):
        return receipt.prior_hold_receipt_id
    return None


def _unknown(failure: str, error: BaseException) -> dict[str, Any]:
    return {
        "state": "unknown",
        "held": None,
        "sources": [],
        "latches": {"host": None, "agent": None, "mandate": None},
        "history": None,
        "failure": failure,
        **_cause(error),
        "redaction_policy": dict(SELF_HOLD_REDACTION_POLICY),
    }


@dataclass(frozen=True)
class SelfHoldSnapshot:
    """One read of the subject's Hold state and a window of its history.

    The read is done once; :meth:`render` projects any prefix of the window,
    so a caller with a bounded result channel shrinks the page without
    re-reading (and re-proving) the history.
    """

    subject_did: str
    current: dict[str, Any]
    effective: EffectiveHoldState | None
    reads: dict[HoldScope, _ScopeRead] | None
    starts: dict[str, HoldReceipt]
    history_error: BaseException | None

    def render(self, page_size: int, *, withhold: bool = False) -> dict[str, Any]:
        """Project the newest ``page_size`` receipts of the window.

        ``withhold`` still advances past the page but omits its contents; it
        exists for a page too large for the caller's channel even at one
        receipt, which is then reported as withheld rather than silently cut.
        """

        if (
            not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or page_size < 1
        ):
            raise ValueError("Hold history page size must be a positive integer")
        result = dict(self.current)
        if self.effective is None:
            return result
        if self.history_error is not None or self.reads is None:
            error = self.history_error or HoldStateError("history not read")
            result["history"] = {"status": "unknown", **_cause(error)}
            return result

        merged = sorted(
            (entry for read in self.reads.values() for entry in read.entries),
            key=lambda entry: entry.feed_seq,
            reverse=True,
        )
        page = merged[: min(page_size, SELF_HOLD_HISTORY_LIMIT)]
        next_positions: dict[HoldScope, Any] = {}
        for scope, read in self.reads.items():
            taken = [entry for entry in page if entry.receipt.scope is scope]
            if read.position is _END or (
                len(taken) == len(read.entries) and not read.has_more
            ):
                next_positions[scope] = _END
            elif taken:
                next_positions[scope] = taken[-1].feed_seq
            else:
                next_positions[scope] = read.position
        next_cursor = (
            None
            if all(position is _END for position in next_positions.values())
            else ":".join(_encode_position(next_positions[s]) for s in _SCOPES)
        )
        history: dict[str, Any] = {
            "status": "read",
            "page_size": len(page),
            "next_cursor": next_cursor,
        }
        if withhold:
            history["status"] = "withheld_oversize"
            history["withheld_receipts"] = len(page)
            history["receipts"] = []
            history["episodes"] = []
        else:
            history["receipts"] = [
                _receipt_view(entry.receipt, subject_did=self.subject_did)
                for entry in page
            ]
            history["episodes"] = self._episodes(page)
        result["history"] = history
        return result

    def _episodes(self, page: list[HoldFeedEntry]) -> list[dict[str, Any]]:
        """Each hold episode, on the page holding its latest receipt.

        An ended hold appears where its ending receipt is; an unended one
        where it began. Pages are contiguous, newest-first slices of each
        scope's history, so an episode is reported on exactly one page.
        """

        if self.effective is None or self.reads is None:
            raise HoldStateError("Hold episodes need a read state and history")
        ended: set[str] = set()
        keyed: list[tuple[int, int, dict[str, Any]]] = []
        for entry in page:
            prior = _ended_hold_id(entry.receipt)
            if prior is None:
                continue
            ended.add(prior)
            status = (
                "released"
                if entry.receipt.action is HoldAction.RELEASE
                else "replaced"
            )
            keyed.append(
                (
                    entry.feed_seq,
                    0,
                    self._episode(
                        self.starts[prior], status=status, ended_by=entry.receipt
                    ),
                )
            )
        for entry in page:
            receipt = entry.receipt
            if not (
                receipt.action is HoldAction.HOLD
                and receipt.disposition is HoldDisposition.APPLIED
            ) or receipt.receipt_id in ended:
                continue
            current_ids = {
                latch.hold_receipt_id
                for latch in (
                    self.effective.host,
                    self.effective.agent,
                    *self.effective.mandates,
                )
                if latch is not None and latch.scope is receipt.scope
            }
            if receipt.receipt_id in current_ids:
                status = "active"
            elif self.reads[receipt.scope].position is None:
                # The latch was read AFTER this history, and no longer names
                # this hold, yet nothing in the newest history ended it: it
                # ended between the two reads. Say so rather than guessing.
                status = "unresolved"
            else:
                # Every receipt newer than this page in its scope was served
                # on an earlier page, and that page carried this episode.
                continue
            keyed.append(
                (entry.feed_seq, 1, self._episode(receipt, status=status, ended_by=None))
            )
        # Ordered by commit position (``feed_seq``), not by the displayed clock.
        keyed.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [episode for _seq, _rank, episode in keyed]

    def _episode(
        self, start: HoldReceipt, *, status: str, ended_by: HoldReceipt | None
    ) -> dict[str, Any]:
        episode: dict[str, Any] = {
            "scope": start.scope.value,
            "status": status,
            "set_at": start.occurred_at,
            "reason": start.reason,
            "set_by_role": actor_role(
                actor_id=start.actor_id,
                authority=start.authority,
                subject_did=self.subject_did,
            ),
            "ended_at": None,
            "ended_by_role": None,
            "end_reason": None,
        }
        if ended_by is not None:
            episode["ended_at"] = ended_by.occurred_at
            episode["ended_by_role"] = actor_role(
                actor_id=ended_by.actor_id,
                authority=ended_by.authority,
                subject_did=self.subject_did,
            )
            episode["end_reason"] = ended_by.reason
        return episode


async def read_self_hold(
    agent: Any, *, cursor: str | None = None
) -> SelfHoldSnapshot:
    """Read the Hold state and one history window that apply to ``agent``.

    ``agent`` is the runtime's own agent object, never a caller's choice:
    the subject DID and the store are both read from it, the same way
    turn-start enforcement reads them. ``cursor`` continues a previous page
    and raises :class:`InvalidHoldHistoryCursor` when it was not issued here.
    """

    positions = parse_history_cursor(cursor)
    store = bound_hold_store(agent)
    if store is None:
        return SelfHoldSnapshot(
            subject_did="",
            current=_unknown(
                "store_unbound",
                SelfHoldStateUnavailable(
                    "No durable Hold store is bound to this agent's runtime"
                ),
            ),
            effective=None,
            reads=None,
            starts={},
            history_error=None,
        )
    try:
        subject_did = _hold_scoped_agent_did(agent)
    except HoldStateError as error:
        return _state_failure(error)

    # History is read BEFORE the latch: a hold the latch still names after
    # this history was taken is genuinely active, not a stale snapshot.
    reads: dict[HoldScope, _ScopeRead] | None = None
    starts: dict[str, HoldReceipt] = {}
    history_error: BaseException | None = None
    try:
        reads = {
            scope: await _read_scope(
                store,
                scope=scope,
                position=positions[scope],
                subject_did=subject_did,
                limit=SELF_HOLD_HISTORY_LIMIT,
            )
            for scope in _SCOPES
        }
        starts = await _episode_starts(store, reads, subject_did=subject_did)
    except HoldStateError as error:
        logger.error(
            "Self Hold introspection could not read history (cause_type=%s)",
            type(error).__name__,
        )
        reads, starts, history_error = None, {}, error

    try:
        effective = await get_effective_hold_state(agent)
        if not isinstance(effective, EffectiveHoldState):
            raise HoldCorruptStateError(
                "Hold store returned no effective state for a bound runtime"
            )
        host_view = await _latch_view(store, effective.host, subject_did=subject_did)
        agent_view = await _latch_view(
            store, effective.agent, subject_did=subject_did
        )
        mandate_views = [
            await _latch_view(store, latch, subject_did=subject_did)
            for latch in effective.mandates
        ]
    except HoldStateError as error:
        return _state_failure(error)

    current: dict[str, Any] = {
        "state": "held" if effective.held else "not_held",
        "held": effective.held,
        "sources": [source.value for source in effective.sources],
        # Independent latches: a host hold never hides an agent hold, and
        # releasing one never reads as releasing the other.
        "latches": {
            "host": host_view,
            "agent": agent_view,
            "mandate": mandate_views,
        },
        "redaction_policy": dict(SELF_HOLD_REDACTION_POLICY),
    }
    return SelfHoldSnapshot(
        subject_did=subject_did,
        current=current,
        effective=effective,
        reads=reads,
        starts=starts,
        history_error=history_error,
    )


def _state_failure(error: HoldStateError) -> SelfHoldSnapshot:
    logger.error(
        "Self Hold introspection could not read current state (cause_type=%s)",
        type(error).__name__,
    )
    return SelfHoldSnapshot(
        subject_did="",
        current=_unknown("read_failed", error),
        effective=None,
        reads=None,
        starts={},
        history_error=None,
    )


async def inspect_self_hold(
    agent: Any,
    *,
    cursor: str | None = None,
    history_limit: int = SELF_HOLD_HISTORY_LIMIT,
) -> dict[str, Any]:
    """Return the Hold state and one history page that apply to ``agent``."""

    if (
        not isinstance(history_limit, int)
        or isinstance(history_limit, bool)
        or history_limit < 1
    ):
        raise ValueError("Hold history limit must be a positive integer")
    snapshot = await read_self_hold(agent, cursor=cursor)
    return snapshot.render(history_limit)


__all__ = [
    "SELF_HOLD_HISTORY_LIMIT",
    "SELF_HOLD_REDACTION_POLICY",
    "SELF_ROLE",
    "InvalidHoldHistoryCursor",
    "SelfHoldSnapshot",
    "SelfHoldStateUnavailable",
    "actor_role",
    "inspect_self_hold",
    "parse_history_cursor",
    "read_self_hold",
]
