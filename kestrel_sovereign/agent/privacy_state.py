"""The agent's privacy mode, recorded durably per agent (#3554).

The privacy mode used to live only in the running process, while the local
model a local-only mode forced was persisted as the agent's model preference. A
restart therefore brought the agent back in NORMAL mode on that local model: a
Sovereign who chose a local-only mode was returned to cloud routes, and one who
had left it stayed on the local model.

The mode is now one ``agent_metadata`` row per agent (key ``privacy_state``)::

    {"version": 1,
     "mode": "deidentified",
     "transition_from": null,
     "saved_preference": {"vendor": "anthropic", "route": "api",
                          "model": "claude-haiku-4-5"}}

``saved_preference`` is the model mandate in force when the agent entered a
local-only mode, restored when it leaves one. ``"model": null`` records that no
mandate was in force (default routing). It is ``null`` outside a local-only
mode.

``transition_from`` marks a pending record: a transition that has started
but has not been recorded as applied. A transition writes three things before
anything in the running process changes: the pending record, then the model
mandate the new mode implies, then the settled record. Only then does the live
mode flip. If any write fails, the live mode stays where it was and the
transition puts back the record it started from; if even that fails, the
pending record stays.

A pending record names two modes and the saved mandate the more restrictive of
them needs. Whoever reads it cannot tell which mode was in force, so it
resolves to the more restrictive one, holding that mandate: boot restores it,
and a later transition starts from it rather than from the live state (see
:func:`transition_origin`). A record that does not validate at all, or one that
cannot be read at boot, could have named any mode, so boot takes the most
restrictive mode there is. None of these ever falls back to NORMAL. Boot then
records the mode it resolved, keeping whatever saved mandate can still be
recovered from the old record, so a later boot that can read the record does
not return to a less restrictive mode than this one ran in. The one exception
is a record boot cannot read even once more: nothing is written over it,
because it may hold the saved mandate, until a transition can read it. A
transition that cannot read the record is refused and writes nothing: an
unreadable record never stands for one with no saved mandate.

The record is control-plane state, not user content: it is written in every
privacy mode, including the volatile ones.

Who wrote the saved mandate is kept beside the record, in the same statement,
as a second row (key ``privacy_saved_preference_provenance``)::

    {"saved_preference": {"vendor": "anthropic", "route": "api",
                          "model": "claude-haiku-4-5"},
     "provenance": {"source": "sovereign", "role": "sovereign", ...}}

Leaving a local-only mode restores the saved mandate under that writer's
authority, so a mandate the agent wrote is still held to the approved provider
chain (#2652). A saved mandate the row does not describe, including one saved
before the row existed, is restored as the agent's. The record's own shape is
unchanged, so an older release still reads it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from kestrel_sovereign.llm.provider_policy import OverrideProvenance
from kestrel_sovereign.privacy import (
    PRIVACY_MODES_MOST_RESTRICTIVE_FIRST,
    PrivacyMode,
    most_restrictive_privacy_mode,
)
from kestrel_sovereign.storage.timestamps import utc_timestamp_parameter
from kestrel_sovereign.execution_custody import is_execution_control_error

logger = logging.getLogger(__name__)

PRIVACY_STATE_KEY = "privacy_state"
PRIVACY_SAVED_PROVENANCE_KEY = "privacy_saved_preference_provenance"
PRIVACY_STATE_VERSION = 1
FAIL_CLOSED_PRIVACY_MODE = PRIVACY_MODES_MOST_RESTRICTIVE_FIRST[0]

PRIVACY_STATE_UNRECORDED_MESSAGE = (
    "Privacy mode change refused: the new mode could not be recorded, so it "
    "would not survive a restart. Staying in the current mode; resolve the "
    "storage error and retry."
)

_RECORD_KEYS = frozenset({"version", "mode", "transition_from", "saved_preference"})
_PREFERENCE_KEYS = frozenset({"vendor", "model", "route"})


class PrivacyStateUnreadableError(ValueError):
    """A recorded privacy state that does not validate."""


@dataclass(frozen=True)
class SavedModelPreference:
    """The model mandate a local-only mode displaced.

    ``model`` ``None`` means no mandate was in force: restoring it returns the
    agent to default routing rather than pinning a route it never chose.

    ``provenance`` is who wrote the mandate (#2652); None restores it as the
    agent's. It does not take part in equality: two records of one mandate
    are the same mandate.
    """

    vendor: Optional[str]
    model: Optional[str]
    route: Optional[str]
    provenance: Optional[OverrideProvenance] = field(default=None, compare=False)

    @classmethod
    def from_preference(
        cls,
        preference: Mapping[str, Any],
        provenance: Optional[OverrideProvenance] = None,
    ) -> "SavedModelPreference":
        """Capture ``LLMService.get_model_preference()`` output, and its writer."""
        model = preference.get("model") or None
        if model is None:
            return cls(vendor=None, model=None, route=None)
        return cls(
            vendor=preference.get("vendor") or None,
            model=model,
            route=preference.get("route") or None,
            provenance=provenance if isinstance(provenance, OverrideProvenance) else None,
        )

    @property
    def pinned(self) -> bool:
        return self.model is not None

    def as_dict(self) -> dict[str, Optional[str]]:
        return {"vendor": self.vendor, "route": self.route, "model": self.model}

    def describe(self) -> str:
        if not self.pinned:
            return "default routing (no mandate)"
        prefix = f"{self.vendor}:{self.route}" if self.route else str(self.vendor)
        return f"{prefix}/{self.model}"


@dataclass(frozen=True)
class PrivacyState:
    """One agent's recorded privacy state."""

    mode: PrivacyMode
    saved_preference: Optional[SavedModelPreference] = None
    transition_from: Optional[PrivacyMode] = None

    @property
    def resolved_mode(self) -> PrivacyMode:
        """The mode this record resolves to: the more restrictive one if pending."""
        if self.transition_from is None:
            return self.mode
        return most_restrictive_privacy_mode(self.mode, self.transition_from)

    def encode(self) -> str:
        return json.dumps(
            {
                "version": PRIVACY_STATE_VERSION,
                "mode": self.mode.value,
                "transition_from": (
                    self.transition_from.value
                    if self.transition_from is not None
                    else None
                ),
                "saved_preference": (
                    self.saved_preference.as_dict()
                    if self.saved_preference is not None
                    else None
                ),
            },
            sort_keys=True,
        )

    @classmethod
    def decode(cls, raw: Any) -> "PrivacyState":
        """Parse a stored record, raising ``PrivacyStateUnreadableError``."""
        if not isinstance(raw, str):
            raise PrivacyStateUnreadableError(
                f"record is {type(raw).__name__}, not JSON text"
            )
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrivacyStateUnreadableError(f"record is not JSON: {exc}") from None
        if not isinstance(record, dict):
            raise PrivacyStateUnreadableError("record is not a JSON object")
        if set(record) != _RECORD_KEYS:
            raise PrivacyStateUnreadableError(
                f"record has keys {sorted(record)}, expected {sorted(_RECORD_KEYS)}"
            )
        version = record["version"]
        if type(version) is not int or version != PRIVACY_STATE_VERSION:
            raise PrivacyStateUnreadableError(f"unsupported record version {version!r}")
        transition_from = record["transition_from"]
        return cls(
            mode=_decode_mode(record["mode"], "mode"),
            saved_preference=_decode_preference(record["saved_preference"]),
            transition_from=(
                None
                if transition_from is None
                else _decode_mode(transition_from, "transition_from")
            ),
        )


def _decode_mode(value: Any, field: str) -> PrivacyMode:
    if not isinstance(value, str):
        raise PrivacyStateUnreadableError(f"{field} is not a mode name: {value!r}")
    try:
        return PrivacyMode(value)
    except ValueError:
        raise PrivacyStateUnreadableError(f"{field} names no mode: {value!r}") from None


def _decode_preference(value: Any) -> Optional[SavedModelPreference]:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _PREFERENCE_KEYS:
        raise PrivacyStateUnreadableError(f"saved_preference is malformed: {value!r}")
    for key in sorted(_PREFERENCE_KEYS):
        if value[key] is not None and not (isinstance(value[key], str) and value[key]):
            raise PrivacyStateUnreadableError(
                f"saved_preference.{key} is not a non-empty string: {value[key]!r}"
            )
    if value["model"] is None and (value["vendor"] or value["route"]):
        raise PrivacyStateUnreadableError(
            "saved_preference names a vendor or route without a model"
        )
    return SavedModelPreference(
        vendor=value["vendor"], model=value["model"], route=value["route"]
    )


async def load_privacy_state(db: Any, agent_id: str) -> Optional[PrivacyState]:
    """Read ``agent_id``'s recorded privacy state; ``None`` when none is recorded.

    Raises ``PrivacyStateUnreadableError`` for a record that does not validate.
    A database error propagates.
    """
    rows = await _fetch_record_rows(db, agent_id)
    if not rows:
        return None
    state = PrivacyState.decode(rows[0][0])
    saved = state.saved_preference
    if saved is None or not saved.pinned:
        return state
    provenance = await _saved_provenance(db, agent_id, saved)
    if provenance is None:
        return state
    return PrivacyState(
        mode=state.mode,
        saved_preference=SavedModelPreference(
            vendor=saved.vendor,
            model=saved.model,
            route=saved.route,
            provenance=provenance,
        ),
        transition_from=state.transition_from,
    )


async def _fetch_record_rows(db: Any, agent_id: str) -> list:
    return await db.fetchall(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
        (agent_id, PRIVACY_STATE_KEY),
    )


async def _saved_provenance(
    db: Any, agent_id: str, saved: SavedModelPreference
) -> Optional[OverrideProvenance]:
    """Who wrote ``saved``, when the provenance row still describes it."""
    rows = await db.fetchall(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
        (agent_id, PRIVACY_SAVED_PROVENANCE_KEY),
    )
    if not rows or not isinstance(rows[0][0], str):
        return None
    try:
        record = json.loads(rows[0][0])
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("saved_preference") != saved.as_dict():
        return None
    return OverrideProvenance.from_dict(record.get("provenance"))


def _encode_saved_provenance(state: PrivacyState) -> str:
    saved = state.saved_preference
    provenance = saved.provenance if saved is not None and saved.pinned else None
    return json.dumps(
        {
            "saved_preference": saved.as_dict() if saved is not None else None,
            "provenance": provenance.to_dict() if provenance is not None else None,
        },
        sort_keys=True,
    )


async def write_privacy_state(db: Any, agent_id: str, state: PrivacyState) -> None:
    """Record ``state`` as ``agent_id``'s privacy state.

    The saved mandate's writer is written in the same statement, so the two
    rows never disagree about which mandate they describe.
    """
    now = utc_timestamp_parameter(db.backend_type, datetime.now(timezone.utc))
    await db.execute(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at) "
        "VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
        (
            agent_id,
            PRIVACY_STATE_KEY,
            state.encode(),
            now,
            agent_id,
            PRIVACY_SAVED_PROVENANCE_KEY,
            _encode_saved_provenance(state),
            now,
        ),
    )


@dataclass(frozen=True)
class RestoredPrivacyState:
    """What boot restores, and whether it had to fail closed to get there.

    ``rewrite`` is True when the record does not name the restored state and
    may be replaced: it does not validate, or it was left mid-transition. Boot
    records the restored state before any turn runs. ``record_read`` is False
    only at boot, when the record could not be read at all: nothing is written
    over it. A transition that cannot read the record is refused instead.
    """

    mode: PrivacyMode
    saved_preference: Optional[SavedModelPreference]
    recorded: bool
    failed_closed: bool = False
    rewrite: bool = False
    record_read: bool = True


def _saved_preference_in(rows: list) -> Optional[SavedModelPreference]:
    """The valid ``saved_preference`` a stored record holds, if any.

    A record that does not validate as a whole may still hold one. Never
    raises: anything unrecoverable is ``None``.
    """
    if not rows or not isinstance(rows[0][0], str):
        return None
    try:
        record = json.loads(rows[0][0])
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    try:
        return _decode_preference(record.get("saved_preference"))
    except PrivacyStateUnreadableError:
        return None


async def _fail_closed(db: Any, agent_id: str) -> RestoredPrivacyState:
    """Fail closed, keeping whatever saved mandate the record still holds.

    Reads the record again: a read that failed may succeed now. A database
    error on that read propagates, because nothing is then known about the
    record: it may hold the saved mandate a local-only mode displaced.
    """
    saved = _saved_preference_in(await _fetch_record_rows(db, agent_id))
    if saved is not None:
        logger.warning(
            "Keeping the saved model mandate %s recovered from the privacy "
            "record for %s",
            saved.describe(),
            agent_id,
        )
    return RestoredPrivacyState(
        mode=FAIL_CLOSED_PRIVACY_MODE,
        saved_preference=saved,
        recorded=True,
        failed_closed=True,
        rewrite=True,
    )


async def resolve_recorded_privacy_state(
    db: Any, agent_id: str, *, initial_mode: PrivacyMode
) -> RestoredPrivacyState:
    """The state a restart would restore from the record as it stands.

    No record: ``initial_mode``. A pending record: the more restrictive of its
    two modes. A record that does not validate: the most restrictive mode,
    holding any saved mandate the record still yields. A database error
    propagates, including one on the second read of a record that does not
    validate: the caller decides what a record it cannot read means. A
    transition refuses; boot fails closed (:func:`restore_privacy_state`).
    """
    try:
        state = await load_privacy_state(db, agent_id)
    except PrivacyStateUnreadableError as exc:
        logger.error(
            "Recorded privacy mode for %s is unreadable (%s); failing closed to %s",
            agent_id,
            exc,
            FAIL_CLOSED_PRIVACY_MODE.value,
        )
        return await _fail_closed(db, agent_id)
    if state is None:
        return RestoredPrivacyState(
            mode=initial_mode, saved_preference=None, recorded=False
        )
    if state.transition_from is not None:
        logger.warning(
            "Privacy transition %s -> %s for %s was never recorded as applied; "
            "resolving to the more restrictive mode, %s",
            state.transition_from.value,
            state.mode.value,
            agent_id,
            state.resolved_mode.value,
        )
    return RestoredPrivacyState(
        mode=state.resolved_mode,
        saved_preference=state.saved_preference,
        recorded=True,
        rewrite=state.transition_from is not None,
    )


async def restore_privacy_state(
    db: Any, agent_id: str, *, initial_mode: PrivacyMode
) -> RestoredPrivacyState:
    """Resolve the privacy mode an agent boots in.

    :func:`resolve_recorded_privacy_state`, except that a read that fails also
    fails closed: to the most restrictive mode, never ``initial_mode``. If the
    record cannot be read even once more, nothing is known about it, so it is
    left as it is rather than overwritten (``record_read`` False): a later boot
    that can read it restores both the mode and the saved mandate it holds.
    """
    try:
        return await resolve_recorded_privacy_state(
            db, agent_id, initial_mode=initial_mode
        )
    except Exception as exc:  # noqa: BLE001 - any read failure must fail closed
        if is_execution_control_error(exc):
            raise
        logger.error(
            "Could not read the recorded privacy mode for %s (%s: %s); failing "
            "closed to %s",
            agent_id,
            type(exc).__name__,
            exc,
            FAIL_CLOSED_PRIVACY_MODE.value,
        )
    try:
        return await _fail_closed(db, agent_id)
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        if is_execution_control_error(exc):
            raise
        logger.error(
            "Could not re-read the privacy record for %s (%s: %s); leaving it "
            "as it is, so this boot's fail-closed mode is not recorded",
            agent_id,
            type(exc).__name__,
            exc,
        )
    return RestoredPrivacyState(
        mode=FAIL_CLOSED_PRIVACY_MODE,
        saved_preference=None,
        recorded=True,
        failed_closed=True,
        record_read=False,
    )


def transition_origin(
    live_mode: PrivacyMode,
    live_saved: Optional[SavedModelPreference],
    recorded: RestoredPrivacyState,
) -> PrivacyState:
    """The settled state a privacy transition starts from.

    Normally the live state, which the record names. They part when a write
    failed: a transition that could not be recorded leaves the live mode where
    it was, while its pending record may resolve to a more restrictive mode;
    a transition the storage wrapper refused after it was recorded may leave
    a less restrictive one. Starting from the more restrictive of the two
    means no transition records less than either already guarantees.

    The saved mandate is the record's whenever it holds one: the record is the
    recovery source of truth, and the live copy can only be a guess boot made
    while it could not read the record. The live copy stands in when the
    record holds none.
    """
    mode = most_restrictive_privacy_mode(recorded.mode, live_mode)
    saved = recorded.saved_preference
    return PrivacyState(mode=mode, saved_preference=saved if saved is not None else live_saved)
