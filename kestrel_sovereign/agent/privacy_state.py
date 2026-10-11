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

``transition_from`` names the mode a transition started from while that
transition is being applied, and is cleared once it has been. A record that
still names it was left by a process that stopped mid-transition: boot cannot
tell which of the two modes was in force, so it takes the more restrictive one.
A record that does not validate at all, or one that cannot be read, could have
named any mode, so boot takes the most restrictive mode there is. Neither case
ever falls back to NORMAL. In both cases boot then records the mode it resolved,
keeping whatever saved mandate can still be recovered from the old record, so a
later boot that can read the record does not return to a less restrictive mode
than this one ran in.

The record is control-plane state, not user content: it is written in every
privacy mode, including the volatile ones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from kestrel_sovereign.privacy import (
    PRIVACY_MODES_MOST_RESTRICTIVE_FIRST,
    PrivacyMode,
    most_restrictive_privacy_mode,
)
from kestrel_sovereign.storage.timestamps import utc_timestamp_parameter

logger = logging.getLogger(__name__)

PRIVACY_STATE_KEY = "privacy_state"
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
    """

    vendor: Optional[str]
    model: Optional[str]
    route: Optional[str]

    @classmethod
    def from_preference(cls, preference: Mapping[str, Any]) -> "SavedModelPreference":
        """Capture ``LLMService.get_model_preference()`` output."""
        model = preference.get("model") or None
        if model is None:
            return cls(vendor=None, model=None, route=None)
        return cls(
            vendor=preference.get("vendor") or None,
            model=model,
            route=preference.get("route") or None,
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
        """The mode boot restores: the more restrictive one mid-transition."""
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
    return PrivacyState.decode(rows[0][0])


async def _fetch_record_rows(db: Any, agent_id: str) -> list:
    return await db.fetchall(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
        (agent_id, PRIVACY_STATE_KEY),
    )


async def write_privacy_state(db: Any, agent_id: str, state: PrivacyState) -> None:
    """Record ``state`` as ``agent_id``'s privacy state."""
    await db.execute(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (
            agent_id,
            PRIVACY_STATE_KEY,
            state.encode(),
            utc_timestamp_parameter(db.backend_type, datetime.now(timezone.utc)),
        ),
    )


@dataclass(frozen=True)
class RestoredPrivacyState:
    """What boot restores, and whether it had to fail closed to get there.

    ``rewrite`` is True when the record does not name the restored state: it
    could not be read (``failed_closed``) or it was left mid-transition. Boot
    records the restored state before any turn runs.
    """

    mode: PrivacyMode
    saved_preference: Optional[SavedModelPreference]
    recorded: bool
    failed_closed: bool = False
    rewrite: bool = False


async def _recoverable_saved_preference(
    db: Any, agent_id: str
) -> Optional[SavedModelPreference]:
    """The saved mandate a record that boot could not use still holds, if any.

    Reads the record again: a read that failed may succeed now, and a record
    that does not validate as a whole may still hold a valid
    ``saved_preference``. Never raises: anything unrecoverable is ``None``.
    """
    try:
        rows = await _fetch_record_rows(db, agent_id)
    except Exception as exc:  # noqa: BLE001 - recovery is best-effort
        logger.warning(
            "Could not re-read the privacy record for %s to recover its saved "
            "model mandate (%s: %s); none is kept",
            agent_id,
            type(exc).__name__,
            exc,
        )
        return None
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
    saved = await _recoverable_saved_preference(db, agent_id)
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


async def restore_privacy_state(
    db: Any, agent_id: str, *, initial_mode: PrivacyMode
) -> RestoredPrivacyState:
    """Resolve the privacy mode an agent boots in.

    No record: ``initial_mode`` (the mode the agent was constructed with), as
    before #3554. A record left mid-transition: the more restrictive of its
    two modes. A record that cannot be read, or a read that fails: the most
    restrictive mode, never ``initial_mode``, holding any saved mandate the
    record still yields.
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
    except Exception as exc:  # noqa: BLE001 - any read failure must fail closed
        logger.error(
            "Could not read the recorded privacy mode for %s (%s: %s); failing "
            "closed to %s",
            agent_id,
            type(exc).__name__,
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
            "Privacy transition %s -> %s was interrupted before it was recorded "
            "as applied; restoring the more restrictive mode, %s",
            state.transition_from.value,
            state.mode.value,
            state.resolved_mode.value,
        )
    return RestoredPrivacyState(
        mode=state.resolved_mode,
        saved_preference=state.saved_preference,
        recorded=True,
        rewrite=state.transition_from is not None,
    )
