"""Host-owned durable authority witnesses for locally spawned agents.

The child's ``spawned_by`` edge is the runtime receipt, but it cannot also be
the only evidence that the child was spawned: deleting that child-owned edge
would otherwise make a delegated identity look like an unrestricted root.
This registry is owned by the host, written before publication, and retained
as a tombstone after terminal retirement.  Terminal removal first advances an
active record to ``retiring``.  That durable intent denies restart across a
crash between routing withdrawal and the final ``retired`` tombstone.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from kestrel_sovereign.spawn.mandate import (
    SpawnMandate,
    validate_spawn_max_child_depth,
)


SPAWN_AUTHORITY_REGISTRY_FILENAME = ".kestrel-spawn-authority.json"
_REGISTRY_VERSION = 2
_READABLE_REGISTRY_VERSIONS = frozenset({1, _REGISTRY_VERSION})
_AUTHORITY_STATES = frozenset({"active", "retiring", "retired"})


def _identity_anchor_birth_status(identity_db: Path) -> bool | None:
    """Return whether a cold SQLite anchor proves identity birth.

    ``False`` is a positive proof that no atomic agent-node birth committed,
    while ``None`` means the slot cannot be inspected safely.  The distinction
    is authority-bearing: an unreadable/corrupt/WAL-bearing anchor retains its
    pending denial, but a schema-only SQLite shell left by interrupted
    inception must not reserve a name and spawn-cap slot forever.
    """

    try:
        resolved = identity_db.resolve(strict=True)
        metadata = resolved.stat()
    except FileNotFoundError:
        return False
    except (OSError, RuntimeError):
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None

    sidecars = (Path(f"{resolved}-wal"), Path(f"{resolved}-shm"))

    def sidecar_state() -> bool | None:
        for sidecar in sidecars:
            try:
                sidecar.stat()
            except FileNotFoundError:
                continue
            except OSError:
                return None
            else:
                return True
        return False

    # Immutable mode guarantees this authority read creates no SQLite
    # sidecars.  It ignores WAL content, so any present or unreadable sidecar
    # makes the result uncertain and preserves the denial.
    if sidecar_state() is not False:
        return None

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        table = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'graph_nodes'"
        ).fetchone()
        if table is None:
            result = False
        else:
            result = (
                connection.execute(
                    "SELECT 1 FROM graph_nodes "
                    "WHERE node_type = 'agent' LIMIT 1"
                ).fetchone()
                is not None
            )
    except sqlite3.Error:
        return None
    finally:
        if connection is not None:
            connection.close()

    # Refuse a stale immutable answer if WAL state appeared during the read.
    if sidecar_state() is not False:
        return None
    return result


def standalone_spawn_manager_base_dir(
    storage_path: str | Path | None,
) -> Path:
    """Return the private manager base owned by one standalone root agent."""

    if storage_path is not None:
        return Path(storage_path).expanduser().resolve().parent.parent

    from kestrel_sovereign.paths import project_dir

    return project_dir()


def spawn_authority_host_base_dir(
    storage_path: str | Path | None,
) -> Path:
    """Return the producing AgentManager base for one child database.

    Every manager creates children below ``<base>/agent_data/<child>``.  Direct
    boot has only the child's database path, so it must walk through both the
    child directory and the manager-owned ``agent_data`` directory to find the
    independent witness.  This is deliberately distinct from the base a root
    uses when it creates a private manager for *its own* descendants.
    """

    if storage_path is not None:
        return Path(storage_path).expanduser().resolve().parent.parent.parent

    from kestrel_sovereign.paths import project_dir

    return project_dir()


def _mandate_wire_json(mandate: SpawnMandate) -> str:
    """Return the exact JSON representation whose fields carry authority."""

    validate_spawn_max_child_depth(mandate.max_child_depth)
    if not isinstance(mandate.ttl_seconds, int) or isinstance(
        mandate.ttl_seconds, bool
    ):
        raise TypeError("spawn mandate TTL must be an integer")
    if not isinstance(mandate.parent_did, str) or not mandate.parent_did:
        raise TypeError("spawn mandate parent DID must be a non-empty string")
    if not isinstance(mandate.child_did, str) or not mandate.child_did:
        raise TypeError("spawn mandate child DID must be a non-empty string")
    if not isinstance(mandate.parent_signature, str) or not mandate.parent_signature:
        raise TypeError("spawn mandate signature must be a non-empty string")
    # Normalize the budget exactly as signing/persistence does, and reject NaN
    # or infinity at this independent authority boundary too.
    mandate._wire_budget_allocation()
    return json.dumps(
        mandate.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_proposal_created_at(
    proposal_created_at: str | None,
    signed_created_at: str,
) -> None:
    """Validate the unsigned inception timestamp retained for crash repair."""

    if proposal_created_at is None:
        return
    if not isinstance(proposal_created_at, str) or not proposal_created_at:
        raise TypeError("spawn authority proposal timestamp must be non-empty")
    try:
        proposal = datetime.fromisoformat(proposal_created_at)
        signed = datetime.fromisoformat(signed_created_at)
    except (TypeError, ValueError) as error:
        raise ValueError("spawn authority timestamps must be ISO-8601") from error
    if proposal.tzinfo is None or signed.tzinfo is None:
        raise ValueError("spawn authority timestamps must include timezone")
    if proposal > signed:
        raise ValueError(
            "spawn authority proposal timestamp cannot follow signed issuance"
        )


def _proposal_wire_json(mandate: SpawnMandate) -> str:
    """Return the exact unsigned proposal reserved before child inception."""

    validate_spawn_max_child_depth(mandate.max_child_depth)
    if not isinstance(mandate.ttl_seconds, int) or isinstance(
        mandate.ttl_seconds, bool
    ):
        raise TypeError("spawn mandate TTL must be an integer")
    if not isinstance(mandate.parent_did, str) or not mandate.parent_did:
        raise TypeError("spawn mandate parent DID must be a non-empty string")
    if mandate.child_did is not None:
        raise ValueError("pending spawn authority cannot name a child DID")
    if mandate.parent_signature is not None:
        raise ValueError("pending spawn authority cannot carry a signature")
    _validate_proposal_created_at(mandate.created_at, mandate.created_at)
    mandate._wire_budget_allocation()
    return json.dumps(
        mandate.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mandate_scope_wire_json(mandate: SpawnMandate) -> str:
    """Return signed fields whose values must survive proposal promotion."""

    payload = mandate.to_dict()
    for field_name in ("child_did", "parent_signature", "created_at"):
        payload.pop(field_name, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class SpawnAuthorityWitness:
    """One exact child authority record retained by its host."""

    child_name: str
    child_did: str
    parent_did: str
    mandate: SpawnMandate
    config: LocalAgentConfig
    proposal_created_at: str | None = None
    state: str = "active"

    @property
    def active(self) -> bool:
        return self.state == "active"

    @property
    def retired(self) -> bool:
        """Compatibility projection for callers interested in finality only."""

        return self.state == "retired"

    def to_payload(self) -> dict[str, object]:
        if self.state not in _AUTHORITY_STATES:
            raise ValueError("spawn authority state is invalid")
        _mandate_wire_json(self.mandate)
        _validate_proposal_created_at(
            self.proposal_created_at,
            self.mandate.created_at,
        )
        return {
            "child_name": self.child_name,
            "child_did": self.child_did,
            "parent_did": self.parent_did,
            "mandate": self.mandate.to_dict(),
            "config": self.config.model_dump(mode="json"),
            "proposal_created_at": self.proposal_created_at,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class PendingSpawnAuthority:
    """A durable restart denial written before child identity inception."""

    reservation_id: str
    child_name: str
    parent_did: str
    mandate: SpawnMandate
    config: LocalAgentConfig

    def to_payload(self) -> dict[str, object]:
        if not self.reservation_id:
            raise ValueError("pending spawn authority reservation ID is empty")
        if not isinstance(self.child_name, str) or not self.child_name:
            raise TypeError("pending spawn authority child name must be non-empty")
        if self.mandate.parent_did != self.parent_did:
            raise ValueError("pending spawn authority parent does not match proposal")
        _proposal_wire_json(self.mandate)
        return {
            "reservation_id": self.reservation_id,
            "child_name": self.child_name,
            "parent_did": self.parent_did,
            "mandate": self.mandate.to_dict(),
            "config": self.config.model_dump(mode="json"),
        }


class SpawnAuthorityRegistry:
    """Atomic host-side registry of active, retiring, and retired authority."""

    def __init__(self, base_data_dir: Path) -> None:
        self._base_data_dir = Path(base_data_dir).expanduser().resolve()
        # Standard container deployments mount ``agent_data`` while the image
        # root is disposable. Keep the host witness on that durable rail so a
        # container recreation cannot erase spawned provenance.
        self.path = (
            self._base_data_dir
            / "agent_data"
            / SPAWN_AUTHORITY_REGISTRY_FILENAME
        )
        self._lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._thread_lock = threading.RLock()
        # Every visible pre-inception reservation owns a separate advisory
        # lock for the lifetime of its producing registry object.  The OS
        # releases that lock on process death, which lets restart recovery
        # distinguish a live producer that has not written its birth record
        # yet from a genuinely orphaned denial.  The registry mutation lock
        # alone cannot carry this liveness signal once its write completes.
        self._owned_pending_locks: dict[str, tuple[int, Any, Path]] = {}

    @staticmethod
    def _lock_descriptor(descriptor: int) -> Any:
        """Take one process-wide exclusive advisory byte-range lock."""

        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class _Overlapped(ctypes.Structure):
                _fields_ = [
                    ("Internal", ctypes.c_size_t),
                    ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE),
                ]

            overlapped = _Overlapped()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.LockFileEx(
                wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
                0x00000002,  # LOCKFILE_EXCLUSIVE_LOCK
                0,
                1,
                0,
                ctypes.byref(overlapped),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return overlapped

        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return None

    @staticmethod
    def _unlock_descriptor(descriptor: int, token: Any) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import ctypes
            import msvcrt
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel32.UnlockFileEx(
                wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
                0,
                1,
                0,
                ctypes.byref(token),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return

        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)

    @staticmethod
    def _try_lock_descriptor(descriptor: int) -> tuple[bool, Any]:
        """Try to claim one owner lock without waiting for a live producer."""

        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class _Overlapped(ctypes.Structure):
                _fields_ = [
                    ("Internal", ctypes.c_size_t),
                    ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE),
                ]

            overlapped = _Overlapped()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            claimed = kernel32.LockFileEx(
                wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
                0x00000002 | 0x00000001,  # EXCLUSIVE | FAIL_IMMEDIATELY
                0,
                1,
                0,
                ctypes.byref(overlapped),
            )
            return bool(claimed), overlapped

        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return False, None
        return True, None

    def _pending_owner_lock_path(self, reservation_id: str) -> Path:
        """Return a path-safe liveness sidecar for one opaque reservation ID."""

        digest = hashlib.sha256(reservation_id.encode("utf-8")).hexdigest()
        return self.path.parent / f".{self.path.name}.pending-{digest}.lock"

    def _acquire_pending_owner_lock(self, reservation_id: str) -> None:
        if reservation_id in self._owned_pending_locks:
            raise RuntimeError("pending spawn authority owner lock is duplicated")
        path = self._pending_owner_lock_path(reservation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            token = self._lock_descriptor(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        self._owned_pending_locks[reservation_id] = (descriptor, token, path)

    def _release_pending_owner_lock(
        self,
        reservation_id: str,
        *,
        unlink: bool,
    ) -> None:
        owned = self._owned_pending_locks.pop(reservation_id, None)
        if owned is None:
            return
        descriptor, token, path = owned
        try:
            self._unlock_descriptor(descriptor, token)
        finally:
            os.close(descriptor)
        if unlink:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        """Release this process object's pending-reservation liveness locks."""

        for reservation_id in tuple(self._owned_pending_locks):
            self._release_pending_owner_lock(reservation_id, unlink=False)

    def __del__(self) -> None:  # pragma: no cover - interpreter cleanup timing
        try:
            self.close()
        except Exception:
            pass

    @contextmanager
    def _exclusive_mutation(self) -> Iterator[None]:
        """Serialize one complete registry read/check/write across processes."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            token = None
            try:
                token = self._lock_descriptor(descriptor)
                yield
            finally:
                try:
                    if token is not None or os.name != "nt":
                        self._unlock_descriptor(descriptor, token)
                finally:
                    os.close(descriptor)

    def _mutate(
        self,
        operation: Callable[
            [dict[str, SpawnAuthorityWitness]],
            tuple[object, bool],
        ],
    ) -> object:
        """Run one checked mutation without a lost-update window."""

        with self._exclusive_mutation():
            records, pending = self._read_state()
            result, changed = operation(records)
            if changed:
                self._save(records, pending)
            return result

    def _mutate_state(
        self,
        operation: Callable[
            [dict[str, SpawnAuthorityWitness], dict[str, PendingSpawnAuthority]],
            tuple[object, bool],
        ],
    ) -> object:
        """Run one checked records-and-pending mutation under the file lock."""

        with self._exclusive_mutation():
            records, pending = self._read_state()
            result, changed = operation(records, pending)
            if changed:
                self._save(records, pending)
            return result

    @staticmethod
    def _parse_record(key: str, raw: object) -> SpawnAuthorityWitness:
        if not isinstance(raw, dict):
            raise TypeError("spawn authority record must be a mapping")
        child_name = raw.get("child_name")
        child_did = raw.get("child_did")
        parent_did = raw.get("parent_did")
        state = raw.get("state")
        if state is None:
            # Read the first deployed draft of this registry without weakening
            # its meaning.  New writes have one explicit three-state field.
            retired = raw.get("retired", False)
            if not isinstance(retired, bool):
                raise TypeError("spawn authority retirement state must be boolean")
            state = "retired" if retired else "active"
        if not isinstance(child_name, str) or not child_name:
            raise TypeError("spawn authority child name must be non-empty")
        if not isinstance(child_did, str) or not child_did:
            raise TypeError("spawn authority child DID must be non-empty")
        if key != child_did:
            raise ValueError("spawn authority record key does not match child DID")
        if not isinstance(parent_did, str) or not parent_did:
            raise TypeError("spawn authority parent DID must be non-empty")
        if not isinstance(state, str) or state not in _AUTHORITY_STATES:
            raise ValueError("spawn authority state is invalid")
        mandate_raw = raw.get("mandate")
        if not isinstance(mandate_raw, dict):
            raise TypeError("spawn authority mandate must be a mapping")
        mandate = SpawnMandate(**mandate_raw)
        if mandate.child_did != child_did or mandate.parent_did != parent_did:
            raise ValueError("spawn authority mandate endpoints do not match record")
        _mandate_wire_json(mandate)
        proposal_created_at = raw.get("proposal_created_at")
        _validate_proposal_created_at(proposal_created_at, mandate.created_at)
        config = LocalAgentConfig.model_validate(raw.get("config"))
        return SpawnAuthorityWitness(
            child_name=child_name,
            child_did=child_did,
            parent_did=parent_did,
            mandate=mandate,
            config=config,
            proposal_created_at=proposal_created_at,
            state=state,
        )

    @staticmethod
    def _parse_pending(key: str, raw: object) -> PendingSpawnAuthority:
        if not isinstance(raw, dict):
            raise TypeError("pending spawn authority must be a mapping")
        reservation_id = raw.get("reservation_id")
        child_name = raw.get("child_name")
        parent_did = raw.get("parent_did")
        if not isinstance(reservation_id, str) or not reservation_id:
            raise TypeError("pending spawn authority reservation ID must be non-empty")
        if key != reservation_id:
            raise ValueError("pending spawn authority key does not match reservation ID")
        if not isinstance(child_name, str) or not child_name:
            raise TypeError("pending spawn authority child name must be non-empty")
        if not isinstance(parent_did, str) or not parent_did:
            raise TypeError("pending spawn authority parent DID must be non-empty")
        mandate_raw = raw.get("mandate")
        if not isinstance(mandate_raw, dict):
            raise TypeError("pending spawn authority proposal must be a mapping")
        mandate = SpawnMandate(**mandate_raw)
        if mandate.parent_did != parent_did:
            raise ValueError("pending spawn authority parent does not match proposal")
        _proposal_wire_json(mandate)
        config = LocalAgentConfig.model_validate(raw.get("config"))
        return PendingSpawnAuthority(
            reservation_id=reservation_id,
            child_name=child_name,
            parent_did=parent_did,
            mandate=mandate,
            config=config,
        )

    def _read_state(
        self,
    ) -> tuple[dict[str, SpawnAuthorityWitness], dict[str, PendingSpawnAuthority]]:
        if not self.path.exists():
            return {}, {}
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if (
            not isinstance(payload, dict)
            or payload.get("version") not in _READABLE_REGISTRY_VERSIONS
        ):
            raise ValueError("unsupported spawn authority registry version")
        raw_records = payload.get("records")
        if not isinstance(raw_records, dict):
            raise TypeError("spawn authority registry records must be a mapping")
        records = {
            key: self._parse_record(key, raw)
            for key, raw in raw_records.items()
            if isinstance(key, str)
        }
        if len(records) != len(raw_records):
            raise TypeError("spawn authority registry keys must be strings")
        raw_pending = payload.get("pending", {})
        if not isinstance(raw_pending, dict):
            raise TypeError("pending spawn authority entries must be a mapping")
        pending = {
            key: self._parse_pending(key, raw)
            for key, raw in raw_pending.items()
            if isinstance(key, str)
        }
        if len(pending) != len(raw_pending):
            raise TypeError("pending spawn authority keys must be strings")
        return records, pending

    def _read(self) -> dict[str, SpawnAuthorityWitness]:
        return self._read_state()[0]

    def _save(
        self,
        records: dict[str, SpawnAuthorityWitness],
        pending: dict[str, PendingSpawnAuthority],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "records": {
                child_did: record.to_payload()
                for child_did, record in sorted(records.items())
            },
            "pending": {
                reservation_id: reservation.to_payload()
                for reservation_id, reservation in sorted(pending.items())
            },
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.exists():
                os.chmod(temporary, self.path.stat().st_mode & 0o7777)
            else:
                os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            if os.name != "nt":
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory = os.open(self.path.parent, flags)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def records(self) -> tuple[SpawnAuthorityWitness, ...]:
        return tuple(self._read().values())

    def get(self, child_did: str) -> SpawnAuthorityWitness | None:
        return self._read().get(child_did)

    def pending(self) -> tuple[PendingSpawnAuthority, ...]:
        return tuple(self._read_state()[1].values())

    def get_pending(self, reservation_id: str) -> PendingSpawnAuthority | None:
        return self._read_state()[1].get(reservation_id)

    def pending_for_slot(
        self,
        *,
        child_name: str,
        config: LocalAgentConfig | None,
    ) -> PendingSpawnAuthority | None:
        """Return the pre-inception denial owning a routing name or data slot."""

        resolved_data_dir = (
            config.resolve_data_dir(self._base_data_dir)
            if isinstance(config, LocalAgentConfig)
            else None
        )
        matches = [
            reservation
            for reservation in self.pending()
            if reservation.child_name.casefold() == child_name.casefold()
            or (
                resolved_data_dir is not None
                and reservation.config.resolve_data_dir(self._base_data_dir)
                == resolved_data_dir
            )
        ]
        if len(matches) > 1:
            raise RuntimeError("spawn authority registry has ambiguous pending slots")
        return matches[0] if matches else None

    def reserve_pending(
        self,
        *,
        child_name: str,
        parent_did: str,
        mandate: SpawnMandate,
        config: LocalAgentConfig,
        max_authority_slots: int | None = None,
    ) -> PendingSpawnAuthority:
        """Durably deny restart before inception can create child identity data."""

        if not isinstance(child_name, str) or not child_name:
            raise TypeError("pending spawn authority child name must be non-empty")
        if not isinstance(mandate, SpawnMandate):
            raise TypeError("pending spawn authority proposal must be a SpawnMandate")
        if not isinstance(config, LocalAgentConfig):
            raise TypeError("pending spawn authority config must be local")
        if max_authority_slots is not None and (
            not isinstance(max_authority_slots, int)
            or isinstance(max_authority_slots, bool)
            or max_authority_slots < 0
        ):
            raise ValueError("pending spawn authority cap must be a non-negative integer")
        proposal = mandate.__class__(**mandate.to_dict())
        if proposal.parent_did != parent_did:
            raise ValueError("pending spawn parent does not match proposal")
        _proposal_wire_json(proposal)
        reservation = PendingSpawnAuthority(
            reservation_id=uuid.uuid4().hex,
            child_name=child_name,
            parent_did=parent_did,
            mandate=proposal,
            config=config.model_copy(deep=True),
        )

        def apply(
            records: dict[str, SpawnAuthorityWitness],
            pending: dict[str, PendingSpawnAuthority],
        ) -> tuple[object, bool]:
            collision = next(
                (
                    record
                    for record in records.values()
                    if record.state != "retired"
                    and (
                        record.child_name.casefold() == child_name.casefold()
                        or self.same_data_slot(record.config, config)
                    )
                ),
                None,
            )
            pending_collision = next(
                (
                    existing
                    for existing in pending.values()
                    if existing.child_name.casefold() == child_name.casefold()
                    or self.same_data_slot(existing.config, config)
                ),
                None,
            )
            if collision is not None or pending_collision is not None:
                raise RuntimeError(
                    "spawn authority child name or data slot is already reserved"
                )
            authority_slots = sum(
                record.state != "retired" for record in records.values()
            ) + len(pending)
            if (
                max_authority_slots is not None
                and authority_slots >= max_authority_slots
            ):
                raise ValueError(
                    "Spawn refused: shared authority reached the spawned-agent cap "
                    f"({max_authority_slots})."
                )
            pending[reservation.reservation_id] = reservation
            return reservation, True

        self._acquire_pending_owner_lock(reservation.reservation_id)
        try:
            return self._mutate_state(apply)  # type: ignore[return-value]
        except BaseException:
            self._release_pending_owner_lock(
                reservation.reservation_id,
                unlink=True,
            )
            raise

    def reap_orphaned_pending_without_birth(
        self,
        *,
        reservation_ids: tuple[str, ...],
    ) -> tuple[PendingSpawnAuthority, ...]:
        """Withdraw selected denials whose producer died before identity birth.

        Absence of an atomically committed agent node is necessary but not
        sufficient: a concurrent producer has the same empty-anchor shape
        while it is between reservation and inception. Claiming its
        non-blocking owner lock under the registry mutation lock proves that no
        producing process still owns that epoch.
        """

        if not isinstance(reservation_ids, tuple) or any(
            not isinstance(reservation_id, str) or not reservation_id
            for reservation_id in reservation_ids
        ):
            raise TypeError("pending spawn authority IDs must be a string tuple")
        selected = frozenset(reservation_ids)
        claimed: list[tuple[int, Any, Path]] = []

        def apply(
            _records: dict[str, SpawnAuthorityWitness],
            pending: dict[str, PendingSpawnAuthority],
        ) -> tuple[object, bool]:
            removed: list[PendingSpawnAuthority] = []
            for reservation_id in selected:
                reservation = pending.get(reservation_id)
                if reservation is None:
                    continue
                path = self._pending_owner_lock_path(reservation_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                descriptor = os.open(path, flags, 0o600)
                claimed_owner, token = self._try_lock_descriptor(descriptor)
                if not claimed_owner:
                    os.close(descriptor)
                    continue
                # Claim liveness before inspecting the birth record.  If a
                # producer were allowed to finish between an empty read and
                # this claim, recovery could reap a newly committed identity.
                claimed.append((descriptor, token, path))
                identity_db = (
                    reservation.config.resolve_data_dir(self._base_data_dir)
                    / "kestrel_prime.db"
                )
                # File creation is not identity birth: AsyncDatabase creates a
                # schema before inception atomically commits the agent node.
                # Reap only a positively absent birth record; every inspection
                # ambiguity preserves the denial.
                if _identity_anchor_birth_status(identity_db) is not False:
                    continue
                del pending[reservation_id]
                removed.append(reservation)
            return tuple(removed), bool(removed)

        try:
            result = self._mutate_state(apply)
        finally:
            for descriptor, token, path in claimed:
                try:
                    self._unlock_descriptor(descriptor, token)
                finally:
                    os.close(descriptor)
                path.unlink(missing_ok=True)
        assert isinstance(result, tuple)
        return result

    def withdraw_pending(
        self,
        *,
        reservation_id: str,
        child_name: str,
    ) -> None:
        """Release a denial only after inception proves no identity was written."""

        def apply(
            _records: dict[str, SpawnAuthorityWitness],
            pending: dict[str, PendingSpawnAuthority],
        ) -> tuple[object, bool]:
            existing = pending.get(reservation_id)
            if existing is None:
                raise RuntimeError("pending spawn authority reservation is missing")
            if existing.child_name != child_name:
                raise RuntimeError("pending spawn authority reservation changed owner")
            del pending[reservation_id]
            return None, True

        self._mutate_state(apply)
        self._release_pending_owner_lock(reservation_id, unlink=True)

    def release_pending_producer(self, *, reservation_id: str) -> None:
        """Relinquish liveness while retaining an ambiguous restart denial.

        A terminal producer can fail after creating a database but before the
        atomic agent-node birth boundary is knowable.  The pending record must
        remain fail-closed, but its advisory lock must not claim that the
        finished producer is still running: a later in-process reconciliation
        needs to be able to prove the shell is ownerless and reap it.
        """

        if not isinstance(reservation_id, str) or not reservation_id:
            raise TypeError("pending spawn authority ID must be a non-empty string")
        self._release_pending_owner_lock(reservation_id, unlink=False)

    def promote_pending(
        self,
        *,
        reservation_id: str,
        child_name: str,
        child_did: str,
        mandate: SpawnMandate,
        config: LocalAgentConfig,
        proposal_created_at: str,
    ) -> SpawnAuthorityWitness:
        """Atomically replace one pre-inception denial with signed authority."""

        if not isinstance(mandate, SpawnMandate):
            raise TypeError("signed spawn authority must be a SpawnMandate")
        if not isinstance(config, LocalAgentConfig):
            raise TypeError("signed spawn authority config must be local")
        if mandate.child_did != child_did:
            raise ValueError("spawn authority child DID does not match mandate")
        expected_wire = _mandate_wire_json(mandate)
        _validate_proposal_created_at(proposal_created_at, mandate.created_at)
        witness = SpawnAuthorityWitness(
            child_name=child_name,
            child_did=child_did,
            parent_did=mandate.parent_did,
            mandate=mandate,
            config=config.model_copy(deep=True),
            proposal_created_at=proposal_created_at,
        )

        def apply(
            records: dict[str, SpawnAuthorityWitness],
            pending: dict[str, PendingSpawnAuthority],
        ) -> tuple[object, bool]:
            reservation = pending.get(reservation_id)
            if reservation is None:
                raise RuntimeError("pending spawn authority reservation is missing")
            if (
                reservation.child_name != child_name
                or reservation.parent_did != mandate.parent_did
                or not self.same_data_slot(reservation.config, config)
                or reservation.mandate.created_at != proposal_created_at
                or _mandate_scope_wire_json(reservation.mandate)
                != _mandate_scope_wire_json(mandate)
            ):
                raise RuntimeError(
                    "signed spawn authority does not match its pending reservation"
                )
            if child_did in records:
                raise RuntimeError("spawn authority child DID is already recorded")
            collision = next(
                (
                    record
                    for record in records.values()
                    if record.state != "retired"
                    and (
                        record.child_name.casefold() == child_name.casefold()
                        or self.same_data_slot(record.config, config)
                    )
                ),
                None,
            )
            if collision is not None:
                raise RuntimeError(
                    "spawn authority child name or data slot is already active"
                )
            # One atomic registry replacement closes both crash directions:
            # there is never a persisted state in which the identity slot is
            # discoverable but neither pending-denied nor actively governed.
            del pending[reservation_id]
            records[child_did] = witness
            assert _mandate_wire_json(witness.mandate) == expected_wire
            return witness, True

        promoted = self._mutate_state(apply)  # type: ignore[assignment]
        self._release_pending_owner_lock(reservation_id, unlink=True)
        return promoted  # type: ignore[return-value]

    def authoritative_for_slot(
        self,
        *,
        child_name: str,
        config: LocalAgentConfig | None,
    ) -> SpawnAuthorityWitness | None:
        """Return a non-final witness owning a routing name or data slot.

        ``retiring`` continues to own its slot: otherwise a crash-ordered
        terminal intent could be bypassed by presenting a replacement DID.
        """

        resolved_data_dir = (
            config.resolve_data_dir(self._base_data_dir)
            if isinstance(config, LocalAgentConfig)
            else None
        )
        matches = [
            record
            for record in self._read().values()
            if record.state != "retired"
            and (
                record.child_name.casefold() == child_name.casefold()
                or (
                    resolved_data_dir is not None
                    and record.config.resolve_data_dir(self._base_data_dir)
                    == resolved_data_dir
                )
            )
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "spawn authority registry has ambiguous active slot ownership"
            )
        return matches[0] if matches else None

    def same_data_slot(
        self,
        first: LocalAgentConfig,
        second: LocalAgentConfig,
    ) -> bool:
        """Whether two operational configs select the same durable identity slot."""

        return first.resolve_data_dir(
            self._base_data_dir
        ) == second.resolve_data_dir(self._base_data_dir)

    def record_active(
        self,
        *,
        child_name: str,
        child_did: str,
        mandate: SpawnMandate,
        config: LocalAgentConfig,
        proposal_created_at: str | None = None,
        max_authority_slots: int | None = None,
    ) -> SpawnAuthorityWitness:
        if mandate.child_did != child_did:
            raise ValueError("spawn authority child DID does not match mandate")
        if max_authority_slots is not None and (
            not isinstance(max_authority_slots, int)
            or isinstance(max_authority_slots, bool)
            or max_authority_slots < 0
        ):
            raise ValueError("spawn authority cap must be a non-negative integer")
        witness = SpawnAuthorityWitness(
            child_name=child_name,
            child_did=child_did,
            parent_did=mandate.parent_did,
            mandate=mandate,
            config=config.model_copy(deep=True),
            proposal_created_at=proposal_created_at,
        )
        expected_wire = _mandate_wire_json(witness.mandate)
        _validate_proposal_created_at(
            witness.proposal_created_at,
            witness.mandate.created_at,
        )

        def apply(
            records: dict[str, SpawnAuthorityWitness],
            pending: dict[str, PendingSpawnAuthority],
        ) -> tuple[object, bool]:
            existing = records.get(child_did)
            if existing is not None:
                if not existing.active:
                    raise RuntimeError("refusing to reactivate terminal spawn authority")
                if (
                    existing.child_name.casefold() != child_name.casefold()
                    or not self.same_data_slot(existing.config, config)
                    or _mandate_wire_json(existing.mandate) != expected_wire
                    or existing.proposal_created_at != proposal_created_at
                ):
                    raise RuntimeError(
                        "spawn authority witness conflicts with existing record"
                    )
                return existing, False
            collision = next(
                (
                    record
                    for record in records.values()
                    if record.state != "retired"
                    and (
                        record.child_name.casefold() == child_name.casefold()
                        or self.same_data_slot(record.config, config)
                    )
                ),
                None,
            )
            if collision is not None:
                raise RuntimeError(
                    "spawn authority child name or data slot is already active"
                )
            authority_slots = sum(
                record.state != "retired" for record in records.values()
            ) + len(pending)
            if (
                max_authority_slots is not None
                and authority_slots >= max_authority_slots
            ):
                raise ValueError(
                    "Spawn refused: shared authority reached the spawned-agent cap "
                    f"({max_authority_slots})."
                )
            records[child_did] = witness
            return witness, True

        return self._mutate_state(apply)  # type: ignore[return-value]

    def withdraw_active(
        self,
        *,
        child_name: str,
        child_did: str,
        mandate: SpawnMandate,
    ) -> None:
        def apply(records: dict[str, SpawnAuthorityWitness]) -> tuple[object, bool]:
            existing = records.get(child_did)
            if existing is None:
                return None, False
            if not existing.active:
                raise RuntimeError("refusing to erase terminal spawn authority")
            if (
                existing.child_name != child_name
                or _mandate_wire_json(existing.mandate) != _mandate_wire_json(mandate)
            ):
                raise RuntimeError(
                    "refusing to erase a changed spawn authority witness"
                )
            del records[child_did]
            return None, True

        self._mutate(apply)

    @staticmethod
    def _with_state(
        existing: SpawnAuthorityWitness,
        state: str,
    ) -> SpawnAuthorityWitness:
        return SpawnAuthorityWitness(
            child_name=existing.child_name,
            child_did=existing.child_did,
            parent_did=existing.parent_did,
            mandate=existing.mandate,
            config=existing.config,
            proposal_created_at=existing.proposal_created_at,
            state=state,
        )

    def admit_retirement(
        self,
        *,
        child_name: str,
        child_did: str,
    ) -> tuple[bool, bool]:
        """Return ``(exists, transitioned)`` for one exact retirement intent."""

        results = self.admit_retirements(
            targets=((child_name, child_did),),
        )
        return results[(child_name, child_did)]

    def admit_retirements(
        self,
        *,
        targets: tuple[tuple[str, str], ...],
    ) -> dict[tuple[str, str], tuple[bool, bool]]:
        """Admit an exact retirement set in one durable registry mutation."""

        if not isinstance(targets, tuple):
            raise TypeError("spawn retirement targets must be a tuple")
        normalized: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        seen_dids: set[str] = set()
        for target in targets:
            if not isinstance(target, tuple) or len(target) != 2:
                raise TypeError("spawn retirement target must be a name/DID tuple")
            child_name, child_did = target
            if not isinstance(child_name, str) or not child_name:
                raise TypeError("retiring spawn name must be a non-empty string")
            if not isinstance(child_did, str) or not child_did:
                raise TypeError("retiring spawn DID must be a non-empty string")
            canonical_name = child_name.casefold()
            if canonical_name in seen_names or child_did in seen_dids:
                raise RuntimeError("spawn retirement target set contains a duplicate")
            seen_names.add(canonical_name)
            seen_dids.add(child_did)
            normalized.append((child_name, child_did))

        def apply(records: dict[str, SpawnAuthorityWitness]) -> tuple[object, bool]:
            results: dict[tuple[str, str], tuple[bool, bool]] = {}
            for child_name, child_did in normalized:
                existing = records.get(child_did)
                if existing is None:
                    results[(child_name, child_did)] = (False, False)
                    continue
                if existing.child_name.casefold() != child_name.casefold():
                    raise RuntimeError(
                        "spawn authority retirement identity does not match"
                    )
                if existing.retired:
                    raise RuntimeError(
                        "refusing retirement intent for finalized authority"
                    )
                results[(child_name, child_did)] = (
                    True,
                    existing.state == "active",
                )

            changed = False
            for child_name, child_did in normalized:
                existing = records.get(child_did)
                if existing is None or existing.state == "retiring":
                    continue
                records[child_did] = self._with_state(existing, "retiring")
                changed = True
            return results, changed

        result = self._mutate(apply)
        assert isinstance(result, dict)
        return result

    def begin_retirement(self, *, child_name: str, child_did: str) -> bool:
        """Durably deny restart before terminal routing withdrawal begins."""

        exists, _transitioned = self.admit_retirement(
            child_name=child_name,
            child_did=child_did,
        )
        return exists

    def cancel_retirement(self, *, child_name: str, child_did: str) -> bool:
        """Restore authority only when a refused terminal removal left it live."""

        def apply(records: dict[str, SpawnAuthorityWitness]) -> tuple[object, bool]:
            existing = records.get(child_did)
            if existing is None:
                return False, False
            if existing.child_name.casefold() != child_name.casefold():
                raise RuntimeError("spawn authority retirement identity does not match")
            if existing.retired:
                raise RuntimeError("refusing to cancel finalized spawn retirement")
            if existing.active:
                return True, False
            records[child_did] = self._with_state(existing, "active")
            return True, True

        return bool(self._mutate(apply))

    def retire(self, *, child_name: str, child_did: str) -> bool:
        """Finalize an active or crash-ordered retiring authority tombstone."""

        def apply(records: dict[str, SpawnAuthorityWitness]) -> tuple[object, bool]:
            existing = records.get(child_did)
            if existing is None:
                return False, False
            if existing.child_name.casefold() != child_name.casefold():
                raise RuntimeError("spawn authority retirement identity does not match")
            if existing.retired:
                return True, False
            records[child_did] = self._with_state(existing, "retired")
            return True, True

        return bool(self._mutate(apply))
