"""The strategy ledger: patterns and blockers, canonical but not prompt-injected.

STRATEGY.yaml is a bootstrap file. Every byte of it is read into the system
prompt on every turn and truncated at ``DEFAULT_MAX_CHARS_PER_FILE`` (20,000)
by head+tail character offset. That is the right treatment for vision,
milestones and stakeholders — a short standing brief. It is the wrong treatment
for an append-only log: Emma's ``patterns_learned`` reached 358 rows and
``blockers`` 80, the file reached 266 KB, and the fraction of it the agent
actually saw fell to under 7% — selected by byte position, not by relevance
(#2954).

So the two growing sections move out into ``STRATEGY_LEDGER.yaml``, which is
deliberately **not** in ``DEFAULT_BOOTSTRAP_FILES``. It stays canonical for the
same reasons STRATEGY.yaml does — human-editable, diffable, survives loss of
the database, travels with the agent's directory rather than its storage
layer — and it is reached by query (:mod:`ledger_index` projects it into the
graph) rather than by injection.

Two properties this file is responsible for, both absent before:

1. **Row identity.** Every row carries an ``id``. ``strategy_resolve_blocker``
   used to match on the ``issue`` key and delete *every* row that shared it —
   one call returned ``removed_count: 10``. A row that cannot be named
   individually cannot be resolved individually.
2. **Supersession.** A row is retired in place (``superseded_at`` for a
   pattern, ``resolved_at`` for a blocker) rather than appended around. The
   history stays; the active view shrinks.

It also keeps the file's edits. The ledger is read once at start, so writing
the whole in-memory copy back would erase anything a human changed in the file
since -- the damage #2273 fixed for STRATEGY.yaml. A write now goes ahead only
over the bytes this ledger last read or wrote; a changed file is re-read and
only the calling tool's own change is applied to it (#3636).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .atomic_file import file_digest, read_file_bytes, replace_atomically

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by the import guard in feature.py
    import yaml
    HAS_YAML = True
except ImportError:  # pragma: no cover
    HAS_YAML = False

#: Not a bootstrap filename. Adding it to ``DEFAULT_BOOTSTRAP_FILES`` would
#: undo the entire point of the split.
LEDGER_FILENAME = "STRATEGY_LEDGER.yaml"

PATTERNS_KEY = "patterns_learned"
BLOCKERS_KEY = "blockers"

#: STRATEGY.yaml's default template shipped an empty ``patterns`` list while
#: every reader and writer used ``patterns_learned``. On Emma both keys were
#: present — one empty, one holding 358 rows. A reader that bound to the wrong
#: one would have reported a truthful zero against nothing. Migration folds any
#: rows found under the stray key into the real one and drops it.
LEGACY_PATTERNS_KEY = "patterns"

#: Written into STRATEGY.yaml when the sections move, so a human opening the
#: file can see where they went instead of concluding they were lost.
LEDGER_POINTER_KEY = "ledger_file"

_PATTERN_ID_PREFIX = "pat_"
_BLOCKER_ID_PREFIX = "blk_"


def _digest(*parts: Any) -> str:
    joined = "\x00".join(str(p or "").strip().lower() for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def pattern_row_id(entry: Dict[str, Any]) -> str:
    """Content-derived id for a pattern row.

    Derived from the pattern text alone so that correcting ``source`` or
    ``implication`` edits the row in place rather than minting a second one —
    the same rule :func:`strategy_decision_node_id` applies to decisions.
    """
    return _PATTERN_ID_PREFIX + _digest(entry.get("pattern"))


def blocker_row_id(entry: Dict[str, Any]) -> str:
    """Content-derived id for a blocker row.

    Includes the title, not just the issue key: Emma carried ten distinct rows
    under ``#2877``. Keying on the issue alone is what made them
    indistinguishable in the first place.
    """
    return _BLOCKER_ID_PREFIX + _digest(entry.get("issue"), entry.get("title"))


def _unique_id(candidate: str, taken: Iterable[str]) -> str:
    """Disambiguate a content-derived id that a sibling row already holds.

    Two rows with byte-identical content are still two rows, and the acceptance
    bar is that resolving one must not resolve the other. Determinism is worth
    keeping for the overwhelmingly common distinct case, so only true
    collisions pay the suffix.
    """
    taken = set(taken)
    if candidate not in taken:
        return candidate
    suffix = 2
    while f"{candidate}-{suffix}" in taken:
        suffix += 1
    return f"{candidate}-{suffix}"


def assign_row_ids(rows: List[Dict[str, Any]], minter) -> int:
    """Give every row a unique ``id``, returning how many were minted.

    Existing ids are never rewritten — an id is an address, and rewriting one
    would break every reference held outside this file, including the graph
    nodes projected from it.
    """
    taken = {
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }
    minted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "").strip():
            continue
        row_id = _unique_id(minter(row), taken)
        row["id"] = row_id
        taken.add(row_id)
        minted += 1
    return minted


def is_active_pattern(row: Dict[str, Any]) -> bool:
    return not str(row.get("superseded_at") or "").strip()


def is_active_blocker(row: Dict[str, Any]) -> bool:
    return not str(row.get("resolved_at") or "").strip()


def active_patterns(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if isinstance(r, dict) and is_active_pattern(r)]


def active_blockers(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if isinstance(r, dict) and is_active_blocker(r)]


#: Process-wide order of confirmed ledger states. Monotonic across every
#: :class:`StrategyLedger` object, not per object: a restarted feature builds a
#: fresh ledger over the same file, and its first snapshot must still order
#: AFTER everything the previous object persisted.
_PERSIST_SEQUENCE = itertools.count(1)

#: Only this module can vouch that a snapshot's rows are the file's rows. A
#: snapshot built anywhere else carries no witness and is therefore unconfirmed,
#: whatever sequence number it claims.
_PERSISTED_WITNESS = object()


@dataclass(frozen=True)
class LedgerSnapshot:
    """An immutable copy of the ledger as it stands ON DISK (#3051, #3320).

    The canonical-assertion producer retracts terminally, so the state it
    reconciles against must be one the file actually holds. Handing it the live
    :class:`StrategyLedger` let it read mutations that were never persisted --
    a supersede whose ``save()`` failed still sat in the shared in-memory data,
    and an overlapping projection retracted the row permanently while the file
    still called it active. A snapshot is taken only where the ledger can vouch
    for it: after :meth:`StrategyLedger.save` succeeds, or on a :meth:`load`
    that minted nothing. The rows are deep-copied at that instant, so no later
    mutation of the ledger can reach them.

    ``readable`` and ``has_canonical_file`` travel with the rows because "no
    rows", "could not be read" and "never existed" are three different facts,
    and the latter two must never authorize a retraction sweep.
    ``persisted_sequence`` orders confirmed states, so a consumer can refuse a
    snapshot older than one it has already acted on; ``path`` names the file
    the order is over.
    """

    path: Optional[Path]
    data: Mapping[str, Any]
    readable: bool
    has_canonical_file: bool
    persisted_sequence: Optional[int] = None
    _witness: Any = field(default=None, repr=False, compare=False)

    @property
    def confirmed_persisted(self) -> bool:
        """Whether the ledger itself vouched that these rows are on disk."""
        return (
            self._witness is _PERSISTED_WITNESS
            and self.persisted_sequence is not None
        )


def _rows(data: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


class LedgerChangeConflict(Exception):
    """A change cannot be applied to the ledger as it now stands.

    Raised by the change a caller hands :meth:`StrategyLedger.commit` when the
    rows it acts on are gone from the file, or no longer in the state it needs:
    a pattern already superseded, a blocker already resolved by hand. Applying
    it anyway would act on a row the caller never saw.
    """


class _LedgerFileError(Exception):
    """The file's bytes cannot be read as a ledger; the message says why."""


@dataclass
class LedgerCommit:
    """Truthful result of :meth:`StrategyLedger.commit`.

    ``error`` is ``None`` exactly when the change reached the file. Otherwise
    ``refused`` says which of two things happened: nothing was written and the
    change was undone in memory too (the file could not take it without losing
    something), or the change is held in memory and only the write failed.
    """

    #: What the change returned on the last pass that ran: after a reload,
    #: computed against the file as it now stands. A refused commit keeps what
    #: the change would have recorded.
    value: Any = None
    error: Optional[str] = None
    refused: bool = False
    #: The file had changed on disk, so it was re-read and only this change
    #: was applied to what it now holds.
    reloaded: bool = False

    @property
    def persisted(self) -> bool:
        return self.error is None


def _parse_ledger(raw: bytes) -> Optional[Dict[str, Any]]:
    """Parse the ledger's bytes; ``None`` for an empty file, which is an empty ledger.

    Raises :class:`_LedgerFileError` for bytes that are not a ledger.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _LedgerFileError(f"could not be read: {e}") from e
    if not HAS_YAML:
        raise _LedgerFileError("PyYAML is not installed, so it cannot be parsed")
    try:
        parsed = yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001
        # Not only YAMLError: the safe loader also raises ValueError,
        # AttributeError and RecursionError on hand edits it cannot read.
        raise _LedgerFileError(f"could not be parsed as YAML: {e}") from e
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise _LedgerFileError("is not a YAML mapping")
    for key in (PATTERNS_KEY, BLOCKERS_KEY):
        value = parsed.get(key)
        if key in parsed and not isinstance(value, list):
            # Do NOT normalize this away. Replacing a malformed section
            # with an empty list makes the next save silently delete
            # whatever was actually in there.
            raise _LedgerFileError(
                f"section {key!r} is a {type(value).__name__}, not a list"
            )
    return parsed


class StrategyLedger:
    """Load/mutate/persist ``STRATEGY_LEDGER.yaml``.

    Carries the same honesty contract as ``StrategicMemoryFeature._save``: a
    write either persisted or it did not, and :meth:`save` says which rather
    than swallowing the failure. A caller that reports success off an
    unpersisted mutation is lying about the agent's own record.
    """

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.data: Dict[str, Any] = {
            "version": 1,
            PATTERNS_KEY: [],
            BLOCKERS_KEY: [],
        }
        #: Set when an existing file could not be read, parsed, or understood.
        #: While it is set the in-memory ledger is NOT the file's contents, so
        #: every write is refused — see :meth:`save`.
        self.load_error: Optional[str] = None
        #: Ids minted in memory that are not yet on disk. Tracked rather than
        #: returned-and-discarded, because a mint the caller forgot to persist
        #: is an address that changes on the next restart.
        self._unsaved_normalization = 0
        #: The last state this ledger confirmed is on disk. Deliberately NOT
        #: refreshed by a mutation, and not by a failed save: see
        #: :class:`LedgerSnapshot`.
        self._persisted_snapshot: Optional[LedgerSnapshot] = None
        #: Digest of the file as this ledger last loaded or wrote it; ``None``
        #: while it has seen no file. A write goes ahead only while the file
        #: on disk still matches it (#3636).
        self._disk_digest: Optional[str] = None
        #: Why the last write was refused over a file that changed on disk or
        #: could not be checked, or ``None``. :meth:`save` keeps its
        #: ``Optional[str]`` contract, so this is how :meth:`commit` tells a
        #: refusal it can recover from by re-reading the file from a write
        #: that failed.
        self._write_refusal: Optional[str] = None
        #: Whether a failed write left a change in memory that the file does
        #: not hold. The next successful write carries it, unless the file
        #: changed meanwhile: a reload applies only the calling change.
        self._holds_unwritten_change = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether there is a path to persist to at all."""
        return self.path is not None and HAS_YAML

    @property
    def readable(self) -> bool:
        """Whether the in-memory ledger faithfully represents the file."""
        return self.load_error is None

    @property
    def has_canonical_file(self) -> bool:
        """Whether a canonical file was actually there to be read.

        :meth:`load` treats a missing file as a NEW ledger rather than an
        error, which is right -- that is how an agent that has never recorded a
        pattern starts. But "new" and "empty" are the same in-memory state, and
        a consumer that reconciles against this ledger cannot tell them apart
        from :attr:`readable` alone: an unmounted volume, a wiped data
        directory, or a ``StrategyLedger(None)`` all present as a readable
        ledger with zero rows, which reads as "every row was deleted".

        The graph index tolerates that because its writes are upserts and the
        next projection rebuilds what it removed. The canonical assertion
        producer does not: retraction is terminal for it, so a keep-set derived
        from a file that was never opened destroys the projection permanently.
        A consumer whose removals are irreversible must check this too.
        """
        return self.path is not None and self.path.exists()

    @property
    def needs_save(self) -> bool:
        """Whether normalization minted ids that are still only in memory."""
        return self._unsaved_normalization > 0

    @property
    def persisted_snapshot(self) -> Optional[LedgerSnapshot]:
        """The last confirmed on-disk state, or ``None`` if none exists yet.

        Never the live rows. A mutation whose save failed is invisible here,
        which is the whole point: a consumer that acts irreversibly on the
        ledger must only ever see what the file holds.
        """
        return self._persisted_snapshot

    def _capture_snapshot(self, confirmed: bool) -> None:
        self._persisted_snapshot = LedgerSnapshot(
            path=self.path,
            data=MappingProxyType(copy.deepcopy(self.data)),
            readable=self.readable,
            has_canonical_file=self.has_canonical_file,
            persisted_sequence=next(_PERSIST_SEQUENCE) if confirmed else None,
            _witness=_PERSISTED_WITNESS if confirmed else None,
        )

    def load(self) -> None:
        """Read the ledger from disk, tolerating absence but not malformation.

        A missing file is a new ledger, not an error — this is how an agent
        that has never recorded a pattern starts.

        A file that exists but cannot be read, parsed, or understood is a
        different thing entirely, and the distinction is load-bearing. The
        earlier version logged the failure and left an empty writable ledger
        behind, so the next ``strategy_add_pattern`` serialized that empty
        ledger straight over the canonical file: one unreadable byte and 358
        rows were gone. The failure is now recorded in :attr:`load_error`, and
        every subsequent write refuses until a human resolves it.
        """
        self._load()
        # A load that minted ids describes a state the file does not hold yet,
        # and an unreadable or absent file describes no state at all. Each
        # still yields a snapshot -- the consumer needs to be told WHY it may
        # not act -- but only a clean read is confirmed.
        self._capture_snapshot(
            confirmed=(
                self.readable and self.has_canonical_file and not self.needs_save
            )
        )

    def _load(self) -> None:
        self.load_error = None
        self._unsaved_normalization = 0
        self._disk_digest = None
        if not self.path:
            return
        try:
            raw = read_file_bytes(self.path)
        except OSError as e:
            self._fail_load(f"could not be read: {e}")
            return
        if raw is None:
            return
        try:
            parsed = _parse_ledger(raw)
        except _LedgerFileError as e:
            self._fail_load(str(e))
            return
        self._disk_digest = file_digest(raw)
        if parsed is None:
            # An empty file is an empty ledger, not a malformed one.
            return
        self.data = parsed
        self.normalize()

    def _fail_load(self, why: str) -> None:
        self.load_error = f"{self.path} {why}"
        logger.error(
            "Strategy ledger %s -- refusing to write over it. Fix or move the "
            "file; no pattern/blocker change will persist until then.",
            self.load_error,
        )

    def normalize(self) -> int:
        """Ensure both sections exist as lists and every row has an id.

        Only ever *adds* the sections when absent. A section that is present
        but malformed is a load failure (see :meth:`load`), never something to
        be quietly replaced with an empty list.
        """
        for key in (PATTERNS_KEY, BLOCKERS_KEY):
            if key not in self.data:
                self.data[key] = []
        self.data.setdefault("version", 1)
        minted = assign_row_ids(self.patterns, pattern_row_id)
        minted += assign_row_ids(self.blockers, blocker_row_id)
        self._unsaved_normalization += minted
        return minted

    def save(self) -> Optional[str]:
        """Persist the ledger. Returns ``None`` on success, else the reason.

        Writes only over the file this ledger last loaded or wrote. The ledger
        is read once at start, so a file edited since would otherwise be
        overwritten with the copy read then, silently erasing the edit
        (#3636). A changed file is refused and left untouched; a caller with a
        change to re-apply to the file as it now stands uses :meth:`commit`.
        """
        self._write_refusal = None
        if not self.path:
            return (
                "No ledger path configured -- strategic memory is not active, "
                "so nothing was persisted."
            )
        if self.load_error:
            # Fail closed. The in-memory ledger is not what the file holds, so
            # writing it is not an update — it is a deletion of everything the
            # parse failed to reach.
            return (
                f"Refusing to write {LEDGER_FILENAME}: {self.load_error}. The "
                "existing file was left untouched -- repair or move it, then "
                "restart."
            )
        if not HAS_YAML:
            return (
                f"PyYAML not installed -- cannot save {LEDGER_FILENAME}."
            )
        return self._write(self.data, expected_digest=self._disk_digest)

    def _refusal(self, reason: str) -> str:
        return (
            f"Refusing to write {LEDGER_FILENAME}: {reason}. The file was left "
            "untouched."
        )

    def _write(
        self, data: Dict[str, Any], expected_digest: Optional[str]
    ) -> Optional[str]:
        """Write ``data`` if the file still has the digest ``expected_digest``.

        Returns ``None`` on success, else the reason; a refusal over a changed
        or uncheckable file is also recorded in :attr:`_write_refusal`. On
        success ``data`` becomes the ledger and its digest the one the next
        write must find. Not a cross-process lock: an edit landing between the
        check and the replace is still lost, a window of one small file write
        rather than the agent's whole uptime.
        """
        self._write_refusal = None
        path = self.path
        try:
            on_disk = read_file_bytes(path)
        except OSError as e:
            self._write_refusal = (
                f"could not read {path} to check it for external edits: {e}"
            )
            return self._refusal(self._write_refusal)
        if file_digest(on_disk) != expected_digest:
            self._write_refusal = (
                f"{path} changed on disk since it was last read or written, "
                "and writing over it would erase that change"
            )
            return self._refusal(self._write_refusal)
        try:
            content = yaml.dump(
                data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=100,
            )
            replace_atomically(path, content)
        except (OSError, UnicodeError, yaml.YAMLError) as e:
            logger.error("Failed to save %s: %s", path, e)
            self._holds_unwritten_change = True
            return f"Failed to save {LEDGER_FILENAME}: {e}"
        self.data = data
        self._disk_digest = file_digest(content.encode("utf-8"))
        self._unsaved_normalization = 0
        self._holds_unwritten_change = False
        self._capture_snapshot(confirmed=True)
        return None

    def commit(self, change: Callable[["StrategyLedger"], Any]) -> LedgerCommit:
        """Apply ``change`` and persist it without erasing edits made on disk.

        ``change`` is one caller's mutation, written against a ledger: it finds
        the rows it acts on itself and raises :class:`LedgerChangeConflict`
        rather than act on rows that are gone or no longer in the state it
        needs. It is applied here and saved. If the file changed on disk since
        this ledger last loaded or wrote it, the in-memory change is undone,
        the file is re-read, and ``change`` alone is applied to what it now
        holds and written -- so the edit is kept rather than overwritten by
        the copy read at start (#3636).

        Refused, leaving the file untouched and the in-memory ledger as it
        was, when the file was removed, cannot be read as a ledger, or no
        longer takes the change. None of those may fall back to the in-memory
        copy: writing it is exactly the overwrite this exists to prevent.

        A write that fails outright (a full disk) keeps the change in memory,
        to be carried by the next successful write.
        """
        before = copy.deepcopy(self.data)
        unsaved = self._unsaved_normalization
        value: Any = None
        try:
            value = change(self)
        except LedgerChangeConflict as e:
            # The copy read at start has no room for the change. The file may
            # have been edited so that it does, and needs no restart for it.
            conflict: Optional[str] = str(e)
        else:
            conflict = None
            # Reset here too: a replaced ``save`` must not inherit a refusal
            # an earlier write recorded.
            self._write_refusal = None
            error = self.save()
            if error is None:
                return LedgerCommit(value=value)
            if self._write_refusal is None:
                # The write failed, or there is no file to write to: nothing
                # on disk is newer than this ledger, so the change is held.
                return LedgerCommit(value=value, error=error)
        self.data = before
        self._unsaved_normalization = unsaved
        if not self.path or self.load_error or not HAS_YAML:
            return LedgerCommit(
                value=value,
                error=self._refusal(f"this change does not apply: {conflict}"),
                refused=True,
            )
        held = self._holds_unwritten_change
        outcome = self._reload_and_reapply(change)
        if held and outcome.reloaded and not outcome.refused:
            logger.warning(
                "%s changed on disk while a change an earlier failed write "
                "left in memory was waiting to be written. The ledger now "
                "holds the file plus only this call's change; the earlier "
                "change was dropped and must be made again.",
                self.path,
            )
        if outcome.value is None:
            outcome.value = value
        return outcome

    def _refused_commit(self, reason: str) -> LedgerCommit:
        error = self._refusal(reason)
        logger.warning(error)
        return LedgerCommit(error=error, refused=True)

    def _reload_and_reapply(
        self, change: Callable[["StrategyLedger"], Any]
    ) -> LedgerCommit:
        """Apply ``change`` to the file as it is now, and write that."""
        path = self.path
        previous = self._disk_digest
        try:
            raw = read_file_bytes(path)
        except OSError as e:
            return self._refused_commit(f"could not read {path}: {e}")
        digest = file_digest(raw)
        reloaded = digest != previous
        fresh = StrategyLedger(path)
        if not reloaded:
            # The file is as this ledger last read or wrote it; only the check
            # failed to read it. The in-memory copy is still the file plus
            # anything a failed write left held, so the change goes on that.
            fresh.data = copy.deepcopy(self.data)
            fresh._unsaved_normalization = self._unsaved_normalization
        elif raw is None:
            return self._refused_commit(
                f"{path} was removed after it was loaded, and recreating it "
                "from the in-memory copy would undo that"
            )
        else:
            try:
                parsed = _parse_ledger(raw)
            except _LedgerFileError as e:
                return self._refused_commit(f"{path} {e}")
            if parsed is not None:
                fresh.data = parsed
            # Rows written by hand without an id get one, exactly as at load.
            fresh.normalize()
        try:
            value = change(fresh)
        except LedgerChangeConflict as e:
            return self._refused_commit(
                f"this change does not apply to {path} as it now stands: {e}"
            )

        if reloaded:
            logger.info(
                "%s changed on disk since it was last loaded or written "
                "(sha256 %s, now %s); re-read it and applied only this change "
                "rather than overwrite the edit with the in-memory copy",
                path,
                previous[:12] if previous else "none",
                digest[:12] if digest else "none",
            )
        error = self._write(fresh.data, expected_digest=digest)
        if error is None:
            return LedgerCommit(value=value, reloaded=reloaded)
        if self._write_refusal is not None:
            # Edited again between the re-read and the write.
            logger.warning(error)
            return LedgerCommit(
                value=value, error=error, refused=True, reloaded=reloaded
            )
        # The write failed. Hold the file as it now stands plus this change in
        # memory -- the same contract as any failed write -- rather than a
        # stale copy the reload replaced.
        self.data = fresh.data
        self._disk_digest = digest
        self._unsaved_normalization = fresh._unsaved_normalization
        return LedgerCommit(value=value, error=error, reloaded=reloaded)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        return _rows(self.data, PATTERNS_KEY)

    @property
    def blockers(self) -> List[Dict[str, Any]]:
        return _rows(self.data, BLOCKERS_KEY)

    def find(self, row_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Locate a row by id across both sections."""
        wanted = str(row_id or "").strip()
        if not wanted:
            return None, None
        for key, rows in ((PATTERNS_KEY, self.patterns), (BLOCKERS_KEY, self.blockers)):
            for row in rows:
                if str(row.get("id") or "") == wanted:
                    return key, row
        return None, None

    def blockers_matching(self, issue: str) -> List[Dict[str, Any]]:
        """Active blockers whose id or issue key matches ``issue``.

        Matching on the id first is what makes an individual row addressable;
        matching on the issue key is retained because that is the handle the
        agent has been using, and dropping it would break every existing call
        site for the sake of a cleaner signature.
        """
        wanted = str(issue or "").strip()
        if not wanted:
            return []
        by_id = [
            row for row in active_blockers(self.blockers)
            if str(row.get("id") or "") == wanted
        ]
        if by_id:
            return by_id
        bare = wanted.lstrip("#")
        return [
            row for row in active_blockers(self.blockers)
            if str(row.get("issue") or "").lstrip("#") == bare
        ]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_pattern(
        self, pattern: str, source: str = "", implication: str = ""
    ) -> Dict[str, Any]:
        rows = self.patterns
        entry = {
            "id": _unique_id(
                pattern_row_id({"pattern": pattern}),
                [str(r.get("id") or "") for r in rows],
            ),
            "pattern": pattern,
            "source": source,
            "implication": implication,
            "recorded_at": str(date.today()),
        }
        self.data.setdefault(PATTERNS_KEY, []).append(entry)
        return entry

    def add_blocker(
        self,
        issue: str,
        title: str,
        severity: str,
        owner: str,
        notes: str = "",
        repo: str = "",
    ) -> Dict[str, Any]:
        """Append one blocker.

        ``repo`` is recorded because ``#42`` names an issue only relative to a
        repository. Without it, reconciliation has to try every configured repo
        and bind to whichever one happens to contain a number 42 — which is how
        a blocker gets resolved against a stranger's closed ticket.
        """
        rows = self.blockers
        entry = {
            "id": _unique_id(
                blocker_row_id({"issue": issue, "title": title}),
                [str(r.get("id") or "") for r in rows],
            ),
            "issue": issue,
            "title": title,
            "severity": severity,
            "owner": owner,
            "repo": repo,
            "blocked_since": str(date.today()),
            "notes": notes,
        }
        self.data.setdefault(BLOCKERS_KEY, []).append(entry)
        return entry

    def resolve_blocker(self, row: Dict[str, Any], resolution: str = "") -> Dict[str, Any]:
        """Retire one blocker in place.

        In place, not removed: the row is the only record that the blocker
        existed, when it started, and who owned it. Deleting it would trade one
        unbounded list for an unbounded loss.
        """
        row["resolved_at"] = str(date.today())
        if resolution:
            row["resolution"] = resolution
        return row

    def supersede_pattern(
        self, row: Dict[str, Any], reason: str = "", superseded_by: str = ""
    ) -> Dict[str, Any]:
        row["superseded_at"] = str(date.today())
        if reason:
            row["superseded_reason"] = reason
        if superseded_by:
            row["superseded_by"] = superseded_by
        return row

    # ------------------------------------------------------------------
    # Migration out of STRATEGY.yaml
    # ------------------------------------------------------------------

    def absorb(self, strategy_data: Dict[str, Any]) -> Dict[str, int]:
        """Copy ledger-owned sections out of loaded STRATEGY.yaml data.

        Does not touch either file — the caller writes the ledger first and
        only strips STRATEGY.yaml once that write is confirmed, so an
        interrupted migration leaves the rows in the old file rather than
        nowhere.

        Idempotent: a row whose id is already in the ledger is skipped, so
        re-running after a partial migration converges instead of doubling.

        Refuses to run at all when the ledger could not be read, because
        absorbing into a ledger whose real contents are unknown is how a
        migration turns into a truncation.
        """
        report = {"patterns": 0, "blockers": 0}
        if self.load_error:
            report["error"] = self.load_error
            return report
        if not isinstance(strategy_data, dict):
            return report

        incoming_patterns = _rows(strategy_data, PATTERNS_KEY) + _rows(
            strategy_data, LEGACY_PATTERNS_KEY
        )
        report["patterns"] = self._absorb_rows(
            incoming_patterns, PATTERNS_KEY, pattern_row_id
        )
        report["blockers"] = self._absorb_rows(
            _rows(strategy_data, BLOCKERS_KEY), BLOCKERS_KEY, blocker_row_id
        )
        return report

    def _absorb_rows(
        self, incoming: List[Dict[str, Any]], key: str, minter
    ) -> int:
        """Copy one section's rows in, preserving multiplicity.

        A content-derived id collision is NOT evidence that a row is already
        migrated — two byte-identical legacy rows are still two rows, and the
        acceptance bar is that resolving one must not resolve the other. The
        earlier version skipped the second one, and since STRATEGY.yaml was
        stripped immediately afterwards, that skip was a deletion.

        So each incoming row is addressed by *occurrence*: the first row with
        a given content id takes the bare id, the second takes ``-2``, and so
        on — the same suffix shape :func:`_unique_id` mints. Comparing those
        deterministic per-occurrence ids against what the ledger already holds
        keeps a re-run idempotent (the same 5 duplicates absorb once) without
        collapsing distinct rows.
        """
        target = self.data.setdefault(key, [])
        taken = {
            str(row.get("id") or "")
            for row in target
            if isinstance(row, dict) and str(row.get("id") or "").strip()
        }
        absorbed = 0
        occurrences: Dict[str, int] = {}
        for row in incoming:
            copied = dict(row)
            declared = str(copied.get("id") or "").strip()
            if declared:
                # A row that already names itself keeps its address.
                if declared in taken:
                    continue
                copied["id"] = declared
            else:
                base = minter(copied)
                seen = occurrences.get(base, 0) + 1
                occurrences[base] = seen
                row_id = base if seen == 1 else f"{base}-{seen}"
                if row_id in taken:
                    # This exact occurrence is already in the ledger from an
                    # earlier run. A later occurrence of the same content is
                    # still absorbed, under its own suffix.
                    continue
                copied["id"] = row_id
            taken.add(copied["id"])
            target.append(copied)
            absorbed += 1
        return absorbed


def strip_ledger_sections(strategy_data: Dict[str, Any]) -> bool:
    """Remove the ledger-owned keys from STRATEGY.yaml's in-memory data.

    Returns whether anything changed. Called only after the ledger write is
    confirmed persisted.
    """
    changed = False
    for key in (PATTERNS_KEY, BLOCKERS_KEY, LEGACY_PATTERNS_KEY):
        if key in strategy_data:
            del strategy_data[key]
            changed = True
    if changed and strategy_data.get(LEDGER_POINTER_KEY) != LEDGER_FILENAME:
        strategy_data[LEDGER_POINTER_KEY] = LEDGER_FILENAME
    return changed


def has_ledger_sections(strategy_data: Dict[str, Any]) -> bool:
    """Whether STRATEGY.yaml still carries sections the ledger now owns."""
    if not isinstance(strategy_data, dict):
        return False
    return any(
        key in strategy_data
        for key in (PATTERNS_KEY, BLOCKERS_KEY, LEGACY_PATTERNS_KEY)
    )
