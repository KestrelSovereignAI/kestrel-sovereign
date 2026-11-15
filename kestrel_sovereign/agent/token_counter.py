"""
Token counting for context management.

Uses tiktoken for accurate OpenAI model token counts,
with fallback estimation for Ollama and other models.

Context limits are sourced from (in priority order):
1. Discovered limits (from API discovery at runtime)
2. Cached limits (persisted from previous discovery runs)
3. ModelCatalogService (model_catalog.toml overrides)
4. DEFAULT_CONTEXT_LIMIT fallback
"""

import contextlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Optional, List, Dict, Any, Iterator, Mapping, Tuple, Union

from kestrel_sovereign.paths import CONTEXT_LIMITS_CACHE_FILE_ENV, runtime_path_env

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_catalog_service = None

def _get_catalog_service():
    """Get catalog service lazily to avoid circular imports."""
    global _catalog_service
    if _catalog_service is None:
        try:
            from kestrel_sovereign.llm.model_catalog import get_catalog_service
            _catalog_service = get_catalog_service()
        except ImportError:
            logger.debug("ModelCatalogService not available")
    return _catalog_service

# Try to import tiktoken, but don't fail if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based estimation")


# --- Discovered limits registry (populated by model_discovery at runtime) ---
_discovered_context_limits: Dict[str, int] = {}

# --- Persistent cache of discovered limits ---
_cached_limits: Optional[Dict[str, int]] = None  # None = not loaded yet
_cached_limits_file: Optional[Path] = None  # the file _cached_limits was read from


def context_limits_cache_file() -> Path:
    """Where discovered context limits persist between runs.

    ``$KESTREL_CONTEXT_LIMITS_CACHE_FILE``, else
    ``~/.kestrel/discovered_context_limits.json``. Resolved on call, not at
    import (#3104), so the environment in force now decides it rather than
    whichever caller imported this module first.
    """
    return Path(
        runtime_path_env(
            CONTEXT_LIMITS_CACHE_FILE_ENV,
            str(Path.home() / ".kestrel" / "discovered_context_limits.json"),
        )
    ).expanduser()


def _load_cached_limits() -> Dict[str, int]:
    """Load cached context limits from disk. Lazy: read once per cache file."""
    global _cached_limits, _cached_limits_file
    cache_file = context_limits_cache_file()
    if _cached_limits is not None and _cached_limits_file == cache_file:
        return _cached_limits
    _cached_limits = {}
    _cached_limits_file = cache_file
    try:
        if cache_file.exists():
            _cached_limits = json.loads(cache_file.read_text())
            logger.info(f"Loaded {len(_cached_limits)} cached context limits from {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to load cached context limits: {e}")
    return _cached_limits


def _persist_cache() -> None:
    """Write discovered limits to disk for cold starts."""
    cache_file = context_limits_cache_file()
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing cache (preserve entries from providers not discovered this run)
        existing = _load_cached_limits()
        merged = {**existing, **_discovered_context_limits}
        cache_file.write_text(json.dumps(merged, indent=2, sort_keys=True))
        logger.debug(f"Persisted {len(merged)} context limits to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to persist context limits cache: {e}")


def register_discovered_limits(models: list) -> None:
    """Register context limits from discovered models and persist to cache.

    Called by model_discovery after API discovery completes.
    Models with a non-None context_limit populate the in-memory registry
    and are written to :func:`context_limits_cache_file`.
    """
    for model in models:
        ctx = getattr(model, 'context_limit', None)
        if ctx:
            _discovered_context_limits[model.id] = ctx
    if _discovered_context_limits:
        logger.info(f"Registered {len(_discovered_context_limits)} discovered context limits")
        _persist_cache()


DEFAULT_CONTEXT_LIMIT = 32768

# Characters per token estimate for non-tiktoken models
CHARS_PER_TOKEN_ESTIMATE = 4


# ---------------------------------------------------------------------------
# Upper bound on a request's input tokens (#4068)
#
# ``count`` uses tiktoken, OpenAI's tokenizer. On 2026-11-13 it counted a
# subagent request at 663,779 tokens that the provider counted at 1,158,906,
# so the gate sent a request the provider rejected. ``upper_bound`` bounds the
# request from its UTF-8 bytes instead:
#
#     bound = ceil(wire_bytes / bpt_min[family]) + overhead[route]
#
# ``bpt_min`` (bytes per token) only moves down and ``overhead`` (tokens the
# route adds that have no bytes on our side) only moves up, both learned from
# the counts providers report for requests whose bytes were captured before
# the send. A family tiktoken really tokenizes uses its tiktoken count plus
# ``overhead``. ``count`` and its callers are unchanged.
# ---------------------------------------------------------------------------

#: Bytes per token a byte-bounded family starts from. The one count we have
#: is 2,616,316 bytes for 1,158,906 tokens (2.26). Provisional until the six
#: fixture classes are recorded (#4085).
STARTING_BYTES_PER_TOKEN = 2.0
#: ``bpt_min`` never goes below this.
MIN_BYTES_PER_TOKEN = 1.0
#: ``overhead[route]`` is capped at this share of the model's context limit.
OVERHEAD_CAP_FRACTION = 0.02
#: A sample moves ``bpt_min`` only when its total input count is at or above
#: the smaller of this and a quarter of the context limit.
SAMPLE_FLOOR_TOKENS = 100_000
SAMPLE_FLOOR_LIMIT_FRACTION = 0.25
#: The ratchet's file, beside :func:`context_limits_cache_file`.
TOKEN_BOUND_CACHE_FILENAME = "token_bound_ratchet.json"
_TOKEN_BOUND_CACHE_VERSION = 1
#: The lock beside the ratchet file that serializes its writers across
#: processes: ``token_bound_ratchet.json.lock``. It is kept after release.
TOKEN_BOUND_LOCK_SUFFIX = ".lock"
#: The total time a writer waits for the lock before its value is held in
#: memory. The wait blocks the event loop every co-hosted agent runs on, so it
#: is at most 250 ms.
TOKEN_BOUND_LOCK_TIMEOUT_SECONDS = 0.25
#: How long a waiting writer sleeps between attempts, within that total.
TOKEN_BOUND_LOCK_POLL_SECONDS = 0.005

#: Content-block types costed by their bytes alone.
BYTE_COSTED_BLOCK_TYPES = frozenset({"text", "tool_use", "tool_result"})
#: Any other block type needs an entry here: the tokens one block of that type
#: costs beyond its bytes, backed by a recorded provider count. None is
#: recorded, so every other type is refused by name.
RECORDED_BLOCK_TOKEN_COSTS: Mapping[str, int] = MappingProxyType({})

_PROMPT_TOO_LONG_RE = re.compile(r"prompt is too long:\s*([0-9][0-9,]*)\s*tokens", re.IGNORECASE)
_FAMILY_RE = re.compile(r"[a-z]+")
_TIKTOKEN_FAMILY = "tiktoken"

#: Events an operator can count: gate refusals, samples that were not
#: learned, and values that could not be published. Read with
#: :func:`token_bound_event_counts`.
_TOKEN_BOUND_EVENTS: Counter = Counter()


def count_token_bound_event(name: str) -> None:
    """Count one token-bound event (a refusal, an unlearned excess)."""
    _TOKEN_BOUND_EVENTS[name] += 1


def token_bound_event_counts() -> Dict[str, int]:
    """A snapshot of the token-bound event counts in this process."""
    return dict(_TOKEN_BOUND_EVENTS)


@dataclass(frozen=True)
class TokenBound:
    """An upper bound on a request's input tokens, and what it was made of.

    ``wire_bytes`` is captured here, before the send, so a sample pairs a
    provider's count with the bytes of the request it counted, not with an
    array that grew afterwards.
    """

    tokens: int
    wire_bytes: int
    model: str
    family: str
    route: str
    #: The tokens before ``overhead``: the byte quotient, or the tiktoken count.
    measured_tokens: int
    overhead: int
    #: ``None`` for a tiktoken family.
    bytes_per_token: Optional[float]


@dataclass(frozen=True)
class MeasurementRefusal:
    """A request ``upper_bound`` cannot bound, and why."""

    reason: str


def _wire_json(part: Any) -> str:
    """One request part serialized as it is measured."""
    return json.dumps(part, ensure_ascii=True, separators=(",", ":"), default=str)


def _bare_model(model: str) -> str:
    """The model id without a ``vendor:route/`` or ``vendor/`` prefix."""
    if ":" in model and "/" in model:
        return model.split("/", 1)[1]
    if "/" in model:
        return model.rsplit("/", 1)[1]
    return model


def model_route(model: str) -> str:
    """``vendor:route`` of a route-qualified model id, else ``""``."""
    if ":" in model and "/" in model:
        return model.split("/", 1)[0]
    return ""


def _tiktoken_encoding(bare_model: str):
    """The encoding tiktoken really has for this model, or ``None``.

    The ``cl100k_base`` fallback ``TokenCounter`` uses for an unknown model is
    not one: it is a guess, so such a model is bounded by its bytes.
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(bare_model)
    except KeyError:
        return None


def model_family(model: str) -> str:
    """The family whose ``bpt_min`` bounds this model's requests."""
    bare = _bare_model(model).lower()
    if "claude" in bare:
        return "claude"
    if _tiktoken_encoding(bare) is not None:
        return _TIKTOKEN_FAMILY
    match = _FAMILY_RE.match(bare)
    return match.group(0) if match else "unknown"


def _request_additions(model: str) -> List[Any]:
    """What a route adds to the request, built in this process, as parts.

    Counted in ``wire_bytes`` for every route that could serve the model, so
    it bounds the route that does:

    * ``claude``: the Claude Code identity system block the plan route
      prepends, and one cache-control content block for each of the four
      breakpoints ``AnthropicAdapter._apply_cache_control`` can place. Each
      such block is at least what wrapping a string costs.
    * ``gpt-5``: the behaviour contract ``prepend_gpt5_overlay`` puts before
      the system prompt, with its separator.

    What a route adds outside this process (a provider's own tool-use prompt,
    the Codex app-server's instructions) has no bytes here and is left to
    ``overhead``.
    """
    additions: List[Any] = []
    if model_family(model) == "claude":
        from kestrel_sovereign.llm.anthropic_adapter import (
            CACHE_CONTROL_EPHEMERAL,
            CLAUDE_CODE_IDENTITY,
        )

        additions.append({
            "type": "text",
            "text": CLAUDE_CODE_IDENTITY,
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        })
        additions.extend(
            [{"type": "text", "text": "", "cache_control": CACHE_CONTROL_EPHEMERAL}] * 4
        )
    from kestrel_sovereign.llm.gpt5_overlay import (
        GPT5_BEHAVIOR_CONTRACT,
        is_gpt5_model_id,
    )

    if is_gpt5_model_id(model):
        additions.append(f"{GPT5_BEHAVIOR_CONTRACT}\n\n")
    return additions


def _unbounded_block_type(content: Any) -> Optional[str]:
    """The first block type in ``content`` with no cost, or ``None``."""
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            return type(block).__name__
        block_type = block.get("type")
        if (
            block_type not in BYTE_COSTED_BLOCK_TYPES
            and block_type not in RECORDED_BLOCK_TOKEN_COSTS
        ):
            return str(block_type)
        nested = _unbounded_block_type(block.get("content"))
        if nested is not None:
            return nested
    return None


def _recorded_block_tokens(content: Any) -> int:
    """Tokens the recorded block costs add for ``content``, beyond its bytes."""
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, dict):
            total += RECORDED_BLOCK_TOKEN_COSTS.get(block.get("type"), 0)
            total += _recorded_block_tokens(block.get("content"))
    return total


def _cause_chain(error: BaseException) -> Iterator[BaseException]:
    """``error`` and each exception it was raised from or during, once each."""
    seen: set = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _prompt_too_long_count(text: str) -> Optional[int]:
    match = _PROMPT_TOO_LONG_RE.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def rejected_prompt_tokens(error: BaseException) -> Optional[int]:
    """The count inside a provider's ``prompt is too long`` rejection.

    The service wraps the provider's error, so the whole cause chain is read.
    """
    for current in _cause_chain(error):
        counted = _prompt_too_long_count(str(current))
        if counted is not None:
            return counted
    return None


def rejected_prompt_samples(error: BaseException) -> List[Tuple[Optional[str], int]]:
    """Each ``prompt is too long`` count in a failed send, with the route that counted it.

    When every route of ``get_response`` failed (the path a subagent's first
    ``generate`` takes) the service raises ``LLMAllProvidersFailedError``,
    which holds each route's own error under the route's name: each count is
    returned under that name. Any other error is read through its cause chain:
    a route walk such as ``generate_with_messages`` raises its own error from
    the last route's, and the service names on each route attempt's error the
    route it was sent to (:func:`~kestrel_sovereign.llm.error_handling.attempt_route`).
    The count is returned under the route of the attempt error that carries
    it. A count no attempt error carries is returned with no route: which
    route counted it is not known, and it is never the route of an earlier
    response.
    """
    from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError, attempt_route

    for current in _cause_chain(error):
        if isinstance(current, LLMAllProvidersFailedError):
            samples: List[Tuple[Optional[str], int]] = []
            for route, route_error in current.errors.items():
                counted = (
                    rejected_prompt_tokens(route_error)
                    if isinstance(route_error, BaseException)
                    else _prompt_too_long_count(str(route_error))
                )
                if counted is not None:
                    samples.append((str(route), counted))
            return samples
    for current in _cause_chain(error):
        route = attempt_route(current)
        if route is not None:
            # The first attempt error in the chain is the one the send failed
            # on; every later link is its own cause.
            counted = rejected_prompt_tokens(current)
            return [] if counted is None else [(route, counted)]
    counted = rejected_prompt_tokens(error)
    return [] if counted is None else [(None, counted)]


def token_bound_cache_file() -> Path:
    """Where the ratchet persists: beside the context-limit cache."""
    return context_limits_cache_file().with_name(TOKEN_BOUND_CACHE_FILENAME)


def _sample_record(model: str, route: str, wire_bytes: int, total_input: int) -> Dict[str, Any]:
    """The sample stored with a value: sizes and counts, no content."""
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "route": route,
        "wire_bytes": wire_bytes,
        "total_input": total_input,
    }


@dataclass(frozen=True)
class PrivacyEpoch:
    """The privacy policy one agent has in force: whose it is, and which epoch.

    A sample observed while persistent writes are not allowed is held in
    memory for its owner's current epoch only. It applies to that owner's
    requests and is dropped when that owner's epoch changes. Co-hosted agents
    share one ratchet, so another agent's epoch never touches it.
    """

    owner: str
    epoch: str


class TokenBoundRatchetBusy(RuntimeError):
    """Another writer held the ratchet file's lock past the wait."""


class _VolatileSamples:
    """What one owner learned in its current privacy epoch, held in memory."""

    __slots__ = ("epoch", "bpt_min", "overhead")

    def __init__(self, epoch: str) -> None:
        self.epoch = epoch
        self.bpt_min: Dict[str, float] = {}
        self.overhead: Dict[str, int] = {}


#: The ratchet's two sections, as named in its file.
_BPT_MIN = "bpt_min"
_OVERHEAD = "overhead"


def _stricter(section: str, value: Union[int, float], than: Union[int, float]) -> bool:
    """Whether ``value`` bounds more conservatively than ``than``."""
    return value < than if section == _BPT_MIN else value > than


def _lock_contended(error: OSError) -> bool:
    """Whether a refused non-blocking lock attempt means another holder."""
    # POSIX flock: EWOULDBLOCK. Windows LockFileEx: ERROR_LOCK_VIOLATION (33).
    return isinstance(error, BlockingIOError) or getattr(error, "winerror", None) == 33


class _TokenBoundRatchet:
    """``bpt_min`` per family and ``overhead`` per route, for one cache file.

    The sample store (#4068 ruling rv-4068-2, S1-S8):

    * **Published.** The file is host-wide: every agent and every process
      using this cache directory shares it. A value describes a model family
      or a route, not an agent, and is stored with the sample that moved it.
    * **Writers.** Any process publishes one value at a time under the lock
      beside the file: it re-reads the file, keeps the stricter of its value
      and the one there (the lower ``bpt_min``, the higher ``overhead``, each
      with its own sample), and replaces the file atomically. The lock wait is
      at most :data:`TOKEN_BOUND_LOCK_TIMEOUT_SECONDS`.
    * **Held.** An entry that could not be published (lock timeout or
      ``OSError``) is held whole, with its sample, and applies in this process
      only. It is retired once the file holds an equal or stricter value for
      its key, checked on every file load and every publish; the next publish
      for its key writes the stricter of the two. It is never retried on its
      own, and it ends with the process.
    * **Volatile.** A sample observed while persistent writes are not allowed
      lives in memory for its owner's privacy epoch (:class:`PrivacyEpoch`),
      applies to that owner's bounds only, and is dropped only when that
      owner's epoch changes. Without an owner and an epoch it is not learned.
    * **Readers** take no lock and apply the strictest of the file's value,
      the held value and the calling owner's volatile value. A reader sees a
      whole file, old or new; the file's value for a key loosens only when the
      operator deletes the file or the entry. Another process's publish is
      visible at this process's next file load, and a held value does not
      survive a restart.
    * **Reset.** The file is re-read whenever it changes on disk, so deleting
      the file or one entry resets what the file held, and the running host
      sees it. A held value that never reached the file is not reset by that;
      the warning on each load names it. A missing or unreadable file means
      the starting values.
    * **Threads.** The in-process state is touched only from one thread, the
      event loop's: the subagent gate measures and feeds on it.
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(path.name + TOKEN_BOUND_LOCK_SUFFIX)
        self._stamp: Any = object()  # never equal to a real stamp
        self._bpt: Dict[str, Dict[str, Any]] = {}
        self._overhead: Dict[str, Dict[str, Any]] = {}
        #: Whole entries (``value`` and ``sample``) that could not be published.
        self._unpublished: Dict[str, Dict[str, Dict[str, Any]]] = {_BPT_MIN: {}, _OVERHEAD: {}}
        self._volatile: Dict[str, _VolatileSamples] = {}

    # -- reading -----------------------------------------------------------

    def _file_stamp(self) -> Any:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        except OSError as error:
            return ("unreadable", type(error).__name__)
        return (stat.st_ino, stat.st_mtime_ns, stat.st_size)

    def _refresh(self) -> None:
        stamp = self._file_stamp()
        if stamp == self._stamp:
            return
        self._stamp = stamp
        if stamp is None:
            logger.warning(
                "Token bound ratchet %s does not exist; using the starting values",
                self.path,
            )
            self._load({}, {})
            return
        self._load(*self._read())

    def _load(
        self, bpt: Dict[str, Dict[str, Any]], overhead: Dict[str, Dict[str, Any]],
    ) -> None:
        """Take the file's entries, and retire each held entry the file now covers."""
        self._bpt, self._overhead = bpt, overhead
        for section, on_disk in ((_BPT_MIN, bpt), (_OVERHEAD, overhead)):
            held = self._unpublished[section]
            for key in [
                key for key, entry in held.items()
                if key in on_disk
                and not _stricter(section, entry["value"], on_disk[key]["value"])
            ]:
                del held[key]
        names = self._held_names()
        if names:
            logger.warning(
                "Token bound ratchet %s: values this process could not publish "
                "still apply here until it exits, whatever the file holds: %s",
                self.path, names,
            )

    def _held_names(self) -> str:
        return ", ".join(
            f"{section}[{key or 'unknown'}]={entry['value']}"
            for section in (_BPT_MIN, _OVERHEAD)
            for key, entry in sorted(self._unpublished[section].items())
        )

    def _read(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """The file's valid entries; none when it is missing or unreadable."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}, {}
        except (OSError, UnicodeDecodeError, ValueError) as error:
            logger.warning(
                "Token bound ratchet %s is unreadable (%s); using the starting values",
                self.path, type(error).__name__,
            )
            return {}, {}
        if not isinstance(data, dict):
            logger.warning(
                "Token bound ratchet %s is not an object; using the starting values",
                self.path,
            )
            return {}, {}
        return (
            self._entries(data.get(_BPT_MIN), _BPT_MIN, self._valid_bpt),
            self._entries(data.get(_OVERHEAD), _OVERHEAD, self._valid_overhead),
        )

    def _entries(self, section: Any, name: str, valid) -> Dict[str, Dict[str, Any]]:
        if section is None:
            return {}
        if not isinstance(section, dict):
            logger.warning("Token bound ratchet %s: %s is not an object; ignored", self.path, name)
            return {}
        kept: Dict[str, Dict[str, Any]] = {}
        for key, entry in section.items():
            if isinstance(entry, dict) and valid(entry.get("value")):
                kept[str(key)] = entry
            else:
                logger.warning(
                    "Token bound ratchet %s: entry %s[%r] is invalid; ignored",
                    self.path, name, key,
                )
        return kept

    @staticmethod
    def _valid_bpt(value: Any) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0
        )

    @staticmethod
    def _valid_overhead(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def _volatile_for(self, privacy_epoch: Optional[PrivacyEpoch]) -> Optional[_VolatileSamples]:
        """The owner's samples for its current epoch, or ``None`` without an epoch.

        An owner whose epoch changed has its own samples dropped; no other
        owner's are touched.
        """
        if privacy_epoch is None:
            return None
        if not isinstance(privacy_epoch, PrivacyEpoch):
            raise TypeError(
                f"privacy_epoch must be a PrivacyEpoch, not {type(privacy_epoch).__name__}"
            )
        held = self._volatile.get(privacy_epoch.owner)
        if held is None or held.epoch != privacy_epoch.epoch:
            held = self._volatile[privacy_epoch.owner] = _VolatileSamples(privacy_epoch.epoch)
        return held

    def bytes_per_token(self, family: str, privacy_epoch: Optional[PrivacyEpoch]) -> float:
        self._refresh()
        candidates = [STARTING_BYTES_PER_TOKEN]
        stored = self._bpt.get(family)
        if stored is not None:
            candidates.append(float(stored["value"]))
        if family in self._unpublished[_BPT_MIN]:
            candidates.append(float(self._unpublished[_BPT_MIN][family]["value"]))
        held = self._volatile_for(privacy_epoch)
        if held is not None and family in held.bpt_min:
            candidates.append(held.bpt_min[family])
        return max(MIN_BYTES_PER_TOKEN, min(candidates))

    def overhead(
        self, route: str, cap: Optional[int], privacy_epoch: Optional[PrivacyEpoch],
    ) -> int:
        """``overhead[route]``; for an unknown route, the largest any route learned."""
        self._refresh()
        held = self._volatile_for(privacy_epoch)
        learned = [
            {key: entry["value"] for key, entry in self._overhead.items()},
            {key: entry["value"] for key, entry in self._unpublished[_OVERHEAD].items()},
            held.overhead if held is not None else {},
        ]
        if route:
            values = [0, *(source[route] for source in learned if route in source)]
        else:
            values = [0, *(value for source in learned for value in source.values())]
        value = max(values)
        return value if cap is None else min(value, cap)

    # -- learning ----------------------------------------------------------

    def record(
        self,
        bound: TokenBound,
        total_input: int,
        *,
        route: str,
        served_model: Optional[str],
        context_limit: Optional[int],
        persist: bool,
        privacy_epoch: Optional[PrivacyEpoch],
    ) -> None:
        """Learn from one provider count for a request bounded at ``bound``.

        ``served_model`` is the model a response says served the request; its
        count is learned only when that model is in the family the bound was
        computed for. ``None`` for the count inside a rejection, which names
        no model.
        """
        if total_input <= 0:
            return
        if served_model is not None:
            served_family = model_family(served_model) if served_model else None
            if served_family != bound.family:
                count_token_bound_event("sample_family_mismatch")
                logger.warning(
                    "Token bound sample not learned: %s served the request, not "
                    "a %s-family model as bounded (route=%s bound model=%s T=%d "
                    "wire_bytes=%d)",
                    f"model {served_model!r}" if served_model else "a response naming no model",
                    bound.family, route or "unknown", bound.model,
                    total_input, bound.wire_bytes,
                )
                return
        cap = int(context_limit * OVERHEAD_CAP_FRACTION) if context_limit else None
        overhead = self.overhead(route, cap, privacy_epoch)
        floor = SAMPLE_FLOOR_TOKENS
        if context_limit:
            floor = min(floor, int(context_limit * SAMPLE_FLOOR_LIMIT_FRACTION))
        if bound.bytes_per_token is not None and total_input >= floor:
            self._learn_bytes_per_token(
                bound, total_input, overhead,
                route=route, persist=persist, privacy_epoch=privacy_epoch,
            )
            return
        if bound.bytes_per_token is None:
            measured = bound.measured_tokens
        else:
            measured = math.ceil(
                bound.wire_bytes / self.bytes_per_token(bound.family, privacy_epoch)
            )
        excess = total_input - (measured + overhead)
        if excess <= 0:
            return
        if cap is None:
            self._unlearned(bound, total_input, route, "the context limit is unknown, so overhead has no cap")
            return
        raised = min(cap, overhead + excess)
        if raised > overhead:
            self._store(_OVERHEAD, route, raised, bound, total_input, route, persist, privacy_epoch)
        if overhead + excess > cap:
            self._unlearned(bound, total_input, route, f"overhead is at its cap of {cap:,} tokens")

    def _learn_bytes_per_token(
        self, bound: TokenBound, total_input: int, overhead: int, *,
        route: str, persist: bool, privacy_epoch: Optional[PrivacyEpoch],
    ) -> None:
        tokens = total_input - overhead
        if tokens <= 0:
            return
        current = self.bytes_per_token(bound.family, privacy_epoch)
        observed = bound.wire_bytes / tokens
        if observed >= current:
            return
        learned = max(MIN_BYTES_PER_TOKEN, observed)
        if learned < current:
            self._store(_BPT_MIN, bound.family, learned, bound, total_input, route, persist, privacy_epoch)
        if observed < MIN_BYTES_PER_TOKEN:
            self._unlearned(
                bound, total_input, route,
                f"bytes per token would fall below {MIN_BYTES_PER_TOKEN}",
            )

    def _unlearned(self, bound: TokenBound, total_input: int, route: str, why: str) -> None:
        count_token_bound_event("unlearned_excess")
        logger.warning(
            "Token bound sample not learned (%s): route=%s model=%s T=%d wire_bytes=%d bound=%d",
            why, route or "unknown", bound.model, total_input, bound.wire_bytes, bound.tokens,
        )

    def _store(
        self, section: str, key: str, value: Union[int, float], bound: TokenBound,
        total_input: int, route: str, persist: bool, privacy_epoch: Optional[PrivacyEpoch],
    ) -> None:
        logger.info(
            "Token bound ratchet: %s[%s] -> %s (route=%s model=%s T=%d wire_bytes=%d)",
            section, key or "unknown", value, route or "unknown", bound.model,
            total_input, bound.wire_bytes,
        )
        if not persist:
            held = self._volatile_for(privacy_epoch)
            if held is None:
                count_token_bound_event("volatile_sample_dropped")
                logger.warning(
                    "Token bound sample not held: persistent writes are not allowed and "
                    "no privacy epoch names whose sample it is (route=%s model=%s)",
                    route or "unknown", bound.model,
                )
                return
            getattr(held, section)[key] = value
            return
        self._publish(section, key, {
            "value": value,
            "sample": _sample_record(bound.model, route, bound.wire_bytes, total_input),
        })

    # -- publishing --------------------------------------------------------

    def _publish(self, section: str, key: str, entry: Dict[str, Any]) -> None:
        """Merge one learned entry into the file, under the writers' lock.

        An entry held for the same key, because an earlier publish failed, is
        merged too: the stricter of the two is written, with its own sample.
        """
        held = self._unpublished[section].get(key)
        if held is not None and _stricter(section, held["value"], entry["value"]):
            entry = held
        try:
            with self._locked():
                bpt, overhead = self._read()
                on_disk = {_BPT_MIN: bpt, _OVERHEAD: overhead}[section]
                current = on_disk.get(key)
                if current is None or _stricter(section, entry["value"], current["value"]):
                    on_disk[key] = entry
                    self._replace_file(bpt, overhead)
                else:
                    logger.info(
                        "Token bound ratchet: %s[%s] is already %s in %s; kept",
                        section, key or "unknown", current["value"], self.path,
                    )
                # The file now holds this key's value or a stricter one, so
                # loading it retires the entry held for this key.
                self._load(bpt, overhead)
                # Stamped while the lock is held, so a later writer's change
                # is seen as a change.
                self._stamp = self._file_stamp()
        except (OSError, TokenBoundRatchetBusy) as error:
            count_token_bound_event("ratchet_publish_failed")
            self._unpublished[section][key] = entry
            logger.warning(
                "Could not persist token bound ratchet %s (%s: %s); %s[%s]=%s "
                "applies in this process until it exits or the file holds a "
                "value as strict",
                self.path, type(error).__name__, error,
                section, key or "unknown", entry["value"],
            )

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the writers' lock beside the file, or raise ``TokenBoundRatchetBusy``.

        The wait is bounded: a writer runs on the event loop, and a holder only
        ever reads and replaces one small file.
        """
        from kestrel_sovereign.private_storage import (
            lock_file_descriptor,
            unlock_file_descriptor,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        try:
            deadline = time.monotonic() + TOKEN_BOUND_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    token = lock_file_descriptor(descriptor, blocking=False)
                    break
                except OSError as error:
                    if not _lock_contended(error):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TokenBoundRatchetBusy(
                            f"{self.lock_path} was held by another writer for "
                            f"{TOKEN_BOUND_LOCK_TIMEOUT_SECONDS}s"
                        ) from error
                    time.sleep(min(TOKEN_BOUND_LOCK_POLL_SECONDS, remaining))
            try:
                yield
            finally:
                # Closing the descriptor below releases the lock even if this fails.
                with contextlib.suppress(OSError):
                    unlock_file_descriptor(descriptor, token)
        finally:
            os.close(descriptor)

    def _replace_file(
        self, bpt: Dict[str, Dict[str, Any]], overhead: Dict[str, Dict[str, Any]],
    ) -> None:
        payload = json.dumps(
            {"version": _TOKEN_BOUND_CACHE_VERSION, _BPT_MIN: bpt, _OVERHEAD: overhead},
            indent=2, sort_keys=True,
        )
        fd, staging = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(staging, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(staging)
            raise


_ratchets: Dict[Path, _TokenBoundRatchet] = {}


def _ratchet() -> _TokenBoundRatchet:
    """The ratchet for the cache file the environment names now."""
    path = token_bound_cache_file()
    ratchet = _ratchets.get(path)
    if ratchet is None:
        ratchet = _ratchets[path] = _TokenBoundRatchet(path)
    return ratchet


def record_input_sample(
    bound: TokenBound,
    total_input: int,
    *,
    route: Optional[str] = None,
    served_model: Optional[str],
    persist: bool,
    privacy_epoch: Optional[PrivacyEpoch],
) -> None:
    """Feed one provider count for the request ``bound`` measured.

    ``total_input`` is the response's input, cache-read and cache-creation
    counts together, or the count inside a ``prompt is too long`` rejection.
    ``served_model`` is the model the response says served the request (``""``
    when it names none), and ``None`` for a rejection's count: a response's
    count is learned only when that model is in the bound's family.
    ``persist`` is whether persistent writes are allowed now: a sample that
    may not be persisted lives in memory for ``privacy_epoch`` only.
    """
    TokenCounter(bound.model).record_sample(
        bound, total_input, route=route, served_model=served_model,
        persist=persist, privacy_epoch=privacy_epoch,
    )


class TokenCounter:
    """
    Counts tokens in text using tiktoken when available,
    falling back to character-based estimation.
    """

    def __init__(self, model: str = "auto"):
        """
        Initialize token counter for a specific model.

        Args:
            model: The model name for tiktoken encoding lookup
        """
        self.model = model
        self.encoder = None
        self._use_tiktoken = False

        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the specific model
                self.encoder = tiktoken.encoding_for_model(model)
                self._use_tiktoken = True
                logger.debug(f"Using tiktoken for model: {model}")
            except KeyError:
                # Model not found, try cl100k_base (GPT-4 family)
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                    self._use_tiktoken = True
                    logger.debug(f"Using cl100k_base encoding for unknown model: {model}")
                except Exception:
                    logger.debug(f"No tiktoken encoding available for model: {model}")

        if not self._use_tiktoken:
            logger.debug(f"Using character estimation for model: {model}")

    def count(self, text: Any) -> int:
        """
        Count tokens in text, or in structured message content.

        Structured content — an Anthropic block array (``text`` /
        ``tool_use`` / ``tool_result``), or any nested payload — reaches the
        provider as JSON, so it is measured the way it is actually sent.
        This mirrors ``_count_tool_schema_tokens``, which already serialises
        tool schemas before counting them.

        Without this, a non-string fell through to the character estimate
        below and was measured as ``len(obj) // 4`` — the number of BLOCKS,
        not their size. A single ``tool_result`` block holding 400 KB counted
        as 7 tokens, so any budget fed block-form content was blind to
        essentially all of it.

        Args:
            text: Text, or structured content, to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if not isinstance(text, str):
            try:
                text = json.dumps(
                    text, sort_keys=True, separators=(",", ":"), default=str
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"content serialisation failed during measurement: {e}"
                )
                text = str(text)

        if self._use_tiktoken and self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, using estimation: {e}")

        # Fallback: estimate based on characters
        return len(text) // CHARS_PER_TOKEN_ESTIMATE

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of chat messages.

        Accounts for message structure overhead (role, separators).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count including overhead
        """
        total = 0

        # Base overhead per message (role name, separators)
        # OpenAI uses ~4 tokens per message for structure
        MESSAGE_OVERHEAD = 4

        for msg in messages:
            content = msg.get("content", "")
            total += self.count(content) + MESSAGE_OVERHEAD

        # Add priming tokens (assistant response start) only when there is a
        # conversation to prime.
        if messages:
            total += 3

        return total

    def upper_bound(
        self,
        system: Any,
        messages: Any,
        tools: Any,
        *,
        route: Optional[str] = None,
        privacy_epoch: Optional[PrivacyEpoch] = None,
    ) -> Union[TokenBound, MeasurementRefusal]:
        """An upper bound on the input tokens of one request, or why there is none.

        Takes the request's structured parts, never a string someone already
        serialized: the system prompt (``None`` when it sits inside
        ``messages``), the message array and the tool schemas. Each is
        serialized exactly once, as ``_wire_json`` does, and its UTF-8 bytes
        are summed with what the route adds in this process
        (:func:`_request_additions`). ``route`` names the route the request
        goes to when the caller knows it; otherwise it is read from a
        route-qualified model id. ``privacy_epoch`` applies the samples its
        owner holds in memory for that epoch.

        A request it cannot bound is returned as a :class:`MeasurementRefusal`,
        never raised: a part that does not serialize, or a content block of a
        type with no recorded cost.
        """
        for content in self._block_contents(system, messages):
            block_type = _unbounded_block_type(content)
            if block_type is not None:
                return MeasurementRefusal(
                    f"content block type {block_type!r} has no recorded token count, "
                    f"so its size cannot be bounded"
                )
        parts = [part for part in (system, messages, tools) if part is not None]
        parts.extend(_request_additions(self.model))
        serialized = []
        for part in parts:
            try:
                serialized.append(_wire_json(part))
            except (TypeError, ValueError, RecursionError) as error:
                return MeasurementRefusal(
                    f"a request part does not serialize ({type(error).__name__}: {error})"
                )
        wire_bytes = sum(len(text.encode("utf-8")) for text in serialized)
        extra = sum(
            _recorded_block_tokens(content)
            for content in self._block_contents(system, messages)
        )
        family = model_family(self.model)
        route = route or model_route(self.model)
        limit = self.resolved_context_limit()
        cap = int(limit * OVERHEAD_CAP_FRACTION) if limit else None
        ratchet = _ratchet()
        overhead = ratchet.overhead(route, cap, privacy_epoch)
        if family == _TIKTOKEN_FAMILY:
            encoder = _tiktoken_encoding(_bare_model(self.model).lower())
            # ``encode_ordinary``: request text is data, and a provider counts
            # a literal such as ``<|endoftext|>`` in it as ordinary text.
            # ``encode`` raises on it instead, and the gate would then refuse a
            # request that fits.
            measured = sum(len(encoder.encode_ordinary(text)) for text in serialized) + extra
            bytes_per_token = None
        else:
            bytes_per_token = ratchet.bytes_per_token(family, privacy_epoch)
            measured = math.ceil(wire_bytes / bytes_per_token) + extra
        return TokenBound(
            tokens=measured + overhead,
            wire_bytes=wire_bytes,
            model=self.model,
            family=family,
            route=route,
            measured_tokens=measured,
            overhead=overhead,
            bytes_per_token=bytes_per_token,
        )

    @staticmethod
    def _block_contents(system: Any, messages: Any) -> List[Any]:
        """Every value that can hold content blocks: the system and each message's content."""
        contents = [system]
        if isinstance(messages, list):
            contents.extend(
                message.get("content") for message in messages if isinstance(message, dict)
            )
        return contents

    def record_sample(
        self,
        bound: TokenBound,
        total_input: int,
        *,
        route: Optional[str] = None,
        served_model: Optional[str],
        persist: bool,
        privacy_epoch: Optional[PrivacyEpoch],
    ) -> None:
        """Feed the ratchet one provider count for the request ``bound`` measured.

        See :func:`record_input_sample`.
        """
        _ratchet().record(
            bound,
            total_input,
            route=route if route is not None else bound.route,
            served_model=served_model,
            context_limit=self.resolved_context_limit(),
            persist=persist,
            privacy_epoch=privacy_epoch,
        )

    def resolved_context_limit(self) -> Optional[int]:
        """
        The context window limit KNOWN for the current model, or None.

        Sources (in order of priority):
        1. Discovered limits (from API discovery this session)
        2. Cached limits (from previous discovery runs on disk)
        3. ModelCatalogService (model_catalog.toml overrides)
        4. DEFAULT_CONTEXT_LIMIT fallback

        Each source is consulted with the full model string first, then
        with the model-name-only portion (post-slash). The LLM service
        routes by ``"vendor:route/model"`` while the cache stores bare
        model IDs (``"claude-opus-4-7"``) — same model reachable via
        multiple routes, so the cache shouldn't carry route prefixes.
        This is normalization, not a workaround: the actual model ID is
        what discovery records and what catalog lookups expect.

        Returns:
            Maximum tokens allowed in context, or ``None`` when no discovery
            source, cache or catalog knows this model.
        """
        # Build candidate keys for discovered/cache/catalog lookup.
        # Route-qualified ``"<vendor>:<route>/<model_name>"`` splits on
        # the FIRST ``/`` so OpenRouter/HF-style model names containing
        # additional slashes (``google/gemini-pro``,
        # ``meta-llama/Llama-3.1-8B-Instruct``) survive intact and the
        # bare-model cache entry is actually found (codex round-7 P2
        # on PR #1396). For vendor-only ``"vendor/model"`` we keep the
        # legacy ``rsplit`` fallback (some cache entries are stored
        # under the bare model name too).
        candidates = [self.model]
        if ":" in self.model and "/" in self.model:
            candidates.append(self.model.split("/", 1)[1])
        elif "/" in self.model:
            candidates.append(self.model.rsplit("/", 1)[1])

        # Route-qualified models: give the catalog FIRST shot at any
        # route-keyed cap (key containing ``:``) before the bare-model
        # discovered/cache fallthrough. Discovery records the model's
        # full context window — it cannot know about route-level
        # per-turn caps (e.g. ChatGPT-Plus's ~20K payload limit on
        # openai:plan), so the manual catalog override is the only
        # authoritative source for those.
        #
        # The discriminator requires BOTH ``:`` AND ``/`` — the kestrel
        # route form is ``"<vendor>:<route>/<model_name>"``. This
        # excludes Ollama tags like ``"gemma2:9b"`` (which have ``:``
        # but no ``/`` — codex round-2 P2 on PR #1396) and vendor-only
        # forms like ``"openai/gpt-5.5"`` (no route to cap). And it
        # consults ONLY route-keyed catalog entries via
        # ``get_route_context_cap`` so bare-model catalog entries
        # cannot bypass discovery on non-capped routes like
        # ``openai:api/gpt-5.5`` (codex round-1 P2 on PR #1396).
        is_route_qualified = ":" in self.model and "/" in self.model
        if is_route_qualified:
            catalog = _get_catalog_service()
            if catalog is not None:
                limit = catalog.get_route_context_cap(self.model)
                if limit is not None:
                    return limit

        # 1. Discovered limits (populated by API discovery this session)
        for key in candidates:
            if key in _discovered_context_limits:
                return _discovered_context_limits[key]

        # 2. Cached limits (from previous discovery runs)
        cached = _load_cached_limits()
        for key in candidates:
            if key in cached:
                return cached[key]

        # 3. Catalog TOML overrides (bare-model fallback path — route
        # cases already handled above).
        catalog = _get_catalog_service()
        if catalog is not None:
            for key in candidates:
                limit = catalog.get_context_limit(key)
                if limit is not None:
                    return limit

        return None

    def get_context_limit(self) -> int:
        """The context limit to USE — the resolved one, or the default.

        Split from :meth:`resolved_context_limit` so callers who must not act
        on a guess can tell "we know this model's window" from "nothing knows
        it, here is 32768". Budget enforcement is such a caller: refusing a
        subagent at the 32768 default when it is actually running on a
        1,000,000-token model would kill working work. Delegating rather than
        re-implementing the lookup keeps the two answers from drifting.
        """
        limit = self.resolved_context_limit()
        if limit is None:
            logger.warning(
                f"Unknown model {self.model}, using default context limit "
                f"{DEFAULT_CONTEXT_LIMIT}"
            )
            return DEFAULT_CONTEXT_LIMIT
        return limit

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        if not text:
            return text

        current_tokens = self.count(text)
        if current_tokens <= max_tokens:
            return text

        if self._use_tiktoken and self.encoder:
            # Precise truncation with tiktoken
            try:
                tokens = self.encoder.encode(text)
                truncated_tokens = tokens[:max_tokens]
                return self.encoder.decode(truncated_tokens)
            except Exception:
                pass

        # Fallback: estimate character limit
        char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        return text[:char_limit]

    def fits_in_context(self, text: str, reserved_tokens: int = 0) -> bool:
        """
        Check if text fits within model's context window.

        Args:
            text: Text to check
            reserved_tokens: Tokens already used or reserved

        Returns:
            True if text fits
        """
        available = self.get_context_limit() - reserved_tokens
        return self.count(text) <= available


def get_token_counter(model: str) -> TokenCounter:
    """
    Factory function to get a token counter for a model.

    Args:
        model: Model name

    Returns:
        TokenCounter instance
    """
    return TokenCounter(model)
