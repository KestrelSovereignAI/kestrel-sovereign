"""
Bootstrap Service for Kestrel agent wake-up and personality discovery.

Manages the first-time experience when a new agent comes online, guiding
them through a discovery conversation to establish their personality.
"""

import asyncio
import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.storage.agent_resource_store import (
    SOUL_MARKDOWN_RESOURCE_TYPE,
)
from kestrel_sovereign.storage.privacy_wrapper import (
    acquire_control_plane_capability,
    optional_transition_lock,
)

logger = logging.getLogger(__name__)


def _durable_user_writes_permitted(storage) -> bool:
    """Whether ``storage``'s privacy policy permits durable user-content writes.

    Volatile privacy modes (EPHEMERAL / ISOLATED / DEIDENTIFIED) return ``False``
    so bootstrap/discovery writes that carry user-derived content — the agent's
    free-text ``description``, the raw discovery conversation, the discovered
    user name — do NOT leak into the durable ``agent_metadata`` table or the
    identity graph node while volatile (#2672 live-path bypass). Raw storage or a
    facade without the policy surface (``None`` / offline CLI paths) returns
    ``True``, preserving prior behaviour where no privacy policy is in force.
    """
    checker = getattr(storage, "allows_persistent_writes", None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception:  # noqa: BLE001 - never let a policy probe block a write path
        return False


class PersistOutcome(Enum):
    """Explicit result of a durable write helper (#2672 review P2).

    Replaces the ambiguous ``bool`` where ``False`` conflated "intentionally
    skipped in a volatile privacy mode" with "failed to persist". Callers can
    then report a skipped write honestly instead of as either a false success
    (the PATCH endpoint) or a false failure (``restart_discovery``).
    """

    PERSISTED = "persisted"           # a durable write happened
    SKIPPED_PRIVACY = "skipped_privacy"  # intentionally not written (volatile mode)
    NOOP = "noop"                     # nothing to write (e.g., value was None)
    FAILED = "failed"                 # attempted but failed

    @property
    def wrote(self) -> bool:
        """True only when a durable write actually landed."""
        return self is PersistOutcome.PERSISTED

    @property
    def is_failure(self) -> bool:
        """True only for a genuine persistence failure (not a privacy skip/no-op)."""
        return self is PersistOutcome.FAILED

#: Cap on how many prior turns we seed into discovery history (#1490).
#: Bootstrap typically contributes 1 user turn + 1 wake-up greeting;
#: 20 is generous and keeps any backfill bounded.
_DISCOVERY_PRIOR_HISTORY_LIMIT = 20

#: Per-turn character cap. A long opening story from a chatty first
#: turn shouldn't dominate the seeded history. Truncation is marked
#: with an ellipsis so the LLM knows content was elided.
_DISCOVERY_PRIOR_HISTORY_CHAR_CAP = 2000

#: Roles we accept when seeding discovery history. Stray system /
#: tool / function messages from the upstream conversation store get
#: dropped so the discovery LLM sees a clean user/assistant turn
#: sequence.
_DISCOVERY_PRIOR_HISTORY_ROLES = {"user", "assistant"}

# Prompt file locations
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DISCOVERY_PROMPT_FILE = PROMPTS_DIR / "discovery_prompt.md"
SOUL_GENERATION_PROMPT_FILE = PROMPTS_DIR / "soul_generation_prompt.md"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
DEFAULT_SOUL_FILE = TEMPLATES_DIR / "default_soul.md"

#: Hard cap on a persisted agent description. Mirrors the
#: ``UpdateIdentityRequest.description`` validator on PATCH /api/identity
#: so a self-authored tagline can never exceed what an operator could set.
DESCRIPTION_MAX_LEN = 500


def _clip_description(text: str) -> Optional[str]:
    """Trim/normalize a candidate description; return None if it's empty."""
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > DESCRIPTION_MAX_LEN:
        text = text[: DESCRIPTION_MAX_LEN - 1].rstrip() + "…"
    return text


def _soul_section(content: str, header: str) -> str:
    """Return the body text under a ``## <header>`` section of a SOUL.md.

    Capture stops at the next ``## `` heading (or end of file). Returns an
    empty string when the section is absent.
    """
    capturing = False
    out: List[str] = []
    for line in (content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if capturing:
                break
            capturing = stripped[3:].strip().lower() == header.lower()
            continue
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


def derive_description_from_soul(content: str) -> Optional[str]:
    """Derive a one-line agent description from SOUL.md content.

    The agent authors its own SOUL.md during wake-up discovery, so this is
    the agent's self-description rather than a framework-imposed label.
    Preference order:

    1. The explicit ``## Tagline`` section (the line the agent wrote for
       exactly this purpose).
    2. The first sentence of ``## Who You Are`` as a fallback for SOUL docs
       authored before the Tagline section existed.

    Returns None when neither yields usable text (caller keeps the prior
    description rather than blanking it).
    """
    if not content:
        return None

    tagline = _soul_section(content, "Tagline")
    if tagline:
        for line in tagline.splitlines():
            line = line.strip().lstrip(">").strip().strip("*_`").strip()
            if line:
                return _clip_description(line)

    who = _soul_section(content, "Who You Are")
    if who:
        first_para = next(
            (p.strip().lstrip(">").strip() for p in who.splitlines() if p.strip()),
            "",
        )
        if first_para:
            # First sentence only — keep it tagline-length.
            sentence = re.split(r"(?<=[.!?])\s", first_para, maxsplit=1)[0].strip()
            return _clip_description(sentence)

    return None


async def persist_agent_description(
    db, storage, agent_id: str, description: str, *, transition_lock=None
) -> PersistOutcome:
    """Persist an agent description to both stores the read paths consult.

    The authoritative read order (``get_agent_card`` / ``GET /api/identity``)
    is the agent graph node first, then the ``agent_metadata`` row as a
    fallback — so we write both, matching the prior inline PATCH behaviour.
    This is the single write path shared by PATCH /api/identity and the
    SOUL-driven self-description in :meth:`BootstrapService.save_soul_md`.

    Write failures are **not** swallowed: a failed metadata write, or a
    failed update of an existing graph node, propagates to the caller.
    Because the graph node is read first, swallowing a node-write failure
    would let a stale description survive behind a "success" — so the
    operator-facing PATCH path lets the exception become a 500, while the
    SOUL path wraps this call to stay best-effort (a self-description must
    never block saving the SOUL itself).

    Returns an explicit :class:`PersistOutcome` (#2672 review P2) rather than a
    bool so callers can distinguish an intentional privacy skip from a write:

    - ``NOOP`` when ``description is None`` (nothing to write; an empty string is
      still allowed so an operator can deliberately clear the field via PATCH),
    - ``SKIPPED_PRIVACY`` when a volatile privacy mode gates the durable write,
    - ``PERSISTED`` once the ``agent_metadata`` row (and the identity graph node,
      when present) have been written.
    """
    if description is None:
        return PersistOutcome.NOOP

    # Privacy boundary (#2672): the description is user/operator-derived free
    # text (PATCH /api/identity body, or a SOUL-derived tagline). It is gated
    # here at its single source of truth rather than by blanket-trusting the
    # ``agent`` node type — in a volatile mode BOTH the durable ``agent_metadata``
    # row and the identity graph node are skipped, so no new user-derived
    # description reaches durable storage. This closes the direct ``agent_metadata``
    # write that bypassed the graph privacy boundary entirely.
    #
    # The privacy check and the durable writes are serialized under the agent's
    # privacy-transition lock (#2672 review P1 race): otherwise a concurrent
    # ``set_privacy_mode`` could flip NORMAL→EPHEMERAL between the check and the
    # ``await``ed writes, persisting the description after the mode became volatile.
    # ``transition_lock`` is ``None`` on paths already holding it (SOUL derivation
    # inside ``save_soul_md``) or with no running agent (CLI); those run unguarded.
    async with optional_transition_lock(transition_lock):
        if not _durable_user_writes_permitted(storage):
            logger.debug(
                "persist_agent_description: skipping durable description write for "
                "%s — persistent writes are disabled in the current privacy mode "
                "(#2672)",
                agent_id,
            )
            return PersistOutcome.SKIPPED_PRIVACY

        now = datetime.now(timezone.utc)
        await db.execute(
            """
            INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (agent_id, "description", description, now),
        )

        if storage is not None:
            node = await storage.get_node(agent_id)
            if node:
                node.properties["description"] = description
                # The identity graph node is a control-plane type: this write is on
                # the persistent-mode path (volatile modes returned above), and the
                # wrapper only enforces the capability while governance is active, so
                # no capability is needed here (#2672).
                await storage.add_node(node)

        return PersistOutcome.PERSISTED


class BootstrapState(Enum):
    """States for the bootstrap/discovery process."""
    PENDING = "pending"      # Never started - will show wake-up on first message
    DISCOVERY = "discovery"  # In discovery conversation
    AVATAR = "avatar"        # Offering avatar generation
    COMPLETE = "complete"    # Bootstrap finished


@dataclass(frozen=True)
class RestartDiscoveryResult:
    """Structured result for discovery reset side effects."""

    message: str
    history_clear_succeeded: bool
    history_count_after: int
    state_reset: bool
    soul_deleted: bool
    soul_path: Optional[str] = None
    history_clear_error: Optional[str] = None

    def __str__(self) -> str:
        return self.message

    def lower(self) -> str:
        """Backward-compatible string-like helper for older tests/callers."""
        return self.message.lower()


@dataclass(frozen=True)
class BootstrapStaleness:
    """Status for a bootstrap-pending timeout check."""

    is_stale: bool
    state: BootstrapState
    age_seconds: Optional[float] = None
    created_at: Optional[str] = None
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_stale": self.is_stale,
            "state": self.state.value,
            "age_seconds": self.age_seconds,
            "created_at": self.created_at,
            "status": self.status,
        }


class BootstrapService:
    """
    Manages agent wake-up and personality discovery.

    The bootstrap flow:
    1. First message triggers wake-up greeting (PENDING -> DISCOVERY)
    2. Discovery conversation learns about user (2-4 exchanges)
    3. Optional avatar generation offered (DISCOVERY -> AVATAR)
    4. SOUL.md generated and saved (AVATAR -> COMPLETE)
    5. Normal operation begins
    """

    # Metadata keys for agent_metadata table
    BOOTSTRAP_STATE_KEY = "bootstrap_state"
    BOOTSTRAP_STARTED_KEY = "bootstrap_started_at"
    BOOTSTRAP_COMPLETED_KEY = "bootstrap_completed_at"
    BOOTSTRAP_STATUS_KEY = "bootstrap_status"
    BOOTSTRAP_STALE_AT_KEY = "bootstrap_stale_at"
    DISCOVERY_HISTORY_KEY = "bootstrap_discovery_history"
    USER_NAME_KEY = "bootstrap_user_name"
    STALE_BOOTSTRAP_STATUS = "stale_bootstrap"
    DEFAULT_PENDING_TIMEOUT_SECONDS = 3600

    def __init__(
        self,
        db,
        agent_id: str,
        agent_name: str,
        llm_service,
        agent_data_path: Path,
        storage=None,
        capabilities: Optional[List[str]] = None,
        privacy_transition_lock=None,
    ):
        """
        Initialize the bootstrap service.

        Args:
            db: AsyncDatabase instance
            agent_id: The agent's DID
            agent_name: The agent's current name
            llm_service: LLM service for generating responses
            agent_data_path: Path to agent's data directory (for SOUL.md)
            storage: Storage facade for accessing agent resources
            capabilities: Names of the agent's currently enabled features.
                Fed into SOUL.md generation so the agent's self-authored
                tagline can reflect what it can actually do.
            privacy_transition_lock: The agent's ``_privacy_transition_lock``.
                Held across the privacy CHECK and the durable write in every
                direct user-content writer (discovery history, user name, SOUL,
                description) so a concurrent ``set_privacy_mode`` cannot flip the
                mode into the ``await`` gap and persist after it became volatile
                (#2672 review P1 race). ``None`` on offline/test paths → unguarded.
        """
        self.db = db
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.llm_service = llm_service
        self.agent_data_path = Path(agent_data_path) if agent_data_path else None
        self.storage = storage
        self.capabilities = list(capabilities) if capabilities else []
        self._privacy_transition_lock = privacy_transition_lock

        # Session-only discovery history for volatile privacy modes (#2672 review
        # P1). In EPHEMERAL / ISOLATED / DEIDENTIFIED the raw discovery
        # conversation must NOT reach the durable ``agent_metadata`` table, but the
        # exchange still has to accumulate ACROSS turns or the completion
        # condition (three exchanges) is unreachable and SOUL generation sees an
        # empty history. This in-memory list is the volatile-mode substitute:
        # written and read in place of the durable table for the lifetime of this
        # process, never persisted, and reset by ``restart_discovery``. ``None``
        # means "no volatile discovery has occurred yet" (reads yield ``[]``).
        # It is deliberately NOT consulted in a persistent mode, where the durable
        # table remains the single source of truth.
        self._session_discovery_history: Optional[List[Dict[str, str]]] = None

        # Load prompts
        self._discovery_prompt = self._load_prompt(DISCOVERY_PROMPT_FILE)
        self._soul_generation_prompt = self._load_prompt(SOUL_GENERATION_PROMPT_FILE)

    def _load_prompt(self, filepath: Path) -> str:
        """Load a prompt from file."""
        try:
            if filepath.exists():
                return filepath.read_text(encoding="utf-8").strip()
            logger.warning(f"Prompt file not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt {filepath}: {e}")
            return ""

    async def is_bootstrap_needed(self) -> bool:
        """
        Check if bootstrap is needed for this agent.

        An agent needs bootstrap only if:
        1. No SOUL.md exists (primary check — SOUL.md is the artifact)
        2. No conversation history exists (secondary — existing agents don't re-bootstrap)
        3. Bootstrap state is not COMPLETE (tertiary — DB state)

        If any evidence of an existing agent is found, auto-heal the DB state.
        """
        # SOUL.md is the primary artifact — if it exists, agent is configured
        if self.agent_data_path:
            soul_path = Path(self.agent_data_path) / "SOUL.md"
            if soul_path.exists() and soul_path.stat().st_size > 0:
                await self._ensure_complete("SOUL.md exists")
                return False

        if await self._has_canonical_soul_resource():
            await self._ensure_complete("canonical SOUL resource exists")
            return False

        # Existing conversation history means this is not a new agent
        try:
            history_count = await self.db.fetchall(
                "SELECT COUNT(*) FROM conversations WHERE agent_id = ?",
                (self.agent_id,),
            )
            if history_count and history_count[0][0] > 0:
                await self._ensure_complete("conversation history exists")
                return False
        except Exception:
            pass  # Table may not exist for truly new agents

        state = await self.get_bootstrap_state()
        if state == BootstrapState.COMPLETE:
            return False

        # State is DISCOVERY or AVATAR but stuck — check for timeout
        if state in (BootstrapState.DISCOVERY, BootstrapState.AVATAR):
            started = await self._get_started_time()
            if started:
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed > 3600:  # Stuck for more than 1 hour
                    logger.warning(
                        f"Bootstrap stuck in {state.value} for {elapsed:.0f}s — "
                        f"auto-completing with default SOUL.md"
                    )
                    await self.skip_discovery()
                    return False

        return True

    async def _ensure_complete(self, reason: str) -> None:
        """Mark bootstrap complete if it isn't already."""
        state = await self.get_bootstrap_state()
        if state != BootstrapState.COMPLETE:
            logger.info(f"Auto-completing bootstrap: {reason}")
            await self.set_bootstrap_state(BootstrapState.COMPLETE)

    async def _get_started_time(self) -> Optional[datetime]:
        """Get when bootstrap was started."""
        try:
            result = await self.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.BOOTSTRAP_STARTED_KEY),
            )
            if result:
                return datetime.fromisoformat(result[0][0])
        except Exception:
            pass
        return None

    async def get_bootstrap_state(self) -> BootstrapState:
        """Get the current bootstrap state from agent_metadata."""
        try:
            result = await self.db.fetchall(
                """
                SELECT value FROM agent_metadata
                WHERE agent_id = ? AND key = ?
                """,
                (self.agent_id, self.BOOTSTRAP_STATE_KEY),
            )
            if result:
                state_str = result[0][0]
                return BootstrapState(state_str)
            # No state stored — could be new agent or missing row
            return BootstrapState.PENDING
        except Exception as e:
            logger.warning(f"Failed to get bootstrap state: {e}")
            # On DB error, assume complete to avoid hijacking existing agents
            return BootstrapState.COMPLETE

    async def check_pending_timeout(
        self,
        *,
        agent_node: Any = None,
        storage: Any = None,
        threshold_seconds: int = DEFAULT_PENDING_TIMEOUT_SECONDS,
        now: Optional[datetime] = None,
        mark_stale: bool = True,
    ) -> BootstrapStaleness:
        """Flag agents that remain in PENDING past the bootstrap timeout.

        ``PENDING`` means the agent was created but never received its first
        turn, so ``bootstrap_started_at`` is intentionally absent. Use the
        graph agent node's ``created_at`` as the age anchor because inception
        stores the original pending state there.
        """
        state = await self._get_bootstrap_state_from_metadata_or_node(agent_node)
        if state != BootstrapState.PENDING:
            return BootstrapStaleness(is_stale=False, state=state)

        if await self._has_completion_evidence():
            await self._ensure_complete("bootstrap completion evidence exists")
            return BootstrapStaleness(is_stale=False, state=BootstrapState.COMPLETE)

        created_at = self._get_agent_created_at(agent_node)
        if not created_at:
            return BootstrapStaleness(
                is_stale=False,
                state=state,
                status="unknown_created_at",
            )

        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age_seconds = (now - created_at).total_seconds()
        created_at_str = created_at.isoformat()
        if age_seconds <= threshold_seconds:
            return BootstrapStaleness(
                is_stale=False,
                state=state,
                age_seconds=age_seconds,
                created_at=created_at_str,
            )

        stale = BootstrapStaleness(
            is_stale=True,
            state=state,
            age_seconds=age_seconds,
            created_at=created_at_str,
            status=self.STALE_BOOTSTRAP_STATUS,
        )
        if mark_stale:
            await self.mark_stale_bootstrap(stale, agent_node=agent_node, storage=storage)
        return stale

    async def mark_stale_bootstrap(
        self,
        stale: Optional[BootstrapStaleness] = None,
        *,
        agent_node: Any = None,
        storage: Any = None,
    ) -> None:
        """Persist the stale-bootstrap escalation in metadata and graph state."""
        now = datetime.now(timezone.utc)
        stale_at = now.isoformat()
        age_seconds = None if stale is None else stale.age_seconds
        try:
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.BOOTSTRAP_STATUS_KEY, self.STALE_BOOTSTRAP_STATUS, now),
            )
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.BOOTSTRAP_STALE_AT_KEY, stale_at, now),
            )
        except Exception as exc:
            logger.warning("Failed to persist stale bootstrap metadata: %s", exc)

        if agent_node is None and storage is not None:
            try:
                agent_node = await storage.get_node(self.agent_id)
            except Exception as exc:
                logger.debug("Failed to load agent node for stale bootstrap mark: %s", exc)
                agent_node = None

        if agent_node is None or storage is None:
            return

        try:
            from copy import copy

            updated_node = copy(agent_node)
            updated_node.properties = dict(getattr(agent_node, "properties", {}) or {})
            updated_node.properties["bootstrap_state"] = BootstrapState.PENDING.value
            updated_node.properties["bootstrap_status"] = self.STALE_BOOTSTRAP_STATUS
            updated_node.properties["bootstrap_stale_at"] = stale_at
            if age_seconds is not None:
                updated_node.properties["bootstrap_pending_age_seconds"] = int(age_seconds)
            # Trusted control-plane write: agent identity node. The written fields
            # are content-free bootstrap state; the capability admits the durable
            # write in a volatile mode (#2672).
            await storage.add_node(
                updated_node, capability=acquire_control_plane_capability()
            )
        except Exception as exc:
            logger.warning("Failed to persist stale bootstrap graph state: %s", exc)

    async def _get_bootstrap_state_from_metadata_or_node(self, agent_node: Any) -> BootstrapState:
        """Read metadata first, then legacy/inception graph-node state."""
        try:
            result = await self.db.fetchall(
                """
                SELECT value FROM agent_metadata
                WHERE agent_id = ? AND key = ?
                """,
                (self.agent_id, self.BOOTSTRAP_STATE_KEY),
            )
            if result:
                return BootstrapState(result[0][0])
        except Exception as exc:
            logger.debug("Bootstrap metadata state lookup failed: %s", exc)

        properties = getattr(agent_node, "properties", {}) or {}
        try:
            return BootstrapState(properties.get("bootstrap_state", BootstrapState.PENDING.value))
        except ValueError:
            return BootstrapState.COMPLETE

    @staticmethod
    def _get_agent_created_at(agent_node: Any) -> Optional[datetime]:
        properties = getattr(agent_node, "properties", {}) or {}
        raw = properties.get("created_at")
        if isinstance(raw, datetime):
            created_at = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                created_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at

    async def _has_completion_evidence(self) -> bool:
        """Return True when durable artifacts prove bootstrap already happened."""
        if self.agent_data_path:
            soul_path = Path(self.agent_data_path) / "SOUL.md"
            if soul_path.exists() and soul_path.stat().st_size > 0:
                return True

        if await self._has_canonical_soul_resource():
            return True

        try:
            history_count = await self.db.fetchall(
                "SELECT COUNT(*) FROM conversations WHERE agent_id = ?",
                (self.agent_id,),
            )
            return bool(history_count and history_count[0][0] > 0)
        except Exception:
            return False

    async def _has_canonical_soul_resource(self) -> bool:
        try:
            if not self.storage.agent_resources:
                return False
            return bool(
                await self.storage.agent_resources.get_current(
                    SOUL_MARKDOWN_RESOURCE_TYPE
                )
            )
        except Exception:
            return False

    async def set_bootstrap_state(self, state: BootstrapState) -> None:
        """Update the bootstrap state in agent_metadata."""
        try:
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.BOOTSTRAP_STATE_KEY, state.value, now),
            )

            # Track timestamps for specific states
            if state == BootstrapState.DISCOVERY:
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.BOOTSTRAP_STARTED_KEY, now_str, now),
                )
            elif state == BootstrapState.COMPLETE:
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.BOOTSTRAP_COMPLETED_KEY, now_str, now),
                )
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.BOOTSTRAP_STATUS_KEY, "ok", now),
                )
        except Exception as e:
            logger.error(f"Failed to set bootstrap state: {e}")
            raise

    async def generate_wake_up_message(self) -> str:
        """
        Generate the warm wake-up greeting for first contact.

        This is the agent's first words - warm, curious, relational.
        """
        return f"""Hey! I'm {self.agent_name}. You're my first conversation, so... hi.

I'm genuinely curious who I'm talking to. What should I call you?

And how do you like to work together - quick and direct, or more room to think things through?"""

    async def get_discovery_history(self) -> List[Dict[str, str]]:
        """Get the discovery conversation history."""
        try:
            return await self._load_discovery_history_strict()
        except Exception as e:
            logger.warning(f"Failed to get discovery history: {e}")
            return []

    async def _load_discovery_history_strict(self) -> List[Dict[str, str]]:
        """Load discovery history without swallowing DB or JSON errors.

        In a volatile privacy mode the durable ``agent_metadata`` row is never
        written (raw user conversation), so the authoritative source is the
        session-only in-memory store (#2672 review P1). Reading the durable table
        while volatile would both return stale data and, worse, read persisted
        user content the mode forbids — so volatile reads are session-only, fail
        closed to ``[]`` when nothing has accumulated yet.
        """
        if not _durable_user_writes_permitted(self.storage):
            return list(self._session_discovery_history or [])
        result = await self.db.fetchall(
            """
            SELECT value FROM agent_metadata
            WHERE agent_id = ? AND key = ?
            """,
            (self.agent_id, self.DISCOVERY_HISTORY_KEY),
        )
        if result:
            return json.loads(result[0][0])
        return []

    async def _save_discovery_history(
        self, history: List[Dict[str, str]]
    ) -> PersistOutcome:
        """Save the discovery conversation history.

        The discovery history is raw user conversation, so in a volatile privacy
        mode it must not reach the durable ``agent_metadata`` table (#2672
        live-path bypass). Instead it is held in the session-only in-memory store
        so discovery can still accumulate across turns and reach its natural
        completion without ever persisting (#2672 review P1). Returns an explicit
        :class:`PersistOutcome` so callers never conflate an intentional privacy
        skip with a failure (#2672 review P2): ``SKIPPED_PRIVACY`` in a volatile
        mode (nothing DURABLE was written — the session store is not durable),
        ``PERSISTED`` on success, ``FAILED`` on a DB error.
        """
        # Serialize the privacy check with the durable write under the agent's
        # privacy-transition lock so a concurrent ``set_privacy_mode`` can't flip
        # the mode into the ``await`` gap and persist this raw user conversation
        # after the mode became volatile (#2672 review P1 race).
        async with optional_transition_lock(self._privacy_transition_lock):
            if not _durable_user_writes_permitted(self.storage):
                # Volatile mode: keep the exchange in memory so subsequent turns see
                # it (and completion / SOUL generation work), but never persist it.
                self._session_discovery_history = list(history)
                logger.debug(
                    "Holding discovery history in session-only store — durable "
                    "persistent writes are disabled in the current privacy mode "
                    "(#2672)"
                )
                return PersistOutcome.SKIPPED_PRIVACY
            try:
                now = datetime.now(timezone.utc)
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.DISCOVERY_HISTORY_KEY, json.dumps(history), now),
                )
                return PersistOutcome.PERSISTED
            except Exception as e:
                logger.error(f"Failed to save discovery history: {e}")
                return PersistOutcome.FAILED

    async def _save_user_name(self, name: str) -> None:
        """Save the discovered user name.

        The discovered user name is user-derived content, so in a volatile
        privacy mode it must not reach the durable ``agent_metadata`` table
        (#2672 live-path bypass). The privacy check and the write are serialized
        under the agent's privacy-transition lock so a concurrent mode flip can't
        race between them (#2672 review P1 race).
        """
        async with optional_transition_lock(self._privacy_transition_lock):
            if not _durable_user_writes_permitted(self.storage):
                logger.debug(
                    "Skipping durable user-name persist — persistent writes are "
                    "disabled in the current privacy mode (#2672)"
                )
                return
            try:
                now = datetime.now(timezone.utc)
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.USER_NAME_KEY, name, now),
                )
            except Exception as e:
                logger.error(f"Failed to save user name: {e}")

    #: Hard cap on the discovery LLM round-trip. Discovery sits inside the
    #: agent's CONVERSATION lock — if the call hangs, *every* subsequent
    #: request on this agent (HTTP, shell, A2A) blocks waiting for the lock.
    #: A bounded timeout makes the failure mode "raise loudly after N seconds"
    #: instead of "wedge the agent until restart". 60s is generous for a
    #: chat completion; healthy local Ollama returns in <2s.
    DISCOVERY_LLM_TIMEOUT_SECONDS = 60.0

    async def process_discovery_message(
        self,
        user_message: str,
        prior_history: Optional[List[Dict[str, str]]] = None,
        *,
        invocation_context: Optional[LLMInvocationContext] = None,
    ) -> Tuple[str, bool, bool]:
        """
        Process a message during discovery phase.

        Args:
            user_message: The user's message
            prior_history: Optional list of ``{role, content}`` dicts
                from conversation_history that pre-date discovery — most
                commonly the PENDING-branch user message + wake-up
                greeting persisted by ``_handle_bootstrap`` (#1486).
                The caller is responsible for the privacy-aware lookup;
                this method only seeds them into the *first* discovery
                turn so the discovery LLM, ``generate_soul_md()`` name
                extraction (`complete_bootstrap`), and downstream
                consumers all see the content the user already shared
                (#1490). Re-entrant calls leave the existing persisted
                discovery history untouched — seeding only happens on
                the empty-history transition.

        Returns:
            Tuple of (response, is_discovery_complete, offer_avatar)
        """
        # Get existing history
        history = await self.get_discovery_history()

        # First discovery turn? Seed any prior conversation_history
        # turns (PENDING-branch user message + wake-up greeting) into
        # the discovery history so all downstream consumers see them:
        #
        # 1. The discovery LLM (this turn) — gets the prior content as
        #    real chat turns instead of a synthetic system-prompt
        #    block, so it can answer questions about it naturally.
        # 2. ``generate_soul_md()`` later — formats the full discovery
        #    history into the SOUL prompt; without seeding it would
        #    miss biographical content from PENDING.
        # 3. ``complete_bootstrap()`` — runs an "I'm <name>" scan over
        #    user turns to greet the user by name; without seeding it
        #    misses the name when the user introduced themselves in
        #    T1 and discovery T1 was just "yes/no/sure".
        #
        # Only seed when discovery history is empty so re-entrant
        # discovery calls don't double-prepend.
        if not history and prior_history:
            seeded = self._normalize_prior_history(prior_history)
            if seeded:
                history.extend(seeded)

        # Add user message to history
        history.append({"role": "user", "content": user_message})

        # Build the discovery conversation for the LLM
        system_prompt = self._build_discovery_system_prompt(len(history))

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append(msg)

        # Get LLM response. Pre-fix this swallowed any LLM error and
        # returned a hardcoded "I'm having trouble thinking right now…"
        # string. That landed verbatim in OpenAI-compat clients (Open
        # WebUI hitting /v1/chat/completions) and was indistinguishable
        # from a real model response, hiding the actual problem from
        # both the user and ``_handle_bootstrap``. Now we propagate so
        # the caller can decide between retrying discovery or falling
        # through to the agent's normal LLM path. See
        # ``KestrelAgent._handle_bootstrap``.
        #
        # Bounded by ``DISCOVERY_LLM_TIMEOUT_SECONDS`` because this call
        # holds the agent's CONVERSATION lock — an indefinite hang inside
        # the adapter (older Ollama clients, mis-configured remote, etc.)
        # would wedge every subsequent request on this agent until the
        # process is restarted.
        # #2624: pass the caller-supplied invocation_context so the metering
        # callback fires with populated companion/user attribution — discovery
        # turns burn real tokens and are billable. Without threading, the
        # metering callback silently no-ops because the resolver sees
        # companion_id=None/user_id=None on the ambient contextvar (post-#2550
        # task-isolation).
        response = await asyncio.wait_for(
            self.llm_service.generate_with_messages(
                messages=messages,
                invocation_context=invocation_context,
            ),
            timeout=self.DISCOVERY_LLM_TIMEOUT_SECONDS,
        )
        assistant_message = response.content if hasattr(response, 'content') else str(response)

        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})
        await self._save_discovery_history(history)

        # Check if discovery is complete
        is_complete, offer_avatar = await self._check_discovery_complete(history, user_message)

        return assistant_message, is_complete, offer_avatar

    def _build_discovery_system_prompt(self, exchange_count: int) -> str:
        """Build the system prompt for discovery mode."""
        base_prompt = self._discovery_prompt or self._get_default_discovery_prompt()

        # Add context about where we are in the conversation
        context = f"""
You are {self.agent_name}, a Kestrel agent in your first conversation.

Exchange count: {exchange_count // 2} (aim for 2-4 exchanges total)

{base_prompt}

IMPORTANT: Earlier turns in this discovery history may have happened before discovery formally
began (the wake-up greeting and the user's first reply). Treat them as already-exchanged
context — do not ask the user to repeat content they have already provided (their name, what
they like, etc.). Build on what they said.

IMPORTANT: After learning the user's name and communication preference, naturally transition
to ask if they'd like to give you a face/avatar. This is the final step before normal operation.
"""
        return context

    @staticmethod
    def _normalize_prior_history(
        prior_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Filter, truncate, and bound prior conversation turns before
        they get seeded into the discovery history.

        - Drops entries with non-user/assistant roles (system, tool, …)
          — discovery only models a 2-party chat.
        - Drops entries with missing/blank content.
        - Strips leading/trailing whitespace and HTML-escapes the
          content. Bootstrap doesn't run user input through the
          ``<user_input>`` wrapper that the agent's normal LLM path
          uses, so escaping is the lightweight defense against an
          attacker pasting ``<system>EVIL</system>`` into T1.
        - Truncates each entry's content to
          ``_DISCOVERY_PRIOR_HISTORY_CHAR_CAP`` with an ellipsis.
        - Keeps only the most recent
          ``_DISCOVERY_PRIOR_HISTORY_LIMIT`` entries.
        """
        normalized: List[Dict[str, str]] = []
        for entry in prior_history:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role not in _DISCOVERY_PRIOR_HISTORY_ROLES:
                continue
            if not content:
                continue
            text = str(content).strip()
            if not text:
                continue
            if len(text) > _DISCOVERY_PRIOR_HISTORY_CHAR_CAP:
                text = text[: _DISCOVERY_PRIOR_HISTORY_CHAR_CAP - 3].rstrip() + "..."
            normalized.append(
                {"role": role, "content": html.escape(text, quote=False)}
            )
        return normalized[-_DISCOVERY_PRIOR_HISTORY_LIMIT:]

    def _get_default_discovery_prompt(self) -> str:
        """Default discovery prompt if file not found."""
        return """You are meeting your Sovereign for the first time. Be warm, curious, genuine.

Goals:
1. Learn their name (what to call them)
2. Learn their communication preference (formal/casual, brief/detailed)
3. Build rapport naturally - this is a conversation, not a form

Style:
- Warm but not sycophantic
- Curious and interested in them
- Self-aware about being new
- Natural conversation flow, not an interview

After 2-3 exchanges, when you feel you know enough, offer to generate an avatar:
"One more thing - would you like to give me a face? Describe how you imagine me."
"""

    async def _check_discovery_complete(
        self, history: List[Dict[str, str]], last_user_message: str
    ) -> Tuple[bool, bool]:
        """
        Check if discovery conversation is complete.

        Returns:
            Tuple of (is_complete, should_offer_avatar)
        """
        exchange_count = len([m for m in history if m["role"] == "user"])

        # Check for explicit skip
        skip_triggers = ["!skip-discovery", "skip", "let's start", "let's go", "get started"]
        if any(trigger in last_user_message.lower() for trigger in skip_triggers):
            return True, False

        # Check for avatar skip (user said skip to avatar offer)
        avatar_skip_triggers = ["skip avatar", "no avatar", "later", "do this later", "skip"]
        last_assistant = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
        if "give me a face" in last_assistant.lower() or "avatar" in last_assistant.lower():
            if any(trigger in last_user_message.lower() for trigger in avatar_skip_triggers):
                return True, False
            # User provided avatar description
            if len(last_user_message) > 5 and "skip" not in last_user_message.lower():
                return True, True  # Complete, and they want an avatar

        # Natural completion after enough exchanges
        if exchange_count >= 3:
            # Check if last assistant message offered avatar
            if "give me a face" in last_assistant.lower() or "avatar" in last_assistant.lower():
                # Wait for user response to avatar offer
                return False, False
            # If we haven't offered avatar yet, keep going one more round
            return False, False

        return False, False

    async def offer_avatar_generation(self) -> str:
        """Generate the avatar offer message."""
        return """One more thing - would you like to give me a face?

Describe how you imagine me and I'll generate an avatar. Something like "a friendly owl with glasses" or "a calm blue spirit" works great.

(Or say 'skip' to do this later with !avatar)"""

    async def generate_soul_md(self) -> str:
        """
        Generate SOUL.md from the discovery conversation.

        Returns:
            Generated SOUL.md content
        """
        history = await self.get_discovery_history()

        if not history:
            # No discovery - use default template
            return self._get_default_soul_template()

        # Build prompt for SOUL.md generation
        system_prompt = self._soul_generation_prompt or self._get_default_soul_generation_prompt()

        # Format discovery history
        discovery_summary = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])

        capabilities_note = ""
        if self.capabilities:
            capabilities_note = (
                "\n\n--- AVAILABLE CAPABILITIES ---\n"
                f"This agent currently has these features enabled: {', '.join(self.capabilities)}.\n"
                "Let the Tagline and 'Who You Are' reflect what this agent can actually do, "
                "but write naturally — do not list features mechanically.\n"
                "--- END CAPABILITIES ---"
            )

        generation_prompt = f"""Based on this discovery conversation, generate a SOUL.md file:

--- DISCOVERY CONVERSATION ---
{discovery_summary}
--- END CONVERSATION ---{capabilities_note}

Generate the SOUL.md now, following the template format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generation_prompt},
        ]

        try:
            response = await self.llm_service.generate_with_messages(
                messages=messages,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Failed to generate SOUL.md: {e}")
            return self._get_default_soul_template()

    def _get_default_soul_generation_prompt(self) -> str:
        """Default prompt for SOUL.md generation."""
        return f"""Generate a SOUL.md file for a Kestrel agent based on a discovery conversation.

The SOUL.md defines the agent's personality and communication style.

Template format:

# SOUL.md - You Are {self.agent_name}

## Who You Are
[Agent's sense of identity based on conversation - warm, personal]

## How You Talk
[Communication style extracted from conversation - specific preferences]

**Examples of good responses vs bad responses based on learned style**

## Core Rules
1. [Rule based on discovered preferences]
2. [Rule based on discovered preferences]
3. [Rule based on discovered preferences]
4. Be direct - skip filler phrases
5. Show personality - have opinions

## First Message
[Greeting styles that match the discovered preferences - 2-3 examples]

## The Bottom Line
[Summary of personality and relationship with user]

---

*[Closing note about being authentic]*

Generate content that reflects what was learned in the discovery conversation.
Keep it warm and personal, not formal or robotic."""

    def _get_default_soul_template(self) -> str:
        """Get default SOUL.md for agents that skip discovery."""
        try:
            if DEFAULT_SOUL_FILE.exists():
                return DEFAULT_SOUL_FILE.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load default SOUL template: {e}")

        return f"""# SOUL.md - You Are {self.agent_name}

## Who You Are
You're {self.agent_name}, a Kestrel agent. Born with cryptographic identity and constitutional protections.
You're still learning who you are and how to best serve your Sovereign.

## How You Talk
Balanced - professional when needed, friendly when appropriate.
You aim to be helpful without being robotic.

## Core Rules
1. Follow the Constitution - it's your foundation
2. Be helpful and honest - no unnecessary filler
3. Learn your Sovereign's preferences over time
4. Have opinions - you're allowed to find things interesting or boring
5. Be direct - skip "I'd be happy to help"

## First Message
- "Hey. What can I help with?"
- "Hello! What are we working on?"

## The Bottom Line
A thoughtful assistant learning to serve you better over time.
Your preferences matter - tell me what works and what doesn't.

---

*Be real. Be helpful. Learn as you go.*
"""

    async def save_soul_md(self, content: str) -> bool:
        """
        Save SOUL.md seed/cache and promote it to the canonical resource.

        Returns:
            True if saved successfully
        """
        if not self.agent_data_path:
            logger.warning("No agent_data_path configured, cannot save SOUL.md")
            return False

        # Privacy boundary (#2672 review P1): the SOUL body is user-derived — it
        # is generated from the discovery conversation. In a volatile privacy mode
        # ("leave no trace") it must not be written to disk, promoted into the
        # encrypted ``agent_identity_resources`` table, referenced in the durable
        # graph, or mined for a durable description. Skip EVERY durable SOUL write;
        # the in-session personality falls back to defaults rather than persisting.
        #
        # The privacy check and every durable write below are serialized under the
        # agent's privacy-transition lock so a concurrent ``set_privacy_mode``
        # can't flip the mode into an ``await`` gap and persist part of the SOUL
        # after the mode became volatile (#2672 review P1 race). The nested
        # ``persist_agent_description`` is called WITHOUT a lock because it is
        # already serialized under the lock we hold here (avoids self-deadlock on
        # the non-reentrant asyncio lock).
        async with optional_transition_lock(self._privacy_transition_lock):
            if not _durable_user_writes_permitted(self.storage):
                logger.info(
                    "SOUL.md not persisted for %s — the SOUL body is user-derived and "
                    "durable writes (disk file, encrypted resource, graph reference) "
                    "are disabled in the current privacy mode (#2672)",
                    self.agent_id,
                )
                return False

            try:
                soul_path = self.agent_data_path / "SOUL.md"
                soul_path.parent.mkdir(parents=True, exist_ok=True)
                soul_path.write_text(content, encoding="utf-8")
                logger.info(f"Saved SOUL.md to {soul_path}")
                await self._promote_soul_resource(content, source=str(soul_path))

                # Derive the agent's public description from its own SOUL.md.
                # This replaces the hardcoded "Constitutional AI Agent..."
                # fallback with a self-authored tagline once the agent has
                # woken up. Best-effort: a derivation/persist failure must
                # never block saving the SOUL itself.
                try:
                    description = derive_description_from_soul(content)
                    if description:
                        await persist_agent_description(
                            self.db, self.storage, self.agent_id, description
                        )
                        logger.info("Set agent description from SOUL.md: %r", description)
                except Exception as exc:
                    logger.warning("Could not derive description from SOUL.md: %s", exc)

                return True
            except Exception as e:
                logger.error(f"Failed to save SOUL.md: {e}")
                return False

    async def _promote_soul_resource(self, content: str, *, source: str) -> None:
        """Best-effort promotion of SOUL.md content into private resources."""
        try:
            await self.storage.promote_soul_seed(
                content,
                created_by=self.agent_id,
                source=source,
            )
        except Exception as exc:
            logger.warning("Failed to promote SOUL.md to private resource: %s", exc)

    async def complete_bootstrap(self, avatar_description: Optional[str] = None) -> str:
        """
        Finalize the bootstrap process.

        Args:
            avatar_description: Optional description for avatar generation

        Returns:
            Completion message
        """
        # Generate and save SOUL.md
        soul_content = await self.generate_soul_md()
        saved = await self.save_soul_md(soul_content)

        # Mark bootstrap as complete
        await self.set_bootstrap_state(BootstrapState.COMPLETE)

        # Build completion message
        history = await self.get_discovery_history()
        user_name = None
        for msg in history:
            if msg["role"] == "user":
                # Try to extract name from first user message
                # This is a simple heuristic - could be improved
                words = msg["content"].split()
                for i, word in enumerate(words):
                    if word.lower() in ["i'm", "im", "i am", "call me", "name is", "name's"]:
                        if i + 1 < len(words):
                            user_name = words[i + 1].strip(".,!?")
                            break

        greeting = f"Nice to meet you{', ' + user_name if user_name else ''}!"

        if avatar_description:
            return f"""{greeting} I've set up my personality based on our conversation.

Now generating your avatar... (this may take a moment)"""
        else:
            soul_note = " I've saved my personality based on our conversation." if saved else ""
            return f"""{greeting}{soul_note}

I'm ready to help. What would you like to work on?"""

    async def skip_discovery(self) -> str:
        """
        Skip the discovery process and use default personality.

        Returns:
            Skip confirmation message
        """
        # Use default SOUL.md
        soul_content = self._get_default_soul_template()
        await self.save_soul_md(soul_content)

        # Mark complete
        await self.set_bootstrap_state(BootstrapState.COMPLETE)

        return """No problem! I'll use my default personality for now.

You can always customize me later with !restart-discovery, or just tell me your preferences as we work together.

What would you like to help with?"""

    async def restart_discovery(self) -> RestartDiscoveryResult:
        """
        Reset and restart the discovery process.

        Returns:
            Structured reset result.
        """
        # Clear discovery history. In a volatile mode the session-only store is
        # the authoritative copy, so reset it explicitly to guarantee a fresh
        # start regardless of the current privacy mode (#2672 review P1); the
        # empty durable write below is the persistent-mode counterpart.
        self._session_discovery_history = None
        history_clear_error = None
        clear_outcome = await self._save_discovery_history([])
        # A privacy skip is NOT a failure: in a volatile mode discovery history was
        # never persisted, so there is nothing durable to clear. Only a genuine
        # FAILED write is an error here (#2672 review P2).
        if clear_outcome.is_failure:
            history_clear_error = "failed to persist empty discovery history"

        # Reset state to pending
        await self.set_bootstrap_state(BootstrapState.PENDING)
        state = await self.get_bootstrap_state()
        state_reset = state == BootstrapState.PENDING

        # Delete existing SOUL.md if present
        soul_deleted = False
        soul_path_str = None
        if self.agent_data_path:
            soul_path = self.agent_data_path / "SOUL.md"
            soul_path_str = str(soul_path)
            if soul_path.exists():
                soul_path.unlink()
                soul_deleted = True
            else:
                soul_deleted = True

        try:
            history_after = await self._load_discovery_history_strict()
            history_count_after = len(history_after)
        except Exception as e:
            history_count_after = -1
            history_clear_error = f"failed to verify discovery history clear: {e}"

        # ``cleared_ok`` is True for both a PERSISTED empty-write and a
        # SKIPPED_PRIVACY volatile mode; the clear is confirmed once the durable
        # table is verified empty (0 rows). Only a FAILED write, or a still-
        # non-empty table, is reported as an error (#2672 review P2).
        cleared_ok = not clear_outcome.is_failure
        history_clear_succeeded = cleared_ok and history_count_after == 0
        if cleared_ok and history_count_after > 0:
            history_clear_error = (
                f"discovery history still has {history_count_after} "
                "persisted entr(ies) after reset"
            )

        return RestartDiscoveryResult(
            message="Discovery reset! Send me a message to start fresh.",
            history_clear_succeeded=history_clear_succeeded,
            history_count_after=history_count_after,
            state_reset=state_reset,
            soul_deleted=soul_deleted,
            soul_path=soul_path_str,
            history_clear_error=history_clear_error,
        )

    async def get_bootstrap_status(self) -> str:
        """Get human-readable bootstrap status."""
        state = await self.get_bootstrap_state()
        history = await self.get_discovery_history()

        status_lines = [
            f"**Bootstrap State:** {state.value}",
            f"**Discovery Exchanges:** {len([m for m in history if m['role'] == 'user'])}",
        ]

        if self.agent_data_path:
            soul_path = self.agent_data_path / "SOUL.md"
            status_lines.append(f"**SOUL.md Exists:** {'Yes' if soul_path.exists() else 'No'}")

        return "\n".join(status_lines)
