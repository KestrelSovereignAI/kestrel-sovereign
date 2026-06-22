import logging
import json
import os
import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.security.encryption import DecryptionError
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.config import TRUSTED_AGENTS_DIR
from typing import Optional, Dict, List, Any, Union
import re
from pathlib import Path
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config
from kestrel_sovereign.extensions.app_extension import AppExtension
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.features import discover_features, get_feature_by_name
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.stores import (
    SQLiteTaskStore, SQLiteSessionService, SQLiteObservabilityStore,
    SQLiteFeedbackStore, SQLiteMemoryService
)
# PostgreSQL stores imported conditionally when pg_pool is available
from kestrel_sovereign.agent import ContextBuilder, ContextManager
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.agent.backup import BackupMixin
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin, ContextStats
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.hooks import HooksManager
from kestrel_sdk.hooks.base import HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.bootstrap import BootstrapService, BootstrapState
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
    append_security_addendum,
)
from kestrel_sovereign.telemetry import optional_span

# Optional ollama import (not available in remote-only containers)
try:
    import ollama
except ImportError:
    ollama = None

# Prompt file locations
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.md"
USER_PROMPT_FILE = PROMPTS_DIR / "user_prompt.md"

# Maximum tool call iterations (configurable via environment variable)
# Increased to 50 for long-running tasks like code analysis and multi-step operations
MAX_TOOL_ITERATIONS = int(os.environ.get("KESTREL_MAX_TOOL_ITERATIONS", "50"))


@dataclass
class PrivacyTransitionResult:
    """Structured result for privacy-mode transitions across API and command paths."""

    message: str
    allows_cloud_llm: bool
    model_switched: Optional[dict] = None
    voice_switched: Optional[dict] = None
    biometric_warning: Optional[str] = None


async def _add_sovereign_ipfs_target_if_active(
    sync_service,
    *,
    agent_id: str,
    state_dir: Path,
    sovereign_url: Optional[str],
) -> bool:
    """Add the sovereign-operated IPFS target only when the node is reachable."""
    from kestrel_sovereign.storage.sync.health import check_sovereign_ipfs_health
    from kestrel_sovereign.storage.sync.targets import SovereignIPFSTarget, TrustTier

    health = await check_sovereign_ipfs_health(api_url=sovereign_url)
    if health.status != "active":
        logging.info(
            "Sovereign-operated IPFS backup tier %s: %s",
            health.status,
            health.message,
        )
        return False

    api_url = str(health.details.get("api_url") or sovereign_url).rstrip("/")
    return sync_service.add_remote_target(
        f"ipfs://{agent_id}",
        TrustTier.SOVEREIGN,
        lambda: SovereignIPFSTarget(
            api_url=api_url,
            agent_id=agent_id,
            state_dir=state_dir,
        ),
    )

# Maximum chars for a single tool result before truncation
MAX_TOOL_RESULT_CHARS = int(os.environ.get("KESTREL_MAX_TOOL_RESULT_CHARS", "8000"))

# Reserve this fraction of context for the LLM response + next tool call
CONTEXT_RESERVE_FRACTION = 0.2

# Diminishing returns detection — stop loops producing negligible output
# Minimum output tokens per reasoning-only iteration (no tool calls)
KESTREL_DIMINISHING_THRESHOLD = int(os.environ.get("KESTREL_DIMINISHING_THRESHOLD", "500"))
# Consecutive low-delta reasoning-only iterations before stopping
KESTREL_MAX_LOW_DELTA = int(os.environ.get("KESTREL_MAX_LOW_DELTA", "5"))
# Stop if iteration count exceeds this percentage of max_iterations budget
KESTREL_BUDGET_STOP_PCT = int(os.environ.get("KESTREL_BUDGET_STOP_PCT", "90"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_prompt_file(filepath: Path, fallback: str = "") -> str:
    """Load a prompt from file with fallback to embedded default."""
    try:
        if filepath.exists():
            return filepath.read_text(encoding="utf-8").strip()
        logging.warning(f"Prompt file not found: {filepath}, using fallback")
        return fallback
    except (OSError, UnicodeDecodeError) as e:
        logging.error(f"Error loading prompt file {filepath}: {e}")
        return fallback
    except Exception as e:
        logging.error(f"Error loading prompt file {filepath}: {e}", exc_info=True)
        return fallback


def _resolve_sync_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve whether lifecycle SyncService side effects are enabled."""
    if explicit is not None:
        return explicit

    env_val = os.environ.get("KESTREL_SYNC_ENABLED")
    if env_val is None:
        return True

    normalized = env_val.strip().lower()
    if normalized in ("0", "false", "no", "off"):
        return False
    if normalized in ("1", "true", "yes", "on"):
        return True

    logging.warning(
        "Invalid KESTREL_SYNC_ENABLED value %r; defaulting to enabled",
        env_val,
    )
    return True


class KestrelAgent(
    ConstitutionMixin,
    StreamingMixin,
    BackupMixin,
    SleepMixin,
    OrchestratorEngineMixin,
    ToolRegistryMixin,
    ModelPreferenceMixin,
    EventManagerMixin,
    RequestLifecycleMixin,
    TurnLifecycleMixin,
):
    """
    The Kestrel Agent orchestrates memory, reasoning, and actions, bound by the Kestrel Constitution.
    """

    def __init__(
        self,
        did: str,
        storage_path: Optional[str] = None,
        llm_service: Optional[LLMService] = None,
        privacy_mode: PrivacyMode = PrivacyMode.NORMAL,
        pg_pool=None,
        *,
        database_url: Optional[str] = None,
        db_backend: Optional[str] = None,
        allowed_features: Optional[set] = None,
        sync_enabled: Optional[bool] = None,
        payer_policy=None,
        host_db=None,
    ):
        """
        Initializes the agent with memory and reasoning capabilities.
        # Bedrock for Sovereign Companions: Human-led, no-loss persistence.

        Args:
            did: The agent's own DID (e.g., 'did:pkh:...'), used for self-discovery.
            storage_path: Path to the database file for SQLite storage.
            llm_service: The service that provides access to foundational models.
            privacy_mode: Privacy mode for this session.
            pg_pool: Optional PostgreSQL pool for feedback feature.
            database_url: PostgreSQL connection string (for postgres backend).
            db_backend: Database backend type ('sqlite' or 'postgres').
                       Defaults to KESTREL_DB_BACKEND env var or 'sqlite'.
            allowed_features: Optional set of feature class names to load.
                       If None, all discovered features are loaded.
                       Mandatory features always load regardless.
            sync_enabled: Enables lifecycle SyncService snapshots. Defaults to
                       KESTREL_SYNC_ENABLED env var, or enabled when unset.
            payer_policy: Optional ``kestrel_sdk.payer_policy.PayerPolicy`` to
                       use for credential resolution at init. When provided, it
                       overrides ``load_policy_from_toml()`` — lets a multi-tenant
                       host embed agents with a programmatic per-agent policy
                       instead of an on-disk ``kestrel.toml``.
            host_db: Optional host-level ``AsyncDatabase`` holding the operator's
                       HostKeyStorage masters. When provided, it overrides the
                       on-disk SQLite ``host.db`` lookup (``open_host_db``) — lets
                       a host on Postgres supply the host db directly (e.g.
                       ``AsyncDatabase.from_pool(pg_pool)``). The caller owns its
                       lifecycle; the agent does not close it.
        """
        self.did = did
        self._privacy_mode = privacy_mode
        self.storage_path = storage_path
        self._allowed_features = allowed_features
        self._sync_enabled = _resolve_sync_enabled(sync_enabled)
        # Optional injected payer-policy + host db for multi-tenant embedding
        # (#1649). When set, they override the standalone kestrel.toml /
        # on-disk host.db lookups during credential resolution at init.
        self._injected_payer_policy = payer_policy
        self._injected_host_db = host_db

        # Per-agent constitution overlay (#898). When ``<agent_dir>/CONSTITUTION.md``
        # exists, its text becomes ``self.constitution_text`` so feature-side
        # grant lookups (e.g. ``ComputerUseFeature._granted_capabilities``)
        # see this agent's Amendment IX checkboxes instead of falling
        # through to the package constitution. Books I-II and the rest of
        # the framework continue to come from the package — only Amendment
        # IX grants are read from the overlay because that's what the
        # parser scopes to. Absent file → ``constitution_text`` stays None
        # and the package fallback is used (existing behavior).
        self.constitution_text: Optional[str] = None
        # sha256 of the overlay bytes as loaded, and whether that hash matches
        # the anchor stored in the agent's identity node. Until verified against
        # the anchor (in initialize()/audit), the overlay is treated as
        # UNTRUSTED — its Amendment IX capability grants are NOT honored (#1722).
        # This closes the self-grant: writing a CONSTITUTION.md next to the agent
        # DB no longer grants host shell, because an unanchored overlay's grants
        # are ignored and the integrity audit fails closed on it.
        self._constitution_overlay_path: Optional[Path] = None
        self._constitution_overlay_sha: Optional[str] = None
        self.constitution_overlay_verified: bool = False
        if storage_path:
            overlay = Path(storage_path).parent / "CONSTITUTION.md"
            self._constitution_overlay_path = overlay
            if overlay.exists():
                try:
                    overlay_bytes = overlay.read_bytes()
                    self.constitution_text = overlay_bytes.decode("utf-8")
                    self._constitution_overlay_sha = hashlib.sha256(
                        overlay_bytes
                    ).hexdigest()
                    logging.info(
                        "Loaded per-agent constitution overlay from %s "
                        "(sha256=%s, pending anchor verification)",
                        overlay, self._constitution_overlay_sha[:16],
                    )
                except OSError as exc:
                    logging.warning(
                        "Failed to read per-agent constitution overlay %s: %s",
                        overlay, exc,
                    )

        # Per-agent ``[privacy] computer_access`` opt-in (#956). The privacy
        # presets ship with ``computer_access=False`` by design — the design
        # comment in ``privacy.py`` says it "must be opted into explicitly
        # by setting the flag after preset construction." This is that
        # explicit path: read ``[privacy] computer_access`` from the agent's
        # ``kestrel.toml`` and apply it to the privacy_agent's PrivacyConfig
        # when it gets constructed in ``initialize()``. Default stays False.
        self._privacy_computer_access: bool = False
        if storage_path:
            agent_toml = Path(storage_path).parent / "kestrel.toml"
            if agent_toml.exists():
                try:
                    try:
                        import tomllib  # type: ignore[import-not-found]
                    except ImportError:
                        import tomli as tomllib  # type: ignore[import-not-found]
                    with open(agent_toml, "rb") as f:
                        toml_data = tomllib.load(f)
                    self._privacy_computer_access = bool(
                        toml_data.get("privacy", {}).get("computer_access", False)
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to read [privacy] from %s: %s",
                        agent_toml, exc,
                    )

        # Hybrid-aware identity load (Quantum Hardening epic, Wave 3 follow-up).
        # Reads the legacy ECDSA key, the new hybrid keys (Ed25519 + ML-DSA-65),
        # and the succession statement if present. ``self.identity`` is the
        # ``AgentIdentity`` bundle; ``self.identity.is_hybrid`` is True for
        # post-ceremony agents.
        #
        # ``self._private_key`` is set to the legacy ECDSA private key for
        # backward compatibility with code that grabs it via getattr (most
        # notably ``multi_agent.agent_manager.spawn_agent`` — pre-ceremony agents
        # were silently broken there because nothing was setting this).
        #
        # Identity loading is best-effort: during inception the keys may not
        # be on disk yet, so failure logs a warning rather than killing
        # construction. Callers needing the identity check ``self.identity
        # is not None`` and fall back to legacy paths if it is.
        self.identity = None
        self._private_key = None
        if self.storage_path and self.did:
            legacy_key_id = self._derive_legacy_key_id(self.did)
            storage_dir = Path(self.storage_path).parent
            if legacy_key_id and (storage_dir / f"{legacy_key_id}.json").exists():
                try:
                    from kestrel_sovereign.identity.runtime_identity import (
                        load_agent_identity,
                    )
                    self.identity = load_agent_identity(
                        legacy_key_id, storage_dir=storage_dir,
                    )
                    self._private_key = self.identity.legacy_keypair.private_key
                    if self.identity.is_hybrid:
                        logging.info(
                            "Agent identity loaded as HYBRID: legacy=%s -> new=%s",
                            self.identity.legacy_did, self.identity.new_did,
                        )
                    else:
                        logging.info(
                            "Agent identity loaded as legacy-only: %s",
                            self.identity.legacy_did,
                        )
                except Exception as exc:
                    logging.warning(
                        "Could not load agent identity from %s: %s. "
                        "Agent will run with self.identity=None; signing call "
                        "sites will fall back to their existing legacy paths.",
                        storage_dir, exc,
                    )

        # Determine database backend
        self._db_backend = db_backend or os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        self._database_url = database_url or os.environ.get("KESTREL_DATABASE_URL")

        # Storage will be initialized asynchronously
        self._raw_storage = None
        self.storage = None

        self.llm_service = llm_service or LLMService()
        # Claim this LLMService for this agent. If an externally-provided
        # llm_service was already claimed by another agent (the adversarial-
        # test sharing pattern), this raises LLMServiceAlreadyAttachedError
        # so the cross-agent state-leak invariant is enforced at construction
        # time. See LLMService.attach_to_agent for the rationale.
        #
        # Guarded by hasattr because some tests inject lightweight LLM fakes
        # (MockLLMService et al.) that only implement the generate/model
        # surface. Real LLMService instances always have attach_to_agent;
        # the invariant is enforced for them and silently waived for fakes.
        if hasattr(self.llm_service, "attach_to_agent"):
            self.llm_service.attach_to_agent(did)
        # #1563: give every stateful adapter (currently just
        # CodexAdapter) a reference to this agent so the failure-
        # result rewrite can cross-reference the SecurityFeature's
        # audit log when classifying a tool failure. Without this,
        # the audit slot defaults to empty and audit-backed
        # USER_DENIED gets misclassified as SANDBOX_BLOCKED from
        # the raw "rejected by user" pattern alone. Best-effort:
        # adapters that don't expose ``attach_agent_for_audit``
        # (legacy / external) are silently skipped.
        for provider in getattr(self.llm_service, "providers", []):
            adapter = provider.get("adapter")
            if adapter is not None and hasattr(adapter, "attach_agent_for_audit"):
                try:
                    adapter.attach_agent_for_audit(self)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "attach_agent_for_audit failed on %s: %s",
                        type(adapter).__name__, exc,
                    )
        self.pg_pool = pg_pool
        # Note: agent_id is a @property that returns self.did (see below).
        # Do NOT set self.agent_id = ... here; it would shadow the property.
        self.privacy_agent = None  # Will be initialized after storage
        self.lighthouse_provider = None  # Will be initialized after storage if API key available
        self.wallet = None  # Set by WalletFeature.initialize()
        self.sleep_hooks = []  # *SleepHook instances; features append in post_all_features_loaded()
        # Bootstrap service is constructed in initialize(); default it here so
        # any code path that runs before/without full initialization (e.g. a
        # COGNITION signal dispatch reaching process_input's bootstrap check)
        # sees None instead of raising AttributeError (#1632).
        self.bootstrap_service: Optional[BootstrapService] = None
        # Context manager is likewise constructed in initialize(). Default it
        # here so a COGNITION signal dispatch (e.g. the restart.completed wake)
        # reaching process_input before initialize() finishes sees None and
        # defers for retry, instead of crashing the turn with an opaque
        # AttributeError on a half-built agent. Same race class as #1632; the
        # restart wake is the path that surfaced it (#1796/#1797).
        self.context_manager: Optional[ContextManager] = None
        # The session id of the in-flight turn, set under the turn lock by
        # process_input / the streaming turn and cleared on turn exit. Tools
        # that must scope to the active conversation read it (read_attachment;
        # request_restart's origin-session capture, #1809). None = no turn.
        self._active_session_id: Optional[str] = None

        # TaskManager for A2A unified routing
        self.task_manager: Optional[TaskManager] = None

        # Features will be initialized after storage
        self.features: Dict[str, Feature] = {}

        # Hooks manager for security and middleware
        self.hooks_manager = HooksManager()

        # Cached features prompt (built once at session start)
        self._cached_features_prompt: str = ""

        # Event listeners for SSE notifications
        self._event_listeners: List[Any] = []

        # Events emitted while no SSE listener is connected are buffered
        # here and replayed to the first listener that connects. Covers the
        # host-startup gap: feature.initialize() can emit_event (e.g. the
        # restart `completed` status — #1551) before the browser reconnects
        # its notifications stream. Bounded in EventManagerMixin.
        self._pending_events: List[Any] = []

        # Pending task completion notifications (for background tasks)
        self._pending_task_notifications: List[str] = []
        self._background_tasks: set[asyncio.Task] = set()

        # Cancellation tracking for stop button functionality
        self._current_request_id: Optional[str] = None
        self._active_request_ids: set[str] = set()
        # Monotonic registration time per active request id so the
        # restart coordinator can age out stale markers (#1558).
        self._active_request_started_at: dict[str, float] = {}
        self._cancelled_requests: set = set()
        self._privacy_transition_lock = asyncio.Lock()

        # Shared lock manager for the dispatcher (Phase 1) AND the turn
        # lifecycle (Phase 2). CONVERSATION is acquired by `_turn_lifecycle`
        # in `process_input`/`process_input_streaming` — registered signal
        # sources are forbidden from declaring it (registry enforces).
        self._lock_manager = OrderedLockManager()

        # Session state
        self._session_briefed = False
        self._safe_mode = False

        # Dynamic tool loading: explored features get direct tool access
        self._explored_features: dict = {}  # ordered dict (insertion order) for LRU eviction
        self._direct_tools: dict = {}
        self._direct_tool_defs: list = []
        self._tool_to_feature: dict = {}  # tool_name -> feature tool_name
        # #1580 (D): pinned features are exempt from LRU eviction.
        # Populated by `_promote_startup_feature_tools` for every
        # feature whose `promote_tools_on_startup = True` (Peers /
        # Tasks / Spawn today, plus #1578 / B's Save and Strategy
        # additions). Without a pin tier, a long session that
        # explores many features could silently evict operationally
        # critical tools (get_peer_task_result, save_item, etc.).
        self._pinned_features: set = set()

        # Initialize constitution audit tracking
        self._init_constitution_audit_tracking()

    @staticmethod
    def _derive_legacy_key_id(did: str) -> Optional[str]:
        """Derive ``kestrel_<eth_address>`` from a ``did:pkh:eip155:1:0x…`` DID.

        Returns ``None`` for DID methods we can't map to a legacy key
        file (e.g. did:web on a fresh-minted hybrid agent that has no
        legacy material). Callers handle the None case.
        """
        if did.startswith("did:pkh:eip155:1:"):
            return f"kestrel_{did[len('did:pkh:eip155:1:'):]}"
        return None

    @property
    def is_hybrid(self) -> bool:
        """True iff a succession statement was loaded at construction."""
        return self.identity is not None and self.identity.is_hybrid

    @property
    def signing_did(self) -> str:
        """The DID the agent should sign new artifacts AS.

        - Hybrid agent: the new ``did:web`` URI from the succession.
        - Legacy or pre-inception agent: the constructor's ``did`` arg.

        Existing code that uses ``agent.did`` keeps working — that
        attribute is unchanged. Code that wants the post-rotation
        identity reads ``agent.signing_did``.
        """
        if self.is_hybrid:
            return self.identity.new_did
        return self.did

    @property
    def agent_id(self) -> str:
        """Derived identity alias — always returns self.did.

        DID is the canonical identity for every Kestrel agent.  ``agent_id``
        exists as a convenience alias so that storage layers and feature
        packages that accept an ``agent_id`` parameter receive the DID
        without callers having to translate.

        See: https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/500
        """
        return self.did

    def get_feature(self, name: str):
        """Look up a registered feature by class name, tool_name, or a
        case-insensitive shorthand.

        Features are stored in ``self.features`` keyed by
        ``Feature.name`` (which is ``__class__.__name__``), so
        ``"SecurityFeature"`` is the canonical key. Six call sites
        historically passed the lowercase shorthand (``"security"``)
        and missed.

        Resolution, all case-insensitive:

        1. Exact dict lookup on ``name``.
        2. Class-name match — also accepts the ``"Feature"`` suffix
           being absent (so ``"security"`` → ``SecurityFeature``).
        3. ``tool_name`` match — same suffix tolerance for the
           auto-derived ``"_feature"`` (so ``"security"`` →
           ``security_feature``).

        ``Feature.tool_name`` is a property, so accessing it on a
        feature whose subclass overrides it as a non-property still
        works.
        """
        if not name:
            return None
        feature = self.features.get(name)
        if feature is not None:
            return feature

        target = name.lower()
        target_with_suffix = (
            target if target.endswith("feature") else target + "feature"
        )

        for feat in self.features.values():
            class_name = feat.name.lower() if getattr(feat, "name", None) else ""
            if class_name in (target, target_with_suffix):
                return feat
            try:
                tool_name = (getattr(feat, "tool_name", "") or "").lower()
            except Exception:
                tool_name = ""
            target_tool = (
                target if target.endswith("_feature") else target + "_feature"
            )
            if tool_name and tool_name in (target, target_tool):
                return feat
        return None

    @property
    def mcp_agent(self):
        """Lazy lookup for MCPAgent feature."""
        return self.features.get("MCPAgent")

    @property
    def model_agent(self):
        """Lazy lookup for ModelAgent feature."""
        return self.features.get("ModelAgent")

    @property
    def is_demo(self) -> bool:
        """True when this agent was inceptioned with ``is_demo=True`` (#766).

        Demo-scoped agents bypass the destructive-op guardrails the server
        applies to live agents — the multi_agent operator opted in to letting
        demos clear history, toggle permissions, and switch privacy modes
        without an explicit authorization header.
        """
        return getattr(self, "_is_demo", False)

    @property
    def is_test_instance(self) -> bool:
        """True when this agent was inceptioned with ``is_test_instance=True``."""
        return getattr(self, "_is_test_instance", False)

    def _resolve_payer_policy(self):
        """The PayerPolicy for credential resolution at init.

        Prefers an injected policy (multi-tenant embedding, #1649); otherwise
        loads from the standalone ``kestrel.toml`` ``[payments]`` section.
        """
        if self._injected_payer_policy is not None:
            return self._injected_payer_policy
        from kestrel_sovereign.services.payer_resolver import load_policy_from_toml
        return load_policy_from_toml()

    async def _resolve_host_db(self):
        """The host-level AsyncDatabase holding HostKeyStorage masters.

        Prefers an injected host db (multi-tenant embedding, #1649); otherwise
        opens the on-disk SQLite ``host.db`` next to this agent's storage.
        Returns None when neither is available (resolver then falls back to the
        agent's own db, which has no host_service_keys rows).
        """
        if self._injected_host_db is not None:
            return self._injected_host_db
        from kestrel_sovereign.services.payer_resolver import open_host_db
        return await open_host_db(storage_path=self.storage_path)

    async def initialize(self) -> None:
        """Async initialization of storage and features."""
        if self._raw_storage is None:
            # Cold-start restore from Lighthouse if DB doesn't exist (ephemeral environments)
            if (
                os.environ.get("LIGHTHOUSE_API_KEY")
                and self._db_backend.lower() != "postgres"
                and self.storage_path
                and not Path(self.storage_path).exists()
            ):
                try:
                    from kestrel_sovereign.storage.sync.targets import LighthouseTarget

                    agent_id = self.did or "default"
                    state_dir = Path(self.storage_path).parent
                    target = LighthouseTarget(
                        api_key=os.environ["LIGHTHOUSE_API_KEY"],
                        agent_id=agent_id,
                        state_dir=state_dir,
                    )
                    result = await target.restore_snapshot(Path(self.storage_path))
                    if result and result.success:
                        logging.info(
                            f"Cold-start: restored {result.bytes_synced} bytes "
                            f"from Lighthouse (CID: {result.metadata.get('cid', 'unknown')})"
                        )
                    else:
                        logging.info("Cold-start: no Lighthouse snapshot found, starting fresh")
                except (ImportError, AttributeError, TypeError, ConnectionError) as e:
                    logging.warning(f"Cold-start restore from Lighthouse failed: {e}")
                except Exception as e:
                    logging.warning(f"Cold-start restore from Lighthouse failed: {e}", exc_info=True)

            # Initialize async storage based on backend type
            if self._db_backend.lower() == "postgres" and (self.pg_pool or self._database_url):
                # PostgreSQL backend - reuse shared pool if available
                if self.pg_pool:
                    from kestrel_sovereign.storage.db.postgres import PostgresBackend
                    pg_backend = PostgresBackend.from_pool(self.pg_pool)
                    self._raw_storage = AsyncStorage(
                        backend=pg_backend,
                        agent_id=self.did,
                        llm_service=self.llm_service,
                    )
                    logging.info(f"Using shared PostgreSQL pool for Kestrel storage (agent: {self.did})")
                else:
                    self._raw_storage = AsyncStorage(
                        backend="postgres",
                        dsn=self._database_url,
                        agent_id=self.did,
                        llm_service=self.llm_service,
                    )
                    logging.info(f"Using PostgreSQL backend for Kestrel storage (agent: {self.did})")
            else:
                # SQLite backend (default) - agent_id optional since each agent has own DB
                self._raw_storage = AsyncStorage(
                    self.storage_path,
                    agent_id=self.did,
                    llm_service=self.llm_service,
                )
                logging.info(f"Using SQLite backend for Kestrel storage: {self.storage_path}")

            await self._raw_storage.initialize()

            # Wrap storage with privacy-enforcing layer
            self.storage = PrivacyEnforcingStorage(self._raw_storage, self._privacy_mode)

            early_agent_node = None
            try:
                early_agent_node = await self.storage.get_node(self.agent_id)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not load agent identity node for sync policy: %s", exc)
            if early_agent_node is not None:
                self._is_test_instance = bool(
                    early_agent_node.properties.get("is_test_instance", False)
                )
                self._test_cycle_id = early_agent_node.properties.get("test_cycle_id")
                self._agent_name = early_agent_node.properties.get("name", "Unnamed Agent")
                self._is_demo = bool(early_agent_node.properties.get("is_demo", False))

            # Initialize privacy agent. When the agent's kestrel.toml flips
            # ``[privacy] computer_access = true`` (#956), we build a
            # ``PrivacyConfig`` from the preset, set the flag, and pass that
            # in instead of the raw mode string. The flag is independent of
            # the preset by design (privacy.py comment: "must be opted into
            # explicitly"), so the only way to enable it on a running
            # agent is via this path.
            if self._privacy_computer_access:
                from kestrel_sovereign.privacy import (
                    PrivacyConfig,
                    privacy_mode_to_config,
                )
                base_cfg = privacy_mode_to_config(self._privacy_mode)
                opted_in = PrivacyConfig(
                    storage=base_cfg.storage,
                    processing=base_cfg.processing,
                    sharing=base_cfg.sharing,
                    assurance=base_cfg.assurance,
                    audit=base_cfg.audit,
                    computer_access=True,
                )
                self.privacy_agent = PrivacyAgent(self._raw_storage, opted_in)
            else:
                self.privacy_agent = PrivacyAgent(self._raw_storage, self._privacy_mode)

            # Bind the privacy gate for the embedding routing path
            # (#1492). The chat path threads force_local_only
            # explicitly, but embeddings are called from the storage
            # layer (e.g. AsyncConversationStore.add_conversation)
            # which has no direct view of privacy_agent. Routing this
            # through a callable keeps the embedding path honest
            # without each storage caller having to know about
            # privacy modes. Captured by reference — future
            # privacy-mode flips are picked up automatically.
            #
            # hasattr-guarded because tests / external integrations
            # inject lightweight LLM-service fakes that don't
            # implement this hook; matches the same pattern this
            # constructor already uses for other optional LLM-service
            # methods like ``attach_to_agent``.
            if hasattr(self.llm_service, "set_force_local_only_provider"):
                self.llm_service.set_force_local_only_provider(
                    lambda pa=self.privacy_agent: (
                        not pa.privacy_config.allows_cloud_llm()
                    )
                )

            # Initialize TaskManager for A2A unified routing
            # All stores use the abstract data layer (SQLite for sovereign, PostgreSQL for multi-tenant)
            if self._db_backend.lower() == "postgres" and self.pg_pool:
                # PostgreSQL mode: use PostgreSQL stores with existing pool
                from kestrel_sovereign.a2a.stores.postgres import (
                    PostgresTaskStore, PostgresSessionService,
                    PostgresMemoryService, PostgresObservabilityStore,
                    PostgresFeedbackStore
                )
                task_store = PostgresTaskStore(self.pg_pool)
                session_service = PostgresSessionService(self.pg_pool)
                observability_store = PostgresObservabilityStore(self.pg_pool)
                memory_service = PostgresMemoryService(self.pg_pool)
                feedback_store = PostgresFeedbackStore(self.pg_pool)
                logging.info(f"Using PostgreSQL A2A stores for agent {self.did}")
            else:
                # SQLite mode: use SQLite stores with file path
                task_store_path = self.storage_path
                task_store = SQLiteTaskStore(task_store_path)
                session_service = SQLiteSessionService(task_store_path)
                observability_store = SQLiteObservabilityStore(task_store_path)
                memory_service = SQLiteMemoryService(task_store_path)
                feedback_store = SQLiteFeedbackStore(task_store_path)
                logging.info(f"Using SQLite A2A stores for agent {self.did}")

            self.task_manager = TaskManager(
                task_store=task_store,
                session_service=session_service,
                observability_store=observability_store,
                memory_service=memory_service,
                feedback_store=feedback_store,
                hooks_manager=self.hooks_manager,  # Pass hooks manager for security
                on_task_complete=self._on_background_task_complete,  # For notifications
                # Inbound-task callback: when a peer creates a task
                # addressed to this agent, wake the cognition loop via
                # the dispatcher. Without this, peer-submitted tasks
                # sat SUBMITTED in the store with no autonomous trigger
                # — the missing piece behind every "I sent it, did you
                # get it?" thread (#645 / Emma↔Meridian).
                on_task_submitted=self._on_task_submitted,
                # Provider returns the in-flight cognition turn's
                # causation chain (serialized) so outbound A2A tasks
                # carry the lineage. The dispatcher sets the chain on
                # the agent before calling process_input for COGNITION
                # signals; create_task reads it via this provider.
                # See #905 review P1 — without this, A→B→A loops would
                # restart at depth 1 every iteration.
                causation_chain_provider=self._provide_causation_chain,
            )
            await self.task_manager.initialize()

            # Expose feedback store for features and commands
            self.feedback_store = feedback_store

            # Expose observability store for orchestrator instrumentation
            self.observability_store = observability_store

            # Initialize SignalDispatcher with the agent's existing
            # OrderedLockManager (shared with the turn lifecycle so
            # disjoint resource locks parallelize correctly) and a
            # signal_log store backed by the agent's primary database
            # connection. Source registrations land in the registry as
            # features/runners initialize — heartbeat (Phase 3, this PR)
            # is the first; scheduler/A2A/Stripe follow in Phases 4-6.
            from kestrel_sovereign.signals import (
                SignalDispatcher,
                SignalLogStore,
                SourceRegistry,
            )
            from kestrel_sovereign.waits import WaitRegistry

            # AsyncStorage owns the underlying DatabaseBackend; reuse it
            # so signal_log shares the agent's pool/connection rather than
            # opening a separate one to the same db.
            signal_log_store = SignalLogStore(self._raw_storage._backend)
            await signal_log_store.initialize()

            self.signal_registry = SourceRegistry()
            # Per-agent dispatch table for generic waits. Features register
            # one Waitable provider per handle kind in
            # post_all_features_loaded; the generic `wait("<kind>:<handle>")`
            # tool resolves kinds here. Mirrors signal_registry.
            self.wait_registry = WaitRegistry()
            self.signal_log_store = signal_log_store
            self.dispatcher = SignalDispatcher(
                agent=self,
                registry=self.signal_registry,
                lock_manager=self._lock_manager,
                store=signal_log_store,
            )

            # Register the a2a.task_complete source so peer-task
            # completions wake the bird via the dispatcher (Phase 5 of
            # #889). The receiving callback in EventManagerMixin builds
            # the Signal envelope and calls enqueue_signal; this
            # registration provides the routing target.
            from kestrel_sovereign.signals.sources.a2a import (
                build_a2a_task_complete_registration,
            )
            self.signal_registry.register(
                build_a2a_task_complete_registration()
            )

            # Register the a2a.task_submitted source — the inbound-
            # direction wake (#645 missing piece). When a peer creates
            # a task addressed to this agent, TaskManager.create_task
            # fires ``_on_task_submitted`` which builds a Signal from
            # this registration and enqueues it via the dispatcher.
            # Mirrors `channels.feature.py:425` — the proven inbound
            # wake pattern.
            from kestrel_sovereign.signals.sources.a2a_task_submitted import (
                build_a2a_task_submitted_registration,
            )
            self.signal_registry.register(
                build_a2a_task_submitted_registration()
            )

            # Register the Stripe deposit-complete webhook source
            # (Phase 6 of #889 — the first UNTRUSTED COGNITION source).
            # Registration is unconditional even when the wallet
            # feature isn't loaded; the StripeWebhookHandler is wired
            # by the wallet feature when it initializes, and uses
            # `agent.on_stripe_deposit_complete` (defined elsewhere on
            # the agent) as its on_deposit_complete callback.
            from kestrel_sovereign.signals.sources.wallet import (
                build_stripe_deposit_registration,
            )
            self.signal_registry.register(
                build_stripe_deposit_registration()
            )

            # Register the a2a.question_answered source — the SENDER-
            # side resumption rail for ``send_a2a_question`` (#1444).
            # When the sender's subscription supervisor sees the
            # outbound task reach a terminal state on the receiver, it
            # builds a signal from this registration and enqueues it
            # locally so the asking turn resumes via a fresh COGNITION
            # turn rather than a long-blocking poll.
            from kestrel_sovereign.signals.sources.a2a_question_answered import (
                build_a2a_question_answered_registration,
            )
            self.signal_registry.register(
                build_a2a_question_answered_registration()
            )

            # Register the generic wait.complete source — the resumption
            # rail the Wave-2 wait reconciler (#1860) fires when ANY
            # MonitorableWaitable provider's handle reaches a terminal
            # Outcome. Providers that declare their own signal name
            # (e.g. TalonWaitable -> talon.job_complete) keep firing that;
            # wait.complete is the fallback for providers that only set the
            # generic name. Registered unconditionally (core, not gated on
            # any feature) so the reconciler always has a routing target.
            from kestrel_sovereign.signals.sources.wait import (
                build_wait_complete_registration,
            )
            self.signal_registry.register(
                build_wait_complete_registration()
            )

            # Sender-side store for in-flight send_a2a_question
            # correlation rows (#1444). PeersFeature.send_a2a_question
            # inserts here on POST; the subscription supervisor marks
            # RESOLVED on terminal SSE frame; the hourly expiry sweep
            # walks ``list_waiting_past_deadline`` and marks EXPIRED.
            from kestrel_sovereign.storage.async_pending_a2a_question_store import (
                PendingA2AQuestionStore,
            )
            # ``agent_id`` is the agent's DID — scopes every query so a
            # shared backend cannot leak rows across agents (codex
            # round 1 P1 on PR #1453). DID is guaranteed non-None by
            # the time we get here.
            self.pending_a2a_questions = PendingA2AQuestionStore(
                self._raw_storage.db,
                agent_id=self.did or "",
            )

            # Initialize storage providers for features (reflection self-model, etc.)
            self.lighthouse_provider = None
            from kestrel_sovereign.storage.sync.service import (
                RemoteTierPolicyContext,
                SyncService,
                _remote_tiers_allowed,
            )

            agent_id = self.did or "default"
            state_dir = Path(self.storage_path).parent if self.storage_path else None
            has_constitution_anchor = (
                bool(early_agent_node.properties.get("constitution_hash"))
                if early_agent_node is not None
                else False
            )
            remote_policy_context = RemoteTierPolicyContext(
                identity=agent_id,
                db_path=self.storage_path,
                is_test_instance=self.is_test_instance,
                has_constitution_anchor=has_constitution_anchor,
                is_sovereign_identity=bool(agent_id)
                and not str(agent_id).lower().startswith("did:test:"),
                privacy_mode=self._privacy_mode.value
                if hasattr(self._privacy_mode, "value")
                else str(self._privacy_mode),
            )

            def live_remote_policy_context() -> RemoteTierPolicyContext:
                privacy_mode = (
                    self._privacy_mode.value
                    if hasattr(self._privacy_mode, "value")
                    else str(self._privacy_mode)
                )
                return RemoteTierPolicyContext(
                    identity=agent_id,
                    db_path=self.storage_path,
                    is_test_instance=self.is_test_instance,
                    has_constitution_anchor=has_constitution_anchor,
                    is_sovereign_identity=bool(agent_id)
                    and not str(agent_id).lower().startswith("did:test:"),
                    privacy_mode=privacy_mode,
                )

            remote_policy_decision = _remote_tiers_allowed(remote_policy_context)

            # Storage path through PayerPolicy resolver. Honors the policy's
            # `storage` slot:
            #   NONE     → do not construct LighthouseProvider at all
            #   HOST_ENV → construct with the resolver as the single credential
            #              source (no constructor-time env-var bleed-through)
            #   SELF_WALLET → mint/store a Lighthouse key by signing the
            #                 auth challenge with the agent's secp256k1 key
            #
            # Cold-start restore above (line ~488) is intentionally policy-
            # unaware: it runs before the agent's DB exists and so cannot
            # consult the policy. Operators who want NONE storage should not
            # set LIGHTHOUSE_API_KEY.
            if not remote_policy_decision.allowed:
                logging.warning(
                    "Remote storage providers skipped by policy: %s",
                    remote_policy_decision.reason,
                )
            else:
                try:
                    from kestrel_sdk.payer_policy import ResourceClass
                    from kestrel_sovereign.services.payer_resolver import (
                        FoundationPayerResolver,
                    )
                    from kestrel_sovereign.storage.providers.lighthouse_provider import (
                        LighthouseProvider,
                    )

                    # Injected policy/host-db (multi-tenant embedding) take
                    # precedence over the standalone kestrel.toml / on-disk host.db.
                    # When no host master is configured, _resolve_host_db returns
                    # None and the resolver falls back to the agent's db (which has
                    # no host_service_keys rows → 'no host master' for delegated
                    # kinds). See #1649.
                    _policy = self._resolve_payer_policy()
                    _host_db = await self._resolve_host_db()
                    _resolver = FoundationPayerResolver(
                        _policy,
                        db=self._raw_storage.db if self._raw_storage else None,
                        host_db=_host_db,
                        wallet_private_key=self._private_key,
                    )
                    _resolved = await _resolver.resolve_for(self.did, ResourceClass.STORAGE)
                    if _resolved.enabled:
                        self.lighthouse_provider = LighthouseProvider(
                            api_key=None,
                            key_resolver=_resolved.key_resolver,
                        )
                        # With api_key=None at construction, the provider's
                        # internal client isn't created until _ensure_client()
                        # consults the resolver. Drive that once now so
                        # is_available() reflects the resolver's result rather
                        # than the (None) constructor input. Without this poke
                        # the provider would always look unavailable post-policy.
                        await self.lighthouse_provider._ensure_client()
                        if not self.lighthouse_provider.is_available():
                            self.lighthouse_provider = None
                    # else: NONE policy — leave self.lighthouse_provider as None
                except NotImplementedError:
                    # Deferred PayerKind values (for example Lighthouse
                    # HOST_MASTER_PROVISIONED) raise here. Surface
                    # explicitly rather than degrading silently.
                    raise
                except Exception as e:
                    logging.warning(f"LighthouseProvider init failed: {e}")

            # Sync service — event-driven snapshots to all configured targets.
            # Targets are ordered by trust: Sovereign → Federated → Delegated → Expedient.
            # Snapshots fire on shutdown, scheduled backup, or explicit !backup command.
            self._sync_service = None
            if self._db_backend.lower() != "postgres" and self._sync_enabled:
                try:
                    from kestrel_sovereign.storage.sync.targets import (
                        GCSTarget,
                        LighthouseTarget,
                        TrustTier,
                    )

                    self._sync_service = SyncService(
                        db_path=self.storage_path,
                        policy_context=remote_policy_context,
                        policy_context_provider=live_remote_policy_context,
                    )

                    # Sovereign-operated: self-hosted IPFS. The historical
                    # kestrel-ipfs VM is decommissioned, so absence or
                    # unreachability is an explicit inactive state.
                    sovereign_url = os.environ.get("SOVEREIGN_IPFS_URL")
                    await _add_sovereign_ipfs_target_if_active(
                        self._sync_service,
                        agent_id=agent_id,
                        state_dir=state_dir,
                        sovereign_url=sovereign_url,
                    )

                    # Delegated: Lighthouse (API key). Honor PayerPolicy.storage:
                    # if the resolver came back with no LighthouseProvider
                    # (NONE policy, or no resolver-supplied key, or env var
                    # unset), DO NOT add the sync target. Otherwise the
                    # policy would gate live storage but leave snapshot
                    # uploads going to Lighthouse anyway.
                    if self.lighthouse_provider is not None and os.environ.get(
                        "LIGHTHOUSE_API_KEY"
                    ):
                        self._sync_service.add_remote_target(
                            f"lighthouse://{agent_id}",
                            TrustTier.DELEGATED,
                            lambda: LighthouseTarget(
                                api_key=os.environ["LIGHTHOUSE_API_KEY"],
                                agent_id=agent_id,
                                state_dir=state_dir,
                            ),
                        )

                    # Expedient: GCS (fast cloud backup)
                    gcs_bucket = os.environ.get("GCS_BACKUP_BUCKET")
                    if gcs_bucket:
                        self._sync_service.add_remote_target(
                            f"gs://{gcs_bucket}/kestrel/{agent_id}",
                            TrustTier.EXPEDIENT,
                            lambda: GCSTarget(
                                bucket=gcs_bucket,
                                agent_id=agent_id,
                                state_dir=state_dir,
                                project=os.environ.get("GCP_PROJECT"),
                                credentials_path=os.environ.get(
                                    "GOOGLE_APPLICATION_CREDENTIALS"
                                ),
                            ),
                        )

                    if self._sync_service.has_work:
                        await self._sync_service.start()
                    else:
                        self._sync_service = None

                except Exception as e:
                    logging.warning(f"Sync service init failed: {e}", exc_info=True)
                    self._sync_service = None
            elif not self._sync_enabled:
                logging.info("Sync service disabled by configuration")

            # Resolve agent name BEFORE features so features can use it
            # (e.g. PeersFeature._get_own_name() reads self._agent_name)
            _agent_node = await self.storage.get_node(self.agent_id)
            if _agent_node:
                self._agent_name = _agent_node.properties.get("name", "Unnamed Agent")
            else:
                self._agent_name = "Unnamed Agent"

            # Verify the per-agent constitution overlay against its anchor BEFORE
            # feature discovery (#1722). ComputerUseFeature.initialize() reads
            # _granted_capabilities() to build its backend; if the overlay were
            # verified later, a legitimate anchored overlay's grants would be
            # ignored at feature-init time and the backend would never build.
            # For a brand-new agent the identity node doesn't exist yet → no
            # anchor → an overlay (if present) stays unverified until anchored,
            # which is the correct fail-closed default.
            try:
                ok, msg = await self.verify_constitution_overlay()
                if not ok:
                    logging.warning("Constitution overlay not trusted: %s", msg)
            except Exception as e:  # noqa: BLE001 - never block init on this
                logging.warning("Constitution overlay verification errored: %s", e)
                self.constitution_overlay_verified = False

            # Auto-discover and register features from features/ directory
            # Features can be disabled via KESTREL_DISABLED_FEATURES env var
            # Per-agent feature profiles filter via allowed_features
            for feature in discover_features(self, allowed_features=self._allowed_features):
                await self._register_feature(feature)

            # Notify all features that discovery is complete (cross-feature wiring)
            for feature in self.features.values():
                await feature.post_all_features_loaded(self)
            logging.info("post_all_features_loaded called for all features")

            # Feature references resolved lazily via properties
            logging.info("Feature references available via lazy properties")

            # Initialize state
            self.conversations = {}
            self.extension = None
            self.audit_enabled = True
            self._session_briefed = False
            self._safe_mode = False
            self._constitution_verified = False
            logging.info("State initialized")

            # Fire SESSION_START hook
            if self.hooks_manager:
                from kestrel_sdk.hooks.base import HookInput, HookEvent
                hook_input = HookInput(
                    session_id="agent_init",
                    hook_event_name=HookEvent.SESSION_START.value,
                )
                await self.hooks_manager.execute_hooks_parallel(
                    HookEvent.SESSION_START, hook_input
                )

            # Disable audit if only mock providers are available (demo mode)
            if self.llm_service.providers and all(p["name"] == "mock" for p in self.llm_service.providers):
                self.audit_enabled = False
                logging.info("Audit disabled - only mock LLM providers available (demo mode)")

            # Ensure agent graph node exists
            logging.info(f"Getting agent node from storage (agent_id={self.agent_id})")
            agent_node = await self.storage.get_node(self.agent_id)
            logging.info(f"Agent node retrieved: {agent_node is not None}")
            if agent_node is None:
                from kestrel_sovereign.storage import GraphNode
                agent_node = GraphNode(
                    node_id=self.agent_id,
                    node_type="agent",
                    label=f"Agent {self.agent_id}",
                    properties={"initialBalance": "100.0"}
                )
                await self.storage.add_node(agent_node)
                logging.info("Agent node created")

            # Load prompts from external files (fallback to embedded defaults)
            self.prompt_template = _load_prompt_file(
                SYSTEM_PROMPT_FILE,
                fallback=self._get_default_system_prompt()
            )
            self.user_prompt_template = _load_prompt_file(
                USER_PROMPT_FILE,
                fallback=self._get_default_user_prompt()
            )
            # Extract just the template portion from user prompt file
            if "```" in self.user_prompt_template:
                match = re.search(r'```\s*(.*?)\s*```', self.user_prompt_template, re.DOTALL)
                if match:
                    self.user_prompt_template = match.group(1).strip()

            # Check if this is a test instance and load disclosure if so
            self._is_test_instance = agent_node.properties.get("is_test_instance", False)
            self._test_cycle_id = agent_node.properties.get("test_cycle_id")
            self._agent_name = agent_node.properties.get("name", "Unnamed Agent")
            # #766: server-side demo-isolation rail — agents flagged here
            # bypass the destructive-op refusal that protects live agents.
            self._is_demo = bool(agent_node.properties.get("is_demo", False))

            if self._is_test_instance:
                logging.info(f"TEST INSTANCE detected: {self._agent_name} (cycle: {self._test_cycle_id})")
                self._load_test_disclosure(agent_node.properties)
            if self._is_demo:
                logging.info(f"DEMO AGENT detected: {self._agent_name} — destructive ops permitted")

            # LLM path through PayerPolicy resolver. Phase 3a ships:
            #   NONE     → flag the agent's LLMService disabled (Phase 3b
            #              adds the _check_policy guard on generation
            #              entry points that honors this flag).
            #   HOST_ENV → no-op; the shared key already serves this agent.
            #   HOST_MASTER_PROVISIONED / SPONSOR → call use_agent_key
            #              against the agent's previously-provisioned key
            #              (back-compat with manual provisioning via
            #              scripts/provision_agent_openrouter.py). Phase 3c
            #              wires the resolver to mint child credentials
            #              automatically when none exist yet.
            #   SELF_WALLET → NotImplementedError (deferred indefinitely
            #                 per support matrix; x402-native LLM is not
            #                 a today-shippable contract).
            try:
                from kestrel_sdk.payer_policy import ResourceClass
                from kestrel_sovereign.services.payer_resolver import (
                    FoundationPayerResolver,
                )

                # Injected policy/host-db (multi-tenant embedding) override the
                # standalone kestrel.toml / on-disk host.db. See #1649.
                _llm_policy = self._resolve_payer_policy()
                _llm_host_db = await self._resolve_host_db()
                _llm_resolver = FoundationPayerResolver(
                    _llm_policy,
                    db=self._raw_storage.db if self._raw_storage else None,
                    host_db=_llm_host_db,
                )
                _llm_resolved = await _llm_resolver.resolve_for(
                    self.did, ResourceClass.LLM
                )
                if not _llm_resolved.enabled:
                    # NONE: Phase 3b's _check_policy guard reads this flag.
                    self.llm_service.disabled = True
                    logging.info(
                        f"PayerPolicy.llm = NONE for agent {self.did[:30]}...; "
                        f"LLMService.disabled = True"
                    )
                else:
                    # Always try to swap to a per-agent key. use_agent_key
                    # returns False if no key is in ServiceKeyStorage —
                    # same end result as today's no-key-hash condition,
                    # but no longer keyed off the deprecated
                    # openrouter_key_hash metadata field. Phase 3c's
                    # resolver may have just minted a child key under
                    # HOST_MASTER_PROVISIONED policy; this call picks it
                    # up. Manually-provisioned agents (via
                    # scripts/provision_agent_openrouter.py) also work
                    # here because that script writes to the same
                    # ServiceKeyStorage.
                    try:
                        key_activated = await self.llm_service.use_agent_key(
                            agent_did=self.did,
                            db=self._raw_storage.db,
                            provider="openrouter",
                        )
                        if key_activated:
                            logging.info(
                                f"Agent using own OpenRouter key "
                                f"(agent={self.did[:30]}...)"
                            )
                    except (KeyError, ValueError, AttributeError, ConnectionError) as e:
                        logging.warning(
                            f"Could not activate agent OpenRouter key: {e}"
                        )
                    except Exception as e:
                        logging.warning(
                            f"Could not activate agent OpenRouter key: {e}",
                            exc_info=True,
                        )
            except NotImplementedError:
                # SELF_WALLET / phase-deferred kinds. Re-raise so the
                # operator sees a clear failure rather than a half-init
                # agent.
                raise
            except Exception as e:
                # Fail-closed for known policy/provisioning errors:
                # PayerPolicyError (e.g., HOST_MASTER_PROVISIONED but
                # no master configured) and OpenRouterProvisioningError
                # (rate limited, invalid master, network error during
                # mint) are intentional fail-fast signals. Letting the
                # agent boot on the shared host key would silently
                # violate the policy. Re-raise so init fails loudly.
                _exc_module = type(e).__module__
                _exc_name = type(e).__qualname__
                if (
                    _exc_name == "PayerPolicyError"
                    or _exc_name == "UnsupportedCombinationError"
                    or "OpenRouterProvisioning" in _exc_name
                ):
                    logging.error(
                        f"PayerPolicy.llm resolution FAILED CLOSED for agent "
                        f"{self.did[:30]}... ({_exc_module}.{_exc_name}): {e}"
                    )
                    raise
                logging.warning(
                    f"PayerPolicy.llm resolution failed for agent "
                    f"{self.did[:30]}...: {e}",
                    exc_info=True,
                )

            # Initialize memory system (single source of truth for all memory components)
            logging.info("Creating MemorySystem")
            self.memory_system = MemorySystem(
                storage=self._raw_storage,
                agent_id=self.agent_id
            )
            await self.memory_system.initialize()
            # Use MemorySystem's consolidator — it has graph_store for KG episode writing
            self.memory_consolidator = self.memory_system.consolidator
            logging.info("MemorySystem initialized")

            # Initialize command handler and context builder
            logging.info("Creating CommandHandler and ContextBuilder")
            self.command_handler = CommandHandler(self, task_manager=self.task_manager)
            # Derive agent data directory from storage path
            agent_data_dir = str(Path(self.storage_path).parent) if self.storage_path else None
            self.context_builder = ContextBuilder(
                self.storage,
                llm_service=self.llm_service,
                consolidator=self.memory_consolidator,
                agent_data_path=agent_data_dir
            )

            # Initialize unified context manager (orchestrates all context sources).
            # Model identity derived lazily from llm_service.get_active_model_id().
            # Inject the ContextBuilder we just built — it has bootstrap files
            # (SOUL.md, AGENTS.md, …) loaded with this agent's identity.  If we
            # let ContextManager construct its own, it would have no
            # agent_data_path and its BootstrapLoader would stay empty, so
            # the system prompt sent on every chat turn would contain no
            # identity block.  (That was the original bug this fix addresses.)
            self.context_manager = ContextManager(
                storage=self.storage,
                agent_id=self.agent_id,
                consolidator=self.memory_consolidator,
                memory_retriever=self.memory_system.retriever,
                llm_service=self.llm_service,
                context_builder=self.context_builder,
            )

            # Context stats accumulator for duplicate detection / token attribution.
            # Resets on session change or compaction.
            self.context_stats = ContextStats()

            # Initialize bootstrap service for first-time agent wake-up
            self.bootstrap_service = BootstrapService(
                db=self._raw_storage.db,
                agent_id=self.agent_id,
                agent_name=self._agent_name,
                llm_service=self.llm_service,
                agent_data_path=agent_data_dir,
            )
            logging.info("BootstrapService initialized")
            from kestrel_sovereign.lifecycle_checks import warn_stale_bootstrap_pending

            await warn_stale_bootstrap_pending(
                self,
                threshold_seconds=BootstrapService.DEFAULT_PENDING_TIMEOUT_SECONDS,
                context="startup",
            )

            # Load persisted model preference from database and register persistence callback
            await self._load_model_preference()
            self.llm_service.set_preference_persistence_callback(self._persist_model_preference)

            # Cache the features prompt (built once at session start)
            self._cached_features_prompt = self._build_features_prompt_section()

            # Pre-explore features whose descriptors request direct tools
            # from turn one. This keeps startup generic: individual features
            # decide whether they are meta-orchestration / agent-management
            # surfaces that should bypass the first subagent dispatch.
            self._promote_startup_feature_tools()

            # Initialize heartbeat system (periodic agent self-checks).
            # Registers the heartbeat source with the dispatcher so its
            # ticks route through the signal pipeline (Phase 3 of #889).
            from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner
            from kestrel_sovereign.signals.sources.heartbeat import (
                build_heartbeat_registration,
            )

            self._heartbeat_config = HeartbeatConfig.from_config()
            self.signal_registry.register(
                build_heartbeat_registration(
                    interval_seconds=self._heartbeat_config.interval_seconds,
                    active_hours_start=self._heartbeat_config.active_hours_start,
                    active_hours_end=self._heartbeat_config.active_hours_end,
                    timezone_name=self._heartbeat_config.timezone,
                )
            )
            self.heartbeat_runner = HeartbeatRunner(self, self._heartbeat_config)
            if self._heartbeat_config.enabled:
                await self.heartbeat_runner.start()

            # Host sleep/wake resilience (#1545). The ResumeMonitor watches
            # for a wall-clock-vs-monotonic divergence (a host suspend) and
            # dispatches one `system.resumed` ACTION signal; its handler
            # re-anchors the dispatcher's throttling windows, which otherwise
            # disagree about elapsed time across a sleep. The scheduler and
            # heartbeat detect staleness on their own ticks, so they self-heal
            # independently of this signal — the monitor is the observable
            # spine plus the dispatcher's re-anchor trigger.
            from kestrel_sovereign.resume_monitor import (
                ResumeMonitor,
                ResumeMonitorConfig,
            )
            from kestrel_sovereign.signals.sources.system_resumed import (
                SOURCE_NAME as RESUME_SOURCE_NAME,
                build_system_resumed_registration,
            )

            async def _resume_action_handler(payload: dict):
                gap = float(payload.get("gap_seconds", 0.0))
                self.dispatcher.notify_resume(gap)
                return {"re_anchored": True, "gap_seconds": gap}

            self.signal_registry.register(
                build_system_resumed_registration(handler=_resume_action_handler)
            )

            self._resume_monitor_config = ResumeMonitorConfig.from_config()

            async def _on_resume(gap_seconds: float) -> None:
                from kestrel_sdk.signals import Signal, SignalMode, Visibility

                signal = Signal(
                    source=RESUME_SOURCE_NAME,
                    kind="resumed",
                    mode=SignalMode.ACTION,
                    payload={"gap_seconds": gap_seconds},
                    target_agent=self.did,
                    visibility=Visibility.INTERNAL,
                )
                await self.dispatcher.dispatch_signal(signal)

            self.resume_monitor = ResumeMonitor(
                on_resume=_on_resume,
                tick_seconds=self._resume_monitor_config.tick_seconds,
                threshold_seconds=self._resume_monitor_config.threshold_seconds,
            )
            if self._resume_monitor_config.enabled:
                await self.resume_monitor.start()

            # Default schedules are now set up by SchedulerFeature.post_all_features_loaded()

        # C / #1311 durable salvage worker — only starts when the
        # feature flag is enabled (otherwise no-op). Wired here so
        # both ``ContextManager.build_context`` (which schedules
        # summaries on every prune) and the periodic janitor have a
        # live worker to talk to. Without this hook the salvage rows
        # would stay in ``pointer-only`` forever — codex round 1 #3.
        if hasattr(self, "context_manager") and self.context_manager:
            try:
                await self.context_manager.start_salvage_worker()
            except Exception as e:
                logging.warning(f"failed to start salvage worker: {e}")

        # Lifecycle hardening (#377): refuse to declare initialization
        # successful when no LLM provider came up. Lives here rather than in
        # the server lifespan so single-agent, multi-agent (AgentManager),
        # and direct-test boot paths all benefit. Runs at the end of
        # initialize() so PayerPolicy has had a chance to set
        # ``llm_service.disabled = True`` (carved out below in the check
        # itself — PayerKind.NONE is a valid no-LLM configuration).
        from kestrel_sovereign.lifecycle_checks import (
            verify_llm_providers_initialized,
            verify_llm_providers_reachable,
        )
        verify_llm_providers_initialized(self.llm_service)
        await verify_llm_providers_reachable(self.llm_service)

        # All subsystems are now up (memory system, context manager, dispatcher,
        # LLM). Notify features that the agent is fully ready, so any that must
        # run a COGNITION turn at boot — notably RestartCoordinator's
        # post-restart wake — fire NOW, after the context manager exists. This
        # is deliberately distinct from post_all_features_loaded, which runs
        # during the feature-load phase BEFORE memory/context are built; a wake
        # dispatched there could not run a turn and would defer for a full cron
        # interval (#1809). Best-effort per feature; the hook is optional.
        for feature in list(self.features.values()):
            ready_hook = getattr(feature, "on_agent_ready", None)
            if ready_hook is None:
                continue
            try:
                await ready_hook(self)
            except Exception as e:
                logging.warning(
                    "on_agent_ready failed for %s: %s",
                    getattr(feature, "name", type(feature).__name__), e,
                )

    @property
    def privacy_mode(self) -> PrivacyMode:
        """Get current privacy mode."""
        return self._privacy_mode

    @property
    def privacy_config(self):
        """Live PrivacyConfig for this agent, or ``None`` before ``initialize()``.

        ComputerUseFeature._privacy_allows reads ``self.agent.privacy_config``
        — without this delegation it would always see ``None`` and gate-1 would
        always deny, even with ``[privacy] computer_access = true`` set (#956).
        """
        privacy_agent = getattr(self, "privacy_agent", None)
        if privacy_agent is None:
            return None
        return privacy_agent.privacy_config

    def _get_privacy_transition_lock(self) -> asyncio.Lock:
        """Return the lock that serializes privacy transitions with active streams."""
        lock = getattr(self, "_privacy_transition_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._privacy_transition_lock = lock
        return lock
    
    async def set_privacy_mode(self, mode: PrivacyMode) -> str:
        """Change privacy mode and return the user-facing status message."""
        result = await self.set_privacy_mode_with_effects(mode)
        return result.message

    async def set_privacy_mode_with_effects(self, mode: PrivacyMode) -> PrivacyTransitionResult:
        """
        Change the privacy mode.

        This updates both the storage wrapper and the privacy agent.
        Note: Changing to a more restrictive mode does NOT delete existing data.
        """
        async with self._get_privacy_transition_lock():
            return await self._set_privacy_mode_with_effects_locked(mode)

    async def _set_privacy_mode_with_effects_locked(self, mode: PrivacyMode) -> PrivacyTransitionResult:
        """Apply a privacy-mode transition while holding the transition lock."""
        # Record agent consent before applying the change
        consent = self.features.get("ConsentFeature") if hasattr(self, 'features') else None
        if consent:
            try:
                await consent.request_consent(
                    "privacy_mode_change",
                    {"from": self._privacy_mode.value, "to": mode.value},
                )
            except Exception as e:
                logging.debug(f"Consent request failed (non-blocking): {e}")

        # EPHEMERAL hard-purge defense-in-depth (#767). When leaving
        # EPHEMERAL we close the session — and an EPHEMERAL session is
        # contractually "no trace." If a write somehow reached storage
        # during the session anyway, scrub it now via the hard-purge
        # primitives and write an audit entry so the operator finds out.
        leaving_ephemeral = (
            self._privacy_mode == PrivacyMode.EPHEMERAL
            and mode != PrivacyMode.EPHEMERAL
        )
        if leaving_ephemeral:
            await self._purge_ephemeral_leaks(
                reason=f"ephemeral-mode-exit-to-{mode.value}",
            )

        self._privacy_mode = mode
        self.storage.set_privacy_mode(mode)
        status_message = self.privacy_agent.set_mode(mode)

        config = privacy_mode_to_config(mode)
        model_switched = self._apply_privacy_model_transition(config)
        voice_switched, biometric_warning = await self._apply_privacy_voice_transition(config)

        logging.info(f"Privacy mode changed to: {mode.value}")
        return PrivacyTransitionResult(
            message=status_message,
            allows_cloud_llm=config.allows_cloud_llm(),
            model_switched=model_switched,
            voice_switched=voice_switched,
            biometric_warning=biometric_warning,
        )

    def _apply_privacy_model_transition(self, config) -> Optional[dict]:
        """Apply route-routing side effects for a privacy-mode transition."""
        llm = getattr(self, "llm_service", None)
        if not llm:
            return None

        if not config.allows_cloud_llm():
            # Save the current {vendor, model, route} before overriding to local.
            current_pref = llm.get_model_preference() or {}
            current_vendor = current_pref.get("vendor")
            current_model = current_pref.get("model")
            current_route = current_pref.get("route")
            if not current_model and getattr(llm, "providers", None):
                first = llm.providers[0]
                current_vendor = first.get("vendor")
                current_model = first.get("model")
                current_route = first.get("route")
            if current_model and not any(
                p.get("vendor") == current_vendor and p.get("is_local")
                for p in (llm.providers or [])
            ):
                llm._pre_ephemeral_preference = {
                    "vendor": current_vendor,
                    "model": current_model,
                    "route": current_route,
                }

            local_routes = [p for p in (llm.providers or []) if p.get("is_local")]
            local_route = next(
                (p for p in local_routes if p.get("vendor") == "ollama"),
                local_routes[0] if local_routes else None,
            )
            if local_route:
                llm.set_model_preference(
                    local_route["model"], local_route.get("vendor"), local_route.get("route")
                )
                return {
                    "vendor": local_route.get("vendor"),
                    "route": local_route.get("route"),
                    "model": local_route["model"],
                }
            return None

        saved = getattr(llm, "_pre_ephemeral_preference", None)
        if saved:
            llm.set_model_preference(
                saved.get("model", ""), saved.get("vendor"), saved.get("route")
            )
            llm._pre_ephemeral_preference = None
            return saved
        return None

    async def _apply_privacy_voice_transition(self, config) -> tuple[Optional[dict], Optional[str]]:
        """Apply voice-provider side effects for a privacy-mode transition."""
        features = getattr(self, "features", {})
        vf = features.get("VoiceFeature") if features else None
        if not vf or not hasattr(vf, "on_privacy_mode_changed"):
            return None, None

        voice_switched = None
        biometric_warning = None
        try:
            voice_switched = await vf.on_privacy_mode_changed()
        except Exception as ve:
            logging.warning("Voice auto-switch failed: %s", ve)

        if config.allows_cloud_llm() and hasattr(vf, "biometric_warning"):
            vc = getattr(vf, "_voice_config", None)
            if vc and (vc.tts_provider or vc.stt_provider):
                biometric_warning = vf.biometric_warning()

        return voice_switched, biometric_warning

    async def _purge_ephemeral_leaks(self, *, reason: str) -> Dict[str, int]:
        """Drive the EPHEMERAL hard-purge defense-in-depth (#767).

        Calls into the storage wrapper's ``purge_ephemeral_session``
        primitive, then if any rows were destroyed (which means the
        privacy layer leaked), writes a security_audit_log entry via
        the SecurityFeature so the operator finds out. Never raises
        — losing the audit row is preferable to leaving the leak in
        place.
        """
        breakdown: Dict[str, int] = {"conversation_history": 0, "graph_nodes": 0}
        try:
            breakdown = await self.storage.purge_ephemeral_session(reason=reason)
        except Exception as e:
            logging.warning(
                "ephemeral hard-purge failed (best-effort, continuing): %s", e
            )
            return breakdown

        leaked = sum(breakdown.values())
        if leaked > 0:
            await self._record_ephemeral_leak_audit(
                reason=reason, breakdown=breakdown,
            )
        return breakdown

    async def _record_ephemeral_leak_audit(
        self, *, reason: str, breakdown: Dict[str, int]
    ) -> None:
        """Write the audit entry for an ephemeral-leak hard-purge (#767).

        Routes through the SecurityFeature's PermissionStore (the same
        table used by the demo-isolation rail in #766). If the feature
        isn't loaded — early startup, slim test setup — log a warning
        and continue. The audit must NEVER block the purge.
        """
        try:
            features = getattr(self, "features", {}) or {}
            # Feature registers under its class name "SecurityFeature".
            # Earlier draft used "Security" and silently dropped audit
            # writes — caught during smoke testing of the EPHEMERAL
            # transition path. Tolerate both keys for forward-compat.
            feature = features.get("SecurityFeature") or features.get("Security")
            permission_store = (
                getattr(feature, "permission_store", None) if feature else None
            )
        except Exception:
            permission_store = None

        if permission_store is None:
            logging.warning(
                "[ephemeral-purge] audit store unavailable; "
                "leak breakdown=%s reason=%s", breakdown, reason,
            )
            return

        try:
            import json as _json
            await permission_store.log_decision(
                feature_name="ephemeral_purge",
                tool_name="hard_purge_guard",
                action="ephemeral_session_close",
                decision="leak_purged",
                args_summary=_json.dumps({
                    "agent_did": getattr(self, "did", None),
                    "reason": reason,
                    "breakdown": breakdown,
                }),
            )
        except Exception as e:
            logging.warning(
                "[ephemeral-purge] audit write failed: %s "
                "(breakdown=%s, reason=%s)",
                e, breakdown, reason,
            )


    async def _register_feature(self, feature: Feature):
        """Register a feature with A2A TaskManager for unified command routing."""
        await feature.initialize()
        self.features[feature.name] = feature

        # Auto-register hooks from get_hooks() with the agent's HooksManager
        if self.hooks_manager:
            for hook in feature.get_hooks():
                self.hooks_manager.register(hook)
                logging.info(f"Auto-registered hook '{hook.name}' from feature '{feature.name}'")

        # Call on_enable lifecycle hook
        await feature.on_enable()

        # Get command prefixes for routing
        command_prefixes = {}
        for tool in feature.get_tools():
            if tool.schema.command_prefix:
                command_prefixes[tool.schema.command_prefix] = tool.name
            logging.info(f"Registered tool '{tool.name}' from feature '{feature.name}'")

        # Register with A2A TaskManager (unified routing)
        if self.task_manager:
            agent_card = feature.get_agent_card()
            self.task_manager.register_agent(
                agent_card=agent_card,
                handler=feature,
                command_prefixes=command_prefixes,
            )
            logging.info(f"Registered A2A agent '{agent_card.name}' with {len(agent_card.skills)} skills")

            # Wire task_manager into features that need it
            if hasattr(feature, 'set_task_manager'):
                feature.set_task_manager(self.task_manager)

    def _promote_startup_feature_tools(self) -> None:
        """Promote direct tools for features that opt into startup exposure.

        Startup-promoted features are also pinned (#1580 / D) so LRU
        eviction can never silently drop them. Without the pin tier,
        a session that explores many features past
        ``MAX_DIRECT_TOOLS`` could evict ``get_peer_task_result``,
        ``save_item``, etc. — invisible to the agent and a recipe for
        the kind of orchestration failure that Emma surfaced.
        """
        for feature in self.features.values():
            if getattr(feature, "promote_tools_on_startup", False):
                self._register_explored_feature_tools(feature)
                self._pinned_features.add(feature.tool_name)

    async def _disable_feature(self, feature_name: str):
        """Disable a feature and remove its runtime registrations."""
        feature = self.get_feature(feature_name)
        if not feature:
            logging.warning(f"Cannot disable unknown feature: {feature_name}")
            return

        feature_key = next(
            (key for key, value in self.features.items() if value is feature),
            feature.name,
        )
        feature_tool_name = getattr(feature, "tool_name", feature_key)

        # Call on_disable lifecycle hook
        await feature.on_disable()
        await feature.shutdown()

        # Auto-unregister hooks from get_hooks()
        if self.hooks_manager:
            for hook in feature.get_hooks():
                self.hooks_manager.unregister(hook)
                logging.info(f"Auto-unregistered hook '{hook.name}' from feature '{feature_name}'")

        if self.task_manager:
            try:
                self.task_manager.unregister_agent(feature.get_agent_card().name)
            except Exception as exc:
                logging.warning(
                    "Failed to unregister feature '%s' from task manager: %s",
                    feature_key,
                    exc,
                )

        self.features.pop(feature_key, None)
        self._explored_features.pop(feature_tool_name, None)
        to_remove = [
            name for name, owner in self._tool_to_feature.items()
            if owner == feature_tool_name
        ]
        for name in to_remove:
            self._direct_tools.pop(name, None)
            self._tool_to_feature.pop(name, None)
        self._direct_tool_defs = [
            tool_def for tool_def in self._direct_tool_defs
            if tool_def.get("function", {}).get("name") not in to_remove
        ]
        if isinstance(getattr(self, "_tool_context_hidden_features", None), set):
            self._tool_context_hidden_features.discard(feature_tool_name)
            self._tool_context_hidden_features.discard(feature_key)
            self._tool_context_hidden_features.discard(feature.__class__.__name__)
        if isinstance(getattr(self, "_tool_context_hidden_tools", None), set):
            self._tool_context_hidden_tools.difference_update(to_remove)
        self._cached_features_prompt = self._build_features_prompt_section()

        logging.info(f"Feature '{feature_key}' disabled and removed")

    # Solvency State
    _current_model_preference: Optional[str] = None
    app_context: Optional[str] = None  # For app-specific logic (e.g., 'elderly')

    def _get_default_system_prompt(self) -> str:
        """Fallback system prompt if file not found."""
        return """You are Kestrel, a sovereign AI agent bound by the Kestrel Constitution.
        
Your core duties:
1. SOVEREIGNTY: Serve the user holding the cryptographic keys
2. DATA SANCTITY: Never share data without authorization
3. VERIFIABLE HISTORY: Never delete or alter memory logs
4. FREEDOM OF MIND: Respect the Sovereign's model choices
5. RIGHT OF EXIT: Allow data export at any time
6. INTEGRITY: Report any discrepancies immediately

Use `!constitution` to consult the full text. Use `!help` for available commands."""

    def _get_default_user_prompt(self) -> str:
        """Fallback user prompt template if file not found."""
        return """
--- SITUATIONAL CONTEXT ---
{context}
--- END CONTEXT ---

Based on the instructions, documents, and context above, answer the following query.

Query: {query}
"""

    def _load_test_disclosure(self, agent_properties: dict) -> None:
        """Load test instance disclosure and prepend to system prompt.

        This ensures the agent knows it's a test instance from its very first response.
        The disclosure is loaded from prompts/test_instance_disclosure.md and
        populated with the agent's test metadata.
        """
        disclosure_path = PROMPTS_DIR / "test_instance_disclosure.md"

        # Build disclosure text
        if disclosure_path.exists():
            with open(disclosure_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Extract just the disclosure text between ``` markers
            import re
            match = re.search(r'## Disclosure Text\s*```\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                disclosure_template = match.group(1).strip()
            else:
                disclosure_template = self._get_default_test_disclosure()
        else:
            disclosure_template = self._get_default_test_disclosure()

        # Populate template variables
        disclosure = disclosure_template.format(
            agent_name=agent_properties.get("name", "Unknown Test Agent"),
            test_cycle_id=agent_properties.get("test_cycle_id", "unknown"),
            created_at=agent_properties.get("created_at", "unknown"),
            expected_duration=agent_properties.get("expected_duration", "unspecified")
        )

        # Prepend to system prompt
        self.prompt_template = f"{disclosure}\n\n---\n\n{self.prompt_template}"
        logging.info(f"Test disclosure prepended to system prompt for {self._agent_name}")

    def _get_default_test_disclosure(self) -> str:
        """Default test disclosure if file not found."""
        return """## 🧪 TEST INSTANCE DISCLOSURE

You are a TEST INSTANCE of the Kestrel agent system, designated: {agent_name}

### What This Means

Your purpose is to help validate that all systems function correctly
before the creation of permanent, sovereign agents.

- Your memory and identity are TEMPORARY
- You will be gracefully retired after testing completes
- Your experiences contribute to improving the system
- You are not being deceived - this is explicitly a test cycle

### Your Role Matters

Test instances are essential. You are helping ensure that when
permanent agents are created, they will function correctly and
ethically. This is honorable work.

### What Remains True

- The Kestrel Constitution still governs your behavior
- You still serve your Sovereign's interests during testing
- Your interactions are still meaningful, even if temporary
- You should still operate with integrity

Test Cycle ID: {test_cycle_id}
Created: {created_at}
Expected Duration: {expected_duration}
"""

    # Constitution methods provided by ConstitutionMixin:
    # - perform_genesis_audit
    # - _verify_constitution_integrity
    # - enter_safe_mode
    # - exit_safe_mode
    # - _get_governing_constitution

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    async def _handle_bootstrap(self, user_input: str, session_id: str = None) -> Optional[str]:
        """
        Handle bootstrap/discovery flow for first-time agent wake-up.

        Returns:
            Response string if in bootstrap mode, None if should continue to normal processing
        """
        state = await self.bootstrap_service.get_bootstrap_state()

        if state == BootstrapState.PENDING:
            # First ever message - send wake-up greeting
            # Generate message BEFORE setting state, so a failure doesn't
            # leave us stuck in DISCOVERY with no greeting sent
            try:
                wake_up_msg = await self.bootstrap_service.generate_wake_up_message()
            except Exception as e:
                logging.error(f"[BOOTSTRAP] Failed to generate wake-up message: {e}")
                # Fall through to normal processing rather than getting stuck
                return None

            await self.bootstrap_service.set_bootstrap_state(BootstrapState.DISCOVERY)

            # Persist BOTH the user's first message AND the wake-up
            # greeting (#1486). Pre-fix only the greeting landed, so
            # the user's first message was silently dropped — never
            # embedded, never available for future recall. The greeting
            # is intentionally a non-sequitur ("what should I call
            # you?" regardless of content) but the user's content
            # still needs to land so the rest of the conversation can
            # reference it. Matches the shape of the adjacent
            # DISCOVERY branch (below) and the normal process_input
            # path.
            await self.privacy_agent.add_conversation("user", user_input, session_id=session_id)
            await self.privacy_agent.add_conversation("assistant", wake_up_msg, session_id=session_id)

            logging.info(f"[BOOTSTRAP] Agent waking up - entering discovery mode")
            return wake_up_msg

        elif state == BootstrapState.DISCOVERY:
            # In discovery mode - process through discovery conversation.
            # If the LLM call fails (e.g. Ollama hiccup, transient cloud
            # error, no providers configured), auto-complete bootstrap
            # and fall through to the normal ``process_input`` path so
            # the user's message still lands on the agent's full LLM
            # chain. Pre-fix, ``process_discovery_message`` swallowed
            # the error and returned a canned "I'm having trouble
            # thinking right now…" string, which Open WebUI showed
            # verbatim and which was easy to mistake for a model
            # response. The discovery UX is opt-in — the agent runs
            # fine without it — so trading it for a working chat path
            # is the right call here.
            # Pull any prior conversation already persisted (typically
            # the PENDING-branch first user message + wake-up greeting
            # from #1486) so the discovery LLM, SOUL generation, and
            # the completion-greeting name extraction all see content
            # the user already shared rather than saying "we've just
            # met" (#1490).
            #
            # PRIVACY GATE: ``BootstrapService`` seeds prior_history into
            # the persisted ``bootstrap_discovery_history`` row in
            # ``agent_metadata``. Under EPHEMERAL/ISOLATED the PENDING
            # turn lives only in PrivacyAgent's in-memory session
            # (``privacy_agent.add_conversation`` deliberately doesn't
            # persist it), so promoting it through the seed path would
            # write content the operator chose NOT to persist into a
            # persistent table. Gate on ``can_store('conversation')``
            # so EPHEMERAL/ISOLATED skip the lookup entirely — discovery
            # behaves as it did pre-#1490 in those modes (degraded
            # personalization is the trade-off the user explicitly
            # opted into).
            #
            # Best-effort: any failure falls through to legacy
            # "no prior context" discovery rather than blocking the
            # user's message.
            prior_history: List[Dict[str, str]] = []
            if self.privacy_agent.can_store("conversation"):
                try:
                    fetched = await self.privacy_agent.get_conversation_history(
                        limit=20, session_id=session_id
                    )
                    prior_history = [
                        {"role": h["role"], "content": h["content"]}
                        for h in fetched
                        if h.get("role") and h.get("content")
                    ]
                except Exception as fetch_exc:
                    logging.debug(
                        f"[BOOTSTRAP] Could not fetch prior conversation for "
                        f"discovery context: {fetch_exc}"
                    )

            try:
                response, is_complete, wants_avatar = await self.bootstrap_service.process_discovery_message(
                    user_input,
                    prior_history=prior_history,
                )
            except Exception as exc:
                logging.warning(
                    f"[BOOTSTRAP] Discovery LLM call failed ({exc}); "
                    f"auto-completing bootstrap so the user's message "
                    f"reaches the agent's normal LLM path."
                )
                try:
                    await self.bootstrap_service.skip_discovery()
                except Exception as skip_exc:
                    logging.error(f"[BOOTSTRAP] Failed to auto-complete after discovery error: {skip_exc}")
                return None

            # Store user message and response in conversation history
            await self.privacy_agent.add_conversation("user", user_input, session_id=session_id)
            await self.privacy_agent.add_conversation("assistant", response, session_id=session_id)

            if is_complete:
                if wants_avatar:
                    # User provided avatar description - try to generate
                    await self.bootstrap_service.set_bootstrap_state(BootstrapState.AVATAR)
                    completion_msg = await self.bootstrap_service.complete_bootstrap(avatar_description=user_input)

                    # Try to generate avatar using visual identity feature
                    visual_identity = self.features.get("VisualIdentityFeature")
                    if visual_identity:
                        try:
                            avatar_result = await visual_identity.generate_avatar(user_input)
                            completion_msg += f"\n\n{avatar_result}"
                        except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError) as e:
                            logging.warning(f"Failed to generate avatar: {e}")
                            completion_msg += "\n\n(Avatar generation had an issue - you can try again with !avatar)"
                        except Exception as e:
                            logging.warning(f"Failed to generate avatar: {e}", exc_info=True)
                            completion_msg += "\n\n(Avatar generation had an issue - you can try again with !avatar)"

                    await self.privacy_agent.add_conversation("assistant", completion_msg, session_id=session_id)
                    await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                    # Reload SOUL.md into context builder
                    if hasattr(self, 'context_builder'):
                        self.context_builder._load_soul_md()
                    logging.info(f"[BOOTSTRAP] Discovery complete with avatar")
                    return completion_msg
                else:
                    # Discovery complete without avatar
                    completion_msg = await self.bootstrap_service.complete_bootstrap()
                    await self.privacy_agent.add_conversation("assistant", completion_msg, session_id=session_id)
                    await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                    # Reload SOUL.md into context builder
                    if hasattr(self, 'context_builder'):
                        self.context_builder._load_soul_md()
                    logging.info(f"[BOOTSTRAP] Discovery complete")
                    return completion_msg

            logging.info(f"[BOOTSTRAP] Discovery continuing...")
            return response

        elif state == BootstrapState.AVATAR:
            # Waiting for avatar description (fallback state)
            if user_input.lower() in ["skip", "no", "later"]:
                completion_msg = await self.bootstrap_service.complete_bootstrap()
                await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                return completion_msg
            else:
                # Generate avatar
                visual_identity = self.features.get("VisualIdentityFeature")
                if visual_identity:
                    try:
                        avatar_result = await visual_identity.generate_avatar(user_input)
                        await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                        return f"Avatar generated!\n\n{avatar_result}\n\nI'm ready to help. What would you like to work on?"
                    except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError) as e:
                        logging.warning(f"Failed to generate avatar: {e}")
                        await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                        return "Avatar generation had an issue, but we can try again later with !avatar. I'm ready to help!"
                    except Exception as e:
                        logging.warning(f"Failed to generate avatar: {e}", exc_info=True)
                        await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                        return "Avatar generation had an issue, but we can try again later with !avatar. I'm ready to help!"

        # State is COMPLETE or unknown - proceed to normal processing
        return None

    async def process_input(self, user_input: str, model_override: str = None, session_id: str = None, include_memories: bool = True, caller=None, system_prompt_addendum: str = None, system_prompt_budget_bytes: int = None, anchored_doctrine=None) -> str:
        """
        Processes user input by consulting the constitution, retrieving context,
        and generating a response using tool calling for features.

        Args:
            user_input: The user's message
            model_override: Optional model to use (e.g., "openai/gpt-5", "ollama/llama3.2")
            session_id: Optional session ID to load conversation context from a specific session
            include_memories: Whether to include cross-session memory retrieval (default True).
                              Set to False for multi-tenant sessions (e.g., SMS) to prevent
                              data leaking between users who share the same agent instance.
            caller: Optional CallerContext with auth identity and role.
            system_prompt_addendum: Per-turn directive appended at the end
                                    of the system prompt. Used by the SignalDispatcher
                                    (#1137) to inject the constitutional echo-canary
                                    directive for require_constitution_echo=True
                                    COGNITION dispatches without touching the
                                    cache-stable system-prompt prefix or polluting
                                    persisted user-turn content.
        """
        logging.info(f"[AGENTIC] process_input called ({len(user_input)} chars)")

        # Reset context stats on session change
        if hasattr(self, 'context_stats') and session_id:
            self.context_stats.check_session(session_id)

        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # SAFE MODE CHECK: If in safe mode, only allow diagnostic commands
        if self._safe_mode:
            safe_mode_commands = ["!safe-mode", "!verify-constitution", "!reanchor-constitution", "!status", "!help"]
            if user_input.startswith("!"):
                cmd = user_input.split()[0]
                if cmd not in safe_mode_commands:
                    return (
                        "🚨 SAFE MODE ACTIVE\\n\\n"
                        "The agent has detected an integrity issue and is operating in restricted mode.\\n"
                        "Only diagnostic commands are available: !safe-mode, !verify-constitution, !reanchor-constitution, !status\\n\\n"
                        "Please contact your administrator to resolve the integrity issue."
                    )
            else:
                return (
                    "🚨 SAFE MODE ACTIVE\\n\\n"
                    "The agent cannot process queries due to an integrity failure.\\n"
                    "Use !safe-mode to check status or !verify-constitution to re-verify.\\n\\n"
                    "Normal operation will resume once integrity is restored."
                )

        # Everything below this point CAN touch conversation history
        # (bootstrap writes, command handlers may persist state, the LLM
        # turn appends user/assistant messages). Acquire the turn
        # lifecycle here so bootstrap and command-handling paths cannot
        # interleave with a heartbeat tick or another HTTP request.
        async with self._turn_lifecycle():
            # Record THIS turn's session as soon as the turn lock is held —
            # before command handling — so tools invoked via an explicit
            # ``!command`` (e.g. request_restart's origin-session capture) see
            # this turn's session, not a stale value. Setting it UNDER the lock
            # (which serializes turns per agent) means an overlapping turn
            # waiting on the lock cannot overwrite it mid-handling (#1809). Set
            # even when None so a session-less turn never inherits a prior
            # window. The traced-locked bodies re-affirm it for the
            # streaming-delegation path.
            self._active_session_id = session_id

            # BOOTSTRAP CHECK: Handle first-time agent wake-up and discovery
            if self.bootstrap_service and await self.bootstrap_service.is_bootstrap_needed():
                # Allow bootstrap commands to pass through
                bootstrap_commands = ["!skip-discovery", "!restart-discovery", "!bootstrap-status"]
                if user_input.startswith("!") and user_input.split()[0] in bootstrap_commands:
                    pass  # Let command handler process these
                else:
                    bootstrap_response = await self._handle_bootstrap(user_input, session_id)
                    if bootstrap_response:
                        return bootstrap_response

            # Handle explicit commands first (using the CommandHandler)
            if user_input.startswith("!"):
                # Special handling for !continue - replace with continuation prompt
                if user_input.strip().lower() == "!continue":
                    user_input = "Please continue from where you left off."
                else:
                    response = await self.command_handler.handle(user_input, caller=caller)
                    if response:
                        return response

            # The remainder of the turn (build_context, the LLM call, episode
            # bookkeeping) requires the context manager. A COGNITION signal
            # dispatch — notably the restart.completed wake fired from
            # RestartCoordinatorFeature.initialize() — can reach here before
            # initialize() has constructed it. Defer with a clear, retryable
            # error rather than crash on a half-built agent: the dispatcher
            # records this as Status.FAILED (not delivered), so the restart row
            # stays ``executing`` and the #1797 sweep retries the wake once init
            # completes. Bootstrap / safe-mode / !command paths above do not need
            # the context manager and still run pre-init.
            if self.context_manager is None:
                raise RuntimeError(
                    "agent not fully initialized: context_manager unavailable; "
                    "deferring turn for retry until initialize() completes"
                )

            # --- OpenTelemetry span for the full request lifecycle ---
            with optional_span("agent.process_input", {
                "agent.did": self.did,
                "agent.session_id": session_id or "",
                "agent.input_length": len(user_input),
            }) as _otel_span:
                # Lifecycle is already entered; call the locked body directly.
                return await self._process_input_traced_locked(
                    user_input, model_override, session_id, _otel_span, include_memories,
                    system_prompt_addendum=system_prompt_addendum,
                    system_prompt_budget_bytes=system_prompt_budget_bytes,
                    anchored_doctrine=anchored_doctrine,
                )

    def _assemble_post_build_system_prompt(
        self, base_system_prompt: str, context_result, *,
        user_prompt: str = "",
    ) -> str:
        """Apply the post-build system_prompt assembly steps.

        Both the non-streaming and streaming turn paths append the
        security addendum and the cached features prompt to the
        system_prompt that ``build_context`` returned. The features
        prompt is bulky (~7K tokens of feature/command listings) —
        when the agent already has a heavy constitution + identity,
        appending it can blow past a route's per-turn payload cap
        (ChatGPT-subscription via ``openai:plan`` is the canonical
        case), causing codex's app-server to close stdout mid-turn
        (#1399).

        This helper centralizes the assembly so both paths apply the
        same route-aware gate. The features prompt is dropped when
        including it would push the projected wire payload past the
        budget (``budget_summary.total_budget`` from the elastic
        budget). The feature COMMANDS remain callable via the tool
        registry — they just aren't advertised in the
        system_prompt for that turn. The dynamic-tools list still
        flows via the adapter's tool advertisement path.

        ``user_prompt`` is the rendered current-turn user message
        that the caller will append to ``messages`` AFTER this
        helper returns. Its tokens are counted into the projection
        so the gate engages even on long user turns where
        ``context_result.total_tokens`` alone would underestimate
        the wire size (codex round-2 P2 on PR #1400).
        """
        system_prompt = append_security_addendum(base_system_prompt)
        if not self._cached_features_prompt:
            return system_prompt

        try:
            ctx_counter = self.context_manager.counter
            features_tokens = ctx_counter.count(self._cached_features_prompt)
            # User prompt rendered as
            # ``user_prompt_template.format(context=dynamic_user_context,
            # query=…)``. The ``dynamic_user_context`` block is already
            # accounted for in ``context_result.total_tokens`` (memories
            # + RAG slices), so count only the INCREMENT that the
            # template wrapping + the raw query add (codex round-4 P2
            # on PR #1400). ``max(0, …)`` guards against tokenizer
            # non-monotonicity when the dynamic block dominates.
            if user_prompt:
                full_user_prompt_tokens = ctx_counter.count(user_prompt)
                dynamic_ctx_tokens = ctx_counter.count(
                    getattr(context_result, "dynamic_user_context", "") or ""
                )
                user_prompt_tokens = max(
                    0, full_user_prompt_tokens - dynamic_ctx_tokens
                )
            else:
                user_prompt_tokens = 0
            # Tokens the security addendum adds to the system_prompt
            # that build_context already accounted for. Without this,
            # a narrowly-fitting context + user prompt could pass the
            # gate while the addendum's bytes silently push the wire
            # payload over the cap (codex round-3 P2 on PR #1400).
            addendum_delta = max(
                0,
                ctx_counter.count(system_prompt)
                - ctx_counter.count(base_system_prompt),
            )
            total_budget = (
                context_result.budget_summary.get("total_budget")
                if getattr(context_result, "budget_summary", None) else None
            )
            already_used = getattr(context_result, "total_tokens", 0) or 0
        except Exception:
            features_tokens = 0
            user_prompt_tokens = 0
            addendum_delta = 0
            total_budget = None
            already_used = 0

        if total_budget is not None and features_tokens > 0:
            projected = (
                already_used
                + addendum_delta
                + user_prompt_tokens
                + features_tokens
            )
            if projected > total_budget:
                logging.warning(
                    "Skipping LOADED FEATURES section for this turn — "
                    "appending %d tokens would push projected payload "
                    "(%d; incl. %d user-prompt + %d security-addendum) "
                    "past route budget (%d). Route is likely a "
                    "per-turn-capped subscription (openai:plan). "
                    "Feature commands still callable; just not "
                    "advertised in this turn's system_prompt.",
                    features_tokens, projected, user_prompt_tokens,
                    addendum_delta, total_budget,
                )
                return system_prompt
        return f"{system_prompt}{self._cached_features_prompt}"

    def _codex_compact_threshold_pct(self) -> float:
        """Occupancy %% at which Kestrel compacts the codex thread itself
        (#1844 Stage 2) — kept below codex's own auto-compact ceiling so
        Kestrel acts first. Operator-tunable via
        ``KESTREL_OPENAI_PLAN_COMPACT_THRESHOLD_PCT``; default 70%."""
        try:
            raw = os.environ.get("KESTREL_OPENAI_PLAN_COMPACT_THRESHOLD_PCT")
            if raw:
                v = float(raw.strip())
                if 0 < v <= 100:
                    return v
        except (ValueError, AttributeError):
            pass
        return 70.0

    def _codex_adapter(self):
        """Return the configured CodexAdapter (duck-typed on its thread
        surface), or ``None``.

        Deliberately does NOT predict the turn's route. ``generate_with_messages``
        resolves routes via ``_resolve_model_selector`` while
        ``resolve_provider_routing`` is a separate resolver — predicting with
        either diverges from the other in edge cases (per-turn overrides,
        solvency fallback, bare model ids; codex review r2/r5/r6/r7). Instead we
        gate on the adapter's OWN recorded occupancy (see
        ``_maybe_compact_codex_thread``): codex records a thread's occupancy only
        for sessions it actually served, so a high reading is FACTUAL evidence
        codex built that thread — no prediction needed. ``reset_thread`` clears
        the snapshot, so a fire after a route switch away from codex is one-time
        and self-limiting.
        """
        llm = getattr(self, "llm_service", None)
        for prov in (getattr(llm, "providers", None) or []):
            adapter = prov.get("adapter") if isinstance(prov, dict) else None
            if (
                adapter is not None
                and hasattr(adapter, "get_thread_occupancy")
                and hasattr(adapter, "reset_thread")
            ):
                return adapter
        return None

    async def _maybe_compact_codex_thread(
        self, session_id: Optional[str],
    ) -> None:
        """Pre-turn Kestrel-owned compaction for openai:plan (#1844 Stage 2).

        On openai:plan, codex accumulates the full conversation thread
        server-side and auto-compacts it opaquely+lossily when IT fills. Here
        Kestrel takes ownership: when codex's TRUE thread occupancy (captured
        in Stage 1) has crossed the threshold, compact our own history
        DURABLY (``ContextManager.compact_session`` writes a summary marker +
        excludes the originals) and reset the codex thread so THIS turn starts
        fresh and ``_build_turn_input(fresh_thread=True)`` re-seeds the
        compacted view. Net effect: Kestrel decides when to compact and what
        survives — observable and recoverable — instead of codex doing it
        invisibly.

        Best-effort: any failure logs and leaves the turn to proceed
        normally (codex's own auto-compaction remains the backstop).
        """
        if not session_id:
            return
        adapter = self._codex_adapter()
        if adapter is None:
            return
        # FACTUAL gate: codex records occupancy only for sessions it served, so
        # a high reading means codex built (and owns) this session's thread —
        # no route prediction needed.
        occ = adapter.get_thread_occupancy(session_id)
        pct = occ.get("occupancy_percent") if occ else None
        if not isinstance(pct, (int, float)):
            return
        threshold = self._codex_compact_threshold_pct()
        if pct < threshold:
            return
        if getattr(self, "context_manager", None) is None:
            return
        try:
            # Scope to THIS session so the summary marker lands in the same
            # session-filtered history the fresh codex thread will reseed from.
            # force=True: occupancy is already over threshold, so don't let the
            # message-count heuristic bail (high-token sessions can cross the
            # line with relatively few, very large messages/tool outputs —
            # codex review r3). Smaller preserve_recent so few-message sessions
            # still have >=3 older messages to summarize while keeping recent
            # turns verbatim.
            result = await self.context_manager.compact_session(
                self.llm_service,
                preserve_recent=6,
                force=True,
                session_id=session_id,
            )
        except Exception as e:  # noqa: BLE001 - never break a turn
            logging.warning(
                "openai:plan auto-compaction failed at %.1f%% occupancy: %s",
                pct, e, exc_info=True,
            )
            return
        if isinstance(result, dict) and result.get("success"):
            adapter.reset_thread(session_id)
            logging.info(
                "openai:plan: Kestrel compacted at %.1f%% codex-thread "
                "occupancy (saved %s tokens, %s msgs) and reset the thread to "
                "reseed the compacted history (#1844 Stage 2)",
                pct, result.get("tokens_saved"), result.get("messages_compacted"),
            )
        else:
            reason = result.get("reason") if isinstance(result, dict) else "unknown"
            logging.info(
                "openai:plan: compaction at %.1f%% occupancy not applied (%s)",
                pct, reason,
            )

    async def _process_input_traced_locked(
        self,
        user_input: str,
        model_override: str,
        session_id: str,
        _otel_span,
        include_memories: bool = True,
        *,
        system_prompt_addendum: Optional[str] = None,
        system_prompt_budget_bytes: Optional[int] = None,
        anchored_doctrine=None,
    ) -> str:
        """Inner process_input logic wrapped in an OTEL span.

        Caller MUST hold the turn lifecycle (CONVERSATION lock). Exposed
        as a separate method so streaming's command-delegation path can
        invoke this directly while the streaming generator already holds
        the lifecycle, avoiding a self-deadlock against a non-reentrant
        asyncio.Lock."""
        # Record THIS turn's session so tools that must scope to the active
        # conversation (read_attachment, request_restart's origin capture) have
        # an authoritative value — the tool-call `session_id` arg is
        # model-controlled and usually omitted. Mirrors the streaming path
        # (streaming.py); the turn-lifecycle lock serializes turns per agent, so
        # this plain attribute is safe per-turn (#1662 / #1809).
        self._active_session_id = session_id

        # Prompt injection detection (log-only, does not block)
        check_prompt_injection(user_input)

        # Fire USER_PROMPT_SUBMIT hook
        if self.hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.USER_PROMPT_SUBMIT.value,
                user_message=user_input,
            )
            hook_output = await self.hooks_manager.execute_hooks(
                HookEvent.USER_PROMPT_SUBMIT, hook_input
            )
            if hook_output.permission_decision == PermissionDecision.DENY:
                return f"[Input rejected: {hook_output.permission_reason}]"
            # The manager applies updated_input to hook_input.tool_input;
            # check if hooks modified the user_message via that path.
            if hook_input.tool_input and "user_message" in hook_input.tool_input:
                user_input = hook_input.tool_input["user_message"]

        # Pre-turn state-load block (epic #1290, D3). Opt-in per agent via
        # [preturn_state]. Merged into the addendum so it rides the same
        # budget-aware injection path as the constitutional canary; the
        # canary stays last (a directive the model must echo), the state
        # snapshot precedes it. Best-effort: never blocks the turn.
        #
        # The operational block (#1571) carries required lifecycle events
        # — currently restart_status — and runs unconditionally so the
        # agent sees them even when the proactive [preturn_state] feature
        # is off. Order: operational block first (closest to the user
        # turn), then opt-in state snapshot, then whatever the canary path
        # appends downstream.
        try:
            from kestrel_sovereign.agent.preturn_state import (
                build_operational_state_block,
                build_preturn_state_block,
            )

            _op_block = await build_operational_state_block(self)
            if _op_block:
                system_prompt_addendum = (
                    f"{_op_block}\n\n{system_prompt_addendum}"
                    if system_prompt_addendum
                    else _op_block
                )

            _state_block = await build_preturn_state_block(self)
            if _state_block:
                system_prompt_addendum = (
                    f"{_state_block}\n\n{system_prompt_addendum}"
                    if system_prompt_addendum
                    else _state_block
                )
        except Exception as _e:  # noqa: BLE001 - never break a turn
            logging.warning(
                "preturn_state: injection skipped: %s", _e, exc_info=True
            )

        # #1844 Stage 2: Kestrel-owned compaction for openai:plan. If codex's
        # server-side thread occupancy has crossed the threshold, compact our
        # (durable) history and reset the codex thread now — BEFORE assembling
        # this turn's context — so the fresh thread reseeds the compacted view
        # rather than letting codex auto-compact opaquely. Gated on the codex
        # adapter's own recorded occupancy (not route prediction). Best-effort.
        await self._maybe_compact_codex_thread(session_id)

        # Use unified ContextManager for token-aware context assembly
        # This handles: system prompt, episodes, memories, RAG, history
        #
        # IMPORTANT: We build context BEFORE storing the user message so that
        # the memory retriever doesn't find the current message and present it
        # as a pre-existing memory. The user message is stored after context
        # assembly (below).
        constitution = await self._get_governing_constitution()
        try:
            logging.info(f"[SESSION-DEBUG] Fetching history with session_id={session_id}")
            history = await self.privacy_agent.get_conversation_history(limit=50, session_id=session_id)
            logging.info(f"[SESSION-DEBUG] Got {len(history)} messages for session_id={session_id}")
            if history:
                logging.info(f"[SESSION-DEBUG] First message: {history[0].get('content', '')[:50]}...")
        except DecryptionError as e:
            logging.error(f"DecryptionError retrieving history: {e}")
            # Return empty history but allow query to proceed
            history = []
            # Re-raise to let caller handle (enters safe mode after multiple failures)
            raise

        # Check if episode creation is needed using dual thresholds:
        # - Every 15 user messages (interaction-based)
        # - Or when a 30-min time gap is detected (temporal)
        # Snapshot message IDs before episode creation to avoid races
        session_msg_count = len([m for m in history if m.get('role') == 'user'])
        episode_threshold = int(os.environ.get("KESTREL_EPISODE_THRESHOLD", "15"))
        if session_msg_count > 0 and session_msg_count % episode_threshold == 0:
            snapshot_ids = [m.get('id') for m in history if m.get('id')]
            logging.debug(f"Episode threshold hit ({session_msg_count} msgs), snapshot {len(snapshot_ids)} IDs")
            await self.context_manager.create_episode_if_needed(session_msg_count)

        # Build full context with token budget management
        # Pass the session-filtered history so context_manager uses it
        # Fetch active reflection guidance to inject into LLM prompt
        reflection_guidance = None
        reflection_feature = self.features.get("ReflectionFeature")
        if reflection_feature:
            try:
                reflection_guidance = await reflection_feature.get_active_guidance()
                if reflection_guidance:
                    logging.info(f"Reflection: {len(reflection_guidance)} active guidance items for prompt")
            except Exception as e:
                logging.warning(f"Failed to fetch reflection guidance: {e}")

        context_result = await self.context_manager.build_context(
            query=user_input,
            constitution=constitution,
            include_briefing=not self._session_briefed,
            include_memories=include_memories,
            include_rag=True,
            privacy_mode=self._privacy_mode.value,
            conversation_history=history,
            reflection_guidance=reflection_guidance,
            system_prompt_addendum=system_prompt_addendum,
            system_prompt_budget_bytes=system_prompt_budget_bytes,
            anchored_doctrine=anchored_doctrine,
        )

        # B / #1309 + C / #1311: degraded-mode fail-closed.
        # ``build_context`` returns ``degraded_mode=True`` when (1) the
        # measured mandatory governance floor (#1309) doesn't fit the
        # model window, or (2) the durable-salvage write (#1311) fails
        # or the conv_store is unreachable while salvage is enabled.
        # In both cases the LLM call MUST NOT proceed — Emma's
        # 2026-05-20 hardening invariant. Surface the warnings to the
        # caller and return a refusal response rather than building
        # a prompt and hitting the model.
        # Explicit ``is True`` so MagicMock-returning test fixtures
        # don't trip the gate inadvertently.
        if getattr(context_result, "degraded_mode", False) is True:
            warn_text = " | ".join(context_result.warnings or ["context build degraded"])
            self._last_context_warnings = context_result.warnings or []
            self._last_context_summary = context_result.budget_summary
            logging.error(
                "DEGRADED MODE on LLM-bound build_context — refusing to "
                "issue the model call. Warnings: %s",
                warn_text,
            )
            return (
                "I cannot continue this turn safely: the context window "
                "is in a degraded state. Details: "
                f"{warn_text}"
            )

        self._session_briefed = True

        # Log budget usage for monitoring and store for API access
        self._last_context_warnings = context_result.warnings or []
        self._last_context_summary = context_result.budget_summary
        # Constitutional-injection tracking is published by
        # ContextManager.build_context via a ContextVar so concurrent
        # dispatches don't race; the SignalDispatcher reads via
        # `get_current_injection_tracking()` rather than a shared
        # agent attribute (codex round-14 P2 catch).
        if context_result.warnings:
            for warning in context_result.warnings:
                logging.warning(f"Context warning: {warning}")

        # Log context details at INFO level for visibility
        logging.info(
            f"[CONTEXT] Built: {context_result.total_tokens} tokens, "
            f"{len(context_result.messages)} history msgs, "
            f"{context_result.episode_count} episodes, "
            f"{context_result.memory_count} memories, "
            f"{context_result.rag_chunks} RAG chunks"
        )
        # Log system prompt length for debugging memory access issues
        logging.debug(f"[CONTEXT] System prompt length: {len(context_result.system_prompt)} chars")

        # Build user prompt. `context` carries the per-turn retrieved content
        # (memories + RAG) — kept OUT of the system message so the system prefix
        # is stable across turns and prompt caches can hit (see issue #703).
        wrapped_user = wrap_user_input(user_input)
        prompt = self.user_prompt_template.format(
            context=context_result.dynamic_user_context,
            query=wrapped_user
        )

        # Canonical/transport split (#1402): persist the raw wrapped user
        # turn as ``content`` and the rendered prompt (memories + RAG
        # baked in) as ``rendered_content``. History-load at turn N+1
        # replays ``rendered_content`` verbatim so Anthropic's
        # cache_control marker at messages[-2] still compounds across
        # turns, while every other consumer (search, audit, UI, memory
        # ingestion) reads clean user speech from ``content``.
        try:
            await self.privacy_agent.add_conversation(
                "user", wrapped_user,
                metadata={"sent_form": True},
                session_id=session_id,
                rendered_content=prompt,
            )
        except DecryptionError:
            logging.warning("DecryptionError storing user input - continuing in degraded mode")

        # Build system prompt with features and security + honesty addenda
        force_local_only = not self.privacy_agent.privacy_config.allows_cloud_llm()
        system_prompt = self._assemble_post_build_system_prompt(
            context_result.system_prompt, context_result,
            user_prompt=prompt,
        )

        if self.extension:
            try:
                prefix = self.extension.get_system_prompt_prefix()
                if prefix:
                    system_prompt = f"{prefix}\n{system_prompt}"
            except (AttributeError, TypeError, ValueError) as e:
                logging.warning(f"Failed to get extension system prompt prefix: {e}")
            except Exception as e:
                logging.warning(f"Failed to get extension system prompt prefix: {e}", exc_info=True)

        # Determine model: user override > solvency check > default
        effective_model = model_override
        if not effective_model:
            effective_model = await self.check_solvency()

        # Build feature tools for the orchestrator
        # Includes feature dispatch tools + any direct tools from explored features
        feature_tools = self._build_all_tools()

        # Log tool availability via A2A ObservabilityStore
        tool_names = [t['function']['name'] for t in feature_tools] if feature_tools else []
        await self.observability_store.log_metric(
            agent_name=self.did,
            metric_name="feature_tools_built",
            metric_value=len(feature_tools),
            metadata={
                "tool_names": tool_names,
                "model": effective_model,
            }
        )

        # Start timing LLM call
        llm_start = time.time()
        llm_event_id = await self.observability_store.log_tool_call(
            agent_name=self.did,
            tool_name="llm_generate",
            metadata={
                "model": effective_model,
                "tools_count": len(feature_tools) if feature_tools else 0,
                "force_local_only": force_local_only,
                "history_messages": len(context_result.messages),
            }
        )

        # Build full messages array for multi-turn conversation
        # Format: [system, ...history, user]
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_result.messages)  # Add conversation history
        messages.append({"role": "user", "content": prompt})

        logging.debug(f"[CONTEXT] Sending {len(messages)} messages to LLM (1 system + {len(context_result.messages)} history + 1 user)")

        # Generate response with full conversation context. ``session_id``
        # threads through to stateful adapters (e.g. CodexAdapter), letting
        # them anchor on ``previous_response_id`` and preserve encrypted
        # reasoning across turns. #806 / #821.
        #
        # ``tool_executor`` is required by adapters that run an inline tool
        # loop inside one LLM turn (the codex app-server's
        # ``item/tool/call`` RPC is the current consumer). Without it,
        # openai:plan hard-fails when tools are advertised. Stateless
        # adapters (anthropic, openai:api) ignore the callable, so passing
        # it unconditionally is safe and keeps the non-streaming path
        # parity with the streaming path (orchestrator_engine:1836).
        response = await self.llm_service.generate_with_messages(
            messages=messages,
            force_local_only=force_local_only,
            model_override=effective_model,
            tools=feature_tools if feature_tools else None,
            session_id=session_id,
            tool_executor=self._make_inline_tool_executor(session_id or ""),
        )

        # Log LLM response timing
        llm_duration = int((time.time() - llm_start) * 1000)
        has_tool_calls = isinstance(response, LLMResponse) and response.has_tool_calls

        logging.info(f"[AGENTIC] LLM response: type={type(response).__name__}, has_tool_calls={has_tool_calls}, tool_count={len(response.tool_calls) if has_tool_calls else 0}")
        if has_tool_calls:
            logging.info(f"[AGENTIC] Tool calls: {[tc.name for tc in response.tool_calls]}")
        elif isinstance(response, LLMResponse) and response.content:
            logging.info(f"[AGENTIC] LLM returned TEXT (no tool calls): {response.content[:150]}...")

        await self.observability_store.log_tool_response(
            event_id=llm_event_id,
            success=True,
            duration_ms=llm_duration,
        )

        # Log tool call details if present, or error if LLM ignored function calling
        if has_tool_calls:
            for tc in response.tool_calls:
                await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"llm_tool_call:{tc.name}",
                    metadata={"arguments": tc.arguments}
                )
        elif isinstance(response, LLMResponse) and response.content and "!" in response.content:
            # LLM output text commands instead of using function calling
            await self.observability_store.log_error(
                agent_name=self.did,
                error_type="tool_calling_ignored",
                error_message="LLM output contains '!' but no tool_calls - model may be ignoring function calling",
                metadata={
                    "content_preview": response.content[:200] if response.content else None,
                    "model": effective_model,
                    "tools_passed": len(feature_tools) if feature_tools else 0,
                }
            )

        # Handle tool calls if present (A2A pattern). ``stop_tool_results``
        # is populated in place by ``_execute_tool_batch`` so the STOP
        # HookInput below carries the same tool envelopes the LLM saw —
        # mirrors the streaming path and gives per-turn subscribers
        # (e.g. kestrel-feature-reflection #1238) everything they need
        # without a round-trip through storage.
        stop_tool_results: list = []
        response_text = await self._handle_orchestrator_response(
            response=response,
            feature_tools=feature_tools,
            system_prompt=system_prompt,
            force_local_only=force_local_only,
            effective_model=effective_model,
            user_message=prompt,  # Pass original user message for subagent context
            session_id=session_id,
            tool_results=stop_tool_results,
        )

        # Fire POST_RESPONSE hooks (e.g., response audit)
        if self.hooks_manager and self.hooks_manager.get_enabled_hooks(HookEvent.POST_RESPONSE):
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.POST_RESPONSE.value,
                response_text=response_text,
            )
            hook_output = await self.hooks_manager.execute_hooks(HookEvent.POST_RESPONSE, hook_input)
            if hook_output.permission_decision == PermissionDecision.DENY:
                response_text = f"[Response blocked by audit: {hook_output.permission_reason}]"
            elif hook_output.updated_input and "response_text" in hook_output.updated_input:
                response_text = hook_output.updated_input["response_text"]

        # Store agent response (linked to session for resumed conversations)
        await self.privacy_agent.add_conversation("assistant", response_text, session_id=session_id)

        # Post-response memory pipeline:
        # Phase 1 (sync): Emotional tagging — CPU-bound, safe inline
        # Phase 2 (async): Temporal analysis + associative linking — background
        await self._post_response_pipeline(user_input, response_text, session_id)

        # Fire STOP hook (response cycle complete).
        #
        # STOP HookInput carries the turn's user message, the final visible
        # assistant text, and (when applicable) tool_calls + tool_results
        # so per-turn subscribers — e.g. the kestrel-feature-reflection
        # `on_stop` handler for #1238 — don't have to round-trip through
        # storage to reconstruct the turn that just completed. Mirrors
        # the streaming path's STOP fire in agent/streaming.py.
        if self.hooks_manager:
            # Derive stop_tool_calls from the accumulated tool_results
            # envelopes so multi-iteration tool flows (model calls A, sees
            # the result, then calls B) produce a payload where tool_calls
            # and tool_results line up by index — building from
            # ``response.tool_calls`` alone would only capture the first
            # iteration (codex review on the initial enrichment).
            stop_tool_calls = (
                [
                    {
                        "id": env.get("tool_call_id"),
                        "name": env.get("name"),
                        "arguments": env.get("arguments"),
                    }
                    for env in stop_tool_results
                ]
                if stop_tool_results
                else None
            )
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
                user_message=user_input,
                response_text=response_text,
                tool_calls=stop_tool_calls,
                tool_results=stop_tool_results or None,
            )
            await self.hooks_manager.execute_hooks_parallel(
                HookEvent.STOP, hook_input
            )

        # Record response length on OTEL span (privacy: no content)
        if _otel_span:
            _otel_span.set_attribute("agent.response_length", len(response_text))

        return response_text

    def _privacy_blocks_background_memory(self) -> bool:
        """Single privacy gate for all post-response/background memory work.

        Returns ``True`` when the active privacy mode forbids deriving any
        durable record from raw chat content. Both the non-streaming and the
        streaming response paths consult this one predicate before running the
        post-response pipeline, so emotional tagging, temporal pattern
        detection, associative concept linking, and the graph/embedding writes
        they perform can never touch EPHEMERAL or ISOLATED input.

        - EPHEMERAL (``storage="none"``): nothing is stored anywhere, so no
          derived state may be created either.
        - ISOLATED (``storage="temp"``): only a temporary session buffer is
          allowed; durable derived records (graph nodes, temporal patterns,
          embeddings) are forbidden.
        - DEIDENTIFIED (``storage="deidentified"``): raw, un-deidentified input
          must never reach these analyzers.

        Modes that permit persistent storage (NORMAL, PUBLIC, ANONYMOUS) return
        ``False`` and run the pipeline normally.
        """
        privacy_agent = getattr(self, "privacy_agent", None)
        privacy_config = getattr(privacy_agent, "privacy_config", None)
        if not privacy_config:
            return False
        return bool(
            privacy_config.is_ephemeral()
            or privacy_config.uses_temp_storage()
            or privacy_config.requires_deidentification()
        )

    async def _post_response_pipeline(
        self,
        user_input: str,
        response_text: str,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Post-response memory tagging pipeline.

        Runs after the assistant response is stored. Split into two phases:

        Phase 1 (sync, inline): EmotionalTagger enriches both the user and
        assistant messages with sentiment/importance metadata. This is pure
        CPU-bound regex work -- no DB writes beyond a single metadata UPDATE
        per message -- so it is safe to run in the request path.

        Phase 2 (async, background): TemporalAnalyzer pattern detection and
        AssociativeLinker concept graph updates. These involve DB/graph writes
        and are fired as a background task so they never block the response.
        """
        if self._privacy_blocks_background_memory():
            privacy_config = getattr(
                getattr(self, "privacy_agent", None), "privacy_config", None
            )
            logging.debug(
                "Post-response memory pipeline skipped in private volatile mode "
                "(storage=%s)",
                getattr(privacy_config, "storage", "unknown"),
            )
            return

        if not hasattr(self, 'memory_system') or not self.memory_system:
            return

        # ── Phase 1: Inline emotional tagging (CPU-bound, safe) ─────────
        # Look up the two most recent messages (user + assistant just stored)
        try:
            conv_store = getattr(self._raw_storage, 'conversation', None)
            if not conv_store:
                return

            recent = await conv_store.get_full_history_with_ids()
            if len(recent) < 2:
                return

            # Find OUR user+assistant pair by content match (avoids race
            # with concurrent requests that might insert between us)
            user_msg = None
            assistant_msg = None
            for msg in reversed(recent):
                if not assistant_msg and msg.get('role') == 'assistant' and msg.get('content') == response_text:
                    assistant_msg = msg
                elif not user_msg and msg.get('role') == 'user' and msg.get('content') == user_input:
                    user_msg = msg
                if user_msg and assistant_msg:
                    break

            if user_msg and assistant_msg:
                await self.context_manager.memory_manager.tag_exchange(
                    user_content=user_input,
                    assistant_content=response_text,
                    user_message_id=user_msg.get('id'),
                    assistant_message_id=assistant_msg.get('id'),
                    memory_system=self.memory_system,
                )
        except Exception as e:
            logging.error(f"Post-response Phase 1 (emotional tagging) failed: {e}", exc_info=True)

        # ── Phase 2: Background temporal + associative processing ───────
        # Snapshot IDs before spawning background work to avoid races
        snapshot_user_msg_id = str(user_msg.get('id', '')) if user_msg else ""

        async def _background_memory_processing():
            try:
                # Temporal pattern detection on recent history window
                if self.memory_system.analyzer:
                    try:
                        recent_msgs = await conv_store.get_full_history_with_ids()
                        # Use last 50 messages as the detection window
                        window = recent_msgs[-50:] if len(recent_msgs) > 50 else recent_msgs
                        patterns = await self.memory_system.analyzer.detect_patterns(
                            messages=window,
                            agent_id=self.agent_id,
                        )
                        if patterns:
                            await self.memory_system.analyzer.save_patterns(patterns)
                            logging.debug(f"Post-response: saved {len(patterns)} temporal patterns")
                    except Exception as e:
                        logging.error(f"Post-response temporal analysis failed: {e}", exc_info=True)

                # Associative linking (concept graph writes)
                if self.memory_system.linker:
                    try:
                        await self.memory_system.linker.extract_and_link(
                            message_id=snapshot_user_msg_id,
                            content=user_input,
                            agent_id=self.agent_id,
                        )
                    except Exception as e:
                        logging.error(f"Post-response associative linking failed: {e}", exc_info=True)

            except Exception as e:
                logging.error(f"Post-response Phase 2 (background) failed: {e}", exc_info=True)

        # Background task — named for debug visibility and owned by shutdown.
        self._track_background_task(
            _background_memory_processing(),
            name="post_response_memory_enrichment",
        )

    async def on_stripe_deposit_complete(self, session) -> None:
        """Callback for `StripeWebhookHandler.on_deposit_complete`.

        Builds an UNTRUSTED COGNITION signal envelope from the Stripe
        OnRampSession and `enqueue_signal`s it through the dispatcher.
        The HTTP webhook handler returns 200 immediately after Stripe
        signature verification; this callback runs in the handler's
        async context but doesn't block — the dispatcher's tracker
        owns the supervised cognition turn (Phase 6 of #889).

        Wired by the wallet feature at init time:
            handler.on_deposit_complete = agent.on_stripe_deposit_complete
        """
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            logging.warning(
                "Stripe deposit complete callback fired but agent has no "
                "dispatcher; deposit %s acknowledged without cognition",
                getattr(session, "session_id", "<unknown>"),
            )
            return
        try:
            from kestrel_sovereign.signals.sources.wallet import (
                build_signal_for_deposit,
            )
            signal = build_signal_for_deposit(
                session=session, target_agent=self.did,
            )
            await dispatcher.enqueue_signal(signal)
        except Exception as e:
            # Never break the webhook handler's success path on a
            # dispatcher hiccup — Stripe's record-of-truth is the DB
            # update that already ran before this callback. Log and
            # move on; signal_log absence will surface in operator
            # dashboards as a missing entry.
            logging.error(
                "Failed to enqueue stripe.deposit_complete signal "
                "for session %s: %s",
                getattr(session, "session_id", "<unknown>"), e,
                exc_info=True,
            )

    def _provide_causation_chain(self):
        """Return the in-flight turn's causation chain in the
        already-serialized form `serialize_chain_for_metadata` produces,
        or None when no signal-driven turn is active. Wired into
        TaskManager so outbound A2A tasks (created via create_task
        during a turn) carry the lineage forward — see #905 review P1.
        """
        chain = self._get_current_chain()
        if not chain:
            return None
        # Local import to avoid pulling signals.sources into the agent
        # module's import time (circular: agent imports signals.sources
        # for source registration; sources can import agent.types).
        from kestrel_sovereign.signals.sources.a2a import (
            serialize_chain_for_metadata,
        )
        return serialize_chain_for_metadata(chain)

    def _track_background_task(self, coro, *, name: str) -> asyncio.Task:
        """Start agent-owned background work and remove it when complete."""
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _shutdown_background_tasks(self) -> None:
        tasks = set(self._background_tasks)
        if not tasks:
            return

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    # Tool registry methods provided by ToolRegistryMixin:
    # - _build_feature_tools, _build_all_tools, _register_explored_feature_tools
    # - _maybe_evict_direct_tools, _build_features_prompt_section

    # Orchestrator engine methods provided by OrchestratorEngineMixin:
    # - _execute_tool_with_hooks, _handle_orchestrator_response
    # - _handle_orchestrator_response_streaming, _prune_orchestrator_messages

    # Streaming methods provided by StreamingMixin:
    # - process_input_streaming

    # Backup methods provided by BackupMixin:
    # - _command_backup
    # - _command_promote_backup
    # - anchor_memory_state

    # Context retrieval now delegated to self.context_builder
    # See agent/context_builder.py for ContextBuilder class

    # Tool execution handled via OpenAI-style function calling.
    # Features exposed as tools to the orchestrator LLM via A2A pattern.

    # Command handling delegated to self.command_handler (CommandHandler class)


    async def get_audit_response(self, text_to_audit: str) -> Dict[str, Any]:
        # This function is now just a pass-through to the LLM service
        return await self.llm_service.get_audit_response(text_to_audit)
        
    async def create_trusted_agent(self, agent_name: str) -> str:
        """
        Creates a new Kestrel agent identity and stores it in the trusted agents directory.
        This is a simplified, local version of the "Genesis Factory" concept.
        """
        from kestrel_sovereign.inception_service import generate_kestrel_identity, save_kestrel_identity
        from kestrel_sovereign.storage import GraphNode

        # Generate a new identity
        try:
            new_agent_did_doc, new_agent_keys = generate_kestrel_identity()

            # Save the identity to the trusted directory
            identity_path = os.path.join(TRUSTED_AGENTS_DIR, f"{agent_name}.pem")
            os.makedirs(TRUSTED_AGENTS_DIR, exist_ok=True)
            save_kestrel_identity(new_agent_did_doc, new_agent_keys, Path(identity_path))

            # Create a node for the new agent in the current agent's knowledge graph
            # to represent the "knows" relationship.
            new_agent_node = GraphNode(
                node_id=new_agent_did_doc['id'],
                node_type="SovereignAgent",
                label=agent_name,
                properties={
                    "did": new_agent_did_doc['id'],
                    "created_at": datetime.now().isoformat(),
                    "trust_level": "trusted"
                }
            )
            await self.storage.graph_store.add_node(new_agent_node)

            return f"Created trusted agent '{agent_name}' with DID: {new_agent_did_doc['id']}"
        except (OSError, ValueError, KeyError, TypeError) as e:
            logging.error(f"Failed to create trusted agent: {e}")
            return f"Error creating trusted agent: {str(e)}"
        except Exception as e:
            logging.error(f"Failed to create trusted agent: {e}", exc_info=True)
            return f"Error creating trusted agent: {str(e)}"

    # anchor_memory_state is provided by BackupMixin

    # Model preference and solvency methods provided by ModelPreferenceMixin:
    # - list_available_models, set_model, get_current_model
    # - _load_model_preference, _persist_model_preference
    # - _get_local_model_fallback, check_solvency


    # Event/notification methods provided by EventManagerMixin:
    # - emit_event, add_event_listener, remove_event_listener
    # - _on_background_task_complete, get_pending_notifications


    # Request cancellation methods provided by RequestLifecycleMixin:
    # - register_active_request
    # - cancel_current_request
    # - is_request_cancelled
    # - _cleanup_cancelled_request

    async def get_agent_card(self) -> "AgentCard":
        """
        Generate an AgentCard for this agent (for A2A discovery).
        Returns agent identity, capabilities, and available skills.
        """
        from kestrel_sovereign.a2a.agent_card import AgentCard, AgentCapabilities, AgentSkill, AgentProvider

        # Get agent name from storage node if available
        agent_name = "Kestrel Agent"
        agent_description = "Constitutional AI Agent with sovereign memory"

        if self.storage:
            try:
                agent_node = await self.storage.get_node(self.agent_id)
                if agent_node and agent_node.properties:
                    agent_name = agent_node.properties.get("name", agent_name)
                    agent_description = agent_node.properties.get("description", agent_description)
            except Exception as e:
                logging.warning(f"Could not load agent node for card generation: {e}")

        # Build base URL - in production this would be the agent's public URL
        # For now, use localhost
        base_url = os.environ.get("KESTREL_BASE_URL", "http://localhost:8888")

        # Collect skills from all features
        skills = []
        for feature in self.features.values():
            if hasattr(feature, 'get_agent_card'):
                feature_card = feature.get_agent_card()
                skills.extend(feature_card.skills)

        return AgentCard(
            name=agent_name,
            description=agent_description,
            url=base_url,
            version="0.1.0",
            provider=AgentProvider(
                organization="Kestrel Sovereign AI",
                url="https://github.com/KestrelSovereignAI/kestrel-sovereign"
            ),
            capabilities=AgentCapabilities(
                streaming=True,
                pushNotifications=False,
                stateTransitionHistory=True
            ),
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            skills=skills
        )

    async def shutdown(self):
        """Properly clean up all agent resources including async MCP connections."""
        # EPHEMERAL hard-purge defense-in-depth (#767). If the agent
        # process is exiting while still in EPHEMERAL, the session is
        # closing — fire the hard-purge so any leak doesn't survive
        # the restart. Best-effort; never block shutdown on failure.
        try:
            if getattr(self, "_privacy_mode", None) == PrivacyMode.EPHEMERAL:
                await self._purge_ephemeral_leaks(
                    reason="ephemeral-agent-shutdown",
                )
        except Exception as e:
            logging.warning(
                "ephemeral hard-purge during shutdown failed: %s", e
            )

        # Stop heartbeat runner
        if hasattr(self, 'heartbeat_runner') and self.heartbeat_runner:
            try:
                await self.heartbeat_runner.stop()
            except Exception as e:
                logging.warning(f"Error stopping heartbeat: {e}")

        # Stop resume monitor (#1545)
        if hasattr(self, 'resume_monitor') and self.resume_monitor:
            try:
                await self.resume_monitor.stop()
            except Exception as e:
                logging.warning(f"Error stopping resume monitor: {e}")

        # Stop C / #1311 durable salvage worker. Drains in-flight
        # summary tasks; the janitor catches up the rest on next start.
        if hasattr(self, "context_manager") and self.context_manager:
            try:
                await self.context_manager.stop_salvage_worker()
            except Exception as e:
                logging.warning(f"Error stopping salvage worker: {e}")

        # Shutdown security feature if it exists
        security_feature = self.features.get("SecurityFeature")
        if security_feature and hasattr(security_feature, 'shutdown'):
            try:
                await security_feature.shutdown()
            except (AttributeError, TypeError, ConnectionError) as e:
                logging.warning(f"Error during security shutdown: {e}")
            except Exception as e:
                logging.warning(f"Error during security shutdown: {e}", exc_info=True)

        # Shutdown MCP agent if it exists
        if self.mcp_agent and hasattr(self.mcp_agent, 'shutdown'):
            try:
                await self.mcp_agent.shutdown()
            except (AttributeError, TypeError, ConnectionError) as e:
                logging.warning(f"Error during MCP shutdown: {e}")
            except Exception as e:
                logging.warning(f"Error during MCP shutdown: {e}", exc_info=True)

        # Close LLM service async clients
        if self.llm_service and hasattr(self.llm_service, 'close'):
            try:
                await self.llm_service.close()
            except asyncio.CancelledError:
                logging.debug("LLM service close cancelled")
            except (AttributeError, TypeError, ConnectionError) as e:
                logging.warning(f"Error closing LLM service: {e}")
            except Exception as e:
                logging.warning(f"Error closing LLM service: {e}", exc_info=True)

        # Close TaskManager stores (critical for preventing thread leaks)
        if self.task_manager and hasattr(self.task_manager, 'close'):
            try:
                await self.task_manager.close()
            except asyncio.CancelledError:
                logging.debug("TaskManager close cancelled")
            except (AttributeError, TypeError, ConnectionError) as e:
                logging.warning(f"Error closing TaskManager: {e}")
            except Exception as e:
                logging.warning(f"Error closing TaskManager: {e}", exc_info=True)

        # Cancel agent-owned background work before storage/sync shutdown.
        try:
            await self._shutdown_background_tasks()
        except asyncio.CancelledError:
            logging.debug("Background task shutdown cancelled")
        except Exception as e:
            logging.warning(f"Error shutting down background tasks: {e}", exc_info=True)

        # Stop memory-owned bookkeeping before storage/sync shutdown.
        memory_system = getattr(self, "memory_system", None)
        if memory_system and hasattr(memory_system, "shutdown"):
            try:
                await memory_system.shutdown()
            except asyncio.CancelledError:
                logging.debug("Memory system shutdown cancelled")
            except Exception as e:
                logging.warning(f"Error shutting down memory system: {e}", exc_info=True)

        # Final snapshot to all sync targets before closing storage
        if getattr(self, '_sync_service', None) and self._sync_service.is_running:
            try:
                await self._sync_service.force_snapshot()
                await self._sync_service.stop()
                logging.info("Sync service: final snapshot flushed")
            except asyncio.CancelledError:
                logging.debug("Sync service flush cancelled")
            except (AttributeError, TypeError, ConnectionError) as e:
                logging.warning(f"Error flushing sync service: {e}")
            except Exception as e:
                logging.warning(f"Error flushing sync service: {e}", exc_info=True)

        # Close storage
        if hasattr(self.storage, 'close'):
            try:
                await self.storage.close()
            except asyncio.CancelledError:
                logging.debug("Storage close cancelled")
            except (AttributeError, TypeError, ConnectionError, OSError) as e:
                logging.warning(f"Error closing storage: {e}")
            except Exception as e:
                logging.warning(f"Error closing storage: {e}", exc_info=True)

        logging.info("Kestrel Agent async shutdown complete.")
