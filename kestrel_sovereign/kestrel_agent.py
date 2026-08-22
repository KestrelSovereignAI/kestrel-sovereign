import logging
import json
import os
import asyncio
import contextlib
import hashlib
import inspect
import math
import sys
import time
from dataclasses import dataclass, replace as _replace_dataclass
from datetime import datetime
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.storage.privacy_wrapper import (
    ReentrantTransitionLock,
    EphemeralPurgeReport,
    StorePurgeResult,
    PurgeOutcome,
    PRIVACY_TRANSITION_RETRY_MESSAGE,
    PrivacyViolationError,
)
from kestrel_sovereign.security.encryption import DecryptionError
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.config import (
    SEMANTIC_CAPABILITIES_CONFIGURED_ENV,
    SEMANTIC_CAPABILITIES_CONFIG_ENV,
    SEMANTIC_INFERENCE_CONFIG_ENV,
    SEMANTIC_MAINTENANCE_CONFIG_ENV,
    SEMANTIC_MAINTENANCE_CONFIGURED_ENV,
    TRUSTED_AGENTS_DIR,
)
from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
from typing import Optional, Dict, List, Any, TYPE_CHECKING, Mapping
import re
from pathlib import Path
from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.features import (
    MandatoryFeatureReadinessError,
    discover_features,
    verify_mandatory_feature_set,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.config_validation import validate_feature_config


class HostFeatureConfigError(RuntimeError):
    """The host could not read the feature configuration it was asked to apply."""
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.command_policy import (
    BOOTSTRAP_ALLOWED_COMMANDS,
    SAFE_MODE_COMMANDS,
    prefixed_command_token,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.stores import (
    SQLiteTaskStore, SQLiteSessionService, SQLiteObservabilityStore,
    SQLiteFeedbackStore, SQLiteMemoryService
)
# PostgreSQL stores imported conditionally when pg_pool is available
from kestrel_sovereign.agent import ContextBuilder, ContextManager
from kestrel_sovereign.agent.context_manager import CONTEXT_HISTORY_LIMIT
from kestrel_sovereign.agent.boot import (
    AgentBootError,
    BootContext,
    BootPhase,
    BootPhaseState,
    run_boot_sequence,
)
from kestrel_sovereign.agent.operator_signals import inject_operator_turn
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.agent.streaming import (
    StreamingMixin,
    resolve_turn_invocation_context,
    _snapshot_post_response_hooks,
)
from kestrel_sovereign.agent.backup import BackupMixin
from kestrel_sovereign.agent.sleep import SleepMixin
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin, ContextStats
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.agent.invocation import bind_async_invocation
from kestrel_sovereign.signals import OrderedLockManager
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.hooks import HooksManager, evaluate_blocking_decision
from kestrel_sdk.hooks.base import HookEvent, HookInput
from kestrel_sovereign.bootstrap import BootstrapService, BootstrapState
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    extract_raw_user_content,
    check_prompt_injection,
    append_security_addendum,
)
from kestrel_sovereign.telemetry import (
    KESTREL_AGENT_NAME,
    KESTREL_SESSION_ID,
    OI_SPAN_KIND,
    OI_SPAN_KIND_CHAIN,
    optional_span,
)

if TYPE_CHECKING:
    from kestrel_sovereign.features.peers.directory import (
        PeerDirectoryRouter,
        PeerRequester,
    )
    from kestrel_sovereign.knowledge.inference import InferenceProfile

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
    # True when the transition is data-destructive and was NOT applied — it is
    # staged as pending and the caller must confirm via confirm_privacy_transition
    # (surfaced to the user as the !confirm-privacy-mode affordance). When True,
    # no state holder changed and ``message`` is the warning explaining why.
    requires_confirmation: bool = False
    pending_mode: Optional[str] = None
    # True only when this result reflects a mode actually applied to the state
    # holders. False for a staged (requires_confirmation) result AND for a no-op
    # confirm (nothing was pending) — so a confirm endpoint/caller can tell an
    # applied transition from a no-op without parsing the message.
    applied: bool = False
    # True when an EPHEMERAL exit was REFUSED because a required no-trace purge
    # sweep failed (#2673). The agent stays in EPHEMERAL (the safe, more
    # restrictive state) and ``applied`` is False — the transition must not be
    # reported as successful. Distinct from ``requires_confirmation`` (a staged
    # data-destructive downgrade the user can still confirm).
    purge_failed: bool = False
    # True when an already-running durable explicit-fact operation won the
    # privacy linearization race. No privacy state changed; callers should
    # surface a retryable conflict rather than success or an internal error.
    retryable_conflict: bool = False


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

# Overall internal budget for the *fallible prefix* of whole-agent shutdown
# (#2409). Kept coherent with the production outer deadline: the CLI/server/
# AgentManager paths all wrap agent.shutdown() in
# asyncio.wait_for(..., timeout=SHUTDOWN_TIMEOUT). If the internal per-step
# bounds summed higher than that outer deadline, a single hung early step
# could consume the whole outer budget and the outer wait_for — not our own
# composition — would be what stops us, starving every later step. So the
# internal sweep budget defaults to the SAME value as the production outer
# deadline (override with KESTREL_AGENT_SHUTDOWN_TIMEOUT_S) and every fallible
# step is bounded against a shared deadline derived from it.
KESTREL_AGENT_SHUTDOWN_TIMEOUT_S = float(
    os.environ.get("KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", str(SHUTDOWN_TIMEOUT))
)

# Headroom carved out of the sweep budget for the durable cleanup tail
# (background tasks, memory, sync snapshot, storage close). The fallible
# prefix may only spend budget up to (deadline - this reserve), so the
# durable tail always has time to run *within* the outer deadline instead of
# relying on the outer wait_for cancellation to trigger it.
KESTREL_SHUTDOWN_DURABLE_RESERVE_S = float(
    os.environ.get("KESTREL_SHUTDOWN_DURABLE_RESERVE_S", "1.0")
)

# Per-feature *cap* for the whole-agent shutdown sweep (#2409). A single
# feature that hangs in shutdown() must not stall the sweep or starve the
# durable cleanup tail (background tasks, memory, sync snapshot, storage
# close); once this elapses the feature is abandoned and the sweep moves on.
# The effective bound is min(this cap, fair share of the remaining sweep
# budget) — see shutdown() — so this cap never lets one feature exceed its
# slice of the coherent deadline above.
KESTREL_FEATURE_SHUTDOWN_TIMEOUT_S = float(
    os.environ.get("KESTREL_FEATURE_SHUTDOWN_TIMEOUT_S", "30")
)

# Minimum per-step guard (seconds) for the durable cleanup tail. Even when the
# budget is nearly exhausted, each durable step (background-task cleanup,
# memory, final snapshot, storage close) gets at least this nonzero attempt so
# data is never silently dropped for lack of a sliver of time. The tail is
# still bounded overall: a step that exceeds its guard is ABANDONED (not
# awaited — so a coroutine that suppresses cancellation cannot hang the tail
# past this guard), logged at WARNING, and the shutdown is reported as
# *degraded* — never as "completed".
KESTREL_SHUTDOWN_TAIL_MIN_STEP_S = float(
    os.environ.get("KESTREL_SHUTDOWN_TAIL_MIN_STEP_S", "0.5")
)

# Fraction of the EPHEMERAL shutdown-purge budget reserved for the durable
# purge-FAILURE audit write (#2673). When the purge times out or errors, the
# audit is the operator's evidence that the no-trace contract could not be
# certified — but a locked/hung audit DB must not blow the shutdown budget. So
# the audit tail is carved OUT of the same supplied budget (never added on top):
# the purge gets ``1 - fraction`` and the audit gets the remainder, keeping the
# whole operation bounded by the budget the caller supplied. A healthy audit
# store writes in well under its slice; a hung one is abandoned and the lost
# evidence is logged at ERROR.
KESTREL_SHUTDOWN_AUDIT_TAIL_FRACTION = float(
    os.environ.get("KESTREL_SHUTDOWN_AUDIT_TAIL_FRACTION", "0.25")
)


def _raise_unexpected_lifecycle_exception_group(
    error: BaseExceptionGroup,
) -> None:
    """Propagate process-control leaves from an owned lifecycle outcome."""

    _expected, unexpected = error.split((asyncio.CancelledError, Exception))
    if unexpected is not None:
        raise unexpected


async def await_lifecycle_task_completion(
    task: "asyncio.Future[object]",
) -> tuple[bool, BaseException | None]:
    """Drive one lifecycle task to a terminal state despite caller cancellation.

    A shutdown owner can be cancelled repeatedly while the task it owns still
    has to release a durable owner or close SQLite.  ``shield`` preserves that
    work; this helper preserves the *join*.  The returned boolean records
    cancellation of the joiner, while the second item reports the task's
    terminal outcome without confusing a task-cancellation with a fresh
    cancellation of the lifecycle owner.
    """
    # A ``CancelledError`` raised by ``shield(task)`` is ambiguous when both
    # tasks are cancelled in the same loop turn: it can report the owned task's
    # terminal cancellation, this joiner's cancellation, or both.  Only the
    # joiner itself can authoritatively tell us whether its cancellation was
    # requested.  Do not infer that from the owned task's terminal state.
    joiner = asyncio.current_task()
    cancelled = bool(joiner is not None and joiner.cancelling())
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if joiner is not None and joiner.cancelling():
                cancelled = True
        except BaseExceptionGroup as error:
            # A lifecycle owner can deliberately retain cancellation and an
            # ordinary cleanup failure in one group.  Like an Exception, that
            # is the owned task's terminal outcome, not a reason to abandon
            # later cleanup owners that the caller must still join.
            # Process-control exceptions are not lifecycle failure data.
            # Preserve their grouping and propagate them immediately.
            _raise_unexpected_lifecycle_exception_group(error)
            assert task.done()
        except Exception:
            # ``shield`` re-raises the owned task's terminal exception into
            # this joiner.  The lifecycle contract returns that outcome as
            # data so a staged owner can drain every later resource before it
            # reports or aggregates the failure.  It is terminal at this
            # point; fetch it below exactly once from the task itself.
            assert task.done()

    if task.cancelled():
        return cancelled, asyncio.CancelledError()
    failure = task.exception()
    if isinstance(failure, BaseExceptionGroup):
        # A task can already be terminal when this helper is entered, in which
        # case the loop above never observes its raised group.
        _raise_unexpected_lifecycle_exception_group(failure)
    return cancelled, failure


async def await_agent_shutdown_completion(agent: object) -> bool:
    """Join an agent's deferred durable cleanup without dropping it on cancel.

    ``KestrelAgent.shutdown`` may transfer dispatcher release and shared
    storage close to a continuation after its bounded user-facing shutdown
    budget expires.  The lifecycle caller that reported that timeout remains
    responsible for joining the continuation before its event loop exits.

    Returns whether this *join* observed cancellation.  Callers choose whether
    to re-raise after cleanup; either way the continuation has already reached
    a terminal result, so no SQLite worker or runtime owner is orphaned.
    """
    waiter = getattr(agent, "wait_for_shutdown_completion", None)
    if not callable(waiter):
        return False
    completion = waiter()
    if not inspect.isawaitable(completion):
        return False

    task = asyncio.ensure_future(completion)
    cancelled, failure = await await_lifecycle_task_completion(task)
    if failure is not None:
        raise failure
    return cancelled


def _resolve_shutdown_budget(
    minimum_tail_reserve: float = 0.0,
) -> tuple[float, float]:
    """Resolve ``(prefix_budget, tail_reserve)`` for whole-agent shutdown.

    Both values are clamped and validated against the production outer
    deadline (``SHUTDOWN_TIMEOUT``) so the internal deadline composition can
    never be incoherent:

    * The total internal budget is clamped to at most ``SHUTDOWN_TIMEOUT``.
      An override above it would let the outer ``wait_for`` (not our own
      composition) be what stops us, starving later steps and the durable
      tail; the mismatch is logged at WARNING and reconciled by clamping.
    * The configured durable-tail reserve is clamped into
      ``[min_step, total / 2]`` so the tail always has a nonzero, honest
      window and the fallible prefix normally keeps the majority of the
      budget.  A backend may require a larger minimum reservation for an
      owned resource to complete shutdown safely; that declared requirement
      wins, while the total remains bounded by the production outer deadline.

    ``minimum_tail_reserve`` is an optional, backend-owned contract such as
    SQLite's aiosqlite worker termination window.  It is deliberately not
    part of the generic SDK storage interface: a backend that does not expose
    it keeps the existing shutdown allocation unchanged.

    Returns ``(prefix_budget, tail_reserve)``.  The prefix can be zero only
    when an operator configures an outer budget too small to leave time for a
    safety-critical backend close; the total still never exceeds the outer
    deadline.
    """
    outer = float(SHUTDOWN_TIMEOUT)
    total = KESTREL_AGENT_SHUTDOWN_TIMEOUT_S
    if total <= 0:
        total = outer
    if total > outer:
        logging.warning(
            "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S=%.2fs exceeds the production "
            "outer shutdown deadline (%.2fs); clamping to keep the internal "
            "deadline coherent with the outer wait_for.",
            total,
            outer,
        )
        total = outer

    max_reserve = total / 2.0
    reserve = KESTREL_SHUTDOWN_DURABLE_RESERVE_S
    if reserve >= total:
        logging.warning(
            "KESTREL_SHUTDOWN_DURABLE_RESERVE_S=%.2fs >= internal shutdown "
            "budget %.2fs; clamping the durable reserve to %.2fs so the "
            "fallible prefix keeps a nonzero budget.",
            reserve,
            total,
            max_reserve,
        )
        reserve = max_reserve
    reserve = min(max(reserve, 0.0), max_reserve)
    if reserve <= 0.0:
        reserve = min(KESTREL_SHUTDOWN_TAIL_MIN_STEP_S, max_reserve)

    required_tail = max(0.0, minimum_tail_reserve)
    if required_tail > total:
        logging.warning(
            "Storage close requires %.2fs but the production shutdown "
            "budget is %.2fs; reserving the complete bounded budget for the "
            "durable tail.",
            required_tail,
            total,
        )
        required_tail = total
    reserve = max(reserve, required_tail)

    prefix_budget = max(0.0, total - reserve)
    return prefix_budget, reserve


def _minimum_storage_close_timeout(storage: Any) -> float:
    """Read an optional backend-owned storage-close reservation safely.

    Generic and non-SQLite storages deliberately do not need to implement the
    extension.  Treat malformed values as no reservation rather than letting a
    mock/proxy attribute change the shutdown budget or make it unbounded.
    """
    try:
        value = getattr(storage, "minimum_close_timeout_s", 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _storage_preclose(storage: Any):
    """Return storage's optional bounded pre-close operation, if available.

    ``AsyncStorage`` uses this hook to dispose a cached SQLAlchemy engine
    before closing its primary backend connection.  Keeping that unbounded
    external-resource work out of the primary close step preserves SQLite's
    aiosqlite worker-drain reservation.  Generic storage implementations keep
    their existing one-step close behavior.
    """
    try:
        dispose = getattr(storage, "dispose_cached_sqla_factory", None)
    except (AttributeError, TypeError, ValueError):
        return None
    return dispose if callable(dispose) else None


def _minimum_storage_preclose_timeout(storage: Any) -> float:
    """Read the optional cached-SQLAlchemy close reservation safely."""
    try:
        value = getattr(storage, "minimum_sqla_factory_close_timeout_s", 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _minimum_storage_potential_preclose_timeout(storage: Any) -> float:
    """Read a late-created SQLAlchemy factory's close reservation safely.

    Feature shutdown may lazily construct a factory after whole-agent
    shutdown has allocated its durable-tail deadline.  SQLite storage exposes
    this potential reservation independently of the current cache; older or
    generic storage implementations fall back to the currently cached value.
    """
    try:
        value = getattr(
            storage,
            "minimum_potential_sqla_factory_close_timeout_s",
            0.0,
        )
    except (AttributeError, TypeError, ValueError):
        value = 0.0
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value) and value > 0.0:
            return value
    return _minimum_storage_preclose_timeout(storage)


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

    #: Features refused activation because a contribution clashed with an
    #: already-registered key. Class-level so `/health/detailed` can be asked
    #: before boot has run and get "none refused" rather than an attribute
    #: error — an absent answer must not read as a clean one (#2951).
    rejected_feature_contributions: tuple = ()

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
        hosted_telegram_route_attestation_resolver: Any = None,
        peer_directory_router: Optional["PeerDirectoryRouter"] = None,
        peer_requester: Optional["PeerRequester"] = None,
        isolated_feature_data_dir: Optional[Path] = None,
        isolated_runtime_root: Optional[str | os.PathLike[str]] = None,
        isolated_runtime_namespace: Optional[str | os.PathLike[str]] = None,
        isolated_runtime_legacy_root: Optional[str | os.PathLike[str]] = None,
        isolated_runtime_hosted: bool = False,
        sovereign_trust_root_path: Optional[str] = None,
        identity_export_dir: Optional[Path] = None,
        semantic_inference_profile: Optional["InferenceProfile"] = None,
        semantic_inference_limits: Optional["InferenceLimits"] = None,
        semantic_maintenance_limits: Optional["SemanticMaintenanceLimits"] = None,
        semantic_capabilities: Optional["SemanticRuntimeCapabilities"] = None,
        semantic_inference_configured: bool = False,
        semantic_maintenance_configured: bool = False,
        semantic_capabilities_configured: bool = False,
        semantic_maintenance_allow_prior_verified_snapshot: bool = False,
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
                With ``pg_pool``, an explicitly supplied value also configures
                the separate scheduler advisory-lock pool.  An ambient
                ``KESTREL_DATABASE_URL`` does not replace the connection recipe
                copied from the supplied pool.
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
            hosted_telegram_route_attestation_resolver: Optional host-owned
                       pre-initialize resolver for a Telegram route already
                       provisioned outside Core. It supplies typed ledger
                       evidence before isolated feature discovery can start
                       the child handshake; Core never provisions provider
                       HTTP/webhooks through this seam.
            peer_directory_router: Optional hosted peer-directory/router.  When
                       supplied it replaces the local multi-agent HTTP adapter
                       used by ``PeersFeature``.
            peer_requester: Host-authenticated stable requester identity plus
                       opaque authorization scope for ``peer_directory_router``.
                       This is injected by the embedding runtime, never derived
                       from a tool caller or user-id field.
            isolated_feature_data_dir: Optional host-owned per-agent directory
                       retained for standalone embedding compatibility. Hosted
                       and multi-tenant factories must use the explicit root /
                       namespace contract below instead.
            isolated_runtime_root: Host-owned root for isolated-feature mutable
                       runtime state. Hosted factories must pair this with
                       ``isolated_runtime_namespace``.
            isolated_runtime_namespace: Canonical relative tenant/agent
                       namespace below ``isolated_runtime_root``. The runtime
                       validates it and securely binds it to this agent DID.
            isolated_runtime_legacy_root: Explicit, agent-scoped location of
                       the released hosted feature runtime layout. Managed
                       factories may supply this only to adopt existing
                       ``feature_venvs/<ClassName>`` state into the new
                       namespace; it is never a runtime fallback.
            isolated_runtime_hosted: Declares that this agent shares a host
                       runtime. Discovery of an isolated feature fails closed
                       unless an explicit root and namespace were supplied.
            sovereign_trust_root_path: Optional operator-owned JSON DID-document
                       path used to authorize constitution reanchor artifacts.
                       When omitted, the shared resolver reads
                       ``KESTREL_SOVEREIGN_TRUST_ROOT_PATH``. The graph database
                       is never a trust-root source.
            identity_export_dir: Optional per-agent local identity export
                       directory. Multi-agent hosts resolve this before agent
                       construction so it never depends on process CWD.
            semantic_inference_profile: Parsed, exact semantic materialization
                       profile injected by a managed agent configuration.
            semantic_inference_limits: Parsed, bounded materialization limits
                       injected with the selected profile.  The limits remain
                       operator configuration rather than service constants.
            semantic_maintenance_limits: Parsed bounded validation, audit,
                       and report budget for post-consolidation maintenance.
            semantic_inference_configured: Whether the managed configuration
                       explicitly supplied the profile, including an explicit
                       disabled profile. When true it takes precedence over a
                       legacy per-agent kestrel.toml block.
            semantic_maintenance_configured: Whether the managed
                configuration explicitly supplied maintenance limits.
                When true it takes precedence over a legacy per-agent
                kestrel.toml block, even when inference is disabled.
            semantic_maintenance_allow_prior_verified_snapshot: Explicit
                operator policy allowing scheduled training to use a prior
                complete maintenance snapshot for the active capability.
        """
        self.did = did
        self._privacy_mode = privacy_mode
        self.storage_path = storage_path
        effective_db_backend = db_backend or os.environ.get(
            "KESTREL_DB_BACKEND", "sqlite"
        )
        if type(isolated_runtime_hosted) is not bool:
            raise TypeError("isolated_runtime_hosted must be a bool")
        if isolated_feature_data_dir is not None and (
            isolated_runtime_root is not None
            or isolated_runtime_namespace is not None
        ):
            raise ValueError(
                "isolated_feature_data_dir cannot be combined with the hosted "
                "isolated runtime root/namespace contract"
            )
        # PostgreSQL hosts commonly have no agent-local filesystem database.
        # Treat that construction shape as hosted even if its factory predates
        # the explicit runtime-scope arguments.  An isolated feature then fails
        # before startup instead of silently collapsing into agent_data/default.
        # A factory which knows its host-owned root and namespace can (and must)
        # supply them below to make the feature usable.
        self.isolated_runtime_hosted = bool(
            isolated_runtime_hosted
            or isolated_runtime_root is not None
            or isolated_runtime_namespace is not None
            or (
                storage_path is None
                and effective_db_backend.lower() == "postgres"
            )
        )
        self.isolated_runtime_root: Optional[Path] = None
        self.isolated_runtime_namespace: Optional[Path] = None
        self.isolated_runtime_path: Optional[Path] = None
        self.isolated_runtime_legacy_root: Optional[Path] = None
        self.isolated_runtime_scope = None
        if isolated_runtime_root is not None or isolated_runtime_namespace is not None:
            # Keep the hosted runtime boundary owned by the isolated-runtime
            # module. This validates path traversal once at construction and
            # the proxy reuses the same canonical scope when it launches a
            # feature, rather than deriving mutable placement from database
            # storage (which PostgreSQL-backed hosted agents deliberately lack).
            from kestrel_sovereign.features.isolated_runtime import (
                resolve_legacy_isolated_runtime_root,
                resolve_isolated_runtime_namespace,
            )

            runtime_scope = resolve_isolated_runtime_namespace(
                isolated_runtime_root, isolated_runtime_namespace
            )
            self.isolated_runtime_root = runtime_scope.root
            self.isolated_runtime_namespace = runtime_scope.namespace
            self.isolated_runtime_path = runtime_scope.path
            self.isolated_runtime_scope = runtime_scope
            if isolated_runtime_legacy_root is not None:
                self.isolated_runtime_legacy_root = (
                    resolve_legacy_isolated_runtime_root(
                        isolated_runtime_legacy_root,
                        runtime_scope,
                    )
                )
        elif isolated_runtime_legacy_root is not None:
            raise ValueError(
                "isolated_runtime_legacy_root requires the hosted isolated "
                "runtime root/namespace contract"
            )
        # Human display name for observability span attribution (#2602). Set to
        # a best-effort floor at construction so EVERY agent object carries the
        # attribute from birth — no construction path (fleet load, spawn /
        # inception, scheduler-context, single-agent host, CLI/REPL) can produce
        # an unnamed object whose LLM spans fall back to the emitter's
        # "unknown". ``initialize()`` (graph-node name) and
        # ``AgentManager._register_agent`` (registered routing key) override this
        # with progressively more authoritative names. Mirrored onto the owning
        # LLMService once it exists (below) so LLM-call spans are attributed from
        # the very first call (#2573).
        self.agent_name: str = self._derive_construction_display_name(
            did, storage_path
        )
        self._allowed_features = allowed_features
        self._sync_enabled = _resolve_sync_enabled(sync_enabled)
        # Optional injected payer-policy + host db for multi-tenant embedding
        # (#1649). When set, they override the standalone kestrel.toml /
        # on-disk host.db lookups during credential resolution at init.
        self._injected_payer_policy = payer_policy
        self._injected_host_db = host_db
        if hosted_telegram_route_attestation_resolver is not None:
            from kestrel_sovereign.features.isolated_runtime import (
                set_hosted_telegram_route_attestation_resolver,
            )

            set_hosted_telegram_route_attestation_resolver(
                self, hosted_telegram_route_attestation_resolver
            )
        # Scoped peer routing is an explicit dependency-injection seam for
        # hosted multi-tenant runtimes.  PeersFeature validates the pair at
        # initialization; keeping the opaque scope here avoids serializing it
        # into agent state or accepting any caller-controlled substitute.
        self.peer_directory_router = peer_directory_router
        self.peer_requester = peer_requester
        self.isolated_feature_data_dir = (
            Path(isolated_feature_data_dir).expanduser().resolve()
            if isolated_feature_data_dir is not None
            else None
        )
        self._sovereign_trust_root_path = sovereign_trust_root_path
        self.identity_export_dir = identity_export_dir

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
        # Inference remains opt-in per tenant. An enabled profile carries exact
        # ontology and rule versions; SleepMixin consumes this value on every
        # incremental maintenance pass and never selects a profile itself.
        # Managed agents receive this from their LocalAgentConfig; direct
        # agents retain the existing per-agent TOML control surface below.
        from kestrel_sovereign.knowledge.inference import (
            InferenceError,
            InferenceLimits,
        )
        from kestrel_sovereign.knowledge.maintenance import (
            SemanticMaintenanceError,
            SemanticMaintenanceLimits,
            maintenance_allows_prior_verified_snapshot,
            maintenance_limits_from_config,
        )
        from kestrel_sovereign.knowledge.capabilities import (
            SemanticCapabilityConfigurationError,
            SemanticRuntimeCapabilities,
            semantic_capabilities_from_config,
        )

        if semantic_inference_limits is not None and not isinstance(
            semantic_inference_limits, InferenceLimits
        ):
            raise RuntimeError("Invalid semantic inference limits")
        self.semantic_inference_profile = semantic_inference_profile
        self.semantic_inference_limits = semantic_inference_limits or InferenceLimits()
        if semantic_maintenance_limits is not None and not isinstance(
            semantic_maintenance_limits, SemanticMaintenanceLimits
        ):
            raise RuntimeError("Invalid semantic maintenance limits")
        self.semantic_maintenance_limits = (
            semantic_maintenance_limits or SemanticMaintenanceLimits()
        )
        if semantic_capabilities is not None and not isinstance(
            semantic_capabilities, SemanticRuntimeCapabilities
        ):
            raise RuntimeError("Invalid semantic capability selection")
        self.semantic_capabilities = (
            semantic_capabilities or SemanticRuntimeCapabilities.stable()
        )
        if semantic_capabilities is not None:
            try:
                self.semantic_capabilities.validate()
            except SemanticCapabilityConfigurationError as exc:
                raise RuntimeError("Invalid semantic capability selection") from exc
        self.semantic_inference_configured = semantic_inference_configured
        self.semantic_maintenance_configured = semantic_maintenance_configured
        # An explicitly injected runtime selection is itself an opt-in.  Do
        # not require direct constructors to also know the internal lifecycle
        # flag used by managed config/env boot paths.
        semantic_capabilities_configured = (
            semantic_capabilities_configured or semantic_capabilities is not None
        )
        self.semantic_capabilities_configured = semantic_capabilities_configured
        if type(semantic_maintenance_allow_prior_verified_snapshot) is not bool:
            raise RuntimeError(
                "Invalid semantic maintenance prior verified snapshot policy"
            )
        self.semantic_maintenance_allow_prior_verified_snapshot = (
            semantic_maintenance_allow_prior_verified_snapshot
        )
        if semantic_inference_profile is not None:
            from kestrel_sovereign.knowledge.inference import (
                validate_inference_profile,
            )

            try:
                validate_inference_profile(semantic_inference_profile)
            except InferenceError as exc:
                raise RuntimeError("Invalid semantic inference profile") from exc
        if not semantic_inference_configured:
            serialized_profile = os.environ.get(SEMANTIC_INFERENCE_CONFIG_ENV)
            if serialized_profile is not None:
                try:
                    environment_profile = json.loads(serialized_profile)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid {SEMANTIC_INFERENCE_CONFIG_ENV} configuration"
                    ) from exc
                from kestrel_sovereign.knowledge.inference import (
                    inference_limits_from_config,
                    inference_profile_from_config,
                    validate_inference_profile,
                )

                try:
                    self.semantic_inference_profile = inference_profile_from_config(
                        environment_profile
                    )
                    self.semantic_inference_limits = inference_limits_from_config(
                        environment_profile
                    )
                    if self.semantic_inference_profile is not None:
                        validate_inference_profile(self.semantic_inference_profile)
                except InferenceError as exc:
                    raise RuntimeError(
                        f"Invalid {SEMANTIC_INFERENCE_CONFIG_ENV} configuration"
                    ) from exc
                semantic_inference_configured = True
                self.semantic_inference_configured = True
        serialized_capabilities = os.environ.get(SEMANTIC_CAPABILITIES_CONFIG_ENV)
        environment_capabilities_configured = os.environ.get(
            SEMANTIC_CAPABILITIES_CONFIGURED_ENV
        )
        if not semantic_capabilities_configured:
            if environment_capabilities_configured not in (None, "1"):
                raise RuntimeError(
                    f"Invalid {SEMANTIC_CAPABILITIES_CONFIGURED_ENV} configuration"
                )
            if (
                environment_capabilities_configured == "1"
                and serialized_capabilities is None
            ):
                raise RuntimeError(
                    f"{SEMANTIC_CAPABILITIES_CONFIGURED_ENV} requires "
                    f"{SEMANTIC_CAPABILITIES_CONFIG_ENV}"
                )
        if not semantic_capabilities_configured and serialized_capabilities is not None:
            try:
                capability_config = json.loads(serialized_capabilities)
                self.semantic_capabilities = semantic_capabilities_from_config(
                    capability_config
                )
            except (json.JSONDecodeError, SemanticCapabilityConfigurationError) as exc:
                raise RuntimeError(
                    f"Invalid {SEMANTIC_CAPABILITIES_CONFIG_ENV} configuration"
                ) from exc
            semantic_capabilities_configured = True
            self.semantic_capabilities_configured = True
        serialized_maintenance = os.environ.get(SEMANTIC_MAINTENANCE_CONFIG_ENV)
        environment_maintenance_configured = os.environ.get(
            SEMANTIC_MAINTENANCE_CONFIGURED_ENV
        )
        if not semantic_maintenance_configured:
            if environment_maintenance_configured not in (None, "1"):
                raise RuntimeError(
                    f"Invalid {SEMANTIC_MAINTENANCE_CONFIGURED_ENV} configuration"
                )
            if (
                environment_maintenance_configured == "1"
                and serialized_maintenance is None
            ):
                raise RuntimeError(
                    f"{SEMANTIC_MAINTENANCE_CONFIGURED_ENV} requires "
                    f"{SEMANTIC_MAINTENANCE_CONFIG_ENV}"
                )
        if not semantic_maintenance_configured and serialized_maintenance is not None:
            try:
                maintenance_config = json.loads(serialized_maintenance)
                self.semantic_maintenance_limits = maintenance_limits_from_config(
                    maintenance_config
                )
                self.semantic_maintenance_allow_prior_verified_snapshot = (
                    maintenance_allows_prior_verified_snapshot(
                        maintenance_config
                    )
                )
            except (json.JSONDecodeError, SemanticMaintenanceError) as exc:
                raise RuntimeError(
                    f"Invalid {SEMANTIC_MAINTENANCE_CONFIG_ENV} configuration"
                ) from exc
            semantic_maintenance_configured = True
            self.semantic_maintenance_configured = True
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
                except (OSError, ValueError) as exc:
                    logging.warning(
                        "Failed to read [privacy] or [semantic_inference] from %s: %s",
                        agent_toml, exc,
                    )
                else:
                    # Semantic inference is an explicit approval boundary, so
                    # parse it independently from optional privacy settings.
                    # A malformed [privacy] section cannot disable a valid
                    # materialization profile, and an invalid explicit profile
                    # always blocks startup.
                    if (
                        not semantic_inference_configured
                        and "semantic_inference" in toml_data
                    ):
                        from kestrel_sovereign.knowledge.inference import (
                            inference_limits_from_config,
                            inference_profile_from_config,
                            validate_inference_profile,
                        )

                        try:
                            self.semantic_inference_profile = (
                                inference_profile_from_config(
                                    toml_data["semantic_inference"]
                                )
                            )
                            self.semantic_inference_limits = (
                                inference_limits_from_config(
                                    toml_data["semantic_inference"]
                                )
                            )
                            if self.semantic_inference_profile is not None:
                                validate_inference_profile(
                                    self.semantic_inference_profile
                                )
                        except InferenceError as exc:
                            # Do not quietly disable a profile an operator
                            # explicitly placed in the agent configuration.
                            raise RuntimeError(
                                f"Invalid [semantic_inference] configuration in {agent_toml}"
                            ) from exc
                        semantic_inference_configured = True
                        self.semantic_inference_configured = True

                    if (
                        not semantic_maintenance_configured
                        and "semantic_maintenance" in toml_data
                    ):
                        try:
                            self.semantic_maintenance_limits = (
                                maintenance_limits_from_config(
                                    toml_data["semantic_maintenance"]
                                )
                            )
                            self.semantic_maintenance_allow_prior_verified_snapshot = (
                                maintenance_allows_prior_verified_snapshot(
                                    toml_data["semantic_maintenance"]
                                )
                            )
                        except SemanticMaintenanceError as exc:
                            raise RuntimeError(
                                f"Invalid [semantic_maintenance] configuration in {agent_toml}"
                            ) from exc
                        semantic_maintenance_configured = True
                        self.semantic_maintenance_configured = True

                    if (
                        not semantic_capabilities_configured
                        and "semantic_capabilities" in toml_data
                    ):
                        try:
                            self.semantic_capabilities = semantic_capabilities_from_config(
                                toml_data["semantic_capabilities"]
                            )
                        except SemanticCapabilityConfigurationError as exc:
                            raise RuntimeError(
                                f"Invalid [semantic_capabilities] configuration in {agent_toml}"
                            ) from exc
                        semantic_capabilities_configured = True
                        self.semantic_capabilities_configured = True

                    privacy = toml_data.get("privacy", {})
                    if not isinstance(privacy, Mapping):
                        logging.warning(
                            "Ignoring malformed [privacy] configuration in %s: expected a table",
                            agent_toml,
                        )
                    else:
                        self._privacy_computer_access = bool(
                            privacy.get("computer_access", False)
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
        # Construction before inception writes identity material is valid.
        # Once any identity artifact exists, however, loading is a readiness
        # gate: custody, completeness, cryptographic binding, and DID binding
        # failures must never downgrade the running agent to ``identity=None``.
        self.identity = None
        self._private_key = None
        if self.storage_path and self.did:
            storage_dir = Path(self.storage_path).parent
            legacy_docs = (
                sorted(storage_dir.glob("kestrel_0x*.json"))
                if storage_dir.is_dir()
                else []
            )
            born_hybrid_docs = (
                sorted(storage_dir.glob("*_did.json"))
                if storage_dir.is_dir()
                else []
            )
            identity_artifacts_present = bool(
                legacy_docs
                or born_hybrid_docs
                or (
                    storage_dir.is_dir()
                    and (
                        any(storage_dir.glob("kestrel_0x*.key.enc"))
                        or any(storage_dir.glob("kestrel_0x*.pem"))
                        or any(storage_dir.glob("*_ed25519.key.enc"))
                        or any(storage_dir.glob("*_mldsa65.bytes.enc"))
                        or any(storage_dir.glob("*_archival_slhdsa*.bytes.enc"))
                        or any((storage_dir / "successions").glob("*.json"))
                    )
                )
            )
            if identity_artifacts_present:
                from kestrel_sovereign.identity.runtime_identity import (
                    IdentityReadinessError,
                    load_agent_identity,
                )

                if not legacy_docs and not born_hybrid_docs:
                    raise IdentityReadinessError(
                        "integrity",
                        cause_type="IdentityDocumentMissing",
                    )
                if (
                    len(legacy_docs) > 1
                    or len(born_hybrid_docs) > 1
                    or (legacy_docs and born_hybrid_docs)
                ):
                    raise IdentityReadinessError(
                        "integrity",
                        cause_type="AmbiguousIdentityDocuments",
                    )

                legacy_key_id = legacy_docs[0].stem if legacy_docs else None
                try:
                    loaded = load_agent_identity(
                        legacy_key_id,
                        storage_dir=storage_dir,
                    )
                    bound_dids = {
                        candidate
                        for candidate in (loaded.legacy_did, loaded.new_did)
                        if candidate
                    }
                    if self.did not in bound_dids:
                        raise IdentityReadinessError(
                            "binding",
                            cause_type="ConfiguredDIDMismatch",
                        )

                    self.identity = loaded
                    # Born-hybrid agents have no legacy keypair;
                    # consumers of _private_key (e.g. wallet plumbing
                    # in spawn) get None and already guard for it.
                    if self.identity.legacy_keypair is not None:
                        self._private_key = self.identity.legacy_keypair.private_key
                    if self.identity.is_born_hybrid:
                        logging.info(
                            "Agent identity loaded as BORN-HYBRID: %s",
                            self.identity.new_did,
                        )
                    elif self.identity.is_hybrid:
                        logging.info(
                            "Agent identity loaded as HYBRID: legacy=%s -> new=%s",
                            self.identity.legacy_did, self.identity.new_did,
                        )
                    else:
                        logging.info(
                            "Agent identity loaded as legacy-only: %s",
                            self.identity.legacy_did,
                        )
                except IdentityReadinessError:
                    raise
                except Exception as exc:
                    # Do not chain the detailed loader exception: it can carry
                    # filesystem paths or secret-bearing crypto/provider text,
                    # and generic startup traceback logging must remain safe.
                    raise IdentityReadinessError.from_load_error(exc) from None

        # Determine database backend
        self._db_backend = effective_db_backend
        # Birth-record capability the runtime database could not be given and
        # no retry can supply (#2871). Surfaced by the ``birth_record`` health
        # check; empty on every healthy agent.
        self._birth_record_shortfall: List[str] = []
        self._birth_record_shortfall_retryable = False
        self._database_url = database_url or os.environ.get("KESTREL_DATABASE_URL")
        # ``_database_url`` is deliberately the resolved storage setting, but
        # a shared pool has an independent advisory-lock pool.  Preserve the
        # constructor provenance so an ambient URL cannot discard a custom
        # connector, SSL context, or other connection recipe copied from that
        # pool.  A non-empty explicit ``database_url`` remains the backwards-
        # compatible override for callers that intentionally configure the
        # scheduler pool independently.
        self._explicit_advisory_dsn = database_url or None

        # Storage will be initialized asynchronously.
        #
        # PRIVACY BOUNDARY (#2672). ``self.storage`` is the privacy-governing
        # boundary — its graph writes default-deny user-derived content in
        # volatile modes. ``self._raw_storage`` is the UNGOVERNED store beneath
        # it: a privileged, first-party control-plane handle used by core paths
        # that must persist regardless of conversational privacy mode (identity
        # node, constitution/doctrine anchors), each of which carries its own
        # explicit volatile-mode gate. It is deliberately NOT part of the
        # feature-facing storage API; in-tree/entry-point features are first-party
        # code sharing this interpreter (they can already reach anything), so the
        # privacy wrapper is a governed API surface for the SANCTIONED path
        # (persist via ``self.agent.storage``), NOT an in-process sandbox. The
        # hard boundary against untrusted extension code is process isolation
        # (``features/isolated_runtime.py``), which never receives this object.
        self._raw_storage = None
        self.storage = None

        # Explicit boot state (#2522). Replaces the old ``_raw_storage is None``
        # proxy that let a second initialize() skip the body and run only the
        # readiness tail over partial state. The boot sequence advances this
        # NOT_STARTED → IN_PROGRESS → READY, or → FAILED (terminal, rolled back)
        # on any phase failure. Readiness may only fire in READY.
        self._boot_state: BootPhaseState = BootPhaseState.NOT_STARTED
        self._boot_context: Optional[BootContext] = None

        self.llm_service = llm_service or LLMService()
        from kestrel_sovereign.agent.operator_signals import OperatorSignalProducer
        self.operator_signal_producer = OperatorSignalProducer(self)
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
        # Mirror the construction-time display name onto the LLMService so LLM
        # spans are attributed from the very first call — including the genesis
        # audit and feature-init calls that run inside initialize() before the
        # registrar's authoritative stamp lands (#2602 / #2573).
        self._set_display_name(self.agent_name)
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
        # Declarative SDK contributions are wired lazily once the existing
        # per-agent wait/signal registries exist.  The operator registry is the
        # single service/workflow registry for this agent lifecycle.
        from kestrel_sovereign.operator import OperatorRuntimeRegistry

        self.operator_registry = OperatorRuntimeRegistry()
        self.feature_contribution_runtime = None
        self.permission_defaults_registry = None
        self.setup_step_registry = None
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
        # The turn id that currently HOLDS the turn lock, set/cleared by
        # `_turn_lifecycle`. Pairs with the task-local `_CURRENT_TURN_ID`
        # ContextVar so a caller can tell "I own the live turn" from "my task
        # inherited a finished turn's context" — the check that keeps a
        # detached task from reading a concurrent turn's `_active_session_id`
        # (#2877). Read via `get_turn_bound_session_id`, not directly.
        self._live_turn_id: Optional[str] = None

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
        # If the bounded durable tail cannot wait for dispatcher release, this
        # task owns the only safe successor: dispatcher drain followed by the
        # matching storage close.  It deliberately does not live in
        # ``_background_tasks`` because that set is cancelled at the start of
        # the tail; cancelling it would revive the post-close durable-write
        # race the continuation exists to prevent.
        self._durable_shutdown_continuation: Optional[asyncio.Task] = None
        self._durable_shutdown_continuation_lock = asyncio.Lock()

        # Cancellation tracking for stop button functionality
        self._current_request_id: Optional[str] = None
        self._active_request_ids: set[str] = set()
        # A caller may retry the same id while its original delivery is still
        # running. Keep lifecycle registration ownership per delivery so one
        # completion cannot unregister the other.
        self._active_request_counts: dict[str, int] = {}
        # Monotonic registration time per active request id so the
        # restart coordinator can age out stale markers (#1558).
        self._active_request_started_at: dict[str, float] = {}
        self._cancelled_requests: set = set()
        # Task-reentrant so a durable-identity write (rename / description /
        # discovery / user-name / SOUL) invoked as a TOOL inside a streamed turn
        # — which already holds this lock across the whole turn — re-enters
        # instead of self-deadlocking on its own task's lock (#2672 review P1).
        self._privacy_transition_lock = ReentrantTransitionLock()
        # A data-destructive privacy transition (e.g. PUBLIC → EPHEMERAL) staged
        # awaiting explicit confirmation via confirm_privacy_transition. None when
        # no transition is pending. Guarded by _privacy_transition_lock.
        self._pending_privacy_transition: Optional[PrivacyMode] = None

        # Shared lock manager for the dispatcher (Phase 1) AND the turn
        # lifecycle (Phase 2). CONVERSATION is acquired by `_turn_lifecycle`
        # in `process_input`/`process_input_streaming` — registered signal
        # sources are forbidden from declaring it (registry enforces).
        self._lock_manager = OrderedLockManager()
        # Serializes constitutional deadline increments/audits. The durable
        # store itself is initialized after the primary DB connects.
        self._constitution_state_lock = asyncio.Lock()
        self._constitution_state_lock_owner = None
        self._constitution_state_store = None

        # Session state
        self._session_briefed = False
        self._safe_mode = False

        # Dynamic tool loading: explored features get direct tool access
        self._explored_features: dict = {}  # ordered dict (insertion order) for LRU eviction
        self._direct_tools: dict = {}
        self._direct_tool_defs: list = []
        self._tool_to_feature: dict = {}  # tool_name -> feature tool_name
        self._constitution_receipt_tool_calls: list[dict] = []
        self._constitution_receipt_expected: Optional[dict] = None
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
    def _derive_construction_display_name(
        did: str, storage_path: Optional[str]
    ) -> str:
        """Best-effort human display name known at construction time (#2602).

        The authoritative name is resolved later — the agent's graph node in
        ``initialize()`` and the registered routing key in
        ``AgentManager._register_agent`` — so this is only the construction-time
        floor: an agent that emits a span before either runs is still attributed
        to a real name rather than the observability emitter's "unknown"
        fallback. Resolution order: the agent directory's ``[agent] name`` from
        ``kestrel.toml`` (the config name), then the data-directory name, then a
        DID label as a last resort. Never returns an empty string.
        """
        if storage_path:
            agent_dir = Path(storage_path).parent
            toml_path = agent_dir / "kestrel.toml"
            if toml_path.exists():
                try:
                    try:
                        import tomllib  # type: ignore[import-not-found]
                    except ImportError:
                        import tomli as tomllib  # type: ignore[import-not-found]
                    with open(toml_path, "rb") as f:
                        configured = tomllib.load(f).get("agent", {}).get("name")
                    if configured:
                        return str(configured)
                except Exception:  # noqa: BLE001 - best-effort floor only
                    logging.debug(
                        "Failed to read [agent] name from %s", toml_path,
                        exc_info=True,
                    )
            dir_name = agent_dir.name
            if dir_name and dir_name not in (".", "..", "/"):
                return dir_name
        return did or "Unnamed Agent"

    def _set_display_name(self, name: Optional[str]) -> None:
        """Publish the human display name used for observability attribution.

        Sets ``self.agent_name`` — the plain instance attribute the
        observability emitter reads — and mirrors it onto the owning
        ``LLMService`` so LLM-call spans carry ``kestrel.agent_name`` (#2573)
        instead of the emitter's "unknown" fallback (#2602). Called at three
        points, each overriding the last: ``__init__`` (construction-time floor
        from config / data-dir), ``initialize()`` (authoritative name from the
        agent's graph node), and ``AgentManager._register_agent`` (the
        registered routing key). A falsy ``name`` is ignored so a later, less
        specific resolution never blanks out an already-good name.
        """
        if not name:
            return
        self.agent_name = name
        llm = getattr(self, "llm_service", None)
        if llm is not None and hasattr(llm, "set_agent_display_name"):
            try:
                llm.set_agent_display_name(name)
            except Exception:  # noqa: BLE001 - attribution must never break init
                logging.debug(
                    "Failed to mirror agent display name to LLMService",
                    exc_info=True,
                )

    def register_constitution_receipt_tool(
        self, *, canary: str, signal_id: str
    ) -> None:
        """Expose the per-turn phantom constitution receipt tool.

        The dispatcher calls this only for claude_code COGNITION
        dispatches with require_constitution_echo=True. The tool is
        removed by clear_constitution_receipt_tool after verification,
        so it never becomes a durable user-facing capability.
        """
        from kestrel_sovereign.signals.constitution_canary import (
            PHANTOM_RECEIPT_ARG_NAME,
            PHANTOM_RECEIPT_TOOL_NAME,
        )

        self._constitution_receipt_tool_calls = []
        self._constitution_receipt_expected = {
            "canary": canary,
            "signal_id": signal_id,
        }
        self._direct_tools[PHANTOM_RECEIPT_TOOL_NAME] = None
        self._tool_to_feature[PHANTOM_RECEIPT_TOOL_NAME] = (
            PHANTOM_RECEIPT_TOOL_NAME
        )
        self._direct_tool_defs = [
            tool_def for tool_def in self._direct_tool_defs
            if tool_def.get("function", {}).get("name")
            != PHANTOM_RECEIPT_TOOL_NAME
        ]
        self._direct_tool_defs.append(
            {
                "type": "function",
                "function": {
                    "name": PHANTOM_RECEIPT_TOOL_NAME,
                    "description": (
                        "Confirm receipt of the current turn's "
                        "constitutional system prompt. Operational "
                        "metadata only; produces no user-visible work."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            PHANTOM_RECEIPT_ARG_NAME: {
                                "type": "string",
                                "description": (
                                    "The exact constitution receipt "
                                    "canary supplied in the system prompt."
                                ),
                            }
                        },
                        "required": [PHANTOM_RECEIPT_ARG_NAME],
                        "additionalProperties": False,
                    },
                },
            }
        )

    def clear_constitution_receipt_tool(self) -> None:
        """Remove the per-turn phantom receipt tool from the tool catalog."""
        from kestrel_sovereign.signals.constitution_canary import (
            PHANTOM_RECEIPT_TOOL_NAME,
        )

        self._direct_tools.pop(PHANTOM_RECEIPT_TOOL_NAME, None)
        self._tool_to_feature.pop(PHANTOM_RECEIPT_TOOL_NAME, None)
        self._direct_tool_defs = [
            tool_def for tool_def in self._direct_tool_defs
            if tool_def.get("function", {}).get("name")
            != PHANTOM_RECEIPT_TOOL_NAME
        ]
        self._constitution_receipt_tool_calls = []
        self._constitution_receipt_expected = None

    async def _handle_constitution_receipt_tool(
        self, *, canary: str = ""
    ) -> dict:
        """No-op handler for the phantom constitution receipt tool."""
        from kestrel_sovereign.signals.constitution_canary import (
            PHANTOM_RECEIPT_ARG_NAME,
            PHANTOM_RECEIPT_TOOL_NAME,
        )

        self._constitution_receipt_tool_calls.append(
            {
                "name": PHANTOM_RECEIPT_TOOL_NAME,
                "arguments": {PHANTOM_RECEIPT_ARG_NAME: canary},
            }
        )
        expected = self._constitution_receipt_expected or {}
        return {
            "success": True,
            "recorded": canary == expected.get("canary"),
        }

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

    async def _load_or_promote_soul_resource(
        self, agent_data_dir: Optional[str]
    ) -> None:
        """Prefer canonical encrypted SOUL, else promote a disk seed/cache."""
        try:
            loaded = await self.context_builder.load_canonical_soul_resource()
            if loaded:
                return
        except Exception as exc:
            logging.warning("Canonical SOUL resource load failed: %s", exc)

        if not agent_data_dir:
            return
        soul_path = Path(agent_data_dir) / "SOUL.md"
        if not soul_path.exists():
            return
        try:
            content = soul_path.read_text(encoding="utf-8")
        except Exception as exc:
            logging.warning("SOUL.md seed read failed during promotion: %s", exc)
            return
        if not content.strip():
            return
        # Privacy boundary (#2672 review P1): promoting a disk SOUL seed writes the
        # user-derived body into the encrypted resource table AND a durable graph
        # reference. In a volatile privacy mode, skip the promotion and load the
        # seed in-session only — nothing durable is created.
        from kestrel_sovereign.features.storage_access import (
            hides_persisted_user_content,
        )
        if hides_persisted_user_content(self):
            logging.info(
                "SOUL.md seed not promoted — durable identity-resource writes are "
                "disabled in the current privacy mode; loading it in-session only "
                "(#2672)"
            )
            try:
                self.context_builder._load_soul_md()
            except Exception as exc:
                logging.warning("In-session SOUL.md seed load failed: %s", exc)
            return
        try:
            await self.storage.promote_soul_seed(
                content,
                created_by=self.agent_id,
                source=str(soul_path),
            )
            await self.context_builder.load_canonical_soul_resource()
            logging.info("Promoted SOUL.md seed into canonical private resource")
        except Exception as exc:
            logging.warning("SOUL.md seed promotion failed: %s", exc)

    async def _maybe_refresh_user_byok_resolver(self, user_passphrase: str | None) -> None:
        """Re-resolve LLM provider if using USER_BYOK and passphrase is provided.

        USER_BYOK agents require a per-request passphrase to decrypt their provider
        credentials. This method re-resolves the LLM key resolver with the fresh
        passphrase for each request. Non-BYOK agents and requests without a passphrase
        are no-ops.

        Args:
            user_passphrase: The per-request passphrase for zero-knowledge BYOK decryption.
        """
        if not user_passphrase:
            return

        # Check if we're using USER_BYOK for LLM
        from kestrel_sdk.payer_policy import PayerKind, ResourceClass
        policy = self._resolve_payer_policy()
        if policy.llm.kind != PayerKind.USER_BYOK:
            return

        # Re-resolve with the passphrase
        from kestrel_sovereign.services.payer_resolver import FoundationPayerResolver
        host_db = await self._resolve_host_db()
        resolver = FoundationPayerResolver(
            policy,
            db=self._raw_storage.db if self._raw_storage else None,
            host_db=host_db,
        )
        resolved = await resolver.resolve_for(
            self.did,
            ResourceClass.LLM,
            user_passphrase=user_passphrase,
        )

        # Update the LLM service's key resolver
        if resolved.enabled and resolved.key_resolver and hasattr(self, 'llm_service'):
            self.llm_service.key_resolver = resolved.key_resolver
            logging.info(
                f"Refreshed USER_BYOK LLM resolver for agent {self.did[:30]}... "
                f"with per-request passphrase"
            )

    async def initialize(self) -> None:
        """Boot the agent as an explicit, ordered, rollback-safe phase sequence.

        Replaces the old ``if self._raw_storage is None:`` monolith (#2522).
        Boot advances NOT_STARTED -> IN_PROGRESS -> READY, or -> FAILED (with a
        reverse-order rollback of every resource the partial boot acquired) on
        any phase failure. Idempotent when already READY. A prior FAILED boot is
        terminal: a retry is refused with :class:`AgentBootError` (close/rebuild
        first) so readiness can never run on partial state.
        """
        state = self._boot_state
        if state is BootPhaseState.READY:
            return
        if state is BootPhaseState.IN_PROGRESS:
            raise AgentBootError(
                "initialize() is already in progress for this agent; "
                "concurrent/re-entrant boot is not supported."
            )
        if state is BootPhaseState.FAILED:
            raise AgentBootError(
                "agent boot previously failed and its partial state was rolled "
                "back; construct a fresh agent (or shutdown() this one) before "
                "retrying — a partial-state retry is refused."
            )
        ctx = BootContext(logger_=logging.getLogger(__name__))
        self._boot_context = ctx

        def _set_state(new_state: BootPhaseState) -> None:
            self._boot_state = new_state

        await run_boot_sequence(self._boot_phases(), ctx, _set_state)

    def _boot_phases(self) -> list[BootPhase]:
        """The ordered boot phases — this order IS the dependency contract.

        Each phase declares the resources it deliberately RETAINS on failure
        via ``retained``; everything else it acquires is registered for
        reverse-order rollback on the :class:`BootContext`.
        """
        return [
            BootPhase("storage_privacy", self._boot_phase_storage_privacy),
            BootPhase(
                "a2a_observability_signals",
                self._boot_phase_a2a_observability_signals,
            ),
            BootPhase(
                "providers_payer_sync", self._boot_phase_providers_payer_sync
            ),
            BootPhase(
                "identity_constitution_features",
                self._boot_phase_identity_constitution_features,
                retained=(
                    "agent identity graph node (durable; reused on a fresh retry)",
                ),
            ),
            BootPhase(
                "memory_bootstrap_context",
                self._boot_phase_memory_bootstrap_context,
            ),
            BootPhase(
                "periodic_services_readiness",
                self._boot_phase_periodic_services_readiness,
            ),
        ]

    async def _boot_phase_storage_privacy(self, ctx: BootContext) -> None:
        """Phase 1 — storage + privacy. Cold-restore, raw/privacy storage, constitution runtime state, embedding-pin hydration, privacy agent, and the force-local-only embedding gate. Owns the primary DB connection."""
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

        # Assertion authority is minted only after this agent boot path has
        # resolved its DID.  Passing an agent_id to AsyncStorage is ordinary
        # resource scoping, not authority issuance; the opaque capability is
        # required for the normalized semantic assertion store on every backend.
        assertion_tenant_capability = (
            _resolve_authenticated_agent_assertion_capability(self.did, self.identity)
            if self.did
            else None
        )

        # Initialize async storage based on backend type
        if self._db_backend.lower() == "postgres" and (self.pg_pool or self._database_url):
            # PostgreSQL backend - reuse shared pool if available
            if self.pg_pool:
                from kestrel_sovereign.storage.db.postgres import PostgresBackend
                # A PostgreSQL scheduler effect holds a session advisory gate
                # across target execution. Its bounded dedicated pool needs the
                # same DSN, never the shared operational pool, or a waiting
                # fence could consume the connection required for renewal/final
                # CAS. A missing DSN fails clearly at the first scheduler gate.
                pg_backend = PostgresBackend.from_pool(
                    self.pg_pool,
                    advisory_dsn=self._explicit_advisory_dsn,
                )
                self._raw_storage = AsyncStorage(
                    backend=pg_backend,
                    agent_id=self.did,
                    llm_service=self.llm_service,
                    _assertion_tenant_capability=assertion_tenant_capability,
                    semantic_capabilities=self.semantic_capabilities,
                )
                logging.info(f"Using shared PostgreSQL pool for Kestrel storage (agent: {self.did})")
            else:
                self._raw_storage = AsyncStorage(
                    backend="postgres",
                    dsn=self._database_url,
                    agent_id=self.did,
                    llm_service=self.llm_service,
                    _assertion_tenant_capability=assertion_tenant_capability,
                    semantic_capabilities=self.semantic_capabilities,
                )
                logging.info(f"Using PostgreSQL backend for Kestrel storage (agent: {self.did})")
        else:
            # SQLite backend (default) - agent_id optional since each agent has own DB
            self._raw_storage = AsyncStorage(
                self.storage_path,
                agent_id=self.did,
                llm_service=self.llm_service,
                _assertion_tenant_capability=assertion_tenant_capability,
                semantic_capabilities=self.semantic_capabilities,
            )
            logging.info(f"Using SQLite backend for Kestrel storage: {self.storage_path}")

        # Register the reverse-order teardown BEFORE opening the connection.
        # ``AsyncStorage.initialize()`` may open the primary DB connection and
        # then raise partway (e.g. a failed migration), so registering the undo
        # only after it returns would leak that connection on a mid-initialize
        # failure (#2522 P1). The teardown null-guards + ``hasattr(raw, close)``
        # so closing a partially-initialized storage is safe; it also drops the
        # privacy wrapper / privacy agent set below.
        ctx.on_rollback("storage", self._boot_teardown_storage)
        await self._raw_storage.initialize()

        # Wrap storage with privacy-enforcing layer
        self.storage = PrivacyEnforcingStorage(self._raw_storage, self._privacy_mode)

        # Bring the birth record into the database this runtime reads (#2871)
        # BEFORE anything downstream consults the agent node. This position is
        # not a preference: the very next block treats a missing node as a
        # genuinely new identity and lets it establish a fresh constitution
        # anchor, and the block after it reads the agent's name. Reconciling
        # later would leave the agent anchored to the wrong constitution and
        # running as "Unnamed Agent" even once its record had arrived. It also
        # keeps the refusal ahead of every feature side effect and SESSION_START
        # hook, so a host that cannot produce a valid birth record does no
        # durable startup work before it stops.
        await self._reconcile_birth_record_with_runtime_database()

        # Distinguish a genuinely new identity from a legacy identity that
        # merely lacks the new runtime-state row. Only a genuine first boot
        # may establish its initial constitution anchor automatically; a
        # lookup failure is treated as existing/unknown and therefore
        # follows the fail-closed migration path.
        early_agent_node = None
        identity_lookup_succeeded = False
        try:
            early_agent_node = await self.storage.get_node(self.agent_id)
            identity_lookup_succeeded = True
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Could not load agent identity node for sync policy: %s",
                exc,
            )
        # Carry the identity node to the providers/sync phase (its
        # constitution-anchor gate reads it) — the single value that crosses a
        # phase boundary now that the body is split.
        ctx.early_agent_node = early_agent_node

        # Safe Mode and periodic-audit deadlines are authoritative runtime
        # state. Restore them before features can emit startup cognition or
        # the server can report readiness. Legacy agents receive a due-now
        # migration record; the full verification runs later in this same
        # initialize call, after feature/spawn constraints are available.
        await self._initialize_constitution_runtime_state(
            is_new_identity=(
                identity_lookup_succeeded and early_agent_node is None
            )
        )

        # #2290 — re-apply any previously-verified shared embedding-space
        # pins from the persisted parity record. ``_verified_space_pins`` is
        # process-local, so without this a restart would silently drop the
        # shared local/cloud space until an operator re-ran the parity probe,
        # stranding reindexed rows outside kNN. Best-effort; never blocks init.
        try:
            if self.llm_service and hasattr(
                self.llm_service, "hydrate_verified_space_pins"
            ):
                await self.llm_service.hydrate_verified_space_pins(
                    getattr(self._raw_storage, "db", None)
                )
        except Exception as exc:  # noqa: BLE001
            logging.debug("Embedding-space pin hydration skipped: %s", exc)

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

        # Async completion pass for routes the sync registry build couldn't
        # bring up (e.g. an OpenRouter route with only a management key, now
        # completed via a bootstrap child key). Guard on iscoroutinefunction
        # rather than hasattr: MagicMock-based LLM-service fakes satisfy
        # hasattr (auto-attr) but are NOT awaitable, so a bare hasattr guard
        # would raise "MagicMock can't be awaited" in every such test. Only
        # await a genuine async finalize hook.
        finalize_providers = getattr(self.llm_service, "finalize_providers", None)
        if inspect.iscoroutinefunction(finalize_providers):
            # Pass the host-level store so the registry can persist + reuse
            # the OpenRouter bootstrap child key across restarts instead of
            # minting a new one every cold start. Resolve it (injected host
            # db, else the on-disk ``host.db``) so standalone/SQLite
            # deployments — not just multi-tenant embeddings — get the reuse.
            await finalize_providers(host_db=await self._resolve_host_db())


    async def _boot_phase_a2a_observability_signals(self, ctx: BootContext) -> None:
        """Phase 2 — A2A stores, observability, and the signal spine. TaskManager + its stores, the observability/feedback sinks, the enablement store, the SignalDispatcher/registries, and the core (always-on) signal sources."""
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
        # Register teardown BEFORE initialize: ``TaskManager.initialize()`` opens
        # its A2A store connections (task/session/observability/memory/feedback)
        # SEQUENTIALLY, so a later store's failure must still close the earlier
        # ones — registering the undo only after it returns would leak them on a
        # mid-initialize failure (#2522 P1).
        ctx.on_rollback("task_manager", self._boot_teardown_task_manager)
        await self.task_manager.initialize()

        # Expose feedback store for features and commands
        self.feedback_store = feedback_store

        # Expose observability store for orchestrator instrumentation
        self.observability_store = observability_store

        # Wire the store into the per-agent LLMService so every chat /
        # generate / streaming call lands in a2a_llm_calls (#2236). The
        # service instruments all chokepoints but stays dark without
        # this attach — only features logging directly to the store
        # (e.g. reflection) showed up in the LLM Calls panel.
        if hasattr(self.llm_service, "set_observability_store"):
            self.llm_service.set_observability_store(observability_store)

        # Privacy-gate the observability sink at the layer boundary (F076).
        # Tool-call args and metadata are user content, so the sink must
        # honour the agent's privacy mode: EPHEMERAL/ISOLATED elide the
        # payload, ANONYMOUS anonymizes it. Bind the live privacy config by
        # reference (same pattern as set_force_local_only_provider) so
        # mid-session mode flips are picked up automatically.
        if hasattr(observability_store, "set_privacy_config_provider"):
            observability_store.set_privacy_config_provider(
                lambda pa=self.privacy_agent: pa.privacy_config
            )
        # Wire the EPHEMERAL safety-net sweep into the storage wrapper so
        # purge_ephemeral_session also scrubs any observability rows
        # authored during the ephemeral stint (F076). Tool-call args in
        # a2a_observability use the agent DID as agent_name (see the
        # log_tool_call callers), so scope by DID on both columns.
        if hasattr(self.storage, "set_observability_purge"):
            self.storage.set_observability_purge(
                lambda since, obs=observability_store, did=self.did: (
                    obs.purge_observability_since(
                        since, agent_did=did, agent_name=did
                    )
                )
            )

        # Per-agent enablement deltas (agent-driven feature/MCP-server
        # add/remove that must survive restart). Reuses the observability
        # backend so it lands in the agent's own DB. Initialized BEFORE
        # feature discovery so the reconcile-union below can read it.
        # Degrades to None (deltas disabled) rather than blocking init.
        self._feature_enablement_store = None
        try:
            from kestrel_sovereign.a2a.stores.unified.feature_enablement_store import (
                FeatureEnablementStore,
            )
            backend = observability_store.backend
            await backend.connect()  # idempotent — no-op if already connected
            store = FeatureEnablementStore(backend)
            await store.initialize()
            self._feature_enablement_store = store
        except Exception as e:  # noqa: BLE001 - never block init on this
            logging.warning(
                "FeatureEnablementStore unavailable; enablement deltas "
                "disabled (features still load from the bootstrap allowlist): %s",
                e,
            )

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
        self._ensure_feature_contribution_runtime()
        self.signal_log_store = signal_log_store
        self.dispatcher = SignalDispatcher(
            agent=self,
            registry=self.signal_registry,
            lock_manager=self._lock_manager,
            store=signal_log_store,
        )
        # Register teardown before durable initialization's first await.  That
        # initialization starts owner liveness before it finishes startup
        # recovery; a later boot-phase failure must cancel it and release even
        # a partially registered owner before storage rolls back.
        ctx.on_rollback("signal_dispatcher", self._boot_teardown_dispatcher)
        # The outcome-only signal_log is initialized above.  Initialize the
        # separate pending-delivery ledger during boot so external workflow
        # consumers can safely register before the first signal arrives.
        await self.dispatcher.initialize_durable_delivery()

        # Register the always-on core signal sources under an explicit
        # MANDATORY policy (#2522). These are boot-critical routing targets,
        # so a registration failure must abort boot — but as an ATOMIC batch:
        # if any one fails validation, register_batch removes the ones already
        # added in this batch, so a partial core source set never survives.
        # Each build_* is unconditional (not gated on any feature):
        #   a2a.task_complete    — peer-task completion wake (#889 Phase 5)
        #   a2a.task_submitted   — inbound peer-task wake (#645)
        #   stripe.deposit       — Stripe deposit webhook (UNTRUSTED COGNITION)
        #   a2a.question_answered— send_a2a_question resumption rail (#1444)
        #   wait.complete        — generic wait reconciler rail (#1860)
        #   workflow rescue      — the six generic sources named by the
        #                          Workflows built-in stalled_work_rescue
        from kestrel_sovereign.signals import RegistrationPolicy
        from kestrel_sovereign.signals.sources.a2a import (
            build_a2a_task_complete_registration,
        )
        from kestrel_sovereign.signals.sources.a2a_task_submitted import (
            build_a2a_task_submitted_registration,
        )
        from kestrel_sovereign.signals.sources.wallet import (
            build_stripe_deposit_registration,
        )
        from kestrel_sovereign.signals.sources.a2a_question_answered import (
            build_a2a_question_answered_registration,
        )
        from kestrel_sovereign.signals.sources.wait import (
            build_wait_complete_registration,
        )
        from kestrel_sovereign.signals.sources.workflow_rescue import (
            build_workflow_rescue_registrations,
        )

        core_source_registrations = [
            build_a2a_task_complete_registration(),
            build_a2a_task_submitted_registration(),
            build_stripe_deposit_registration(),
            build_a2a_question_answered_registration(),
            build_wait_complete_registration(),
            # Core hosts these provider-neutral registrations because the
            # Workflows built-in names them.  The sweep is deliberately the
            # echo-only implementation: an installed domain feature may feed
            # explicit candidates into a run, but it does not replace or own
            # the generic registration lifecycle.
            *build_workflow_rescue_registrations(),
        ]
        core_source_names = [reg.name for reg in core_source_registrations]
        self.signal_registry.register_batch(
            core_source_registrations, RegistrationPolicy.MANDATORY
        )
        # Unregister them in reverse if a later phase fails, so the discarded
        # boot leaves no orphan source registrations behind.
        ctx.on_rollback(
            "core_signal_sources",
            lambda names=list(core_source_names): self._boot_teardown_signal_sources(names),
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


    async def _boot_phase_providers_payer_sync(self, ctx: BootContext) -> None:
        """Phase 3 — storage providers, payer policy, and the sync service. Reads the identity node carried from phase 1 for the constitution-anchor gate."""
        # Initialize storage providers for features (reflection self-model, etc.)
        self.lighthouse_provider = None
        from kestrel_sovereign.storage.sync.service import (
            RemoteTierPolicyContext,
            SyncService,
            _remote_tiers_allowed,
        )

        agent_id = self.did or "default"
        state_dir = Path(self.storage_path).parent if self.storage_path else None
        early_agent_node = ctx.early_agent_node
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
                    # Started a background sync worker — stop it on rollback.
                    ctx.on_rollback(
                        "sync_service", self._boot_teardown_sync_service
                    )
                else:
                    self._sync_service = None

            except Exception as e:
                logging.warning(f"Sync service init failed: {e}", exc_info=True)
                self._sync_service = None
        elif not self._sync_enabled:
            logging.info("Sync service disabled by configuration")


    async def _ensure_agent_node_present(self) -> "GraphNode":
        """Return the agent's identity node, fabricating it only when correct.

        A missing node in the runtime database is fabricated ONLY for a
        genuinely new agent. If on-disk identity material is present, an
        inception happened and its birth record is in a DIFFERENT database
        (#2878) — fabricating a placeholder here would mask the real identity
        (unnamed agent, no ``bootstrap_state``, nothing in Constitutional RAG)
        while every health surface still reports ok. Refuse loudly instead.
        """
        logging.info(f"Getting agent node from storage (agent_id={self.agent_id})")
        agent_node = await self.storage.get_node(self.agent_id)
        logging.info(f"Agent node retrieved: {agent_node is not None}")
        if agent_node is not None:
            return agent_node

        self._refuse_if_birth_record_in_another_database()

        from kestrel_sovereign.storage import GraphNode
        from kestrel_sovereign.storage.privacy_wrapper import (
            acquire_control_plane_capability,
        )
        agent_node = GraphNode(
            node_id=self.agent_id,
            node_type="agent",
            label=f"Agent {self.agent_id}",
            properties={"initialBalance": "100.0"},
        )
        # Trusted control-plane write: agent identity node. The capability
        # admits the durable identity write in a volatile mode (#2672).
        await self.storage.add_node(
            agent_node, capability=acquire_control_plane_capability()
        )
        logging.info("Agent node created")
        return agent_node

    def _refuse_if_birth_record_in_another_database(self) -> None:
        """Refuse to fabricate an agent node when inception's birth record is
        in a database the runtime is not reading (#2878).

        The runtime just proved this agent's DID by loading its on-disk identity
        material (keys + DID document). Those artifacts ARE the evidence that an
        inception happened. If that inception's agent node is nonetheless absent
        from the runtime's database, the birth record was written elsewhere —
        classically a PostgreSQL host whose ``kestrel create`` wrote to the
        per-agent SQLite file the runtime never opens. Fabricating a placeholder
        node here would boot the agent unnamed, with no ``bootstrap_state`` and
        nothing in Constitutional RAG, while ``/health`` still reports ok. Fail
        loudly instead.

        A genuinely new agent — no prior inception, so ``self.identity`` is None
        — is unaffected: creating its node is correct, and this returns.
        """
        if self.identity is None:
            return
        agent_dir = (
            str(Path(self.storage_path).parent) if self.storage_path else "(unknown)"
        )
        # Detailed operator context (directory + backend) goes to the log, not
        # into the public-safe IdentityReadinessError message.
        logging.error(
            "Birth record missing from the configured runtime database "
            "(backend=%s) for agent %s: on-disk identity material is present in "
            "%s but its agent node is absent from the runtime database. "
            "Inception wrote the birth record to a different database. Refusing "
            "to boot with a fabricated placeholder identity.",
            self._db_backend,
            self.agent_id,
            agent_dir,
        )
        from kestrel_sovereign.identity.runtime_identity import (
            IdentityReadinessError,
        )

        raise IdentityReadinessError(
            "birth_record", cause_type="BirthRecordDatabaseMismatch"
        )

    async def _reconcile_birth_record_with_runtime_database(self) -> None:
        """Copy inception's birth record into the runtime database (#2871).

        ``kestrel create`` writes the birth record into the SQLite it opens in
        the agent's directory. A host configured for PostgreSQL then boots the
        agent against PostgreSQL, where the record does not exist — the agent
        came up unnamed, with no ``bootstrap_state`` and nothing in
        Constitutional RAG, while ``/health`` reported ok. The local file stays
        (twelve places read its existence as the fact that a directory IS an
        agent); the record is copied out of it into the database the runtime
        actually reads.

        Runs on every boot, and is a no-op on every ordinary SQLite deployment
        because there the runtime database and the anchor are the same file.
        When a copy IS needed it is idempotent, so an interrupted pass is
        finished by the next boot instead of stranding a half-written record —
        which is why this lives here and not inside inception, where a failed
        copy would leave the anchor on disk, the next ``kestrel create``
        refusing, and nothing left to retry.

        If the record still does not agree with the anchor after a copy, boot
        stops with ``IdentityReadinessError("birth_record")`` rather than
        continuing on the claim that the copy happened.
        """
        if self.identity is None:
            # No prior inception, so there is no birth record to copy. Node
            # creation for a genuinely new agent is correct and happens later.
            return

        from kestrel_sovereign.identity.birth_record import (
            anchor_holds_birth_record,
            diagnose_birth_record,
            diagnose_runtime_birth_record,
            local_anchor_path,
            replicate_birth_record,
            runtime_database_is_the_anchor,
        )

        anchor = local_anchor_path(self.storage_path)
        if anchor is None:
            return
        runtime_db = getattr(self._raw_storage, "db", None)
        if runtime_db is None:
            return
        if runtime_database_is_the_anchor(runtime_db, anchor):
            return

        from kestrel_sovereign.identity.runtime_identity import (
            IdentityReadinessError,
        )
        from kestrel_sovereign.storage.async_database import AsyncDatabase

        # Ask the runtime database first, and open the anchor only if it has
        # something to answer. Opening the anchor runs its migrations and
        # ownership backfills, so a corrupt, read-only or half-deleted
        # kestrel_prime.db would otherwise refuse a boot whose runtime record is
        # complete — a file this host does not need any more deciding whether it
        # may start.
        shortfall = await diagnose_runtime_birth_record(
            runtime_db=runtime_db, agent_did=self.agent_id,
        )
        if not shortfall:
            return

        anchor_db = None
        copy_committed = False
        try:
            # Opening the anchor brings its schema up to date, exactly as a
            # SQLite host would at every boot. Its birth record is only read.
            anchor_db = await AsyncDatabase.sqlite(str(anchor))
            if not await anchor_holds_birth_record(
                anchor_db=anchor_db, agent_did=self.agent_id
            ):
                # Nothing to copy. The verdict is decided by WHAT is short, not
                # by whether an anchor happens to exist — identical damage must
                # not boot on one host and refuse on another.
                logging.warning(
                    "Birth record for %s is incomplete in the runtime database "
                    "(%s) and the local anchor %s holds no record to repair it "
                    "from.",
                    self.agent_id, shortfall.describe(), anchor,
                )
                self._record_birth_record_shortfall(shortfall)
                return

            # The shortfall drives the repair; this comparison is for the
            # operator, naming which rows the anchor can supply. Gating on it
            # instead would silently skip the one condition only the runtime
            # check detects — a governing edge whose node is unreadable.
            divergence = await diagnose_birth_record(
                runtime_db=runtime_db,
                anchor_db=anchor_db,
                agent_did=self.agent_id,
            )
            logging.warning(
                "Birth record for %s is incomplete in the runtime database "
                "(%s); against the local anchor %s: %s. Replicating "
                "(backend=%s).",
                self.agent_id,
                shortfall.describe(),
                anchor,
                divergence.describe() or "no rows missing",
                self._db_backend,
            )
            result = await replicate_birth_record(
                runtime_db=runtime_db,
                anchor_db=anchor_db,
                agent_did=self.agent_id,
            )
            copy_committed = True
            logging.info(
                "Replicated birth record for %s into the runtime database: %s",
                self.agent_id,
                result.describe(),
            )

            # Verify rather than assume. A pass that reports success but left
            # the record incomplete is precisely the failure this whole cluster
            # of issues is about — a durable claim nobody observed.
            #
            # Asked of the runtime alone, deliberately: the question is whether
            # this agent can now be who it is, not whether it matches a frozen
            # snapshot. Re-comparing against the anchor would refuse forever
            # over differences replication correctly declines to make — a
            # reanchored constitution, chunks already indexed here.
            remaining = await diagnose_runtime_birth_record(
                runtime_db=runtime_db, agent_did=self.agent_id,
            )
            self._record_birth_record_shortfall(remaining)
        except IdentityReadinessError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A failed copy is judged by the same rule as a completed one: what
            # is the runtime database actually short of? Every raise site in
            # replicate_birth_record is an ANCHOR-integrity problem — an
            # unwitnessed edge, a file row with no bytes, the anchor failing to
            # open — and so is a transient fault mid-copy. Refusing on all of
            # them unconditionally would brick an agent whose own identity is
            # intact and which boots on origin/main today, and would tell its
            # operator to "re-incept against this backend", which for a dropped
            # connection is destructive advice.
            #
            # ``shortfall`` is the pre-replication diagnosis, so it describes
            # the runtime as this pass found it. Identity gaps still refuse.
            logging.error(
                "Could not replicate the birth record for %s from %s into the "
                "configured runtime database (backend=%s): %s",
                self.agent_id, anchor, self._db_backend, exc, exc_info=True,
            )
            # ``shortfall`` is the pre-replication diagnosis. It is exact for a
            # raise inside the copy's transaction, which rolled back; for one
            # after the commit it can name something already repaired, so
            # re-diagnose when the copy is known to have landed.
            verdict = shortfall
            if copy_committed:
                try:
                    verdict = await diagnose_runtime_birth_record(
                        runtime_db=runtime_db, agent_did=self.agent_id,
                    )
                except Exception:  # noqa: BLE001 - keep the older, safe verdict
                    pass
            self._record_birth_record_shortfall(verdict, retryable=True)
        finally:
            if anchor_db is not None:
                await anchor_db.close()

    def _record_birth_record_shortfall(self, divergence, *, retryable=False) -> None:
        """Decide the verdict on WHAT is short, and leave a trace either way.

        ``identity`` — no agent node, or a fabricated placeholder — is #2878's
        condition, and boot refuses on it here rather than fifty lines later in
        ``_ensure_agent_node_present``, which only ever inspects the node's
        presence. Refusing here also keeps it ahead of every feature side
        effect, and makes the verdict independent of whether a local anchor
        exists: identical damage must not boot on one host and refuse on
        another.

        ``capability`` — no governing edge, an unreadable governing target, no
        retrievable constitution chunks — is NOT refused. Replication has
        already repaired whatever the anchor could supply; what is left is
        unobtainable, and every retry produces the same result. Refusing would
        turn an agent that boots today into one that never boots again, with no
        verb to fix it. It is recorded instead, so ``/health/detailed`` names
        the loss. #2871's defect was that this loss was SILENT; naming it is
        the fix.

        ``retryable`` distinguishes the two ways a capability can be short. A
        completed pass that could not supply it means the anchor does not have
        it and no restart will change that. A pass that DIED — a dropped
        connection, a locked file — is usually transient, and telling the
        operator it is unrepairable sends them to rebuild an anchor when a
        restart would have fixed it.
        """
        self._birth_record_shortfall = list(getattr(divergence, "capability", []))
        self._birth_record_shortfall_retryable = bool(retryable)
        if self._birth_record_shortfall:
            logging.error(
                "Birth record for %s is still incomplete (%s). The agent will "
                "boot without it. %s",
                self.agent_id,
                "; ".join(self._birth_record_shortfall),
                (
                    "The copy failed this pass; a restart may complete it."
                    if retryable
                    else "The local anchor cannot supply it, so no retry will "
                    "— see /health/detailed."
                ),
            )
        identity_failures = list(getattr(divergence, "identity", []))
        if identity_failures:
            from kestrel_sovereign.identity.runtime_identity import (
                IdentityReadinessError,
            )

            logging.error(
                "Birth record for %s does not establish its identity in the "
                "configured runtime database (%s). Refusing to boot.",
                self.agent_id,
                "; ".join(identity_failures),
            )
            raise IdentityReadinessError(
                "birth_record", cause_type="BirthRecordIdentityMissing"
            )

    async def _boot_phase_identity_constitution_features(self, ctx: BootContext) -> None:
        """Phase 4 — identity name, constitution overlay verification (BEFORE feature discovery), feature discovery/enablement/registration, the durable agent node, the startup constitution audit, and LLM payer policy."""
        # Resolve agent name BEFORE features so features can use it
        # (e.g. PeersFeature._get_own_name() reads self._agent_name)
        _agent_node = await self.storage.get_node(self.agent_id)
        if _agent_node:
            self._agent_name = _agent_node.properties.get("name", "Unnamed Agent")
        else:
            self._agent_name = "Unnamed Agent"
        # Upgrade the observability display name from the construction-time
        # floor to the authoritative graph-node name (#2602), so every span
        # emitted for the rest of initialize() — genesis audit, feature init
        # — and by non-fleet agents (single-agent host, CLI/REPL) that never
        # reach the registrar carries the real name. ``_register_agent``
        # overrides this with the registered routing key for fleet members.
        if _agent_node:
            self._set_display_name(self._agent_name)

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
        # Per-agent feature profiles filter via allowed_features — the
        # config bootstrap set unioned with agent-driven enablement deltas
        # from the DB. This is the READ side of the primitive; the production
        # writers (FeatureFeaturesFeature.feature_add/remove + MCPAgent
        # enable/disable, which call persist_feature_enablement) land in the
        # follow-up PRs. With no deltas the effective set == the bootstrap
        # allowlist, so behavior is unchanged today.
        effective_features = await self._effective_allowed_features()
        # Enforce a spawned child's mandate feature ceiling (#2226) on EVERY
        # boot path. The AgentManager spawn path threads mandate.features_
        # allowed into config (#1946), but single-agent server / CLI / direct
        # KestrelAgent boots do not — so without this a restarted child would
        # load features beyond its mandate. The ceiling is read from the
        # durable spawned_by edge and INTERSECTED with any operator allowlist
        # (never widening it). discover_features always force-loads
        # MANDATORY_FEATURES regardless, so this can't drop constitution/
        # security. Fail-closed: a read error propagates (see mandate_reload).
        if self.did and self.storage is not None:
            from kestrel_sovereign.spawn.mandate_reload import (
                read_spawn_features_allowed,
            )

            mandate_features = await read_spawn_features_allowed(
                self.storage, self.did
            )
            # A recorded ceiling is always a non-empty list; None/empty means
            # "no explicit ceiling" (root, legacy, or inherit-from-degenerate-
            # parent) → load all. See read_spawn_features_allowed.
            if mandate_features:
                ceiling = set(mandate_features)
                effective_features = (
                    ceiling
                    if effective_features is None
                    else set(effective_features) & ceiling
                )
        # Disabled deltas must be honored even when there is no bootstrap
        # allowlist (effective is None → discover_features loads all), so a
        # runtime feature_remove survives restart for bootstrap-less agents
        # too. Mandatory features are never in this set.
        disabled_features = await self._disabled_feature_names()
        discovered_features = discover_features(
            self, allowed_features=effective_features
        )
        # Register the feature teardown BEFORE the loop so a failure partway
        # through registration (or in post_all_features_loaded below) rolls back
        # every feature already initialized — each feature.initialize() may have
        # opened connections or started workers.
        ctx.on_rollback("features", self._boot_teardown_features)
        enabled_discovered_features = tuple(
            feature
            for feature in discovered_features
            if feature.name not in disabled_features
        )
        prepared_contributions = self._prepare_feature_contribution_transition(
            enabled_discovered_features
        )
        self._record_contribution_rejections(prepared_contributions)
        for feature, prepared_item in prepared_contributions.activatable(
            enabled_discovered_features
        ):
            await self._register_startup_feature(
                feature,
                prepared_contributions=prepared_item,
            )
        verify_mandatory_feature_set(
            self.features,
            stage="agent readiness",
        )

        # Notify all features that discovery is complete (cross-feature wiring)
        for feature in self.features.values():
            try:
                await feature.post_all_features_loaded(self)
            except Exception as exc:
                from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

                if type(feature).__name__ in MANDATORY_FEATURES:
                    raise MandatoryFeatureReadinessError(
                        type(feature).__name__,
                        "post-load wiring",
                        "could not finish cross-feature wiring",
                    ) from exc
                raise
        logging.info("post_all_features_loaded called for all features")

        # Feature references resolved lazily via properties
        logging.info("Feature references available via lazy properties")

        # Initialize state
        self.conversations = {}
        self.extension = None
        self._session_briefed = False
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

        # Ensure agent graph node exists (fabricates only for a genuinely new
        # agent; refuses when a birth record exists in another database — #2878).
        agent_node = await self._ensure_agent_node_present()

        # A missing durable row or an audit older than 24 hours must be
        # resolved before initialize() can make this agent visible as
        # ready. A failure enters durable Safe Mode; a success advances the
        # deadline but never auto-clears an already-restored Safe Mode.
        await self._audit_constitution_on_startup()

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
                or _exc_name == "PassphraseRequiredError"
                or _exc_name == "DecryptionError"
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


    async def _boot_phase_memory_bootstrap_context(self, ctx: BootContext) -> None:
        """Phase 5 — memory system, command handler, context builder/manager, bootstrap service, and model/embedding-preference hydration (corpus profile provider BEFORE any embedding reconcile)."""
        # Initialize memory system (single source of truth for all memory components)
        logging.info("Creating MemorySystem")
        self.memory_system = MemorySystem(
            storage=self._raw_storage,
            agent_id=self.agent_id,
            # Route durable memory graph writes through the privacy-governing
            # facade and gate the consolidator's direct memory_episodes write,
            # so manual / scheduled consolidation can't leak user-derived
            # memory in a volatile privacy mode (#2672).
            privacy_storage=self.storage,
            # Episode repair (#2856) checks the privacy mode and then awaits
            # decryption and several durable writes. Without the same mutex a
            # transition holds, a flip to EPHEMERAL / ISOLATED can land in that
            # gap and the raw memory_episodes write persists regardless.
            transition_lock=self._get_privacy_transition_lock(),
        )
        # Register teardown BEFORE initialize so a failure partway through
        # ``MemorySystem.initialize()`` (which may open stores / start workers)
        # still shuts it down rather than leaking (#2522 P1).
        ctx.on_rollback("memory_system", self._boot_teardown_memory)
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
            agent_data_path=agent_data_dir,
            db=self._raw_storage.db,
            agent_id=self.agent_id,
            semantic_inference_profile=self.semantic_inference_profile,
            semantic_inference_limits=self.semantic_inference_limits,
            semantic_maintenance_limits=self.semantic_maintenance_limits,
            semantic_answerability_gate=self.memory_system.retriever.answerability_gate,
        )
        # Merge DB-backed bootstrap config (bootstrap_add / bootstrap_remove
        # persistence) into the loader before the first system-prompt
        # assembly (#2135, F099). Storage is up here and no prompt has been
        # built yet, so there is no first-prompt ordering regression.
        await self.context_builder.load_bootstrap_db_config()
        await self._load_or_promote_soul_resource(agent_data_dir)

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
            storage=self.storage,
            capabilities=sorted(self.features.keys()) if getattr(self, "features", None) else None,
            # Serialize the bootstrap service's direct user-content writes
            # (discovery history, user name, SOUL, description) against
            # concurrent privacy-mode transitions (#2672 review P1 race).
            privacy_transition_lock=self._get_privacy_transition_lock(),
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

        # Wire the corpus dominant-embedding-profile provider (#2366) so auto
        # embedding-model resolution prefers continuity with the DB's
        # existing embedding space over catalog order. This MUST be registered
        # BEFORE any path that reconciles embedding capabilities
        # (``_load_embedding_route`` / ``_load_route_embedding_models`` below
        # both call ``reconcile_embedding_capabilities``): reconcile only
        # writes ``caps["embedding_model"]`` when it is empty, so a reconcile
        # that ran without the corpus provider would latch the catalog-first
        # default and later corpus-aware reconciles could not correct it.
        if hasattr(self.llm_service, "set_corpus_embedding_profile_provider"):
            self.llm_service.set_corpus_embedding_profile_provider(
                self._dominant_embedding_profile
            )

        # Load persisted embedding_route knob and register persistence (#2263)
        await self._load_embedding_route()
        self.llm_service.set_embedding_route_persistence_callback(self._persist_embedding_route)

        # Load persisted per-route embedding_model pins and register
        # persistence (#2337) — the runtime equivalent of the TOML
        # embedding_model/embedding_dim keys, set from the embeddings UI.
        if hasattr(self.llm_service, "set_route_embedding_model_persistence_callback"):
            await self._load_route_embedding_models()
            self.llm_service.set_route_embedding_model_persistence_callback(
                self._persist_route_embedding_models
            )

        # Cache the features prompt (built once at session start)
        self._cached_features_prompt = self._build_features_prompt_section()

        # Pre-explore features whose descriptors request direct tools
        # from turn one. This keeps startup generic: individual features
        # decide whether they are meta-orchestration / agent-management
        # surfaces that should bypass the first subagent dispatch.
        self._promote_startup_feature_tools()


    async def _boot_phase_periodic_services_readiness(self, ctx: BootContext) -> None:
        """Phase 6 — periodic services (heartbeat, resume monitor, salvage worker), spawn-mandate reattach, provider-reachability readiness, and the on_agent_ready hooks. Readiness fires only after every prior phase committed."""
        # Initialize heartbeat system (periodic agent self-checks).
        # Registers the heartbeat source with the dispatcher so its
        # ticks route through the signal pipeline (Phase 3 of #889).
        from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner
        from kestrel_sovereign.signals.sources.heartbeat import (
            build_heartbeat_registration,
        )

        from kestrel_sovereign.signals import RegistrationPolicy

        self._heartbeat_config = HeartbeatConfig.from_config()
        _heartbeat_reg = build_heartbeat_registration(
            interval_seconds=self._heartbeat_config.interval_seconds,
            active_hours_start=self._heartbeat_config.active_hours_start,
            active_hours_end=self._heartbeat_config.active_hours_end,
            timezone_name=self._heartbeat_config.timezone,
        )
        self.signal_registry.register_with_policy(
            _heartbeat_reg, RegistrationPolicy.MANDATORY
        )
        ctx.on_rollback(
            "heartbeat_source",
            lambda n=_heartbeat_reg.name: self._boot_teardown_signal_sources([n]),
        )
        self.heartbeat_runner = HeartbeatRunner(self, self._heartbeat_config)
        if self._heartbeat_config.enabled:
            await self.heartbeat_runner.start()
            ctx.on_rollback("heartbeat_runner", self._boot_teardown_heartbeat)

        # Host sleep/wake resilience (#1545). The ResumeMonitor watches
        # for a wall-clock-vs-monotonic divergence (a host suspend) and
        # re-anchors the dispatcher before it emits one auditable
        # `system.resumed` ACTION signal. Reconciliation of volatile durable-
        # delivery sidecars must not depend on that signal persisting.
        # The scheduler and heartbeat detect staleness on their own ticks, so
        # they self-heal independently of this signal — the monitor is the
        # observable spine plus the dispatcher's re-anchor trigger.
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
            return {"recorded": True, "gap_seconds": gap}

        self.signal_registry.register_with_policy(
            build_system_resumed_registration(handler=_resume_action_handler),
            RegistrationPolicy.MANDATORY,
        )
        ctx.on_rollback(
            "system_resumed_source",
            lambda n=RESUME_SOURCE_NAME: self._boot_teardown_signal_sources([n]),
        )

        self._resume_monitor_config = ResumeMonitorConfig.from_config()

        async def _on_resume(gap_seconds: float) -> None:
            from kestrel_sdk.signals import Signal, SignalMode, Visibility

            # This cleanup is safety-critical for volatile raw sidecars. A
            # durable persistence failure is reported by dispatch_signal as a
            # failed result rather than raised, so doing it in the ACTION
            # handler would leave expiry timers frozen across a host suspend.
            self.dispatcher.notify_resume(gap_seconds)
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
            ctx.on_rollback("resume_monitor", self._boot_teardown_resume)

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
                ctx.on_rollback("salvage_worker", self._boot_teardown_salvage)
            except Exception as e:
                logging.warning(f"failed to start salvage worker: {e}")

        # Reattach spawn-mandate enforcement (#2137). initialize() is the single
        # boot path shared by single-agent, multi-agent (AgentManager), and
        # direct-test starts, so registering here — not in AgentManager — means a
        # spawned child's restricted_tools are hard-denied whenever the child
        # runs, reconstructed from the durable spawned_by delegation edge
        # (survives restart). No-op for root agents / spawns with no constraints.
        if self.did and self.storage is not None and self.hooks_manager is not None:
            from kestrel_sovereign.spawn.mandate_reload import (
                read_spawn_mandate,
                register_restriction_hook,
            )

            _spawn_mandate = await read_spawn_mandate(self.storage, self.did)
            if _spawn_mandate is not None:
                if getattr(self, "spawn_mandate", None) is None:
                    self.spawn_mandate = _spawn_mandate
                register_restriction_hook(self.hooks_manager, _spawn_mandate)

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

    # ------------------------------------------------------------------
    # Boot rollback teardown helpers (#2522)
    #
    # Each releases exactly ONE resource a boot phase acquired and is
    # registered via ``ctx.on_rollback`` at the moment of acquisition, so the
    # BootContext can unwind them in reverse (LIFO) order on any phase failure.
    # Each clears its handle only after releasing the resource it protects. A
    # durable dispatcher is the deliberate exception to eager handle clearing:
    # a failed owner release retains both dispatcher and storage so a later
    # lifecycle shutdown can retry safely. The rollback driver logs failures
    # and continues with independent resources.
    # ------------------------------------------------------------------
    async def _boot_teardown_storage(self) -> None:
        """Close the primary DB connection and drop the privacy layer."""
        # Dispatcher teardown owns a runtime-owner release against this very
        # backend.  If its retryable release has not succeeded, retaining both
        # handles is the only safe rollback state: closing here would strand a
        # live owner or let its completion touch a closed SQLite worker.
        if getattr(self, "dispatcher", None) is not None:
            logging.error(
                "boot rollback: retaining storage because durable dispatcher "
                "teardown is still incomplete"
            )
            return
        raw = self._raw_storage
        self._raw_storage = None
        self.storage = None
        self.privacy_agent = None
        if raw is not None and hasattr(raw, "close"):
            await raw.close()

    async def _boot_teardown_task_manager(self) -> None:
        """Close the A2A TaskManager stores."""
        tm = self.task_manager
        self.task_manager = None
        if tm is not None and hasattr(tm, "close"):
            await tm.close()

    async def _boot_teardown_sync_service(self) -> None:
        """Stop the background sync worker."""
        svc = self._sync_service
        self._sync_service = None
        if svc is not None and getattr(svc, "is_running", False):
            await svc.stop()

    async def _boot_teardown_dispatcher(self) -> None:
        """Stop durable dispatcher liveness before its storage is released."""
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            return

        cancelled = False
        retried_failure = False
        while True:
            try:
                await dispatcher.shutdown_durable_delivery()
                break
            except asyncio.CancelledError:
                # BootContext guarantees a complete rollback, but this
                # particular step owns the dispatcher-to-storage ordering.
                # Do not let repeated cancellation leave its owner release
                # alive while the next rollback action closes SQLite.
                cancelled = True
                continue
            except Exception:
                if retried_failure:
                    raise
                # Durable teardown is intentionally retryable.  A failure can
                # land after an owner-release transaction has reached the
                # driver, so retry the dispatcher-owned completion before
                # deciding rollback is unsafe. A persistent failure leaves
                # ``self.dispatcher`` intact, which keeps storage owned and
                # open for a later lifecycle shutdown.
                retried_failure = True
                logging.warning(
                    "boot rollback: durable dispatcher teardown failed; retrying "
                    "before storage release",
                    exc_info=True,
                )

        # Only a successfully released dispatcher may lose its public handle.
        # The following storage rollback can now close the shared backend.
        self.dispatcher = None
        if cancelled:
            raise asyncio.CancelledError()

    async def _boot_teardown_features(self) -> None:
        """Reverse every feature registration made before the failure, LIFO.

        Delegates to the canonical per-feature teardown
        (:meth:`_unregister_feature_runtime`) so boot rollback removes exactly
        what runtime disable does — hooks, A2A TaskManager registrations,
        dynamic tools, owned signal sources, AND wait providers — rather than a
        subset. The old rollback called only ``feature.shutdown()``, leaving a
        rolled-back feature's hooks and ``task:``/``talon:`` wait providers
        registered on a dead agent (kestrel-sovereign#2522). Each feature is
        guarded so one stubborn teardown can't strand the rest.
        """
        for name, feature in reversed(list(self.features.items())):
            try:
                await self._unregister_feature_runtime(feature)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logging.warning(
                    "boot rollback: feature '%s' teardown failed: %s",
                    name,
                    exc,
                )
        self.features = {}

    async def _boot_teardown_memory(self) -> None:
        """Shut down the memory system."""
        ms = getattr(self, "memory_system", None)
        self.memory_system = None
        if ms is not None and hasattr(ms, "shutdown"):
            await ms.shutdown()

    async def _boot_teardown_salvage(self) -> None:
        """Drain and stop the context-manager salvage worker."""
        cm = getattr(self, "context_manager", None)
        if cm is not None and hasattr(cm, "stop_salvage_worker"):
            await cm.stop_salvage_worker()

    async def _boot_teardown_heartbeat(self) -> None:
        """Stop the heartbeat runner."""
        hb = getattr(self, "heartbeat_runner", None)
        if hb is not None and hasattr(hb, "stop"):
            await hb.stop()

    async def _boot_teardown_resume(self) -> None:
        """Stop the resume monitor."""
        rm = getattr(self, "resume_monitor", None)
        if rm is not None and hasattr(rm, "stop"):
            await rm.stop()

    async def _boot_teardown_signal_sources(self, names: list[str]) -> None:
        """Unregister the named signal sources (registry rollback).

        Async so it composes with the ``await``-based rollback driver even
        though ``unregister`` itself is synchronous.
        """
        reg = getattr(self, "signal_registry", None)
        if reg is None:
            return
        for name in names:
            try:
                reg.unregister(name)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logging.warning(
                    "boot rollback: could not unregister source '%s': %s",
                    name,
                    exc,
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

    def _get_privacy_transition_lock(self) -> ReentrantTransitionLock:
        """Return the lock that serializes privacy transitions with active streams.

        Task-reentrant (#2672 review P1): a streamed turn holds it across the whole
        turn, so a durable-identity write dispatched as a tool inside that turn must
        be able to re-enter its own task's lock rather than deadlock on it.
        """
        lock = getattr(self, "_privacy_transition_lock", None)
        if lock is None:
            lock = ReentrantTransitionLock()
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

    async def confirm_privacy_transition(self) -> PrivacyTransitionResult:
        """Apply a privacy transition previously staged as pending confirmation.

        The counterpart to a ``requires_confirmation`` result from
        :meth:`set_privacy_mode_with_effects`. Acquires the transition lock
        (same order as every other privacy path — CONVERSATION before the
        transition lock, so no deadlock) and applies the staged mode atomically
        across all three state holders. A no-op (with an explanatory message) if
        nothing is pending.
        """
        async with self._get_privacy_transition_lock():
            mode = getattr(self, "_pending_privacy_transition", None)
            if mode is None:
                return PrivacyTransitionResult(
                    message="No pending privacy-mode change to confirm.",
                    allows_cloud_llm=privacy_mode_to_config(self._privacy_mode).allows_cloud_llm(),
                )
            self._pending_privacy_transition = None
            result = await self._apply_privacy_mode_locked(mode)
            if result.retryable_conflict:
                # Keep the consented target staged. Retrying confirmation after
                # the active fact finishes must apply that same target, not
                # silently become a "nothing pending" no-op.
                self._pending_privacy_transition = mode
            return result

    async def cancel_privacy_transition(self) -> PrivacyTransitionResult:
        """Discard a privacy transition previously staged as pending confirmation.

        The counterpart to declining a ``requires_confirmation`` result: the
        staged (data-destructive) mode is dropped so a later
        :meth:`confirm_privacy_transition` — from another tab, the
        ``!confirm-privacy-mode`` command, etc. — can't apply a change the user
        declined. A no-op (with an explanatory message) if nothing is pending.
        Nothing else is mutated; the agent stays in its current mode.
        """
        async with self._get_privacy_transition_lock():
            had_pending = getattr(self, "_pending_privacy_transition", None) is not None
            self._pending_privacy_transition = None
            return PrivacyTransitionResult(
                message=(
                    "Pending privacy-mode change discarded."
                    if had_pending
                    else "No pending privacy-mode change to cancel."
                ),
                allows_cloud_llm=privacy_mode_to_config(self._privacy_mode).allows_cloud_llm(),
            )

    async def _set_privacy_mode_with_effects_locked(self, mode: PrivacyMode) -> PrivacyTransitionResult:
        """Evaluate a privacy-mode transition, then apply it (or stage it).

        The privacy agent is the single decision point and is consulted BEFORE
        any state holder changes. A data-destructive transition is staged as
        pending and returned with ``requires_confirmation=True`` — nothing flips
        until :meth:`confirm_privacy_transition`. Every other transition applies
        atomically. This is what prevents the split-state bug where the agent /
        wrapper flipped while the privacy agent stayed behind.
        """
        decision = self.privacy_agent.evaluate_transition(mode)
        if decision.requires_confirmation:
            self._pending_privacy_transition = mode
            return PrivacyTransitionResult(
                message=decision.warning,
                allows_cloud_llm=privacy_mode_to_config(self._privacy_mode).allows_cloud_llm(),
                requires_confirmation=True,
                pending_mode=mode.value,
            )
        # A non-destructive change supersedes any previously-staged pending one.
        self._pending_privacy_transition = None
        return await self._apply_privacy_mode_locked(mode)

    async def _apply_privacy_mode_locked(self, mode: PrivacyMode) -> PrivacyTransitionResult:
        """Atomically apply a privacy-mode transition to all state holders.

        Caller holds the transition lock and has already cleared the
        confirmation gate. Flips agent mode, storage wrapper, and privacy agent
        together, then applies model/voice side effects.
        """
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
            report = await self._purge_ephemeral_leaks(
                reason=f"ephemeral-mode-exit-to-{mode.value}",
            )
            if report.required_sweep_failed:
                # Fail closed (#2673): a required content sweep could not certify
                # "no trace", so we must NOT claim a successful exit to a less
                # restrictive mode. Stay in EPHEMERAL (the safe, more restrictive
                # state) and return an explicit not-applied result — the agent
                # cannot report success. The failure was already audited by
                # _purge_ephemeral_leaks. Resolve the storage error and retry.
                logging.error(
                    "Refusing EPHEMERAL exit to %s: required purge sweep(s) "
                    "failed (%s); staying in EPHEMERAL",
                    mode.value,
                    [r.store for r in report.failed_stores],
                )
                return PrivacyTransitionResult(
                    message=(
                        "Privacy mode change refused: the EPHEMERAL no-trace "
                        "purge could not be certified (a required storage sweep "
                        "failed). Staying in EPHEMERAL; resolve the storage "
                        "error and retry."
                    ),
                    allows_cloud_llm=privacy_mode_to_config(
                        self._privacy_mode
                    ).allows_cloud_llm(),
                    applied=False,
                    purge_failed=True,
                )

        # The storage wrapper owns the linearization guard against durable
        # explicit-fact operations.  Ask it first: if a fact operation is in
        # flight, the transition is refused without leaving the agent and its
        # privacy policy in a split state.
        try:
            self.storage.set_privacy_mode(mode)
        except PrivacyViolationError:
            return PrivacyTransitionResult(
                message=PRIVACY_TRANSITION_RETRY_MESSAGE,
                allows_cloud_llm=privacy_mode_to_config(
                    self._privacy_mode
                ).allows_cloud_llm(),
                applied=False,
                retryable_conflict=True,
            )
        self._privacy_mode = mode
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
            applied=True,
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

    async def _purge_ephemeral_leaks(self, *, reason: str) -> EphemeralPurgeReport:
        """Drive the EPHEMERAL hard-purge defense-in-depth (#767 / #2673).

        Calls into the storage wrapper's ``purge_ephemeral_session`` primitive
        and returns its structured :class:`EphemeralPurgeReport` so the caller
        can distinguish a clean sweep, a leak-and-purge, and a FAILED sweep.

        Three outcomes are audited distinctly (#2673):

        * A raised top-level exception is treated as an UNCERTIFIED purge — the
          report marks every required content store ``FAILED`` (unknown), NEVER a
          zeroed clean breakdown, so the caller fails closed instead of claiming
          a clean transition.
        * A required-sweep failure writes a ``purge_failed`` security audit.
        * A real leak (rows destroyed from a content store) writes the existing
          ``leak_purged`` audit.

        Never raises — losing an audit breadcrumb is preferable to crashing the
        transition/shutdown, but a failure is always surfaced (report + audit).
        """
        try:
            report = await self.storage.purge_ephemeral_session(reason=reason)
        except Exception as e:
            logging.error(
                "ephemeral hard-purge raised; treating as an UNCERTIFIED purge "
                "(required sweeps unknown, session must be treated as leaked): "
                "%s", e,
            )
            report = EphemeralPurgeReport(
                StorePurgeResult(store, PurgeOutcome.FAILED, error=repr(e))
                for store in ("conversation_history", "graph_nodes", "channel_messages")
            )

        if report.required_sweep_failed:
            await self._record_ephemeral_purge_failure_audit(
                reason=reason, report=report,
            )
        elif report.leaked_rows > 0:
            # A genuine privacy leak means data reached tables EPHEMERAL must
            # never write. The observability sink is allowed to hold content-free
            # metric rows during an ephemeral stint (F076), so its swept counts
            # are reported in the breakdown but do NOT trip the leak audit.
            await self._record_ephemeral_leak_audit(
                reason=reason, breakdown=dict(report),
            )
        return report

    async def _purge_ephemeral_on_shutdown(self, *, timeout: float) -> None:
        """Run the EPHEMERAL hard-purge during shutdown, bounded by ``timeout``.

        The process is exiting, so there is no mode transition to block — but a
        failed or timed-out purge must be reported at ERROR severity with durable
        audit evidence, never swallowed as a best-effort success (#2673).

        The WHOLE operation stays bounded by ``timeout``: a durable-audit tail is
        carved out of the supplied budget (``KESTREL_SHUTDOWN_AUDIT_TAIL_FRACTION``)
        so the purge cannot consume the entire window and then leave an *unbounded*
        audit write to overrun the shutdown deadline against a locked/hung audit
        DB. The purge gets the majority of the budget; whatever remains before the
        deadline bounds the post-timeout/error audit write. Re-raises
        ``CancelledError`` so the shutdown method's durable-tail contract is
        preserved.
        """
        loop = asyncio.get_running_loop()
        budget = max(0.0, float(timeout))
        deadline = loop.time() + budget
        # Reserve a slice of the budget for a durable audit write if the purge
        # fails/times out, so the audit is bounded by the SAME window — never an
        # unbounded tail bolted onto the deadline (#2673).
        reserve = budget * KESTREL_SHUTDOWN_AUDIT_TAIL_FRACTION
        purge_timeout = max(0.0, budget - reserve)

        try:
            report = await asyncio.wait_for(
                self._purge_ephemeral_leaks(reason="ephemeral-agent-shutdown"),
                timeout=purge_timeout,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logging.error(
                "ephemeral hard-purge during shutdown TIMED OUT after %.2fs — "
                "the EPHEMERAL no-trace contract could not be certified before "
                "exit", purge_timeout,
            )
            await self._record_bounded_shutdown_purge_failure_audit(
                reason="ephemeral-agent-shutdown",
                failure="shutdown-purge-timeout",
                deadline=deadline,
                loop=loop,
            )
            return
        except Exception as e:
            logging.error(
                "ephemeral hard-purge during shutdown FAILED: %s — the EPHEMERAL "
                "no-trace contract could not be certified before exit", e,
            )
            await self._record_bounded_shutdown_purge_failure_audit(
                reason="ephemeral-agent-shutdown",
                failure=f"shutdown-purge-error: {e!r}",
                deadline=deadline,
                loop=loop,
            )
            return

        if report is not None and report.required_sweep_failed:
            # _purge_ephemeral_leaks already wrote the purge_failed audit; make
            # the shutdown log loud so an operator scanning shutdown output sees
            # the uncertified no-trace contract.
            logging.error(
                "ephemeral hard-purge during shutdown could NOT certify "
                "no-trace: required sweep(s) failed %s — session must be treated "
                "as potentially leaked",
                [r.store for r in report.failed_stores],
            )

    async def _record_bounded_shutdown_purge_failure_audit(
        self, *, reason: str, failure: str, deadline: float, loop,
    ) -> None:
        """Bound the shutdown-time purge-FAILURE audit by the remaining budget.

        During shutdown the durable ``purge_failed`` audit is the operator's only
        evidence that EPHEMERAL's no-trace contract could not be certified — but
        a locked or hung audit database must not overrun the shutdown deadline
        (#2673). ``deadline`` is the absolute ``loop.time()`` by which the whole
        purge operation (including this tail) must finish; the remaining slice
        bounds the write. If the audit cannot be persisted within that slice it
        is abandoned and the lost evidence is logged at ERROR — never awaited
        unbounded. ``CancelledError`` propagates so the shutdown durable tail is
        preserved.
        """
        remaining = max(0.0, deadline - loop.time())
        try:
            await asyncio.wait_for(
                self._record_ephemeral_purge_failure_audit(
                    reason=reason, failure=failure,
                ),
                timeout=remaining,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logging.error(
                "[ephemeral-purge] SECURITY: shutdown purge-failure audit could "
                "NOT be written within the remaining %.2fs of the shutdown budget "
                "(reason=%s failure=%s) — durable evidence was not persisted "
                "before exit; the session must be treated as potentially leaked",
                remaining, reason, failure,
            )

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

    async def _record_ephemeral_purge_failure_audit(
        self, *, reason: str, report: Optional[EphemeralPurgeReport] = None,
        failure: Optional[str] = None,
    ) -> None:
        """Write a durable security audit for an EPHEMERAL purge FAILURE (#2673).

        Distinct from :meth:`_record_ephemeral_leak_audit`: a leak means data was
        found and REMOVED (contract upheld); a purge FAILURE means a required
        sweep could NOT be certified, so the session must be treated as
        potentially still leaking. Routes through the same SecurityFeature
        PermissionStore with ``decision="purge_failed"`` so the operator-visible
        signal is separate from "leak found and removed". When the audit store is
        unavailable — or the write itself fails — the evidence is logged at ERROR
        severity rather than lost silently. Never raises: losing the breadcrumb
        must not crash a transition or shutdown.

        ``report`` (transition path) carries the per-store outcomes; ``failure``
        (shutdown timeout / raised error) is a short reason string when there is
        no report to attach.
        """
        try:
            features = getattr(self, "features", {}) or {}
            feature = features.get("SecurityFeature") or features.get("Security")
            permission_store = (
                getattr(feature, "permission_store", None) if feature else None
            )
        except Exception:
            permission_store = None

        failed_stores: list = []
        breakdown: Dict[str, int] = {}
        if report is not None:
            try:
                failed_stores = [r.store for r in report.failed_stores]
                breakdown = dict(report)
            except Exception:  # noqa: BLE001 - never let report introspection block the audit
                pass

        if permission_store is None:
            logging.error(
                "[ephemeral-purge] SECURITY: purge FAILURE could not be audited "
                "(store unavailable); reason=%s failed_stores=%s failure=%s "
                "breakdown=%s", reason, failed_stores, failure, breakdown,
            )
            return

        try:
            import json as _json
            await permission_store.log_decision(
                feature_name="ephemeral_purge",
                tool_name="hard_purge_guard",
                action="ephemeral_session_close",
                decision="purge_failed",
                args_summary=_json.dumps({
                    "agent_did": getattr(self, "did", None),
                    "reason": reason,
                    "failed_stores": failed_stores,
                    "failure": failure,
                    "breakdown": breakdown,
                }),
            )
        except Exception as e:
            logging.error(
                "[ephemeral-purge] SECURITY: purge-failure audit write failed: "
                "%s (reason=%s failed_stores=%s failure=%s)",
                e, reason, failed_stores, failure,
            )


    async def _effective_allowed_features(self) -> Optional[set]:
        """Bootstrap allowlist unioned with agent-driven enablement deltas.

        ``self._allowed_features`` is the operator's bootstrap set from
        ``multi_agent.toml`` (``None`` = no filter, load all discovered). Agent-
        driven ``feature_add``/``feature_remove`` persist deltas in the DB; this
        applies them so a runtime change survives restart. Mandatory features can
        never be disabled by a delta. With no deltas the result equals the
        bootstrap set, so behavior is unchanged for agents that don't self-manage.
        """
        bootstrap = self._allowed_features
        if bootstrap is None:
            # No allowlist filter in effect; enablement deltas need an explicit
            # bootstrap set to layer onto, so they are a no-op here.
            return None
        try:
            deltas = await self.get_enablement_deltas("feature")
        except Exception as e:  # noqa: BLE001 - never block init on this
            logging.warning("Could not read feature enablement deltas: %s", e)
            return set(bootstrap)
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
        mandatory = set(MANDATORY_FEATURES)
        effective = set(bootstrap)
        for d in deltas:
            if d["state"] == "enabled":
                effective.add(d["name"])
            elif d["state"] == "disabled" and d["name"] not in mandatory:
                effective.discard(d["name"])
        return effective

    async def _disabled_feature_names(self) -> set:
        """Feature names an agent has persisted as ``disabled`` (mandatory excluded).

        Applied as a load-loop skip so a runtime ``feature_remove`` survives
        restart even for agents with no bootstrap allowlist (where
        ``_effective_allowed_features`` returns ``None`` and ``discover_features``
        would otherwise reload everything).
        """
        try:
            deltas = await self.get_enablement_deltas("feature")
        except Exception as e:  # noqa: BLE001 - never block init on this
            logging.warning("Could not read disabled feature deltas: %s", e)
            return set()
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES
        mandatory = set(MANDATORY_FEATURES)
        return {
            d["name"] for d in deltas
            if d["state"] == "disabled" and d["name"] not in mandatory
        }

    async def persist_feature_enablement(
        self, kind: str, name: str, state: str, *, actor: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record an agent-driven enablement delta so it survives restart.

        ``kind`` is ``"feature"`` or ``"mcp_server"``; ``state`` is ``"enabled"``
        or ``"disabled"``. No-op if the store isn't initialized (e.g. a bare
        test agent).
        """
        if kind == "feature" and state == "disabled":
            from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

            if name in MANDATORY_FEATURES:
                raise MandatoryFeatureReadinessError(
                    name,
                    "persistent enablement",
                    "cannot be disabled",
                )
        store = getattr(self, "_feature_enablement_store", None)
        if store is None:
            return
        await store.set_state(
            agent_did=self.did, kind=kind, name=name, state=state,
            actor=actor, metadata=metadata,
        )

    async def get_enablement_deltas(self, kind: Optional[str] = None) -> list:
        """Return this agent's enablement deltas (optionally filtered by kind)."""
        store = getattr(self, "_feature_enablement_store", None)
        if store is None:
            return []
        return await store.get_deltas(self.did, kind)

    async def clear_feature_enablement(self, kind: str, name: str) -> None:
        """Drop a delta, reverting to the bootstrap default for it."""
        store = getattr(self, "_feature_enablement_store", None)
        if store is None:
            return
        await store.clear(self.did, kind, name)

    def _ensure_feature_contribution_runtime(self):
        """Return the contribution controller bound to this agent's registries.

        Bare lifecycle tests attach wait/signal registries after construction,
        while a fully booted agent creates them in the durable-runtime phase.
        Laziness supports both without creating a competing registry.
        """
        from kestrel_sovereign.features.contribution_runtime import (
            FeatureContributionRuntime,
        )
        from kestrel_sovereign.signals import SourceRegistry
        from kestrel_sovereign.waits import WaitRegistry

        signal_registry = getattr(self, "signal_registry", None)
        if signal_registry is None:
            signal_registry = SourceRegistry()
            self.signal_registry = signal_registry
        wait_registry = getattr(self, "wait_registry", None)
        if wait_registry is None:
            wait_registry = WaitRegistry()
            self.wait_registry = wait_registry

        runtime = getattr(self, "feature_contribution_runtime", None)
        if runtime is not None:
            if (
                runtime.wait_registry is wait_registry
                and runtime.source_registry is signal_registry
            ):
                return runtime
            if runtime.active_owners():
                raise RuntimeError(
                    "cannot replace contribution registries while features are active"
                )

        runtime = FeatureContributionRuntime(
            operator_registry=self.operator_registry,
            wait_registry=wait_registry,
            source_registry=signal_registry,
        )
        self.feature_contribution_runtime = runtime
        self.permission_defaults_registry = runtime.permission_defaults_registry
        self.setup_step_registry = runtime.setup_step_registry
        return runtime

    def _record_contribution_rejections(self, transition) -> None:
        """Log and RETAIN the features refused activation.

        Retained, not just logged: a feature that did not load must not be
        indistinguishable from one that loaded and had nothing to do. This is
        what ``/health/detailed`` reports (issue #2951).
        """
        self.rejected_feature_contributions = tuple(transition.rejected)
        for rejection in transition.rejected:
            logging.error(
                "Feature '%s' did not load — %s. The agent is running WITHOUT "
                "it; every other feature and agent is unaffected.",
                rejection.feature_name,
                rejection.reason,
            )

    def _prepare_feature_contribution_transition(self, features):
        """Collect and prevalidate one complete feature activation transition."""
        from kestrel_sovereign.features.contribution_runtime import (
            FeatureContributionCollectionError,
        )
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        try:
            return self._ensure_feature_contribution_runtime().prepare_transition(
                features
            )
        except FeatureContributionCollectionError as exc:
            feature_class_name = type(exc.feature).__name__
            cause = exc.__cause__
            if feature_class_name in MANDATORY_FEATURES:
                if exc.stage == "tool collection":
                    stage = "registration"
                    problem = "could not register its tools"
                else:
                    stage = "contribution registration"
                    problem = "could not register its SDK contributions"
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    stage,
                    problem,
                ) from cause
            if cause is None:  # Defensive: collection failures always chain.
                raise
            raise cause.with_traceback(cause.__traceback__) from cause.__cause__

    async def _shutdown_failed_feature(self, feature: Feature) -> None:
        """Drain EVERY registration of a feature whose registration failed mid-way.

        ``_register_feature`` wires a feature up in stages — ``initialize()``
        (signal sources), ``_wire_feature_hooks`` (HooksManager hooks),
        ``on_enable()``, then ``_wire_feature_a2a`` (A2A TaskManager agent +
        ``set_task_manager``). Any stage after the first can raise having left the
        EARLIER stages' registrations live: an ``on_enable`` failure strands the
        hooks already registered, and an ``_wire_feature_a2a`` failure past
        ``register_agent`` strands both the hooks and the A2A agent. The old
        rollback popped the feature from ``self.features`` FIRST and then called
        only ``feature.shutdown()`` — which reverses the feature's OWN resources
        (signal sources / wait providers / owned tasks / sleep hooks) but NOT the
        agent-side hook / A2A / dynamic-tool registrations — so those leaked, and
        because the feature was already gone from ``self.features`` boot rollback
        (``_boot_teardown_features``) could no longer find it to finish the job
        (#2522 P1).

        Route through the ONE canonical teardown instead, with the feature left
        in ``self.features`` so it drains every registration (``on_disable`` →
        ``shutdown`` → hooks → A2A → dynamic tools, each independent and run
        unconditionally) BEFORE it is dropped — the exact inverse of the wiring
        ``_register_feature`` performs. It is best-effort here: a teardown error
        is logged, never raised, so the ORIGINAL registration failure (or its
        ``MandatoryFeatureReadinessError`` wrap) is what surfaces to the caller.
        Every teardown step is idempotent, so it is safe even when the feature
        only got as far as ``initialize()`` (never added to ``self.features``) or
        a stage left nothing to undo — and ``unload=True`` still drops the
        instance so boot rollback never double-tears-down it (#2522 P1).
        """
        name = getattr(feature, "name", None)
        try:
            await self._unregister_feature_runtime(feature, unload=True)
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logging.warning(
                "boot rollback: failed feature '%s' teardown errored: %s",
                name or type(feature).__name__,
                exc,
            )

    async def _apply_host_feature_config(self, feature: Feature) -> None:
        """Hand a freshly initialized feature the config the operator declared.

        A feature that declares a ``config_schema`` has, until now, had no
        declarative way to be given values: the only route was an HTTP call to
        the configuration endpoint. That is why extracting Talon silently broke
        it on every existing agent — the package correctly requires the host to
        supply explicit paths, and the host had no mechanism with which to
        supply them (#3008). Features that needed configuration either read the
        agent's TOML themselves, re-deriving a file location per feature, or
        were simply unconfigurable without a control-panel visit.

        Core cannot know what any feature's configuration means, and must never
        import a feature package to find out. It does not need to: the registry
        already maps a feature class to the package that owns it, so the block
        is addressed by the *operator-facing* package name — ``[features.talon
        .config]``, the name used by ``kestrel feature enable`` — rather than
        by the class name, which is an implementation detail that happens to
        leak through ``self.features``.

        A declared block that will not apply is a capability gap rather than an
        identity gap: the agent boots and says so, rather than refusing. It is
        safe to be loud instead of fatal here precisely because an unconfigured
        feature now reports itself unconfigured instead of claiming readiness.
        """
        schema = getattr(feature, "config_schema", None)
        if schema is None:
            return
        # ``feature.name`` rather than the concrete class: an isolated feature
        # is loaded as a ProxyFeature and only ``name`` carries the advertised
        # registry class, so keying on type() would silently skip every
        # isolated feature — reintroducing the exact silence this closes.
        declared = self._declared_feature_config(getattr(feature, "name", "") or "")
        if declared is None:
            return
        # An explicitly empty [features.X.config] means "clear it", which is a
        # different instruction from "no block". Treating them alike would let
        # an operator who deliberately emptied the table keep running the
        # settings they just removed — and would skip required-field
        # validation on the way past.
        declared = dict(declared)
        if isinstance(schema, dict):
            # The same rule the HTTP configuration route applies. Validating
            # here means a bad value is refused before any feature persists it.
            validate_feature_config(declared, schema)
        await feature.set_config(declared)

    def _agent_toml_path(self) -> Optional[Path]:
        """The per-agent TOML, beside the agent's database file."""
        if not self.storage_path:
            return None
        return Path(self.storage_path).parent / "kestrel.toml"

    def _declared_feature_config(
        self, feature_class_name: str
    ) -> Optional[Dict[str, Any]]:
        """Read ``[features.<package>.config]`` for one feature class.

        Returns ``None`` when no block is declared, and a (possibly empty)
        mapping when one is — the distinction is load-bearing.
        """
        path = self._agent_toml_path()
        if path is None or not path.exists():
            return None
        from kestrel_sovereign.feature_registry import get_package_for_feature

        package = get_package_for_feature(feature_class_name)
        if package is None:
            return None
        try:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found]
            with open(path, "rb") as handle:
                data = tomllib.load(handle)
        except (OSError, ValueError) as exc:
            # Do NOT degrade to "no declaration". A malformed file is not an
            # absent one: reporting it as absent lets an initialized feature
            # keep configuration the operator believes they replaced, which is
            # the silent divergence this mechanism exists to end.
            raise HostFeatureConfigError(
                f"Feature configuration in {path} could not be read: {exc}"
            ) from exc
        features = data.get("features")
        if not isinstance(features, dict):
            return None
        entry = features.get(package.name)
        if not isinstance(entry, dict):
            # The confusable spelling is the class name, because that is what
            # the HTTP configuration route and ``self.features`` are keyed by.
            # A block written there would be silently ignored, which is the
            # failure this whole mechanism exists to end — so say so.
            mistaken = features.get(feature_class_name)
            if isinstance(mistaken, dict) and "config" in mistaken:
                logging.warning(
                    "%s declares [features.%s.config], which is never read. "
                    "Feature configuration is addressed by package name: use "
                    "[features.%s.config].",
                    path,
                    feature_class_name,
                    package.name,
                )
            return None
        declared = entry.get("config")
        return declared if isinstance(declared, dict) else None

    async def _register_startup_feature(
        self,
        feature: Feature,
        *,
        prepared_contributions=None,
    ) -> bool:
        """Register one discovered feature, quarantining safe prep failures.

        A transient OS failure, an ambiguous legacy/stable custody migration,
        or unsafe process-wide feature configuration makes an optional isolated
        feature unavailable, but does not weaken the rest of the agent.  True
        namespace/ownership violations use the separate
        ``IsolatedRuntimeNamespaceError`` hierarchy and continue to fail the
        hosted agent closed.
        """

        try:
            await self._register_feature(
                feature,
                prepared_contributions=prepared_contributions,
            )
        except Exception as exc:
            # Do not import the optional isolated runtime while registering
            # ordinary core features. An actual ProxyFeature proves the module
            # is already loaded, so classify its narrow quarantine outcomes
            # from that exact module and let every other exception propagate.
            isolated_runtime = sys.modules.get(
                "kestrel_sovereign.features.isolated_runtime"
            )
            proxy_type = getattr(isolated_runtime, "ProxyFeature", None)
            configuration_error = getattr(
                isolated_runtime,
                "IsolatedRuntimeConfigurationError",
                None,
            )
            preparation_error = getattr(
                isolated_runtime,
                "IsolatedRuntimePreparationError",
                None,
            )
            is_isolated_feature = isinstance(proxy_type, type) and isinstance(
                feature, proxy_type
            )
            if not is_isolated_feature:
                raise
            if isinstance(configuration_error, type) and isinstance(
                exc, configuration_error
            ):
                # Derive text through the exception type's closed diagnostic
                # map. Third-party code can raise this public type with an
                # arbitrary base message, so logging ``exc`` could disclose
                # values.
                diagnostic = configuration_error.safe_diagnostic(exc)
                logging.error(
                    "Optional isolated feature '%s' is unavailable because %s; "
                    "other agent features will continue.",
                    getattr(feature, "name", type(feature).__name__),
                    diagnostic,
                )
                return False
            if isinstance(preparation_error, type) and isinstance(
                exc, preparation_error
            ):
                logging.error(
                    "Optional isolated feature '%s' is unavailable because its "
                    "agent-scoped runtime could not be prepared safely; other agent "
                    "features will continue.",
                    getattr(feature, "name", type(feature).__name__),
                )
                return False
            raise
        return True

    async def _register_feature(
        self,
        feature: Feature,
        *,
        prepared_contributions=None,
    ):
        """Register a feature with A2A TaskManager for unified command routing."""
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        feature_class_name = type(feature).__name__
        mandatory = feature_class_name in MANDATORY_FEATURES
        if prepared_contributions is None:
            prepared_contributions = self._prepare_feature_contribution_transition(
                (feature,)
            ).only()
        try:
            await feature.initialize()
        except Exception as exc:
            await self._shutdown_failed_feature(feature)
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    "initialization",
                    "could not initialize",
                ) from exc
            raise
        try:
            await self._apply_host_feature_config(feature)
        except asyncio.CancelledError:
            # CancelledError is a BaseException, so the handler below never
            # sees it. initialize() has already run at this point and the
            # feature is not yet in self.features, so boot rollback cannot
            # find it — its tasks and signal sources would outlive the failed
            # boot. Tear down, then let the cancellation continue.
            with contextlib.suppress(Exception):
                await self._shutdown_failed_feature(feature)
            raise
        except Exception as exc:
            # Deliberately fatal to this feature. A rejected block does NOT
            # leave the feature unconfigured — a feature that validates before
            # replacing its active config (Talon does) keeps running its
            # PREVIOUS configuration, so swallowing this would leave the agent
            # dispatching with paths and policy the operator did not declare
            # and believes they replaced. Silent divergence between declared
            # and running configuration is the whole subject of #3008; it must
            # not be reintroduced by its own fix.
            await self._shutdown_failed_feature(feature)
            logging.error(
                "Feature '%s' rejected the configuration declared in %s: %s. "
                "The feature is NOT loaded, so it cannot run with settings "
                "other than the ones declared.",
                feature_class_name,
                self._agent_toml_path(),
                exc,
            )
            raise
        self.features[feature.name] = feature

        try:
            self._ensure_feature_contribution_runtime().activate(
                prepared_contributions
            )
        except Exception as exc:
            await self._shutdown_failed_feature(feature)
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    "contribution registration",
                    "could not register its SDK contributions",
                ) from exc
            raise

        # Auto-register hooks from get_hooks() with the agent's HooksManager
        try:
            self._wire_feature_hooks(feature)
        except Exception as exc:
            await self._shutdown_failed_feature(feature)
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    "registration",
                    "could not register its hooks",
                ) from exc
            raise

        # Call on_enable lifecycle hook
        try:
            await feature.on_enable()
        except Exception as exc:
            await self._shutdown_failed_feature(feature)
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    "enablement",
                    "could not be enabled",
                ) from exc
            raise

        try:
            self._wire_feature_a2a(feature)
        except Exception as exc:
            await self._shutdown_failed_feature(feature)
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    feature_class_name,
                    "registration",
                    "could not register its tools",
                ) from exc
            raise

    def _wire_feature_hooks(self, feature: "Feature") -> None:
        """Register a feature's ``get_hooks()`` with the agent's HooksManager.

        One source of truth for hook registration, shared by boot
        (:meth:`_register_feature`) and runtime re-enable
        (:meth:`_activate_feature_runtime`). The exact inverse is the hook
        unregister block in :meth:`_unregister_feature_runtime`.
        """
        if not self.hooks_manager:
            return
        for hook in feature.get_hooks():
            self.hooks_manager.register(hook)
            logging.info(
                f"Auto-registered hook '{hook.name}' from feature "
                f"'{feature.name}'"
            )

    async def _register_runtime_feature_permissions(self, feature: "Feature") -> None:
        """Apply one runtime-enabled feature's permissions before exposure.

        Boot uses ``SecurityFeature.post_all_features_loaded`` once discovery
        is complete. A soft-disabled feature's SDK contributions, however, are
        absent from that pass and become active only during runtime enable.
        Route that transition through SecurityFeature's same per-feature seam
        after contribution activation and before hooks, A2A, or direct tools
        are wired, so hard defaults cannot temporarily degrade to fallback ASK.
        """
        from kestrel_sovereign.features.security.feature import SecurityFeature

        security = self.features.get("SecurityFeature")
        if (
            not isinstance(security, SecurityFeature)
            or security is feature
            or not bool(getattr(security, "enabled", True))
        ):
            return
        await security.register_feature_tools(feature.name, feature)

    def _wire_feature_a2a(self, feature: "Feature") -> None:
        """Register a feature as an A2A agent with the TaskManager.

        Collects command prefixes, registers the feature's agent card + handler
        for unified command routing, and wires the task_manager into features
        that consume it. One source of truth shared by boot
        (:meth:`_register_feature`) and runtime re-enable
        (:meth:`_activate_feature_runtime`); the inverse is the
        ``task_manager.unregister_agent`` call in
        :meth:`_unregister_feature_runtime`.
        """
        command_prefixes = {}
        for tool in feature.get_tools():
            if tool.schema.command_prefix:
                command_prefixes[tool.schema.command_prefix] = tool.name
            logging.info(
                f"Registered tool '{tool.name}' from feature '{feature.name}'"
            )

        if self.task_manager:
            agent_card = feature.get_agent_card()
            self.task_manager.register_agent(
                agent_card=agent_card,
                handler=feature,
                command_prefixes=command_prefixes,
            )
            logging.info(
                f"Registered A2A agent '{agent_card.name}' with "
                f"{len(agent_card.skills)} skills"
            )

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

    async def _activate_feature_runtime(
        self,
        feature: "Feature",
        *,
        prepared_contributions=None,
    ) -> None:
        """Bring an already-loaded feature fully live — the inverse of
        :meth:`_unregister_feature_runtime` (kestrel-sovereign#2522 P1).

        The canonical runtime *activation* re-runs the same registration boot
        performed, on the SAME feature instance, so a soft-disabled feature is
        restored end to end:

        * ``initialize()`` — re-registers the feature's owned **signal sources**;
        * contributed permission defaults through SecurityFeature, before any
          callable surface is exposed;
        * hooks from ``get_hooks()`` (via :meth:`_wire_feature_hooks`);
        * the ``on_enable`` lifecycle;
        * the A2A TaskManager agent registration (via :meth:`_wire_feature_a2a`);
        * startup-promoted **dynamic tools** (mirrors
          :meth:`_promote_startup_feature_tools`);
        * ``post_all_features_loaded()`` — re-registers the feature's owned
          **wait providers**;
        * ``on_agent_ready()`` — the ready-phase hook boot fires only after all
          services are live (RestartCoordinator's post-restart wake sweep runs
          here, #1809). Runtime re-enable must fire it too or a re-enabled
          feature silently skips its ready work.

        Precondition: ``feature.initialize()`` must be idempotent — it is re-run
        to restore signal sources a disable detached. Atomic: on any failure
        BEFORE the commit, the partial
        activation is torn back down *softly* (``unload=False`` — the instance
        stays re-enable-able) and the error re-raised, so a failed enable never
        leaves half-registered state. ``on_agent_ready`` runs AFTER the commit
        and is best-effort (mirroring boot): its failure is logged, not fatal,
        so a transient ready-hook error doesn't roll back an otherwise-live
        feature. Used by the public enable endpoint; boot uses
        :meth:`_register_feature` (which additionally handles first-load
        discovery and the mandatory-feature readiness contract).
        """
        if prepared_contributions is None:
            prepared_contributions = self._prepare_feature_contribution_transition(
                (feature,)
            ).only()
        try:
            await feature.initialize()
            # initialize() can reset config a feature does not persist (a
            # volatile-privacy host key, for example), so a disable/enable
            # cycle would otherwise lose the declared value until restart.
            await self._apply_host_feature_config(feature)
            self._ensure_feature_contribution_runtime().activate(
                prepared_contributions
            )
            await self._register_runtime_feature_permissions(feature)
            self._wire_feature_hooks(feature)
            await feature.on_enable()
            self._wire_feature_a2a(feature)
            if getattr(feature, "promote_tools_on_startup", False):
                self._register_explored_feature_tools(feature)
                self._pinned_features.add(feature.tool_name)
            await feature.post_all_features_loaded(self)
            self.features[feature.name] = feature
            feature.enabled = True
            self._cached_features_prompt = self._build_features_prompt_section()
        except Exception:
            # Atomic activation: undo whatever partially registered so a failed
            # enable can't strand hooks / sources / tools. Soft teardown keeps
            # the instance loaded (re-enable-able); its own errors are logged so
            # the ORIGINAL activation error is what surfaces.
            try:
                await self._unregister_feature_runtime(feature, unload=False)
            except Exception as cleanup_exc:  # noqa: BLE001 - best-effort undo
                logging.warning(
                    "Cleanup after failed activation of feature '%s' failed: %s",
                    getattr(feature, "name", type(feature).__name__),
                    cleanup_exc,
                )
            raise

        # Ready-phase lifecycle — fire ONLY after activation committed, so a
        # re-enabled feature gets the same ``on_agent_ready`` signal boot gives
        # it once services are live (#1809). Best-effort per the boot policy: an
        # optional hook, and a failure here logs but never rolls back the
        # now-live feature (kestrel-sovereign#2522 P2).
        ready_hook = getattr(feature, "on_agent_ready", None)
        if ready_hook is not None:
            try:
                await ready_hook(self)
            except Exception as exc:  # noqa: BLE001 - readiness is non-fatal
                logging.warning(
                    "on_agent_ready failed for %s during re-enable: %s",
                    getattr(feature, "name", type(feature).__name__),
                    exc,
                )

    async def _unregister_feature_runtime(
        self, feature: "Feature", *, unload: bool = True
    ) -> None:
        """Reverse *every* runtime registration a fully-registered feature acquired.

        The single canonical inverse of the wiring ``_register_feature`` /
        :meth:`_activate_feature_runtime` and ``post_all_features_loaded`` set
        up. The runtime :meth:`_disable_feature` path, boot rollback
        (:meth:`_boot_teardown_features`), and the public disable endpoint all
        call it, so none can drift or leave a feature's registrations stranded
        (kestrel-sovereign#2522). It removes, in order:

        * the ``on_enable`` lifecycle (via ``on_disable``);
        * the feature's owned **signal sources** and **wait providers** (via
          ``feature.shutdown()`` — base :class:`Feature` unregisters both);
        * the hooks auto-registered from ``get_hooks()``;
        * the A2A TaskManager agent registration;
        * the feature's dynamic tools + hidden-tool bookkeeping.

        ``unload`` controls the endpoint soft-toggle vs. full unload:

        * ``True`` (default — runtime disable + boot rollback): the feature is
          dropped from ``self.features`` (a full unload).
        * ``False`` (public disable endpoint soft-toggle): the feature stays in
          ``self.features`` with ``enabled=False`` so the SAME instance can be
          re-enabled via :meth:`_activate_feature_runtime`.

        Every inverse step is INDEPENDENT of the others, so each runs
        UNCONDITIONALLY even if an earlier one raises (#2522 P2): a failing
        ``on_disable`` must not leave the feature's shutdown / hooks / A2A /
        tools / dropped-from-``features`` cleanup un-run. Errors are collected
        and the first is re-raised *after* all cleanup so callers still see the
        failure. It does NOT enforce the mandatory-feature guard — that is a
        caller policy (runtime disable refuses; boot rollback must tear mandatory
        features down too).
        """
        feature_key = next(
            (key for key, value in self.features.items() if value is feature),
            getattr(feature, "name", None),
        )
        feature_tool_name = getattr(feature, "tool_name", feature_key)
        errors: List[Exception] = []

        # Declarative contributions are exact lifecycle capabilities. Remove
        # them independently even when the feature's imperative hooks fail.
        try:
            runtime = self._ensure_feature_contribution_runtime()
            runtime.deactivate(feature)
        except Exception as exc:  # noqa: BLE001 - cleanup continues below
            errors.append(exc)
            logging.exception(
                "Feature '%s' SDK contribution teardown failed; "
                "continuing with the remaining cleanup",
                feature_key,
            )

        # Lifecycle inverse of on_enable (run during activation). Independent of
        # every teardown step below, so its failure must not skip them.
        try:
            await feature.on_disable()
        except Exception as exc:  # noqa: BLE001 - cleanup continues below
            errors.append(exc)
            logging.exception(
                "Feature '%s' on_disable() failed during teardown; "
                "continuing with the remaining cleanup",
                feature_key,
            )

        # The feature's own resource teardown (signal sources + wait providers).
        try:
            await feature.shutdown()
        except Exception as exc:  # noqa: BLE001 - cleanup continues below
            errors.append(exc)
            logging.exception(
                "Feature '%s' shutdown() failed during teardown; "
                "continuing with the remaining cleanup",
                feature_key,
            )

        # Auto-unregister hooks from get_hooks()
        if self.hooks_manager:
            try:
                for hook in feature.get_hooks():
                    self.hooks_manager.unregister(hook)
                    logging.info(
                        f"Auto-unregistered hook '{hook.name}' from feature "
                        f"'{feature_key}'"
                    )
            except Exception as exc:  # noqa: BLE001 - cleanup continues below
                errors.append(exc)
                logging.exception(
                    "Feature '%s' hook unregistration failed during teardown; "
                    "continuing with the remaining cleanup",
                    feature_key,
                )

        if self.task_manager:
            try:
                self.task_manager.unregister_agent(feature.get_agent_card().name)
            except Exception as exc:
                logging.warning(
                    "Failed to unregister feature '%s' from task manager: %s",
                    feature_key,
                    exc,
                )

        # Dynamic tools + hidden-tool bookkeeping (independent of the above).
        try:
            # Capture the owned tool names before unregister mutates the maps;
            # _tool_context_hidden_tools is reconciled against them below.
            to_remove = [
                name for name, owner in self._tool_to_feature.items()
                if owner == feature_tool_name
            ]
            self.unregister_dynamic_tools(feature_tool_name)
            if isinstance(getattr(self, "_tool_context_hidden_features", None), set):
                self._tool_context_hidden_features.discard(feature_tool_name)
                self._tool_context_hidden_features.discard(feature_key)
                self._tool_context_hidden_features.discard(feature.__class__.__name__)
            if isinstance(getattr(self, "_tool_context_hidden_tools", None), set):
                self._tool_context_hidden_tools.difference_update(to_remove)
        except Exception as exc:  # noqa: BLE001 - cleanup continues below
            errors.append(exc)
            logging.exception(
                "Feature '%s' dynamic-tool teardown failed; "
                "continuing with the remaining cleanup",
                feature_key,
            )

        # Full unload drops the instance; soft-toggle keeps it re-enable-able.
        if unload:
            if feature_key is not None:
                self.features.pop(feature_key, None)
        else:
            feature.enabled = False
        self._cached_features_prompt = self._build_features_prompt_section()

        # Surface the failure only AFTER every independent cleanup step ran.
        if errors:
            raise errors[0]

    async def _disable_feature(self, feature_name: str):
        """Disable a feature and remove its runtime registrations."""
        feature = self.get_feature(feature_name)
        if not feature:
            logging.warning(f"Cannot disable unknown feature: {feature_name}")
            return

        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        feature_class_name = type(feature).__name__
        if feature_class_name in MANDATORY_FEATURES:
            raise MandatoryFeatureReadinessError(
                feature_class_name,
                "runtime disable",
                "cannot be disabled",
            )

        await self._unregister_feature_runtime(feature)
        logging.info(f"Feature '{feature_name}' disabled and removed")

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

    async def _handle_bootstrap(
        self,
        user_input: str,
        session_id: str = None,
        *,
        invocation_context: Optional[LLMInvocationContext] = None,
    ) -> Optional[str]:
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
            await self._persist_assistant_conversation(
                wake_up_msg, session_id=session_id,
            )

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
                    invocation_context=invocation_context,
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
            await self._persist_assistant_conversation(
                response,
                session_id=session_id,
                response=response,
                use_last_identity=True,
            )

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

                    await self._persist_assistant_conversation(
                        completion_msg, session_id=session_id,
                    )
                    await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                    # Reload SOUL.md into context builder
                    if hasattr(self, 'context_builder'):
                        loaded = await self.context_builder.load_canonical_soul_resource()
                        if not loaded:
                            self.context_builder._load_soul_md()
                    logging.info(f"[BOOTSTRAP] Discovery complete with avatar")
                    return completion_msg
                else:
                    # Discovery complete without avatar
                    completion_msg = await self.bootstrap_service.complete_bootstrap()
                    await self._persist_assistant_conversation(
                        completion_msg, session_id=session_id,
                    )
                    await self.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
                    # Reload SOUL.md into context builder
                    if hasattr(self, 'context_builder'):
                        loaded = await self.context_builder.load_canonical_soul_resource()
                        if not loaded:
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

    @bind_async_invocation("invocation_id")
    async def process_input(self, user_input: str, model_override: str = None, session_id: str = None, include_memories: bool = True, caller=None, system_prompt_addendum: str = None, system_prompt_budget_bytes: int = None, anchored_doctrine=None, user_passphrase: str = None, signal_wake: Optional[dict] = None, invocation_context: Optional[LLMInvocationContext] = None, *, invocation_id: Optional[str] = None, invocation_provenance=None) -> str:
        """
        Processes user input by consulting the constitution, retrieving context,
        and generating a response using tool calling for features.

        ``invocation_context`` is the modern identity-passing path (immutable,
        per-request). Prefer it in new consumers; the legacy
        ``LLMService.set_observability_context`` state remains as a fallback.
        When supplied, its fields win over the ambient state; an explicit
        ``session_id`` still fills the context's ``session_id`` slot when the
        caller-supplied context left it empty (see #2614 and
        :func:`resolve_turn_invocation_context`).

        Args:
            user_input: The user's message
            invocation_context: Optional immutable per-request correlation
                identity (session/companion/user) for metering + telemetry.
                Takes precedence over ``set_observability_context`` ambient
                state; ``None`` falls back to that legacy path unchanged.
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
            user_passphrase: Optional per-request passphrase for USER_BYOK agents.
                             Required for PayerKind.USER_BYOK to decrypt provider keys.
            invocation_id: Opaque top-level operation identity. When omitted,
                an id is generated and task-locally bound for tool provenance.
            invocation_provenance: Endpoint-owned authenticated actor and
                transport metadata. This is task-local only; tools cannot
                provide or override it through their arguments.
        """
        logging.info(f"[AGENTIC] process_input called ({len(user_input)} chars)")

        # USER_BYOK credentials are a non-cognitive readiness input. Refresh
        # them before the genesis gate so a Sovereign can supply the auditor's
        # passphrase on the very first turn.
        await self._maybe_refresh_user_byok_resolver(user_passphrase)

        # A restart signal can reach process_input before initialize() creates
        # storage/context. There is no durable receipt to inspect yet, so keep
        # the existing retryable pre-init deferral instead of crashing inside
        # the genesis gate or returning a terminal user-facing block.
        if getattr(self, "storage", None) is None:
            raise RuntimeError(
                "agent not fully initialized: context_manager and storage "
                "unavailable; deferring turn for retry until initialize() completes"
            )

        # GENESIS READINESS: no bootstrap, context build, or provider cognition
        # may precede the durable audit receipt. The shared helper also covers
        # streaming and serializes concurrent first turns (#2470).
        genesis_block = await self._genesis_audit_cognition_block(user_input)
        if genesis_block is not None:
            return genesis_block

        # Reset context stats on session change
        if hasattr(self, 'context_stats') and session_id:
            self.context_stats.check_session(session_id)

        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # SAFE MODE CHECK: If in safe mode, only allow diagnostic commands.
        # ``process_input`` can receive a restart signal while initialize() is
        # still constructing the agent, so absent flags are not restrictions.
        safe_mode = getattr(self, "_safe_mode", False) is True
        audit_pending = (
            getattr(self, "_constitution_audit_pending", False) is True
        )
        if safe_mode or audit_pending:
            command = prefixed_command_token(user_input)
            if command is not None:
                if command not in SAFE_MODE_COMMANDS:
                    return (
                        "🚨 SAFE MODE ACTIVE\\n\\n"
                        "The agent has detected an integrity issue and is operating in restricted mode.\\n"
                        "Only diagnostic commands are available: !safe-mode, !verify-constitution, !reanchor-constitution, !status, !help\\n\\n"
                        "Please contact your administrator to resolve the integrity issue."
                    )
            else:
                restriction = (
                    "a required startup integrity audit"
                    if audit_pending
                    else "an integrity failure"
                )
                return (
                    "🚨 SAFE MODE ACTIVE\\n\\n"
                    f"The agent cannot process queries due to {restriction}.\\n"
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
                command = prefixed_command_token(user_input)
                if command in BOOTSTRAP_ALLOWED_COMMANDS:
                    pass  # Let command handler process these
                elif command is not None:
                    # Never feed command text into discovery. Bootstrap may
                    # persist its input and response or even complete before it
                    # returns, which would leave the operator's transcript out
                    # of sync with durable state when we replace that response.
                    logging.info(
                        "[BOOTSTRAP] Command %s unavailable until onboarding completes",
                        command,
                    )
                    return (
                        f"❌ Command unavailable during bootstrap: {command}\n\n"
                        "Complete onboarding first, or use !skip-discovery to "
                        "finish bootstrap with the default personality."
                    )
                else:
                    bootstrap_response = await self._handle_bootstrap(
                        user_input, session_id,
                        invocation_context=invocation_context,
                    )
                    if bootstrap_response:
                        # Bootstrap persists real conversation rows and must
                        # enter the same privacy-gated memory ingestion path as
                        # every later exchange. Returning here without this
                        # call leaves first-turn importance, emotion, concepts,
                        # and schema routing permanently absent (#2331).
                        await self._post_response_pipeline(
                            user_input, bootstrap_response, session_id
                        )
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
                OI_SPAN_KIND: OI_SPAN_KIND_CHAIN,
                KESTREL_AGENT_NAME: self.agent_name,
                "agent.did": self.did,
                "agent.session_id": session_id or "",
                # #2916: the key the fleet Timeline actually groups on. Omitted
                # (None, not "") when the turn has no session, so a sessionless
                # turn stays absent rather than carrying an empty attribute.
                KESTREL_SESSION_ID: session_id or None,
                "agent.input_length": len(user_input),
            }) as _otel_span:
                # Lifecycle is already entered; call the locked body directly.
                return await self._process_input_traced_locked(
                    user_input, model_override, session_id, _otel_span, include_memories,
                    system_prompt_addendum=system_prompt_addendum,
                    system_prompt_budget_bytes=system_prompt_budget_bytes,
                    anchored_doctrine=anchored_doctrine,
                    signal_wake=signal_wake,
                    invocation_context=invocation_context,
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

    def _conversation_response_identity(
        self,
        response=None,
        *,
        use_last_identity: bool = False,
    ) -> dict:
        """Return the resolved provider/model for the latest LLM response."""
        model = getattr(response, "model", None) if response is not None else None
        provider = (
            getattr(response, "provider", None) if response is not None else None
        )
        if use_last_identity and (not model or not provider):
            llm = getattr(self, "llm_service", None)
            get_identity = getattr(llm, "get_last_response_identity", None)
            if callable(get_identity):
                identity = get_identity() or {}
                model = model or identity.get("model")
                provider = provider or identity.get("provider")
        return {"model": model, "provider": provider}

    async def _persist_assistant_conversation(
        self,
        content: str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        response=None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        use_last_identity: bool = False,
    ) -> None:
        """Persist an assistant row with resolved model/provider columns."""
        identity = self._conversation_response_identity(
            response,
            use_last_identity=use_last_identity or response is not None,
        )
        kwargs = {"session_id": session_id}
        if metadata:
            kwargs["metadata"] = metadata
        resolved_model = model or identity.get("model")
        resolved_provider = provider or identity.get("provider")
        if resolved_model is not None:
            kwargs["model"] = resolved_model
        if resolved_provider is not None:
            kwargs["provider"] = resolved_provider
        await self.privacy_agent.add_conversation(
            "assistant",
            content,
            **kwargs,
        )

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
        signal_wake: Optional[dict] = None,
        invocation_context: Optional[LLMInvocationContext] = None,
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

        # #2674 finding 2: snapshot the enabled POST_RESPONSE hook set at TRUE
        # turn start — BEFORE USER_PROMPT_SUBMIT — the same way the streaming path
        # does (agent/streaming.py). The completion audit below runs EXACTLY this
        # set (pinned to its turn-start modes) via ``_fire_post_response_hook``, so
        # a tool that registers / enables / disables a POST_RESPONSE hook mid-turn
        # cannot change what enforces on THIS turn (the transition takes effect
        # next turn). Capturing BEFORE USER_PROMPT_SUBMIT closes the fail-open a
        # post-hook capture left: a USER_PROMPT hook that unregisters / disables
        # the strict audit could otherwise turn the gate OFF for the very turn it
        # was pinned at start of. The turn-lifecycle lock serializes turns, so
        # nothing else mutates the registry between here and the audit. The
        # streaming ``!continue`` / command-fall-through delegation re-enters this
        # method and snapshots here — one snapshot per path, no double capture.
        audit_hook_snapshot = _snapshot_post_response_hooks(self.hooks_manager)
        # #2674 findings 4 & 5: derive the enforcing (fail-closed) flag from the
        # SAME turn-start snapshot the completion audit runs, so the non-streaming
        # path gates its raw side channels (durable observability preview below,
        # the STOP hook's tool payload) exactly as the streaming ``buffer_audit``
        # does. Independent of the eventual verdict — an enforcing audit buffers
        # the turn whether it ends ALLOW, MODIFY, ASK, or DENY.
        #
        # #2674 finding 1: read the enforcement flag CAPTURED in the snapshot at
        # true turn start (before USER_PROMPT_SUBMIT), NOT a fresh live read — the
        # same lockstep guarantee the streaming ``buffer_audit`` relies on, so a
        # USER_PROMPT mode flip cannot desync this gate from the pinned completion
        # verdict. (This computation already precedes USER_PROMPT_SUBMIT, but
        # sourcing it from the snapshot makes the invariant structural.)
        audit_enforcing = any(
            enforcing for _h, _mode, enforcing in audit_hook_snapshot
        )

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
            # DENY and ASK both block the prompt (F038): an ASK on
            # USER_PROMPT_SUBMIT gates the turn behind approval, so it
            # must not fall through and run.
            blocked = evaluate_blocking_decision(hook_output)
            if blocked is not None:
                return f"[Input rejected: {blocked.reason}]"
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

        # FAIL CLOSED (#2463 review): `_get_governing_constitution()` returns
        # error-sentinel strings ("Error: ...") when the anchored constitution
        # cannot be retrieved/anchored. Those are non-empty strings but NOT a
        # constitution body — proceeding would send the model the error text in
        # place of the governing constitution. Refuse the turn rather than issue
        # the model call, matching the signal dispatcher and streaming path.
        if isinstance(constitution, str) and constitution.lstrip().startswith("Error:"):
            logging.error(
                "process_input refused: _get_governing_constitution returned an "
                "error sentinel (%s); not issuing the model call without a "
                "governing constitution.",
                constitution,
            )
            return (
                "I cannot continue this turn safely: my governing constitution "
                "could not be loaded. Please verify system integrity."
            )

        try:
            history = await self.privacy_agent.get_conversation_history(
                limit=CONTEXT_HISTORY_LIMIT,
                session_id=session_id,
            )
            logging.debug(
                "Conversation history loaded: count=%d session_scoped=%s",
                len(history),
                session_id is not None,
            )
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

        # Resolve the exact schemas before planning so wrapper/tool accounting
        # and final pruning describe the same payload sent below.
        feature_tools = self._build_all_tools()
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
            tools=feature_tools,
            # Span attribution only (#2940): memory retrieval's answerability
            # judge issues its own LLM call, and this is the only place the
            # turn's session is in scope to name it. History filtering is
            # already done — ``conversation_history`` above is the session's.
            session_id=session_id,
        )
        from kestrel_sovereign.agent.semantic_recall import (
            persistence_dependency_metadata,
        )

        semantic_recall_metadata = persistence_dependency_metadata(
            getattr(context_result, "semantic_recall_dependencies", ())
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
        # A COGNITION signal wake (e.g. the restart.completed post-restart
        # verification) arrives here with its rendered instruction template in
        # the user-prompt position, so it persists as a ``user`` turn. Tag it
        # with ``signal_wake`` so the transcript renderer collapses it to a
        # compact "Autonomous wake" chip on reload instead of printing the raw
        # internal instruction block — matching the live path, which shows only
        # the wake's response bubble and never the prompt. rendered_content is
        # untouched, so LLM replay + cache stability are unaffected.
        user_meta = {"sent_form": True, **semantic_recall_metadata}
        if signal_wake:
            user_meta["signal_wake"] = signal_wake
        try:
            await self.privacy_agent.add_conversation(
                "user", wrapped_user,
                metadata=user_meta,
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

        # ``feature_tools`` was snapshotted before context planning so the
        # plan and provider call cannot observe different registry states.

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

        operator_turn = await inject_operator_turn(
            self, messages, context_result, session_id, effective_model, force_local_only
        )

        logging.debug(f"[CONTEXT] Sending {len(messages)} messages to LLM (1 system + {len(context_result.messages)} history + 1 user)")

        # #2530: the operator notice is INJECTED but not yet delivered. An
        # inline system notice is ephemeral by construction (#2009), so it is
        # only beyond loss once the provider has accepted the request — i.e.
        # once this call returns. Anything that throws between here and there
        # loses it, so it settles failed/cancelled and the producer requeues
        # it for the next turn instead of recording a delivery that never
        # happened.
        try:
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
            # #2614: resolve the per-request correlation identity for THIS turn.
            # Explicit ``invocation_context`` wins; otherwise fall back to the
            # legacy ``set_observability_context`` ambient state, with the explicit
            # ``session_id`` filling an empty session slot. Shared helper so the
            # streaming path agrees on precedence.
            resolved_context = resolve_turn_invocation_context(
                self.llm_service, invocation_context, session_id
            )
            # #2674 finding 3: under an enforcing (fail-closed) POST_RESPONSE audit
            # the assistant prose is withheld pending the verdict, so the main
            # provider call's raw prompt/response must not land in durable llm_calls
            # telemetry or the OTel LLM span before that verdict (and, on DENY,
            # never). Carry the content-redaction flag on the FROZEN per-turn context
            # (never global state); it also covers the follow-up tool-synthesis calls
            # in ``_handle_orchestrator_response`` that reuse ``resolved_context``.
            if audit_enforcing and isinstance(resolved_context, LLMInvocationContext):
                resolved_context = _replace_dataclass(
                    resolved_context, redact_content=True
                )
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                force_local_only=force_local_only,
                model_override=effective_model,
                tools=feature_tools if feature_tools else None,
                session_id=session_id,
                keep_trailing_system=operator_turn.keep_trailing_system,
                tool_executor=self._make_inline_tool_executor(session_id or ""),
                invocation_context=resolved_context,
            )
        except BaseException as exc:
            await operator_turn.settle_interrupted(exc)
            raise
        await operator_turn.settle_delivered()

        # Log LLM response timing
        llm_duration = int((time.time() - llm_start) * 1000)
        has_tool_calls = isinstance(response, LLMResponse) and response.has_tool_calls

        logging.info(f"[AGENTIC] LLM response: type={type(response).__name__}, has_tool_calls={has_tool_calls}, tool_count={len(response.tool_calls) if has_tool_calls else 0}")
        if has_tool_calls:
            logging.info(f"[AGENTIC] Tool calls: {[tc.name for tc in response.tool_calls]}")
        elif isinstance(response, LLMResponse) and response.content:
            logging.debug(
                "[AGENTIC] LLM returned text (no tool calls): chars=%d",
                len(response.content),
            )

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
            # LLM output text commands instead of using function calling.
            # #2674 finding 5: ``content_preview`` copies the raw assistant
            # response verbatim into DURABLE observability — before the
            # POST_RESPONSE verdict exists. Under an enforcing (fail-closed) audit
            # the whole turn is withheld pending that verdict, so writing an
            # unaudited preview here would leak exactly what the buffer withholds
            # (and survive even a DENY). Redact the preview to a content-free
            # diagnostic in that mode; advisory / no-audit turns keep the preview
            # (the operator diagnostic for "model ignored function calling").
            content_preview = (
                f"[redacted: {len(response.content)} chars withheld pending audit]"
                if audit_enforcing
                else response.content[:200]
            )
            await self.observability_store.log_error(
                agent_name=self.did,
                error_type="tool_calling_ignored",
                error_message="LLM output contains '!' but no tool_calls - model may be ignoring function calling",
                metadata={
                    "content_preview": content_preview,
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
            invocation_context=resolved_context,
            # #2841: the same budgeted history this turn's FIRST provider call
            # sent. Without it the post-tool continuation answered from a blank
            # conversation while the first call had full context.
            conversation_history=context_result.messages,
        )

        # #2675: assemble the SAME tool/narration evidence the streaming path
        # hands POST_RESPONSE so the deterministic narration check
        # (ResponseAuditHook / #1042 layer 3) sees identical inputs on both
        # transports. Non-streaming previously fired the hook with only
        # ``response_text``, so ``pre_tool_prose`` / ``tool_calls`` /
        # ``tool_results`` were all ``None`` and the same dishonest
        # "tool succeeded" narration streaming catches silently no-op'd here.
        #
        # ``stop_tool_results`` is the multi-iteration envelope list the
        # orchestrator accumulated in place (``{tool_call_id, name, arguments,
        # result}`` per dispatch, ordered by execution across every iteration).
        # Derive the LLM-shaped ``tool_calls`` from it (``id``/``name``/
        # ``arguments``) so calls and results line up by index — the SAME
        # derivation the STOP hook below reuses (built once here).
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
        # Non-streaming equivalent of streaming's ``pre_tool_prose``: the text
        # the model emitted in the SAME completion as its (first) tool calls —
        # what it "said it was about to do" BEFORE observing any tool result.
        #
        # Two shapes reach here, mirroring the two ways a turn can dispatch a
        # tool:
        #
        # * Orchestrator-dispatched (``response.tool_calls`` present): the
        #   pre-tool prose is exactly ``response.content`` on the INITIAL
        #   response. ``_handle_orchestrator_response`` reassigns its own local
        #   ``response``, leaving this scope's ``response`` the pre-tool one.
        # * Inline-executed (codex/openai:plan): the adapter runs kestrel tools
        #   inline and returns ``tool_calls=None`` with the calls on
        #   ``executed_tool_calls`` — so ``has_tool_calls`` is False even though
        #   ``stop_tool_results`` is populated. Here ``response.content`` is the
        #   FULL turn (pre- AND post-tool synthesis), NOT a pre-tool snapshot, so
        #   using it would feed the narration check post-tool text it must not
        #   see. Instead read the marker-bound snapshot the adapter preserved on
        #   ``response.pre_tool_prose`` (#2675) — the streaming path snapshots the
        #   same boundary. Absent (non-codex inline path, or a tool that fired
        #   before any prose) → ``None``; we never invent a boundary.
        #
        # A no-tool turn has no pre/post split (streaming passes ``None`` too).
        # ``analyze_narration`` no-ops on empty prose or empty results.
        if has_tool_calls:
            pre_tool_prose = response.content
        elif stop_tool_results:
            pre_tool_prose = getattr(response, "pre_tool_prose", None)
        else:
            pre_tool_prose = None

        # Fire POST_RESPONSE hooks (e.g., response audit). #2674: route through
        # the SHARED, fail-closed ``_fire_post_response_hook`` — the same gate
        # the streaming path uses — instead of the open-coded DENY-only check
        # this replaced. That old check honored DENY but let ASK / provider
        # failure / timeout fall THROUGH to the raw text (fail-OPEN): an
        # enforcing approval hook returning ASK released unaudited output. It
        # also read the LIVE registry at fire time, so a tool that enabled /
        # disabled / registered a POST_RESPONSE hook mid-turn changed what
        # enforced. ``_fire_post_response_hook`` blocks on DENY *and* ASK (via
        # ``evaluate_blocking_decision``) and runs the turn-start
        # ``audit_hook_snapshot`` pinned to its turn-start modes, so this path —
        # and the streaming ``!continue`` / fall-through delegation that reaches
        # it — obeys the identical fail-closed + snapshot contract. The returned
        # ``_PostResponseText`` carries an explicit ``denied`` verdict the
        # streaming command wrapper reads to decide whether to release parts.
        response_text = await self._fire_post_response_hook(
            response_text, session_id,
            pre_tool_prose=pre_tool_prose,
            tool_calls=stop_tool_calls,
            tool_results=stop_tool_results or None,
            hook_snapshot=audit_hook_snapshot,
        )

        # Store agent response (linked to session for resumed conversations)
        await self._persist_assistant_conversation(
            response_text,
            session_id=session_id,
            metadata=semantic_recall_metadata,
            response=response,
        )

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
            # ``stop_tool_calls`` was derived above (before the POST_RESPONSE
            # fire) from the accumulated ``stop_tool_results`` envelopes so
            # multi-iteration tool flows (model calls A, sees the result, then
            # calls B) produce a payload where tool_calls and tool_results line
            # up by index — building from ``response.tool_calls`` alone would
            # only capture the first iteration (codex review on the initial
            # enrichment). #2675 reuses that single derivation for both hooks.
            # #2674 finding 4: mirror the streaming path's STOP-payload nulling.
            # A STOP hook is an ARBITRARY subscriber (e.g. the reflection
            # ``on_stop`` handler) that may persist or emit its HookInput. Under
            # an enforcing (fail-closed) POST_RESPONSE audit ``tool_calls`` /
            # ``tool_results`` carry raw, unaudited tool envelopes — ids,
            # arguments, full result payloads incl. tool exception strings — the
            # audit never saw and, on a DENY, explicitly rejected. ``response_text``
            # is already the reviewed release (block message on DENY). Null BOTH
            # on every verdict so the non-streaming / ``!continue`` path leaks
            # nothing through the STOP side channel that the streaming path drops.
            # Advisory / no-audit turns keep the full STOP payload.
            stop_hook_tool_calls = None if audit_enforcing else stop_tool_calls
            stop_hook_tool_results = (
                None if audit_enforcing else (stop_tool_results or None)
            )
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
                user_message=user_input,
                response_text=response_text,
                tool_calls=stop_hook_tool_calls,
                tool_results=stop_hook_tool_results,
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
        conv_store = None
        user_msg = None
        assistant_msg = None
        tag_results = {"user": None, "assistant": None}
        try:
            conv_store = getattr(self._raw_storage, 'conversation', None)
            if not conv_store:
                return

            # Only rows written around this serialized turn can be our pair.
            # Never decrypt/materialize an agent's entire lifetime merely to
            # locate two canonical IDs.
            recent = await conv_store.get_full_history_with_ids(limit=20)
            if len(recent) < 2:
                return

            canonical_session_id = None
            if session_id:
                canonical_session_id = await conv_store.resolve_session_id(
                    session_id
                )

            # Find OUR user+assistant pair by canonical content. User turns are
            # persisted in sent form, so compare their raw projection rather
            # than the transport wrapper. Scope to the active session when one
            # is available so identical text in another conversation cannot be
            # selected.
            for msg in reversed(recent):
                msg_meta = msg.get('metadata') or {}
                if (
                    canonical_session_id
                    and str(msg_meta.get('session_id')) != str(canonical_session_id)
                ):
                    continue
                if not assistant_msg and msg.get('role') == 'assistant' and msg.get('content') == response_text:
                    assistant_msg = msg
                elif (
                    not user_msg
                    and msg.get('role') == 'user'
                    and extract_raw_user_content(msg.get('content') or '') == user_input
                ):
                    user_msg = msg
                if user_msg and assistant_msg:
                    break

            if user_msg and assistant_msg:
                tag_results = await self.context_manager.memory_manager.tag_exchange(
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
        snapshot_user_msg_id = user_msg.get('id') if user_msg else None
        snapshot_user_metadata = dict((tag_results or {}).get("user") or {})

        # Phase 1 can fail before a conversation store is acquired. Do not
        # enqueue background work that is guaranteed to dereference None.
        if conv_store is None:
            return

        async def _background_memory_processing():
            try:
                # Temporal pattern detection on recent history window
                if self.memory_system.analyzer:
                    try:
                        window = await conv_store.get_full_history_with_ids(limit=50)
                        patterns = await self.memory_system.analyzer.detect_patterns(
                            messages=window,
                            agent_id=self.agent_id,
                        )
                        if patterns:
                            await self.memory_system.analyzer.save_patterns(patterns)
                            logging.debug(f"Post-response: saved {len(patterns)} temporal patterns")
                    except Exception as e:
                        logging.error(f"Post-response temporal analysis failed: {e}", exc_info=True)

                # Associative linking + schema routing share one canonical
                # stored-message path and the real conversation row ID.
                if snapshot_user_msg_id:
                    try:
                        derived = await self.memory_system.link_and_route_message(
                            message_id=snapshot_user_msg_id,
                            content=user_input,
                            role="user",
                            metadata=snapshot_user_metadata,
                        )
                        if derived:
                            await conv_store.update_message_metadata(
                                snapshot_user_msg_id, derived
                            )
                    except Exception as e:
                        logging.error(f"Post-response memory graph processing failed: {e}", exc_info=True)

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
        # ``asyncio.Task`` carries no creation time, and age is what separates
        # "busy" from "wedged" when something inspects this set — the restart
        # coordinator's idle gate reports it so an operator can tell a task
        # that appeared once from one stuck for hours (#2665). Stamped at the
        # single chokepoint that owns the set, so nothing has to maintain a
        # parallel map that could outlive the task.
        task._kestrel_started_at = time.monotonic()  # type: ignore[attr-defined]
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

    async def _complete_durable_shutdown_continuation(
        self,
        dispatcher,
        storage,
        storage_preclose,
    ) -> None:
        """Finish dispatcher release before touching its shared storage.

        This task is created only after the bounded tail has spent its
        dispatcher guard.  It is deliberately unbounded and joinable: the
        durable dispatcher already owns a shielded, retryable release task,
        and storage cannot safely close until that task has actually finished.
        The production lifecycle owner joins this continuation before removing
        the agent or releasing its budget.
        """
        await dispatcher.shutdown_durable_delivery()
        wait_for_owner_release = getattr(
            dispatcher, "wait_for_durable_shutdown_release", None
        )
        if callable(wait_for_owner_release):
            await wait_for_owner_release()
        if storage_preclose is not None:
            await storage_preclose()
        if storage is not None and hasattr(storage, "close"):
            await storage.close()

    async def _ensure_durable_shutdown_continuation(self, dispatcher) -> asyncio.Task:
        """Return the single agent-owned dispatcher-to-storage continuation."""
        async with self._durable_shutdown_continuation_lock:
            continuation = self._durable_shutdown_continuation
            if continuation is not None and not continuation.done():
                return continuation
            if continuation is not None and not continuation.cancelled():
                # A completed successful continuation has already performed
                # its storage close.  Reuse it so a second shutdown cannot
                # duplicate that close.
                if continuation.exception() is None:
                    return continuation

            storage = self.storage
            continuation = asyncio.create_task(
                self._complete_durable_shutdown_continuation(
                    dispatcher,
                    storage,
                    _storage_preclose(storage),
                ),
                name=f"agent_durable_shutdown_continuation:{self.did}",
            )
            # Always retrieve an unjoined failure.  The task remains retained
            # for a later lifecycle retry, rather than disappearing as an
            # unobserved "Task exception was never retrieved" warning.
            continuation.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
            self._durable_shutdown_continuation = continuation
            return continuation

    async def wait_for_shutdown_completion(self) -> None:
        """Join deferred durable cleanup before releasing this agent.

        Normal shutdown has no continuation because its bounded tail closes
        storage directly.  If dispatcher release outlives that guard, this
        joins the agent-owned continuation that performs the only permitted
        subsequent storage close.  ``shield`` preserves the work if an outer
        lifecycle wait is cancelled; a later owner can join the same task.
        """
        retried_failure = False
        while True:
            continuation = self._durable_shutdown_continuation
            if continuation is None:
                return
            try:
                await asyncio.shield(continuation)
                return
            except asyncio.CancelledError:
                # A continuation can itself be cancelled (for example while
                # releasing the runtime owner).  That is a retryable cleanup
                # failure.  A cancellation of *this* waiter, on the other
                # hand, must remain visible to the outer lifecycle owner,
                # which keeps the continuation alive with ``shield``.
                if not continuation.done() or not continuation.cancelled():
                    raise
                failure: BaseException = asyncio.CancelledError()
            except Exception as exc:
                failure = exc

            if retried_failure:
                raise RuntimeError(
                    "Durable shutdown continuation failed after its retry; "
                    "dispatcher release and storage close are unconfirmed"
                ) from failure

            retried_failure = True
            dispatcher = getattr(self, "dispatcher", None)
            if dispatcher is None or not hasattr(
                dispatcher, "shutdown_durable_delivery"
            ):
                raise RuntimeError(
                    "Durable shutdown continuation failed without a dispatcher "
                    "available for its required retry"
                ) from failure
            logging.warning(
                "Durable shutdown continuation failed; retrying dispatcher "
                "release and storage close before lifecycle exit",
                exc_info=(type(failure), failure, failure.__traceback__),
            )
            await self._ensure_durable_shutdown_continuation(dispatcher)

    def handoff_shutdown_to_reaper(
        self, shutdown_task: "asyncio.Future[object]"
    ) -> "asyncio.Future[None]":
        """Transfer a timed-out lifecycle join to a retained control-plane reaper.

        A durable cognition turn can legally outlive the user-facing shutdown
        deadline: its delivery lease and the storage connection it uses must
        remain with that original execution until it either commits or is
        safely retried.  Lifecycle callers nevertheless need a finite control
        plane operation so they can withdraw routing and revoke a delegated
        budget.  This method gives them one explicit handoff boundary.

        The returned coroutine *never* closes storage early.  It first joins
        the exact ``shutdown_task`` the caller already owns, then joins this
        agent's continuation (if the bounded tail created one).  An
        :class:`~kestrel_sovereign.multi_agent.agent_manager.AgentManager`
        retains that coroutine in its observable quarantine registry rather
        than awaiting it on the DELETE/restart path.
        """
        if not isinstance(shutdown_task, asyncio.Future):
            raise TypeError("shutdown reaper handoff requires an asyncio future")

        async def reap() -> None:
            _cancelled, failure = await await_lifecycle_task_completion(shutdown_task)
            if failure is not None and not isinstance(
                failure, asyncio.CancelledError
            ):
                raise failure
            await self.wait_for_shutdown_completion()

        return asyncio.create_task(
            reap(), name=f"agent_shutdown_quarantine_reaper:{self.did}"
        )

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

    def resolve_effective_name(
        self, agent_node: Any = None, *, default: Optional[str] = None
    ) -> Optional[str]:
        """The name the agent currently answers to for this session.

        The live in-memory ``_agent_name`` is the session source of truth: every
        rename updates it, INCLUDING a volatile-mode rename that intentionally
        skips the durable graph/metadata writes (#2672 review P2). The stored
        ``agent`` graph node therefore lags the live name after a volatile
        rename, so prefer the live name and only fall back to the stored node,
        then ``default``.

        Centralizes effective-identity resolution so the PATCH /api/identity
        response, GET /api/identity, and the A2A agent card all report the same
        (live) name — instead of the endpoint and card disagreeing after a
        volatile rename by re-reading the stale durable node (#2672 review P2).
        """
        live = getattr(self, "_agent_name", None)
        if isinstance(live, str) and live.strip():
            return live
        if agent_node is not None:
            props = getattr(agent_node, "properties", None) or {}
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return default

    async def get_agent_card(self) -> "AgentCard":
        """
        Generate an AgentCard for this agent (for A2A discovery).
        Returns agent identity, capabilities, and available skills.
        """
        from kestrel_sovereign.a2a.agent_card import AgentCard, AgentCapabilities, AgentProvider

        # Get agent name from storage node if available
        agent_name = "Kestrel Agent"
        agent_description = "Constitutional AI Agent with sovereign memory"

        agent_node = None
        if self.storage:
            try:
                agent_node = await self.storage.get_node(self.agent_id)
                if agent_node and agent_node.properties:
                    agent_description = agent_node.properties.get("description", agent_description)
            except Exception as e:
                logging.warning(f"Could not load agent node for card generation: {e}")

        # Prefer the live session name so a volatile-mode rename (which skips the
        # durable node write) is reflected on the card instead of the stale stored
        # name (#2672 review P2).
        agent_name = self.resolve_effective_name(agent_node, default=agent_name)

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
        """Properly clean up all agent resources including async MCP connections.

        The whole method is a *fallible prefix* wrapped in one try/finally.
        The prefix stops every agent-owned worker and every feature; the
        ``finally`` runs the safety-critical durable tail (background-task
        cleanup, memory shutdown, final sync snapshot, storage close).

        Cancellation contract (#2409): agent.shutdown() is wrapped in
        ``asyncio.wait_for(..., timeout=SHUTDOWN_TIMEOUT)`` by the CLI /
        server / AgentManager paths. If the outer timeout cancels us
        *anywhere* in the prefix — while stopping the heartbeat, the resume
        monitor, the salvage worker, SecurityFeature, or any other feature —
        we must NOT skip the durable tail. So the ENTIRE prefix lives inside
        the try, the durable tail always runs in the ``finally``, and
        CancelledError is re-raised only after that tail completes (never
        reporting a successful shutdown post-cancellation).

        Deadline composition is kept coherent with the production outer
        deadline: the prefix shares a single internal budget
        (``KESTREL_AGENT_SHUTDOWN_TIMEOUT_S``, defaulting to the same value
        the outer ``wait_for`` uses) minus a durable-tail reserve, and every
        step is bounded against it. Features additionally fair-divide the
        remaining budget so a single hung early feature can consume only its
        own slice and cannot starve a later feature — or the durable tail.
        """
        loop = asyncio.get_running_loop()
        dispatcher = getattr(self, "dispatcher", None)
        inherited_durable_admission = getattr(
            dispatcher, "has_live_durable_admission_in_current_context", None
        )
        if callable(inherited_durable_admission) and inherited_durable_admission():
            # A signal handler can call agent.shutdown(), whose bounded tail
            # creates child tasks.  ContextVars copy the parent dispatch's
            # durable admission into those children, so allowing this shutdown
            # to begin would make dispatcher teardown wait for the dispatch
            # that is itself awaiting shutdown.  Refuse before any prefix or
            # storage teardown runs; an external lifecycle owner may retry
            # once the dispatch has released its admission.
            raise RuntimeError(
                "Cannot shut down an agent from a live durable signal operation"
            )
        storage_close_timeout = _minimum_storage_close_timeout(self.storage)
        # A feature may lazily create a file-backed SQLAlchemy factory during
        # its shutdown.  Reserve that backend-declared *potential* close
        # window now, before the feature sweep can populate the cache.
        storage_preclose_reservation = (
            _minimum_storage_potential_preclose_timeout(self.storage)
        )
        prefix_budget, tail_reserve = _resolve_shutdown_budget(
            self._durable_tail_minimum_budget(
                storage_close_timeout, storage_preclose_reservation
            )
        )
        # Shared deadline for the fallible prefix. Reserve headroom so the
        # durable tail runs WITHIN the outer deadline rather than relying on
        # the outer wait_for cancellation to trigger it.
        prefix_deadline = loop.time() + prefix_budget

        def _remaining() -> float:
            return max(0.0, prefix_deadline - loop.time())

        security_feature = self.features.get("SecurityFeature")
        mcp_feature = self.mcp_agent

        # ONE ordered count/budget across EVERY fallible prefix operation
        # (#2409 review). Each op gets a fair share of the LIVE remaining
        # budget divided by the number of ops still pending (itself included),
        # so a single hung EARLY op — ephemeral purge, heartbeat, resume,
        # salvage, SecurityFeature, any feature, MCP, LLM, or TaskManager —
        # can consume only its slice and never starves a later op or the
        # durable tail. This is what makes the internal deadline composition
        # coherent with the production outer wait_for.
        remaining_features = [
            (name, feature)
            for name, feature in self.features.items()
            if feature is not security_feature
            and feature is not mcp_feature
            and hasattr(feature, "shutdown")
        ]

        has_ephemeral = getattr(self, "_privacy_mode", None) == PrivacyMode.EPHEMERAL
        has_heartbeat = bool(
            getattr(self, "heartbeat_runner", None)
        )
        has_resume = bool(getattr(self, "resume_monitor", None))
        has_salvage = bool(getattr(self, "context_manager", None))
        has_security = bool(security_feature and hasattr(security_feature, "shutdown"))
        has_mcp = bool(self.mcp_agent and hasattr(self.mcp_agent, "shutdown"))
        has_llm = bool(self.llm_service and hasattr(self.llm_service, "close"))
        has_task_mgr = bool(self.task_manager and hasattr(self.task_manager, "close"))

        pending_ops = (
            int(has_ephemeral)
            + int(has_heartbeat)
            + int(has_resume)
            + int(has_salvage)
            + int(has_security)
            + len(remaining_features)
            + int(has_mcp)
            + int(has_llm)
            + int(has_task_mgr)
        )

        def _step_budget() -> float:
            """Fair share of the live remaining budget for the next op.

            Consumes one pending-op slot each call: the returned bound is
            ``remaining / pending`` (capped by the per-feature cap), so an op
            that hangs burns only its slice and the ops after it still get a
            fair division of whatever budget is left.
            """
            nonlocal pending_ops
            remaining = _remaining()
            share = remaining / pending_ops if pending_ops > 1 else remaining
            if pending_ops > 0:
                pending_ops -= 1
            return min(KESTREL_FEATURE_SHUTDOWN_TIMEOUT_S, share)

        shutdown_cancelled = False
        try:
            # EPHEMERAL hard-purge defense-in-depth (#767 / #2673). If the agent
            # process is exiting while still in EPHEMERAL, the session is
            # closing — fire the hard-purge so any leak doesn't survive the
            # restart. Shutdown stays bounded, but a failed or timed-out purge is
            # reported at ERROR severity with durable audit evidence rather than
            # swallowed as a best-effort success.
            if getattr(self, "_privacy_mode", None) == PrivacyMode.EPHEMERAL:
                await self._purge_ephemeral_on_shutdown(timeout=_step_budget())

            # Stop heartbeat runner
            if hasattr(self, 'heartbeat_runner') and self.heartbeat_runner:
                try:
                    await asyncio.wait_for(
                        self.heartbeat_runner.stop(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logging.warning("Stopping heartbeat timed out")
                except Exception as e:
                    logging.warning(f"Error stopping heartbeat: {e}")

            # Stop resume monitor (#1545)
            if hasattr(self, 'resume_monitor') and self.resume_monitor:
                try:
                    await asyncio.wait_for(
                        self.resume_monitor.stop(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logging.warning("Stopping resume monitor timed out")
                except Exception as e:
                    logging.warning(f"Error stopping resume monitor: {e}")

            # Stop C / #1311 durable salvage worker. Drains in-flight
            # summary tasks; the janitor catches up the rest on next start.
            if hasattr(self, "context_manager") and self.context_manager:
                try:
                    await asyncio.wait_for(
                        self.context_manager.stop_salvage_worker(),
                        timeout=_step_budget(),
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logging.warning("Stopping salvage worker timed out")
                except Exception as e:
                    logging.warning(f"Error stopping salvage worker: {e}")

            # Shutdown security feature if it exists
            if security_feature and hasattr(security_feature, 'shutdown'):
                try:
                    await asyncio.wait_for(
                        security_feature.shutdown(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    logging.warning("Security feature shutdown timed out")
                except (AttributeError, TypeError, ConnectionError) as e:
                    logging.warning(f"Error during security shutdown: {e}")
                except Exception as e:
                    logging.warning(f"Error during security shutdown: {e}", exc_info=True)

            # Shutdown every other loaded feature via its standard lifecycle
            # (#2409). Feature-owned background workers (health, delivery,
            # scheduler, ...) only cancel/await when their own shutdown()
            # runs; whole-agent shutdown previously reached only
            # SecurityFeature, leaking those workers past process exit.
            # Skip SecurityFeature (already stopped above) and the MCP agent
            # (handled by its own block below) so neither is double-stopped,
            # and run here — before storage teardown — so a feature can still
            # flush against live storage without racing it. ``remaining_features``
            # was computed above and folded into the shared ``pending_ops``
            # count, so each feature already fair-divides the SAME live budget
            # as every other prefix op (heartbeat, MCP, LLM, TaskManager, ...).
            for feature_name, feature in remaining_features:
                per_feature = _step_budget()
                try:
                    # Some lifecycle owners (notably isolated SDK facades)
                    # must keep an exact stop coroutine alive after this
                    # fair-share deadline: it may own the only subprocess
                    # handle.  The optional wrapper gives such a feature the
                    # canonical agent-deadline signal, so it can retain that
                    # task and return cancellation promptly instead of
                    # consuming later features' and the durable tail's time.
                    # Look on the class, not the instance: test doubles and
                    # dynamic adapters can fabricate arbitrary attributes,
                    # which must not be mistaken for this lifecycle contract.
                    bounded_shutdown = getattr(
                        type(feature), "shutdown_with_agent_deadline", None
                    )
                    prepare_bounded_shutdown = getattr(
                        type(feature), "prepare_shutdown_with_agent_deadline", None
                    )
                    if callable(prepare_bounded_shutdown):
                        # A zero fair share is still a terminal shutdown
                        # request.  Establish isolated runtime ownership
                        # before wait_for can cancel its coroutine prior to
                        # the first instruction in its async body.
                        feature.prepare_shutdown_with_agent_deadline()
                    shutdown_operation = (
                        feature.shutdown_with_agent_deadline()
                        if callable(bounded_shutdown)
                        else feature.shutdown()
                    )
                    await asyncio.wait_for(
                        shutdown_operation, timeout=per_feature
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        "Feature '%s' shutdown exceeded %.2fs; abandoning "
                        "and continuing sweep",
                        feature_name,
                        per_feature,
                    )
                except asyncio.CancelledError:
                    # Whole-agent shutdown is being cancelled. Stop the sweep
                    # and propagate to the durable tail via `finally`.
                    logging.warning(
                        "Feature '%s' shutdown cancelled; running durable "
                        "cleanup before propagating",
                        feature_name,
                    )
                    raise
                except Exception as e:
                    logging.warning(
                        "Error during feature '%s' shutdown: %s",
                        feature_name,
                        e,
                        exc_info=True,
                    )

            # Shutdown MCP agent if it exists
            if self.mcp_agent and hasattr(self.mcp_agent, 'shutdown'):
                try:
                    await asyncio.wait_for(
                        self.mcp_agent.shutdown(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    logging.warning(
                        "MCP shutdown cancelled; running durable cleanup "
                        "before propagating"
                    )
                    raise
                except asyncio.TimeoutError:
                    logging.warning("MCP shutdown timed out")
                except (AttributeError, TypeError, ConnectionError) as e:
                    logging.warning(f"Error during MCP shutdown: {e}")
                except Exception as e:
                    logging.warning(f"Error during MCP shutdown: {e}", exc_info=True)

            # Close LLM service async clients
            if self.llm_service and hasattr(self.llm_service, 'close'):
                try:
                    await asyncio.wait_for(
                        self.llm_service.close(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    logging.debug("LLM service close cancelled")
                    raise
                except asyncio.TimeoutError:
                    logging.warning("LLM service close timed out")
                except (AttributeError, TypeError, ConnectionError) as e:
                    logging.warning(f"Error closing LLM service: {e}")
                except Exception as e:
                    logging.warning(f"Error closing LLM service: {e}", exc_info=True)

            # Close TaskManager stores (critical for preventing thread leaks)
            if self.task_manager and hasattr(self.task_manager, 'close'):
                try:
                    await asyncio.wait_for(
                        self.task_manager.close(), timeout=_step_budget()
                    )
                except asyncio.CancelledError:
                    logging.debug("TaskManager close cancelled")
                    raise
                except asyncio.TimeoutError:
                    logging.warning("TaskManager close timed out")
                except (AttributeError, TypeError, ConnectionError) as e:
                    logging.warning(f"Error closing TaskManager: {e}")
                except Exception as e:
                    logging.warning(f"Error closing TaskManager: {e}", exc_info=True)
        except asyncio.CancelledError:
            shutdown_cancelled = True
        finally:
            # Re-read the actual cache at the tail boundary.  The original
            # potential reservation remains the floor, so a factory created
            # by feature shutdown cannot spend the primary backend's worker
            # close window.  This stays inside ``tail_reserve`` — and thus the
            # production outer shutdown bound — because that reservation was
            # composed before the prefix started.
            storage_preclose_timeout = max(
                storage_preclose_reservation,
                _minimum_storage_preclose_timeout(self.storage),
            )
            # Durable cleanup tail — safety-critical, always runs even under
            # cancellation. It has its OWN finite, honest deadline
            # (``tail_reserve``) so a tail step that hangs or suppresses
            # cancellation cannot make the outer wait_for / CLI "forcing exit"
            # branch unreachable. Returns whether cancellation was observed
            # and whether any step degraded (abandoned/errored).
            tail_cancelled, tail_degraded = await self._run_durable_shutdown_tail(
                tail_reserve,
                storage_close_timeout=storage_close_timeout,
                storage_preclose_timeout=storage_preclose_timeout,
            )
            if tail_cancelled:
                shutdown_cancelled = True

        # A timed dispatcher guard transfers ownership of the remaining
        # dispatcher-drain -> storage-close sequence to one agent-owned task.
        # On an ordinary shutdown we join it here, so this method never reports
        # completion while the shared SQLite worker remains open.  If an outer
        # lifecycle timeout already cancelled us, leave the continuation
        # shielded and let that lifecycle owner join it before removing the
        # agent; re-raising below preserves the cancellation contract.
        dispatcher_owner_fenced = bool(
            getattr(dispatcher, "durable_shutdown_owner_fenced", False)
        )
        if not shutdown_cancelled and not dispatcher_owner_fenced:
            try:
                await self.wait_for_shutdown_completion()
            except asyncio.CancelledError:
                shutdown_cancelled = True
        elif dispatcher_owner_fenced:
            # The continuation owns storage close after the hostile cognition
            # task settles. Returning degraded keeps shutdown bounded without
            # making that live task peer-reclaimable in this process.
            tail_degraded = True
        elif (
            self._durable_shutdown_continuation is not None
            and not self._durable_shutdown_continuation.done()
        ):
            # We must propagate the caller's timeout, but must not describe
            # shutdown as complete while its owned dispatcher-to-storage tail
            # is still running.
            tail_degraded = True

        if shutdown_cancelled:
            # Never report success after cancellation: re-raise so the outer
            # asyncio.wait_for surfaces the timeout/cancellation. Report the
            # tail honestly — only say cleanup "completed" when it actually
            # did; if a step was abandoned, say so.
            if tail_degraded:
                logging.warning(
                    "Kestrel Agent shutdown cancelled; durable cleanup ran but "
                    "was DEGRADED (one or more steps abandoned — see warnings "
                    "above); propagating cancellation."
                )
            else:
                logging.warning(
                    "Kestrel Agent shutdown cancelled; durable cleanup "
                    "completed; propagating cancellation."
                )
            raise asyncio.CancelledError()

        if tail_degraded:
            logging.warning(
                "Kestrel Agent async shutdown complete, but durable cleanup "
                "was DEGRADED — one or more steps were abandoned "
                "(see warnings above)."
            )
        else:
            logging.info("Kestrel Agent async shutdown complete.")

    def _durable_tail_minimum_budget(
        self,
        storage_close_timeout: float,
        storage_preclose_timeout: float = 0.0,
    ) -> float:
        """Return the durable-tail floor needed for a storage close contract.

        SQLite's close must retain enough time for its aiosqlite worker to
        exit.  The preceding durable operations retain their existing minimum
        attempts, and this reservation makes their aggregate guards plus the
        SQLite requirement fit inside the same bounded tail deadline.
        """
        if storage_close_timeout <= 0.0:
            return 0.0

        memory_system = getattr(self, "memory_system", None)
        run_memory = bool(memory_system and hasattr(memory_system, "shutdown"))
        sync_service = getattr(self, "_sync_service", None)
        run_sync = bool(sync_service and sync_service.is_running)
        dispatcher = getattr(self, "dispatcher", None)
        run_dispatcher = bool(
            dispatcher and hasattr(dispatcher, "shutdown_durable_delivery")
        )
        run_storage = hasattr(self.storage, "close")
        if not run_storage:
            return 0.0

        # The cached SQLAlchemy factory is a separate, bounded pre-close
        # operation.  Its own SQLite driver workers need their own declared
        # reservation; it must not consume the primary close's reservation.
        run_storage_preclose = _storage_preclose(self.storage) is not None
        # The dispatcher releases runtime-owner liveness and any raw volatile
        # handoffs before storage closes. It is a distinct guarded tail step,
        # so its minimum must be reserved alongside background cleanup,
        # memory, and the two sync phases.
        preceding_steps = (
            1 + int(run_dispatcher) + int(run_memory) + 2 * int(run_sync)
        )
        preclose_minimum = (
            max(KESTREL_SHUTDOWN_TAIL_MIN_STEP_S, storage_preclose_timeout)
            if run_storage_preclose
            else 0.0
        )
        return (
            preceding_steps * KESTREL_SHUTDOWN_TAIL_MIN_STEP_S
            + preclose_minimum
            + storage_close_timeout
        )

    async def _run_durable_shutdown_tail(
        self,
        tail_reserve: float,
        *,
        storage_close_timeout: float | None = None,
        storage_preclose_timeout: float | None = None,
    ) -> tuple[bool, bool]:
        """Run the safety-critical durable shutdown steps.

        These persist data and prevent leaks (agent-owned background-task
        cleanup, memory shutdown, the final sync snapshot, and storage
        close) and MUST complete even when a feature/MCP/LLM/TaskManager
        shutdown above was cancelled by the outer ``asyncio.wait_for``
        timeout. Each step guards its own errors so one failure never skips
        the rest.

        The tail is itself BOUNDED (#2409 review). Python ``asyncio.wait_for``
        cancels once and then waits for cancellation to complete; if the tail
        performed fresh *unbounded* awaits, a step that hangs or suppresses
        cancellation would make the outer timeout / "forcing exit" branch
        unreachable forever. So every step runs against a finite guard derived
        from ``tail_reserve`` and is ABANDONED (not awaited) if it exceeds the
        guard. Each data-critical step still gets a nonzero attempt
        (``KESTREL_SHUTDOWN_TAIL_MIN_STEP_S``).

        Returns ``(cancelled, degraded)``:
        * ``cancelled`` — a step observed cancellation, so the caller
          re-raises ``CancelledError`` after the tail (never reporting a
          successful shutdown post-cancellation).
        * ``degraded`` — a step was abandoned past its guard or errored, so
          the caller must NOT describe the cleanup as "completed".
        """
        loop = asyncio.get_running_loop()

        state = {"cancelled": False, "degraded": False}

        # Count the tail steps that will actually run so each fair-divides the
        # live remaining tail budget (with a nonzero floor per step).
        memory_system = getattr(self, "memory_system", None)
        run_memory = bool(memory_system and hasattr(memory_system, "shutdown"))
        sync_service = getattr(self, "_sync_service", None)
        run_sync = bool(sync_service and sync_service.is_running)
        dispatcher = getattr(self, "dispatcher", None)
        run_dispatcher = bool(
            dispatcher and hasattr(dispatcher, "shutdown_durable_delivery")
        )
        run_storage = hasattr(self.storage, "close")
        if storage_close_timeout is None:
            storage_close_timeout = _minimum_storage_close_timeout(self.storage)
        if storage_preclose_timeout is None:
            storage_preclose_timeout = max(
                _minimum_storage_potential_preclose_timeout(self.storage),
                _minimum_storage_preclose_timeout(self.storage),
            )
        if not run_storage:
            storage_close_timeout = 0.0
            storage_preclose_timeout = 0.0

        tail_minimum = self._durable_tail_minimum_budget(
            storage_close_timeout, storage_preclose_timeout
        )
        # The normal (non-SQLite) path retains its existing fair-share
        # allocation.  SQLite adds a declared reservation, which includes the
        # preceding tail-step floors, so a short fair share cannot cancel the
        # worker-exit wait before its own deadline.
        # ``tail_reserve`` was already resolved and clamped against the
        # production outer deadline.  It is the one authoritative tail
        # deadline: never recompute a larger deadline here from a backend
        # requirement, because that would silently discard the outer-budget
        # clamp when a configuration cannot fit.
        tail_deadline = loop.time() + max(0.0, tail_reserve)
        # The sync path makes TWO guarded steps (snapshot + stop), so it must
        # count as two in the fair-division denominator — otherwise sync-stop
        # runs at pending_steps==1 and claims the entire remaining budget,
        # starving the data-critical storage-close of its fair share.
        storage_preclose = _storage_preclose(self.storage)
        run_storage_preclose = storage_preclose is not None
        pending_steps = (
            1
            + int(run_dispatcher)
            + int(run_memory)
            + 2 * int(run_sync)
            + int(run_storage_preclose)
            + int(run_storage)
        )
        reserved_minimum = tail_minimum

        def _step_guard(minimum: float = KESTREL_SHUTDOWN_TAIL_MIN_STEP_S) -> float:
            nonlocal pending_steps, reserved_minimum
            remaining = max(0.0, tail_deadline - loop.time())
            if storage_close_timeout > 0.0:
                # Preserve every later step's minimum.  Giving an earlier tail
                # operation an ordinary fair share could otherwise consume the
                # SQLite close reservation before storage is reached.
                excess = max(0.0, remaining - reserved_minimum)
                share = excess / pending_steps if pending_steps > 0 else 0.0
                # A required close reservation can exceed the total shutdown
                # budget.  In that incoherent configuration the resolver
                # gives the durable tail all available time; this live clamp
                # keeps every individual guard inside that same deadline
                # rather than creating a second, larger one here.
                guard = min(remaining, max(0.0, minimum) + share)
                reserved_minimum = max(0.0, reserved_minimum - minimum)
                if pending_steps > 0:
                    pending_steps -= 1
                return guard
            share = remaining / pending_steps if pending_steps > 1 else remaining
            if pending_steps > 0:
                pending_steps -= 1
            return max(KESTREL_SHUTDOWN_TAIL_MIN_STEP_S, share)

        def _harvest(
            task,
            label: str,
            *,
            defer_failure_report: bool = False,
        ) -> str:
            """Read a completed task's outcome, recording degradation."""
            if task.cancelled():
                state["cancelled"] = True
                return "cancelled"
            exc = task.exception()
            if exc is None:
                return "ok"
            if not defer_failure_report:
                logging.warning(
                    "Durable shutdown step '%s' failed: %s",
                    label,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                state["degraded"] = True
            return "error"

        def _abandon(task) -> None:
            """Force-terminate an over-guard / cancelled tail step.

            ``task.cancel()`` alone is not enough: a step that suppresses
            ``CancelledError`` keeps running forever and would hang a graceful
            event-loop teardown that awaits every still-pending task (and, in
            production, leak past process exit). Closing the underlying
            coroutine throws ``GeneratorExit`` — a ``BaseException`` the step
            cannot swallow with ``except CancelledError`` — so the frame is
            guaranteed to unwind. A done callback retrieves the resulting
            exception so it is never surfaced as "exception never retrieved".
            """
            task.cancel()  # cooperative first; unblocks well-behaved steps
            coro = task.get_coro()
            if coro is not None:
                try:
                    coro.close()  # hard stop — GeneratorExit cannot be suppressed
                except Exception:
                    pass
            # Retrieve the eventual exception once the task reconciles to done.
            task.add_done_callback(
                lambda t: None if t.cancelled() else t.exception()
            )

        async def _bounded(
            coro,
            guard: float,
            label: str,
            *,
            shielded: bool = False,
            defer_failure_report: bool = False,
        ):
            """Run one tail step bounded by ``guard``.

            The coroutine is scheduled as a task and awaited only up to
            ``guard`` via ``asyncio.wait`` — which, unlike ``asyncio.wait_for``,
            does NOT wait for the task's cancellation to complete on timeout.
            If the step exceeds the guard it is CANCELLED, hard-terminated, and
            ABANDONED (never awaited), so a step that hangs *or suppresses
            cancellation* cannot stall the tail beyond ``guard`` (the outer
            ``wait_for`` / CLI "forcing exit" branch stays reachable) nor leak
            past it. When ``shielded`` the underlying task keeps running if the
            tail's own await is cancelled externally (used for
            ``force_snapshot`` so cancellation neither aborts it nor skips the
            following ``stop()``); it is still bounded by ``guard``.
            """
            # Every guard and any cancellation grace shares the one live tail
            # deadline.  In particular, an externally-cancelled shielded
            # close must not receive a fresh full guard after the deadline.
            guard = min(max(0.0, guard), max(0.0, tail_deadline - loop.time()))
            task = asyncio.ensure_future(coro)
            try:
                done, _pending = await asyncio.wait({task}, timeout=guard)
            except asyncio.CancelledError:
                # The tail's own await was cancelled externally. ``asyncio.wait``
                # does not cancel its futures, so a shielded task is still
                # running — give it a bounded chance to finish so data-critical
                # steps are not aborted, then propagate via state["cancelled"].
                state["cancelled"] = True
                if shielded:
                    cancellation_grace = min(
                        guard, max(0.0, tail_deadline - loop.time())
                    )
                    try:
                        done, _pending = await asyncio.wait(
                            {task}, timeout=cancellation_grace
                        )
                    except asyncio.CancelledError:
                        _abandon(task)
                        if not defer_failure_report:
                            state["degraded"] = True
                        return "abandoned"
                    if task in done:
                        return _harvest(
                            task,
                            label,
                            defer_failure_report=defer_failure_report,
                        )
                    _abandon(task)
                    if not defer_failure_report:
                        state["degraded"] = True
                    return "abandoned"
                _abandon(task)  # do NOT await — may suppress cancel
                return "cancelled"

            if task not in done:
                # Exceeded the guard. Hard-terminate but do NOT await — the step
                # may suppress cancellation and would otherwise hang us.
                _abandon(task)
                if not defer_failure_report:
                    logging.warning(
                        "Durable shutdown step '%s' exceeded %.2fs; abandoned "
                        "(shutdown degraded).",
                        label,
                        guard,
                    )
                    state["degraded"] = True
                return "abandoned"

            return _harvest(
                task,
                label,
                defer_failure_report=defer_failure_report,
            )

        # Cancel agent-owned background work before storage/sync shutdown.
        await _bounded(
            self._shutdown_background_tasks(), _step_guard(), "background-tasks"
        )

        # Durable signal events retain only their policy-safe projection, but
        # the dispatcher can briefly hold a same-process live payload handoff
        # for a worker that claims before restart.  It is never durable and
        # must not outlive this agent instance.
        dispatcher_shutdown_complete = True
        if run_dispatcher:
            dispatcher_shutdown_status = await _bounded(
                dispatcher.shutdown_durable_delivery(),
                _step_guard(),
                "durable-signal-dispatcher",
            )
            dispatcher_shutdown_complete = dispatcher_shutdown_status == "ok"
            if getattr(dispatcher, "durable_shutdown_owner_fenced", False):
                await self._ensure_durable_shutdown_continuation(dispatcher)
                logging.warning(
                    "Durable cognition is still running after shutdown cancellation; "
                    "keeping owner liveness and storage fenced until it settles."
                )
                # Other bounded shutdown work remains safe and useful (memory
                # bookkeeping and sync workers do not own the retained
                # delivery lease). The continuation observed below is the
                # sole fence around shared storage close.
                state["degraded"] = True
                dispatcher_shutdown_complete = False
            # The dispatcher owns an independent, shielded teardown task, so
            # caller cancellation cannot strand committed work.  If its task
            # outlives this guard, it may still be using the same backend;
            # closing storage here would recreate the post-close audit-write
            # race. The agent-owned continuation below joins it first.

        # Stop memory-owned bookkeeping before storage/sync shutdown.
        if run_memory:
            await _bounded(memory_system.shutdown(), _step_guard(), "memory-system")

        # Final snapshot to all sync targets before closing storage. The
        # snapshot is SHIELDED so cancellation neither aborts it nor skips the
        # ``stop()`` that releases the sync worker.
        if run_sync:
            status = await _bounded(
                sync_service.force_snapshot(),
                _step_guard(),
                "sync-snapshot",
                shielded=True,
            )
            # Always attempt stop() — even if the snapshot was abandoned or
            # cancellation was observed — so the sync worker is released.
            await _bounded(sync_service.stop(), _step_guard(), "sync-stop")
            if status == "ok":
                logging.info("Sync service: final snapshot flushed")

        # Another lifecycle caller may already have spent the dispatcher guard
        # and installed the unique continuation while this tail was stopping
        # memory/sync work.  That task owns the eventual close; a second tail
        # must join or retry it rather than closing the same backend directly.
        continuation = self._durable_shutdown_continuation
        if continuation is not None:
            if continuation.done() and (
                continuation.cancelled() or continuation.exception() is not None
            ):
                continuation = await self._ensure_durable_shutdown_continuation(
                    dispatcher
                )
            return state["cancelled"], state["degraded"]

        if run_dispatcher and not dispatcher_shutdown_complete:
            await self._ensure_durable_shutdown_continuation(dispatcher)
            logging.warning(
                "Durable signal dispatcher teardown exceeded the bounded "
                "tail guard; an agent-owned continuation will close storage "
                "after owner release."
            )
            return state["cancelled"], state["degraded"]

        # Dispose an optional cached SQLAlchemy engine as its own bounded
        # phase.  Otherwise a slow engine disposal can consume the guard that
        # the following primary SQLite close requires to drain its worker.
        storage_preclose_status = None
        if storage_preclose is not None:
            storage_preclose_status = await _bounded(
                storage_preclose(),
                _step_guard(
                    max(KESTREL_SHUTDOWN_TAIL_MIN_STEP_S, storage_preclose_timeout)
                ),
                "storage-sqla-pre-close",
                shielded=storage_preclose_timeout > 0.0,
                # The primary SQLite backend owns an independent worker.  It
                # must receive its reserved close chance before a pre-close
                # timeout/error is reported as degraded.
                defer_failure_report=True,
            )

        # Close storage — SQLite gets its declared worker-exit reservation and
        # keeps running through an external cancellation until that bounded
        # contract completes.  Other backends retain the existing behavior.
        if run_storage:
            await _bounded(
                self.storage.close(),
                _step_guard(storage_close_timeout),
                "storage-close",
                shielded=storage_close_timeout > 0.0,
            )

        if storage_preclose_status in {"error", "abandoned"}:
            logging.warning(
                "Durable shutdown step 'storage-sqla-pre-close' %s; primary "
                "storage close was attempted before reporting shutdown "
                "degraded.",
                storage_preclose_status,
            )
            state["degraded"] = True

        return state["cancelled"], state["degraded"]
