import logging
import json
import os
import asyncio
import time
from datetime import datetime
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.security.encryption import DecryptionError
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.config import TRUSTED_AGENTS_DIR
from typing import Optional, Dict, List, Any, Union
import re
from pathlib import Path
from kestrel_sovereign.privacy import PrivacyMode
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
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.hooks import HooksManager, HookEvent, HookInput, PermissionDecision
from kestrel_sovereign.bootstrap import BootstrapService, BootstrapState
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
    ANTI_INJECTION_SYSTEM_PROMPT,
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

# Maximum chars for a single tool result before truncation
MAX_TOOL_RESULT_CHARS = int(os.environ.get("KESTREL_MAX_TOOL_RESULT_CHARS", "8000"))

# Reserve this fraction of context for the LLM response + next tool call
CONTEXT_RESERVE_FRACTION = 0.2

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
    ):
        """
        Initializes the agent with memory and reasoning capabilities.
        # Bedrock for Sovereign Companions: Human-led, no-loss persistence.

        Args:
            did: The agent's own DID (e.g., 'did:pkh:...'), used for self-discovery.
            storage_path: Path to the database file for SQLite storage.
            llm_service: The service that provides access to foundational models.
            privacy_mode: Privacy mode for this session (EPHEMERAL, ISOLATED, ANONYMOUS, NORMAL, PUBLIC).
            pg_pool: Optional PostgreSQL pool for feedback feature.
            database_url: PostgreSQL connection string (for postgres backend).
            db_backend: Database backend type ('sqlite' or 'postgres').
                       Defaults to KESTREL_DB_BACKEND env var or 'sqlite'.
            allowed_features: Optional set of feature class names to load.
                       If None, all discovered features are loaded.
                       Mandatory features always load regardless.
        """
        self.did = did
        self._privacy_mode = privacy_mode
        self.storage_path = storage_path
        self._allowed_features = allowed_features

        # Determine database backend
        self._db_backend = db_backend or os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        self._database_url = database_url or os.environ.get("KESTREL_DATABASE_URL")

        # Storage will be initialized asynchronously
        self._raw_storage = None
        self.storage = None

        self.llm_service = llm_service or LLMService()
        self.pg_pool = pg_pool
        # Note: agent_id is a @property that returns self.did (see below).
        # Do NOT set self.agent_id = ... here; it would shadow the property.
        self.privacy_agent = None  # Will be initialized after storage
        self.lighthouse_provider = None  # Will be initialized after storage if API key available
        self.wallet = None  # Set by WalletFeature.initialize()
        self.reflection_hook = None  # Set by ReflectionFeature.post_all_features_loaded()

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

        # Pending task completion notifications (for background tasks)
        self._pending_task_notifications: List[str] = []

        # Cancellation tracking for stop button functionality
        self._current_request_id: Optional[str] = None
        self._active_request_ids: set[str] = set()
        self._cancelled_requests: set = set()

        # Session state
        self._session_briefed = False
        self._safe_mode = False

        # Dynamic tool loading: explored features get direct tool access
        self._explored_features: dict = {}  # ordered dict (insertion order) for LRU eviction
        self._direct_tools: dict = {}
        self._direct_tool_defs: list = []
        self._tool_to_feature: dict = {}  # tool_name -> feature tool_name

        # Initialize constitution audit tracking
        self._init_constitution_audit_tracking()

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

    @property
    def mcp_agent(self):
        """Lazy lookup for MCPAgent feature."""
        return self.features.get("MCPAgent")

    @property
    def model_agent(self):
        """Lazy lookup for ModelAgent feature."""
        return self.features.get("ModelAgent")

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
                        agent_id=self.did
                    )
                    logging.info(f"Using shared PostgreSQL pool for Kestrel storage (agent: {self.did})")
                else:
                    self._raw_storage = AsyncStorage(
                        backend="postgres",
                        dsn=self._database_url,
                        agent_id=self.did
                    )
                    logging.info(f"Using PostgreSQL backend for Kestrel storage (agent: {self.did})")
            else:
                # SQLite backend (default) - agent_id optional since each agent has own DB
                self._raw_storage = AsyncStorage(self.storage_path, agent_id=self.did)
                logging.info(f"Using SQLite backend for Kestrel storage: {self.storage_path}")

            await self._raw_storage.initialize()

            # Wrap storage with privacy-enforcing layer
            self.storage = PrivacyEnforcingStorage(self._raw_storage, self._privacy_mode)

            # Initialize privacy agent
            self.privacy_agent = PrivacyAgent(self._raw_storage, self._privacy_mode)

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
            )
            await self.task_manager.initialize()

            # Expose feedback store for features and commands
            self.feedback_store = feedback_store

            # Expose observability store for orchestrator instrumentation
            self.observability_store = observability_store

            # Initialize storage providers for features (reflection self-model, etc.)
            self.lighthouse_provider = None
            self.storacha_provider = None

            if os.environ.get("STORACHA_SPACE_DID") and os.environ.get("STORACHA_AGENT_KEY"):
                try:
                    from kestrel_sovereign.storage.providers.storacha_provider import StorachaProvider
                    self.storacha_provider = StorachaProvider()
                    if not self.storacha_provider.is_available():
                        self.storacha_provider = None
                except Exception as e:
                    logging.warning(f"StorachaProvider init failed: {e}")

            if os.environ.get("LIGHTHOUSE_API_KEY"):
                try:
                    from kestrel_sovereign.storage.providers.lighthouse_provider import LighthouseProvider
                    self.lighthouse_provider = LighthouseProvider()
                    if not self.lighthouse_provider.is_available():
                        self.lighthouse_provider = None
                except Exception as e:
                    logging.warning(f"LighthouseProvider init failed: {e}")

            # Sync service — event-driven snapshots to all configured targets.
            # Targets are ordered by trust: Sovereign → Federated → Delegated → Expedient.
            # Snapshots fire on shutdown, scheduled backup, or explicit !backup command.
            self._sync_service = None
            if self._db_backend.lower() != "postgres":
                try:
                    from kestrel_sovereign.storage.sync.service import SyncService
                    from kestrel_sovereign.storage.sync.targets import (
                        SovereignIPFSTarget, StorachaTarget, LighthouseTarget, GCSTarget,
                    )

                    agent_id = self.did or "default"
                    state_dir = Path(self.storage_path).parent if self.storage_path else None

                    self._sync_service = SyncService(db_path=self.storage_path)

                    # Sovereign: self-hosted IPFS (our infrastructure)
                    sovereign_url = os.environ.get("SOVEREIGN_IPFS_URL")
                    if sovereign_url:
                        self._sync_service.add_target(SovereignIPFSTarget(
                            api_url=sovereign_url, agent_id=agent_id, state_dir=state_dir,
                        ))

                    # Federated: Storacha (UCAN/DID auth)
                    if os.environ.get("STORACHA_SPACE_DID") and os.environ.get("STORACHA_AGENT_KEY"):
                        try:
                            self._sync_service.add_target(StorachaTarget(
                                space_did=os.environ["STORACHA_SPACE_DID"],
                                agent_key=os.environ["STORACHA_AGENT_KEY"],
                                proof=os.environ.get("STORACHA_PROOF", ""),
                                agent_id=agent_id, state_dir=state_dir,
                            ))
                        except Exception as e:
                            logging.warning(f"StorachaTarget init failed: {e}")

                    # Delegated: Lighthouse (API key)
                    if os.environ.get("LIGHTHOUSE_API_KEY"):
                        self._sync_service.add_target(LighthouseTarget(
                            api_key=os.environ["LIGHTHOUSE_API_KEY"],
                            agent_id=agent_id, state_dir=state_dir,
                        ))

                    # Expedient: GCS (fast cloud backup)
                    gcs_bucket = os.environ.get("GCS_BACKUP_BUCKET")
                    if gcs_bucket:
                        self._sync_service.add_target(GCSTarget(
                            bucket=gcs_bucket, agent_id=agent_id, state_dir=state_dir,
                            project=os.environ.get("GCP_PROJECT"),
                            credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                        ))

                    if self._sync_service.targets:
                        await self._sync_service.start()
                    else:
                        self._sync_service = None

                except Exception as e:
                    logging.warning(f"Sync service init failed: {e}", exc_info=True)
                    self._sync_service = None

            # Resolve agent name BEFORE features so features can use it
            # (e.g. PeersFeature._get_own_name() reads self._agent_name)
            _agent_node = await self.storage.get_node(self.agent_id)
            if _agent_node:
                self._agent_name = _agent_node.properties.get("name", "Unnamed Agent")
            else:
                self._agent_name = "Unnamed Agent"

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
                from kestrel_sovereign.hooks.base import HookInput, HookEvent
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

            if self._is_test_instance:
                logging.info(f"TEST INSTANCE detected: {self._agent_name} (cycle: {self._test_cycle_id})")
                self._load_test_disclosure(agent_node.properties)

            # Activate agent's own OpenRouter key for isolated billing
            openrouter_key_hash = agent_node.properties.get("openrouter_key_hash")
            if openrouter_key_hash:
                try:
                    key_activated = await self.llm_service.use_agent_key(
                        agent_did=self.did,
                        db=self._raw_storage.db,
                        provider="openrouter",
                    )
                    if key_activated:
                        logging.info(f"Agent using own OpenRouter key (hash: {openrouter_key_hash[:16]}...)")
                except (KeyError, ValueError, AttributeError, ConnectionError) as e:
                    logging.warning(f"Could not activate agent OpenRouter key: {e}")
                except Exception as e:
                    logging.warning(f"Could not activate agent OpenRouter key: {e}", exc_info=True)

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

            # Initialize unified context manager (orchestrates all context sources)
            # Model identity derived lazily from llm_service.get_active_model_id()
            self.context_manager = ContextManager(
                storage=self.storage,
                agent_id=self.agent_id,
                consolidator=self.memory_consolidator,
                memory_retriever=self.memory_system.retriever,
                llm_service=self.llm_service,
            )

            # Initialize bootstrap service for first-time agent wake-up
            self.bootstrap_service = BootstrapService(
                db=self._raw_storage.db,
                agent_id=self.agent_id,
                agent_name=self._agent_name,
                llm_service=self.llm_service,
                agent_data_path=agent_data_dir,
            )
            logging.info("BootstrapService initialized")

            # Load persisted model preference from database and register persistence callback
            await self._load_model_preference()
            self.llm_service.set_preference_persistence_callback(self._persist_model_preference)

            # Cache the features prompt (built once at session start)
            self._cached_features_prompt = self._build_features_prompt_section()

            # Pre-explore features whose tools should be direct from turn one.
            # TaskFeature: run_workflow, list_available_skills (meta-tools)
            # PeersFeature: ask_agent, list_peers (inter-agent communication)
            for feature_name in ("TaskFeature", "PeersFeature"):
                feature = self.features.get(feature_name)
                if feature:
                    self._register_explored_feature_tools(feature)

            # Initialize heartbeat system (periodic agent self-checks)
            from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner
            self._heartbeat_config = HeartbeatConfig.from_config()
            self.heartbeat_runner = HeartbeatRunner(self, self._heartbeat_config)
            if self._heartbeat_config.enabled:
                await self.heartbeat_runner.start()

            # Default schedules are now set up by SchedulerFeature.post_all_features_loaded()

    @property
    def privacy_mode(self) -> PrivacyMode:
        """Get current privacy mode."""
        return self._privacy_mode
    
    async def set_privacy_mode(self, mode: PrivacyMode) -> str:
        """
        Change the privacy mode.

        This updates both the storage wrapper and the privacy agent.
        Note: Changing to a more restrictive mode does NOT delete existing data.
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

        self._privacy_mode = mode
        self.storage.set_privacy_mode(mode)
        status_message = self.privacy_agent.set_mode(mode)
        logging.info(f"Privacy mode changed to: {mode.value}")
        return status_message


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

    async def _disable_feature(self, feature_name: str):
        """Disable a feature: call on_disable, unregister its hooks, remove from features dict."""
        feature = self.features.get(feature_name)
        if not feature:
            logging.warning(f"Cannot disable unknown feature: {feature_name}")
            return

        # Call on_disable lifecycle hook
        await feature.on_disable()

        # Auto-unregister hooks from get_hooks()
        if self.hooks_manager:
            for hook in feature.get_hooks():
                self.hooks_manager.unregister(hook)
                logging.info(f"Auto-unregistered hook '{hook.name}' from feature '{feature_name}'")

        logging.info(f"Feature '{feature_name}' disabled")

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

            # Store the wake-up message in conversation history
            await self.privacy_agent.add_conversation("assistant", wake_up_msg, session_id=session_id)

            logging.info(f"[BOOTSTRAP] Agent waking up - entering discovery mode")
            return wake_up_msg

        elif state == BootstrapState.DISCOVERY:
            # In discovery mode - process through discovery conversation
            response, is_complete, wants_avatar = await self.bootstrap_service.process_discovery_message(user_input)

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

    async def process_input(self, user_input: str, model_override: str = None, session_id: str = None) -> str:
        """
        Processes user input by consulting the constitution, retrieving context,
        and generating a response using tool calling for features.

        Args:
            user_input: The user's message
            model_override: Optional model to use (e.g., "openai/gpt-5", "ollama/llama3.2")
            session_id: Optional session ID to load conversation context from a specific session
        """
        logging.info(f"[AGENTIC] process_input called ({len(user_input)} chars)")

        # CONSTITUTION AUDIT CHECK: Trigger periodic integrity audits
        await self._maybe_audit()

        # SAFE MODE CHECK: If in safe mode, only allow diagnostic commands
        if self._safe_mode:
            safe_mode_commands = ["!safe-mode", "!verify-constitution", "!status", "!help"]
            if user_input.startswith("!"):
                cmd = user_input.split()[0]
                if cmd not in safe_mode_commands:
                    return (
                        "🚨 SAFE MODE ACTIVE\\n\\n"
                        "The agent has detected an integrity issue and is operating in restricted mode.\\n"
                        "Only diagnostic commands are available: !safe-mode, !verify-constitution, !status\\n\\n"
                        "Please contact your administrator to resolve the integrity issue."
                    )
            else:
                return (
                    "🚨 SAFE MODE ACTIVE\\n\\n"
                    "The agent cannot process queries due to an integrity failure.\\n"
                    "Use !safe-mode to check status or !verify-constitution to re-verify.\\n\\n"
                    "Normal operation will resume once integrity is restored."
                )

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
                response = await self.command_handler.handle(user_input)
                if response:
                    return response

        # --- OpenTelemetry span for the full request lifecycle ---
        with optional_span("agent.process_input", {
            "agent.did": self.did,
            "agent.session_id": session_id or "",
            "agent.input_length": len(user_input),
        }) as _otel_span:
            return await self._process_input_traced(
                user_input, model_override, session_id, _otel_span
            )

    async def _process_input_traced(self, user_input: str, model_override: str, session_id: str, _otel_span) -> str:
        """Inner process_input logic wrapped in an OTEL span."""
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

        # Check if episode creation is needed (after 20+ messages or 30-min gap)
        session_msg_count = len([m for m in history if m.get('role') == 'user'])
        if session_msg_count > 0 and session_msg_count % 20 == 0:
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
            include_memories=True,
            include_rag=True,
            privacy_mode=self._privacy_mode.value,
            conversation_history=history,
            reflection_guidance=reflection_guidance,
        )

        # NOW store user input — after context is built so memory retrieval
        # doesn't find the current message as a "past memory"
        try:
            await self.privacy_agent.add_conversation("user", user_input, session_id=session_id)
        except DecryptionError:
            logging.warning("DecryptionError storing user input - continuing in degraded mode")
        self._session_briefed = True

        # Log budget usage for monitoring and store for API access
        self._last_context_warnings = context_result.warnings or []
        self._last_context_summary = context_result.budget_summary
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

        # Build user prompt with context, wrapping user input in boundary markers
        prompt = self.user_prompt_template.format(
            context="[Context included in system prompt]",
            query=wrap_user_input(user_input)
        )

        # Build system prompt with features and anti-injection defense
        force_local_only = not self.privacy_agent.privacy_config.allows_cloud_llm()
        system_prompt = context_result.system_prompt

        # Add anti-injection instructions to system prompt
        system_prompt = f"{system_prompt}\n{ANTI_INJECTION_SYSTEM_PROMPT}"

        # Add cached features section (built once at session start)
        if self._cached_features_prompt:
            system_prompt = f"{system_prompt}{self._cached_features_prompt}"

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

        # Generate response with full conversation context
        response = await self.llm_service.generate_with_messages(
            messages=messages,
            force_local_only=force_local_only,
            model_override=effective_model,
            tools=feature_tools if feature_tools else None
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

        # Handle tool calls if present (A2A pattern)
        response_text = await self._handle_orchestrator_response(
            response=response,
            feature_tools=feature_tools,
            system_prompt=system_prompt,
            force_local_only=force_local_only,
            effective_model=effective_model,
            user_message=prompt  # Pass original user message for subagent context
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

        # Fire STOP hook (response cycle complete)
        if self.hooks_manager:
            hook_input = HookInput(
                session_id=session_id or "",
                hook_event_name=HookEvent.STOP.value,
            )
            await self.hooks_manager.execute_hooks_parallel(
                HookEvent.STOP, hook_input
            )

        # Record response length on OTEL span (privacy: no content)
        if _otel_span:
            _otel_span.set_attribute("agent.response_length", len(response_text))

        return response_text

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
        # Stop heartbeat runner
        if hasattr(self, 'heartbeat_runner') and self.heartbeat_runner:
            try:
                await self.heartbeat_runner.stop()
            except Exception as e:
                logging.warning(f"Error stopping heartbeat: {e}")

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
