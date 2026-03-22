import logging
import json
import os
import asyncio
import time
from datetime import datetime
from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage
from kestrel_sovereign.storage.encryption import DecryptionError
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.config import TRUSTED_AGENTS_DIR
from typing import Optional, Dict, List, Any, Union
import re
from pathlib import Path
from kestrel_sovereign.privacy import PrivacyMode
from decimal import Decimal, getcontext
from kestrel_sovereign.extensions.app_extension import AppExtension
from kestrel_sovereign.features.wallet import WalletAgent
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
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.hooks import HooksManager, HookEvent, HookInput, HookOutput, PermissionDecision
from kestrel_sovereign.bootstrap import BootstrapService, BootstrapState
from kestrel_sovereign.security.input_guardrails import (
    wrap_user_input,
    check_prompt_injection,
    validate_tool_arguments,
    ANTI_INJECTION_SYSTEM_PROMPT,
)

# Optional ollama import (not available in remote-only containers)
try:
    import ollama
except ImportError:
    ollama = None

# Set precision for Decimal calculations
getcontext().prec = 18

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


class KestrelAgent(ConstitutionMixin, StreamingMixin, BackupMixin, SleepMixin):
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
        """
        self.did = did
        self._privacy_mode = privacy_mode
        self.storage_path = storage_path

        # Determine database backend
        self._db_backend = db_backend or os.environ.get("KESTREL_DB_BACKEND", "sqlite")
        self._database_url = database_url or os.environ.get("KESTREL_DATABASE_URL")

        # Storage will be initialized asynchronously
        self._raw_storage = None
        self.storage = None

        self.llm_service = llm_service or LLMService()
        self.pg_pool = pg_pool
        self.agent_id = did
        self.privacy_agent = None  # Will be initialized after storage
        self.lighthouse_provider = None  # Will be initialized after storage if API key available

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

                    agent_id = self.did or getattr(self, 'agent_id', None) or "default"
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

                    agent_id = self.did or self.agent_id or "default"
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

            # Auto-discover and register features from features/ directory
            # Features can be disabled via KESTREL_DISABLED_FEATURES env var
            for feature in discover_features(self):
                await self._register_feature(feature)

            # Set up feature references
            self.mcp_agent = self.features.get("MCPAgent")
            self.model_agent = self.features.get("ModelAgent")
            logging.info("Feature references set up")

            # Register all tools with SecurityFeature AFTER all features are loaded
            security = self.features.get("SecurityFeature")
            if security and hasattr(security, '_register_all_tools'):
                await security._register_all_tools()
                logging.info("Security permissions registered for all features")

            # Wire reflection into sleep cycle (runs pre/post-consolidation analysis)
            from kestrel_sovereign.features.reflection.hooks import create_reflection_hook
            self.reflection_hook = create_reflection_hook(self)
            if self.reflection_hook:
                logging.info("Reflection hook enabled for sleep cycle")

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

            # Initialize wallet with genesis budget
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

            initial_balance_str = agent_node.properties.get("initialBalance", "100.0")
            logging.info(f"Creating WalletAgent with db_path={self.storage_path}")
            self.wallet = WalletAgent(
                agent_id=self.agent_id,
                initial_balance=Decimal(initial_balance_str),
                db_path=self.storage_path
            )
            await self.wallet.initialize()
            logging.info("WalletAgent initialized")

            # Initialize memory consolidator for episode management
            logging.info("Creating MemoryConsolidator")
            self.memory_consolidator = MemoryConsolidator(
                db=self._raw_storage.db,
                agent_id=self.agent_id
            )
            logging.info("MemoryConsolidator created")

            # Initialize memory system for semantic/emotional memory retrieval
            logging.info("Creating MemorySystem")
            self.memory_system = MemorySystem(
                storage=self._raw_storage,
                agent_id=self.agent_id
            )
            await self.memory_system.initialize()
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

            # Auto-schedule reflection + training (idempotent — skips if already scheduled)
            await self._setup_default_schedules()

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

    async def _setup_default_schedules(self):
        """Register default scheduled tasks for self-improvement.

        Idempotent — checks for existing tasks before adding.
        Requires both SchedulerFeature and ReflectionFeature to be loaded.
        """
        scheduler = self.features.get("SchedulerFeature")
        reflection = self.features.get("ReflectionFeature")
        if not scheduler or not reflection:
            return

        # Check what's already scheduled
        existing = await scheduler.schedule_list()
        existing_names = {t["task_name"] for t in existing.get("tasks", [])}

        defaults = [
            ("reflect", "0 */4 * * *", '{"scope":"all","depth":"normal"}'),
            ("training_cycle", "0 3 * * *", '{"iterations":3,"depth":"normal"}'),
            ("backup_snapshot", "0 */4 * * *", "{}"),
        ]

        for task_name, cron, args in defaults:
            if task_name in existing_names:
                logging.debug(f"Schedule '{task_name}' already exists, skipping")
                continue
            result = await scheduler.schedule_add(
                cron_expression=cron, task_name=task_name, args_json=args,
            )
            if result.get("success"):
                logging.info(f"Scheduled '{task_name}' ({cron}), next: {result.get('next_run_at')}")
            else:
                logging.warning(f"Failed to schedule '{task_name}': {result.get('error')}")

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
        logging.info(f"[AGENTIC] process_input called with: {user_input[:100]}...")

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
        context_result = await self.context_manager.build_context(
            query=user_input,
            constitution=constitution,
            include_briefing=not self._session_briefed,
            include_memories=True,
            include_rag=True,
            privacy_mode=self._privacy_mode.value,
            conversation_history=history,
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

        return response_text

    def _build_feature_tools(self) -> List[Dict[str, Any]]:
        """
        Build the list of feature tools for the orchestrator LLM.

        Each feature is exposed as a high-level tool that the orchestrator
        can call. The feature then handles the task using its own tools
        and context (A2A pattern).

        Returns:
            List of tools in OpenAI function calling format
        """
        tools = []
        failed_features = []

        for feature in self.features.values():
            try:
                # Skip subagent dispatcher for pre-explored features
                # (their individual tools are already in the direct tool list)
                if feature.tool_name in self._explored_features:
                    continue
                tool_def = feature.to_orchestrator_tool()
                tools.append(tool_def)
                logging.info(f"[AGENTIC] Added tool: {feature.tool_name}")
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                logging.error(f"[AGENTIC] FAILED to build tool for {feature.name}: {e}")
                failed_features.append({"name": feature.name, "error": str(e)})
            except Exception as e:
                logging.error(f"[AGENTIC] FAILED to build tool for {feature.name}: {e}", exc_info=True)
                failed_features.append({"name": feature.name, "error": str(e)})

        # Log summary
        logging.info(f"[AGENTIC] Built {len(tools)} tools from {len(self.features)} features")
        if failed_features:
            logging.error(f"[AGENTIC] Failed features: {failed_features}")

        return tools

    # --- Dynamic Tool Loading ---

    MAX_DIRECT_TOOLS = 60

    def _build_all_tools(self) -> list:
        """Build combined tool list: feature dispatchers + explored individual tools."""
        tools = self._build_feature_tools()
        tools.extend(self._direct_tool_defs)
        return tools

    def _register_explored_feature_tools(self, feature) -> None:
        """Register a feature's individual tools for direct calling.

        After a successful subagent dispatch, the feature's @tool methods
        become available for the orchestrator to call directly without
        a subagent LLM hop.
        """
        if feature.tool_name in self._explored_features:
            return
        self._explored_features[feature.tool_name] = True
        registered = 0
        for tool in feature.get_tools():
            if tool.name in self._direct_tools:
                name = f"{feature.tool_name}__{tool.name}"
            else:
                name = tool.name
            self._direct_tools[name] = tool
            tool_def = tool.schema.to_openai_format()
            tool_def["function"]["name"] = name
            self._direct_tool_defs.append(tool_def)
            self._tool_to_feature[name] = feature.tool_name
            registered += 1
        self._maybe_evict_direct_tools()
        logging.info(
            f"[DYNAMIC-TOOLS] Explored {feature.tool_name}, "
            f"registered {registered} direct tools. "
            f"Total: {len(self._direct_tools)}"
        )

    def _maybe_evict_direct_tools(self) -> None:
        """Evict least-recently-explored feature's tools if over limit."""
        while len(self._direct_tools) > self.MAX_DIRECT_TOOLS:
            oldest = next(iter(self._explored_features))
            del self._explored_features[oldest]
            to_remove = [k for k, v in self._tool_to_feature.items() if v == oldest]
            for name in to_remove:
                del self._direct_tools[name]
                del self._tool_to_feature[name]
            self._direct_tool_defs = [
                d for d in self._direct_tool_defs
                if d["function"]["name"] not in to_remove
            ]
            logging.info(f"[DYNAMIC-TOOLS] Evicted {len(to_remove)} tools from {oldest}")

    def _build_features_prompt_section(self) -> str:
        """
        Build a dynamic system prompt section describing loaded features.

        This informs the LLM about what features/subagents are available,
        their capabilities, and the commands they provide.

        Returns:
            Formatted string describing loaded features
        """
        if not self.features:
            return ""

        sections = ["\n\n## LOADED FEATURES (Active Subagents)\n"]
        sections.append("These are your ACTIVE subagents. They are loaded and ready to use RIGHT NOW:\n")

        for feature in self.features.values():
            try:
                # Feature name and description
                sections.append(f"\n### {feature.name}")
                sections.append(f"**Capabilities:** {feature.tool_description}")

                # List the feature's tools/commands
                tools = feature.get_tools()
                if tools:
                    sections.append("\n**Available commands:**")
                    for tool in tools:
                        cmd_prefix = tool.schema.command_prefix or ""
                        if cmd_prefix:
                            sections.append(f"- `{cmd_prefix}` - {tool.schema.description}")
                        else:
                            sections.append(f"- {tool.name}: {tool.schema.description}")
            except (AttributeError, TypeError, KeyError) as e:
                logging.warning(f"Failed to build prompt section for feature {feature.name}: {e}")
            except Exception as e:
                logging.warning(f"Failed to build prompt section for feature {feature.name}: {e}", exc_info=True)

        sections.append("\n\n**CRITICAL:** When asked about your subagents, capabilities, or available tools, LIST the features above by name. They ARE your active subagents. Never say 'no active subagents' - that is incorrect.")
        return "\n".join(sections)

    async def _execute_tool_with_hooks(
        self,
        tool_name: str,
        feature_name: str,
        args: dict,
        session_id: str,
        execute_fn,
    ) -> dict:
        """
        Execute a tool with PRE_TOOL_USE and POST_TOOL_USE hook enforcement.

        This is the single entry point for all tool execution in the orchestrator
        loop, ensuring security hooks (permissions, audit logging) are always
        invoked regardless of whether the tool is a feature subagent dispatch
        or a direct tool call.

        Args:
            tool_name: Name of the tool being called
            feature_name: Name of the owning feature (for permission lookup)
            args: Arguments to pass to the tool
            session_id: Session ID for hook context
            execute_fn: Async callable that performs the actual tool execution.
                        Called with no arguments; should return the tool result.

        Returns:
            Tool result dict, or an error dict if permission was denied.
        """
        # --- PRE_TOOL_USE hooks ---
        hook_input = HookInput(
            session_id=session_id,
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=feature_name,
        )

        hook_output = await self.hooks_manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            hook_input,
        )

        if hook_output.permission_decision == PermissionDecision.DENY:
            reason = hook_output.permission_reason or "Blocked by security policy"
            logging.info(f"[HOOKS] Tool denied: {feature_name}.{tool_name} - {reason}")
            return {"success": False, "error": f"Permission denied: {reason}"}

        # If hooks modified the input, update args (callers that need it can
        # inspect the returned result; the execute_fn closure already captured
        # the original args, so we pass updated_input through the result).
        if hook_output.updated_input:
            args = hook_output.updated_input

        # --- Execute the tool ---
        exec_start = time.time()
        result = await execute_fn()
        exec_duration_ms = int((time.time() - exec_start) * 1000)

        # --- POST_TOOL_USE hooks (parallel, non-blocking) ---
        post_hook_input = HookInput(
            session_id=session_id,
            hook_event_name=HookEvent.POST_TOOL_USE.value,
            tool_name=tool_name,
            tool_input=args,
            feature_name=feature_name,
            tool_response=result if isinstance(result, dict) else {"result": str(result)},
            execution_time_ms=exec_duration_ms,
        )
        await self.hooks_manager.execute_hooks_parallel(
            HookEvent.POST_TOOL_USE,
            post_hook_input,
        )

        return result

    async def _handle_orchestrator_response(
        self,
        response: Union[str, LLMResponse],
        feature_tools: List[Dict[str, Any]],
        system_prompt: str,
        force_local_only: bool,
        effective_model: str,
        max_iterations: int = None,
        user_message: str = None
    ) -> str:
        """
        Handle the orchestrator's response, executing any tool calls.

        If the LLM returns tool_calls, we dispatch them to the appropriate
        features (as subagents), then continue the conversation with results.

        Args:
            response: The LLM response (string or LLMResponse)
            feature_tools: The available feature tools
            system_prompt: System prompt for continuation
            force_local_only: Whether to force local-only LLM
            effective_model: The model to use
            max_iterations: Maximum tool call iterations (default: KESTREL_MAX_TOOL_ITERATIONS env var or 5)
            user_message: Original user message to provide context to subagents

        Returns:
            Final text response after all tool calls are processed
        """
        # Use module constant if not explicitly specified
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        # If response is just a string, return it directly
        if isinstance(response, str):
            return response

        # If response has no tool calls, return the content
        if not response.has_tool_calls:
            return response.content or ""

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add the user's original message so the LLM knows what to do with tool results
        if user_message:
            messages.append({"role": "user", "content": user_message})
            logging.debug(f"[ORCHESTRATOR] Added user message to context: {user_message[:100]}...")
        else:
            logging.warning(f"[ORCHESTRATOR] No user_message provided - LLM won't have context for tool results!")

        # Add initial assistant response with tool calls
        assistant_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # ollama library expects arguments as dict, not string
                        "arguments": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments) if tc.arguments else {}
                    }
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # Build feature lookup by tool_name
        features_by_tool_name = {f.tool_name: f for f in self.features.values()}

        # Build known tool allowlist for argument validation
        known_tools = set(features_by_tool_name.keys()) | set(self._direct_tools.keys())

        for iteration in range(max_iterations):
            # Warn when approaching iteration limit
            if iteration >= max_iterations * 0.8:  # 80% threshold
                logging.warning(f"[ORCHESTRATOR] Approaching max iterations: {iteration + 1}/{max_iterations}")

            # Execute each tool call by dispatching to features
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

                # Validate tool arguments before execution
                is_valid, validation_error = validate_tool_arguments(
                    tool_name, args, known_tools=known_tools
                )
                if not is_valid:
                    logging.warning(f"[ORCHESTRATOR] Tool validation failed: {validation_error}")
                    result = {"success": False, "error": f"Tool validation failed: {validation_error}"}
                    from kestrel_sovereign.features.base import _serialize_tool_result
                    result_json = json.dumps(_serialize_tool_result(result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json
                    })
                    continue

                # Log tool dispatch start
                dispatch_start = time.time()
                dispatch_event_id = await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"feature_dispatch:{tool_name}",
                    metadata={"arguments": args, "iteration": iteration}
                )

                # Find the feature for this tool
                feature = features_by_tool_name.get(tool_name)
                if feature:
                    # Determine feature_name for hooks (class name)
                    hook_feature_name = type(feature).__name__

                    # --- PRE_SUBAGENT_CALL hooks (feature-level security) ---
                    subagent_hook_input = HookInput(
                        session_id="orchestrator",
                        hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
                        tool_name=tool_name,
                        tool_input=args,
                        feature_name=hook_feature_name,
                    )
                    subagent_hook_output = await self.hooks_manager.execute_hooks(
                        HookEvent.PRE_SUBAGENT_CALL, subagent_hook_input
                    )
                    if subagent_hook_output.permission_decision == PermissionDecision.DENY:
                        reason = subagent_hook_output.permission_reason or "Subagent call blocked by policy"
                        logging.warning(f"[HOOKS] Subagent denied: {hook_feature_name}.{tool_name} - {reason}")
                        result = {"success": False, "error": f"Permission denied: {reason}"}

                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=False,
                            duration_ms=dispatch_duration,
                            error_message=reason,
                        )
                    else:
                        # Subagent hook allowed — proceed with tool execution

                        async def _exec_feature(f=feature, a=args):
                            task = a.get("task", "")
                            context = a.get("context")
                            if not context and user_message:
                                context = f"User's original request: {user_message}"
                            logging.info(f"Dispatching to feature subagent: {f.tool_name}")
                            r = await f.execute_as_subagent(task=task, context=context)
                            self._register_explored_feature_tools(f)
                            return r

                        try:
                            result = await self._execute_tool_with_hooks(
                                tool_name=tool_name,
                                feature_name=hook_feature_name,
                                args=args,
                                session_id="orchestrator",
                                execute_fn=_exec_feature,
                            )

                            # Log success
                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=True,
                                duration_ms=dispatch_duration,
                            )

                            # Fire POST_SUBAGENT_CALL hook (non-blocking, parallel)
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response=result if isinstance(result, dict) else {"result": str(result)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )
                        except (ConnectionError, TimeoutError, ValueError, KeyError, TypeError, AttributeError) as e:
                            logging.error(f"Feature {tool_name} execution failed: {e}")
                            result = {"success": False, "error": str(e)}

                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=False,
                                duration_ms=dispatch_duration,
                                error_message=str(e),
                            )

                            # Fire POST_SUBAGENT_CALL hook on failure
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response={"success": False, "error": str(e)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )
                        except Exception as e:
                            logging.error(f"Feature {tool_name} execution failed: {e}", exc_info=True)
                            result = {"success": False, "error": str(e)}

                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=False,
                                duration_ms=dispatch_duration,
                                error_message=str(e),
                            )

                            # Fire POST_SUBAGENT_CALL hook on failure
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response={"success": False, "error": str(e)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )
                elif tool_name in self._direct_tools:
                    # Direct tool execution — no subagent LLM hop
                    tool = self._direct_tools[tool_name]
                    hook_feature_name = self._tool_to_feature.get(tool_name, tool_name)

                    async def _exec_direct(t=tool, a=args):
                        return await t.execute(**a)

                    try:
                        result = await self._execute_tool_with_hooks(
                            tool_name=tool_name,
                            feature_name=hook_feature_name,
                            args=args,
                            session_id="orchestrator",
                            execute_fn=_exec_direct,
                        )

                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=True,
                            duration_ms=dispatch_duration,
                        )
                        logging.info(f"[DIRECT-TOOL] {tool_name} ({dispatch_duration}ms)")
                    except Exception as e:
                        logging.error(f"[DIRECT-TOOL] {tool_name} failed: {e}")
                        result = {"success": False, "error": str(e)}

                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=False,
                            duration_ms=dispatch_duration,
                            error_message=str(e),
                        )

                else:
                    result = {"success": False, "error": f"Unknown feature tool: {tool_name}"}

                    # Log unknown tool error
                    dispatch_duration = int((time.time() - dispatch_start) * 1000)
                    await self.observability_store.log_tool_response(
                        event_id=dispatch_event_id,
                        success=False,
                        duration_ms=dispatch_duration,
                        error_message=f"Unknown feature tool: {tool_name}",
                    )
                    await self.observability_store.log_error(
                        agent_name=self.did,
                        error_type="unknown_feature_tool",
                        error_message=f"Unknown feature tool: {tool_name}",
                        metadata={"tool_name": tool_name, "available": list(features_by_tool_name.keys())}
                    )

                # Add tool result to messages (serialize dataclasses, enums, etc.)
                from kestrel_sovereign.features.base import _serialize_tool_result
                result_json = json.dumps(_serialize_tool_result(result))

                # Truncate oversized tool results to prevent context blowout
                if len(result_json) > MAX_TOOL_RESULT_CHARS:
                    truncated_len = len(result_json)
                    result_json = result_json[:MAX_TOOL_RESULT_CHARS] + f'\n... [truncated {truncated_len - MAX_TOOL_RESULT_CHARS} chars]'
                    logging.warning(f"[ORCHESTRATOR] Truncated tool result from {truncated_len} to {MAX_TOOL_RESULT_CHARS} chars")

                logging.info(f"[ORCHESTRATOR] Adding tool result ({len(result_json)} chars): {result_json[:200]}...")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_json
                })

            # Continue conversation with tool results (include newly explored direct tools)
            all_tools = self._build_all_tools()

            # Context windowing: prune old tool exchanges if approaching limit
            messages = self._prune_orchestrator_messages(messages, all_tools)

            logging.info(f"[ORCHESTRATOR] Calling LLM with {len(messages)} messages, {len(all_tools)} tools")
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                tools=all_tools or None,
                force_local_only=force_local_only,
                model_override=effective_model
            )

            # If response is string or has no more tool calls, we're done
            if isinstance(response, str):
                logging.info(f"[ORCHESTRATOR] Final response (string): {response[:300]}...")
                return response

            if not response.has_tool_calls:
                final_content = response.content or ""
                logging.info(f"[ORCHESTRATOR] Final response (no more tool calls): {final_content[:300]}...")
                return final_content

            # Add assistant response with new tool calls to messages
            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # ollama library expects arguments as dict, not string
                            "arguments": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments) if tc.arguments else {}
                        }
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

        logging.warning("Max tool call iterations reached")
        return response.content or "Error: Maximum tool call iterations exceeded"

    def _prune_orchestrator_messages(
        self, messages: list, tools: list, context_limit: int = None
    ) -> list:
        """Prune orchestrator messages to stay within context limits.

        Strategy: keep system + user messages (first 2), keep the most recent
        assistant+tool exchange, and progressively drop older tool results
        (replacing with summaries) until we fit.

        Uses char-based estimation: ~4 chars per token.
        """
        if context_limit is None:
            # Try to get from LLM service's provider info
            context_limit = getattr(self.llm_service, '_context_limit', None) or 131072

        # Estimate tokens from chars (conservative: 1 token ≈ 3.5 chars)
        chars_per_token = 3.5
        max_chars = int(context_limit * chars_per_token * (1 - CONTEXT_RESERVE_FRACTION))

        # Also account for tool definitions (~500 chars each)
        tool_chars = sum(len(json.dumps(t)) for t in tools) if tools else 0
        max_message_chars = max_chars - tool_chars

        def _total_chars(msgs):
            total = 0
            for m in msgs:
                total += len(m.get("content", "") or "")
                for tc in m.get("tool_calls", []):
                    total += len(json.dumps(tc.get("function", {}).get("arguments", {})))
            return total

        current = _total_chars(messages)

        if current <= max_message_chars:
            return messages  # Fits fine

        logging.warning(
            f"[ORCHESTRATOR] Context pressure: ~{int(current / chars_per_token)}tok "
            f"messages + ~{int(tool_chars / chars_per_token)}tok tools "
            f"vs {context_limit}tok limit. Pruning old tool results."
        )

        # Keep system (idx 0) and user (idx 1) messages always.
        # Prune from idx 2 forward, oldest tool results first.
        # We need to keep assistant+tool pairs together for API validity.
        protected = messages[:2]  # system + user
        middle = messages[2:]

        # Find tool messages (pruneable) — go from oldest to newest
        for i, msg in enumerate(middle):
            if _total_chars(protected + middle) <= max_message_chars:
                break
            if msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                original_len = len(msg["content"])
                # Replace with a compact summary
                middle[i] = {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": f"[Result truncated: was {original_len} chars. Tool completed successfully.]"
                }

        result = protected + middle
        new_total = _total_chars(result)
        logging.info(
            f"[ORCHESTRATOR] After pruning: ~{int(new_total / chars_per_token)}tok "
            f"(removed ~{int((current - new_total) / chars_per_token)}tok)"
        )
        return result

    async def _handle_orchestrator_response_streaming(
        self,
        response,
        feature_tools: list,
        system_prompt: str,
        force_local_only: bool,
        effective_model: str,
        max_iterations: int = None,
        user_message: str = None,
        tool_events: list = None,
    ):
        """
        Streaming version of _handle_orchestrator_response.

        Executes tool calls synchronously, then streams the final LLM response.
        Tool execution cannot be streamed (we need complete results), but the
        final text response streams token-by-token.

        Yields:
            Text chunks as they arrive from the LLM
        """
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        # If response is just a string, yield it directly
        if isinstance(response, str):
            yield response
            return

        # If response has no tool calls, yield the content
        if not response.has_tool_calls:
            yield response.content or ""
            return

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if user_message:
            messages.append({"role": "user", "content": user_message})
            logging.debug(f"[ORCHESTRATOR-STREAM] Added user message to context: {user_message[:100]}...")
        else:
            logging.warning(f"[ORCHESTRATOR-STREAM] No user_message provided!")

        # Add initial assistant response with tool calls
        assistant_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments) if tc.arguments else {}
                    }
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        features_by_tool_name = {f.tool_name: f for f in self.features.values()}

        # Build known tool allowlist for argument validation
        known_tools = set(features_by_tool_name.keys()) | set(self._direct_tools.keys())

        for iteration in range(max_iterations):
            # Execute tool calls - stream activity indicators to user
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

                # Validate tool arguments before execution
                is_valid, validation_error = validate_tool_arguments(
                    tool_name, args, known_tools=known_tools
                )
                if not is_valid:
                    logging.warning(f"[ORCHESTRATOR-STREAM] Tool validation failed: {validation_error}")
                    result = {"success": False, "error": f"Tool validation failed: {validation_error}"}
                    from kestrel_sovereign.features.base import _serialize_tool_result
                    result_json = json.dumps(_serialize_tool_result(result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json
                    })
                    continue

                # Stream tool start indicator to user
                if tool_events is not None:
                    tool_events.append({'type': 'start', 'tool': tool_name})
                yield f"🔧 Calling {tool_name}...\n"

                dispatch_start = time.time()
                dispatch_event_id = await self.observability_store.log_tool_call(
                    agent_name=self.did,
                    tool_name=f"feature_dispatch:{tool_name}",
                    metadata={"arguments": args, "iteration": iteration}
                )

                feature = features_by_tool_name.get(tool_name)
                if feature:
                    hook_feature_name = type(feature).__name__

                    # --- PRE_SUBAGENT_CALL hooks (feature-level security) ---
                    subagent_hook_input = HookInput(
                        session_id="orchestrator",
                        hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
                        tool_name=tool_name,
                        tool_input=args,
                        feature_name=hook_feature_name,
                    )
                    subagent_hook_output = await self.hooks_manager.execute_hooks(
                        HookEvent.PRE_SUBAGENT_CALL, subagent_hook_input
                    )
                    if subagent_hook_output.permission_decision == PermissionDecision.DENY:
                        reason = subagent_hook_output.permission_reason or "Subagent call blocked by policy"
                        logging.warning(f"[HOOKS] Subagent denied: {hook_feature_name}.{tool_name} - {reason}")
                        result = {"success": False, "error": f"Permission denied: {reason}"}


                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=False,
                            duration_ms=dispatch_duration,
                            error_message=reason,
                        )

                        if tool_events is not None:
                            tool_events.append({'type': 'error', 'tool': tool_name, 'error': reason[:200]})
                        yield f"🚫 {tool_name} blocked by policy: {reason[:100]}\n"
                    else:
                        # Subagent hook allowed — proceed with tool execution

                        async def _exec_feature_stream(f=feature, a=args):
                            task = a.get("task", "")
                            context = a.get("context")
                            if not context and user_message:
                                context = f"User's original request: {user_message}"
                            logging.info(f"[STREAM] Dispatching to feature subagent: {f.tool_name}")
                            r = await f.execute_as_subagent(task=task, context=context)
                            self._register_explored_feature_tools(f)
                            return r

                        try:
                            result = await self._execute_tool_with_hooks(
                                tool_name=tool_name,
                                feature_name=hook_feature_name,
                                args=args,
                                session_id="orchestrator",
                                execute_fn=_exec_feature_stream,
                            )

                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=True,
                                duration_ms=dispatch_duration,
                            )

                            # Fire POST_SUBAGENT_CALL hook (non-blocking, parallel)
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response=result if isinstance(result, dict) else {"result": str(result)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )

                            if tool_events is not None:
                                tool_events.append({'type': 'complete', 'tool': tool_name, 'ms': dispatch_duration})
                            yield f"✓ {tool_name} complete ({dispatch_duration}ms)\n"
                        except (ConnectionError, TimeoutError, ValueError, KeyError, TypeError, AttributeError) as e:
                            logging.error(f"Feature {tool_name} execution failed: {e}")
                            result = {"success": False, "error": str(e)}
                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=False,
                                duration_ms=dispatch_duration,
                                error_message=str(e),
                            )

                            # Fire POST_SUBAGENT_CALL hook on failure
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response={"success": False, "error": str(e)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )

                            if tool_events is not None:
                                tool_events.append({'type': 'error', 'tool': tool_name, 'error': str(e)[:200]})
                            yield f"❌ {tool_name} failed: {str(e)[:100]}\n"
                        except Exception as e:
                            logging.error(f"Feature {tool_name} execution failed: {e}", exc_info=True)
                            result = {"success": False, "error": str(e)}
                            dispatch_duration = int((time.time() - dispatch_start) * 1000)
                            await self.observability_store.log_tool_response(
                                event_id=dispatch_event_id,
                                success=False,
                                duration_ms=dispatch_duration,
                                error_message=str(e),
                            )

                            # Fire POST_SUBAGENT_CALL hook on failure
                            post_hook_input = HookInput(
                                session_id="orchestrator",
                                hook_event_name=HookEvent.POST_SUBAGENT_CALL.value,
                                tool_name=tool_name,
                                tool_input=args,
                                feature_name=hook_feature_name,
                                tool_response={"success": False, "error": str(e)},
                                execution_time_ms=dispatch_duration,
                            )
                            await self.hooks_manager.execute_hooks_parallel(
                                HookEvent.POST_SUBAGENT_CALL, post_hook_input
                            )

                            if tool_events is not None:
                                tool_events.append({'type': 'error', 'tool': tool_name, 'error': str(e)[:200]})
                            yield f"❌ {tool_name} failed: {str(e)[:100]}\n"

                elif tool_name in self._direct_tools:
                    # Direct tool execution — no subagent LLM hop
                    tool = self._direct_tools[tool_name]
                    hook_feature_name = self._tool_to_feature.get(tool_name, tool_name)

                    async def _exec_direct_stream(t=tool, a=args):
                        return await t.execute(**a)

                    try:
                        result = await self._execute_tool_with_hooks(
                            tool_name=tool_name,
                            feature_name=hook_feature_name,
                            args=args,
                            session_id="orchestrator",
                            execute_fn=_exec_direct_stream,
                        )

                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=True,
                            duration_ms=dispatch_duration,
                        )
                        if tool_events is not None:
                            tool_events.append({'type': 'complete', 'tool': tool_name, 'ms': dispatch_duration})
                        yield f"⚡ {tool_name} (direct, {dispatch_duration}ms)\n"
                    except Exception as e:
                        logging.error(f"[DIRECT-TOOL] {tool_name} failed: {e}")
                        result = {"success": False, "error": str(e)}

                        dispatch_duration = int((time.time() - dispatch_start) * 1000)
                        await self.observability_store.log_tool_response(
                            event_id=dispatch_event_id,
                            success=False,
                            duration_ms=dispatch_duration,
                            error_message=str(e),
                        )
                        if tool_events is not None:
                            tool_events.append({'type': 'error', 'tool': tool_name, 'error': str(e)[:200]})
                        yield f"❌ {tool_name} failed: {str(e)[:100]}\n"

                else:
                    result = {"success": False, "error": f"Unknown feature tool: {tool_name}"}
                    dispatch_duration = int((time.time() - dispatch_start) * 1000)
                    await self.observability_store.log_tool_response(
                        event_id=dispatch_event_id,
                        success=False,
                        duration_ms=dispatch_duration,
                        error_message=f"Unknown feature tool: {tool_name}",
                    )
                    if tool_events is not None:
                        tool_events.append({'type': 'error', 'tool': tool_name, 'error': f'Unknown feature tool: {tool_name}'})
                    yield f"❌ Unknown tool: {tool_name}\n"

                from kestrel_sovereign.features.base import _serialize_tool_result
                result_json = json.dumps(_serialize_tool_result(result))

                # Truncate oversized tool results
                if len(result_json) > MAX_TOOL_RESULT_CHARS:
                    truncated_len = len(result_json)
                    result_json = result_json[:MAX_TOOL_RESULT_CHARS] + f'\n... [truncated {truncated_len - MAX_TOOL_RESULT_CHARS} chars]'
                    logging.warning(f"[ORCHESTRATOR-STREAM] Truncated tool result from {truncated_len} to {MAX_TOOL_RESULT_CHARS} chars")

                logging.info(f"[ORCHESTRATOR-STREAM] Tool result ({len(result_json)} chars): {result_json[:200]}...")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_json
                })

            # Check if we need more tool calls with non-streaming first
            all_tools = self._build_all_tools()

            # Context windowing: prune old tool exchanges if approaching limit
            messages = self._prune_orchestrator_messages(messages, all_tools)

            logging.info(f"[ORCHESTRATOR-STREAM] Checking for more tool calls with {len(messages)} messages, {len(all_tools)} tools")
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                tools=all_tools or None,
                force_local_only=force_local_only,
                model_override=effective_model
            )

            if isinstance(response, str):
                # No more tool calls, but we got string - yield it
                yield response
                return

            if not response.has_tool_calls:
                # No more tool calls - now stream the final response
                logging.info(f"[ORCHESTRATOR-STREAM] Streaming final response")
                yield "\n---\n"  # Visual separator before final response
                async for chunk in self.llm_service.stream_with_messages(
                    messages=messages,
                    force_local_only=force_local_only,
                    model_override=effective_model
                ):
                    yield chunk
                return

            # More tool calls - add to messages and continue loop
            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments) if tc.arguments else {}
                        }
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

        logging.warning("Max tool call iterations reached")
        yield "Error: Maximum tool call iterations exceeded"

    # Streaming methods provided by StreamingMixin:
    # - process_input_streaming

    # Backup methods provided by BackupMixin:
    # - _command_backup
    # - _command_promote_backup
    # - anchor_memory_state

    # Context retrieval now delegated to self.context_builder
    # See agent/context_builder.py for ContextBuilder class

    # NOTE: The _execute_commands_in_response hack has been removed.
    # Tool execution is now handled properly via OpenAI-style function calling.
    # Features are exposed as tools to the orchestrator LLM, which can call them
    # as subagents via the A2A (Agent-to-Agent) pattern.
    # See _build_feature_tools() and _handle_orchestrator_response() methods.

    # Command handling is now delegated to self.command_handler (CommandHandler class)
    # See command_handler.py for implementation

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

    async def list_available_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        List all available models from configured LLM providers.

        Args:
            use_cache: If True, use cached models if available (default: True)

        Returns:
            List of model dictionaries with id, provider, name, description
        """
        return await self.model_agent.list_models(use_cache=use_cache)

    def set_model(self, model_id: str) -> str:
        """
        Set the LLM model for this agent.

        Args:
            model_id: The model ID to use (e.g., "gpt-5", "claude-sonnet-4-5")

        Returns:
            Confirmation message
        """
        return self.model_agent.set_model_preference(model_id)

    def get_current_model(self) -> str:
        """
        Get the current LLM model being used by this agent.

        Delegates to LLMService.get_active_model_id() as the single
        source of truth, then formats with provider prefix.

        Returns:
            Current model ID (provider/model format)
        """
        model_id = self.llm_service.get_active_model_id()

        # Find the provider for this model
        pref = self.llm_service.get_model_preference()
        if pref.get("provider"):
            return f"{pref['provider']}/{model_id}"

        if self.llm_service.providers:
            provider = self.llm_service.providers[0].get('name', '')
            if provider:
                return f"{provider}/{model_id}"

        return model_id

    MODEL_PREFERENCE_KEY = "model_preference"

    async def _load_model_preference(self) -> None:
        """Load persisted model preference from agent_metadata table."""
        try:
            result = await self._raw_storage.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.MODEL_PREFERENCE_KEY),
            )
            if result:
                import json
                pref = json.loads(result[0][0])
                model = pref.get("model")
                provider = pref.get("provider")
                if model and model != "auto":
                    self.llm_service.set_model_preference(model, provider)
                    logging.info(f"Loaded persisted model preference: {provider}/{model}" if provider else f"Loaded persisted model preference: {model}")
        except Exception as e:
            logging.warning(f"Failed to load model preference: {e}")

    async def _persist_model_preference(self, model: str | None, provider: str | None) -> None:
        """Persist model preference to agent_metadata table."""
        try:
            import json
            from datetime import datetime, timezone
            value = json.dumps({"model": model, "provider": provider})
            await self._raw_storage.db.execute(
                """INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (self.agent_id, self.MODEL_PREFERENCE_KEY, value, datetime.now(timezone.utc)),
            )
        except Exception as e:
            logging.warning(f"Failed to persist model preference: {e}")

    def _get_local_model_fallback(self) -> str:
        """Get the configured local (ollama) model for economy/solvency fallback."""
        # Check ollama provider in the providers list
        for provider in self.llm_service.providers:
            if provider.get("name") == "ollama":
                return provider.get("model", "auto")
        # Fall back to config
        if hasattr(self.llm_service, 'config'):
            return self.llm_service.config.get("ollama", {}).get("model", "auto")
        return "auto"

    async def check_solvency(self) -> str:
        """
        Checks the agent's wallet balance and determines the economic operating mode.
        Returns the model preference based on solvency.

        Uses total USD-equivalent value across all currencies so that agents holding
        ETH, MATIC, or other non-FIL assets are correctly classified as solvent.
        FIL balance is checked as a fallback for FIL-only wallets.
        """
        try:
            fil_balance = self.wallet.get_balance()
            usd_balance = self.wallet.get_total_balance_usd()

            # Green Zone: > $5 USD equivalent (or > 10 FIL for FIL-only wallets)
            if usd_balance > Decimal("5.0") or fil_balance > Decimal("10.0"):
                if self._current_model_preference != "NORMAL":
                    logging.info(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"Operating in NORMAL mode."
                    )
                    self._current_model_preference = "NORMAL"
                return None  # No override, use default/mandated models

            # Yellow Zone: > $0.50 USD equivalent (or > 1 FIL)
            elif usd_balance > Decimal("0.50") or fil_balance > Decimal("1.0"):
                if self._current_model_preference != "ECONOMY":
                    logging.warning(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"Switching to ECONOMY mode (Local Models)."
                    )
                    self._current_model_preference = "ECONOMY"
                return self._get_local_model_fallback()

            # Red Zone: Critical (< $0.50 USD and < 1 FIL)
            else:
                if self._current_model_preference != "CRITICAL":
                    logging.error(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"CRITICAL SOLVENCY. Forced to minimal model."
                    )
                    self._current_model_preference = "CRITICAL"
                return self._get_local_model_fallback()

        except Exception as e:
            logging.error(f"Solvency check failed: {e}", exc_info=True)
            return None

    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to all registered listeners (for SSE notifications).

        Args:
            event_type: Type of event (e.g., 'approval_request')
            data: Event data to send
        """
        for listener in self._event_listeners:
            try:
                await listener(event_type, data)
            except (TypeError, AttributeError, ConnectionError) as e:
                logging.warning(f"Failed to emit event to listener: {e}")
            except Exception as e:
                logging.warning(f"Failed to emit event to listener: {e}", exc_info=True)

    def add_event_listener(self, listener) -> None:
        """Add an event listener for SSE notifications."""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener) -> None:
        """Remove an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def _on_background_task_complete(self, task) -> None:
        """
        Callback invoked when a background task completes.

        Queues a notification message to be included in the next chat response.
        This is called by TaskManager when tasks reach terminal states
        (COMPLETED, FAILED, CANCELED).
        """
        from kestrel_sovereign.a2a.types import TaskState

        state = task.status.state
        task_id = task.id

        # Get task description from metadata
        agent_id = task.metadata.get("agent_id", "unknown") if task.metadata else "unknown"
        skill_id = task.metadata.get("skill", "task") if task.metadata else "task"

        # Format notification based on state
        if state == TaskState.COMPLETED:
            msg = f"✅ Background task completed: {agent_id}/{skill_id} (task: {task_id[:8]})"
        elif state == TaskState.FAILED:
            error_msg = ""
            if task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if hasattr(part, 'text'):
                        error_msg = f": {part.text}"
                        break
            msg = f"❌ Background task failed: {agent_id}/{skill_id}{error_msg} (task: {task_id[:8]})"
        elif state == TaskState.CANCELED:
            msg = f"⚠️ Background task canceled: {agent_id}/{skill_id} (task: {task_id[:8]})"
        else:
            return  # Don't notify for non-terminal states

        self._pending_task_notifications.append(msg)
        logging.info(f"Queued task notification: {msg}")

    def get_pending_notifications(self) -> List[str]:
        """
        Get and clear pending task completion notifications.

        Called by the chat endpoint to include notifications in responses.
        """
        notifications = self._pending_task_notifications.copy()
        self._pending_task_notifications.clear()
        return notifications

    # =========================================================================
    # Request Cancellation (Stop Button Support)
    # =========================================================================

    def register_active_request(self, request_id: str) -> None:
        """Track an active request for later cancellation and cleanup."""
        if not hasattr(self, "_active_request_ids"):
            self._active_request_ids = set()
        self._active_request_ids.add(request_id)
        # Preserve the legacy "current request" fallback for callers that
        # do not yet pass an explicit request ID.
        self._current_request_id = request_id

    def cancel_current_request(self, request_id: Optional[str] = None) -> bool:
        """
        Cancel the current streaming request.
        
        Returns:
            True if a request was cancelled, False if no request was active.
        """
        active_request_ids = getattr(self, "_active_request_ids", set())
        target_request_id = request_id or self._current_request_id
        if target_request_id and (
            target_request_id in active_request_ids
            or target_request_id == self._current_request_id
        ):
            self._cancelled_requests.add(target_request_id)
            logging.info(f"Cancelled request: {target_request_id}")
            return True
        return False

    def is_request_cancelled(self, request_id: Optional[str] = None) -> bool:
        """Check if a request has been cancelled."""
        rid = request_id or self._current_request_id
        return rid in self._cancelled_requests if rid else False

    def _cleanup_cancelled_request(self, request_id: str):
        """Remove a request from the cancelled set after it's been handled."""
        active_request_ids = getattr(self, "_active_request_ids", None)
        if active_request_ids is not None:
            active_request_ids.discard(request_id)
        self._cancelled_requests.discard(request_id)
        if self._current_request_id == request_id:
            self._current_request_id = next(iter(active_request_ids), None) if active_request_ids else None

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
