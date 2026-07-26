"""Isolated feature runtime proxy and per-agent venv provisioning.

A feature distribution opts into out-of-venv execution via its pyproject:

    [tool.kestrel.feature]
    runtime = "isolated-venv"
    service = "kestrel-whatsapp-web"   # runnable: a console-script name, or "module:func"
    project = "service"                # install target for the venv (path/dist); defaults to the distribution
    # venv  = "/abs/path/.venv"        # optional explicit venv-path override

`service` is the thing to RUN (resolved from the per-agent venv's bin/ as a
console script, or executed as a "module:func" callable). `project` is the
thing to INSTALL. They are deliberately distinct so the runnable is never
mistaken for a pip target or a `python -m` module.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from kestrel_sdk.channels import ChannelAdapter
from kestrel_sdk.isolated_feature import (
    CONFIG_TRANSITION_APPLIED,
    ConfigTransitionResult,
)
from kestrel_sdk.tools.base import AgentTool, ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.base import Feature, UIContributions

logger = logging.getLogger(__name__)

# Upper bound on a single supervision health probe. A wedged child that never
# answers health() must not silently kill supervision forever (F013) — treat a
# probe that exceeds this as unhealthy and fall through to the restart path.
_HEALTH_PROBE_TIMEOUT = 5.0

# ``ToolExecutionTrigger`` validates identifiers by UTF-8 byte length, not
# Python character count.  Schedule IDs normally fit unchanged, but legacy
# imports can contain arbitrarily long IDs while the SDK context permits at
# most 512 bytes.  Keep the full ID in SchedulerExecution/storage and expose a
# deterministic, plainly tagged provenance surrogate only at that bounded wire
# boundary.
_SDK_CONTEXT_IDENTIFIER_MAX_BYTES = 512
_SCHEDULE_TRIGGER_SOURCE_HASH_PREFIX = "schedule-sha256:"

# Transition metadata lives beside the existing ``config`` property on the
# feature-config graph node.  The keys are deliberately implementation-private:
# callers continue to read the long-standing ``config`` dictionary, while the
# proxy uses the opaque generation token to prove that a later promotion is its
# own write rather than a same-shaped write from another hosted replica.
_CONFIG_GENERATION_KEY = "_isolated_config_generation"
_PENDING_GENERATION_KEY = "_isolated_pending_generation"
_PENDING_OWNER_KEY = "_isolated_pending_owner"
_PENDING_LEASE_EXPIRES_AT_KEY = "_isolated_pending_lease_expires_at"

# A staged config must survive a short process pause, but it must not turn an
# interrupted deploy or process death into a permanent write lock.  Readers
# wait an additional skew allowance before takeover: a replica whose clock is
# ahead cannot steal a healthy writer's lease merely because its wall clock is
# fast.  The durable timestamp is always UTC and is parsed fail-closed.
_PENDING_CONFIG_LEASE_TTL = timedelta(minutes=2)
_PENDING_CONFIG_CLOCK_SKEW = timedelta(seconds=30)
_PENDING_CLEANUP_WRITE_ATTEMPTS = 2
_TERMINAL_TRAFFIC_ERROR = "isolated feature traffic is unavailable"


class _ConfigTransitionLeaseLost(RuntimeError):
    """The durable lease for an in-flight lifecycle transition was not renewed."""


class _TrafficGateTerminalError(RuntimeError):
    """The proxy has entered a terminal no-admission lifecycle state."""

    def __init__(self) -> None:
        super().__init__(_TERMINAL_TRAFFIC_ERROR)


class _TrafficGateClosedError(RuntimeError):
    """A non-waiting admission arrived during a finite transition."""


async def _await_task_until_complete(
    task: asyncio.Task[Any],
    *,
    preserve_cancellation: bool,
) -> Any:
    """Wait for a shielded task without letting a later cancellation orphan it.

    Lifecycle cleanup, including the traffic-gate boundary, must settle before
    its owner releases ``_reload_lock``.  ``Task.result()`` after completion is
    intentional: it avoids one final cancellation point after the task has
    already changed durable/client/gate state.
    """

    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # A cancellation request is not a reason to abandon the child
            # before it records its own SDK fence or cleanup outcome.
            if task.done() and task.cancelled():
                break
            cancellation = exc
            continue
    # A child cancellation is a result of the child operation, not an empty
    # successful result.  In particular, lifecycle callers must not mistake a
    # cancelled terminal cleanup for a completed one and release ownership
    # while its child or gate state remains uncertain.
    if task.cancelled():
        raise asyncio.CancelledError()
    result = task.result()
    if cancellation is not None and not preserve_cancellation:
        raise cancellation
    return result


class _TrafficGate:
    """A small reader/writer gate around externally visible child traffic.

    Normal tool and event delivery takes the shared side only long enough to
    keep its selected child stable.  A config transition takes the exclusive
    side: it closes admission first, drains calls that were already executing,
    and does not reopen until the caller has reconciled client and durable
    state.  This is deliberately not the reload lock: serialising every normal
    tool call would both hurt throughput and make a long tool execution block
    unrelated calls when no transition is taking place.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._closed = False
        self._sealed = False
        self._active = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def sealed(self) -> bool:
        return self._sealed

    async def close_and_drain(self) -> None:
        async with self._condition:
            if self._sealed:
                raise _TrafficGateTerminalError()
            self._closed = True
            while self._active:
                await self._condition.wait()
            if self._sealed:
                raise _TrafficGateTerminalError()

    async def seal_and_drain(self) -> None:
        """Make admission terminal before waiting for admitted calls to finish.

        The state flip and notification occur under the same condition lock, so
        waiters from a finite transition cannot be left asleep when quarantine
        or shutdown becomes permanent.  Already admitted calls retain their
        client until they return; the caller can then retire it safely.
        """

        async with self._condition:
            self._sealed = True
            self._closed = True
            self._condition.notify_all()
            while self._active:
                await self._condition.wait()

    async def reopen(self) -> None:
        async with self._condition:
            # A terminal proxy may only be reset by an explicit successful
            # initialize.  Cleanup for a cancelled transition must never turn
            # a quarantine/shutdown into an accidental reopen.
            if not self._sealed:
                self._closed = False
                self._condition.notify_all()

    async def reset_and_reopen(self) -> None:
        """Reset terminal admission after a durable child initialization."""

        async with self._condition:
            self._sealed = False
            self._closed = False
            self._condition.notify_all()

    async def _release_admission(self) -> None:
        async with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    @asynccontextmanager
    async def admit(self, *, wait_for_open: bool = True):
        async with self._condition:
            while self._closed and not self._sealed:
                if not wait_for_open:
                    raise _TrafficGateClosedError()
                await self._condition.wait()
            if self._sealed:
                raise _TrafficGateTerminalError()
            self._active += 1
        try:
            yield
        finally:
            release = asyncio.create_task(self._release_admission())
            await _await_task_until_complete(release, preserve_cancellation=False)


def _utc_now() -> datetime:
    """Return the UTC wall clock used for durable config lease decisions."""

    return datetime.now(timezone.utc)


@dataclass
class _ConfigState:
    """One authoritative snapshot of an isolated feature's config node."""

    properties: Optional[Dict[str, Any]] = field(repr=False)
    config: Dict[str, Any] = field(repr=False)
    has_pending: bool = False
    pending_generation: Optional[str] = None
    pending_owner: Optional[str] = None
    pending_lease_expires_at: Optional[datetime] = None


@dataclass
class _ConfigTransition:
    """A generation-owned stage → promote transaction.

    ``expected_properties`` and ``staged_properties`` are exact graph-store
    snapshots.  They are never reconstructed from the proxy cache: a hosted
    replica must only promote the pending generation it actually staged.
    """

    active_config: Dict[str, Any] = field(repr=False)
    next_config: Dict[str, Any] = field(repr=False)
    persistent: bool
    storage: Any = field(repr=False)
    expected_properties: Optional[Dict[str, Any]] = field(repr=False)
    staged_properties: Optional[Dict[str, Any]] = field(repr=False)
    promoted_properties: Optional[Dict[str, Any]] = field(repr=False)
    generation: Optional[str] = None
    owner: Optional[str] = None


@dataclass
class _ConfigWriteResult:
    """The direct result of one graph write, before ambiguity is reconciled."""

    committed: bool
    error: Optional[BaseException] = field(default=None, repr=False)


@dataclass
class _PromotionResolution:
    """Durable outcome of a promotion after any ambiguous write is re-read."""

    state: _ConfigState
    committed: bool
    error: Optional[BaseException] = field(default=None, repr=False)
    storage_error: bool = False


@dataclass
class _PendingCleanupResolution:
    """Outcome of a generation-scoped pending-state cleanup attempt."""

    state: _ConfigState
    cleared: bool


class SchedulerExecutionContextUnavailable(RuntimeError):
    """A scheduled isolated call cannot carry its trusted effect identity.

    A normal interactive isolated-tool invocation remains compatible with
    legacy SDK services.  A scheduler delivery does not: omitting its
    occurrence identity would let an isolated tool perform an effect without
    the idempotency key that makes lease recovery safe.
    """


def _scheduler_trigger_source_id(schedule_id: str) -> str:
    """Return an SDK-safe, stable source ID for a scheduler occurrence.

    The SDK's execution context uses a 512-byte UTF-8 identifier limit.  A
    migrated schedule can legitimately have a longer database ID, so passing
    it through would reject an otherwise safe execution *after* it was
    claimed.  Retain fitting IDs verbatim for compatibility; hash only the
    oversized representation.  Invalid persisted IDs fail closed rather than
    being erased or silently replaced.
    """

    if not isinstance(schedule_id, str) or not schedule_id:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has an invalid schedule_id"
        )
    if len(schedule_id.encode("utf-8")) <= _SDK_CONTEXT_IDENTIFIER_MAX_BYTES:
        return schedule_id
    digest = hashlib.sha256(schedule_id.encode("utf-8")).hexdigest()
    return f"{_SCHEDULE_TRIGGER_SOURCE_HASH_PREFIX}{digest}"


def _scheduled_tool_execution_context() -> Any | None:
    """Translate the active scheduler occurrence into the public SDK context.

    Keep this lookup lazy so the core continues to start with the currently
    published SDK.  Once a scheduled isolated invocation is attempted, the
    missing SDK contract is a safety failure rather than a reason to smuggle
    scheduler fields into user-controlled tool arguments.
    """

    from kestrel_sovereign.features.scheduler.runner import (
        get_current_scheduler_execution,
    )

    execution = get_current_scheduler_execution()
    if execution is None:
        return None

    try:
        from kestrel_sdk.isolated_feature import (
            ToolExecutionContext,
            ToolExecutionTrigger,
        )
    except ImportError as exc:
        raise SchedulerExecutionContextUnavailable(
            "scheduled isolated tool calls require an SDK with "
            "ToolExecutionContext support"
        ) from exc

    try:
        scheduled_for = datetime.fromisoformat(
            execution.scheduled_for.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has an invalid scheduled_for timestamp"
        ) from exc
    if scheduled_for.tzinfo is None or scheduled_for.utcoffset() is None:
        raise SchedulerExecutionContextUnavailable(
            "scheduled occurrence has a timezone-naive scheduled_for timestamp"
        )

    return ToolExecutionContext(
        invocation_id=execution.id,
        idempotency_key=execution.idempotency_key,
        attempt=execution.attempt,
        trigger=ToolExecutionTrigger(
            kind="scheduler",
            id=execution.id,
            source_id=_scheduler_trigger_source_id(execution.schedule_id),
            triggered_at=datetime.now(timezone.utc),
            scheduled_for=scheduled_for.astimezone(timezone.utc),
        ),
    )


def _host_sdk_version() -> str:
    """The kestrel-sdk version resolved in the *host* process, used as the
    provisioning stamp for isolated venvs so a host SDK upgrade forces the
    per-agent venv to reprovision instead of pinning a stale wire contract.

    The ``kestrel_sdk`` import package is shipped by the ``kestrel-sovereign-sdk``
    distribution, so resolve the distribution from the import name rather than
    guessing — a hardcoded wrong name would silently stamp ``unknown`` forever
    and defeat stale detection.
    """
    try:
        candidates = importlib_metadata.packages_distributions().get("kestrel_sdk")
    except Exception:  # noqa: BLE001
        candidates = None
    for dist in list(candidates or []) + ["kestrel-sovereign-sdk", "kestrel-sdk"]:
        try:
            return importlib_metadata.version(dist)
        except Exception:  # noqa: BLE001
            continue
    return "unknown"


# Probe run *inside* a feature venv to report the kestrel-sdk version actually
# installed there — mirrors _host_sdk_version's distribution resolution.
_CHILD_SDK_PROBE = (
    "from importlib import metadata as m\n"
    "def v():\n"
    "    try: c = m.packages_distributions().get('kestrel_sdk')\n"
    "    except Exception: c = None\n"
    "    for d in list(c or []) + ['kestrel-sovereign-sdk', 'kestrel-sdk']:\n"
    "        try: return m.version(d)\n"
    "        except Exception: continue\n"
    "    return 'unknown'\n"
    "print(v())\n"
)


def _venv_sdk_version(python_path: Path) -> str:
    """The kestrel-sdk version resolved *inside* the feature venv (may differ
    from the host when the feature pins the dependency)."""
    try:
        res = subprocess.run(
            [str(python_path), "-c", _CHILD_SDK_PROBE],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_key(feature_name: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", feature_name).upper()
    return f"KESTREL_FEATURE_{normalized}_{suffix}"


def _agent_data_dir(agent: Any) -> Path:
    storage_path = getattr(agent, "storage_path", None)
    if storage_path:
        return Path(storage_path).expanduser().resolve().parent
    return (Path.cwd() / "agent_data" / "default").resolve()


def _venv_bin_dir(venv_path: Path) -> Path:
    return venv_path / ("Scripts" if os.name == "nt" else "bin")


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


# Interpreter-behavior env vars that would let the HOST Python installation
# shadow the isolated venv's packages, defeating the isolation the runtime
# exists for (F023). Feature config/secrets ride through the general environment
# intentionally (KESTREL_FEATURE_* is the documented config channel), so we STRIP
# these specific interpreter vars rather than allowlisting the whole environment.
_SHADOWING_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV")


def _isolated_child_env(venv_path: Optional[Path]) -> Dict[str, str]:
    """Build the launch environment for the isolated service subprocess.

    Inherits the host environment (so feature config/secrets pass through) but
    strips the interpreter-behavior vars in ``_SHADOWING_ENV_VARS`` so a stray
    host ``PYTHONPATH``/``VIRTUAL_ENV`` can't resolve the service's imports
    against host site-packages. When a venv is used, re-point ``VIRTUAL_ENV`` at
    it and prepend its bin dir to ``PATH`` so child processes bind to the
    isolated venv.
    """
    env = dict(os.environ)
    for var in _SHADOWING_ENV_VARS:
        env.pop(var, None)
    if venv_path is not None:
        env["VIRTUAL_ENV"] = str(venv_path)
        bin_dir = str(_venv_bin_dir(venv_path))
        env["PATH"] = os.pathsep.join(
            [bin_dir, env.get("PATH", "")]
        ).rstrip(os.pathsep)
    return env


def _coerce_category(value: Any) -> ToolCategory:
    if isinstance(value, ToolCategory):
        return value
    if value is None:
        return ToolCategory.SYSTEM
    text = str(value)
    for category in ToolCategory:
        if text == category.value or text.upper() == category.name:
            return category
    return ToolCategory.SYSTEM


def _meta_get(metadata: Any, key: str, default: Any = None) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _input_schema_to_parameters(input_schema: Any) -> List[ToolParameter]:
    """Convert an isolated service's advertised JSON-Schema ``input_schema``
    (the SDK wire contract, ``ToolMetadata.input_schema``) into the
    ``List[ToolParameter]`` a host ``ToolSchema`` expects.

    Without this the proxied tool reaches the LLM with an empty parameter list,
    so the model cannot supply arguments (F004). Passing the raw dict through is
    also wrong: ``ToolSchema.to_openai_format`` iterates ``ToolParameter``
    objects and would crash on a dict/string.
    """
    if not isinstance(input_schema, dict):
        return []
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return []
    raw_required = input_schema.get("required")
    required = set(raw_required) if isinstance(raw_required, (list, tuple, set)) else set()

    params: List[ToolParameter] = []
    for pname, pdef in properties.items():
        pdef = pdef if isinstance(pdef, dict) else {}
        ptype = pdef.get("type")
        if isinstance(ptype, (list, tuple)):
            # JSON-Schema union, e.g. ["string", "null"] for an Optional — take
            # the first non-null member.
            ptype = next((t for t in ptype if t != "null"), None)
        params.append(
            ToolParameter(
                name=str(pname),
                type=str(ptype or "string"),
                description=str(pdef.get("description", "")),
                required=pname in required,
                default=pdef.get("default"),
                enum=pdef.get("enum"),
                items=pdef.get("items"),
            )
        )
    return params


def _maybe_await(value: Any) -> Awaitable[Any]:
    if inspect.isawaitable(value):
        return value

    async def _wrapped() -> Any:
        return value

    return _wrapped()


class IsolatedFeatureTool(AgentTool):
    """Tool wrapper that forwards execution to an isolated feature service."""

    def __init__(self, feature: "ProxyFeature", metadata: Any):
        self._feature = feature
        self._metadata = metadata

    @property
    def name(self) -> str:
        return str(_meta_get(self._metadata, "name", ""))

    @property
    def schema(self) -> ToolSchema:
        input_schema = _meta_get(self._metadata, "input_schema", None)
        if input_schema is None:
            # camelCase spelling tolerated on the wire (see protocol.from_dict)
            input_schema = _meta_get(self._metadata, "inputSchema", None)
        return ToolSchema(
            name=self.name,
            description=str(_meta_get(self._metadata, "description", "")),
            category=_coerce_category(_meta_get(self._metadata, "category")),
            parameters=_input_schema_to_parameters(input_schema),
            command_prefix=_meta_get(self._metadata, "command_prefix"),
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self._feature.call_isolated_tool(self.name, kwargs)


class ProxyFeature(Feature):
    """Feature contract adapter backed by an SDK isolated-feature client."""

    def __init__(
        self,
        agent: Any,
        runtime: InstalledFeatureRuntime,
        *,
        client_factory: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(agent)
        self.runtime = runtime
        self.name = runtime.class_name
        self._client_factory = client_factory
        self._client: Any = None
        self._tools: List[AgentTool] = []
        self._supervision_task: Optional[asyncio.Task] = None
        # A terminal cleanup owns sealing, lifecycle serialization, and child
        # retirement as one transaction.  Keep one shared task so repeated
        # caller cancellation cannot strand a second cleanup behind
        # ``_reload_lock`` after the first caller has unwound.
        self._terminal_cleanup_task: Optional[asyncio.Task[None]] = None
        self._stopping = False
        # Coordinate ``set_config``'s reload with the health supervisor so they
        # never stop/start the client concurrently. ``_reloading`` skips probes
        # during a reload; ``_reload_lock`` serializes the actual stop/start of
        # reload vs. a supervisor restart; ``_reload_gen`` lets the supervisor
        # detect that a reload cycled the client around its (now-stale) probe and
        # skip restarting the freshly launched one.
        self._reloading = False
        self._reload_lock = asyncio.Lock()
        self._reload_gen = 0
        self._traffic_gate = _TrafficGate()
        self._fenced_recovery_failed = False
        self._venv_path: Optional[Path] = None
        self._bin_path: Optional[Path] = None
        self._host_config: Dict[str, Any] = {}
        # Process-local identity for the durable pending lease.  A new proxy
        # instance (including one after a crash/restart) never impersonates an
        # earlier writer; it may only reclaim that writer after its lease has
        # conservatively expired.
        self._config_transition_owner = uuid4().hex
        # ``{}`` is a valid, fully-loaded config.  Keep its loaded state
        # separate from its truthiness so a concurrent read never falls back to
        # durable transition state while a service is still running the empty
        # config.
        self._host_config_loaded = False
        self._channel_adapter: Optional["ProxyChannelAdapter"] = None
        # Channel-link plumbing (#2081): the bridged channel type and the name of
        # its pairing tool. When that tool runs on the streaming turn, the host
        # emits a persisted ``channel_link`` typed part so the pairing card rides
        # the conversation that asked for it (survives refresh) instead of
        # orphaning as a live SSE bubble.
        self._channel_type: Optional[str] = None
        self._link_tool: Optional[str] = None

    @property
    def tool_description(self) -> str:
        if self.runtime.description:
            return self.runtime.description
        return f"Isolated feature service for {self.name}"

    @property
    def config_schema(self) -> Optional[Dict]:
        return self.runtime.config_schema

    async def initialize(self):
        """Initialize a child as an all-or-terminal lifecycle transaction.

        ``_connect_client`` intentionally publishes before event registration
        completes.  A caller cancellation delivered at the following gate
        reset used to return from here with that published child unsupervised.
        Run the initialization body independently, then terminally retire any
        published state before reporting *any* failure to the caller.
        """

        task = asyncio.create_task(
            self._initialize_uninterrupted(),
            name=f"isolated-initialize:{self.name}",
        )
        try:
            await _await_task_until_complete(task, preserve_cancellation=False)
        except BaseException:
            # This also covers a cancellation observed after the inner task
            # successfully reset the gate.  A cancelled initialize is never a
            # successful publication from its caller's perspective.
            await self._quarantine_unreconciled_client()
            raise

    async def _initialize_uninterrupted(self) -> None:
        """Build and publish a fresh child while holding lifecycle ownership."""

        async with self._reload_lock:
            # A completed shutdown/quarantine transaction belongs to the old
            # enable cycle.  A later explicit initialize gets a fresh terminal
            # transaction if this new cycle subsequently fails.
            if self._terminal_cleanup_task is not None and self._terminal_cleanup_task.done():
                self._terminal_cleanup_task = None
            # Reset lifecycle state to the fresh-start baseline BEFORE the client or
            # supervisor start. ``shutdown()`` latches ``_stopping=True`` to unwind
            # the health supervisor; runtime re-enable re-runs this SAME instance's
            # ``initialize()`` (``_activate_feature_runtime``), so without the reset
            # the new ``_supervise()`` task sees a stale ``_stopping`` and exits on
            # its first ``while not self._stopping`` check — leaving a re-enabled
            # service with no health supervisor (kestrel-sovereign#2522 P2).
            self._stopping = False
            # A previous enable cycle may have left an intentional empty config (or
            # a stopped client) on this same object. A fresh initialize must never
            # let that in-memory state stand in for the durable read below.
            self._host_config = {}
            self._host_config_loaded = False
            self._venv_path, self._bin_path = self.resolve_runtime_paths()
            if self._bin_path is None:
                self.ensure_venv()
            # Resolve persisted/UI host config BEFORE building the client so it can be
            # forwarded to the isolated service through the initialize handshake (the
            # service is otherwise launched bare, with only env vars).
            await self._ensure_host_config_loaded()
            await self._connect_client()
            # A previously quarantined instance is only made reachable after its
            # fresh child was initialized from durable config.
            await self._reset_traffic_gate_after_initialize()
            self._supervision_task = self._start_supervision()

    async def _run_traffic_gate_operation(
        self,
        operation: Awaitable[None],
        *,
        name: str,
        preserve_cancellation: bool = False,
    ) -> None:
        """Complete a gate mutation before its lifecycle owner can proceed.

        A cancellation can arrive while a close/drain waits for an admitted
        tool.  Running the boundary in a shielded task ensures that cancellation
        is reported only after the gate has reached a coherent state, so no task
        remains after ``_reload_lock`` with authority to mutate admission.
        """

        task = asyncio.create_task(operation, name=f"isolated-traffic-{name}:{self.name}")
        await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    async def _close_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.close_and_drain(),
            name="close",
        )

    async def _reopen_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.reopen(),
            name="reopen",
        )

    async def _seal_traffic_gate(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.seal_and_drain(),
            name="seal",
        )

    async def _reset_traffic_gate_after_initialize(self) -> None:
        await self._run_traffic_gate_operation(
            self._traffic_gate.reset_and_reopen(),
            name="initialize",
        )

    async def _connect_client(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        register_channel_bridge: bool = True,
    ) -> None:
        """Build + start the isolated client from the current ``_host_config``,
        then wire event handling, tools, and the channel bridge.

        Shared by ``initialize`` (first launch) and ``reload`` (re-launch after a
        config change). ``_build_client`` snapshots ``_host_config`` into the
        service's initialize handshake, so rebuilding here is how new config
        actually reaches a running service.
        """
        client, tools = await self._start_detached_client(config)
        self._publish_client(
            client,
            tools,
            register_channel_bridge=register_channel_bridge,
        )
        try:
            await self._register_event_handler(client)
        except BaseException:
            # A client whose event registration failed must not remain
            # reachable through host tools while its caller unwinds.
            self._unpublish_client(client)
            await self._retire_detached_client(client)
            raise

    async def _refresh_published_client_inventory(self) -> None:
        """Republish tools and channel capability after a live apply.

        A negotiated ``applied`` hook keeps the same process, but its config can
        change which tools it advertises and which channel adapter should be
        registered.  The caller holds both the reload lock and the closed
        traffic gate, so the replacement inventory becomes visible as one host
        state before any new call or event is admitted.
        """

        client = self._client
        if client is None:
            raise RuntimeError("isolated feature client is unavailable after live config apply")
        advertised_tools = await _maybe_await(client.list_tools())
        refreshed_tools = [IsolatedFeatureTool(self, meta) for meta in advertised_tools]
        if self._client is not client:
            raise RuntimeError("isolated feature client changed during inventory refresh")
        self._unregister_channel_bridge()
        self._tools = refreshed_tools
        self._register_channel_bridge()

    async def _start_detached_client(
        self, config: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, List[AgentTool]]:
        """Start a child without making it reachable through this proxy.

        Keep startup and host publication separate so a newly started child is
        never reachable through tools, channel bridges, or event handlers until
        its caller has established the authoritative lifecycle state.
        """

        child_config = self._host_config if config is None else config
        client = self._build_client(config=child_config)
        try:
            await _maybe_await(client.start())
            advertised_tools = await _maybe_await(client.list_tools())
        except BaseException:
            await self._retire_detached_client(client)
            raise
        return client, [IsolatedFeatureTool(self, meta) for meta in advertised_tools]

    def _publish_client(
        self,
        client: Any,
        tools: List[AgentTool],
        *,
        register_channel_bridge: bool,
    ) -> None:
        """Atomically make a started child available to host traffic.

        Callers hold ``_reload_lock`` whenever replacing a live child.  The
        paired assignments deliberately happen before any event registration:
        once an event can enter the host, the child and its advertised tools are
        already the single live proxy state.
        """

        self._client = client
        self._tools = tools
        if register_channel_bridge:
            self._register_channel_bridge()

    def _unpublish_client(self, expected_client: Any = None) -> Any:
        """Remove the current child from host-visible proxy state.

        ``expected_client`` prevents an error path for an old detached child
        from removing a newer restored child.
        """

        if expected_client is not None and self._client is not expected_client:
            return None
        self._unregister_channel_bridge()
        client = self._client
        self._client = None
        self._tools = []
        return client

    async def _retire_detached_client(self, client: Any) -> None:
        """Stop a child which was never published to the host."""

        try:
            await _maybe_await(client.stop())
        except BaseException:
            logger.error(
                "Isolated feature %s could not stop its detached client",
                self.name,
            )
            raise

    async def reload(self) -> None:
        """Restart the isolated service so the current ``_host_config`` takes
        effect (config is forwarded only at the initialize handshake, so a live
        config change requires a re-launch). Guarded so the health supervisor
        doesn't treat the intentional stop as a crash and double-restart."""
        async with self._reload_lock:
            self._begin_reload()
            # Set this before the await: a cancelled drain still closed the
            # gate and must take the matching final boundary below.
            gate_closed = True
            replacement_started = False
            try:
                await self._close_traffic_gate()
                replacement_started = True
                await self._replace_client()
            except BaseException:
                # ``_replace_client`` retires the old child before starting a
                # candidate.  If the candidate fails or this reload is
                # cancelled, reopening to the old (now stopped) child would
                # make its stale tools callable.  Terminal quarantine is the
                # only honest outcome until a later explicit initialize builds
                # a coherent child from durable config.
                # Cancellation while the gate is merely draining has not
                # touched the old child, so it can still safely reopen.  Once
                # replacement begins, however, publication is removed before
                # the first stop await and only terminal quarantine is honest.
                if replacement_started:
                    await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                raise
            finally:
                self._end_reload()
                if gate_closed:
                    if self._stopping or self._client is None:
                        await self._seal_traffic_gate()
                    else:
                        await self._reopen_traffic_gate()

    def _begin_reload(self) -> None:
        """Fence health supervision while this proxy owns client lifecycle."""

        self._reloading = True
        self._reload_gen += 1

    def _end_reload(self) -> None:
        self._reloading = False

    async def _replace_client(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        register_channel_bridge: bool = True,
    ) -> None:
        """Stop the current child and start one using ``_host_config``.

        Callers must already hold ``_reload_lock`` and have established whether
        an advertised config-transition lifecycle hook must run first.

        ``config`` selects an explicit effective config for a normal
        replacement. Fenced recovery calls this only after its pending
        generation has been durably promoted (or after it has read another
        authoritative active generation).
        """

        # The old child must lose every host-visible handle before a stop that
        # can succeed while candidate startup fails.  Restoring it only on a
        # stop failure gives terminal quarantine one last best-effort retirement
        # handle; it never restores its tools or channel bridge.
        previous_client = self._unpublish_client()
        if previous_client is not None:
            try:
                await _maybe_await(previous_client.stop())
            except BaseException:
                self._client = previous_client
                raise
        await self._connect_client(
            config,
            register_channel_bridge=register_channel_bridge,
        )

    async def shutdown(self):
        # Set the latch before scheduling the transaction so a health probe
        # cannot decide to restart the child in the tiny interval before seal.
        self._stopping = True
        await self._complete_terminal_cleanup()

    async def _complete_terminal_cleanup(
        self,
        *,
        best_effort: bool = False,
        lifecycle_lock_held: bool = False,
    ) -> None:
        """Finish terminal teardown before propagating caller cancellation.

        The task remains owned by this proxy while it waits on ``_reload_lock``.
        Shielded waiting is intentional: a cancellation is returned only after
        the gate is sealed, supervision is settled, and publication/retirement
        have reached their final state.
        """

        self._stopping = True
        if lifecycle_lock_held:
            # A set-config/reload owner cannot await the shared terminal task:
            # that task may already be waiting behind this very lock (for
            # example when shutdown sealed while a reload was in progress).
            # Complete this terminal portion under the existing ownership in a
            # separate shielded task instead.  It does not acquire the lock, so
            # neither the cleanup nor the original shutdown can be orphaned in
            # a lock cycle.
            task = asyncio.create_task(
                self._terminal_cleanup_uninterrupted(
                    best_effort=best_effort,
                    lifecycle_lock_held=True,
                ),
                name=f"isolated-terminal-cleanup-owned:{self.name}",
            )
            await _await_task_until_complete(task, preserve_cancellation=False)
            return

        task = self._terminal_cleanup_task
        if task is None or task.done():
            task = asyncio.create_task(
                self._terminal_cleanup_uninterrupted(
                    best_effort=best_effort,
                    lifecycle_lock_held=False,
                ),
                name=f"isolated-terminal-cleanup:{self.name}",
            )
            self._terminal_cleanup_task = task
        await _await_task_until_complete(task, preserve_cancellation=False)

    async def _terminal_cleanup_uninterrupted(
        self,
        *,
        best_effort: bool,
        lifecycle_lock_held: bool,
    ) -> None:
        """Seal, serialize, unpublish, and retire without caller interruption."""

        self._stopping = True
        # Seal before lifecycle ownership so finite-transition waiters become
        # terminal even while another reload currently holds the lock.
        await self._seal_traffic_gate()
        async def unpublish_and_retire() -> None:
            supervision_task = self._supervision_task
            if supervision_task is not None:
                self._supervision_task = None
                if supervision_task is not asyncio.current_task():
                    supervision_task.cancel()
                    try:
                        await supervision_task
                    except asyncio.CancelledError:
                        pass

            client = self._unpublish_client()
            if client is None:
                return
            try:
                await _maybe_await(client.stop())
            except BaseException:
                if not best_effort:
                    raise
                logger.error(
                    "Isolated feature %s could not stop its unreconciled client; "
                    "the proxy has been quarantined",
                    self.name,
                )

        if lifecycle_lock_held:
            await unpublish_and_retire()
            return
        async with self._reload_lock:
            await unpublish_and_retire()

    def get_tools(self) -> List[AgentTool]:
        return list(self._tools)

    def get_router(self):
        if self._client is not None and hasattr(self._client, "get_router"):
            return self._client.get_router()
        return None

    def get_ui_contributions(self) -> Optional[UIContributions]:
        """Forward UI contributions an isolated service reports over the SDK
        init handshake (design option (a) of ticket #2043).

        The out-of-process service advertises its UI assets in its
        ``initialize`` capabilities under ``ui_contributions`` — modules/css
        plus an absolute ``static_dir`` that lives on the same host. The host
        then mounts and serves those assets through the same single asset path
        as in-process features, so isolated-venv features can contribute UI
        without the host proxying every static request.
        """
        caps = self._client_capabilities()
        ui = caps.get("ui_contributions") or caps.get("ui")
        if not isinstance(ui, dict):
            return None
        modules = ui.get("modules")
        if not isinstance(modules, list) or not modules:
            return None
        css = ui.get("css")
        return UIContributions(
            modules=[str(m) for m in modules],
            css=[str(c) for c in css] if isinstance(css, list) else [],
            static_dir=ui.get("static_dir"),
            capability=ui.get("capability"),
        )

    async def get_config(self) -> Dict:
        """Return the feature's current host config.

        The SDK client exposes no ``get_config`` (config only flows host→service
        at initialize), so read from the in-memory host config, falling back to
        the persisted ``feature_config:<name>`` node — NOT an empty passthrough,
        which made the config API/UI show blank and drop write-only secrets on a
        partial PATCH (#2214).
        """
        # Hosted replicas must not answer from their process-local cache: a
        # second replica can commit a credential rotation after this proxy
        # initialized.  In particular, cached ``{}`` is a valid loaded config,
        # not evidence that no durable config exists.  A pending transition
        # deliberately exposes its *active* config here; the candidate remains
        # private until promotion.
        storage = getattr(self.agent, "storage", None)
        if storage is not None and await self._persistent_config_writes_allowed(storage):
            state = await self._read_config_state(storage)
            # Do not overwrite the local candidate while its traffic gate is
            # closed during an in-process applied hook.  The return value is
            # nevertheless authoritative for this read.
            if not self._reloading:
                self._host_config = dict(state.config)
                self._host_config_loaded = True
            return dict(state.config)

        # Volatile privacy mode intentionally has no durable node.  Its local
        # empty config is still distinct from an unloaded state.
        await self._ensure_host_config_loaded()
        return dict(self._host_config)

    async def set_config(
        self,
        config: Dict,
        *,
        _preserve_secret_fields: set[str] | None = None,
        _validate_effective_config: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        """Persist an effective config and apply it to the running service.

        The previous implementation forwarded to ``self._client.set_config`` —
        which the SDK client does not implement — so config set via the API/UI
        was silently dropped: never persisted (so lost on restart) and never
        applied (#2214). Persist to the ``feature_config:<name>`` graph node (the
        same node ``_load_host_config`` reads at startup). A service that
        advertises the SDK config-transition capability receives the full next
        effective config while it still owns the old one. Its successful typed
        result decides whether to replace the process or retain it after a
        live apply. Legacy services retain the existing safe replacement path.

        The candidate is durably staged as ``pending_config`` while the active
        ``config`` remains unchanged.  This is deliberately not implemented
        through :meth:`Feature.persist_config`: that compatibility helper
        swallows storage failures, whereas an apply may only continue after a
        durable write succeeds (apart from an intentional volatile-privacy
        no-op). A failed hook conditionally removes its own pending generation;
        if storage cannot prove that cleanup, the local proxy is quarantined
        rather than left reachable on an uncertain config.

        Each durable update is a generation-owned ``stage → promote`` protocol:
        a conditional graph write stages the candidate, and a second conditional
        write promotes only that exact pending generation.  A write exception is
        not evidence of rollback — cloud storage can commit before a connection
        breaks or a task is cancelled — so every uncertain promotion is re-read
        and the child is reconciled to that authoritative state.  In particular,
        fenced SDK recovery promotes before it starts a replacement child: child
        startup may create external resources and must never run ahead of the
        durable active configuration.
        """
        cfg = dict(config) if isinstance(config, dict) else {}
        async with self._reload_lock:
            self._begin_reload()
            # This intent marker deliberately precedes the await below. A
            # cancelled close/drain has already made the gate finite-closed,
            # so its finally must perform a cancellation-safe reopen or seal.
            gate_closed = True
            self._fenced_recovery_failed = False
            transition_attempted = False
            transition_succeeded = False
            transition: Optional[_ConfigTransition] = None
            promotion: Optional[_PromotionResolution] = None
            transition_settled = False
            lifecycle_result: Optional[ConfigTransitionResult] = None
            local_authoritative = False
            try:
                # Admission must close before the candidate is staged, not just
                # before a replacement.  A successful in-process hook may have
                # adopted its candidate by the time it returns, so tools,
                # channel sends, and inbound callbacks must all be drained
                # before the hook begins.
                await self._close_traffic_gate()
                # A caller may invoke set_config after a failed startup or
                # before normal initialization. Reload the authoritative
                # durable value first; otherwise a partial PATCH could stage
                # an empty config over a write-only secret.
                await self._ensure_host_config_loaded()
                # Stage from a fresh graph snapshot, never the in-memory
                # cache. A Cloud Run replica that read an older config cannot
                # overwrite a newer generation through the old add-node
                # upsert path.
                transition = await self._stage_pending_config(
                    cfg,
                    preserve_secret_fields=_preserve_secret_fields,
                    validate_effective_config=_validate_effective_config,
                )
                await self._reconcile_client_to_authoritative_config(
                    transition.active_config,
                    force=False,
                )
                if self._client is None:
                    promotion = await self._promote_config(transition)
                    if not promotion.committed:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=False,
                            preserve_cancellation=False,
                        )
                        transition_settled = True
                        self._raise_promotion_failure(promotion)
                    transition_settled = True
                    self._host_config = dict(transition.next_config)
                    self._host_config_loaded = True
                    local_authoritative = True
                    if promotion.error is not None:
                        self._raise_promotion_failure(promotion)
                    return

                if self._supports_config_transition():
                    transition_attempted = True
                    lifecycle_result = await self._prepare_config_transition_with_lease(
                        transition
                    )
                    transition_succeeded = True

                # A legacy service has no preparation phase; a supported
                # service reaches this point only after its preparation
                # completed.  Either way, make the next config durable before
                # exposing it through the host or launching a normal child.
                promotion = await self._promote_config(transition)
                if not promotion.committed:
                    # A negotiated hook may already have mutated the old
                    # child. Replacing it from the active config after the
                    # generation-owned abort is the only safe response.
                    await self._run_owned_transition_cleanup(
                        transition,
                        force=lifecycle_result is not None,
                        preserve_cancellation=False,
                    )
                    transition_settled = True
                    self._raise_promotion_failure(promotion)

                transition_settled = True
                self._host_config = dict(transition.next_config)
                self._host_config_loaded = True

                if lifecycle_result is None:
                    # Legacy SDK/service: no negotiated hook, so preserve the
                    # established stop-and-replace behavior.
                    await self._replace_client()
                elif lifecycle_result.action == CONFIG_TRANSITION_APPLIED:
                    # The service atomically adopted the config in-process.
                    # Its channel bridge still carries host-side config (enabled
                    # and sender filters), so refresh that forwarding adapter.
                    await self._refresh_published_client_inventory()
                else:
                    # The SDK validates result actions; the non-live outcome is
                    # the normal prepare-then-restart protocol.
                    await self._replace_client()
                local_authoritative = True
                # A connection failure/cancellation may have been raised after
                # the promote committed. The child is now coherent with the
                # durable state, but the original caller still receives its
                # transport outcome rather than a false success.
                if promotion.error is not None:
                    self._raise_promotion_failure(promotion)
            except asyncio.CancelledError:
                if transition is not None:
                    # Every await after staging enters this path. The cleanup
                    # task is shielded so a second cancellation cannot strand
                    # this generation; it either proves durable state and
                    # reconciles the child or quarantines the proxy itself.
                    if self._client_requires_replacement():
                        # The SDK fences a cancelled lifecycle RPC precisely
                        # because it may have reached the child.  Do not revert
                        # to the old config first: promote this generation (or
                        # prove another durable winner) before any child is
                        # started, then preserve the caller's cancellation.
                        await self._recover_fenced_transition(
                            transition,
                            asyncio.CancelledError(),
                            preserve_cancellation=True,
                        )
                    else:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=(not local_authoritative)
                            and (
                                transition_attempted
                                or lifecycle_result is not None
                                or transition_settled
                            ),
                            preserve_cancellation=True,
                        )
                raise
            except BaseException as transition_error:
                if transition is not None and not transition_settled:
                    if transition_attempted and not transition_succeeded:
                        if (
                            self._client_requires_replacement()
                            or isinstance(transition_error, _ConfigTransitionLeaseLost)
                        ):
                            await self._recover_fenced_transition(
                                transition,
                                transition_error,
                            )
                        else:
                            await self._run_owned_transition_cleanup(
                                transition,
                                force=False,
                                preserve_cancellation=False,
                            )
                    else:
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=lifecycle_result is not None,
                            preserve_cancellation=False,
                        )
                elif (
                    transition is not None
                    and transition_settled
                    and promotion is not None
                    and promotion.committed
                    and not local_authoritative
                ):
                    # An await after the durable promotion (for example child
                    # replacement) failed.  Re-read/reconcile the active
                    # generation before surfacing it; no pending stage remains.
                    await self._run_owned_transition_cleanup(
                        transition,
                        force=True,
                        preserve_cancellation=False,
                    )
                raise
            finally:
                try:
                    if self._fenced_recovery_failed:
                        await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
                finally:
                    self._end_reload()
                    # A quarantined proxy remains fail-closed. Every other path
                    # reaches this point only after promotion or owned cleanup
                    # has reconciled the active child with durable state. These
                    # operations are themselves shielded to a final condition
                    # state before this reload releases its lock.
                    if gate_closed:
                        if self._stopping:
                            await self._seal_traffic_gate()
                        else:
                            await self._reopen_traffic_gate()

    async def set_config_with_secret_preservation(
        self,
        incoming: Dict[str, Any],
        secret_fields: set[str],
        validate: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Atomically preserve omitted write-only fields for the generic API.

        The endpoint must not read a secret on one hosted replica and later
        stage that stale value over another replica's rotation.  Preservation
        is therefore performed from the very same authoritative snapshot that
        becomes the stage CAS predicate; validation also runs on that effective
        value before the lifecycle hook can observe it.
        """

        await self.set_config(
            incoming,
            _preserve_secret_fields=set(secret_fields),
            _validate_effective_config=validate,
        )

    async def _run_owned_transition_cleanup(
        self,
        transition: _ConfigTransition,
        *,
        force: bool,
        preserve_cancellation: bool,
    ) -> None:
        """Run owned abort/reconciliation while protecting its durable outcome.

        The cleanup task either proves durable state and reconciles the child or
        quarantines the proxy.  Its caller chooses whether a new cancellation
        should immediately propagate (ordinary exception unwinding) or defer to
        an already-caught original ``CancelledError``.
        """

        async def cleanup() -> None:
            try:
                await self._abort_and_reconcile_uncommitted_transition(
                    transition,
                    force=force,
                )
            except BaseException:  # noqa: BLE001 - cancellation cleanup fences the proxy
                self._host_config_loaded = False
                await self._quarantine_unreconciled_client(lifecycle_lock_held=True)

        task = asyncio.create_task(cleanup())
        # Shielding alone is insufficient: a *second* cancellation used to let
        # this method return while ``task`` still held a durable cleanup or
        # quarantine operation, releasing ``_reload_lock`` to the next reload.
        # Keep waiting through every cancellation; the caller either re-raises
        # its original cancellation after cleanup or receives the newly caught
        # one only once the state can no longer mutate in the background.
        await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    async def _recover_fenced_transition(
        self,
        transition: _ConfigTransition,
        _original_error: BaseException,
        *,
        preserve_cancellation: bool = False,
    ) -> None:
        """Finish fenced recovery before releasing lifecycle ownership.

        The caller may already be unwinding a cancellation.  In that case a
        later cancellation or recovery error must not let a cleanup task escape
        behind ``_reload_lock``; quarantine is complete first and the original
        cancellation remains the public result.
        """

        # Treat recovery as unsafe until its task has completed normally.  This
        # also gives the enclosing finally a fail-closed marker if an exception
        # is raised at any await boundary below.
        self._fenced_recovery_failed = True
        task = asyncio.create_task(
            self._recover_fenced_transition_uninterrupted(transition),
            name=f"isolated-fenced-recovery:{self.name}",
        )
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
                continue
        try:
            await task
        except BaseException as recovery_error:
            self._fenced_recovery_failed = True
            # The uninterrupted body has already tried generation-scoped
            # cleanup.  Repeat through the standard fenced cleanup path so a
            # generic failure (including old-client stop failure) cannot leave
            # our pending generation wedged.  That helper never clears another
            # owner's state and quarantines on uncertainty.
            await self._run_owned_transition_cleanup(
                transition,
                force=True,
                preserve_cancellation=True,
            )
            await self._clear_owned_pending_before_quarantine(transition)
            # A recovery operation itself failed (for example an old child
            # refused to stop or the replacement could not start).  Even if
            # durable cleanup succeeded, do not leave a client whose lifecycle
            # outcome is unknown reachable to a later supervisor restart.
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            # ``_quarantine_unreconciled_client`` is deliberately best-effort
            # about process retirement; publication itself is not best-effort.
            # Keep the host boundary closed even if a non-SDK test/client
            # implementation mutates its stop path unexpectedly.
            self._client = None
            self._tools = []
            self._stopping = True
            if preserve_cancellation:
                logger.error(
                    "Isolated feature %s fenced recovery failed; proxy was reconciled "
                    "or quarantined before cancellation propagated",
                    self.name,
                )
                return
            raise recovery_error
        if cancellation is not None and not preserve_cancellation:
            raise cancellation
        self._fenced_recovery_failed = False

    async def _recover_fenced_transition_uninterrupted(
        self,
        transition: _ConfigTransition,
    ) -> None:
        """Recover an SDK-fenced hook outcome without exposing pending config.

        An SDK fence means the hook may have reached the child, so first remove
        that child from host traffic.  Promotion remains generation-scoped; if
        it cannot be proved, abort only this staged generation and restart from
        the durable active config.  No child ever starts from a pending value.
        """

        active_client = self._unpublish_client()
        try:
            if active_client is not None:
                try:
                    await _maybe_await(active_client.stop())
                except BaseException:
                    # Put it back only so the standard cleanup can retire or
                    # quarantine it.  It remains unpublished throughout.
                    self._client = active_client
                    raise

            promotion = await self._promote_config(transition)
            if promotion.committed:
                target_config = transition.next_config
            else:
                state = await self._abort_and_reconcile_uncommitted_transition(
                    transition,
                    force=False,
                )
                # ``state.config`` is always the active value; pending_config
                # is never used to initialize a recovery child.
                target_config = state.config

            await self._connect_client(target_config)
        except BaseException:
            # A partially restored old client must be visible only to the
            # cleanup/quarantine path, never to traffic (the gate is closed).
            if active_client is not None and self._client is None:
                self._client = active_client
            # In particular, a failed old-client stop must not prevent the
            # generation's durable cleanup.  This happens before quarantine so
            # a later replica does not inherit a needless pending wedge.
            await self._clear_owned_pending_before_quarantine(transition)
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            raise

        self._host_config = dict(target_config)
        self._host_config_loaded = True

    async def _clear_owned_pending_before_quarantine(
        self,
        transition: _ConfigTransition,
    ) -> None:
        """Make one last client-independent attempt to retire our stage.

        Recovery can fail while stopping the old child.  The normal cleanup
        path also reconciles that child and can therefore abort before its
        post-cleanup result is observable.  Durable pending removal must not
        depend on a process stop: this helper uses only generation-scoped CAS
        and never touches a state owned by another replica.  Any uncertainty is
        left quarantined rather than guessed away.
        """

        if not transition.persistent:
            return
        try:
            state = await self._read_config_state(transition.storage)
            if not self._state_matches_pending_generation(
                state,
                generation=transition.generation,
                owner=transition.owner,
            ):
                return
            cleanup = await self._clear_pending_generation(
                transition.storage,
                state,
                generation=transition.generation,
                owner=transition.owner,
            )
            if cleanup.cleared:
                self._host_config = dict(cleanup.state.config)
                self._host_config_loaded = True
        except BaseException:  # noqa: BLE001 - quarantine remains the fallback
            self._host_config_loaded = False

    async def _quarantine_unreconciled_client(
        self,
        *,
        lifecycle_lock_held: bool = False,
    ) -> None:
        """Fail closed when a recovery child cannot be reconciled to storage.

        The SDK subprocess client stops/terminates its child even when graceful
        RPC shutdown fails.  Other client implementations may not provide that
        guarantee, so remove the client and its host-visible tools before
        attempting best-effort retirement.  No config values enter logs or
        exception messages.
        """

        # This shares the terminal transaction with shutdown.  In particular,
        # a cancellation delivered after seal cannot skip unpublication,
        # adapter removal, supervision cancellation, or best-effort retirement
        # while the task is still waiting on another lifecycle owner.
        self._stopping = True
        await self._complete_terminal_cleanup(
            best_effort=True,
            lifecycle_lock_held=lifecycle_lock_held,
        )

    async def _stage_pending_config(
        self,
        pending_config: Dict[str, Any],
        *,
        preserve_secret_fields: set[str] | None = None,
        validate_effective_config: Callable[[Dict[str, Any]], None] | None = None,
    ) -> _ConfigTransition:
        """CAS-stage one candidate from a fresh authoritative graph snapshot.

        ``add_node`` is an upsert and is therefore unsafe on hosted replicas.
        Persistent transitions require ``compare_and_swap_node``; a storage
        surface without that atomic contract fails before any lifecycle hook is
        invoked.  Volatile privacy mode remains the intentional non-durable
        path above.
        """

        storage = getattr(self.agent, "storage", None)
        if storage is None:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: storage is unavailable"
            )

        persistent = await self._persistent_config_writes_allowed(storage)
        if not persistent:
            # Privacy modes intentionally have no durable state to contend
            # over. Keeping the transition in-memory is valid only because the
            # policy forbids a durable config node in the first place.
            effective_pending_config = dict(pending_config)
            for key in preserve_secret_fields or set():
                if key not in effective_pending_config and key in self._host_config:
                    effective_pending_config[key] = self._host_config[key]
            if validate_effective_config is not None:
                validate_effective_config(dict(effective_pending_config))
            return _ConfigTransition(
                active_config=dict(self._host_config),
                next_config=effective_pending_config,
                persistent=False,
                storage=storage,
                expected_properties=None,
                staged_properties=None,
                promoted_properties=None,
            )

        transition: Optional[_ConfigTransition] = None
        try:
            while True:
                state = await self._read_config_state(storage)
                if state.has_pending:
                    if not self._pending_lease_is_expired(state):
                        await self._reconcile_client_to_authoritative_config(
                            state.config,
                            force=False,
                        )
                        raise RuntimeError(
                            f"Cannot apply config for isolated feature {self.name}: "
                            "config transition is already in progress"
                        )

                    # An expired lease is abandoned work, not a candidate we
                    # may promote.  First CAS-remove exactly that generation,
                    # retain its active config, then start over from a fresh
                    # snapshot.  No child sees the abandoned pending config.
                    cleanup = await self._clear_pending_generation(
                        storage,
                        state,
                        generation=state.pending_generation,
                        owner=state.pending_owner,
                    )
                    if cleanup.cleared:
                        continue
                    if self._state_matches_pending_generation(
                        cleanup.state,
                        generation=state.pending_generation,
                        owner=state.pending_owner,
                    ):
                        raise RuntimeError(
                            f"Cannot apply config for isolated feature {self.name}: "
                            "could not clear an expired config transition"
                        )
                    # A concurrent replica changed the node. Re-read before
                    # deciding whether its state is active, pending, or ours.
                    continue

                generation = uuid4().hex
                owner = self._config_transition_owner
                lease_expires_at = _utc_now() + _PENDING_CONFIG_LEASE_TTL
                # Reconstitute omitted write-only fields immediately before the
                # stage CAS from *this* snapshot.  A concurrent credential
                # rotation can only make our predicate fail; the next loop
                # observes its winner and preserves that value instead.
                effective_pending_config = dict(pending_config)
                for key in preserve_secret_fields or set():
                    if key not in effective_pending_config and key in state.config:
                        effective_pending_config[key] = state.config[key]
                if validate_effective_config is not None:
                    validate_effective_config(dict(effective_pending_config))
                staged_properties = dict(state.properties or {})
                staged_properties["config"] = dict(state.config)
                staged_properties["pending_config"] = dict(effective_pending_config)
                staged_properties[_PENDING_GENERATION_KEY] = generation
                staged_properties[_PENDING_OWNER_KEY] = owner
                staged_properties[_PENDING_LEASE_EXPIRES_AT_KEY] = (
                    lease_expires_at.isoformat()
                )

                promoted_properties = self._promoted_properties_from_staged(
                    staged_properties,
                    config=effective_pending_config,
                    generation=generation,
                )

                transition = _ConfigTransition(
                    active_config=dict(state.config),
                    next_config=dict(effective_pending_config),
                    persistent=True,
                    storage=storage,
                    expected_properties=(
                        dict(state.properties) if state.properties is not None else None
                    ),
                    staged_properties=staged_properties,
                    promoted_properties=promoted_properties,
                    generation=generation,
                    owner=owner,
                )
                write = await self._write_config_state(
                    storage,
                    transition.expected_properties,
                    staged_properties,
                )
                if write.committed:
                    return transition

                # A connection can fail after the conditional write committed.
                # Read before deciding this is a failed stage or a concurrent
                # winner. Cancellation follows the same owned-abort path as
                # every later await boundary.
                observed = await self._read_config_state(storage)
                if observed.properties == staged_properties:
                    if isinstance(write.error, asyncio.CancelledError):
                        await self._run_owned_transition_cleanup(
                            transition,
                            force=False,
                            preserve_cancellation=True,
                        )
                        raise write.error
                    return transition

                if write.error is None and preserve_secret_fields:
                    # An atomic PATCH preservation attempt deliberately retries
                    # from the newer durable winner.  This is the only path
                    # where a stale pre-stage read may be repaired without
                    # surfacing a conflict: each loop re-merges omitted
                    # write-only fields from the exact CAS predicate snapshot,
                    # so it can never reintroduce the stale secret that lost.
                    continue

                await self._reconcile_client_to_authoritative_config(
                    observed.config,
                    force=False,
                )
                if write.error is not None:
                    self._raise_storage_write_error(write.error)
                raise RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "config transition conflicts with a newer durable state"
                )
        except asyncio.CancelledError:
            if transition is not None:
                await self._run_owned_transition_cleanup(
                    transition,
                    force=False,
                    preserve_cancellation=True,
                )
            raise

    def _pending_lease_is_expired(self, state: _ConfigState) -> bool:
        """Return whether a validated pending lease is safely reclaimable."""

        expires_at = state.pending_lease_expires_at
        if expires_at is None:
            # ``_read_config_state`` rejects malformed pending metadata. Keep
            # this guard fail-closed if a future caller constructs a state.
            return False
        return expires_at <= (_utc_now() - _PENDING_CONFIG_CLOCK_SKEW)

    async def _renew_transition_lease(self, transition: _ConfigTransition) -> None:
        """Extend exactly this staged generation's lease with a CAS write.

        The lifecycle hook may legitimately take longer than the original
        takeover interval.  Renewal keeps a healthy owner unstealable without
        relaxing recovery of an actually abandoned stage.  A predicate failure
        is a hard fence: another durable outcome won and this proxy must stop
        treating its child as authoritative.
        """

        if not transition.persistent:
            return
        staged = transition.staged_properties
        generation = transition.generation
        owner = transition.owner
        if (
            not isinstance(staged, dict)
            or not isinstance(generation, str)
            or not generation
            or not isinstance(owner, str)
            or not owner
        ):
            raise _ConfigTransitionLeaseLost("isolated config transition lease is invalid")

        current_state = await self._read_config_state(transition.storage)
        if not self._state_matches_pending_generation(
            current_state,
            generation=generation,
            owner=owner,
        ) or current_state.properties != staged:
            raise _ConfigTransitionLeaseLost("isolated config transition lease was lost")

        renewed = dict(staged)
        renewed[_PENDING_LEASE_EXPIRES_AT_KEY] = (
            _utc_now() + _PENDING_CONFIG_LEASE_TTL
        ).isoformat()
        write = await self._write_config_state(transition.storage, staged, renewed)
        if write.committed:
            transition.staged_properties = renewed
            transition.promoted_properties = self._promoted_properties_from_staged(
                renewed,
                config=transition.next_config,
                generation=generation,
            )
            return

        observed = await self._read_config_state(transition.storage)
        if observed.properties == renewed:
            # The CAS committed before its caller lost the response.  Preserve
            # the refreshed predicate for a later promotion.
            transition.staged_properties = renewed
            transition.promoted_properties = self._promoted_properties_from_staged(
                renewed,
                config=transition.next_config,
                generation=generation,
            )
            return
        raise _ConfigTransitionLeaseLost("isolated config transition lease was lost")

    @staticmethod
    def _lease_heartbeat_interval() -> float:
        """Renew well before expiry while keeping a minimum testable cadence."""

        return max(0.01, _PENDING_CONFIG_LEASE_TTL.total_seconds() / 3)

    async def _run_transition_lease_heartbeat(
        self,
        transition: _ConfigTransition,
        stop: asyncio.Event,
    ) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self._lease_heartbeat_interval()
                )
                return
            except asyncio.TimeoutError:
                await self._renew_transition_lease(transition)

    async def _await_task_completion(
        self,
        task: asyncio.Task[Any],
        *,
        preserve_cancellation: bool,
    ) -> Any:
        """Wait for a shielded task without leaving it behind on cancellation."""

        return await _await_task_until_complete(
            task,
            preserve_cancellation=preserve_cancellation,
        )

    @staticmethod
    def _state_matches_pending_generation(
        state: _ConfigState,
        *,
        generation: Optional[str],
        owner: Optional[str],
    ) -> bool:
        """Whether ``state`` still names one exact pending owner/generation."""

        return (
            state.has_pending
            and generation is not None
            and owner is not None
            and state.pending_generation == generation
            and state.pending_owner == owner
        )

    @staticmethod
    def _without_pending_metadata(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the active config while removing a completed/abandoned stage."""

        cleared = dict(properties)
        cleared.pop("pending_config", None)
        cleared.pop(_PENDING_GENERATION_KEY, None)
        cleared.pop(_PENDING_OWNER_KEY, None)
        cleared.pop(_PENDING_LEASE_EXPIRES_AT_KEY, None)
        return cleared

    @staticmethod
    def _promoted_properties_from_staged(
        staged_properties: Dict[str, Any],
        *,
        config: Dict[str, Any],
        generation: str,
    ) -> Dict[str, Any]:
        """Build the exact promotion state for one current leased stage."""

        promoted = dict(staged_properties)
        promoted["config"] = dict(config)
        promoted[_CONFIG_GENERATION_KEY] = generation
        promoted.pop("pending_config", None)
        promoted.pop(_PENDING_GENERATION_KEY, None)
        promoted.pop(_PENDING_OWNER_KEY, None)
        promoted.pop(_PENDING_LEASE_EXPIRES_AT_KEY, None)
        return promoted

    async def _clear_pending_generation(
        self,
        storage: Any,
        state: _ConfigState,
        *,
        generation: Optional[str],
        owner: Optional[str],
    ) -> _PendingCleanupResolution:
        """CAS-clear exactly one pending generation and reconcile ambiguity.

        The full properties snapshot is the predicate.  Matching owner and
        generation before every write is defense in depth: a delayed cleanup
        can never erase a newer writer's stage.  A write exception is ambiguous
        and is always followed by a durable read; one retry covers a transient
        pre-commit failure without turning cleanup into an unbounded loop.
        """

        if (
            state.properties is None
            or not self._state_matches_pending_generation(
                state,
                generation=generation,
                owner=owner,
            )
        ):
            return _PendingCleanupResolution(state=state, cleared=False)

        expected_properties = dict(state.properties)
        cleared_properties = self._without_pending_metadata(expected_properties)
        observed = state
        for _ in range(_PENDING_CLEANUP_WRITE_ATTEMPTS):
            write = await self._write_config_state(
                storage,
                expected_properties,
                cleared_properties,
            )
            if write.committed:
                return _PendingCleanupResolution(
                    state=_ConfigState(
                        properties=cleared_properties,
                        config=dict(state.config),
                    ),
                    cleared=True,
                )

            observed = await self._read_config_state(storage)
            if observed.properties == cleared_properties:
                return _PendingCleanupResolution(state=observed, cleared=True)
            if not self._state_matches_pending_generation(
                observed,
                generation=generation,
                owner=owner,
            ):
                return _PendingCleanupResolution(state=observed, cleared=False)
            # A retry remains scoped to the exact initial properties. If
            # anything beyond a retry is needed, leave it for the lease/takeover
            # protocol instead of risking a writer that changed the state.
            if observed.properties != expected_properties:
                return _PendingCleanupResolution(state=observed, cleared=False)

        return _PendingCleanupResolution(state=observed, cleared=False)

    async def _abort_and_reconcile_uncommitted_transition(
        self,
        transition: _ConfigTransition,
        *,
        force: bool,
    ) -> _ConfigState:
        """Abort this pending stage or quarantine if its durable outcome is unknown."""

        if not transition.persistent:
            state = _ConfigState(properties=None, config=dict(transition.active_config))
            await self._reconcile_client_to_authoritative_config(state.config, force=force)
            return state

        try:
            state = await self._read_config_state(transition.storage)
            if self._state_matches_pending_generation(
                state,
                generation=transition.generation,
                owner=transition.owner,
            ):
                cleanup = await self._clear_pending_generation(
                    transition.storage,
                    state,
                    generation=transition.generation,
                    owner=transition.owner,
                )
                state = cleanup.state
                if not cleanup.cleared and self._state_matches_pending_generation(
                    state,
                    generation=transition.generation,
                    owner=transition.owner,
                ):
                    raise RuntimeError(
                        f"Cannot apply config for isolated feature {self.name}: "
                        "could not reconcile config transition cleanup"
                    )
            await self._reconcile_client_to_authoritative_config(state.config, force=force)
            return state
        except BaseException:
            self._host_config_loaded = False
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            raise

    async def _promote_config(
        self, transition: _ConfigTransition
    ) -> _PromotionResolution:
        """Promote only ``transition``'s staged generation, then reconcile.

        Every non-successful write result — including ``CancelledError`` and a
        transport exception after commit — is followed by a durable read. The
        caller must use the returned state rather than assuming an exception
        means the old config is still active.
        """

        if not transition.persistent:
            return _PromotionResolution(
                state=_ConfigState(
                    properties=None,
                    config=dict(transition.next_config),
                ),
                committed=True,
            )

        try:
            permitted = await self._persistent_config_writes_allowed(
                transition.storage
            )
        except BaseException as probe_error:
            # A completed SDK hook may already have adopted next_config. Even
            # though no promotion write was attempted, a failed policy probe
            # leaves the staged durable state authoritative, so reconcile it
            # before returning the probe/cancellation outcome.
            observed = await self._read_config_state_after_promotion_failure(
                transition.storage,
                probe_error,
            )
            return _PromotionResolution(
                state=observed,
                committed=False,
                error=probe_error,
                storage_error=True,
            )
        if not permitted:
            observed = await self._read_config_state_after_promotion_failure(
                transition.storage,
                RuntimeError("persistent config writes became unavailable"),
            )
            return _PromotionResolution(
                state=observed,
                committed=False,
                error=RuntimeError(
                    f"Cannot apply config for isolated feature {self.name}: "
                    "persistent config writes became unavailable"
                ),
            )

        write = await self._write_config_state(
            transition.storage,
            transition.staged_properties,
            transition.promoted_properties,
        )
        if write.committed:
            return _PromotionResolution(
                state=_ConfigState(
                    properties=dict(transition.promoted_properties or {}),
                    config=dict(transition.next_config),
                ),
                committed=True,
            )

        observed = await self._read_config_state_after_promotion_failure(
            transition.storage,
            write.error,
        )
        # The generation stamp makes this proof specific to this transition;
        # matching only ``config`` would let a different replica's same-valued
        # write be mistaken for our promotion.
        committed = observed.properties == transition.promoted_properties
        return _PromotionResolution(
            state=observed,
            committed=committed,
            error=write.error,
            storage_error=write.error is not None,
        )

    async def _write_config_state(
        self,
        storage: Any,
        expected_properties: Optional[Dict[str, Any]],
        properties: Optional[Dict[str, Any]],
    ) -> _ConfigWriteResult:
        """Conditionally write one complete transition state without swallowing.

        The graph store's atomic compare-and-swap is the durable protocol.
        There is intentionally no ``add_node`` fallback: an upsert cannot prove
        ownership on hosted replicas and would let a stale reader overwrite a
        newer config.
        """

        from kestrel_sovereign.storage.async_graph_store import GraphNode

        node = GraphNode(
            node_id=self._config_node_id(),
            node_type=self._CONFIG_NODE_TYPE,
            label=f"{self.name} config",
            properties=dict(properties or {}),
        )
        compare_and_swap = getattr(storage, "compare_and_swap_node", None)
        if not callable(compare_and_swap):
            return _ConfigWriteResult(
                committed=False,
                error=RuntimeError(
                    "persistent isolated config transitions require "
                    "compare_and_swap_node"
                ),
            )
        try:
            result = await _maybe_await(
                compare_and_swap(self._config_node_id(), expected_properties, node)
            )
            return _ConfigWriteResult(committed=result == "swapped")
        except BaseException as exc:
            # A write boundary may raise after commit. Its caller performs the
            # authoritative read needed to classify the result.
            return _ConfigWriteResult(committed=False, error=exc)

    async def _persistent_config_writes_allowed(self, storage: Any) -> bool:
        """Return the current privacy-policy permission for config persistence."""

        allows_persistent_writes = getattr(storage, "allows_persistent_writes", None)
        if not callable(allows_persistent_writes):
            return True
        try:
            return bool(await _maybe_await(allows_persistent_writes()))
        except Exception as exc:  # noqa: BLE001 - policy probe is a hard boundary
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "could not determine persistence policy"
            ) from exc

    async def _read_config_state(self, storage: Any) -> _ConfigState:
        """Read one complete config-node snapshot without consulting the cache."""

        get_node = getattr(storage, "get_node", None)
        if not callable(get_node):
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "storage cannot read config transition state"
            )
        try:
            node = await _maybe_await(get_node(self._config_node_id()))
        except Exception as exc:  # noqa: BLE001 - durable read is a hard boundary
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "failed to load config transition state"
            ) from exc
        if node is None:
            return _ConfigState(properties=None, config={})

        raw_properties = getattr(node, "properties", None)
        if not isinstance(raw_properties, dict):
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored config transition state is invalid"
            )
        properties = dict(raw_properties)
        raw_config = properties.get("config")
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except (TypeError, ValueError):
                raw_config = None
        config = dict(raw_config) if isinstance(raw_config, dict) else {}
        pending_keys = (
            "pending_config",
            _PENDING_GENERATION_KEY,
            _PENDING_OWNER_KEY,
            _PENDING_LEASE_EXPIRES_AT_KEY,
        )
        has_pending = any(key in properties for key in pending_keys)
        if not has_pending:
            return _ConfigState(properties=properties, config=config)

        pending_config = properties.get("pending_config")
        generation = properties.get(_PENDING_GENERATION_KEY)
        owner = properties.get(_PENDING_OWNER_KEY)
        raw_expires_at = properties.get(_PENDING_LEASE_EXPIRES_AT_KEY)
        if (
            not isinstance(pending_config, dict)
            or not isinstance(generation, str)
            or not generation
            or not isinstance(owner, str)
            or not owner
            or not isinstance(raw_expires_at, str)
            or not raw_expires_at
        ):
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config transition metadata is invalid"
            )
        try:
            expires_at = datetime.fromisoformat(raw_expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config lease is invalid"
            ) from exc
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "stored pending config lease is invalid"
            )
        return _ConfigState(
            properties=properties,
            config=config,
            has_pending=True,
            pending_generation=generation,
            pending_owner=owner,
            pending_lease_expires_at=expires_at.astimezone(timezone.utc),
        )

    async def _read_config_state_after_promotion_failure(
        self,
        storage: Any,
        write_error: Optional[BaseException],
    ) -> _ConfigState:
        """Read durable state after an ambiguous promotion or quarantine."""

        try:
            return await self._read_config_state(storage)
        except BaseException as read_error:
            # We cannot prove which config won, so no live child may remain.
            self._host_config_loaded = False
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            if isinstance(write_error, asyncio.CancelledError):
                raise write_error
            raise RuntimeError(
                f"Cannot apply config for isolated feature {self.name}: "
                "could not reconcile durable config after a promotion failure"
            ) from read_error

    async def _reconcile_client_to_authoritative_config(
        self,
        config: Dict[str, Any],
        *,
        force: bool,
    ) -> None:
        """Make the local child match a config freshly read from storage."""

        authoritative_config = dict(config)
        try:
            # Forced reconciliation owns recovery after a lifecycle operation
            # whose outcome made the current publication unsafe.  In
            # particular, ``_replace_client`` removes the old child before it
            # starts a candidate; if that candidate fails after durable
            # promotion, there is intentionally no client here.  Rebuild from
            # the freshly read *active* config before the finite traffic gate
            # can reopen.  A failed rebuild still follows the quarantine path
            # below while the caller retains the lifecycle lock.
            if force or (
                self._client is not None and self._host_config != authoritative_config
            ):
                await self._replace_client(authoritative_config)
        except BaseException:
            self._host_config = authoritative_config
            self._host_config_loaded = True
            await self._quarantine_unreconciled_client(lifecycle_lock_held=True)
            raise
        self._host_config = authoritative_config
        self._host_config_loaded = True

    def _raise_storage_write_error(self, error: BaseException) -> None:
        """Surface storage failure without leaking feature config or secrets."""

        if isinstance(error, asyncio.CancelledError):
            raise error
        raise RuntimeError(
            f"Cannot apply config for isolated feature {self.name}: "
            "failed to persist config"
        ) from error

    def _raise_promotion_failure(self, promotion: _PromotionResolution) -> None:
        """Raise the classified promotion outcome after local reconciliation."""

        if promotion.error is not None:
            if promotion.storage_error:
                self._raise_storage_write_error(promotion.error)
            raise promotion.error
        raise RuntimeError(
            f"Cannot apply config for isolated feature {self.name}: "
            "config transition conflicts with a newer durable state"
        )

    async def _prepare_config_transition(
        self, next_config: Dict[str, Any]
    ) -> ConfigTransitionResult | None:
        """Run the public SDK lifecycle hook when the live client opted in.

        Capability negotiation is intentionally limited to the SDK client's
        typed property and lifecycle method. The host neither knows nor sends
        feature-private RPC method names.
        """

        if self._client is None:
            return None
        if not self._supports_config_transition():
            return None

        prepare = getattr(self._client, "prepare_config_transition", None)
        if not callable(prepare):
            raise RuntimeError(
                f"Isolated feature {self.name} advertised config-transition support "
                "without the SDK lifecycle method"
            )

        result = await _maybe_await(prepare(next_config))
        if not isinstance(result, ConfigTransitionResult):
            raise RuntimeError(
                f"Isolated feature {self.name} returned an invalid config-transition result"
            )
        return result

    async def _prepare_config_transition_with_lease(
        self, transition: _ConfigTransition
    ) -> ConfigTransitionResult | None:
        """Run the external hook while continuously proving stage ownership."""

        if not transition.persistent:
            return await self._prepare_config_transition(transition.next_config)

        stop_heartbeat = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._run_transition_lease_heartbeat(transition, stop_heartbeat),
            name=f"isolated-config-lease:{self.name}",
        )
        hook = asyncio.create_task(
            self._prepare_config_transition(transition.next_config),
            name=f"isolated-config-hook:{self.name}",
        )
        try:
            done, _ = await asyncio.wait(
                {hook, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                # A healthy heartbeat only finishes when asked to stop below;
                # before that, its completion is a lost lease or storage error.
                heartbeat.result()
                raise _ConfigTransitionLeaseLost(
                    "isolated config transition lease heartbeat stopped unexpectedly"
                )
            return hook.result()
        finally:
            # If the lease cannot be renewed, cancellation may reach the SDK
            # child; wait for its cancellation to settle before recovery makes
            # a replacement decision.  No lifecycle task may outlive the
            # reload lock.
            if not hook.done():
                hook.cancel()
                try:
                    await self._await_task_completion(
                        hook, preserve_cancellation=True
                    )
                except asyncio.CancelledError:
                    pass
            stop_heartbeat.set()
            try:
                await self._await_task_completion(
                    heartbeat, preserve_cancellation=True
                )
            except asyncio.CancelledError:
                pass

    def _supports_config_transition(self) -> bool:
        """Whether the initialized SDK client explicitly opted into transitions."""

        return (
            self._client is not None
            and getattr(self._client, "supports_config_transition", False) is True
        )

    def _client_requires_replacement(self) -> bool:
        """Whether the SDK fenced the current child after an unknown outcome."""

        return getattr(self._client, "replacement_required", False) is True

    async def _load_host_config(self) -> Dict[str, Any]:
        """Resolve persisted/UI host config to forward into the service.

        Reads the same graph-store node the in-process Feature base persists to
        (``feature_config:<name>``). An absent node is an intentional empty
        config. A failed read is not: starting with ``{}`` would make that
        transient failure authoritative and could overwrite a write-only secret
        in a later partial update, so initialization fails until storage recovers.
        """
        try:
            persisted = await self.load_persisted_config(raise_on_error=True)
        except Exception as exc:  # noqa: BLE001 - durable read is a hard boundary
            raise RuntimeError(
                f"Cannot initialize isolated feature {self.name}: failed to load persisted config"
            ) from exc
        return persisted if isinstance(persisted, dict) else {}

    async def _ensure_host_config_loaded(self) -> None:
        """Load durable config exactly before it may become host-authoritative."""

        if self._host_config_loaded:
            return
        self._host_config = await self._load_host_config()
        self._host_config_loaded = True

    # ------------------------------------------------------------------
    # Channel bridge
    # ------------------------------------------------------------------

    def _client_capabilities(self) -> Dict[str, Any]:
        # Prefer the wrapper's passthrough; fall back to the inner JSON-RPC
        # client, where SubprocessIsolatedFeatureClient stores capabilities after
        # initialize (covers SDK builds without the wrapper-level property).
        caps = getattr(self._client, "capabilities", None)
        if not isinstance(caps, dict) or not caps:
            inner = getattr(self._client, "client", None)
            inner_caps = getattr(inner, "capabilities", None)
            if isinstance(inner_caps, dict):
                caps = inner_caps
        return caps if isinstance(caps, dict) else {}

    def _supports_tool_execution_context(self, context: Any) -> bool:
        """Whether the initialized service accepts this SDK context version.

        New SDK clients expose a typed boolean property.  Reading the raw
        initialize capability as a fallback keeps compatible wrappers usable,
        while malformed or legacy capability data fails closed for scheduler
        delivery.
        """

        supported = getattr(self._client, "supports_tool_execution_context", None)
        if isinstance(supported, bool):
            return supported

        capability = self._client_capabilities().get("tool_execution_context")
        versions = capability.get("versions") if isinstance(capability, dict) else None
        return (
            isinstance(versions, list)
            and not isinstance(getattr(context, "version", None), bool)
            and getattr(context, "version", None) in versions
        )

    def _register_channel_bridge(self) -> None:
        """If the service advertises a channel capability, register a forwarding
        ``ChannelAdapter`` so the generic channels API works against this
        isolated feature (otherwise only the feature's own tools + inbound
        events work, not ``channels_send``/``channels_list``)."""
        channel = self._client_capabilities().get("channel")
        if not isinstance(channel, dict):
            return
        channel_type = channel.get("channel_type")
        send_tool = channel.get("send_tool")
        if not channel_type or not send_tool:
            logger.warning(
                "Isolated feature %s advertised an incomplete channel capability: %r",
                self.name,
                channel,
            )
            return

        channel_feature = self._channel_feature()
        registry = getattr(channel_feature, "registry", None) if channel_feature else None
        if registry is None:
            logger.warning(
                "Isolated channel feature %s cannot bridge: ChannelFeature/registry "
                "unavailable; channels_send will not see channel '%s'",
                self.name,
                channel_type,
            )
            return

        adapter = ProxyChannelAdapter(
            self,
            channel_type=str(channel_type),
            send_tool=str(send_tool),
            status_tool=channel.get("status_tool"),
            config=self._channel_config(str(channel_type)),
        )
        registry.register(adapter)
        self._channel_adapter = adapter
        # Remember the pairing tool so ``call_isolated_tool`` can emit the
        # persisted ``channel_link`` part when it runs (#2081). Prefer an
        # explicitly advertised ``link_tool``; otherwise fall back to the
        # ``<channel_type>_link`` naming convention (e.g. ``whatsapp_link``).
        self._channel_type = str(channel_type)
        link_tool = channel.get("link_tool")
        self._link_tool = str(link_tool) if link_tool else f"{channel_type}_link"
        logger.info(
            "Bridged isolated feature %s into ChannelFeature.registry as channel '%s'",
            self.name,
            channel_type,
        )

    def _unregister_channel_bridge(self) -> None:
        if self._channel_adapter is None:
            return
        channel_feature = self._channel_feature()
        registry = getattr(channel_feature, "registry", None) if channel_feature else None
        if registry is not None:
            # Only remove our own adapter: a reload or a native adapter may have
            # since replaced this channel_type, and we must not evict it.
            getter = getattr(registry, "get", None)
            if not callable(getter) or getter(self._channel_adapter.channel_type) is self._channel_adapter:
                registry.unregister(self._channel_adapter.channel_type)
        self._channel_adapter = None

    def _channel_feature(self) -> Any:
        features = getattr(self.agent, "features", None)
        if isinstance(features, dict):
            return features.get("ChannelFeature")
        getter = getattr(features, "get", None)
        return getter("ChannelFeature") if callable(getter) else None

    def _channel_config(self, channel_type: str):
        """Build a ChannelConfig from host config for sender-filtering / enabled.

        Inbound sender filtering and the disabled-channel guard both read the
        registered adapter's ``config``, so mirror the host-side feature config
        onto the forwarding adapter.
        """
        from kestrel_sdk.channels import ChannelConfig

        cfg = self._host_config or {}
        return ChannelConfig(
            channel_type=channel_type,
            agent_id=str(cfg.get("agent_id", "") or ""),
            enabled=bool(cfg.get("enabled", True)),
            allowed_senders=list(cfg.get("allowed_senders") or []),
        )

    async def call_isolated_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        context = _scheduled_tool_execution_context()
        # The gate keeps a selected client alive through the complete RPC.  It
        # is shared (not a reload mutex), so unrelated calls remain concurrent
        # whenever no config transition is active.
        try:
            async with self._traffic_gate.admit():
                # This preflight belongs *inside* admission. Otherwise a
                # terminal shutdown with no published client would leak a
                # scheduler-specific error instead of the stable fail-closed
                # result used by every other new tool/channel call.
                if context is not None and not self._supports_tool_execution_context(context):
                    raise SchedulerExecutionContextUnavailable(
                        "scheduled isolated tool calls require a service that advertises "
                        "ToolExecutionContext support"
                    )
                return await self._call_isolated_tool_admitted(name, args, context)
        except _TrafficGateTerminalError:
            # Unlike an ordinary RPC error, terminal admission has no client to
            # retry against. Keep the public result stable and secret-free for
            # both direct tools and ProxyChannelAdapter sends.
            return {
                "status": "error",
                "error": _TERMINAL_TRAFFIC_ERROR,
                "tool": name,
                "success": False,
            }

    async def _call_isolated_tool_admitted(
        self,
        name: str,
        args: Dict[str, Any],
        context: Any | None,
    ) -> Dict[str, Any]:
        """Dispatch one tool after :meth:`call_isolated_tool` admitted traffic."""

        try:
            if self._client is None:
                raise RuntimeError("isolated feature client is unavailable")
            if context is None:
                # Preserve the existing wire format for chat/API/legacy calls.
                result = await _maybe_await(self._client.call_tool(name, args))
            else:
                # Context is reserved JSON-RPC metadata.  It must not be merged
                # into ``args``: those remain entirely user-controlled tool
                # input, while the isolated SDK authenticates this envelope.
                result = await _maybe_await(
                    self._client.call_tool(name, args, context=context)
                )
            from kestrel_sovereign.features.base import is_flat_toolresult_envelope
            if is_flat_toolresult_envelope(result):
                # Service returned the flat ToolResult envelope. Pass it through
                # TOP-LEVEL (unified shape #F025) rather than nesting it under
                # ``result`` with a hardcoded ``success: True`` — that hid a
                # service-side ``ToolResult.failed`` behind success and made the
                # honesty layer read the isolated tool as always succeeding
                # (#F018). Derive ``success`` from the service's status.
                envelope = dict(result)
                envelope["tool"] = name
                envelope["success"] = result.get("status") != "error"
                if envelope["success"]:
                    self._maybe_emit_channel_link_part(name)
                return envelope
            # Non-envelope return (a raw value from a legacy service) — keep the
            # wrapped legacy shape so existing readers still see it.
            self._maybe_emit_channel_link_part(name)
            return {
                "success": True,
                "result": result,
                "tool": name,
            }
        except SchedulerExecutionContextUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            if context is not None:
                # No scheduler effect may proceed if the negotiated context was
                # rejected or the context-aware RPC could not be delivered.
                # Raising lets SchedulerRunner persist a failed occurrence and
                # retain its stable idempotency key for recovery.
                raise SchedulerExecutionContextUnavailable(
                    "scheduled isolated tool call rejected its execution context"
                ) from exc
            logger.warning("Isolated feature tool %s.%s failed: %s", self.name, name, exc)
            # Transport/RPC failure — emit the flat error envelope so callers
            # and the honesty layer read a top-level ``status: error``.
            return {
                "status": "error",
                "error": str(exc),
                "tool": name,
                "success": False,
            }

    def _maybe_emit_channel_link_part(self, tool_name: str) -> None:
        """Emit a persisted ``channel_link`` typed part when the bridged
        channel's pairing tool ran on the active streaming turn (#2081).

        The part carries only a reference (``{channel_type}``), not the QR
        bytes: the chat card resolves the current QR state live from
        ``/api/agent/channels/<type>/link-qr.png``. Emitting here — inside the
        tool's host-side execution, which runs within the turn's part-collector
        contextvar — makes the card ride the message that requested the link so
        it persists in that conversation and survives a refresh. ``emit_part``
        is a no-op off a streaming turn, so this is safe on any call path.
        """
        if not self._link_tool or tool_name != self._link_tool or not self._channel_type:
            return
        try:
            from kestrel_sovereign.agent.parts import emit_part

            emit_part("channel_link", {"channel_type": self._channel_type})
        except Exception as exc:  # noqa: BLE001
            logger.debug("channel_link emit_part failed for %s: %s", self.name, exc)

    def resolve_runtime_paths(self) -> tuple[Path, Optional[Path]]:
        bin_override = os.environ.get(_env_key(self.name, "BIN"))
        if bin_override:
            return self._default_venv_path(), Path(bin_override).expanduser().resolve()

        venv_override = os.environ.get(_env_key(self.name, "VENV"))
        if venv_override:
            return Path(venv_override).expanduser().resolve(), None

        if self.runtime.venv:
            return Path(self.runtime.venv).expanduser().resolve(), None

        return self._default_venv_path(), None

    def _venv_is_overridden(self) -> bool:
        """True when the venv path was supplied by the operator (KESTREL_FEATURE_
        <NAME>_VENV env or the pyproject ``venv =``) rather than provisioned by
        the host at the default path. An operator-supplied venv is NOT ours to
        mutate — see ensure_venv."""
        return bool(
            os.environ.get(_env_key(self.name, "VENV"))
            or self.runtime.venv
        )

    def _default_venv_path(self) -> Path:
        return _agent_data_dir(self.agent) / "feature_venvs" / self.name / ".venv"

    def _provision_manifest_path(self) -> Path:
        # Inside the venv dir, not its parent: explicit venv overrides can share
        # a parent directory, and a parent-scoped manifest would let sibling
        # features clobber each other's stamp and reinstall on every startup.
        assert self._venv_path is not None
        return self._venv_path / ".kestrel_provision.json"

    def _read_provision_manifest(self) -> Dict[str, Any]:
        try:
            return json.loads(self._provision_manifest_path().read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt manifest ⇒ reprovision
            return {}

    def _write_provision_manifest(
        self, install_target: str, host_sdk: str, child_sdk: str
    ) -> None:
        path = self._provision_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "install_target": install_target,
                    # The host SDK we provisioned AGAINST — staleness keys on a
                    # change here, so a genuinely SDK-pinned feature reinstalls
                    # once per host bump, not on every startup.
                    "provisioned_against_host_sdk": host_sdk,
                    # The SDK version that actually landed in the venv (may lag
                    # host_sdk if the feature pins it); recorded for diagnosis.
                    "child_sdk_version": child_sdk,
                },
                indent=2,
            )
        )

    def _provision_is_stale(self, install_target: str) -> bool:
        """A provisioned venv is stale if the install target changed or the host
        has upgraded kestrel-sdk since we last provisioned against it (F019: a
        stale wire contract — e.g. pre-0.28 serial dispatch — must not silently
        survive a host update)."""
        manifest = self._read_provision_manifest()
        if manifest.get("install_target") != install_target:
            return True
        if manifest.get("provisioned_against_host_sdk") != _host_sdk_version():
            return True
        return False

    def ensure_venv(self) -> None:
        assert self._venv_path is not None
        python_path = _venv_python(self._venv_path)

        # Install the PROJECT (path/dist), never the `service` runnable — the
        # latter is a console-script name or "module:func", not a pip target.
        install_target = self.runtime.project or self.runtime.distribution
        if not install_target:
            raise RuntimeError(
                f"Isolated feature {self.name} has no project/distribution to install"
            )

        exists = python_path.exists()

        # A PREBUILT operator-supplied (override) venv is NOT ours to mutate:
        # running `uv pip install --upgrade` into it would rewrite a prebuilt/
        # pinned environment the operator deliberately provided (and hard-fail
        # the whole feature at startup if the index is unreachable). We recognize
        # a prebuilt override as one that exists at an override path AND carries
        # no provision manifest of ours — i.e. we did not create it. Verify SDK
        # compatibility and warn on a mismatch (See Something Say Something), but
        # leave it untouched and stamp nothing. An override venv WE created
        # earlier (our manifest present) keeps the full reprovision lifecycle, as
        # do host-owned default venvs — both fall through below.
        if (
            exists
            and self._venv_is_overridden()
            and not self._provision_manifest_path().exists()
        ):
            self._warn_on_sdk_mismatch(python_path)
            return

        # An operator-supplied (override) venv that already exists is NOT ours to
        # mutate: running `uv pip install --upgrade` into it would rewrite a
        # prebuilt/pinned environment the operator deliberately provided (and
        # hard-fail the whole feature at startup if the index is unreachable).
        # Verify SDK compatibility and warn on a mismatch (See Something Say
        # Something), but leave the venv untouched and do not stamp a manifest we
        # don't own. Host-owned default venvs (and a not-yet-created override
        # path we bootstrap below) keep the full reprovision lifecycle.
        if not exists:
            self._run(["uv", "venv", str(self._venv_path)])
        elif not self._provision_is_stale(install_target):
            return

        # Fresh venv, changed install target, or host SDK upgraded since the
        # venv was provisioned. On an existing venv, upgrade in place so a stale
        # kestrel-sdk is replaced; then stamp the manifest so the next startup
        # can tell whether another reprovision is due.
        cmd = ["uv", "pip", "install", "--python", str(python_path)]
        if exists:
            cmd.append("--upgrade")
        cmd.append(install_target)
        self._run(cmd)

        # Verify what actually landed: a feature that pins an older SDK can
        # install "successfully" while keeping the stale wire contract. Surface
        # that rather than silently stamping the venv as fresh (See Something
        # Say Something) — staleness still keys on the host transition so we
        # don't thrash reinstalling a genuinely pinned feature every startup.
        host_sdk = _host_sdk_version()
        child_sdk = _venv_sdk_version(python_path)
        self._warn_on_sdk_mismatch(python_path, host_sdk=host_sdk, child_sdk=child_sdk)
        self._write_provision_manifest(install_target, host_sdk, child_sdk)

    def _warn_on_sdk_mismatch(
        self, python_path: Path, *, host_sdk: str = None, child_sdk: str = None
    ) -> None:
        host_sdk = host_sdk if host_sdk is not None else _host_sdk_version()
        child_sdk = child_sdk if child_sdk is not None else _venv_sdk_version(python_path)
        if child_sdk != host_sdk and "unknown" not in (child_sdk, host_sdk):
            logger.warning(
                "Isolated feature %s venv resolved kestrel-sdk %s but host is %s — "
                "the feature may pin an incompatible wire contract",
                self.name,
                child_sdk,
                host_sdk,
            )

    def _run(self, cmd: List[str]) -> None:
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(f"Required executable not found: {cmd[0]}")
        subprocess.run(cmd, check=True)

    def _build_client(self, config: Optional[Dict[str, Any]] = None) -> Any:
        factory = self._client_factory
        if factory is None:
            from kestrel_sdk.isolated_feature import SubprocessIsolatedFeatureClient

            factory = SubprocessIsolatedFeatureClient

        child_config = self._host_config if config is None else config

        kwargs = {
            "feature_name": self.name,
            "service": self.runtime.service,
            "venv_path": str(self._venv_path) if self._venv_path else None,
            "python": str(_venv_python(self._venv_path)) if self._venv_path else None,
            "executable": str(self._bin_path) if self._bin_path else None,
            "event_handler": self._handle_event,
            "notification_handler": self._handle_event,
            # An empty object is an explicit effective config: the SDK sends
            # ``config`` only when this value is not ``None``, and its service
            # then calls ``configure({})``. Do not collapse it into a missing
            # config field.
            "config": child_config,
            # Launch env with interpreter-shadowing vars stripped (F023) so the
            # host PYTHONPATH/VIRTUAL_ENV can't defeat the venv isolation.
            "env": _isolated_child_env(self._venv_path),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        try:
            signature = inspect.signature(factory)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return factory(**kwargs)

        accepted = {key: value for key, value in kwargs.items() if key in params}

        # Keyword-only factory (named params, no positional `command`): deliver the
        # accepted keyword args directly (config/event handlers/etc.).
        if "command" not in params:
            if accepted:
                return factory(**accepted)
            return factory()

        # Positional-command constructor (SubprocessIsolatedFeatureClient): pass the
        # launch argv plus whatever keyword extras the factory accepts (notably
        # `config`, so host config reaches the service via the initialize handshake).
        accepted.pop("command", None)
        try:
            return factory(self._service_command(), **accepted)
        except (TypeError, ValueError):
            return factory(self._service_command())

    def _service_command(self) -> List[str]:
        """Build the argv to launch the isolated service.

        Resolution order:
          1. explicit BIN override (``self._bin_path``);
          2. ``service`` of the form ``module:func`` -> ``<venv-python> -c ...``;
          3. ``service`` as a console-script name -> ``<venv>/bin/<script>``.
        The ``service`` runnable is NEVER treated as a ``python -m`` module
        (it may be a path/dist), which is what previously broke startup.
        """
        if self._bin_path is not None:
            return [str(self._bin_path)]

        service = self.runtime.service
        if not service:
            raise RuntimeError(
                f"Isolated feature {self.name} has no `service` runnable configured"
            )

        if ":" in service:  # module:func callable
            module, _, func = service.partition(":")
            python = (
                str(_venv_python(self._venv_path)) if self._venv_path else "python"
            )
            return [python, "-c", f"from {module} import {func}; {func}()"]

        # console-script installed into the venv's bin/Scripts dir
        if self._venv_path is not None:
            return [str(_venv_bin_dir(self._venv_path) / service)]
        return [service]

    async def _register_event_handler(self, client: Any = None) -> None:
        """Attach the host event handler to a published client.

        Accepting the client explicitly lets fenced recovery keep a started
        candidate detached until durable promotion has completed.
        """

        target = self._client if client is None else client
        register = (
            getattr(target, "set_event_handler", None)
            or getattr(target, "add_event_handler", None)
            or getattr(target, "subscribe", None)
        )
        if register is not None:
            await _maybe_await(register(self._handle_event))
            return

        on_event = getattr(target, "on_event", None)
        if on_event is None:
            return
        try:
            signature = inspect.signature(on_event)
            params = list(signature.parameters.values())
        except (TypeError, ValueError):
            return
        if not params:
            return
        first_name = params[0].name
        if first_name in {"handler", "callback", "event_handler"}:
            await _maybe_await(on_event(self._handle_event))

    def _start_supervision(self) -> asyncio.Task:
        """Start the supervision loop, registered with the agent's background-task
        lifecycle when available so normal agent shutdown cancels it (otherwise
        the child process + task leak — agent shutdown does not call every
        feature's ``shutdown()``). Falls back to a bare task (e.g. under test
        doubles whose ``_track_background_task`` doesn't return a real Task)."""
        name = f"isolated-feature:{self.name}"
        coro = self._supervise()
        tracker = getattr(self.agent, "_track_background_task", None)
        if callable(tracker):
            try:
                task = tracker(coro, name=name)
            except Exception:  # noqa: BLE001
                task = None
            if isinstance(task, asyncio.Task):
                return task
        return asyncio.create_task(coro, name=name)

    async def _supervise(self) -> None:
        try:
            backoff = 1.0
            while not self._stopping:
                await asyncio.sleep(backoff)
                # A ``set_config`` reload intentionally stops/starts the client;
                # don't probe (and "restart") a service that is mid-reload.
                if self._reloading:
                    backoff = 1.0
                    continue
                # Snapshot the reload generation BEFORE probing: if a reload cycles
                # the client while this (now-stale) probe is in flight, we must not
                # then "restart" the freshly launched client.
                gen = self._reload_gen
                try:
                    health = await asyncio.wait_for(
                        _maybe_await(self._client.health()),
                        timeout=_HEALTH_PROBE_TIMEOUT,
                    )
                    healthy = self._is_healthy_response(health)
                    if healthy:
                        backoff = 1.0
                        continue
                except asyncio.TimeoutError:
                    logger.warning(
                        "Isolated feature %s health probe exceeded %ss — treating as "
                        "wedged and restarting",
                        self.name,
                        _HEALTH_PROBE_TIMEOUT,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Isolated feature %s health check failed: %s", self.name, exc)

                if self._stopping:
                    break
                # Serialize the restart against a concurrent reload, and re-check
                # under the lock: a reload may have started during our probe (so
                # the failed probe was expected) or completed with a fresh healthy
                # client. Either way the reload owns the lifecycle — skip.
                async with self._reload_lock:
                    if self._stopping or self._reloading or self._reload_gen != gen:
                        backoff = 1.0
                        continue
                    # Mark ownership before draining. A supervisor cancellation
                    # while an admitted tool is active must still run the final
                    # gate boundary below instead of leaving callers blocked.
                    gate_closed = True
                    try:
                        await self._close_traffic_gate()
                        try:
                            await _maybe_await(self._client.stop())
                        except Exception:  # noqa: BLE001
                            pass
                        await asyncio.sleep(backoff)
                        try:
                            await _maybe_await(self._client.start())
                            backoff = 1.0
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Isolated feature %s restart failed: %s", self.name, exc)
                            backoff = min(backoff * 2, 30.0)
                    finally:
                        if gate_closed:
                            if self._stopping:
                                await self._seal_traffic_gate()
                            else:
                                await self._reopen_traffic_gate()
        finally:
            # If the task is cancelled (e.g. agent shutdown cancelling tracked
            # background tasks) rather than stopped via shutdown(), make sure the
            # child process is still torn down so it can't outlive the agent.
            # This is terminal rather than a finite restart: release pending
            # admissions with the stable fail-closed result before retirement.
            if not self._stopping:
                self._stopping = True
                await self._seal_traffic_gate()
                client = self._unpublish_client()
                if client is not None:
                    try:
                        await _maybe_await(client.stop())
                    except Exception:  # noqa: BLE001
                        pass

    @staticmethod
    def _is_healthy_response(health: Any) -> bool:
        """Interpret the SDK health envelope without treating an error as ready.

        Legacy client doubles may return a boolean. SDK clients return an
        object, including ``{\"status\": \"restart-required\", \"ready\": false}``
        after a child has been fenced. A non-empty mapping is not evidence of
        readiness, so unknown envelopes fail closed rather than suppressing a
        required replacement.
        """

        if not isinstance(health, dict):
            return bool(health)
        if health.get("replacement_required") is True:
            return False
        if "ready" in health:
            return health["ready"] is True
        if "ok" in health:
            return health["ok"] is True
        if "healthy" in health:
            return health["healthy"] is True
        status = health.get("status")
        if isinstance(status, str):
            return status.lower() in {"ready", "ok", "healthy", "running"}
        return False

    async def _handle_event(self, event: Any) -> None:
        # SDK event callbacks are externally visible traffic too: an inbound
        # channel message can wake the agent and trigger effects.  Keep it on
        # the same gate as tools so a candidate hook cannot route an event
        # before its config becomes durable. Events are deliberately *dropped*
        # rather than queued during a finite close: they originated from the
        # old child and replaying them after promotion could apply stale input
        # under a new configuration. Terminal drops are also silent because an
        # SDK callback has no caller to handle a deliberate shutdown result.
        try:
            async with self._traffic_gate.admit(wait_for_open=False):
                await self._handle_event_admitted(event)
        except (_TrafficGateClosedError, _TrafficGateTerminalError):
            return

    async def _handle_event_admitted(self, event: Any) -> None:
        kind = _meta_get(event, "type") or _meta_get(event, "event") or _meta_get(event, "kind")
        payload = _meta_get(event, "payload", event)
        if kind == "feature/event":
            event_name = _meta_get(payload, "name") or _meta_get(payload, "event")
            data = _meta_get(payload, "data", payload)
        else:
            event_name = kind
            data = payload

        if event_name in {"channel.inbound", "inbound", "message.inbound"}:
            await self._route_inbound(data)
        elif event_name in {"channel.link_qr", "link_qr", "channel.qr"}:
            await self._route_link_qr(data)
        elif event_name in {"channel.link_cleared", "link_cleared"}:
            await self._route_link_cleared(data)

    async def _route_link_cleared(self, payload: Any) -> None:
        """Retract a channel pairing QR once the channel is linked.

        Removes the persisted PNG so the ``channel_link`` card (#2081) resolves
        to "expired or already linked" on its next fetch. No sticky/SSE state to
        clear anymore — the card is a persisted typed part, not a live bubble.
        """
        if not isinstance(payload, dict):
            return
        channel_type = str(payload.get("channel_type") or "").strip().lower()
        if not channel_type or not re.fullmatch(r"[a-z0-9_]{1,32}", channel_type):
            return
        png = (
            _agent_data_dir(self.agent)
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            png.unlink(missing_ok=True)
        except OSError:
            pass

    async def _route_link_qr(self, payload: Any) -> None:
        """Persist the latest channel pairing-QR PNG under the agent data dir.

        Isolated channel features emit ``channel.link_qr`` with the QR rendered
        as a PNG (base64) each time a pairing code is produced (it rotates
        ~20s). The host writes the latest under the agent data dir, served by
        ``/api/agent/channels/{type}/link-qr.png``. The chat's persisted
        ``channel_link`` card (#2081) fetches that endpoint on render/refresh —
        so the QR is no longer pushed as a live SSE bubble (which orphaned on
        refresh, #1918); the PNG is simply the current state the card resolves.
        """
        if not isinstance(payload, dict):
            return
        import base64

        channel_type = str(payload.get("channel_type") or "").strip().lower()
        png_b64 = payload.get("png_b64") or payload.get("png")
        if (
            not channel_type
            or not re.fullmatch(r"[a-z0-9_]{1,32}", channel_type)
            or not png_b64
        ):
            logger.warning("Dropping malformed channel.link_qr from %s", self.name)
            return
        try:
            png_bytes = base64.b64decode(png_b64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("channel.link_qr from %s had undecodable PNG: %s", self.name, exc)
            return

        out = (
            _agent_data_dir(self.agent)
            / "channel_link_artifacts"
            / f"{channel_type}_link_qr.png"
        )
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png_bytes)
        except OSError as exc:
            logger.warning("Failed to persist channel.link_qr PNG for %s: %s", self.name, exc)
            return

    async def _route_inbound(self, payload: Any) -> None:
        channel = self._channel_feature()
        if channel is None or not hasattr(channel, "handle_inbound"):
            logger.warning(
                "Inbound notification from %s dropped: ChannelFeature unavailable",
                self.name,
            )
            return

        from kestrel_sovereign.features.channels.models import ChannelMessage

        message = payload
        if isinstance(payload, dict):
            # from_dict coerces the wire shape (string direction/timestamp) back
            # into typed fields and ignores unknown keys.
            message = ChannelMessage.from_dict(payload)
        await channel.handle_inbound(message)


def _send_outcome(envelope: Dict[str, Any], transport_ok: bool):
    """Classify a send tool's envelope into a DeliveryStatus.

    Recognizes two wire shapes a channel send tool may return:
      * a plain ``{"ok": bool, ...}`` envelope, and
      * the framework ``ToolResult`` shape ``{"status": "ok"|"error"|
        "partial", ...}`` (ok->SUCCESS, error->FAILURE, partial->PENDING).
    An explicit error/non-OK envelope must NOT be reported as a successful
    delivery just because the JSON-RPC transport itself succeeded.
    """
    from kestrel_sdk.channels import DeliveryStatus

    if not transport_ok:
        return DeliveryStatus.FAILURE
    if "ok" in envelope:
        return DeliveryStatus.SUCCESS if envelope["ok"] else DeliveryStatus.FAILURE
    status = envelope.get("status")
    if isinstance(status, str):
        token = status.lower()
        if token in {"error", "failure", "failed"}:
            return DeliveryStatus.FAILURE
        if token in {"partial", "pending"}:
            return DeliveryStatus.PENDING
        if token in {"ok", "success"}:
            return DeliveryStatus.SUCCESS
    if "success" in envelope:
        return DeliveryStatus.SUCCESS if envelope["success"] else DeliveryStatus.FAILURE
    # No tool-level signal: trust the transport outcome.
    return DeliveryStatus.SUCCESS


def _delivery_receipt_from_result(channel_type: str, result: Dict[str, Any]):
    """Map a forwarded isolated-tool result onto a ``DeliveryReceipt``.

    ``call_isolated_tool`` returns the tool's ToolResult envelope TOP-LEVEL
    (unified shape #F025): ``{status, confirmation, error, data, tool, success}``.
    The envelope IS the result; a legacy raw return still arrives wrapped as
    ``{success, result}`` and is tolerated below.
    """
    import uuid as _uuid

    from kestrel_sdk.channels import DeliveryReceipt, DeliveryStatus

    result = result if isinstance(result, dict) else {}
    transport_ok = bool(result.get("success", True))
    # Flat envelope (status top-level) is the result itself; tolerate a legacy
    # nested ``result`` payload for any un-migrated raw-return service.
    envelope = result
    if result.get("status") is None and isinstance(result.get("result"), dict):
        envelope = result["result"]
    data = envelope.get("data")
    data = data if isinstance(data, dict) else {}
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    message_id = str(
        envelope.get("message_id")
        or envelope.get("id")
        or data.get("message_id")
        or data.get("id")
        or receipt.get("message_id")
        or _uuid.uuid4()
    )
    status = _send_outcome(envelope, transport_ok)
    if status is DeliveryStatus.FAILURE:
        error = envelope.get("error") or result.get("error") or "send failed"
        return DeliveryReceipt(
            message_id=message_id,
            status=status,
            channel_type=channel_type,
            error=str(error),
        )
    return DeliveryReceipt(message_id=message_id, status=status, channel_type=channel_type)


class ProxyChannelAdapter(ChannelAdapter):
    """Forwarding ``ChannelAdapter`` backed by an isolated feature service.

    Registered into ``ChannelFeature.registry`` for isolated features that
    advertise a channel capability, so the generic channels API routes through
    the out-of-venv service. Sends forward to the service's ``send_tool``;
    inbound flows independently via ``channel.inbound`` events (handled by the
    proxy's event handler), so ``on_message`` is a no-op here.
    """

    def __init__(
        self,
        proxy: "ProxyFeature",
        *,
        channel_type: str,
        send_tool: str,
        status_tool: Optional[str] = None,
        config: Any = None,
    ):
        super().__init__(config)
        self._proxy = proxy
        self._channel_type = channel_type
        self._send_tool = send_tool
        self._status_tool = status_tool
        # Registered only after the service reports ready; treat as connected
        # until told otherwise. The send receipt carries the real per-send truth.
        self._connected = True

    @property
    def channel_type(self) -> str:
        return self._channel_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def on_message(self, callback) -> None:
        # Inbound is delivered to ChannelFeature.handle_inbound via the proxy's
        # event handler, not through an adapter-held callback.
        return None

    async def send_message(self, to: str, content: str, **kwargs):
        result = await self._proxy.call_isolated_tool(
            self._send_tool, {"to": to, "message": content}
        )
        return _delivery_receipt_from_result(self._channel_type, result)
