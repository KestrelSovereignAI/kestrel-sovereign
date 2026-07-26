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
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from kestrel_sdk.channels import ChannelAdapter
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
        self._venv_path: Optional[Path] = None
        self._bin_path: Optional[Path] = None
        self._host_config: Dict[str, Any] = {}
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
        # Reset lifecycle state to the fresh-start baseline BEFORE the client or
        # supervisor start. ``shutdown()`` latches ``_stopping=True`` to unwind
        # the health supervisor; runtime re-enable re-runs this SAME instance's
        # ``initialize()`` (``_activate_feature_runtime``), so without the reset
        # the new ``_supervise()`` task sees a stale ``_stopping`` and exits on
        # its first ``while not self._stopping`` check — leaving a re-enabled
        # service with no health supervisor (kestrel-sovereign#2522 P2).
        self._stopping = False
        self._venv_path, self._bin_path = self.resolve_runtime_paths()
        if self._bin_path is None:
            self.ensure_venv()
        # Resolve persisted/UI host config BEFORE building the client so it can be
        # forwarded to the isolated service through the initialize handshake (the
        # service is otherwise launched bare, with only env vars).
        self._host_config = await self._load_host_config()
        await self._connect_client()
        self._supervision_task = self._start_supervision()

    async def _connect_client(self) -> None:
        """Build + start the isolated client from the current ``_host_config``,
        then wire event handling, tools, and the channel bridge.

        Shared by ``initialize`` (first launch) and ``reload`` (re-launch after a
        config change). ``_build_client`` snapshots ``_host_config`` into the
        service's initialize handshake, so rebuilding here is how new config
        actually reaches a running service.
        """
        self._client = self._build_client()
        await _maybe_await(self._client.start())
        await self._register_event_handler()
        advertised_tools = await _maybe_await(self._client.list_tools())
        self._tools = [IsolatedFeatureTool(self, meta) for meta in advertised_tools]
        self._register_channel_bridge()

    async def reload(self) -> None:
        """Restart the isolated service so the current ``_host_config`` takes
        effect (config is forwarded only at the initialize handshake, so a live
        config change requires a re-launch). Guarded so the health supervisor
        doesn't treat the intentional stop as a crash and double-restart."""
        async with self._reload_lock:
            self._reloading = True
            self._reload_gen += 1
            try:
                self._unregister_channel_bridge()
                if self._client is not None:
                    await _maybe_await(self._client.stop())
                await self._connect_client()
            finally:
                self._reloading = False

    async def shutdown(self):
        self._stopping = True
        self._unregister_channel_bridge()
        if self._supervision_task:
            self._supervision_task.cancel()
            try:
                await self._supervision_task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await _maybe_await(self._client.stop())

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
        if self._client is not None and hasattr(self._client, "get_config"):
            live = await _maybe_await(self._client.get_config())
            if live:
                return dict(live)
        if self._host_config:
            return dict(self._host_config)
        persisted = await self.load_persisted_config()
        return dict(persisted) if isinstance(persisted, dict) else {}

    async def set_config(self, config: Dict) -> None:
        """Persist the config AND apply it to the running service.

        The previous implementation forwarded to ``self._client.set_config`` —
        which the SDK client does not implement — so config set via the API/UI
        was silently dropped: never persisted (so lost on restart) and never
        applied (#2214). Persist to the ``feature_config:<name>`` graph node (the
        same node ``_load_host_config`` reads at startup), update the in-memory
        host config, then ``reload`` so the new values reach the running service
        through a fresh initialize handshake.
        """
        cfg = dict(config) if isinstance(config, dict) else {}
        await self.persist_config(cfg)
        self._host_config = cfg
        if self._client is not None:
            await self.reload()

    async def _load_host_config(self) -> Dict[str, Any]:
        """Resolve persisted/UI host config to forward into the service.

        Reads the same graph-store node the in-process Feature base persists to
        (``feature_config:<name>``). Best-effort: any failure yields an empty
        config rather than blocking feature startup.
        """
        try:
            persisted = await self.load_persisted_config()
        except Exception as exc:  # noqa: BLE001
            logger.debug("No persisted config for isolated feature %s: %s", self.name, exc)
            return {}
        return persisted if isinstance(persisted, dict) else {}

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
        if context is not None and not self._supports_tool_execution_context(context):
            raise SchedulerExecutionContextUnavailable(
                "scheduled isolated tool calls require a service that advertises "
                "ToolExecutionContext support"
            )
        try:
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

    def _build_client(self) -> Any:
        factory = self._client_factory
        if factory is None:
            from kestrel_sdk.isolated_feature import SubprocessIsolatedFeatureClient

            factory = SubprocessIsolatedFeatureClient

        kwargs = {
            "feature_name": self.name,
            "service": self.runtime.service,
            "venv_path": str(self._venv_path) if self._venv_path else None,
            "python": str(_venv_python(self._venv_path)) if self._venv_path else None,
            "executable": str(self._bin_path) if self._bin_path else None,
            "event_handler": self._handle_event,
            "notification_handler": self._handle_event,
            "config": self._host_config or None,
            # Launch env with interpreter-shadowing vars stripped (F023) so the
            # host PYTHONPATH/VIRTUAL_ENV can't defeat the venv isolation.
            "env": _isolated_child_env(self._venv_path),
        }
        kwargs = {key: value for key, value in kwargs.items() if value}

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

    async def _register_event_handler(self) -> None:
        register = (
            getattr(self._client, "set_event_handler", None)
            or getattr(self._client, "add_event_handler", None)
            or getattr(self._client, "subscribe", None)
        )
        if register is not None:
            await _maybe_await(register(self._handle_event))
            return

        on_event = getattr(self._client, "on_event", None)
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
                    healthy = bool(health)
                    if isinstance(health, dict):
                        healthy = bool(health.get("ok", health.get("healthy", True)))
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
            # If the task is cancelled (e.g. agent shutdown cancelling tracked
            # background tasks) rather than stopped via shutdown(), make sure the
            # child process is still torn down so it can't outlive the agent.
            if not self._stopping and self._client is not None:
                try:
                    await _maybe_await(self._client.stop())
                except Exception:  # noqa: BLE001
                    pass

    async def _handle_event(self, event: Any) -> None:
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
