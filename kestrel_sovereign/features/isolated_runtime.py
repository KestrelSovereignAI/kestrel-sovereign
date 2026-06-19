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
import inspect
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from kestrel_sdk.channels import ChannelAdapter
from kestrel_sdk.tools.base import AgentTool, ToolCategory, ToolSchema

from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.base import Feature

logger = logging.getLogger(__name__)


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
        return ToolSchema(
            name=self.name,
            description=str(_meta_get(self._metadata, "description", "")),
            category=_coerce_category(_meta_get(self._metadata, "category")),
            parameters=_meta_get(self._metadata, "parameters", {}) or {},
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
        self._venv_path: Optional[Path] = None
        self._bin_path: Optional[Path] = None
        self._host_config: Dict[str, Any] = {}
        self._channel_adapter: Optional["ProxyChannelAdapter"] = None

    @property
    def tool_description(self) -> str:
        if self.runtime.description:
            return self.runtime.description
        return f"Isolated feature service for {self.name}"

    @property
    def config_schema(self) -> Optional[Dict]:
        return self.runtime.config_schema

    async def initialize(self):
        self._venv_path, self._bin_path = self.resolve_runtime_paths()
        if self._bin_path is None:
            self.ensure_venv()
        # Resolve persisted/UI host config BEFORE building the client so it can be
        # forwarded to the isolated service through the initialize handshake (the
        # service is otherwise launched bare, with only env vars).
        self._host_config = await self._load_host_config()
        self._client = self._build_client()
        await _maybe_await(self._client.start())
        await self._register_event_handler()
        advertised_tools = await _maybe_await(self._client.list_tools())
        self._tools = [IsolatedFeatureTool(self, meta) for meta in advertised_tools]
        self._register_channel_bridge()
        self._supervision_task = self._start_supervision()

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

    async def get_config(self) -> Dict:
        if self._client is not None and hasattr(self._client, "get_config"):
            return await _maybe_await(self._client.get_config())
        return {}

    async def set_config(self, config: Dict) -> None:
        if self._client is not None and hasattr(self._client, "set_config"):
            await _maybe_await(self._client.set_config(config))

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
        try:
            result = await _maybe_await(self._client.call_tool(name, args))
            return {
                "success": True,
                "result": result,
                "tool": name,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isolated feature tool %s.%s failed: %s", self.name, name, exc)
            return {
                "success": False,
                "error": str(exc),
                "tool": name,
            }

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

    def _default_venv_path(self) -> Path:
        return _agent_data_dir(self.agent) / "feature_venvs" / self.name / ".venv"

    def ensure_venv(self) -> None:
        assert self._venv_path is not None
        python_path = _venv_python(self._venv_path)
        created = False
        if not python_path.exists():
            self._run(["uv", "venv", str(self._venv_path)])
            created = True

        if not created:
            return

        # Install the PROJECT (path/dist), never the `service` runnable — the
        # latter is a console-script name or "module:func", not a pip target.
        install_target = self.runtime.project or self.runtime.distribution
        if not install_target:
            raise RuntimeError(
                f"Isolated feature {self.name} has no project/distribution to install"
            )
        self._run(["uv", "pip", "install", "--python", str(python_path), install_target])

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
        }
        kwargs = {key: value for key, value in kwargs.items() if value}

        try:
            signature = inspect.signature(factory)
            params = signature.parameters
        except (TypeError, ValueError):
            params = {}

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return factory(**kwargs)

        # Positional-command constructor (SubprocessIsolatedFeatureClient): pass the
        # launch argv plus whatever keyword extras the factory accepts (notably
        # `config`, so host config reaches the service via the initialize handshake).
        extras = {key: value for key, value in kwargs.items() if key in params}
        # `command` is supplied positionally; never also pass it by keyword.
        extras.pop("command", None)
        try:
            return factory(self._service_command(), **extras)
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
                try:
                    health = await _maybe_await(self._client.health())
                    healthy = bool(health)
                    if isinstance(health, dict):
                        healthy = bool(health.get("ok", health.get("healthy", True)))
                    if healthy:
                        backoff = 1.0
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Isolated feature %s health check failed: %s", self.name, exc)

                if self._stopping:
                    break
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

    ``call_isolated_tool`` wraps the tool's own envelope as
    ``{"success": bool, "result": <envelope>}``.
    """
    import uuid as _uuid

    from kestrel_sdk.channels import DeliveryReceipt, DeliveryStatus

    transport_ok = bool(result.get("success"))
    envelope = result.get("result")
    envelope = envelope if isinstance(envelope, dict) else {}
    data = envelope.get("data")
    data = data if isinstance(data, dict) else {}
    receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
    message_id = str(
        data.get("message_id")
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
