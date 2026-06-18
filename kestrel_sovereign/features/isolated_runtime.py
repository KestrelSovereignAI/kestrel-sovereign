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
        self._client = self._build_client()
        await _maybe_await(self._client.start())
        await self._register_event_handler()
        advertised_tools = await _maybe_await(self._client.list_tools())
        self._tools = [IsolatedFeatureTool(self, meta) for meta in advertised_tools]
        self._supervision_task = self._start_supervision()

    async def shutdown(self):
        self._stopping = True
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
        }
        kwargs = {key: value for key, value in kwargs.items() if value}

        try:
            signature = inspect.signature(factory)
            if any(
                param.kind is inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            ):
                return factory(**kwargs)
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            if accepted:
                return factory(**accepted)
        except (TypeError, ValueError):
            pass

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
        channel = (
            self.agent.features.get("ChannelFeature")
            if hasattr(self.agent, "features")
            else None
        )
        if channel is None or not hasattr(channel, "handle_inbound"):
            logger.warning(
                "Inbound notification from %s dropped: ChannelFeature unavailable",
                self.name,
            )
            return

        from kestrel_sovereign.features.channels.models import ChannelMessage

        message = payload
        if isinstance(payload, dict):
            message = ChannelMessage(**payload)
        await channel.handle_inbound(message)
