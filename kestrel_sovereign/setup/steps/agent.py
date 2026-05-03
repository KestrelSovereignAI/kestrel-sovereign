"""Create (or recognise) a sovereign agent.

This step is the wizard's bridge to the existing
:func:`kestrel_sovereign.inception_service.create_kestrel_identity_async`.
We do not duplicate inception logic — we just collect the inputs
(name + port + autostart) and call it.

Idempotence rules:

  - If an agent directory already contains ``kestrel_prime.db``, we do
    not re-incept; we just make sure rookery.toml lists it.
  - Port collisions with existing rookery rows trigger a fresh probe
    starting at :data:`DEFAULT_AGENT_START_PORT`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from dataclasses import dataclass

from kestrel_sovereign.rookery.config import (
    DEFAULT_AGENT_START_PORT,
    LocalAgentConfig,
    ROOKERY_CONFIG_FILENAME,
    RookeryConfig,
)
from kestrel_sovereign.setup.context import Flow, SetupContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreateAgentResult:
    """Outcome of :func:`create_agent`."""

    name: str
    did: str | None  # None if agent already existed
    port: int
    autostart: bool
    db_path: Path
    already_existed: bool


def create_agent(
    *,
    name: str,
    project_dir: Path,
    agent_data_root: Path,
    autostart: bool = True,
    port: int | None = None,
) -> CreateAgentResult:
    """Idempotent agent creation: incept if needed, then register in rookery.

    Used by both the wizard's ``agent`` step and ``kestrel create``. If
    the agent's ``kestrel_prime.db`` already exists, inception is skipped
    and the existing agent is just (re-)registered with the requested
    port/autostart.
    """
    rookery_path = project_dir / ROOKERY_CONFIG_FILENAME
    rookery = RookeryConfig.load(rookery_path)

    if port is None:
        existing_entry = rookery.agents.get(name)
        if existing_entry is not None:
            port = existing_entry.port
        else:
            port = _next_free_port(rookery)

    agent_dir = agent_data_root / name
    db_path = agent_dir / "kestrel_prime.db"

    did: str | None = None
    already_existed = db_path.exists()
    if not already_existed:
        agent_dir.mkdir(parents=True, exist_ok=True)
        creds = _run_inception(agent_dir, name)
        did = creds.agent_did

    rookery.agents[name] = LocalAgentConfig(
        data_dir=Path("agent_data") / name,
        port=port,
        autostart=autostart,
    )
    rookery.save(rookery_path)

    return CreateAgentResult(
        name=name,
        did=did,
        port=port,
        autostart=autostart,
        db_path=db_path,
        already_existed=already_existed,
    )


def _next_free_port(rookery: RookeryConfig) -> int:
    used = {rookery.host.port}
    for cfg in rookery.get_local_agents().values():
        used.add(cfg.port)
    port = max(
        DEFAULT_AGENT_START_PORT,
        max((cfg.port for cfg in rookery.get_local_agents().values()), default=DEFAULT_AGENT_START_PORT - 1) + 1,
    )
    while port in used:
        port += 1
    return port


def run(ctx: SetupContext) -> None:
    """Make sure at least one agent exists in the rookery."""
    rookery_path = ctx.project_dir / ROOKERY_CONFIG_FILENAME
    rookery = RookeryConfig.load(rookery_path)

    existing = rookery.get_local_agents()
    if existing and ctx.flow in (Flow.QUICKSTART, Flow.CHECK):
        # In quickstart, the presence of any agent is enough.
        if ctx.flow is Flow.CHECK and not existing:
            ctx.block("No agents in rookery — run `kestrel setup agent`")
        return

    if existing and ctx.flow is Flow.INTERACTIVE:
        wants_more = ctx.prompter.confirm(
            f"Rookery already has {len(existing)} agent(s) "
            f"({', '.join(existing.keys())}). Add another?",
            default=False,
        )
        if not wants_more:
            return

    if ctx.flow is Flow.CHECK:
        if not existing:
            ctx.block("No agents in rookery — run `kestrel setup agent`")
        return

    name = _prompt_name(ctx, existing)
    if not name:
        ctx.block("Agent name not provided — skipped")
        return

    autostart = _prompt_autostart(ctx)

    if not (ctx.agent_data_root / name / "kestrel_prime.db").exists():
        ctx.prompter.info(f"Running inception for '{name}' — generating DID + DB...")
    try:
        result = create_agent(
            name=name,
            project_dir=ctx.project_dir,
            agent_data_root=ctx.agent_data_root,
            autostart=autostart,
        )
    except Exception as exc:  # noqa: BLE001 — surface inception failures verbatim
        ctx.block(f"Inception failed for '{name}': {exc}")
        return

    if result.already_existed:
        ctx.record(f"Agent '{name}' already existed; rookery row refreshed")
    else:
        ctx.record(f"Created agent '{name}' with DID {result.did}")
    ctx.record(
        f"Registered '{name}' in rookery.toml on port {result.port} "
        f"({'autostart' if result.autostart else 'manual'})"
    )


def _prompt_name(ctx: SetupContext, existing: dict) -> str:
    """Pick an agent name. Quickstart picks 'Kestrel' if available, else suffixes."""
    if ctx.flow is Flow.QUICKSTART:
        candidate = "Kestrel"
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"Kestrel{suffix}"
        return candidate

    while True:
        name = ctx.prompter.text(
            "Agent name", default="Kestrel" if "Kestrel" not in existing else ""
        ).strip()
        if not name:
            return ""
        if name in existing:
            ctx.prompter.info(f"'{name}' is already in the rookery — pick another.")
            continue
        if not name.replace("_", "").replace("-", "").isalnum():
            ctx.prompter.info("Names must be alphanumeric (with optional _ or -).")
            continue
        return name


def _prompt_autostart(ctx: SetupContext) -> bool:
    """Quickstart → autostart on. Interactive → ask."""
    if ctx.flow is Flow.QUICKSTART:
        return True
    return ctx.prompter.confirm(
        "Autostart this agent when `kestrel start` runs?", default=True
    )


def _run_inception(agent_dir: Path, name: str):
    """Call into inception_service. Imported lazily — heavy module."""
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    agent_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        create_kestrel_identity_async(
            output_dir=str(agent_dir),
            agent_name=name,
        )
    )
