"""Create (or recognise) a sovereign agent.

This step is the wizard's bridge to the existing
:func:`kestrel_sovereign.inception_service.create_kestrel_identity_async`.
We do not duplicate inception logic — we just collect the inputs
(name + port + autostart) and call it.

Idempotence rules:

  - If an agent directory already contains ``kestrel_prime.db``, we do
    not re-incept; we just make sure multi_agent.toml lists it.
  - Port collisions with existing multi_agent rows trigger a fresh probe
    starting at :data:`DEFAULT_AGENT_START_PORT`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from dataclasses import dataclass

from kestrel_sovereign.constitution.emancipation import (
    EmancipationConfigError,
    EmancipationContract,
    parse_emancipation_block,
)
from kestrel_sovereign.multi_agent.config import (
    DEFAULT_AGENT_START_PORT,
    LocalAgentConfig,
    MULTI_AGENT_CONFIG_FILENAME,
    MultiAgentConfig,
)
from kestrel_sovereign.setup.context import Flow, SetupContext
from kestrel_sovereign.setup.toml_file import read_toml

logger = logging.getLogger(__name__)

DEFAULT_QUICKSTART_AGENT_NAME = "Kestrel"


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
    emancipation_contract: EmancipationContract | None = None,
    is_test_instance: bool = False,
    genesis_auditor=None,
    genesis_audit_provenance: str | None = None,
) -> CreateAgentResult:
    """Idempotent agent creation: incept if needed, then register in multi_agent.

    Used by both the wizard's ``agent`` step and ``kestrel create``. If
    the agent's ``kestrel_prime.db`` already exists, inception is skipped
    and the existing agent is just (re-)registered with the requested
    port/autostart.

    ``is_test_instance`` propagates to
    :func:`kestrel_sovereign.inception_service.create_kestrel_identity_async`
    so the new agent is tagged with ``is_test_instance=True`` and an
    auto-generated ``test_cycle_id`` on its properties node. Ignored when
    the agent already exists (no re-inception).
    """
    multi_agent_path = project_dir / MULTI_AGENT_CONFIG_FILENAME
    multi_agent = MultiAgentConfig.load(multi_agent_path)

    if port is None:
        existing_entry = multi_agent.agents.get(name)
        if existing_entry is not None:
            port = existing_entry.port
        else:
            port = _next_free_port(multi_agent)

    agent_dir = agent_data_root / name
    db_path = agent_dir / "kestrel_prime.db"

    did: str | None = None
    already_existed = db_path.exists()
    if not already_existed:
        agent_dir.mkdir(parents=True, exist_ok=True)
        if genesis_auditor is None and _configured_genesis_auditor_available(
            project_dir
        ):
            genesis_auditor = _build_configured_genesis_auditor(project_dir)
            genesis_audit_provenance = "setup:configured_llm"
        creds = _run_inception(
            agent_dir,
            name,
            emancipation_contract,
            is_test_instance=is_test_instance,
            genesis_auditor=genesis_auditor,
            genesis_audit_provenance=genesis_audit_provenance,
        )
        did = creds.agent_did

    multi_agent.agents[name] = LocalAgentConfig(
        data_dir=Path("agent_data") / name,
        port=port,
        autostart=autostart,
    )
    multi_agent.save(multi_agent_path)

    return CreateAgentResult(
        name=name,
        did=did,
        port=port,
        autostart=autostart,
        db_path=db_path,
        already_existed=already_existed,
    )


def _next_free_port(multi_agent: MultiAgentConfig) -> int:
    used = {multi_agent.host.port}
    for cfg in multi_agent.get_local_agents().values():
        used.add(cfg.port)
    port = max(
        DEFAULT_AGENT_START_PORT,
        max((cfg.port for cfg in multi_agent.get_local_agents().values()), default=DEFAULT_AGENT_START_PORT - 1) + 1,
    )
    while port in used:
        port += 1
    return port


def run(ctx: SetupContext) -> None:
    """Make sure at least one agent exists in the multi_agent."""
    multi_agent_path = ctx.project_dir / MULTI_AGENT_CONFIG_FILENAME
    multi_agent = MultiAgentConfig.load(multi_agent_path)

    existing = multi_agent.get_local_agents()
    if existing and ctx.flow in (Flow.QUICKSTART, Flow.CHECK):
        # In quickstart, the presence of any agent is enough.
        if ctx.flow is Flow.CHECK and not existing:
            ctx.block("No agents in multi_agent — run `kestrel setup agent`")
        return

    if existing and ctx.flow is Flow.INTERACTIVE:
        wants_more = ctx.prompter.confirm(
            f"MultiAgent already has {len(existing)} agent(s) "
            f"({', '.join(existing.keys())}). Add another?",
            default=False,
        )
        if not wants_more:
            return

    if ctx.flow is Flow.CHECK:
        if not existing:
            ctx.block("No agents in multi_agent — run `kestrel setup agent`")
        return

    name = _prompt_name(ctx, existing)
    if not name:
        ctx.block("Agent name not provided — skipped")
        return

    autostart = _prompt_autostart(ctx)

    contract = _load_emancipation_contract(ctx)
    if ctx.blockers and any("[emancipation]" in b for b in ctx.blockers):
        # Invalid block already surfaced as a blocker; stop before
        # inception so we never anchor a malformed contract.
        return

    if not _ensure_did_web_domain(ctx):
        return

    incept_needed = not (ctx.agent_data_root / name / "kestrel_prime.db").exists()
    if incept_needed:
        # Centralize custody at the *actual* inception boundary (#2468): resolve,
        # persist (round-trip-verified) and export the one effective
        # KESTREL_DATA_KEY right before inception, so encrypt-key == persist-key
        # regardless of how we got here. This closes two gaps the keys step alone
        # could not:
        #   * a single-step ``kestrel setup agent`` skips the keys step entirely;
        #   * ``--reset`` moves ``.env`` aside at wizard start, so a key resolved
        #     *before* reset would be lost — resolving here (after reset) persists
        #     the effective key that the immediate next boot will actually load.
        # Never incept an identity we would encrypt with the wrong — or no — key.
        from kestrel_sovereign.setup.steps.keys import ensure_effective_data_key

        _, custody_conflict, _, _ = ensure_effective_data_key(
            ctx.env_path, generate_if_missing=True
        )
        if custody_conflict:
            ctx.block(
                f"Skipped inception for '{name}': {custody_conflict}"
            )
            return

    if incept_needed:
        suffix = " (test instance)" if ctx.is_test_instance else ""
        ctx.prompter.info(
            f"Running inception for '{name}'{suffix} — generating DID + DB..."
        )
    try:
        result = create_agent(
            name=name,
            project_dir=ctx.project_dir,
            agent_data_root=ctx.agent_data_root,
            autostart=autostart,
            emancipation_contract=contract,
            is_test_instance=ctx.is_test_instance,
        )
    except Exception as exc:  # noqa: BLE001 — surface inception failures verbatim
        ctx.block(f"Inception failed for '{name}': {exc}")
        return

    if result.already_existed:
        ctx.record(f"Agent '{name}' already existed; multi_agent row refreshed")
    else:
        test_suffix = " [test instance]" if ctx.is_test_instance else ""
        ctx.record(f"Created agent '{name}' with DID {result.did}{test_suffix}")
    ctx.record(
        f"Registered '{name}' in multi_agent.toml on port {result.port} "
        f"({'autostart' if result.autostart else 'manual'})"
    )


def _prompt_name(ctx: SetupContext, existing: dict) -> str:
    """Pick an agent name. Quickstart picks 'Kestrel' if available, else suffixes."""
    if ctx.flow is Flow.QUICKSTART:
        candidate = DEFAULT_QUICKSTART_AGENT_NAME
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"{DEFAULT_QUICKSTART_AGENT_NAME}{suffix}"
        return candidate

    while True:
        name = ctx.prompter.text(
            "Agent name",
            default=(
                DEFAULT_QUICKSTART_AGENT_NAME
                if DEFAULT_QUICKSTART_AGENT_NAME not in existing
                else ""
            ),
        ).strip()
        if not name:
            return ""
        if name in existing:
            ctx.prompter.info(f"'{name}' is already in the multi_agent — pick another.")
            continue
        if not name.replace("_", "").replace("-", "").isalnum():
            ctx.prompter.info("Names must be alphanumeric (with optional _ or -).")
            continue
        return name


def _ensure_did_web_domain(ctx: SetupContext) -> bool:
    """Born-hybrid inception (#2397) needs a did:web domain. Resolve it
    before inception so the wizard surfaces a clear blocker (or prompt)
    instead of a mid-inception traceback.

    Returns True when inception can proceed (domain available, or the
    operator explicitly configured the classical did:pkh method).
    """
    import os

    from kestrel_sovereign.inception_service import (
        DID_WEB_DOMAIN_ENV,
        IDENTITY_METHOD_DID_WEB,
        resolve_identity_method,
    )

    try:
        method = resolve_identity_method(None)
    except ValueError as exc:
        ctx.block(str(exc))
        return False
    if method != IDENTITY_METHOD_DID_WEB or os.environ.get(DID_WEB_DOMAIN_ENV):
        return True

    if ctx.flow is Flow.QUICKSTART:
        # Quickstart's contract is a zero-config LOCAL bootstrap. A
        # fresh user has no domain yet, and a classical fallback would
        # mint the quantum-vulnerable identity this epic eliminates —
        # so default the did:web domain to "localhost" (spec-legal,
        # unmistakably local, unique per agent via the slug's entropy
        # suffix). Verification is local-custody anchored either way;
        # when a real domain arrives, the rotation ceremony migrates
        # the agent to it with full succession continuity.
        from kestrel_sovereign.setup.env_file import write_env

        os.environ[DID_WEB_DOMAIN_ENV] = "localhost"
        write_env(ctx.env_path, {DID_WEB_DOMAIN_ENV: "localhost"})
        ctx.record(
            f"Quickstart: {DID_WEB_DOMAIN_ENV} defaulted to 'localhost' — "
            f"the agent's did:web identity is local-only. Set a real "
            f"domain in .env and run a rotation ceremony to publish it."
        )
        return True

    if ctx.flow is Flow.INTERACTIVE:
        domain = ctx.prompter.text(
            "did:web domain for new agents' DID documents "
            "(e.g. agents.example.com; published at "
            "https://<domain>/<agent>/did.json)",
            default="",
        ).strip()
        if domain:
            from kestrel_sovereign.setup.env_file import write_env

            os.environ[DID_WEB_DOMAIN_ENV] = domain
            write_env(ctx.env_path, {DID_WEB_DOMAIN_ENV: domain})
            ctx.record(f"Saved {DID_WEB_DOMAIN_ENV}={domain} to {ctx.env_path}")
            return True

    ctx.block(
        f"Born-hybrid inception requires {DID_WEB_DOMAIN_ENV} (the domain "
        f"for new agents' did:web DID documents). Set it in .env, or set "
        f"KESTREL_IDENTITY_METHOD=did:pkh to mint classical identities."
    )
    return False


def _prompt_autostart(ctx: SetupContext) -> bool:
    """Quickstart → autostart on. Interactive → ask."""
    if ctx.flow is Flow.QUICKSTART:
        return True
    return ctx.prompter.confirm(
        "Autostart this agent when `kestrel start` runs?", default=True
    )


def _run_inception(
    agent_dir: Path,
    name: str,
    emancipation_contract: EmancipationContract | None,
    *,
    is_test_instance: bool = False,
    genesis_auditor=None,
    genesis_audit_provenance: str | None = None,
):
    """Call into inception_service. Imported lazily — heavy module."""
    from kestrel_sovereign.inception_service import create_kestrel_identity_async

    agent_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        create_kestrel_identity_async(
            output_dir=str(agent_dir),
            agent_name=name,
            emancipation_contract=emancipation_contract,
            is_test_instance=is_test_instance,
            genesis_auditor=genesis_auditor,
            genesis_audit_provenance=genesis_audit_provenance,
        )
    )


def _configured_genesis_auditor_available(project_dir: Path) -> bool:
    """Return whether the configured route set has a usable audit lane.

    A declared route is not enough: the stock config declares Ollama even on
    hosts where it is absent. Cloud routes require a persisted/exported
    credential; Ollama must answer its normal reachability probe. If neither is
    currently usable, inception records ``pending`` and first cognition remains
    gated until the operator configures one.
    """
    import os

    from kestrel_sovereign.setup.env_file import read_env
    from kestrel_sovereign.setup.steps import llm as llm_step

    config = read_toml(project_dir / "kestrel.toml")
    llm_config = config.get("llm") or {}
    vendors = llm_config.get("vendors") or {}
    persisted_env = read_env(project_dir / ".env")

    for route_id in llm_config.get("route_priority", []) or []:
        vendor_key, separator, route_key = str(route_id).partition(":")
        if not separator:
            continue
        route = (
            ((vendors.get(vendor_key) or {}).get("routes") or {}).get(route_key)
            or {}
        )
        accepted = llm_step.accepted_credential_envs(route_id, route)
        if accepted:
            if any(os.environ.get(name) or persisted_env.get(name) for name in accepted):
                return True
            continue
        adapter = str(route.get("adapter", ""))
        if adapter == "OllamaAdapter":
            if llm_step._is_ollama_reachable(
                str(route.get("host") or "http://localhost:11434")
            ):
                return True
            continue
        # A credential-free non-Ollama route is explicitly configured and does
        # not expose a generic reachability probe; let the attempted audit be
        # the authoritative fail-closed check.
        if route:
            return True
    return False


def _build_configured_genesis_auditor(project_dir: Path):
    """Build a one-shot auditor using the target home's configured LLM lane."""

    async def _audit(prompt: str):
        # The setup wizard may have written a credential to .env moments ago.
        # Load that target deliberately; inception_service itself must remain
        # free of import-time/current-directory dotenv behavior (#2468).
        import os

        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sovereign.paths import load_project_env, reset_cache

        load_project_env(project_dir)
        # LLMService resolves kestrel.toml during its synchronous constructor.
        # Pin that instant to the explicit create_agent target, then restore the
        # host environment before the first await so concurrent agents cannot
        # observe a process-global KESTREL_HOME detour.
        previous_home = os.environ.get("KESTREL_HOME")
        os.environ["KESTREL_HOME"] = str(project_dir)
        reset_cache()
        try:
            service = LLMService()
        finally:
            if previous_home is None:
                os.environ.pop("KESTREL_HOME", None)
            else:
                os.environ["KESTREL_HOME"] = previous_home
            reset_cache()
        try:
            return await service.get_audit_response(prompt)
        finally:
            await service.close()

    return _audit


def _load_emancipation_contract(ctx: SetupContext) -> EmancipationContract | None:
    """Read ``[emancipation]`` from kestrel.toml and return the parsed contract.

    Returns None if the block is absent (dormant by omission) **or** if
    the block fails validation. In the failure case a blocker is
    recorded so the wizard reports the problem and ``run`` aborts before
    inception — never raises, so the caller can continue without a
    try/except wrapper around this call.
    """
    if not ctx.kestrel_toml_path.exists():
        return None
    try:
        data = read_toml(ctx.kestrel_toml_path)
        return parse_emancipation_block(data)
    except EmancipationConfigError as exc:
        ctx.block(
            f"[emancipation] block in kestrel.toml is invalid: {exc}. "
            f"Inception aborted to avoid anchoring an unsigned contract."
        )
        return None
