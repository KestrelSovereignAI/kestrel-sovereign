"""
PayerResolver — concrete foundation impl of `kestrel_sdk.payer_policy.PayerResolver`.

Resolves an agent's `PayerPolicy` slot to a credential resolver at
agent-init time. Side effects (provisioning a child credential,
signing a wallet auth message, persisting into ServiceKeyStorage)
happen here so providers downstream stay simple.

This Phase 2 implementation ships the two simplest paths:

- ``HOST_ENV`` — wraps today's `KeyResolutionService` and returns it.
  This is the back-compat path that preserves every standalone
  Kestrel deployment's existing behavior.
- ``NONE`` — returns ``ResolvedResource.disabled()``. The agent-init
  layer (Phase 3 of the PayerPolicy plan) is required to skip
  provider construction entirely for this slot — returning a sentinel
  resolver is insufficient against constructor-time env-var
  fallbacks. Plan v11 spells this out for the Lighthouse path
  specifically.

All other `PayerKind` values (``HOST_MASTER_PROVISIONED``,
``USER_MASTER_PROVISIONED``, ``SPONSOR``, ``SELF_WALLET``) raise
`NotImplementedError` with a clear message naming the (resource_class,
vendor, kind) triple. Phase 3 fills in the OpenRouter provisioning and
Phase 3.5 the Lighthouse wallet-signed key flow. The wizard step in
Phase 4 reads the same SUPPORT_MATRIX the resolver consults, so an
operator never picks a path the resolver cannot honor.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, Optional

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerSpec,
    ResolvedResource,
    ResourceClass,
    SUPPORT_MATRIX,
    SupportStatus,
    UnsupportedCombinationError,
    is_offerable,
    status_for,
)

from kestrel_sovereign.services.key_resolution import KeyResolutionService

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


class FoundationPayerResolver:
    """Concrete `PayerResolver` for standalone Kestrel deployments.

    Constructor takes the policy explicitly so tests and the wizard
    can pass synthesized policies without depending on kestrel.toml.
    Production callers (kestrel_agent.py, Phase 3) load the policy
    from kestrel.toml `[payments]` via ``load_policy_from_toml`` and
    pass the result in.

    The optional ``db`` is the agent's own database; when provided,
    HOST_ENV resolution wires up agent-scoped ServiceKeyStorage so
    per-agent credentials override the env-var fallback (today's
    behavior). When absent, only the env-var path is consulted.
    """

    # Process-global per-agent mint locks. CLASS-LEVEL (not instance)
    # because kestrel_agent.py constructs a fresh FoundationPayerResolver
    # on each agent init — two concurrent inits for the same DID would
    # each get their own empty per-instance lock map, defeating the
    # serialization. Class-level shares the locks across resolver
    # instances within a single Python process. (Cross-process
    # concurrency would require DB-level enforcement; that's a future
    # follow-up if/when multi-process agent init becomes a thing.)
    _GLOBAL_MINT_LOCKS: Dict[str, asyncio.Lock] = {}

    def __init__(
        self,
        policy: PayerPolicy,
        *,
        db: Optional["AsyncDatabase"] = None,
    ) -> None:
        self._policy = policy
        self._db = db

    # ------------------------------------------------------------------
    # Public surface — matches kestrel_sdk.payer_policy.PayerResolver
    # ------------------------------------------------------------------

    async def resolve_for(
        self,
        agent_did: str,
        resource_class: ResourceClass,
    ) -> ResolvedResource:
        if not agent_did:
            raise ValueError("agent_did is required for resolve_for")

        spec = self._spec_for(resource_class)

        # Hard-fail on unsupported combinations BEFORE attempting any
        # side effects. The wizard refuses to write these in the first
        # place; tripping here means the policy was hand-edited or
        # somehow constructed past validation.
        status = status_for(resource_class, spec.vendor, spec.kind)
        if status is not SupportStatus.READY:
            raise UnsupportedCombinationError(
                resource_class=resource_class,
                vendor=spec.vendor,
                kind=spec.kind,
                status=status,
            )

        # NONE: explicit disabled. Agent-init layer must skip provider
        # construction entirely (see plan v11 §"NONE wiring").
        if spec.kind is PayerKind.NONE:
            logger.debug(
                f"PayerPolicy.{resource_class.value}: NONE for agent "
                f"{agent_did[:30]}..."
            )
            return ResolvedResource.disabled()

        # HOST_ENV: today's behavior — agent's ServiceKeyStorage with
        # env-var fallback. Always safe; no side effects required.
        #
        # HOST_MASTER_PROVISIONED / USER_MASTER_PROVISIONED / SPONSOR:
        # share the same agent-side surface in Phase 3a for LLM only.
        # The agent-init layer detects pre-existing per-agent credentials
        # via the deprecated `openrouter_key_hash` metadata field and
        # calls use_agent_key. Phase 3c switches to resolver-driven
        # minting for LLM when that field is absent.
        #
        # For STORAGE / COMPUTE / TOOLS / COMMS, delegated-master kinds
        # are NOT yet wired (no analogous "child credential already
        # provisioned" path; minting requires Phase 3.5+). The SDK
        # SUPPORT_MATRIX marks those triples as NOT_IMPLEMENTED so the
        # matrix-validation gate above raises
        # UnsupportedCombinationError before reaching this code. We
        # ALSO keep an explicit defense-in-depth check here so a stale
        # or accidentally-regressed SDK matrix cannot silently fall
        # through to LIGHTHOUSE_API_KEY env var (which would bill the
        # operator's master key as the agent's storage — exact policy
        # violation these kinds exist to prevent).
        delegated_master_kinds = {
            PayerKind.HOST_MASTER_PROVISIONED,
            PayerKind.USER_MASTER_PROVISIONED,
            PayerKind.SPONSOR,
        }
        if (
            spec.kind in delegated_master_kinds
            and resource_class is not ResourceClass.LLM
        ):
            raise NotImplementedError(
                f"PayerKind.{spec.kind.name} resolution for "
                f"({resource_class.value}, vendor={spec.vendor!r}) is not "
                "yet implemented (delegated-master kinds for non-LLM "
                "resources land in Phase 3.5+). The SDK support matrix "
                "should already mark this combination NOT_IMPLEMENTED — "
                "if you're reaching this gate, the matrix is stale or "
                "the SDK pin is too loose."
            )

        # Phase 3c: side-effect for HOST_MASTER_PROVISIONED on OpenRouter.
        # Mint a per-agent child key against the host's master OpenRouter
        # key if one doesn't already exist. Idempotent.
        if (
            spec.kind is PayerKind.HOST_MASTER_PROVISIONED
            and resource_class is ResourceClass.LLM
            and spec.vendor == "openrouter"
        ):
            await self._maybe_mint_openrouter_child(agent_did, spec)

        if spec.kind is PayerKind.HOST_ENV or spec.kind in delegated_master_kinds:
            logger.debug(
                f"PayerPolicy.{resource_class.value}: HOST_ENV for agent "
                f"{agent_did[:30]}..."
            )
            storage = None
            if self._db is not None:
                # Late import to avoid a hard dependency at module import time;
                # ServiceKeyStorage construction can fail if KESTREL_DATA_KEY
                # is misconfigured, and the wizard's verify-step is the right
                # place to surface that, not import time.
                from kestrel_sovereign.security.service_key_storage import (
                    ServiceKeyStorage,
                )
                from kestrel_sovereign.security.exceptions import (
                    MasterKeyNotConfiguredError,
                )
                try:
                    storage = ServiceKeyStorage(self._db, agent_did)
                except MasterKeyNotConfiguredError as e:
                    logger.warning(
                        f"ServiceKeyStorage not available for "
                        f"{agent_did[:30]}...: {e}"
                    )
            resolver = KeyResolutionService(storage=storage, agent_did=agent_did)
            return ResolvedResource(enabled=True, key_resolver=resolver)

        # All other kinds are stubbed in Phase 2; Phase 3 / 3.5 fill them in.
        # We raise here rather than silently degrading because a wizard
        # that respects SUPPORT_MATRIX should never land us here.
        raise NotImplementedError(
            f"PayerKind.{spec.kind.name} resolution for "
            f"({resource_class.value}, vendor={spec.vendor!r}) is not yet "
            "implemented in the foundation resolver. Phase 3 of the "
            "PayerPolicy plan adds HOST_MASTER_PROVISIONED for OpenRouter; "
            "Phase 3.5 adds Lighthouse SELF_WALLET. See "
            "docs/architecture/PAYER_POLICY_FOUNDATION.md."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _maybe_mint_openrouter_child(
        self,
        agent_did: str,
        spec: PayerSpec,
    ) -> None:
        """Mint a per-agent OpenRouter child key under the host master,
        if the agent doesn't already have one. Idempotent.

        Called from resolve_for on (LLM, openrouter, HOST_MASTER_PROVISIONED).
        Reads the host master key from HostKeyStorage, calls
        OpenRouterProvisioningService.create_agent_key with the agent's
        DID and the policy's monthly_cap_usd, stores the resulting child
        key in ServiceKeyStorage. Subsequent agent inits and
        LLMService.use_agent_key calls find it there.

        Raises:
            PayerPolicyError: If the host master key is not configured
                (operator must run setup wizard or provision manually
                via scripts/manage_openrouter_keys.py).
            OpenRouterProvisioningError: If the OpenRouter API call fails
                (rate limited, network error, invalid master, etc.).
        """
        if self._db is None:
            # No agent storage to check or write to. Caller (agent-init
            # layer) should have provided db; skipping here is safer than
            # silently using the host master directly.
            logger.warning(
                "PayerResolver: _maybe_mint_openrouter_child called without "
                "db; skipping mint. Per-agent key isolation degraded."
            )
            return

        # Per-agent lock prevents two concurrent agent-init paths from
        # both passing has_key() and both calling create_agent_key —
        # which would orphan one of the two remote OpenRouter keys.
        # Class-level so the lock is shared across resolver instances
        # within the same process.
        lock = self._GLOBAL_MINT_LOCKS.setdefault(agent_did, asyncio.Lock())
        async with lock:
            await self._maybe_mint_openrouter_child_locked(agent_did, spec)

    async def _maybe_mint_openrouter_child_locked(
        self,
        agent_did: str,
        spec: PayerSpec,
    ) -> None:
        """Inside the per-agent mint lock — see _maybe_mint_openrouter_child."""
        # Late imports: keep module-level deps minimal so this resolver
        # is importable even on deployments that haven't installed the
        # OpenRouter provisioning surface (which depends on httpx).
        from kestrel_sovereign.security.host_key_storage import HostKeyStorage
        from kestrel_sovereign.security.service_key_storage import (
            ServiceKeyStorage,
        )
        from kestrel_sovereign.features.llm_keys.openrouter_provisioning import (
            OpenRouterProvisioningService,
        )

        # Idempotent: skip if agent already has an OpenRouter key.
        # Re-checked under the lock so a concurrent path that minted
        # while we were waiting is honored.
        agent_storage = ServiceKeyStorage(self._db, agent_did)
        if await agent_storage.has_key("openrouter"):
            logger.debug(
                f"PayerResolver: agent {agent_did[:30]}... already has "
                "OpenRouter key; skipping mint."
            )
            return

        # Look up the host master.
        host_storage = HostKeyStorage(self._db)
        if not await host_storage.has_key("openrouter"):
            from kestrel_sdk.payer_policy import PayerPolicyError
            raise PayerPolicyError(
                "PayerPolicy.llm.kind = HOST_MASTER_PROVISIONED for openrouter, "
                "but no host master key is configured in HostKeyStorage. "
                "Run the setup wizard or use scripts/manage_openrouter_keys.py "
                "to provision the operator's master key before agent init."
            )
        master_key = await host_storage.get_key("openrouter")

        # Mint the child.
        provisioning = OpenRouterProvisioningService(management_key=master_key)
        try:
            # Per-agent monthly cap from the policy spec; default $100/mo
            # mirrors the deprecated provision_agent_openrouter.py default.
            limit_usd = float(spec.monthly_cap_usd) if spec.monthly_cap_usd is not None else 100.0
            key_info = await provisioning.create_agent_key(
                agent_name=agent_did,
                limit_usd=limit_usd,
                limit_reset="monthly",
            )
        finally:
            await provisioning.close()

        # Persist the child in ServiceKeyStorage. Subsequent
        # use_agent_key calls find it via key_storage.get_key("openrouter").
        await agent_storage.store_key(
            provider_id="openrouter",
            api_key=key_info.key,
        )

        # Persist the key_hash to graph_nodes.properties.openrouter_key_hash
        # so retirement_service.py (which reads from there via
        # agent_info.get("openrouter_key_hash")) can revoke this child key
        # when the agent retires. Without this, resolver-minted keys
        # leak: ServiceKeyStorage forgets them on retirement, but
        # OpenRouter still bills them.
        await self._persist_openrouter_key_hash(agent_did, key_info.key_hash)

        logger.info(
            f"PayerResolver: minted OpenRouter child key for agent "
            f"{agent_did[:30]}... (hash {key_info.key_hash[:16]}..., "
            f"limit ${limit_usd:.2f}/mo)"
        )

    async def _persist_openrouter_key_hash(
        self,
        agent_did: str,
        key_hash: str,
    ) -> None:
        """Write openrouter_key_hash to graph_nodes.properties for the
        agent. Mirrors scripts/provision_agent_openrouter.py:119-138 so
        retirement_service.py can revoke the key on agent retirement.

        graph_nodes.properties is the SOLE retirement-readable location;
        retirement_service.get_agent_info() reads ONLY from there. An
        earlier draft fell back to agent_metadata when no graph_nodes
        agent row existed, but codex round 2 caught that retirement
        wouldn't see those entries — leaking the remote OpenRouter
        child key on retirement. Now we fail loudly instead: the
        precondition (graph_nodes row exists for this agent) is created
        by inception_service before any agent-init path reaches this
        resolver, so a missing row indicates a deeper inconsistency.

        Idempotent: overwrites if already present.

        Raises:
            PayerPolicyError: If no graph_nodes agent row exists for
                this agent_did. The mint call must NOT have been made
                in that state, but the remote child key was already
                created — the caller surfaces the error so an operator
                can investigate.
        """
        import json

        # Look up the agent's graph node and current properties.
        rows = await self._db.fetchall(
            "SELECT properties FROM graph_nodes WHERE node_id = ? LIMIT 1",
            (agent_did,),
        )
        if not rows:
            from kestrel_sdk.payer_policy import PayerPolicyError
            raise PayerPolicyError(
                f"PayerResolver: cannot persist openrouter_key_hash for "
                f"agent {agent_did[:30]}... — no graph_nodes row exists. "
                "retirement_service reads the hash from graph_nodes.properties "
                "only; without a row, the remote OpenRouter child key would "
                "leak at retirement. Inception is expected to create the "
                "graph_nodes row before agent init reaches this resolver."
            )

        properties_json = rows[0][0]
        properties = json.loads(properties_json) if properties_json else {}
        properties["openrouter_key_hash"] = key_hash
        await self._db.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
            (json.dumps(properties), agent_did),
        )

    def _spec_for(self, resource_class: ResourceClass) -> PayerSpec:
        if resource_class is ResourceClass.LLM:
            return self._policy.llm
        if resource_class is ResourceClass.STORAGE:
            return self._policy.storage
        if resource_class is ResourceClass.COMPUTE:
            return self._policy.compute
        if resource_class is ResourceClass.TOOLS:
            return self._policy.tools
        if resource_class is ResourceClass.COMMS:
            return self._policy.comms
        # ResourceClass is a closed StrEnum, so this is unreachable
        # unless someone adds a member without updating this dispatch.
        raise ValueError(f"unknown ResourceClass: {resource_class!r}")


# ----------------------------------------------------------------------
# Policy loader
# ----------------------------------------------------------------------


def load_policy_from_toml() -> PayerPolicy:
    """Load PayerPolicy from kestrel.toml ``[payments]`` table.

    Returns ``PayerPolicy.host_env_default()`` if the section is
    missing or empty. This is the back-compat path: deployments that
    have never run the payments wizard see today's behavior unchanged.

    Raises:
        pydantic.ValidationError: If the ``[payments]`` table is
            present but malformed (unknown keys, master_did mismatch,
            etc.). Surfaces during agent init / wizard verification.
    """
    from kestrel_sovereign.config import load_section

    section = load_section("payments")
    if not section:
        return PayerPolicy.host_env_default()
    return PayerPolicy.from_toml_section(section)


__all__ = [
    "FoundationPayerResolver",
    "load_policy_from_toml",
]
