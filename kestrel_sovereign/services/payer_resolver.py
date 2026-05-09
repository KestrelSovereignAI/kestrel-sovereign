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

import logging
from typing import TYPE_CHECKING, Optional

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
        # env-var fallback. No side effects required at this point.
        if spec.kind is PayerKind.HOST_ENV:
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
