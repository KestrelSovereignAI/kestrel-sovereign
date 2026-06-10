"""
PayerResolver — concrete foundation impl of `kestrel_sdk.payer_policy.PayerResolver`.

Resolves an agent's `PayerPolicy` slot to a credential resolver at
agent-init time. Side effects (provisioning a child credential,
signing a wallet auth message, persisting into ServiceKeyStorage)
happen here so providers downstream stay simple.

Implemented paths:

- ``HOST_ENV`` — wraps today's `KeyResolutionService` and returns it.
  This is the back-compat path that preserves every standalone
  Kestrel deployment's existing behavior.
- ``NONE`` — returns ``ResolvedResource.disabled()``. The agent-init
  layer (Phase 3 of the PayerPolicy plan) is required to skip
  provider construction entirely for this slot — returning a sentinel
  resolver is insufficient against constructor-time env-var
  fallbacks. Plan v11 spells this out for the Lighthouse path
  specifically.
- ``HOST_MASTER_PROVISIONED`` for OpenRouter LLM — mints a per-agent
  child key under the host's master OpenRouter key.
- ``SELF_WALLET`` for Lighthouse storage — signs Lighthouse's wallet
  auth message with the agent's secp256k1 key, creates an API key, and
  stores it in ServiceKeyStorage.

Unsupported combinations raise before side effects. The setup wizard
reads the same SUPPORT_MATRIX the resolver consults, so an operator
never picks a path the resolver cannot honor.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

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


class _GraphNodeVanishedError(Exception):
    """Internal sentinel: graph_nodes agent row was present at
    pre-flight but missing by the time _persist_openrouter_key_hash
    ran. The caller (mint loop) catches this to revoke the
    just-minted remote key and surface a PayerPolicyError to the
    operator.
    """

    def __init__(self, agent_did: str) -> None:
        super().__init__(
            f"graph_nodes row for agent {agent_did[:30]}... "
            "vanished between pre-flight and persist"
        )
        self.agent_did = agent_did


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
        host_db: Optional["AsyncDatabase"] = None,
        wallet_private_key: Optional[Any] = None,
    ) -> None:
        """
        Args:
            db: Agent's own database. Used for ServiceKeyStorage
                (per-agent child credentials) and for graph_nodes
                metadata writes during minting.
            host_db: Shared host-level database (one per Kestrel
                deployment) that holds operator-master credentials in
                HostKeyStorage. The wizard's payments step persists
                masters here; the resolver reads them here at mint
                time. If None, falls back to ``db`` (single-DB tests
                and standalone deployments where one DB serves both
                roles).
        """
        self._policy = policy
        self._db = db
        # Fall back to db if host_db not explicitly provided. In
        # production, kestrel_agent.py wires both with distinct DBs
        # (agent's vs the shared host.db); in tests one db often
        # serves both roles.
        self._host_db = host_db if host_db is not None else db
        self._wallet_private_key = wallet_private_key

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
        # provisioned" path; minting requires payer-wallet custody and
        # consent for host/user/sponsor wallets). The SDK
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
                "resources need payer-wallet custody/consent). The SDK support matrix "
                "should already mark this combination NOT_IMPLEMENTED — "
                "if you're reaching this gate, the matrix is stale or "
                "the SDK pin is too loose."
            )

        # Side-effect for delegated-master OpenRouter LLM: mint a per-agent
        # child key against the funding master (host's, or a user's for
        # USER_MASTER_PROVISIONED) if one doesn't already exist. Idempotent.
        # The master source is chosen by spec.kind in _fetch_openrouter_master.
        if (
            spec.kind in (
                PayerKind.HOST_MASTER_PROVISIONED,
                PayerKind.USER_MASTER_PROVISIONED,
            )
            and resource_class is ResourceClass.LLM
            and spec.vendor == "openrouter"
        ):
            await self._maybe_mint_openrouter_child(agent_did, spec)

        if (
            spec.kind is PayerKind.SELF_WALLET
            and resource_class is ResourceClass.STORAGE
            and spec.vendor == "lighthouse"
        ):
            await self._maybe_mint_lighthouse_self_wallet_key(agent_did)
            logger.debug(
                f"PayerPolicy.{resource_class.value}: SELF_WALLET for agent "
                f"{agent_did[:30]}..."
            )
            from kestrel_sovereign.security.service_key_storage import (
                ServiceKeyStorage,
            )

            storage = ServiceKeyStorage(self._db, agent_did)
            resolver = KeyResolutionService(storage=storage, agent_did=agent_did)
            return ResolvedResource(enabled=True, key_resolver=resolver)

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

        # All other kinds are stubbed.
        # We raise here rather than silently degrading because a wizard
        # that respects SUPPORT_MATRIX should never land us here.
        raise NotImplementedError(
            f"PayerKind.{spec.kind.name} resolution for "
            f"({resource_class.value}, vendor={spec.vendor!r}) is not yet "
            "implemented in the foundation resolver. See "
            "docs/architecture/PAYER_POLICY_FOUNDATION.md."
        )

    async def _maybe_mint_lighthouse_self_wallet_key(self, agent_did: str) -> None:
        """Mint/store an agent-scoped Lighthouse key via SELF_WALLET.

        Lighthouse authenticates API-key creation by asking an EVM wallet
        to sign a challenge. Kestrel agents already have a secp256k1 key
        for the legacy did:pkh identity, so SELF_WALLET signs with that
        key and stores the resulting Lighthouse API key in
        ServiceKeyStorage under the agent DID.
        """
        if self._db is None:
            from kestrel_sdk.payer_policy import PayerPolicyError

            raise PayerPolicyError(
                "PayerPolicy.storage.kind = SELF_WALLET for lighthouse, "
                "but no agent database was provided. The minted Lighthouse "
                "API key would have nowhere to be stored."
            )
        if self._wallet_private_key is None:
            from kestrel_sdk.payer_policy import PayerPolicyError

            raise PayerPolicyError(
                "PayerPolicy.storage.kind = SELF_WALLET for lighthouse, "
                "but the agent's secp256k1 wallet private key is unavailable."
            )

        lock = self._GLOBAL_MINT_LOCKS.setdefault(
            f"{agent_did}:storage:lighthouse", asyncio.Lock()
        )
        async with lock:
            await self._maybe_mint_lighthouse_self_wallet_key_locked(agent_did)

    async def _maybe_mint_lighthouse_self_wallet_key_locked(
        self, agent_did: str
    ) -> None:
        from kestrel_sovereign.security.service_key_storage import (
            ServiceKeyStorage,
        )
        from kestrel_sovereign.storage.providers.lighthouse_rest import (
            LighthouseRestClient,
        )

        agent_storage = ServiceKeyStorage(self._db, agent_did)
        if await agent_storage.has_key("lighthouse"):
            logger.debug(
                f"PayerResolver: agent {agent_did[:30]}... already has "
                "Lighthouse key; skipping wallet auth."
            )
            return

        address = self._evm_address_from_private_key()
        client = LighthouseRestClient(api_key="")
        try:
            message = await client.get_auth_message(address)
            signature = self._sign_eth_message(message)
            api_key = await client.create_api_key(address, signature)
        finally:
            await client.close()

        await agent_storage.store_key(
            provider_id="lighthouse",
            api_key=api_key,
        )
        logger.info(
            f"PayerResolver: minted Lighthouse SELF_WALLET key for agent "
            f"{agent_did[:30]}... using wallet {address[:10]}..."
        )

    def _evm_address_from_private_key(self) -> str:
        account = self._eth_account_from_private_key()
        return str(account.address)

    def _sign_eth_message(self, message: str) -> str:
        try:
            from eth_account.messages import encode_defunct
        except ImportError as exc:
            from kestrel_sdk.payer_policy import PayerPolicyError

            raise PayerPolicyError(
                "Lighthouse SELF_WALLET requires eth-account. Install the "
                "wallet extra before using PayerPolicy.storage.self_wallet."
            ) from exc

        account = self._eth_account_from_private_key()
        signed = account.sign_message(encode_defunct(text=message))
        signature = signed.signature.hex()
        return signature if signature.startswith("0x") else f"0x{signature}"

    def _eth_account_from_private_key(self) -> Any:
        try:
            from eth_account import Account
        except ImportError as exc:
            from kestrel_sdk.payer_policy import PayerPolicyError

            raise PayerPolicyError(
                "Lighthouse SELF_WALLET requires eth-account. Install the "
                "wallet extra before using PayerPolicy.storage.self_wallet."
            ) from exc

        private_key = self._wallet_private_key
        if isinstance(private_key, str):
            key_hex = private_key if private_key.startswith("0x") else f"0x{private_key}"
        elif isinstance(private_key, bytes):
            key_hex = f"0x{private_key.hex()}"
        elif hasattr(private_key, "private_numbers"):
            key_int = private_key.private_numbers().private_value
            key_hex = f"0x{key_int.to_bytes(32, 'big').hex()}"
        else:
            from kestrel_sdk.payer_policy import PayerPolicyError

            raise PayerPolicyError(
                "Unsupported Lighthouse SELF_WALLET private key type: "
                f"{type(private_key).__name__}"
            )
        return Account.from_key(key_hex)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _maybe_mint_openrouter_child(
        self,
        agent_did: str,
        spec: PayerSpec,
    ) -> None:
        """Mint a per-agent OpenRouter child key under the funding master,
        if the agent doesn't already have one. Idempotent.

        Called from resolve_for on (LLM, openrouter) for the delegated-master
        kinds HOST_MASTER_PROVISIONED and USER_MASTER_PROVISIONED. Reads the
        funding master (host's, or the user's keyed by spec.master_did) via
        _fetch_openrouter_master, calls
        OpenRouterProvisioningService.create_agent_key with the agent's
        DID and the policy's monthly_cap_usd, stores the resulting child
        key in ServiceKeyStorage. Subsequent agent inits and
        LLMService.use_agent_key calls find it there.

        Raises:
            PayerPolicyError: If the relevant master key is not configured
                (host master via the setup wizard, or the user master the
                funding user must provision).
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

        # PRE-FLIGHT: validate the graph_nodes agent row exists BEFORE
        # any side effects (mint + local store). Codex Phase 3c round 3
        # finding: previously the missing-row check ran AFTER mint/store,
        # so a failed persist left the agent with a working local key
        # but no retirement-visible hash; subsequent inits would skip
        # mint entirely (has_key=True) and the hash would never get
        # written. Pre-flight makes that race impossible.
        if not await self._agent_graph_node_exists(agent_did):
            from kestrel_sdk.payer_policy import PayerPolicyError
            raise PayerPolicyError(
                f"PayerResolver: cannot mint OpenRouter child key for "
                f"agent {agent_did[:30]}... — no graph_nodes row exists. "
                "retirement_service reads openrouter_key_hash from "
                "graph_nodes.properties only; without a row, the remote "
                "OpenRouter child key would leak at retirement. Inception "
                "is expected to create the graph_nodes row before agent "
                "init reaches this resolver."
            )

        # Per-agent lock prevents two concurrent agent-init paths from
        # both passing has_key() and both calling create_agent_key —
        # which would orphan one of the two remote OpenRouter keys.
        # Class-level so the lock is shared across resolver instances
        # within the same process.
        lock = self._GLOBAL_MINT_LOCKS.setdefault(agent_did, asyncio.Lock())
        async with lock:
            await self._maybe_mint_openrouter_child_locked(agent_did, spec)

    async def _agent_graph_node_exists(self, agent_did: str) -> bool:
        """True iff a graph_nodes row exists for this agent_did."""
        rows = await self._db.fetchall(
            "SELECT 1 FROM graph_nodes WHERE node_id = ? LIMIT 1",
            (agent_did,),
        )
        return bool(rows)

    async def _fetch_openrouter_master(self, spec: PayerSpec) -> str:
        """Resolve the OpenRouter master key for a delegated-master mint.

        HOST_MASTER_PROVISIONED reads the operator's master from
        ``HostKeyStorage``; USER_MASTER_PROVISIONED reads the funding user's
        master from ``UserMasterKeyStorage`` (keyed by ``spec.master_did``).
        Both live in the shared ``host_db``. Raises ``PayerPolicyError`` if the
        relevant master is not configured.
        """
        from kestrel_sdk.payer_policy import PayerPolicyError

        if spec.kind is PayerKind.USER_MASTER_PROVISIONED:
            from kestrel_sovereign.security.user_master_key_storage import (
                UserMasterKeyStorage,
            )

            storage = UserMasterKeyStorage(self._host_db, spec.master_did)
            if not await storage.has_key("openrouter"):
                raise PayerPolicyError(
                    "PayerPolicy.llm.kind = USER_MASTER_PROVISIONED for "
                    f"openrouter (master_did={spec.master_did[:30]}...), but no "
                    "user master key is configured in UserMasterKeyStorage. The "
                    "user must provision their OpenRouter master key before the "
                    "agent mints a child against it."
                )
            return await storage.get_key("openrouter")

        # HOST_MASTER_PROVISIONED (the default delegated-master path)
        from kestrel_sovereign.security.host_key_storage import HostKeyStorage

        storage = HostKeyStorage(self._host_db)
        if not await storage.has_key("openrouter"):
            raise PayerPolicyError(
                "PayerPolicy.llm.kind = HOST_MASTER_PROVISIONED for openrouter, "
                "but no host master key is configured in HostKeyStorage. "
                "Run the setup wizard or use scripts/manage_openrouter_keys.py "
                "to provision the operator's master key before agent init."
            )
        return await storage.get_key("openrouter")

    async def _maybe_mint_openrouter_child_locked(
        self,
        agent_did: str,
        spec: PayerSpec,
    ) -> None:
        """Inside the per-agent mint lock — see _maybe_mint_openrouter_child."""
        # Late imports: keep module-level deps minimal so this resolver
        # is importable even on deployments that haven't installed the
        # OpenRouter provisioning surface (which depends on httpx).
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

        # Resolve the funding master from the shared host_db: the host's
        # master for HOST_MASTER_PROVISIONED, or the user's master (keyed by
        # spec.master_did) for USER_MASTER_PROVISIONED. Raises PayerPolicyError
        # if the relevant master is not configured.
        master_key = await self._fetch_openrouter_master(spec)

        # Mint the child remotely. Keep the provisioning service open
        # past create_agent_key so we can revoke if the local persist
        # fails.
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

            # Persist the key_hash to graph_nodes.properties FIRST,
            # before storing locally. If the graph_nodes row vanished
            # between pre-flight and now (concurrent delete during
            # init), we revoke the remote child immediately and raise.
            # Local store happens only on success — so a failed mint
            # leaves no inconsistent state: no local key, no remote
            # key, retry sees has_key=False and starts fresh.
            try:
                await self._persist_openrouter_key_hash(
                    agent_did, key_info.key_hash, require_row=True
                )
            except _GraphNodeVanishedError:
                # Revoke the remote key so it doesn't leak. Use a
                # narrow except so unexpected exceptions still propagate.
                logger.error(
                    f"PayerResolver: graph_nodes row for agent "
                    f"{agent_did[:30]}... vanished mid-mint; revoking "
                    f"remote OpenRouter key (hash {key_info.key_hash[:16]}...)"
                )
                try:
                    await provisioning.delete_key(key_info.key_hash)
                except Exception as revoke_err:
                    # Revoke failed — the remote key is now orphaned.
                    # Surface clearly so an operator can clean up via
                    # scripts/manage_openrouter_keys.py delete --hash ...
                    logger.error(
                        f"PayerResolver: revoke FAILED for orphaned key "
                        f"{key_info.key_hash[:16]}...: {revoke_err}. "
                        "Manual cleanup required: "
                        "scripts/manage_openrouter_keys.py delete --hash "
                        f"{key_info.key_hash}"
                    )
                from kestrel_sdk.payer_policy import PayerPolicyError
                raise PayerPolicyError(
                    f"PayerResolver: graph_nodes row for agent "
                    f"{agent_did[:30]}... disappeared mid-mint. Remote "
                    "child key revoked (or revoke logged for manual "
                    "cleanup). No local state was changed; retry is safe."
                )

            # Persist locally. Subsequent use_agent_key calls find it
            # via key_storage.get_key("openrouter").
            await agent_storage.store_key(
                provider_id="openrouter",
                api_key=key_info.key,
            )
        finally:
            await provisioning.close()

        logger.info(
            f"PayerResolver: minted OpenRouter child key for agent "
            f"{agent_did[:30]}... (hash {key_info.key_hash[:16]}..., "
            f"limit ${limit_usd:.2f}/mo)"
        )

    async def _persist_openrouter_key_hash(
        self,
        agent_did: str,
        key_hash: str,
        *,
        require_row: bool = False,
    ) -> None:
        """Write openrouter_key_hash to graph_nodes.properties.

        Args:
            require_row: When True, raise _GraphNodeVanishedError if
                no graph_nodes row exists (treats the missing-row case
                as a hard failure so the caller can revoke the remote
                key). When False, log and return (used by callers that
                are tolerant of the row being absent).
        """
        import json

        rows = await self._db.fetchall(
            "SELECT properties FROM graph_nodes WHERE node_id = ? LIMIT 1",
            (agent_did,),
        )
        if not rows:
            if require_row:
                raise _GraphNodeVanishedError(agent_did)
            logger.error(
                f"PayerResolver: graph_nodes row not found for agent "
                f"{agent_did[:30]}...; openrouter_key_hash NOT persisted."
            )
            return

        properties_json = rows[0][0]
        properties = json.loads(properties_json) if properties_json else {}
        properties["openrouter_key_hash"] = key_hash
        # AsyncDatabase.execute returns cursor.rowcount on both backends.
        # If the row was deleted between SELECT above and this UPDATE
        # (concurrent retirement), rowcount is 0 — that's the same leak
        # shape as the missing-row case codex round 4 closed at the
        # SELECT, so treat it the same way.
        rows_affected = await self._db.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
            (json.dumps(properties), agent_did),
        )
        if rows_affected == 0:
            if require_row:
                raise _GraphNodeVanishedError(agent_did)
            logger.error(
                f"PayerResolver: graph_nodes row vanished between SELECT "
                f"and UPDATE for agent {agent_did[:30]}...; "
                "openrouter_key_hash NOT persisted."
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


def _find_host_db(start: "Path") -> Optional["Path"]:
    """Walk up from ``start`` looking for an ``agent_data`` directory
    that contains a ``host.db`` file. Returns the first such
    ``host.db`` path, or None.

    Handles both layouts:
    - Multi-agent: ``<project>/agent_data/<name>/kestrel_prime.db``
    - Flat:       ``<project>/agent_data/<name>.db``

    And both symlink directions:
    - Case A: ``<project>/agent_data`` is a symlink to a mount.
      Lexical walk finds 'agent_data' segment; resolved walk would
      lose it.
    - Case B: storage_path is a symlink alias outside agent_data that
      points into ``<project>/agent_data``. Lexical walk fails (no
      'agent_data' segment); resolved walk wins.
    - Case C: alias path itself has an 'agent_data' segment that's
      NOT the real root (e.g. ``~/agent_data/current.db`` symlink
      target ``<project>/agent_data/<name>.db``). Both walks find an
      'agent_data' but only one has the host.db. Returning the FIRST
      ancestor named 'agent_data' would pick the wrong one — instead
      we keep walking until host.db is actually present, across both
      candidates.

    Stops at filesystem root for each candidate.
    """
    from pathlib import Path

    candidates = [Path(start).absolute()]
    try:
        resolved = Path(start).resolve()
        if resolved != candidates[0]:
            candidates.append(resolved)
    except (OSError, RuntimeError):
        # resolve() can raise on broken symlinks or recursion. Skip
        # the resolved fallback rather than failing the whole walk.
        pass

    for cur in candidates:
        if cur.is_file() or not cur.exists():
            cur = cur.parent
        while cur != cur.parent:
            if cur.name == "agent_data":
                candidate_host_db = cur / "host.db"
                if candidate_host_db.exists():
                    return candidate_host_db
            cur = cur.parent
    return None


async def open_host_db(
    *,
    storage_path: "Path | str | None" = None,
    agent_data_dir: "Path | None" = None,
    project_dir: "Path | None" = None,
) -> Optional["AsyncDatabase"]:
    """Open the deployment-wide host database for HostKeyStorage.

    Lives at ``<agent_data_dir>/host.db`` by convention (the same path
    the wizard's payments step writes to). Returns None if the file
    doesn't exist — the resolver falls back to the agent's own db,
    which is the correct behavior for deployments that have never run
    the payments wizard.

    Args:
        storage_path: An agent's storage path. Preferred when called
            from KestrelAgent.initialize() — we walk up looking for
            ``agent_data``, which handles both the multi-agent layout
            (``<project>/agent_data/<name>/kestrel_prime.db``) and the
            flat layout (``<project>/agent_data/<name>.db``).
        agent_data_dir: Explicit path to the deployment's agent_data
            directory. Used by callers that already know where it is
            (e.g. SetupContext.agent_data_root).
        project_dir: Project root; host.db is at ``<project_dir>/agent_data/host.db``.
            Defaults to cwd when none of the above are given.
    """
    from pathlib import Path
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    host_db_path: Optional[Path]
    if storage_path is not None:
        # _find_host_db walks both lexical and resolved paths, and
        # only returns an agent_data ancestor that actually contains
        # a host.db. Returns None if none of the candidates yields one.
        host_db_path = _find_host_db(Path(storage_path))
    elif agent_data_dir is not None:
        candidate = Path(agent_data_dir) / "host.db"
        host_db_path = candidate if candidate.exists() else None
    else:
        root = project_dir if project_dir is not None else Path.cwd()
        candidate = root / "agent_data" / "host.db"
        host_db_path = candidate if candidate.exists() else None

    if host_db_path is None:
        return None
    return await AsyncDatabase.sqlite(str(host_db_path))


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
