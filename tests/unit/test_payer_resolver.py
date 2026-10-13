"""Unit tests for FoundationPayerResolver.

Phase 2c of the PayerPolicy foundation work.

Coverage:
- HOST_ENV → returns enabled ResolvedResource with a working
  KeyResolutionService.
- NONE → ResolvedResource.disabled() with no resolver.
- Unsupported combinations from the matrix raise
  UnsupportedCombinationError BEFORE any side effects.
- Phase-deferred kinds (HOST_MASTER_PROVISIONED, USER_MASTER_PROVISIONED,
  SPONSOR, SELF_WALLET) raise NotImplementedError with a clear message.
- Empty agent_did rejected.
- Each ResourceClass slot in PayerPolicy is dispatched to the right
  PayerSpec.
- load_policy_from_toml returns host_env_default when no kestrel.toml
  [payments] section exists.
"""
from __future__ import annotations

import base64
from typing import Iterator

import pytest
import pytest_asyncio

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerSpec,
    ResolvedResource,
    ResourceClass,
    UnsupportedCombinationError,
)

from kestrel_sovereign.security.exceptions import (
    DecryptionError,
    KeyNotConfiguredError as StoredKeyNotConfiguredError,
)
from kestrel_sovereign.security.service_key_storage import (
    ServiceKeyStorage,
    StoredKeyCiphertextMissingError,
)
from kestrel_sovereign.services.key_resolution import (
    KeyResolutionService,
    KeyUnreadableError,
    resolve_key,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.services.payer_resolver import (
    FoundationPayerResolver,
    load_policy_from_toml,
)


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv(
        "KESTREL_DATA_KEY",
        "test-master-key-32-bytes-fixed--",
    )
    yield


def _all_host_env(
    *, llm_kind: PayerKind = PayerKind.HOST_ENV
) -> PayerPolicy:
    """Build a policy that's host_env everywhere except optionally LLM."""
    policy = PayerPolicy.host_env_default()
    if llm_kind is PayerKind.HOST_ENV:
        return policy
    # Replace the LLM slot with the requested kind, using vendor that
    # matches the matrix for that kind.
    return PayerPolicy(
        llm=PayerSpec(vendor="openrouter", kind=llm_kind),
        storage=policy.storage,
        compute=policy.compute,
        tools=policy.tools,
        comms=policy.comms,
    )


class TestHostEnvResolution:
    @pytest.mark.asyncio
    async def test_host_env_returns_enabled_with_resolver(self) -> None:
        resolver = FoundationPayerResolver(PayerPolicy.host_env_default())
        result = await resolver.resolve_for("did:test:agent-a", ResourceClass.LLM)
        assert isinstance(result, ResolvedResource)
        assert result.enabled is True
        assert isinstance(result.key_resolver, KeyResolutionService)

    @pytest.mark.asyncio
    async def test_host_env_resolver_falls_back_to_env_var(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-or-from-env")
        resolver = FoundationPayerResolver(PayerPolicy.host_env_default())
        result = await resolver.resolve_for("did:test:agent-a", ResourceClass.LLM)
        # The resolver returned by HOST_ENV is a real KeyResolutionService
        # with no agent storage configured (db=None) — env var is the
        # only source.
        key = await result.key_resolver.resolve_key("openrouter", require=False)
        assert key == "sk-test-or-from-env"


class TestNoneResolution:
    @pytest.mark.asyncio
    async def test_none_returns_disabled(self) -> None:
        # Build a policy with llm = NONE.
        policy = PayerPolicy(
            llm=PayerSpec(vendor="openrouter", kind=PayerKind.NONE),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        result = await resolver.resolve_for("did:test:agent-a", ResourceClass.LLM)
        assert result.enabled is False
        assert result.key_resolver is None


class TestUnsupportedCombinations:
    @pytest.mark.asyncio
    async def test_unknown_vendor_raises_unsupported(self) -> None:
        # An LLM vendor not in the matrix and with no wildcard fallback
        # for the LLM resource class. status_for returns NOT_IMPLEMENTED.
        policy = PayerPolicy(
            llm=PayerSpec(vendor="bogus-vendor-xyz", kind=PayerKind.HOST_ENV),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        with pytest.raises(UnsupportedCombinationError) as excinfo:
            await resolver.resolve_for("did:test:agent-a", ResourceClass.LLM)
        assert excinfo.value.vendor == "bogus-vendor-xyz"


class TestEnabledKindsReturnResolver:
    """Phase 3a expanded the set of kinds that return an enabled
    ResolvedResource: HOST_ENV plus all three delegated-master kinds
    (HOST_MASTER_PROVISIONED, USER_MASTER_PROVISIONED, SPONSOR). They
    share the same agent-side surface; the difference is provisioning
    side-effects (Phase 3c for HOST_MASTER, 3.5 for SELF_WALLET). Until
    the side-effects land, the agent-init layer detects pre-existing
    per-agent credentials via the deprecated openrouter_key_hash
    metadata field and calls use_agent_key uniformly.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kind",
        [
            PayerKind.HOST_MASTER_PROVISIONED,
            PayerKind.USER_MASTER_PROVISIONED,
            PayerKind.SPONSOR,
        ],
    )
    async def test_enabled_kind_returns_resolver(
        self, kind: PayerKind
    ) -> None:
        spec_kwargs = {"vendor": "openrouter", "kind": kind}
        if kind in (PayerKind.USER_MASTER_PROVISIONED, PayerKind.SPONSOR):
            spec_kwargs["master_did"] = "did:test:master"
        policy = PayerPolicy(
            llm=PayerSpec(**spec_kwargs),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        result = await resolver.resolve_for(
            "did:test:agent-a", ResourceClass.LLM
        )
        assert result.enabled is True
        assert result.key_resolver is not None


class TestDelegatedStorageNotImplemented:
    """Delegated-master kinds (HOST_MASTER_PROVISIONED, USER_MASTER_PROVISIONED,
    SPONSOR) are LLM-only in Phase 3a. For storage they raise
    NotImplementedError until Phase 3.5 ships the Lighthouse wallet-signed
    key minting flow.

    This is the regression guard for codex Phase 3a round 2: returning
    enabled for delegated storage kinds would let LighthouseProvider
    fall through to LIGHTHOUSE_API_KEY env var, billing the operator's
    master key as the agent's storage — the exact policy violation
    these kinds exist to prevent.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kind",
        [
            PayerKind.HOST_MASTER_PROVISIONED,
            PayerKind.USER_MASTER_PROVISIONED,
            PayerKind.SPONSOR,
        ],
    )
    async def test_delegated_storage_raises(self, kind: PayerKind) -> None:
        spec_kwargs = {"vendor": "lighthouse", "kind": kind}
        if kind in (PayerKind.USER_MASTER_PROVISIONED, PayerKind.SPONSOR):
            spec_kwargs["master_did"] = "did:test:master"
        policy = PayerPolicy(
            llm=PayerSpec(vendor="openrouter", kind=PayerKind.HOST_ENV),
            storage=PayerSpec(**spec_kwargs),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        # Two paths can raise here, both correct:
        # (a) Primary: SDK SUPPORT_MATRIX marks the triple
        #     NOT_IMPLEMENTED → matrix-validation gate raises
        #     UnsupportedCombinationError. This is the steady-state
        #     path once the SDK 0.12.0+ pin is in effect.
        # (b) Defense-in-depth: even if a stale SDK ships a regressed
        #     matrix that says READY, the resolver still gates
        #     delegated-master kinds for non-LLM resources and raises
        #     NotImplementedError. Codex round 4 of Phase 3a insisted
        #     on the belt-and-suspenders pattern.
        # Either error type satisfies the test; what matters is that
        # the resolver does NOT silently return enabled.
        with pytest.raises(
            (UnsupportedCombinationError, NotImplementedError)
        ):
            await resolver.resolve_for(
                "did:test:agent-a", ResourceClass.STORAGE
            )


class TestSelfWalletDeferred:
    """SELF_WALLET for LLM is explicitly deferred per the support matrix.
    The resolver raises NotImplementedError there. SELF_WALLET for
    storage (lighthouse) lands in Phase 3.5.
    """

    @pytest.mark.asyncio
    async def test_self_wallet_for_llm_deferred(self) -> None:
        # The matrix marks (LLM, openrouter, SELF_WALLET) as
        # NOT_IMPLEMENTED, so resolve_for raises
        # UnsupportedCombinationError BEFORE reaching the
        # NotImplementedError for the kind. Both are valid surfaces;
        # both indicate "this combination is not offered."
        policy = PayerPolicy(
            llm=PayerSpec(vendor="openrouter", kind=PayerKind.SELF_WALLET),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        from kestrel_sdk.payer_policy import UnsupportedCombinationError
        with pytest.raises((NotImplementedError, UnsupportedCombinationError)):
            await resolver.resolve_for("did:test:agent-a", ResourceClass.LLM)


class TestArgValidation:
    @pytest.mark.asyncio
    async def test_empty_agent_did_raises(self) -> None:
        resolver = FoundationPayerResolver(PayerPolicy.host_env_default())
        with pytest.raises(ValueError):
            await resolver.resolve_for("", ResourceClass.LLM)


class TestResourceClassDispatch:
    @pytest.mark.asyncio
    async def test_each_slot_dispatches_to_its_spec(self) -> None:
        # Each ResourceClass should resolve under its own slot's spec,
        # not someone else's. Build a policy where each slot has a
        # detectably different shape (different vendor strings).
        policy = PayerPolicy(
            llm=PayerSpec(vendor="openrouter", kind=PayerKind.HOST_ENV),
            storage=PayerSpec(vendor="lighthouse", kind=PayerKind.HOST_ENV),
            compute=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            tools=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
            comms=PayerSpec(vendor="*", kind=PayerKind.HOST_ENV),
        )
        resolver = FoundationPayerResolver(policy)
        for rc in ResourceClass:
            result = await resolver.resolve_for("did:test:agent-a", rc)
            # All slots are HOST_ENV in this fixture so all return enabled.
            assert result.enabled is True


class TestLoadPolicyFromToml:
    def test_no_section_returns_host_env_default(
        self, monkeypatch, tmp_path
    ) -> None:
        # Point cwd at a tmp dir with no kestrel.toml; load_section returns
        # an empty dict, and we expect the host_env_default fallback.
        monkeypatch.chdir(tmp_path)
        # Also clear any KESTREL_PROJECT_DIR override, if such exists.
        monkeypatch.delenv("KESTREL_PROJECT_DIR", raising=False)
        policy = load_policy_from_toml()
        assert policy == PayerPolicy.host_env_default()


# How an agent's stored key becomes unreadable, and what its lookup raises:
# a ciphertext that does not decrypt (a damaged row or another
# KESTREL_DATA_KEY), or an active row whose ciphertext is empty.
UNREADABLE = {
    "undecryptable": (
        base64.b64encode(b"not a ciphertext").decode(),
        DecryptionError,
    ),
    "empty": ("", StoredKeyCiphertextMissingError),
}


async def _make_unreadable(
    db, agent_did: str, provider: str, how: str = "undecryptable"
) -> None:
    """The agent's stored key stops being readable, as ``UNREADABLE[how]``."""
    await db.execute(
        "UPDATE agent_service_keys SET encrypted_key = ? "
        "WHERE agent_did = ? AND provider_id = ?",
        (UNREADABLE[how][0], agent_did, provider),
    )


class TestStrictStorageResolution:
    """``strict_storage`` separates a key the agent holds but cannot read from
    one it does not hold. Only the absent key falls back to the environment;
    without the flag every storage failure still does (#2874)."""

    @pytest_asyncio.fixture
    async def db(self, tmp_path):
        database = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
        yield database
        await database.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", sorted(UNREADABLE))
    async def test_a_non_lighthouse_caller_still_falls_back_on_a_storage_error(
        self, db, monkeypatch, how
    ) -> None:
        agent_did = "did:test:agent-llm"
        await ServiceKeyStorage(db, agent_did).store_key("openrouter", "sk-or-agent")
        await _make_unreadable(db, agent_did, "openrouter", how)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-host")
        result = await FoundationPayerResolver(
            PayerPolicy.host_env_default(), db=db
        ).resolve_for(agent_did, ResourceClass.LLM)

        assert await result.key_resolver.resolve_key("openrouter", require=False) == (
            "sk-or-host"
        )
        assert await resolve_key(
            "openrouter", storage=ServiceKeyStorage(db, agent_did), require=False
        ) == "sk-or-host"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("how", sorted(UNREADABLE))
    async def test_strictly_an_unreadable_key_raises_instead_of_the_hosts(
        self, db, monkeypatch, how
    ) -> None:
        agent_did = "did:test:agent-storage"
        await ServiceKeyStorage(db, agent_did).store_key("lighthouse", "lh-agent")
        await _make_unreadable(db, agent_did, "lighthouse", how)
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "lh-host")
        result = await FoundationPayerResolver(
            PayerPolicy.host_env_default(), db=db
        ).resolve_for(agent_did, ResourceClass.STORAGE)

        with pytest.raises(KeyUnreadableError, match="cannot be read") as raised:
            await result.key_resolver.resolve_key(
                "lighthouse", require=False, strict_storage=True
            )
        assert raised.value.provider == "lighthouse"
        assert isinstance(raised.value.__cause__, UNREADABLE[how][1])
        with pytest.raises(KeyUnreadableError):
            await resolve_key(
                "lighthouse",
                storage=ServiceKeyStorage(db, agent_did),
                require=False,
                strict_storage=True,
            )
        # Without the flag the same lookup still falls back.
        assert await result.key_resolver.resolve_key("lighthouse", require=False) == (
            "lh-host"
        )

    @pytest.mark.asyncio
    async def test_strictly_an_absent_key_still_falls_back_to_the_host(
        self, db, monkeypatch
    ) -> None:
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "lh-host")
        result = await FoundationPayerResolver(
            PayerPolicy.host_env_default(), db=db
        ).resolve_for("did:test:agent-without-key", ResourceClass.STORAGE)

        assert await result.key_resolver.resolve_key(
            "lighthouse", require=False, strict_storage=True
        ) == "lh-host"

    @pytest.mark.asyncio
    async def test_an_empty_key_row_still_reads_as_no_key_to_direct_callers(
        self, db
    ) -> None:
        """A caller of ``ServiceKeyStorage.get_key`` that catches the storage
        layer's ``KeyNotConfiguredError`` (the LLM service's agent-key
        activation, the keys feature) still reads an empty row as no key. Only
        the type it raises tells the damaged row from the absent one."""
        agent_did = "did:test:agent-direct"
        storage = ServiceKeyStorage(db, agent_did)
        with pytest.raises(StoredKeyNotConfiguredError) as absent:
            await storage.get_key("openrouter")
        assert not isinstance(absent.value, StoredKeyCiphertextMissingError)

        await storage.store_key("openrouter", "sk-or-agent")
        await _make_unreadable(db, agent_did, "openrouter", "empty")
        with pytest.raises(StoredKeyNotConfiguredError) as empty:
            await storage.get_key("openrouter")
        assert isinstance(empty.value, StoredKeyCiphertextMissingError)

        # A removed key is absent, not damaged.
        await storage.deactivate_key(provider_id="openrouter")
        with pytest.raises(StoredKeyNotConfiguredError) as removed:
            await storage.get_key("openrouter")
        assert not isinstance(removed.value, StoredKeyCiphertextMissingError)
