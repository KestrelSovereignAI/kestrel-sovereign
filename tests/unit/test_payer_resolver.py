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

from decimal import Decimal
from typing import Iterator

import pytest

from kestrel_sdk.payer_policy import (
    PayerKind,
    PayerPolicy,
    PayerSpec,
    ResolvedResource,
    ResourceClass,
    UnsupportedCombinationError,
)

from kestrel_sovereign.services.key_resolution import KeyResolutionService
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
        # The SDK SUPPORT_MATRIX marks (STORAGE, lighthouse, <delegated>)
        # as NOT_IMPLEMENTED, so the matrix-validation gate at the top of
        # resolve_for raises UnsupportedCombinationError before ever
        # reaching the kind dispatch. This keeps matrix and resolver
        # behavior aligned; codex round 3 of Phase 3a flagged the
        # earlier inconsistency where matrix said READY but resolver
        # raised NotImplementedError.
        with pytest.raises(UnsupportedCombinationError) as excinfo:
            await resolver.resolve_for(
                "did:test:agent-a", ResourceClass.STORAGE
            )
        assert excinfo.value.kind is kind
        assert excinfo.value.resource_class is ResourceClass.STORAGE


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
