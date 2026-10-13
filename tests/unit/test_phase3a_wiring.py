"""Unit tests for Phase 3a — Lighthouse + LLM resolver wiring.

These tests pin the contracts the agent-init layer relies on:

- `LighthouseProvider.__init__(api_key=None, key_resolver=...)` does NOT
  silently consult `LIGHTHOUSE_API_KEY` at construction time when a
  resolver is supplied. The resolver must be the single credential
  source, so a `PayerPolicy` slot of `NONE` actually means NONE even
  on a host where the env var happens to be set.

- `LighthouseProvider.__init__(api_key=None, key_resolver=None)`
  preserves the standalone env-var fallback (today's behavior).

- `LLMService` carries a `disabled` flag (default False) that the
  agent-init layer sets to True when the LLM policy is NONE. Phase 3b
  adds the `_check_policy()` guard that reads this flag on every
  generation entry point; Phase 3a only plumbs the flag.
"""
from __future__ import annotations

from typing import Iterator

import pytest

from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.storage.providers.lighthouse_provider import (
    LighthouseProvider,
)


@pytest.fixture(autouse=True)
def _kestrel_data_key(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv(
        "KESTREL_DATA_KEY",
        "test-master-key-32-bytes-fixed--",
    )
    yield


class TestLighthouseConstructorContract:
    def test_no_args_falls_back_to_env_var(self, monkeypatch) -> None:
        # Standalone path: no api_key, no resolver → env var is consulted.
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "lh-from-env")
        provider = LighthouseProvider()
        assert provider.api_key == "lh-from-env"

    def test_explicit_api_key_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "lh-from-env")
        provider = LighthouseProvider(api_key="lh-explicit")
        assert provider.api_key == "lh-explicit"

    def test_resolver_supplied_does_not_seed_from_env(self, monkeypatch) -> None:
        # Critical: when a resolver is in charge, the constructor must
        # NOT silently consult os.environ. Otherwise a `PayerPolicy`
        # slot of NONE on a host with `LIGHTHOUSE_API_KEY` set would
        # still construct an env-keyed provider.
        monkeypatch.setenv("LIGHTHOUSE_API_KEY", "lh-from-env")

        # A sentinel resolver — its identity isn't material here; we're
        # only asserting the constructor does NOT fall back to os.environ
        # when ANY resolver is supplied.
        sentinel = object()
        provider = LighthouseProvider(api_key=None, key_resolver=sentinel)
        assert provider.api_key is None
        assert provider._key_resolver is sentinel
        # is_available reflects "no key captured at construction" which
        # is exactly right — _ensure_client() is the place that consults
        # the resolver, not __init__.
        assert provider.is_available() is False


class _Resolver:
    """A key resolver whose answers a test sets one call at a time."""

    def __init__(self, *answers):
        self._answers = list(answers)

    async def resolve_key(self, provider, require=True, *, strict_storage=False):
        # Strictly: a key the agent holds but cannot read is never read past
        # to the host's key (#2874).
        assert (provider, require, strict_storage) == ("lighthouse", False, True)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Client:
    def __init__(self, api_key=None, gateway_url=None):
        self.api_key = api_key
        self.closed = False

    async def close(self):
        self.closed = True


class TestLighthouseResolverOwnsTheKey:
    """With a resolver, the key it resolves now is the only key the provider
    calls Lighthouse with; a removed key is not used again (#2874)."""

    @pytest.fixture(autouse=True)
    def _client(self, monkeypatch):
        monkeypatch.setattr(
            "kestrel_sovereign.storage.providers.lighthouse_provider.LighthouseRestClient",
            _Client,
        )

    async def test_a_key_the_resolver_stops_supplying_is_dropped(self, tmp_path):
        provider = LighthouseProvider(
            cache_dir=str(tmp_path), key_resolver=_Resolver("old", None, "new")
        )

        assert await provider.resolve_api_key() == "old"
        old_client = provider._client
        assert await provider.resolve_api_key() is None
        assert old_client.closed
        assert (provider.api_key, provider.is_available()) == (None, False)
        # A key that resolves again is used again.
        assert await provider.resolve_api_key() == "new"
        assert provider._client.api_key == "new"

    async def test_a_resolver_failure_raises_and_drops_the_key(self, tmp_path):
        provider = LighthouseProvider(
            cache_dir=str(tmp_path),
            key_resolver=_Resolver("old", RuntimeError("key store unreadable")),
        )
        assert await provider.resolve_api_key() == "old"
        old_client = provider._client

        with pytest.raises(RuntimeError, match="key store unreadable"):
            await provider.publish(b"x", filename="f", tag="t")

        assert old_client.closed
        assert (provider.api_key, provider.is_available()) == (None, False)

    async def test_without_a_resolver_the_constructed_key_is_used(self, tmp_path):
        provider = LighthouseProvider(api_key="given", cache_dir=str(tmp_path))

        assert await provider.resolve_api_key() == "given"
        assert provider.is_available()


class TestLLMServiceDisabledFlag:
    def test_disabled_default_is_false(self) -> None:
        svc = LLMService()
        assert svc.disabled is False

    def test_disabled_is_settable(self) -> None:
        svc = LLMService()
        svc.disabled = True
        assert svc.disabled is True

    def test_disabled_is_per_instance(self) -> None:
        # Two services don't share state — same per-agent invariant the
        # attach_to_agent contract enforces.
        a = LLMService()
        b = LLMService()
        a.disabled = True
        assert a.disabled is True
        assert b.disabled is False
