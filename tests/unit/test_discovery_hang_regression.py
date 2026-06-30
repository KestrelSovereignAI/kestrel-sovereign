"""Regression tests for the post-#1110 discovery wedge.

Three independent bugs reported the day after #1110 merged:

1. **Provider model never resolved on fresh setup.**
   Routes are seeded with ``model = "auto"`` in ``kestrel.toml``. The disk
   cache is empty on first ``--quickstart``, so the
   ``_load_from_disk_cache`` path inside ``LLMService.__init__`` doesn't
   call ``_resolve_auto_providers``. Nothing else triggers
   ``discover_all_models`` until either the model-picker UI or
   ``ModelFeature.list_models`` runs. The very first chat call therefore
   reaches the adapter with the literal string ``"auto"`` — Ollama 404s
   on recent versions, and older versions can hang the request
   indefinitely while the SDK retries.

2. **No upper bound on the discovery LLM call.**
   ``BootstrapService.process_discovery_message`` runs inside the
   agent's CONVERSATION lock. If the LLM call hangs, every subsequent
   request on the agent (HTTP, shell, A2A) blocks waiting for the
   lock — the agent stays wedged until restart.

3. **ConstitutionFeature reads a CWD-relative path.**
   ``with open("docs/principles/KESTREL_CONSTITUTION.md")`` only works
   from a source clone with the right CWD. Pip-installed users boot
   with a noisy ``[Errno 2] No such file or directory`` from
   ``ConstitutionFeature.initialize``.

Each test below pins a regression guard for one of the three fixes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.bootstrap.service import BootstrapService, BootstrapState


@pytest.mark.asyncio
async def test_generate_with_messages_lazy_resolves_auto_models():
    """First call with ``model='auto'`` triggers ``discover_all_models``,
    then proceeds with the resolved id — never sends ``"auto"`` to the
    adapter."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._backend = MagicMock(name="not-remote")
    svc._remote_client = None
    svc._disabled_routes = {}
    svc._mandate_preference = {"vendor": None, "model": None, "route": None}

    adapter = MagicMock()
    adapter.get_response = AsyncMock(return_value="resolved-model-said-hello")

    seed_provider = {
        "name": "ollama:local",
        "vendor": "ollama",
        "model": "auto",
        "adapter": adapter,
        "client": MagicMock(),
        "is_local": True,
    }
    svc.providers = [seed_provider]

    async def _fake_discover(use_cache: bool = True):
        seed_provider["model"] = "llama3.2:3b"
        return []

    svc.discover_all_models = AsyncMock(side_effect=_fake_discover)
    svc._check_model_tool_support = MagicMock(side_effect=lambda providers, tools, model_override: tools)
    svc._resolve_model_selector = MagicMock(return_value={"provider": None, "model": None})
    svc._filter_providers_by_selector = MagicMock(return_value=[seed_provider])
    svc._maybe_disable_route = MagicMock()

    response = await svc.generate_with_messages(messages=[{"role": "user", "content": "hi"}])

    assert response == "resolved-model-said-hello"
    svc.discover_all_models.assert_awaited_once()
    # The adapter must NOT have been called with "auto".
    sent_model = adapter.get_response.await_args.kwargs.get("model")
    assert sent_model == "llama3.2:3b", f"adapter received {sent_model!r}, expected resolved id"


@pytest.mark.asyncio
async def test_streaming_path_lazy_resolves_auto_models():
    """#2069: the STREAMING entry points must trigger the same lazy
    discovery warm-up as ``generate_with_messages``.

    Pre-fix, only ``generate_with_messages`` warmed discovery; the streaming
    paths (``get_streaming_response`` / ``stream_with_messages`` /
    ``stream_with_tool_detection``) called ``resolve_provider_routing``
    directly. On a fresh boot with a cold cache, every route fails
    ``_resolve_concrete_model`` and the walk surfaces the LAST route's
    ``ModelNotAvailableForRoute`` (e.g. ``ollama:local``) as a hard error —
    even though a key-backed vendor is configured. This pins that the shared
    ``_resolve_routing_with_discovery`` warm-up runs on the streaming path."""
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.streaming import RoutingResolution

    svc = LLMService.__new__(LLMService)
    svc._disabled_routes = {}

    seed_provider = {
        "name": "ollama:local",
        "vendor": "ollama",
        "model": "auto",
    }
    svc.providers = [seed_provider]

    async def _fake_discover(use_cache: bool = True):
        seed_provider["model"] = "llama3.2:3b"
        return []

    svc.discover_all_models = AsyncMock(side_effect=_fake_discover)

    captured = {}

    def _fake_resolve(*, model_override=None, force_local_only=False):
        # Capture the provider model AT RESOLUTION TIME so the test proves the
        # warm-up ran BEFORE routing, not after.
        captured["model_at_resolve"] = seed_provider["model"]
        return RoutingResolution([], None, (False, set()))

    svc.resolve_provider_routing = MagicMock(side_effect=_fake_resolve)

    resolution = await svc._resolve_routing_with_discovery(
        model_override=None, force_local_only=False,
    )

    svc.discover_all_models.assert_awaited_once()
    assert captured["model_at_resolve"] == "llama3.2:3b", (
        "discovery warm-up must run before resolve_provider_routing so the "
        "route walk sees a concrete model, not 'auto'"
    )
    assert isinstance(resolution, RoutingResolution)


@pytest.mark.asyncio
async def test_ensure_models_discovered_skips_when_already_resolved():
    """The warm-up is a no-op once every route has a concrete model — it must
    not fire discovery on every turn after the first."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._disabled_routes = {}
    svc.providers = [{"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"}]
    svc.discover_all_models = AsyncMock(return_value=[])

    await svc._ensure_models_discovered()

    svc.discover_all_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_models_discovered_never_runs_global_discovery_when_local_only():
    """#2069 codex r2 (privacy): a ``force_local_only`` turn must NEVER call the
    global ``discover_all_models`` — it enumerates every configured vendor
    (incl. cloud) and writes the shared/disk cache, a leak + cache-poisoning
    risk for an ISOLATED/EPHEMERAL session. A cloud-only ``auto`` route warms
    nothing (the force-local filter drops it from the turn)."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._disabled_routes = {}
    svc.providers = [{"name": "openai:api", "vendor": "openai", "model": "auto"}]
    svc.discover_all_models = AsyncMock(return_value=[])
    svc._resolve_local_auto_routes = AsyncMock()

    await svc._ensure_models_discovered(force_local_only=True)
    svc.discover_all_models.assert_not_awaited()
    svc._resolve_local_auto_routes.assert_not_awaited()  # no LOCAL auto route

    # Sanity: the SAME cold-auto state DOES run global discovery when not
    # local-only — so the gate above is real, not a dead branch.
    await svc._ensure_models_discovered(force_local_only=False)
    svc.discover_all_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_models_discovered_resolves_local_auto_when_local_only():
    """#2069 codex r3: a ``force_local_only`` turn with a cold ``auto`` LOCAL
    route — in BOTH all-local and mixed cloud/local configs — must resolve that
    local route via the local-scoped path (no cloud contact, no cache write),
    not fail. It must NOT fall back to the global ``discover_all_models``."""
    from kestrel_sovereign.llm.service import LLMService

    # Mixed config: a cloud route AND a local auto route. This is codex's case.
    svc = LLMService.__new__(LLMService)
    svc._disabled_routes = {}
    svc.providers = [
        {"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"},
        {"name": "ollama:local", "vendor": "ollama", "model": "auto", "is_local": True},
    ]
    svc.discover_all_models = AsyncMock(return_value=[])
    svc._resolve_local_auto_routes = AsyncMock()

    await svc._ensure_models_discovered(force_local_only=True)
    svc._resolve_local_auto_routes.assert_awaited_once()
    svc.discover_all_models.assert_not_awaited()  # never the cloud-contacting path


@pytest.mark.asyncio
async def test_resolve_local_auto_routes_contacts_local_routes_only():
    """``_resolve_local_auto_routes`` must scope discovery to LOCAL routes only
    (never a cloud route) and resolve autos in place without writing the cache."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    cloud = {"name": "openai:api", "vendor": "openai", "model": "auto", "is_local": False}
    local = {"name": "ollama:local", "vendor": "ollama", "model": "auto", "is_local": True}
    svc.providers = [cloud, local]
    svc._select_discovery_routes = MagicMock(
        return_value=[("openai", cloud), ("ollama", local)]
    )
    discovered = {"models": ["local-model-info"]}
    contacted = []

    async def _discover(vendor, route):
        contacted.append(vendor)
        return discovered["models"]

    svc._discover_for_vendor_route = AsyncMock(side_effect=_discover)
    svc._resolve_auto_providers = MagicMock()

    await svc._resolve_local_auto_routes()

    assert contacted == ["ollama"], "must contact ONLY the local route, never cloud"
    # Resolution is scoped to the local providers only — never the cloud route.
    svc._resolve_auto_providers.assert_called_once_with(
        ["local-model-info"], only_providers=[local]
    )


@pytest.mark.asyncio
async def test_resolve_local_auto_routes_does_not_mutate_cloud_sharing_vendor():
    """#2069 codex r4: a cloud route that SHARES a vendor with a local route
    must not be resolved to a locally-discovered model id (else a later
    non-local request would send a local-only model to the cloud route).
    End-to-end through the real _resolve_auto_providers."""
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo

    svc = LLMService.__new__(LLMService)
    # Same vendor "acme", two routes: cloud (auto) and local (auto).
    cloud = {"name": "acme:api", "vendor": "acme", "model": "auto", "is_local": False,
             "selection_hints": []}
    local = {"name": "acme:local", "vendor": "acme", "model": "auto", "is_local": True,
             "selection_hints": []}
    svc.providers = [cloud, local]
    svc._select_discovery_routes = MagicMock(return_value=[("acme", local)])
    svc._ensure_route_catalogs_sync = MagicMock()
    svc._route_catalogs = {}
    local_model = ModelInfo(
        id="acme-local-7b", provider="acme", display_name="Acme Local 7B",
        category=ModelCategory.CHAT, is_featured=True, is_hidden=False,
    )
    svc._discover_for_vendor_route = AsyncMock(return_value=[local_model])

    await svc._resolve_local_auto_routes()

    assert local["model"] == "acme-local-7b", "local route must resolve"
    assert cloud["model"] == "auto", "cloud route sharing the vendor must stay 'auto'"


@pytest.mark.asyncio
async def test_process_discovery_message_times_out_on_llm_hang():
    """A hung LLM call is bounded by ``DISCOVERY_LLM_TIMEOUT_SECONDS``.

    Pre-fix this would hold the agent's CONVERSATION lock forever and
    wedge every subsequent chat / shell / A2A request. The whole point
    of the bound is that ``asyncio.wait_for`` cancels the inner task
    on timeout — the discovery LLM never gets to hold the lock past the
    deadline."""

    class HangingLLM:
        async def generate_with_messages(self, *, messages):
            await asyncio.sleep(60)  # well past the test timeout

    db = MagicMock()
    db.fetchall = AsyncMock(return_value=[])
    db.execute = AsyncMock()

    mock_storage = MagicMock()
    service = BootstrapService(
        db=db,
        agent_id="did:test:hang",
        agent_name="HangAgent",
        llm_service=HangingLLM(),
        agent_data_path=None,
        storage=mock_storage,
    )
    # Tighten the timeout for the test so we don't actually wait 60s.
    service.DISCOVERY_LLM_TIMEOUT_SECONDS = 0.5

    with pytest.raises(asyncio.TimeoutError):
        await service.process_discovery_message("hello")


def test_constitution_feature_finds_package_shipped_constitution(tmp_path, monkeypatch):
    """When CWD has no ``docs/principles/KESTREL_CONSTITUTION.md`` (the
    pip-install case), the loader falls back to the
    ``kestrel_sovereign/data/KESTREL_CONSTITUTION.md`` shipped in the
    wheel."""
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.features.constitution import ConstitutionFeature

    monkeypatch.chdir(tmp_path)  # CWD has no docs/ dir

    text = ConstitutionFeature._read_canonical_constitution()

    assert "Kestrel" in text or "Constitution" in text or "Article" in text or "Book" in text, (
        f"package-shipped constitution at {CONSTITUTION_PATH} read but content "
        f"didn't match any expected token: first 200 chars = {text[:200]!r}"
    )


def test_constitution_feature_prefers_source_clone_path(tmp_path, monkeypatch):
    """Source clones with a real ``docs/principles/KESTREL_CONSTITUTION.md``
    in CWD still get THAT file (so ongoing edits are picked up without a
    rebuild)."""
    from kestrel_sovereign.features.constitution import ConstitutionFeature

    docs = tmp_path / "docs" / "principles"
    docs.mkdir(parents=True)
    (docs / "KESTREL_CONSTITUTION.md").write_text("MARKER-SOURCE-CLONE-WINS\n")
    monkeypatch.chdir(tmp_path)

    text = ConstitutionFeature._read_canonical_constitution()

    assert "MARKER-SOURCE-CLONE-WINS" in text


def test_constitution_feature_reports_path_when_neither_exists(tmp_path, monkeypatch):
    """Both candidate paths missing → raise with a clear list, not a
    bare FileNotFoundError that hides the second candidate."""
    from kestrel_sovereign.features import constitution as const_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(const_mod, "CONSTITUTION_PATH", str(tmp_path / "nope.md"), raising=False)
    # Patch the import-inside-function so the helper sees our fake CONSTITUTION_PATH.
    import kestrel_sovereign.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONSTITUTION_PATH", str(tmp_path / "nope.md"))

    with pytest.raises(FileNotFoundError, match="not found at any of"):
        const_mod.ConstitutionFeature._read_canonical_constitution()
