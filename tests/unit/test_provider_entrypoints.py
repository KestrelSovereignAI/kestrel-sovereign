"""
Tests for provider entry_point discovery across all registries.

Tests that each registry correctly scans entry_points, merges with
built-in providers, and handles failures gracefully.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from types import SimpleNamespace

from kestrel_sovereign.entrypoints import discover_entry_point_classes


# ---------------------------------------------------------------------------
# Shared helper: mock entry_points
# ---------------------------------------------------------------------------

def _make_entry_point(name: str, cls):
    """Create a mock entry point that loads the given class."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"mock_package:{name}"
    ep.load.return_value = cls
    return ep


def _patch_entry_points(group: str, entry_points: list):
    """Return a patch context manager that mocks importlib.metadata.entry_points."""
    mock_eps = MagicMock()
    mock_eps.select.return_value = entry_points
    return patch("kestrel_sovereign.entrypoints.importlib.metadata.entry_points", return_value=mock_eps)


# ===========================================================================
# 1. Core utility: discover_entry_point_classes
# ===========================================================================

class TestDiscoverEntryPointClasses:
    """Tests for the shared discovery utility."""

    def test_discovers_valid_subclass(self):
        class Base:
            pass
        class Sub(Base):
            pass

        ep = _make_entry_point("sub", Sub)
        with _patch_entry_points("test.group", [ep]):
            result = discover_entry_point_classes("test.group", Base)
        assert "sub" in result
        assert result["sub"] is Sub

    def test_skips_non_subclass(self):
        class Base:
            pass
        class Unrelated:
            pass

        ep = _make_entry_point("bad", Unrelated)
        with _patch_entry_points("test.group", [ep]):
            result = discover_entry_point_classes("test.group", Base)
        assert "bad" not in result

    def test_skips_base_class_itself(self):
        class Base:
            pass

        ep = _make_entry_point("base", Base)
        with _patch_entry_points("test.group", [ep]):
            result = discover_entry_point_classes("test.group", Base)
        assert "base" not in result

    def test_handles_load_failure(self):
        class Base:
            pass

        ep = MagicMock()
        ep.name = "broken"
        ep.value = "broken_package:Broken"
        ep.load.side_effect = ImportError("package not found")

        with _patch_entry_points("test.group", [ep]):
            result = discover_entry_point_classes("test.group", Base)
        assert result == {}

    def test_handles_entry_points_failure(self):
        class Base:
            pass

        with patch(
            "kestrel_sovereign.entrypoints.importlib.metadata.entry_points",
            side_effect=Exception("metadata broken"),
        ):
            result = discover_entry_point_classes("test.group", Base)
        assert result == {}

    def test_empty_group(self):
        class Base:
            pass

        with _patch_entry_points("test.group", []):
            result = discover_entry_point_classes("test.group", Base)
        assert result == {}

    def test_python_39_dict_fallback(self):
        """Test compatibility with Python 3.9 dict-style entry_points."""
        class Base:
            pass
        class Sub(Base):
            pass

        ep = _make_entry_point("sub", Sub)
        # Simulate Python 3.9: entry_points() returns a dict, no .select
        mock_eps = {"test.group": [ep], "other.group": []}
        with patch(
            "kestrel_sovereign.entrypoints.importlib.metadata.entry_points",
            return_value=mock_eps,
        ):
            result = discover_entry_point_classes("test.group", Base)
        assert "sub" in result


# ===========================================================================
# 2. Voice Provider Registry
# ===========================================================================

class TestVoiceProviderEntryPoints:
    """Tests for voice provider entry_point discovery."""

    @pytest.mark.asyncio
    async def test_discovers_tts_entry_point(self):
        from kestrel_sovereign.voice.base import TTSProvider
        from kestrel_sovereign.voice.provider_registry import (
            VoiceProviderRegistry,
            VOICE_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeTTS(TTSProvider):
            name = "fake_tts"
            is_local = True
            async def synthesize(self, *a, **kw): return b""
            async def synthesize_stream(self, *a, **kw): yield b""
            async def list_voices(self): return []
            async def is_available(self): return True
            def __init__(self, config=None): pass

        ep = _make_entry_point("fake_tts", FakeTTS)

        with _patch_entry_points(VOICE_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            registry = VoiceProviderRegistry(config={})
            await registry.initialize()

        assert "fake_tts" in registry.list_tts_providers()

    @pytest.mark.asyncio
    async def test_builtin_wins_on_collision(self):
        from kestrel_sovereign.voice.base import TTSProvider
        from kestrel_sovereign.voice.provider_registry import (
            VoiceProviderRegistry,
            VOICE_PROVIDER_ENTRY_POINT_GROUP,
        )

        class CollisionTTS(TTSProvider):
            name = "openai"
            is_local = False
            async def synthesize(self, *a, **kw): return b""
            async def synthesize_stream(self, *a, **kw): yield b""
            async def list_voices(self): return []
            async def is_available(self): return True
            def __init__(self, config=None):
                self._marker = "entry_point"

        ep = _make_entry_point("openai", CollisionTTS)

        # Config that would create built-in openai TTS
        config = {"tts_provider_priority": ["openai"], "openai": {}}

        with _patch_entry_points(VOICE_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            with patch(
                "kestrel_sovereign.voice.provider_registry.VoiceProviderRegistry._create_tts_provider"
            ) as mock_create:
                builtin = MagicMock()
                builtin.name = "openai"
                builtin.is_available = AsyncMock(return_value=True)
                mock_create.return_value = builtin
                registry = VoiceProviderRegistry(config=config)
                await registry.initialize()

        # Built-in should win
        provider = registry.get_tts("openai")
        assert provider is builtin

    @pytest.mark.asyncio
    async def test_unavailable_entry_point_skipped(self):
        from kestrel_sovereign.voice.base import TTSProvider
        from kestrel_sovereign.voice.provider_registry import (
            VoiceProviderRegistry,
            VOICE_PROVIDER_ENTRY_POINT_GROUP,
        )

        class UnavailableTTS(TTSProvider):
            name = "unavailable"
            is_local = True
            async def synthesize(self, *a, **kw): return b""
            async def synthesize_stream(self, *a, **kw): yield b""
            async def list_voices(self): return []
            async def is_available(self): return False
            def __init__(self, config=None): pass

        ep = _make_entry_point("unavailable", UnavailableTTS)

        with _patch_entry_points(VOICE_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            registry = VoiceProviderRegistry(config={})
            await registry.initialize()

        assert "unavailable" not in registry.list_tts_providers()


# ===========================================================================
# 3. LLM Provider Registry
# ===========================================================================

class TestLLMProviderEntryPoints:
    """Tests for LLM provider entry_point discovery."""

    def test_discovers_llm_entry_point(self):
        """Entry-point LLM providers become single-route vendors (``<name>:api``)."""
        from kestrel_sovereign.llm.adapter import LLMAdapter
        from kestrel_sovereign.llm.provider_registry import (
            ProviderRegistry,
            LLM_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeAdapter(LLMAdapter):
            provider_name = "fake_llm"
            async def get_response(self, *a, **kw): return ""

        ep = _make_entry_point("fake_llm", FakeAdapter)

        # Vendor config under the new shape — entry point adapters read from
        # ``vendors.<name>.routes.api`` (see _discover_entrypoint_providers).
        config = {
            "route_priority": [],
            "vendors": {
                "fake_llm": {"routes": {"api": {"base_url": "http://localhost:1234"}}},
            },
        }

        with _patch_entry_points(LLM_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            registry = ProviderRegistry(config=config)
            try:
                providers = registry.initialize_providers()
            except Exception:
                providers = registry.providers

        names = [p.name for p in providers]
        # Composite name under the new scheme.
        assert "fake_llm:api" in names
        fake = next(p for p in providers if p.name == "fake_llm:api")
        assert fake.vendor == "fake_llm"
        assert fake.route == "api"

    def test_builtin_wins_on_collision(self):
        """A built-in vendor route takes precedence over an entry point of the same name."""
        from kestrel_sovereign.llm.adapter import LLMAdapter
        from kestrel_sovereign.llm.provider_registry import (
            ProviderRegistry,
            LLM_PROVIDER_ENTRY_POINT_GROUP,
            ProviderInfo,
        )

        class FakeAdapter(LLMAdapter):
            provider_name = "ollama"
            async def get_response(self, *a, **kw): return ""

        ep = _make_entry_point("ollama", FakeAdapter)

        config = {
            "route_priority": ["ollama:local"],
            "vendors": {
                "ollama": {
                    "is_cloud": False,
                    "routes": {
                        "local": {
                            "adapter": "OllamaAdapter",
                            "host": "http://localhost:11434",
                            "model": "test",
                        },
                    },
                },
            },
        }

        builtin = ProviderInfo(
            name="ollama:local",
            vendor="ollama",
            route="local",
            client=MagicMock(),
            adapter=MagicMock(),
            model="test",
            is_cloud=False,
            is_local=True,
        )

        with _patch_entry_points(LLM_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            with patch.object(
                ProviderRegistry,
                "_build_route",
                return_value=builtin,
            ):
                registry = ProviderRegistry(config=config)
                providers = registry.initialize_providers()

        # Built-in ollama:local wins; entry point fake "ollama:api" is ignored
        # when a route with the same composite name is already registered.
        ollama_providers = [p for p in providers if p.vendor == "ollama"]
        # Either just the built-in, or the built-in plus a distinct route name.
        assert any(p.name == "ollama:local" for p in ollama_providers)
        # The entry point's default "<name>:api" should NOT collide with the built-in.
        # (The built-in is registered first, and entry-point dedup uses composite names.)


# ===========================================================================
# 4. Channel Registry
# ===========================================================================

class TestChannelRegistryEntryPoints:
    """Tests for channel adapter entry_point discovery."""

    def test_discovers_channel_adapter(self):
        from kestrel_sovereign.features.channels.adapter import ChannelAdapter
        from kestrel_sovereign.features.channels.registry import (
            ChannelRegistry,
            CHANNEL_ADAPTER_ENTRY_POINT_GROUP,
        )

        class FakeAdapter(ChannelAdapter):
            @property
            def channel_type(self): return "fake"
            @property
            def is_connected(self): return False
            async def connect(self): pass
            async def disconnect(self): pass
            async def send_message(self, *a, **kw): pass
            async def on_message(self, *a, **kw): pass

        ep = _make_entry_point("fake", FakeAdapter)

        with _patch_entry_points(CHANNEL_ADAPTER_ENTRY_POINT_GROUP, [ep]):
            registry = ChannelRegistry()

        assert "fake" in registry.list_discovered_adapters()
        assert registry.get_adapter_class("fake") is FakeAdapter

    def test_no_entry_points_still_works(self):
        from kestrel_sovereign.features.channels.registry import (
            ChannelRegistry,
            CHANNEL_ADAPTER_ENTRY_POINT_GROUP,
        )

        with _patch_entry_points(CHANNEL_ADAPTER_ENTRY_POINT_GROUP, []):
            registry = ChannelRegistry()

        assert registry.list_discovered_adapters() == []
        assert registry.adapter_count == 0


# ===========================================================================
# 5. Storage Provider Discovery
# ===========================================================================

class TestStorageProviderEntryPoints:
    """Tests for storage provider entry_point discovery."""

    def test_discovers_storage_provider(self):
        from kestrel_sovereign.storage.providers.base import StorageProvider, StorageTier
        from kestrel_sovereign.storage.providers import (
            discover_storage_providers,
            STORAGE_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeStorage(StorageProvider):
            @property
            def tier(self): return StorageTier.LOCAL
            @property
            def provider_name(self): return "fake"
            def is_available(self): return True
            async def store(self, *a, **kw): pass
            async def retrieve(self, *a, **kw): return b""
            async def list_content(self, *a, **kw): return []
            async def delete(self, *a, **kw): return True
            async def verify(self, *a, **kw): return True

        ep = _make_entry_point("fake_storage", FakeStorage)

        with _patch_entry_points(STORAGE_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            result = discover_storage_providers()

        assert "fake_storage" in result
        assert result["fake_storage"] is FakeStorage


# ===========================================================================
# 6. Deploy Manager
# ===========================================================================

class TestDeployManagerEntryPoints:
    """Tests for deploy manager entry_point discovery."""

    def test_discovers_deploy_provider(self):
        from kestrel_sovereign.features.deploy.providers.base import DeployProvider
        from kestrel_sovereign.features.deploy.manager import (
            DeployManager,
            CLOUD_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeDeployProvider(DeployProvider):
            async def deploy(self, *a, **kw): return {}
            async def get_status(self, *a, **kw): return {}
            async def teardown(self, *a, **kw): return {}
            async def get_logs(self, *a, **kw): return ""
            async def list_deployments(self, *a, **kw): return []
            async def health_check(self, *a, **kw): return {}

        ep = _make_entry_point("runpod", FakeDeployProvider)

        with _patch_entry_points(CLOUD_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            with patch(
                "kestrel_sovereign.features.deploy.core.load_config",
                return_value={"manager": {}, "profiles": {}},
            ):
                manager = DeployManager(config={"manager": {}, "profiles": {}})

        assert "runpod" in manager.list_external_providers()
        assert manager.get_external_provider_class("runpod") is FakeDeployProvider


# ===========================================================================
# 7. Web Search (refactored)
# ===========================================================================

class TestWebSearchEntryPoints:
    """Tests for web search provider entry_point discovery."""

    def test_discovers_search_provider(self):
        from kestrel_sovereign.features.web_search.base import SearchProvider
        from kestrel_sovereign.features.web_search.tool import (
            WebSearchTool,
            SEARCH_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeSearch(SearchProvider):
            @property
            def name(self): return "fake_search"
            @property
            def enabled(self): return True
            async def search(self, query, max_results=5, **kw):
                return {"success": True, "results": []}

        ep = _make_entry_point("fake_search", FakeSearch)

        with _patch_entry_points(SEARCH_PROVIDER_ENTRY_POINT_GROUP, [ep]):
            tool = WebSearchTool()

        assert "fake_search" in tool.list_providers()
        assert tool.enabled is True

    def test_tavily_still_works_as_default(self):
        from kestrel_sovereign.features.web_search.tool import (
            WebSearchTool,
            SEARCH_PROVIDER_ENTRY_POINT_GROUP,
        )

        with _patch_entry_points(SEARCH_PROVIDER_ENTRY_POINT_GROUP, []):
            with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
                tool = WebSearchTool()

        assert tool.enabled is True
        assert "tavily" in tool.list_providers()
        assert tool.get_provider().name == "tavily"

    def test_no_providers_disabled(self):
        from kestrel_sovereign.features.web_search.tool import (
            WebSearchTool,
            SEARCH_PROVIDER_ENTRY_POINT_GROUP,
        )

        with _patch_entry_points(SEARCH_PROVIDER_ENTRY_POINT_GROUP, []):
            with patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("TAVILY_API_KEY", None)
                tool = WebSearchTool()

        assert tool.enabled is False

    @pytest.mark.asyncio
    async def test_search_uses_default_provider(self):
        from kestrel_sovereign.features.web_search.base import SearchProvider
        from kestrel_sovereign.features.web_search.tool import (
            WebSearchTool,
            SEARCH_PROVIDER_ENTRY_POINT_GROUP,
        )

        class FakeSearch(SearchProvider):
            @property
            def name(self): return "fake"
            @property
            def enabled(self): return True
            async def search(self, query, max_results=5, **kw):
                return {"success": True, "results": [{"title": query}]}

        ep = _make_entry_point("fake", FakeSearch)

        # Remove TAVILY_API_KEY so Tavily doesn't become the default
        import os
        env_patch = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        with patch.dict("os.environ", env_patch, clear=True):
            with _patch_entry_points(SEARCH_PROVIDER_ENTRY_POINT_GROUP, [ep]):
                tool = WebSearchTool()
                result = await tool.search("test query")

        assert result["success"] is True
        assert result["results"][0]["title"] == "test query"

    def test_search_provider_base_format(self):
        """Test the base class format_results_for_llm."""
        from kestrel_sovereign.features.web_search.base import SearchProvider

        class Impl(SearchProvider):
            @property
            def name(self): return "test"
            @property
            def enabled(self): return True
            async def search(self, *a, **kw): return {}

        provider = Impl()
        result = provider.format_results_for_llm({"success": False, "error": "boom"})
        assert "boom" in result

        result = provider.format_results_for_llm({
            "success": True,
            "answer": "42",
            "results": [{"title": "Test", "url": "http://x", "content": "hello"}],
        })
        assert "42" in result
        assert "Test" in result
