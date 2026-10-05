"""Opt-in live smoke: model discovery against this host's configured vendors.

Skipped unless ``KESTREL_LIVE_TESTS=1``. What it observes is which vendors are
configured and reachable today, which is a property of the host rather than of
the discovery code (#3270). The hermetic contract, including the 404 and
missing-credential outcomes, is ``tests/unit/test_model_discovery.py``.
"""

import os

import pytest

from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.llm import model_catalog
from kestrel_sovereign.llm.model_catalog import ModelCatalogService
from kestrel_sovereign.llm.model_metadata import ModelInfo
from kestrel_sovereign.llm.service import LLMService

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("KESTREL_LIVE_TESTS") != "1",
        reason="live discovery requires explicit opt-in: set KESTREL_LIVE_TESTS=1",
    ),
]


@pytest.mark.asyncio
async def test_live_discovery_returns_only_model_info(tmp_path, monkeypatch):
    """Whatever the vendors answer, only framework ``ModelInfo`` reaches the catalog.

    A vendor that fails is excluded and named in ``_discovery_failures``
    instead of failing this test.
    """
    # The default catalog writes its disk cache beside the package source.
    monkeypatch.setattr(
        model_catalog,
        "_catalog_service",
        ModelCatalogService(cache_path=tmp_path / "model_discovery_cache.json"),
    )
    # Discovery registers every context limit it sees in a process-wide map.
    monkeypatch.setattr(token_counter, "_discovered_context_limits", {})
    service = LLMService()
    try:
        models = await service.discover_all_models(use_cache=False)
        failures = dict(service._discovery_failures)
    finally:
        await service.close()

    assert models, f"No models discovered; vendor failures: {failures}"
    not_framework = [
        (type(m).__module__, m.provider, m.id)
        for m in models
        if not isinstance(m, ModelInfo)
    ]
    assert not_framework == []
    assert all(isinstance(reason, str) for reason in failures.values())
