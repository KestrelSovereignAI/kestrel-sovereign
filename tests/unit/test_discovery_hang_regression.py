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

    service = BootstrapService(
        db=db,
        agent_id="did:test:hang",
        agent_name="HangAgent",
        llm_service=HangingLLM(),
        agent_data_path=None,
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
