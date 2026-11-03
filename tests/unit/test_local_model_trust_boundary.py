"""Local embedding models stay outside the Kestrel process (#3718).

GHSA-jhr6-gm9c-rqjv (CVE-2026-68770): before 6.0, sentence-transformers
imports the custom module class a local model directory's ``modules.json``
names, even with ``trust_remote_code=False``. Its 5.6 and 5.7 releases only
warn; 6.0 refuses before importing anything.

Core never imports sentence-transformers, although the ``local`` extra
installs it. Core's local embedding path sends the configured model *name*
to an Ollama server over HTTP, and that server, a separate process, owns the
weights. These tests pin that boundary with a fabricated model directory
shaped like the advisory's exploit, named as the embedding model.

Nothing here loads a real model, downloads one, or runs the fixture's code.
In the Core path tests a tripwire on the import system records and refuses
every attempt to import ``sentence_transformers``, so even a regression cannot
reach the real loader. The one test that uses the real library runs it in a
child interpreter, and only after confirming the installed release refuses.
The fixture's module, if anything ever imported it, would only write a
marker file beside itself; every test asserts the marker is absent.
"""
from __future__ import annotations

import ast
import importlib.abc
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import ollama
import pytest
from packaging.version import Version

from kestrel_sovereign.llm.embedding_service import (
    EmbeddingService,
    ProviderEmbeddingService,
)
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from tests.unit.test_dependency_contract import SECURITY_FLOORS
from tests.utils.process_local_llm_service import process_local_service

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "kestrel_sovereign"

#: First sentence-transformers release that refuses a local directory's
#: custom module class without ``trust_remote_code=True``.
REFUSING_RELEASE = Version(SECURITY_FLOORS["sentence-transformers"])

PROBE_MODULE = "modeling_kestrel_3718_probe"
MARKER = "PROBE_IMPORTED"
VECTOR = [0.25, 0.5, 0.75]


def _is_sentence_transformers(name: str) -> bool:
    return name == "sentence_transformers" or name.startswith(
        "sentence_transformers."
    )


class _ImportTripwire(importlib.abc.MetaPathFinder):
    """Records, then refuses, every attempt to import ``sentence_transformers``.

    A finder sees an attempt even when the importer catches the failure and
    degrades, which a ``sys.modules`` stub would not.
    """

    def __init__(self) -> None:
        self.attempts: List[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if _is_sentence_transformers(fullname):
            self.attempts.append(fullname)
            raise ImportError(f"#3718 tripwire: {fullname} was imported")
        return None


@pytest.fixture
def tripwire(monkeypatch) -> Iterator[_ImportTripwire]:
    for name in [name for name in sys.modules if _is_sentence_transformers(name)]:
        monkeypatch.delitem(sys.modules, name)
    finder = _ImportTripwire()
    sys.meta_path.insert(0, finder)
    try:
        yield finder
    finally:
        sys.meta_path.remove(finder)


def _fabricate_untrusted_model_dir(root: Path) -> Path:
    """A local model directory shaped like the advisory's exploit.

    ``modules.json`` names a class from a repository-local ``modeling_*.py``,
    which is what sentence-transformers before 6.0 imports from an existing
    local path regardless of ``trust_remote_code``.
    """
    model_dir = root / "untrusted-model"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text(
        json.dumps(
            [{"idx": 0, "name": "0", "path": "", "type": f"{PROBE_MODULE}.ProbeModule"}]
        ),
        encoding="utf-8",
    )
    (model_dir / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "SentenceTransformer"}), encoding="utf-8"
    )
    (model_dir / f"{PROBE_MODULE}.py").write_text(
        "# Fabricated #3718 test fixture. Being imported is the failure under\n"
        "# test; all it does is leave a marker beside itself.\n"
        "import pathlib\n"
        f"pathlib.Path(__file__).with_name({MARKER!r}).write_text('imported')\n"
        "\n"
        "\n"
        "class ProbeModule:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return model_dir


def _assert_fixture_never_imported(model_dir: Path) -> None:
    assert not (model_dir / MARKER).exists(), (
        "the fabricated model directory's custom module was imported"
    )
    assert not [name for name in sys.modules if PROBE_MODULE in name]


class _OllamaDaemon:
    """A stand-in Ollama server: it answers ``/api/embed`` and records requests."""

    def __init__(self) -> None:
        self.requests: List[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        self.requests.append(body)
        count = len(body["input"]) if isinstance(body["input"], list) else 1
        return httpx.Response(
            200, json={"model": body["model"], "embeddings": [VECTOR] * count}
        )

    def async_client(self) -> ollama.AsyncClient:
        return ollama.AsyncClient(
            host="http://ollama.test:11434", transport=httpx.MockTransport(self.handle)
        )

    def client(self) -> ollama.Client:
        return ollama.Client(
            host="http://ollama.test:11434", transport=httpx.MockTransport(self.handle)
        )


def _local_ollama_route(client: ollama.AsyncClient, model: str) -> dict[str, Any]:
    return {
        "name": "ollama:local",
        "vendor": "ollama",
        "route": "local",
        "adapter": OllamaAdapter(embedding_model=model, embedding_dim=len(VECTOR)),
        "client": client,
        "model": "auto",
        "is_local": True,
        "is_cloud": False,
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": model,
            "embedding_dim": len(VECTOR),
        },
    }


def _llm_service(provider: dict[str, Any]):
    service = process_local_service([provider])
    service._observability_store = MagicMock()
    service._observability_store.log_llm_call = AsyncMock()
    service._owner_agent_did = "did:example:agent"
    service._track_model_usage = AsyncMock()
    return service


def _sentence_transformers_references(
    source_root: Path, base: Path = REPO_ROOT
) -> List[str]:
    """Every import of, or module-name string naming, ``sentence_transformers``."""
    hits: List[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # importlib.import_module("..."), find_spec("..."), entry points
                names = [node.value]
            else:
                continue
            hits.extend(
                f"{path.relative_to(base).as_posix()}:{node.lineno}: {name}"
                for name in names
                if _is_sentence_transformers(name)
            )
    return hits


def test_core_has_no_in_process_sentence_transformers_loader() -> None:
    """The reachability map behind #3718: no Core module imports the library.

    A new in-process loader brings the advisory's trust boundary into Core.
    It needs a trust review, and the boundary section of
    LLM_SERVICE_ARCHITECTURE.md updated, before this test may change. A name
    computed at runtime evades this scan; the path tests below catch that.
    """
    assert _sentence_transformers_references(PACKAGE_ROOT) == []


def test_the_reference_scan_sees_imports_and_module_name_strings(tmp_path: Path) -> None:
    source = tmp_path / "pkg"
    source.mkdir()
    (source / "loader.py").write_text(
        "import sentence_transformers\n"
        "from sentence_transformers.util import misc\n"
        "import importlib\n"
        "importlib.import_module('sentence_transformers')\n"
        "NOTE = 'mentions sentence_transformers in prose only'\n",
        encoding="utf-8",
    )
    hits = _sentence_transformers_references(source, base=tmp_path)

    assert sorted(hit.split(": ", 1)[0] for hit in hits) == [
        "pkg/loader.py:1",
        "pkg/loader.py:2",
        "pkg/loader.py:4",
    ]


@pytest.mark.asyncio
async def test_provider_route_sends_a_local_model_directory_to_ollama_as_a_name(
    tmp_path: Path, tripwire: _ImportTripwire
) -> None:
    """The production path: ``LLMService.get_embedding_service()`` on Ollama.

    The configured model names an existing local directory holding the
    advisory's payload shape. Core forwards that string in the HTTP request
    and imports nothing, from the directory or from sentence-transformers.
    """
    model_dir = _fabricate_untrusted_model_dir(tmp_path)
    daemon = _OllamaDaemon()
    client = daemon.async_client()
    try:
        embed = _llm_service(
            _local_ollama_route(client, str(model_dir))
        ).get_embedding_service()
        assert isinstance(embed, ProviderEmbeddingService)

        document = await embed.aembed("a stored memory")
        query = await embed.aembed_query("what do I remember?")
        batch = await embed.aembed_batch(["first", "second"])
    finally:
        await client._client.aclose()

    assert (document, query, batch) == (VECTOR, VECTOR, [VECTOR, VECTOR])
    assert [request["model"] for request in daemon.requests] == [str(model_dir)] * 3
    assert tripwire.attempts == []
    _assert_fixture_never_imported(model_dir)


@pytest.mark.asyncio
async def test_legacy_ollama_service_sends_a_local_model_directory_as_a_name(
    tmp_path: Path, tripwire: _ImportTripwire
) -> None:
    model_dir = _fabricate_untrusted_model_dir(tmp_path)
    daemon = _OllamaDaemon()
    service = EmbeddingService(model=str(model_dir), base_url="http://ollama.test:11434")
    service._client = daemon.client()
    service._async_client = daemon.async_client()
    try:
        results = [
            service.embed("a"),
            service.embed_batch(["b", "c"]),
            await service.aembed("d"),
            await service.aembed_query("e"),
            await service.aembed_batch(["f"]),
        ]
    finally:
        service._client._client.close()
        await service._async_client._client.aclose()

    assert results == [VECTOR, [VECTOR, VECTOR], VECTOR, VECTOR, [VECTOR]]
    assert [request["model"] for request in daemon.requests] == [str(model_dir)] * 5
    assert tripwire.attempts == []
    _assert_fixture_never_imported(model_dir)


def test_tripwire_records_and_refuses_an_import(tripwire: _ImportTripwire) -> None:
    """The guard the two path tests rely on actually fires."""
    with pytest.raises(ImportError, match="#3718 tripwire"):
        __import__("sentence_transformers")

    assert tripwire.attempts == ["sentence_transformers"]


#: Run in a child interpreter: importing the library imports torch, which
#: writes process environment variables and starts native threads.
_LOAD_UNTRUSTED_DIRECTORY = """
import sys
from sentence_transformers import SentenceTransformer

try:
    SentenceTransformer(sys.argv[1], trust_remote_code=False)
except ValueError as exc:
    print(f"REFUSED: {exc}")
else:
    print("LOADED")
"""


def test_installed_sentence_transformers_refuses_local_custom_code(
    tmp_path: Path,
) -> None:
    """On a host with the ``local`` extra, the library itself holds the line.

    The version is checked before anything is loaded, so a vulnerable
    installation fails here without touching the fixture. CI does not
    install the extra, so this runs only where the library is present.
    """
    try:
        installed = Version(importlib.metadata.version("sentence-transformers"))
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sentence-transformers (the local extra) is not installed")
    assert installed >= REFUSING_RELEASE, (
        f"sentence-transformers {installed} imports a local model directory's "
        f"custom code despite trust_remote_code=False (GHSA-jhr6-gm9c-rqjv); "
        f"upgrade to >={REFUSING_RELEASE} before loading any model"
    )
    model_dir = _fabricate_untrusted_model_dir(tmp_path)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    environment = {
        name: value for name, value in os.environ.items() if name != "PYTHONPATH"
    }
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(tmp_path / "hf-home"),
    )

    child = subprocess.run(
        [sys.executable, "-c", _LOAD_UNTRUSTED_DIRECTORY, str(model_dir)],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.startswith("REFUSED:"), child.stdout + child.stderr
    # The refusal is the advisory's gate, naming the fixture's custom class.
    assert f"{PROBE_MODULE}.ProbeModule" in child.stdout
    assert "trust_remote_code=True" in child.stdout
    assert not (model_dir / MARKER).exists(), (
        "the fabricated model directory's custom module was imported"
    )
