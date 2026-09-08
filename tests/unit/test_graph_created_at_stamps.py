"""Every graph ``created_at`` stamp is a UTC ISO-8601 string (#3256).

``storage/async_graph_store.py``'s module contract requires every persisted
``properties.created_at`` to be ``datetime.now(timezone.utc).isoformat()``:
range filters, ordering and the scoped EPHEMERAL purge compare it
lexicographically, so a naive local-time stamp sorts hours away from the UTC
watermark on any non-UTC host. Two writers stamped ``datetime.now()`` bare;
this module pins both and scans the package for the literal naive shapes
(``datetime.now().isoformat()``, ``datetime.utcnow().isoformat()``,
``str(datetime.now())``) assigned to a ``created_at`` key, so a third copy of
that shape cannot land silently. A stamp routed through a variable is
outside the scan; the two site tests, not the scan, are the proof for a
given writer.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The feature packages live under the core package; there is no second root.
SCAN_ROOT = ROOT / "kestrel_sovereign"

# A naive stamp assigned to a created_at key, allowing whitespace/newlines:
# a bare now(), a naive utcnow(), or str() of either.
_NAIVE_CREATED_AT = re.compile(
    r"(?:[\"']created_at[\"']\s*:|\bcreated_at\s*=)\s*"
    r"(?:datetime\.(?:now|utcnow)\(\)\.isoformat\(\)|str\(datetime\.(?:now|utcnow)\(\)\))"
)


def _sources():
    sources = sorted(SCAN_ROOT.rglob("*.py"))
    # The register must be able to fail: an empty walk is a broken scan,
    # not a clean tree.
    assert len(sources) > 500, len(sources)
    return sources


@pytest.fixture
def new_york_clock(monkeypatch):
    """A non-UTC process zone, undone in the right order: ``monkeypatch``
    restores ``TZ`` at teardown but ``tzset()`` is what the C library reads,
    and on glibc the cached zone outlives the variable."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_scanner_recognises_the_removed_shape():
    """Positive control: the exact text this ticket removed is caught, and
    the sibling naive shapes with it."""
    assert _NAIVE_CREATED_AT.search('                    "created_at": datetime.now().isoformat(),')
    assert _NAIVE_CREATED_AT.search('created_at=datetime.now().isoformat()')
    assert _NAIVE_CREATED_AT.search('"created_at": datetime.utcnow().isoformat()')
    assert _NAIVE_CREATED_AT.search('"created_at": str(datetime.now())')
    assert not _NAIVE_CREATED_AT.search('"created_at": datetime.now(timezone.utc).isoformat()')


def test_no_graph_created_at_is_stamped_naive():
    offenders = [
        f"{path.relative_to(ROOT)}:{text[:m.start()].count(chr(10)) + 1}"
        for path in _sources()
        for text in [path.read_text(encoding="utf-8", errors="replace")]
        for m in _NAIVE_CREATED_AT.finditer(text)
    ]
    assert offenders == [], offenders


def _assert_utc_iso(value: str):
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, value
    assert value.endswith("+00:00"), value


@pytest.mark.asyncio
async def test_spawned_agent_node_created_at_is_utc(monkeypatch, tmp_path, new_york_clock):
    """The spawn tool's SovereignAgent node carries a UTC stamp, whatever the
    host's local zone is."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import kestrel_sovereign.inception_service as inception
    import kestrel_sovereign.kestrel_agent as agent_module
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    monkeypatch.setattr(agent_module, "TRUSTED_AGENTS_DIR", str(tmp_path / "trusted"))
    monkeypatch.setattr(inception, "generate_kestrel_identity", lambda: ({"id": "did:kestrel:child"}, object()))
    monkeypatch.setattr(inception, "save_kestrel_identity", lambda doc, keys, path: None)

    stored = []
    stub = SimpleNamespace(storage=SimpleNamespace(graph_store=SimpleNamespace(add_node=AsyncMock(side_effect=stored.append))))
    result = await KestrelAgent.create_trusted_agent(stub, "child")

    assert "did:kestrel:child" in result
    assert len(stored) == 1 and stored[0].node_type == "SovereignAgent"
    _assert_utc_iso(stored[0].properties["created_at"])


@pytest.mark.asyncio
async def test_sovereignty_receipt_created_at_is_utc(monkeypatch, new_york_clock):
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature
    from kestrel_sovereign.storage.providers.base import StorageResult, StorageTier

    stored = []
    storage = MagicMock()
    storage.create_backup_blob = AsyncMock(return_value=b"backup-bytes")
    storage.record_backup_artifact = AsyncMock(return_value="hash123")
    storage.add_node = AsyncMock(side_effect=lambda node: stored.append(deepcopy(node)))
    wallet = MagicMock(); wallet.can_afford.return_value = True; wallet.transfer = AsyncMock()
    agent = SimpleNamespace(agent_id="did:kestrel:test", features={}, storage=storage, wallet=wallet)
    adapter = MagicMock()
    adapter.store_content.return_value = StorageResult(
        content_hash="hash123", cid="bafybackup", tier=StorageTier.IPFS, provider="ipfs",
        encrypted=True, encryption_key_hash="keyhash123", size_bytes=12,
    )
    with patch("kestrel_sovereign.features.sovereignty.feature.FilecoinAdapter", return_value=adapter):
        await SovereigntyFeature(agent).export_sovereignty(storage_tier="ipfs", encrypt=True)

    assert len(stored) == 1 and stored[0].node_type == "sovereignty_receipt"
    _assert_utc_iso(stored[0].properties["created_at"])
