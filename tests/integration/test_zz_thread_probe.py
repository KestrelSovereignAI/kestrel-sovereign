import pytest
from tests.integration.test_sovereignty_lighthouse_attribution import _host, AGENT_A, _status, MESSAGES
from tests.integration.test_sovereignty_lighthouse_attribution import kubo, lighthouse, data_key  # noqa
from kestrel_sovereign.features.sovereignty.feature import SovereigntyFeature


@pytest.mark.asyncio
async def test_probe_status(tmp_path, monkeypatch, kubo, lighthouse, data_key):
    async with _host(tmp_path / "h", monkeypatch, did=AGENT_A, lighthouse=False) as (agent, raw):
        await _status(agent)


@pytest.mark.asyncio
async def test_probe_export_local(tmp_path, monkeypatch, kubo, lighthouse, data_key):
    async with _host(tmp_path / "h", monkeypatch, did=AGENT_A, lighthouse=False) as (agent, raw):
        for role, content in MESSAGES:
            await raw.add_conversation(role, content, metadata={})
        await SovereigntyFeature(agent).export_sovereignty(storage_tier="ipfs", encrypt=True)
