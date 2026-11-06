"""The real Kite command path owns durable admission before native fact writes."""

import pytest

from kestrel_sovereign.agent.invocation import request_provenance
from kestrel_sovereign.endpoints.agent import _kite_runtime_observation
from tests.shared.booted_agent import booted_agent


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_kite_native_save_and_delete_have_original_journal_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv('KESTREL_HOME', str(tmp_path))
    monkeypatch.setenv('KESTREL_DB_PATH', str(tmp_path))
    monkeypatch.setenv('KESTREL_KITE_RELEASE_EVIDENCE', '1')
    monkeypatch.setenv('KESTREL_KITE_RELEASE_EVIDENCE_ROOT', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'kestrel.toml').write_text('[llm]\nroute_priority=[]\n')
    async with booted_agent(tmp_path, name='OwnedKiteJournal',
            provenance='test:3679-kite-native-journal', features={'MemoryAgencyFeature'}) as agent:
        provenance = request_provenance(source_kind='http_request', source_locator='/agent/invoke')
        with pytest.raises(RuntimeError, match='Kite runtime command did not return a successful typed result'):
            await _kite_runtime_observation(agent,
                request_id='owned-kite-delete-missing', provenance=provenance,
                request={'operation': 'delete', 'nonce': '3' * 64})
        operation, observation = await _kite_runtime_observation(agent,
            request_id='owned-kite-save', provenance=provenance,
            request={'operation': 'save', 'nonce': '1' * 64,
                     'value': 'kite-evidence-owned-native-journal-region'})
        assert (operation, observation) == ('save', {'fact_write_count': 1})
        operation, observation = await _kite_runtime_observation(agent,
            request_id='owned-kite-delete', provenance=provenance,
            request={'operation': 'delete', 'nonce': '2' * 64})
        assert (operation, observation) == ('delete', {'fact_delete_count': 1})
        assert await agent.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_claims') == [
                ('RETURNED_ERROR',), ('RETURNED_OK',), ('RETURNED_OK',)]
        assert await agent.tool_effect_store.database.fetchall(
            'SELECT state FROM tool_effect_operations') == [('CLOSED',)] * 3
        assert not agent._active_request_ids
