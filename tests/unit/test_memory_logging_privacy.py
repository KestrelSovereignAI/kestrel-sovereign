"""Privacy contracts for routine memory/request diagnostics (#2332)."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_request_logs_do_not_embed_history_or_response_snippets():
    agent_source = (ROOT / "kestrel_sovereign/kestrel_agent.py").read_text()
    orchestrator_source = (
        ROOT / "kestrel_sovereign/agent/orchestrator_engine.py"
    ).read_text()

    assert "[SESSION-DEBUG]" not in agent_source
    assert "history[0].get('content'" not in agent_source
    assert "response.content[:150]" not in agent_source
    assert "final_content[:300]" not in orchestrator_source
    assert "response[:300]" not in orchestrator_source
    assert "result_json[:200]" not in orchestrator_source
