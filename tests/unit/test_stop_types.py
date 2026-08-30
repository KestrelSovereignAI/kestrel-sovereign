"""Canonical cooperative Stop vocabulary (#3137)."""

import json

from kestrel_sovereign.multi_agent.process_manager import ProcessManager
from kestrel_sovereign.stop import StopScope


def test_stop_scope_has_stable_lowercase_wire_values() -> None:
    assert {scope.value for scope in StopScope} == {
        "host",
        "agent",
        "turn",
        "tool_call",
    }
    for scope in StopScope:
        assert StopScope(scope.value) is scope
        assert json.loads(json.dumps(scope)) == scope.value


def test_process_termination_cannot_be_wired_through_stop_vocabulary() -> None:
    assert not hasattr(ProcessManager, "stop_all")
    assert not hasattr(ProcessManager, "stop_agent")
    assert callable(ProcessManager.terminate_agent)
    assert callable(ProcessManager.terminate_all)
