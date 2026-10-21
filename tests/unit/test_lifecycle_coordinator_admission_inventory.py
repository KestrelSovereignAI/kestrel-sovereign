"""All canonical lifecycle admissions use receipts, including coordinators."""
import ast
from pathlib import Path


def test_direct_lifecycle_creators_publish_an_admission_receipt():
    root = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
    calls = []
    for relative in ("server.py", "signals/dispatcher.py", "multi_agent/agent_manager.py"):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_owned_task":
                calls.append((relative, node.lineno))
                assert any(keyword.arg == "admission" for keyword in node.keywords), (relative, node.lineno)
    assert calls
    tree = ast.parse((root / "kestrel_agent.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "await_agent_shutdown_completion")
    owners = [node for node in ast.walk(helper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_owned_task"]
    assert owners and all(any(keyword.arg == "admission" for keyword in node.keywords) for node in owners)
    # Structural inventory only; actual accepted-Task and physical-lock
    # transitions have separate runtime/native tests. Not every raw/alias call.
