"""Explicit native Task sites must publish an acceptance receipt before callbacks."""
import ast
from pathlib import Path


def test_explicit_native_owned_task_admissions_have_original_receipts():
    package = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
    missing, inspected = [], 0
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "create_owned_task":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if "retirement_resource" not in keywords:
                continue
            inspected += 1
            if "admission" not in keywords:
                missing.append(f"{path.relative_to(package)}:{node.lineno}")
    assert inspected >= 20, "inventory did not inspect the native Task admission surface"
    assert not missing, "Native Task acceptance lacks original receipt: " + ", ".join(missing)
