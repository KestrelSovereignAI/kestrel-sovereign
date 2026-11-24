"""Explicit native Task sites must publish an acceptance receipt before callbacks."""
import ast
from pathlib import Path


_WORK = {
    ("current_runpod_quote.py", "_CurrentRunpodQuoteMutation.commit_quote"),
    ("current_visual_step.py", "_CurrentVisualStepMutation.commit_step"),
    ("db/sqlite.py", "SQLiteBackend._acquire_connection"),
    ("db/sqlite.py", "SQLiteBackend._fetch_on_connection"),
    ("memory_retriever.py", "MemoryRetriever._schedule_access_update"),
    ("original_acquisition_result.py", "_OriginalAcquisitionResult._transition"),
    ("original_action_result.py", "_OriginalActionResultCustody.not_dispatched"),
    ("original_tool_effect_finish.py", "_OriginalToolEffectFinish.finish"),
    ("privacy_wrapper.py", "ReentrantTransitionLock.start_owned_native_work"),
    ("privacy_wrapper.py", "PrivacyEnforcingStorage.store_deidentified_records"),
}
# These private helpers only execute inside the named admitting phase. Their
# production callsites are checked below; a new caller must preserve that phase.
_INHERITED = {
    ("db/original_result_custody.py", "_drain_original_results"),
    ("db/sqlite.py", "_retain_aiosqlite_close"),
    ("db/sqlite.py", "_close_acquired_sqlite_connection"),
    ("db/sqlite.py", "_finish_aiosqlite_connection_close"),
}


class _AdmissionPhases(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.phase = False
        self.calls = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        original = self.phase
        # Defining a nested function in cleanup is not evidence that a later
        # caller will execute it there. Classify its admitting frame itself.
        self.phase = False
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        self.phase = original

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node):
        original = self.phase
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == "bind_execution_cleanup":
                    self.phase = True
                elif call.func.id == "_disable_owned_task_lending" and call.args:
                    self.phase |= isinstance(call.args[0], ast.Constant) and call.args[0].value is True
        self.generic_visit(node)
        self.phase = original

    visit_AsyncWith = visit_With

    def visit_Call(self, node):
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        self.calls.append((".".join(self.stack), name, self.phase, node.lineno))
        self.generic_visit(node)


def test_storage_native_creators_have_an_explicit_work_or_retirement_phase():
    storage = Path(__file__).resolve().parents[2] / "kestrel_sovereign" / "storage"
    creators = {"create_owned_task", "OwnedOperation", "await_owned_operation"}
    inherited_names = {function.split(".")[-1] for _, function in _INHERITED}
    missing, inspected = [], 0
    for path in sorted(storage.rglob("*.py")):
        relative = path.relative_to(storage).as_posix()
        census = _AdmissionPhases()
        census.visit(ast.parse(path.read_text(), filename=str(path)))
        for function, callee, phase, line in census.calls:
            key = relative, function
            if callee in creators:
                inspected += 1
                if not phase and key not in _WORK and key not in _INHERITED:
                    missing.append(f"{relative}:{line} unclassified {function}/{callee}")
            if callee in inherited_names and not phase and key not in _INHERITED:
                missing.append(f"{relative}:{line} calls {callee} outside retirement phase")
    assert inspected >= 25, "storage creator census lost native admission sites"
    assert not missing, "Native phase inventory requires explicit review: " + "; ".join(missing)


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


def test_nonstorage_physical_retirement_admits_inside_its_phase():
    package = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
    entries = {
        "hold/state.py": {( "_close_postgres_preflight_backends", "create_owned_task")},
        "kestrel_agent.py": {("KestrelAgent._run_durable_shutdown_tail._bounded", "create_owned_task")},
        "a2a/native_inprocess_http.py": {
            ("_Exchange.close", "create_owned_task"),
            ("_Exchange.retire_request", "retire"),
        },
    }
    for relative, required in entries.items():
        census = _AdmissionPhases()
        census.visit(ast.parse((package / relative).read_text(), filename=relative))
        phased = {(function, callee) for function, callee, phase, _ in census.calls if phase}
        assert required <= phased, f"{relative}: retirement admitting frame lost its phase: {required - phased}"
