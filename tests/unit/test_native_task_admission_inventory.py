"""Explicit native Task sites must publish an acceptance receipt before callbacks."""
import ast
from pathlib import Path

_RETIREMENT_ENTRIES = {
    "execution_custody.py": {("close_native_execution_stream", "create_owned_task")},
    "hold/state.py": {( "_close_postgres_preflight_backends", "create_owned_task")},
    "kestrel_agent.py": {("KestrelAgent._run_durable_shutdown_tail._bounded", "create_owned_task")},
    "llm/anthropic_adapter.py": {("_anthropic_stream_with_retry.close_acquired", "create_owned_task")},
    "llm/lifecycle.py": {("start_llm_close_task", "create_owned_task")},
    "llm/cancellation.py": {("await_or_cancelled", "OwnedOperation")},
    "a2a/task_manager.py": {("TaskManager.close", "create_owned_task")},
    "host_features/context.py": {("_close_partial_host_resources", "create_owned_task")},
    "streams/tap.py": {("_TapResponseOwner.aclose", "create_owned_task")},
    "artifacts/preview.py": {
        ("DockerArtifactRenderer._command_native", "OwnedOperation"),
        ("DockerArtifactRenderer._cleanup", "create_owned_task"),
    },
    "visual_computer/docker.py": {("DockerBoundary._invoke", "OwnedOperation")},
    "visual_computer/local_worker.py": {("_LocalWorkerResources.close", "create_owned_task")},
    "mcp_server/publication.py": {("NativeMCPPublication.close", "OwnedOperation")},
    "server.py": {("_shutdown_phoenix", "create_owned_task")},
    "stop/invocation.py": {("DistributedInvocationRegistry.close", "create_owned_task")},
    "features/isolated_runtime.py": {
        ("ProxyFeature._admit_terminal_cleanup", "start"),
        ("ProxyFeature._complete_terminal_cleanup", "start"),
    },
    "a2a/native_inprocess_http.py": {
        ("_Exchange.close", "create_owned_task"),
        ("_Exchange.retire_request", "retire"),
    },
}


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
    missing = []
    for relative, required in _RETIREMENT_ENTRIES.items():
        census = _AdmissionPhases()
        census.visit(ast.parse((package / relative).read_text(), filename=relative))
        phased = {(function, callee) for function, callee, phase, _ in census.calls if phase}
        if not required <= phased:
            missing.append(f"{relative}: retirement admitting frame lost its phase: {required - phased}")
    assert not missing, "\n".join(missing)


def test_only_facade_stop_callers_select_cleanup_admission():
    package = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
    tree = ast.parse((package / "features/isolated_runtime.py").read_text())
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_await_owned_facade_lifecycle_operation":
            continue
        operation = node.args[0]
        assert isinstance(operation, ast.Call) and isinstance(operation.func, ast.Attribute)
        method = operation.func.attr
        cleanup = next((keyword.value for keyword in node.keywords if keyword.arg == "cleanup"), None)
        if method == "stop":
            assert isinstance(cleanup, ast.Constant) and cleanup.value is True, (
                f"facade stop at {node.lineno} lost physical cleanup admission")
        else:
            assert cleanup is None, f"facade {method} at {node.lineno} bypassed Work"
        seen[method] = seen.get(method, 0) + 1
    assert seen == {"stop": 4, "_call_exact_external_ingress_lifecycle": 2, "start": 1}


def test_retirement_inventory_detects_removed_cleanup_admission_guards():
    """Structural wiring mutations, not a native container-retirement claim."""
    package = Path(__file__).resolve().parents[2] / "kestrel_sovereign"
    for relative, required in _RETIREMENT_ENTRIES.items():
        tree = ast.parse((package / relative).read_text(), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "bind_execution_cleanup":
                    node.func.id = "_removed_cleanup_phase"
        census = _AdmissionPhases()
        census.visit(tree)
        phased = {(function, callee) for function, callee, phase, _ in census.calls if phase}
        assert not required <= phased, f"{relative}: guard mutation escaped the admitting-frame inventory"
