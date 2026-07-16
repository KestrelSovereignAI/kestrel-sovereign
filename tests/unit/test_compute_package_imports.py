"""Cold-import contracts for the compute package boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


_EXPECTED_EXPORTS = [
    "ComputeFeature",
    "ComputePolicy",
    "ComputeScript",
    "DenialResponse",
    "ExecutionRecord",
    "ScriptState",
    "SecurityFinding",
    "SuggestedFix",
    "calculate_risk_score",
    "ScriptStore",
    "ScriptSigner",
    "ScriptAnalyzer",
    "AnalysisResult",
    "analyze_script",
    "DestructiveOperationPolicy",
    "rewrite_script_for_safety",
    "TrashManager",
    "TrashItem",
    "get_trash_manager",
    "ComputeSecurityHook",
    "ComputeDebugHook",
    "BaseExecutor",
    "ExecutionError",
    "ExecutionTimeoutError",
    "UvExecutor",
    "DockerExecutor",
    "LocalExecutor",
]


def _run_in_fresh_interpreter(source: str) -> subprocess.CompletedProcess[str]:
    """Run an import probe without inheriting pytest's populated module cache."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_probe_passed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr


def test_compute_package_and_models_import_without_runtime_graph() -> None:
    """Pure submodules must not load stores, hooks, or execution backends."""
    result = _run_in_fresh_interpreter(
        """
        import importlib
        import shutil
        import subprocess
        import sys

        def reject_docker_probe(*args, **kwargs):
            raise AssertionError(f"unexpected Docker probe: {args!r} {kwargs!r}")

        shutil.which = reject_docker_probe
        subprocess.run = reject_docker_probe

        package_name = "kestrel_sovereign.features.compute"
        models_name = f"{package_name}.models"

        importlib.import_module(package_name)
        loaded = {
            name
            for name in sys.modules
            if name == package_name or name.startswith(f"{package_name}.")
        }
        assert loaded == {package_name}, loaded

        models = importlib.import_module(models_name)
        assert models.ComputeScript.__module__ == models_name
        loaded = {
            name
            for name in sys.modules
            if name == package_name or name.startswith(f"{package_name}.")
        }
        assert loaded == {package_name, models_name}, loaded
        """
    )

    _assert_probe_passed(result)


def test_compute_presenters_import_without_runtime_graph() -> None:
    """Presenter imports must not wake feature, store, or executor modules."""
    result = _run_in_fresh_interpreter(
        """
        import importlib
        import shutil
        import subprocess
        import sys

        def reject_docker_probe(*args, **kwargs):
            raise AssertionError(f"unexpected Docker probe: {args!r} {kwargs!r}")

        shutil.which = reject_docker_probe
        subprocess.run = reject_docker_probe

        package_name = "kestrel_sovereign.features.compute"
        models_name = f"{package_name}.models"
        presenters_name = f"{package_name}.presenters"

        presenters = importlib.import_module(presenters_name)
        assert presenters.present_script_list.__module__ == presenters_name
        loaded = {
            name
            for name in sys.modules
            if name == package_name or name.startswith(f"{package_name}.")
        }
        assert loaded == {package_name, models_name, presenters_name}, loaded
        """
    )

    _assert_probe_passed(result)


def test_model_reexport_is_lazy() -> None:
    """A lightweight public name must load its defining module only."""
    result = _run_in_fresh_interpreter(
        """
        import sys

        from kestrel_sovereign.features.compute import ComputePolicy

        package_name = "kestrel_sovereign.features.compute"
        loaded = {
            name
            for name in sys.modules
            if name == package_name or name.startswith(f"{package_name}.")
        }
        assert loaded == {package_name, f"{package_name}.models"}, loaded
        assert ComputePolicy.__module__ == f"{package_name}.models"
        """
    )

    _assert_probe_passed(result)


def test_all_is_a_literal_for_static_star_import_consumers() -> None:
    """Mypy must be able to discover every name exported by ``import *``."""
    package_file = (
        Path(__file__).parents[2]
        / "kestrel_sovereign"
        / "features"
        / "compute"
        / "__init__.py"
    )
    tree = ast.parse(package_file.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, (ast.List, ast.Tuple))
    assert [ast.literal_eval(element) for element in value.elts] == _EXPECTED_EXPORTS


def test_reload_discards_cached_exports() -> None:
    """Reload must bind exports from their current defining modules."""
    result = _run_in_fresh_interpreter(
        """
        import importlib

        import kestrel_sovereign.features.compute as compute
        from kestrel_sovereign.features.compute import models

        original = compute.ComputePolicy
        replacement = object()
        models.ComputePolicy = replacement

        reloaded = importlib.reload(compute)

        assert reloaded is compute
        assert compute.ComputePolicy is replacement
        assert compute.ComputePolicy is not original
        """
    )

    _assert_probe_passed(result)


def test_static_introspection_tradeoff_is_explicit_and_bounded() -> None:
    """Cold static inspection omits values until dynamic access resolves them."""
    result = _run_in_fresh_interpreter(
        f"""
        import inspect

        import kestrel_sovereign.features.compute as compute

        expected = {_EXPECTED_EXPORTS!r}
        assert compute.__all__ == expected
        assert not set(expected).intersection(vars(compute))
        cold_static = dict(inspect.getmembers_static(compute))
        assert not set(expected).intersection(cold_static)

        dynamic = dict(inspect.getmembers(compute))
        assert all(name in dynamic for name in expected)
        warm_static = dict(inspect.getmembers_static(compute))
        assert all(name in warm_static for name in expected)
        """
    )

    _assert_probe_passed(result)


def test_historical_compute_reexports_remain_compatible() -> None:
    """All package-level names still resolve to their canonical definitions."""
    result = _run_in_fresh_interpreter(
        """
        import importlib

        import kestrel_sovereign.features.compute as compute
        from kestrel_sovereign.features.compute import *

        star_exports = {name: globals()[name] for name in compute.__all__}

        expected_groups = {
            "kestrel_sovereign.features.compute.feature": ("ComputeFeature",),
            "kestrel_sovereign.features.compute.models": (
                "ComputePolicy",
                "ComputeScript",
                "DenialResponse",
                "ExecutionRecord",
                "ScriptState",
                "SecurityFinding",
                "SuggestedFix",
                "calculate_risk_score",
            ),
            "kestrel_sovereign.features.compute.script_store": ("ScriptStore",),
            "kestrel_sovereign.features.compute.script_signer": ("ScriptSigner",),
            "kestrel_sovereign.features.compute.script_analyzer": (
                "ScriptAnalyzer",
                "AnalysisResult",
                "analyze_script",
            ),
            "kestrel_sovereign.features.compute.destructive_policy": (
                "DestructiveOperationPolicy",
                "rewrite_script_for_safety",
            ),
            "kestrel_sovereign.features.compute.trash_manager": (
                "TrashManager",
                "TrashItem",
                "get_trash_manager",
            ),
            "kestrel_sovereign.features.compute.security_hook": (
                "ComputeSecurityHook",
                "ComputeDebugHook",
            ),
            "kestrel_sovereign.features.compute.executors": (
                "BaseExecutor",
                "ExecutionError",
                "ExecutionTimeoutError",
                "UvExecutor",
                "DockerExecutor",
                "LocalExecutor",
            ),
        }
        expected = {}
        for module_name, names in expected_groups.items():
            module = importlib.import_module(module_name)
            for name in names:
                expected[name] = getattr(module, name)

        assert compute.__all__ == list(expected)
        assert star_exports == expected
        assert all(name in dir(compute) for name in expected)
        for name, canonical in expected.items():
            assert getattr(compute, name) is canonical, name
            assert vars(compute)[name] is canonical, name

        try:
            compute.not_a_public_export
        except AttributeError as error:
            assert "not_a_public_export" in str(error)
        else:
            raise AssertionError("unknown export did not raise AttributeError")
        """
    )

    _assert_probe_passed(result)
