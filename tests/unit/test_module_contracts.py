"""Structural contracts for the modular runtime boundary scaffold."""

from pathlib import Path
import inspect
import tomllib

from kestrel_sovereign.features import FEATURE_ENTRY_POINT_GROUP
from kestrel_sovereign.features.base import Feature


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULAR_RUNTIME_DOC = PROJECT_ROOT / "docs/architecture/core/MODULAR_RUNTIME.md"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _doc_text() -> str:
    return MODULAR_RUNTIME_DOC.read_text()


def test_modular_runtime_doc_exists_and_names_core_contract_terms():
    text = _doc_text()

    assert "# Modular Runtime Boundary" in text
    for term in [
        "Runtime Kernel",
        "Module",
        "Stable Module ID",
        "Provided Capabilities",
        "Required Capabilities",
        "Owned State",
        "Startup Hooks",
        "Router Contribution",
        "Export and Import Fragments",
    ]:
        assert f"## {term}" in text


def test_modular_runtime_doc_points_to_current_entry_point_groups():
    text = _doc_text()
    pyproject = tomllib.loads(PYPROJECT.read_text())
    entry_points = pyproject["project"]["entry-points"]

    expected_groups = {
        FEATURE_ENTRY_POINT_GROUP,
        "kestrel_sovereign.cloud_providers",
        "kestrel_sovereign.voice_providers",
        "kestrel_sovereign.storage_providers",
    }

    assert expected_groups <= set(entry_points)
    for group in expected_groups:
        assert group in text


def test_feature_base_exposes_documented_module_contribution_seams():
    expected_members = {
        "initialize",
        "shutdown",
        "on_enable",
        "on_disable",
        "on_remove",
        "get_hooks",
        "get_router",
        "post_all_features_loaded",
        "get_config",
        "set_config",
    }

    for member in expected_members:
        assert hasattr(Feature, member)

    assert isinstance(Feature.promote_tools_on_startup, property)
    assert isinstance(Feature.config_schema, property)


def test_documented_lifecycle_seams_are_subclass_extension_points():
    for member in [
        "initialize",
        "shutdown",
        "on_enable",
        "on_disable",
        "on_remove",
        "get_hooks",
        "get_router",
        "post_all_features_loaded",
        "get_config",
        "set_config",
    ]:
        attr = inspect.getattr_static(Feature, member)
        assert inspect.isfunction(attr)


def test_modular_runtime_doc_preserves_core_to_feature_dependency_direction():
    text = _doc_text()

    assert "Core must not import" in text
    assert "kestrel_feature_*" in text
    assert "the framework does not depend on optional packages" in text
