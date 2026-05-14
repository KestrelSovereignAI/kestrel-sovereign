"""Guardrails for the current tools architecture docs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DOCS = [
    PROJECT_ROOT / "docs/architecture/tools/AGENT_TOOLS.md",
    PROJECT_ROOT / "docs/architecture/tools/AGENT_TOOLS_ARCHITECTURE.md",
    PROJECT_ROOT / "docs/architecture/tools/AGENT_TOOLS_IMPLEMENTATION.md",
]


def _combined_text() -> str:
    return "\n".join(path.read_text() for path in TOOLS_DOCS)


def test_tools_docs_no_longer_carry_deprecation_banners():
    text = _combined_text()

    assert "DEPRECATED" not in text
    assert "removed architecture" not in text
    assert "#1047" not in text


def test_tools_docs_do_not_reference_removed_tool_architecture_files():
    text = _combined_text()

    for removed in [
        "AgentToolMixin",
        "kestrel_agent_tools.py",
        "/tools/web_search.py",
        "/tools/feedback_tool.py",
        "tools/registry.py",
    ]:
        assert removed not in text


def test_tools_docs_name_current_sdk_and_feature_package_contracts():
    text = _combined_text()

    for required in [
        "kestrel_sdk.features.base.tool",
        "kestrel_sdk.tools.base.ToolCategory",
        "kestrel_sdk.tools.base.ToolSchema",
        "kestrel_sdk.tools.result.ToolResult",
        "kestrel_sovereign.features",
        "kestrel_sovereign.cloud_providers",
        "kestrel_sovereign.voice_providers",
        "kestrel_sovereign.storage_providers",
    ]:
        assert required in text


def test_tools_docs_crosslink_core_feature_framework_and_sdk_readme():
    text = _combined_text()

    assert "../core/FEATURE_AGENT_FRAMEWORK.md" in text
    assert "kestrel-sovereign-sdk#readme" in text
