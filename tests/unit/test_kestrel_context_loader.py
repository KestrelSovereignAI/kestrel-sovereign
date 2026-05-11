import json

from kestrel_sovereign.kestrel_context import KestrelContextLoader


def write_project_file(project_path, relative_path, content):
    path = project_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_load_context_uses_current_standard_doc_paths(tmp_path):
    write_project_file(tmp_path, "AGENTS.md", "# Agent instructions")
    write_project_file(tmp_path, "PROJECT_STATUS.md", "# Current state")
    write_project_file(tmp_path, "README.md", "# Kestrel")
    write_project_file(tmp_path, "docs/PROJECT_VISION.md", "# Vision")
    write_project_file(
        tmp_path,
        "docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md",
        "# Feature framework",
    )
    write_project_file(
        tmp_path,
        "docs/architecture/security/PRIVACY_MODES.md",
        "# Privacy modes",
    )
    write_project_file(
        tmp_path,
        "docs/architecture/security/CRYPTOGRAPHIC_ANCHORING.md",
        "# Anchoring",
    )

    loader = KestrelContextLoader(str(tmp_path))

    context = loader.load_context()
    prompt = loader.create_prompt("Summarize the project")

    assert context["AGENTS.md"] == "# Agent instructions"
    assert "docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md" in context
    assert "docs/architecture/security/PRIVACY_MODES.md" in context
    assert "docs/architecture/security/CRYPTOGRAPHIC_ANCHORING.md" in context
    assert "docs/architecture/FEATURE_AGENT_FRAMEWORK.md" not in context
    assert "## Mode: Platform Extension" not in prompt
    assert "Summarize the project" in prompt


def test_platform_extension_mode_loads_explicit_extension_docs(tmp_path):
    write_project_file(tmp_path, "AGENTS.md", "# Agent instructions")
    write_project_file(tmp_path, "extension/PRD.md", "# Extension PRD")
    write_project_file(tmp_path, "extension/schema.sql", "create table events ();")

    loader = KestrelContextLoader(
        str(tmp_path),
        platform_extension=True,
        extension_docs=("extension/PRD.md", "extension/schema.sql"),
    )

    context = loader.load_context()
    prompt = loader.create_prompt()

    assert context["extension/PRD.md"] == "# Extension PRD"
    assert context["extension/schema.sql"] == "create table events ();"
    assert "## Mode: Platform Extension" in prompt
    assert "- You are working on the platform extension" in prompt
    assert "- Extension PRD and database schema have been loaded" in prompt


def test_extract_mcp_config_from_agents_json_block(tmp_path):
    mcp_config = {"mcpServers": {"memory": {"command": "uvx", "args": ["server"]}}}
    write_project_file(
        tmp_path,
        "AGENTS.md",
        "Use this MCP config:\n```json\n"
        + json.dumps(mcp_config)
        + "\n```\n",
    )

    loader = KestrelContextLoader(str(tmp_path))
    loader.load_context()

    assert loader.extract_mcp_config() == mcp_config
