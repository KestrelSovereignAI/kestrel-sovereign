"""Doc-visibility assertions for #1946.

Several tools documented their allowed enum/format only in the Python
docstring (invisible to the LLM at tool-selection time). These tests pin
the promoted values into the agent-visible ``@tool(description=...)`` so a
future edit can't silently drop them back into a docstring.
"""

from kestrel_sovereign.features.response_audit.feature import ResponseAuditFeature
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.constitution import ConstitutionFeature
from kestrel_sovereign.features.cli.feature import CliFeature
from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.features.web_search.feature import WebSearchFeature


def _desc(method) -> str:
    return method._tool_schema["description"]


def test_audit_enable_lists_modes():
    desc = _desc(ResponseAuditFeature.enable_audit)
    assert "warn" in desc and "strict" in desc


def test_approve_lists_scopes():
    desc = _desc(SecurityFeature.approve_request)
    assert "once" in desc and "session" in desc and "always" in desc


def test_constitution_documents_two_slot_grammar():
    desc = _desc(ConstitutionFeature.get_constitution)
    for kw in ("book", "amendment", "article", "search", "summary"):
        assert kw in desc


def test_git_tools_document_ref_path_conventions():
    diff = _desc(CliFeature.git_diff)
    assert ".." in diff and "pathspec" in diff and "allowed repo roots" in diff

    show = _desc(CliFeature.git_show_file)
    assert ".." in show and "pathspec" in show

    status = _desc(CliFeature.git_status)
    assert "allowed repo roots" in status

    merge_base = _desc(CliFeature.git_merge_base)
    assert "left_ref" in merge_base and "right_ref" in merge_base

    log = _desc(CliFeature.git_log)
    assert "100" in log


def test_bootstrap_descriptions_carry_conventions():
    add = _desc(BootstrapFeature.bootstrap_add)
    assert "data dir" in add and "BASENAME" in add

    remove = _desc(BootstrapFeature.bootstrap_remove)
    assert "basename" in remove

    list_desc = _desc(BootstrapFeature.bootstrap_list)
    for status in ("loaded", "partial", "not found", "skipped (budget)"):
        assert status in list_desc

    rename = _desc(BootstrapFeature.rename_agent)
    assert "1-64" in rename


def test_heartbeat_aliases_steer_to_canonical():
    assert "use health instead" in _desc(HealthFeature.heartbeat_check_alias)
    assert "use health_history instead" in _desc(HealthFeature.heartbeat_status_alias)
    assert "use health_interval instead" in _desc(HealthFeature.heartbeat_interval_alias)


def test_web_search_description_documents_max_results_and_disabled():
    desc = _desc(WebSearchFeature.search)
    assert "max_results" in desc
    assert "TAVILY_API_KEY" in desc
