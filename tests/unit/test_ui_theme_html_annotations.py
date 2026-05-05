"""Verify HTML annotations match the legacy theme file (epic #986, #990).

Two regression catchers:

1. Every `data-label-key="<key>"` and `data-label-attr-X="<key>"` in
   `index.html` references a key that exists in `themes/legacy/en.toml`.
   Drift between the HTML and the theme file would silently produce
   un-themed labels in production — this test fails fast.

2. For [data-label-key] elements, the inline text content matches the
   legacy theme value byte-for-byte. The legacy theme is supposed to be
   visually equivalent to the un-hydrated rendering — if someone edits
   the HTML or the TOML without keeping them in lockstep, this catches
   it.

Note: lightweight regex-based extraction. We're not parsing HTML in full.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kestrel_sovereign.ui.theme_loader import load_theme

INDEX_HTML = (
    Path(__file__).resolve().parents[2]
    / "kestrel_sovereign"
    / "static"
    / "index.html"
)

# Match data-label-key="..." and data-label-attr-X="..." (X = title, placeholder, alt, aria-label, ...)
_KEY_PATTERN = re.compile(r'\bdata-label-key="([^"]+)"')
_ATTR_PATTERN = re.compile(r'\bdata-label-attr-[a-z-]+="([^"]+)"')


@pytest.fixture(scope="module")
def html_text() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def legacy_labels() -> dict[str, str]:
    return load_theme("legacy", "en").labels


def test_index_html_has_annotations(html_text):
    """Sanity check: the annotation pass actually produced annotations."""
    keys = _KEY_PATTERN.findall(html_text)
    attrs = _ATTR_PATTERN.findall(html_text)
    assert len(keys) > 50, f"expected many data-label-key annotations, got {len(keys)}"
    assert len(attrs) > 5, f"expected several data-label-attr-* annotations, got {len(attrs)}"


def test_every_data_label_key_exists_in_legacy_theme(html_text, legacy_labels):
    """Every textContent key referenced in HTML must exist in legacy/en.toml."""
    keys = set(_KEY_PATTERN.findall(html_text))
    missing = sorted(k for k in keys if k not in legacy_labels)
    assert not missing, (
        f"data-label-key values referenced in index.html but missing from "
        f"legacy/en.toml: {missing}"
    )


def test_every_data_label_attr_value_exists_in_legacy_theme(html_text, legacy_labels):
    """Every attribute key referenced in HTML must exist in legacy/en.toml."""
    keys = set(_ATTR_PATTERN.findall(html_text))
    missing = sorted(k for k in keys if k not in legacy_labels)
    assert not missing, (
        f"data-label-attr-* values referenced in index.html but missing from "
        f"legacy/en.toml: {missing}"
    )


def test_inline_text_for_a_few_keys_matches_legacy_value(html_text, legacy_labels):
    """Inline text for a sample of theme keys must match the legacy theme.

    This is the byte-equivalence check for the legacy theme — if the
    HTML inline text drifts away from legacy/en.toml, the un-hydrated
    rendering will look different from the hydrated rendering, breaking
    the "legacy is invisible" promise.

    Spot-check: pick a handful of distinctive labels rather than every
    annotation; the broader check (every key exists) catches structural
    drift, this catches value drift on the high-signal labels.
    """
    samples = [
        ("tab_identity", "Identity"),
        ("sidebar_agents", "Agents"),
        ("chat_thinking", "Thinking..."),
        ("memories_title", "Knowledge Graph"),
        ("spawn_title", "Spawn Manager"),
        ("constitution_title", "Kestrel Constitution"),
        ("metrics_title", "Metrics Dashboard"),
        ("features_title", "Feature Store"),
        ("security_title", "Security Permissions"),
    ]
    for key, expected in samples:
        # 1. Legacy theme value matches what we expect
        assert legacy_labels.get(key) == expected, (
            f"legacy/en.toml: {key} = {legacy_labels.get(key)!r}, expected {expected!r}"
        )
        # 2. The inline HTML text for that key matches the legacy value.
        #    Find any element annotated with this key, capture its inline text.
        pattern = re.compile(
            r'data-label-key="' + re.escape(key) + r'"[^>]*>([^<]*)<',
            re.MULTILINE,
        )
        matches = pattern.findall(html_text)
        assert matches, f"no element with data-label-key={key!r} found in index.html"
        for inline in matches:
            assert inline == expected, (
                f"inline text drift for {key!r}: html={inline!r} "
                f"vs legacy={expected!r}"
            )


def test_attribute_inline_values_match_legacy(html_text, legacy_labels):
    """Sample check: attributes annotated with data-label-attr-* should also
    have their inline attribute value matching the legacy theme."""
    samples = [
        ("placeholder", "chat_input_placeholder", "Ask me anything or use !commands..."),
        ("title", "btn_send", "Send"),
        ("title", "btn_collapse", "Collapse"),
    ]
    for attr, key, expected in samples:
        assert legacy_labels.get(key) == expected, (
            f"legacy/en.toml: {key} = {legacy_labels.get(key)!r}, expected {expected!r}"
        )
        # Match: data-label-attr-<attr>="<key>" ... <attr>="<value>"
        # The two attributes are on the same element, so they appear close.
        # We allow any chars (including other attrs) between them as long as
        # we don't cross a tag boundary.
        pattern = re.compile(
            rf'data-label-attr-{attr}="{re.escape(key)}"[^>]*?{attr}="([^"]*)"'
        )
        match = pattern.search(html_text)
        assert match, (
            f"no element pairs data-label-attr-{attr}={key!r} with {attr}="
        )
        assert match.group(1) == expected, (
            f"attribute drift for {attr}={key!r}: "
            f"html={match.group(1)!r} vs legacy={expected!r}"
        )


def test_theme_js_is_loaded_in_index_html(html_text):
    """theme.js must be referenced in index.html or hydration never runs."""
    assert "js/theme.js" in html_text, "index.html must include js/theme.js"


def test_no_orphan_label_keys_in_legacy_theme(html_text, legacy_labels):
    """Inverse check: every theme-class key in legacy/en.toml should be
    referenced by the HTML. An orphan key suggests we lost an annotation
    somewhere or a key was added speculatively.

    NOTE: locale-class keys (loading_*, btn_*, ...) may be referenced in
    JS-injected content rather than the static HTML — those won't appear
    in this scan and shouldn't fail it. We only check theme-class keys
    that are clearly tied to specific UI surfaces.
    """
    surface_anchored_keys = {
        "tab_identity", "tab_chat", "tab_constitution", "tab_memories",
        "tab_tasks", "tab_sovereignty", "tab_resources", "tab_metrics",
        "tab_spawn", "tab_features", "tab_security",
        "sidebar_agents", "sidebar_conversations",
        "chat_history_title", "chat_welcome_message", "chat_thinking",
        "chat_input_placeholder",
        "constitution_title", "memories_title", "tasks_title",
        "tasks_view_tasks", "tasks_view_activity",
        "sovereignty_title", "sovereignty_tagline",
        "sovereignty_export_history", "sovereignty_local_cache",
        "sovereignty_db_explorer", "sovereignty_ipfs_network",
        "resources_title", "resources_agent_keys_title",
        "resources_user_keys_title", "resources_platform_title",
        "resources_wallet_title", "resources_usage_title",
        "metrics_title", "metrics_event_timeline", "metrics_tool_duration",
        "metrics_event_distribution", "metrics_recent_errors",
        "spawn_title", "spawn_active_children", "spawn_delegation_chain",
        "spawn_budget_allocation", "spawn_history_title",
        "features_title",
        "security_title", "security_pending_approvals",
        "security_permission_tree", "security_session_controls",
        "security_audit_log",
        "document_title",
    }
    referenced = set(_KEY_PATTERN.findall(html_text))
    referenced |= set(_ATTR_PATTERN.findall(html_text))

    orphans = sorted(k for k in surface_anchored_keys if k not in referenced)
    assert not orphans, (
        f"surface-anchored theme keys present in legacy/en.toml but never "
        f"referenced in index.html: {orphans}"
    )
