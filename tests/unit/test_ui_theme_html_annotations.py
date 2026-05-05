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


def test_theme_picker_js_is_loaded_in_index_html(html_text):
    """theme_picker.js must be referenced or the picker dropdowns sit dead."""
    assert "js/theme_picker.js" in html_text, "index.html must include js/theme_picker.js"


def test_theme_picker_dom_elements_present(html_text):
    """The picker's anchor elements must exist; theme_picker.js targets them by id."""
    assert 'id="theme-picker-theme"' in html_text, "theme dropdown anchor missing"
    assert 'id="theme-picker-locale"' in html_text, "locale dropdown anchor missing"
    assert 'id="theme-picker-status"' in html_text, "status line anchor missing"


def test_picker_section_labels_exist_in_legacy(legacy_labels):
    """The picker chrome itself is themable — these keys must exist."""
    for key in ("sovereignty_display", "sovereignty_display_description",
                "theme_picker_label", "locale_picker_label"):
        assert key in legacy_labels, f"{key} missing from legacy/en.toml"


def test_dynamic_elements_are_not_themed(html_text):
    """Elements that JS mutates after first paint must NOT carry
    data-label-key — otherwise theme switches and re-hydrations would
    overwrite the dynamic value with the legacy placeholder.

    Each entry here is a specific (id, mutating_module) pair that was
    found to be a real bug by codex CLI review. Add new entries when a
    new dynamic element gets accidentally annotated.
    """
    # <title> is mutated by identity.js to include the agent name
    title_tag = re.search(r"<title[^>]*>", html_text)
    assert title_tag, "no <title> element found"
    assert "data-label-key" not in title_tag.group(0), (
        "<title> must not carry data-label-key — identity.js sets it "
        "dynamically per agent"
    )

    # key-source-badge is mutated by resources.js to show the active key
    # provider ('Agent Key' / 'Your Key (BYOK)' / 'Platform' / 'Unknown')
    badge_match = re.search(
        r'<span[^>]*id="key-source-badge"[^>]*>',
        html_text,
    )
    assert badge_match, "no #key-source-badge element found"
    assert "data-label-key" not in badge_match.group(0), (
        "#key-source-badge must not carry data-label-key — resources.js "
        "writes the active key source into it dynamically"
    )


def test_theme_picker_resyncs_on_themechange():
    """theme_picker.js must update its dropdowns when a themechange event
    fires — otherwise the picker shows the wrong theme on first paint
    when the user has a non-legacy theme stored (theme.js's initial
    applyTheme() resolves async, after the picker's init runs).
    """
    picker_path = (
        Path(__file__).resolve().parents[2]
        / "kestrel_sovereign" / "static" / "js" / "theme_picker.js"
    )
    src = picker_path.read_text(encoding="utf-8")
    # Must register a themechange listener.
    assert "addEventListener('themechange'" in src or 'addEventListener("themechange"' in src, (
        "theme_picker.js must listen to themechange to keep the dropdowns "
        "in sync with the actual applied theme"
    )
    # That listener must update the select values (not just the status line).
    # Approximate check: themeSelect.value or localeSelect.value assigned
    # inside the file. The exact form lives in the listener.
    assert (
        re.search(r"themeSelect\.value\s*=", src)
        and re.search(r"localeSelect\.value\s*=", src)
    ), (
        "theme_picker.js must assign themeSelect.value and localeSelect.value "
        "in response to themechange; otherwise the dropdowns drift from "
        "the actual applied theme on first paint"
    )


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
        "sovereignty_display", "sovereignty_display_description",
        "theme_picker_label", "locale_picker_label",
    }
    referenced = set(_KEY_PATTERN.findall(html_text))
    referenced |= set(_ATTR_PATTERN.findall(html_text))

    orphans = sorted(k for k in surface_anchored_keys if k not in referenced)
    assert not orphans, (
        f"surface-anchored theme keys present in legacy/en.toml but never "
        f"referenced in index.html: {orphans}"
    )
