"""Tests for ``context_builder.extract_raw_user_content``.

The helper unwraps a stored user-turn sent-form back to the raw text so
UI previews, exports, and any future memory-indexing of user content see
the actual user message rather than the retrieved_context + <user_input>
wrappers that history-replay requires.

It must also be a no-op on legacy rows (raw text, no wrappers) so callers
can apply it unconditionally during the transition period.
"""
from kestrel_sovereign.agent.context_builder import extract_raw_user_content
from kestrel_sovereign.security.input_guardrails import wrap_user_input


def test_strips_full_sent_form_with_retrieved_context():
    sent = (
        "<retrieved_context>\n<memories>\nM1\n</memories>\n<documents>\nD1\n</documents>\n"
        "</retrieved_context>\n<user_input>\nhello world\n</user_input>"
    )
    assert extract_raw_user_content(sent) == "hello world"


def test_strips_sent_form_without_retrieved_context():
    """Empty {context} substitutes as "" → template becomes "\\n{query}",
    producing a leading newline before the <user_input> wrap."""
    sent = "\n<user_input>\nhello\n</user_input>"
    assert extract_raw_user_content(sent) == "hello"


def test_strips_sent_form_round_trips_wrap_user_input():
    """Verify round-trip against the actual wrap_user_input helper so any
    future format change to wrap_user_input is caught."""
    raw = "multi\nline\ninput"
    # No retrieved_context → just the leading newline + user_input wrap
    sent = "\n" + wrap_user_input(raw)
    assert extract_raw_user_content(sent) == raw


def test_legacy_raw_content_unchanged():
    """Rows written before the sent-form contract have no wrappers — the
    helper must be idempotent so callers can apply it unconditionally."""
    assert extract_raw_user_content("plain text") == "plain text"
    assert extract_raw_user_content("") == ""


def test_preserves_inner_newlines_in_raw():
    sent = "<user_input>\nline1\nline2\nline3\n</user_input>"
    assert extract_raw_user_content(sent) == "line1\nline2\nline3"


def test_preserves_inner_tag_like_content():
    """A raw user message that itself looks like a tag must NOT be
    double-stripped. Only the outer wrap is removed."""
    sent = "<user_input>\n<user_input>nested</user_input>\n</user_input>"
    assert extract_raw_user_content(sent) == "<user_input>nested</user_input>"


def test_malformed_partial_wrappers_left_alone():
    """If the prefix is present but the suffix isn't (or vice versa), do
    NOT strip — we'd be mangling content. Return as-is."""
    assert extract_raw_user_content("<user_input>\nhello") == "<user_input>\nhello"
    assert extract_raw_user_content("hello\n</user_input>") == "hello\n</user_input>"


def test_retrieved_context_only_no_user_input_wrap():
    """Defensive: if only the retrieved_context block is present (no
    user_input wrap — shouldn't happen in practice but be robust), strip
    it and return what remains."""
    sent = "<retrieved_context>\n<memories>\nM\n</memories>\n</retrieved_context>\nhello"
    assert extract_raw_user_content(sent) == "hello"
