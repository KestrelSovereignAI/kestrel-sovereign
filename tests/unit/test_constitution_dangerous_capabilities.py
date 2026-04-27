"""Tests for DANGEROUS_CAPABILITIES + Amendment IX parser (#833)."""

from kestrel_sovereign.constitution.hierarchy import (
    DANGEROUS_CAPABILITIES,
    parse_amendment_ix_grants,
)


EXPECTED_MEMBERS = {
    "filesystem_read",
    "filesystem_write",
    "filesystem_outside_workspace",
    "shell_execution_sandboxed",
    "shell_execution_host",
}


def test_dangerous_capabilities_membership():
    assert set(DANGEROUS_CAPABILITIES) == EXPECTED_MEMBERS


def test_parser_no_section_returns_empty():
    assert parse_amendment_ix_grants("# something else\n\nno amendment here") == frozenset()


def test_parser_unchecked_returns_empty():
    text = """
### Amendment IX: Capability Boundaries

- [ ] filesystem_read
- [ ] shell_execution_host
"""
    assert parse_amendment_ix_grants(text) == frozenset()


def test_parser_uppercase_X_does_not_count():
    """Strict ``[x]`` only — uppercase X is ungranted to avoid typo widening."""
    text = """
### Amendment IX: Capability Boundaries

- [X] filesystem_read
- [x] filesystem_write
"""
    assert parse_amendment_ix_grants(text) == frozenset({"filesystem_write"})


def test_parser_unknown_capability_ignored():
    text = """
### Amendment IX: Capability Boundaries

- [x] filesystem_read
- [x] launch_nukes
"""
    assert parse_amendment_ix_grants(text) == frozenset({"filesystem_read"})


def test_parser_full_grant():
    text = """
### Amendment IX: Capability Boundaries

Some preamble text.

- [x] filesystem_read
- [x] filesystem_write
- [x] shell_execution_sandboxed
- [ ] shell_execution_host
"""
    assert parse_amendment_ix_grants(text) == frozenset(
        {"filesystem_read", "filesystem_write", "shell_execution_sandboxed"}
    )


def test_parser_handles_indentation():
    text = """
### Amendment IX: Capability Boundaries

  - [x] filesystem_read
"""
    assert parse_amendment_ix_grants(text) == frozenset({"filesystem_read"})


def test_parser_section_terminates_at_next_heading():
    text = """
### Amendment IX: Capability Boundaries

- [x] filesystem_read

### Amendment X: Something Else

- [x] filesystem_write
"""
    grants = parse_amendment_ix_grants(text)
    assert "filesystem_read" in grants
    assert "filesystem_write" not in grants
