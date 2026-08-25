"""The agent can read back its own prior writes (#3107).

The defect this closes was not a missing record. ``security_audit_log`` already
held 86,633 rows on the live agent, including BOTH halves of the duplicate that
motivated the ticket: kestrel-talon#228 filed at 19:31Z and #229 at 21:04Z, the
same defect, ninety-three minutes apart, each with its complete body in
``args_summary``. The first was on disk and queryable the entire time the second
was being written.

What was missing was a read. The ``security_audit`` tool exposes the table but
filters to ``("feature", "tool", "decision", "user_choice", "timestamp")`` — a
deliberate privacy decision, because dumping every recent row's arguments into
the model's context is disclosure nobody asked for. That filter also removed the
only field that can answer "have I already done this?", and nobody noticed the
second effect.

``security_audit_search`` asks the other question. The caller must describe what
it is looking for, so a match is disclosure of something the caller already
named — a different profile from an unbounded listing, which is why returning
``args_summary`` here is defensible where returning it there was not. These
tests pin that boundary in both directions: matches come back with their
arguments, and a row the query does not describe stays invisible.
"""

import pytest

from kestrel_sovereign.features.security.permissions import PermissionStore


# The two real filings, trimmed. Same defect, same reasoning chain,
# independently re-derived — the titles share words but the bodies are what a
# substantive dedupe check has to see.
_FILING_ONE = (
    '{"repo": "KestrelSovereignAI/kestrel-talon", "title": "Killing a tracked '
    'job kills the wrapper and orphans the worker \\u2014 reports success, '
    'leaves work running", "body": "## The defect\\n\\nThe job registry stores '
    'the wrapper PID. The wrapper does not exec."}'
)
_FILING_TWO = (
    '{"repo": "KestrelSovereignAI/kestrel-talon", "title": "Killing a tracked '
    'Talon job kills the wrapper and orphans the worker \\u2014 the kill '
    'reports success and the work keeps running", "body": "## Summary"}'
)


@pytest.fixture
async def store(tmp_path):
    store = PermissionStore(str(tmp_path / "audit.db"))
    await store.initialize()
    return store


async def _log_filing(store, args_summary, tool="create_github_issue"):
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name=tool,
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=args_summary,
    )


@pytest.mark.asyncio
async def test_a_later_turn_finds_the_filing_it_already_made(store):
    """The reproduction. Before filing the second, a search for the defect
    finds the first — which is all that was needed to prevent the duplicate."""
    await _log_filing(store, _FILING_ONE)

    matches = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1
    assert matches[0]["tool"] == "create_github_issue"
    assert "kestrel-talon" in matches[0]["args_summary"]


@pytest.mark.asyncio
async def test_the_match_carries_the_arguments_not_just_the_tool_name(store):
    """The whole point. ``security_audit`` returns which tool ran; that cannot
    distinguish this filing from any other ``create_github_issue`` call."""
    await _log_filing(store, _FILING_ONE)
    await _log_filing(
        store,
        '{"repo": "KestrelSovereignAI/kestrel-sovereign", "title": '
        '"Something entirely unrelated about embeddings"}',
    )

    matches = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1, (
        "two calls to the same tool must be distinguishable by their arguments"
    )
    assert "orphans the worker" in matches[0]["args_summary"]


@pytest.mark.asyncio
async def test_a_row_the_query_does_not_describe_stays_invisible(store):
    """The privacy boundary, asserted rather than assumed.

    Query-scoping is the entire reason returning ``args_summary`` here is
    defensible. If an unrelated row came back, this would just be the
    unbounded dump wearing a query parameter."""
    await _log_filing(store, _FILING_ONE)
    await _log_filing(
        store,
        '{"path": "/Users/someone/private/notes.txt", "operation": "read"}',
        tool="read_file",
    )

    matches = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1
    blob = " ".join(m["args_summary"] or "" for m in matches)
    assert "private/notes.txt" not in blob


@pytest.mark.asyncio
async def test_like_wildcards_in_the_query_cannot_widen_the_match(store):
    """``%`` is a LIKE wildcard. Passed through unescaped it turns any query
    containing one into "return everything", which is the unbounded listing
    this tool exists to avoid — reachable by typing a punctuation mark."""
    await _log_filing(store, _FILING_ONE)
    await _log_filing(store, _FILING_TWO)

    # "%" would be "match everything" unescaped; escaped it is a literal
    # percent sign, which appears in neither row.
    assert await store.search_audit_log("%") == []
    # A query that wraps a real term in wildcards must find nothing rather
    # than silently behaving like the bare term.
    assert await store.search_audit_log("%orphans%") == []
    # "_" is LIKE's any-single-character. Escaped it is a literal underscore,
    # which BOTH rows contain in "create_github_issue" — so the assertion that
    # proves escaping is that it matches as a character, not as a wildcard.
    assert len(await store.search_audit_log("_")) == 2
    await store.log_decision(
        feature_name="ShellFeature",
        tool_name="shell",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"command": "ls"}',
    )
    # ...and the row with no underscore anywhere is not swept in by it.
    underscore_hits = await store.search_audit_log("_")
    assert all(h["tool"] != "shell" for h in underscore_hits)


@pytest.mark.asyncio
async def test_an_empty_query_returns_nothing_rather_than_everything(store):
    """A blank query must not become a door to the listing this tool
    deliberately is not."""
    await _log_filing(store, _FILING_ONE)

    assert await store.search_audit_log("") == []
    assert await store.search_audit_log("   ") == []


@pytest.mark.asyncio
async def test_search_matches_the_tool_name_too(store):
    """"Did I call create_github_issue today" must work without the caller
    knowing any argument."""
    await _log_filing(store, _FILING_ONE)

    matches = await store.search_audit_log("create_github_issue")

    assert len(matches) == 1


@pytest.mark.asyncio
async def test_a_masked_secret_stays_masked_in_the_search_result(store):
    """The read-back must not become a way to recover what ``summarize_args``
    masked on the way in. The ceiling is whatever was persisted."""
    from kestrel_sovereign.features.security.args_summary import summarize_args

    await store.log_decision(
        feature_name="WalletAgent",
        tool_name="send_payment",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=summarize_args(
            {"api_key": "sk-live-SECRET", "memo": "orphans the worker"}
        ),
    )

    matches = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1
    assert "sk-live-SECRET" not in matches[0]["args_summary"]
    assert "***MASKED***" in matches[0]["args_summary"]


@pytest.mark.asyncio
async def test_tool_name_and_days_narrow_the_search(store):
    await _log_filing(store, _FILING_ONE)
    await _log_filing(store, _FILING_ONE, tool="add_github_issue_comment")

    only_issues = await store.search_audit_log(
        "orphans the worker", tool_name="create_github_issue"
    )
    assert len(only_issues) == 1
    assert only_issues[0]["tool"] == "create_github_issue"

    recent = await store.search_audit_log("orphans the worker", days=1)
    assert len(recent) == 2, "both rows were written just now"
