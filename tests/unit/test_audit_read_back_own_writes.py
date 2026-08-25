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


# ---------------------------------------------------------------------------
# Review round 1 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_search_does_not_return_its_own_act_of_searching(tmp_path):
    """The defect my first tests could not see, because they called the store
    directly and production does not.

    `SecurityHook` runs at PRE_TOOL_USE and writes an audit row carrying the
    tool's arguments — for this tool, the query itself — BEFORE the tool body
    executes. Without an exclusion, every search matches its own invocation:
    the no-match branch becomes unreachable in production and a brand-new
    search reads as prior work. That is exactly the failure this tool exists
    to prevent, manufactured by the tool.

    Driven through the real hook rather than a stand-in, because the stand-in
    is what hid it."""
    from kestrel_sdk.hooks import HookInput
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.hooks import SecurityHook
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SEARCH_TOOL_NAME,
    )

    store = PermissionStore(str(tmp_path / "hooked.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", SEARCH_TOOL_NAME, PermissionLevel.ALLOW
    )
    hook = SecurityHook(store, ApprovalQueue(permission_store=store))

    query = "a phrase that appears in no prior filing"
    await hook.execute(HookInput(
        session_id="s",
        hook_event_name="PreToolUse",
        tool_name=SEARCH_TOOL_NAME,
        feature_name="SecurityFeature",
        tool_input={"query": query},
    ))

    # The hook really did record this call, query and all — the precondition
    # is real rather than assumed.
    logged = await store.get_audit_log(10)
    assert any(
        row["tool"] == SEARCH_TOOL_NAME and query in (row["args_summary"] or "")
        for row in logged
    ), "precondition: the hook must have logged the search with its query"

    assert await store.search_audit_log(query) == [], (
        "a search must not return itself; otherwise no query can ever come "
        "back empty and every novel search looks like prior work"
    )
    assert await store.count_audit_matches(query) == 0


@pytest.mark.asyncio
async def test_a_query_too_broad_to_be_a_description_returns_no_arguments(
    tmp_path,
):
    """The disclosure bound. Query-scoping only justifies returning arguments
    while the query describes one prior action; `"e"` describes nothing and
    would otherwise page out the log."""
    from kestrel_sovereign.features.security.feature import (
        MAX_DISCLOSING_MATCHES,
    )

    store = PermissionStore(str(tmp_path / "broad.db"))
    await store.initialize()
    for i in range(MAX_DISCLOSING_MATCHES + 5):
        await store.log_decision(
            feature_name="GitHubFeature",
            tool_name="create_github_issue",
            action="tool_execution",
            decision="auto_mode_allowed",
            args_summary=f'{{"title": "issue number {i} about everything"}}',
        )

    total = await store.count_audit_matches("e")
    assert total > MAX_DISCLOSING_MATCHES

    narrow = await store.count_audit_matches("issue number 3 about")
    assert 0 < narrow <= MAX_DISCLOSING_MATCHES, (
        "a real description must still be answerable"
    )


@pytest.mark.asyncio
async def test_a_denied_attempt_is_returned_as_denied(store):
    """A row is an AUTHORIZATION, not a completion — the hook logs before the
    body runs, and it logs refusals too. A caller that reads presence as "the
    work happened" would suppress a retry it needs."""
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_denied",
        args_summary=_FILING_ONE,
    )

    matches = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1
    assert matches[0]["decision"] == "auto_denied", (
        "the decision must survive to the caller; a denied attempt is not a "
        "prior write"
    )


@pytest.mark.asyncio
async def test_days_window_does_not_drop_legacy_timestamps(tmp_path):
    """`created_at` is ISO today and space-separated on legacy rows (F092).
    Compared as text, " " (32) sorts below "T" (84), so an ISO cutoff excludes
    EVERY legacy row on the cutoff's own date regardless of its time — a
    filter that silently drops the rows it was asked to include."""
    import sqlite3
    from datetime import datetime, timedelta, timezone

    db_path = tmp_path / "legacy_ts.db"
    store = PermissionStore(str(db_path))
    await store.initialize()

    # The row must land on the CUTOFF'S OWN DATE for this to bite. " " sorts
    # below "T" at index 10, so the comparison only goes wrong once the first
    # ten characters are equal; a row on any later date compares correctly by
    # its date alone and the bug hides. Put it one hour INSIDE the window,
    # which is the same calendar date as the cutoff.
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    legacy_stamp = (cutoff + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO security_audit_log "
        "(feature_name, tool_name, action, decision, user_choice, "
        " args_summary, created_at) VALUES (?,?,?,?,?,?,?)",
        ("GitHubFeature", "create_github_issue", "tool_execution",
         "auto_mode_allowed", None, _FILING_ONE, legacy_stamp),
    )
    raw.commit()
    raw.close()

    matches = await store.search_audit_log("orphans the worker", days=1)

    assert len(matches) == 1, (
        f"a legacy row stamped {legacy_stamp} is inside a 1-day window and "
        "must not be dropped for wearing the older timestamp format"
    )


@pytest.mark.asyncio
async def test_the_tool_withholds_arguments_for_a_broad_query(tmp_path):
    """The gate lives in the tool, so it is tested there. Past the bound the
    caller gets a count and a instruction to narrow — and no arguments."""
    from kestrel_sovereign.features.security.feature import (
        MAX_DISCLOSING_MATCHES,
        SecurityFeature,
    )

    store = PermissionStore(str(tmp_path / "toolgate.db"))
    await store.initialize()
    for i in range(MAX_DISCLOSING_MATCHES + 5):
        await store.log_decision(
            feature_name="GitHubFeature",
            tool_name="create_github_issue",
            action="tool_execution",
            decision="auto_mode_allowed",
            args_summary=f'{{"title": "issue number {i}", "secretish": "xyz"}}',
        )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    broad = await feature.security_audit_search(query="e")
    assert broad.data["too_broad"] is True
    assert broad.data["count"] > MAX_DISCLOSING_MATCHES
    assert broad.data["matches"] == []
    assert "xyz" not in broad.confirmation, (
        "a query too broad to be a description must disclose no arguments"
    )

    narrow = await feature.security_audit_search(query="issue number 3\"")
    assert narrow.data["too_broad"] is False
    assert narrow.data["matches"], "a real description is still answerable"


@pytest.mark.asyncio
async def test_raising_limit_cannot_defeat_the_breadth_gate(tmp_path):
    """The bound is the TOTAL-match gate, not the page size.

    An earlier draft clamped ``limit`` and called that the bound. Mutation
    testing showed removing the clamp changed nothing, and it was right: the
    gate runs on the total before any page is fetched, so a caller raising
    ``limit`` to 500 still gets a count and no arguments. This pins the
    property that actually holds, rather than the one that read well."""
    from kestrel_sovereign.features.security.feature import (
        MAX_DISCLOSING_MATCHES,
        SecurityFeature,
    )

    store = PermissionStore(str(tmp_path / "limitcap.db"))
    await store.initialize()
    for i in range(MAX_DISCLOSING_MATCHES + 5):
        await store.log_decision(
            feature_name="GitHubFeature",
            tool_name="create_github_issue",
            action="tool_execution",
            decision="auto_mode_allowed",
            args_summary=f'{{"title": "issue number {i}"}}',
        )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="e", limit=500)
    assert result.data["too_broad"] is True
    assert result.data["matches"] == []
    assert result.data.get("limit_requested", MAX_DISCLOSING_MATCHES) <= (
        MAX_DISCLOSING_MATCHES
    )
