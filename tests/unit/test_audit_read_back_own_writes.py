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

import asyncio
import json

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

    matches, _ = await store.search_audit_log("orphans the worker")

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

    matches, _ = await store.search_audit_log("orphans the worker")

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

    matches, _ = await store.search_audit_log("orphans the worker")

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
    assert (await store.search_audit_log("%"))[0] == []
    # A query that wraps a real term in wildcards must find nothing rather
    # than silently behaving like the bare term.
    assert (await store.search_audit_log("%orphans%"))[0] == []
    # "_" is LIKE's any-single-character. Escaped it is a literal underscore,
    # which BOTH rows contain in "create_github_issue" — so the assertion that
    # proves escaping is that it matches as a character, not as a wildcard.
    assert len((await store.search_audit_log("_"))[0]) == 2
    await store.log_decision(
        feature_name="ShellFeature",
        tool_name="shell",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"command": "ls"}',
    )
    # ...and the row with no underscore anywhere is not swept in by it.
    underscore_hits, _ = await store.search_audit_log("_")
    assert all(h["tool"] != "shell" for h in underscore_hits)


@pytest.mark.asyncio
async def test_an_empty_query_returns_nothing_rather_than_everything(store):
    """A blank query must not become a door to the listing this tool
    deliberately is not."""
    await _log_filing(store, _FILING_ONE)

    assert (await store.search_audit_log(""))[0] == []
    assert (await store.search_audit_log("   "))[0] == []


@pytest.mark.asyncio
async def test_search_matches_the_tool_name_too(store):
    """"Did I call create_github_issue today" must work without the caller
    knowing any argument."""
    await _log_filing(store, _FILING_ONE)

    matches, _ = await store.search_audit_log("create_github_issue")

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

    matches, _ = await store.search_audit_log("orphans the worker")

    assert len(matches) == 1
    assert "sk-live-SECRET" not in matches[0]["args_summary"]
    assert "***MASKED***" in matches[0]["args_summary"]


@pytest.mark.asyncio
async def test_tool_name_and_days_narrow_the_search(store):
    await _log_filing(store, _FILING_ONE)
    await _log_filing(store, _FILING_ONE, tool="add_github_issue_comment")

    only_issues, _ = await store.search_audit_log(
        "orphans the worker", tool_name="create_github_issue"
    )
    assert len(only_issues) == 1
    assert only_issues[0]["tool"] == "create_github_issue"

    recent, _ = await store.search_audit_log("orphans the worker", days=1)
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

    assert (await store.search_audit_log(query))[0] == [], (
        "a search must not return itself; otherwise no query can ever come "
        "back empty and every novel search looks like prior work"
    )
    assert (await store.search_audit_log(query))[1] is False


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

    _rows, too_broad = await store.search_audit_log(
        "e", limit=MAX_DISCLOSING_MATCHES,
    )
    assert too_broad is True

    narrow, narrow_too_broad = await store.search_audit_log(
        "issue number 3 about", limit=MAX_DISCLOSING_MATCHES,
    )
    assert narrow and narrow_too_broad is False, (
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

    matches, _ = await store.search_audit_log("orphans the worker")

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
    legacy_stamp = (cutoff + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    # The construction above must land on the cutoff's own date or the test
    # stops exercising the bug (one hour ahead crossed midnight for one hour
    # in every 24 and passed against the reverted predicate). Say so loudly.
    assert legacy_stamp[:10] == cutoff.strftime("%Y-%m-%d"), legacy_stamp
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

    matches, _ = await store.search_audit_log("orphans the worker", days=1)

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
    assert broad.data["matches"] == []
    # No exact count either: reporting "412 matched" is itself a small
    # disclosure about the log, and the caller only needs to know to narrow.
    assert broad.data["count"] is None
    assert "xyz" not in broad.confirmation, (
        "a query too broad to be a description must disclose no arguments"
    )

    # A real description, not a serialization artifact: the searchable
    # projection is now decoded values, so JSON punctuation is deliberately
    # absent from it.
    narrow = await feature.security_audit_search(query="issue number 3")
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
    assert result.data["matches"] == [], (
        "the store is asked for MAX_DISCLOSING_MATCHES regardless of what the "
        "caller passed, so limit cannot widen what is disclosed"
    )


# ---------------------------------------------------------------------------
# Review round 2 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_natural_language_query_matches_json_escaped_storage(store):
    """`summarize_args` persists `json.dumps(...)`, which is ASCII-escaped by
    default, so "Échec — café" is on disk as "\\u00c9chec \\u2014 caf\\u00e9".
    A LIKE built from the literal query finds nothing — and it fails for
    exactly the fragments a caller reaches for. BOTH real filings behind this
    ticket carried an em dash in their titles."""
    from kestrel_sovereign.features.security.args_summary import summarize_args

    stored = summarize_args(
        {"title": "Échec — the wrapper orphans the worker", "repo": "x"}
    )
    assert "\\u2014" in stored, (
        "precondition: the em dash must actually be escaped on disk"
    )
    assert "—" not in stored

    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=stored,
    )

    for fragment in ("Échec — the wrapper", "Échec", "—"):
        matches, _ = await store.search_audit_log(fragment)
        assert len(matches) == 1, (
            f"a query containing non-ASCII ({fragment!r}) must match the "
            "escaped form the writer actually persisted"
        )


@pytest.mark.asyncio
async def test_a_subagent_dispatch_is_a_request_not_a_prior_action(tmp_path):
    """`SecurityHook` runs on PRE_SUBAGENT_CALL as well as PRE_TOOL_USE, so a
    feature-as-subagent dispatch writes a row carrying the whole requested task
    text. When that dispatch is what reached this very tool, the search phrase
    is inside it and the enclosing call comes back as prior work.

    The rule is not "special-case my own caller" — it is that a dispatch
    records what was ASKED FOR and the inner tool rows record what was DONE.
    Driven through the real hook, on the real event."""
    from kestrel_sdk.hooks.base import HookEvent, HookInput
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.hooks import SecurityHook
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "subagent.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", "security_feature", PermissionLevel.ALLOW
    )
    hook = SecurityHook(store, ApprovalQueue(permission_store=store))

    phrase = "whether the wrapper orphans the worker"
    await hook.execute(HookInput(
        session_id="s",
        hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
        tool_name="security_feature",
        feature_name="SecurityFeature",
        tool_input={"task": f"Search the audit log for {phrase}"},
    ))

    logged = await store.get_audit_log(10)
    assert any(
        row["action"] == SUBAGENT_DISPATCH_ACTION
        and phrase in (row["args_summary"] or "")
        for row in logged
    ), "precondition: the dispatch envelope must be recorded, and marked as one"

    matches, _ = await store.search_audit_log(phrase)
    assert matches == [], (
        "a dispatch envelope is a request; returning it as prior work makes "
        "novel work look already done"
    )


@pytest.mark.asyncio
async def test_an_inner_tool_call_is_still_found_after_a_dispatch(tmp_path):
    """The other end of the same boundary: excluding envelopes must not
    exclude the work. A dispatch that actually ran its inner tool leaves a
    `tool_execution` row, and THAT is the prior action."""
    from kestrel_sovereign.features.security.permissions import (
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "inner.db"))
    await store.initialize()
    phrase = "orphans the worker"
    await store.log_decision(
        feature_name="SecurityFeature",
        tool_name="security_feature",
        action=SUBAGENT_DISPATCH_ACTION,
        decision="auto_mode_allowed",
        args_summary=f'{{"task": "file an issue about {phrase}"}}',
    )
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=f'{{"title": "the wrapper {phrase}"}}',
    )

    matches, _ = await store.search_audit_log(phrase)

    assert len(matches) == 1
    assert matches[0]["tool"] == "create_github_issue"


# ---------------------------------------------------------------------------
# Review round 3 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lowercase_query_matches_an_accented_capital(store):
    """SQLite's LOWER folds ASCII only, and the JSON escapes differ in a
    character it will not touch: "É" is \\u00c9 and "é" is \\u00e9. So matching
    the escaped form case-insensitively silently fails for every non-English
    prior write — the case-insensitivity the tool promises stopped at the
    ASCII boundary without saying so."""
    from kestrel_sovereign.features.security.args_summary import summarize_args

    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=summarize_args({"title": "Échec — the worker is orphaned"}),
    )

    for query in ("échec", "ÉCHEC", "Échec — the worker"):
        matches, _ = await store.search_audit_log(query)
        assert len(matches) == 1, f"{query!r} must match the stored capital form"


@pytest.mark.asyncio
async def test_a_truncated_summary_is_still_searchable(store):
    """`fold_stored_summary` decodes JSON before folding, and a summary truncated
    mid-escape is not valid JSON. Falling back to the raw text keeps the row in
    the corpus; dropping it would silently shrink what the caller believes it
    searched — the same class of quiet incompleteness this tool exists to
    remove."""
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"title": "the worker is orphaned", "body": "## Deta...',
    )

    matches, _ = await store.search_audit_log("worker is orphaned")
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_the_approval_gated_path_also_labels_a_dispatch(tmp_path):
    """Round 2 taught ONE door the rule. `ApprovalQueue` writes audit rows on
    paths the hook never returns through, and it hard-coded
    `action="tool_execution"` — including for the ASK level, which is the
    DEFAULT for the feature-as-subagent dispatcher. So the exclusion added in
    round 2 missed exactly the case it was added for."""
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "asked.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", "security_feature", PermissionLevel.DENY
    )
    queue = ApprovalQueue(permission_store=store)

    phrase = "whether the wrapper orphans the worker"
    approved, _scope = await queue.request_approval(
        feature_name="SecurityFeature",
        tool_name="security_feature",
        tool_args={"task": f"Search the audit log for {phrase}"},
        audit_action=SUBAGENT_DISPATCH_ACTION,
    )
    assert approved is False  # DENY short-circuits, and still audits

    logged = await store.get_audit_log(10)
    assert logged and logged[0]["action"] == SUBAGENT_DISPATCH_ACTION, (
        "the queue must label a dispatch envelope the same way the hook does; "
        "two writers, one invariant"
    )

    matches, _ = await store.search_audit_log(phrase)
    assert matches == []


@pytest.mark.asyncio
async def test_the_deferred_decision_path_also_labels_a_dispatch(tmp_path):
    """And the OTHER door inside the queue.

    `ApprovalQueue` writes its audit row from two places: an early return for
    the levels it can decide alone, and `_persist_decision` for everything that
    reaches a real decision — timeout, cancellation, headless no-approver, or a
    human answering minutes later on another task. Mutation testing caught that
    the previous test only covered the first: neutering `_persist_decision`'s
    label left it passing.

    This drives the headless path, which is the one an unattended agent
    actually takes."""
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "deferred.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", "security_feature", PermissionLevel.ASK
    )
    queue = ApprovalQueue(permission_store=store)

    phrase = "whether the wrapper orphans the worker"
    approved, scope = await queue.request_approval(
        feature_name="SecurityFeature",
        tool_name="security_feature",
        tool_args={"task": f"Search the audit log for {phrase}"},
        allow_blocking=False,
        audit_action=SUBAGENT_DISPATCH_ACTION,
    )
    assert approved is False and scope == "no_approver", (
        "precondition: this must reach _persist_decision, not an early return"
    )

    logged = await store.get_audit_log(10)
    assert logged and logged[0]["action"] == SUBAGENT_DISPATCH_ACTION
    assert logged[0]["decision"] == "no_approver"

    matches, _ = await store.search_audit_log(phrase)
    assert matches == []


@pytest.mark.asyncio
async def test_the_caller_s_limit_is_honoured_within_the_bound(tmp_path):
    """The gate needs the full bound to decide, so the query always asks for
    MAX_DISCLOSING_MATCHES — but the caller's smaller limit must still shrink
    what is disclosed. Round 2 echoed `limit` in the result and returned
    everything."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "limit.db"))
    await store.initialize()
    for i in range(5):
        await store.log_decision(
            feature_name="GitHubFeature",
            tool_name="create_github_issue",
            action="tool_execution",
            decision="auto_mode_allowed",
            args_summary=f'{{"title": "orphaned worker report {i}"}}',
        )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="orphaned worker", limit=2)
    assert result.data["too_broad"] is False
    assert len(result.data["matches"]) == 2, (
        "asking for 2 must disclose 2, not the whole bound"
    )


@pytest.mark.asyncio
async def test_the_hook_tells_the_queue_which_kind_of_call_it_was(tmp_path):
    """End to end through the seam that actually runs in production.

    The previous two tests called `request_approval` directly and passed the
    label themselves, so neither noticed when the HOOK stopped forwarding it —
    mutation testing found that gap. This drives `SecurityHook.execute` on
    PRE_SUBAGENT_CALL at a level that reaches the queue."""
    import asyncio

    from kestrel_sdk.hooks.base import HookEvent, HookInput
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.hooks import SecurityHook
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "hook_to_queue.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", "security_feature", PermissionLevel.ASK
    )
    hook = SecurityHook(store, ApprovalQueue(permission_store=store))

    phrase = "whether the wrapper orphans the worker"
    # session_id="scheduler" is NON_INTERACTIVE_SESSION_IDS, so the queue
    # returns a no_approver denial instead of waiting forever for a human —
    # and that is the unattended path an agent actually runs on.
    await asyncio.wait_for(hook.execute(HookInput(
        session_id="scheduler",
        hook_event_name=HookEvent.PRE_SUBAGENT_CALL.value,
        tool_name="security_feature",
        feature_name="SecurityFeature",
        tool_input={"task": f"Search the audit log for {phrase}"},
    )), timeout=5.0)

    logged = await store.get_audit_log(10)
    assert logged, "precondition: the ASK path must have written a row"
    assert logged[0]["action"] == SUBAGENT_DISPATCH_ACTION, (
        "the hook knows which event fired; the queue is the one that writes. "
        "If the fact does not cross that boundary the label is decorative."
    )

    matches, _ = await store.search_audit_log(phrase)
    assert matches == []


@pytest.mark.asyncio
async def test_a_decision_answered_later_still_labels_the_dispatch(tmp_path):
    """The third door: a request that waits for a human and is decided on a
    different task, minutes later. `_persist_decision` reads the label off the
    request rather than an argument in scope, so the request has to have
    carried it — mutation testing showed nothing covered that."""
    import asyncio

    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.permissions import (
        PermissionLevel,
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "answered.db"))
    await store.initialize()
    await store.register_tool(
        "SecurityFeature", "security_feature", PermissionLevel.ASK
    )
    queue = ApprovalQueue(permission_store=store)

    async def approve_later():
        for _ in range(50):
            await asyncio.sleep(0.01)
            pending = queue.pending_requests
            if pending:
                await queue.submit_decision(pending[0].id, True, "once")
                return
        raise AssertionError("the request never reached the queue")

    task = asyncio.create_task(approve_later())
    phrase = "whether the wrapper orphans the worker"
    approved, _scope = await queue.request_approval(
        feature_name="SecurityFeature",
        tool_name="security_feature",
        tool_args={"task": f"Search the audit log for {phrase}"},
        audit_action=SUBAGENT_DISPATCH_ACTION,
    )
    await task
    assert approved is True

    logged = await store.get_audit_log(10)
    row = next(r for r in logged if r["decision"] == "user_approved")
    assert row["action"] == SUBAGENT_DISPATCH_ACTION

    matches, _ = await store.search_audit_log(phrase)
    assert matches == []


# ---------------------------------------------------------------------------
# Review round 4 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_inner_dispatch_envelope_is_excluded_too(tmp_path):
    """Labelling by hook EVENT could not cover this, by construction.

    `orchestrator_engine` fires PRE_SUBAGENT_CALL for a dispatch and then a
    second PRE_TOOL_USE around `execute_as_subagent` with the same arguments —
    deliberately, so PRE_TOOL_USE-only hooks see chat-path and inline-path
    dispatches alike (PR #1385). Only the first row carries the dispatch event,
    so the second passed the exclusion, contained the current query, and came
    back as prior work.

    Both rows name the feature's dispatch ENTRY. That is the fact they share,
    and it is what the exclusion keys on now."""
    from kestrel_sovereign.features.security.permissions import (
        SUBAGENT_DISPATCH_ACTION,
    )

    store = PermissionStore(str(tmp_path / "inner_envelope.db"))
    await store.initialize()
    store.mark_dispatch_entry("security_feature")

    phrase = "whether the wrapper orphans the worker"
    task_args = f'{{"task": "Search the audit log for {phrase}"}}'
    # The pair the orchestrator actually writes.
    await store.log_decision(
        feature_name="SecurityFeature", tool_name="security_feature",
        action=SUBAGENT_DISPATCH_ACTION, decision="auto_mode_allowed",
        args_summary=task_args,
    )
    await store.log_decision(
        feature_name="SecurityFeature", tool_name="security_feature",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary=task_args,
    )

    matches, _ = await store.search_audit_log(phrase)
    assert matches == [], (
        "the inner PRE_TOOL_USE envelope carries the same task text and must "
        "not read as a prior attempt at the work it is still requesting"
    )


@pytest.mark.asyncio
async def test_an_inner_tool_row_survives_the_name_exclusion(tmp_path):
    """The other end again: excluding a feature's dispatch ENTRY must not
    exclude the tools that feature actually ran."""
    store = PermissionStore(str(tmp_path / "inner_survives.db"))
    await store.initialize()
    store.mark_dispatch_entry("security_feature")

    phrase = "orphans the worker"
    await store.log_decision(
        feature_name="SecurityFeature", tool_name="security_feature",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary=f'{{"task": "file an issue about {phrase}"}}',
    )
    await store.log_decision(
        feature_name="GitHubFeature", tool_name="create_github_issue",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary=f'{{"title": "the wrapper {phrase}"}}',
    )

    matches, _ = await store.search_audit_log(phrase)
    assert len(matches) == 1
    assert matches[0]["tool"] == "create_github_issue"


@pytest.mark.asyncio
async def test_a_truncated_summary_still_matches_non_ascii(store):
    """The escape-decoding and the truncation fallback met in the worst place.

    `summarize_args` cuts at 500 characters, mid-structure, so a long issue
    body is not valid JSON — and long issue bodies are exactly the motivating
    case. Falling back to raw text there left every escape undecoded, which is
    the round-3 bug one row-shape over: `échec` could not match a stored
    `\\u00c9chec` even when it sat at character 12."""
    import json

    long_body = "## Detail. " * 80
    stored = json.dumps(
        {"title": "Échec — the worker is orphaned", "body": long_body}
    )[:500]
    with pytest.raises(ValueError):
        json.loads(stored)  # precondition: genuinely truncated

    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary=stored,
    )

    for query in ("échec", "Échec — the worker", "—"):
        matches, _ = await store.search_audit_log(query)
        assert len(matches) == 1, f"{query!r} must match inside a truncated row"


@pytest.mark.asyncio
async def test_registering_a_feature_marks_its_dispatch_entry(tmp_path):
    """The wiring, not the mechanism.

    Mutation testing found that every test above called `mark_dispatch_entry`
    itself, so nothing noticed when `register_feature_tools` stopped calling
    it. The exclusion would then be correct and never armed — a guard that is
    right about nothing because it was never told what to guard.

    Third time this session that a fix and its test covered the same one of two
    doors."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    class _FeatureWithDispatch:
        tool_name = "pretend_feature"
        tools = []

        def get_tools(self):
            return []

    store = PermissionStore(str(tmp_path / "registration.db"))
    await store.initialize()
    assert store.dispatch_entries == frozenset(), "precondition: nothing marked"

    security = SecurityFeature.__new__(SecurityFeature)
    security.permission_store = store
    security.agent = None
    await security.register_feature_tools(
        "PretendFeature", _FeatureWithDispatch(),
    )

    assert "pretend_feature" in store.dispatch_entries, (
        "registration is the only place that knows a name is a dispatch "
        "entry; if it does not say so, the exclusion never arms"
    )


# ---------------------------------------------------------------------------
# Review round 5 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_security_features_own_dispatch_entry_is_marked(tmp_path):
    """The fourth instance of the same miss, and the sharpest.

    Round 4 added the registry and a test — but the test registered a STUB
    feature, and `_register_all_tools` explicitly `continue`s past
    SecurityFeature. So the one dispatch entry that actually reaches
    `security_audit_search` was the only one never marked, and the exclusion
    was armed for every feature except the one that matters.

    The double was the disguise: a stub feature goes through the branch the
    real one skips."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    class _Other:
        tool_name = "other_feature"

        def get_tools(self):
            return []

    class _Agent:
        def __init__(self, security):
            self.features = {"SecurityFeature": security, "OtherFeature": _Other()}

    store = PermissionStore(str(tmp_path / "own_entry.db"))
    await store.initialize()

    security = SecurityFeature.__new__(SecurityFeature)
    security.permission_store = store
    # ``tool_name`` is a read-only property deriving snake_case from ``name``,
    # so the dispatch entry is set by naming the feature — the same way it is
    # derived in production rather than assigned.
    security.name = "SecurityFeature"
    assert security.tool_name == "security_feature"
    security.agent = _Agent(security)
    await security._register_all_tools()

    assert "security_feature" in store.dispatch_entries, (
        "SecurityFeature's tool rows are skipped, but its DISPATCH entry is "
        "the envelope that reaches this very tool — skipping it leaves the "
        "exclusion armed for every feature except the one that matters"
    )


@pytest.mark.asyncio
async def test_a_truncated_emoji_does_not_break_every_search(store):
    """Severity note: this is not a missed match, it is a dead tool.

    `summarize_args` can cut a 500-character summary between an emoji's two
    surrogate escapes. Decoding that half alone yields a lone surrogate, and a
    lone surrogate returned from a SQLite scalar function raises inside the
    engine — failing the WHOLE query, so one such row makes every search error
    out regardless of what it was looking for."""
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"title": "the worker is orphaned", "note": "hi \\ud83d',
    )

    matches, _ = await store.search_audit_log("worker is orphaned")
    assert len(matches) == 1, (
        "a row truncated mid-surrogate must be searchable, not fatal"
    )


def test_a_whole_emoji_still_folds_to_one_character():
    """The other end: leaving lone surrogates escaped must not stop a COMPLETE
    pair from rejoining, or every emoji becomes two escapes nobody can match."""
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    # The projection is decoded keys and values, not re-serialized JSON — so
    # the assertion is that the emoji survives as ONE character in it, not that
    # the JSON round-trips.
    folded = fold_stored_summary('{"t": "hi \\ud83d\\ude00"}')
    assert "😀" in folded
    assert "\\ud83d" not in folded


@pytest.mark.asyncio
async def test_the_self_exclusion_does_not_swallow_similarly_named_tools(store):
    """`security_audit_search` contains underscores, and LIKE treats each as
    "any character" — so the unescaped self-exclusion also hid unrelated tools
    whose names differ only in those positions. Same wildcard bug already fixed
    for the query, left standing in the exclusion."""
    await store.log_decision(
        feature_name="IndexFeature",
        tool_name="security-audit-search-index",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"note": "the worker is orphaned"}',
    )

    matches, _ = await store.search_audit_log("worker is orphaned")
    assert len(matches) == 1, (
        "only this tool's own rows are excluded, not every name that matches "
        "it once underscores are read as wildcards"
    )


# ---------------------------------------------------------------------------
# Review round 6 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_legacy_unmasked_secret_is_masked_on_the_way_out(store):
    """The guarantee has to live on the READ path.

    `summarize_args` masks on the way in, which makes the stored value safe
    only for rows written by a path that had that masking — `ApprovalQueue`
    kept an unmasked copy until F252. This tool cannot verify the provenance of
    tens of thousands of historical rows, so it must not depend on it.

    On Emma's live corpus (15,727 rows with args) there are zero parseable rows
    carrying an unmasked sensitive key, so this is a defence against a
    mechanism rather than an observed leak. It is still the right shape: it
    converts a claim about every historical writer into a property of the one
    path we control."""
    await store.log_decision(
        feature_name="WalletAgent",
        tool_name="send_payment",
        action="tool_execution",
        decision="auto_mode_allowed",
        # Written raw, exactly as a pre-F252 queue row would be.
        args_summary='{"api_key": "sk-live-LEGACY-LEAK", "memo": "orphaned"}',
    )

    # The search read path — new in #3107 — re-masks in the STORE, so every
    # caller of it sees the same text the searchable projection matched on
    # (round 14: a re-mask that lived only in the tool left the next caller
    # to leak what the tool hid).
    matches, _ = await store.search_audit_log("orphaned")
    assert len(matches) == 1
    shown = matches[0]["args_summary"]
    assert "sk-live-LEGACY-LEAK" not in shown
    assert "***MASKED***" in shown and "orphaned" in shown

    # The store's OTHER read path over the same column, get_audit_log
    # (/api/security/audit), masks too: "one door in the store" is only true
    # if both doors do (round 17 review).
    recent = await store.get_audit_log(limit=5)
    assert recent and all("sk-live-LEGACY-LEAK" not in (r.get("args_summary") or "") for r in recent)
    assert any("***MASKED***" in (r.get("args_summary") or "") for r in recent)


@pytest.mark.asyncio
async def test_an_unmaskable_row_naming_a_secret_is_not_searchable(tmp_path):
    """Stricter than round 6, and deliberately so.

    Round 6 masked on the way OUT, which closed the display and left the MATCH
    open: a caller could compare a query against the raw stored secret and read
    it back a character at a time from hit/no-hit, while every returned row
    showed ***MASKED***. So the searchable projection is the masked one now.

    A row too truncated to parse cannot be masked field-by-field. Round 13:
    the sensitive VALUE is masked through the end of its (cut) JSON string
    and the rest of the row stays searchable — the oracle is closed at the
    value, and a benign field beside it no longer leaves the corpus."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "unmaskable.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="WalletAgent", tool_name="send_payment",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"memo": "orphaned worker", "api_key": "sk-live-CUT',
    )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphaned worker")

    assert result.data["count"] == 1, (
        "the benign field beside a masked secret must stay searchable"
    )
    assert "sk-live-CUT" not in result.confirmation
    assert "sk-live-CUT" not in str(result.data)

    # And the oracle is closed at the source: the query never reaches the raw
    # value, so hit/no-hit cannot be used to walk it out.
    from kestrel_sovereign.features.security.permissions import fold_stored_summary
    folded = fold_stored_summary('{"memo": "orphaned worker", "api_key": "sk-live-CUT')
    assert "sk-live" not in folded and "cut" not in folded
    # ...but the benign field beside it stays searchable: dropping the whole
    # row silently shrank the corpus (round 13).
    assert "orphaned worker" in folded


@pytest.mark.asyncio
async def test_a_refused_attempt_is_not_described_as_authorized(tmp_path):
    """The headline used to say every match "was authorized" while the row
    beside it read `auto_denied`. In a tool whose whole purpose is telling an
    agent what it has already done, that is the failure mode itself."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "refused.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHubFeature", tool_name="create_github_issue",
        action="tool_execution", decision="auto_denied",
        args_summary='{"title": "the worker is orphaned"}',
    )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="worker is orphaned")

    assert "an authorization to run" in result.confirmation
    assert result.data["matches"][0]["decision"] == "auto_denied"


@pytest.mark.asyncio
async def test_an_unrecognised_decision_is_not_called_authorized(tmp_path):
    """The inversion, tested at the point it matters.

    `record_tool_rejection` writes `decision="blocked"` and demo isolation
    writes `refused`; a hand-listed refusal set had neither, so both read as
    "authorized to run" — the direction that suppresses a retry of work that
    never happened. This agent's own log carries fifteen distinct decision
    values. Listing the refusals is a list that grows; listing the
    authorizations means anything new is safe by default."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "unknown_decision.db"))
    await store.initialize()
    for decision in ("blocked", "refused", "a_value_invented_next_year"):
        await store.log_decision(
            feature_name="GitHubFeature", tool_name="create_github_issue",
            action="tool_execution", decision=decision,
            args_summary=f'{{"title": "orphaned worker {decision}"}}',
        )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphaned worker")

    assert result.data["count"] == 3
    assert "an authorization to run" in result.confirmation
    assert result.confirmation.count("✗") == 3, (
        "every unrecognised decision must be marked as not-authorized"
    )


@pytest.mark.asyncio
async def test_a_removed_features_dispatch_name_still_filters(tmp_path):
    """`_dispatch_entries` is rebuilt from CURRENTLY LOADED features, but the
    audit log outlives the feature list. A feature removed or renamed since its
    envelope rows were written would drop out of the exclusion and its old
    REQUESTS would start reading as prior attempts."""
    db_path = tmp_path / "durable_entries.db"

    boot_one = PermissionStore(str(db_path))
    await boot_one.initialize()
    boot_one.mark_dispatch_entry("retired_feature")
    await boot_one.sync_dispatch_entries()
    await boot_one.log_decision(
        feature_name="RetiredFeature", tool_name="retired_feature",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"task": "look into whether the worker is orphaned"}',
    )

    # A later boot: the feature is gone, so nothing marks it this time.
    boot_two = PermissionStore(str(db_path))
    await boot_two.initialize()
    assert "retired_feature" not in boot_two.dispatch_entries

    matches, _ = await boot_two.search_audit_log("worker is orphaned")
    assert matches == [], (
        "the name outlives the feature because the rows do"
    )


@pytest.mark.asyncio
async def test_a_legacy_secret_cannot_be_walked_out_by_hit_or_miss(store):
    """The oracle, closed at the match rather than at the display.

    Round 6 masked on the way out and I called the leak fixed. It was not: the
    LIKE predicate still compared the query against the RAW stored value, so a
    caller could test `sk-live-L`, `sk-live-LE`, `sk-live-LEA` and read a
    credential out one character at a time from hit/no-hit, while every
    returned row dutifully displayed ***MASKED***. Masking the display closed
    the half a human looks at and left the half a program can use.

    Mutation testing caught that removing the mask from the searchable
    projection broke nothing — because every existing test covered the
    unparseable case. This one covers the parseable case, which is the common
    one."""
    await store.log_decision(
        feature_name="WalletAgent",
        tool_name="send_payment",
        action="tool_execution",
        decision="auto_mode_allowed",
        # Written raw, as a pre-F252 ApprovalQueue row would have been.
        args_summary='{"api_key": "sk-live-ORACLE-9Q7", "memo": "orphaned"}',
    )

    # The row is findable by its non-sensitive content...
    found, _ = await store.search_audit_log("orphaned")
    assert len(found) == 1

    # ...and by no prefix of the secret, at any length.
    for probe in ("sk-live-O", "sk-live-ORACLE", "sk-live-ORACLE-9Q7", "9Q7"):
        hits, _ = await store.search_audit_log(probe)
        assert hits == [], (
            f"searching {probe!r} must not confirm it; a hit/no-hit answer is "
            "a read of the value even when the row renders masked"
        )


# ---------------------------------------------------------------------------
# Review round 8 (#3107)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_query_containing_key_still_searches(store):
    """The defect my round-7 fix created, and the ninth of its kind.

    Round 7 gave `fold_searchable` a stored-summary rule — drop anything naming
    a sensitive key — and the QUERY went through the same function. So `monkey`
    folded to the empty string, `_like("")` became `%%`, and the search matched
    every row in the table: a broken result AND a disclosure of unrelated
    summaries.

    I changed a shared function without asking who else called it. That is the
    step I had written down that morning and did not take."""
    from kestrel_sovereign.features.security.permissions import fold_query

    await _log_filing(store, _FILING_ONE)

    for query in ("monkey", "password reset", "API key rotation"):
        assert fold_query(query), f"{query!r} must not fold away"
        matches, _ = await store.search_audit_log(query)
        assert matches == [], (
            f"{query!r} matches nothing here — it must return nothing, not "
            "everything"
        )

    # Scoping masking to KEY POSITIONS already saves the plain cases above,
    # which is why re-conflating the two functions survived a first mutation
    # pass. The split is still load-bearing, and this is where: a query that
    # is itself JSON with a sensitive key folds away under the stored-summary
    # rule (which masks that value) and becomes "%%".
    hostile = '{"api_key": "rotation"}'
    from kestrel_sovereign.features.security.permissions import (
        fold_stored_summary,
    )
    folded = fold_stored_summary(hostile)
    assert "rotation" not in folded and "***masked***" in folded, (
        "precondition: the summary rule DOES mask the value after a key "
        "position — that is correct for a stored row and wrong for a query"
    )
    assert "rotation" in fold_query(hostile), "the query rule must not"
    matches, _ = await store.search_audit_log(hostile)
    assert matches == [], "and the search must return nothing, not everything"


@pytest.mark.asyncio
async def test_a_benign_value_containing_key_stays_searchable(store):
    """Sensitive-key detection has to look at KEY POSITIONS. Scanning the whole
    serialized text meant `{"title": "orphaned keyboard worker"}` contained
    `key` and silently left the corpus — defeating the long truncated summaries
    the fallback exists to support."""
    await store.log_decision(
        feature_name="GitHubFeature",
        tool_name="create_github_issue",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"title": "orphaned keyboard worker", "body": "## Det',
    )

    matches, _ = await store.search_audit_log("orphaned keyboard worker")
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_a_truncated_fragment_with_quotes_still_matches(store):
    """Decoding only `\\uXXXX` left a pre-cut fragment containing a quote,
    newline or backslash stored escaped while the natural query carries the
    decoded character — a false absence, in a tool whose empty result promises
    only that detail *past* the cut is invisible."""
    await store.log_decision(
        feature_name="ComputeFeature",
        tool_name="run_script",
        action="tool_execution",
        decision="auto_mode_allowed",
        args_summary='{"cmd": "say \\"hello worker\\"", "body": "## Det',
    )

    matches, _ = await store.search_audit_log('say "hello worker"')
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_the_authorization_split_counts_rows_the_limit_hides(tmp_path):
    """Slicing before counting produced a false 'none authorized'.

    20 recent denials plus one older authorized call, with the default limit of
    20 — which sits INSIDE the 25-row bound, so no unusual input is needed. The
    older authorized row was sliced away before the split was computed, so the
    tool reported that nothing had been authorized and invited exactly the
    duplicate work it exists to prevent."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "sliced.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHubFeature", tool_name="create_github_issue",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"title": "orphaned worker THE REAL ONE"}',
    )
    for i in range(20):
        await store.log_decision(
            feature_name="GitHubFeature", tool_name="create_github_issue",
            action="tool_execution", decision="auto_denied",
            args_summary=f'{{"title": "orphaned worker denied {i}"}}',
        )

    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphaned worker")

    assert result.data["count"] == 21
    assert result.data["shown"] == 20
    assert result.data["omitted"] == 1
    assert "1 older not shown" in result.confirmation
    assert "authorized" in result.confirmation
    assert "none of the" not in result.confirmation, (
        "one authorized call exists; saying none did is the false conclusion"
    )


# --------------------------------------------------------------------------
# Review round 9 (opus gate, codex credits exhausted). Four findings, each with
# the test that was missing.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_completion_row_is_never_reported_as_work_that_may_not_have_happened(tmp_path):
    """`_AUTHORIZED_DECISIONS` lists permission decisions, so an outcome row —
    action='tool_outcome', decision='filed_and_dispatched' — is not in it, and
    the split folded it into "refused". A query matching only the outcome row
    (an issue number, a job id) then said the work "may never have happened"
    about the strongest completion evidence in the table."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "outcomes.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="talon_feature", tool_name="talon_file_and_claim.outcome",
        action="tool_outcome", decision="filed_and_dispatched",
        args_summary='{"reason_code": "OK", "filed": true, "issue_number": 1479}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="1479")

    assert result.data["count"] == 1
    assert result.data["outcomes"] == 1
    assert result.data["refused"] == 0
    assert "may never have happened" not in result.confirmation
    assert "1 completion(s) recorded" in result.confirmation
    assert "✓" in result.confirmation

    # A genuine refusal still reads as one.
    await store.log_decision(
        feature_name="talon_feature", tool_name="talon_file_and_claim",
        action="tool_execution", decision="blocked",
        args_summary='{"issue_number": 1480}',
    )
    result = await feature.security_audit_search(query="1480")
    assert result.data["refused"] == 1 and result.data["outcomes"] == 0
    assert "may never have happened" in result.confirmation

    # An outcome row whose decision this tool does not classify — a failure
    # (live: 6 `filing_failed` rows) or a success value the allowlist has
    # not learned — is neither a completion (no ✓) nor a refusal: it is an
    # outcome record to READ. Calling it "work that may never have happened"
    # turned a completion record into the false absence this tool exists to
    # prevent (round 19 review).
    for decision in ("filing_failed", "issue_created"):
        await store.log_decision(
            feature_name="talon_feature", tool_name="talon_file_and_claim.outcome",
            action="tool_outcome", decision=decision,
            args_summary=f'{{"decision": "{decision}", "issue_number": 1481}}',
        )
    result = await feature.security_audit_search(query="1481")
    assert result.data["outcomes"] == 0 and result.data["refused"] == 0
    assert result.data["unclassified_outcomes"] == 2
    assert "✓" not in result.confirmation and "?" in result.confirmation
    assert "may never have happened" not in result.confirmation
    assert "does not classify" in result.confirmation and "read them" in result.confirmation


@pytest.mark.asyncio
async def test_an_unparseable_row_without_a_sensitive_key_is_still_shown(tmp_path):
    """`summarize_args` truncates at 500 chars mid-structure, so every long
    issue body — the filings that motivated this tool — was unparseable and
    `remask_summary` withheld all of them. One rule at both projections now:
    shown unless a sensitive name sits in key position, exactly what the
    searchable projection already did."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.feature import SecurityFeature

    truncated = '{"title": "orphans the worker", "body": "' + "x" * 40  # cut mid-value
    # Repaired (the cut string and object closed), nothing masked, nothing
    # lost — and marked as reconstructed.
    assert remask_summary(truncated) == truncated + '"}...'
    leaking = '{"memo": "orphaned worker", "api_key": "sk-live-CUT'
    shown = remask_summary(leaking)
    assert "sk-live-CUT" not in shown
    # Round 13: the sensitive VALUE is masked and the rest is shown, rather
    # than withholding the whole row.
    assert "***MASKED***" in shown and "orphaned worker" in shown

    store = PermissionStore(str(tmp_path / "truncated.db"))
    await store.initialize()
    await _log_filing(store, truncated)
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphans the worker")
    assert result.data["count"] == 1
    assert "orphans the worker" in result.confirmation
    assert "not shown" not in result.confirmation


@pytest.mark.asyncio
async def test_a_name_marked_during_a_flush_is_not_dropped_by_it(tmp_path, monkeypatch):
    """`sync_dispatch_entries` snapshotted the dirty set for the write and then
    cleared the WHOLE set after commit. A name marked while the executemany or
    commit awaited was never written and no longer dirty — absent from the
    durable table, so after a restart its envelope rows read as prior actions."""
    import aiosqlite

    store = PermissionStore(str(tmp_path / "race.db"))
    await store.initialize()
    store._dispatch_entries_dirty.add("feat_a")

    real_executemany = aiosqlite.Connection.executemany

    async def late_mark_then_write(self, *args, **kwargs):
        # A concurrent mark_dispatch_entry whose own flush has not run yet.
        store._dispatch_entries_dirty.add("feat_late")
        return await real_executemany(self, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "executemany", late_mark_then_write)
    await store.sync_dispatch_entries()
    monkeypatch.undo()

    assert "feat_late" in store._dispatch_entries_dirty, (
        "the flush must discard only what it wrote"
    )
    durable = await store.sync_dispatch_entries()
    assert {"feat_a", "feat_late"} <= durable

    boot_two = PermissionStore(str(tmp_path / "race.db"))
    await boot_two.initialize()
    assert {"feat_a", "feat_late"} <= await boot_two.sync_dispatch_entries()


@pytest.mark.asyncio
async def test_marking_a_dispatch_entry_flushes_it_without_a_search(tmp_path):
    """The docstring promised durability would not wait for the first search;
    deleting the whole create_task block left every test green because each
    one called sync_dispatch_entries() itself."""
    db_path = tmp_path / "autoflush.db"
    store = PermissionStore(str(db_path))
    await store.initialize()

    store.mark_dispatch_entry("flushed_by_itself")
    assert store._dispatch_flush_tasks, "the mark must schedule its own flush"
    await asyncio.gather(*store._dispatch_flush_tasks)

    boot_two = PermissionStore(str(db_path))
    await boot_two.initialize()
    assert "flushed_by_itself" in await boot_two.sync_dispatch_entries()


@pytest.mark.asyncio
async def test_the_tool_masks_a_legacy_row_on_the_way_out(tmp_path):
    """The read-path re-mask is the guarantee; this drives it through the TOOL.
    `test_a_legacy_unmasked_secret_is_masked_on_the_way_out` calls
    remask_summary directly, so replacing the tool's call with the identity
    function left every test green."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "legacy.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="WalletAgent", tool_name="send_payment",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"api_key": "sk-live-LEGACY-LEAK", "memo": "orphaned"}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="orphaned")

    assert result.data["count"] == 1
    assert "sk-live-LEGACY-LEAK" not in result.confirmation
    assert "sk-live-LEGACY-LEAK" not in str(result.data)
    assert "***MASKED***" in result.confirmation


@pytest.mark.asyncio
async def test_a_parseable_row_with_a_lone_surrogate_does_not_kill_every_search(tmp_path):
    """Round 5 closed this on the unparseable branch only. json.dumps stores a
    lone surrogate happily; handing it back to SQLite from the scalar function
    raised, and a raised scalar fails the WHOLE query — one poisoned row and
    every search errored, permanently."""
    import json

    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "surrogate.db"))
    await store.initialize()
    await _log_filing(store, json.dumps({"cmd": "grep -E stalled logs/host.log"}))
    await _log_filing(store, json.dumps({"title": "note \ud83d"}))  # poisoned row
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="stalled")

    assert result.status is not ToolResultStatus.ERROR, result.error
    assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_a_query_with_a_backslash_matches_the_command_the_agent_ran(store):
    """The stored side of a parseable row is already decoded by json.loads; a
    query that went through the standard-escape decoder turned ``\\b`` into a
    backspace and reported the agent's own grep as never run."""
    import json

    command = 'grep -E "\\bstalled\\b" logs/host.log'
    await _log_filing(store, json.dumps({"cmd": command}))

    matches, _ = await store.search_audit_log(command)

    assert len(matches) == 1


@pytest.mark.asyncio
async def test_the_hook_door_keeps_the_shared_500_char_cap(tmp_path):
    """Round 3 removed the hook's private 200-char override so both writers
    truncate at the shared 500 and the read-back can state one bound.
    Restoring ``max_length=200`` left every test green — the claim was
    user-visible and unenforced."""
    from kestrel_sdk.hooks import HookInput
    from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
    from kestrel_sovereign.features.security.hooks import SecurityHook
    from kestrel_sovereign.features.security.permissions import PermissionLevel

    store = PermissionStore(str(tmp_path / "cap.db"))
    await store.initialize()
    await store.register_tool("GitHubFeature", "create_github_issue", PermissionLevel.ALLOW)
    hook = SecurityHook(store, ApprovalQueue(permission_store=store))

    body = ("x" * 260) + " the phrase past the old cap " + ("y" * 40)
    await hook.execute(HookInput(
        session_id="s",
        hook_event_name="PreToolUse",
        tool_name="create_github_issue",
        feature_name="GitHubFeature",
        tool_input={"title": "t", "body": body},
    ))

    logged = await store.get_audit_log(10)
    rows = [r for r in logged if r["tool"] == "create_github_issue"]
    assert rows and len(rows[0]["args_summary"] or "") > 200
    matches, _ = await store.search_audit_log("the phrase past the old cap")
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_a_tool_name_that_is_excluded_by_design_is_answered_not_searched(tmp_path):
    """`tool_name=<dispatch entry>` contradicted the unconditional exclusion in
    the WHERE clause, so the ordinary question "did I dispatch Talon about
    this?" always got the plain no-match text that blames truncation."""
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import SEARCH_TOOL_NAME

    store = PermissionStore(str(tmp_path / "excluded.db"))
    await store.initialize()
    store.mark_dispatch_entry("talon_feature")
    await store.log_decision(
        feature_name="TalonFeature", tool_name="talon_feature",
        action="subagent_dispatch", decision="auto_mode_allowed",
        args_summary='{"task": "look into whether the worker is orphaned"}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="orphaned", tool_name="talon_feature")
    assert result.data["excluded_by_design"] is True
    assert "dispatch envelope" in result.confirmation
    assert "No recorded tool call matched" not in result.confirmation

    result = await feature.security_audit_search(query="orphaned", tool_name=SEARCH_TOOL_NAME)
    assert result.data["excluded_by_design"] is True
    assert "own rows" in result.confirmation


@pytest.mark.asyncio
async def test_an_absurd_days_window_is_refused_not_an_overflow(tmp_path):
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "days.db"))
    await store.initialize()
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="anything", days=10**9)
    assert result.status is ToolResultStatus.ERROR
    assert "days must be <=" in (result.error or "")


@pytest.mark.asyncio
async def test_a_truncated_row_with_a_masked_token_is_still_searchable(tmp_path):
    """The ticket's own motivating shape: a filing carrying a token (masked at
    write time) beside a long body, cut mid-structure at 500. The row left the
    corpus for naming a sensitive key, and the no-match text blamed truncation
    for a phrase that sat INSIDE the cut."""
    from kestrel_sovereign.features.security.args_summary import (
        remask_summary, summarize_args,
    )
    from kestrel_sovereign.features.security.feature import SecurityFeature

    stored = summarize_args({
        "token": "ghp_secret_value",
        "repo": "KestrelSovereignAI/kestrel-talon",
        "body": "Killing a tracked job orphans the worker. " * 20,
    })
    assert len(stored) == 500 and "ghp_secret" not in stored

    store = PermissionStore(str(tmp_path / "masked_token.db"))
    await store.initialize()
    await _log_filing(store, stored)
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="orphans the worker")
    assert result.data["count"] == 1
    assert "orphans the worker" in result.confirmation
    assert "ghp_secret" not in result.confirmation

    # The oracle stays closed at the VALUE: a legacy raw secret beside a
    # benign field is masked through the end of its (cut) string.
    shown = remask_summary('{"memo": "orphaned worker", "api_key": "sk-live-CUT')
    assert "sk-live" not in shown and "orphaned worker" in shown


@pytest.mark.asyncio
async def test_days_window_excludes_an_iso_row_older_than_the_window(tmp_path):
    """The cutoff clause normalizes created_at before comparing; a plain
    `created_at >= ?` left every test green while a ~44h-old ISO row leaked
    into a 1-day window ('T' sorts above ' ')."""
    import sqlite3
    from datetime import datetime, timedelta, timezone

    db_path = tmp_path / "iso_ts.db"
    store = PermissionStore(str(db_path))
    await store.initialize()
    # Same calendar date as the cutoff, earlier in the day: compared as raw
    # text, "T" (84) at index 10 sorts above the cutoff's " " (32), so an
    # unnormalized comparison admits a row that is OUTSIDE the window.
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    old_stamp = cutoff.replace(hour=0, minute=0, second=1, microsecond=0).isoformat()
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO security_audit_log "
        "(feature_name, tool_name, action, decision, user_choice, "
        " args_summary, created_at) VALUES (?,?,?,?,?,?,?)",
        ("GitHubFeature", "create_github_issue", "tool_execution",
         "auto_mode_allowed", None, _FILING_ONE, old_stamp),
    )
    raw.commit()
    raw.close()

    matches, _ = await store.search_audit_log("orphans the worker", days=1)
    assert matches == [], f"a row stamped {old_stamp} is outside a 1-day window"
    matches, _ = await store.search_audit_log("orphans the worker", days=2)
    assert len(matches) == 1


# --------------------------------------------------------------------------
# Round 14: a sensitive key whose value is a container is masked through the
# balanced close (or to the cut), in both projections.
# --------------------------------------------------------------------------

def test_a_truncated_row_masks_every_value_shape_and_keeps_the_rest():
    """One structural masker for truncated rows too: the cut is repaired,
    parsed and masked by key position. A text scanner had to know every
    value shape and missed one per round (container, escaped string, prose)."""
    from kestrel_sovereign.features.security.args_summary import remask_summary

    cases = [
        # a dict value; the scanner stopped at its first comma
        '{"memo": "orphaned worker", "secrets": {"a": "AAA-LEAK", "b": "sk-live-BBB-LEAK"}, "body": "## Deta',
        # a list value with a nested object and a brace inside a string
        '{"api_keys": ["sk-1-LEAK", {"k": "sk-2-LEAK}"}], "memo": "orphaned worker"',
        # a container the cut left open
        '{"memo": "orphaned worker", "secrets": {"a": "AAA-LEAK", "b": "sk-live-BB-LEAK',
        # a string, an unterminated string, a bare scalar, an escaped quote
        '{"memo": "orphaned worker", "token": "abc-LEAK", "n": 1',
        '{"memo": "orphaned worker", "token": "ab-LEAK',
        '{"memo": "orphaned worker", "token": 12345, "n": 1',
        '{"memo": "orphaned worker", "token": "a\\"b-LEAK", "n": 1',
        # the round-16 P1: a nested JSON string whose value is escaped
        '{"payload": "{\\"password\\": \\"correct horse battery staple-LEAK\\"}", "memo": "orphaned worker", "body": "deploy the widget',
        '{"memo": "orphaned worker", "credit_card": "4111 1111 1111 1111", "body": "deploy the wi',
    ]
    for cut in cases:
        shown = remask_summary(cut)
        assert "LEAK" not in shown and "12345" not in shown and "1111" not in shown, (cut, shown)
        assert "***MASKED***" in shown, (cut, shown)
        assert "orphaned worker" in shown, (cut, shown)


@pytest.mark.asyncio
async def test_a_container_valued_secret_in_an_unmaskable_row_is_not_searchable(tmp_path):
    """The round-7 oracle one value-shape over: with the dict's tail surviving
    the mask, ``sk-live-B`` / ``sk-live-BB`` hit and ``zz`` missed, walking
    the credential out a character at a time while the display showed
    ***MASKED*** for the first entry only."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "container.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="WalletAgent", tool_name="send_payment",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"memo": "orphaned worker", "secrets": {"a": "AAA-LEAK", "b": "sk-live-BBB-LEAK"}, "body": "## Deta',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    shown = await feature.security_audit_search(query="orphaned worker")
    assert shown.data["count"] == 1
    assert "LEAK" not in shown.confirmation and "LEAK" not in str(shown.data)
    assert '"secrets": "***MASKED***"' in str(shown.data)
    for probe in ("sk-live-B", "sk-live-BB", "AAA-LE"):
        result = await feature.security_audit_search(query=probe)
        assert result.data["count"] == 0, f"{probe!r} matched: the oracle is open"

    from kestrel_sovereign.features.security.permissions import fold_stored_summary
    folded = fold_stored_summary('{"memo": "orphaned worker", "secrets": {"a": "AAA-LEAK", "b": "sk-live-BBB-LEAK"}, "body": "## Deta')
    assert "leak" not in folded and "orphaned worker" in folded and "## deta" in folded


def test_the_dead_sensitive_key_regex_is_gone():
    # Round 13 replaced drop-the-row with mask-and-keep; the regex that
    # decided the drop had no caller left and a comment describing the old
    # rule sat above its alias.
    import kestrel_sovereign.features.security.args_summary as args_summary
    import kestrel_sovereign.features.security.permissions as permissions
    assert not hasattr(args_summary, "SENSITIVE_JSON_KEY")
    assert not hasattr(permissions, "_SENSITIVE_JSON_KEY")


# --------------------------------------------------------------------------
# Round 15: masking substitutes, never removes; the store's own read path is
# the re-mask door; a JSON-encoded payload string is masked in every branch.
# --------------------------------------------------------------------------

def _talon_outcome_row(length: int = 900) -> str:
    # The shape kestrel_feature_talon persists at [:1000]: a long body, then
    # the fields the read-back exists to recover, past any 500-char cut.
    body = "## Details\\n" + ("the worker orphaned its lease " * 40)
    row = json.dumps({
        "action": "filed_and_dispatched",
        "body": body,
        "phrase": "unmistakable-tail-phrase",
        "issue_number": 3107,
        "job_id": "job-77",
    })
    assert len(row) >= length, len(row)
    return row


def test_remask_summary_never_shrinks_a_row():
    from kestrel_sovereign.features.security.args_summary import remask_summary

    row = _talon_outcome_row()
    shown = remask_summary(row)
    assert "unmistakable-tail-phrase" in shown and '"issue_number": 3107' in shown
    # An unparseable long row keeps its tail too (repair may drop up to eight
    # characters AT the cut to reach a parseable point, never more).
    cut = row[:850]
    assert cut[-48:-8] in remask_summary(cut)


@pytest.mark.asyncio
async def test_a_match_past_500_characters_is_shown_with_the_fields_it_matched_on(tmp_path):
    """A read-side cap of 500 returned a matched Talon outcome that showed
    neither the phrase the match was made on nor issue_number — the fields
    the tool exists to recover — and then blamed the writer for the cut."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "long.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="TalonFeature", tool_name="talon_file_and_claim.outcome",
        action="tool_outcome", decision="filed_and_dispatched",
        args_summary=_talon_outcome_row(),
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    result = await feature.security_audit_search(query="unmistakable-tail-phrase")
    assert result.data["count"] == 1
    shown = result.data["matches"][0]["args_summary"]
    assert "unmistakable-tail-phrase" in shown
    assert '"issue_number": 3107' in shown and "job-77" in shown

    miss = await feature.security_audit_search(query="zz-no-such-phrase")
    assert "500 characters now" not in miss.confirmation
    assert "other writers set their own cap" in miss.confirmation


@pytest.mark.asyncio
async def test_the_store_read_path_masks_a_legacy_row_for_every_caller(tmp_path):
    """The re-mask lived only in the tool; the store returned its own rows
    raw, so the next caller of search_audit_log would have leaked what the
    tool hid. One door, in the store."""
    store = PermissionStore(str(tmp_path / "store-door.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="WalletAgent", tool_name="send_payment",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary='{"api_key": "sk-live-LEAKY", "memo": "orphans the worker"}',
    )
    rows, _ = await store.search_audit_log("orphans the worker")
    assert len(rows) == 1
    assert "sk-live-LEAKY" not in rows[0]["args_summary"]
    assert "***MASKED***" in rows[0]["args_summary"]
    assert "orphans the worker" in rows[0]["args_summary"]


@pytest.mark.asyncio
async def test_a_json_encoded_payload_string_is_masked_in_every_branch(tmp_path):
    """mask_sensitive only saw dict keys, so a sensitive key inside a
    string-encoded payload survived the parseable branch (and its fold was
    searchable) while the unparseable branch masked the same bytes."""
    from kestrel_sovereign.features.security.args_summary import (
        mask_sensitive, remask_summary, summarize_args,
    )
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    nested = json.dumps({"api_key": "sk-NESTED"})
    row = json.dumps({"payload": nested, "memo": "orphans the worker"})
    assert "sk-NESTED" not in remask_summary(row)
    assert "sk-nested" not in fold_stored_summary(row)
    assert "sk-NESTED" not in summarize_args({"payload": nested})
    assert mask_sensitive({"payload": nested})["payload"] != nested
    assert mask_sensitive("orphaned keyboard worker") == "orphaned keyboard worker"

    store = PermissionStore(str(tmp_path / "nested.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="WalletAgent", tool_name="send_payment",
        action="tool_execution", decision="auto_mode_allowed", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    shown = await feature.security_audit_search(query="orphans the worker")
    assert shown.data["count"] == 1 and "sk-NESTED" not in str(shown.data)
    assert (await feature.security_audit_search(query="sk-NEST")).data["count"] == 0


# --------------------------------------------------------------------------
# Round 16: the region mask is for JSON, not prose — the write path must not
# destroy a body that merely quotes a sensitive-looking key.
# --------------------------------------------------------------------------

_BODY_QUOTING_A_KEY = (
    "## The defect\n\nThe `\"api_key\":` handling is the red herring here.\n"
    "Killing a tracked job kills the wrapper and orphans the worker.\n"
    "The job registry stores the wrapper PID; the wrapper does not exec.\n"
)


def test_a_benign_body_that_quotes_a_key_survives_the_write_path_intact():
    from kestrel_sovereign.features.security.args_summary import mask_sensitive, summarize_args

    assert mask_sensitive(_BODY_QUOTING_A_KEY) == _BODY_QUOTING_A_KEY
    stored = summarize_args({"title": "t", "body": _BODY_QUOTING_A_KEY})
    assert json.loads(stored)["body"] == _BODY_QUOTING_A_KEY
    # A JSON-shaped string is still region-masked.
    assert "sk-NESTED" not in mask_sensitive(json.dumps({"api_key": "sk-NESTED"}))
    assert "sk-NESTED" not in mask_sensitive("  [" + json.dumps({"api_key": "sk-NESTED"}) + "]")


@pytest.mark.asyncio
async def test_a_benign_body_that_quotes_a_key_is_still_searchable_end_to_end(tmp_path):
    """Reproduced through the real hook path's summarizer: three of four body
    lines were gone from the audit row permanently, and the phrases the
    read-back exists to find matched nothing."""
    from kestrel_sovereign.features.security.args_summary import summarize_args
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "prose.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHub", tool_name="create_github_issue",
        action="tool_execution", decision="auto_mode_allowed",
        args_summary=summarize_args({"title": "orphaned worker", "body": _BODY_QUOTING_A_KEY}),
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    for phrase in ("orphans the worker", "wrapper does not exec", "red herring"):
        result = await feature.security_audit_search(query=phrase)
        assert result.data["count"] == 1, phrase
        assert phrase in result.data["matches"][0]["args_summary"]


def test_a_truncated_row_quoting_a_key_in_prose_loses_nothing():
    """The unparseable branch used to scan raw text and could not tell a
    structural key from one quoted inside a string value. Repairing and
    parsing the cut row masks by key position, so prose is never touched."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    cut = '{"title": "t", "body": "The `\\"api_key\\":` handling is the red herring here. Killing a job orphans the worker'
    shown = remask_summary(cut)
    assert "red herring" in shown and "orphans the worker" in shown and "***MASKED***" not in shown
    folded = fold_stored_summary(cut)
    assert "red herring" in folded and "orphans the worker" in folded
    # ...while a real bare scalar after a structural key is still masked.
    assert "12345" not in remask_summary('{"token": 12345, "n": 1')
    assert "12345" not in remask_summary('{"token": 12345')


# --------------------------------------------------------------------------
# Round 17: the text scanner is gone. A truncated row is REPAIRED (its open
# string and containers closed), parsed, and masked by the one structural
# masker every parseable row gets; a JSON-valued string is decided by parsing.
# --------------------------------------------------------------------------

def test_complete_truncated_json_closes_what_the_cut_left_open():
    from kestrel_sovereign.features.security.args_summary import complete_truncated_json as c

    assert c('{"a": 1, "b": {"c": [1, 2') == {"a": 1, "b": {"c": [1, 2]}}
    assert c('{"a": "unterminated str') == {"a": "unterminated str"}
    assert c('{"a": "x", "api_k') == {"a": "x"}                 # a key the cut split is dropped
    assert c('{"a": "x", "api_key"') == {"a": "x"}              # a key with no value is dropped
    assert c('{"a": "x", "api_key":') == {"a": "x", "api_key": None}
    # ...and a cut or dangling key LONGER than the eight-character trim: only
    # the key-position drop can repair these (trimming cannot reach the comma).
    assert c('{"a": "x", "social_security_numb') == {"a": "x"}
    assert c('{"a": "x", "social_security_number"') == {"a": "x"}
    assert c('{"a": "x",') == {"a": "x"}
    assert c('{"a": tr') == {"a": None}   # a cut bare literal: a null PLACEHOLDER (shown with the marker)
    assert c('{"a": "caf\\u00') == {"a": "caf"}                # a cut inside an escape
    assert c('{"a": "back\\') == {"a": "back"}                 # a cut right after a backslash
    assert c('{"a": "x"}...') == {"a": "x"}                      # summarize_args' marker
    assert c('{"a": "x", "b": "y') == {"a": "x", "b": "y"}
    assert c('{"list": [{"k": "v"}, {"k') == {"list": [{"k": "v"}, {}]}   # the cut key is dropped
    # not JSON: prose, a bare scalar, misnested brackets
    assert c("orphaned keyboard worker") is None
    assert c('"just a string"') is None
    assert c('{"a": [1}') is None
    assert c("") is None and c(None) is None


def test_the_escaped_nested_secret_is_masked_and_not_searchable_end_to_end(tmp_path):
    """Round 16's P1: a nested JSON string's value begins with an escaped
    quote, which the scanner read as a bare scalar and masked only to the
    first space — the tail reached the model and the hit/no-hit oracle."""
    import asyncio
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = '{"payload": "{\\"password\\": \\"correct horse battery staple\\"}", "body": "deploy the widget BBB'
    assert "horse" not in fold_stored_summary(row)
    assert "deploy the widget" in fold_stored_summary(row)

    async def scenario():
        store = PermissionStore(str(tmp_path / "escaped.db"))
        await store.initialize()
        await store.log_decision(
            feature_name="Legacy", tool_name="deploy_widget", action="tool_execution",
            decision="auto_mode_allowed", args_summary=row,
        )
        feature = SecurityFeature.__new__(SecurityFeature)
        feature.permission_store = store
        shown = await feature.security_audit_search(query="deploy the widget")
        assert shown.data["count"] == 1 and "horse" not in str(shown.data)
        for probe in ("horse battery staple", "horse battery stapleX", "correct horse"):
            assert (await feature.security_audit_search(query=probe)).data["count"] == 0, probe

    asyncio.run(scenario())


def test_a_markdown_body_starting_with_a_bracket_survives_the_write_path():
    """Round 16's P2: deciding "is this JSON?" by first character made a body
    that opened with a link or an [x] checkbox eat its own prose."""
    from kestrel_sovereign.features.security.args_summary import mask_sensitive, summarize_args

    body = (
        "[Talon run 41](https://example/41)\n\n"
        'The config block opens with `"api_key": {` and is never closed in the\n'
        "snippet below. Killing the tracked job orphans the worker, and the\n"
        "kill reports success while the work keeps running.\n"
    )
    assert mask_sensitive(body) == body
    assert mask_sensitive("[x] rotate the api_key: done") == "[x] rotate the api_key: done"
    stored = summarize_args({"repo": "r", "title": "orphaned worker", "body": body})
    assert json.loads(stored)["body"] == body
    # ...and a string that IS JSON is still masked, whatever it starts with.
    assert "sk-NESTED" not in mask_sensitive(" \n" + json.dumps([{"api_key": "sk-NESTED"}]))
    assert "sk-NESTED" not in mask_sensitive(json.dumps({"api_key": "sk-NESTED"})[:-3])  # cut


def test_a_tool_audit_row_keeps_its_reason_and_masks_its_embedded_args():
    """tool_audit writes '<reason> | args=<json>'. The prose prefix is kept,
    the embedded JSON is repaired and masked; a prose-only row (no args) is
    shown and folded as it is, having no key position at all."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = 'tool not in palette [wallet] | args={"memo": "orphaned worker", "api_key": "sk-live-LEAK", "body": "deplo'
    shown = remask_summary(row)
    assert shown.startswith("tool not in palette [wallet] | args=")
    assert "sk-live-LEAK" not in shown and "orphaned worker" in shown and "***MASKED***" in shown
    folded = fold_stored_summary(row)
    assert "sk-live" not in folded and "orphaned worker" in folded and "not in palette" in folded

    prose = "refused: tool not in palette"
    assert remask_summary(prose) == prose
    assert fold_stored_summary(prose) == prose


def test_a_row_whose_json_cannot_be_repaired_is_withheld_not_shown_raw():
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    broken = '{"api_key": "sk-live-LEAK"] "memo": "orphaned worker"'   # misnested: no repair
    assert remask_summary(broken) == "(summary truncated past repair; not shown)"
    assert fold_stored_summary(broken) == ""


# --------------------------------------------------------------------------
# Round 18: the prefix before a repairable region is never handed back raw;
# both store read paths mask.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unrepairable_region_before_a_repairable_one_is_withheld(tmp_path):
    """The repair loop scanned up to eight bracket positions and returned the
    text before the first one that repaired RAW — including an earlier JSON
    region that did not repair: shown verbatim, and walked out by hit/no-hit."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = '{"api_key": "sk-live-LEAK"] trailing {"b": 1}'
    assert remask_summary(row) == "(summary truncated past repair; not shown)"
    assert fold_stored_summary(row) == ""

    store = PermissionStore(str(tmp_path / "two-regions.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="Legacy", tool_name="t", action="tool_execution",
        decision="auto_mode_allowed", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    for probe in ("sk-live-L", "sk-live-LEAK", "trailing"):
        assert (await feature.security_audit_search(query=probe)).data["count"] == 0, probe
    # A prose prefix that merely brackets a word still keeps its embedded JSON.
    assert "***MASKED***" in remask_summary('refused [wallet] | args={"api_key": "sk-LEAK", "m": "x"')
    # The guard is not keyed on a double-quoted key position: unrepairable
    # structure in any quoting before a repairable region withholds the row.
    for shape in (
        "{'api_key': 'sk-live-LEAK'} then {\"b\": 1}",
        '["sk-live-LEAK"] trailing {"b": 1}',
        "{'k': 1} then {\"b\": 1}",
    ):
        assert remask_summary(shape) == "(summary truncated past repair; not shown)", shape
        assert fold_stored_summary(shape) == "", shape


# --------------------------------------------------------------------------
# Round 20: a BLOB row cannot poison every search; repair slack belongs to
# the read path; a JSON prefix keeps its prose tail.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_blob_args_summary_row_does_not_kill_every_search(tmp_path):
    """The column has TEXT affinity but stores bytes as bytes; a BLOB reached
    the registered scalar function, raised, and a raised scalar fails the
    WHOLE query — every search errored, permanently."""
    import sqlite3

    db_path = tmp_path / "blob.db"
    store = PermissionStore(str(db_path))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHub", tool_name="create_github_issue", action="tool_execution",
        decision="auto_mode_allowed", args_summary='{"title": "orphans the worker"}',
    )
    raw = sqlite3.connect(db_path)
    # Same tool name as the real row, so a tool-name match pulls the BLOB row
    # into the result set and it reaches the display re-mask too — the fold
    # guard alone left get_audit_log and every tool-name search raising
    # (round 22 review).
    raw.execute(
        "INSERT INTO security_audit_log (feature_name, tool_name, action, decision, args_summary)"
        " VALUES (?,?,?,?,?)", ("Foreign", "create_github_issue", "tool_execution", "auto_mode_allowed", b"\x00orphans the worker"),
    )
    raw.execute(
        "INSERT INTO security_audit_log (feature_name, tool_name, action, decision, args_summary)"
        " VALUES (?,?,?,?,?)", ("Foreign", "create_github_issue", "tool_execution", "auto_mode_allowed", 3),
    )
    raw.commit(); raw.close()

    matches, _ = await store.search_audit_log("orphans the worker")
    assert [m["args_summary"] for m in matches] == ['{"title": "orphans the worker"}']
    matches, _ = await store.search_audit_log("create_github_issue")
    assert len(matches) == 3 and all("orphans" not in (m["args_summary"] or "") or m["args_summary"].startswith("{") for m in matches)
    assert {m["args_summary"] for m in matches} >= {"(summary not text; not shown)"}
    recent = await store.get_audit_log(10)
    assert len(recent) == 3 and all(isinstance(r["args_summary"], str) for r in recent)


def test_the_write_path_keeps_prose_after_a_json_prefix_and_never_trims():
    from kestrel_sovereign.features.security.args_summary import mask_sensitive, summarize_args

    assert mask_sensitive('{"api_key": "x"} yes') == '{"api_key": "***MASKED***"} yes'
    assert mask_sensitive('{"token": 1} FAILED') == '{"token": "***MASKED***"} FAILED'
    assert mask_sensitive('  {"token": 1}...') == '  {"token": "***MASKED***"}...'
    assert json.loads(summarize_args({"body": '{"api_key": "x"} yes'}))["body"] == '{"api_key": "***MASKED***"} yes'
    # A cut nested payload is NOT repaired on the write path (nothing was cut
    # there): it is prose and passes through unchanged rather than trimmed.
    cut = '{"api_key": "x", "n": tr'
    assert mask_sensitive(cut) == cut


def test_the_read_path_still_repairs_a_cut_nested_payload_with_slack():
    """A nested payload cut inside a bare literal needs the trim slack to
    parse; the read path passes it, so the secret inside is still masked."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = '{"payload": "{\\"password\\": \\"sk-live-LEAK\\", \\"n\\": tr", "b": 1'
    assert "LEAK" not in remask_summary(row) and "***MASKED***" in remask_summary(row)
    assert "leak" not in fold_stored_summary(row)


@pytest.mark.asyncio
async def test_a_cut_nested_payload_inside_a_parseable_row_is_masked_in_both_projections(tmp_path):
    """The outer row parses, so both read paths took the json.loads branch and
    called mask_sensitive with no repair slack; the nested payload's cut
    string never parsed and its secret stayed raw — shown, and walkable
    from hit/no-hit (round 20 review)."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = json.dumps({"payload": '{"api_key": "sk-live-NESTEDCUT', "memo": "orphans the worker"})
    assert "NESTEDCUT" not in remask_summary(row) and "***MASKED***" in remask_summary(row)
    assert "nestedcut" not in fold_stored_summary(row) and "orphans the worker" in fold_stored_summary(row)

    store = PermissionStore(str(tmp_path / "nested-cut.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="Legacy", tool_name="t", action="tool_execution",
        decision="auto_mode_allowed", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    shown = await feature.security_audit_search(query="orphans the worker")
    assert shown.data["count"] == 1 and "NESTEDCUT" not in str(shown.data)
    for probe in ("sk-live-N", "sk-live-NESTEDCUT"):
        assert (await feature.security_audit_search(query=probe)).data["count"] == 0, probe


# --------------------------------------------------------------------------
# Round 22: one bucket per row; NULL columns stay in the corpus; a top-level
# JSON-string row is masked like any other.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_outcome_row_carrying_an_authorization_value_is_counted_once(tmp_path):
    """The buckets were not disjoint: an outcome row whose decision was an
    authorization value counted as authorized AND unclassified, and refused,
    derived by subtraction, came out -1 — "-1 NOT authorized" about a row
    the same line called authorized."""
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "once.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="Foreign", tool_name="t.outcome", action="tool_outcome",
        decision="allowed", args_summary='{"note": "orphaned worker"}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphaned worker")
    d = result.data
    assert (d["authorized"], d["outcomes"], d["unclassified_outcomes"], d["refused"]) == (0, 0, 1, 0)
    assert d["authorized"] + d["outcomes"] + d["unclassified_outcomes"] + d["refused"] == d["count"]
    assert "-1" not in result.confirmation and "?" in result.confirmation


@pytest.mark.asyncio
async def test_rows_with_a_null_action_or_tool_name_are_still_searchable(tmp_path):
    """`NULL <> x`, `NULL NOT LIKE x` and `NULL NOT IN (...)` are all NULL, so
    the exclusion predicates silently dropped a foreign writer's row and the
    caller was told absence was weak evidence — for a row that was there."""
    import sqlite3

    db_path = tmp_path / "nulls.db"
    store = PermissionStore(str(db_path))
    await store.initialize()
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO security_audit_log (feature_name, tool_name, action, decision, args_summary)"
        " VALUES (?,?,?,?,?)", ("Foreign", "create_github_issue", None, "auto_mode_allowed", '{"title": "orphans the worker"}'),
    )
    raw.execute(
        "INSERT INTO security_audit_log (feature_name, tool_name, action, decision, args_summary)"
        " VALUES (?,?,?,?,?)", ("Foreign", None, "tool_execution", "auto_mode_allowed", '{"title": "orphans the worker too"}'),
    )
    raw.commit(); raw.close()

    rows, _ = await store.search_audit_log("orphans the worker")
    assert len(rows) == 2, rows
    rows, _ = await store.search_audit_log("orphans the worker too")
    assert len(rows) == 1 and rows[0]["tool"] is None


@pytest.mark.asyncio
async def test_a_top_level_json_string_row_is_masked_in_both_projections(tmp_path):
    """Both read paths short-circuited on a non-dict parse, so a row that IS a
    JSON-encoded payload (the same bytes the walker masks one level down)
    was shown raw and searchable raw."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = json.dumps(json.dumps({"api_key": "sk-live-TOPLEVEL", "memo": "orphans the worker — Échec"}))
    assert "TOPLEVEL" not in remask_summary(row) and "***MASKED***" in remask_summary(row)
    folded = fold_stored_summary(row)
    assert "toplevel" not in folded and "orphans the worker" in folded
    # Its escapes are decoded for matching, as a nested payload's are.
    assert "— échec" in folded, folded
    assert remask_summary(json.dumps("just prose")) == json.dumps("just prose")

    store = PermissionStore(str(tmp_path / "toplevel.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="Foreign", tool_name="t", action="tool_execution",
        decision="auto_mode_allowed", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    shown = await feature.security_audit_search(query="orphans the worker")
    assert shown.data["count"] == 1 and "TOPLEVEL" not in str(shown.data)
    assert (await feature.security_audit_search(query="worker — Échec")).data["count"] == 1
    assert (await feature.security_audit_search(query="sk-live-TOP")).data["count"] == 0


# --------------------------------------------------------------------------
# Round 23: escapes inside a nested payload are decoded for matching; the
# headline never drops the unclassified clause.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_nested_payloads_unicode_escapes_are_searchable(tmp_path):
    """json.loads decodes the OUTER row; a string value that is itself a
    serialized JSON document still carries literal \\u2014, and fold_query
    decodes the query side — neither spelling matched, a false absence in
    the tool that exists to prevent one."""
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    nested = json.dumps({"title": "Killing a tracked job — orphans the worker", "city": "Échec"})
    row = json.dumps({"repo": "K/kt", "payload": nested})
    assert "\\u2014" in row  # the escape survives the outer dump
    folded = fold_stored_summary(row)
    assert "job — orphans" in folded and "échec" in folded

    store = PermissionStore(str(tmp_path / "nested-escapes.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHub", tool_name="create_github_issue", action="tool_execution",
        decision="auto_mode_allowed", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    for phrase in ("Killing a tracked job — orphans", "Échec"):
        assert (await feature.security_audit_search(query=phrase)).data["count"] == 1, phrase


@pytest.mark.asyncio
async def test_the_headline_keeps_the_unclassified_clause_beside_an_authorization(tmp_path):
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "headline.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="talon_feature", tool_name="talon_file_and_claim", action="tool_execution",
        decision="auto_mode_allowed", args_summary='{"issue": "orphan worker alpha"}',
    )
    await store.log_decision(
        feature_name="talon_feature", tool_name="talon_file_and_claim.outcome", action="tool_outcome",
        decision="issue_created", args_summary='{"issue": "orphan worker alpha", "issue_number": 7}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    result = await feature.security_audit_search(query="orphan worker alpha")
    assert result.data["authorized"] == 1 and result.data["unclassified_outcomes"] == 1
    assert "does not classify" in result.confirmation and "read them" in result.confirmation
    assert "which is not proof it succeeded" not in result.confirmation


# --------------------------------------------------------------------------
# Round 24: a value nested past the interpreter's limit is refused at every
# door, never raised; a reconstructed row is visibly reconstructed.
# --------------------------------------------------------------------------

def test_a_deeply_nested_value_never_raises_at_any_door():
    """RecursionError is a RuntimeError: it escaped every ValueError guard. On
    the write path the hook failed closed and the call was wrongly DENIED
    with no audit row; on the read path the registered scalar raised and
    every search failed."""
    from kestrel_sovereign.features.security.args_summary import mask_sensitive, remask_summary, summarize_args
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    deep = "[" * 1000 + "]" * 1000
    stored = summarize_args({"repo": "o/r", "title": "orphans the worker", "body": deep})
    assert json.loads(stored)["title"] == "orphans the worker"
    # The decoder accepts 1000 levels; the walk cannot follow them, so the
    # payload is withheld rather than raised or left raw.
    assert json.loads(stored)["body"] == "(payload nested past the limit; not shown)"
    assert isinstance(summarize_args({"body": "[" * 100000}), str)
    # A REAL structure (not a string) nested past the limit: the string branch
    # cannot absorb it, so summarize_args' own guard is the door.
    deep_obj: list = []
    cur = deep_obj
    for _ in range(1000):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    assert summarize_args({"body": deep_obj}) == "(args could not be summarized)"
    # A row that is itself one structure nested past the limit parses (the C
    # decoder follows it) and overflows the walk: withheld, not raised.
    assert remask_summary(deep) == "(summary could not be re-masked; not shown)"
    assert fold_stored_summary(deep) == ""
    # A parseable row whose string VALUE is nested past the limit still masks
    # its other fields.
    row = json.dumps({"api_key": "sk-live-LEAK", "body": "[" * 5000})
    assert "sk-live-LEAK" not in remask_summary(row) and "***MASKED***" in remask_summary(row)
    assert "sk-live" not in fold_stored_summary(row)
    assert "sk-live-LEAK" not in mask_sensitive(row)


def test_a_repaired_row_is_shown_with_the_truncation_marker():
    """The repair closed what the cut left open and rendered a cut field as
    null; without a marker both read paths handed the reader a well-formed
    record and `"draft": null` read as a fact about the prior call."""
    from kestrel_sovereign.features.security.args_summary import remask_summary

    for cut in ('{"repo": "o/r", "title": "orphans the worker", "issue_number":', '{"ok": tru', '{"a": 1, "n": 1.2e'):
        shown = remask_summary(cut)
        assert shown.endswith("..."), shown
        assert json.loads(shown[:-3]), shown
    assert remask_summary('{"repo": "o/r", "title": "orphans the worker", "issue_number":') == '{"repo": "o/r", "title": "orphans the worker", "issue_number": null}...'
    # A row that parses as written carries no marker.
    assert remask_summary('{"repo": "o/r"}') == '{"repo": "o/r"}'
    # tool_audit's prose prefix keeps its place before the repaired JSON.
    assert remask_summary('refused [wallet] | args={"api_key": "sk-LEAK", "m": "x"').endswith('"m": "x"}...')


# --------------------------------------------------------------------------
# Round 25: prose that closes its bracket is prose; only a reconstruction
# is marked.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_guardrail_reason_with_a_bracketed_argument_name_keeps_its_row(tmp_path):
    """input_guardrails writes "argument 'files[0]' exceeds maximum length"
    and tool_audit stores it as "<reason> | args=<json>". The prefix guard
    scanned past the closing bracket to the quote and withheld the intact
    row from both read paths — a false absence for the refused call."""
    from kestrel_sovereign.features.security.args_summary import remask_summary
    from kestrel_sovereign.features.security.feature import SecurityFeature
    from kestrel_sovereign.features.security.permissions import fold_stored_summary

    row = ("Tool 'create_github_issue' argument 'files[0]' exceeds maximum length (10001 > 10000)"
           ' | args={"files": ["a.txt"], "api_key": "sk-LEAK", "memo": "orphans the worker"}')
    shown = remask_summary(row)
    assert shown.startswith("Tool 'create_github_issue' argument 'files[0]'") and "sk-LEAK" not in shown
    assert "***MASKED***" in shown and "orphans the worker" in shown
    assert "orphans the worker" in fold_stored_summary(row) and "sk-leak" not in fold_stored_summary(row)

    store = PermissionStore(str(tmp_path / "guardrail.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="Guardrail", tool_name="create_github_issue", action="tool_execution",
        decision="blocked", args_summary=row,
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store
    assert (await feature.security_audit_search(query="orphans the worker")).data["count"] == 1
    # Unrepaired structure INSIDE a bracket still withholds the row.
    assert remask_summary("{'api_key': 'sk-live-LEAK'} then {\"b\": 1}") == "(summary truncated past repair; not shown)"


def test_an_intact_row_reaching_the_repair_path_is_not_marked_as_reconstructed():
    """tool_audit's '<reason> | args=<json>' fails json.loads only for its
    prose prefix; its JSON parses as written, and marking it made a plain
    record indistinguishable from a cut one."""
    from kestrel_sovereign.features.security.args_summary import remask_summary, repair_json_text

    row = 'Unknown feature tool: wallet_feature | args={"task": "pay rent"}'
    assert remask_summary(row) == row
    assert remask_summary('refused [wallet] | args={"api_key": "sk-LEAK", "m": "x"}') == 'refused [wallet] | args={"api_key": "***MASKED***", "m": "x"}'
    # ...and the same prefix with a CUT payload is marked.
    assert remask_summary('Unknown feature tool: wallet_feature | args={"task": "pay re').endswith("...")
    assert repair_json_text('{"a": 1}') == ({"a": 1}, False)
    assert repair_json_text('{"a": 1}...') == ({"a": 1}, True)
    assert repair_json_text('{"a": 1') == ({"a": 1}, True)
    assert repair_json_text('{"a": tr') == ({"a": None}, True)


# --------------------------------------------------------------------------
# Round 26: a nested reconstruction is marked like a top-level one.
# --------------------------------------------------------------------------

def test_a_reconstructed_nested_payload_is_marked_in_the_display():
    """The read path repairs a cut nested payload (closing its string,
    inventing null for a cut key) and showed the result unmarked while the
    identical reconstruction at the top level carried the marker — the
    false precision the marker exists to prevent."""
    from kestrel_sovereign.features.security.args_summary import remask_summary

    cut_inside = json.dumps({"payload": '{"api_key": "sk-live-SECRET", "body": "the quick brown fo', "note": "x"})
    shown = remask_summary(cut_inside)
    assert "SECRET" not in shown and "***MASKED***" in shown
    assert shown.endswith("..."), shown
    assert json.loads(shown[:-3])["note"] == "x"
    # The same shape carried as a top-level JSON string.
    top = json.dumps('{"api_key": "sk-live-SECRET", "body": "the quick brown fo')
    assert remask_summary(top).endswith("...") and "SECRET" not in remask_summary(top)
    # An INTACT nested payload is masked without a marker; a row with no
    # nested payload at all is untouched.
    intact = json.dumps({"payload": json.dumps({"api_key": "sk-live-SECRET", "body": "fine"}), "note": "x"})
    assert not remask_summary(intact).endswith("...") and "SECRET" not in remask_summary(intact)
    plain = json.dumps({"note": "x", "body": "the quick brown fo"})
    assert remask_summary(plain) == plain
    # A cut nested payload with NOTHING to mask stays as written and is
    # marked; an intact sibling payload beside it does not unset that.
    two = json.dumps({"a": '{"body": "the quick brown fo', "b": json.dumps({"k": "v"})})
    shown = remask_summary(two)
    assert shown.endswith("...") and json.loads(shown[:-3])["a"] == '{"body": "the quick brown fo'


@pytest.mark.asyncio
async def test_a_non_positive_or_non_integer_limit_or_days_is_refused_not_a_false_absence(tmp_path):
    """limit is applied before the empty check: limit=0 turned a real match
    into 'No recorded tool call matched … absence is weak evidence', and
    days=0 (cutoff = now) did the same. The `< 1` guards prevent it and had
    no test; a mutation deleting either killed nothing."""
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.features.security.feature import SecurityFeature

    store = PermissionStore(str(tmp_path / "limits.db"))
    await store.initialize()
    await store.log_decision(
        feature_name="GitHub", tool_name="create_github_issue", action="tool_execution",
        decision="auto_mode_allowed", args_summary='{"title": "orphans the worker"}',
    )
    feature = SecurityFeature.__new__(SecurityFeature)
    feature.permission_store = store

    for kwargs in ({"limit": 0}, {"limit": -3}, {"limit": "x"}, {"days": 0}, {"days": "x"}):
        result = await feature.security_audit_search(query="orphans the worker", **kwargs)
        assert result.status is ToolResultStatus.ERROR, (kwargs, result)
        assert "No recorded tool call matched" not in (result.error or "") + (result.confirmation or "")
    ok = await feature.security_audit_search(query="orphans the worker", limit=1, days=1)
    assert ok.data["count"] == 1
