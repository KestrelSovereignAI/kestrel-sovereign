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
    """`fold_searchable` decodes JSON before folding, and a summary truncated
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
    from kestrel_sovereign.features.security.permissions import fold_searchable

    assert fold_searchable('{"t": "hi \\ud83d\\ude00"}') == '{"t": "hi 😀"}'.casefold()


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
