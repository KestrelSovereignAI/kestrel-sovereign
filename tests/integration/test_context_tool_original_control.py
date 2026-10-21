"""Actual PG context producers, localhost SDK summaries and outer tool control.

Typed compatibility carriers wrap observed native originals. The owner/request
projection is a component input, not Agent genesis/READY or authenticated HTTP.
All SDK traffic uses the fixture's loopback server and inert credentials.
"""

import pytest

from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
async def test_actual_context_summary_local_sdk_fixture(native_memory_owners):
    n = native_memory_owners
    n.service.providers[0]["model"] = "gpt-4o"
    n.service.set_model_preference(
        "gpt-4o",
        vendor="openai",
        route="local-fixture",
        validate=False,
    )
    result = await n.service.generate(
        system_prompt="Local SDK fixture.",
        user_prompt="Return the local summary.",
    )
    assert (getattr(result, "content", None) or result) == "local summary"
    assert len(n.requests) == 1 and not n.requests[0].get("stream")
    assert (
        await n.observer.fetchval(
            f'SELECT total_tokens FROM "{n.schema}".model_usage WHERE model_id=$1',
            "gpt-4o",
        )
        == 4
    )


# Each carrier represents one actual native original explicitly carried through
# a typed compatibility boundary, not a fabricated driver or close receipt.
_CASES = [
    ("history", "native"),
    ("history", "conn"),
    ("history", "value"),
    ("select-last", "value"),
    ("select-ids", "value"),
    ("resolve", "value"),
    ("summary-selection", "native"),
    ("summary-selection", "value"),
    ("summary-run", "native"),
    ("summary-run", "conn"),
    ("summary-run", "value"),
    ("mark", "native"),
    ("mark", "attr"),
    ("compact-dry", "native"),
    ("compact-dry", "value"),
    ("compact-run", "native"),
    ("compact-run", "conn"),
    ("compact-run", "value"),
    ("exclude", "native"),
    ("exclude", "attr"),
    ("restore", "native"),
    ("restore", "value"),
    ("restore", "attr"),
    ("stash", "native"),
    ("stash", "value"),
    ("stash", "attr"),
    ("pop", "native"),
    ("pop", "value"),
    ("apply", "native"),
    ("apply", "value"),
    ("list", "native"),
    ("list", "attr"),
    ("drop", "native"),
    ("drop", "value"),
    ("save", "native"),
    ("save", "conn"),
    ("save", "value"),
    ("save", "import"),
    ("peek", "native"),
    ("peek", "value"),
    ("hierarchical", "native"),
    ("hierarchical", "conn"),
    ("hierarchical", "value"),
    ("recursive-last", "native"),
    ("recursive-last", "value"),
    ("recursive-last", "attr"),
    ("recursive-last", "key"),
    ("recursive-marker", "key"),
]
_CARRIERS = {
    "conn": ConnectionError,
    "value": ValueError,
    "attr": AttributeError,
    "key": KeyError,
    "import": ImportError,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("door,carrier", _CASES, ids=[f"{d}-{c}" for d, c in _CASES])
@pytest.mark.parametrize("kind", ["native-control", "ordinary"])
async def test_actual_context_producers_and_tool_consumers_preserve_control(
    native_memory_owners,
    monkeypatch,
    door,
    carrier,
    kind,
):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from kestrel_sovereign.agent.context_manager import ContextManager
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody,
        _explicit_control_chain,
        execution_commit_outcome,
        is_execution_control_error,
    )
    from kestrel_sovereign.features.context.feature import ContextFeature
    from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
    from tests.integration.test_execution_authority_postgres import GenerationFence

    n = native_memory_owners
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    monkeypatch.setattr(n.pg_storage, "llm_service", n.service, raising=False)
    n.service.providers[0]["model"] = "gpt-4o"
    n.service.set_model_preference(
        "gpt-4o",
        vendor="openai",
        route="local-fixture",
        validate=False,
    )
    store, db = n.pg_storage.conversation, n.pg_storage.db
    metadata = {}
    if door in {"pop", "apply", "list", "drop", "save", "peek"}:
        metadata = {
            "stashed": True,
            "stash_id": "native-context",
            "stash_name": "native",
        }
    elif door == "restore":
        metadata = {"excluded_from_context": True}
    for index in range(4):
        await store.add_conversation(
            "user",
            f"Native stored context row {index}. " * 4,
            session_id="native-context",
            metadata=metadata,
        )
    ids = [
        r[0]
        for r in await db.fetchall(
            "SELECT id FROM conversation_history WHERE agent_id=? ORDER BY id",
            (n.did,),
        )
    ]
    assert len(ids) == 4
    import json

    before_metadata = [
        json.loads(r[0])
        for r in await n.observer.fetch(
            f'SELECT metadata FROM "{n.schema}".conversation_history WHERE agent_id=$1 ORDER BY id',
            n.did,
        )
    ]
    context = ContextManager(
        n.pg_storage, model="gpt-4o", agent_id=n.did, llm_service=n.service
    )
    # Explicit component projection; not full Agent construction/admission/READY.
    agent = SimpleNamespace(
        did=n.did,
        agent_id=n.did,
        storage=n.pg_storage,
        _raw_storage=n.pg_storage,
        llm_service=n.service,
        context_manager=context,
    )
    feature = ContextFeature(agent)
    original = OSError("actual context producer native release acknowledgement lost")
    reached, sources, fired = [], [], []
    pool, armed = n.backend._pool, False

    @asynccontextmanager
    async def lose_ack(native):
        async with native as connection:
            yield connection
        if armed and not fired:
            fired.append(True)
            raise original

    write_doors = {"mark", "exclude", "restore", "stash", "pop", "apply", "drop"}
    insert_doors = {"summary-run", "compact-run", "hierarchical"}
    target = (
        "UPDATE CONVERSATION_HISTORY"
        if door in write_doors
        else "INSERT INTO CONVERSATION_HISTORY"
        if door in insert_doors
        else "INSERT INTO SAVED_ITEMS"
        if door == "save"
        else "FROM CONVERSATION_HISTORY"
    )
    methods = {}
    for name in ("execute", "execute_commit", "fetchone", "fetchall"):
        actual = getattr(db, name)
        methods[name] = actual

        async def intercept(sql, *args, _actual=actual, **kwargs):
            nonlocal armed
            selected = target in " ".join(sql.upper().split()) and not reached
            if selected:
                reached.append(sql)
                if kind == "native-control":
                    armed = True
            try:
                result = await _actual(sql, *args, **kwargs)
            except BaseException as error:
                if selected:
                    sources.append(error)
                    if carrier != "native":
                        raise _CARRIERS[carrier](
                            "declared typed native context carrier"
                        ) from error
                raise
            if selected and kind == "ordinary":
                error_type = _CARRIERS.get(carrier, OSError)
                raise error_type(
                    "ordinary context callback unavailable after acknowledged SQL"
                )
            return result

        monkeypatch.setattr(db, name, intercept)
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    if kind == "native-control":
        n.backend._pool = PostgresPoolAdapter(pool, lose_ack)
    failure = result = None
    try:
        try:
            if door == "history":
                result = await context.conversation_manager.get_conversation_history()
            elif door.startswith("select-"):
                result = await context.get_messages_for_selection(
                    "messages" if door == "select-ids" else "last_n",
                    ",".join(map(str, ids)) if door == "select-ids" else "4",
                )
            elif door == "resolve":
                result = await feature._resolve_target_messages("last_4")
            elif door == "summary-selection":
                result = await feature.summarize_section("time_range", "before_today")
            elif door == "summary-run":
                result = await feature.summarize_section("last_n", "4")
            elif door == "mark":
                result = await feature.mark_content("protect", "last_4")
            elif door == "compact-dry":
                result = await feature.compact_context(dry_run=True)
            elif door == "compact-run":
                result = await feature.compact_context(keep_recent=0, force=True)
            elif door == "exclude":
                result = await feature.exclude_from_context(
                    "last_4", "native component test"
                )
            elif door == "restore":
                result = await feature.restore_excluded()
            elif door == "stash":
                result = await feature.context_stash("last_4")
            elif door == "pop":
                result = await feature.context_stash_pop("native-context")
            elif door == "apply":
                result = await feature.context_stash_apply("native-context")
            elif door == "list":
                result = await feature.context_stash_list()
            elif door == "drop":
                result = await feature.context_stash_drop("native-context")
            elif door == "save":
                result = await feature.context_stash_save("native-context")
            elif door == "peek":
                result = await feature.context_stash_peek("native-context")
            elif door == "hierarchical":
                result = await feature.hierarchical_compact(
                    chunk_size=50, keep_recent=0, max_depth=1
                )
            elif door == "recursive-marker":
                result = await feature.recursive_query(f"compacted:{ids[0]}", "native?")
            else:
                result = await feature.recursive_query("last_4", "native?")
        except BaseException as error:
            failure = error
        assert reached, (
            f"actual context source was not reached: {door} {failure!r} {result!r}"
        )
        if kind == "native-control":
            assert (
                fired
                and sources
                and execution_commit_outcome(sources[0]) == "committed"
            )
            assert original in tuple(_explicit_control_chain(sources[0]))
            assert failure is not None and is_execution_control_error(failure), (
                f"actual context producer/tool control became ordinary fallback: "
                f"door={door} carrier={carrier} failure={failure!r} result={result!r}"
            )
            assert sources[0] in tuple(_explicit_control_chain(failure))
            if door in insert_doors:
                assert (
                    await n.observer.fetchval(
                        f'SELECT count(*) FROM "{n.schema}".conversation_history WHERE agent_id=$1',
                        n.did,
                    )
                    == 5
                ), "first marker COMMIT occurred; no audit/next effect may follow"
            elif door in write_doors:
                # The first actual row UPDATE committed; original control must
                # stop the loop before a second row is marked/audited.
                rows = await n.observer.fetch(
                    f'SELECT metadata FROM "{n.schema}".conversation_history WHERE agent_id=$1 ORDER BY id',
                    n.did,
                )
                changed = [
                    json.loads(r[0]) != before
                    for r, before in zip(rows, before_metadata)
                ]
                assert sum(changed) == 1
        else:
            from kestrel_sdk.tools.result import ToolResultStatus

            assert failure is None, (
                f"ordinary availability behavior changed: {failure!r}"
            )
            if door == "history" or door.startswith("select-"):
                assert result == []
            elif door == "resolve":
                assert result == ([], None)
            else:
                assert result.status is (
                    ToolResultStatus.OK
                    if door == "compact-dry"
                    else ToolResultStatus.ERROR
                )
        if door not in insert_doors and door != "save":
            assert not n.requests, "cheap context operations must not call a model"
    finally:
        n.backend._pool = pool
        n.backend._execution_custody = None


@pytest.mark.asyncio
async def test_actual_pg_stash_save_serializes_native_timestamp_without_mutating_rows(
    native_memory_owners,
    monkeypatch,
):
    import json
    from datetime import datetime

    from kestrel_sovereign.agent.context_manager import ContextManager

    n = native_memory_owners
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    monkeypatch.setattr(n.pg_storage, "llm_service", n.service, raising=False)
    for index in range(2):
        await n.pg_storage.conversation.add_conversation(
            "user",
            f"Native saved stash {index}.",
            metadata={"stashed": True, "stash_id": "native-save"},
        )
    original = await n.pg_storage.conversation.get_stashed_messages("native-save")
    assert len(original) == 2 and all(
        isinstance(m["created_at"], datetime) for m in original
    )
    context = ContextManager(
        n.pg_storage, model="gpt-4o", agent_id=n.did, llm_service=n.service
    )
    result = failure = None
    try:
        result = await context.stash_save("native-save", summary="Local native stash.")
    except Exception as error:
        failure = error
    assert failure is None and result["success"], (
        f"real PG stash cannot be saved: {failure!r} {result!r}"
    )
    saved = await n.pg_saved.get_by_id(result["saved_item_id"])
    payload = json.loads(saved.content)
    assert [datetime.fromisoformat(m["created_at"]) for m in payload["messages"]] == [
        m["created_at"] for m in original
    ]
    again = await n.pg_storage.conversation.get_stashed_messages("native-save")
    assert again == original and all(
        isinstance(m["created_at"], datetime) for m in again
    )


@pytest.mark.asyncio
async def test_actual_healthy_mark_stash_list_pop_retains_native_rows(
    native_memory_owners,
    monkeypatch,
):
    import json
    from types import SimpleNamespace

    from kestrel_sdk.tools.result import ToolResultStatus

    from kestrel_sovereign.agent.context_manager import ContextManager
    from kestrel_sovereign.execution_custody import ExecutionCustody
    from kestrel_sovereign.features.context.feature import ContextFeature
    from tests.integration.test_execution_authority_postgres import GenerationFence

    n = native_memory_owners
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    store = n.pg_storage.conversation
    for index in range(4):
        await store.add_conversation("user", f"Native healthy context {index}.")
    ids = [
        r[0]
        for r in await n.pg_storage.db.fetchall(
            "SELECT id FROM conversation_history WHERE agent_id=? ORDER BY id",
            (n.did,),
        )
    ]
    context = ContextManager(
        n.pg_storage, model="gpt-4o", agent_id=n.did, llm_service=n.service
    )
    feature = ContextFeature(
        SimpleNamespace(
            did=n.did,
            agent_id=n.did,
            storage=n.pg_storage,
            llm_service=n.service,
            context_manager=context,
        )
    )
    n.backend._execution_custody = ExecutionCustody(GenerationFence())
    target = "ids:" + ",".join(map(str, ids))
    marked = await feature.mark_content("droppable", target)
    assert marked.status is ToolResultStatus.OK and marked.data["marked_count"] == 4
    stashed = await feature.context_stash(target, name="Native healthy")
    assert stashed.status is ToolResultStatus.OK and stashed.data["stashed_count"] == 4
    listed = await feature.context_stash_list()
    assert listed.status is ToolResultStatus.OK and len(listed.data["stashes"]) == 1
    popped = await feature.context_stash_pop(stashed.data["stash_id"])
    assert popped.status is ToolResultStatus.OK and popped.data["restored_count"] == 4
    rows = await n.observer.fetch(
        f'SELECT id,metadata FROM "{n.schema}".conversation_history WHERE agent_id=$1 ORDER BY id',
        n.did,
    )
    originals = [json.loads(r[1]) for r in rows if r[0] in ids]
    assert len(originals) == 4 and all(
        m["context_priority"] == "droppable" and m["stashed"] is False
        for m in originals
    )
    assert not n.requests, "healthy cheap context operations must not call a provider"
