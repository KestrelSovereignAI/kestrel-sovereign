"""Actual vector/metadata consumers with declared original native lease fault inputs."""

import pytest

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.storage.vector.python import PurePythonBackend
from tests.integration.native_memory_original_control import actual_native_loss
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.native_memory_original_control import (
    native_vector_consumers as native_vector_consumers,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["retriever", "consolidator", "saved-items"])
async def test_actual_vector_read_keeps_original_native_lease_control(
    native_vector_consumers, monkeypatch, door
):
    memory, saved, factory, service, requests, lease, did = native_vector_consumers
    actual = PurePythonBackend.knn
    readbacks, originals = [], []

    async def after_actual_native_knn(backend, *args, **kwargs):
        result = await actual(backend, *args, **kwargs)
        readbacks.append(result)
        # Declared post-read callback boundary forwarding actual vector SQL;
        # no synthetic session, SQL result, ACK, scope or authority exception.
        actual_native_loss(lease, originals)
        return result

    monkeypatch.setattr(PurePythonBackend, "knn", after_actual_native_knn)
    result = failure = None
    try:
        if door == "retriever":
            result = await memory.retriever._semantic_similarities_via_vector_backend(
                [1.0, 0.0, 0.0], did, None, 5
            )
        elif door == "saved-items":
            result = await saved._search_via_vector_backend(
                factory, "local fixture", [1.0, 0.0, 0.0], None, 5
            )
        else:
            result = await memory.consolidator._knn_episode_ids("local fixture", 5)
    except BaseException as error:
        failure = error
    assert readbacks and originals, (
        "did not forward actual SQLite SQLAlchemy kNN + original retired native lease"
    )
    assert failure is not None and all(
        error in tuple(_explicit_control_chain(failure)) for error in originals
    ), f"native vector control became fallback {result!r}"
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "door",
    [
        "consolidator",
        "saved-vector",
        "saved-legacy",
        "saved-text",
        "saved-write",
        "retriever",
    ],
)
async def test_actual_profile_callback_keeps_original_native_lease_control(
    native_vector_consumers, monkeypatch, door
):
    memory, saved, factory, service, requests, lease, did = native_vector_consumers
    actual = ProviderEmbeddingService.current_profile_id
    profiles, originals = [], []

    def after_actual_profile(embedding):
        result = actual(embedding)
        profiles.append(result)
        # Declared metadata callback fault from genuine retired PG authority;
        # not evidence that the built-in pure profile hash performs native IO.
        actual_native_loss(lease, originals)
        return result

    monkeypatch.setattr(
        ProviderEmbeddingService, "current_profile_id", after_actual_profile
    )
    result = failure = None
    try:
        if door == "consolidator":
            result = await memory.consolidator._knn_episode_ids("local fixture", 5)
        elif door == "retriever":
            result = await memory.retriever.retrieve(
                "local fixture", did, read_only=True
            )
        elif door == "saved-vector":
            result = await saved._search_via_vector_backend(
                factory, "local fixture", [1.0, 0.0, 0.0], None, 5
            )
        elif door == "saved-write":
            result = await saved.save_item(
                item_type="excerpt",
                name="profile write fixture",
                content="local fixture",
                pin_to_ipfs=False,
            )
        elif door == "saved-legacy":
            result = await saved._legacy_in_python_search(
                "local fixture", [1.0, 0.0, 0.0], None, 5
            )
        else:
            result = await saved._text_search("local fixture", None, 5)
    except BaseException as error:
        failure = error
    assert profiles and all(profiles) and originals
    if door == "saved-write":
        assert (
            await saved.db.fetchval(
                "SELECT COUNT(*) FROM saved_items WHERE name=?",
                ("profile write fixture",),
            )
            == 1
        )
    assert failure is not None and all(
        error in tuple(_explicit_control_chain(failure)) for error in originals
    ), f"native profile control became fallback {result!r}"
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["consolidator", "saved-items"])
async def test_actual_route_metadata_callback_denial_is_not_unavailable(
    native_vector_consumers, monkeypatch, door
):
    memory, saved, factory, service, requests, lease, did = native_vector_consumers
    originals = []

    def declared_space_callback(adapter):
        # Supported adapter metadata extension, no fake SDK/native receipt.
        actual_native_loss(lease, originals)

    monkeypatch.setattr(
        OpenAIAdapter, "embedding_space_id", declared_space_callback, raising=False
    )
    result = failure = None
    try:
        result = (
            memory.consolidator if door == "consolidator" else saved
        )._get_embedding_service()
    except BaseException as error:
        failure = error
    assert originals, (
        "actual production provider construction did not invoke metadata callback"
    )
    assert failure is not None and originals[0] in tuple(
        _explicit_control_chain(failure)
    ), f"native metadata denial became unavailable {result!r}"
    assert is_execution_control_error(failure)


@pytest.mark.asyncio
async def test_actual_saved_profile_registry_keeps_original_native_control(
    native_vector_consumers, monkeypatch
):
    from kestrel_sovereign.storage.sqla import embedding_profile

    memory, saved, factory, service, requests, lease, did = native_vector_consumers
    actual = embedding_profile.upsert_embedding_profile
    writes, originals = [], []

    async def after_actual_registry(*args, **kwargs):
        result = await actual(*args, **kwargs)
        writes.append(True)
        actual_native_loss(lease, originals)
        return result

    monkeypatch.setattr(
        embedding_profile, "upsert_embedding_profile", after_actual_registry
    )
    result = failure = None
    try:
        result = await saved.save_item(
            item_type="excerpt",
            name="registry write fixture",
            content="local fixture",
            pin_to_ipfs=False,
        )
    except BaseException as error:
        failure = error
    assert writes and originals and requests
    assert await saved.db.fetchval("SELECT COUNT(*) FROM embedding_profiles") == 1
    assert failure is not None and originals[0] in tuple(
        _explicit_control_chain(failure)
    ), f"native registry control became successful save {result!r}"
    assert is_execution_control_error(failure)
