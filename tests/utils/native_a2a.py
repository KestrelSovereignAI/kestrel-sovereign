"""Owned database scopes shared by the native A2A storage contracts."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.db.postgres import PostgresBackend
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.utils.postgres_schema import disposable_postgres_schema, with_search_path


def signing_identity(did):
    """Build actual hybrid keys and their public DID verification document."""
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    pair = generate_hybrid_keypair()
    return pair, {
        "id": did,
        "verificationMethod": build_verification_methods(did, pair.public_keys()),
    }


def manager_for(backend, scope, *, owns_backend=True, **callbacks):
    """Construct the canonical native stores on one owned test backend."""
    from kestrel_sovereign.a2a.task_manager import TaskManager

    return TaskManager.from_backend(
        backend,
        scope=scope,
        owns_backend=owns_backend,
        include_memory=False,
        include_feedback=False,
        **callbacks,
    )


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def native_a2a_backend_factory(request, tmp_path):
    """Independent connections to one disposable database scope per test."""
    if request.param == "sqlite":
        yield lambda: SQLiteBackend(str(tmp_path / "native-context.db"))
        return
    # Do not inherit production DATABASE_URL/KESTREL_DATABASE_URL here.
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is required for the PostgreSQL contract")
    admin = PostgresBackend(url, min_pool_size=1, max_pool_size=2)
    await admin.connect()
    try:
        async with disposable_postgres_schema(admin, "native_a2a_context") as schema:
            yield lambda: PostgresBackend(
                with_search_path(url, schema), min_pool_size=1, max_pool_size=2
            )
    finally:
        await admin.close()
