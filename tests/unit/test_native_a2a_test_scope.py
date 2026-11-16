"""The native PostgreSQL fixture exposes required types without sharing tables."""

import pytest


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_disposable_native_scope_can_resolve_original_pgvector_type(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        schema, = await backend.fetch_one("SELECT current_schema()")
        assert schema.startswith("native_a2a_context_")
        assert await backend.fetch_one("SELECT vector_dims('[1,2,3]'::vector)") == (3,)
    finally:
        await backend.close()
