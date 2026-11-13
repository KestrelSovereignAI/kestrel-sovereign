"""Stored-reply UI errors cannot consume typed control."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request

from kestrel_sovereign.endpoints.conversations import get_message
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.storage.db.interface import QueryError
from kestrel_sovereign.storage.encryption import DecryptionError


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ordinary", "control", "decrypt", "decrypt_control"])
async def test_optional_read_normalization_retains_control_and_ordinary_contract(kind):
    error = (DecryptionError("decrypt read failed") if kind.startswith("decrypt")
             else QueryError("stored reply read failed"))
    if "control" in kind:
        error.__cause__ = ExecutionAuthorityError("original read authority lost")

    async def fail(*args):
        raise error

    storage = SimpleNamespace(get_conversation_message=fail)
    agent = SimpleNamespace(storage=storage, privacy_agent=SimpleNamespace(can_store=lambda _kind: True))
    app = FastAPI()
    app.state.agent = agent
    request = Request({"type": "http", "app": app, "method": "GET", "path": "/"})
    if "control" in kind:
        with pytest.raises(type(error)) as caught:
            await get_message(request, 1)
        assert caught.value is error
    else:
        with pytest.raises(HTTPException) as caught:
            await get_message(request, 1)
        assert caught.value.status_code == (409 if kind == "decrypt" else 500)
