"""Positive static controls for the public native/generic context boundary."""

from types import TracebackType
from typing import Literal, assert_type

import httpx
import httpx2

from kestrel_sovereign.execution_custody import owned_native_http_client


class SuppressingClient:
    async def __aenter__(self) -> str:
        return "entered generic client"

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return True


class HTTPXChild(httpx.AsyncClient):
    pass


class HTTPX2Child(httpx2.AsyncClient):
    pass


async def value() -> str:
    return "assigned"


async def native_httpx_assignment(client: httpx.AsyncClient, fail: bool) -> str:
    context = owned_native_http_client(client)
    async with context as entered:
        assert_type(entered, httpx.AsyncClient)
        if fail:
            raise RuntimeError("before assignment")
        result = await value()
    return result


async def native_httpx2_assignment(client: httpx2.AsyncClient, fail: bool) -> str:
    context = owned_native_http_client(client)
    async with context as entered:
        assert_type(entered, httpx2.AsyncClient)
        if fail:
            raise RuntimeError("before assignment")
        result = await value()
    return result


async def preserved_subtypes(first: HTTPXChild, second: HTTPX2Child) -> None:
    async with owned_native_http_client(first) as entered_first:
        assert_type(entered_first, HTTPXChild)
    async with owned_native_http_client(second) as entered_second:
        assert_type(entered_second, HTTPX2Child)


async def generic_contract(client: SuppressingClient) -> None:
    context = owned_native_http_client(client)
    assert_type(await context.__aenter__(), str)
    assert_type(await context.__aexit__(None, None, None), bool)


async def native_exit_contracts(first: httpx.AsyncClient, second: httpx2.AsyncClient) -> None:
    first_context = owned_native_http_client(first)
    second_context = owned_native_http_client(second)
    await first_context.__aenter__()
    assert_type(await first_context.__aexit__(None, None, None), Literal[False] | None)
    await second_context.__aenter__()
    assert_type(await second_context.__aexit__(None, None, None), Literal[False] | None)
