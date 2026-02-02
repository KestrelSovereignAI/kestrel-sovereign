"""
Async wait utilities to replace hardcoded sleep statements.

These utilities provide condition-based waiting with proper timeouts,
making tests faster and more reliable than fixed sleeps.

Usage:
    # Instead of: await asyncio.sleep(5)
    # Use:
    await wait_until(
        lambda: storage.file_exists(hash),
        timeout=5.0,
        message="File not stored"
    )
"""
import asyncio
import inspect
from typing import Callable, TypeVar, Awaitable, Optional, Union, Any

T = TypeVar('T')


class WaitTimeoutError(Exception):
    """Raised when wait_until times out."""
    pass


async def wait_until(
    condition: Callable[[], Union[bool, Awaitable[bool]]],
    timeout: float = 5.0,
    interval: float = 0.1,
    message: str = "Condition not met"
) -> None:
    """
    Wait until a condition becomes True.

    Supports both sync and async condition functions.

    Args:
        condition: Callable that returns True when ready (sync or async)
        timeout: Maximum time to wait in seconds
        interval: Time between condition checks in seconds
        message: Error message if timeout occurs

    Raises:
        WaitTimeoutError: If condition not met within timeout

    Examples:
        # Sync condition
        await wait_until(lambda: file_path.exists(), timeout=5.0)

        # Async condition
        await wait_until(
            lambda: storage.file_exists(hash),
            timeout=5.0,
            message="File not stored"
        )
    """
    start = asyncio.get_event_loop().time()

    while True:
        # Handle both sync and async conditions
        result = condition()
        if inspect.isawaitable(result):
            result = await result

        if result:
            return

        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            raise WaitTimeoutError(f"{message} (waited {elapsed:.2f}s)")

        await asyncio.sleep(interval)


async def wait_for_value(
    getter: Callable[[], Union[T, Awaitable[T]]],
    expected: T,
    timeout: float = 5.0,
    interval: float = 0.1,
    message: Optional[str] = None
) -> T:
    """
    Wait until a value matches expected.

    Args:
        getter: Callable that returns the value to check (sync or async)
        expected: The expected value
        timeout: Maximum time to wait
        interval: Time between checks
        message: Custom error message

    Returns:
        The final value (which equals expected)

    Examples:
        count = await wait_for_value(
            lambda: db.count_rows("messages"),
            expected=10,
            timeout=3.0
        )
    """
    async def check_value():
        result = getter()
        if inspect.isawaitable(result):
            result = await result
        return result == expected

    msg = message or f"Value did not reach {expected}"
    await wait_until(check_value, timeout=timeout, interval=interval, message=msg)

    # Return final value
    result = getter()
    if inspect.isawaitable(result):
        result = await result
    return result


async def wait_for_not_none(
    getter: Callable[[], Union[Optional[T], Awaitable[Optional[T]]]],
    timeout: float = 5.0,
    interval: float = 0.1,
    message: str = "Value remained None"
) -> T:
    """
    Wait until a value is not None.

    Args:
        getter: Callable that returns the value
        timeout: Maximum wait time
        interval: Check interval
        message: Error message

    Returns:
        The non-None value

    Examples:
        result = await wait_for_not_none(
            lambda: cache.get("key"),
            timeout=2.0
        )
    """
    async def check_not_none():
        result = getter()
        if inspect.isawaitable(result):
            result = await result
        return result is not None

    await wait_until(check_not_none, timeout=timeout, interval=interval, message=message)

    result = getter()
    if inspect.isawaitable(result):
        result = await result
    return result


async def wait_for_http_ready(
    client: Any,
    url: str,
    timeout: float = 30.0,
    interval: float = 0.5,
    expected_status: int = 200
) -> None:
    """
    Wait for HTTP endpoint to return expected status.

    Replaces: time.sleep(1) in server startup loops

    Args:
        client: httpx.AsyncClient or similar
        url: URL to check
        timeout: Maximum wait time
        interval: Check interval
        expected_status: Expected HTTP status code

    Examples:
        await wait_for_http_ready(client, "http://localhost:8000/health")
    """
    async def check_ready():
        try:
            response = await client.get(url, timeout=interval * 2)
            return response.status_code == expected_status
        except Exception:
            return False

    await wait_until(
        check_ready,
        timeout=timeout,
        interval=interval,
        message=f"HTTP endpoint {url} not ready (expected status {expected_status})"
    )


async def wait_for_db_connection(
    connect_func: Callable[[], Awaitable[Any]],
    timeout: float = 10.0,
    interval: float = 0.5
) -> Any:
    """
    Wait for database to become available.

    Replaces: time.sleep(5) in database startup waits

    Args:
        connect_func: Async function that attempts connection
        timeout: Maximum wait time
        interval: Retry interval

    Returns:
        The connection object returned by connect_func

    Examples:
        conn = await wait_for_db_connection(
            lambda: asyncpg.connect(DATABASE_URL),
            timeout=10.0
        )
    """
    last_error = None

    async def try_connect():
        nonlocal last_error
        try:
            await connect_func()
            return True
        except Exception as e:
            last_error = e
            return False

    try:
        await wait_until(
            try_connect,
            timeout=timeout,
            interval=interval,
            message="Database connection not available"
        )
    except WaitTimeoutError:
        raise WaitTimeoutError(
            f"Database connection not available after {timeout}s. "
            f"Last error: {last_error}"
        )

    return await connect_func()


async def wait_for_process_ready(
    process: Any,
    check_func: Callable[[], Union[bool, Awaitable[bool]]],
    timeout: float = 30.0,
    interval: float = 0.5
) -> None:
    """
    Wait for a subprocess to be ready (e.g., server started).

    Args:
        process: subprocess.Popen object
        check_func: Function to check if process is ready
        timeout: Maximum wait time
        interval: Check interval

    Raises:
        WaitTimeoutError: If process not ready in time
        RuntimeError: If process exits unexpectedly
    """
    start = asyncio.get_event_loop().time()

    while True:
        # Check if process died
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"Process exited with code {process.returncode}. "
                f"stderr: {stderr.decode() if stderr else 'N/A'}"
            )

        # Check readiness
        result = check_func()
        if inspect.isawaitable(result):
            result = await result

        if result:
            return

        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            raise WaitTimeoutError(f"Process not ready after {elapsed:.2f}s")

        await asyncio.sleep(interval)


async def poll_with_backoff(
    func: Callable[[], Union[T, Awaitable[T]]],
    check: Callable[[T], bool],
    timeout: float = 60.0,
    initial_interval: float = 0.5,
    max_interval: float = 5.0,
    backoff_factor: float = 1.5
) -> T:
    """
    Poll a function with exponential backoff until check passes.

    Useful for waiting on external services that may take variable time.

    Args:
        func: Function to poll
        check: Function to check if result is acceptable
        timeout: Maximum total wait time
        initial_interval: Starting poll interval
        max_interval: Maximum poll interval
        backoff_factor: Multiplier for interval on each iteration

    Returns:
        The final result from func

    Examples:
        pod_status = await poll_with_backoff(
            lambda: runpod.get_pod(pod_id),
            check=lambda p: p['status'] == 'RUNNING',
            timeout=300.0,
            initial_interval=5.0
        )
    """
    start = asyncio.get_event_loop().time()
    interval = initial_interval

    while True:
        result = func()
        if inspect.isawaitable(result):
            result = await result

        if check(result):
            return result

        elapsed = asyncio.get_event_loop().time() - start
        if elapsed >= timeout:
            raise WaitTimeoutError(f"Polling timed out after {elapsed:.2f}s")

        await asyncio.sleep(interval)
        interval = min(interval * backoff_factor, max_interval)
