"""The live-route deadline and probes turn a silent route into a named outcome (#3269).

The route here is real: a local socket that accepts every connection and never
answers, called through the real ``ollama`` client, which (like the Ollama
adapter's chat client) carries no timeout of its own. Without the helpers in
``live_routes`` each of these calls waits forever.
"""

import asyncio
import os
import re
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import ollama
import pytest

from tests.integration import live_routes
from tests.integration.live_routes import (
    LiveCallBudget,
    bounded_live_call,
    require_answering_chat_route,
    require_answering_discovery,
    require_answering_ollama,
)

#: Short enough to keep the module fast, long enough to connect on any runner.
DEADLINE = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def silent_route():
    """The URL of an endpoint that accepts connections and never answers.

    Nothing ever reads from the listening socket: the kernel completes each
    TCP handshake, so a client connects and sends its request, then waits on
    a reply that never comes.
    """
    with socket.create_server(("127.0.0.1", 0), backlog=16) as listener:
        host, port = listener.getsockname()[:2]
        yield f"http://{host}:{port}"


def _budget(timeout: float = 60.0) -> LiveCallBudget:
    """A budget whose timeout started now and is too long to cut ``DEADLINE``."""
    return LiveCallBudget(started=time.monotonic(), timeout=timeout)


class _Routes:
    """The LLMService surface the probes use, over routes the test defines.

    ``routes`` maps a route name to the coroutine function that route's call
    runs. Discovery gathers every route; a turn starts with the first.
    """

    def __init__(self, routes):
        self._routes = routes
        self.discovered_with_cache = None

    def _select_discovery_routes(self):
        return [(name.split(":", 1)[0], {"name": name}) for name in self._routes]

    async def discover_all_models(self, use_cache=True):
        self.discovered_with_cache = use_cache
        return await asyncio.gather(*(call() for call in self._routes.values()))

    def resolve_provider_routing(self):
        return [{"name": name} for name in self._routes], None

    async def generate(self, *, system_prompt, user_prompt):
        return await next(iter(self._routes.values()))()


def _silent(url):
    async def call():
        return await ollama.AsyncClient(host=url).list()

    return call


async def _answers():
    return []


async def _refuses():
    raise RuntimeError("the route answered with an error")


async def _answers_after(seconds):
    await asyncio.sleep(seconds)
    return "answered"


async def test_deadline_fails_naming_the_test_call_and_route(silent_route):
    client = ollama.AsyncClient(host=silent_route)

    # The shape of a turn: process_input runs it in an isolated child task, and
    # discovery gathers its vendors.
    async def gathered():
        return await asyncio.gather(
            client.chat(model="qwen2.5:0.5b", messages=[{"role": "user", "content": "hi"}]),
            client.list(),
        )

    async def turn():
        return await asyncio.create_task(gathered(), name="isolated-turn")

    started = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as failed:
        await bounded_live_call(
            turn(),
            budget=_budget(),
            what="turn('hi')",
            waiting_on=f"route ollama:local at {silent_route}",
            deadline=DEADLINE,
        )

    assert time.monotonic() - started < DEADLINE + live_routes.CANCEL_GRACE_SECONDS
    message = str(failed.value)
    assert message.startswith(
        "tests/integration/test_live_routes.py::"
        "test_deadline_fails_naming_the_test_call_and_route: "
        f"turn('hi') did not return within 0.5s; it was waiting on route "
        f"ollama:local at {silent_route}."
    )
    assert "it unwound" in message
    # The await chain follows the child task and both gathered calls down to
    # the socket reads the silent route never satisfies.
    assert "task 'isolated-turn':" in message
    assert "AsyncClient.chat" in message
    assert "AsyncClient.list" in message
    assert message.count("SocketStream.receive") == 2


async def test_deadline_does_not_wait_on_a_call_that_ignores_cancellation(monkeypatch):
    monkeypatch.setattr(live_routes, "CANCEL_GRACE_SECONDS", 0.2)
    release = asyncio.Event()

    async def ignores_cancellation():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()

    try:
        with pytest.raises(pytest.fail.Exception) as failed:
            await bounded_live_call(
                ignores_cancellation(),
                budget=_budget(),
                what="ignores_cancellation()",
                waiting_on="nothing",
                deadline=DEADLINE,
            )
    finally:
        release.set()

    assert "was still running 0.2s later" in str(failed.value)


async def test_deadline_passes_the_call_outcome_through():
    budget = _budget()
    assert await bounded_live_call(_answers(), budget=budget, what="answers()", waiting_on="-") == []
    with pytest.raises(RuntimeError, match="answered with an error"):
        await bounded_live_call(_refuses(), budget=budget, what="refuses()", waiting_on="-")


async def test_a_slow_call_shortens_the_next_calls_deadline_to_beat_the_timeout(
    silent_route, monkeypatch
):
    """A slow but successful call, then a silent one, fail by name before the timeout.

    Each call's own deadline (75s) is far longer than the test's timeout, so
    only the shared budget can make the second deadline fire in time. The
    ``asyncio.timeout`` stands in for pytest-timeout: had the second call
    kept its own deadline, it would fire instead of the named failure.
    """
    monkeypatch.setattr(live_routes, "CANCEL_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(live_routes, "TEARDOWN_RESERVE_SECONDS", 0.3)
    budget = _budget(timeout=2.0)
    times_out_at = budget.started + budget.timeout

    first = await bounded_live_call(
        _answers_after(1.0), budget=budget, what="first()", waiting_on="a slow route"
    )
    assert first == "answered"

    try:
        async with asyncio.timeout(times_out_at - time.monotonic()):
            with pytest.raises(pytest.fail.Exception) as failed:
                await bounded_live_call(
                    ollama.AsyncClient(host=silent_route).list(),
                    budget=budget,
                    what="second()",
                    waiting_on=f"route silent:api at {silent_route}",
                )
    except TimeoutError:
        pytest.fail("the test's timeout fired before the second call's deadline")

    message = str(failed.value)
    assert re.search(
        r": second\(\) did not return within 0\.\d+s, all that was left of 75s once "
        r"1(\.\d+)?s of the test's 2s timeout had gone \(first\(\) 1(\.\d+)?s, "
        r"setup and the rest of the test [\d.]+s\), and its last 0\.5s is kept to "
        r"cancel a call and tear down; it was waiting on route silent:api at ",
        message,
    ), message
    assert "it unwound" in message
    assert [label for label, _ in budget.spent] == ["first()", "second()"]


async def test_a_spent_budget_fails_the_call_without_running_it(monkeypatch):
    monkeypatch.setattr(live_routes, "CANCEL_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(live_routes, "TEARDOWN_RESERVE_SECONDS", 0.3)
    budget = LiveCallBudget(started=time.monotonic() - 1.8, timeout=2.0)
    ran = False

    async def call():
        nonlocal ran
        ran = True

    with pytest.raises(pytest.fail.Exception) as failed:
        await bounded_live_call(call(), budget=budget, what="late()", waiting_on="a route")

    assert not ran
    assert re.search(
        r": no time left to run late\(\), which would wait on a route: "
        r"1\.8\d?s of the test's 2s timeout had gone \(setup and the rest of the "
        r"test 1\.8\d?s\), and its last 0\.5s is kept to cancel a call and tear down\.$",
        str(failed.value),
    ), str(failed.value)


_TIMEOUT_RACE = """
import asyncio
import time

import pytest

from tests.integration import live_routes
from tests.integration.live_routes import bounded_live_call

pytestmark = pytest.mark.timeout({timeout})


@pytest.fixture(autouse=True)
def short_reserves(monkeypatch):
    monkeypatch.setattr(live_routes, "CANCEL_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(live_routes, "TEARDOWN_RESERVE_SECONDS", 1.0)


@pytest.fixture
def slow_setup():
    time.sleep({setup})


async def answer_slowly():
    await asyncio.sleep({first})
    return "answered"


async def test_slow_then_silent(slow_setup, live_call_budget):
    first = await bounded_live_call(
        answer_slowly(), budget=live_call_budget, what="first()", waiting_on="a slow route"
    )
    assert first == "answered"
    await bounded_live_call(
        asyncio.Event().wait(),
        budget=live_call_budget,
        what="second()",
        waiting_on="a silent route",
    )
"""


def test_the_shared_budget_beats_pytest_timeout_counting_setup(tmp_path):
    """Under real pytest-timeout, the silent second call fails by name first.

    The child test runs a 2s setup and a 2s successful call under a 6s
    timeout, with 1.5s kept for cancelling and teardown, so the silent second
    call gets 0.5s and fails at about 4.5s. Its budget comes from the
    integration conftest's hook and fixture. If the budget ignored setup, the
    second call would get 2.5s, and the 6s timeout would fire before it.
    """
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nasyncio_mode = auto\nasyncio_default_fixture_loop_scope = function\n"
    )
    (tmp_path / "conftest.py").write_text(
        "from tests.integration.conftest import (  # noqa: F401 - the wiring under test\n"
        "    live_call_budget,\n"
        "    pytest_timeout_set_timer,\n"
        ")\n"
    )
    (tmp_path / "test_race.py").write_text(
        textwrap.dedent(_TIMEOUT_RACE).format(timeout=6, setup=2.0, first=2.0)
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    run = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "pytest_timeout", "-p", "pytest_asyncio.plugin",
            "-p", "no:cacheprovider", "-q", "--no-header", "test_race.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = run.stdout + run.stderr
    assert run.returncode == 1, output
    assert "1 failed" in output, output
    assert "from pytest-timeout" not in output, output
    assert re.search(
        r"second\(\) did not return within 0\.\d+s, all that was left of 75s once "
        r"4(\.\d+)?s of the test's 6s timeout had gone \(first\(\) 2(\.\d+)?s, "
        r"setup and the rest of the test 2(\.\d+)?s\)",
        output,
    ), output


async def test_ollama_probe_skips_a_daemon_that_never_answers(silent_route, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", silent_route)
    started = time.monotonic()

    with pytest.raises(pytest.skip.Exception, match="Ollama not available: ReadTimeout"):
        require_answering_ollama(deadline=DEADLINE)

    assert time.monotonic() - started < DEADLINE + 2


async def test_discovery_probe_skips_naming_where_discovery_is_parked(silent_route):
    service = _Routes({"ollama:local": _answers, "silent:api": _silent(silent_route)})

    with pytest.raises(pytest.skip.Exception) as skipped:
        await require_answering_discovery(service, budget=_budget(), deadline=DEADLINE)

    # Live discovery, not a catalog some earlier call left cached.
    assert service.discovered_with_cache is False
    reason = str(skipped.value)
    assert reason.startswith(
        "!model-list cannot answer here: model discovery (discovery routes "
        "[ollama:local, silent:api]) did not finish within 0.5s"
    )
    # Only the silent route's branch is still waiting, and it is named.
    assert "(parked in _silent.<locals>.call (test_live_routes.py:" in reason


async def _assert_runs(probe):
    """A probe that skips here fails the test, rather than reading as a skip."""
    try:
        await probe
    except pytest.skip.Exception as skipped:
        pytest.fail(f"the probe skipped routes that answered: {skipped}")


async def test_discovery_probe_runs_the_test_when_every_route_answers():
    service = _Routes({"ollama:local": _answers, "erroring:api": _refuses})

    await _assert_runs(require_answering_discovery(service, budget=_budget(), deadline=DEADLINE))


async def test_chat_probe_skips_naming_the_turns_routes(silent_route):
    service = _Routes({"silent:local": _silent(silent_route), "ollama:local": _answers})

    with pytest.raises(pytest.skip.Exception) as skipped:
        await require_answering_chat_route(service, budget=_budget(), deadline=DEADLINE)

    reason = str(skipped.value)
    assert reason.startswith(
        "No answer to a one-word prompt from chat routes "
        "[silent:local, ollama:local] model auto within 0.5s"
    )
    assert "(parked in _silent.<locals>.call (test_live_routes.py:" in reason


async def test_a_probe_spends_the_budget_its_test_calls_share():
    budget = _budget()

    await _assert_runs(
        require_answering_chat_route(
            _Routes({"first:api": lambda: _answers_after(0.2)}), budget=budget, deadline=DEADLINE
        )
    )

    [(label, seconds)] = budget.spent
    assert label == "the chat-route probe"
    assert seconds >= 0.2


@pytest.mark.parametrize("answer", [_answers, _refuses])
async def test_chat_probe_runs_the_test_when_the_route_answers(answer):
    await _assert_runs(
        require_answering_chat_route(
            _Routes({"first:api": answer}), budget=_budget(), deadline=DEADLINE
        )
    )
