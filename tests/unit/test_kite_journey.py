"""The Kite journey driver's own logic: freshness, scrubbing, ordering, verdict."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts/ci/kite_journey.py"
_SPEC = importlib.util.spec_from_file_location("kite_journey", _SCRIPT)
assert _SPEC and _SPEC.loader
kj = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves a class's module through sys.modules while decorating.
sys.modules[_SPEC.name] = kj
_SPEC.loader.exec_module(kj)


class _Recorder(kj.Journey):
    """A journey with no processes: records log lines, scrubs one secret."""

    def __init__(self, tmp_path: Path, secrets: tuple[str, ...] = ()):
        super().__init__(tmp_path, chat_timeout=1, telegram_window=">=0.1.9,<1")
        self._secrets = kj.secret_values({}, secrets)
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)


# -- freshness -----------------------------------------------------------------


@pytest.mark.parametrize("state", kj.STATE_PATHS)
def test_refuses_a_checkout_that_already_has_an_install(tmp_path, state):
    (tmp_path / state).mkdir()
    with pytest.raises(kj.JourneyError, match=state):
        kj.refuse_unfresh(tmp_path, {})


@pytest.mark.parametrize("name", kj.REFUSED_ENV)
def test_refuses_environment_that_points_at_a_live_install(tmp_path, name):
    with pytest.raises(kj.JourneyError, match=name):
        kj.refuse_unfresh(tmp_path, {name: "x"})


def test_fresh_checkout_with_clean_environment_is_accepted(tmp_path):
    kj.refuse_unfresh(tmp_path, {"KESTREL_NONINTERACTIVE": "1", "PATH": "/bin"})


# -- secrets -------------------------------------------------------------------


def test_credentials_from_dotenv_are_scrubbed_and_plain_config_is_not(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        'KESTREL_API_KEY="abc123secret"\n'
        "KESTREL_DATA_KEY=datakey999\n"
        "TELEGRAM_BOT_TOKEN=123:tok\n"
        "KESTREL_DID_WEB_DOMAIN=localhost\n"
        "# KESTREL_API_KEY=commented\n",
        encoding="utf-8",
    )
    secrets = kj.secret_values(kj.read_dotenv(env))
    text = "key abc123secret data datakey999 tok 123:tok domain localhost"
    assert kj.redact(text, secrets) == "key *** data *** tok *** domain localhost"


def test_longer_secret_is_scrubbed_before_a_shorter_one_it_contains():
    secrets = kj.secret_values({"A_KEY": "abc", "B_KEY": "abcdef"})
    assert kj.redact("abcdef abc", secrets) == "*** ***"


def test_a_credential_from_the_environment_is_scrubbed_before_setup(tmp_path, monkeypatch):
    """The CI model key arrives in the environment; .env does not exist yet."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-ci-secret")
    monkeypatch.setenv("KESTREL_NONINTERACTIVE", "1")
    journey = kj.Journey(tmp_path, 1, ">=0")
    assert journey.scrub("key=sk-or-v1-ci-secret mode=1") == "key=*** mode=1"


def test_step_detail_and_data_are_scrubbed(tmp_path):
    journey = _Recorder(tmp_path, secrets=("s3cret",))
    step = kj.Step("leaky", lambda j: kj.failed("saw s3cret", echoed="s3cret"))
    (result,) = kj.run_steps(journey, [step])
    assert "s3cret" not in result.detail
    assert "s3cret" not in json.dumps(result.data)


# -- roster ----------------------------------------------------------------------


def test_roster_reads_host_port_and_first_agent(tmp_path):
    path = tmp_path / "multi_agent.toml"
    path.write_text(
        "[host]\nport = 8888\n\n[agents.Kestrel]\nport = 8801\ndata_dir = 'agent_data/Kestrel'\n",
        encoding="utf-8",
    )
    assert kj.read_roster(path) == kj.Roster(8888, "Kestrel", 8801)


@pytest.mark.parametrize(
    "body, message",
    [
        ("[agents.K]\nport = 1\n", "host"),
        ("[host]\nport = 1\n", "no agents"),
        ("[host]\nport = 1\n[agents.K]\ndata_dir = 'x'\n", "no port for agent K"),
    ],
)
def test_roster_names_what_is_missing(tmp_path, body, message):
    path = tmp_path / "multi_agent.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(kj.JourneyError, match=message):
        kj.read_roster(path)


# -- host feature manifest -------------------------------------------------------


def test_manifest_entry_is_appended_once_and_keeps_existing_text(tmp_path):
    path = tmp_path / kj.HOST_FEATURE_MANIFEST
    path.write_text('# operator note\n[[feature]]\nname = "other"\npypi = ">=1"', encoding="utf-8")
    assert kj.add_manifest_entry(path, "pkg", ">=0.1,<1") is True
    assert kj.add_manifest_entry(path, "pkg", ">=0.1,<1") is False
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# operator note\n")
    entries = tomllib.loads(text)["feature"]
    assert entries == [
        {"name": "other", "pypi": ">=1"},
        {"name": "pkg", "pypi": ">=0.1,<1"},
    ]


def test_manifest_is_created_when_absent(tmp_path):
    path = tmp_path / kj.HOST_FEATURE_MANIFEST
    assert kj.add_manifest_entry(path, "pkg", ">=0.1") is True
    assert tomllib.loads(path.read_text(encoding="utf-8"))["feature"] == [
        {"name": "pkg", "pypi": ">=0.1"}
    ]


# -- ordering and verdict --------------------------------------------------------


def test_a_step_whose_requirement_failed_is_skipped_and_not_run(tmp_path):
    ran: list[str] = []

    def record(name, outcome):
        def run(_j):
            ran.append(name)
            return outcome
        return run

    steps = [
        kj.Step("a", record("a", kj.failed("broke"))),
        kj.Step("b", record("b", kj.passed("ok")), requires=("a",)),
        kj.Step("c", record("c", kj.passed("ok"))),
    ]
    results = kj.run_steps(_Recorder(tmp_path), steps)
    assert ran == ["a", "c"]
    assert [r.status for r in results] == [kj.FAIL, kj.SKIP, kj.PASS]
    assert results[1].detail == "requires a"


def test_journey_error_inside_a_step_is_a_failure_not_a_crash(tmp_path):
    def boom(_j):
        raise kj.JourneyError("multi_agent.toml has no [host].port")

    (result,) = kj.run_steps(_Recorder(tmp_path), [kj.Step("roster", boom)])
    assert result.status == kj.FAIL
    assert "no [host].port" in result.detail


def test_an_unexpected_exception_propagates_and_earlier_results_survive(tmp_path):
    def bug(_j):
        raise KeyError("driver bug")

    results: list = []
    steps = [kj.Step("first", lambda j: kj.passed("ok")), kj.Step("x", bug)]
    with pytest.raises(KeyError):
        kj.run_steps(_Recorder(tmp_path), steps, into=results)
    assert [r.name for r in results] == ["first"]


def test_a_later_batch_sees_the_requirements_an_earlier_batch_met(tmp_path):
    results: list = []
    journey = _Recorder(tmp_path)
    kj.run_steps(journey, [kj.Step("roster", lambda j: kj.passed("ok"))], into=results)
    kj.run_steps(journey, [kj.Step("stop", lambda j: kj.passed("ok"), requires=("roster",))],
                 into=results)
    assert [r.status for r in results] == [kj.PASS, kj.PASS]


def test_console_safe_survives_a_cp1252_console():
    text = "Run 'kestrel feature list' → ✅ done"
    safe = kj.console_safe(text, "cp1252")
    safe.encode("cp1252")
    assert "kestrel feature list" in safe and "\\u2192" in safe


def test_log_never_raises_on_an_unencodable_console(tmp_path, monkeypatch):
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(kj.sys, "stdout", stream)
    kj.Journey(tmp_path, 1, ">=0").log("arrow → here")
    stream.flush()
    assert b"arrow" in stream.buffer.getvalue()


def test_the_refusal_command_is_read_from_the_update_message():
    stderr = (
        "kestrel update: refusing to start: kestrel.exe is running, ...\n"
        "Run the update through Python instead, which holds no launcher open:\n"
        "    cmd:         \"C:\\v\\Scripts\\python.exe\" -P -m kestrel_sovereign.cli update --no-pull\n"
        "    PowerShell:  & 'C:\\v\\Scripts\\python.exe' -P -m kestrel_sovereign.cli update --no-pull\n"
        "If another `kestrel` command is also running, close it first.\n"
    )
    assert kj.refusal_command(stderr) == (
        '"C:\\v\\Scripts\\python.exe" -P -m kestrel_sovereign.cli update --no-pull')
    assert kj.refusal_command("error: something else\n") == ""


@pytest.mark.parametrize(
    "row, expected",
    [
        ("○ channel_telegram   kestrel-channel-telegram   Telegram transport", "channel_telegram"),
        ("● channel_telegram kestrel-channel-telegram", "channel_telegram"),
        ("○ kestrel-channel-telegram only", ""),
    ],
)
def test_the_feature_name_is_read_past_the_status_glyph(row, expected):
    assert kj.feature_name_in_row(row) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("PONG", True), ("pong.", True), ("The genesis audit is pending.", False), ("", False)],
)
def test_a_model_answer_is_recognised_and_a_refusal_is_not(text, expected):
    assert kj.mentions_pong(text) is expected


def _result(status, gate=True, detail="d"):
    return kj.StepResult("s", status, gate, 0.0, detail, {})


@pytest.mark.parametrize(
    "results, expected",
    [
        ([_result(kj.PASS)], 0),
        ([_result(kj.FAIL)], 1),
        ([_result(kj.FAIL, gate=False)], 0),
        ([_result(kj.SKIP, detail="requires roster")], 1),
        ([_result(kj.SKIP, detail="TELEGRAM_TEST_BOT_TOKEN not provided")], 0),
    ],
)
def test_exit_status(results, expected):
    assert kj.exit_status(results) == expected


def test_markdown_lists_every_step_and_expands_failures():
    results = [
        kj.StepResult("setup", kj.PASS, True, 1.0, "ok", {}),
        kj.StepResult("update", kj.FAIL, True, 2.0, "exit 1 | pipe\nos error 32", {}),
    ]
    text = kj.render_markdown(results, {"platform": "Windows"})
    assert "| setup | ✅ pass |" in text
    assert "exit 1 \\| pipe" in text
    assert "### update" in text and "os error 32" in text


def test_every_requirement_names_an_earlier_step():
    seen: set[str] = set()
    for step in kj.STEPS:
        assert set(step.requires) <= seen, step.name
        seen.add(step.name)
