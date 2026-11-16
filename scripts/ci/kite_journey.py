"""Kite journey: walk a fresh Kestrel install the way a new user does.

The clean-install gate proves an agent boots and answers ``/health``. By
design it skips the genesis audit, never talks to a model, and runs the CLI
as ``python -m kestrel_sovereign.cli`` so the command is identical on every
OS. A user does none of that. They run the ``kestrel`` console script (on
Windows, ``kestrel.exe``), let quickstart find their local model, chat with
the agent, add a channel, and later run ``kestrel update`` while the agent is
still up. Each of those has broken on Windows with the clean-install gate
green.

This driver walks that journey in the checkout it runs from and records one
result per step. It was written to find Windows defects and runs unchanged on
Linux and macOS, so a finding can be checked on another OS.

Run it from a FRESH checkout. It writes ``.env``, ``kestrel.toml``,
``multi_agent.toml`` and ``agent_data/`` there, as quickstart does for a
user, and refuses to start if any of them already exist::

    uv run python scripts/ci/kite_journey.py \
        --report journey.json --artifacts journey-artifacts

Stdlib only. Exit status 0 when every gating step passed, 1 otherwise, 2 when
the checkout is not fresh.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

# Files quickstart creates. Their presence means this is somebody's install,
# and walking the journey would overwrite their agent.
STATE_PATHS = (".env", "kestrel.toml", "multi_agent.toml", "agent_data")

# Environment that would point the journey at a live install or database.
REFUSED_ENV = (
    "KESTREL_HOME",
    "KESTREL_DATA_KEY",
    "KESTREL_MULTI_AGENT_CONFIG",
    "DATABASE_URL",
)

# .env keys whose values are credentials. Their values are scrubbed from the
# report and every copied log, because the CI run that publishes them is on a
# public repository.
_SECRET_KEY_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE)", re.IGNORECASE)
# Credentials handed to the journey in its environment. The runner image's
# own variables (PGPASSWORD=root, ...) are not ours, and scrubbing short
# common values corrupts every log line that contains them.
CREDENTIAL_ENV = ("OPENROUTER_API_KEY", "TELEGRAM_TEST_BOT_TOKEN")
MIN_SECRET_LENGTH = 8

TELEGRAM_PACKAGE = "kestrel-channel-telegram"
TELEGRAM_FEATURE_CLASS = "TelegramFeature"
TELEGRAM_SERVICE_IMAGE = "kestrel-telegram-service"
TELEGRAM_TOKEN_ENV = "TELEGRAM_TEST_BOT_TOKEN"
HOST_FEATURE_MANIFEST = ".kestrel-host-features.toml"

TAIL_LINES = 60
TEARDOWN_RESERVE_SECONDS = 180


@dataclasses.dataclass
class Outcome:
    status: str
    detail: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Step:
    name: str
    run: Callable[[Journey], Outcome]
    requires: tuple[str, ...] = ()
    gate: bool = True


@dataclasses.dataclass
class StepResult:
    name: str
    status: str
    gate: bool
    seconds: float
    detail: str
    data: dict[str, Any]


class JourneyError(RuntimeError):
    """The checkout or environment cannot host a journey."""


def passed(detail: str, **data: Any) -> Outcome:
    return Outcome(PASS, detail, data)


def failed(detail: str, **data: Any) -> Outcome:
    return Outcome(FAIL, detail, data)


def skipped(detail: str, **data: Any) -> Outcome:
    return Outcome(SKIP, detail, data)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def refuse_unfresh(project_dir: Path, environ: dict[str, str]) -> None:
    """Raise unless ``project_dir`` is a fresh checkout and env is clean."""
    present = [p for p in STATE_PATHS if (project_dir / p).exists()]
    if present:
        raise JourneyError(
            "the journey needs a fresh checkout; found existing "
            + ", ".join(present)
        )
    leaked = [name for name in REFUSED_ENV if environ.get(name)]
    if leaked:
        raise JourneyError(
            "unset " + ", ".join(leaked) + " before running the journey; "
            "it would point the journey at an existing install"
        )


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal dotenv reader: KEY=value lines, one layer of quotes stripped."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def secret_values(dotenv: dict[str, str], extra: tuple[str, ...] = ()) -> list[str]:
    """Credential values to scrub, longest first so overlaps scrub fully."""
    found = {v for k, v in dotenv.items() if _SECRET_KEY_RE.search(k) and v}
    found.update(v for v in extra if v)
    found = {v for v in found if len(v) >= MIN_SECRET_LENGTH}
    return sorted(found, key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> str:
    for value in secrets:
        text = text.replace(value, "***")
    return text


def tail(text: str, lines: int = TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def console_safe(text: str, encoding: str | None) -> str:
    """``text`` with anything ``encoding`` cannot represent escaped.

    A Windows console or pipe defaults to cp1252, which has no ``→`` or emoji;
    printing Kestrel's own output there raises UnicodeEncodeError.
    """
    codec = encoding or "utf-8"
    return text.encode(codec, errors="backslashreplace").decode(codec)


def feature_name_in_row(row: str) -> str:
    """The catalog short name in a ``kestrel feature list`` row.

    Rows lead with a status glyph (``○``/``●``); the short name is the first
    lowercase identifier mentioning telegram (the package name has hyphens).
    """
    for token in row.split():
        if re.fullmatch(r"[a-z][a-z0-9_]*", token) and "telegram" in token:
            return token
    return ""


def json_object_in(text: str) -> dict[str, Any]:
    """The first JSON object in ``text`` (a ``!command`` prints its result as JSON), or {}.

    A substring test cannot read a status: the payload always carries the key
    ``"connected"``, true or false.
    """
    start = text.find("{")
    while start != -1:
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return value if isinstance(value, dict) else {}
    return {}


def mentions_pong(text: str) -> bool:
    """The reply a model gives when asked to answer with exactly PONG."""
    return "pong" in text.lower()


@dataclasses.dataclass(frozen=True)
class Roster:
    host_port: int
    agent_name: str
    agent_port: int


def read_roster(path: Path) -> Roster:
    """The host port and the first local agent quickstart registered."""
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    host_port = (config.get("host") or {}).get("port")
    agents = config.get("agents") or {}
    if not isinstance(host_port, int):
        raise JourneyError(f"{path.name} has no [host].port")
    if not agents:
        raise JourneyError(f"{path.name} registers no agents")
    name, entry = next(iter(agents.items()))
    port = (entry or {}).get("port")
    if not isinstance(port, int):
        raise JourneyError(f"{path.name} has no port for agent {name}")
    return Roster(host_port=host_port, agent_name=name, agent_port=port)


def add_manifest_entry(path: Path, package: str, pypi_window: str) -> bool:
    """Append a ``[[feature]]`` entry for ``package``; False if already listed.

    Appends text rather than rewriting the file, so an operator's comments and
    ordering survive, as they would if a person had added the entry by hand.
    """
    if path.exists():
        with open(path, "rb") as handle:
            existing = tomllib.load(handle).get("feature") or []
        if any(entry.get("name") == package for entry in existing):
            return False
        prefix = path.read_text(encoding="utf-8")
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
    else:
        prefix = ""
    entry = f'\n[[feature]]\nname = "{package}"\npypi = "{pypi_window}"\n'
    path.write_text(prefix + entry, encoding="utf-8")
    return True


def scrub_data(value: Any, scrub: Callable[[str], str]) -> Any:
    """``value`` with every string scrubbed, structure kept."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {key: scrub_data(item, scrub) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_data(item, scrub) for item in value]
    return value


def run_steps(
    journey: Journey, steps: list[Step], into: list[StepResult] | None = None
) -> list[StepResult]:
    """Run ``steps`` in order; a step whose requirement did not pass is skipped.

    Each result is appended to ``into`` as soon as it exists, so the steps that
    ran are still there for the report when a later step raises.
    """
    collected = [] if into is None else into
    results: dict[str, StepResult] = {r.name: r for r in collected}
    for step in steps:
        unmet = [r for r in step.requires if results.get(r) is None or results[r].status != PASS]
        started = time.monotonic()
        if unmet:
            outcome = skipped("requires " + ", ".join(unmet))
        else:
            try:
                outcome = step.run(journey)
            except JourneyError as exc:
                outcome = failed(str(exc))
            except subprocess.TimeoutExpired as exc:
                outcome = failed(f"timed out after {exc.timeout:.0f}s: {exc.cmd!r}"[:600])
        result = StepResult(
            name=step.name,
            status=outcome.status,
            gate=step.gate,
            seconds=round(time.monotonic() - started, 1),
            detail=journey.scrub(outcome.detail),
            data=scrub_data(outcome.data, journey.scrub),
        )
        results[step.name] = result
        collected.append(result)
        journey.log(f"[{outcome.status.upper()}] {step.name} ({result.seconds}s)")
        if outcome.status != PASS:
            journey.log(result.detail)
    return collected


def exit_status(results: list[StepResult]) -> int:
    """1 when a gating step failed or was skipped for an unmet requirement."""
    for result in results:
        if not result.gate:
            continue
        if result.status == FAIL:
            return 1
        if result.status == SKIP and result.detail.startswith("requires "):
            return 1
    return 0


def render_markdown(results: list[StepResult], header: dict[str, str]) -> str:
    icon = {PASS: "✅", FAIL: "❌", SKIP: "⏭️"}
    lines = ["## Kite journey", ""]
    lines += [f"- **{key}:** {value}" for key, value in header.items()]
    lines += ["", "| step | result | seconds | detail |", "|---|---|---|---|"]
    for result in results:
        first = result.detail.splitlines()[0] if result.detail else ""
        first = first.replace("|", "\\|")[:160]
        gate = "" if result.gate else " (info)"
        lines.append(
            f"| {result.name}{gate} | {icon[result.status]} {result.status} "
            f"| {result.seconds} | {first} |"
        )
    failures = [r for r in results if r.status == FAIL]
    for result in failures:
        lines += ["", f"### {result.name}", "", "```", result.detail, "```"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The journey: process, HTTP and filesystem effects
# ---------------------------------------------------------------------------


class Journey:
    def __init__(self, project_dir: Path, chat_timeout: int, telegram_window: str,
                 deadline_seconds: float = 75 * 60):
        self.project_dir = project_dir
        self.chat_timeout = chat_timeout
        self.telegram_window = telegram_window
        self.kestrel = Path(sys.executable).parent / (
            "kestrel.exe" if os.name == "nt" else "kestrel"
        )
        self.roster: Roster | None = None
        self.api_key = ""
        self.host_process: subprocess.Popen[str] | None = None
        self.telegram_name = ""
        self.telegram_installed = False
        # Credentials handed to the journey in its environment (the CI model
        # key, the Telegram token) before quickstart copies any into .env.
        # The job is killed at its timeout before `finally` can write the
        # report, so every wait is held inside this one deadline instead.
        self.deadline = time.monotonic() + deadline_seconds
        self._secrets: list[str] = secret_values(
            {}, tuple(os.environ.get(name, "") for name in CREDENTIAL_ENV))

    # -- output --------------------------------------------------------------

    def budget(self, timeout: float) -> float:
        """``timeout``, cut to what the deadline leaves after a teardown reserve."""
        remaining = self.deadline - time.monotonic() - TEARDOWN_RESERVE_SECONDS
        if remaining <= 0:
            raise JourneyError("journey deadline reached")
        return min(timeout, remaining)

    def log(self, message: str) -> None:
        print(console_safe(message, sys.stdout.encoding), flush=True)

    def scrub(self, text: str) -> str:
        return redact(text, self._secrets)

    def learn_secrets(self) -> None:
        dotenv = read_dotenv(self.project_dir / ".env")
        self._secrets = secret_values(dotenv, tuple(self._secrets))

    # -- processes -----------------------------------------------------------

    def kestrel_cli(self, *args: str, timeout: int) -> subprocess.CompletedProcess[str]:
        """Run the ``kestrel`` console script exactly as a user types it.

        Not ``python -m kestrel_sovereign.cli``: on Windows the console script
        is a separate ``kestrel.exe`` launcher, and how that launcher behaves
        (and what it holds open) is part of what the journey tests.
        """
        return subprocess.run(
            [str(self.kestrel), *args],
            cwd=self.project_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="backslashreplace",
            timeout=self.budget(timeout),
            check=False,
        )

    def cli_outcome(self, label: str, proc: subprocess.CompletedProcess[str]) -> Outcome:
        # Tailed separately: the CLI's verdict is on stdout and its logging on
        # stderr, and a combined tail of a chatty run holds only the logging.
        output = (
            f"--- stdout ---\n{tail(proc.stdout or '', 40)}\n"
            f"--- stderr ---\n{tail(proc.stderr or '', 20)}"
        )
        if proc.returncode != 0:
            return failed(f"{label} exited {proc.returncode}\n{output}", rc=proc.returncode)
        return passed(f"{label} exited 0", output_tail=output)

    # -- HTTP ----------------------------------------------------------------

    def base_url(self) -> str:
        assert self.roster is not None
        return f"http://127.0.0.1:{self.roster.host_port}"

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30
    ) -> tuple[int | None, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(
            self.base_url() + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, _decode(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def wait_healthy(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, _ = self.request("GET", "/health", timeout=5)
            if status == 200:
                return True
            time.sleep(2)
        return False

    def invoke(self, text: str) -> tuple[int | None, Any, float]:
        assert self.roster is not None
        started = time.monotonic()
        status, body = self.request(
            "POST",
            f"/api/agents/{self.roster.agent_name}/api/agent/invoke",
            {"input": text},
            timeout=self.budget(self.chat_timeout),
        )
        return status, body, round(time.monotonic() - started, 1)

    # -- process table -------------------------------------------------------

    def running_images(self) -> list[str]:
        """Command lines (POSIX) or image names (Windows) of running processes."""
        if os.name == "nt":
            proc = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, encoding="utf-8", errors="replace", check=False,
            )
            return [line.split('","')[0].strip('"') for line in proc.stdout.splitlines() if line]
        proc = subprocess.run(
            ["ps", "-A", "-o", "args="],
            capture_output=True, encoding="utf-8", errors="replace", check=False,
        )
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def port_open(self, port: int) -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", port)) == 0


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _response_text(body: Any) -> str:
    if isinstance(body, dict):
        value = body.get("response")
        return value if isinstance(value, str) else ""
    return ""


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_entry_point(j: Journey) -> Outcome:
    if not j.kestrel.exists():
        return failed(f"no kestrel console script next to {sys.executable}")
    return passed(f"using {j.kestrel.name}", path=str(j.kestrel))


def step_setup(j: Journey) -> Outcome:
    proc = j.kestrel_cli("setup", "--quickstart", timeout=900)
    j.learn_secrets()
    outcome = j.cli_outcome("kestrel setup --quickstart", proc)
    toml_path = j.project_dir / "kestrel.toml"
    if outcome.status == PASS and toml_path.exists():
        with open(toml_path, "rb") as handle:
            llm = tomllib.load(handle).get("llm") or {}
        outcome.data["route_priority"] = llm.get("route_priority")
    return outcome


def step_doctor(j: Journey) -> Outcome:
    return j.cli_outcome("kestrel doctor", j.kestrel_cli("doctor", timeout=300))


def step_roster(j: Journey) -> Outcome:
    j.roster = read_roster(j.project_dir / "multi_agent.toml")
    j.api_key = read_dotenv(j.project_dir / ".env").get("KESTREL_API_KEY", "")
    if not j.api_key:
        return failed(".env has no KESTREL_API_KEY")
    return passed(
        f"agent {j.roster.agent_name} on :{j.roster.agent_port}, host :{j.roster.host_port}",
        **dataclasses.asdict(j.roster),
    )


def step_host_start(j: Journey) -> Outcome:
    """``kestrel start`` with no name: the multi-agent host, as a user runs it.

    The launcher writes to a file, not a pipe: a host that detaches inherits
    the launcher's handles, and reading a pipe it holds would wait for the
    host to exit rather than for the launcher.
    """
    launcher_log = j.project_dir / "logs" / "kite-journey-start.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    with open(launcher_log, "w", encoding="utf-8") as sink:
        j.host_process = subprocess.Popen(
            [str(j.kestrel), "start"],
            cwd=j.project_dir,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
    try:
        j.host_process.wait(timeout=300)
        launcher = f"launcher exited {j.host_process.returncode}"
        foreground = False
    except subprocess.TimeoutExpired:
        launcher = "launcher still running after 300s (foreground host)"
        foreground = True
    healthy = j.wait_healthy(180)
    out = launcher_log.read_text(encoding="utf-8", errors="backslashreplace")
    data = {"launcher": launcher, "foreground": foreground}
    if not healthy:
        return failed(f"host not healthy on :{j.roster.host_port}; {launcher}\n{tail(out)}", **data)
    if not foreground and j.host_process.returncode != 0:
        return failed(f"host healthy but {launcher}\n{tail(out)}", **data)
    return passed(f"host healthy; {launcher}", output_tail=tail(out), **data)


def step_first_contact(j: Journey) -> Outcome:
    """A new agent's first turn is its scripted first-contact greeting.

    It answers whatever the user says with an introduction and does not call
    a model, so it is checked for a reply only. The model check follows.
    """
    status, body, seconds = j.invoke("Hi")
    text = _response_text(body)
    if status != 200 or not text.strip():
        return failed(f"first contact returned HTTP {status}: {str(body)[:1500]}")
    return passed(f"greeted in {seconds}s", response=text[:300])


def step_chat(j: Journey) -> Outcome:
    status, body, seconds = j.invoke("Reply with exactly one word: PONG")
    text = _response_text(body)
    data = {
        "http": status,
        "seconds": seconds,
        "provider": body.get("provider") if isinstance(body, dict) else None,
        "model": body.get("model") if isinstance(body, dict) else None,
        "response": text[:300],
    }
    # HTTP 200 with text is not an answer: a turn the agent refused (genesis
    # pending, no usable route) also comes back 200 with an explanation.
    if status != 200 or not mentions_pong(text) or not data["provider"]:
        return failed(
            f"no model answer: HTTP {status}, provider={data['provider']}, "
            f"response={text[:1500]!r}" + ("" if text else f" body={str(body)[:1500]}"),
            **data,
        )
    return passed(f"{data['provider']}/{data['model']} answered in {seconds}s", **data)


def step_bang_command(j: Journey) -> Outcome:
    status, body, seconds = j.invoke("!help")
    text = _response_text(body)
    if status != 200 or not text.strip():
        return failed(f"!help returned HTTP {status}: {str(body)[:2000]}", http=status)
    return passed(f"!help answered in {seconds}s", response=text[:300])


def step_cli_ask(j: Journey) -> Outcome:
    assert j.roster is not None
    proc = j.kestrel_cli(
        "ask", j.roster.agent_name, "Reply with exactly one word: PONG",
        timeout=j.chat_timeout + 60,
    )
    outcome = j.cli_outcome("kestrel ask", proc)
    if outcome.status == PASS and not mentions_pong(proc.stdout or ""):
        return failed(f"kestrel ask exited 0 without a model answer:\n{tail(proc.stdout or '')}")
    return outcome


def step_ollama_reachable(j: Journey) -> Outcome:
    """Can this machine reach Ollama at the URL quickstart configured?

    Times the same request under both loopback spellings and two timeouts.
    Kestrel's probes use 1.5 s. Informational: it explains a chat failure.
    """
    with open(j.project_dir / "kestrel.toml", "rb") as handle:
        config = tomllib.load(handle)
    route = (
        (((config.get("llm") or {}).get("vendors") or {}).get("ollama") or {})
        .get("routes", {}).get("local", {})
    )
    configured = str(route.get("host") or "")
    if not route:
        return skipped("quickstart configured no Ollama route")
    timings: dict[str, str] = {}
    for base in dict.fromkeys([configured, "http://localhost:11434", "http://127.0.0.1:11434"]):
        if not base:
            continue
        for timeout in (1.5, 10.0):
            started = time.monotonic()
            try:
                with urllib.request.urlopen(base.rstrip("/") + "/api/tags", timeout=timeout) as resp:
                    result = f"HTTP {resp.status}"
            except (urllib.error.URLError, OSError) as exc:
                result = f"{type(exc).__name__}: {exc}"
            timings[f"{base} @{timeout}s"] = f"{result} in {time.monotonic() - started:.2f}s"
    detail = "; ".join(f"{k} -> {v}" for k, v in timings.items())
    ok = timings.get(f"{configured} @1.5s", "").startswith("HTTP 200")
    return (passed if ok else failed)(
        f"configured host {configured or '(none)'}: {detail}", configured=configured, **timings
    )


def step_telegram_registry(j: Journey) -> Outcome:
    """A user asks for "telegram", and on "Unknown feature" checks the list."""
    proc = j.kestrel_cli("feature", "info", "telegram", timeout=120)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "Unknown feature" not in output:
        j.telegram_name = "telegram"
        return passed("kestrel feature info telegram resolves", output_tail=tail(output))
    listing = j.kestrel_cli("feature", "list", timeout=120)
    rows = [line for line in (listing.stdout or "").splitlines() if "telegram" in line.lower()]
    if not rows:
        return failed(
            "kestrel feature info telegram: Unknown feature, and `kestrel feature list` "
            f"has no row mentioning telegram ({len((listing.stdout or '').splitlines())} lines)"
        )
    j.telegram_name = feature_name_in_row(rows[0])
    if not j.telegram_name:
        return failed(f"`kestrel feature list` row has no feature name: {rows[0]!r}")
    return passed(f"listed as {j.telegram_name!r}", rows=rows[:5])


def installed_version(package: str) -> str | None:
    """``package``'s version in this venv, read by a fresh interpreter."""
    probe = subprocess.run(
        [sys.executable, "-c",
         f"import importlib.metadata as m; print(m.version({package!r}))"],
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def step_telegram_install(j: Journey) -> Outcome:
    """``kestrel feature install`` when the CLI knows the feature (the user path).

    Otherwise the host feature manifest plus ``kestrel feature sync``, the
    path an operator who read the README takes. Either way the step records
    which path ran, so a pass here never hides a missing user path.
    """
    if j.telegram_name:
        path = f"kestrel feature install {j.telegram_name}"
        proc = j.kestrel_cli("feature", "install", j.telegram_name, timeout=900)
    else:
        path = f"{HOST_FEATURE_MANIFEST} + kestrel feature sync"
        add_manifest_entry(j.project_dir / HOST_FEATURE_MANIFEST, TELEGRAM_PACKAGE,
                           j.telegram_window)
        proc = j.kestrel_cli("feature", "sync", timeout=900)
    outcome = j.cli_outcome(path, proc)
    if outcome.status != PASS:
        return outcome
    version = installed_version(TELEGRAM_PACKAGE)
    if version is None:
        return failed(f"{path} exited 0 but {TELEGRAM_PACKAGE} is not installed\n"
                      f"{outcome.data.get('output_tail', '')}", path=path)
    j.telegram_installed = True
    return passed(f"{TELEGRAM_PACKAGE} {version} installed via {path}", path=path,
                  version=version)


def step_restart(j: Journey) -> Outcome:
    proc = j.kestrel_cli("restart", timeout=600)
    outcome = j.cli_outcome("kestrel restart", proc)
    if outcome.status != PASS:
        return outcome
    if not j.wait_healthy(180):
        return failed(f"host not healthy after restart\n{outcome.data.get('output_tail', '')}")
    return outcome


def step_telegram_loaded(j: Journey) -> Outcome:
    assert j.roster is not None
    status, body = j.request("GET", f"/api/agents/{j.roster.agent_name}/api/features/installed")
    listing = json.dumps(body)
    if status != 200:
        return failed(f"installed-features listing returned HTTP {status}: {listing[:1500]}")
    if TELEGRAM_FEATURE_CLASS not in listing:
        return failed(f"{TELEGRAM_FEATURE_CLASS} not loaded: {listing[:1500]}")
    deadline = time.monotonic() + 120
    service_running = False
    while time.monotonic() < deadline and not service_running:
        service_running = any(TELEGRAM_SERVICE_IMAGE in name for name in j.running_images())
        if not service_running:
            time.sleep(3)
    cmd_status, cmd_body, _ = j.invoke("!telegram_status")
    data = {
        "service_process": service_running,
        "telegram_status_http": cmd_status,
        "telegram_status": _response_text(cmd_body)[:600] or str(cmd_body)[:600],
    }
    if not service_running:
        return failed(
            f"{TELEGRAM_FEATURE_CLASS} is listed but no {TELEGRAM_SERVICE_IMAGE} process "
            "started (isolated venv service)", **data,
        )
    return passed("feature loaded and its isolated service is running", **data)


def step_telegram_token(j: Journey) -> Outcome:
    token = os.environ.get(TELEGRAM_TOKEN_ENV, "")
    if not token:
        return skipped(f"{TELEGRAM_TOKEN_ENV} not provided")
    assert j.roster is not None
    status, body = j.request(
        "PATCH",
        f"/api/agents/{j.roster.agent_name}/api/features/{TELEGRAM_FEATURE_CLASS}/config",
        {"config": {"enabled": True, "token": token}},
    )
    if status != 200:
        return failed(f"config PATCH returned HTTP {status}: {str(body)[:1500]}")
    deadline = time.monotonic() + 120
    last = ""
    while time.monotonic() < deadline:
        cmd_status, cmd_body, _ = j.invoke("!telegram_status")
        last = _response_text(cmd_body) or str(cmd_body)
        status_doc = json_object_in(last)
        if cmd_status == 200 and status_doc.get("connected") is True:
            return passed("bot connected", telegram_status=last[:600])
        time.sleep(5)
    return failed(f"bot never reported connected: {last[:1500]}")


def force_stop_tree(pid_file: Path) -> None:
    """Harness-only: end the process tree a PID record names. Never used as a pass."""
    try:
        pid = int(json.loads(pid_file.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                       capture_output=True, check=False)
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            return
    time.sleep(3)


def refusal_command(stderr: str) -> str:
    """The command a Windows update refusal says to run instead, or ""."""
    lines = stderr.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Run the update through Python instead"):
            for following in lines[index + 1:]:
                if following.strip().startswith("cmd:"):
                    return following.strip()[len("cmd:"):].strip()
    return ""


def run_update(j: Journey, *args: str) -> tuple[Outcome, str]:
    """``kestrel update`` as a Windows user meets it.

    First the ``kestrel`` command. On Windows it refuses before changing
    anything, because it cannot replace the ``kestrel.exe`` it runs in (#2923),
    and prints the command that can; the journey then runs exactly that.
    """
    label = "kestrel update " + " ".join(args)
    outcome, output = _run_update_once(j, label, *args)
    if outcome.status == PASS and j.telegram_installed and installed_version(TELEGRAM_PACKAGE) is None:
        # #4127: uv sync prunes a feature `kestrel feature install` never recorded.
        j.telegram_installed = False
        return failed(f"{label} removed {TELEGRAM_PACKAGE}, which was installed before it "
                      f"ran\n{tail(output)}", **outcome.data), output
    return outcome, output


def _run_update_once(j: Journey, label: str, *args: str) -> tuple[Outcome, str]:
    proc = j.kestrel_cli("update", *args, timeout=1800)
    output = (proc.stdout or "") + (proc.stderr or "")
    command = refusal_command(proc.stderr or "")
    data: dict[str, Any] = {"file_lock": "os error 32" in output and not command,
                            "refused": bool(command)}
    if proc.returncode == 0 or not command:
        outcome = j.cli_outcome(label, proc)
        outcome.data.update(data)
        return outcome, output
    if proc.returncode != 2:
        return failed(f"{label} printed a command but exited {proc.returncode}\n{tail(output)}",
                      **data), output
    data["ran_instead"] = command
    rerun = subprocess.run(command if os.name == "nt" else shlex.split(command),
                           cwd=j.project_dir, stdin=subprocess.DEVNULL, capture_output=True,
                           encoding="utf-8", errors="backslashreplace",
                           timeout=j.budget(1800), check=False)
    outcome = j.cli_outcome(f"the refusal's command ({label})", rerun)
    data["file_lock"] = "os error 32" in (rerun.stdout or "") + (rerun.stderr or "")
    if outcome.status != PASS and j.roster is not None:
        # What still holds the host port when the update's restart gives up.
        data["ownership_after"] = listener_ownership(j.project_dir / "logs" / ".host.pid",
                                                     j.roster.host_port)
        data["processes_after"] = [name for name in j.running_images()
                                   if "python" in name.lower() or "kestrel" in name.lower()]
    outcome.data.update(data)
    return outcome, (rerun.stdout or "") + (rerun.stderr or "")


def step_update_while_stopped(j: Journey) -> Outcome:
    """``kestrel update`` with nothing running, then start the host again.

    #2923 blames the running agent for the ``os error 32`` on ``kestrel.exe``.
    The update's own ``kestrel.exe`` launcher is also alive while ``uv sync``
    replaces it. Stopping everything first tells the two apart: if this step
    also hits the lock, the update is locking itself.
    """
    stopped = step_terminate(j)
    forced = False
    if stopped.status != PASS:
        # `kestrel terminate` is a separate finding (terminate-start-cycle reports it).
        # The question here needs nothing running, so the harness ends the
        # recorded process tree itself and says so in the result.
        force_stop_tree(j.project_dir / "logs" / ".host.pid")
        forced = True
        if j.roster and j.port_open(j.roster.host_port):
            return failed(f"could not stop before updating, even by force: {stopped.detail}")
    os.utime(j.project_dir / "pyproject.toml", None)
    outcome, _output = run_update(j, "--no-pull", "--no-restart")
    outcome.data["harness_forced_stop"] = forced
    restarted = step_host_start(j)
    if outcome.status == PASS and restarted.status != PASS:
        return failed(f"updated, then the host would not start: {restarted.detail}")
    if forced and outcome.status == PASS:
        return failed(f"kestrel terminate failed, so the harness forced the stop; the update "
                      f"then passed. Terminate: {stopped.detail}", **outcome.data)
    return outcome


def step_update_while_running(j: Journey) -> Outcome:
    """``kestrel update`` with the agent up, across a change to pyproject.toml.

    Every release bumps the version in pyproject.toml, which makes ``uv sync``
    rebuild and reinstall the project, replacing the ``kestrel`` console
    script. Touching the file reproduces that without a second revision.
    """
    pyproject = j.project_dir / "pyproject.toml"
    os.utime(pyproject, None)
    outcome, output = run_update(j, "--no-pull")
    if outcome.status != PASS:
        return outcome
    if not j.wait_healthy(180):
        return failed(f"host not healthy after update\n{tail(output)}")
    telegram = installed_version(TELEGRAM_PACKAGE)
    status, body, seconds = j.invoke("!help")
    if status != 200 or not _response_text(body).strip():
        return failed(f"agent did not respond after update: HTTP {status} {str(body)[:1500]}")
    return passed(f"updated and restarted; agent responds ({seconds}s); telegram {telegram}",
                  output_tail=tail(output))


def listener_ownership(pid_file: Path, port: int) -> dict[str, Any]:
    """Who the PID record names versus who actually listens on ``port``.

    ``kestrel stop`` refuses unless the recorded PID owns the listener. On
    Windows a venv's ``python.exe`` is a launcher that runs the interpreter
    as a child process, so the two can differ. Uses psutil, a core
    dependency, which this venv has; the driver itself stays stdlib.
    """
    import psutil

    try:
        recorded = pid_file.read_text(encoding="utf-8")[:400]
    except OSError as exc:
        recorded = f"unreadable: {exc}"
    owners = []
    try:
        connections = psutil.net_connections(kind="inet")
    except psutil.Error as exc:
        # macOS refuses the system-wide table to a non-root user.
        return {"pid_record": recorded, "error": f"{type(exc).__name__}: {exc}"}
    for conn in connections:
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
            owners.append(conn.pid)
    detail = {}
    for pid in sorted({p for p in owners if p}):
        try:
            proc = psutil.Process(pid)
            detail[str(pid)] = {"exe": proc.exe(), "ppid": proc.ppid()}
        except psutil.Error as exc:
            detail[str(pid)] = {"error": type(exc).__name__}
    recorded_exe = ""
    try:
        recorded_exe = psutil.Process(int(json.loads(recorded)["pid"])).exe()
    except (ValueError, KeyError, TypeError, psutil.Error) as exc:
        recorded_exe = f"unavailable: {type(exc).__name__}"
    pyvenv = Path(sys.prefix) / "pyvenv.cfg"
    try:
        home = next((line for line in pyvenv.read_text(encoding="utf-8").splitlines()
                     if line.strip().lower().startswith("home")), "")
    except OSError as exc:
        home = f"unreadable: {exc}"
    return {"pid_record": recorded, "recorded_exe": recorded_exe, "pyvenv_home": home,
            "listener_pids": owners, "listeners": detail}


def step_terminate_start_cycle(j: Journey) -> Outcome:
    """Terminate and start again on an untouched install, before ``update`` runs."""
    stopped = step_terminate(j)
    if stopped.status != PASS:
        return stopped
    started = step_host_start(j)
    if started.status != PASS:
        return failed(f"terminated cleanly, then could not start again: {started.detail}",
                      **started.data)
    return passed("terminated cleanly and started again", **stopped.data)


def step_stop_in_flight(j: Journey) -> Outcome:
    """``kestrel stop --all`` halts in-flight work; the host stays up.

    ``stop`` is Kestrel's cooperative halt, not a shutdown (that is
    ``terminate``). It must prove it is talking to this checkout's host before
    it sends the operator credential, and on Windows it could not.
    """
    assert j.roster is not None
    proc = j.kestrel_cli("stop", "--all", timeout=300)
    outcome = j.cli_outcome("kestrel stop --all", proc)
    if outcome.status != PASS:
        outcome.data["ownership"] = listener_ownership(
            j.project_dir / "logs" / ".host.pid", j.roster.host_port)
        return outcome
    if not j.wait_healthy(30):
        return failed("kestrel stop --all exited 0, but the host is no longer healthy")
    return passed("stop accepted; host still healthy", output_tail=outcome.data.get("output_tail"))


def leftover_processes(j: Journey) -> list[str]:
    """Kestrel launchers and isolated services still running."""
    names = [
        name for name in j.running_images()
        if TELEGRAM_SERVICE_IMAGE in name
        # Windows lists the image (kestrel.exe); POSIX the argv (python .../bin/kestrel ...).
        or any(Path(part).stem == "kestrel" for part in name.split()[:2])
    ]
    if j.host_process is not None and j.host_process.poll() is None:
        names.append(f"launcher pid {j.host_process.pid}")
    return names


def step_terminate(j: Journey) -> Outcome:
    """``kestrel terminate`` leaves no listener and no isolated service behind.

    On Windows the recorded host PID is the venv launcher; ending it must also
    end the interpreter it started, or the server outlives its record.
    """
    if j.roster is None:
        return skipped("no roster; nothing was started")
    owners_before = listener_ownership(j.project_dir / "logs" / ".host.pid",
                                       j.roster.host_port)
    proc = j.kestrel_cli("terminate", timeout=300)
    outcome = j.cli_outcome("kestrel terminate", proc)
    deadline = time.monotonic() + 30
    while True:
        open_ports = [p for p in (j.roster.host_port, j.roster.agent_port) if j.port_open(p)]
        leftovers = leftover_processes(j)
        if (not open_ports and not leftovers) or time.monotonic() > deadline:
            break
        time.sleep(2)
    data = {"open_ports": open_ports, "leftover_processes": leftovers[:20],
            "ownership_before": owners_before}
    if outcome.status != PASS:
        outcome.data.update(data)
        return outcome
    if open_ports or leftovers:
        return failed(f"30s after terminate: ports {open_ports} open, "
                      f"processes {leftovers[:20]}", **data)
    return passed("terminated; no listeners or service processes left", **data)


def step_launcher_rename_probe(j: Journey) -> Outcome:
    """Can a running ``kestrel.exe`` be renamed? Informational, for #2923.

    The update's fix renames its own launcher aside so the install can write a
    new one. Windows allows renaming a running image unless something holds an
    open handle without delete sharing. Measured both with no ``kestrel.exe``
    running and while a ``kestrel ask`` turn runs.
    """
    if os.name != "nt":
        return skipped("Windows only")
    launcher = Path(sys.executable).parent / "kestrel.exe"
    probe = launcher.with_name("kestrel.exe.rename-probe")

    def try_rename() -> str:
        try:
            launcher.rename(probe)
        except OSError as exc:
            return f"refused: {exc}"
        probe.rename(launcher)
        return "renamed and restored"

    idle = try_rename()
    assert j.roster is not None
    # A model turn keeps a real kestrel.exe running for ~10 s on this runner.
    follower = subprocess.Popen(
        [str(j.kestrel), "ask", j.roster.agent_name,
         "Write the numbers from one to forty in words, one per line."],
        cwd=j.project_dir, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        time.sleep(3)
        running = try_rename() if follower.poll() is None else (
            f"kestrel ask exited {follower.returncode} before the probe")
    finally:
        follower.kill()
        follower.wait(timeout=30)
    data = {"no_kestrel_exe_running": idle, "while_kestrel_exe_runs": running}
    return passed(f"idle: {idle}; while running: {running}", **data)


STEPS: list[Step] = [
    Step("kestrel-entry-point", step_entry_point),
    Step("setup-quickstart", step_setup, requires=("kestrel-entry-point",)),
    Step("doctor", step_doctor, requires=("setup-quickstart",)),
    Step("roster", step_roster, requires=("setup-quickstart",)),
    Step("ollama-reachable", step_ollama_reachable, requires=("setup-quickstart",), gate=False),
    Step("host-start", step_host_start, requires=("roster",)),
    Step("first-contact", step_first_contact, requires=("host-start",)),
    Step("chat-invoke", step_chat, requires=("first-contact",)),
    Step("bang-command", step_bang_command, requires=("host-start",)),
    Step("cli-ask", step_cli_ask, requires=("host-start",)),
    Step("telegram-in-registry", step_telegram_registry, requires=("setup-quickstart",)),
    Step("telegram-install", step_telegram_install, requires=("setup-quickstart",)),
    Step("restart-with-telegram", step_restart, requires=("host-start", "telegram-install")),
    Step("telegram-loaded", step_telegram_loaded, requires=("restart-with-telegram",)),
    Step("telegram-token", step_telegram_token, requires=("telegram-loaded",)),
    Step("stop-in-flight", step_stop_in_flight, requires=("host-start",)),
    Step("terminate-start-cycle", step_terminate_start_cycle, requires=("host-start",)),
    Step("launcher-rename-probe", step_launcher_rename_probe, requires=("host-start",), gate=False),
    Step("update-while-stopped", step_update_while_stopped, requires=("host-start",)),
    Step("update-while-running", step_update_while_running, requires=("host-start",)),
]

TEARDOWN = Step("terminate", step_terminate)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def collect_artifacts(j: Journey, dest: Path) -> list[str]:
    """Copy logs and config (never .env or databases), credentials scrubbed."""
    dest.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = []
    for name in ("kestrel.toml", "multi_agent.toml", HOST_FEATURE_MANIFEST):
        if (j.project_dir / name).is_file():
            sources.append(j.project_dir / name)
    for root in ("logs", "agent_data"):
        base = j.project_dir / root
        if base.is_dir():
            sources += [p for p in base.rglob("*.log") if p.is_file()]
    copied = []
    for src in sources:
        rel = src.relative_to(j.project_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8", errors="backslashreplace")
        target.write_text(j.scrub(text), encoding="utf-8")
        copied.append(str(rel))
    return copied


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    parser.add_argument("--artifacts", type=Path, required=True, help="log/config copy dir")
    parser.add_argument("--chat-timeout", type=int, default=900,
                        help="seconds to wait for one agent turn (CPU inference is slow)")
    parser.add_argument("--telegram-window", default=">=0.1.9,<1",
                        help="PyPI version window for the Telegram manifest entry")
    args = parser.parse_args(argv)

    project_dir = Path.cwd()
    try:
        refuse_unfresh(project_dir, dict(os.environ))
    except JourneyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    journey = Journey(project_dir, args.chat_timeout, args.telegram_window)
    results: list[StepResult] = []
    try:
        run_steps(journey, STEPS, into=results)
    finally:
        # A driver bug propagates, but not before the host is stopped and the
        # steps that did run are reported: that is what the run exists for.
        try:
            journey.deadline = time.monotonic() + TEARDOWN_RESERVE_SECONDS + 300
            run_steps(journey, [TEARDOWN], into=results)
        finally:
            write_report(journey, results, args.report, collect_artifacts(journey, args.artifacts))
    return exit_status(results)


def write_report(j: Journey, results: list[StepResult], path: Path, copied: list[str]) -> None:
    header = {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "kestrel": j.kestrel.name,
    }
    report = {"header": header, "steps": [dataclasses.asdict(r) for r in results],
              "artifacts": copied}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = render_markdown(results, header)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary)
    j.log(summary)


if __name__ == "__main__":
    raise SystemExit(main())
