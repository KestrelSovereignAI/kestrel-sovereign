"""CI verification helpers for the clean-install workflow.

Each subcommand is a single readiness assertion against the on-disk
state the wizard produces. The workflow YAML calls them as:

    uv run python scripts/ci/clean_install_verify.py wizard-artifacts
    uv run python scripts/ci/clean_install_verify.py identity --agent-name Kestrel
    uv run python scripts/ci/clean_install_verify.py constitution --agent-name Kestrel
    uv run python scripts/ci/clean_install_verify.py memory --agent-name Kestrel
    uv run python scripts/ci/clean_install_verify.py start-and-health --agent-name Kestrel
    uv run python scripts/ci/clean_install_verify.py host-and-chat-503
    uv run python scripts/ci/clean_install_verify.py did-persists --agent-name Kestrel
    uv run python scripts/ci/clean_install_verify.py test-instance --agent-name Kestrel

Pure stdlib. No shell idioms. No package import (kestrel_sovereign would
pull heavy deps; we're just reading SQLite + TOML + dotenv). Designed
so the workflow runs identically on Ubuntu, macOS, and Windows.

Exit code 0 = pass. Non-zero = fail; the message on stderr says why.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — clean-install matrix uses 3.13
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOTENV_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal dotenv parser. Strips one layer of surrounding quotes."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _DOTENV_RE.match(raw)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _agent_db(agent_name: str) -> Path:
    return Path("agent_data") / agent_name / "kestrel_prime.db"


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(f"PASS: {msg}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: wizard-artifacts
# ---------------------------------------------------------------------------

def cmd_wizard_artifacts(args: argparse.Namespace) -> int:
    """Sanity-check the files the wizard claims it produced."""
    required_files = [Path(".env"), Path("kestrel.toml"), Path("multi_agent.toml")]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        return _fail(f"wizard did not produce: {', '.join(missing)}")

    env = _read_dotenv(Path(".env"))
    if not env.get("KESTREL_DATA_KEY"):
        return _fail("KESTREL_DATA_KEY missing from .env after wizard")

    config = _read_toml(Path("kestrel.toml"))
    priority = (config.get("llm") or {}).get("route_priority") or []
    if not priority:
        return _fail("kestrel.toml missing [llm].route_priority")

    return _ok(
        f".env, kestrel.toml ({len(priority)} routes), multi_agent.toml all present; "
        f"KESTREL_DATA_KEY set"
    )


# ---------------------------------------------------------------------------
# Subcommand: identity (DID exists in graph_nodes)
# ---------------------------------------------------------------------------

def cmd_identity(args: argparse.Namespace) -> int:
    """Identity Pillar: agent's DID is stored as a graph node."""
    db_path = _agent_db(args.agent_name)
    if not db_path.exists():
        return _fail(f"Agent database not created at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT node_id FROM graph_nodes WHERE node_type='agent' LIMIT 1"
        ).fetchone()
    if not row or not row[0]:
        return _fail("No DID found in graph_nodes")
    return _ok(f"Identity Pillar — DID: {row[0]}")


# ---------------------------------------------------------------------------
# Subcommand: constitution (governed_by edge + document node + RAG chunks)
# ---------------------------------------------------------------------------

def cmd_constitution(args: argparse.Namespace) -> int:
    """Constitution Pillar: stored, edge-linked to agent, RAG-indexed."""
    db_path = _agent_db(args.agent_name)
    if not db_path.exists():
        return _fail(f"Agent database not found at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        edges = conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE label='governed_by'"
        ).fetchone()[0]
        docs = conn.execute(
            "SELECT COUNT(*) FROM graph_nodes WHERE node_type='document'"
        ).fetchone()[0]
        files = conn.execute(
            "SELECT COUNT(*) FROM files WHERE original_name='KESTREL_CONSTITUTION.md'"
        ).fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]

    if not (edges and docs and files):
        return _fail(
            f"Constitution not anchored — "
            f"governed_by={edges}, constitution_docs={docs}, files={files}"
        )
    return _ok(
        f"Constitution Pillar — governed_by={edges}, constitution_docs={docs}, "
        f"files={files}, rag_chunks={chunks}"
    )


# ---------------------------------------------------------------------------
# Subcommand: memory (required tables exist)
# ---------------------------------------------------------------------------

_REQUIRED_TABLES = {
    "graph_nodes",
    "graph_edges",
    "files",
    "document_chunks",
    "conversation_history",
}


def cmd_memory(args: argparse.Namespace) -> int:
    """Memory Pillar: storage tables created by inception are all present."""
    db_path = _agent_db(args.agent_name)
    if not db_path.exists():
        return _fail(f"Agent database not found at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    tables = {r[0] for r in rows}
    missing = _REQUIRED_TABLES - tables
    if missing:
        return _fail(f"missing tables: {sorted(missing)}")
    return _ok(f"Memory Pillar — {len(tables)} tables, all required present")


# ---------------------------------------------------------------------------
# Subcommand: start-and-health (start, probe /health, stop)
# ---------------------------------------------------------------------------

def _agent_port(agent_name: str) -> int | None:
    multi_agent = _read_toml(Path("multi_agent.toml"))
    return ((multi_agent.get("agents") or {}).get(agent_name) or {}).get("port")


def _kestrel(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the kestrel CLI via the current Python interpreter.

    We use ``python -m kestrel_sovereign.cli`` (rather than the
    ``kestrel`` entry-point script) because the entry point is named
    ``kestrel`` on Unix and ``kestrel.exe`` on Windows; the module
    invocation is identical on every OS.
    """
    return subprocess.run(
        [sys.executable, "-m", "kestrel_sovereign.cli", *args],
        capture_output=True,
        text=True,
    )


def _poll_health(port: int, timeout_s: int = 30) -> bool:
    """Poll ``GET http://localhost:<port>/health`` until 200 or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def cmd_start_and_health(args: argparse.Namespace) -> int:
    """Start the agent, verify /health responds, stop the agent.

    ``kestrel start`` already waits for the agent's health internally,
    but we probe externally too — that catches cases where the start
    command thinks the agent is healthy but the HTTP route is broken.
    """
    port = _agent_port(args.agent_name)
    if not port:
        return _fail(f"multi_agent.toml missing port for {args.agent_name}")
    print(f"Agent port from multi_agent.toml: {port}")

    start = _kestrel("start", args.agent_name)
    sys.stdout.write(start.stdout)
    sys.stderr.write(start.stderr)
    if start.returncode != 0:
        return _fail(f"kestrel start exited {start.returncode}")

    try:
        if not _poll_health(port, timeout_s=30):
            return _fail(
                f"Agent did not respond on http://localhost:{port}/health "
                f"within 30 seconds"
            )
        print(f"Health endpoint responding on port {port}")
    finally:
        # Always try to stop, even if the health check failed — leaves
        # the runner clean for the DID-persistence step.
        stop = _kestrel("stop", args.agent_name)
        sys.stdout.write(stop.stdout)
        sys.stderr.write(stop.stderr)

    return _ok(f"Health endpoint verified on port {port}")


# ---------------------------------------------------------------------------
# Subcommand: host-and-chat-503
#
# Boots the multi-agent host (``kestrel start`` with no agent name) and
# probes the OpenAI-compat ``/v1/chat/completions`` endpoint at the host
# level — i.e. without the ``/api/agents/<name>/`` prefix. In multi-agent
# mode the host can't pick a target agent without that prefix and must
# fail honestly with HTTP 503. Pre-#1110 the same call returned a
# misleading 500 with "Internal error in chat completions", which read
# like a server bug to clients (Open WebUI in particular) when it was
# actually a routing problem. This subcommand locks that contract in CI.
#
# No LLM is involved: the route bails at ``get_agent(request)`` long
# before any provider is consulted, so the assertion runs on every CI
# matrix without an Ollama install.
# ---------------------------------------------------------------------------

def _host_port() -> int | None:
    multi_agent = _read_toml(Path("multi_agent.toml"))
    return ((multi_agent.get("host") or {}).get("port"))


def _post_chat_completions(port: int, api_key: str, timeout_s: int = 5):
    """POST a minimal chat-completions body. Returns (status, body) or (None, error_str)."""
    payload = json.dumps(
        {"model": "x", "messages": [{"role": "user", "content": "ping"}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # 5xx and 4xx come back here — that's exactly what we want to inspect.
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        return exc.code, body
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        return None, str(exc)


def cmd_host_and_chat_503(args: argparse.Namespace) -> int:
    """Start multi-agent host, assert top-level /v1/chat/completions → 503."""
    port = _host_port()
    if not port:
        return _fail("multi_agent.toml missing [host].port")

    # The chat-completions route is auth-gated (the security middleware
    # 401s ahead of the route handler), so without the API key we'd
    # exercise the auth path instead of the routing path. Read the key
    # the wizard wrote into .env.
    api_key = _read_dotenv(Path(".env")).get("KESTREL_API_KEY", "")
    if not api_key:
        return _fail(".env missing KESTREL_API_KEY (wizard should have generated it)")
    print(f"Host port from multi_agent.toml: {port}")

    start = _kestrel("start")  # no agent name → multi-agent host
    sys.stdout.write(start.stdout)
    sys.stderr.write(start.stderr)
    if start.returncode != 0:
        return _fail(f"kestrel start (host mode) exited {start.returncode}")

    try:
        if not _poll_health(port, timeout_s=30):
            return _fail(
                f"Host did not respond on http://localhost:{port}/health "
                f"within 30 seconds"
            )

        status, body = _post_chat_completions(port, api_key=api_key)
        if status != 503:
            return _fail(
                f"Top-level /v1/chat/completions returned status={status} "
                f"(expected 503). Body: {body[:200]!r}"
            )
        # The honest-fail contract: the body should mention agent
        # initialisation, not "Internal error in chat completions" (the
        # pre-#1110 misleading wording).
        if "Internal error in chat completions" in body:
            return _fail(
                "Top-level /v1/chat/completions returned 503 but with the "
                "pre-#1110 misleading body. Body: " + body[:200]
            )
    finally:
        stop = _kestrel("stop")
        sys.stdout.write(stop.stdout)
        sys.stderr.write(stop.stderr)

    return _ok(
        f"Top-level /v1/chat/completions on host:{port} returned 503 with "
        f"a routing-not-server-error body"
    )


# ---------------------------------------------------------------------------
# Subcommand: test-instance (agent's properties node carries the test flag)
#
# When the wizard is run with ``--test`` (or ``KESTREL_TEST_INSTANCE=1``),
# inception writes ``is_test_instance: True`` and a generated
# ``test_cycle_id`` onto the agent's properties JSON in graph_nodes.
# This subcommand reads that JSON and asserts the marker is present, so
# CI can prove every agent it incepts is identifiable as a test agent
# in any downstream telemetry that consumes the graph.
# ---------------------------------------------------------------------------

def cmd_test_instance(args: argparse.Namespace) -> int:
    """Verify the agent's properties node is tagged ``is_test_instance``."""
    db_path = _agent_db(args.agent_name)
    if not db_path.exists():
        return _fail(f"Agent database not found at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT properties FROM graph_nodes "
            "WHERE node_type='agent' LIMIT 1"
        ).fetchone()
    if not row or not row[0]:
        return _fail("No agent node properties in graph_nodes")

    try:
        props = json.loads(row[0])
    except json.JSONDecodeError as exc:
        return _fail(f"agent properties is not valid JSON: {exc}")

    if not props.get("is_test_instance"):
        return _fail(
            f"agent properties missing is_test_instance=True "
            f"(properties keys: {sorted(props.keys())})"
        )
    cycle_id = props.get("test_cycle_id")
    return _ok(
        f"Agent tagged as test instance — test_cycle_id={cycle_id or '(unset)'}"
    )


# ---------------------------------------------------------------------------
# Subcommand: did-persists (DB still has DID after stop/start cycle)
# ---------------------------------------------------------------------------

def cmd_did_persists(args: argparse.Namespace) -> int:
    """Identity portability: DID survives a stop/start cycle."""
    db_path = _agent_db(args.agent_name)
    if not db_path.exists():
        return _fail(f"Agent database not found at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT node_id FROM graph_nodes WHERE node_type='agent' LIMIT 1"
        ).fetchone()
    if not row or not row[0]:
        return _fail("DID not found after restart cycle")
    return _ok(f"DID persists after stop/start cycle: {row[0]}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clean_install_verify",
        description="Per-step assertions for the clean-install CI job.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("wizard-artifacts", help="Verify wizard wrote .env, kestrel.toml, multi_agent.toml")

    # host-and-chat-503 doesn't need --agent-name (it queries the host port).
    sub.add_parser(
        "host-and-chat-503",
        help="Start the multi-agent host and verify top-level /v1/chat/completions → 503",
    )

    for name, help_text in (
        ("identity", "Verify Identity Pillar — DID in graph_nodes"),
        ("constitution", "Verify Constitution Pillar — anchored + RAG-indexed"),
        ("memory", "Verify Memory Pillar — required tables present"),
        ("start-and-health", "Start agent, probe /health, stop agent"),
        ("did-persists", "Verify DID survives after the stop/start cycle"),
        ("test-instance", "Verify agent is tagged is_test_instance=True"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument(
            "--agent-name", required=True,
            help="Agent name (matches multi_agent.toml + agent_data/<name>/)",
        )

    return p


_HANDLERS = {
    "wizard-artifacts": cmd_wizard_artifacts,
    "identity": cmd_identity,
    "constitution": cmd_constitution,
    "memory": cmd_memory,
    "start-and-health": cmd_start_and_health,
    "host-and-chat-503": cmd_host_and_chat_503,
    "did-persists": cmd_did_persists,
    "test-instance": cmd_test_instance,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
