"""
Unified Kestrel CLI for host and agent management.

This is the single entry point for managing the Kestrel Host and all agents.
It subsumes main.py's interactive chat into `kestrel shell <name>`.

Commands:
    kestrel start                  # start all agents in-process (default)
    kestrel start <name>           # start just one agent (standalone process)
    kestrel stop                   # stop everything (agents first, then host)
    kestrel stop <name>            # stop just one agent
    kestrel status                 # table: host + all agents with ports, PIDs, status
    kestrel logs <name>            # tail agent logs (or "host" for host logs)
    kestrel list                   # list multi_agent agents, ports, data dirs
    kestrel create <name>          # inception: generate DID, create agent folder, add to multi_agent.toml
    kestrel shell <name>           # interactive CLI chat (what main.py does today)
    kestrel health                 # run health check
    kestrel identity harden-exports  # privatize eligible legacy identity exports
    kestrel config <agent_dir>     # show/edit agent config
    kestrel deploy <profile>       # deploy agent to Cloud Run / Azure Container Apps
    kestrel verify-install [TESTS...]  # run the 5-test clean-install matrix in throwaway venvs
    kestrel demo run <name>        # run a demo against an isolated demo agent
    kestrel agent docker create    # create a Docker-isolated agent (KESTREL_DATA_KEY rail)
    kestrel agent docker chat      # interactive chat with a Docker-isolated agent
    kestrel agent docker retire    # retire a Docker-isolated agent
    kestrel docker remote build    # build the lightweight remote-LLM image
    kestrel docker remote run      # run the lightweight remote-LLM container
    kestrel docker build <preset>  # build a Cloud-Build / GCR specialty image
    kestrel ipfs {build,deploy,pin}    # self-hosted IPFS node lifecycle (Kubo + GCS)
    kestrel runpod {deploy,status,stop,kill}   # RunPod GPU pod lifecycle (LoRA training)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from kestrel_sovereign import __version__
from kestrel_sovereign.paths import load_project_env, spawned_agent_env
from kestrel_sovereign.multi_agent.config import (
    MultiAgentConfig,
    LocalAgentConfig,
    MULTI_AGENT_CONFIG_FILENAME,
    DEFAULT_AGENT_START_PORT,
)
from kestrel_sovereign.multi_agent.process_manager import ProcessManager


# Tokens that terminate an interactive `kestrel shell` session. Matched
# case-insensitively against the *trimmed* input — typing "  EXIT  " is
# the same as typing "/exit". Both ``_run_shell`` (local in-process) and
# ``_run_http_shell`` (HTTP-routed) honor this same list so users don't
# have to remember which mode they're in. #658.
_SHELL_EXIT_TOKENS = frozenset({"!quit", "/exit", "exit", "quit", "q", ":q"})
_SHELL_EXIT_HINT = (
    "Type !quit, /exit, exit, quit, or q to leave the shell."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_dir() -> Path:
    """Get the project root directory.

    Delegates to :func:`kestrel_sovereign.paths.project_dir`, which honours
    ``KESTREL_HOME``, walks up from CWD looking for marker files
    (``multi_agent.toml``, ``kestrel.toml``, ``.env``), and falls back to
    ``~/.kestrel`` for pip-installed users with no project in CWD. Crucially,
    this no longer returns ``site-packages/`` when the package is pip-installed
    — that was silent data loss on ``pip install --upgrade``.
    """
    from kestrel_sovereign.paths import project_dir

    return project_dir()


def _apply_target_data_key_custody(project_dir: Path) -> Optional[str]:
    """Resolve the single effective ``KESTREL_DATA_KEY`` for this target home
    and make it authoritative in the process, mirroring the wizard ``keys`` step
    (#2468).

    Used by the CLI paths that can reach inception **without** running the keys
    step — chiefly ``kestrel create`` (never runs it). It previously relied on
    the import-time ``load_dotenv()`` in ``inception_service``; with that removed
    it lost all ``.env`` awareness, so inception could fail or encrypt with the
    wrong key.

    Loads the target home's non-key environment first (so an existing
    ``KESTREL_DID_WEB_DOMAIN`` etc. is honored, not overwritten) — but *excludes*
    the data key from that pre-load so a stale persisted value can't mask an
    exported⇄persisted conflict — then funnels the data key through the shared
    ``ensure_effective_data_key`` primitive: it persists (round-trip-verified)
    and exports the one effective key, generating a fresh one if the home has
    none. Returns an error message to refuse on (conflict / invalid / non
    round-trippable material) — the same custody block the keys step raises — or
    ``None`` on success, at which point ``os.environ[KESTREL_DATA_KEY]`` holds
    exactly the key persisted in the target ``.env`` (encrypt-key == persist-key).
    """
    from kestrel_sovereign.setup.steps.keys import (
        DATA_KEY_ENV,
        ensure_effective_data_key,
    )

    load_project_env(project_dir, exclude=(DATA_KEY_ENV,))
    _, conflict, _, _ = ensure_effective_data_key(
        project_dir / ".env", generate_if_missing=True
    )
    return conflict


def cmd_list(args) -> int:
    """List all agents in multi_agent."""
    project_dir = _get_project_dir()
    multi_agent = MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)

    local_agents = multi_agent.get_local_agents()
    remote_agents = multi_agent.get_remote_agents()

    if not local_agents and not remote_agents:
        print("No agents configured")
        print("Create one: kestrel create <name>")
        return 0

    if local_agents:
        print(f"  {'NAME':12} {'PORT':>6}   {'DATA DIR':30} {'AUTOSTART'}")
        for name, cfg in local_agents.items():
            autostart_str = "autostart" if cfg.autostart else "manual"
            print(f"  {name:12} {cfg.port:>6}   {str(cfg.data_dir):30} {autostart_str}")

    if remote_agents:
        print(f"\n  {'NAME':12} {'URL'}")
        for name, cfg in remote_agents.items():
            print(f"  {name:12} {cfg.url}")

    return 0


def cmd_create(args) -> int:
    """Create a new agent via inception (thin wrapper over the agent step)."""
    from kestrel_sovereign.constitution.emancipation import (
        EmancipationConfigError,
        parse_emancipation_block,
    )
    from kestrel_sovereign.setup.steps.agent import create_agent
    from kestrel_sovereign.setup.toml_file import read_toml

    project_dir = _get_project_dir()

    name = args.name
    agent_data_dir = project_dir / "agent_data" / name

    if (agent_data_dir / "kestrel_prime.db").exists():
        print(f"Agent '{name}' already exists at {agent_data_dir}")
        return 1

    # inception_service no longer loads .env at import time (that leaked the
    # current-directory key into os.environ and could encrypt with the wrong
    # KESTREL_DATA_KEY — #2468). Resolve the *resolved target home's* effective
    # data key deliberately, mirroring the wizard keys step: refuse a split
    # brain (exported key A ⇄ persisted key B) before inception rather than
    # encrypt with one key while the home persists another. This runs only
    # after the already-exists guard — an existing agent aborts before any
    # key resolution, since no inception (and no encryption) will occur.
    custody_error = _apply_target_data_key_custody(project_dir)
    if custody_error:
        print(custody_error, file=sys.stderr)
        return 1

    # Pick up any [emancipation] block authored in kestrel.toml so the
    # CLI path anchors the same contract the wizard would. Without this
    # an authored contract is silently ignored at inception.
    contract = None
    kestrel_toml = project_dir / "kestrel.toml"
    if kestrel_toml.exists():
        try:
            contract = parse_emancipation_block(read_toml(kestrel_toml))
        except EmancipationConfigError as exc:
            print(f"[emancipation] block in kestrel.toml is invalid: {exc}")
            print("Inception aborted to avoid anchoring an unsigned contract.")
            return 1

    is_test_instance = bool(getattr(args, "test", False))
    suffix = " (test instance)" if is_test_instance else ""
    print(f"\U0001F985 Creating new Kestrel agent: {name}{suffix}")
    if contract is not None and contract.enabled:
        print("   Amendment VIII active \u2014 Sovereign-authored contract will be anchored.")

    try:
        result = create_agent(
            name=name,
            project_dir=project_dir,
            agent_data_root=project_dir / "agent_data",
            autostart=True,
            port=args.port,
            emancipation_contract=contract,
            is_test_instance=is_test_instance,
        )
    except Exception as exc:  # noqa: BLE001 \u2014 surface inception failure verbatim
        print(f"Inception failed: {exc}")
        return 1

    print(f"   DID: {result.did or '(unknown)'}")
    print(f"   Data dir: agent_data/{name}/")
    print(f"   Port: {result.port} (next available)")
    print(f"   Added to {MULTI_AGENT_CONFIG_FILENAME}")
    print(f"\u2705 Agent created. Start with: kestrel start {name}")
    return 0


def _detect_running_agent_server(
    agent_name: str,
    agent_cfg: LocalAgentConfig,
    multi_agent: MultiAgentConfig,
) -> Optional[tuple[str, str]]:
    """Probe for a running server that hosts this agent.

    Returns ``(base_url, api_key)`` if found — where ``base_url`` is what
    the shell should POST ``/api/agent/invoke`` against — or ``None`` if no
    server is up for this agent. Caller uses ``None`` to fall back to
    the local in-process shell.

    Tries two candidates in order, matching the two modes ``kestrel start``
    supports:

    1. **Standalone / subprocess mode** — agent runs on its own port
       (``agent_cfg.port``). Base URL is that port's root; no path prefix.
    2. **In-process multi-agent mode** — all agents share the host port
       (``multi_agent.host.port``) under ``/api/agents/{name}/``. Base URL is
       host:port + that prefix.

    Health probe uses ``GET /health`` (public, no auth). Key fetch uses
    ``GET /api/auth/key`` (public). Network errors fall through silently
    — no server is a normal case, not an error.
    """
    import httpx

    candidates = [
        (f"http://localhost:{agent_cfg.port}", ""),
        (
            f"http://localhost:{multi_agent.host.port}",
            f"/api/agents/{agent_name}",
        ),
    ]
    for origin, prefix in candidates:
        try:
            health = httpx.get(f"{origin}/health", timeout=1.0)
        except httpx.RequestError:
            continue
        if health.status_code != 200:
            continue
        try:
            key_resp = httpx.get(f"{origin}/api/auth/key", timeout=2.0)
        except httpx.RequestError:
            continue
        api_key = ""
        if key_resp.status_code == 200:
            try:
                api_key = key_resp.json().get("key", "") or ""
            except ValueError:
                api_key = ""
        # In multi-agent mode, confirm the named agent is actually routed
        # by this server. The routing middleware returns 404 for unknown
        # names; hit the prefix's /health proxy to verify before declaring
        # success. (Standalone mode has no prefix, and /health above
        # already confirmed the single-agent server is alive.)
        if prefix:
            try:
                scoped = httpx.get(
                    f"{origin}{prefix}/health", timeout=1.0,
                    headers={"X-API-Key": api_key} if api_key else {},
                )
            except httpx.RequestError:
                continue
            if scoped.status_code != 200:
                continue
        return (f"{origin}{prefix}", api_key)
    return None


# Default read timeout (seconds) for talking to a running agent. An agentic
# turn can fan out into several provider round-trips + tool executions (a
# Codex/openai:plan turn alone carries a ~300s server-side watchdog), so the
# old hardcoded 600s flat timeout would abort a legitimately-long turn. The
# read timeout is configurable via ``KESTREL_ASK_TIMEOUT_SECONDS``; connect /
# write / pool stay short so a dead server still fails fast.
_DEFAULT_ASK_READ_TIMEOUT = 1800.0


def _agent_http_timeout(read_override: Optional[float] = None):
    """``httpx.Timeout`` for agent calls: fast connect, long (configurable) read.

    Precedence for the read timeout: explicit ``read_override`` (e.g. the
    ``--timeout`` flag) > ``KESTREL_ASK_TIMEOUT_SECONDS`` env > default. A
    value <= 0 disables the read timeout (wait indefinitely) for very long
    batch turns.
    """
    import httpx

    read = read_override
    if read is None:
        env = os.environ.get("KESTREL_ASK_TIMEOUT_SECONDS", "").strip()
        try:
            read = float(env) if env else _DEFAULT_ASK_READ_TIMEOUT
        except ValueError:
            read = _DEFAULT_ASK_READ_TIMEOUT
    read_timeout = None if read is not None and read <= 0 else read
    return httpx.Timeout(read_timeout, connect=10.0, write=30.0, pool=10.0)


def _run_http_shell(
    agent_name: str,
    base_url: str,
    api_key: str,
) -> int:
    """Interactive chat loop that POSTs to a running agent's HTTP API.

    Used when ``_detect_running_agent_server`` finds a live server for
    the named agent. The in-process shell (``_run_shell``) is only used
    as a fallback when no server is running — see #654.
    """
    import httpx

    print(f"✅ Connected to running agent at {base_url}")
    print(f"   Agent: {agent_name}")
    print(f"   {_SHELL_EXIT_HINT}")

    headers = {"X-API-Key": api_key} if api_key else {}
    with httpx.Client(base_url=base_url, headers=headers,
                      timeout=_agent_http_timeout()) as client:
        while True:
            try:
                user_input = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in _SHELL_EXIT_TOKENS:
                break
            try:
                resp = client.post("/api/agent/invoke", json={"input": user_input})
            except httpx.RequestError as e:
                print(f"\nConnection error: {e}")
                continue
            if resp.status_code != 200:
                print(f"\nHTTP {resp.status_code}: {resp.text[:200]}")
                continue
            try:
                body = resp.json()
            except ValueError:
                print(f"\nKestrel returned non-JSON response: {resp.text[:200]}")
                continue
            print(f"\nKestrel: {body.get('response', '')}")
    return 0


def cmd_shell(args) -> int:
    """Interactive CLI chat. Routes to a running server when one is up
    for this agent; falls back to an in-process agent instance if not.
    """
    project_dir = _get_project_dir()
    # The in-process fallback builds a whole agent here — its own LLMService,
    # its own storage — so this command needs the home's environment, and it
    # used to get it as a side effect of constructing that LLMService. That is
    # exactly the accident #2896 removed. Without it `KESTREL_DATA_KEY` is
    # absent (the key hierarchy has no `.env` fallback), the first encrypted
    # read raises, and the shell reports a decryption problem for what is an
    # env-loading regression.
    load_project_env(project_dir)
    multi_agent = MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    local_agents = multi_agent.get_local_agents()

    if args.name not in local_agents:
        print(f"Agent '{args.name}' not found in multi_agent config")
        print(f"Available agents: {', '.join(local_agents.keys()) or '(none)'}")
        return 1

    agent_cfg = local_agents[args.name]
    agent_dir = (project_dir / agent_cfg.data_dir).resolve()

    if not (agent_dir / "kestrel_prime.db").exists():
        print(f"Agent database not found at {agent_dir}")
        print(f"Create the agent first: kestrel create {args.name}")
        return 1

    # Extension loading (e.g., ElderlyExtension) is only supported in the
    # local-process shell because it mutates the live agent object. Skip
    # HTTP routing when the user passed --app so extensions still work.
    use_extension = bool(getattr(args, "app", None))
    if not use_extension:
        server = _detect_running_agent_server(args.name, agent_cfg, multi_agent)
        if server is not None:
            base_url, api_key = server
            return _run_http_shell(args.name, base_url, api_key)

    # Fall back to in-process agent when no server is running (or when
    # an extension is requested — see comment above).
    return asyncio.run(_run_shell(agent_dir, args))


def _run_http_ask(
    agent_name: str,
    base_url: str,
    api_key: str,
    message: str,
    session_id: Optional[str],
    as_json: bool,
    timeout_seconds: Optional[float] = None,
) -> int:
    """One-shot POST to a running agent's HTTP API; print the reply.

    Non-interactive sibling of ``_run_http_shell`` — same endpoint, same
    auth, no REPL. Deliberately does NOT boot an in-process agent: the
    point of ``kestrel ask`` is talking to the *warm running* agent for
    scripting/validation; a cold in-process boot is the multi-minute trap
    this command exists to avoid (see #1287).
    """
    import httpx

    payload = {"input": message}
    if session_id:
        payload["session_id"] = session_id
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        with httpx.Client(base_url=base_url, headers=headers,
                          timeout=_agent_http_timeout(timeout_seconds)) as client:
            resp = client.post("/api/agent/invoke", json=payload)
    except httpx.RequestError as e:
        print(f"Connection error talking to '{agent_name}' at {base_url}: {e}",
              file=sys.stderr)
        return 1
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return 1
    try:
        body = resp.json()
    except ValueError:
        print(f"Non-JSON response: {resp.text[:300]}", file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(body))
    else:
        print(body.get("response", ""))
    return 0


def cmd_ask(args) -> int:
    """Send ONE message to an already-running agent and print its reply.

    Non-interactive, scriptable counterpart of ``kestrel shell``. Reuses
    the same server detection + ``/api/agent/invoke`` path. If no server
    hosts the agent it errors (non-zero) rather than cold-booting an
    in-process instance — that fallback is the latency trap ``ask`` is
    built to sidestep (#1287).
    """
    project_dir = _get_project_dir()
    multi_agent = MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    local_agents = multi_agent.get_local_agents()

    if args.name not in local_agents:
        print(f"Agent '{args.name}' not found in multi_agent config", file=sys.stderr)
        print(f"Available agents: {', '.join(local_agents.keys()) or '(none)'}",
              file=sys.stderr)
        return 1

    agent_cfg = local_agents[args.name]
    server = _detect_running_agent_server(args.name, agent_cfg, multi_agent)
    if server is None:
        print(
            f"No running server hosts agent '{args.name}'. "
            f"`kestrel ask` only talks to a live agent — start it first "
            f"(`kestrel start {args.name}` or `kestrel start host`). "
            f"For an offline in-process session use `kestrel shell {args.name}`.",
            file=sys.stderr,
        )
        return 1

    base_url, api_key = server
    return _run_http_ask(
        args.name,
        base_url,
        api_key,
        args.message,
        getattr(args, "session", None),
        bool(getattr(args, "json", False)),
        getattr(args, "timeout", None),
    )


async def _run_shell(agent_dir: Path, args) -> int:
    """Run the interactive chat shell for an agent."""
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.security.encryption import DecryptionError
    from kestrel_sovereign.kestrel_agent import (
        KestrelAgent,
        await_agent_shutdown_completion,
    )
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT
    import logging

    logger = logging.getLogger(__name__)

    # Load agent DID
    db_path = agent_dir / "kestrel_prime.db"
    storage = AsyncStorage(str(db_path))
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        if not agent_nodes:
            print("No agent found in the database. Run inception service first.")
            return 1
        agent_did = agent_nodes[0].node_id
    finally:
        await storage.close()

    # Initialize agent
    llm_service = LLMService()
    agent = KestrelAgent(
        did=agent_did,
        storage_path=str(db_path),
        llm_service=llm_service,
    )
    await agent.initialize()

    # Load extension if requested
    if hasattr(args, 'app') and args.app:
        if args.app == 'elderly':
            from kestrel_sovereign.extensions.elderly_extension import ElderlyExtension
            agent.extension = ElderlyExtension(agent)
            agent.app_context = args.app
            print(f"   Extension Loaded: {args.app}")

    print("\u2705 Kestrel Agent Initialized.")
    print(f"   DID: {agent.agent_id}")
    print(f"   Memory: {agent_dir}")
    print(f"   {_SHELL_EXIT_HINT}")

    decryption_error_count = 0
    MAX_DECRYPTION_ERRORS = 3

    try:
        while True:
            user_input = input("\n> ")
            if user_input.strip().lower() in _SHELL_EXIT_TOKENS:
                break
            try:
                response = await agent.process_input(user_input)
                decryption_error_count = 0
                print(f"\nKestrel: {response}")
            except DecryptionError:
                decryption_error_count += 1
                print(f"\n\U0001f510 DECRYPTION ERROR: Cannot read encrypted data.")
                print(f"   This usually means KESTREL_DATA_KEY is incorrect or missing.")
                print(f"   Error count: {decryption_error_count}/{MAX_DECRYPTION_ERRORS}")

                if decryption_error_count >= MAX_DECRYPTION_ERRORS:
                    print("\n\u26a0\ufe0f  Too many decryption errors. Entering safe mode.")
                    print("   The agent cannot access encrypted memories.")
                    print("   Please verify KESTREL_DATA_KEY and restart.")
                    print("   Use !quit to exit.")
                    if hasattr(agent, 'enter_safe_mode'):
                        await agent.enter_safe_mode(
                            "Repeated encrypted-state decryption failures"
                        )

    except KeyboardInterrupt:
        print("\nDeactivating agent...")
    finally:
        cancelled = False
        try:
            await asyncio.wait_for(agent.shutdown(), timeout=SHUTDOWN_TIMEOUT)
            print("Agent deactivated.")
        except asyncio.TimeoutError:
            print(
                f"Agent shutdown timed out ({SHUTDOWN_TIMEOUT}s); "
                "waiting for durable cleanup."
            )
        except asyncio.CancelledError:
            cancelled = True
            print("Agent shutdown cancelled.")
        except Exception:
            print("Agent deactivated (with errors).")
        cancelled = await await_agent_shutdown_completion(agent) or cancelled
        if cancelled:
            raise asyncio.CancelledError()

    return 0


def cmd_health(args) -> int:
    """Deprecated alias for ``kestrel doctor``. Removed in a future release."""
    print("(`kestrel health` is deprecated — use `kestrel doctor`)")
    return cmd_doctor(args)


def cmd_doctor(args) -> int:
    """Diagnose readiness without making any changes."""
    from kestrel_sovereign.doctor import diagnose, format_report

    project_dir = _get_project_dir()
    report = diagnose(project_dir)
    print(format_report(report))
    return 0 if report.ready else 1


def cmd_identity(args) -> int:
    """Dispatch local identity-custody maintenance commands."""

    if args.identity_command != "harden-exports":
        print("Usage: kestrel identity harden-exports")
        return 1

    from kestrel_sovereign.identity.protected_export import (
        IdentityExportSecurityError,
        effective_identity_export_roots,
        harden_legacy_identity_exports,
    )

    project_dir = _get_project_dir()
    roots = effective_identity_export_roots(project_dir)
    try:
        result = harden_legacy_identity_exports(roots)
    except IdentityExportSecurityError as exc:
        print(f"Identity export permission hardening refused: {exc}")
        return 1
    print(
        "Identity export permission hardening: "
        f"hardened={result.hardened}, "
        f"already_private={result.already_private}, "
        f"refused={result.refused}, "
        f"missing_roots={result.missing_roots}. "
        "Package contents were not read."
    )
    return 1 if result.refused else 0


def cmd_storage(args) -> int:
    """Dispatch ``kestrel storage`` subcommands."""
    storage_commands = {
        "health": cmd_storage_health,
    }
    handler = storage_commands.get(args.storage_command)
    if handler is None:
        print("Usage: kestrel storage {health}")
        return 1
    return handler(args)


async def _run_auth_login(args) -> int:
    """Run an interactive provider login flow with SIGINT cancellation."""
    route = (args.route or "").strip().lower()
    if route != "openai:plan":
        print(
            "error: only `--route openai:plan` has an interactive login "
            "flow in this release.",
            file=sys.stderr,
        )
        return 2

    import signal

    from kestrel_sovereign.llm.cancellation import AuthCancellationToken
    from kestrel_sovereign.llm.codex_app_server import (
        CodexAppServerCancelled,
        CodexAppServerClient,
    )

    token = AuthCancellationToken()
    loop = asyncio.get_running_loop()
    old_handler = None
    used_loop_signal_handler = False

    def _cancel_login() -> None:
        token.cancel()

    try:
        try:
            loop.add_signal_handler(signal.SIGINT, _cancel_login)
            used_loop_signal_handler = True
        except (NotImplementedError, RuntimeError):
            old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda _signum, _frame: _cancel_login())

        client = CodexAppServerClient()
        try:
            await client.ensure_started(cancellation_token=token)
        except CodexAppServerCancelled:
            print(f"{route} login cancelled.")
            return 130
        finally:
            await client.aclose()
    finally:
        if used_loop_signal_handler:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass
        elif old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)

    print(f"{route} login ready.")
    return 0


def cmd_auth(args) -> int:
    """Dispatch auth subcommands."""
    if args.auth_command == "login":
        return asyncio.run(_run_auth_login(args))
    print("Usage: kestrel auth login --route openai:plan")
    return 1


def cmd_tool_log(args) -> int:
    """Query structured tool dispatch logs."""
    subcommands = {
        "failure-rate": cmd_tool_log_failure_rate,
        "recent-failures": cmd_tool_log_recent_failures,
    }
    handler = subcommands.get(args.tool_log_command)
    if handler is None:
        print("Usage: kestrel tool-dispatches {failure-rate,recent-failures} ...")
        return 1
    return handler(args)


def _resolve_agent_db_path(agent: str, db_path: Optional[str] = None) -> Path:
    if db_path:
        return Path(db_path).expanduser().resolve()

    project_dir = _get_project_dir()
    multi_agent = MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    local_agents = multi_agent.get_local_agents()
    if agent not in local_agents:
        available = ", ".join(sorted(local_agents)) or "(none)"
        raise ValueError(f"Agent '{agent}' not found. Available agents: {available}")
    return (project_dir / local_agents[agent].data_dir / "kestrel_prime.db").resolve()


async def _query_tool_dispatches(args, query_name: str):
    from kestrel_sovereign.a2a.stores.observability_store import (
        SQLiteObservabilityStore,
    )

    db_path = _resolve_agent_db_path(args.agent, getattr(args, "db_path", None))
    store = SQLiteObservabilityStore(str(db_path))
    await store.initialize()
    try:
        agent_did = args.agent_did or await _resolve_agent_did_from_db(
            store.backend, args.agent
        )
        if query_name == "failure-rate":
            return await store.tool_failure_rate(agent_did, args.last_n_turns)
        return await store.recent_failures(agent_did, args.limit)
    finally:
        await store.close()


async def _resolve_agent_did_from_db(backend, agent_name: str) -> str:
    row = await backend.fetch_one(
        """
        SELECT node_id
        FROM graph_nodes
        WHERE node_type = 'agent'
          AND (label = ? OR json_extract(properties, '$.agent_name') = ?)
        ORDER BY node_id
        LIMIT 1
        """,
        (agent_name, agent_name),
    )
    if row and row[0]:
        return row[0]
    raise ValueError(
        f"Could not resolve DID for agent '{agent_name}'. "
        "Pass --agent-did explicitly."
    )


def cmd_tool_log_failure_rate(args) -> int:
    try:
        result = asyncio.run(_query_tool_dispatches(args, "failure-rate"))
    except Exception as exc:
        print(f"Error querying tool dispatches: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"Agent {result['agent_did']}: {result['failure_calls']}/"
            f"{result['total_calls']} non-success calls over "
            f"{result['turns_observed']} turns "
            f"({result['failure_rate']:.1%})"
        )
        for row in result["dominant_failures"]:
            print(
                f"{row['count']:4d} {row['rate']:.1%} "
                f"{row['tool_name']} / {row['error_class']}"
            )
    return 0


def cmd_tool_log_recent_failures(args) -> int:
    try:
        result = asyncio.run(_query_tool_dispatches(args, "recent-failures"))
    except Exception as exc:
        print(f"Error querying tool dispatches: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for row in result:
            print(
                f"#{row['id']} {row['ts']} {row['tool_name']} "
                f"{row['result_status']} {row['error_class'] or ''}: "
                f"{row['error_message'] or ''}"
            )
    return 0


def cmd_storage_health(args) -> int:
    """Show backup tier health with honest tier labels."""
    from datetime import timedelta

    from kestrel_sovereign.storage.sync.health import (
        build_storage_health_report,
        load_env_file,
    )

    project_dir = _get_project_dir()
    env = load_env_file(project_dir / ".env")
    report = asyncio.run(
        build_storage_health_report(
            agent_id=args.agent_id,
            env=env,
            lighthouse_grace=timedelta(hours=args.lighthouse_grace_hours),
            gcs_prefix=args.gcs_prefix,
        )
    )
    data = report.to_dict()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"Storage health: {report.status}")
        for target in (report.sovereign_ipfs, report.lighthouse, report.gcs):
            configured = "configured" if target.configured else "not configured"
            print(
                f"  {target.name:15} {target.status:14} "
                f"{configured} [{target.label}]"
            )
            print(f"    {target.message}")
            if target.details:
                cid = target.details.get("cid")
                if cid:
                    print(f"    cid: {cid}")
                if target.name == "sovereign_ipfs":
                    api_url = target.details.get("api_url")
                    if api_url:
                        print(f"    api: {api_url}")
                if target.name == "lighthouse":
                    deal_count = target.details.get("deal_count")
                    age_seconds = target.details.get("age_seconds")
                    if age_seconds is not None:
                        print(f"    age: {age_seconds // 3600}h")
                    if deal_count is not None:
                        print(f"    deals: {deal_count}")
                if target.name == "gcs":
                    print(f"    bucket: {target.details.get('bucket')}")
                    print(f"    latest: {target.details.get('latest_blob')}")
    return 1 if report.status == "warning" else 0


def cmd_constitution(args) -> int:
    """Dispatch ``kestrel constitution`` subcommands."""
    constitution_commands = {
        "reanchor": cmd_constitution_reanchor,
        "anchor-overlay": cmd_constitution_anchor_overlay,
    }
    handler = constitution_commands.get(args.constitution_command)
    if handler is None:
        print("Usage: kestrel constitution {reanchor|anchor-overlay}")
        return 1
    return handler(args)


def cmd_constitution_reanchor(args) -> int:
    """Reanchor an agent to the current canonical constitution."""
    import asyncio

    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.multi_agent.config import (
        MULTI_AGENT_CONFIG_FILENAME, MultiAgentConfig,
    )
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    project_dir = _get_project_dir()

    # Load the *target home's* environment before anything resolves a
    # database. The agent's backend, its DSN, and its data key all live in
    # that .env; without this the reanchor target is decided by whatever the
    # operator happens to have exported, so the same command against the same
    # agent could write PostgreSQL from one shell and the local anchor from
    # another — and on a PostgreSQL host the anchor is a database the runtime
    # never reads (#2890). ``setdefault`` semantics keep a genuinely-exported
    # value authoritative.
    #
    # NOT ``_apply_target_data_key_custody``: that generates a data key when
    # the home has none, which is an inception concern and wrong for a repair.
    # The trade-off is real — this seeds a persisted ``KESTREL_DATA_KEY``
    # without detecting an exported⇄persisted conflict (#2468), so an operator
    # with a stale exported key can still write a constitution blob the agent
    # cannot decrypt. That hazard predates this change; what changes is that
    # the key now gets loaded at all.
    load_project_env(project_dir)

    # ...but the *database target* is resolved separately, with the launcher's
    # precedence, where the project ``.env`` outranks a conflicting export.
    # ``load_project_env``'s ``setdefault`` is the opposite, and the two
    # disagreeing is not academic: ``kestrel doctor`` reads the database the
    # agents will actually open (``paths.spawned_agent_env``, the body of
    # ``ProcessManager._load_env``). With a stale DSN exported in the shell,
    # doctor would report drift in database A while the repair it prescribes
    # rewrote database B — leaving the finding standing and modifying a copy
    # nobody reads. A finding and its remedy have to mean the same database.
    #
    # Targeting somewhere else deliberately is still possible; it just has to
    # be deliberate, which is what these explicit arguments are.
    launch_env = spawned_agent_env(project_dir)
    runtime_backend = launch_env.get("KESTREL_DB_BACKEND")
    runtime_dsn = launch_env.get("KESTREL_DATABASE_URL")

    multi_agent = MultiAgentConfig.load(
        project_dir / MULTI_AGENT_CONFIG_FILENAME, auto_discover_fallback=False,
    )
    agents = multi_agent.get_local_agents()

    if args.agent_name not in agents:
        print(
            f"error: '{args.agent_name}' not in multi_agent. "
            f"Available: {', '.join(agents.keys()) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    agent_dir = (project_dir / agents[args.agent_name].data_dir).resolve()
    canonical = Path(args.constitution_path or CONSTITUTION_PATH)

    # Pre-flight check: agent must not be running. SQLite WAL locking
    # would corrupt mid-write. We check the multi_agent's PID file rather
    # than probing the network — same source-of-truth as `kestrel stop`.
    if _agent_appears_running(project_dir, args.agent_name, agents[args.agent_name]):
        print(
            f"error: agent '{args.agent_name}' appears to be running. "
            f"Run `kestrel stop {args.agent_name}` first to avoid DB corruption.",
            file=sys.stderr,
        )
        return 2

    result = asyncio.run(
        reanchor_constitution(
            agent_name=args.agent_name,
            agent_dir=agent_dir,
            canonical_path=canonical,
            force=args.force,
            authorization=f"kestrel constitution reanchor (cli, {args.agent_name})",
            kestrel_toml_path=project_dir / "kestrel.toml",
            amendment_artifact_path=(
                Path(args.signed_artifact) if args.signed_artifact else None
            ),
            sovereign_trust_root_path=(
                Path(args.trust_root) if args.trust_root else None
            ),
            runtime_backend=runtime_backend,
            runtime_dsn=runtime_dsn,
        )
    )

    # Every outcome names the database it describes. A reanchor that does not
    # say where it read or wrote is the ambiguity #2890 is about.
    target = f"  Target:  {result.target_label}"
    if result.backup_path is not None:
        backup_line = f"  Backup:  {result.backup_path}"
    else:
        backup_line = f"  Backup:  {result.backup_unavailable_reason or '(none)'}"
    if result.target_backend == "sqlite":
        planned_backup = (
            f"The DB will be backed up to "
            f"{result.db_path.name}.backup-<timestamp> before any write."
        )
    else:
        planned_backup = (
            f"Governance for this agent lives in {result.target_label}; there "
            f"is no local file to back up. Snapshot that database first if you "
            f"want to be able to undo a successful reanchor."
        )

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    if result.unchanged:
        print(
            f"{result.agent_name}: already anchored to current constitution "
            f"({result.new_hash[:12]}…) — nothing to do.\n"
            f"{target}"
        )
        return 0
    if result.drift_unforced:
        if result.governance_edge_drift and result.old_hash == result.new_hash:
            # Edge-only drift (#2616): the hash is current but the
            # governed_by edge set is inconsistent — integrity proof 2
            # will safe-mode the agent at boot.
            stale = ", ".join(f"{t[:12]}…" for t in result.stale_edge_targets)
            print(
                f"{result.agent_name}: governance-edge drift detected.\n"
                f"  Anchor: {result.new_hash[:12]}… (current)\n"
                f"  governed_by edge(s): {stale or '(none)'}\n"
                f"{target}\n"
                f"\n"
                f"The fail-closed integrity audit (proof 2) will safe-mode "
                f"this agent at next boot. Re-run with --force to repair the "
                f"governance edge. {planned_backup}"
            )
            return 1
        print(
            f"{result.agent_name}: constitution drift detected.\n"
            f"  Stored: {result.old_hash[:12]}…\n"
            f"  File:   {result.new_hash[:12]}… ({result.canonical_path})\n"
            f"{target}\n"
            f"\n"
            f"Re-run with --force to update the agent's anchor. {planned_backup}"
        )
        return 1
    # Reanchored (or same-hash governance repair).
    if result.governance_edge_drift and result.old_hash == result.new_hash:
        print(
            f"{result.agent_name}: governance edge repaired.\n"
            f"  Anchor:  {result.new_hash[:12]}… (unchanged)\n"
            f"  Removed: {', '.join(f'{t[:12]}…' for t in result.stale_edge_targets) or '(none)'}\n"
            f"  Source:  {result.canonical_path}\n"
            f"{target}\n"
            f"{backup_line}"
        )
        return 0
    print(
        f"{result.agent_name}: reanchored.\n"
        f"  Old: {result.old_hash[:12]}…\n"
        f"  New: {result.new_hash[:12]}…\n"
        f"  Source:  {result.canonical_path}\n"
        f"{target}\n"
        f"{backup_line}"
    )
    return 0


def cmd_constitution_anchor_overlay(args) -> int:
    """Anchor an agent's per-agent CONSTITUTION.md overlay (#1722).

    Establishes trust in the overlay so ComputerUseFeature honors its
    Amendment IX grants. Refuses to run against a live agent (SQLite WAL safety),
    same as ``reanchor``."""
    import asyncio

    from kestrel_sovereign.multi_agent.config import (
        MULTI_AGENT_CONFIG_FILENAME, MultiAgentConfig,
    )
    from kestrel_sovereign.setup.overlay_anchor import anchor_overlay

    project_dir = _get_project_dir()
    # Same rule as reanchor (#2890): the agent's own .env decides which
    # database this anchor lands in. The overlay hash is read by the runtime,
    # so writing the local file on a PostgreSQL host leaves every Amendment IX
    # grant in the overlay permanently denied while reporting success.
    load_project_env(project_dir)

    multi_agent = MultiAgentConfig.load(
        project_dir / MULTI_AGENT_CONFIG_FILENAME, auto_discover_fallback=False,
    )
    agents = multi_agent.get_local_agents()

    if args.agent_name not in agents:
        print(
            f"error: '{args.agent_name}' not in multi_agent. "
            f"Available: {', '.join(agents.keys()) or '(none)'}",
            file=sys.stderr,
        )
        return 2

    if _agent_appears_running(project_dir, args.agent_name, agents[args.agent_name]):
        print(
            f"error: agent '{args.agent_name}' appears to be running. "
            f"Run `kestrel stop {args.agent_name}` first to avoid DB corruption.",
            file=sys.stderr,
        )
        return 2

    agent_dir = (project_dir / agents[args.agent_name].data_dir).resolve()
    result = asyncio.run(
        anchor_overlay(agent_name=args.agent_name, agent_dir=agent_dir)
    )

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    if result.unchanged:
        print(
            f"{result.agent_name}: overlay already anchored "
            f"({result.new_hash[:12]}…) — nothing to do.\n"
            f"  Target:  {result.target_label}"
        )
        return 0
    print(
        f"{result.agent_name}: constitution overlay anchored.\n"
        f"  Old: {(result.old_hash[:12] + '…') if result.old_hash else '(none)'}\n"
        f"  New: {result.new_hash[:12]}…\n"
        f"  Overlay: {result.overlay_path}\n"
        f"  Target:  {result.target_label}"
    )
    return 0


def cmd_migrate_llm_config(args) -> int:
    """One-shot: merge legacy ``llm_config.toml`` into ``kestrel.toml [llm]``."""
    from kestrel_sovereign.setup.migrate_llm_config import migrate_llm_config

    project_dir = (
        Path(args.project_dir).resolve()
        if args.project_dir
        else _get_project_dir()
    )

    result = migrate_llm_config(project_dir, force=args.force)

    if result.action == "no_source":
        print(
            f"Nothing to migrate: {result.source_path} does not exist.\n"
            f"Configure LLM providers in {result.kestrel_toml_path} under "
            f"the [llm] section instead."
        )
        return 0

    if result.action == "parse_error":
        print(
            f"error: {result.source_path.name} is not valid TOML.\n"
            f"  {result.error}\n"
            f"\n"
            f"The source file has NOT been touched. Fix the syntax error "
            f"and re-run `kestrel migrate-llm-config`.",
            file=sys.stderr,
        )
        return 1

    if result.action == "diverged":
        print(
            f"error: kestrel.toml [llm] differs from {result.source_path.name}.\n"
            f"\n{result.diff}\n"
            f"Re-run with --force to let llm_config.toml win, or hand-edit "
            f"kestrel.toml first.",
            file=sys.stderr,
        )
        return 1

    if result.action == "already_clean":
        print(
            f"{result.kestrel_toml_path.name} [llm] already matches "
            f"{result.source_path.name}. Renamed source to "
            f"{result.bak_path.name}; nothing else to do."
        )
        return 0

    # action == "migrated"
    backup_msg = (
        f"  Backup of prior kestrel.toml: {result.backup_path.name}\n"
        if result.backup_path
        else "  (no prior kestrel.toml; created fresh)\n"
    )
    print(
        f"Migrated {result.source_path.name} -> {result.kestrel_toml_path.name} [llm].\n"
        f"  Source renamed to: {result.bak_path.name}\n"
        f"{backup_msg}"
        f"You can now remove {result.bak_path.name} once you've confirmed "
        f"the agent loads correctly."
    )
    return 0


def cmd_migrate_config(args) -> int:
    """Merge legacy model config files into unified ``kestrel.toml``."""
    from kestrel_sovereign.setup.migrate_config import migrate_config

    project_dir = (
        Path(args.project_dir).resolve()
        if args.project_dir
        else _get_project_dir()
    )

    result = migrate_config(project_dir)

    if result.action == "parse_error":
        errors = [
            (
                f"  {source.source_path.name}: {source.error}"
                if source.section_path == "destination"
                else (
                    f"  {source.source_path.name} -> "
                    f"[{source.section_path}]: {source.error}"
                )
            )
            for source in result.sources
            if source.action == "parse_error"
        ]
        print(
            "error: one or more model config files are not valid TOML.\n"
            + "\n".join(errors)
            + "\n\nNo files were changed. Fix the syntax error and re-run "
            "`kestrel migrate-config`.",
            file=sys.stderr,
        )
        return 1

    migrated = [
        source for source in result.sources if source.action == "migrated"
    ]
    already_clean = [
        source for source in result.sources if source.action == "already_clean"
    ]
    missing = [
        source for source in result.sources if source.action == "no_source"
    ]

    if result.action == "migrated":
        lines = [
            f"Migrated legacy model config into {result.kestrel_toml_path.name}:"
        ]
        lines.extend(
            f"  {source.source_path.name} -> [{source.section_path}]"
            for source in migrated
        )
        lines.extend(
            f"  skipped [{source.section_path}] (already present)"
            for source in already_clean
        )
        lines.extend(
            f"  skipped {source.source_path.name} (not found)"
            for source in missing
        )
        lines.append(
            f"  Backup of prior kestrel.toml: {result.backup_path.name}"
            if result.backup_path
            else "  (no prior kestrel.toml; created fresh)"
        )
        print("\n".join(lines))
        return 0

    if result.action == "already_clean":
        print(
            f"{result.kestrel_toml_path.name} already has the unified model "
            "config sections; nothing to migrate."
        )
        return 0

    print(
        "Nothing to migrate: model_mandate.toml and model_catalog.toml were "
        f"not found. Configure them in {result.kestrel_toml_path.name} under "
        "[llm.mandate] and [llm.catalog]."
    )
    return 0


def cmd_migrate_encryption(args) -> int:
    """One-shot: encrypt pre-migration plaintext rows at rest (#1401).

    Thin wrapper — the real logic lives in
    ``kestrel_sovereign.security.encryption_backfill.cli_run`` so the
    test suite can drive it without paying the full ``cli.py`` import
    cost (which transitively pulls KestrelAgent + LLM stack and trips
    on in-flight sibling work like ``ProviderCapabilities``).
    """
    from kestrel_sovereign.security.encryption_backfill import cli_run
    return cli_run(args)


def _agent_appears_running(project_dir, agent_name, agent_cfg) -> bool:
    """Best-effort check that the agent process isn't holding the DB."""
    try:
        from kestrel_sovereign.multi_agent.process_manager import ProcessManager

        resolved_dir = (project_dir / agent_cfg.data_dir).resolve()
        pid_file = ProcessManager.agent_pid_file(resolved_dir)
        pid = ProcessManager.read_pid(pid_file)
        if pid is None:
            return False
        return ProcessManager.is_process_running(pid)
    except Exception:
        # If we can't tell, err on the side of letting the user proceed
        # — they get a clear error from the storage layer if it's locked.
        return False


def cmd_setup(args) -> int:
    """Run the setup wizard."""
    from kestrel_sovereign.setup.context import Flow
    from kestrel_sovereign.setup.wizard import build_context, run_wizard

    if args.check and args.reset:
        # --check is read-only by contract; --reset writes (moves files
        # aside). Combining them silently moved .env / kestrel.toml to
        # backups before the check phase ever ran. Reject the combo
        # rather than picking a winner.
        print(
            "error: --check and --reset are mutually exclusive. "
            "--check is read-only; --reset moves files to backups.",
            file=sys.stderr,
        )
        return 2

    project_dir = _get_project_dir()

    if args.check:
        flow = Flow.CHECK
    elif args.quickstart:
        flow = Flow.QUICKSTART
    else:
        flow = Flow.INTERACTIVE

    is_test_instance = bool(args.test) or os.environ.get(
        "KESTREL_TEST_INSTANCE", ""
    ).lower() in ("1", "true", "yes")

    ctx = build_context(
        project_dir,
        flow=flow,
        reset=args.reset,
        is_test_instance=is_test_instance,
    )

    # Load the target home's *non-key* environment before the wizard runs so an
    # existing home's config (e.g. KESTREL_DID_WEB_DOMAIN=agents.example.com) is
    # honored instead of being ignored and defaulted to localhost (#2468).
    # Deliberately excludes KESTREL_DATA_KEY — its custody is resolved by the
    # keys/agent step so an exported⇄persisted conflict is still detected — and
    # is skipped entirely under --reset (which regenerates config from scratch)
    # and CHECK (read-only). Inception itself no longer loads .env at import
    # time, so this is the sole target-aware load for the wizard path.
    if not args.reset and flow is not Flow.CHECK:
        from kestrel_sovereign.setup.steps.keys import DATA_KEY_ENV

        load_project_env(project_dir, exclude=(DATA_KEY_ENV,))

    return run_wizard(
        ctx,
        only_step=args.step,
        core_only=getattr(args, "core_only", False),
    )


def _maybe_first_run_setup(project_dir: Path) -> Optional[int]:
    """If this looks like a truly fresh checkout, offer to run setup.

    Fires only when **all** of these are true:

      1. We are NOT inside a git worktree (worktrees never carry the
         user's gitignored state — ``.env``, ``multi_agent.toml`` — and the
         hook would always misfire there). Detected by ``.git`` being a
         FILE (gitdir pointer) rather than a directory.
      2. ``.env`` is absent, AND
      3. There are no agents registered in the multi_agent.

    A user who has already inceptioned an agent (``kestrel create``) has
    done deliberate setup; we must not block ``kestrel start`` for them
    even if their ``.env`` is missing — inception falls back to plaintext
    key storage when ``KESTREL_DATA_KEY`` is unset, and the CI clean-
    install workflow exercises exactly that path.

    Returns:
        ``None`` to proceed with start as normal.
        ``0`` if the user just finished setup successfully — caller should
            re-read multi_agent and continue starting.
        Non-zero if the user declined / setup failed — caller exits.

    Honors ``KESTREL_SKIP_FIRST_RUN=1`` to bypass entirely.
    """
    if os.environ.get("KESTREL_SKIP_FIRST_RUN", "").lower() in ("1", "true", "yes"):
        return None

    # Worktree detection: the user's deliberate state (.env, multi_agent.toml)
    # lives in the main checkout, not here. A worktree's .git is a file
    # containing a `gitdir:` pointer; a main checkout's .git is a directory.
    # Without this guard, running any `kestrel` command from inside a
    # worktree (a normal dev workflow) misfires the wizard on real users
    # who already have a valid setup at the main checkout.
    git_marker = project_dir / ".git"
    if git_marker.is_file():
        return None

    env_path = project_dir / ".env"
    if env_path.exists():
        return None

    # Only treat a missing .env as "fresh checkout" if no agents exist.
    multi_agent_path = project_dir / MULTI_AGENT_CONFIG_FILENAME
    if multi_agent_path.exists():
        try:
            existing_multi_agent = MultiAgentConfig.load(
                multi_agent_path, auto_discover_fallback=False
            )
            if existing_multi_agent.get_local_agents():
                return None
        except Exception:
            # If multi_agent parsing fails, fall through to the prompt path.
            pass

    from kestrel_sovereign.setup.prompts import is_tty

    if not is_tty():
        print("No .env found. Run: kestrel setup --quickstart")
        return 1

    print("No .env found — looks like a fresh checkout.")
    try:
        answer = input("Run `kestrel setup` now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    if answer in ("", "y", "yes"):
        from kestrel_sovereign.setup.context import Flow
        from kestrel_sovereign.setup.wizard import build_context, run_wizard

        ctx = build_context(project_dir, flow=Flow.INTERACTIVE, reset=False)
        rc = run_wizard(ctx)
        if rc != 0:
            return rc
        return None  # Proceed with start
    print("Skipped. Re-run later with: kestrel setup")
    return 1


# ---------------------------------------------------------------------------
# Feature CLI commands
# ---------------------------------------------------------------------------

def _load_kestrel_toml(project_dir: Path) -> dict:
    """Load kestrel.toml from project dir. Returns empty dict if not found."""
    import toml
    toml_path = project_dir / "kestrel.toml"
    if not toml_path.exists():
        return {}
    try:
        return toml.load(toml_path)
    except Exception:
        return {}


def _save_kestrel_toml(project_dir: Path, data: dict) -> None:
    """Save data to kestrel.toml in project dir, preserving existing content."""
    import toml
    toml_path = project_dir / "kestrel.toml"

    # Load existing content and merge
    existing = _load_kestrel_toml(project_dir)
    existing.update(data)

    with open(toml_path, "w", encoding="utf-8") as f:
        toml.dump(existing, f)


def _get_toml_disabled_features(project_dir: Path) -> list:
    """Read disabled features list from kestrel.toml [features] section."""
    data = _load_kestrel_toml(project_dir)
    return data.get("features", {}).get("disabled", [])


def _set_toml_disabled_features(project_dir: Path, disabled: list) -> None:
    """Write disabled features list to kestrel.toml [features] section."""
    data = _load_kestrel_toml(project_dir)
    if "features" not in data:
        data["features"] = {}
    data["features"]["disabled"] = sorted(set(disabled))
    _save_kestrel_toml(project_dir, data)


def cmd_skills(args) -> int:
    """Dispatch skills subcommands."""
    skills_commands = {
        "search": cmd_skills_search,
    }

    handler = skills_commands.get(args.skills_command)
    if handler is None:
        print("Usage: kestrel skills {search}")
        return 1
    return handler(args)


def cmd_skills_search(args) -> int:
    """Search all skills by name or tag."""
    from kestrel_sovereign.feature_registry import load_registry

    query = args.query.lower()
    registry = load_registry()

    matches = []
    for pkg_name, info in registry.items():
        for skill in info.skills:
            if (query in skill.name.lower()
                    or query in skill.description.lower()
                    or query in skill.category.lower()
                    or any(query in t.lower() for t in skill.tags)):
                matches.append((pkg_name, skill))

    if not matches:
        print(f"No skills matching '{args.query}'")
        return 0

    print()
    print(f"  {'SKILL':<24} {'PACKAGE':<16} {'DESCRIPTION'}")
    print(f"  {'─' * 60}")
    for pkg_name, skill in matches:
        print(f"  {skill.name:<24} {pkg_name:<16} {skill.description}")
    print()
    return 0


def cmd_config(args) -> int:
    """Show or edit agent config."""
    from kestrel_sovereign.agent_config import AgentConfig, find_agent_dir

    agent_dir = find_agent_dir(args.agent_dir)
    if not agent_dir:
        print("No agent directory found")
        return 1

    config = AgentConfig.from_directory(agent_dir)

    if args.set_port:
        config.port = args.set_port
        config.save()
        print(f"Port set to {args.set_port}")

    if args.set_name:
        config.name = args.set_name
        config.save()
        print(f"Name set to {args.set_name}")

    if args.init:
        config.save()
        print(f"Config created: {config.config_file}")

    # Show current config
    print(f"\n{config.name}")
    print(f"   Directory: {config.agent_dir}")
    print(f"   Port:      {config.port}")
    print(f"   Host:      {config.host}")
    print(f"   Config:    {config.config_file}")
    print(f"   Exists:    {'Yes' if config.config_file.exists() else 'No'}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Lifecycle commands live in cli_lifecycle.py (#1678); re-export the public
# handlers + helpers the dispatch table and test suite resolve via
# `kestrel_sovereign.cli.<name>` (patched git/uv update helpers included).
from kestrel_sovereign.cli_lifecycle import (  # noqa: E402
    cmd_start,
    cmd_stop,
    cmd_restart,
    cmd_update,
    cmd_status,
    cmd_logs,
    _format_uptime,
    _host_pid_file,
    _host_log_file,
    _query_agents_api,
    _reap_orphans_on_port,
    _start_inprocess_mode,
    _resolve_source_checkout,
    _project_dir_is_git,
    _git_working_tree_dirty,
    _run_git_pull,
    _git_reattach_if_safely_detached,
    _run_uv_sync,
    _run_uv_pip_install_editable,
    _GitFailedError,
)


# Feature commands live in cli_features.py (#1678); re-export the public
# handlers + _resolve_feature_name so `from kestrel_sovereign.cli import ...`
# call sites (the dispatch table below and the test suite) keep resolving.
from kestrel_sovereign.cli_features import (  # noqa: E402
    cmd_feature,
    cmd_feature_list,
    cmd_feature_install,
    cmd_feature_upgrade,
    cmd_feature_sync,
    cmd_feature_status,
    cmd_feature_enable,
    cmd_feature_disable,
    cmd_feature_info,
    cmd_feature_scaffold,
    cmd_feature_skills,
    _resolve_feature_name,
    # Feature-domain helpers exercised directly (or patched) by the test
    # suite via ``kestrel_sovereign.cli.<name>`` — re-exported so those seams
    # keep resolving. The two patched-and-invoked-in-command helpers
    # (_extension_install_run, _query_agent_feature_catalog) are called back
    # through the ``cli`` module inside cli_features so the patch takes effect.
    _extension_install_run,
    _query_agent_feature_catalog,
    _editable_install_path,
    _installed_extension_distributions,
    _registry_info_for,
    _load_host_manifest,
    _host_manifest_path,
    _parse_pip_installed_version,
    _pip_spec,
    _toml_basic_string,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the unified CLI."""
    parser = argparse.ArgumentParser(
        prog="kestrel",
        description="Kestrel Sovereign Agent Manager",
    )
    parser.add_argument(
        "--version", action="version", version=f"kestrel {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # kestrel embeddings audit | reindex  (#1477)
    from kestrel_sovereign.cli_embeddings import add_embeddings_subparser
    add_embeddings_subparser(subparsers)

    # kestrel serve list|up|down|switch|status  (local model servers)
    from kestrel_sovereign.cli_serve import add_serve_subparser
    add_serve_subparser(subparsers)

    # kestrel start|stop|restart|update|status|logs
    from kestrel_sovereign.cli_lifecycle import add_lifecycle_subparsers
    add_lifecycle_subparsers(subparsers)

    # kestrel list
    subparsers.add_parser("list", help="List all agents in multi_agent")

    # kestrel create <name>
    create_p = subparsers.add_parser("create", help="Create a new agent")
    create_p.add_argument("name", help="Agent name")
    create_p.add_argument("--port", type=int, help="Override port assignment")
    create_p.add_argument(
        "--test",
        action="store_true",
        help="Tag the inceptioned agent as a test instance "
        "(is_test_instance=True + auto test_cycle_id), so it can be "
        "graduated/retired later. Ignored if the agent already exists.",
    )

    # kestrel shell <name>
    shell_p = subparsers.add_parser("shell", help="Interactive CLI chat")
    shell_p.add_argument("name", help="Agent name")
    shell_p.add_argument(
        "--app", type=str, default=None, choices=["elderly"],
        help="Load an application extension",
    )

    # kestrel ask <name> "<message>" — non-interactive one-shot to a
    # RUNNING agent (no in-process boot; scriptable). See #1287.
    ask_p = subparsers.add_parser(
        "ask",
        help="Send one message to a running agent and print its reply "
             "(non-interactive)",
    )
    ask_p.add_argument("name", help="Agent name")
    ask_p.add_argument("message", help="Message to send")
    ask_p.add_argument(
        "--session", type=str, default=None,
        help="Session id to thread continuity across calls",
    )
    ask_p.add_argument(
        "--json", action="store_true",
        help="Emit the full {response, session_id} envelope as JSON",
    )
    ask_p.add_argument(
        "--timeout", type=float, default=None, metavar="SECONDS",
        help="Read timeout for the agent's reply (default 1800, or "
             "KESTREL_ASK_TIMEOUT_SECONDS); <=0 waits indefinitely. "
             "Connect stays fast so a dead server fails immediately.",
    )

    # kestrel health (deprecated alias for doctor)
    subparsers.add_parser("health", help="(deprecated) alias for `kestrel doctor`")

    # kestrel doctor
    subparsers.add_parser("doctor", help="Diagnose readiness; no changes")

    # kestrel identity harden-exports
    identity_p = subparsers.add_parser(
        "identity",
        help="Maintain private local identity-export custody",
    )
    identity_sub = identity_p.add_subparsers(dest="identity_command")
    identity_sub.add_parser(
        "harden-exports",
        help="Set eligible legacy identity export directories/files to 0700/0600",
    )

    # kestrel storage {health}
    storage_p = subparsers.add_parser(
        "storage", help="Storage health and operations"
    )
    storage_sub = storage_p.add_subparsers(dest="storage_command")
    storage_health_p = storage_sub.add_parser(
        "health",
        help="Check Lighthouse deal lag and GCS fallback readiness",
    )
    storage_health_p.add_argument(
        "--agent-id",
        default="default",
        help="Agent identifier used by sync targets (default: default)",
    )
    storage_health_p.add_argument(
        "--lighthouse-grace-hours",
        type=float,
        default=24.0,
        help="Hours before no-deal Lighthouse snapshots warn (default: 24)",
    )
    storage_health_p.add_argument(
        "--gcs-prefix",
        default="kestrel/",
        help="GCS prefix used by GCSTarget (default: kestrel/)",
    )
    storage_health_p.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )

    # kestrel auth login --route openai:plan
    auth_p = subparsers.add_parser(
        "auth", help="Interactive authentication helpers"
    )
    auth_sub = auth_p.add_subparsers(dest="auth_command")
    auth_login_p = auth_sub.add_parser(
        "login", help="Run an interactive provider login flow"
    )
    auth_login_p.add_argument(
        "--route",
        required=True,
        help="Provider route to authenticate (currently: openai:plan)",
    )

    # kestrel tool-dispatches {failure-rate,recent-failures}
    tool_log_p = subparsers.add_parser(
        "tool-dispatches",
        aliases=["tool-log"],
        help="Query structured tool/subagent dispatch logs",
    )
    tool_log_sub = tool_log_p.add_subparsers(dest="tool_log_command")
    failure_rate_p = tool_log_sub.add_parser(
        "failure-rate",
        help="Summarize per-tool failure rates over the last N turns",
    )
    failure_rate_p.add_argument("agent", help="Agent name from multi_agent.toml")
    failure_rate_p.add_argument(
        "--agent-did",
        help="Agent DID stored in a2a_tool_dispatches.agent_did",
    )
    failure_rate_p.add_argument(
        "--last-n-turns", type=int, default=100, help="Turn window (default: 100)"
    )
    failure_rate_p.add_argument("--db-path", default=None, help="Override DB path")
    failure_rate_p.add_argument("--json", action="store_true")

    recent_failures_p = tool_log_sub.add_parser(
        "recent-failures", help="Show recent non-success tool call rows"
    )
    recent_failures_p.add_argument("agent", help="Agent name from multi_agent.toml")
    recent_failures_p.add_argument(
        "--agent-did",
        help="Agent DID stored in a2a_tool_dispatches.agent_did",
    )
    recent_failures_p.add_argument(
        "--limit", type=int, default=20, help="Maximum rows (default: 20)"
    )
    recent_failures_p.add_argument("--db-path", default=None, help="Override DB path")
    recent_failures_p.add_argument("--json", action="store_true")

    # kestrel setup [step]
    setup_p = subparsers.add_parser(
        "setup", help="Run the setup wizard (idempotent, re-runnable)"
    )
    setup_p.add_argument(
        "step", nargs="?",
        help="Run only this step (default: run all in order). "
             "Installed feature steps and optional steps only run when named "
             "explicitly.",
    )
    setup_p.add_argument(
        "--quickstart", action="store_true",
        help="Accept defaults; only prompt for missing secrets",
    )
    setup_p.add_argument(
        "--check", action="store_true",
        help="Report readiness only, never write or prompt",
    )
    setup_p.add_argument(
        "--core-only", action="store_true",
        help="Run only built-in setup/recovery steps without loading providers",
    )
    setup_p.add_argument(
        "--reset", action="store_true",
        help="Move existing .env and kestrel.toml aside before regenerating",
    )
    setup_p.add_argument(
        "--test", action="store_true",
        help="Mark the inceptioned agent as a test instance "
             "(is_test_instance=True on the agent's properties node, "
             "auto-generated test_cycle_id). Honoured automatically when "
             "KESTREL_TEST_INSTANCE=1 is set in the environment — useful for "
             "CI runners that want every agent they incept tagged as a test.",
    )

    # kestrel migrate-encryption — backfill plaintext rows at rest (#1401)
    migrate_enc_p = subparsers.add_parser(
        "migrate-encryption",
        help="One-shot: encrypt pre-migration plaintext rows at rest "
             "in conversation_history + files (#1401)",
    )
    migrate_enc_p.add_argument(
        "--data-dir", required=True,
        help="Agent data directory containing kestrel_prime.db "
             "(e.g. agent_data/meridian).",
    )
    migrate_enc_p.add_argument(
        "--agent-id", default=None,
        help="Agent DID to scope conversation_history backfill. "
             "Defaults to the DID stored in graph_nodes.",
    )
    migrate_enc_p.add_argument(
        "--dry-run", action="store_true",
        help="Report counts without writing. Safe to run on a live DB.",
    )

    # kestrel migrate-llm-config
    migrate_llm_p = subparsers.add_parser(
        "migrate-llm-config",
        help="One-shot: merge legacy llm_config.toml into kestrel.toml [llm]",
    )
    migrate_llm_p.add_argument(
        "--force", action="store_true",
        help="Overwrite kestrel.toml [llm] when it differs from llm_config.toml. "
             "Without --force, divergent content is reported and the file is left alone.",
    )
    migrate_llm_p.add_argument(
        "--project-dir", default=None,
        help="Project root containing llm_config.toml and kestrel.toml "
             "(defaults to the Kestrel repo root).",
    )

    # kestrel migrate-config
    migrate_config_p = subparsers.add_parser(
        "migrate-config",
        help="One-shot: merge legacy model_mandate.toml/model_catalog.toml "
             "into kestrel.toml [llm.mandate]/[llm.catalog]",
    )
    migrate_config_p.add_argument(
        "--project-dir", default=None,
        help="Project root containing legacy model config files and "
             "kestrel.toml (defaults to the Kestrel repo root).",
    )

    # kestrel constitution {reanchor}
    constitution_p = subparsers.add_parser(
        "constitution",
        help="Constitution lifecycle (reanchor an agent to the current file)",
    )
    constitution_sub = constitution_p.add_subparsers(dest="constitution_command")

    reanchor_p = constitution_sub.add_parser(
        "reanchor",
        help="Reanchor agent to the current canonical KESTREL_CONSTITUTION.md",
    )
    reanchor_p.add_argument(
        "--agent-name", required=True,
        help="Name of the agent (must be in multi_agent.toml)",
    )
    reanchor_p.add_argument(
        "--force", action="store_true",
        help="Required to actually write. Without --force, drift is reported but the DB is untouched.",
    )
    reanchor_p.add_argument(
        "--constitution-path", default=None,
        help="Override the canonical constitution path (defaults to package's KESTREL_CONSTITUTION.md)",
    )
    reanchor_p.add_argument(
        "--signed-artifact", default=None,
        help=(
            "Detached Sovereign-signed reanchor artifact. Required with "
            "--force; drift-only inspection does not require it."
        ),
    )
    reanchor_p.add_argument(
        "--trust-root", default=None,
        help=(
            "Operator-owned JSON DID document used to verify the artifact. "
            "Defaults to KESTREL_SOVEREIGN_TRUST_ROOT_PATH; conflicting "
            "sources fail closed."
        ),
    )

    anchor_overlay_p = constitution_sub.add_parser(
        "anchor-overlay",
        help="Anchor an agent's per-agent CONSTITUTION.md overlay so its Amendment IX grants are trusted",
    )
    anchor_overlay_p.add_argument(
        "--agent-name", required=True,
        help="Name of the agent (must be in multi_agent.toml)",
    )

    # kestrel config <agent_dir>
    config_p = subparsers.add_parser("config", help="Show/edit agent config")
    config_p.add_argument("agent_dir", nargs="?", help="Agent directory")
    config_p.add_argument("--init", action="store_true", help="Create kestrel.toml")
    config_p.add_argument("--set-port", type=int, help="Set port")
    config_p.add_argument("--set-name", type=str, help="Set name")

    # kestrel feature {list|install|upgrade|sync|status|enable|disable|info|scaffold|skills}
    from kestrel_sovereign.cli_features import add_feature_subparser
    add_feature_subparser(subparsers)

    # kestrel skills {search}
    skills_p = subparsers.add_parser("skills", help="Search and manage skills")
    skills_sub = skills_p.add_subparsers(dest="skills_command")

    skills_search = skills_sub.add_parser("search", help="Search skills by name/tag")
    skills_search.add_argument("query", help="Search query")

    # kestrel release {sign,verify} — Wave 5 of Quantum Hardening (#920).
    # Self-contained module; the import is local so an operator who never
    # touches release commands doesn't pay for the SLH-DSA suite import.
    from kestrel_sovereign.cli_release import add_release_subcommands
    add_release_subcommands(subparsers)

    # kestrel deploy <profile> — sub-PR 1.1 of epic #1050 (bash-to-Python
    # port). Same locality reason: keeps cloud SDK imports out of the
    # hot path for operators who just run `kestrel start`.
    from kestrel_sovereign.cli_deploy import add_deploy_subcommands
    add_deploy_subcommands(subparsers)

    # kestrel verify-install [TESTS...] — sub-PR 2.2 of epic #1050.
    # Local import for the same reason: tempdir/venv/uvicorn machinery
    # is dead weight for operators who never run install verification.
    from kestrel_sovereign.cli_verify_install import (
        add_verify_install_subcommand,
    )
    add_verify_install_subcommand(subparsers)

    # kestrel demo run <name> — sub-PR 3.1 of epic #1050 (port of
    # demos/run.sh). Local import — playwright/uvicorn machinery is
    # dead weight for operators who never run demos.
    from kestrel_sovereign.cli_demo import add_demo_subcommand
    add_demo_subcommand(subparsers)

    # kestrel agent docker {create,chat,retire} — sub-PR 3.2 of epic
    # #1050 (port of scripts/sovereign-agent.sh). Local import — docker
    # subprocess shell-out is dead weight for operators who never use
    # the Docker-isolated lifecycle.
    from kestrel_sovereign.cli_agent_docker import (
        add_agent_docker_subcommand,
    )
    add_agent_docker_subcommand(subparsers)

    # kestrel docker remote {build,run} — sub-PR 3.3 of epic #1050
    # (port of scripts/{build,run}_docker_remote.sh). Local import for
    # the same reason.
    from kestrel_sovereign.cli_docker_remote import add_docker_subcommand
    add_docker_subcommand(subparsers)

    # kestrel docker build <preset> — sub-PR 4 of epic #1050 (port of
    # scripts/docker/build_*.sh). Hangs under the same ``kestrel
    # docker`` parent as ``remote`` via the shared
    # ``get_or_create_docker_subparsers`` helper.
    from kestrel_sovereign.cli_docker_build import (
        add_docker_build_subcommand,
    )
    add_docker_build_subcommand(subparsers)

    # kestrel ipfs {build,deploy,pin} — sub-PR 4 of epic #1050 (port of
    # scripts/ipfs/{build,deploy,pin_agents}.sh). Local import — gcloud
    # / urllib / sqlite machinery is dead weight for operators who
    # never run a self-hosted IPFS node.
    from kestrel_sovereign.cli_ipfs import add_ipfs_subcommand
    add_ipfs_subcommand(subparsers)

    # kestrel runpod {deploy,status,stop,kill} — sub-PR 4 of epic #1050
    # (port of scripts/runpod/deploy_lora_trainer.sh). Local import —
    # the kestrel-cloud-runpod package may not be installed.
    from kestrel_sovereign.cli_runpod import add_runpod_subcommand
    add_runpod_subcommand(subparsers)

    # Feature-contributed CLI groups — discovered via the
    # ``kestrel_sovereign.cli`` entry-point group so a feature package can add
    # a `kestrel <feature> ...` subcommand with zero edits here (#2046). Runs
    # last so the core groups above are reserved and win any name collision.
    from kestrel_sovereign.cli_extensions import register_cli_extensions
    register_cli_extensions(subparsers)

    # kestrel help [command] — a real command, not just a flag, so users
    # aren't required to already know `-h`/`--help` syntax to ask for help.
    help_p = subparsers.add_parser("help", help="Show help for kestrel or a specific command")
    help_p.add_argument(
        "topic", nargs="?", help="Command to show help for (e.g. `kestrel help feature`)",
    )

    # Stashed so `kestrel help <topic>` can look up the topic's own parser
    # without re-walking the registration above.
    parser._kestrel_subparsers = subparsers  # type: ignore[attr-defined]

    return parser


def _ensure_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 so emoji output doesn't crash on
    Windows consoles that default to cp1252.

    Without this, ``kestrel create`` and other commands that print emoji
    raise ``UnicodeEncodeError: 'charmap' codec can't encode character ...``
    on stock Windows PowerShell / cmd. Safe no-op on Linux/macOS.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            # Older Python or already-detached stream — nothing to do.
            pass


def main() -> int:
    """Main entry point for the kestrel CLI."""
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "help":
        topic = getattr(args, "topic", None)
        subparsers = getattr(parser, "_kestrel_subparsers", None)
        topic_parser = subparsers.choices.get(topic) if subparsers and topic else None
        if topic and topic_parser is None:
            print(f"kestrel help: no such command '{topic}'\n")
            parser.print_help()
            return 1
        (topic_parser or parser).print_help()
        return 0

    from kestrel_sovereign.cli_release import cmd_release
    from kestrel_sovereign.cli_deploy import cmd_deploy
    from kestrel_sovereign.cli_verify_install import cmd_verify_install
    from kestrel_sovereign.cli_demo import cmd_demo
    from kestrel_sovereign.cli_agent_docker import cmd_agent
    from kestrel_sovereign.cli_docker_remote import cmd_docker
    from kestrel_sovereign.cli_ipfs import cmd_ipfs
    from kestrel_sovereign.cli_runpod import cmd_runpod
    from kestrel_sovereign.cli_embeddings import run as cmd_embeddings
    from kestrel_sovereign.cli_serve import run as cmd_serve

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "update": cmd_update,
        "status": cmd_status,
        "logs": cmd_logs,
        "list": cmd_list,
        "create": cmd_create,
        "shell": cmd_shell,
        "ask": cmd_ask,
        "health": cmd_health,
        "doctor": cmd_doctor,
        "identity": cmd_identity,
        "storage": cmd_storage,
        "auth": cmd_auth,
        "tool-dispatches": cmd_tool_log,
        "tool-log": cmd_tool_log,
        "setup": cmd_setup,
        "constitution": cmd_constitution,
        "migrate-llm-config": cmd_migrate_llm_config,
        "migrate-config": cmd_migrate_config,
        "migrate-encryption": cmd_migrate_encryption,
        "config": cmd_config,
        "feature": cmd_feature,
        "skills": cmd_skills,
        "release": cmd_release,
        "deploy": cmd_deploy,
        "verify-install": cmd_verify_install,
        "demo": cmd_demo,
        "agent": cmd_agent,
        "docker": cmd_docker,
        "ipfs": cmd_ipfs,
        "runpod": cmd_runpod,
        "embeddings": cmd_embeddings,
        "serve": cmd_serve,
    }

    # Core dispatch dict first; then the extension registry. Feature-contributed
    # CLI groups (discovered via the ``kestrel_sovereign.cli`` entry-point group)
    # register their handler through argparse's ``set_defaults(_handler=...)`` —
    # the same convention cli_serve / cli_embeddings use — which we drain here
    # for any command core doesn't own (#2046).
    handler = commands.get(args.command)
    if handler is None:
        handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
