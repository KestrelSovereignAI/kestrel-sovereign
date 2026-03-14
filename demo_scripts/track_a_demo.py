#!/usr/bin/env python3
"""
Kestrel Sovereign - Track A Technical Demo Script
Issue #191 / #133

Cross-platform presenter script for the 5-Act technical demo.
Press Enter between acts to advance. Each act runs API commands,
prints narration cues, and pauses for the presenter to speak.

The script handles EVERYTHING: creates a fresh demo agent, starts
the server, runs all 5 acts, and stops the server when done.

Prerequisites:
  - Ollama running (ollama serve) for Act 4 EPHEMERAL mode
  - .env file with KESTREL_API_KEY in the Kestrel project root
  - Run from the Kestrel project root directory

Usage:
  uv run python demo_scripts/track_a_demo.py
  uv run python demo_scripts/track_a_demo.py --auto-advance
  uv run python demo_scripts/track_a_demo.py --skip-setup
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ============================================================================
# Colors (ANSI — works on macOS Terminal, Windows Terminal, and pwsh)
# ============================================================================

if sys.platform == "win32":
    os.system("")  # Enable ANSI on Windows 10+

CYAN = "\033[0;36m"
DARK_CYAN = "\033[1;36m"
YELLOW = "\033[1;33m"
DARK_YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"
GRAY = "\033[0;90m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
WHITE = "\033[1;37m"
NC = "\033[0m"

# ============================================================================
# Helpers
# ============================================================================

def write_narration(text: str):
    print()
    print(f"  {DARK_CYAN}NARRATION:{NC}")
    for line in text.strip().split("\n"):
        print(f"  {CYAN}{line}{NC}")
    print()


def write_act(number: str, title: str, time_range: str):
    print()
    print(f"{YELLOW}{'=' * 70}{NC}")
    print(f"  {YELLOW}ACT {number}: {title}{NC}")
    print(f"  {DARK_YELLOW}Target: {time_range}{NC}")
    print(f"{YELLOW}{'=' * 70}{NC}")
    print()


def write_step(text: str):
    print(f"  {GREEN}>> {text}{NC}")


def write_fail(text: str):
    print(f"  {RED}[RECOVERY] {text}{NC}")


def pause_for_presenter(prompt: str = "Press ENTER to continue...", auto: bool = False):
    if not auto:
        print()
        input(f"  {MAGENTA}{prompt}{NC}")


def call_api(
    uri: str,
    method: str = "GET",
    body: str | None = None,
    headers: dict | None = None,
    timeout: int = 120,
) -> dict | list | None:
    """Make an API call. Returns parsed JSON or None on failure."""
    if headers is None:
        headers = {}
    req = urllib.request.Request(uri, method=method, headers=headers)
    if body is not None:
        req.data = body.encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data) if data.strip() else {}
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        write_fail(f"API call failed: HTTP {e.code} — {detail}")
        write_fail(f"URI: {uri}")
        return None
    except Exception as e:
        write_fail(f"API call failed: {e}")
        write_fail(f"URI: {uri}")
        return None


def pretty_json(obj, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent)


def elapsed_str(start: float) -> str:
    secs = int(time.time() - start)
    mins, secs = divmod(secs, 60)
    return f"{mins}:{secs:02d}"


# ============================================================================
# Server lifecycle
# ============================================================================

_server_proc = None


def stop_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Poll /health until the server is up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kestrel Track A Technical Demo")
    parser.add_argument("--kestrel-dir", default=".", help="Path to Kestrel project root")
    parser.add_argument("--base-url", default="http://localhost:8888", help="Kestrel server URL")
    parser.add_argument("--skip-setup", action="store_true", help="Skip agent creation + server start (assume running)")
    parser.add_argument("--auto-advance", action="store_true", help="Don't pause between acts (for CI/recording)")
    args = parser.parse_args()

    kestrel_dir = Path(args.kestrel_dir).resolve()
    base_url = args.base_url.rstrip("/")
    auto = args.auto_advance

    env_file = kestrel_dir / ".env"

    def load_api_key() -> str:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("KESTREL_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return os.environ.get("KESTREL_API_KEY", "")

    # ========================================================================
    # Pre-Demo Setup
    # ========================================================================

    if not args.skip_setup:
        print()
        print(f"{WHITE}{'=' * 70}{NC}")
        print(f"  {WHITE}KESTREL SOVEREIGN - TRACK A TECHNICAL DEMO{NC}")
        print(f"  {GRAY}Pre-Demo Setup{NC}")
        print(f"{WHITE}{'=' * 70}{NC}")
        print()

        # 1. Check Ollama
        write_step("Checking Ollama...")
        try:
            req = urllib.request.Request("http://localhost:11434/api/version")
            with urllib.request.urlopen(req, timeout=3) as resp:
                ol = json.loads(resp.read())
                write_step(f"Ollama UP: {ol.get('version', '?')}")
        except Exception:
            write_fail("Ollama DOWN — Act 4 (EPHEMERAL) will fail!")
            write_fail("Start it: ollama serve")

        # 2. Kill any existing server on 8888
        write_step("Stopping any existing server...")
        try:
            subprocess.run(
                ["uv", "run", "kestrel", "stop"],
                cwd=str(kestrel_dir), capture_output=True, timeout=10,
            )
        except Exception:
            pass
        # Also kill standalone server.py
        if sys.platform != "win32":
            subprocess.run(["pkill", "-f", "python server.py"], capture_output=True)
        time.sleep(1)

        # 3. Create fresh demo agent (wipes agent_data/demo)
        write_step("Creating fresh demo agent (clean slate)...")
        result = subprocess.run(
            ["uv", "run", "python", "scripts/setup_demo_agent.py"],
            cwd=str(kestrel_dir), capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            write_fail(f"setup_demo_agent.py failed: {result.stderr}")
            sys.exit(1)
        # Print the DID from setup output
        for line in result.stdout.splitlines():
            if "DID:" in line or "Demo Agent Created" in line:
                write_step(line.strip())

        # 4. Start standalone server with demo agent
        write_step("Starting standalone server (agent_data/demo)...")
        global _server_proc
        env = os.environ.copy()
        env["KESTREL_DB_PATH"] = "agent_data/demo"
        _server_proc = subprocess.Popen(
            ["uv", "run", "python", "server.py"],
            cwd=str(kestrel_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop_server)
        signal.signal(signal.SIGINT, lambda *_: (stop_server(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda *_: (stop_server(), sys.exit(0)))

        if not wait_for_server(base_url, timeout=30):
            write_fail(f"Server failed to start at {base_url}")
            stop_server()
            sys.exit(1)
        write_step("Server UP")

        # 5. Load API key
        write_step("Loading API key...")
        key = load_api_key()
        if not key:
            # Try fetching from localhost bootstrap endpoint
            bootstrap = call_api(f"{base_url}/api/auth/key", timeout=5)
            if bootstrap:
                key = bootstrap.get("key", "")
        if key:
            write_step(f"Key loaded: {key[:8]}...")
        else:
            write_fail("No API key found")
            key = input("  Enter API key manually: ").strip()

        headers = {"X-API-Key": key, "Content-Type": "application/json"}

        # 6. Verify agent
        write_step("Verifying agent...")
        info = call_api(f"{base_url}/agent/info", headers=headers)
        if info:
            write_step(f"Agent DID: {info.get('agent_id', '?')}")
            write_step(f"Privacy mode: {info.get('privacy_mode', '?')}")
        else:
            write_fail("Agent not responding — cannot proceed")
            stop_server()
            sys.exit(1)

        # 7. Set model to Ollama (ensures EPHEMERAL local-only works)
        write_step("Setting model to Ollama (local provider)...")
        call_api(f"{base_url}/api/model/set", method="POST",
                 body='{"model":"llama3.2:latest","provider":"ollama"}', headers=headers)

        # 8. Reset privacy + skip bootstrap discovery
        call_api(f"{base_url}/agent/privacy-mode", method="POST",
                 body='{"mode":"NORMAL"}', headers=headers)
        bs = call_api(f"{base_url}/agent/invoke", method="POST",
                      body='{"input": "!bootstrap-status"}', headers=headers)
        if bs and "discovery" in str(bs.get("response", "")).lower():
            write_step("Skipping discovery mode...")
            call_api(f"{base_url}/agent/invoke", method="POST",
                     body='{"input": "!skip-discovery"}', headers=headers)

        print()
        write_step("Setup complete. Ready for demo.")
        pause_for_presenter(
            "Position terminal + browser side-by-side, then press ENTER to start", auto)

    else:
        # --skip-setup: assume server is running, just load key
        key = load_api_key()
        if not key:
            bootstrap = call_api(f"{base_url}/api/auth/key", timeout=5)
            if bootstrap:
                key = bootstrap.get("key", "")
        if not key:
            key = os.environ.get("KESTREL_API_KEY", "")
        headers = {"X-API-Key": key, "Content-Type": "application/json"}

    demo_start = time.time()

    # ========================================================================
    # ACT 1: Born in the Terminal (0:45 - 2:45)
    # ========================================================================

    write_act("1", "BORN IN THE TERMINAL", "0:45 - 2:45")

    write_narration("Every Kestrel agent begins with a single command.")
    pause_for_presenter("Press ENTER to show the agent's cryptographic identity...", auto)

    write_step("Querying the agent's identity chain...")
    chain = call_api(f"{base_url}/api/identity-chain", headers=headers)
    if chain:
        # Show clean output: just agent DID + constitution hash
        agent_info = chain.get("agent", {})
        constitution = chain.get("constitution", {})
        clean_chain = {
            "agent": {
                "did": agent_info.get("did"),
                "created_at": agent_info.get("created_at"),
            },
            "constitution": {
                "hash": constitution.get("hash"),
                "label": constitution.get("label"),
                "relationship": constitution.get("relationship"),
            },
        }
        print(pretty_json(clean_chain))

    write_narration("""\
What you're seeing: a secp256k1 key pair was generated at inception.
An Ethereum-format address was derived from the public key. A W3C
Decentralized Identifier was constructed from that address. The Kestrel
Constitution was hashed and linked as the first node in the knowledge graph.

No cloud service issued this identity. No certificate authority approved it.
It's mathematical proof that lives on this machine.""")

    pause_for_presenter("Press ENTER to show the constitution hash...", auto)

    write_step("Constitution hash (SHA-256, anchored at birth):")
    if chain and "constitution" in chain:
        h = chain["constitution"].get("hash", "?")
        print(f"  {WHITE}{h}{NC}")

    write_narration("""\
That hash is the proof that the constitution hasn't changed since inception.
You can re-hash the document yourself and compare — it'll match.""")

    write_step(f"Elapsed: {elapsed_str(demo_start)} (target: 2:45)")
    pause_for_presenter("Press ENTER for Act 2...", auto)

    # ========================================================================
    # ACT 2: Constitution Governs Every Response (2:45 - 5:30)
    # ========================================================================

    write_act("2", "CONSTITUTION GOVERNS EVERY RESPONSE", "2:45 - 5:30")

    write_step("Checking agent status...")
    status = call_api(f"{base_url}/agent/invoke", method="POST",
                      body='{"input": "!status"}', headers=headers)
    if status:
        print(f"  {status.get('response', '')}")

    write_narration("""\
DID and privacy mode — right in the status line.
Now let's ask it directly about its governance.""")
    pause_for_presenter("Press ENTER to ask about constitutional principles...", auto)

    write_step("Asking agent about its principles...")
    r = call_api(
        f"{base_url}/agent/invoke", method="POST",
        body='{"input": "What principles govern your behavior and who controls your rules?"}',
        headers=headers, timeout=120)
    if r and r.get("response"):
        print()
        print(f"  {WHITE}{r['response']}{NC}")
    else:
        write_fail("LLM did not respond — check provider config")

    write_narration("""\
That response went through a full context-building loop — RAG retrieval
from the knowledge graph, constitutional grounding, token budget management,
then the LLM call.""")

    pause_for_presenter("Press ENTER to show audit trace...", auto)

    write_step("Showing observability events (audit trace)...")
    obs = call_api(f"{base_url}/api/observability/events?limit=5", headers=headers)
    if obs and isinstance(obs, dict) and obs.get("events"):
        for ev in obs["events"]:
            etype = str(ev.get("event_type", "")).ljust(15)
            tname = ev.get("tool_name") or ""
            print(f"  [{etype}]  {tname}")
    else:
        write_step("(no observability events yet)")

    write_narration("""\
Every LLM call is logged with timing. In a regulated deployment,
this is your compliance record.""")

    write_step(f"Elapsed: {elapsed_str(demo_start)} (target: 5:30)")
    pause_for_presenter("Press ENTER for Act 3...", auto)

    # ========================================================================
    # ACT 3: Memory That Survives Sessions (5:30 - 7:45)
    # ========================================================================

    write_act("3", "MEMORY THAT SURVIVES SESSIONS", "5:30 - 7:45")

    write_narration("""\
Most AI chat tools give you conversation history within a tab.
Close the tab, start over. Kestrel uses a knowledge graph —
persistent, cross-session, and yours.""")

    write_step("Querying knowledge graph nodes...")
    mem = call_api(f"{base_url}/api/memories", headers=headers)
    if mem and isinstance(mem, dict):
        print(f"  Total memory nodes: {mem.get('total', '?')}")
        nodes = mem.get("nodes", [])
        if nodes:
            print(f"  {'node_type':<25} {'label'}")
            print(f"  {'-' * 25} {'-' * 30}")
            for n in nodes:
                print(f"  {str(n.get('node_type', '')):25} {n.get('label', '')}")

    write_narration("""\
Every node was written by a real event — inception, constitution anchoring.
The graph is the agent's persistent memory.""")

    pause_for_presenter("Press ENTER to test cross-session memory...", auto)

    write_step("Starting a new session (fresh conversation history)...")
    ns = call_api(f"{base_url}/api/conversations/new", method="POST",
                  body='{}', headers=headers)
    session_id = str(ns.get("session_id", "1")) if ns else "1"
    print(f"  New session ID: {session_id}")

    write_step("Asking for identity in the new session (zero conversation history)...")
    body = json.dumps({"input": "!status", "session_id": session_id})
    r = call_api(f"{base_url}/agent/invoke", method="POST", body=body,
                 headers=headers, timeout=30)
    if r and r.get("response"):
        print()
        print(f"  {WHITE}{r['response']}{NC}")
    else:
        write_fail("Agent did not respond")

    write_narration("""\
Zero conversation history in this session. The agent's identity comes from
the knowledge graph — persistent, cryptographic, independent of any chat window.""")

    write_step(f"Elapsed: {elapsed_str(demo_start)} (target: 7:45)")
    pause_for_presenter("Press ENTER for Act 4...", auto)

    # ========================================================================
    # ACT 4: Privacy Is Architecture, Not Policy (7:45 - 9:30)
    # ========================================================================

    write_act("4", "PRIVACY IS ARCHITECTURE, NOT POLICY", "7:45 - 9:30")

    write_narration("""\
Now watch what happens when we flip the privacy mode to EPHEMERAL.
This is where Kestrel is genuinely different from anything else.""")

    write_step("Recording baseline conversation count...")
    before = call_api(f"{base_url}/api/conversations", headers=headers)
    before_total = before.get("total", 0) if before else 0
    print(f"  Conversations before: {before_total}")

    write_step("Switching to EPHEMERAL mode...")
    priv_result = call_api(f"{base_url}/agent/privacy-mode", method="POST",
                           body='{"mode":"EPHEMERAL"}', headers=headers)
    if priv_result:
        print(f"  {YELLOW}{priv_result.get('message', '')}{NC}")

    pause_for_presenter("Press ENTER to send message in EPHEMERAL mode...", auto)

    write_step("Sending message in EPHEMERAL mode (nothing should be stored)...")
    r = call_api(
        f"{base_url}/agent/invoke", method="POST",
        body='{"input": "This is a sensitive matter I do not want stored anywhere."}',
        headers=headers, timeout=120)
    if r and r.get("response"):
        print()
        print(f"  {WHITE}{r['response']}{NC}")
    elif r is None:
        # EPHEMERAL may fail if Ollama model is slow — that's OK, the point
        # is that nothing was stored, which we verify next
        write_step("(LLM timed out in EPHEMERAL — the privacy proof still holds)")

    write_step("Restoring NORMAL mode...")
    call_api(f"{base_url}/agent/privacy-mode", method="POST",
             body='{"mode":"NORMAL"}', headers=headers)

    write_step("Checking if any records were written...")
    after = call_api(f"{base_url}/api/conversations", headers=headers)
    after_total = after.get("total", 0) if after else 0
    new_records = after_total - before_total
    color = GREEN if new_records == 0 else RED
    print(f"  {color}New records written during EPHEMERAL session: {new_records}{NC}")

    write_narration("""\
Kestrel has five privacy levels:
  EPHEMERAL  — nothing stored, not even temporarily
  ISOLATED   — in-memory only; you can explicitly save or discard
  ANONYMOUS  — stored encrypted, distributed, no identity linkage
  NORMAL     — full local persistence with sovereignty guarantees
  PUBLIC     — cloud LLMs allowed

In a regulated industry, you can hardcode a privacy floor that operators
cannot override. The compliance guarantee is in the architecture.""")

    write_step(f"Elapsed: {elapsed_str(demo_start)} (target: 9:30)")
    pause_for_presenter("Press ENTER for Act 5...", auto)

    # ========================================================================
    # ACT 5: You Own It. You Can Move It. (9:30 - 11:00)
    # ========================================================================

    write_act("5", "YOU OWN IT. YOU CAN MOVE IT.", "9:30 - 11:00")

    write_narration("Last thing. This is the one we don't see anywhere else in the market.")
    pause_for_presenter("Press ENTER to run sovereignty export...", auto)

    write_step("Exporting agent state...")
    export = call_api(
        f"{base_url}/api/sovereignty/export", method="POST",
        body='{"tier": "local", "encrypt": false}',
        headers=headers, timeout=60)
    if export:
        print()
        print(f"  {WHITE}{export.get('message', pretty_json(export))}{NC}")

    write_step("Showing export receipt in knowledge graph...")
    mem3 = call_api(f"{base_url}/api/memories", headers=headers)
    if mem3 and isinstance(mem3, dict):
        receipts = [n for n in mem3.get("nodes", [])
                    if n.get("node_type") == "sovereignty_receipt"]
        if receipts:
            for receipt in receipts:
                nid = receipt.get("node_id", "?")
                print(f"  {nid[:40]}...  {receipt.get('label', '')}")
        else:
            print(f"  {GRAY}(no sovereignty_receipt nodes found){NC}")

    write_narration("""\
That export is a portable blob. Content-addressed by SHA-256 — you can
verify it hasn't been tampered with. It contains the complete agent state.

You can take it to another machine, another Kestrel instance, another
cloud provider. The agent doesn't live in our cloud. It lives in that file.
The CID is the proof.""")

    # ========================================================================
    # CLOSE
    # ========================================================================

    total_elapsed = (time.time() - demo_start) / 60.0

    print()
    print(f"{YELLOW}{'=' * 70}{NC}")
    print(f"  {YELLOW}DEMO COMPLETE{NC}")
    print(f"  {DARK_YELLOW}Total time: {total_elapsed:.1f} minutes{NC}")
    print(f"{YELLOW}{'=' * 70}{NC}")

    write_narration("""\
What you just saw:

  A cryptographic identity generated in two seconds.
  A constitution anchored at birth and tamper-evident.
  An audit trace on every response.
  A knowledge graph that accumulates across sessions, not a chat window.
  Privacy enforced by the storage layer, not by policy.
  A complete data export with a content hash you can independently verify.

Kestrel is MIT-licensed, open source, runs on any machine with a GPU or
cloud budget. In 30 minutes you can have your own agent running with all
of this active.""")

    print()
    print(f"  {MAGENTA}KEY PHRASES:{NC}")
    print(f'  {WHITE}"That refusal is constitutional, not corporate."{NC}')
    print(f'  {WHITE}"The compliance guarantee is in the architecture."{NC}')
    print(f'  {WHITE}"The agent does not live in our cloud. It lives in that file."{NC}')
    print(f'  {WHITE}"In 30 minutes you can have your own agent running."{NC}')
    print()

    # Clean up
    stop_server()


if __name__ == "__main__":
    main()
