#!/usr/bin/env python3
"""
The main entry point for the Kestrel Agent.
"""
import argparse
import asyncio
import os
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.security.encryption import DecryptionError
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
)
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.paths import load_project_env, project_dir
import logging

from kestrel_sovereign.kestrel_config.constants import SHUTDOWN_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_agent_did_async(
    storage_dir: str,
    *,
    db_backend: str | None = None,
    database_url: str | None = None,
) -> str:
    """Resolve which agent this directory is, for server startup.

    **Identity comes from the anchor; governance comes from the runtime
    database.** ``agent_data/<Name>/kestrel_prime.db`` is where inception
    writes the birth record on every backend, and twelve places across seven
    modules already read its existence as the fact that a directory *is* an
    agent (#2871). The runtime database is where the agent is governed.

    This function used to ask the runtime database on PostgreSQL, which
    inverted that and blocked boot outright: ``kestrel create`` writes the
    birth record to the anchor, PostgreSQL has no tables at all until the
    agent boots, and the replication that fills it (#2871) runs inside
    ``KestrelAgent.initialize()`` — downstream of this gate. The gate refused
    first, so the boot that would have repaired the gap never happened, and
    ``kestrel start`` reported 503 for its whole window (#2894).

    The runtime database is still consulted when the anchor cannot answer —
    an ephemeral container whose disk carries no identity, which is the case
    #2472 added the PostgreSQL branch for. That is not a fallback in the sense
    this codebase refuses: the two sources answer different questions, and
    when both can answer, disagreeing about who this agent is is a custody
    failure that gets refused rather than resolved.

    Args:
        storage_dir: Directory containing local identity files and, for SQLite,
            ``kestrel_prime.db``.
        db_backend: Explicit storage backend. Defaults to KESTREL_DB_BACKEND.
        database_url: PostgreSQL DSN. Defaults to KESTREL_DATABASE_URL.
    """
    from kestrel_sovereign.identity.local_anchor import (
        AgentDIDLookupMode,
        AnchorAbsent,
        read_anchor_agent_did,
    )

    backend = (db_backend or os.environ.get("KESTREL_DB_BACKEND", "sqlite")).lower()
    if backend != "postgres":
        # The anchor IS the runtime database here, so there is one answer and
        # one reader for it.
        return await read_anchor_agent_did(
            storage_dir, mode=AgentDIDLookupMode.INITIALIZATION
        )

    dsn = database_url or os.environ.get("KESTREL_DATABASE_URL")
    if not dsn:
        raise ValueError(
            "KESTREL_DATABASE_URL is required when KESTREL_DB_BACKEND=postgres"
        )

    anchored_did: str | None = None
    try:
        anchored_did = await read_anchor_agent_did(
            storage_dir, mode=AgentDIDLookupMode.INITIALIZATION
        )
    except AnchorAbsent as exc:
        # *Only* absence. A corrupt file, a permission denial, or two agent
        # roots all mean an anchor is present and could not be read, and
        # falling through to the runtime database there would boot this
        # directory as whichever agent that database happens to hold. Those
        # propagate. This branch is for the container whose disk genuinely
        # carries no identity, which is the case #2472 added it for.
        logger.info(
            "No local identity anchor in %s (%s); asking the runtime database.",
            storage_dir, exc,
        )

    storage = AsyncStorage(backend="postgres", dsn=dsn)
    await storage.initialize()
    try:
        agent_nodes = await storage.get_nodes_by_type("agent")
        runtime_dids = [node.node_id for node in agent_nodes]
    finally:
        await storage.close()

    if anchored_did is not None:
        if runtime_dids and anchored_did not in runtime_dids:
            # This directory belongs to one agent and the database to another.
            # Booting either identity against the other's governance is the
            # "wrong database" failure this cluster is about; naming both is
            # the only safe answer.
            # Naming a remedy the operator can actually reach: the standalone
            # launcher takes one host-wide KESTREL_DATABASE_URL — there is no
            # per-agent DSN to point anywhere (that is #2843). The in-process
            # host runs a whole fleet against one PostgreSQL happily, because
            # it resolves each agent's identity from its own anchor.
            raise ValueError(
                f"Identity conflict: the local anchor in {storage_dir} names "
                f"{anchored_did}, but the configured PostgreSQL database holds "
                f"{', '.join(sorted(runtime_dids))}. Durable single-agent "
                "custody requires a dedicated database per agent, and the "
                "standalone launcher has only one host-wide "
                "KESTREL_DATABASE_URL. Run the fleet in-process instead — "
                "`kestrel start` with no agent name — or give this agent its "
                "own database. Per-agent custody is #2843."
            )
        if len(runtime_dids) > 1:
            raise ValueError(
                "Durable single-agent PostgreSQL custody requires exactly "
                "one agent node; use a dedicated database"
            )
        # Zero rows is the freshly-incepted case: boot proceeds and the birth
        # record is replicated into the runtime database by #2871.
        return anchored_did

    if not runtime_dids:
        raise ValueError(
            "No agent found in the database. Please run inception service first."
        )
    if len(runtime_dids) > 1:
        raise ValueError(
            "Durable single-agent PostgreSQL custody requires exactly "
            "one agent node; use a dedicated database"
        )
    return runtime_dids[0]

# This is a placeholder for a more robust discovery mechanism.
# In a real system, this would involve a more complex lookup.
async def get_agent_by_did(did: str) -> KestrelAgent:
    """
    Retrieves an agent instance based on its DID.
    """
    storage_path = os.environ.get("KESTREL_DB_PATH", os.getcwd())
    llm_service = LLMService()
    agent = KestrelAgent(did=did, storage_path=storage_path, llm_service=llm_service)
    from kestrel_sovereign.hold import build_bound_host_context

    # This legacy helper transfers both agent and context lifetime to its
    # caller.  Main process paths below close their context explicitly.
    context = await build_bound_host_context(agent)
    agent._standalone_hold_context = context
    try:
        await agent.initialize()
    except BaseException as error:
        from kestrel_sovereign.hold import close_bound_host_context

        try:
            await close_bound_host_context(context)
        except BaseException as close_error:
            error.add_note(
                "Standalone Hold context cleanup also failed: "
                f"{type(close_error).__name__}: {close_error}"
            )
        finally:
            agent._standalone_hold_context = None
        raise
    return agent

async def main():
    # This module is a process entry point (``python -m kestrel_sovereign.main``
    # — the container's interactive shell, per ``docker_entrypoint.sh``), so it
    # loads the project home's ``.env`` itself. It used to get one for free from
    # ``LLMService.__init__``, which called ``load_dotenv()`` on every
    # construction; that is removed in #2896 because a library constructor
    # resurrecting deliberately-unset variables is not something the rest of the
    # process can defend against. An entry point loading its own environment,
    # once, before it reads any of it, is.
    #
    # One deliberate consequence: ``db_path`` below falls back to
    # ``KESTREL_DB_PATH``, which the old constructor-time load could never
    # reach because it ran later. A home whose ``.env`` sets it now selects
    # that agent instead of printing "path not specified". Inert in the
    # container, which exports the variable already.
    load_project_env(project_dir())

    parser = argparse.ArgumentParser(description="Kestrel Sovereign Agent Interface")
    parser.add_argument(
        "db_path",
        nargs='?',
        type=str,
        default=None,
        help="Path to the agent's memory database. Overrides KESTREL_DB_PATH."
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        choices=['elderly'],
        help="Load an application extension."
    )
    args = parser.parse_args()

    db_path = args.db_path or os.environ.get("KESTREL_DB_PATH")
    if not db_path:
        print("❌ Error: Agent database path not specified.")
        return

    if not os.path.exists(db_path):
        print(f"❌ Error: Database file not found at '{db_path}'.")
        return

    storage_dir = db_path
    agent_did = await get_agent_did_async(storage_dir)
    if not agent_did:
        print(f"❌ Error: Could not determine agent's DID from '{storage_dir}'.")
        return

    # Construct full database file path from directory (same as get_agent_did_async)
    storage_path = os.path.join(storage_dir, "kestrel_prime.db")
    llm_service = LLMService()
    agent = KestrelAgent(did=agent_did, storage_path=storage_path, llm_service=llm_service)
    from kestrel_sovereign.hold import (
        HOLD_TURN_CONSOLE_MESSAGE,
        HoldTurnRefusal,
        build_bound_host_context,
        close_bound_host_context,
    )

    hold_context = await build_bound_host_context(agent)
    await agent.initialize()

    if args.app:
        extension_class = None
        if args.app == 'elderly':
            from kestrel_sovereign.extensions.elderly_extension import ElderlyExtension
            extension_class = ElderlyExtension
        
        if extension_class:
            agent.extension = extension_class(agent)
            agent.app_context = args.app
            print(f"   Extension Loaded: {args.app}")

    print("✅ Kestrel Agent Initialized.")
    print(f"   DID: {agent.agent_id}")
    print(f"   Memory: {db_path}")

    decryption_error_count = 0
    MAX_DECRYPTION_ERRORS = 3

    try:
        while True:
            user_input = input("\n> ")
            if user_input.lower() == '!quit':
                break
            try:
                response = await agent.process_input(user_input)
                decryption_error_count = 0  # Reset on success
                print(f"\nKestrel: {response}")
            except HoldTurnRefusal:
                print(f"\n{HOLD_TURN_CONSOLE_MESSAGE}")
            except DecryptionError as e:
                decryption_error_count += 1
                logger.error(f"DecryptionError during processing: {e}")
                print(f"\n🔐 DECRYPTION ERROR: Cannot read encrypted data.")
                print(f"   This usually means KESTREL_DATA_KEY is incorrect or missing.")
                print(f"   Error count: {decryption_error_count}/{MAX_DECRYPTION_ERRORS}")

                if decryption_error_count >= MAX_DECRYPTION_ERRORS:
                    print("\n⚠️  Too many decryption errors. Entering safe mode.")
                    print("   The agent cannot access encrypted memories.")
                    print("   Please verify KESTREL_DATA_KEY and restart.")
                    print("   Use !quit to exit.")
                    # Trigger safe mode in agent
                    if hasattr(agent, 'enter_safe_mode'):
                        # An availability failure of stored MEMORY — not a
                        # failed verification, and not governance state
                        # either. The constitution was never read, so nothing
                        # about it was found wrong, and the runtime-state
                        # store is answering fine; pointing the operator at
                        # either would send them to the wrong place (#2920).
                        from kestrel_sovereign.agent.constitution import (
                            SafeModeCause,
                        )

                        await agent.enter_safe_mode(
                            "Repeated encrypted-state decryption failures",
                            cause=SafeModeCause.MEMORY_UNREADABLE.value,
                        )

    except KeyboardInterrupt:
        print("\nDeactivating agent...")
    finally:
        cancelled = False
        # Graceful shutdown with timeout
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
        except Exception as e:
            logger.debug(f"Error during shutdown: {e}")
            print("Agent deactivated (with errors).")
        cancelled = await await_agent_shutdown_completion(agent) or cancelled
        await close_bound_host_context(hold_context)
        if cancelled:
            raise asyncio.CancelledError()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle double Ctrl+C gracefully
        print("\nForced exit.")
