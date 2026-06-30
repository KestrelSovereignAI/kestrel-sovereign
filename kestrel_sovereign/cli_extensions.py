"""CLI extension discovery via the ``kestrel_sovereign.cli`` entry-point group.

The ``kestrel`` CLI is argparse-based and every core subcommand group is a
hardcoded import in :func:`kestrel_sovereign.cli.build_parser`. This module
lets a feature package contribute a ``kestrel <feature> ...`` subcommand with
**zero** core edits, mirroring the backend feature discovery that already runs
over the ``kestrel_sovereign.features`` group.

How a feature registers a CLI group
------------------------------------

In its ``pyproject.toml``::

    [project.entry-points."kestrel_sovereign.cli"]
    myfeature = "my_package.cli:add_myfeature_subparser"

The entry-point name (``myfeature``) is the subcommand name and the value
resolves to a callable matching the existing core convention exactly::

    def add_myfeature_subparser(subparsers):
        parser = subparsers.add_parser("myfeature", help="...")
        sub = parser.add_subparsers(dest="myfeature_command")
        sub.add_parser("status", help="Show feature status")
        # Register the dispatch handler the same way cli_serve / cli_embeddings
        # do — the CLI dispatcher drains ``args._handler`` for extensions.
        parser.set_defaults(_handler=run)

    def run(args):
        ...

Critical constraint — out-of-process
-------------------------------------

The CLI runs **host-side, out-of-process from the live agent**. A feature CLI
command **cannot** touch in-process feature state. It operates against host
config or the agent's HTTP API — exactly like ``cli_features`` does today. So a
feature CLI command is a **thin client over the feature's router** (the real
primitive): ``kestrel myfeature status`` should hit the feature's own router
endpoint over HTTP, not call an in-process tool. See
``cli_extension_example.py`` for a reference implementation.

Isolation
---------

Discovery and registration are defensive: a missing group is a no-op, an
extension whose name collides with a core command is rejected (core wins,
logged), and an extension that raises at registration is logged and skipped so
one broken extension never breaks the whole CLI.
"""

from __future__ import annotations

import argparse
import logging

from kestrel_sovereign.entrypoints import discover_entry_point_callables

logger = logging.getLogger(__name__)

CLI_ENTRY_POINT_GROUP = "kestrel_sovereign.cli"


def register_cli_extensions(
    subparsers: argparse._SubParsersAction,
    group: str = CLI_ENTRY_POINT_GROUP,
) -> list[str]:
    """Discover and register CLI extension subcommand groups.

    Invoked by :func:`kestrel_sovereign.cli.build_parser` after the core
    command groups are registered, so ``subparsers.choices`` already holds
    every reserved core name.

    Each discovered entry point resolves to an ``add_<name>_subparser`` callable
    that registers its own subparser (and its ``_handler`` dispatch default).

    Collision handling: an extension whose entry-point name matches a core
    command (or an already-registered extension) is rejected with a clear log
    message — **core wins**. Registration failures are logged and skipped.

    Returns:
        The list of extension command names successfully registered.
    """
    # Snapshot the names already claimed by core groups (and their aliases).
    reserved = set(subparsers.choices)
    registered: list[str] = []

    for name, add_subparser in discover_entry_point_callables(group):
        if name in reserved:
            logger.error(
                "CLI extension '%s' collides with an existing command; "
                "skipping it (core wins).",
                name,
            )
            continue
        try:
            add_subparser(subparsers)
        except Exception as e:
            logger.error(
                "CLI extension '%s' failed to register, skipping: %s",
                name, e,
            )
            continue
        reserved.add(name)
        registered.append(name)
        logger.info("Registered CLI extension command: %s", name)

    return registered
