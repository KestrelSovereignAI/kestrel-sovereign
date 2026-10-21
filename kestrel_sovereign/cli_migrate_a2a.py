"""Offline, reviewed A2A cutover command. It never boots an agent."""

import asyncio
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from kestrel_sovereign.a2a.legacy_cutover import (
    apply_legacy_migration,
    prepare_legacy_migration,
)
from kestrel_sovereign.a2a.migrate_legacy import LegacyTaskMigrationError
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnership
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.multi_agent.liveness import serving_holder
from kestrel_sovereign.storage.db.interface import DatabaseError


def add_migrate_a2a_subcommand(subparsers):
    command = subparsers.add_parser(
        "migrate-a2a", help="Preview or apply the offline native A2A data cutover"
    )
    actions = command.add_subparsers(dest="migration_action", required=True)
    for action in ("preview", "apply"):
        parser = actions.add_parser(action)
        target = parser.add_mutually_exclusive_group(required=True)
        target.add_argument(
            "--sqlite",
            type=Path,
            help="Existing agent database; its host must be stopped",
        )
        target.add_argument(
            "--postgres-dsn-env",
            help="Name of an exported PostgreSQL DSN variable; all host connections must be stopped",
        )
        if action == "preview":
            parser.add_argument(
                "--owners",
                required=True,
                type=Path,
                help="Reviewed task ID to ContextOwnership JSON mapping",
            )
            parser.add_argument(
                "--output",
                required=True,
                type=Path,
                help="New file for the concrete migration plan",
            )
        else:
            parser.add_argument(
                "--plan",
                required=True,
                type=Path,
                help="Previously reviewed preview file",
            )


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Migration JSON contains duplicate keys")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Migration JSON requires finite values")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=invalid_constant,
    )


def _ownership(value):
    if not isinstance(value, dict):
        raise ValueError("Reviewed task provenance must be a JSON object")
    names = {field.name for field in fields(ContextOwnership)}
    result = {}
    for task_id, owner in value.items():
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(owner, dict)
            or set(owner) != names
        ):
            raise ValueError(
                "Every task needs explicit context, tenant, recipient, and creator provenance"
            )
        if any(not isinstance(item, str) for item in owner.values()):
            raise ValueError("Reviewed provenance fields must be strings")
        result[task_id] = ContextOwnership(**owner)
    return result


def _plan_ownership(review):
    if (
        not isinstance(review, dict)
        or set(review)
        != {"revision", "migration_id", "source_digest", "tables", "tasks"}
        or review["revision"] != 1
    ):
        raise ValueError("Unsupported reviewed A2A migration plan")
    if not isinstance(review["tasks"], list):
        raise ValueError("Reviewed migration tasks must be an array")
    owners = {}
    for task in review["tasks"]:
        if not isinstance(task, dict) or set(task) != {
            "task_id",
            "ownership",
            "native_sha256",
        }:
            raise ValueError("Malformed reviewed task conversion")
        if task["task_id"] in owners:
            raise ValueError("Reviewed migration repeats a task ID")
        owners[task["task_id"]] = task["ownership"]
    return _ownership(owners)


async def _run(args):
    if args.sqlite is not None:
        selected = args.sqlite.expanduser()
        path = selected.resolve(strict=True)
        if not path.is_file():
            raise ValueError("The selected SQLite database must be an existing file")
        # Agents register the profile directory, even when its database file
        # is a symlink. Check both that spelling and the canonical target.
        for directory in {selected.parent.resolve(), path.parent}:
            holder = serving_holder(directory)
            if holder is not None:
                raise ValueError(
                    f"Stop the selected database's host before offline migration: {holder.evidence}; {holder.remedy}"
                )
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(path), cold_read=args.migration_action == "preview")
    else:
        dsn = os.environ.get(args.postgres_dsn_env)
        if not dsn:
            raise ValueError("The selected PostgreSQL DSN variable is not exported")
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(dsn, min_pool_size=1, max_pool_size=1)
    try:
        await backend.connect()
        if args.migration_action == "preview":
            owners = _ownership(_read(args.owners))
            plan = await prepare_legacy_migration(backend, ownership=owners)
            if args.sqlite is not None:
                backend.assert_cold_read_still_valid()
            descriptor = os.open(
                args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(plan.review(), output, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            return {
                "action": "preview",
                "plan": str(args.output),
                "review": plan.review(),
            }
        review = _read(args.plan)
        owners = _plan_ownership(review)
        if args.sqlite is None:
            other = await backend.fetch_one(
                "SELECT 1 FROM pg_stat_activity WHERE datname = current_database() "
                "AND pid != pg_backend_pid() AND backend_type = 'client backend' LIMIT 1"
            )
            if other is not None:
                raise ValueError(
                    "Stop all other connections to the selected database before applying the offline A2A cutover"
                )
        plan = await prepare_legacy_migration(
            backend, ownership=owners, migration_id=review["migration_id"]
        )
        if plan.review() != review:
            raise LegacyTaskMigrationError(
                "The reviewed plan no longer matches this database; create and review a new preview"
            )
        archives = await apply_legacy_migration(backend, plan)
        return {
            "action": "applied",
            "migration_id": plan.migration_id,
            "archives": archives,
        }
    finally:
        await backend.close()


def run(args):
    try:
        result = asyncio.run(_run(args))
    except (OSError, ValueError, TypeError, DatabaseError) as error:
        if is_execution_control_error(error):
            raise
        sys.stderr.write(f"A2A migration refused: {error}\n")
        return 1
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0
