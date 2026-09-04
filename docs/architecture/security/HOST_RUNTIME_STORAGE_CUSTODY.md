---
type: Security Guide
title: Host Runtime Storage Custody
description: Private placement, SQLite creation, migration, and database-separation rules for fleet and host-feature state.
status: active
owner: security
canonical: true
privacy: public
---

# Host runtime storage custody

The multi-agent host opens a fleet-scoped SQLite backend for host features.
That database can contain cross-agent entities, workflow state, operational
metadata, and feature-owned records. Kestrel treats it as sensitive runtime
state, not source-tree output.

## Shared private host-data root

Implicit host services use one placement rule:

1. `<KESTREL_HOME>/host-data` when `KESTREL_HOME` is explicitly set;
2. `~/.kestrel/host-data` otherwise.

This resolver deliberately ignores source markers and the current working
directory. Launching from a clone does not make that clone a runtime-data root.
Phoenix uses `<host-data>/phoenix`; fleet/host features use
`<host-data>/host-features.db`.

`KESTREL_HOST_DB_PATH` remains the explicit host-feature database override. Its
parent is the custody boundary: Kestrel creates a missing parent as `0700`, but
an existing parent must already be a real, dedicated `0700` directory. Kestrel
will not chmod a shared operator directory such as `/data` or `/tmp`. The
database leaf and SQLite auxiliaries must be regular single-link files; symbolic
links, hard links, and special files fail closed.

The supported SQLite Docker images set this override beneath their persistent
agent-data mount (`/app/agent_data/host-data/host-features.db`, or
`/data/host-data/host-features.db` for the sovereign image). Recreating a
container therefore preserves the active Hold database and its adjacent
history/pending-publication witnesses with the agent database. Custom images
must provide an equivalent persistent mount; the process-home default is not a
durability boundary inside a replaceable container.

## Secure SQLite creation

Kestrel opens or pre-creates the main database with `O_NOFOLLOW` where the
platform provides it, verifies the opened descriptor is a single-link regular
file, and applies `0600` before SQLite receives the path. Existing DB, WAL, SHM,
and rollback-journal files are hardened before connection.

On POSIX, SQLite's standard Unix VFS creates `-wal`, `-journal`, and `-shm`
files with exactly the main database mode, ignoring the process umask for those
auxiliaries. See the comments and implementation around `robust_open` and
`findCreateFileMode` in the [upstream SQLite Unix VFS source](https://sqlite.org/src/doc/trunk/src/os_unix.c).
Kestrel validates the live family after opening; if the VFS or filesystem does
not preserve the private contract, the partial context is closed and host
features receive no database rather than an insecure one.

The POSIX mode contract is `0700` for the parent and `0600` for the main DB,
WAL, SHM, and journal. Windows does not expose equivalent POSIX mode semantics;
link/type validation still applies and ACL policy remains the operator's
responsibility.

## Two host databases, two responsibilities

The similarly named databases are intentionally distinct:

| Database | Owner and purpose | Discovery contract |
|---|---|---|
| `<host-data>/host-features.db` | Multi-agent host; fleet-scoped host-feature entities and operational state | `build_host_context`; optional `KESTREL_HOST_DB_PATH` |
| `<project>/agent_data/host.db` | Payment/key subsystem; deployment credential records agents must discover from their storage path | setup payment step and `open_host_db` |

There is no fallback, merge, or automatic copying between these databases.
Renaming the fleet store to `host-features.db` makes that boundary explicit;
moving payment/key records into the fleet-feature store would break agent-side
credential discovery and expand access to credential tables. Operators must
back up and govern each store according to its owner.

## Legacy migration

Older releases defaulted fleet state to `<project_dir>/kestrel_host.db`. On the
first default-path startup, Kestrel:

1. restricts the legacy main DB and any auxiliaries to `0600` in place;
2. refuses migration if WAL, SHM, or journal files remain, because an older
   host may still own the database;
3. refuses to guess or merge when both legacy and new databases exist;
4. atomically renames a stopped single-file database on one filesystem; or
5. copies it through a private, fsynced staging file across filesystems,
   publishes the destination, and removes the source only after validation and
   destination-directory fsync.

If sidecars remain, stop every older Kestrel host cleanly and restart. If both
stores exist, stop Kestrel, back up both, inspect them through an
operator-controlled SQLite process, select the authoritative fleet-feature
history, and move the other aside. Custody errors disable the host database;
host routes that do not need persistence may continue, but no insecure SQLite
connection is returned.

The repository ignore patterns for `*.db-wal`, `*.db-shm`, and
`*.db-journal` are defense in depth for legacy or explicit checkout paths.
Ignore rules do not provide confidentiality; descriptor validation, private
parent custody, and the SQLite creation mode are the controls.
