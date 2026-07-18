---
type: Security Guide
title: Phoenix Trace Custody
description: Private placement, creation, migration, and recovery rules for the host-supervised Phoenix trace store.
status: active
owner: security
canonical: true
privacy: public
---

# Phoenix trace custody

Phoenix stores observability data that can contain user prompts, model outputs,
tool inputs and results, provider metadata, and exception details. Kestrel treats
the complete Phoenix working directory as sensitive host data rather than
source-tree output.

## Storage resolution

Kestrel resolves the working directory in this order:

1. `KESTREL_PHOENIX_WORKING_DIR`, when explicitly set;
2. `<KESTREL_HOME>/host-data/phoenix`, when `KESTREL_HOME` is explicitly set;
3. `~/.kestrel/host-data/phoenix` otherwise.

The default intentionally does not use marker discovery or the current source
checkout. Launching Kestrel from a clone must not make that clone a trace-data
root. Set `KESTREL_HOME` or `KESTREL_PHOENIX_WORKING_DIR` when an operator wants
an explicit data volume.

## Creation and startup gate

Before Kestrel advertises the local Phoenix OTLP endpoint to the host or spawned
agents, the supervisor performs a custody preflight:

- the host-data and Phoenix directories must be real directories, never links,
  and are created or restricted to mode `0700`;
- every existing directory below the store is restricted to `0700`;
- every existing regular file is restricted to `0600`;
- symbolic links, multiply linked files, and special files fail closed;
- `phoenix.log` and `phoenix.pid` are opened without following links and made
  `0600` before data is written; and
- the Phoenix child starts with `umask 077`, so `phoenix.db`, `-wal`, `-shm`,
  temporary files, and future child-created files are private from their first
  inode.

If the preflight cannot establish custody, Kestrel does not auto-wire
`OTEL_EXPORTER_OTLP_ENDPOINT`, does not start Phoenix, and leaves `/phoenix`
returning `503`. An operator-supplied external OTLP endpoint is not changed.

## Legacy source-checkout migration

Older releases wrote to `<project_dir>/phoenix`. On first startup with the new
default, Kestrel detects that store and restricts it in place before attempting
migration. Migration proceeds only when:

- no live PID is recorded in the legacy store; and
- the new destination does not already contain a separate store.

On one filesystem the secured directory is renamed atomically. Across
filesystems Kestrel copies a stopped store through a private staging directory,
validates the destination modes, publishes it, and removes the old copy only
after the destination succeeds.

If both old and new stores exist, Kestrel will not guess which SQLite history is
authoritative or merge live databases. Stop Kestrel, back up both directories,
inspect them through an operator-controlled process, move the non-authoritative
directory aside, and restart. If the legacy PID is still live, stop that process
before restarting. The startup error reports both resolved paths and the exact
reason tracing stayed disabled.

The repository `/phoenix/` ignore rule is defense in depth for old or manually
configured checkouts. Ignore rules do not establish privacy; the custody
preflight and child umask are the security controls.
