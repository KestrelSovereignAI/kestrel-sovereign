---
type: Architecture Guide
title: Server Launch Contract
description: Canonical direct-server command, bind precedence, and managed-launch boundaries.
resource: /docs/architecture/core/SERVER_LAUNCH_CONTRACT.md
tags:
- docs
- architecture
- operations
- security
timestamp: '2026-07-19T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Server Launch Contract

Kestrel has one human-operated direct-server command:

```bash
KESTREL_DB_PATH=./agent_data/myagent uv run python -m kestrel_sovereign.server \
  --host 127.0.0.1 --port 8888
```

Use an explicit loopback host for local development. The compatibility host
default remains `0.0.0.0` because container platforms require an all-interface
bind, but relying on that default can expose a development agent to other
machines on the network.

## Precedence

The module resolves each value independently before opening a socket:

| Setting | Highest precedence | Environment | Compatibility default |
|---|---|---|---|
| Bind host | `--host HOST` | `KESTREL_SERVER_HOST` | `0.0.0.0` |
| TCP port | `--port PORT` | `PORT` | `8888` |

CLI values always win. For example, `PORT=8080 ... --port 9999` binds port
`9999`. Kestrel mirrors the resolved values into its runtime environment so
Uvicorn's socket and the host-feature `HostContext` report the same address.

In a managed multi-agent host, extension-specific host settings live beneath
`[host.features.<name>]` in `multi_agent.toml`. Kestrel copies each explicitly
selected mapping into the public `HostContext.config` under `<name>`; it does
not expose the whole registry or agent definitions to extensions. The context
is host-wide: every installed host feature can read these mappings. Put only
non-secret routing and policy values here; keep credentials in the feature's
private store, a referenced private config file, or the deployment secret
manager. For example:

```toml
[host.features.talon]
operator_state_path = "/srv/kestrel/state/talon"

[host.features.talon.runtime]
project_parent = "/srv/kestrel/projects"
running_agent_source_root = "/srv/kestrel/kestrel-sovereign"
```

The keys `host_bind`, `host_port`, `agents`, and
`observability_tenant_resolver` are platform-owned and cannot be used as
feature names.

Hosts must be non-empty. Ports must be integers from 1 through 65535. Unknown
options and invalid values print argparse usage, exit with status 2, and never
start Uvicorn. This means a misspelled or unsupported security-sensitive option
cannot look like a successful launch.

## `kestrel start` is a different contract

`kestrel start` is the normal fleet launcher. It reads the host bind and port
from the `[host]` section of `multi_agent.toml`; `kestrel start <name>` reads the
selected agent's registered port. The registry is the one the launched host
loads (`kestrel_sovereign.multi_agent.config.launched_host_registry`): the file
`KESTREL_MULTI_AGENT_CONFIG` names when it exists, else the project's
`multi_agent.toml`. `kestrel terminate` and the deploy gates read the same one,
so a named restart cannot stop and start an agent whose database no gate
judged (#3832). The command intentionally does not accept
direct-server `--host` or `--port` options. Edit the registry when changing a
managed fleet address.

Both forms launch the command built by
`kestrel_sovereign.multi_agent.process_manager.server_command`, which passes
`--timeout-graceful-shutdown` (`SERVER_GRACEFUL_SHUTDOWN_SECONDS`). After a
stop signal uvicorn waits at most that long for open connections, such as a
console's event stream, then cancels them and runs the application's lifespan
shutdown. `kestrel terminate` waits up to `SERVER_TERMINATE_TIMEOUT_SECONDS`
for the process to exit before it sends SIGKILL, and says so when it has to,
then allows `SERVER_KILL_GRACE_SECONDS` before it judges. The restart
coordinator watches a `kestrel restart` it dispatched for as long as that stop
can last, so a restart that fails to stop the host is caught when it exits.
A host with no PID record is stopped through its port, under the same
deadline: uvicorn frees the port before its shutdown has run, so the wait is
for the process, not the port. A host or agent the stop signal could not
reach is reported at once rather than waited for.

`SERVER_TERMINATE_TIMEOUT_SECONDS` is the host's one shutdown budget,
`HOST_SHUTDOWN_BUDGET_SECONDS` (#4028). The lifespan shutdown takes its own
deadline from the same value: `HOST_SHUTDOWN_REMOTE_WORK_SECONDS`, the budget
less `HOST_SHUTDOWN_RESERVE_SECONDS` (the graceful window and the time to
finish after it). Agent shutdown starts no remote snapshot upload. It
records the snapshot as owed unless the database is unchanged since the start
of the last snapshot that reached every target and began with no other
snapshot in flight: a write committed while that snapshot was uploading is a
change, and a snapshot that begins while an earlier one uploads records
nothing, since the earlier upload can land last. The next scheduled
`backup_snapshot` (seeded every four hours) sends an
owed snapshot, even if the database looks unchanged by then. A record
the sync state file cannot take fails the shutdown step `sync-snapshot-owed`,
naming the file, and the agent's cleanup is reported degraded; the state file
keeps what it held before. A
snapshot no backup has sent within `KESTREL_OWED_SNAPSHOT_DEADLINE_S` of the
record (default five hours) is sent by the agent's sync service itself, at its
next start if the agent was stopped by then, so an agent that schedules no
backup still sends it. If a stopping agent is still waiting on an upload some
other task started, such as a scheduled backup in progress, the shutdown
cancels it at that deadline and logs `snapshot to <target> cancelled at
shutdown budget`. This is the fleet's terminal drain on a multi-agent host, or
the server's own join for a single agent, including a join of cleanup an
earlier teardown left running. A GCS upload runs on a thread of its own; the
cancellation shuts down the sockets it is using and joins the thread, so the
upload does not keep the process alive after the lifespan returns. The agent's
local cleanup is then joined as before, so the lifespan
returns before SIGKILL whenever that cleanup and the later phases fit in the
reserve. SIGKILL is left for a process whose local cleanup is wedged.

The direct `python -m kestrel_sovereign.server` command uses the same graceful
window. Each launch first moves a log of `LOG_ROTATION_MAX_BYTES` or more aside
(`host.log` becomes `host.log.1`, keeping `LOG_ROTATION_BACKUPS` generations).
The container entrypoints below do not bound the graceful window yet
([#3791](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3791)).

## Managed deployments

Kestrel's process manager and container entrypoints invoke Uvicorn directly
with explicit `--host` and `--port` values. That is an internal launch boundary,
not a second user CLI:

- Cloud Run and Azure Container Apps inject `PORT`; their entrypoints bind
  `0.0.0.0` and pass that exact port to Uvicorn.
- `kestrel start` passes the bind and port already validated from
  `multi_agent.toml`.
- The live-agent dogfooding runbook uses direct Uvicorn intentionally to mirror
  the managed process byte-for-byte.

Those paths do not parse the module CLI and are unchanged by its precedence
rules. The repo-root `server.py` remains a source-clone compatibility shim and
delegates script execution to this packaged contract rather than maintaining a
second parser.
