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
selected agent's registered port. The command intentionally does not accept
direct-server `--host` or `--port` options. Edit the registry when changing a
managed fleet address.

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
