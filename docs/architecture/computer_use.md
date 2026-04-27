# Computer Use & File System

Optional capability that lets a sovereign agent read, list, write, and edit files on the user's machine and run shell commands. Disabled by default. Enabling it requires three independent actions — any single one missing, every tool call is refused.

## The three gates

Every tool call in `ComputerUseFeature` runs through these gates in order. The audit log records which gate was the last to allow (or refuse) the call.

| Gate | Where it lives | What enables it |
|---|---|---|
| **Privacy** | `PrivacyConfig.computer_access` | `agent.privacy_config.computer_access = True`. Defaults `False` on every preset (ephemeral … public); never inherited from a preset name. |
| **Constitution** | Book II Amendment IX | The agent's constitution markdown contains a `### Amendment IX` section with `[x]` checked next to the capability name. Parser is strict: only lowercase `[x]` counts. |
| **Approval** | `SecurityFeature.approval_queue` | A human approves the per-call request inside the approval timeout (default 5 min). Reads inside the allow-list auto-approve; writes, edits, and shell always require human approval. |

All three are mandatory. The friction is the feature: if you don't want to flip a checkbox in your constitution, you don't want this capability turned on.

## Backends

Filesystem reads, writes, and listings always run on the host — the agent is editing your files; that is the point. The backend split governs *shell exec*:

- **`docker`** (default) — wraps the existing `kestrel_sovereign/features/compute/executors/docker_executor.py`. Each shell command runs in a fresh container with `--read-only`, `--network=none`, `--security-opt=no-new-privileges`, memory and PID limits. Same blast radius as the long-running compute feature.
- **`local`** — direct `asyncio.subprocess.exec` on the host. Refuses to construct unless Amendment IX grants both `shell_execution_sandboxed` *and* `shell_execution_host`. The two-grant model exists so a single permissive amendment cannot covertly widen sandboxed access into host access.

The backend is selected once at feature init and cannot be swapped at runtime.

## Tools

| Tool | Capability checked | Approval policy |
|---|---|---|
| `!fs-read <path>` | `filesystem_read` | Auto-approve inside allow-list; human approval outside |
| `!fs-list <path>` | `filesystem_read` | Auto-approve inside allow-list; human approval outside |
| `!fs-write <path>` | `filesystem_write` | Always human approval; payload includes diff preview |
| `!shell <cmd>` | `shell_execution_sandboxed` (docker) or `shell_execution_host` (local) | Always human approval; binary deny-list hard-rejects before approval |

A path that matches the deny-list is rejected before the approval queue ever sees it; the approver cannot accidentally widen the deny-list.

## Configuration

```toml
[features.computer_use]
enabled = false                       # opt-in
backend = "docker"                    # docker | local
allowed_paths = ["~/projects", "~/Documents"]
deny_paths = ["~/.ssh", "~/.aws", "~/.config", "~/.gnupg"]
max_read_bytes = 5_000_000
allowed_binaries = ["git", "ls", "cat", "rg", "uv", "node", "python"]
denied_binaries = ["rm", "dd", "mkfs", "shutdown", "sudo", "ssh"]
auto_approve_read = true              # only inside allowed_paths
audit_log_path = ".kestrel/computer_use_audit.jsonl"
```

The feature can also be disabled via `KESTREL_DISABLED_FEATURES=ComputerUseFeature`.

## Audit log

Append-only JSONL at `audit_log_path`. One record per tool call (allowed, denied, or errored), `fsync` per write, also forwarded to the existing `record_tool_usage` feedback hook so the knowledge graph stays in sync.

```json
{
  "ts": "2026-04-27T14:11:02.421Z",
  "agent_did": "did:kestrel:0x07D6…",
  "tool": "fs-write",
  "backend": "docker",
  "args": {"path": "/Users/me/projects/foo/bar.py", "bytes": 1234},
  "allowed_by": ["privacy", "constitution", "approval:once:human"],
  "outcome": "ok",
  "duration_ms": 42,
  "error": null
}
```

## Threat model

**What this mitigates:**

- **Path traversal** — `..` segments rejected before any I/O.
- **Symlink escape** — realpath of any candidate must live under an allow-list root; symlinks pointing outside are rejected.
- **Deny-list bypass** — deny matches short-circuit the approval flow; the approver never sees a deny-listed path or binary.
- **Accidental host shell exec** — the `docker` backend is the default and reuses the compute feature's vetted container flags. Choosing `local` requires a *separate* constitutional grant.
- **Silent capability widening** — adding a capability requires editing the constitution, which changes its hash, which changes the agent's identity record. There is no way to enable this feature without that change being visible in the inception/continuity audit.

**What this does *not* mitigate:**

- A determined operator can edit Amendment IX, sign the new hash, and grant themselves `shell_execution_host`. The audit log is then your evidence trail. This is the design — sovereignty includes the right to widen one's own grants — and the constitutional and approval layers make every step of that decision a recorded action.
- Approval-fatigue. A user who clicks "approve" on everything trains themselves to approve a hostile-looking call. The deny-list is the safety rail; keep the binary and path deny-lists tight.
- Container escape from the docker backend. The container hardening is what `compute/executors/docker_executor.py` provides; this feature inherits whatever guarantees that gives, no more.

## Enabling, end-to-end

1. Open `kestrel.toml`, set `[features.computer_use] enabled = true`, configure `allowed_paths` and `denied_binaries`.
2. Open `kestrel_sovereign/data/KESTREL_CONSTITUTION.md`, find Amendment IX, change `[ ]` to `[x]` for the capabilities you want to grant.
3. If the agent has been inception-anchored to a previous constitution hash, run `!reanchor-constitution <new_hash_prefix>` so identity records reflect the change.
4. Set `agent.privacy_config.computer_access = True` for the session(s) where you want the feature available.
5. The first tool call will trigger an approval prompt unless it's a read inside the allow-list.

## Out of scope

- Browser automation
- Anthropic computer-use beta API (screen / mouse / keyboard)
- Web-based file picker UI
- Cross-platform native daemon packaging
