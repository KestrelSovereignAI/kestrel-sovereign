# Computer Use & File System

Optional capability that lets a sovereign agent read, list, write, and edit files on the user's machine and run shell commands. Disabled by default. Enabling it requires three independent actions — and even then, every tool call is gated.

## The gate sequence

Every tool call runs through the same sequence. The audit log records every stage that passed and the stage that refused (if any) as a chain like `["privacy", "constitution", "path_safety", "policy", "input_validation", "approval:once:human"]`.

| # | Stage | Where it lives | What enables it |
|---|---|---|---|
| 1 | **Readiness** | `[features.computer_use].enabled` in `kestrel.toml`, plus a successfully-constructed backend | Toml file present and feature `enabled = true`; backend constructed without `CapabilityBlocked` |
| 2 | **Privacy** | `PrivacyConfig.computer_access` | `agent.privacy_config.computer_access = True`. Defaults `False` on every preset; never inherited from a preset name. |
| 3 | **Constitution** | Book II Amendment IX | The agent's constitution markdown contains a `### Amendment IX` section with `[x]` checked next to the capability name. Parser is strict: only lowercase `[x]` counts. |
| 4 | **Path safety** | `path_safety.resolve_realpath` | Reject `..` traversal and embedded NUL bytes. Symlinks are *resolved*, not rejected — the realpath is what the human approver later sees. |
| 5 | **Policy** | `PathPolicy.evaluate` / `BinaryPolicy.evaluate` | Allow-list / deny-list. Deny-list always wins. Allow-list match → `ALLOW` (read with `auto_approve_read`) or `REQUIRE_APPROVAL` (write). No match → `REQUIRE_APPROVAL` so the human can vouch. |
| 6 | **Input validation** *(optional)* | A tool's `pre_approval` hook on `_run_gates` | Tools that need to touch the file before approval (diff previews for `fs-write`, occurrence-finding for `fs-edit`) do that work here. **All file I/O for `fs-write` and `fs-edit` happens inside this stage** so an unauthorized caller never triggers a read. Refusals raise `_PreApprovalRefusal` and are audited. |
| 7 | **Approval** | `SecurityFeature.approval_queue` | A human approves the per-call request inside the approval timeout (default 5 min). |

Privacy and constitution come first because they're call-level and input-independent — they decide whether the call is *eligible* at all, without leaking anything about path layout or policy contents. Path-safety and policy are input validation on the path itself; input-validation is content-dependent validation on the file (diff, encoding, substring presence) and only runs after path is authorized.

Two side effects of this order:

- A missing constitutional grant always surfaces as `constitution:Amendment IX missing grant 'X'`, never as `path_safety:...`, `policy:...`, or `input_validation:...`. A probing agent can't map your allow-list, your symlink targets, or whether a substring exists in a file before it has a grant.
- The `fs-edit` substring oracle (which would let an unauthorized agent confirm "is `password=hunter2` in this file?" via the difference between "old_text not found" and other errors) is closed: every gate failure before stage 6 returns the same constitution/privacy/policy error regardless of file contents.

## Backends

Filesystem reads, writes, and listings always run on the host — the agent is editing your files; that is the point. The backend split governs *shell exec*:

- **`docker`** (default) — wraps the existing `kestrel_sovereign/features/compute/executors/docker_executor.py`. Each shell command runs in a fresh container with `--read-only`, `--network=none`, `--security-opt=no-new-privileges`, memory and PID limits. Same blast radius as the long-running compute feature.
- **`local`** — direct `asyncio.subprocess.exec` on the host. Refuses to construct unless Amendment IX grants both `shell_execution_sandboxed` *and* `shell_execution_host`. The two-grant model exists so a single permissive amendment cannot covertly widen sandboxed access into host access.

The backend is selected once at feature init and cannot be swapped at runtime.

## Tools

| Tool | Capability checked | Approval policy |
|---|---|---|
| `!fs-read <path>` | `filesystem_read` | Auto-approve inside allow-list; human approval outside (the human sees the resolved realpath, including symlink targets) |
| `!fs-list <path>` | `filesystem_read` | Auto-approve inside allow-list; human approval outside |
| `!fs-write <path>` | `filesystem_write` | Always human approval; payload includes a unified-diff preview |
| `!fs-edit <path> <old> <new>` | `filesystem_write` | Always human approval; replaces the Nth occurrence of `old_text` with `new_text`; payload includes a unified-diff preview |
| `!shell <cmd>` | `shell_execution_sandboxed` (docker) or `shell_execution_host` (local) | Always human approval; binary deny-list hard-rejects before approval |

A path or binary that matches the deny-list is rejected before the approval queue ever sees it; the approver cannot accidentally widen the deny-list.

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

The toml is looked up in this order: (1) the agent's `storage_path` directory, (2) `<storage_path>/..`, (3) walking up from the source file to the repo root. Per-agent kestrel.toml under `agent_data/<name>/` is the typical home.

The feature can also be disabled via `KESTREL_DISABLED_FEATURES=ComputerUseFeature`.

## Audit log

Append-only JSONL at `audit_log_path`. **One record per tool call**, regardless of whether it succeeded, was denied at any stage, or errored during execution. `fsync` per write. Also forwarded to the existing `record_tool_usage` feedback hook so the knowledge graph stays in sync.

```json
{
  "ts": "2026-04-27T14:11:02.421Z",
  "agent_did": "did:kestrel:0x07D6…",
  "tool": "fs-write",
  "backend": "docker",
  "args": {"path": "/Users/me/projects/foo/bar.py", "bytes": 1234, "diff_preview": "..."},
  "allowed_by": ["privacy", "constitution", "path_safety", "policy", "approval:once:human"],
  "outcome": "ok",
  "duration_ms": 42,
  "error": null
}
```

Denials append `denied:<stage>` to `allowed_by` and set `outcome="denied"`:

```json
{"allowed_by": ["denied:privacy"], "outcome": "denied"}
{"allowed_by": ["privacy", "denied:constitution"], "outcome": "denied"}
{"allowed_by": ["privacy", "constitution", "denied:path_safety"], "outcome": "denied", "error": "..."}
{"allowed_by": ["privacy", "constitution", "path_safety", "denied:policy"], "outcome": "denied"}
{"allowed_by": ["privacy", "constitution", "path_safety", "policy", "denied:input_validation:encoding"], "outcome": "denied"}
{"allowed_by": ["privacy", "constitution", "path_safety", "policy", "denied:input_validation:missing_text"], "outcome": "denied"}
{"allowed_by": ["privacy", "constitution", "path_safety", "policy", "input_validation", "denied:approval:denied"], "outcome": "denied"}
```

Reading those rows back is the canonical way to reconstruct what happened.

## Threat model

**What this mitigates:**

- **Path traversal** — `..` segments and embedded NUL bytes rejected before any I/O.
- **Symlink escape** — realpath resolution happens *before* the human approver sees the request. A symlink inside an allow-listed directory pointing to `/etc/passwd` is shown to the approver as `/etc/passwd`, not as the symlink path.
- **Deny-list bypass** — deny-list matches short-circuit the approval flow; the approver never sees a deny-listed path or binary.
- **Accidental host shell exec** — the `docker` backend is the default and reuses the compute feature's vetted container flags. Choosing `local` requires a *separate* constitutional grant.
- **Silent capability widening** — adding a capability requires editing the constitution, which changes its hash, which changes the agent's identity record. There is no way to enable this feature without that change being visible in the inception/continuity audit.
- **Information disclosure via gate ordering** — privacy and constitution stages run before any input is examined. An agent without a constitutional grant cannot probe path layout or policy contents through error messages.
- **Pre-gate file I/O leak** — `fs-write` and `fs-edit` defer all file reads (existence checks, size, encoding, substring search, diff preview) into a `pre_approval` hook that runs only after privacy/constitution/path-safety/policy authorize the call. An unauthorized caller cannot use these tools as an oracle to probe file existence, readability, encoding, or substring presence.

**What this does *not* mitigate:**

- A determined operator can edit Amendment IX, sign the new hash, and grant themselves `shell_execution_host`. The audit log is then your evidence trail. This is the design — sovereignty includes the right to widen one's own grants — and the constitutional and approval layers make every step of that decision a recorded action.
- Approval-fatigue. A user who clicks "approve" on everything trains themselves to approve a hostile-looking call. The deny-list is the safety rail; keep the binary and path deny-lists tight.
- Container escape from the docker backend. The container hardening is what `compute/executors/docker_executor.py` provides; this feature inherits whatever guarantees that gives, no more.

## Enabling, end-to-end

1. Open the agent's `kestrel.toml` (typically `agent_data/<name>/kestrel.toml`), set `[features.computer_use] enabled = true`, configure `allowed_paths` and `denied_binaries`.
2. Open `kestrel_sovereign/data/KESTREL_CONSTITUTION.md`, find Amendment IX, change `[ ]` to `[x]` for the capabilities you want to grant.
3. If the agent has been inception-anchored to a previous constitution hash, run `!reanchor-constitution <new_hash_prefix>` so identity records reflect the change.
4. Set `agent.privacy_config.computer_access = True` for the session(s) where you want the feature available.
5. The first tool call will trigger an approval prompt unless it's a read inside the allow-list.

## Out of scope

- Browser automation
- Anthropic computer-use beta API (screen / mouse / keyboard)
- Web-based file picker UI
- Cross-platform native daemon packaging
