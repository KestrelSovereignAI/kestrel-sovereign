# Approval paths

Closes #1582. Reference for operators and future agents on the three independent approval systems an LLM tool call can ride, and which paths auto-mode covers today.

## TL;DR

| Path | Triggered by | Auto-mode covers? | Audit surface |
|---|---|---|---|
| **1. Codex-native sandbox shell** | LLM uses codex's built-in `shell` tool | **Yes** (since #1575 / PR #1583) | `codex_decline_events` (since #1581 / PR #1588), `security_audit_log` via the bridge's queue |
| **2. Kestrel internal `computer_use` gate** | A direct `cu.shell()` method call from inside a feature (e.g., `talon_file_and_claim`) | Yes — by **both** the global auto-mode short-circuit AND the scoped `[[security.auto_approve.shell]]` allowlist (see flow below) | `auto_approve_audit` (scoped path) + `security_audit_log` (auto-mode path) |
| **3. Kestrel LLM tool call (`ComputerUseFeature.shell`)** | LLM advertises and calls the `ComputerUseFeature.shell` tool | Yes — `PermissionLevel.AUTO` via auto-mode | `SecurityHook` PRE_TOOL_USE → `security_audit_log` |

> **Note on doc/code sync:** the operator-facing statements about `codex_decline_events` and the operational-state-block decline summary describe shipped behavior **after #1581 (PR #1588) merges into main**. This doc PR is intentionally written for the post-merge state; if you're reading it before #1588 lands, those claims are accurate for that PR's branch but not yet for main.

If a tool call doesn't go through one of these three, no kestrel-side audit will surface it. There is no fourth path today.

---

## Background

Until #1575 (PR #1583) merged on 2026-06-07, **Path 1 was silently declined** by the hardcoded `_DEFAULT_APPROVAL_REPLIES` table in [`kestrel_sovereign/llm/codex_app_server.py`](../../kestrel_sovereign/llm/codex_app_server.py). An auto-mode agent could try `gh issue create` via codex's native shell and get back `Rejected("rejected by user")` even though no human had decided anything and her `ComputerUseFeature.shell` permission row was `'allow'`. Emma surfaced this in her 2026-06-06 orchestration session; the docstring in [`kestrel_sovereign/security/auto_approve.py:35-44`](../../kestrel_sovereign/security/auto_approve.py) had only ever documented paths 2 vs 3.

This page exists so neither the Sovereign nor a future agent is surprised by which gate fires for which shape of tool call.

---

## Path 1: Codex-native sandbox shell

**Entry point:** the LLM, mid-turn through `openai:plan` (the codex app-server route), uses codex's built-in `shell` tool — NOT `ComputerUseFeature.shell`.

**Flow:**

1. Codex's app-server runs in `sandbox="read-only"` by default ([`codex_adapter.py:_ensure_thread`](../../kestrel_sovereign/llm/codex_adapter.py)). Any write/exec via the native shell triggers a server→client RPC: `item/commandExecution/requestApproval`. File mutations trigger `item/fileChange/requestApproval`.
2. The bridge handler registered by `CodexAdapter._make_codex_approval_handler` (#1575) intercepts these RPCs at the (`method`, `thread_id=None`) global slot.
3. The handler runs two gates in order:
   - **Policy gate** — `_codex_policy_precheck` runs `BinaryPolicy.evaluate(argv)` for commandExecution and `PathPolicy.evaluate(path, write=True)` for fileChange. Hard-DENY → `{"decision": "decline"}` immediately. Without this, auto-mode could blanket-approve any binary or path.
   - **ApprovalQueue gate** — `agent.security_feature.approval_queue.request_approval(feature_name="codex_native", tool_name=<kind>, tool_args=params, timeout=120s)`. In auto mode with a non-DENY permission row, this returns `(True, "auto")` silently. Otherwise the request goes through the normal modal/queue.
4. Approve → `{"decision": "accept"}`. Decline → `{"decision": "decline"}` + a typed `codex_decline_events` row (#1581) so the agent sees it on her next turn under the `--- OPERATIONAL STATE ---` block.

**Auto-mode covers** this path AFTER #1575. Before #1575 it was silently declined regardless of permission level.

**Failure modes that still decline (fail-closed by design):**
- Missing `ComputerUseFeature` (no policy gate available).
- Missing `_binary_policy` / `_path_policy`.
- Empty or malformed approval params (no command / no path / non-dict entry / missing argv).
- `policy.evaluate` raises.
- Any unexpected exception in the bridge.

All of these are surfaced as `codex_decline_events` rows with descriptive `reason` codes (`policy_deny:<rule>`, `no_binary_policy`, `no_command_in_params`, `binary_eval_failed:...`, etc.). Emma's next turn sees a one-line summary in the operational state block.

**One RPC is STILL hardcoded to decline by `_DEFAULT_APPROVAL_REPLIES`** that the bridge intentionally doesn't cover: `mcpServer/elicitation/request` (`{"action": "decline"}`). The other two non-bridged RPCs in the default table return *non-decline* shapes — `item/permissions/requestApproval` returns `{"permissions": {}, "scope": "turn"}` (empty grant, single-turn scope) and `item/tool/requestUserInput` returns `{"answers": {}}` (empty answers). Those aren't declines so they don't surface as decline events.

When an audit agent is attached, the `mcpServer/elicitation/request` decline IS surfaced as a `reason="auto_default"` row in `codex_decline_events` (#1581). The audit recorder handles both decline reply shapes — `{"decision": "decline"}` (used by `commandExecution`/`fileChange` when the bridge isn't attached, e.g., tests or headless callers) AND `{"action": "decline"}` — so it would also catch any future RPC the default table is extended to cover.

---

## Path 2: Kestrel internal `computer_use` gate

**Entry point:** a feature calls `cu.shell(...)` directly as a Python method (NOT as an LLM tool). The chief example is `talon_file_and_claim` ([`kestrel_sovereign/features/talon/coordinator.py`](../../kestrel_sovereign/features/talon/coordinator.py)) for `gh issue create`.

**Flow:**

1. `ComputerUseFeature.shell(argv=...)` runs `BinaryPolicy.evaluate(argv)` directly.
2. `REQUIRE_APPROVAL` → `ApprovalQueue.request_approval(feature_name="computer_use", tool_name="shell", tool_args={"argv": [...]})`.
3. Inside `request_approval`, the order is: **(a)** explicit DENY in the permission store → hard stop; **(b)** global auto-mode + non-DENY permission level → `(True, "auto")` immediately, logged as `decision="auto_mode_allowed"` in `security_audit_log` ([`approval_queue.py:217-234`](../../kestrel_sovereign/features/security/approval_queue.py)); **(c)** scoped `[[security.auto_approve.shell]]` allowlist match → `(True, f"auto_approve:{audit_id}")` with the full row in `auto_approve_audit` ([`approval_queue.py:242-321`](../../kestrel_sovereign/features/security/approval_queue.py)). The scoped allowlist matches ONLY on the `argv` shape — the internal-gate signature — because the LLM tool path (#3) passes `{"command": ...}` instead.
4. The audit id from the scoped path rides the `allowed_by` chain so `computer_use._audit_run` finalizes the exit code.
5. No path matched → human approval. Permission policy DENY → silent denial (logged).

**Auto-mode covers** this path via either the global short-circuit (`PermissionLevel.AUTO`) OR the scoped allowlist, with the scoped path remaining the only one that produces an `auto_approve_audit` row tied to the command/repo for fine-grained operator review.

**Why the `argv` discrimination matters:** the internal gate is the only auto-approve surface that gets to write+finalize a full audit row (command, agent DID, timestamp, exit code). The LLM-tool path can't thread an audit id through to a finalizer, so auto-approving there would either dangle an unfinalizable audit row or require POST_TOOL_USE finalization machinery. Designed boundary, not an oversight.

---

## Path 3: Kestrel LLM tool call (`ComputerUseFeature.shell`)

**Entry point:** the LLM advertises `ComputerUseFeature.shell` (or any other `ComputerUseFeature` tool) and the orchestrator dispatches it.

**Flow:**

1. PRE_TOOL_USE hooks fire. `SecurityHook.execute` ([`kestrel_sovereign/features/security/hooks.py`](../../kestrel_sovereign/features/security/hooks.py)) looks up `PermissionLevel`.
2. `PermissionLevel.ALLOW` → silent allow (audit row written, no modal).
3. `PermissionLevel.AUTO` (the auto-mode promotion of any non-DENY stored level) → silent allow with `decision="auto_mode_allowed"`.
4. `PermissionLevel.DENY` → silent deny with `decision="auto_denied"`.
5. `PermissionLevel.ASK` / `SESSION` → queue and wait for the modal.
6. The downstream `ComputerUseFeature.shell` then runs the same `BinaryPolicy` / `PathPolicy` check internally — failure-modes there mirror Path 2's gates.

**Auto-mode covers** this path via the `_global_auto_mode and level != PermissionLevel.DENY → PermissionLevel.AUTO` promotion in [`features/security/permissions.py:432-437`](../../kestrel_sovereign/features/security/permissions.py).

The LLM-tool path is what an agent should prefer when an equivalent kestrel-side tool exists. For `gh` specifically, `kestrel-feature-github`'s `create_issue` (over `httpx`) is the supported alternative; it goes through Path 3 (the feature pipeline) and bypasses `BinaryPolicy` entirely. See #1579.

---

## Operator runbook

### Enable auto-approve for a specific command pattern (Path 2)

```toml
[[security.auto_approve.shell]]
pattern = "^gh issue (create|comment) -R KestrelSovereignAI/kestrel-sovereign(?:\\s|$)"
repo_scope = "KestrelSovereignAI/kestrel-sovereign"
audit = true
agent = "Emma"   # optional
```

The pattern matches the `argv` shape (Path 2 only). LLM-driven `ComputerUseFeature.shell` calls (Path 3) ride the `PermissionLevel` system instead.

### Verify auto-mode coverage end-to-end

```bash
kestrel ask Emma --json "what are your enabled features?"
```

Look for `github` (the kestrel-feature-github surface) if you want Path 3 routing for GitHub writes. If absent, enable per #1579's runbook.

### Inspect declines

Path 1 declines: `SELECT * FROM codex_decline_events WHERE agent_id = ? ORDER BY created_at DESC LIMIT 20`. Surfaced automatically in the next turn's `--- OPERATIONAL STATE ---` block.

Path 2/3 decisions: `SELECT * FROM security_audit_log WHERE feature_name = ?`.

---

## Cross-reference

- [#1575](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1575) — bridged the codex sandbox-approval RPCs through `ApprovalQueue` (Path 1 auto-mode coverage).
- [#1581](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1581) — typed `codex_decline_events` table + operational state block summary.
- [#1579](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1579) — `kestrel-feature-github`'s `create_issue` over httpx (alternative to Path 1/2 shelling for GitHub writes).
- [auto_approve.py docstring](../../kestrel_sovereign/security/auto_approve.py) — the original honest-but-incomplete write-up that documented Paths 2 vs 3.
