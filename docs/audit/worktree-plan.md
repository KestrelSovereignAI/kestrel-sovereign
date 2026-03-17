# Audit Worktree Plan

Use dedicated worktrees to keep domain audits isolated and parallelizable.

Do not place full worktree checkouts inside the repository root. Nested worktrees
pollute `git status`, confuse repo-wide tooling, and can cause duplicate scans.

Suggested worktrees:

- `/tmp/kestrel-sovereign-worktrees/audit-foundation` on branch `codex/audit-foundation`
- `/tmp/kestrel-sovereign-worktrees/audit-runtime` on branch `codex/audit-runtime`
- `/tmp/kestrel-sovereign-worktrees/audit-security` on branch `codex/audit-security`
- `/tmp/kestrel-sovereign-worktrees/audit-platform` on branch `codex/audit-platform`

Domain split:

- `audit-foundation`: constitution, identity, sovereignty, storage, memory
- `audit-runtime`: agent loop, context, commands, tools, A2A, observability
- `audit-security`: privacy, permissions, keys, webhooks, auth, adversarial suites
- `audit-platform`: LLM providers, model mandate, APIs, UI, CLI, deployment

Rules:

- Keep each worktree aligned to one audit domain at a time.
- Do not mix unrelated fixes in the same branch.
- Put the failing proof in the same branch as the root-cause fix.
- If two domains touch the same source of truth, stop and resolve ownership before editing.
