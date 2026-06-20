---
type: Architecture Spec
title: Feature-Owned CLI Adapters
description: 'Status: **Active**. Introduced by [#1185](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1185),
  extended by [#1192](https://github.com/KestrelSovereignAI/kestrel-s...'
resource: /docs/architecture/FEATURE_CLI_ADAPTERS.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# Feature-Owned CLI Adapters

> Status: **Active**. Introduced by [#1185](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1185),
> extended by [#1192](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1192),
> and refocused on local `git` inspection after [#1206](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1206).
>
> **GitHub access does not belong here.** The `gh`-based GitHub adapter that
> originally shipped in this feature was removed: GitHub is a remote integration,
> not a local CLI workflow, and core must not carry `gh` tooling. GitHub access
> lives in the optional `kestrel-feature-github` package (httpx against
> `api.github.com`), discovered through the `kestrel_sovereign.features`
> entry-point registry. See
> [GITHUB_FEATURE_DESIGN.md](GITHUB_FEATURE_DESIGN.md).

Feature-owned CLI adapters let Kestrel features use installed command-line
tools through a shared terminal substrate while reusing the user's existing
CLI authentication. Read-only local repository inspection through `git` is the
reference adapter, but the pattern is intentionally broader: cloud CLIs,
package-manager CLIs, and coding CLIs can all use the same contract.

This is not arbitrary shell access. A feature exposes explicit adapter
methods, each method builds an argument vector for a known command, and tools
call those methods through normal `@tool` surfaces.

## Core Shape

The shipped code lives under [`kestrel_sovereign/features/cli/`](../../kestrel_sovereign/features/cli/):

- [`terminal.py`](../../kestrel_sovereign/features/cli/terminal.py) provides
  `TerminalExecutionService`, command result models, output caps, redaction
  helpers, and `CliRisk`.
- [`adapters.py`](../../kestrel_sovereign/features/cli/adapters.py) contains
  `FeatureCliAdapter` plus the local `git` reference adapter.
- [`feature.py`](../../kestrel_sovereign/features/cli/feature.py) exposes
  user/agent tools via `CliFeature`.

The relationship is:

```text
CliFeature tool
  -> FeatureCliAdapter method
    -> TerminalExecutionService.run(TerminalCommandRequest)
      -> executable argv, no shell interpolation
```

Adapters declare their executables and registered command metadata:

```python
class GitCliAdapter(FeatureCliAdapter):
    adapter_id = "git"
    tools = (CliToolDeclaration("git", required=True),)
    commands = (
        CliCommandDefinition(
            "git.status",
            "Read local repository status using `git status --short --branch`.",
            CliRisk.READ_ONLY,
            ("git",),
        ),
    )
```

## Design Rules

### Use argv, never shell strings

Always call `TerminalExecutionService.run()` with `TerminalCommandRequest.argv`
as a list. Do not use shell interpolation, `shell=True`, shell pipes, shell
redirection, or ad hoc string concatenation.

Good:

```python
TerminalCommandRequest(
    argv=["git", "--no-optional-locks", "-C", repo_path, "diff", ref],
    risk=CliRisk.READ_ONLY,
    command_id="git.diff",
)
```

Bad:

```python
TerminalCommandRequest(argv=["sh", "-c", f"git -C {repo_path} diff {ref}"])
```

If a CLI needs complex composition, build that composition inside Python and
pass the final values as argv elements.

### Validate every user-controlled argument

Every value that can come from a user, model, command prefix, or third-party
payload must be validated before argv construction. Validation belongs at the
adapter boundary, not only in the `CliFeature` tool method.

The `git` adapter validates:

- refs as non-empty strings matching a safe ref grammar, rejecting leading `-`,
  NUL bytes, and revision ranges (`..`)
- pathspecs as relative paths without empty, `.`, or `..` segments, and not
  shaped like a command option
- local repository paths, resolved and constrained to allowed roots (see below)
- `max_count` as a positive integer, capped before use

New adapters should add equivalent validators for their own command grammar.

### Parse structured output before redaction

For commands that return JSON, parse raw stdout first and then recursively
redact the parsed values. Regex redaction can corrupt JSON if it runs before
parsing. Build a small adapter-local helper that raises on non-zero exit,
rejects truncation, and parses, then walk the parsed structure applying
`redact_secrets()` to string leaves.

For plain text output (the `git` adapter's case), return
`result.redacted_stdout` or `result.redacted_stderr` directly.

### Treat truncation as non-authoritative

`TerminalExecutionService` caps stdout/stderr while reading process pipes. If a
command result has `truncated_stdout` or `truncated_stderr`, adapter methods
must not present that output as complete evidence.

For authoritative outputs such as diffs, file contents, and status reads,
reject truncation:

```python
if result.truncated_stdout or result.truncated_stderr:
    raise CliAdapterError("command output exceeded the capture limit")
```

If a future tool intentionally returns a preview, name it as a preview and
include `truncated: true` in the data.

### Redact all user-visible command output

Do not return raw stdout/stderr in `ToolResult`, exception messages, logs, or
confirmation text. Use:

- `result.redacted_stdout`
- `result.redacted_stderr`
- `redact_secrets()` for any decoded or recomposed text

This includes failure paths: adapter error messages use redacted stderr.

### Keep risk explicit

Every registered command has a `CliRisk`:

- `READ_ONLY`
- `LOCAL_MUTATION`
- `REMOTE_MUTATION`
- `DESTRUCTIVE`
- `CREDENTIAL_AFFECTING`
- `EXTERNAL_TRANSMISSION`
- `FINANCIAL_OR_BILLING`
- `UNKNOWN`

The shipped `git` adapter exposes read-only commands only. The terminal
substrate fails closed for every non-`READ_ONLY` risk unless an explicit
approval callback approves the exact command request. `CliFeature` wires that
callback to `SecurityFeature.approval_queue` when the security feature is
available.

Approval payloads must be safe to display in the UI before execution. The
shared `CliFeature` approval callback sends a conservative argv summary:
command name and option names are shown, ordinary values become `[ARG]`, and
sensitive option values become `[REDACTED]`.

Do not smuggle mutation into a command declared as `READ_ONLY`. Future mutating
commands must use the correct risk class so the substrate can pause for
approval before process execution.

### Keep availability checks diagnostic

`check_availability()` should detect installed executables and versions via
`which`. It must not read credential stores directly or print secrets.

## Adding a New Adapter

1. Add adapter methods in `kestrel_sovereign/features/cli/adapters.py`, or move
   adapters into a package if the file becomes crowded.
2. Declare required executables with `CliToolDeclaration`.
3. Declare every callable command with `CliCommandDefinition`.
4. Add adapter methods that validate inputs, run argv commands, parse output,
   reject truncation, and redact returned data.
5. Expose feature tools from `CliFeature` in `feature.py`.
6. Add the tool names to `component.yaml` and
   `kestrel_sovereign/data/feature_registry.toml`.
7. Add unit tests in `tests/unit/test_cli_adapter_feature.py` or a dedicated
   adapter test module. Cover the real executable end-to-end (see the Test
   Checklist), not just argv shape against a mocked terminal.
8. Run focused tests and compile checks.

Adapters here are for **local** CLI tools. A capability that talks to a remote
service over the network (GitHub, cloud providers, package indexes) belongs in
its own feature package that calls the service API directly — not in a `gh`/CLI
wrapper inside core.

## Test Checklist

Every adapter should cover:

- the real executable runs end-to-end against a fixture (e.g. a temp git repo),
  not only argv construction against a mocked terminal
- command construction uses the expected argv list
- command metadata uses the correct `CliRisk`
- success path parses and returns structured data
- non-zero command exit returns a useful redacted error
- invalid or malformed output fails clearly
- stdout/stderr truncation is rejected or explicitly marked as a preview
- token-like stdout, stderr, and decoded contents are redacted
- every user-controlled argument rejects option-like values and traversal-shaped
  values when relevant
- command-prefix positional parsing works for every exposed `!command`
- non-read-only commands fail closed without approval and do not spawn
- missing or malformed CLI payloads fail closed

## Reference Adapter

The local `git` adapter is the current reference pattern for read-only
repository inspection:

- `git_status`
- `git_diff`
- `git_log`
- `git_show_file`
- `git_merge_base`

These tools validate local repository paths, refs, and pathspecs before
constructing argv. Pathspecs are separated with `--` where the command grammar
allows it, and revision ranges are rejected for the initial read-only surface.
The adapter also disables optional git locks and external diff/textconv hooks
for diff reads. Local repository paths are constrained to the current process
working directory plus any roots configured through
`KESTREL_CLI_ALLOWED_REPO_ROOTS`.

## Future Work

Mutating commands such as `git fetch`, cloud deploys, or package publishes can
reuse the substrate approval gate, but each adapter still needs
command-specific validation, focused approval payloads, and tests that prove
the command cannot run before approval.
