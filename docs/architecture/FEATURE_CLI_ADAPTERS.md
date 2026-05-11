# Feature-Owned CLI Adapters

> Status: **Active**. Introduced by [#1185](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1185),
> extended by [#1192](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1192),
> and piloted for local `git` inspection after [#1206](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/1206).

Feature-owned CLI adapters let Kestrel features use installed command-line
tools through a shared terminal substrate while reusing the user's existing
CLI authentication. GitHub PR review through `gh` and local repository
inspection through `git` are the reference adapters, but the pattern is
intentionally broader: cloud CLIs, package-manager CLIs, and coding CLIs can
all use the same contract.

This is not arbitrary shell access. A feature exposes explicit adapter
methods, each method builds an argument vector for a known command, and tools
call those methods through normal `@tool` surfaces.

## Core Shape

The shipped code lives under [`kestrel_sovereign/features/cli/`](../../kestrel_sovereign/features/cli/):

- [`terminal.py`](../../kestrel_sovereign/features/cli/terminal.py) provides
  `TerminalExecutionService`, command result models, output caps, redaction
  helpers, and `CliRisk`.
- [`adapters.py`](../../kestrel_sovereign/features/cli/adapters.py) contains
  `FeatureCliAdapter` plus the GitHub reference adapter.
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
class GitHubCliAdapter(FeatureCliAdapter):
    adapter_id = "github"
    tools = (CliToolDeclaration("gh", required=True),)
    commands = (
        CliCommandDefinition(
            "github.pr_view",
            "Read pull request metadata, files, commits, and status rollup.",
            CliRisk.READ_ONLY,
            ("gh",),
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
    argv=["gh", "pr", "view", "--repo", repo, str(number), "--json", fields],
    risk=CliRisk.READ_ONLY,
    command_id="github.pr_view",
)
```

Bad:

```python
TerminalCommandRequest(argv=["sh", "-c", f"gh pr view {number} --repo {repo}"])
```

If a CLI needs complex composition, build that composition inside Python and
pass the final values as argv elements.

### Validate every user-controlled argument

Every value that can come from a user, model, command prefix, or third-party
payload must be validated before argv construction. Validation belongs at the
adapter boundary, not only in the `CliFeature` tool method.

The GitHub adapter validates:

- repo names as exactly `owner/name`
- PR numbers as positive integers
- repository file paths as relative paths without empty, `.`, or `..` segments
- refs as non-empty strings without NUL bytes

New adapters should add equivalent validators for their own command grammar.
For example, a future `git` adapter should validate refs, pathspecs, revision
ranges, and working directories before constructing argv.

### Parse structured output before redaction

For commands that return JSON, parse raw stdout first and then recursively
redact parsed values. Regex redaction can corrupt JSON if it runs before
parsing.

Pattern:

```python
payload = _json_or_raise(result)
return redact_json(payload)
```

For plain text output, return `result.redacted_stdout` or
`result.redacted_stderr`.

### Treat truncation as non-authoritative

`TerminalExecutionService` caps stdout/stderr while reading process pipes. If a
command result has `truncated_stdout` or `truncated_stderr`, adapter methods
must not present that output as complete evidence.

For authoritative outputs such as diffs, JSON payloads, status checks, and file
contents, reject truncation:

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
- `redact_json()`
- `redact_secrets()` for decoded file contents

This includes failure paths. The auth-status tool is a useful example: the
structured data and the partial error string both use redacted stderr.

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

The initial shipped adapters expose read-only commands only. The terminal
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

`check_availability()` should detect installed executables and versions. It
must not read credential stores directly or print secrets. Prefer native status
commands such as `gh auth status` for authentication checks.

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
   adapter test module.
8. Run focused tests and compile checks.

## Test Checklist

Every adapter should cover:

- command construction uses the expected argv list
- command metadata uses the correct `CliRisk`
- success path parses and returns structured data
- non-zero command exit returns a useful redacted error
- invalid JSON or malformed output fails clearly
- stdout/stderr truncation is rejected or explicitly marked as a preview
- token-like stdout, stderr, JSON fields, diffs, and decoded file contents are
  redacted
- every user-controlled argument rejects option-like values and traversal-shaped
  values when relevant
- command-prefix positional parsing works for every exposed `!command`
- non-read-only commands fail closed without approval and do not spawn
- missing or malformed CLI payloads fail closed

## Reference Adapters

The GitHub adapter exposes the current reference pattern:

- `github_cli_auth_status`
- `github_pr_view`
- `github_pr_diff`
- `github_pr_files`
- `github_pr_checks`
- `github_read_file_at_ref`
- `github_read_file_at_pr_head`
- `github_pr_review_context`

These tools let an agent inspect real PR heads, changed files, diffs, check
rollups, and bounded file contents through the user's authenticated `gh`
session. That closes the original PR-review gap without giving the model a
general-purpose terminal.

The local `git` adapter pilots the same pattern for read-only repository
inspection:

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

Mutating commands such as `git fetch`, `gh pr merge`, cloud deploys, or package
publishes can now reuse the substrate approval gate, but each adapter still
needs command-specific validation, focused approval payloads, and tests that
prove the command cannot run before approval.
