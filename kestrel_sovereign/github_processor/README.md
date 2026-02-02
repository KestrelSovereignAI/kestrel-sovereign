# GitHub Ticket Processor

Autonomous GitHub issue processing using Claude Agent SDK. Analyzes issues, asks clarifying questions, implements changes, and creates pull requests.

## Quick Start

```bash
# Process a specific issue
uv run kestrel-github claim --repo owner/repo --issue 42

# Use isolated worktree (recommended for parallel processing)
uv run kestrel-github claim --repo owner/repo --issue 42 --worktree

# Process all 'enhancement' issues
uv run kestrel-github claim --repo owner/repo --label enhancement

# Dry run (no changes)
uv run kestrel-github claim --repo owner/repo --dry-run
```

### Parallel Processing with Worktrees

The `--worktree` flag creates isolated git worktrees for each issue, allowing
multiple agents to work on different issues simultaneously without conflicts:

```bash
# Terminal 1: Process issue 42 in its own worktree
uv run kestrel-github claim --repo owner/repo --issue 42 --worktree

# Terminal 2: Process issue 43 in a separate worktree (parallel)
uv run kestrel-github claim --repo owner/repo --issue 43 --worktree

# Terminal 3: Process issue 44 in another worktree (parallel)
uv run kestrel-github claim --repo owner/repo --issue 44 --worktree
```

Worktrees are created in the parent directory by default. Use `--worktree-base`
to specify a different location.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        ISSUE SUBMITTED                          │
│                    (with 'enhancement' label)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLARIFICATION PHASE                          │
│                                                                 │
│  Agent analyzes issue for clarity:                              │
│  • Are requirements specific?                                   │
│  • Is scope well-defined?                                       │
│  • Are there ambiguous choices?                                 │
│                                                                 │
│  Label: agent-analyzing                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      NEEDS CLARITY      │     │     READY TO BUILD      │
│                         │     │                         │
│  Posts checkbox survey: │     │  Proceeds directly to   │
│  - Option A             │     │  implementation         │
│  - Option B             │     │                         │
│  - Other: ___           │     │  (or skip with          │
│                         │     │   'agent-ready' label)  │
│  Label: agent-clarifying│     │                         │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              │ PM answers questions          │
              │ + reacts 👍 or comments       │
              │ "ready"                       │
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION PHASE                         │
│                                                                 │
│  1. Create branch: issue-{number}-{slug}                        │
│  2. Run Claude Agent SDK (autonomous mode)                      │
│  3. Make code changes, run tests                                │
│  4. Commit with issue reference                                 │
│                                                                 │
│  Label: agent-claimed                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CI PHASE                                │
│                                                                 │
│  1. Push branch to remote                                       │
│  2. Wait for CI checks                                          │
│  3. If CI fails → attempt fix (up to 3 retries)                 │
│  4. If still failing → mark blocked                             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│       CI PASSED         │     │       CI FAILED         │
│                         │     │                         │
│  Create PR              │     │  Post failure details   │
│  Link to issue          │     │  Request human review   │
│                         │     │                         │
│  Label: agent-complete  │     │  Label: agent-blocked   │
└─────────────────────────┘     └─────────────────────────┘
```

## Labels

| Label | Meaning |
|-------|---------|
| `agent-analyzing` | Agent is reviewing the issue for clarity |
| `agent-clarifying` | Agent posted questions, waiting for answers |
| `agent-ready` | Issue is pre-approved, skip clarification |
| `agent-claimed` | Agent is actively implementing |
| `agent-blocked` | Agent needs human help |
| `agent-complete` | PR created, ready for review |

## Clarification Questions

When an issue is vague, the agent posts structured questions with checkboxes:

```markdown
## Clarification Needed

Before I begin implementation, I need to clarify a few things:

### 1. Which caching backend should we use?

- [ ] Redis (recommended for production)
- [ ] In-memory (simpler, good for development)
- [ ] HTTP cache headers only
- [ ] Other: ___

### 2. What endpoints need caching?

- [ ] All GET endpoints
- [ ] High-traffic endpoints only
- [ ] Other: ___

---
*React with :+1: or comment 'ready' when answered.*
```

To continue:
1. Check your preferred options
2. React with 👍 on the comment OR reply "ready"
3. Agent will resume implementation

## Skip Clarification

For well-defined issues, skip the clarification phase:

1. **Add `agent-ready` label** to the issue before processing
2. **Use `--skip-clarification` flag**: `kestrel-github claim --repo owner/repo --skip-clarification`

## GitHub Issue Templates

Use the included issue templates for structured requests:

- **Feature Request** (`feature-request.yml`) - Structured feature proposals
- **Bug Report** (`bug-report.yml`) - Bug reports with reproduction steps
- **Agent Task** (`agent-task.yml`) - Pre-clarified tasks ready for implementation

Templates are in `.github/ISSUE_TEMPLATE/`.

## Environment Variables

```bash
# Required
GITHUB_TOKEN=ghp_...           # GitHub PAT with repo access
ANTHROPIC_API_KEY=sk-ant-...   # Claude API key

# Optional
GITHUB_HUMAN_REVIEWER=username # Human reviewer for blocked issues
GITHUB_BOT_ASSIGNEE=claude-bot # Bot username (legacy mode)
```

## Configuration

Key settings in `config.py`:

```python
# Processing
max_turns: int = 50           # Max agent iterations
max_ci_retries: int = 3       # CI fix attempts
ci_timeout: int = 600         # CI wait timeout (seconds)

# Behavior
skip_clarification: bool = False    # Skip clarification phase
create_draft_pr: bool = False       # Create draft PRs
post_plan_comment: bool = True      # Post implementation plan
```

## Architecture

```
github_processor/
├── __init__.py
├── README.md              # This file
├── agent_runner.py        # Claude Agent SDK integration
│   ├── AgentRunner        # Main agent executor
│   ├── analyze_issue()    # Clarification analysis
│   ├── run()              # Full implementation
│   └── run_fix_ci()       # CI failure fixing
├── config.py              # Configuration dataclass
├── github_client.py       # GitHub API wrapper
│   ├── GitHubClient       # Issues, PRs, labels, comments
│   └── GitOperations      # Git commands (branch, commit, push)
├── models.py              # Data models
│   ├── IssueContext       # Issue with comments, labels
│   ├── ProcessingResult   # Outcome of processing
│   ├── ClarificationQuestion  # Structured question
│   └── ClarificationRequest   # Set of questions
├── ticket_processor.py    # Main workflow orchestrator
│   └── TicketProcessor    # Full workflow: analyze → implement → PR
└── cli.py                 # Command-line interface
```

## Testing

```bash
# Run E2E tests (creates real GitHub issues)
uv run pytest tests/e2e/test_clarification_workflow.py -v

# Run specific test
uv run pytest tests/e2e/test_clarification_workflow.py::TestClarificationWorkflow::test_vague_issue_triggers_clarification -v
```

## Docker Support

Run agents in containers:

```bash
cd docker/claude-agent

# Export OAuth token from macOS Keychain
./export-token.sh

# Run agent
./run-agent.sh --repo owner/repo --issue 42
```

See `docker/claude-agent/README.md` for details.
