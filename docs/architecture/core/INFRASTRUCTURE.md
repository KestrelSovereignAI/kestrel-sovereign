---
type: Architecture Spec
title: Kestrel Development Infrastructure
description: Complete toolkit for accelerated parallel development with Claude Code.
resource: /docs/architecture/core/INFRASTRUCTURE.md
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

# Kestrel Development Infrastructure

Complete toolkit for accelerated parallel development with Claude Code.

## Quick Start

```bash
# Launch parallel development across 3 feature branches
/parallel-work

# Test all worktrees
/test-worktrees

# Integrate completed features
/merge-features
```

## Infrastructure Components

### 🤖 Subagents (`.claude/agents/`)

Specialized agents for focused work:

| Subagent | Purpose | Worktree | Branch |
|----------|---------|----------|--------|
| **llm-service-specialist** | LLM service hardening | ../kestrel-llm-service | feature/llm-service-hardening |
| **constitution-verifier** | Constitution validation | ../kestrel-verify-constitution | feature/verify-constitution-inception |
| **privacy-architect** | Privacy modes | ../kestrel-privacy-settings | feature/privacy-settings-ui |

**Usage:**
```
"Use the llm-service-specialist subagent to add streaming support"
"Use the constitution-verifier subagent to validate inception"
"Use the privacy-architect subagent to implement EPHEMERAL mode"
```

### ⚡ Slash Commands (`.claude/commands/`)

Workflow automation:

| Command | Description |
|---------|-------------|
| `/worktree-create <name>` | Create new worktree for feature |
| `/parallel-work` | Launch 3 subagents in parallel |
| `/test-worktrees` | Run tests across all worktrees |
| `/constitution-check` | Verify constitution embedding |
| `/privacy-test` | Test all 5 privacy modes |
| `/merge-features` | Integrate feature branches safely |

## Parallel Development Workflow

### 1. Setup Phase
```bash
# Worktrees already created:
git worktree list

# Output:
# ./ (main)
# ./-llm-service (feature/llm-service-hardening)
# ./-verify-constitution (feature/verify-constitution-inception)
# ./-privacy-settings (feature/privacy-settings-ui)
```

### 2. Launch Parallel Work
```
/parallel-work
```

This command:
1. Launches **llm-service-specialist** in llm-service worktree
2. Launches **constitution-verifier** in verify-constitution worktree
3. Launches **privacy-architect** in privacy-settings worktree
4. All work happens **in parallel**
5. Each agent returns a comprehensive report

### 3. Monitor Progress
Each agent works independently and returns:
```json
{
  "agent_id": "llm-service-specialist",
  "status": "completed",
  "worktree": "./-llm-service",
  "branch": "feature/llm-service-hardening",
  "files_modified": ["llm/service.py", "llm/adapter.py"],
  "files_created": ["llm/llm_service_router.py"],
  "tests_added": 45,
  "tests_passing": 42,
  "tests_failing": 3,
  "blockers": [],
  "integration_notes": "Requires pillow dependency"
}
```

### 4. Test Everything
```
/test-worktrees
```

Runs pytest across all worktrees and generates consolidated report.

### 5. Integration
```
/merge-features
```

Safely merges all feature branches with:
- Conflict detection
- Test verification before each merge
- Proper merge order (constitution → privacy → llm)

## Individual Feature Workflows

### LLM Service Work

**Manual:**
```
"Use the llm-service-specialist subagent to implement structured output"
```

**Reference:**
- Implement: Streaming, vision, Pydantic models
- Test: Parameterized tests with real providers

### Constitution Work

**Manual:**
```
/constitution-check
```

**Verification:**
- Constitution stored as first file ✓
- Hash in agent properties ✓
- "governed_by" edge ✓
- Genesis self-audit ✓
- DID format valid ✓

### Privacy Work

**Manual:**
```
/privacy-test
```

**Implementation:**
- 5 privacy modes (EPHEMERAL to PUBLIC)
- Agent commands (!set-privacy-mode)
- UI components (indicator, selector)
- Storage layer integration
- PII filtering

## Benefits

### ⚡ Speed
- 3 features developed simultaneously
- Separate context windows (no pollution)
- Independent test cycles

### 🎯 Focus
- Each subagent has single, clear responsibility
- Specialized expertise per area
- No context switching overhead

### 🔍 Traceability
- Each feature on separate branch
- Git history clear and organized
- Easy to rollback individual features

### 🧪 Quality
- Real integration tests (NO MOCKS)
- Fail-fast with pytest -x
- Comprehensive test coverage
- Professional coding standards enforced

## Architecture Principles

### Subagent Design
- **Focused**: One responsibility per subagent
- **Autonomous**: Works independently in worktree
- **Reportable**: Returns structured status
- **Testable**: Includes comprehensive tests

### Slash Command Design
- **Clear**: Explicit purpose and usage
- **Safe**: Includes validation and checks
- **Documented**: Usage examples provided
- **Chained**: Can be combined in workflows

## Integration with Existing Tools

### Git Worktrees
- Each feature in separate working directory
- Shares same git repository
- Independent branches
- Clean isolation

### Pytest
- Fail-fast with -x flag
- Real services (NO MOCKS)
- Parameterized tests
- >90% coverage target

### Professional Standards
- From `PROFESSIONAL_CODING_STANDARDS.md`
- No hacks, workarounds, or spackle
- Fail fast philosophy
- Test failures stop everything

## Next Steps

1. **Test the infrastructure:**
   ```
   /parallel-work
   ```

2. **Monitor progress** - Each subagent reports back

3. **Review results** - Check agent reports

4. **Test integration:**
   ```
   /test-worktrees
   ```

5. **Merge features:**
   ```
   /merge-features
   ```

## Troubleshooting

**Subagent not activating?**
- Use explicit invocation: "Use the [agent-name] subagent to..."
- Check worktree exists and is on correct branch

**Tests failing?**
- Each subagent reports its test results
- Use pytest -x for fail-fast debugging
- Fix in worktree, re-run tests

**Merge conflicts?**
- `/merge-features` detects conflicts before merging
- Resolve conflicts in feature branch
- Re-run merge process

## File Locations

```
kestrel/
├── .claude/
│   ├── agents/
│   │   ├── llm-service-specialist.md    (NEW)
│   │   ├── constitution-verifier.md      (NEW)
│   │   └── privacy-architect.md          (NEW)
│   ├── commands/
│   │   ├── parallel-work.md              (NEW)
│   │   ├── test-worktrees.md             (NEW)
│   │   ├── merge-features.md             (NEW)
│   │   ├── constitution-check.md         (NEW)
│   │   ├── privacy-test.md               (NEW)
│   │   └── worktree-create.md            (NEW)
├── INFRASTRUCTURE.md                      (THIS FILE)
└── AGENTS.md                              (REFERENCE DOC)
```

## Success Metrics

- **Development Speed**: 3x faster (parallel vs sequential)
- **Code Quality**: >90% test coverage, all tests passing
- **Integration Risk**: Low (independent branches, tested before merge)
- **Context Management**: Clean (separate worktrees, focused subagents)
- **Team Scalability**: High (can add more subagents/worktrees)

---

**Ready to accelerate Kestrel development!** 🚀
