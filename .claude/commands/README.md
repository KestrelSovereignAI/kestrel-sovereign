# Kestrel Slash Commands

Custom workflow commands for Kestrel framework development.

## Available Commands

### Git Worktree Management

**`/worktree-create <feature-name>`**
Create a new git worktree for parallel development.
```
Example: /worktree-create llm-hardening
Creates: ../kestrel-llm-hardening on feature/llm-hardening branch
```

### Parallel Development

**`/parallel-work`**
Launch 3 specialist subagents to work in parallel across worktrees:
- LLM Service Hardening (llm-service-specialist)
- Constitution Verification (constitution-verifier)
- Privacy Settings (privacy-architect)

### Testing

**`/test-worktrees`**
Run pytest across all active worktrees in parallel.
Generates consolidated test report.

**`/constitution-check`**
Verify Kestrel Constitution embedding in all agents.
Validates governance system integrity.

**`/privacy-test`**
Test all 5 privacy modes end-to-end with UI validation.

### Integration

**`/merge-features`**
Safely integrate multiple feature branches into main.
Includes conflict detection and test verification.

### Quality Reports

**`/code-quality`**
Generate comprehensive code quality report analyzing:
- Large files (decomposition candidates)
- Spackle/quick-fix markers (TODO, HACK, FIXME)
- Hardcoded values needing configuration
- Fallback patterns masking issues
- Duplication opportunities

**`/doc-quality`**
Audit documentation for staleness and organization issues:
- Orphaned documents not linked from indexes
- Stale/completed plans needing archival
- Duplicate/overlapping content
- Misplaced documents
- Broken internal links
- Archive recommendations (never delete)

**`/test-quality`**
Audit test suite for quality, consistency, and best practices:
- Test organization (classes, naming, file size)
- Fixture usage (shared fixtures vs inline setup)
- Resource cleanup (leaks, missing close/shutdown)
- Test isolation (global state, proper markers)
- Assertion quality (messages, pytest style)
- Coverage gaps and anti-patterns
- Beautiful test examples and recommendations

## Workflow Examples

### Parallel Feature Development
```bash
# 1. Create worktrees for 3 features
/worktree-create llm-hardening
/worktree-create constitution-verify
/worktree-create privacy-ui

# 2. Launch parallel work
/parallel-work

# 3. Test all features
/test-worktrees

# 4. Integrate when complete
/merge-features
```

### Constitution Verification
```bash
# After modifying inception_service.py
/constitution-check

# Should pass all 5 checks before deployment
```

### Privacy System Validation
```bash
# Test complete privacy system
/privacy-test

# Should show 100% pass rate for production
```

## Command Guidelines

- Commands expand inline as prompts
- Use specific commands for focused tasks
- Chain commands for complex workflows
- All test commands use pytest -x (fail-fast)

## Adding New Commands

1. Create `.md` file in `.claude/commands/`
2. Write clear prompt with steps and constraints
3. Include usage examples
4. Test with sample invocation
5. Update this README

## Related Tools

- **Subagents**: Specialized agents in `.claude/agents/`
- **Hooks**: Event-driven automation in `.claude/settings.json`
