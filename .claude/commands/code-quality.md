# Code Quality Report

Generate a comprehensive code quality report for the codebase, identifying areas that need attention.

## Instructions

Analyze the codebase and generate a report with the following sections. This is a READ-ONLY analysis - do not make any changes, just report findings.

### 1. Large Files (>1000 lines)

Run this command to find large Python files:
```bash
find . -name "*.py" -not -path "./.venv/*" -not -path "./node_modules/*" -exec wc -l {} + | sort -rn | head -30
```

List any files exceeding 1000 lines as decomposition candidates.

### 2. Spackle / Quick Fixes

Search for temporary fix markers:
```bash
grep -rn --include="*.py" -E "(TODO|FIXME|HACK|XXX|WORKAROUND|TEMPORARY|KLUDGE)" . --exclude-dir=.venv --exclude-dir=node_modules
```

Categorize findings by severity:
- **HACK/KLUDGE** - High priority, should be addressed soon
- **FIXME** - Medium priority, known issues
- **TODO** - Lower priority, planned improvements
- **WORKAROUND/TEMPORARY** - Track for removal when proper fix available

### 3. Hardcoded Values

Search for potential hardcoded values that should be configurable:
```bash
# Magic numbers (excluding common ones like 0, 1, -1)
grep -rn --include="*.py" -E "= [0-9]{2,}" . --exclude-dir=.venv --exclude-dir=node_modules

# Hardcoded URLs
grep -rn --include="*.py" -E "(http://|https://)[^\s\"']+" . --exclude-dir=.venv --exclude-dir=node_modules

# Hardcoded paths
grep -rn --include="*.py" -E "(/usr/|/var/|/home/|/tmp/|C:\\\\)" . --exclude-dir=.venv --exclude-dir=node_modules
```

Flag items that should be environment variables or config values.

### 4. Fallback Patterns

Look for defensive fallbacks that might mask real issues:
```bash
grep -rn --include="*.py" -E "(or None|or \[\]|or \{\}|or ''|or \"\"|\.get\(.*,.*None\)|except:$|except Exception:|pass$)" . --exclude-dir=.venv --exclude-dir=node_modules
```

Review each for:
- Silent exception swallowing
- Default values hiding configuration problems
- Overly broad exception handling

### 5. Duplication Candidates

Look for repeated patterns:
```bash
# Similar function signatures
grep -rn --include="*.py" "def " . --exclude-dir=.venv --exclude-dir=node_modules | cut -d: -f3 | sort | uniq -c | sort -rn | head -20

# Repeated imports across files
grep -rn --include="*.py" "^from\|^import" . --exclude-dir=.venv --exclude-dir=node_modules | cut -d: -f3 | sort | uniq -c | sort -rn | head -20
```

Identify opportunities to consolidate similar code into shared utilities.

### 6. Large Functions/Classes

Use AST analysis or manual review to identify:
- Functions longer than 50 lines
- Classes with more than 10 methods
- Deeply nested code (3+ levels of indentation)

### 7. Consolidation Targets

Look for scattered similar logic:
- Multiple files doing similar validation
- Repeated error handling patterns
- Similar data transformation logic

## Report Format

Generate output as markdown with:

```markdown
# Code Quality Report - [Date]

## Summary
- X large files identified
- Y spackle/quick fix markers found
- Z hardcoded values flagged

## Detailed Findings

### Large Files (Decomposition Candidates)
| File | Lines | Recommendation |
|------|-------|----------------|
| path/to/file.py | 1500 | Split into X and Y |

### Spackle / Quick Fixes
#### High Priority (HACK/KLUDGE)
- `file.py:123` - Description

#### Medium Priority (FIXME)
- `file.py:456` - Description

### Hardcoded Values
| File:Line | Value | Recommendation |
|-----------|-------|----------------|
| config.py:42 | 3600 | Move to env var TIMEOUT_SECONDS |

### Fallback Patterns to Review
- `service.py:89` - Bare `except:` swallows all errors

### Duplication Opportunities
- Pattern X appears in files A, B, C - consolidate to utils

### Consolidation Recommendations
- Error handling in X, Y, Z could use shared decorator
```

## Notes

This is an analysis command only. After reviewing the report, create separate tasks/issues to address findings based on priority.

## Processing Issues

When creating a GitHub issue from this report and using the GitHub processor to implement changes, always use the `--worktree` flag for isolation:

```bash
# Create issue from findings, then process in isolated worktree
uv run kestrel-github claim --repo owner/repo --issue <number> --worktree
```

This keeps your main working directory clean while the agent works.
