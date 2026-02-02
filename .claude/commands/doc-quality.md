# Documentation Quality Review

Audit documentation for staleness, orphans, and organization issues. This is a READ-ONLY analysis - recommend archival, never deletion.

## Instructions

Thoroughly read and analyze documentation files, then generate a report identifying cleanup opportunities.

### 1. Orphaned Documents

First, get a complete inventory of ALL markdown files in the repository:
```bash
# Complete markdown inventory (excluding venv/node_modules)
echo "=== All Markdown Files ==="
find . -name "*.md" -type f -not -path "./.venv/*" -not -path "./node_modules/*" -not -path "./.git/*" | sort

# Count by location
echo ""
echo "=== Breakdown ==="
echo "Repo root: $(ls *.md 2>/dev/null | wc -l)"
echo "docs/: $(find docs -name "*.md" -type f 2>/dev/null | wc -l)"
echo "Other locations: $(find . -name "*.md" -type f -not -path "./docs/*" -not -path "./.venv/*" -not -path "./node_modules/*" -not -path "./.git/*" -not -path "./*.md" 2>/dev/null | wc -l)"
```

Find markdown files in docs/ not linked from any index or README:
```bash
# Check for orphans in docs/
for f in $(find docs -name "*.md" -type f); do
  basename=$(basename "$f")
  if ! grep -rql "$basename" docs --include="*.md" 2>/dev/null | grep -v "$f" > /dev/null; then
    echo "ORPHAN: $f"
  fi
done
```

Find markdown files outside docs/ that may need integration (excluding standard root files):
```bash
# Non-standard markdown files outside docs/
find . -name "*.md" -type f \
  -not -path "./docs/*" \
  -not -path "./.venv/*" \
  -not -path "./node_modules/*" \
  -not -path "./.git/*" \
  -not -name "README.md" \
  -not -name "CHANGELOG.md" \
  -not -name "CONTRIBUTING.md" \
  -not -name "LICENSE.md" \
  -not -name "CLAUDE.md" \
  -not -name "CODE_OF_CONDUCT.md" | sort
```

For files in subdirectories (like `.claude/`, `scripts/`, `kestrel_sovereign/`), assess if they should:
- Stay where they are (component-specific docs)
- Move to `docs/` (general documentation)
- Be linked from relevant index files

### 2. Stale Plans

Look for plans that may be completed or obsolete:
```bash
# Find all plan documents
find docs/plans docs/planning -name "*.md" -type f 2>/dev/null | sort

# Look for completion indicators
grep -rn --include="*.md" -E "(COMPLETED|DONE|IMPLEMENTED|OBSOLETE|DEPRECATED|Phase [0-9]+ Complete)" docs/plans docs/planning 2>/dev/null
```

For each plan document, READ the content and assess:
- Is this plan fully implemented? Check against actual code.
- Is this plan abandoned or superseded?
- Does this plan reference features that now exist differently?
- Is this dated more than 6 months ago with no recent updates?

### 3. Duplicate/Overlapping Content

Look for documents covering similar topics:
```bash
# Find similar filenames
find docs -name "*.md" -exec basename {} \; | sort | uniq -d

# Look for repeated headings across files
grep -rh --include="*.md" "^# " docs | sort | uniq -c | sort -rn | head -20

# Search for similar topic coverage
grep -rn --include="*.md" -E "(architecture|privacy|constitution|identity|memory|agent)" docs | cut -d: -f1 | sort | uniq -c | sort -rn
```

Read overlapping documents and identify:
- Content that should be consolidated
- Documents that reference the same concepts differently
- Outdated versions of the same topic

### 4. Misplaced Documents

Check if documents are in the correct folder per the README structure:
- Architecture docs should be in `docs/architecture/`
- Plans should be in `docs/plans/` or `docs/planning/`
- User guides should be in `docs/user-documentation/`
- Business docs should be in `docs/business/`

```bash
# Find documents that might be misplaced based on content
grep -l "architecture" docs/*.md docs/*/*.md 2>/dev/null | grep -v "architecture/"
grep -l "business\|revenue\|pricing" docs/*.md 2>/dev/null
grep -l "user guide\|how to\|tutorial" docs/*.md 2>/dev/null | grep -v "user-documentation/"
```

### 5. Broken Internal Links

Check for broken references between documents:
```bash
# Find all markdown links
grep -roh --include="*.md" '\[.*\](.*\.md)' docs | grep -oE '\(.*\.md\)' | tr -d '()' | sort | uniq -c | sort -rn
```

For each linked file, verify it exists at the referenced path.

### 6. Empty or Stub Documents

Find documents that are placeholders or nearly empty:
```bash
find docs -name "*.md" -type f -exec sh -c 'lines=$(wc -l < "$1"); if [ "$lines" -lt 10 ]; then echo "$lines lines: $1"; fi' _ {} \;
```

### 7. Root-Level Doc Sprawl

Check for accumulation of docs at the repo root and docs root that should be organized:
```bash
# Root directory markdown files (excluding standard ones)
echo "=== Repo Root Markdown Files ==="
ls -la *.md 2>/dev/null

# docs/ root level (not in subfolders)
echo "=== docs/ Root Level ==="
ls -la docs/*.md 2>/dev/null

# Count totals
echo "=== Counts ==="
echo "Repo root .md files: $(ls *.md 2>/dev/null | wc -l)"
echo "docs/ root .md files: $(ls docs/*.md 2>/dev/null | wc -l)"
```

For each root-level markdown file, READ it and assess:
- **Repo root files**: Should this be in `docs/`? Is it a standard file (README, CHANGELOG, CONTRIBUTING, LICENSE)?
- **docs/ root files**: Should this be categorized into a subfolder (architecture/, plans/, user-documentation/)?
- Is this a one-off document that accumulated over time?
- Does it duplicate content that exists elsewhere?

Standard root files to ignore: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE.md`, `CLAUDE.md`, `CODE_OF_CONDUCT.md`

### 8. Archive Candidates

Based on all findings, identify documents that should be moved to `docs/archive/`:
- Completed plans
- Superseded documentation
- Historical records no longer actively referenced
- Draft documents that were never finalized

## Report Format

Generate output as markdown:

```markdown
# Documentation Quality Report - [Date]

## Summary
- Total markdown files: X (Y in docs/, Z in root, W elsewhere)
- X orphaned documents found
- Y stale/completed plans identified
- Z documents recommended for archival
- N documents need relocation
- R root-level files need categorization

## Detailed Findings

### Orphaned Documents
Documents not linked from any index:
| Document | Recommendation |
|----------|----------------|
| docs/old-plan.md | Archive or link from planning/README.md |

### Stale Plans (Archive Candidates)
Plans that appear completed or obsolete:
| Plan | Status | Evidence | Recommendation |
|------|--------|----------|----------------|
| docs/plans/phase-1.md | Completed | All tasks implemented | Archive to docs/archive/plans/ |

### Duplicate/Overlapping Content
| Documents | Overlap | Recommendation |
|-----------|---------|----------------|
| A.md, B.md | Both cover agent architecture | Consolidate into A.md |

### Misplaced Documents
| Document | Current Location | Should Be In |
|----------|------------------|--------------|
| api-guide.md | docs/ | docs/user-documentation/ |

### Broken Links
| Source | Broken Link | Fix |
|--------|-------------|-----|
| README.md | ./old-doc.md | Remove or update reference |

### Empty/Stub Documents
| Document | Lines | Recommendation |
|----------|-------|----------------|
| placeholder.md | 3 | Complete or remove |

### Root-Level Sprawl

#### Repo Root (non-standard files)
| File | Purpose | Recommendation |
|------|---------|----------------|
| NOTES.md | Development notes | Move to docs/planning/ or archive |

#### docs/ Root (uncategorized)
| File | Topic | Should Be In |
|------|-------|--------------|
| CRITICAL_CODE_REVIEW.md | Code issues | docs/architecture/ or docs/planning/ |

#### Other Locations
| File | Location | Recommendation |
|------|----------|----------------|
| scripts/DEPLOY.md | scripts/ | Keep (deployment-specific) or move to docs/

## Archive Actions
Proposed archive moves (to docs/archive/):
```bash
# Copy these commands to execute archival
mkdir -p docs/archive/plans
mv docs/plans/completed-plan.md docs/archive/plans/
# Update any references before moving
```

## Integration Actions
Documents needing integration into existing structure:
| Document | Target | Action |
|----------|--------|--------|
| standalone-guide.md | user-documentation/README.md | Add link and context |
```

## Notes

- **Never delete** - always archive to preserve history
- Read documents thoroughly before recommending action
- Check git history for last modification dates
- Verify recommendations against actual code state
- Update index files (README.md) after any moves

## Processing Issues

When creating a GitHub issue from this report and using the GitHub processor to implement changes, always use the `--worktree` flag for isolation:

```bash
# Create issue from findings, then process in isolated worktree
uv run kestrel-github claim --repo owner/repo --issue <number> --worktree
```

This keeps your main working directory clean while the agent works.
