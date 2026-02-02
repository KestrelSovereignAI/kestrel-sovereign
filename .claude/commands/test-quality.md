# Test Quality Review

Audit the test suite for quality, consistency, proper fixture usage, and resource cleanup. This is a READ-ONLY analysis - recommend improvements, not changes.

## Philosophy

Good tests are:
- **Clear** - obvious what they test and why they might fail
- **Isolated** - no shared state, no order dependencies
- **Clean** - use fixtures, clean up resources, no leaks
- **Fast** - unit tests run in milliseconds, not seconds
- **Consistent** - follow project conventions

## Instructions

Thoroughly analyze the test suite and generate a quality report.

### 1. Test Organization

Check test file organization and naming:
```bash
# Count tests per file (flag files with >30 tests)
echo "=== Tests Per File ==="
for f in $(find tests -name "test_*.py" -type f); do
  count=$(grep -c "def test_" "$f" 2>/dev/null || echo 0)
  if [ "$count" -gt 30 ]; then
    echo "LARGE: $f ($count tests)"
  fi
done

# Check for proper test class organization
echo ""
echo "=== Test Files Without Classes (consider organizing) ==="
for f in $(find tests -name "test_*.py" -type f); do
  if ! grep -q "^class Test" "$f" 2>/dev/null; then
    test_count=$(grep -c "def test_" "$f" 2>/dev/null || echo 0)
    if [ "$test_count" -gt 5 ]; then
      echo "$f ($test_count tests, no Test classes)"
    fi
  fi
done
```

For each finding, assess:
- Should tests be grouped into logical Test classes?
- Are test names descriptive (test_<behavior>_<scenario>)?
- Is the file focused on one component or scattered?

### 2. Fixture Usage

Check for proper fixture utilization:
```bash
# Find tests creating their own temp directories (should use fixtures)
echo "=== Inline Temp Directory Creation (use temp_dir fixture) ==="
grep -rn --include="test_*.py" "tempfile.mkdtemp\|tempfile.TemporaryDirectory\|tmp_path.*mkdir" tests

# Find tests not using async fixtures properly
echo ""
echo "=== Async Tests Without Proper Fixtures ==="
grep -rn --include="test_*.py" -A3 "@pytest.mark.asyncio" tests | grep -E "LLMService\(\)|KestrelAgent\(" | head -20

# Find duplicate fixture definitions (should be in conftest)
echo ""
echo "=== Duplicate Fixtures (should be in conftest.py) ==="
grep -rh --include="test_*.py" "@pytest.fixture" tests | grep -v "conftest.py" | sort | uniq -c | sort -rn | head -10
```

Verify tests use these shared fixtures from conftest.py:
- `temp_dir` - temporary directories with auto-cleanup
- `temp_file` - temporary files with auto-cleanup
- `temp_db` - temporary SQLite databases
- `agent_data_dir` - mock agent environment
- `async_llm_service` - properly closed LLM service
- `async_kestrel_agent` - properly shutdown agent

### 3. Resource Cleanup

Check for resource leaks and cleanup issues:
```bash
# Find tests that might not clean up
echo "=== Potential Resource Leaks ==="
grep -rn --include="test_*.py" -E "(open\(|sqlite3\.connect|aiosqlite\.connect|httpx\.Client|aiohttp\.Client)" tests | grep -v "with " | head -20

# Find missing cleanup in async tests
echo ""
echo "=== Async Resources Without Cleanup ==="
grep -rn --include="test_*.py" -B5 "await.*close\(\)" tests | grep -E "def test_" | wc -l
echo "Tests with explicit close() calls"

# Check for subprocess/thread cleanup
echo ""
echo "=== Subprocess/Thread Usage (verify cleanup) ==="
grep -rn --include="test_*.py" -E "(subprocess\.|threading\.|multiprocessing\.)" tests | head -10
```

For each potential leak, verify:
- Is there a corresponding cleanup (close, shutdown, terminate)?
- Should this use a context manager or fixture instead?
- Is the cleanup in a finally block or fixture teardown?

### 4. Test Isolation

Check for test pollution:
```bash
# Find global state modifications
echo "=== Global State Modifications ==="
grep -rn --include="test_*.py" -E "(os\.environ\[|setattr\(.*\,|patch\()" tests | grep -v "monkeypatch\." | head -20

# Find tests modifying shared resources
echo ""
echo "=== Shared Resource Modifications ==="
grep -rn --include="test_*.py" -E "(\.write\(|\.mkdir|shutil\.)" tests | grep -v "temp_dir\|tmp_path\|temp_file" | head -15

# Find tests without proper markers
echo ""
echo "=== Tests Missing Required Markers ==="
grep -rn --include="test_*.py" -B2 "async def test_" tests | grep -v "@pytest.mark.asyncio" | grep "async def test_" | head -10
```

Verify:
- Environment modifications use `monkeypatch` fixture
- File operations use temp directories
- Async tests have `@pytest.mark.asyncio`
- Integration tests have proper markers

### 5. Assertion Quality

Check assertion practices:
```bash
# Find bare asserts without messages
echo "=== Bare Asserts (consider adding messages) ==="
grep -rn --include="test_*.py" "assert [^,]*$" tests | grep -v '"""' | head -20

# Find multiple asserts without clear purpose
echo ""
echo "=== Tests With Many Asserts (consider splitting) ==="
for f in $(find tests -name "test_*.py" -type f); do
  grep -n "def test_" "$f" | while read line; do
    linenum=$(echo "$line" | cut -d: -f1)
    name=$(echo "$line" | grep -oE "def test_[a-zA-Z0-9_]+")
    # Count asserts in next 50 lines
    count=$(sed -n "${linenum},$((linenum+50))p" "$f" | grep -c "assert " || echo 0)
    if [ "$count" -gt 10 ]; then
      echo "$f:$linenum - $name has $count asserts"
    fi
  done
done 2>/dev/null | head -10

# Check for assertEqual vs assert == (pytest style)
echo ""
echo "=== Non-Pytest Style Assertions ==="
grep -rn --include="test_*.py" -E "(self\.assert|assertEquals|assertTrue|assertFalse)" tests | head -10
```

Best practices:
- Use descriptive assertion messages for complex conditions
- One logical assertion per test when possible
- Use pytest's native assertions (not unittest style)
- Use `pytest.raises` for exception testing

### 6. Test Coverage Gaps

Identify untested areas:
```bash
# Find source files without corresponding tests
echo "=== Source Files Potentially Missing Tests ==="
for src in $(find kestrel_sovereign -name "*.py" -type f | grep -v __pycache__ | grep -v __init__); do
  basename=$(basename "$src" .py)
  if ! find tests -name "test_*${basename}*.py" -type f 2>/dev/null | grep -q .; then
    echo "No test file for: $src"
  fi
done | head -20

# Find public functions without test coverage hints
echo ""
echo "=== Public Functions (verify test coverage) ==="
grep -rh --include="*.py" "^def [a-z]" kestrel_sovereign | grep -v "^def _" | cut -d: -f2 | sort | uniq -c | sort -rn | head -15
```

### 7. Test Speed & Markers

Check for proper test categorization:
```bash
# Find slow tests without markers
echo "=== Potentially Slow Tests (verify markers) ==="
grep -rn --include="test_*.py" -E "(time\.sleep|asyncio\.sleep)" tests | head -10

# Check marker usage
echo ""
echo "=== Marker Distribution ==="
echo "asyncio: $(grep -r "@pytest.mark.asyncio" tests --include="test_*.py" | wc -l)"
echo "slow: $(grep -r "@pytest.mark.slow" tests --include="test_*.py" | wc -l)"
echo "integration: $(grep -r "@pytest.mark.integration" tests --include="test_*.py" | wc -l)"
echo "cloud_resource: $(grep -r "@pytest.mark.cloud_resource" tests --include="test_*.py" | wc -l)"
echo "parametrize: $(grep -r "@pytest.mark.parametrize" tests --include="test_*.py" | wc -l)"

# Find tests in wrong directories
echo ""
echo "=== Tests Possibly in Wrong Directory ==="
grep -l "httpx\|aiohttp\|requests" tests/unit/*.py 2>/dev/null | head -5
echo "(Unit tests should not make real HTTP calls)"
```

### 8. Docstrings & Documentation

Check test documentation:
```bash
# Find test files without module docstrings
echo "=== Test Files Missing Module Docstrings ==="
for f in $(find tests -name "test_*.py" -type f); do
  first_line=$(head -1 "$f" | tr -d '[:space:]')
  if [ "$first_line" != '"""' ] && [ "$first_line" != "'''" ]; then
    echo "$f"
  fi
done | head -15

# Find test classes without docstrings
echo ""
echo "=== Test Classes Missing Docstrings ==="
grep -rn --include="test_*.py" -A1 "^class Test" tests | grep -v '"""' | grep "class Test" | head -10
```

### 9. Anti-Patterns

Check for common test anti-patterns:
```bash
# Hardcoded test values that should be fixtures
echo "=== Hardcoded Test Data (consider fixtures) ==="
grep -rn --include="test_*.py" -E '(localhost|127\.0\.0\.1|test@|password|secret)' tests | head -15

# Tests depending on execution order
echo ""
echo "=== Potential Order Dependencies ==="
grep -rn --include="test_*.py" -E "(global |_state|_cache|_instance)" tests | head -10

# Empty or trivial tests
echo ""
echo "=== Possibly Trivial Tests ==="
grep -rn --include="test_*.py" -A3 "def test_" tests | grep -E "pass$|assert True|assert 1" | head -10
```

### 10. Consistency Check

Compare against project conventions:
```bash
# Check naming consistency
echo "=== Naming Inconsistencies ==="
# Test files should be test_*.py
find tests -name "*.py" -type f | grep -v "test_\|conftest\|__init__\|/utils/\|/shared/" | head -10

# Check import style consistency
echo ""
echo "=== Import Style ==="
echo "from...import style: $(grep -r "^from " tests --include="test_*.py" | wc -l)"
echo "import style: $(grep -r "^import " tests --include="test_*.py" | wc -l)"
```

## Report Format

Generate output as markdown:

```markdown
# Test Quality Report - [Date]

## Summary
- Total test files: X
- Total test functions: Y
- Files needing attention: Z
- Fixture usage score: A/10
- Cleanup compliance: B%

## Health Indicators

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Tests with docstrings | X% | 80% | OK/NEEDS WORK |
| Fixture usage | X% | 90% | OK/NEEDS WORK |
| Proper markers | X% | 100% | OK/NEEDS WORK |
| Resource cleanup | X% | 100% | OK/NEEDS WORK |

## Detailed Findings

### Organization Issues
| File | Issue | Recommendation |
|------|-------|----------------|
| test_large.py | 45 tests, no classes | Split into TestFeatureA, TestFeatureB |

### Fixture Improvements
| File:Line | Current Pattern | Better Pattern |
|-----------|-----------------|----------------|
| test_x.py:42 | `tempfile.mkdtemp()` | Use `temp_dir` fixture |

### Resource Leaks
| File:Line | Resource | Missing Cleanup |
|-----------|----------|-----------------|
| test_y.py:100 | SQLite connection | Add `await conn.close()` or use fixture |

### Isolation Issues
| File:Line | Issue | Fix |
|-----------|-------|-----|
| test_z.py:50 | Modifies os.environ directly | Use `monkeypatch.setenv()` |

### Missing Markers
| File:Line | Test | Missing Marker |
|-----------|------|----------------|
| test_async.py:20 | test_fetch_data | `@pytest.mark.asyncio` |

### Coverage Gaps
| Source File | Test Status |
|-------------|-------------|
| kestrel_sovereign/new_feature.py | No test file found |

### Anti-Patterns Found
| File:Line | Pattern | Why It's Bad | Fix |
|-----------|---------|--------------|-----|
| test_x.py:10 | Hardcoded "password123" | Security, maintainability | Use fixture or env var |

## Recommendations

### High Priority (Fix Now)
1. Resource leaks causing test hangs
2. Missing async markers causing silent failures
3. Tests modifying global state

### Medium Priority (Fix Soon)
1. Large test files needing decomposition
2. Inline temp directories - use fixtures
3. Missing test docstrings

### Low Priority (Continuous Improvement)
1. Add descriptive assertion messages
2. Improve test naming consistency
3. Increase parametrized test coverage

## Beautiful Test Example

Reference implementation showing all best practices:

    """
    Tests for FeatureName component.

    Tests the core functionality including edge cases and error handling.
    """
    import pytest
    from unittest.mock import AsyncMock

    from kestrel_sovereign.feature import FeatureName


    class TestFeatureNameBasics:
        """Tests for basic FeatureName operations."""

        @pytest.fixture
        def feature(self, temp_dir):
            """Create a feature instance with test configuration."""
            return FeatureName(data_dir=temp_dir)

        def test_creates_output_file_in_data_directory(self, feature, temp_dir):
            """Feature should create output in the configured data directory."""
            result = feature.process("input")

            output_file = temp_dir / "output.txt"
            assert output_file.exists(), "Output file should be created"
            assert result.success, "Processing should succeed"

        @pytest.mark.asyncio
        async def test_handles_network_timeout_gracefully(self, feature):
            """Feature should return error result on network timeout, not raise."""
            feature.client = AsyncMock(side_effect=TimeoutError())

            result = await feature.fetch_remote()

            assert not result.success
            assert "timeout" in result.error.lower()
```

## Notes

- This is an analysis command only - generate report, don't make changes
- Focus on patterns that cause real problems (flaky tests, hangs, pollution)
- Balance thoroughness with pragmatism
- Celebrate good patterns found, not just problems
- Tests are first-class code - they deserve the same care as production code

## Processing Issues

When creating a GitHub issue from this report and using the GitHub processor to implement changes, always use the `--worktree` flag for isolation:

```bash
# Create issue from findings, then process in isolated worktree
uv run kestrel-github claim --repo owner/repo --issue <number> --worktree
```

This keeps your main working directory clean while the agent works.
