# Run All Tests

Run comprehensive test suite with fail-fast behavior and detailed reporting.

## Arguments
- `$ARGUMENTS` - Optional: specific test file or directory to run

## Execution

1. **Check services first:**
```bash
# Check PostgreSQL
docker ps | grep kestrel-postgres || echo "WARNING: PostgreSQL not running"

# Check Redis
docker ps | grep kestrel-redis || echo "WARNING: Redis not running"

# Check API (if running)
curl -s http://localhost:7777/health 2>/dev/null | head -c 100 || echo "INFO: Kestrel API not running (optional for unit tests)"
```

2. **Run tests based on arguments:**

If `$ARGUMENTS` is provided, run that specific test:
```bash
cd ./
uv run pytest $ARGUMENTS -x -v --tb=short
```

If no arguments, run full suite in order (fail-fast stops at first failure):

```bash
cd ./

# Run unit tests first (fast)
echo "=== Unit Tests ==="
uv run pytest tests/unit/ -x -v --tb=short

# If unit passes, run integration tests
echo "=== Integration Tests ==="
uv run pytest tests/integration/ -x -v --tb=short

# If integration passes, run LLM tests
echo "=== LLM Tests ==="
uv run pytest tests/llm/ -x -v --tb=short
```

3. **On failure:**
- STOP immediately (that's what -x does)
- Show the failing test output
- Do NOT continue to other test suites
- Suggest fix based on error

4. **On success:**
- Report total tests passed
- Report execution time

## Flags Used
- `-x` - Stop on first failure (MANDATORY per coding standards)
- `-v` - Verbose output
- `--tb=short` - Short tracebacks (readable but not overwhelming)
- `--ignore=tests/load/` - Skip load tests by default (use --run-load explicitly)

## Examples
```
/run_all_tests                           # Run all tests
/run_all_tests tests/integration/        # Run only integration tests
/run_all_tests kestrel/tests/integration/test_auth_e2e.py  # Run specific file
/run_all_tests -k "test_user"            # Run tests matching pattern
```
