---
type: Architecture Spec
title: Kestrel Test Strategy Guide
description: A comprehensive guide to running and writing tests for Kestrel Sovereign.
resource: /docs/architecture/testing/TESTING_GUIDE.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-07-24T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Kestrel Test Strategy Guide

A comprehensive guide to running and writing tests for Kestrel Sovereign.

> For where tests sit in the **Agent/Talon review loop** (targeted tests
> during implementation, independent verification during review, CI
> before merge) and the structured result states the reviewer reports,
> see [`TEST_EVIDENCE_GATES.md`](TEST_EVIDENCE_GATES.md).
>
> Semantic-KB cutovers additionally use the immutable, content-free
> conformance, parity, erasure, benchmark, diagnostics, and Kite evidence
> catalog in [`SEMANTIC_RELEASE_EVIDENCE.md`](SEMANTIC_RELEASE_EVIDENCE.md).
> A generated template is deliberately non-ready until each catalog-bound
> observation and artifact digest has been independently recorded.

## Test Pyramid Strategy

Tests are organized in a pyramid structure - run from the bottom up:

1. **Unit Tests** - Fast, no external dependencies
2. **Integration Tests** - Requires services (SQLite, optionally PostgreSQL/Redis)
3. **E2E/UI Tests** - Requires running server + Playwright browser automation

## Running Tests

### Quick Start

```bash
# Run all tests with the test runner
./run_tests.py --unit          # Unit tests only
./run_tests.py --integration   # Integration tests
./run_tests.py --ci --skip-check  # Full suite + canonical-package coverage
```

A fresh `uv sync` installs `pytest-cov` through the default development group.
The `--ci` command measures the canonical `kestrel_sovereign` package and
enforces the measured ratchet in `.coveragerc`. The current ratchet is 73%, not
80%: a 2026-07-25 full-unit-suite measurement produced 73.53% combined
line/branch coverage (68,079 covered lines and 19,863 covered branches). The
weekly job includes that unit suite plus the remaining repository tests. An 80%
floor remains the next target and must not be presented as current coverage.

The scheduled weekly analysis runs the same coverage gate with xdist. CI
disables fail-fast for this lane so pytest-cov can combine every worker file and
write terminal, HTML, JSON, and XML diagnostics even when tests fail. The job
remains failed when tests or coverage fail, while feedback/coverage uploads and
the issue-analysis job still run. Missing artifacts are tolerated so an early
test-runner failure does not hide the original failure or prevent analysis.

For a focused coverage report during development, override the repository-wide
floor so untested modules outside the selected test do not fail the command:

```bash
uv run pytest tests/unit/test_docs_verify.py --cov=kestrel_sovereign \
  --cov-report=term --cov-fail-under=0
```

### The `run_tests.py` Script

The project includes a comprehensive test runner with smart features:

| Flag | Purpose |
|------|---------|
| `--unit` | Run unit tests only |
| `--integration` | Run integration tests only |
| `--llm` | Run LLM-dependent tests |
| `--ci` | Full CI mode (parallel + canonical-package branch coverage, 73% measured ratchet) |
| `-x` | Fail fast (stop on first failure) |
| `--failed` | Re-run only last failed tests |
| `--skip-check` | Skip DB/Redis health checks |
| `-k "pattern"` | Run tests matching pattern |
| `--parallel auto` | Auto-detect worker count |

### Recommended Workflow

1. **Run unit tests first** (fast, catches obvious issues):
   ```bash
   ./run_tests.py --unit --skip-check
   ```

2. **On failure - fix and re-run only failed**:
   ```bash
   ./run_tests.py --unit --failed
   ```

3. **Search for similar issues**:
   ```bash
   grep -r "pattern" tests/unit/
   ```

4. **Move to integration tests**:
   ```bash
   ./run_tests.py --integration --skip-check
   ```

5. **E2E tests** (requires running server):
   ```bash
   uv run python server.py &
   cd tests/e2e && npx playwright test
   ```

## Writing Tests

### SQLite WAL Mode Tests

When testing SQLite sync features, remember that **WAL files are checkpointed when all connections close**. This means if you:

1. Create a database
2. Close the connection
3. Open a new connection to write

...the WAL file will be empty because it was checkpointed when the first connection closed.

**Solution: Use a keeper connection**

```python
@pytest.fixture
def temp_db_with_keeper(tmp_path) -> Tuple[Path, sqlite3.Connection]:
    """Database with keeper connection to prevent WAL checkpoint."""
    db_path = tmp_path / "test.db"
    keeper = sqlite3.connect(str(db_path))
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    keeper.commit()
    # Return both - test must close keeper when done
    yield db_path, keeper
    keeper.close()

async def test_wal_monitoring(self, temp_db_with_keeper, tmp_path):
    temp_db, keeper = temp_db_with_keeper

    # Keeper keeps WAL file alive even when other connections close
    conn = sqlite3.connect(str(temp_db))
    conn.execute("INSERT INTO test (value) VALUES ('data')")
    conn.commit()
    conn.close()  # WAL file still exists!

    # ... test WAL listener ...
```

### Docker-Dependent Tests

Tests requiring Docker should gracefully skip when Docker is unavailable:

```python
@pytest.fixture
def check_docker():
    """Skip test if Docker is not available."""
    try:
        import docker
        from docker.credentials.errors import StoreError
        client = docker.from_env()
        client.ping()
    except ImportError as e:
        pytest.skip(f"Docker SDK not installed: {e}")
    except docker.credentials.errors.StoreError as e:
        pytest.skip(f"Docker credential store not available: {e}")
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")
```

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with real services
├── e2e/            # Browser/UI tests
│   ├── playwright.config.cjs
│   └── test_*.spec.cjs
└── conftest.py     # Shared fixtures
```

## E2E/Playwright Tests

### Setup

```bash
cd tests/e2e
npm install
npx playwright install
```

### Running

```bash
# Start server first
uv run python server.py &

# Run tests
npx playwright test

# Run specific test
npx playwright test test_chat_and_models.spec.cjs

# Re-run failed tests
npx playwright test --last-failed

# View report
npx playwright show-report
```

### Configuration

- Base URL: `http://localhost:8888` (or `KESTREL_URL` env var)
- Timeout: 120 seconds (LLM calls can be slow)
- Reports: `tests/e2e/playwright-report/`

## CI/CD Integration

The GitHub Actions workflow runs:

1. Lint and import validation
2. Unit tests
3. Integration tests (with PostgreSQL/Redis services)
4. LLM tests (with API keys from secrets)

E2E tests require a running server and are run manually.

## Troubleshooting

### "Docker credential store not available"
Docker Desktop isn't running or isn't in PATH. Tests will skip automatically.

### "WAL file doesn't exist"
Use `temp_db_with_keeper` fixture to prevent WAL checkpoint.

### Integration tests fail with "service unavailable"
Use `--skip-check` to skip health checks, or start required services.

### Playwright tests timeout
- Increase timeout in `playwright.config.cjs`
- Check server is running at correct port
- Check `KESTREL_URL` environment variable
