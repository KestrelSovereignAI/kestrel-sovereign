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

### PostgreSQL Tests

A dual-backend test runs its PostgreSQL case when `TEST_POSTGRES_URL` is set.
That database outlives the run, and a developer's is often reused between runs
or shared. Only an xdist worker gets a schema of its own
([`tests/shared/postgres_worker_isolation.py`](../../../tests/shared/postgres_worker_isolation.py));
a serial run uses the URL's schema as it is. A PostgreSQL case must therefore
leave alone every row and column it did not create:

- A case whose operation spans a whole table (a backfill, a migration, a
  table-wide count) runs in a schema of its own:
  `disposable_postgres_schema` from
  [`tests/utils/postgres_schema.py`](../../../tests/utils/postgres_schema.py),
  dropped on exit. Deleting the case's own rows at teardown does not undo a
  write to rows it never inserted.
- Create that schema through a connection opened with a `schema_initializer`
  that issues no DDL. The default initializer boots the URL's schema, and the
  startup sequence rewrites rows already there.
- A case that needs pgvector puts the schema `pgvector_schema()` returns on its
  search path: the extension lives in whichever schema first installed it.

`tests/unit/test_embedding_vec_backfill.py` is a worked example, including a
regression test that the case leaves a reused database unchanged (#3404).

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

### CI smoke subset

Pull-request CI runs one Playwright project, `console-smoke` in
`tests/e2e/playwright.config.cjs`, in the `console-smoke` job of
`.github/workflows/ci.yml`. It proves the shipped console boots, bootstraps its
own API key, routes to the right agent, shows its DID, and answers `!status`
through the chat. It calls no LLM. Run the same subset locally with:

```bash
npx playwright install chromium   # once
uv run kestrel demo smoke         # fresh temp home, port 8910
uv run kestrel demo smoke --home /tmp/smoke-home --port 8920 --keep-server
```

`kestrel demo smoke` never targets your server or agents. It creates a fresh
instance (agent, SQLite database, and throwaway data key) in a home that must
not exist yet or must be empty, and starts it on a loopback port that is never
8888. The instance inherits no `KESTREL_*` setting, no `OTEL_*` exporter
setting, and no credential-shaped variable. Its tracing is off
(`KESTREL_TRACING_ENABLED=0`), its server reads no dotenv file
(`KESTREL_SKIP_DOTENV=1`), and its only LLM route is local Ollama. Before the
browser starts, the runner checks that the server imports this checkout's
`kestrel_sovereign` and serves the DID this run's inception minted. It writes
those facts to
`<home>/console-smoke-instance.json`; the spec checks them again and refuses
any other target. The server is stopped on success, failure, and Ctrl-C. Its
log stays in `<home>/server.log`.

The `console-smoke` project is registered only when `kestrel demo smoke` sets
`KESTREL_CONSOLE_SMOKE_MANIFEST`, and the default `chromium` project skips the
smoke spec, so a plain `npx playwright test` against your running server is
unaffected.

### Configuration

- Base URL: `http://localhost:8888` (or `KESTREL_URL` env var)
- Timeout: 120 seconds (LLM calls can be slow); the `console-smoke` project
  allows 60 seconds and no retries
- Reports: `tests/e2e/playwright-report/`

## CI/CD Integration

The GitHub Actions workflow runs:

1. Syntax and import validation (the `lint-and-imports` job: ruff F811 on
   changed files, Python 3.11 `compileall`, and module import checks)
2. Unit tests
3. Integration tests (with PostgreSQL/Redis services)
4. LLM tests (with API keys from secrets)
5. The Sovereign Console smoke (`console-smoke`, see
   [CI smoke subset](#ci-smoke-subset)): one Chromium browser against a fresh
   isolated instance, no secrets, no LLM

The rest of the E2E suite requires a running server and is run manually.

The `lint-and-imports` job is not a repository-wide ruff or mypy gate. Broad
lint and type enforcement is staged under the structural quality campaign
([#2480](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/2480)),
which adds narrow regression gates as each baseline is paid down rather than an
always-red tree-wide check.

The `Typing :: Typed` classifier is backed by the PEP 561 marker
`kestrel_sovereign/py.typed`. `tests/unit/test_packaging_typed_marker.py`
builds the wheel, checks that its metadata and contents agree, and confirms a
clean install exposes the marker.

The clean-install matrix and `scripts/ci/clean_install_local.sh` validate
package installation, quickstart files, identity/constitution/memory storage,
server health, host routing, and DID persistence. They create an explicitly
marked test instance with `KESTREL_AUDIT_MODE=skip`: its hash-bound genesis
receipt is **pending**, not passed, and first cognition remains blocked. The
harness makes no audit or embedding model call, even if the developer runs
Ollama locally. The local script requires a fresh checkout, isolates host
state and chooses disposable loopback ports; it does not reuse an operator's
agent or serve as evidence that a real agent passed genesis. Live acceptance
requires a configured auditor's completed low/medium-risk result and the
normal durable integrity checks for the exact governing source.

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
