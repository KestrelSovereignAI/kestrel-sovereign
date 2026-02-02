# Kestrel Test Suite

Commands to run the various test categories for the Kestrel Sovereign AI Agent framework.

## Prerequisites

```bash
# Use uv directly (recommended - handles venv automatically)
uv run pytest ...

# Or activate virtual environment manually
source .venv/bin/activate
pytest ...
```

## Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific unit test file
uv run pytest tests/unit/test_key_storage.py -v

# Run with fail-fast (stop on first failure)
uv run pytest tests/unit/ -x
```

### Unit Test Files
- `test_key_storage.py` - Secure key storage (AES-256-GCM)
- `test_privacy_agent.py` - Privacy agent functionality
- `test_privacy_wrapper.py` - Privacy enforcing storage wrapper
- `test_pii_detector.py` - PII detection (spaCy NER)
- `test_wallet_agent.py` - Wallet and economics
- `test_compute_feature.py` - Compute/script execution
- `test_github_feature.py` - GitHub code introspection
- `test_feature_discovery.py` - Auto-discovery of features
- `test_hooks.py` - Security hooks
- `test_docstring_parser.py` - Tool parameter parsing
- `test_model_metadata.py` - LLM model metadata
- `test_adapter_list_models.py` - Model listing adapters
- `test_streaming_audit.py` - Streaming with audit
- `test_mcp_healthcheck.py` - MCP container health
- `test_storage_providers.py` - Storage provider tests
- `storage/test_db_backends.py` - Database backend tests

## Integration Tests

```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run with fail-fast
uv run pytest tests/integration/ -x

# Run specific integration test
uv run pytest tests/integration/test_genesis_audit_e2e.py -v

# Run tests requiring Docker
uv run pytest tests/integration/ -m docker

# Run adversarial security tests
uv run pytest tests/integration/ -m adversarial
```

### Integration Test Files
- `test_genesis_audit_e2e.py` - Agent inception with constitution audit
- `test_agent_tools_e2e.py` - Agent tools framework
- `test_tool_calling_e2e.py` - LLM tool calling
- `test_mcp_tools_e2e.py` - MCP server integration
- `test_agent_mcp_workflow_e2e.py` - Agent + MCP workflow
- `test_dynamic_features.py` - Dynamic feature loading
- `test_compute_security_integration.py` - Compute with security
- `test_llm_model_pulling_e2e.py` - Model pulling from Ollama
- `test_llm_space_mgmt_e2e.py` - Disk space management
- `test_model_mandate_e2e.py` - Model mandate enforcement
- `test_orchestration_e2e.py` - Multi-agent orchestration
- `test_ensemble_removal_e2e.py` - Ensemble mode removal
- `test_solvency.py` - Economic solvency checks
- `test_docker.py` - Docker container tests
- `test_runpod_feature.py` - RunPod GPU integration
- `test_reflection_e2e.py` - Agent self-reflection

## LLM Adapter Tests

```bash
# Run LLM adapter tests
uv run pytest tests/llm/ -v

# Test specific adapter
uv run pytest tests/llm/test_adapter.py -v
uv run pytest tests/llm/test_vertex_adapter.py -v
```

## Load Tests (Opt-in)

```bash
# Run load/stress tests (requires --run-load flag)
uv run pytest tests/load/ --run-load -v

# Specific load test
uv run pytest tests/load/test_load_storage.py --run-load -v
```

### Load Tests
- Concurrent write throughput (10 writers, 1000 messages)
- Mixed read/write workloads (70/30 ratio)
- RAG indexing at scale (1000 document chunks)
- Large conversation history (10K messages)
- Concurrent agent sessions (5 parallel agents)

## Cloud Resource Tests (Opt-in)

```bash
# Run tests that use cloud resources (requires --run-cloud flag)
uv run pytest tests/ --run-cloud -v -m cloud_resource
```

## Playwright E2E Tests (Sovereign Console UI)

```bash
# Install Playwright browsers (first time)
cd tests/e2e
npm install
npx playwright install

# Run all E2E tests
npx playwright test

# Run specific test file
npx playwright test test_sovereign_console.spec.cjs

# Run with UI mode (interactive)
npx playwright test --ui

# Run headed (see the browser)
npx playwright test --headed

# Generate test report
npx playwright show-report
```

### E2E Test Files
- `test_sovereign_console.spec.cjs` - Main console UI
- `test_sovereignty_modals.spec.cjs` - Export/import modals
- `test_chat_and_models.spec.cjs` - Chat and model selection

### Environment Variables
```bash
# Override base URL (default: http://localhost:8888)
KESTREL_URL=http://localhost:8888 npx playwright test

# Set API key (optional - tests fetch from /api/auth/key)
KESTREL_API_KEY=your-key npx playwright test
```

## Pytest Markers

```bash
# Run only tests with specific marker
uv run pytest -m integration tests/
uv run pytest -m adversarial tests/
uv run pytest -m docker tests/
uv run pytest -m dual_backend tests/
uv run pytest -m slow tests/

# Exclude slow tests
uv run pytest tests/ -m "not slow"
```

### Available Markers
- `integration` - Integration tests with real services
- `adversarial` - Security bypass attempt tests
- `dual_backend` - Tests on both SQLite and PostgreSQL
- `docker` - Tests requiring Docker
- `load` - Performance tests (requires `--run-load`)
- `cloud_resource` - Tests using RunPod, GPU (requires `--run-cloud`)
- `slow` - Slow-running tests

## Test Infrastructure

```bash
# Infrastructure tests (RunPod, containers)
uv run pytest tests/infrastructure/ -v

# Build test script
./tests/infrastructure/build_test.sh
```

## Common Options

```bash
# Verbose output
uv run pytest -v

# Extra verbose
uv run pytest -vv

# Show print statements
uv run pytest -s

# Fail fast (stop on first failure)
uv run pytest -x

# Run last failed tests
uv run pytest --lf

# Run tests matching pattern
uv run pytest -k "test_genesis"

# Show test durations
uv run pytest --durations=10

# Coverage report
uv run pytest --cov=. --cov-report=html
```

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   └── storage/             # Storage-specific unit tests
├── integration/             # Integration tests (real services)
│   └── mcp_test_server/     # MCP test server for integration
├── llm/                     # LLM adapter tests
├── load/                    # Load/stress tests (opt-in)
├── infrastructure/          # Infrastructure tests
├── e2e/                     # Playwright E2E tests
│   └── playwright.config.cjs
├── shared/                  # Shared test utilities
│   ├── resource_registry.py # Crash-safe resource tracking
│   ├── cost_tracker.py      # Cloud cost estimation
│   └── pytest_cleanup_plugin.py
└── utils/                   # Test utilities
    ├── async_waits.py       # Condition-based async waits
    ├── parallel_support.py  # Parallel test support
    └── feedback_bridge.py   # Test-to-reflection integration
```
