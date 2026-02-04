# Kestrel Sovereign — Agent Instructions

> **🐢 See [/Volumes/data2/projects/AGENTS.md](../AGENTS.md) for the Tortoise Philosophy and global coding standards.**

---

## Project Overview

Kestrel Sovereign is a Constitutional AI Agent Framework with cryptographic identity (DIDs). It provides:
- FastAPI server with agent endpoints
- Multi-LLM support (Anthropic, OpenAI, Gemini, Ollama)
- Privacy-preserving agent memory
- Constitutional protections
- GPU cloud deployment (RunPod, Vast.ai, GCP)

## Key Directories

- `kestrel_sovereign/` - Main package
- `endpoints/` - FastAPI route handlers
- `tests/` - Test suite (pytest)
- `docker/` - Docker configurations
- `scripts/` - Utility scripts

## Running Tests

See the full testing guide: [`docs/architecture/testing/TESTING_GUIDE.md`](docs/architecture/testing/TESTING_GUIDE.md)

### Quick Start

```bash
# Unit tests (fast, no dependencies)
./run_tests.py --unit --skip-check

# Integration tests
./run_tests.py --integration --skip-check

# Re-run only failed tests
./run_tests.py --unit --failed

# E2E tests (requires running server)
uv run python -m kestrel_sovereign.server &
cd tests/e2e && npx playwright test
```

### Test Pyramid Strategy

Run tests in order: Unit → Integration → E2E. Fix failures before moving up.

## GitHub Ticket Processor

Autonomous GitHub issue processing with clarification workflow. See full documentation: [`kestrel_sovereign/github_processor/README.md`](kestrel_sovereign/github_processor/README.md)

### Quick Start

```bash
# Process a specific issue
uv run kestrel-github claim --repo owner/repo --issue 42

# Process all 'enhancement' issues
uv run kestrel-github claim --repo owner/repo --label enhancement

# Skip clarification phase
uv run kestrel-github claim --repo owner/repo --issue 42 --skip-clarification
```

### Workflow

1. **Clarification Phase** - Agent analyzes issue, posts checkbox questions if unclear
2. **Implementation Phase** - Agent implements changes autonomously
3. **CI Phase** - Push, wait for CI, auto-fix failures (up to 3 retries)
4. **PR Phase** - Create pull request linked to issue

### Labels

| Label | Meaning |
|-------|---------|
| `agent-analyzing` | Reviewing issue for clarity |
| `agent-clarifying` | Posted questions, waiting for answers |
| `agent-ready` | Skip clarification (pre-approved) |
| `agent-claimed` | Actively implementing |
| `agent-blocked` | Needs human help |
| `agent-complete` | PR ready for review |

### Environment Variables

```bash
GITHUB_TOKEN=ghp_...              # GitHub PAT with repo access
ANTHROPIC_API_KEY=sk-ant-...      # Claude API key
GITHUB_HUMAN_REVIEWER=username    # Human for blocked issues (optional)
```

## Common Tasks

### Adding a new endpoint
1. Create handler in `endpoints/`
2. Register in `server.py`
3. Add tests in `tests/`

### Modifying agent behavior
1. Check `kestrel_sovereign/agent.py`
2. Review constitutional protections in `kestrel_sovereign/constitution.py`
3. Run constitution-verifier tests

### Working with LLM providers
1. Config in `llm_config.toml`
2. Provider implementations in `kestrel_sovereign/llm/`

---

## 📚 Lessons Learned (Case Studies)

### Model Selection System - What NOT to Do

The model selection system became a maintenance nightmare because quick fixes accumulated:

**What happened:**
1. Someone needed to set a model → added `set_default_model()`
2. Later, mandate system needed → added `set_model_preference()` in mixin
3. Later, simpler API needed → added ANOTHER `set_model_preference()` that shadows the first
4. Each endpoint/feature reimplemented "get current model" logic slightly differently
5. Nobody consolidated → 3 ways to set, 4 places with duplicate fallback logic

**The result:**
- User sets model via UI → calls one method
- User sets model via command → calls different method
- API endpoint checks → uses yet another path
- Actual LLM routing → checks a fourth place
- Nothing stays in sync!

**The fix (Feb 2026):**
- Consolidated to ONE source of truth: `llm_service.get_model_preference()`
- All callers now use the same method
- Removed duplicate fallback logic from 4 places
- Deleted legacy methods that used wrong APIs

**The lesson:**
Before adding a new method, SEARCH for existing ones. If one exists, EXTEND it.
If you find yourself copying logic, EXTRACT it to a shared function first.
