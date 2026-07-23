# Contributing to Kestrel

Thank you for your interest in contributing. Kestrel is sovereign AI infrastructure — contributions here affect how people own and control their AI agents. We take that responsibility seriously.

<!-- canonical-contributing-notice:start -->
> **Canonical contributor guide:** this root file is the source of truth. The
> public-repository copy at `scripts/public_repo_files/CONTRIBUTING.md` is
> synchronized from it. After editing this guide, run
> `uv run python scripts/sync_public_repo_files.py --check` before opening a
> pull request.
<!-- canonical-contributing-notice:end -->

---

## Table of Contents

- [Getting started](#getting-started)
- [Running tests](#running-tests)
- [Making changes](#making-changes)
- [Pull requests](#pull-requests)
- [Where we need help](#where-we-need-help)
- [Constitutional alignment](#constitutional-alignment)
- [Reporting issues](#reporting-issues)

---

## Getting started

**Prerequisites:** Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/).
Ollama is optional and only needed when you want to exercise a local model
route.

```bash
git clone https://github.com/KestrelSovereignAI/kestrel-sovereign.git
cd kestrel-sovereign
uv sync                    # install all dependencies including dev/test deps
cp .env.example .env       # configure your local environment
uv run kestrel setup       # interactive: writes kestrel.toml [llm] + .env
# or hand-edit: cp kestrel.toml.example kestrel.toml
```

To exercise a local Ollama route, start it in a separate terminal:

```bash
ollama serve
ollama pull llama3.2:3b
```

Verify the setup:

```bash
uv run kestrel doctor
```

---

## Running tests

```bash
# Unit tests (fast, no external provider required)
./run_tests.py --unit --skip-check

# Integration tests (configure any required external provider/service first)
./run_tests.py --integration --skip-check

# A specific test file
uv run pytest tests/unit/test_kestrel_agent.py -v

# Check the repository's documentation invariants
uv run python scripts/generate_feature_docs.py --check
uv run python scripts/docs_verify.py audit --check
uv run python scripts/sync_public_repo_files.py --check
```

The test inventory changes frequently; use the commands above and the
[testing guide](docs/architecture/testing/TESTING_GUIDE.md) instead of relying
on a fixed test count. Before submitting a PR, run the targeted tests for your
change and the relevant repository gate. Run integration tests when your change
or configuration affects an integration path.

---

## Making changes

### Branch naming

```
feature/short-description      # new functionality
fix/issue-number-description   # bug fix
docs/what-you-changed          # documentation only
refactor/what-you-refactored   # refactoring without behavior change
```

### Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
feat(privacy): add per-session memory budget cap
fix(llm): include llama_cpp in local-only provider filter
docs(oss): add SOVEREIGNTY.md architecture guide
test(agent): add constitution audit failure integration test
refactor(storage): extract graph traversal into GraphQuery helper
```

Keep the subject line under 72 characters. Use the body for "why", not "what".

### Code style

- **Python**: PEP 8. Type hints on all public function signatures. Docstrings on public classes and functions. `async/await` for all I/O.
- **No new global state.** Pass dependencies explicitly.
- **No silent fallbacks** that hide errors. Log with appropriate level. Let callers decide how to handle exceptions.
- **Tests required for new functionality.** Bug fixes should include a regression test.

---

## Pull requests

1. Fork the repo and create your branch from `main`.
2. Make your changes with tests.
3. Run the test suite locally — all unit tests must pass.
4. Open a PR with a clear description: what changed, why, and how to verify it.
5. Reference any related issues (`Closes #123`).

PRs are reviewed on a rolling basis. We aim for a first response within a few days. No guaranteed merge timeline — complex changes may go through multiple rounds.

### Contributor License Agreement

By submitting a PR, you confirm that your contributions are made under the Apache 2.0 license and that you have the right to contribute them.

---

## Where we need help

Browse the current
[open issues](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues)
and the `agent-ready` label for work that has a current owner, scope, and
acceptance criteria.

### Core infrastructure

| Area | What's needed | Skill level |
|------|--------------|-------------|
| Privacy modes | Storage-policy tests, UX clarity, and documentation | Intermediate |
| Constitution | Authoring and governance tooling | Intermediate |
| LLM service | Provider and route adapters | Advanced |
| Memory / RAG | Retrieval quality improvements, embedding model options | Advanced |
| Cryptographic anchoring | Audit trail UI, key rotation support | Advanced |

### Documentation & DX

| Area | What's needed | Skill level |
|------|--------------|-------------|
| Installation validation | Test on clean supported platforms | Beginner |
| Tutorials | "Build your first companion" walkthrough | Intermediate |
| API reference | Auto-generated docs from type hints | Intermediate |

### Testing

| Area | What's needed | Skill level |
|------|--------------|-------------|
| Frontend auth | Unit tests for console auth flows | Intermediate |
| Load testing | Concurrent agent sessions under load | Advanced |
| E2E | Clean-install automation (Mac/Linux/Windows) | Intermediate |

If you're not sure where to start, open an issue and say what you'd like to work on — we'll point you to the right place.

---

## Constitutional alignment

Kestrel exists to give people sovereignty over their AI agents. All contributions should align with these principles:

1. **User sovereignty first** — the user controls their agent, their data, and their keys. Contributions that reduce user control require strong justification.
2. **Privacy by default** — new features should default to privacy-preserving behavior. Don't store what you don't need.
3. **Transparency and auditability** — behavior should be explainable and observable. Avoid black-box side effects.
4. **Cryptographic integrity** — don't bypass or weaken the identity or constitution systems to make something easier.
5. **No platform lock-in** — don't introduce dependencies that tie the agent to a single cloud provider.

See [docs/principles/KESTREL_CONSTITUTION.md](docs/principles/KESTREL_CONSTITUTION.md) for the full governance document.

---

## Reporting issues

- **Security vulnerabilities**: Report privately to security@kestrelsovereign.com — see [SECURITY.md](SECURITY.md).
- **Bugs**: Open an issue with steps to reproduce, expected behavior, and actual behavior. Include your OS and Python version.
- **Feature requests**: Open an issue describing the use case. The more concrete, the better.

---

## Getting help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions, ideas, design conversations
- **Email**: hello@kestrelsovereign.com for private inquiries

---

*See [docs/SOVEREIGNTY.md](docs/SOVEREIGNTY.md) for the architectural context behind these principles.*
