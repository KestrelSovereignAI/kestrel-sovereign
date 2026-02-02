# Contributing to Kestrel

Thank you for your interest in contributing to the Kestrel Sovereign AI Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How to Contribute

### Reporting Issues

- **Security vulnerabilities**: Please report security issues privately via email to unclesaurus@proton.me. Do not create public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for details.
- **Bugs**: Open an issue on GitHub with a clear description, steps to reproduce, and expected vs actual behavior.
- **Feature requests**: Open an issue describing the feature and its use case.

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Follow the coding style** used throughout the project.
3. **Write tests** for any new functionality.
4. **Update documentation** as needed.
5. **Sign your commits** (see below).
6. **Submit a pull request** with a clear description of the changes.

### Commit Signing

We encourage (but don't require) signing your commits with GPG or SSH keys. This helps verify the authenticity of contributions.

### Contributor License Agreement

By submitting a pull request, you agree that your contributions are made under the Apache 2.0 license and that you have the right to make such contributions.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend development)
- Docker (for containerized development)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/Kestrel-Sovereign-AI/kestrel.git
cd kestrel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy example environment file
cp .env.example .env

# Run tests
pytest tests/
```

### Using Dev Containers

We provide a dev container configuration for VS Code:

1. Install the "Dev Containers" extension
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted

## Coding Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Use async/await for I/O operations

### Testing

- Write unit tests for new functionality
- Integration tests for component interactions
- Aim for meaningful coverage, not 100% coverage

### Documentation

- Update README.md if adding new features
- Add docstrings to new modules and functions
- Update architecture docs for significant changes

## Constitutional Alignment

Kestrel is built on a constitutional governance model. Contributions should:

1. **Respect user sovereignty** - Users control their data and agents
2. **Preserve privacy** - Default to privacy-preserving behavior
3. **Maintain transparency** - Behavior should be auditable
4. **Support cryptographic integrity** - Changes shouldn't weaken security

When in doubt, refer to the [Kestrel Constitution](docs/principles/KESTREL_CONSTITUTION.md).

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` for beginner-friendly tasks.

### High-Impact Areas

- **LLM Adapters**: Support for new model providers
- **Storage Backends**: Additional storage implementations
- **Privacy Features**: Enhanced privacy mode support
- **Documentation**: Tutorials, examples, and guides
- **Testing**: Improved test coverage

## Getting Help

- **GitHub Discussions**: For questions and community discussion
- **GitHub Issues**: For bugs and feature requests
- **Email**: unclesaurus@proton.me for private matters

## Recognition

Contributors are recognized in our release notes. Significant contributors may be invited to join the project's governance.

## Pseudonymous Contributions

Following the project's values, we welcome and support pseudonymous contributions. You don't need to use your real name - your code speaks for itself.

---

Thank you for helping make Kestrel better!
