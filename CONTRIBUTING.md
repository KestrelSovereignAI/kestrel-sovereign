# Contributing to Kestrel

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting Issues

- **Security vulnerabilities**: Report privately to security@kestrelsovereign.com. See [SECURITY.md](SECURITY.md) for details.
- **Bugs**: Open an issue with clear description, steps to reproduce, and expected vs actual behavior.
- **Feature requests**: Open an issue describing the feature and use case.

## Pull Requests

Pull requests are reviewed on a case-by-case basis. No guarantee of merge timeline or acceptance.

If submitting a PR:

1. Fork the repository and create your branch from `main`.
2. Follow the existing coding style (PEP 8 for Python).
3. Write tests for any new functionality.
4. Update documentation as needed.
5. Submit PR with clear description of changes.

### Contributor License Agreement

By submitting a pull request, you agree that your contributions are made under the Apache 2.0 license and that you have the right to make such contributions.

## Coding Standards

### Python
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Use async/await for I/O operations

### Testing
- Write unit tests for new functionality
- Integration tests for component interactions
- See [docs/architecture/testing/TESTING_GUIDE.md](docs/architecture/testing/TESTING_GUIDE.md)

## Constitutional Alignment

Kestrel is built on a constitutional governance model. Contributions should:

1. **Respect user sovereignty** - Users control their data and agents
2. **Preserve privacy** - Default to privacy-preserving behavior
3. **Maintain transparency** - Behavior should be auditable
4. **Support cryptographic integrity** - Don't weaken security

Refer to the [Kestrel Constitution](docs/principles/KESTREL_CONSTITUTION.md) for details.

## Getting Help

- **GitHub Discussions**: Questions and community discussion
- **GitHub Issues**: Bug reports and feature requests
- **Email**: hello@kestrelsovereign.com for private inquiries

---

For security research opportunities, see [SECURITY.md](SECURITY.md).
