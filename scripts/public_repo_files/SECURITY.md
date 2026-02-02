# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately via email to:

**unclesaurus@proton.me**

Do NOT create public GitHub issues for security vulnerabilities.

## What to Include

When reporting, please include:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

## Response Timeline

* **Acknowledgment**: Within 48 hours
* **Initial assessment**: Within 7 days
* **Resolution target**: Within 30 days for critical issues

## Supported Versions

| Version | Supported |
| ------- | --------- |
| Latest  | Yes       |
| < 1.0   | No        |

## Security Best Practices

When using Kestrel:

1. Keep API keys in environment variables, never in code
2. Use the encryption features for sensitive data
3. Enable privacy modes appropriate to your use case
4. Regularly rotate cryptographic keys
5. Review agent permissions before deployment

## Responsible Disclosure

We follow responsible disclosure practices. After a fix is released,
we will publicly acknowledge the reporter (unless anonymity is requested).

Thank you for helping keep Kestrel secure.
