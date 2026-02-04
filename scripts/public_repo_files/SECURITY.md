# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately via email to:

**security@kestrelsovereign.com**

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

---

## Security Research Opportunities

We're interested in working with qualified security researchers to audit Kestrel's core security systems.

### Areas of Interest

- **DID (Decentralized Identity) implementation** - W3C did:pkh format verification
- **Constitutional verification system** - Genesis audit integrity checks
- **Cryptographic key management** - Agent wallet security, key rotation
- **Privacy mode enforcement** - EPHEMERAL through PUBLIC data isolation
- **Agent-to-agent (A2A) protocol** - Cross-agent communication security

### Qualification

We're looking for security researchers with:

- Published CVEs or security research
- Contributions to security tools or frameworks
- Verifiable track record in application security or cryptography
- Experience with Python security audits

### Engagement

- **Bug bounty**: We're evaluating platforms like HackerOne for public bug bounty program
- **Paid audit**: Specific scope, fixed engagement for experienced researchers
- **Academic collaboration**: Research partnerships for published work

If you have a strong security background and want to audit Kestrel's implementation, email security@kestrelsovereign.com with:

1. Brief background (CVEs, publications, tools, previous audits)
2. Area of interest from list above
3. Preferred engagement type (bounty, paid audit, research collaboration)

We're building public infrastructure for sovereign AI. Help us make it secure.

---

Thank you for helping keep Kestrel secure.
