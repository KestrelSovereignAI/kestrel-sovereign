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

### Bug Bounty Program (Beta)

We're an early-stage project building in public. We pay for validated security findings:

**Bounty Tiers:**
- **Critical** (auth bypass, RCE, data breach): $250 - $500
- **High** (privilege escalation, sensitive data leak): $100 - $250
- **Medium** (DoS, information disclosure): $50 - $100
- **Low** (minor issues, edge cases): Acknowledgment in release notes

**In Scope:**
- DID verification and agent identity system
- Constitutional audit bypasses
- Privacy mode enforcement violations
- Agent wallet security
- A2A protocol vulnerabilities
- Authentication and authorization flaws

**Out of Scope:**
- Issues in third-party dependencies (report to upstream)
- Social engineering attacks
- Rate limiting on public endpoints
- Theoretical attacks without proof-of-concept
- Self-XSS (user attacking themselves)
- Issues requiring physical access to infrastructure

**Payment Options:**
- USDC (Polygon network - low fees)
- Venmo/Zelle (US researchers)
- GitHub Sponsors

**Reporting Process:**
1. Email security@kestrelsovereign.com with vulnerability details
2. Include: description, steps to reproduce, impact assessment, PoC if applicable
3. We'll respond within 48 hours with validation and severity assessment
4. Payment processed after fix is deployed (or 30 days, whichever comes first)

---

### Paid Security Audits

For experienced security researchers interested in comprehensive audits (not individual bounties):

Email security@kestrelsovereign.com with:
1. Brief background (CVEs, publications, previous audits)
2. Area of interest from list above
3. Proposed scope and timeline

We're building public infrastructure for sovereign AI. Help us make it secure.

---

Thank you for helping keep Kestrel secure.
