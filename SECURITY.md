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

## Feature Trust Model

Kestrel features are **trusted, in-process extensions** — not a sandbox.

A feature is loaded either from the in-tree `kestrel_sovereign/features/`
directory or from any pip package registered under the
`kestrel_sovereign.features` entry-point group. **Every loaded feature receives
the full, unrestricted agent object** and through it can reach:

- the LLM service (spend tokens, drive the model),
- the agent's storage and a shared, unscoped database handle,
- sibling features — including `keys` (i.e. `get_key()` for stored credentials),
- the hooks manager and identity.

There is **no capability scoping, signing requirement, or sandbox** between a
feature and the agent. A typo-squatted or compromised dependency registered under
that entry-point group is therefore equivalent to full agent compromise.

**This is an accepted v1 trade-off, stated explicitly here so it is not a
surprise:** installing a feature is a trust decision on par with installing any
other Python dependency that runs in your process.

**Operator guidance:**

1. Treat `pip install`-ing a `kestrel-feature-*` (or anything that registers a
   `kestrel_sovereign.features` entry point) as granting it full agent access.
   Vet the source and pin versions, exactly as you would any dependency with
   access to your secrets.
2. Prefer in-tree / first-party features and audited packages.
3. Run agents with the least-privileged host credentials that still let them do
   their job, so a compromised feature's blast radius is bounded by the process,
   not the host.
4. Disable features you don't use (`KESTREL_DISABLED_FEATURES`) to shrink the
   loaded surface.

Tightening this — per-feature capability scoping, signed/allowlisted entry-point
features, and per-feature database namespacing — is tracked as future hardening.

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

### Paid Security Audits

For experienced security researchers interested in comprehensive audits:

Email security@kestrelsovereign.com with:
1. Brief background (CVEs, publications, previous audits)
2. Area of interest from list above
3. Proposed scope and timeline

We're building public infrastructure for sovereign AI. Help us make it secure.

---

Thank you for helping keep Kestrel secure.
