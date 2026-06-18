---
type: Principle Document
title: The Tortoise Doctrine
description: '*"Slow is smooth, smooth is fast."*'
resource: /docs/TORTOISE_DOCTRINE.md
tags:
- docs
- principle-document
timestamp: '2026-06-18T00:00:00Z'
status: active
owner: documentation
canonical: false
generated: false
privacy: public
---

# The Tortoise Doctrine

> *"Slow is smooth, smooth is fast."*

This is the canonical statement of how we build Kestrel Sovereign. The repo's
[AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md) both link here — read
this file once at the start of every session.

---

## Bet on the tortoise. Always.

We prioritize doing things *right* over doing things *fast*. Technical debt
compounds exponentially — a "quick fix" today becomes 10 hours of debugging
next week, becomes a rewrite next month.

We build software that lasts.

---

## Core Principles

### 1. One Source of Truth

Every concept has ONE canonical implementation. Not two. Not "one here and a
helper over there."

- If you find 3 ways to do the same thing → consolidate to 1
- If similar logic exists in multiple places → extract to a shared function
- NEVER copy-paste-modify. Extract and parameterize.

### 2. Fix Root Causes, Not Symptoms

When something breaks, resist the urge to patch it. Understand it.

- If something doesn't work → understand WHY before changing anything
- If a test fails → the test is telling you about a real bug
- If you need a workaround → STOP and ask for guidance

*The symptom is not the disease.*

### 3. Design Before Implementation

Think end-to-end before writing code.

- Consider: "Where else might this be called from?"
- Ask: "What happens when this fails?"
- Draw it out if needed: `UI → API → Service → Storage`

### 4. Interfaces Over Implementations

Build contracts between layers. Honor them.

- Changes to implementation shouldn't break consumers
- If you need to change an interface → update ALL callers

### 5. Technical Debt is Real Debt

Every shortcut has interest payments.

- "We'll fix it later" usually means "We'll rewrite it later"
- If you can't do it right now, create a GitHub issue with full context

### 6. When You See Something, Say Something

While working on task A, you'll often pass by an unrelated bug, broken test,
stale doc, or sketchy pattern. **Don't walk past it.**

- **Don't ignore an error because "this session didn't cause it."** The repo
  is a shared system. An error you saw and ignored is now an error you own.
- **Fix easy things in the same PR.** A typo, a one-line dead import, a
  comment that lies about what the code does — fix it. The cost of fixing is
  lower than the cost of context-switching back to it later.
- **File a ticket for not-easy things.** If the problem is bigger than the
  current change can absorb, open a GitHub issue with full reproduction
  steps and a link to where you saw it. Then keep moving.
- **Surface it in the PR.** A "noticed in passing" line in the PR description
  is how unrelated drift becomes visible to reviewers and gets routed.

The bar is: a future engineer reading the diff or the issue tracker should
be able to act on what you saw, even if you didn't fix it yourself. Silent
bystanders are how rot spreads.

---

## Anti-Patterns to Avoid

| Don't | Why |
|-------|-----|
| Add a new method when an existing one could be extended | Creates confusion and divergence |
| Reimplement logic that exists elsewhere | Bugs won't be fixed everywhere |
| Shadow a base class method with incompatible behavior | Creates traps for future developers |
| Use hardcoded values "just for now" | "Now" never ends |
| Skip tests because "it's a simple change" | Simple changes break complex systems |
| Catch generic exceptions to "make it work" | Hides bugs instead of fixing them |
| Walk past an error you didn't cause | You saw it, you own it. Fix or file. |

---

## When You're Unsure

1. **Read the existing code thoroughly** — the answer is often already there
2. **Search for similar patterns** — `grep -r` is your friend
3. **Ask: "How would a senior engineer approach this?"**
4. **If still unsure → STOP AND ASK**

---

## Professional Coding Standards

### NO HACKS, WORKAROUNDS, OR 'SPACKLE'

- Do NOT work around issues — fix them properly
- Do NOT create mock tests or temporary solutions
- If something isn't working — STOP AND ASK FOR GUIDANCE
- NO fallbacks in production code — fail fast and fail clearly

### NO AMATEUR CODING PRACTICES

- NO hardcoded values — use environment variables or configuration files
- NO `print()` statements for debugging — use proper logging
- NO commented-out code — delete it or use version control
- NO duplicate code — extract to functions or modules
- NO catching generic exceptions — handle specific errors
- NO global variables — use proper dependency injection
- NO mixed concerns — separate business logic from I/O
- NO untested error paths — test failures, not just success
- NO absolute local paths in committed code — derive from `__file__`, env
  vars, or repo-relative

### TESTING IS NOT OPTIONAL

- EVERY code file MUST have corresponding tests
- Run tests after EVERY change
- Fix failing tests before proceeding
- Test failures should stop everything

### FAIL FAST PHILOSOPHY

- Applications should fail immediately on errors
- No silent failures or fallbacks
- Clear error messages that explain the problem

### CHALLENGE WHEN WRONG

- If something is wrong, say so clearly with evidence
- Don't accept bad patterns or practices
- Provide better alternatives when you disagree
