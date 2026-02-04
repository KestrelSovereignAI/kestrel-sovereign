---
applyTo: '**'
---
# Global Standards

**🐢 See /Volumes/data2/projects/AGENTS.md for the Tortoise Philosophy and complete coding standards.**

Always read AGENTS.md (local and global) when starting a new session or after compaction.

Quick reference - the 5 core principles:
1. ONE SOURCE OF TRUTH - Every concept has ONE canonical implementation
2. FIX ROOT CAUSES - The symptom is not the disease
3. DESIGN BEFORE IMPLEMENTATION - Think end-to-end before writing code
4. INTERFACES OVER IMPLEMENTATIONS - Build contracts, honor them
5. TECHNICAL DEBT IS REAL DEBT - Every shortcut has interest payments

# Project Context

This is **Kestrel Sovereign** - a Constitutional AI Agent Framework. See local AGENTS.md for project-specific instructions including:
- Key directories and file locations
- Testing commands and strategies
- GitHub ticket processor workflow
- Common tasks and patterns

# Critical Rules

- NO hacks, workarounds, or temporary solutions
- NO hardcoded values, print debugging, or commented-out code
- ALWAYS run tests after changes: `./run_tests.py --unit --skip-check`
- If something isn't working - STOP AND ASK FOR GUIDANCE
- Challenge bad patterns - don't be agreeable when wrong
