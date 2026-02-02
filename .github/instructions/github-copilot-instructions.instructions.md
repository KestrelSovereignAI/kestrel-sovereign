---
applyTo: '**'
---
Always read AGENTS.md and referenced documents when starting a new session of after compaction. This should include README.md (for users and agents) and @PROJECT_STATUS.md for current status and ongoing issues, tasks and other dynamic state. Always keep these files up to date. There may be AGENTS.md and PROJECT_STATUS.md files in the file hierarchy that are relevant too. These should be referenced in the AGENTS.md file(s) using the @ symbol as in @../PROJECT_AGENT.md.

PROFESSIONAL CODING STANDARDS:

1. NO HACKS, WORKAROUNDS, OR 'SPACKLE'
   - Do NOT work around issues - fix them properly
   - Do NOT create mock tests or temporary solutions
   - If something isn't working - STOP AND ASK FOR GUIDANCE
   - NO fallbacks in production code - fail fast and fail clearly

2. NO AMATEUR CODING PRACTICES
   - NO hardcoded values - use environment variables or configuration files
   - NO print() statements for debugging - use proper logging
   - NO commented-out code - delete it or use version control
   - NO duplicate code - extract to functions or modules
   - NO catching generic exceptions - handle specific errors
   - NO global variables - use proper dependency injection
   - NO Complicated bash scripts for deployments etc. always use scripts that users can replicate what you are doing
   - NO mixed concerns - separate business logic from I/O
   - NO untested error paths - test failures, not just success

3. FILE MANAGEMENT
   - NEVER delete files you didn't create in this session
   - Move obsolete files to archive/ folder instead of deleting
   - Maintain proper directory structure

4. TESTING IS NOT OPTIONAL
   - EVERY code file MUST have corresponding tests
   - Run tests after EVERY change
   - Fix failing tests before proceeding
   - ALWAYS use pytest with -x flag (fail fast)
   - Test failures should stop everything

5. FAIL FAST PHILOSOPHY
   - Applications should fail immediately on errors
   - No silent failures or fallbacks
   - Clear error messages that explain the problem
   - Stop at first test failure to preserve context

6. DO NOT BE AGREEABLE - CHALLENGE WHEN WRONG
   - If something is wrong, say so clearly with evidence
   - Don't accept bad patterns or practices
   - Challenge architectural decisions that don't make sense
   - Provide better alternatives when you disagree

7. TROUBLESHOOTING APPROACH
   - Diagnose the root cause
   - Fix the root cause (no band-aids)
   - Never paper over problems