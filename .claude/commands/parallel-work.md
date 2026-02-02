Launch 3 specialist subagents to work in parallel on separate worktrees.

This command orchestrates parallel development across three Kestrel feature areas:

1. **LLM Service Hardening** (llm-service-specialist agent)
   - Worktree: ../kestrel-llm-service
   - Branch: feature/llm-service-hardening
   - Task: Implement streaming, structured output, vision support

2. **Constitution Verification** (constitution-verifier agent)
   - Worktree: ../kestrel-verify-constitution
   - Branch: feature/verify-constitution-inception
   - Task: Verify and document constitution embedding

3. **Privacy Settings** (privacy-architect agent)
   - Worktree: ../kestrel-privacy-settings
   - Branch: feature/privacy-settings-ui
   - Task: Implement 5-level privacy system with UI

Process:
1. Verify all worktrees exist (or create them)
2. Launch all 3 subagents in parallel (single message, 3 Task calls)
3. Monitor progress
4. Collect reports from each agent
5. Coordinate integration of completed work

Use this command when you have multiple independent features to develop simultaneously.
