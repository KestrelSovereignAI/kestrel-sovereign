---
type: Developer Note
title: 'Scout-Plan-Code: Agentic Development Workflow'
description: '**Date:** November 30, 2025 **Status:** Proposed **Integration:** A2A
  Protocol, Feedback System, Task Manager'
resource: /docs/development/SCOUT_PLAN_CODE.md
tags:
- docs
- development
- developer-note
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: internal
---

# Scout-Plan-Code: Agentic Development Workflow

**Date:** November 30, 2025  
**Status:** Proposed  
**Integration:** A2A Protocol, Feedback System, Task Manager

---

## Problem Statement

AI agents (and humans working with them) often:
1. Make assumptions without verifying code
2. Miss existing implementations
3. Document features that don't exist
4. Skip systematic investigation
5. Create duplicate or conflicting code

**Evidence:** Every time the user asked "are you sure?" today, the agent found something new.

---

## Proposed Workflow: Scout → Plan → Code

### Phase 1: SCOUT (Investigation)

**Goal:** Systematically discover what actually exists before planning changes.

**Actions:**
1. **File discovery** - Find all relevant files by pattern
2. **Grep verification** - Search for specific implementations
3. **Import chain** - Verify dependencies are actually used
4. **Runtime check** - Confirm code executes as expected
5. **Document findings** - Create audit trail

**Output:** Scout Report (structured JSON or markdown)

```python
# Example Scout Report structure
{
    "topic": "token_counting",
    "searched_patterns": ["tiktoken", "count_token", "num_token", "max_tokens"],
    "files_checked": 150,
    "findings": {
        "tiktoken_installed": True,
        "tiktoken_imported": False,
        "fake_counting_locations": ["kestrel/endpoints/openai_compat.py:203"],
        "real_counting_locations": []
    },
    "confidence": "high",
    "verified_by": ["grep", "import_check", "runtime_test"]
}
```

### Phase 2: PLAN (Design)

**Goal:** Design implementation based on verified facts.

**Actions:**
1. **Gap analysis** - What's missing vs what exists
2. **Integration points** - Where does new code connect
3. **Risk assessment** - What could break
4. **Effort estimation** - Realistic time/complexity
5. **Review existing patterns** - Follow project conventions

**Output:** Implementation Plan with:
- Files to modify (with line numbers)
- New files to create
- Tests to write
- Migration/rollback plan

### Phase 3: CODE (Implementation)

**Goal:** Execute plan with verification at each step.

**Actions:**
1. **Implement incrementally** - Small, testable changes
2. **Verify each change** - Run tests, check imports
3. **Update documentation** - Keep docs accurate
4. **Submit feedback** - Log issues discovered during implementation

---

## Integration with Existing Systems

### A2A Protocol Integration

The Scout-Plan-Code workflow maps to A2A task states:

| Phase | A2A State | A2A Components Used |
|-------|-----------|---------------------|
| Scout | SUBMITTED → WORKING | `ObservabilityStore.log_tool_call()` |
| Plan | INPUT_REQUIRED | `FeedbackStore.submit_feedback()` |
| Code | WORKING → COMPLETED | `TaskStore`, `SessionService` |

**Implementation:**

```python
# a2a/stores/orchestration_store.py (new or extend)
from a2a.types import TaskState
from a2a.stores import TaskManager, FeedbackStore

class ScoutPlanCodeOrchestrator:
    """Orchestrate Scout-Plan-Code workflow as A2A tasks."""
    
    def __init__(self, task_manager: TaskManager, feedback_store: FeedbackStore):
        self.task_manager = task_manager
        self.feedback_store = feedback_store
    
    async def start_investigation(self, topic: str, session_id: str) -> str:
        """Start a Scout phase for a topic."""
        task = await self.task_manager.create_task(
            params=TaskSendParams(
                sessionId=session_id,
                message=Message(role="user", parts=[TextPart(text=f"Scout: {topic}")])
            ),
            agent_name="scout_agent"
        )
        return task.id
    
    async def submit_scout_report(self, task_id: str, report: dict) -> None:
        """Submit scout findings and transition to plan phase."""
        await self.feedback_store.submit_feedback(
            agent_name="scout_agent",
            source=FeedbackSource.AGENT,
            category=FeedbackCategory.IMPROVEMENT,
            severity=FeedbackSeverity.MEDIUM,
            title=f"Scout Report: {report['topic']}",
            description=json.dumps(report, indent=2),
            context=report
        )
```

### Feedback System Integration

Use existing `/feedback/` directory and `FeedbackStore`:

| Feedback Type | Use Case |
|---------------|----------|
| `AGENT` source | Agent self-discovered issues during scout |
| `BUG` category | Code that claims to work but doesn't |
| `IMPROVEMENT` | Missing features identified |
| `CONFUSION` | Unclear documentation |

**File-based feedback** (current `/feedback/*.md` pattern):
- Scout reports → `feedback/scouts/YYYY-MM-DD_topic.md`
- Implementation plans → `feedback/plans/YYYY-MM-DD_topic.md`
- Post-mortems → `feedback/postmortems/YYYY-MM-DD_topic.md`

---

## Scout Checklist Template

```markdown
## Scout Report: [TOPIC]

**Date:** YYYY-MM-DD
**Investigator:** [Agent/Human]
**Confidence:** [Low/Medium/High]

### Search Patterns Used
- [ ] Pattern 1: `grep -rn "pattern" --include="*.py"`
- [ ] Pattern 2: ...

### Files Checked
- Total files scanned: N
- Relevant files found: M

### Findings

#### Exists (Verified)
| What | Location | Verification |
|------|----------|--------------|
| ... | file:line | grep/import/runtime |

#### Missing (Verified Absent)
| What | Searched For | Verification |
|------|--------------|--------------|
| ... | patterns | grep returned 0 |

#### Uncertain (Needs More Investigation)
| What | Issue | Next Step |
|------|-------|-----------|
| ... | ... | ... |

### Dependencies
- Installed but unused: ...
- Used but not installed: ...
- Version mismatches: ...

### Recommendations
1. ...
2. ...
```

---

## Example: Chat Context Management Scout

See `/feedback/chat_session_management.md` for the completed scout that triggered this workflow design.

**Key Findings:**
- `tiktoken` installed but never imported
- Fake token counting using word count × 1.3
- Character-based limits (50K chars) unrelated to actual tokens
- No compaction, no UI gauge, no `!compact` command

---

## Implementation Priority

### Phase 1: Scout Tooling
- [ ] Create `scripts/scout.py` - Automated codebase investigation
- [ ] Create `/feedback/scouts/` directory structure
- [ ] Add scout report template to `features/feedback/`

### Phase 2: A2A Integration
- [ ] Extend `TaskManager` with scout/plan/code task types
- [ ] Add `ScoutReport` type to `a2a/types.py`
- [ ] Integrate with existing `FeedbackStore`

### Phase 3: Agent Training
- [ ] Document workflow in AGENTS.md
- [ ] Create example scouts for common topics
- [ ] Add "are you sure?" checkpoint to agent prompts

---

## Success Metrics

1. **Fewer false claims** - Agents verify before stating
2. **Audit trail** - All investigations documented
3. **Faster debugging** - Scout reports show what was checked
4. **Pattern library** - Reusable search patterns accumulate
5. **Reduced rework** - Plan before code prevents mistakes

---

## Related Files

**A2A Protocol:**
- `a2a/task_manager.py` - Task lifecycle
- `a2a/stores/feedback_store.py` - Feedback persistence
- `a2a/stores/observability_store.py` - Activity logging

**Existing Feedback:**
- `feedback/*.md` - Current feedback documents
- `features/feedback/feature.py` - Feedback agent feature

**Documentation:**
- `AGENTS.md` - Agent instructions (add workflow reference)
- `docs/IMPLEMENTATION_PLAN.md` - Add scout phase requirement
