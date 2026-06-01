# Authoring multi-agent Workflow scripts

When a Claude session uses the Workflow tool to fan out across subagents — typically `observation → diagnosis → synthesis` — the synthesis stage tends to overreach. It reads upstream findings, draws a confident conclusion about the world, and produces a clean bottom-line… that doesn't match what observation actually returned.

This document is the convention for writing those scripts so synthesis stays honest.

> Filed against [#1484](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/1484). Concrete incident: a triage workflow for `kestrel-sovereign#1479` concluded *"the fix landed on main and shipped in v0.21.0"* while the upstream observation agent had returned `prs_referencing_1479: none, issue_state: OPEN, agent-blocked: true`. The branch was orphaned. The bug shape (file path, line number, lambda) was diagnosed correctly. The state assertion was fabricated.

---

## Vocabulary

| Term | Meaning |
|---|---|
| **Observation** | Stage that fetches facts about the world (gh state, file contents, test results). Returns structured data. |
| **Diagnosis** | Stage that reasons about *what the bug is* given the observations. Conclusions here are model judgments about code shape. |
| **Synthesis** | Stage that produces the bottom-line report. Combines diagnosis with state for the caller's next action. |
| **State assertion** | A claim about what has happened in the world: "X is merged", "the issue is closed", "v0.21.0 contains the fix". |
| **Diagnosis assertion** | A claim about what the bug or code is: "the silent-drop lambda on line 290 is the root cause". |

State and diagnosis are different categories. **Synthesis is allowed to be confident about diagnosis. Synthesis must be strictly observational about state.**

---

## Rules

### 1. Quote upstream observation fields verbatim before drawing state conclusions.

The synthesis prompt must require the agent to **echo** the observation field that justifies each state assertion. Not paraphrase — quote.

**Bad synthesis output:**

> The fix is on main and the issue is resolved.

**Good synthesis output:**

> gh-state observation returned `issue_state: OPEN` and `prs_referencing_1479: none`. Therefore the fix is **not** merged and the issue is **not** resolved. Talon's branch (`issue-1479-fix-workflows-accept-object-ar` at `ab097f78`) is orphaned.

If the agent can't quote the field, it can't make the assertion.

### 2. No state assertions beyond what observation returned.

If observation did not return a `merged_sha` field, synthesis must not claim something merged. If observation returned `pr_open: false`, synthesis must not claim a PR exists. The full set of state claims is a subset of what observation reported.

This rule is what "narrative smoothing" violates: synthesis fills in a satisfying conclusion because the diagnosis was tidy, not because evidence supports it.

### 3. Distinguish diagnosis from state in the prompt.

Tell the synthesis agent the two categories exist and that they're judged by different standards. The prompt should contain something like:

> Diagnosis is a model conclusion — be confident if the upstream signals support it. State is a fact about the world — restrict yourself to what observation observed. Do not infer state from diagnosis.

### 4. Structured `state_evidence` block (optional but recommended).

Have synthesis output a machine-checkable block alongside its prose:

```json
{
  "state_evidence": [
    {"claim": "issue is open",            "source_agent": "gh-state", "source_field": "issue_state",       "source_value": "OPEN"},
    {"claim": "no PR references the issue", "source_agent": "gh-state", "source_field": "prs_referencing_1479", "source_value": "[]"}
  ]
}
```

Then a deterministic post-check can flag any narrative state claim with no entry. The model will learn that ungrounded claims get caught.

---

## Reference synthesis prompt

A reusable shape for the synthesis stage in any `observation → diagnosis → synthesis` workflow:

```text
You are synthesizing observations into a single report for the caller.

Two categories of claims:
- DIAGNOSIS claims describe what the bug or code is. You may be confident about
  these if upstream diagnosis output supports them.
- STATE claims describe what has happened in the world: merge status, issue
  status, file existence, deployment status, ticket labels. You must be
  strictly observational about state.

Strict rules for STATE claims:
1. Quote the upstream observation field that justifies each state claim.
   Format: "<observation-agent> reported `<field>: <value>`. Therefore <claim>."
2. If no observation returned the field you would need, do not make the claim.
   Say "no observation covered this" instead.
3. Do not infer state from diagnosis. A correct-looking bug fix does not imply
   the fix has shipped.

Output one prose summary, followed by a machine-checkable state_evidence list
where each entry maps a state claim to its source_agent + source_field +
source_value.
```

Inline this template in any script where synthesis has to combine observation and diagnosis outputs.

---

## When this matters most

- **Triage workflows** — the caller is about to comment on a ticket, close it, or open a PR based on synthesis. Wrong state → wrong action.
- **Migration / cleanup audits** — synthesis decides whether a sweep is "done."
- **Rescue workflows** — synthesis decides whether an orphaned branch needs picking up. (The `#1479` incident.)

Diagnosis-heavy workflows (pure research, pure design) are less exposed because they don't drive an action against external state.

---

## Anti-patterns to refuse in review

- Synthesis output that says "X is done" / "X has shipped" / "X is merged" without quoting an observation field.
- Synthesis output that rounds an "open" issue to "resolved" because diagnosis is satisfying.
- Synthesis output whose state claims drift more confident than the upstream evidence (observation: `unclear` → diagnosis: `likely X` → synthesis: `X`).

If you see any of these in a workflow run, treat them as you would a hallucinated function name: ground-truth them against the observation outputs before acting.
