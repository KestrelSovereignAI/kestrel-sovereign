# Spawn Agent Demo — Narrative Beats

> Narrated walkthrough of Kestrel's agent delegation system.
> Each beat corresponds to a screenshot in `demo-output/`.

---

## Beat 1: "Claw receives a complex research request"

The user sends a multi-faceted research question that naturally benefits from
parallel investigation. The agent receives the message and begins reasoning.

**Screenshot:** `01-research-request.png`
**What to see:** Chat panel with the user's complex question visible.

---

## Beat 2: "Claw decides to parallelize"

The agent analyzes the task, determines it has multiple independent axes of
research, and decides to spawn specialist workers rather than tackle everything
sequentially. The agent's response indicates it is delegating work.

**Screenshot:** `02-agent-decides-to-spawn.png`
**What to see:** Agent response explaining its decision to spawn child agents.

---

## Beat 3: "Two workers appear in the Console"

Navigating to the Spawn panel reveals newly created child agents in the active
children table. Each child has a name, purpose description, status (RUNNING),
its own DID, TTL countdown, and budget allocation.

**Screenshot:** `03-spawn-panel-children.png`
**What to see:** Spawn panel with active children table showing 2+ workers.

---

## Beat 4: "Workers begin independent research"

The delegation chain tree visualization shows the parent-child relationship.
Each worker's status is RUNNING as they process their assigned sub-tasks
independently.

**Screenshot:** `04-delegation-chain.png`
**What to see:** Delegation chain tree with parent at root and children below.

---

## Beat 5: "Budget meters tick"

The budget progress bars and Chart.js stacked bar chart show budget allocation
per child. As workers make LLM calls, the "Spent" portion grows while
"Remaining" shrinks.

**Screenshot:** `05-budget-meters.png`
**What to see:** Budget bars with non-zero spent amounts, chart visible.

---

## Beat 6: "First worker reports back"

One worker completes its research and reports results back to the parent.
Its status changes to COMPLETED in the children table. Budget accounting
shows final consumption.

**Screenshot:** `06-first-worker-done.png`
**What to see:** At least one child with completed/terminated status.

---

## Beat 7: "Second worker reports back"

The remaining worker(s) also complete. All children now show terminal status.
The budget chart reflects total consumption across all workers.

**Screenshot:** `07-all-workers-done.png`
**What to see:** All children in terminal state, budget fully accounted.

---

## Beat 8: "Workers auto-terminate"

TTL/completion cleanup occurs. Unspent budget is returned to the parent.
The active children count drops as ephemeral workers are cleaned up.

**Screenshot:** `08-workers-terminated.png`
**What to see:** Children list empty or showing terminated status, budget returned.

---

## Beat 9: "Claw synthesizes results"

Back in the Chat panel, the parent agent combines findings from all workers
into a unified, comprehensive response to the user's original question.

**Screenshot:** `09-synthesized-response.png`
**What to see:** Chat showing the agent's synthesized multi-source answer.

---

## Beat 10: "Spawn history shows the full story"

The spawn history timeline in the Spawn panel shows the complete lifecycle:
spawn events, completion events, timestamps, budget consumed per worker.

**Screenshot:** `10-spawn-history.png`
**What to see:** History timeline with spawn and termination entries.

---

## Running the Demo

```bash
# Start the Kestrel server
uv run kestrel start DemoAgent

# Run the narrated demo
cd demos/spawn
npx playwright test --config=demo_config.cjs

# Screenshots land in demos/spawn/demo-output/
# Narration transcript generated as demos/spawn/demo-output/narration.md
```
