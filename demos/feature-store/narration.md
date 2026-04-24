# Feature Store Demo — Narration

Kestrel agents aren't monolithic — their capabilities come from feature packages. Each feature bundles a set of skills (procedural behaviors the agent can invoke). The Feature Store is where those packages are browsed, searched, and installed.

## Beats

### Act 1: The Feature Store panel
Open Features. A grid of feature cards, each showing name, description, install status, and skill count. "Think of it as App Store for agent capability."

### Act 2: Search and filter
Type "memory" into the search bar. The grid narrows to memory-related features. Switch filter from All → Installed. "Only what's active on this agent."

### Act 3: Drill into a feature
Click one feature. A detail view opens: full description, author, version, and the list of skills the feature provides. Each skill has a one-line summary.

### Act 4: Skills view
Highlight the skills list. "Skills are the unit the orchestrator calls. Each has a schema; the LLM picks the one that matches the user's intent."

### Act 5: Back to the grid
Return to the grid view. "Features ship independently, skills extend behavior, and the agent's core stays stable."
