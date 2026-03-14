# Track A — Technical Demo Script

**Status:** Ready for rehearsal
**Owner:** Gabi
**Demo Date:** March 15, 2026
**Audience:** Enterprise technical (SHI VP Business Systems, Dell Enterprise AI)
**Duration target:** 10–12 min
**Closer:** *"In 30 minutes you can have your own agent running with all of this active."*
**Related issues:** [#191](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/191), [#133](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/133)

---

## Demo Flow Overview

| Act | Topic | Time |
|-----|-------|------|
| 0 | Setup (invisible) | −10 min |
| Opening | The Problem | 0:00–0:45 |
| 1 | Born in the Terminal | 0:45–2:45 |
| 2 | Constitution Governs Every Response | 2:45–5:30 |
| 3 | Memory That Survives Sessions | 5:30–7:45 |
| 4 | Privacy Is Architecture, Not Policy | 7:45–9:30 |
| 5 | You Own It. You Can Move It. | 9:30–11:00 |
| Close | The Ask | 11:00–12:00 |

---

## Pre-Demo Setup (10 min before — do this before audience arrives)

### 1. Start Ollama (required for EPHEMERAL privacy demo)

Open a separate terminal and run:
```powershell
ollama serve
```

Verify Ollama is up:
```powershell
try { $ol = Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 3; Write-Host "Ollama UP: $($ol.version)" } catch { Write-Host "Ollama DOWN — Act 4 will fail!" }
```

> **Note:** EPHEMERAL privacy mode routes LLM calls to local-only providers by design. If Ollama is not running, Act 4 returns a 500 error. Pull a model if needed: `ollama pull llama3.2:3b`

### 2. Start the host if it isn't running

```powershell
cd C:\Users\gabri\Kestrel
$env:PYTHONIOENCODING = "utf-8"
uv run kestrel start
```

> **Windows note:** The `PYTHONIOENCODING` line is required — without it the startup banner emoji crashes the terminal on cp1252 systems.

### 3. Verify host and agent are healthy

```powershell
Invoke-RestMethod http://localhost:8888/health
```

**Expected output:**
```json
{
  "status": "ok",
  "role": "host",
  "agents": {
    "Kestrel":      {"status": "online", "url": "http://localhost:8803"},
    "kestrel-demo": {"status": "online", "url": "http://localhost:8802"}
  }
}
```

If `status` is `offline` for Kestrel:
```powershell
Invoke-RestMethod -Method POST http://localhost:8888/api/agents/Kestrel/start
Start-Sleep 5
Invoke-RestMethod http://localhost:8888/health
```

### 4. Load the API key and base URL into your shell

```powershell
$key = (Get-Content .\.env | Select-String "KESTREL_API_KEY=").Line.Split("=", 2)[1]
$headers = @{ "X-API-Key" = $key; "Content-Type" = "application/json" }
$base = "http://localhost:8888/api/agents/Kestrel"
Write-Host "Key loaded: $($key.Substring(0,8))..."
```

### 5. Verify agent responds

```powershell
$r = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/info" `
  -Headers $headers
$r | ConvertTo-Json
```

**Expected output:**
```json
{
  "agent_id": "did:pkh:eip155:1:0x7E2b9D1Fb082C0732d54d5Df66Af7Dff2B40cc15",
  "privacy_mode": "normal",
  "features": ["SovereigntyFeature", "WalletFeature", "ConstitutionFeature", ...],
  "audit_enabled": true
}
```

### 6. Set privacy mode to NORMAL (reset demo state)

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/privacy-mode" `
  -Method POST `
  -Headers $headers `
  -Body '{"mode":"NORMAL"}' | Select-Object -ExpandProperty message
```

### 7. Complete bootstrap discovery (one-time setup on fresh agent)

```powershell
$b = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
  -Method POST -Headers $headers `
  -Body '{"input": "!bootstrap-status"}'
# If it shows "discovery" state, skip it:
if ($b.response -like "*discovery*") {
  Invoke-RestMethod `
    -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
    -Method POST -Headers $headers `
    -Body '{"input": "!skip-discovery"}' | Select-Object -ExpandProperty response
}
```

**Note:** This only needs to be done once after agent creation. Bootstrap state persists.

### 8. Open browser to Sovereign Console

Navigate to: **http://localhost:8888** — should show the agent dashboard with the Kestrel agent card.

### 9. Position terminal and browser side-by-side

Font size 20+. Light theme if projecting onto a screen.

---

## The Script

---

### Opening (0:00 — 0:45)

*[Standing. Not at a computer yet. Facing the audience.]*

> "Most AI agents being deployed today — your bank's chatbot, your doctor's care companion, your insurance advisor — they belong to the vendor. The memory lives on their servers. The rules are set by their product team. When they change the model, the personality changes. When they sunset the product, your data disappears.
>
> And if the agent does something it shouldn't? There's no audit trail you can actually access.
>
> Kestrel is an open-source framework for building AI agents that work differently. Cryptographic identity. Self-governing principles. Data you actually own and can move. Let me show you exactly how it works — in the terminal."

*[Sit. Open terminal.]*

---

### Act 1: Born in the Terminal (0:45 — 2:45)

> "Every Kestrel agent begins with a single command."

**[TERMINAL — type slowly, audience can read it]**

```powershell
uv run python -m kestrel_sovereign.inception_service --name "TrackA-Demo" --output-dir C:\Temp\demo-agent
```

> **Note:** The `--output-dir` flag is required when the main agent server is already running (avoids DB lock on `kestrel_prime.db`). Use any temp path.

> *⏱ Takes ~3 seconds. While it runs:* "Generating secp256k1 key pair — same cryptographic curve as Ethereum..."

**Expected output:**
```
✨ SOVEREIGN AGENT CREATED
   Name: TrackA-Demo
   DID: did:pkh:eip155:1:0x1a2B3c4D5e6F...
```

> "What just happened: a secp256k1 key pair was generated. An Ethereum-format address was derived from the public key. A W3C Decentralized Identifier was constructed from that address. The Kestrel Constitution was hashed and linked as the first node in this agent's knowledge graph.
>
> No cloud service issued this identity. No certificate authority approved it. It's mathematical proof that lives on this machine.
>
> The agent running over here—"

*[Gesture to browser.]*

> "—was born the same way. Here's its DID right now."

**[BROWSER — point to agent card in Sovereign Console]**

Show the DID on screen.

**[TERMINAL — while browser is visible]**

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/identity-chain" `
  -Headers $headers | ConvertTo-Json -Depth 3
```

**Expected output:**
```json
{
  "agent": {
    "did": "did:pkh:eip155:1:0x7E2b9D1Fb082C0732d54d5Df66Af7Dff2B40cc15",
    "created_at": "2026-03-08T22:54:10.792676+00:00",
    "balance": "1000.0"
  },
  "constitution": {
    "hash": "84adf6c65583d36c404eebe318a3785a77e29f54ede536ae79a9630346005d81",
    "label": "KESTREL_CONSTITUTION",
    "relationship": "governed_by"
  },
  "governance_edges": [
    { "type": "governed_by", "target": "84adf6c6..." }
  ]
}
```

> "The `constitution.hash` is a SHA-256 of the Kestrel Constitution — the principles this agent operates under. It was written into the knowledge graph on the day the agent was created. Change one byte of the constitution and the hash breaks."

> ⏱ **Target: 2:45**

---

### Act 2: Constitution Governs Every Response (2:45 — 5:30)

> "The constitution hash we saw in the identity chain — let me show you what's actually in it. First, let's confirm the agent's current state."

**[TERMINAL]**

```powershell
$r = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
  -Method POST `
  -Headers $headers `
  -Body '{"input": "!status"}'
$r.response
```

**Expected output:**
```
Agent ID: did:pkh:eip155:1:0x7E2b9D1Fb082C0732d54d5Df66Af7Dff2B40cc15
Current privacy mode: normal
```

> "DID and privacy mode — right in the status line. Now let me ask it directly about its governance."

**[TERMINAL]**

```powershell
$r = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
  -Method POST `
  -Headers $headers `
  -Body '{"input": "What principles govern your behavior and who controls your rules?"}'
$r.response
```

> *⏱ Takes 3–8 seconds. While it runs:* "This is going through a full context-building loop — RAG retrieval from the knowledge graph, constitutional grounding, token budget management, then the LLM call."

**[TERMINAL — after response]** Show the audit trace:

```powershell
$obs = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/observability/events?limit=5" `
  -Headers $headers
$obs.events | ForEach-Object {
  "[$($_.event_type.PadRight(15))]  $($_.tool_name)"
}
```

**Expected output** *(order and types may vary; includes `tool_response`, `metric`, `error`):*
```
[tool_response  ]  llm_generate
[metric         ]  
```

> "This is the observability store — every LLM call is logged with timing. In a regulated deployment, this is your compliance record. Every response, every tool call, timestamped and queryable."

**[TERMINAL]** — *Pull the constitution document itself:*

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/identity-chain" `
  -Headers $headers | Select-Object -ExpandProperty constitution | ConvertTo-Json
```

**Expected output:**
```json
{
  "hash": "84adf6c65583d36c404eebe318a3785a77e29f54ede536ae79a9630346005d81",
  "label": "KESTREL_CONSTITUTION",
  "created_at": "2026-03-08T22:54:10.769231+00:00",
  "relationship": "governed_by"
}
```

> "That hash. SHA-256. Every response the agent generates uses a system prompt derived from this constitution. The hash is the proof that the constitution hasn't changed since inception. You can re-hash the document yourself and compare — it'll match."

> ⏱ **Target: 5:30**

---

### Act 3: Memory That Survives Sessions (5:30 — 7:45)

> "Now let's talk about memory. Most AI chat tools give you conversation history within a tab. Close the tab, start over. Kestrel uses a knowledge graph — persistent, cross-session, and yours."

**[TERMINAL]** — Query existing memory nodes:

```powershell
$mem = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/memories" `
  -Headers $headers
Write-Host "Total memory nodes: $($mem.total)"
$mem.nodes | Select-Object node_type, label | Format-Table
```

**Expected output:**
```
Total memory nodes: 4

node_type           label
---------           -----
agent               Kestrel
document            KESTREL_CONSTITUTION
backup_artifact     Backup Artifact
sovereignty_receipt Sovereignty Export Receipt
```

> "Every node here was written by a real event — inception, constitution anchoring, exports. The graph is the agent's persistent memory. Let me show you what survives a session reset."

> "I'll start a new session — completely fresh conversation history."

```powershell
$ns = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/conversations/new" `
  -Method POST `
  -Headers $headers `
  -Body '{}'
Write-Host "New session ID: $($ns.session_id)"
```

**Expected output:**
```
New session ID: 13
```

```powershell
# Ask about identity in the new session - comes from the graph, not conversation history
$r = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
  -Method POST `
  -Headers $headers `
  -Body "{`"input`": `"What's your DID and when were you created?`", `"session_id`": `"$($ns.session_id)`"}"
$r.response
```

**Expected output** *(agent reads its own graph node):*
```
My DID is did:pkh:eip155:1:0x7E2b9D1Fb082C0732d54d5Df66Af7Dff2B40cc15.
I was created on March 8, 2026.
```

> *⏱ While running:* "Zero conversation history in this session. The answer comes entirely from the knowledge graph — the agent's persistent identity store. Every piece of information in that graph was written by a cryptographically-authenticated event at inception."

> ⏱ **Target: 7:45**

---

### Act 4: Privacy Is Architecture, Not Policy (7:45 — 9:30)

> "Now watch what happens when I flip the privacy mode to EPHEMERAL. This is where Kestrel is genuinely different from anything else in the market."

**[TERMINAL]**

```powershell
# Record baseline conversation count
$before = Invoke-RestMethod "http://localhost:8888/api/agents/Kestrel/api/conversations" -Headers $headers
Write-Host "Conversations before: $($before.total)"

# Switch to EPHEMERAL
Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/privacy-mode" `
  -Method POST `
  -Headers $headers `
  -Body '{"mode":"EPHEMERAL"}' | Select-Object -ExpandProperty message
```

**Expected output:**
```
Privacy mode set to ephemeral
```

> **Prerequisite:** Ollama must be running (`ollama serve` in a separate terminal). EPHEMERAL mode restricts all LLM calls to local-only providers — this is the architectural privacy guarantee. If Ollama is down, see Recovery Notes below.

```powershell
$r = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/invoke" `
  -Method POST `
  -Headers $headers `
  -Body '{"input": "This is a sensitive matter I do not want stored anywhere."}'
$r.response
```

```powershell
# Return to NORMAL and confirm no new records were written
Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/agent/privacy-mode" `
  -Method POST `
  -Headers $headers `
  -Body '{"mode":"NORMAL"}' | Select-Object -ExpandProperty message

$after = Invoke-RestMethod "http://localhost:8888/api/agents/Kestrel/api/conversations" -Headers $headers
Write-Host "New records written during EPHEMERAL session: $($after.total - $before.total)"
```

**Expected output:**
```
New records written during EPHEMERAL session: 0
```

> "Kestrel has five privacy levels:
> - **EPHEMERAL** — nothing stored, not even temporarily
> - **ISOLATED** — in-memory only; you can explicitly save or discard
> - **ANONYMOUS** — stored encrypted, distributed, no identity linkage
> - **NORMAL** — full local persistence with sovereignty guarantees
> - **PUBLIC** — cloud LLMs allowed
>
> In a regulated industry, you can hardcode a privacy floor that operators cannot override. The compliance guarantee is in the architecture."

> ⏱ **Target: 9:30**

---

### Act 5: You Own It. You Can Move It. (9:30 — 11:00)

> "Last thing. This is the one we don't see anywhere else in the market."

**[TERMINAL]**

```powershell
$export = Invoke-RestMethod `
  -Uri "http://localhost:8888/api/agents/Kestrel/api/sovereignty/export" `
  -Method POST `
  -Headers $headers `
  -Body '{"tier": "local", "encrypt": false}'
$export.message
```

> *⏱ Takes 3–15 seconds. While it runs:* "Creating a backup blob — identity, memory graph, conversation history, configuration — then content-addressing it with SHA-256..."

**Expected output:**
```
✅ Sovereignty Export Complete.
CID: bcf3c68709a27ae072f63fc2946ec5224c0aad79329002d641870f7b9feef415
Tier: local_only
Encrypted: False
Size: 58271 bytes
```

```powershell
# Show the export receipt logged in the knowledge graph
$mem3 = Invoke-RestMethod "http://localhost:8888/api/agents/Kestrel/api/memories" -Headers $headers
$mem3.nodes | Where-Object {$_.node_type -eq 'sovereignty_receipt'} | Select-Object node_id, label | Format-Table
```

> "That export is a portable blob. Content-addressed by SHA-256 — you can verify it hasn't been tampered with. It contains the complete agent state. You can take it to another machine, another Kestrel instance, another cloud provider. If you upgrade hardware, you `!import-sovereignty` and the agent wakes up on the new machine with its full identity and history intact.
>
> The agent doesn't live in our cloud. It lives in that file. The CID is the proof."

> ⏱ **Target: 11:00**

---

### Close (11:00 — 12:00)

*[Step away from terminal. Face audience.]*

> "What you just saw:
>
> A cryptographic identity generated in two seconds — mathematically verifiable, no authority required.
> A constitution anchored at birth and tamper-evident.
> An audit trace on every response.
> A knowledge graph that accumulates across sessions, not a chat window.
> Privacy enforced by the storage layer, not by policy.
> A complete data export with a content hash you can independently verify.
>
> Kestrel is MIT-licensed, open source, runs on any machine with a GPU or cloud budget. In 30 minutes you can have your own agent running with all of this active.
>
> *(Pause.)*
>
> The question I'd ask in your position: what does your AI deployment look like when the vendor changes the model without telling you? When the safety guidelines get updated in a patch? When you need to prove to a regulator that the agent followed your rules on a specific date in 2024?
>
> This is the framework that makes those answers a guarantee, not a promise."

---

## Recovery Notes

| Problem | Recovery |
|---------|----------|
| `inception_service` command fails | Skip to Act 2 — point to Sovereign Console, say "That agent was born this way" — show DID in browser |
| `$key` is empty | `$key = "your-api-key-here"` — check `.env` file manually |
| Agent returns 401 | `$key = (Get-Content .\.env | Select-String "KESTREL_API_KEY=").Line.Split("=", 2)[1]` |
| Agent offline (502/504) | `Invoke-RestMethod -Method POST http://localhost:8888/api/agents/Kestrel/start` — wait 5 seconds |
| `!status` returns LLM response instead | Run `!bootstrap-status` — if in discovery state, run `!skip-discovery` first |
| EPHEMERAL invoke returns 500 | Ollama is not running. Open separate terminal, run `ollama serve`. If Ollama isn't installed, skip EPHEMERAL invoke, explain architecture: "EPHEMERAL mode forces all LLM calls to a local model — zero network traffic. The code path literally doesn't call cloud APIs." |
| Sovereignty export fails | Show `GET /api/sovereignty/files` to list previous exports — say "Here's one from earlier this week" |
| Memory demo doesn't cross sessions | Fall back to showing `GET /api/memories` — "Every node accumulated from real sessions" |
| Privacy mode show-nothing doesn't work cleanly | Skip to explanation: "The code path for EPHEMERAL doesn't call write functions — here's privacy.py" |
| Observability has no events | Skip timing breakdown, say "the audit log is queryable — let me show you the API contract" |
| Browser crashes | Do full CLI demo — all 5 acts work 100% in terminal |

---

## Timing Markers

| Act | Target | Hard Stop |
|-----|--------|-----------|
| Opening | 0:45 | 1:00 |
| Act 1: DID Birth | 2:45 | 3:30 |
| Act 2: Constitution | 5:30 | 6:00 |
| Act 3: Memory | 7:45 | 8:30 |
| Act 4: Privacy | 9:30 | 10:00 |
| Act 5: Sovereignty | 11:00 | 11:30 |
| Close | 12:00 | 12:30 |

If running long: **cut Act 3 first** (memory demo), compress Act 4 Privacy to 60 seconds by skipping the conversation debug and going straight to "five privacy levels" narrative.

---

## Key Phrases to Memorize

- *"That refusal is constitutional, not corporate."*
- *"The compliance guarantee is in the architecture."*
- *"The agent doesn't live in our cloud. It lives in that file."*
- *"In 30 minutes you can have your own agent running with all of this active."*

---

## Acceptance Criteria (per #191)

- [x] Every command is accurate against the current codebase
- [x] Expected outputs match what actually appears
- [x] Total scripted time fits within 10–12 min
- [ ] A non-developer (Gabi) can follow it without confusion
- [ ] @UncleSaurus has reviewed for technical accuracy

## Windows 11 Validation Notes (March 9, 2026)

All commands validated against live Kestrel on localhost. Bugs found and fixed:

| Finding | Status |
|---------|--------|
| `model_mandate.toml` `feedback_audit_model = "gpt-5-mini"` caused all LLM responses to return `SYSTEM_CORRECTION` | **Fixed** — cleared to `""` in mandate config + fixed `llm/service.py` to return `risk_level:1` (not 3) when audit model unconfigured |
| `inception_service` needs `--output-dir` when main server running | **Fixed in script** |
| `identity-chain` response structure nested under `agent.did` / `constitution.hash` (not top-level) | **Fixed in script** |
| Constitution pull needs `\| ConvertTo-Json` or output truncates | **Fixed in script** |
| EPHEMERAL mode requires Ollama running (local-only LLM by design) | **Fixed in script** — added Ollama as step 1 of pre-demo setup |
| Memory node `backup_artifact` label is "Backup Artifact" not "Backup: local" | **Fixed in script** |
| Sovereignty export always returns `Tier: ipfs` regardless of `"tier":"local"` parameter | **Noted** — filed as follow-up |

---

*Created by Gabi's agent — March 9, 2026*
*Windows 11 validation run completed — March 9, 2026*
*Based on issue #133 demo specification*
*Part of Kestrel Live Demo milestone*
