#Requires -Version 5.1
<#
.SYNOPSIS
    Kestrel Sovereign - Track A Technical Demo Script
    Issue #191 / #133

.DESCRIPTION
    Automated presenter script for the 5-Act technical demo.
    Press Enter between acts to advance. Each act runs its commands,
    prints narration cues, and pauses for the presenter to speak.

    Prerequisites:
      - Kestrel server running (uv run kestrel start)
      - Ollama running (ollama serve) for Act 4 EPHEMERAL mode
      - .env file with KESTREL_API_KEY in the Kestrel project root

.PARAMETER KestrelDir
    Path to the Kestrel project root (default: current directory)

.PARAMETER BaseUrl
    Kestrel server URL (default: http://localhost:8888)

.PARAMETER AgentName
    Agent name for multi-agent routing (default: Kestrel)

.PARAMETER SkipSetup
    Skip the pre-demo setup checks (use if server is already verified)

.PARAMETER AutoAdvance
    Don't pause between acts (for CI/recording)

.EXAMPLE
    .\track_a_demo.ps1
    .\track_a_demo.ps1 -KestrelDir "C:\Users\gabri\Kestrel"
    .\track_a_demo.ps1 -SkipSetup -AutoAdvance
#>

param(
    [string]$KestrelDir = ".",
    [string]$BaseUrl = "http://localhost:8888",
    [string]$AgentName = "Kestrel",
    [switch]$SkipSetup,
    [switch]$AutoAdvance
)

$ErrorActionPreference = "Continue"
$agentBase = "$BaseUrl/api/agents/$AgentName"

# ============================================================================
# Helpers
# ============================================================================

function Write-Narration {
    param([string]$Text)
    Write-Host ""
    Write-Host "  NARRATION:" -ForegroundColor DarkCyan
    foreach ($line in ($Text -split "`n")) {
        Write-Host "  $line" -ForegroundColor Cyan
    }
    Write-Host ""
}

function Write-Act {
    param([string]$Number, [string]$Title, [string]$Time)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Yellow
    Write-Host "  ACT ${Number}: ${Title}" -ForegroundColor Yellow
    Write-Host "  Target: ${Time}" -ForegroundColor DarkYellow
    Write-Host ("=" * 70) -ForegroundColor Yellow
    Write-Host ""
}

function Write-Step {
    param([string]$Text)
    Write-Host "  >> $Text" -ForegroundColor Green
}

function Write-Expected {
    param([string]$Text)
    Write-Host "  [Expected] $Text" -ForegroundColor DarkGray
}

function Write-Fail {
    param([string]$Text)
    Write-Host "  [RECOVERY] $Text" -ForegroundColor Red
}

function Pause-ForPresenter {
    param([string]$Prompt = "Press ENTER to continue...")
    if (-not $AutoAdvance) {
        Write-Host ""
        Write-Host "  $Prompt" -ForegroundColor Magenta
        Read-Host | Out-Null
    }
}

function Invoke-Api {
    param(
        [string]$Uri,
        [string]$Method = "GET",
        [string]$Body = $null
    )
    $params = @{
        Uri     = $Uri
        Method  = $Method
        Headers = $script:headers
    }
    if ($Body) {
        $params.Body = $Body
    }
    try {
        Invoke-RestMethod @params
    }
    catch {
        Write-Fail "API call failed: $($_.Exception.Message)"
        Write-Fail "URI: $Uri"
        $null
    }
}

# ============================================================================
# Pre-Demo Setup
# ============================================================================

if (-not $SkipSetup) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor White
    Write-Host "  KESTREL SOVEREIGN - TRACK A TECHNICAL DEMO" -ForegroundColor White
    Write-Host "  Pre-Demo Setup (run before audience arrives)" -ForegroundColor DarkGray
    Write-Host ("=" * 70) -ForegroundColor White
    Write-Host ""

    # Check Ollama
    Write-Step "Checking Ollama..."
    try {
        $ol = Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 3
        Write-Step "Ollama UP: $($ol.version)"
    }
    catch {
        Write-Fail "Ollama DOWN - Act 4 (EPHEMERAL) will fail!"
        Write-Fail "Start it: ollama serve"
    }

    # Check Kestrel health
    Write-Step "Checking Kestrel server..."
    try {
        $health = Invoke-RestMethod "$BaseUrl/health" -TimeoutSec 5
        Write-Step "Server: $($health.status), agent_initialized: $($health.agent_initialized)"
    }
    catch {
        Write-Fail "Kestrel server not responding at $BaseUrl"
        Write-Fail "Start it: cd $KestrelDir && uv run kestrel start"
        Write-Host ""
        Read-Host "Fix the server and press ENTER to retry, or Ctrl+C to abort"
        $health = Invoke-RestMethod "$BaseUrl/health" -TimeoutSec 5
    }

    # Load API key
    Write-Step "Loading API key..."
    $envFile = Join-Path $KestrelDir ".env"
    if (Test-Path $envFile) {
        $keyLine = Get-Content $envFile | Select-String "KESTREL_API_KEY="
        if ($keyLine) {
            $key = $keyLine.Line.Split("=", 2)[1]
            Write-Step "Key loaded: $($key.Substring(0, [Math]::Min(8, $key.Length)))..."
        }
        else {
            Write-Fail "No KESTREL_API_KEY in .env"
        }
    }
    else {
        Write-Fail "No .env file found at $envFile"
        $key = Read-Host "Enter API key manually"
    }

    $script:headers = @{ "X-API-Key" = $key; "Content-Type" = "application/json" }

    # Verify agent responds
    Write-Step "Verifying agent..."
    $info = Invoke-Api -Uri "$agentBase/agent/info"
    if ($info) {
        Write-Step "Agent DID: $($info.agent_id)"
        Write-Step "Privacy mode: $($info.privacy_mode)"
    }

    # Reset privacy to NORMAL
    Write-Step "Resetting privacy mode to NORMAL..."
    $null = Invoke-Api -Uri "$agentBase/agent/privacy-mode" -Method POST -Body '{"mode":"NORMAL"}'

    # Skip bootstrap if needed
    Write-Step "Checking bootstrap state..."
    $bs = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST -Body '{"input": "!bootstrap-status"}'
    if ($bs -and $bs.response -like "*discovery*") {
        Write-Step "Skipping discovery mode..."
        $null = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST -Body '{"input": "!skip-discovery"}'
    }

    Write-Host ""
    Write-Step "Setup complete. Ready for demo."
    Pause-ForPresenter "Position terminal + browser side-by-side, then press ENTER to start"
}
else {
    # Minimal setup - just load key
    $envFile = Join-Path $KestrelDir ".env"
    $keyLine = Get-Content $envFile | Select-String "KESTREL_API_KEY="
    $key = $keyLine.Line.Split("=", 2)[1]
    $script:headers = @{ "X-API-Key" = $key; "Content-Type" = "application/json" }
}

$demoStart = Get-Date

# ============================================================================
# ACT 1: Born in the Terminal (0:45 - 2:45)
# ============================================================================

Write-Act "1" "BORN IN THE TERMINAL" "0:45 - 2:45"

Write-Narration "Every Kestrel agent begins with a single command."
Pause-ForPresenter "Press ENTER to run inception_service..."

Write-Step "Creating a new agent identity..."
$tempDir = if ($IsWindows -or $env:OS -eq "Windows_NT") { "C:\Temp\demo-agent" } else { "/tmp/demo-agent" }

# Clean previous demo agent
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }

& uv run python -m kestrel_sovereign.inception_service --name "TrackA-Demo" --output-dir $tempDir

Write-Narration @"
What just happened: a secp256k1 key pair was generated. An Ethereum-format
address was derived from the public key. A W3C Decentralized Identifier was
constructed from that address. The Kestrel Constitution was hashed and linked
as the first node in this agent's knowledge graph.

No cloud service issued this identity. No certificate authority approved it.
It's mathematical proof that lives on this machine.
"@

Pause-ForPresenter "Point to browser DID, then press ENTER to show identity-chain..."

Write-Step "Querying the running agent's identity chain..."
$chain = Invoke-Api -Uri "$agentBase/api/identity-chain"
if ($chain) {
    $chain | ConvertTo-Json -Depth 3
}

Write-Narration @"
The constitution.hash is a SHA-256 of the Kestrel Constitution. It was written
into the knowledge graph on the day the agent was created. Change one byte of
the constitution and the hash breaks.
"@

$elapsed = [Math]::Round(((Get-Date) - $demoStart).TotalSeconds)
Write-Step "Elapsed: ${elapsed}s (target: 2:45)"
Pause-ForPresenter "Press ENTER for Act 2..."

# ============================================================================
# ACT 2: Constitution Governs Every Response (2:45 - 5:30)
# ============================================================================

Write-Act "2" "CONSTITUTION GOVERNS EVERY RESPONSE" "2:45 - 5:30"

Write-Step "Checking agent status..."
$status = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST -Body '{"input": "!status"}'
if ($status) { $status.response }

Write-Narration "DID and privacy mode -- right in the status line. Now let me ask it directly about its governance."
Pause-ForPresenter "Press ENTER to ask about constitutional principles..."

Write-Step "Asking agent about its principles..."
$r = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST `
    -Body '{"input": "What principles govern your behavior and who controls your rules?"}'
if ($r) {
    Write-Host ""
    Write-Host $r.response -ForegroundColor White
}

Write-Narration @"
This went through a full context-building loop -- RAG retrieval from the
knowledge graph, constitutional grounding, token budget management, then
the LLM call.
"@

Pause-ForPresenter "Press ENTER to show audit trace..."

Write-Step "Showing observability events (audit trace)..."
$obs = Invoke-Api -Uri "$agentBase/api/observability/events?limit=5"
if ($obs -and $obs.events) {
    $obs.events | ForEach-Object {
        "[$($_.event_type.PadRight(15))]  $($_.tool_name)"
    }
}

Write-Narration @"
This is the observability store -- every LLM call is logged with timing.
In a regulated deployment, this is your compliance record.
"@

Write-Step "Pulling constitution hash..."
$chain2 = Invoke-Api -Uri "$agentBase/api/identity-chain"
if ($chain2) {
    $chain2 | Select-Object -ExpandProperty constitution | ConvertTo-Json
}

Write-Narration @"
That hash. SHA-256. The hash is the proof that the constitution hasn't changed
since inception. You can re-hash the document yourself and compare -- it'll match.
"@

$elapsed = [Math]::Round(((Get-Date) - $demoStart).TotalSeconds)
Write-Step "Elapsed: ${elapsed}s (target: 5:30)"
Pause-ForPresenter "Press ENTER for Act 3..."

# ============================================================================
# ACT 3: Memory That Survives Sessions (5:30 - 7:45)
# ============================================================================

Write-Act "3" "MEMORY THAT SURVIVES SESSIONS" "5:30 - 7:45"

Write-Narration @"
Most AI chat tools give you conversation history within a tab.
Close the tab, start over. Kestrel uses a knowledge graph --
persistent, cross-session, and yours.
"@

Write-Step "Querying knowledge graph nodes..."
$mem = Invoke-Api -Uri "$agentBase/api/memories"
if ($mem) {
    Write-Host "  Total memory nodes: $($mem.total)"
    $mem.nodes | Select-Object node_type, label | Format-Table
}

Write-Narration @"
Every node here was written by a real event -- inception, constitution
anchoring, exports. The graph is the agent's persistent memory.
Let me show you what survives a session reset.
"@

Pause-ForPresenter "Press ENTER to create new session and test cross-session memory..."

Write-Step "Starting a new session (fresh conversation history)..."
$ns = Invoke-Api -Uri "$agentBase/api/conversations/new" -Method POST -Body '{}'
if ($ns) {
    Write-Host "  New session ID: $($ns.session_id)"
}

Write-Step "Asking about identity in the new session (zero history)..."
$sessionId = if ($ns) { $ns.session_id } else { "1" }
$body = @{ input = "What's your DID and when were you created?"; session_id = "$sessionId" } | ConvertTo-Json
$r = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST -Body $body
if ($r) {
    Write-Host ""
    Write-Host $r.response -ForegroundColor White
}

Write-Narration @"
Zero conversation history in this session. The answer comes entirely from
the knowledge graph. Every piece of information in that graph was written
by a cryptographically-authenticated event at inception.
"@

$elapsed = [Math]::Round(((Get-Date) - $demoStart).TotalSeconds)
Write-Step "Elapsed: ${elapsed}s (target: 7:45)"
Pause-ForPresenter "Press ENTER for Act 4..."

# ============================================================================
# ACT 4: Privacy Is Architecture, Not Policy (7:45 - 9:30)
# ============================================================================

Write-Act "4" "PRIVACY IS ARCHITECTURE, NOT POLICY" "7:45 - 9:30"

Write-Narration @"
Now watch what happens when I flip the privacy mode to EPHEMERAL.
This is where Kestrel is genuinely different from anything else in the market.
"@

Write-Step "Recording baseline conversation count..."
$before = Invoke-Api -Uri "$agentBase/api/conversations"
$beforeTotal = if ($before) { $before.total } else { 0 }
Write-Host "  Conversations before: $beforeTotal"

Write-Step "Switching to EPHEMERAL mode..."
$privResult = Invoke-Api -Uri "$agentBase/agent/privacy-mode" -Method POST -Body '{"mode":"EPHEMERAL"}'
if ($privResult) {
    Write-Host "  $($privResult.message)" -ForegroundColor Yellow
}

Pause-ForPresenter "Press ENTER to send message in EPHEMERAL mode..."

Write-Step "Sending message in EPHEMERAL mode (nothing should be stored)..."
$r = Invoke-Api -Uri "$agentBase/agent/invoke" -Method POST `
    -Body '{"input": "This is a sensitive matter I do not want stored anywhere."}'
if ($r) {
    Write-Host ""
    Write-Host $r.response -ForegroundColor White
}

Write-Step "Restoring NORMAL mode..."
$null = Invoke-Api -Uri "$agentBase/agent/privacy-mode" -Method POST -Body '{"mode":"NORMAL"}'

Write-Step "Checking if any records were written..."
$after = Invoke-Api -Uri "$agentBase/api/conversations"
$afterTotal = if ($after) { $after.total } else { 0 }
$newRecords = $afterTotal - $beforeTotal
Write-Host "  New records written during EPHEMERAL session: $newRecords" -ForegroundColor $(if ($newRecords -eq 0) { "Green" } else { "Red" })

Write-Narration @"
Kestrel has five privacy levels:
  EPHEMERAL  -- nothing stored, not even temporarily
  ISOLATED   -- in-memory only; you can explicitly save or discard
  ANONYMOUS  -- stored encrypted, distributed, no identity linkage
  NORMAL     -- full local persistence with sovereignty guarantees
  PUBLIC     -- cloud LLMs allowed

In a regulated industry, you can hardcode a privacy floor that operators
cannot override. The compliance guarantee is in the architecture.
"@

$elapsed = [Math]::Round(((Get-Date) - $demoStart).TotalSeconds)
Write-Step "Elapsed: ${elapsed}s (target: 9:30)"
Pause-ForPresenter "Press ENTER for Act 5..."

# ============================================================================
# ACT 5: You Own It. You Can Move It. (9:30 - 11:00)
# ============================================================================

Write-Act "5" "YOU OWN IT. YOU CAN MOVE IT." "9:30 - 11:00"

Write-Narration "Last thing. This is the one we don't see anywhere else in the market."
Pause-ForPresenter "Press ENTER to run sovereignty export..."

Write-Step "Exporting agent state..."
$export = Invoke-Api -Uri "$agentBase/api/sovereignty/export" -Method POST `
    -Body '{"tier": "local", "encrypt": false}'
if ($export) {
    Write-Host ""
    Write-Host $export.message -ForegroundColor White
}

Write-Step "Showing export receipt in knowledge graph..."
$mem3 = Invoke-Api -Uri "$agentBase/api/memories"
if ($mem3) {
    $mem3.nodes | Where-Object { $_.node_type -eq 'sovereignty_receipt' } |
        Select-Object node_id, label | Format-Table
}

Write-Narration @"
That export is a portable blob. Content-addressed by SHA-256 -- you can
verify it hasn't been tampered with. It contains the complete agent state.

You can take it to another machine, another Kestrel instance, another
cloud provider. The agent doesn't live in our cloud. It lives in that file.
The CID is the proof.
"@

# ============================================================================
# CLOSE
# ============================================================================

$totalElapsed = [Math]::Round(((Get-Date) - $demoStart).TotalMinutes, 1)

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host "  DEMO COMPLETE" -ForegroundColor Yellow
Write-Host "  Total time: $totalElapsed minutes" -ForegroundColor DarkYellow
Write-Host ("=" * 70) -ForegroundColor Yellow

Write-Narration @"
What you just saw:

  A cryptographic identity generated in two seconds.
  A constitution anchored at birth and tamper-evident.
  An audit trace on every response.
  A knowledge graph that accumulates across sessions, not a chat window.
  Privacy enforced by the storage layer, not by policy.
  A complete data export with a content hash you can independently verify.

Kestrel is MIT-licensed, open source, runs on any machine with a GPU or
cloud budget. In 30 minutes you can have your own agent running with all
of this active.
"@

Write-Host ""
Write-Host "  KEY PHRASES:" -ForegroundColor Magenta
Write-Host '  - "That refusal is constitutional, not corporate."' -ForegroundColor White
Write-Host '  - "The compliance guarantee is in the architecture."' -ForegroundColor White
Write-Host '  - "The agent does not live in our cloud. It lives in that file."' -ForegroundColor White
Write-Host '  - "In 30 minutes you can have your own agent running."' -ForegroundColor White
Write-Host ""
