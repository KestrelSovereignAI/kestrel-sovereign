# Plan: Sovereign GPU Integration (RunPod)

## 1. Overview
This document outlines the architecture for integrating ephemeral GPU compute (RunPod) into the Kestrel Sovereign Agent. The goal is to allow the agent to "upgrade its brain" on demand, moving from local/cloud inference to a high-performance private GPU pod for complex tasks or image generation, while maintaining full sovereignty and control.

## 2. Core Architecture

The system relies on three main components:

1.  **`RunPodManager`**: The infrastructure layer. Handles the "metal" (spinning up/down pods).
2.  **`BrainRouter`**: The cognitive layer. A dynamic proxy that routes LLM requests to the active backend (Cloud, Local, or Remote GPU).
3.  **`RunPodFeature` + Tools**: The interface layer. Allows the Agent (and User) to control the infrastructure through the existing feature + command system.

### 2.1. RunPodManager (`features/runpod/runpod_manager.py`)
A unified interface for GPU provisioning that supports two modes of operation:

*   **Direct Mode (Sovereign)**: Uses the user's own `RUNPOD_API_KEY`. Kestrel talks directly to RunPod. No middleman.
*   **Managed Mode (Kestrel)**: Uses the Kestrel Platform API. For users who don't want to manage their own cloud accounts.

**Key Responsibilities:**
*   Abstracts the provider (Direct vs. Kestrel).
*   Manages Pod Lifecycle: `start`, `stop`, `status`.
*   Enforces "Stateless" design: Pods are compute nodes, not storage nodes.
*   Profiles:
    *   `profile="llm"`: Optimized for Text (e.g., Llama 3 70B).
    *   `profile="image"`: Optimized for Generation (e.g., Flux/ComfyUI).

### 2.2. BrainRouter (`llm/router.py`)
A dynamic proxy sitting behind the standard `agent.llm` interface.

**Key Responsibilities:**
*   Maintains `current_backend` state: `CLOUD` | `LOCAL` | `REMOTE_GPU`.
*   **Hot Swapping**: Switches the active backend without requiring a restart or code changes in the rest of the agent.
*   **Auto-Fallback**: If the GPU pod dies (timeout/crash), automatically catches the connection error and reverts to `CLOUD` or `LOCAL` for the next request.
*   **Context Management**: Adapts the context window size based on the active backend (e.g., shrinking context when moving from 128k Cloud to 8k Local).

### 2.3. RunPodFeature & Tooling (`features/runpod/feature.py`)
Implements the feature surface area by subclassing `features.base.Feature` and exposing tools via the `@tool` decorator so the `ToolRegistry` can auto-discover them without touching core agent logic.

**Key Responsibilities:**
*   Provide a `RunPodFeature.initialize()` hook that wires the feature to a shared `RunPodManager` (and later the `BrainRouter`).
*   Expose a primary tool such as `manage_gpu(action, pod_type, task_profile, ttl_seconds, model_name, prompt=None)` that reports pod id, target URL, TTL, and estimated cost.
*   Register user-facing commands through `command_prefix` values matching the existing command system:
    *   `!gpu on [model] [profile] [ttl]` → `action="start"` (LLM pods)
    *   `!gpu off` → `action="stop"`
    *   `!gpu status` → `action="status"`
    *   `!dream <prompt>` → `action="one_shot_image"` (spins up `profile="image"`, runs prompt, tears down)
*   Emit structured responses in the standard `{"success": bool, "result": {...}}` envelope and raise explicit errors when policy checks fail (missing env vars, TTL too long, unsupported profile).
*   Surface telemetry (cost, TTL remaining, backend health) so higher-level automation and the user UI can react without extra API calls.

### 2.4. Command Registry Integration (Tool Routing)
*   `ToolRegistry` (`tools/registry.py`) matches commands by prefix, so every RunPod command must set `command_prefix` and keep positional parameters deterministic for `parse_command_args`.
*   Use `ToolCategory.SYSTEM` for now; if GPU orchestration needs its own grouping, extend `ToolCategory` in `tools/base.py` and update this plan before implementation.
*   Provide usage strings/examples (``!gpu on llama-3-70b h100-single 1800``) within each tool schema so `tool_registry.get_help_text()` automatically documents the commands alongside the existing ones in `kestrel_agent.py`.
*   Integration tests should exercise `tool_registry.route_command("!gpu status")` to guarantee routing works and that `validate_parameters` enforces enum/TTL bounds.

## 3. Implementation Plan

### Phase 1: The Infrastructure (RunPodManager)
*   **Task**: Create `features/runpod/runpod_manager.py`.
*   **Details**:
    *   Define `GPUProvider` abstract base class.
    *   Implement `DirectRunPodProvider` using the `runpod` python library.
    *   Implement `KestrelRunPodProvider` using `requests` to the Kestrel API.
    *   Implement `RunPodManager` as a factory/wrapper that enforces stateless pods and tracks the active session record (pod id, backend URL, TTL, owning feature).
    *   **Safety**: Implement client-side tracking of session intent ("GPU reserved until X") plus guardrails for TTL/cost ceilings so the feature can reject out-of-policy requests before provisioning.
*   **Deliverable**: Unit/integration tests verifying provider start/stop/status calls (real in dev env, controlled double in CI) and session tracking persistence.

### Phase 2: The "Brain" (BrainRouter)
*   **Task**: Refactor `llm/service.py` or create `llm/router.py`.
*   **Details**:
    *   Implement the `BrainRouter` class.
    *   Update `KestrelAgent` to initialize `self.llm` as a `BrainRouter` instance.
    *   Implement the "Hot Swap" method `switch_backend(type, url, model)`.
    *   Implement the "Auto-Fallback" logic in `generate()`.
*   **Deliverable**: Integration test where the agent switches backends mid-conversation.

### Phase 3: The Feature + Commands (RunPodFeature)
*   **Task**: Create `features/runpod/feature.py` that subclasses `Feature`, exposes `@tool` methods, and registers the command prefixes.
*   **Details**:
    *   Provide a `manage_gpu` tool whose schema lines up with `ToolRegistry.parse_command_args` ordering so `!gpu on ...` works without custom parsing.
    *   Implement a `dream_image` helper/tool (exposed as `!dream`) that wraps `manage_gpu` with `task_profile="image"` and auto-tears down the pod once the one-shot action completes.
    *   Update `kestrel_agent.KestrelAgent.__init__` to `_register_feature(RunPodFeature(self))` so the commands auto-load like existing features.
    *   Add polling logic so `!gpu on` only returns success after `status="READY"` and the `BrainRouter` has the endpoint info it needs to call `switch_backend`.
    *   Ensure responses carry TTL remaining, pod id, backend URI, and cost estimate for UI/LLM consumption, plus explicit failure info when providers error out.
*   **Deliverable**: The agent can successfully execute `!gpu on` / `!gpu status` / `!gpu off` / `!dream` via the command system with schema validation and registry routing covered by tests.

### Phase 4: UX & Safety
*   **Task**: Polish the interaction.
*   **Details**:
    *   **Dead Man's Switch (Server-Side)**: Ensure RunPod/Kestrel kills pods after TTL or idle timeout.
    *   **Session Tracking (Client-Side)**: Kestrel tracks "GPU reserved until X". If pod dies early, handle gracefully.
    *   **User Feedback**: "Spinning up GPU... (ETA 2m)" messages.
    *   **Cost Awareness**: Agent should know roughly how much the requested pod costs ($/hr).

## 4. Technical Specifications

### 4.1. Data Models

```python
class PodStatus(Enum):
    OFFLINE = "offline"
    PROVISIONING = "provisioning"
    LOADING = "loading"  # Container up, model downloading
    READY = "ready"
    TERMINATING = "terminating"

class GPUProfile(BaseModel):
    id: str  # e.g., "h100-single"
    name: str
    vram_gb: int
    cost_per_hr: float
    supported_models: List[str]
    task_type: str # "llm" or "image"
```

### 4.2. Environment Variables

```bash
# Direct Mode
RUNPOD_API_KEY=rp_...

# Managed Mode
kestrel_API_KEY=... (Existing auth)

# Configuration
GPU_DEFAULT_TTL_SECONDS=1800  # 30 minutes
GPU_DEFAULT_MODEL="llama-3-70b-instruct"
```

### 4.3. Tool Schema & Command Mapping

| Command | Tool Method | Parameters (in order) | Notes |
|---------|-------------|------------------------|-------|
| `!gpu on <model> <profile> <ttl>` | `manage_gpu` | `model_name` (str), `task_profile` (enum: `llm`/`image`), `ttl_seconds` (int), optional `pod_type` (enum) | Converts to `action="start"`; TTL validated against `GPU_DEFAULT_TTL_SECONDS` ceiling. |
| `!gpu status` | `manage_gpu` | none (defaults `action="status"`) | Returns pod id, `PodStatus`, TTL remaining, backend URL for `BrainRouter`. |
| `!gpu off` | `manage_gpu` | none (defaults `action="stop"`) | Tears down pods and resets `BrainRouter` to previous backend. |
| `!dream <prompt>` | `dream_image` helper | `prompt` (str), optional `model_name`, `ttl_seconds` | Wraps `manage_gpu` with `task_profile="image"`, ensures pod stops after render. |

Implementation Notes:
*   Each tool uses the `@tool` decorator with `command_prefix` so `ToolRegistry.route_command` can match inputs with no bespoke parsing logic.
*   `ToolParameter` metadata should include `enum` lists for `task_profile` and `pod_type` plus sensible defaults (e.g., TTL from env, default `pod_type` = `h100-single`).
*   Error responses must describe the command that failed and the provider message (`{"success": False, "error": "TTL exceeds policy", "tool": "manage_gpu"}`).
*   `tool_registry.get_help_text()` should list these commands; verify by running the helper in tests and asserting the RunPod entries appear alongside `!list-models`, `!export-sovereignty`, etc.

## 5. Open Questions / Decisions
*   **Image Baking**: We will start with a "Baked" strategy (Model inside Docker image) for speed, but support "Dynamic" loading for power users later.
    *   *Baked*: Fast start (~30s), limited selection.
    *   *Dynamic*: Slower start (~3m), infinite selection.
*   **Persistence**: Pods are ephemeral. No volume mounting in V1.

## 6. Out of Scope for V1

To keep the first implementation focused and shippable, V1 explicitly does **not** include:

*   Multi-user GPU pooling or shared GPU pods across different Kestrel users.
*   Long-lived GPU sessions that survive Kestrel restarts or user logouts.
*   GPU-hosted memory stores or vector databases (all long-term state remains in Kestrel).
*   Complex scheduling or queueing of GPU jobs beyond basic TTL/idle policies.

These can be revisited in a later iteration once the core `RunPodManager` + `BrainRouter` + `RunPodFeature` flow is stable.

## 7. Minimal Acceptance Criteria (V1)

V1 is considered complete when all of the following are true:

*   `!gpu on` using a dev or mock provider starts a pod, and the BrainRouter switches the active backend to `REMOTE_GPU` when status reaches `READY`.
*   `!gpu status` reports correct pod status, model, and remaining TTL, based on `RunPodManager`/provider responses.
*   When the GPU pod is terminated (TTL expiry or manual stop), the next generation attempt from the agent:
    *   Detects the failure,
    *   Falls back to the default backend (`CLOUD`/`LOCAL` as configured), and
    *   Optionally retries once and informs the user that the GPU session ended.
*   `!dream` successfully triggers an image-profile pod via the tool (real or mocked provider), performs a one-shot image generation, and shuts the pod down.
*   `tool_registry.route_command("!gpu status")` resolves to the RunPod tool, and `tool_registry.get_help_text()` lists the RunPod commands alongside existing ones without manual wiring inside `KestrelAgent`.
*   **Log Retrieval**: `!gpu logs` retrieves the last N lines of logs from the active pod via SSH.

## 8. Feature Agent Alignment (Non-Negotiable)

To comply with the Feature Agent framework (see `docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md`) the RunPod effort MUST ship as its own feature agent with the standard lifecycle:

1. **PRD** – keep this document as the authoritative PRD and reference it from `docs/architecture/RUNPOD_FEATURE.md` once created.
2. **Implementation** – all GPU management logic lives under `features/runpod/` (e.g., `features/runpod/feature.py`, `runpod_manager.py`, tool definitions). The core `KestrelAgent` only registers the feature; no GPU-specific logic should be embedded directly into the agent class.
3. **Tests** – add `tests/integration/test_runpod_feature.py` (real or simulated provider) producing a visible artifact such as the session ledger output. Optional unit tests can mock providers only after the integration path passes.

Delivering outside this structure violates the agent governance rules; if we discover blockers (e.g., router refactor touches multiple features) raise them early so we can allocate dedicated planning compute.

## 9. Referral Program & Revenue Economics

RunPod offers a tiered referral/affiliate program that Kestrel can leverage for passive revenue.

### Referral Program Tiers

| Tier | Commission | Duration | Requirements |
|------|------------|----------|--------------|
| Standard Referral | 3% Pod + 5% Serverless | 6 months | None |
| Affiliate Program | 10% all spend (cash) | Lifetime | 25+ referred users |
| Template Creator | 1% compute | Lifetime | Publish template to Hub |
| Hub Maintainer | Up to 7% compute | Monthly tiers | Maintain repos on Hub |

### Key Details

- **Sign-up Bonus**: Both referrer AND referred user get $5-500 random bonus after first $10 load
- **Grandfathered Referrals**: Pre-June 2025 referrals earn lifetime 3%/5% commission
- **Affiliate Payouts**: Via PartnerStack (PayPal or Stripe), verified monthly
- **Payment Schedule**: Commissions earned in February → paid March 13th

### Configuration

```bash
# Add to .env
RUNPOD_REFERRAL_CODE=your_referral_code_here
```

### Integration with Two-Mode Architecture

**Direct Mode (Sovereign):**
- User creates their own RunPod account via referral URL
- User pays RunPod directly
- Kestrel earns 3-10% referral commission
- URL format: `https://runpod.io/?ref=YOUR_CODE`

**Managed Mode (Kestrel):**
- User pays Kestrel for GPU access
- Kestrel uses platform RunPod account
- Kestrel earns margin (20-40%) instead of referral
- No referral code needed (user never sees RunPod)

### Revenue Projections

| Users | Direct Mode Users | Monthly Referral Revenue |
|-------|-------------------|-------------------------|
| 100 | 30 (30%) | $105 ($3.50 avg) |
| 500 | 150 (30%) | $525 |
| 1,000 | 300 (30%) | $1,050 |

**Path to Affiliate Program:**
- Need 25+ referred paying users
- Upgrade from 5% credits → 10% cash (lifetime)
- Significant revenue increase at scale

### Future Opportunities

1. **Template Publishing**: Create Kestrel-optimized RunPod templates (1% ongoing)
2. **Hub Maintenance**: Maintain LLM repos on RunPod Hub (up to 7%)
3. **Enterprise Volume**: Negotiate custom rates for high-volume users

*See [Provider Economics](PROVIDER_ECONOMICS.md) for cross-provider strategy.*

