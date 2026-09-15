/**
 * Kestrel Sovereign Console - Agent List Component (#2278 / design #2166)
 *
 * ONE agent/companion list surface, in the SAME contract family as chat's
 * `mount()` and the conversations module (`mountConversations` /
 * `mountConversationsPane`, #2149 / #2199 / #2216 / #2222). Before this module
 * the standalone console hand-rolled the agent loop inside identity.js's
 * `loadAgents`, and every embedding host (Frinz's companion list) maintained a
 * bespoke copy. This module owns the shared list logic — fetch-via-adapter,
 * the owned card shell, selection routing, the per-card `agent-card-actions`
 * slot, refresh, and the active-selection highlight — while card RENDERING is a
 * config hook so a host can supply its own skin.
 *
 * The division is exactly the conversations pane's owned-rows-vs-host-callbacks
 * split: the component owns chrome + orchestration; presentation is pluggable.
 *
 *   - `mountAgentList(containerEl, config)` — the embeddable LIST surface.
 *   - `createDefaultAgentAdapter(api)` — the default `/api/agents` adapter for
 *     the standalone console (avatar via `avatar_hash` → `/api/files/<hash>`).
 *
 * The machine-readable contract these build to is
 * `docs/proposals/agent-list-component/contract.js` (AgentListItem,
 * AgentListAdapter, AgentCardRenderer, AgentListConfig, AgentListHandle).
 */

import API from './api.js';
import { escapeHtml as sharedEscapeHtml } from './ui.js';
import { UI } from './ui-ext/registry.js';
import { storeGet, storeSet } from './ui_state.mjs';
import { validateHostStopEnvelope } from './stop_evidence.js';
import { createKebabButton, openMenuAt, positionFromEvent } from './kebab_menu.js';

// One pane owns one set of component listeners. A host may remount into
// adopted chrome without first retaining/destroying the old handle; carrying
// ownership on the container lets the new mount retire the old listeners
// before adopting the same buttons (#3155).
const AGENT_LIST_PANE_OWNER = Symbol.for('kestrel.agentListPane.owner');
const AGENT_LIST_STOP_ALL_OPERATION = Symbol.for('kestrel.agentListPane.stopAllOperation');
const AGENT_LIST_STOP_ALL_RETRY = Symbol.for('kestrel.agentListPane.stopAllRetry');

function newOperationId(prefix) {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
        return `${prefix}:${globalThis.crypto.randomUUID()}`;
    }
    return `${prefix}:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

function newStopAllCorrelationId() {
    return newOperationId('ui-host-stop');
}

// ============================================================================
// Default adapter — the standalone console's `/api/agents` data source
// ============================================================================

/**
 * The default {@link AgentListAdapter} over `API.getAgents()` (design question
 * 1: standalone reads `/api/agents`). It normalizes each agent-card record onto
 * the {@link AgentListItem} shape and resolves the portrait URL from
 * `avatar_hash` → `/api/files/<hash>` (the component never builds an avatar URL
 * itself). `mode` / `serverDemoMode` mirror the response fields so a host can
 * read them back after `listAgents()` resolves (identity.js drives its
 * demo-banner + misconfig-gated auto-select off them).
 */
export function createDefaultAgentAdapter(api = API) {
    const adapter = {
        mode: 'multi_agent',
        serverDemoMode: false,
        // POST /api/agents support, mirrored from the response — ABSENT on
        // hosts (subprocess) whose payload predates the flag: treat as false.
        canCreateAgents: false,
        // False until a /api/agents payload has actually been parsed —
        // consumers gating SAFETY decisions (demo rail, create-agent flow)
        // must fail CLOSED while this is false rather than trust the
        // defaults above (codex P1 on #2358).
        classificationLoaded: false,
        loadGeneration: 0,
        lastPayload: null,
        async listAgents() {
            const generation = ++adapter.loadGeneration;
            // Revoke the adapter's cached classification before awaiting the
            // authenticated discovery request. Security-sensitive consumers
            // must use API.canManageHostAgentLifecycle(); these mirrors remain
            // fail-closed for older presentation consumers as well.
            adapter.classificationLoaded = false;
            adapter.canCreateAgents = false;
            const data = await api.getAgents();
            if (generation === adapter.loadGeneration) {
                adapter.lastPayload = data;
                adapter.mode = data.mode === 'standalone' ? 'standalone' : 'multi_agent';
                adapter.serverDemoMode = data.server_demo_mode === true;
                adapter.canCreateAgents = data.can_create_agents === true;
                adapter.classificationLoaded = true;
            }
            const agents = data.agents || [];
            return agents.map((a) => {
                // `routing_name` is the AgentManager's immutable registration
                // key — what `/api/agents/{key}/…` paths MUST be built from.
                // `name` is the live/effective DISPLAY name, which a volatile
                // rename updates in the session without a durable write (#2672
                // review P2). The list item's `name` drives selection/routing
                // (select → setHostAgent → applyHostAgentPrefix), so it must be
                // the routing key; `displayName` carries the live name for the
                // visible card. Fall back to `a.name` when `routing_name` is
                // absent (standalone payloads / older hosts).
                const routingKey = a.routing_name || a.name;
                return {
                    name: routingKey,
                    id: a.id || a.did || routingKey,
                    displayName: a.name,
                    description: a.description,
                    avatarUrl: a.avatar_hash ? `/api/files/${a.avatar_hash}` : undefined,
                    status: a.status,
                    isDemo: a.is_demo === true,
                    raw: a,
                };
            });
        },
    };
    return adapter;
}

// ============================================================================
// Default card renderer — the CONSOLE ROW (matches today's `.agent-item`)
// ============================================================================

// Build the default console-row body: the per-agent thinking pulse and the
// name/description block. Returns a DocumentFragment so the
// children become DIRECT children of the `.agent-item` flex row (a wrapping
// div would break the row layout), letting the component prepend the
// component-owned status dot and append its shared controls around it. The
// status dot and Stop affordance are component-owned, so this renderer does
// not draw either one.
function makeConsoleRenderer() {
    return (item) => {
        const doc = typeof document !== 'undefined' ? document : null;
        const frag = doc.createDocumentFragment();
        const name = item.displayName || item.name || 'Unnamed Agent';

        const thinkingDot = doc.createElement('span');
        thinkingDot.className = 'agent-thinking-dot';
        // Tooltips show the live DISPLAY name (`name` local, displayName-first),
        // not the routing key, so a session rename reads correctly (#2672 P2).
        thinkingDot.title = `${name} is thinking`;
        frag.appendChild(thinkingDot);

        const info = doc.createElement('div');
        info.className = 'agent-info';
        const nameEl = doc.createElement('div');
        nameEl.className = 'agent-name';
        nameEl.textContent = name;
        const desc = doc.createElement('div');
        desc.className = 'agent-description';
        desc.textContent = item.description || 'No description';
        info.appendChild(nameEl);
        info.appendChild(desc);
        frag.appendChild(info);

        return frag;
    };
}

// The identities a card may be MATCHED by in a host payload. A lookup key set,
// not an authority — the default adapter falls back to the routing key for
// `id`, so one of these can be a name. What an operation is ADDRESSED to comes
// from the matched host record (`agent_id` / `resolved_target`) or from the
// routing key captured at render, never from whichever candidate matched.
function cardTargetIds(item) {
    return [
        item.id,
        item.raw && item.raw.did,
        item.raw && item.raw.id,
    ].filter((value) => typeof value === 'string' && value);
}

const STOP_DISPOSITION_LABELS = {
    stopped: 'Stopped',
    already_complete: 'Already complete',
    refused: 'Stop refused',
    unreachable: 'Stop unreachable',
    indeterminate: 'Stop indeterminate',
};

// One renderer for the typed Stop result, so the Stop button and the kebab's
// "Stop and hold" cannot describe the same receipt differently.
function renderStopOutcome(outcomeEl, item, routedTarget, result) {
    const outcomes = Array.isArray(result?.outcomes)
        ? result.outcomes
        : (
            Array.isArray(result?.stop_outcomes)
                ? result.stop_outcomes
                : (
                    Array.isArray(result?.response?.stop_outcomes)
                        ? result.response.stop_outcomes
                        : []
                )
        );
    const targetIds = [routedTarget, ...cardTargetIds(item)]
        .filter((value) => typeof value === 'string' && value);
    const outcome = outcomes.find((candidate) => (
        candidate
        && targetIds.includes(candidate.resolved_target || candidate.agent_id)
    )) || outcomes[0] || null;
    const disposition = outcome && typeof outcome.disposition === 'string'
        ? outcome.disposition
        : (result === true ? 'stopped' : 'unreachable');
    outcomeEl.hidden = false;
    outcomeEl.dataset.disposition = disposition;
    outcomeEl.textContent = STOP_DISPOSITION_LABELS[disposition] || `Stop: ${disposition}`;
    outcomeEl.title = outcome && typeof outcome.detail === 'string'
        ? outcome.detail
        : outcomeEl.textContent;
    return disposition;
}

function renderStopFailure(outcomeEl, error) {
    outcomeEl.hidden = false;
    outcomeEl.dataset.disposition = 'unreachable';
    outcomeEl.textContent = 'Stop unreachable';
    outcomeEl.title = error && error.message
        ? error.message
        : 'Cooperative Stop request failed';
}

// Stop is behavior, not card presentation. Keep the control on the component-
// owned seam so a host's custom portrait renderer cannot replace cancellation
// when it replaces the console body. Default rows receive these nodes directly;
// custom cards receive them in the actions anchor that the host positions.
function makeStopControls(doc, item, onStop) {
    const frag = doc.createDocumentFragment();
    const displayName = item.displayName || item.name || 'Unnamed Agent';
    const stopBtn = doc.createElement('button');
    stopBtn.className = 'agent-stop-btn';
    stopBtn.title = `Stop ${displayName}`;
    stopBtn.setAttribute('aria-label', `Stop ${displayName}`);
    stopBtn.innerHTML = '&times;';

    const outcomeEl = doc.createElement('span');
    outcomeEl.className = 'agent-stop-outcome';
    outcomeEl.setAttribute('role', 'status');
    outcomeEl.setAttribute('aria-live', 'polite');
    outcomeEl.hidden = true;

    // Capture the immutable routing key at render time. Display-name changes
    // and selection changes during an awaited Stop may never retarget retries.
    const routedTarget = item.name;
    const runStop = async () => {
        if (stopBtn.disabled || typeof onStop !== 'function') return null;
        stopBtn.disabled = true;
        outcomeEl.hidden = false;
        outcomeEl.dataset.disposition = 'requested';
        outcomeEl.textContent = 'Stopping…';
        try {
            const disposition = renderStopOutcome(
                outcomeEl,
                item,
                routedTarget,
                await onStop(routedTarget),
            );
            return disposition;
        } catch (error) {
            renderStopFailure(outcomeEl, error);
            return null;
        } finally {
            stopBtn.disabled = false;
        }
    };
    stopBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        void runStop();
    });
    frag.appendChild(stopBtn);
    frag.appendChild(outcomeEl);
    // `tail` stays a live reference after the fragment is appended, so Hold's
    // nodes can be inserted immediately after Stop's whenever they arrive —
    // whether that is at build time or once the host confirms the Hold door.
    return { nodes: frag, runStop, tail: outcomeEl };
}

// ============================================================================
// Hold — the durable latch, drawn on the CARD (not on the Stop button)
// ============================================================================
//
// `.agent-stop-btn` is `display:none` until the row is thinking, and Hold's
// primary case is an IDLE agent about to heartbeat — so a gesture on Stop is
// unreachable exactly when Hold is most needed (#3135 §7). Hold therefore
// lives on the card's own kebab, a visible focusable control, with the row's
// `contextmenu` as an accelerator onto the same menu. A held card renders a
// persistent badge and puts Resume in the Stop slot, so a hold set on Tuesday
// is still legible on Friday with nothing in flight.

const HOLD_DISPOSITION_LABELS = {
    applied: 'Held',
    already_in_state: 'Already held',
};

const RESUME_DISPOSITION_LABELS = {
    applied: 'Resumed',
    already_in_state: 'Not held',
    refused_stale: 'Hold changed — refreshed',
};

function formatHoldTime(value) {
    if (typeof value !== 'string' || !value) return '';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    try { return parsed.toLocaleString(); } catch (_) { return value; }
}

// A mutation's `current` is the host's authoritative post-mutation latch for
// that exact scope and target, under EVERY disposition — the fresh latch, the
// pre-existing one, `null` once a release lands, or the superseding latch a
// stale release was refused against. It is a committed fact the follow-up read
// may never get to confirm, so the card applies it directly; `undefined` here
// means the response carried no usable evidence and nothing is applied.
function mutationLatch(response) {
    if (!response || typeof response !== 'object') return undefined;
    if (!('current' in response)) return undefined;
    const current = response.current;
    if (current === null) return null;
    if (typeof current !== 'object' || Array.isArray(current)) return undefined;
    return typeof current.hold_receipt_id === 'string' && current.hold_receipt_id
        ? current
        : undefined;
}

// The latch a card is held by, from the host's own inventory. A card with no
// entry has no resolved identity, so it offers no Hold at all rather than
// inventing a target the latch table would never be read for.
function holdEntryForItem(item, byAgent) {
    if (!byAgent || typeof byAgent.get !== 'function') return null;
    let resolved = null;
    for (const candidate of cardTargetIds(item)) {
        const entry = byAgent.get(candidate);
        if (!entry) continue;
        if (resolved && resolved.agent_id !== entry.agent_id) return null; // ambiguous
        resolved = entry;
    }
    return resolved;
}

function makeHoldControls(doc, item, ctx) {
    const displayName = item.displayName || item.name || 'Unnamed Agent';

    const badge = doc.createElement('span');
    badge.className = 'agent-hold-badge';
    badge.setAttribute('role', 'status');
    badge.hidden = true;
    const badgeLabel = doc.createElement('span');
    badgeLabel.className = 'agent-hold-badge-label';
    badgeLabel.textContent = 'Held';
    const badgeReason = doc.createElement('span');
    badgeReason.className = 'agent-hold-badge-reason';
    const badgeActor = doc.createElement('span');
    badgeActor.className = 'agent-hold-badge-actor';
    const badgeTime = doc.createElement('span');
    badgeTime.className = 'agent-hold-badge-time';
    badge.appendChild(badgeLabel);
    badge.appendChild(badgeReason);
    badge.appendChild(badgeActor);
    badge.appendChild(badgeTime);

    const outcomeEl = doc.createElement('span');
    outcomeEl.className = 'agent-hold-outcome';
    outcomeEl.setAttribute('role', 'status');
    outcomeEl.setAttribute('aria-live', 'polite');
    outcomeEl.hidden = true;

    const resumeBtn = doc.createElement('button');
    resumeBtn.type = 'button';
    resumeBtn.className = 'agent-resume-btn';
    resumeBtn.textContent = 'Resume';
    resumeBtn.title = `Resume ${displayName}`;
    resumeBtn.setAttribute('aria-label', `Resume ${displayName}`);

    const state = { entry: null, canHold: false, hostHold: null, busy: false };

    function renderOutcome(labels, disposition, detail) {
        outcomeEl.hidden = false;
        outcomeEl.dataset.disposition = disposition;
        outcomeEl.textContent = labels[disposition] || disposition;
        outcomeEl.title = detail || outcomeEl.textContent;
    }

    function renderFailure(message) {
        outcomeEl.hidden = false;
        outcomeEl.dataset.disposition = 'unreachable';
        outcomeEl.textContent = 'Hold unreachable';
        outcomeEl.title = message || 'The durable Hold request failed';
    }

    async function withBusy(run) {
        if (state.busy) return;
        state.busy = true;
        resumeBtn.disabled = true;
        try {
            await run();
        } finally {
            state.busy = false;
            resumeBtn.disabled = false;
            await ctx.refreshHoldState();
        }
    }

    // Hold BEFORE Stop: latching willingness first closes the window in which
    // a heartbeat could start a fresh turn between the two operations. They
    // stay two typed requests with two receipts — the gesture is compound,
    // the operations are not.
    async function hold({ alsoStop }) {
        const entry = state.entry;
        if (!entry || !state.canHold) return;
        const reason = ctx.askReason(
            alsoStop
                ? `Stop ${displayName} and hold it. Reason:`
                : `Hold ${displayName}. Reason:`,
            '',
        );
        if (!reason) return;
        await withBusy(async () => {
            let response;
            try {
                response = await ctx.setHold({
                    scope: 'agent',
                    target_id: entry.agent_id,
                    reason,
                    operation_id: newOperationId('ui-hold'),
                });
            } catch (error) {
                renderFailure(error && error.message);
                return;
            }
            const receipt = response && response.receipt;
            // BOTH Hold dispositions leave a latch, so a response carrying no
            // current latch contradicts its own receipt. Say indeterminate
            // rather than reading that receipt as a hold.
            const current = mutationLatch(response);
            const latched = !!current;
            // Paint the committed latch NOW, from the mutation that committed
            // it. The refresh in `withBusy`'s finally is confirmation, not the
            // source of truth: if that read fails, a card whose Hold demonstrably
            // landed must still render as held rather than reverting to the
            // pre-mutation reading while the outcome says "Held".
            // MUTANT
            const disposition = latched && receipt && typeof receipt.disposition === 'string'
                ? receipt.disposition
                : 'indeterminate';
            renderOutcome(
                HOLD_DISPOSITION_LABELS,
                disposition,
                receipt && receipt.receipt_id
                    ? `Hold receipt ${receipt.receipt_id}`
                    : null,
            );
            // Only proceed to the Stop half once the latch is demonstrably
            // active; a bare Stop after a failed Hold is a different action
            // from the one the operator asked for.
            if (alsoStop && latched) {
                await ctx.runStop();
            }
        });
    }

    async function resume() {
        const entry = state.entry;
        const latch = entry && entry.agent_hold;
        if (!latch || !state.canHold) return;
        const reason = ctx.askReason(
            `Resume ${displayName}. Reason:`,
            'Resumed from the agent card',
        );
        if (!reason) return;
        await withBusy(async () => {
            let response;
            try {
                response = await ctx.releaseHold({
                    scope: 'agent',
                    target_id: entry.agent_id,
                    reason,
                    operation_id: newOperationId('ui-resume'),
                    // The receipt the operator SAW. A hold replaced since the
                    // last poll is refused as stale rather than released.
                    expected_hold_receipt_id: latch.hold_receipt_id,
                });
            } catch (error) {
                renderFailure(error && error.message);
                return;
            }
            const receipt = response && response.receipt;
            // Same rule in the other direction, and it covers `refused_stale`
            // too: the latch that came back is the one that actually holds this
            // agent now, so the badge and the next Resume's compare-and-set both
            // name it without waiting on a read that may not arrive.
            const current = mutationLatch(response);
            if (current !== undefined) ctx.applyLatch(entry.agent_id, current);
            const disposition = receipt && typeof receipt.disposition === 'string'
                ? receipt.disposition
                : 'indeterminate';
            renderOutcome(
                RESUME_DISPOSITION_LABELS,
                disposition,
                receipt && receipt.receipt_id
                    ? `Release receipt ${receipt.receipt_id}`
                    : null,
            );
        });
    }

    function menuItems() {
        const entry = state.entry;
        if (!entry || !state.canHold) return [];
        const items = [];
        const thinking = ctx.isThinking();
        if (entry.agent_hold) {
            items.push({ label: `Resume ${displayName}`, action: 'resume', onSelect: () => { void resume(); } });
        } else {
            items.push({ label: `Hold ${displayName}…`, action: 'hold', onSelect: () => { void hold({ alsoStop: false }); } });
            if (thinking) {
                items.push({
                    label: 'Stop and hold…',
                    action: 'stop-and-hold',
                    onSelect: () => { void hold({ alsoStop: true }); },
                });
            }
        }
        if (thinking) {
            items.push({
                label: `Stop ${displayName}`,
                action: 'stop',
                separatorBefore: true,
                onSelect: () => { void ctx.runStop(); },
            });
        }
        return items;
    }

    const kebabBtn = createKebabButton(menuItems, {
        className: 'agent-card-kebab',
        ariaLabel: `Actions for ${displayName}`,
        title: `${displayName} actions`,
    });
    kebabBtn.disabled = true;

    resumeBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        void resume();
    });

    // The kebab button is the visible, focusable path; `contextmenu` is an
    // accelerator onto the SAME menu, never the only way in. It is bound and
    // unbound with the nodes, so a card without the Hold surface keeps the
    // browser's own context menu.
    function onContextMenu(event) {
        if (typeof event.preventDefault === 'function') event.preventDefault();
        if (kebabBtn.disabled) return;
        openMenuAt(menuItems(), positionFromEvent(event));
    }

    function apply(shell, { entry, canHold, hostHold, stale }) {
        state.entry = entry || null;
        state.canHold = canHold === true;
        state.hostHold = hostHold || null;
        const unconfirmed = stale === true;
        const agentHold = entry && entry.agent_hold;
        const sources = Array.isArray(entry && entry.sources) ? entry.sources : [];
        const held = !!(entry && entry.held === true);
        const primary = agentHold || (sources.includes('host') ? state.hostHold : null);

        kebabBtn.disabled = !(state.entry && state.canHold);
        kebabBtn.title = kebabBtn.disabled
            ? 'Hold controls are unavailable'
            : `${displayName} actions`;

        if (shell && shell.classList) {
            shell.classList.toggle('agent-held', held);
        }
        badge.hidden = !held;
        if (!held) {
            delete badge.dataset.holdSources;
            delete badge.dataset.holdStale;
            resumeBtn.hidden = true;
            return;
        }
        badge.dataset.holdSources = sources.join(' ');
        if (unconfirmed) badge.dataset.holdStale = 'true';
        else delete badge.dataset.holdStale;
        badgeLabel.textContent = sources.includes('agent') ? 'Held' : 'Held by host';
        badgeReason.textContent = (primary && primary.reason) || '';
        badgeActor.textContent = (primary && primary.actor_id) || '';
        badgeTime.textContent = formatHoldTime(primary && primary.set_at);
        const scopeText = sources.includes('agent')
            ? 'Held'
            : 'Held by the host-wide Hold';
        const detail = primary
            ? `${scopeText} by ${primary.actor_id} at ${primary.set_at} — ${primary.reason}`
            : scopeText;
        // Say WHICH it is: a latch the host just confirmed, or the last one it
        // did. Silently presenting the second as the first is the lie.
        badge.title = unconfirmed
            ? `${detail} (last confirmed reading; the host Hold state is currently unreadable)`
            : detail;
        // Resume releases ONLY the latch this card owns. A card held solely by
        // the host latch gets no Resume here: releasing that one from an agent
        // card would release a latch the operator did not intend.
        resumeBtn.hidden = !(agentHold && state.canHold);
        resumeBtn.title = `Resume ${displayName} — releases this agent's hold only`;
    }

    return { nodes: [badge, resumeBtn, outcomeEl, kebabBtn], apply, onContextMenu };
}

// ============================================================================
// mountAgentList — the embeddable list surface
// ============================================================================

/**
 * Mount the agent/companion list surface into `containerEl`. Returns a handle:
 * `{ element, refresh, select, setActiveName, getActive, destroy }`.
 *
 * The component NEVER fetches directly — it calls `config.adapter.listAgents()`
 * (defaulting to the `/api/agents` adapter). Selecting a card drives the shared
 * host-agent selection path — `API.setHostAgent(name)` in multi-agent mode ONLY
 * (standalone must not install a route prefix; see the identity.js `loadAgents`
 * note) — then invokes the host `onSelect`. Card RENDERING is `config.renderCard`
 * (default = the console row); the component owns the outer shell, the status
 * dot, and the per-card `agent-card-actions` slot anchor.
 *
 * Hold (#3164) is component-owned too, so every host that adopts this list —
 * the console and Frinz's companions pane — inherits the same latch surface.
 * Whether a host HAS that door is a server fact, so the component asks the
 * server: the API client exposing `getHostHoldState` / `setHostHold` /
 * `releaseHostHold` only makes the question askable (the standard
 * `createApiClient()` exposes all three regardless of what the backend mounts),
 * and no Hold affordance is drawn until `GET /api/host/hold` answers. A host
 * that answers "no such route" retires the surface and stops polling, so its
 * cards are exactly today's cards.
 *   - `hold` — skip the probe: `false` never offers Hold, `true` asserts the
 *     door exists (for a host that knows its own backend).
 *   - `holdStatusIntervalMs` — authoritative latch-poll cadence (default 5s).
 *   - `askHoldReason(message, defaultValue)` — reason prompt override.
 */
export function mountAgentList(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountAgentList requires a container element');
    const doc = containerEl.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (!doc) throw new Error('mountAgentList requires a document');

    const api = config.api || API;
    const adapter = config.adapter || createDefaultAgentAdapter(api);
    const escapeFn = typeof config.escapeHtml === 'function' ? config.escapeHtml : sharedEscapeHtml;
    const hostRenderCard = typeof config.renderCard === 'function' ? config.renderCard : null;
    const usingDefaultRenderer = !hostRenderCard;
    const showStatusDot = config.showStatusDot !== false; // default true = console behavior
    const isThinking = typeof config.isThinking === 'function' ? config.isThinking : () => false;
    const onStop = typeof config.onStop === 'function' ? config.onStop : null;
    const onSelect = typeof config.onSelect === 'function'
        ? config.onSelect
        : (adapter && typeof adapter.onSelect === 'function' ? adapter.onSelect : null);
    const renderCard = hostRenderCard || makeConsoleRenderer();

    let items = [];
    let activeName = config.selectedName || null;
    let refreshSeq = 0;

    // --- Hold: the durable latch surface (#3164) ---------------------------
    // Two different questions, deliberately kept apart. `holdCallable` is only
    // "can this client ASK" — the standard createApiClient() always exposes the
    // three methods, so treating their presence as proof the backend mounts
    // /api/host/hold would opt an embedding host in by accident. `holdSupported`
    // is the answer: null until the host has replied, true once it has, false
    // once it has said there is no such door.
    const holdCallable = !!(api
        && typeof api.getHostHoldState === 'function'
        && typeof api.setHostHold === 'function'
        && typeof api.releaseHostHold === 'function');
    let holdSupported = null;
    if (!holdCallable || config.hold === false) holdSupported = false;
    else if (config.hold === true) holdSupported = true;
    const askHoldReason = typeof config.askHoldReason === 'function'
        ? config.askHoldReason
        : ((message, defaultValue) => (
            typeof window !== 'undefined' && typeof window.prompt === 'function'
                ? window.prompt(message, defaultValue)
                : null
        ));
    // A Hold and a Resume are receipted governance acts, so both carry an
    // operator reason. A blank or cancelled answer aborts rather than
    // silently substituting one.
    function askReason(message, defaultValue) {
        const answer = askHoldReason(message, defaultValue);
        return typeof answer === 'string' && answer.trim() ? answer.trim() : null;
    }
    // The browser never derives "held" — the host composes the two
    // independent latches and says so. Until a read succeeds the controls
    // stay inert rather than drawing an agent as un-held.
    let holdState = {
        loaded: false,
        stale: false,
        canHold: false,
        hostHold: null,
        byAgent: new Map(),
    };
    let holdSeq = 0;
    let holdPromise = null;
    const holdViews = [];

    // Hold is durable state a card must keep showing with nothing in flight,
    // and it can be set from another tab, the CLI, or a mandate holder. Poll
    // the authoritative latch rather than trusting this document's memory.
    // The poll is retired with the surface: a host with no Hold door must not
    // be asked for one every few seconds forever.
    const holdIntervalMs = Number.isFinite(config.holdStatusIntervalMs)
        ? Math.max(250, config.holdStatusIntervalMs)
        : 5000;
    const setIntervalFn = doc.defaultView && doc.defaultView.setInterval;
    const clearIntervalFn = doc.defaultView && doc.defaultView.clearInterval;
    let holdInterval = holdSupported !== false && typeof setIntervalFn === 'function'
        ? setIntervalFn.call(doc.defaultView, () => { void refreshHoldState(); }, holdIntervalMs)
        : null;
    function stopHoldPolling() {
        if (holdInterval === null) return;
        if (typeof clearIntervalFn === 'function') {
            clearIntervalFn.call(doc.defaultView, holdInterval);
        }
        holdInterval = null;
    }

    if (containerEl.classList) containerEl.classList.add('agent-list-component');
    const root = doc.createElement('div');
    root.className = 'agent-list-root';
    containerEl.innerHTML = '';
    containerEl.appendChild(root);

    function mode() { return (adapter && adapter.mode) || 'multi_agent'; }
    function isStandalone() { return mode() === 'standalone'; }
    function findItem(name) { return items.find((i) => i && i.name === name) || null; }

    // Portrait/signed-URL hosts can mint the avatar URL per render via an adapter
    // `avatarUrl(item)` override; otherwise the adapter precomputed it in
    // `listAgents`. The component never builds an avatar URL itself.
    function resolveAvatar(item) {
        if (adapter && typeof adapter.avatarUrl === 'function') {
            try { return adapter.avatarUrl(item); } catch (_) { /* fall through */ }
        }
        return item ? item.avatarUrl : undefined;
    }

    // Put a card's Hold nodes immediately after its Stop nodes — the position
    // they occupy when the host's answer arrives before the card is built — so
    // the surface lands in one place whichever race wins.
    function attachHoldView(view) {
        if (view.attached) return;
        view.attached = true;
        let cursor = view.anchor;
        const parent = (cursor && cursor.parentNode) || view.shell;
        for (const node of view.controls.nodes) {
            parent.insertBefore(node, cursor ? cursor.nextSibling : null);
            cursor = node;
        }
        view.shell.addEventListener('contextmenu', view.controls.onContextMenu);
    }

    function detachHoldView(view) {
        if (!view.attached) return;
        view.attached = false;
        for (const node of view.controls.nodes) {
            if (node.parentNode) node.parentNode.removeChild(node);
        }
        view.shell.removeEventListener('contextmenu', view.controls.onContextMenu);
        if (view.shell.classList) view.shell.classList.remove('agent-held');
    }

    // The host says there is no Hold door. Take the surface off every card,
    // stop asking, and build no more of it — an operator must not be left a
    // permanently disabled control and a poll against a route that 404s.
    function retireHoldSurface() {
        if (holdSupported === false) return;
        holdSupported = false;
        holdSeq++; // orphan any in-flight read
        holdState = {
            loaded: true, stale: false, canHold: false, hostHold: null, byAgent: new Map(),
        };
        stopHoldPolling();
        for (const view of holdViews) detachHoldView(view);
        holdViews.length = 0;
    }

    // "No such route" is a structural answer, not a blip: the door is absent,
    // and retrying cannot make it appear. Every other failure (auth, 5xx,
    // offline) leaves the question open and the last reading standing.
    function holdRouteAbsent(error) {
        const status = error && error.status;
        return status === 404 || status === 405 || status === 501;
    }

    // Repaint every card's Hold affordances in place. Like renderHighlight,
    // this never rebuilds the list — a latch poll must not tear down the
    // per-card slot anchors and their contributions.
    function renderHoldState() {
        for (const view of holdViews) {
            if (holdSupported === true) attachHoldView(view);
            if (!view.attached) continue;
            view.controls.apply(view.shell, {
                entry: holdEntryForItem(view.item, holdState.byAgent),
                canHold: holdState.canHold,
                hostHold: holdState.hostHold,
                stale: holdState.stale === true,
            });
        }
    }

    async function refreshHoldState() {
        if (!holdCallable || holdSupported === false) return false;
        if (holdPromise) return holdPromise;
        const request = (async () => {
            const seq = ++holdSeq;
            try {
                const payload = await api.getHostHoldState();
                if (seq !== holdSeq) return false;
                const byAgent = new Map();
                const entries = Array.isArray(payload && payload.agents)
                    ? payload.agents
                    : [];
                for (const entry of entries) {
                    if (entry && typeof entry.agent_id === 'string' && entry.agent_id) {
                        byAgent.set(entry.agent_id, entry);
                    }
                }
                // The host answered, so the door exists. This is the only place
                // support is established; method presence never establishes it.
                holdSupported = true;
                holdState = {
                    loaded: true,
                    stale: false,
                    canHold: !!(payload && payload.can_hold === true),
                    hostHold: (payload && payload.host_hold) || null,
                    byAgent,
                };
            } catch (error) {
                if (seq !== holdSeq) return false;
                if (holdRouteAbsent(error)) {
                    retireHoldSurface();
                    return false;
                }
                // A failed read is not "nothing is held" — erasing the badge on
                // a network blip is exactly the defect Hold exists to prevent
                // (a latch whose only record is the operator's memory). Keep
                // the last observed latch, mark it unconfirmed, and withdraw
                // authority: a mutation needs a receipt this read did not get.
                holdState = { ...holdState, loaded: true, canHold: false, stale: true };
            }
            renderHoldState();
            return holdState.canHold;
        })();
        holdPromise = request;
        try {
            return await request;
        } finally {
            if (holdPromise === request) holdPromise = null;
        }
    }

    // Write the latch a mutation committed into the state the cards render
    // from. `held`/`sources` are composed by the SAME rule the host applies
    // (EffectiveHoldState): held if either independent latch is set, in
    // host-then-agent order. The host latch is untouched by an agent mutation,
    // so the last reading of it remains the best evidence there is.
    function applyLatch(agentId, latch) {
        if (typeof agentId !== 'string' || !agentId) return;
        const previous = holdState.byAgent.get(agentId) || null;
        const sources = [];
        if (holdState.hostHold) sources.push('host');
        if (latch) sources.push('agent');
        const byAgent = new Map(holdState.byAgent);
        byAgent.set(agentId, {
            ...(previous || {}),
            agent_id: agentId,
            held: sources.length > 0,
            sources,
            agent_hold: latch || null,
        });
        holdState = { ...holdState, loaded: true, byAgent };
        renderHoldState();
    }

    function buildCard(item) {
        const selected = item.name != null && item.name === activeName;
        const offline = item.status === 'offline';
        const thinking = !!isThinking(item.name);

        const shell = doc.createElement('div');
        const classes = [];
        // The default renderer IS the console row, so tag the shell `.agent-item`
        // for its row layout. Every renderer retains `.agent-card`, which is the
        // shared live-state selector used by refreshAgentThinkingDot. A host
        // renderer therefore gets a clean shell without console-row layout.
        if (usingDefaultRenderer) classes.push('agent-item');
        classes.push('agent-card');
        if (selected) classes.push('selected');
        if (offline) classes.push('offline');
        if (thinking) classes.push('agent-thinking');
        shell.className = classes.join(' ');
        // Always carry the real agent name — thinking-dot / stop lookups depend
        // on it in every mode (see chat.js refreshAgentThinkingDot).
        shell.dataset.agentName = item.name || '';

        // The per-card `agent-card-actions` slot anchor (design question 3),
        // component-created and handed to the renderer as `ctx.actionsAnchor`.
        const actionsAnchor = doc.createElement('div');
        actionsAnchor.dataset.slot = 'agent-card-actions';
        actionsAnchor.className = 'agent-card-actions';
        const stopControls = makeStopControls(doc, item, onStop);
        // Hold rides the same component-owned seam as Stop, so a host renderer
        // that replaces the card body cannot replace the latch surface either.
        // Its nodes are built here but ATTACHED only once the host has answered
        // that it has a Hold door — see attachHoldView.
        const holdControls = holdSupported !== false
            ? makeHoldControls(doc, item, {
                isThinking: () => !!isThinking(item.name),
                askReason,
                setHold: (payload) => api.setHostHold(payload),
                releaseHold: (payload) => api.releaseHostHold(payload),
                runStop: () => stopControls.runStop(),
                refreshHoldState: () => refreshHoldState(),
                applyLatch,
            })
            : null;
        if (!usingDefaultRenderer) {
            actionsAnchor.appendChild(stopControls.nodes);
        }

        // Component-owned status dot — a config flag (`showStatusDot`, default
        // true = console behavior); a host renderCard may omit it entirely.
        let statusDot = null;
        if (showStatusDot) {
            statusDot = doc.createElement('span');
            statusDot.className = `agent-status-dot ${offline ? 'offline' : 'online'}`;
        }

        // Selection: only online agents in multi-agent mode are clickable —
        // standalone must NOT install a host-agent prefix (it 404s the
        // un-prefixed routes; the identity.js note at loadAgents), and offline
        // agents are not selectable. `select()` is still available programmatically.
        const selectable = !offline && !isStandalone();
        if (selectable) {
            shell.addEventListener('click', () => select(item.name));
        }

        if (adapter && typeof adapter.avatarUrl === 'function') {
            item.avatarUrl = resolveAvatar(item);
        }

        const ctx = {
            selected,
            standalone: isStandalone(),
            escapeHtml: escapeFn,
            actionsAnchor,
        };
        const body = renderCard(item, ctx);

        // Component owns the shell layout: status dot first, then the renderer's
        // body, then the actions anchor — UNLESS a host renderer already placed
        // the anchor inside its own layout (portrait cards position it under the
        // portrait), in which case the component leaves it where the host put it.
        if (statusDot) shell.appendChild(statusDot);
        if (body) shell.appendChild(body);
        if (usingDefaultRenderer) {
            shell.appendChild(stopControls.nodes);
        }
        if (!actionsAnchor.parentNode) shell.appendChild(actionsAnchor);

        if (holdControls) {
            const view = {
                shell,
                item,
                controls: holdControls,
                // Stop's last node, live in whichever parent the card put it in
                // (the shell for the console row, the actions anchor for a host
                // renderer). Hold's nodes follow it.
                anchor: stopControls.tail,
                attached: false,
            };
            holdViews.push(view);
            if (holdSupported === true) attachHoldView(view);
        }

        // NOTE: the `agent-card-actions` slot is rendered by `renderList` AFTER
        // this shell is appended to the live `root`, NOT here. Slot code
        // (e.g. voice's mountAgentVoiceControls → refreshAgentVoiceCard) locates
        // its row via `document.querySelector`, which returns null against a
        // detached card and silently skips the initial state paint. Mounting the
        // slot only once the card is in the document restores the pre-#2278
        // ordering (card live → slot mounts).
        return { shell, actionsAnchor, item };
    }

    // Render the per-card actions slot INTO the anchor via the existing ui-ext
    // registry (unchanged from identity.js today). MUST run only after the card
    // is attached to the live DOM so slot code that does `document.querySelector`
    // for its row resolves. `standalone` comes from `adapter.mode`; `agentName`
    // is the item's name (the voice session key). The registry tears the anchor's
    // contributions down on the next list rebuild.
    function mountActionsSlot(actionsAnchor, item) {
        try {
            UI.renderSlot('agent-card-actions', {
                element: actionsAnchor,
                api,
                agentName: item.name,
                standalone: isStandalone(),
            });
        } catch (_) { /* a missing/misbehaving slot must not break the row */ }
    }

    function renderList() {
        root.innerHTML = '';
        holdViews.length = 0;
        if (!items.length) {
            const empty = doc.createElement('p');
            empty.className = 'agent-list-empty empty-state';
            empty.textContent = config.emptyText || 'No agents available';
            root.appendChild(empty);
            return;
        }
        // Append every card to the live DOM FIRST, then mount each card's actions
        // slot — so slot code that queries `document` for its row resolves.
        for (const item of items) {
            const { shell, actionsAnchor } = buildCard(item);
            root.appendChild(shell);
            mountActionsSlot(actionsAnchor, item);
        }
        // A rebuilt card starts un-held; repaint from the latch state already
        // read so a list refresh never blanks a hold badge for a poll interval.
        renderHoldState();
    }

    // Repaint the active-selection highlight only — no rebuild, so per-card slot
    // anchors and listeners survive (mirrors the conversations pane's
    // setActiveSessionId, #2222).
    function renderHighlight() {
        const cards = root.querySelectorAll('.agent-card');
        cards.forEach((c) => {
            c.classList.toggle('selected', c.dataset.agentName === activeName);
        });
    }

    // In multi-agent mode, select the first online agent when none is selected
    // (matches identity.js). Gated by `autoSelectFirst`; the standalone console
    // keeps its own demo-misconfig gate host-side (via `onLoaded`).
    function maybeAutoSelect() {
        if (!config.autoSelectFirst) return;
        if (isStandalone()) return;
        if (activeName) return;
        const firstOnline = items.find((i) => i && i.status !== 'offline');
        if (firstOnline) select(firstOnline.name);
    }

    async function refresh() {
        // Seq-guard like the conversations list so a stale response never wins.
        const seq = ++refreshSeq;
        try {
            const next = await adapter.listAgents();
            if (seq !== refreshSeq) return;
            items = Array.isArray(next) ? next : [];
            renderList();
            if (typeof config.onLoaded === 'function') {
                config.onLoaded(items, { mode: mode() });
            }
            maybeAutoSelect();
        } catch (e) {
            if (seq !== refreshSeq) return;
            root.innerHTML = '';
            const err = doc.createElement('p');
            err.className = 'agent-list-error';
            err.textContent = config.errorText || 'Failed to load agents';
            root.appendChild(err);
            if (typeof config.onError === 'function') config.onError(e);
        }
    }

    // Drive the shared host-agent selection path: pin routing
    // (`API.setHostAgent`) in multi-agent mode ONLY, repaint the active
    // highlight, then invoke the host `onSelect` (chat mount / product state).
    function select(name) {
        activeName = name;
        if (!isStandalone() && api && typeof api.setHostAgent === 'function') {
            api.setHostAgent(name);
        }
        renderHighlight();
        if (typeof onSelect === 'function') {
            onSelect(findItem(name) || { name }, { standalone: isStandalone() });
        }
    }

    // Override the active highlight WITHOUT firing selection (host reconciling
    // its own notion of current agent).
    function setActiveName(name) {
        activeName = name;
        renderHighlight();
    }

    function destroy() {
        refreshSeq++; // orphan any in-flight refresh
        holdSeq++;    // and any in-flight latch read
        holdViews.length = 0;
        stopHoldPolling();
        containerEl.innerHTML = '';
    }

    if (config.autoLoad !== false) refresh();
    // Ask the host for its latch immediately: the answer both establishes that
    // the Hold door exists and paints a reloaded page's held cards as held on
    // first render, rather than a poll interval later.
    void refreshHoldState();

    return {
        element: root,
        refresh,
        refreshHoldState,
        select,
        setActiveName,
        getActive: () => findItem(activeName),
        destroy,
    };
}

// ============================================================================
// mountAgentListPane — the full collapsible pane unit (design #2166 §4)
// ============================================================================

// Best-effort localStorage persistence now lives in the shared ui_state.mjs
// module (#2298) — `storeGet`/`storeSet` are imported above. The raw-string
// contract is unchanged, so the pane's collapse/width on-disk format is
// byte-for-byte identical (no stored state migrates or breaks).

/**
 * Mount the full collapsible agent/companion PANE — the embeddable list surface
 * (`mountAgentList`) PLUS the surrounding pane chrome. This is the agent
 * analogue of `mountConversationsPane`, sharing the SAME chrome contract (#2199
 * / #2216) so the standalone console and any embed consume ONE pane
 * implementation; a host provides only a container + config and gets:
 *
 *   - a `<` chevron that fully HIDES the pane (`display:none`, no leftover rail;
 *     #2216) with `open()` / `close()` / `toggle()` and an `onToggle(collapsed)`
 *     callback so a host toolbar trigger can reopen it;
 *   - a drag-resize handle with min/max width + `localStorage` persistence
 *     (width under `:width`, collapsed state under `:collapsed`), guarded to a
 *     no-op when localStorage is unavailable;
 *   - a component-owned "+ New" header action wired via `config.onNew` (Frinz →
 *     Add-a-Companion; standalone console → new-agent / spawn flow, an
 *     affordance it gains here, same-everywhere rule).
 *
 * Chrome is ADOPT-or-BUILD: when the container already looks like a pane (the
 * console's static `#agents-pane` with its `.pane-header` / `.resize-handle`),
 * those elements are reused; when the container is bare (the embedder contract —
 * a host hands over just a `<div>`), the full chrome is built inside it.
 * `destroy()` removes only chrome this mount built; adopted chrome is left.
 *
 * Config (all optional except where the list needs them):
 *   - api, adapter, renderCard, showStatusDot, isThinking, onStop, onSelect,
 *     onLoaded, onError, autoLoad, autoSelectFirst, selectedName, escapeHtml,
 *     emptyText, errorText — forwarded verbatim to `mountAgentList`.
 *   - onNew()          — the "+ New" header action (Add-a-Companion / new agent).
 *                        The New button is only built/adopted when this is a fn.
 *   - onPrepareStopAll(items) — REQUIRED to enable Stop All: synchronous
 *                        browser-work fence returning an optional settlement
 *                        callback invoked with (response, error, correlationId).
 *   - confirmStopAll(message) — host confirmation override (defaults to confirm).
 *   - stopAllStatusIntervalMs — authoritative host-status refresh cadence.
 *   - newLabel         — accessible label / tooltip for the New button.
 *   - collapsed        — initial collapsed state (overridden by persistence).
 *   - storageKey       — persistence namespace (default 'kestrel:agents-pane').
 *   - title            — pane header title (default 'Agents').
 *   - onToggle(bool)   — fired after every collapse/expand with the new state.
 *   - minWidth/maxWidth — resize clamps (default 200 / 500, matching the CSS).
 *
 * Returns a handle:
 *   `{ element, list, refresh, select, setActiveName, getActive,
 *      open, close, toggle, collapsed, destroy }`
 * where `list` is the inner `mountAgentList` handle.
 */
export function mountAgentListPane(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountAgentListPane requires a container element');
    const doc = containerEl.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (!doc) throw new Error('mountAgentListPane requires a document');

    const priorOwner = containerEl[AGENT_LIST_PANE_OWNER];
    if (priorOwner && typeof priorOwner.destroy === 'function') {
        priorOwner.destroy();
    }

    const storageKey = config.storageKey || 'kestrel:agents-pane';
    const api = config.api || API;
    const KEY_WIDTH = `${storageKey}:width`;
    const KEY_COLLAPSED = `${storageKey}:collapsed`;
    const minWidth = Number.isFinite(config.minWidth) ? config.minWidth : 200;
    const maxWidth = Number.isFinite(config.maxWidth) ? config.maxWidth : 500;

    // The container IS the pane element; tag it so it inherits the pane-sidebar
    // chrome CSS whether it was already a pane (adopt) or a bare div (build).
    if (containerEl.classList) {
        containerEl.classList.add('pane-sidebar', 'agent-list-pane');
    }
    const paneEl = containerEl;

    // --- Header (adopt existing .pane-header, else build one) ---------------
    let header = paneEl.querySelector('.pane-header');
    let builtHeader = false;
    if (!header) {
        header = doc.createElement('div');
        header.className = 'pane-header';
        const h3 = doc.createElement('h3');
        h3.className = 'agent-list-pane-title';
        h3.textContent = config.title || 'Agents';
        header.appendChild(h3);
        paneEl.insertBefore(header, paneEl.firstChild);
        builtHeader = true;
    }

    // --- Collapse rail (adopt existing .collapse-btn, else build one) -------
    let collapseBtn = header.querySelector('.collapse-btn');
    if (!collapseBtn) {
        collapseBtn = doc.createElement('button');
        collapseBtn.type = 'button';
        collapseBtn.className = 'collapse-btn';
        collapseBtn.title = 'Collapse';
        collapseBtn.setAttribute('aria-label', 'Collapse agents pane');
        collapseBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
            ? window.kicon('chevron-left')
            : '<span class="ki ki-chevron-left" aria-hidden="true"></span>';
        header.appendChild(collapseBtn);
    }

    // --- List body (adopt existing #agents-list / .pane-content) -----------
    let body = paneEl.querySelector('#agents-list')
        || paneEl.querySelector('.agent-list-pane-body');
    if (!body) {
        body = doc.createElement('div');
        body.className = 'pane-content agent-list-pane-body';
        // Insert before any existing resize handle so the handle stays last.
        const existingHandle = paneEl.querySelector('.resize-handle');
        if (existingHandle) paneEl.insertBefore(body, existingHandle);
        else paneEl.appendChild(body);
    }

    // A BUILT header sits directly above the body, not pinned to
    // paneEl.firstChild. When the pane carries foreign leading children — e.g.
    // an embedder nests its own chrome (a user panel) INSIDE the pane so it
    // collapses together with the list — pinning the header to the very top
    // would sandwich that chrome between the header and the list, splitting the
    // component's two-part unit. Placing the built header adjacent to the body
    // keeps header + list contiguous below any pre-existing chrome. Adopted
    // headers (the console's static #agents-pane) are never repositioned — their
    // placement is the host's to own.
    // The `body.parentNode === paneEl` guard matters because `body` may have
    // been ADOPTED via a descendant querySelector (an existing #agents-list can
    // be nested inside foreign chrome, not a direct pane child); insertBefore
    // requires a direct child, so in that unusual shape we leave the built
    // header where it is rather than throw.
    if (builtHeader && body.parentNode === paneEl && header.nextSibling !== body) {
        paneEl.insertBefore(header, body);
    }

    // --- Resize handle (adopt existing .resize-handle, else build one) ------
    let resizeHandle = paneEl.querySelector('.resize-handle');
    let builtResizeHandle = false;
    if (!resizeHandle) {
        resizeHandle = doc.createElement('div');
        resizeHandle.className = 'resize-handle agent-list-resize-handle';
        paneEl.appendChild(resizeHandle);
        builtResizeHandle = true;
    }

    // --- Mount the shared list surface into the body -----------------------
    let loadedItems = [];
    // Set once and never cleared. A failed list fetch must not permanently
    // retire the fleet Stop control: its authority comes from the host status
    // endpoint, not from the agent list, so once the pane has loaded at all,
    // polling stays eligible and a later successful status read re-enables it.
    let listEverLoaded = false;
    let stopAllPending = false;
    let refreshStopAllState = async () => false;
    let invalidateStopAllState = () => {};
    const listHandle = mountAgentList(body, {
        api: config.api,
        adapter: config.adapter,
        renderCard: config.renderCard,
        showStatusDot: config.showStatusDot,
        isThinking: config.isThinking,
        onStop: config.onStop,
        onSelect: config.onSelect,
        onLoaded: (items, meta) => {
            loadedItems = Array.isArray(items) ? items : [];
            listEverLoaded = true;
            if (!stopAllPending && !containerEl[AGENT_LIST_STOP_ALL_OPERATION]) {
                void refreshStopAllState();
            }
            if (typeof config.onLoaded === 'function') config.onLoaded(items, meta);
        },
        onError: (error) => {
            invalidateStopAllState();
            if (typeof config.onError === 'function') config.onError(error);
        },
        autoLoad: config.autoLoad,
        autoSelectFirst: config.autoSelectFirst,
        selectedName: config.selectedName,
        escapeHtml: config.escapeHtml,
        emptyText: config.emptyText,
        errorText: config.errorText,
        askHoldReason: config.askHoldReason,
        holdStatusIntervalMs: config.holdStatusIntervalMs,
    });

    // --- Fleet cooperative Stop (andon cord) -------------------------------
    // This control belongs to the shared pane, not identity.js, so every host
    // that adopts or embeds mountAgentListPane receives the same behavior. The
    // server owns target resolution and fan-out; this component submits one
    // host request and renders every typed result without collapsing partial
    // refusal/unreachability into success.
    const stopAllOptIn = typeof config.onPrepareStopAll === 'function'
        && api && typeof api.getHostStopStatus === 'function'
        && typeof api.stopHost === 'function';
    let stopAllBtn = header.querySelector('.agent-stop-all-btn');
    const builtStopAllBtn = stopAllOptIn && !stopAllBtn;
    if (stopAllOptIn && !stopAllBtn) {
        stopAllBtn = doc.createElement('button');
        stopAllBtn.type = 'button';
        stopAllBtn.className = 'agent-stop-all-btn';
        stopAllBtn.textContent = 'Stop all';
        // collapseBtn came from header.querySelector, a DESCENDANT query, so
        // an adopted header may nest its chevron in a wrapper. insertBefore on
        // `header` then throws NotFoundError and aborts the whole mount --
        // before the "+ New" wiring, the collapse/resize listeners, and the
        // pane owner handle. Insert relative to the button's own parent.
        (collapseBtn.parentNode || header).insertBefore(stopAllBtn, collapseBtn);
    }
    if (stopAllBtn) {
        stopAllBtn.hidden = !stopAllOptIn;
        stopAllBtn.disabled = !stopAllOptIn;
        stopAllBtn.title = 'Cooperatively stop all in-flight agent work';
        stopAllBtn.setAttribute('aria-label', 'Stop all in-flight agents');
    }

    const stopAllResults = stopAllOptIn ? doc.createElement('div') : null;
    if (stopAllResults) {
        stopAllResults.className = 'agent-stop-all-results';
        stopAllResults.setAttribute('role', 'status');
        stopAllResults.setAttribute('aria-live', 'polite');
        stopAllResults.hidden = true;
        body.insertBefore(stopAllResults, listHandle.element);
    }

    let stopAllStatusSeq = 0;
    let stopAllStatusPromise = null;
    let stopAllStatus = { loaded: false, canStop: false, inFlightCount: 0 };
    function renderStopAllState() {
        if (!stopAllOptIn || !stopAllBtn) return;
        const { loaded, canStop, inFlightCount } = stopAllStatus;
        const retryPending = typeof containerEl[AGENT_LIST_STOP_ALL_RETRY] === 'string';
        stopAllBtn.disabled = stopAllPending
            || Boolean(containerEl[AGENT_LIST_STOP_ALL_OPERATION])
            || !listEverLoaded || !loaded || !canStop
            || (inFlightCount === 0 && !retryPending);
        stopAllBtn.dataset.inFlightCount = String(inFlightCount);
        if (!loaded) {
            stopAllBtn.title = 'Checking cooperative Stop availability';
        } else if (!canStop) {
            stopAllBtn.title = 'Sovereign host authority is required to Stop all agents';
        } else if (retryPending && inFlightCount === 0) {
            stopAllBtn.title = 'Recover the durable result of the prior Stop All request';
        } else if (inFlightCount > 0) {
            stopAllBtn.title = `Cooperatively stop ${inFlightCount} in-flight agent${inFlightCount === 1 ? '' : 's'}`;
        } else {
            stopAllBtn.title = 'No agent work is currently in flight';
        }
    }
    invalidateStopAllState = () => {
        stopAllStatus = { loaded: false, canStop: false, inFlightCount: 0 };
        renderStopAllState();
    };
    refreshStopAllState = async () => {
        if (!stopAllOptIn || !listEverLoaded) return false;
        if (stopAllStatusPromise) return stopAllStatusPromise;
        const request = (async () => {
            const seq = ++stopAllStatusSeq;
            try {
                const status = await api.getHostStopStatus();
                if (seq !== stopAllStatusSeq) return false;
                const rawCount = status && status.in_flight_count;
                const validCount = Number.isSafeInteger(rawCount) && rawCount >= 0;
                stopAllStatus = {
                    loaded: true,
                    canStop: status && status.can_stop === true,
                    inFlightCount: validCount ? rawCount : 0,
                };
            } catch (_) {
                if (seq !== stopAllStatusSeq) return false;
                // Status is an authority and inventory gate. A failed read must not
                // fall back to browser-local cards or expose a knowingly doomed
                // control to an unauthorized caller.
                stopAllStatus = { loaded: true, canStop: false, inFlightCount: 0 };
            }
            renderStopAllState();
            return stopAllStatus.canStop;
        })();
        stopAllStatusPromise = request;
        try {
            return await request;
        } finally {
            if (stopAllStatusPromise === request) stopAllStatusPromise = null;
        }
    };

    function displayTarget(outcome) {
        const ids = [outcome && outcome.agent_id, outcome && outcome.resolved_target]
            .filter((value) => typeof value === 'string' && value);
        const item = loadedItems.find((candidate) => {
            if (!candidate) return false;
            const candidateIds = [
                candidate.id,
                candidate.name,
                candidate.raw && candidate.raw.did,
                candidate.raw && candidate.raw.id,
            ];
            return candidateIds.some((value) => ids.includes(value));
        });
        return (item && (item.displayName || item.name)) || ids[0] || 'Unknown target';
    }

    function renderStopAllOutcomes(response, expectedCorrelationId) {
        const evidence = validateHostStopEnvelope(response, expectedCorrelationId);
        stopAllResults.hidden = false;
        stopAllResults.textContent = '';
        if (!evidence) {
            stopAllResults.textContent = 'Cooperative Stop evidence was malformed or incomplete; outcome is indeterminate.';
            return;
        }

        const counts = new Map();
        const list = doc.createElement('ul');
        for (const outcome of evidence.outcomes) {
            const disposition = outcome.disposition;
            counts.set(disposition, (counts.get(disposition) || 0) + 1);
            const row = doc.createElement('li');
            row.dataset.disposition = disposition;
            row.textContent = `${displayTarget(outcome)}: ${disposition}`;
            if (outcome && typeof outcome.detail === 'string' && outcome.detail) {
                row.textContent += ` — ${outcome.detail}`;
            }
            list.appendChild(row);
        }
        const summary = doc.createElement('p');
        summary.textContent = `Stop All results: ${Array.from(counts.entries())
            .map(([name, count]) => `${count} ${name}`)
            .join(', ')}.`;
        stopAllResults.appendChild(summary);
        stopAllResults.appendChild(list);
    }

    const confirmStopAll = typeof config.confirmStopAll === 'function'
        ? config.confirmStopAll
        : ((message) => (
            typeof window !== 'undefined' && typeof window.confirm === 'function'
                ? window.confirm(message)
                : false
        ));
    const onStopAllClick = async () => {
        if (!stopAllOptIn || !listEverLoaded || stopAllPending
            || containerEl[AGENT_LIST_STOP_ALL_OPERATION]) return;
        const count = stopAllStatus.inFlightCount;
        const retryCorrelationId = typeof containerEl[AGENT_LIST_STOP_ALL_RETRY] === 'string'
            ? containerEl[AGENT_LIST_STOP_ALL_RETRY]
            : null;
        const recoverBeforeFresh = retryCorrelationId !== null && count > 0;
        const operation = {
            correlationId: retryCorrelationId && !recoverBeforeFresh
                ? retryCorrelationId
                : newStopAllCorrelationId(),
            recoveryCorrelationId: recoverBeforeFresh ? retryCorrelationId : null,
        };
        const retryPending = retryCorrelationId !== null;
        const noun = count === 1 ? 'agent' : 'agents';
        const confirmation = retryPending && count === 0
            ? 'Recover the durable result of the prior Stop All request?'
            : `Stop all ${count} in-flight ${noun}?`;
        if (!confirmStopAll(confirmation)) return;

        containerEl[AGENT_LIST_STOP_ALL_OPERATION] = operation;
        stopAllPending = true;
        renderStopAllState();
        // Browser-owned queues and streams must be fenced synchronously before
        // even the fresh status read can yield. The callback may return a
        // settlement hook that reconciles locally-addressed streams with the
        // typed host outcomes.
        let settleLocalStop = null;
        let response = null;
        let stopError = null;
        try {
            settleLocalStop = config.onPrepareStopAll(loadedItems);
            containerEl[AGENT_LIST_STOP_ALL_RETRY] = operation.correlationId;
            if (operation.recoveryCorrelationId) {
                // A prior response may have been lost. Its immutable operation
                // must be recovered without settling the fence for work that
                // appeared afterwards; a fresh operation below addresses that
                // current work. Recovery failure cannot safely substitute its
                // stale identity for the new Stop attempt.
                try {
                    const recovery = api.stopHost({
                        reason: config.stopAllReason || 'Stopped from the agents banner',
                        correlation_id: operation.recoveryCorrelationId,
                    });
                    if (recovery && typeof recovery.catch === 'function') {
                        void recovery.catch(() => {});
                    }
                } catch (_) { /* the fresh operation remains authoritative */ }
            }
            response = await api.stopHost({
                reason: config.stopAllReason || 'Stopped from the agents banner',
                correlation_id: operation.correlationId,
            });
            // A response, even malformed or unreceipted, is terminal for this
            // attempt. Only a transport-ambiguous failure replays one durable
            // correlation; a terminal refusal/conflict gets a fresh operation.
            delete containerEl[AGENT_LIST_STOP_ALL_RETRY];
            if (!destroyed && containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
                renderStopAllOutcomes(response, operation.correlationId);
            }
            if (!destroyed && containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation
                && typeof config.onStopAllOutcomes === 'function') {
                config.onStopAllOutcomes(response);
            }
        } catch (error) {
            stopError = error;
            if (Number.isSafeInteger(error && error.status)
                && error.status >= 400 && error.status < 500) {
                delete containerEl[AGENT_LIST_STOP_ALL_RETRY];
            }
            if (!destroyed && containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
                stopAllResults.hidden = false;
                stopAllResults.textContent = `Stop All failed: ${error && error.message ? error.message : 'request failed'}`;
            }
        } finally {
            if (typeof settleLocalStop === 'function') {
                try {
                    settleLocalStop(response, stopError, operation.correlationId);
                } catch (error) {
                    if (!destroyed
                        && containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
                        stopAllResults.hidden = false;
                        stopAllResults.textContent = `Local Stop settlement failed: ${error && error.message ? error.message : 'unknown error'}`;
                    }
                }
            }
            stopAllPending = false;
            if (containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
                delete containerEl[AGENT_LIST_STOP_ALL_OPERATION];
            }
            const currentOwner = containerEl[AGENT_LIST_PANE_OWNER];
            if (!destroyed && currentOwner === handle) {
                await refreshStopAllState();
            } else if (currentOwner && typeof currentOwner.refreshStopAllState === 'function') {
                void currentOwner.refreshStopAllState();
            }
        }
    };
    if (stopAllOptIn) stopAllBtn.addEventListener('click', onStopAllClick);

    // Host work can originate in another tab, through the API, or from a
    // signal. Poll the authoritative read-only status instead of treating this
    // document's `.agent-thinking` classes as fleet truth.
    const statusIntervalMs = Number.isFinite(config.stopAllStatusIntervalMs)
        ? Math.max(250, config.stopAllStatusIntervalMs)
        : 2000;
    const setIntervalFn = doc.defaultView && doc.defaultView.setInterval;
    const clearIntervalFn = doc.defaultView && doc.defaultView.clearInterval;
    const statusInterval = stopAllOptIn && typeof setIntervalFn === 'function'
        ? setIntervalFn.call(doc.defaultView, () => {
            if (listEverLoaded && !stopAllPending
                && !containerEl[AGENT_LIST_STOP_ALL_OPERATION]) {
                void refreshStopAllState();
            }
        }, statusIntervalMs)
        : null;
    renderStopAllState();
    void refreshStopAllState();

    // --- "+ New" header action (adopt existing, else build) ----------------
    // Component-owned so embed hosts — which never run the console's
    // DOMContentLoaded wiring — still get it, and so the standalone console
    // GAINS a new-agent affordance it lacks today (same-everywhere rule). The
    // action is entirely host-defined via `onNew`, so the button is only
    // built/adopted when `onNew` is a function. The console's static
    // `#new-agent-sidebar-btn` (if present) is adopted in place; otherwise a
    // fresh `ki-plus` button is built. Same adopt-or-build pattern as the
    // conversations pane's New button.
    const hasNew = typeof config.onNew === 'function';
    let newBtn = null;
    let builtNewBtn = false;
    let onNewClick = null;
    if (hasNew) {
        newBtn = header.querySelector('#new-agent-sidebar-btn')
            || header.querySelector('.new-agent-btn');
        if (!newBtn) {
            newBtn = doc.createElement('button');
            newBtn.type = 'button';
            newBtn.className = 'new-agent-btn btn-icon';
            newBtn.title = config.newLabel || 'New Agent';
            newBtn.setAttribute('aria-label', config.newLabel || 'New agent');
            newBtn.innerHTML = (typeof window !== 'undefined' && typeof window.kicon === 'function')
                ? window.kicon('plus')
                : '<span class="ki ki-plus" aria-hidden="true"></span>';
            // Sit just after the title, before the collapse chevron.
            const titleEl = header.querySelector('.agent-list-pane-title')
                || header.querySelector('h3');
            if (titleEl && titleEl.nextSibling) header.insertBefore(newBtn, titleEl.nextSibling);
            else header.insertBefore(newBtn, header.firstChild);
            builtNewBtn = true;
        }
        onNewClick = () => { config.onNew(); };
        newBtn.addEventListener('click', onNewClick);
    }

    // ---- Collapse state ---------------------------------------------------
    // #2216 two-state: open (full pane) and fully HIDDEN (`display:none`, zero
    // width — NO collapsed chevron rail). The `.collapsed` class is kept in
    // lock-step with `display:none` as the single state marker. Unlike the
    // conversations pane, the agents pane is the primary navigation surface, so
    // its first-run default is OPEN (a persisted value or explicit
    // `config.collapsed` still wins).
    function isCollapsed() {
        return !!(paneEl.classList && paneEl.classList.contains('collapsed'));
    }
    function applyCollapsed(collapsed, persist) {
        if (!paneEl.classList) return;
        paneEl.classList.toggle('collapsed', collapsed);
        paneEl.style.display = collapsed ? 'none' : '';
        if (persist) storeSet(KEY_COLLAPSED, collapsed ? '1' : '0');
        if (typeof config.onToggle === 'function') config.onToggle(collapsed);
    }
    function open() { if (isCollapsed()) applyCollapsed(false, true); }
    function close() { if (!isCollapsed()) applyCollapsed(true, true); }
    function toggle() { applyCollapsed(!isCollapsed(), true); }

    // The chevron `<` CLOSES the pane (fully hides it); a host trigger reopens.
    const onCollapseClick = () => close();
    collapseBtn.addEventListener('click', onCollapseClick);

    // Initial state: a persisted value wins; otherwise `config.collapsed`;
    // otherwise OPEN. Always applied so the pane's `display` reflects the state
    // from mount, whichever branch wins.
    const persistedCollapsed = storeGet(KEY_COLLAPSED);
    const startCollapsed = persistedCollapsed !== null
        ? persistedCollapsed === '1'
        : (config.collapsed !== undefined ? !!config.collapsed : false);
    applyCollapsed(startCollapsed, false);

    // ---- Resize (min/max + persistence) -----------------------------------
    const persistedWidth = parseInt(storeGet(KEY_WIDTH), 10);
    if (Number.isFinite(persistedWidth)) {
        paneEl.style.width = `${Math.max(minWidth, Math.min(maxWidth, persistedWidth))}px`;
    }
    let startX = 0;
    let startWidth = 0;
    function onMouseMove(e) {
        const diff = e.clientX - startX;
        const w = Math.max(minWidth, Math.min(maxWidth, startWidth + diff));
        paneEl.style.width = `${w}px`;
    }
    function onMouseUp() {
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseup', onMouseUp);
        if (doc.body) {
            doc.body.style.cursor = '';
            doc.body.style.userSelect = '';
        }
        storeSet(KEY_WIDTH, String(paneEl.offsetWidth || parseInt(paneEl.style.width, 10) || startWidth));
    }
    function onResizeDown(e) {
        startX = e.clientX;
        startWidth = paneEl.offsetWidth || parseInt(paneEl.style.width, 10) || minWidth;
        if (doc.body) {
            doc.body.style.cursor = 'col-resize';
            doc.body.style.userSelect = 'none';
        }
        doc.addEventListener('mousemove', onMouseMove);
        doc.addEventListener('mouseup', onMouseUp);
    }
    resizeHandle.addEventListener('mousedown', onResizeDown);

    let destroyed = false;
    let handle = null;
    function destroy() {
        if (destroyed) return;
        destroyed = true;
        if (paneEl[AGENT_LIST_PANE_OWNER] === handle) {
            delete paneEl[AGENT_LIST_PANE_OWNER];
        }
        collapseBtn.removeEventListener('click', onCollapseClick);
        if (stopAllBtn) stopAllBtn.removeEventListener('click', onStopAllClick);
        stopAllStatusSeq++;
        if (statusInterval !== null && typeof clearIntervalFn === 'function') {
            clearIntervalFn.call(doc.defaultView, statusInterval);
        }
        if (newBtn && onNewClick) newBtn.removeEventListener('click', onNewClick);
        resizeHandle.removeEventListener('mousedown', onResizeDown);
        doc.removeEventListener('mousemove', onMouseMove);
        doc.removeEventListener('mouseup', onMouseUp);
        try { listHandle.destroy(); } catch (_) { /* best-effort */ }
        // Remove only chrome this mount built; adopted chrome is left in place.
        if (builtHeader && header.parentNode) header.parentNode.removeChild(header);
        if (builtNewBtn && newBtn) newBtn.remove();
        if (builtStopAllBtn && stopAllBtn) stopAllBtn.remove();
        if (stopAllResults && stopAllResults.parentNode) stopAllResults.remove();
        // The built resize handle too (codex P2): a leaked absolutely-positioned
        // .resize-handle overlays the container edge and gets ADOPTED by the
        // next mount into the same container, doubling listeners over time.
        if (builtResizeHandle && resizeHandle) resizeHandle.remove();
    }

    handle = {
        element: paneEl,
        list: listHandle,
        refresh: (...a) => listHandle.refresh(...a),
        refreshHoldState: (...a) => listHandle.refreshHoldState(...a),
        select: (...a) => listHandle.select(...a),
        setActiveName: (...a) => listHandle.setActiveName(...a),
        getActive: (...a) => listHandle.getActive(...a),
        open,
        close,
        toggle,
        refreshStopAllState,
        get collapsed() { return isCollapsed(); },
        destroy,
    };
    paneEl[AGENT_LIST_PANE_OWNER] = handle;
    return handle;
}
