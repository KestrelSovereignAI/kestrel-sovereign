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
// A fleet Hold gesture outlives the mount that started it: "Stop all and hold"
// awaits a mutation, then a confirming read, and only then stops. A host may
// remount the pane across any of those awaits, so the in-flight operation is
// carried on the CONTAINER — the one thing both mounts share — rather than in
// a closure the retired mount takes with it. That is what lets a replacement
// mount inherit the fence, and what lets the retired continuation discover it
// no longer owns the gesture (#3165).
//
// "Stop all and hold" is ONE operator action that spans TWO lanes, and the two
// symbols above are per-lane. A gesture that takes them one at a time has a gap
// between the two claims: another surface takes the Stop lane while the Hold
// half is still awaiting, and the Stop half then either declines silently — a
// gesture that reports a Hold and never stops — or runs a second Stop behind
// the competitor with a stale in-flight count. So the compound gesture takes
// ONE reservation covering both lanes, synchronously, before its first request
// leaves, and releases both exactly once when the whole gesture settles: the
// same operation object is written into both slots, and `endFleetHoldGesture`
// is the only place either is cleared.
const AGENT_LIST_FLEET_HOLD_OPERATION = Symbol.for('kestrel.agentListPane.fleetHoldOperation');

function newOperationId(prefix) {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
        return `${prefix}:${globalThis.crypto.randomUUID()}`;
    }
    return `${prefix}:${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

function newStopAllCorrelationId() {
    return newOperationId('ui-host-stop');
}

// A Hold and a Resume are receipted governance acts, so both carry an operator
// reason. A blank or cancelled answer aborts rather than silently substituting
// one. Card-scoped and fleet-scoped Hold share this asker so the two cannot
// drift into different ideas of what counts as a reason.
function makeReasonAsker(override) {
    const ask = typeof override === 'function'
        ? override
        : ((message, defaultValue) => (
            typeof window !== 'undefined' && typeof window.prompt === 'function'
                ? window.prompt(message, defaultValue)
                : null
        ));
    return (message, defaultValue) => {
        const answer = ask(message, defaultValue);
        return typeof answer === 'string' && answer.trim() ? answer.trim() : null;
    };
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
            if (current !== undefined) ctx.applyLatch(entry.agent_id, current);
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
        // The same document every sibling node above is built in. An embedding
        // host mounts into ITS document, so a kebab (and the menu it opens)
        // built in the console's would belong to a tree that host never shows.
        ownerDocument: doc,
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
        // Disabled FIRST: an accelerator may only take the browser's own menu
        // when it has one of ours to put there. A non-sovereign caller, a
        // stale reading, or a card the host's inventory does not name all
        // leave the kebab disabled — suppressing the native menu for those
        // turns a right-click into a gesture that does nothing at all.
        if (kebabBtn.disabled) return;
        if (typeof event.preventDefault === 'function') event.preventDefault();
        openMenuAt(menuItems(), positionFromEvent(event), { ownerDocument: doc });
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
 *   - `onHoldState(snapshot)` — republishes the host's latch table after every
 *     read and every committed mutation, so a surface OUTSIDE the list (the
 *     pane's banner held count, #3165) renders host verdicts rather than
 *     composing "held" a second time.
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
    const askReason = makeReasonAsker(config.askHoldReason);
    // The banner's held count and fleet menu are drawn by the PANE, which owns
    // no latch state of its own — it renders what the host said, republished
    // here after every read and after every mutation this document committed.
    const onHoldState = typeof config.onHoldState === 'function' ? config.onHoldState : null;
    // The browser never derives "held" — the host composes the two
    // independent latches and says so. Until a read succeeds the controls
    // stay inert rather than drawing an agent as un-held.
    //
    // `stale` and `composed` are deliberately two fields, because they are two
    // different failures of confirmation and only one of them is visible as a
    // failure. `stale` says a READ did not arrive. `composed` says this
    // document painted a committed mutation over the last reading before any
    // read confirmed it — which leaves `stale` false while the table is no
    // longer purely the host's. A surface that presents entries as EVIDENCE
    // (the fleet fan-out, #3165) needs both to be clear; one that merely
    // renders the best current knowledge (a badge) needs only `stale`.
    let holdState = {
        loaded: false,
        stale: false,
        composed: false,
        canHold: false,
        hostHold: null,
        byAgent: new Map(),
    };
    let holdSeq = 0;
    let holdPromise = null;
    let holdPromiseSeq = 0;
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

    // The public projection of the host's latch table: the entries the host
    // itself named, its own composed `held` verdict for each, and the tallies a
    // banner renders. The count is a SUM of host verdicts, never a second
    // composition rule — the browser derives "held" in exactly one place
    // (applyLatch / applyHostLatch, for the mutation it just committed) and
    // nowhere else.
    //
    // `confirmed` is the load-bearing field for any consumer presenting these
    // entries as evidence: it is true only when every entry here came from ONE
    // host read that this document has not painted over. A false `confirmed`
    // does not mean the entries are wrong — it means they are this document's
    // best guess at a table only the host can state, and the fleet membership
    // or an agent's own latch may have moved since the read they were composed
    // from.
    function holdSnapshot() {
        const agents = Array.from(holdState.byAgent.values());
        return {
            supported: holdSupported === true,
            loaded: holdState.loaded === true,
            stale: holdState.stale === true,
            composed: holdState.composed === true,
            confirmed: holdState.loaded === true
                && holdState.stale !== true
                && holdState.composed !== true,
            canHold: holdState.canHold === true,
            hostHold: holdState.hostHold || null,
            agents,
            targetCount: agents.length,
            heldCount: agents.filter((entry) => entry && entry.held === true).length,
        };
    }

    function publishHoldState() {
        if (!onHoldState) return;
        // A host callback is not allowed to break the list it is watching.
        try { onHoldState(holdSnapshot()); } catch (_) { /* best-effort */ }
    }

    // The host says there is no Hold door. Take the surface off every card,
    // stop asking, and build no more of it — an operator must not be left a
    // permanently disabled control and a poll against a route that 404s.
    function retireHoldSurface() {
        if (holdSupported === false) return;
        holdSupported = false;
        holdSeq++; // orphan any in-flight read
        holdState = {
            loaded: true,
            stale: false,
            composed: false,
            canHold: false,
            hostHold: null,
            byAgent: new Map(),
        };
        stopHoldPolling();
        for (const view of holdViews) detachHoldView(view);
        holdViews.length = 0;
        // The banner hangs off the same answer: a host with no Hold door must
        // lose its fleet menu and held count too, not keep a dead one.
        publishHoldState();
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
        publishHoldState();
    }

    // The stored read is the join target for every coalescing caller, and it is
    // cleared when it SETTLES — which is precisely what a read that never
    // settles does not do. `api.getHostHoldState()` hands the browser's fetch
    // no signal, so a host that accepts the connection and never answers leaves
    // one promise standing in front of every later poll for the life of the
    // mount: each tick joins a read that cannot land, no request is issued, and
    // the badge's count and Hold authority never recover even once the host
    // does.
    //
    // A caller that has stopped waiting retires it here. The request is
    // deliberately left in flight — a late answer still converges through the
    // sequence fence, which is strictly more than aborting would leave behind —
    // but it stops standing in for a read nobody is going to get, so the next
    // poll issues one. The check is by identity: a caller giving up on ITS read
    // must not retire a later read that is still doing its job.
    //
    // Clearing the promise IS the whole retirement. `holdPromiseSeq` is read
    // only alongside a stored promise, so it goes stale here exactly as it does
    // when a read settles normally below — one rule for when that fence means
    // anything, rather than a second write that cannot be observed.
    function makeHoldReadAbandon(request) {
        return () => {
            if (holdPromise !== request) return false;
            holdPromise = null;
            return true;
        };
    }

    /**
     * Read the host's latch table.
     *
     * `{ fresh: true }` demands a read that STARTS now. Plain coalescing is
     * wrong for a caller whose own evidence the read has to postdate: an
     * in-flight read may have been issued before the mutation that caller just
     * committed, and joining it reports a confirmation that never happened.
     * Worse, a mutation orphans an in-flight read by sequence, so joining one
     * is frequently joining a promise that resolves to nothing at all — which
     * is what left the post-mutation refresh a no-op until the next poll tick.
     *
     * `{ onReadStarted }` is handed a function that retires THIS read as the
     * join target, for a caller that bounds how long it will wait for it. It is
     * called only when this call actually starts a read, never when it joins
     * one: a joiner did not start the read and may not retire it.
     */
    async function refreshHoldState(options = {}) {
        if (!holdCallable || holdSupported === false) return false;
        const fresh = !!(options && options.fresh === true);
        // Join only a read that can still land, and only when the caller did
        // not ask for one strictly newer than itself.
        if (holdPromise && !fresh && holdPromiseSeq === holdSeq) return holdPromise;
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
                // One read, wholesale: the entries, the host latch and the
                // membership are all the host's, so nothing this document
                // composed earlier survives into the table. That is what makes
                // the snapshot presentable as evidence again.
                holdState = {
                    loaded: true,
                    stale: false,
                    composed: false,
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
        // The IIFE above takes its sequence synchronously, so this is THIS
        // read's fence. A later mutation moving `holdSeq` past it is precisely
        // how a joiner learns the in-flight read can no longer land.
        holdPromiseSeq = holdSeq;
        if (options && typeof options.onReadStarted === 'function') {
            // Synchronous, and before the first await, so a caller that arms a
            // leash on the next line already holds the key to this read.
            options.onReadStarted(makeHoldReadAbandon(request));
        }
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
        // A read that started BEFORE this mutation carries a pre-mutation
        // reading of the very latch just committed, so letting it land would
        // undo a committed fact with a stale one. Orphan it — the same
        // sequence fence retireHoldSurface uses. Reads started after this
        // point take a higher sequence and are still free to confirm.
        holdSeq++;
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
        // `composed`: a paint, not a reading. The badge may render it (that is
        // what it is for), but nothing may present it as the host's own answer
        // until a read confirms the whole table.
        holdState = { ...holdState, loaded: true, composed: true, byAgent };
        renderHoldState();
    }

    // The same rule on the OTHER independent axis, for a host-scope mutation
    // the banner committed. Every entry's `held`/`sources` is recomposed from
    // (host latch, that entry's own agent latch) — which is why releasing the
    // host latch here leaves an agent someone held individually still held,
    // exactly as EffectiveHoldState leaves it server-side. Nothing about an
    // agent's own latch is read from, or written by, the host mutation.
    //
    // What this CANNOT do is learn anything the last read did not contain. The
    // membership it iterates, and every per-agent latch it carries forward, are
    // as old as that read; if an agent joined the fleet or somebody held one
    // individually since, this recomposition is confidently wrong about them.
    // That is why it marks the table `composed` — see holdSnapshot.
    function applyHostLatch(latch) {
        if (latch !== null && (typeof latch !== 'object' || Array.isArray(latch))) return;
        holdSeq++; // orphan a read that started before this mutation committed
        const hostHold = latch || null;
        const byAgent = new Map();
        for (const [agentId, previous] of holdState.byAgent) {
            const agentHold = (previous && previous.agent_hold) || null;
            const sources = [];
            if (hostHold) sources.push('host');
            if (agentHold) sources.push('agent');
            byAgent.set(agentId, {
                ...(previous || {}),
                agent_id: agentId,
                held: sources.length > 0,
                sources,
                agent_hold: agentHold,
            });
        }
        holdState = { ...holdState, loaded: true, composed: true, hostHold, byAgent };
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
                // A card's post-mutation confirmation is subject to exactly the
                // same rule as the banner's: it has to be a read that started
                // after the latch this card just committed.
                refreshHoldState: () => refreshHoldState({ fresh: true }),
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
    publishHoldState();
    void refreshHoldState();

    return {
        element: root,
        refresh,
        refreshHoldState,
        getHoldState: holdSnapshot,
        applyHostLatch,
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
 *     onLoaded, onError, onHoldState, autoLoad, autoSelectFirst, selectedName,
 *     escapeHtml, emptyText, errorText, hold, holdStatusIntervalMs,
 *     askHoldReason —
 *     forwarded verbatim to `mountAgentList`. This list is the pane's whole
 *     forwarding surface, so an option `mountAgentList` reads and this call
 *     site omits is silently ignored — the shape that dropped `hold` (#3164).
 *     The drift test in tests/frontend/agent_hold_controls.test.mjs holds the
 *     two lists together.
 *   - onNew()          — the "+ New" header action (Add-a-Companion / new agent).
 *                        The New button is only built/adopted when this is a fn.
 *   - onPrepareStopAll(items) — REQUIRED to enable Stop All: synchronous
 *                        browser-work fence returning an optional settlement
 *                        callback invoked with (response, error, correlationId).
 *   - confirmStopAll(message) — host confirmation override (defaults to confirm).
 *   - stopAllStatusIntervalMs — authoritative host-status refresh cadence.
 *   - stopAllReason / fleetHoldReason — wording carried on the fleet requests.
 *   - fleetHoldConfirmTimeoutMs — how long a fleet gesture waits for the
 *                        confirming latch read that draws its fan-out before
 *                        reporting that fan-out unconfirmed (default 5s). The
 *                        read has no timeout of its own, and presentation
 *                        evidence may not fence an operator control.
 *   - newLabel         — accessible label / tooltip for the New button.
 *   - collapsed        — initial collapsed state (overridden by persistence).
 *   - storageKey       — persistence namespace (default 'kestrel:agents-pane').
 *   - title            — pane header title (default 'Agents').
 *   - onToggle(bool)   — fired after every collapse/expand with the new state.
 *   - minWidth/maxWidth — resize clamps (default 200 / 500, matching the CSS).
 *
 * The banner also carries the FLEET Hold surface (#3165): a persistent held
 * count and a menu with "Hold all", "Stop all and hold", and the release of the
 * host latch. Both are pane-owned for the same reason Stop All is — the pane is
 * what the console and Frinz's companions pane both mount, so one
 * implementation serves two products.
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

    // Declared here, not beside `destroy()` at the foot of the mount, because
    // the inner list publishes its Hold state SYNCHRONOUSLY from inside
    // `mountAgentList` below — a guard reading `destroyed` from that callback
    // would hit the temporal dead zone and throw during mount. Every
    // lifecycle-fenced continuation in this mount reads these two.
    let destroyed = false;
    let handle = null;

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
    // The banner's own view of the host latch table. It is never composed here:
    // every field arrives from the inner list's republication of what the host
    // said (`onHoldState`). Until the host answers, the banner knows nothing —
    // which is a distinct state from "nothing is held".
    let holdBanner = {
        supported: false,
        loaded: false,
        stale: false,
        canHold: false,
        hostHold: null,
        agents: [],
        targetCount: 0,
        heldCount: 0,
    };
    // Assigned once the header chrome exists. The inner mount can publish
    // synchronously, before the banner nodes are built — the same stub shape
    // `refreshStopAllState` uses for `onLoaded`.
    let renderFleetHoldState = () => {};
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
        onHoldState: (snapshot) => {
            // A retired mount's list handle can still resolve a read it issued
            // before `destroy()` — the inner list orphans in-flight reads by
            // sequence but a request already past that fence still republishes.
            // Painting detached chrome would be harmless; telling the EMBEDDING
            // HOST that a destroyed pane has fleet state is not, because the
            // host has no way to tell which mount is speaking.
            if (destroyed) return;
            holdBanner = snapshot;
            renderFleetHoldState();
            if (typeof config.onHoldState === 'function') config.onHoldState(snapshot);
        },
        autoLoad: config.autoLoad,
        autoSelectFirst: config.autoSelectFirst,
        selectedName: config.selectedName,
        escapeHtml: config.escapeHtml,
        emptyText: config.emptyText,
        errorText: config.errorText,
        askHoldReason: config.askHoldReason,
        hold: config.hold,
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

    // --- Peer Stop circuit breaker (#3170) ---------------------------------
    // Repeated peer Stop against one agent is Hold through the back door, so
    // the host stops honoring it past a threshold and tells a human -- here.
    // The circuits arrive on the same host status read Stop All polls, and
    // only a sovereign caller is sent them; Reset is that caller's door.
    const circuitOptIn = stopAllOptIn && typeof api.resetPeerStopCircuit === 'function';
    const circuitBanner = circuitOptIn ? doc.createElement('div') : null;
    if (circuitBanner) {
        circuitBanner.className = 'agent-peer-stop-circuits';
        circuitBanner.setAttribute('role', 'status');
        circuitBanner.setAttribute('aria-live', 'polite');
        circuitBanner.hidden = true;
        body.insertBefore(circuitBanner, listHandle.element);
    }
    const askCircuitResetReason = makeReasonAsker(config.askCircuitResetReason);
    const circuitResetsPending = new Set();
    const circuitResetErrors = new Map();
    let peerStopCircuit = null;

    function renderPeerStopCircuits() {
        if (!circuitBanner) return;
        circuitBanner.textContent = '';
        const state = peerStopCircuit;
        if (!state || typeof state !== 'object') {
            circuitBanner.hidden = true;
            return;
        }
        if (state.available !== true) {
            // An unreadable breaker is not a clear one.
            const line = doc.createElement('p');
            line.className = 'agent-peer-stop-circuit-unavailable';
            line.textContent = 'Peer Stop circuit state is unavailable';
            circuitBanner.appendChild(line);
            circuitBanner.hidden = false;
            return;
        }
        const open = Array.isArray(state.open)
            ? state.open.filter((entry) => entry && typeof entry.target_agent_id === 'string'
                && entry.target_agent_id)
            : [];
        circuitBanner.hidden = open.length === 0;
        for (const entry of open) {
            const target = entry.target_agent_id;
            const name = displayIdentity([target]);
            const row = doc.createElement('div');
            row.className = 'agent-peer-stop-circuit';
            row.dataset.target = target;
            const label = doc.createElement('span');
            label.className = 'agent-peer-stop-circuit-label';
            label.textContent = `peer Stop circuit open: ${name}`;
            const count = Number.isSafeInteger(entry.admitted_count) ? entry.admitted_count : null;
            const windowSeconds = Number.isSafeInteger(entry.window_seconds) ? entry.window_seconds : null;
            if (count !== null && windowSeconds !== null) {
                label.title = `${count} honored peer Stops within ${windowSeconds}s; further peer Stops are refused`;
            }
            const reset = doc.createElement('button');
            reset.type = 'button';
            reset.className = 'agent-peer-stop-circuit-reset';
            reset.textContent = 'Reset';
            reset.title = `Reset the peer Stop circuit for ${name}`;
            reset.disabled = circuitResetsPending.has(target);
            reset.addEventListener('click', () => { void resetPeerStopCircuit(target, name); });
            row.appendChild(label);
            row.appendChild(reset);
            const error = circuitResetErrors.get(target);
            if (error) {
                const failure = doc.createElement('span');
                failure.className = 'agent-peer-stop-circuit-error';
                failure.textContent = error;
                row.appendChild(failure);
            }
            circuitBanner.appendChild(row);
        }
    }

    async function resetPeerStopCircuit(target, name) {
        if (!circuitOptIn || destroyed || circuitResetsPending.has(target)) return;
        const reason = askCircuitResetReason(
            `Reset the peer Stop circuit for ${name}? Peers will be able to stop it again. Reason:`,
            'Reviewed the repeated peer Stops',
        );
        if (reason === null) return;
        circuitResetsPending.add(target);
        circuitResetErrors.delete(target);
        renderPeerStopCircuits();
        try {
            await api.resetPeerStopCircuit({ target, reason });
        } catch (error) {
            circuitResetErrors.set(
                target,
                `Reset failed: ${(error && error.message) || 'request refused'}`,
            );
        } finally {
            circuitResetsPending.delete(target);
        }
        if (destroyed) return;
        renderPeerStopCircuits();
        await refreshStopAllState();
    }

    let stopAllStatusSeq = 0;
    let stopAllStatusPromise = null;
    let stopAllStatus = { loaded: false, canStop: false, inFlightCount: 0 };
    function renderStopAllState() {
        if (!stopAllOptIn || !stopAllBtn) return;
        const { loaded, canStop, inFlightCount } = stopAllStatus;
        const retryPending = typeof containerEl[AGENT_LIST_STOP_ALL_RETRY] === 'string';
        // A fleet gesture holding this lane is the same fence as a Stop All
        // holding it — one slot, read once — but it is a different SENTENCE, and
        // a control disabled for a reason the operator cannot read is the thing
        // a busy lane must not look like.
        const laneHolder = containerEl[AGENT_LIST_STOP_ALL_OPERATION];
        stopAllBtn.disabled = stopAllPending
            || Boolean(laneHolder)
            || !listEverLoaded || !loaded || !canStop
            || (inFlightCount === 0 && !retryPending);
        stopAllBtn.dataset.inFlightCount = String(inFlightCount);
        if (laneHolder && laneHolder.reservesStopAll === true) {
            stopAllBtn.title = 'A "Stop all and hold" gesture already owns this Stop';
        } else if (!loaded) {
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
                peerStopCircuit = status && status.peer_stop_circuit
                    ? status.peer_stop_circuit
                    : null;
            } catch (_) {
                if (seq !== stopAllStatusSeq) return false;
                // Status is an authority and inventory gate. A failed read must not
                // fall back to browser-local cards or expose a knowingly doomed
                // control to an unauthorized caller.
                stopAllStatus = { loaded: true, canStop: false, inFlightCount: 0 };
                // A circuit this caller was shown cannot be read now: that is
                // "unknown", never "cleared" and never the stale open list.
                if (peerStopCircuit) peerStopCircuit = { available: false, open: [] };
            }
            renderStopAllState();
            renderPeerStopCircuits();
            return stopAllStatus.canStop;
        })();
        stopAllStatusPromise = request;
        try {
            return await request;
        } finally {
            if (stopAllStatusPromise === request) stopAllStatusPromise = null;
        }
    };

    // A host record names an agent by identity; the card names it by the
    // display name an operator recognises. Both the Stop fan-out and the Hold
    // fan-out address agents by identity, so both read them back through here.
    function displayIdentity(ids) {
        const known = ids.filter((value) => typeof value === 'string' && value);
        const item = loadedItems.find((candidate) => {
            if (!candidate) return false;
            const candidateIds = [
                candidate.id,
                candidate.name,
                candidate.raw && candidate.raw.did,
                candidate.raw && candidate.raw.id,
            ];
            return candidateIds.some((value) => known.includes(value));
        });
        return (item && (item.displayName || item.name)) || known[0] || 'Unknown target';
    }

    function displayTarget(outcome) {
        return displayIdentity([
            outcome && outcome.agent_id,
            outcome && outcome.resolved_target,
        ]);
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
    // `confirm` is false for the compound "Stop all and hold" gesture, whose
    // single gate is the reason prompt the Hold already asked — the same shape
    // the card's "Stop and hold" uses. Everything else about the fan-out is
    // identical, so the two gestures cannot report a fan-out differently.
    // `onSettled` fires where this function releases its OWN operation token —
    // the moment the durable Stop is over, and deliberately before the
    // convergence read below, which is bookkeeping a hung host can stall
    // forever. A caller whose gesture wraps this one (the compound "Stop all
    // and hold") ends its fence there for exactly the same reason.
    //
    // `reservation` is that wrapping gesture's already-taken claim on this lane.
    // A compound gesture cannot claim the Stop lane HERE — by the time it gets
    // here its Hold has committed, and a lane taken in between is a race it can
    // only lose — so it takes both lanes up front and hands its reservation
    // down. Under one, this function neither re-claims the lane nor releases it:
    // one reservation, released once, by the gesture that took it.
    //
    // Returns TRUE only when it actually held the lane and ran a Stop to
    // settlement. A wrapping gesture may not read a decline as a Stop it made.
    async function runStopAll({ confirm = true, onSettled = null, reservation = null } = {}) {
        if (!stopAllOptIn || !listEverLoaded || stopAllPending) return false;
        // One sentence for both doors: this lane must hold nothing except the
        // reservation this call was handed. With no reservation that reads as
        // the familiar "the lane must be free" — which is what refuses a Stop
        // All clicked while a compound gesture holds it. With one, it also
        // admits that exact object and nothing else, so a gesture can never
        // stop the fleet under a claim it does not hold.
        const laneHolder = containerEl[AGENT_LIST_STOP_ALL_OPERATION];
        if (laneHolder && laneHolder !== reservation) return false;
        const count = stopAllStatus.inFlightCount;
        const retryCorrelationId = typeof containerEl[AGENT_LIST_STOP_ALL_RETRY] === 'string'
            ? containerEl[AGENT_LIST_STOP_ALL_RETRY]
            : null;
        const recoverBeforeFresh = retryCorrelationId !== null && count > 0;
        // ONE token holds this lane for the whole run — the wrapping gesture's
        // reservation when there is one, otherwise this call's own record — so
        // every ownership recheck below reads the same slot against the same
        // value whichever gesture is driving.
        const operation = reservation || {};
        operation.correlationId = retryCorrelationId && !recoverBeforeFresh
            ? retryCorrelationId
            : newStopAllCorrelationId();
        operation.recoveryCorrelationId = recoverBeforeFresh ? retryCorrelationId : null;
        const retryPending = retryCorrelationId !== null;
        const noun = count === 1 ? 'agent' : 'agents';
        const confirmation = retryPending && count === 0
            ? 'Recover the durable result of the prior Stop All request?'
            : `Stop all ${count} in-flight ${noun}?`;
        if (confirm && !confirmStopAll(confirmation)) return false;

        // A no-op re-assignment under a reservation: the slot already holds this
        // exact object, claimed before the gesture's first request.
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
            // A reservation spans two lanes and is released ONCE, by the gesture
            // that took it — `onSettled` below is where that happens. Clearing
            // half of it here would split one release across two places, which
            // is the shape that let the two lanes drift apart to begin with.
            if (!reservation && containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
                delete containerEl[AGENT_LIST_STOP_ALL_OPERATION];
            }
            if (typeof onSettled === 'function') {
                try { onSettled(); } catch (_) { /* a fence is not a reporting surface */ }
            }
            const currentOwner = containerEl[AGENT_LIST_PANE_OWNER];
            if (!destroyed && currentOwner === handle) {
                await refreshStopAllState();
            } else if (currentOwner && typeof currentOwner.refreshStopAllState === 'function') {
                void currentOwner.refreshStopAllState();
            }
        }
        return true;
    }
    const onStopAllClick = () => { void runStopAll({ confirm: true }); };
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

    // --- Fleet Hold: the banner latch surface (#3165) ----------------------
    // Hold is a LATCH, so the banner's job is different from Stop All's: Stop
    // All reports a momentary fan-out and goes quiet, while the held count is
    // a RESTING reading of durable state — it must still be there tomorrow,
    // after a reload, with nothing in flight. It is therefore drawn from the
    // host's own per-agent verdicts (republished by the list), never from this
    // document's memory of what somebody clicked.
    //
    // The menu's Hold is HOST-scope: one latch, every agent. That is what makes
    // its release safe to offer from a fleet control — a host resume releases
    // exactly the latch it set and leaves an agent someone held individually
    // still held, which the per-agent rows below say out loud.
    const askFleetReason = makeReasonAsker(config.askHoldReason);
    const fleetHoldReason = typeof config.fleetHoldReason === 'string'
        && config.fleetHoldReason.trim()
        ? config.fleetHoldReason.trim()
        : 'Held from the agents banner';
    // How long a fleet gesture will wait for the confirming latch read that
    // draws its fan-out before reporting that fan-out unconfirmed. See
    // `confirmFleetProjection` for why a bound is required rather than tuned.
    const fleetConfirmTimeoutMs = Number.isFinite(config.fleetHoldConfirmTimeoutMs)
        ? Math.max(0, config.fleetHoldConfirmTimeoutMs)
        : 5000;
    // Every leash still outstanding, so destroy() can settle them. A retired
    // mount has no business waiting for evidence it may no longer paint, and a
    // timer nobody cancels keeps its host's event loop alive.
    const fleetConfirmWaits = new Set();

    // Pending is READ from the container, never stored beside it. A boolean in
    // this closure answers "did THIS mount start a fleet gesture", and that is
    // the wrong question in both directions: a replacement mount would answer
    // "no" while a retired continuation is still mid-gesture and unfence its
    // controls, and the retired mount would answer "yes" forever if its own
    // `finally` were skipped. One container-scoped operation object answers the
    // question every mount actually has — "is a fleet gesture in flight here".
    function fleetHoldIsPending() {
        return Boolean(containerEl[AGENT_LIST_FLEET_HOLD_OPERATION]);
    }
    // The Stop-All lane, read once, in the one place both the menu's offer and
    // the compound gesture's claim consult it. Free means nothing holds it: not
    // a bare Stop All, and not another fleet reservation.
    function fleetStopLaneIsFree() {
        return stopAllOptIn && !containerEl[AGENT_LIST_STOP_ALL_OPERATION];
    }
    // ONE reservation, every lane the gesture will touch, taken synchronously
    // before its first request leaves. `reservesStopAll` is the record of which
    // lanes it covers, so the release below and the Stop-All button's own
    // rendering read the same fact rather than re-deriving it.
    function claimFleetGesture(operation) {
        containerEl[AGENT_LIST_FLEET_HOLD_OPERATION] = operation;
        if (operation.reservesStopAll) {
            containerEl[AGENT_LIST_STOP_ALL_OPERATION] = operation;
        }
        renderFleetHoldState();
        renderStopAllState();
    }
    // Ownership, checked after every await. `destroyed` and the token are two
    // independent losses: this mount can be retired without the gesture being
    // taken over (destroy leaves the token so a replacement inherits the
    // fence), and the token can be taken over without this mount being
    // destroyed. AUTHORISING anything requires BOTH still to hold.
    function ownsFleetHold(operation) {
        return !destroyed && containerEl[AGENT_LIST_FLEET_HOLD_OPERATION] === operation;
    }
    // Conducting a gesture and painting its receipt are two different
    // permissions, and the compound gesture is where they come apart: its fence
    // ends when the durable Stop settles, while the confirming read it started
    // may land after that. Requiring gesture OWNERSHIP to paint would silently
    // drop the fan-out from the host's own per-agent rows to nothing at all.
    // What the panel actually requires is narrower: this mount is still on
    // screen, and no LATER gesture has claimed the panel out from under it.
    //
    // The rule across both gestures: `ownsFleetHold` authorises an ACTION,
    // `mayPublishFleetOutcome` authorises a PAINT. One panel, one predicate.
    function mayPublishFleetOutcome(operation) {
        if (destroyed) return false;
        const current = containerEl[AGENT_LIST_FLEET_HOLD_OPERATION];
        return current === undefined || current === operation;
    }
    // The gesture's fence ends HERE and nowhere else, so every path ends it the
    // same way and none of them can drift into a different idea of ownership.
    //
    // Idempotent by identity, and it has to be: the compound gesture ends its
    // fence the moment its Stop settles, and the `finally` that wraps that Stop
    // still runs afterwards — on a path where a hung convergence read means
    // "afterwards" may be never. Clearing only our own token is also what keeps
    // a retired continuation from unfencing a gesture it no longer owns; no
    // path can currently hand the token to a second gesture, since the entry
    // fence admits one at a time, so that half is an invariant guard rather
    // than a tested branch.
    function endFleetHoldGesture(operation) {
        let released = false;
        if (containerEl[AGENT_LIST_FLEET_HOLD_OPERATION] === operation) {
            delete containerEl[AGENT_LIST_FLEET_HOLD_OPERATION];
            released = true;
        }
        // Both lanes, one release, keyed by the same identity. A reservation
        // released by halves leaves the Stop All button disabled with no gesture
        // behind it — the fence outliving the thing it was fencing.
        if (containerEl[AGENT_LIST_STOP_ALL_OPERATION] === operation) {
            delete containerEl[AGENT_LIST_STOP_ALL_OPERATION];
            released = true;
        }
        if (!released) return;
        releaseFleetFenceThroughOwner();
    }
    // Clearing the fence is only half of unfencing: the DISABLED control is on
    // whichever mount is showing the banner now, and after a remount that is
    // not this one. A retired continuation that cleared the token without this
    // handoff would leave the replacement's fleet menu disabled for the life of
    // the page, with no gesture in flight to justify it.
    //
    // BOTH lanes, because one reservation fenced both: handing back only the
    // fleet menu would leave the Stop All button dead.
    //
    // The Stop lane gets a fresh READING, not just a repaint, and that is not
    // tidiness. Status reads are suppressed while an operation holds that lane —
    // the count is in flux mid-Stop — so a mount that arrived DURING the
    // reservation has never read it: repainting alone would show a replacement
    // pane a permanently disabled Stop All until the next poll. The gesture may
    // also have just stopped the very work the old count described.
    function releaseFleetFenceThroughOwner() {
        const currentOwner = containerEl[AGENT_LIST_PANE_OWNER];
        if (!destroyed && currentOwner === handle) {
            renderFleetHoldState();
            renderStopAllState();
            void refreshStopAllState();
            return;
        }
        if (currentOwner && typeof currentOwner.refreshFleetHoldFence === 'function') {
            currentOwner.refreshFleetHoldFence();
        }
    }
    // The latch table must still converge after a mount is retired mid-gesture,
    // so the refresh goes to whoever owns the pane NOW — exactly the handoff
    // Stop All performs in its own `finally`. Asking our own destroyed list
    // handle would re-publish state from a retired mount, which is the thing
    // the fence exists to prevent.
    function refreshFleetHoldThroughOwner() {
        const currentOwner = containerEl[AGENT_LIST_PANE_OWNER];
        if (!destroyed && currentOwner === handle) {
            void listHandle.refreshHoldState({ fresh: true });
            return;
        }
        if (currentOwner && typeof currentOwner.refreshHoldState === 'function') {
            void currentOwner.refreshHoldState({ fresh: true });
        }
    }

    let holdCountEl = header.querySelector('.agent-hold-count');
    const builtHoldCountEl = !holdCountEl;
    if (!holdCountEl) {
        holdCountEl = doc.createElement('span');
        holdCountEl.className = 'agent-hold-count';
        holdCountEl.setAttribute('role', 'status');
        holdCountEl.setAttribute('aria-live', 'polite');
        holdCountEl.hidden = true;
        // Left of Stop All when that exists, so the banner reads
        // "<held count> <Stop all> <fleet menu> <collapse>". Same
        // parent-relative insert as Stop All: an adopted header may nest its
        // chevron in a wrapper, and `header.insertBefore` would throw.
        const anchor = stopAllBtn && stopAllBtn.parentNode ? stopAllBtn : collapseBtn;
        (anchor.parentNode || header).insertBefore(holdCountEl, anchor);
    }

    // Always built, never adopted: `createKebabButton` binds its own click
    // listener, so adopting a previous mount's button would leave two menus
    // racing one gesture. It is removed again by this mount's destroy().
    const fleetKebab = createKebabButton(() => fleetMenuItems(), {
        className: 'agent-fleet-kebab',
        ariaLabel: 'Fleet Hold actions',
        title: 'Fleet actions',
        // The embedding host's document, not the console's — a menu built in
        // the wrong tree is a control nobody can reach.
        ownerDocument: doc,
    });
    fleetKebab.hidden = true;
    fleetKebab.disabled = true;
    (collapseBtn.parentNode || header).insertBefore(fleetKebab, collapseBtn);

    const fleetHoldResults = doc.createElement('div');
    fleetHoldResults.className = 'agent-fleet-hold-results';
    fleetHoldResults.setAttribute('role', 'status');
    fleetHoldResults.setAttribute('aria-live', 'polite');
    fleetHoldResults.hidden = true;
    // Above the Stop fan-out's results, because "Stop all and hold" holds
    // first: the two receipts read top-to-bottom in the order they happened.
    body.insertBefore(fleetHoldResults, stopAllResults || listHandle.element);

    const FLEET_HOLD_LABELS = {
        applied: 'Held',
        already_in_state: 'Already held',
        unreachable: 'Hold unreachable',
        indeterminate: 'Hold indeterminate',
    };
    const FLEET_RELEASE_LABELS = {
        applied: 'Host hold released',
        already_in_state: 'No host hold was set',
        refused_stale: 'Host hold changed — release refused',
        unreachable: 'Resume unreachable',
        indeterminate: 'Resume indeterminate',
    };
    // A disposition that left the fleet's latch table unknown or unchanged by
    // the operator's intent. Kept apart from partial/empty so "some agents are
    // held" is never read as "the request was refused".
    const FLEET_REFUSED_DISPOSITIONS = ['unreachable', 'indeterminate', 'refused_stale'];

    function fleetHoldState(snapshot = holdBanner) {
        if (!snapshot || !snapshot.supported || !snapshot.loaded) return 'unknown';
        if (snapshot.targetCount === 0) return 'empty';
        if (snapshot.heldCount === 0) return 'none';
        if (snapshot.heldCount < snapshot.targetCount) return 'partial';
        return 'all';
    }

    function fleetMenuItems() {
        if (destroyed || !holdBanner.supported || !holdBanner.canHold
            || fleetHoldIsPending()) return [];
        const total = holdBanner.targetCount;
        const noun = total === 1 ? 'agent' : 'agents';
        const items = [];
        if (holdBanner.hostHold) {
            items.push({
                label: 'Resume — release the host hold…',
                action: 'resume-host-hold',
                onSelect: () => { void releaseFleetHold(); },
            });
            return items;
        }
        items.push({
            label: `Hold all ${total} ${noun}…`,
            action: 'hold-all',
            onSelect: () => { void holdFleet({ alsoStop: false }); },
        });
        // Only while there is work to stop, for the same reason the card offers
        // "Stop and hold" only on a thinking row: with nothing in flight the
        // compound gesture IS the plain Hold above it.
        //
        // A Stop All already in flight reaches the same answer by a different
        // road: this gesture cannot reserve a lane somebody else holds, so it
        // would refuse at the claim and the operator would get a prompt for a
        // gesture that never ran. "Hold all" above is exactly the half that
        // remains, so nothing is taken away by not offering it.
        if (stopAllStatus.canStop && stopAllStatus.inFlightCount > 0
            && fleetStopLaneIsFree()) {
            items.push({
                label: 'Stop all and hold…',
                action: 'stop-all-and-hold',
                separatorBefore: true,
                onSelect: () => { void holdFleet({ alsoStop: true }); },
            });
        }
        return items;
    }

    renderFleetHoldState = () => {
        const state = fleetHoldState();
        fleetKebab.hidden = !holdBanner.supported;
        fleetKebab.disabled = !holdBanner.supported
            || !holdBanner.canHold
            || fleetHoldIsPending();
        fleetKebab.title = holdBanner.supported && !holdBanner.canHold
            ? 'Sovereign host authority is required to Hold agents'
            : 'Fleet actions';

        holdCountEl.dataset.holdState = state;
        holdCountEl.dataset.heldCount = String(holdBanner.heldCount);
        holdCountEl.dataset.targetCount = String(holdBanner.targetCount);
        if (holdBanner.stale) holdCountEl.dataset.holdStale = 'true';
        else delete holdCountEl.dataset.holdStale;

        const visible = state === 'partial' || state === 'all';
        holdCountEl.hidden = !visible;
        if (!visible) {
            holdCountEl.textContent = '';
            holdCountEl.removeAttribute('title');
            return;
        }
        holdCountEl.textContent = state === 'all'
            ? `All ${holdBanner.targetCount} held`
            : `${holdBanner.heldCount} of ${holdBanner.targetCount} held`;
        const detail = holdBanner.hostHold
            ? `A host-wide Hold is set by ${holdBanner.hostHold.actor_id} — ${holdBanner.hostHold.reason}`
            : 'Held by their own independent agent holds';
        // Say WHICH it is: a reading the host just confirmed, or the last one
        // it did. Presenting the second as the first is the lie.
        holdCountEl.title = holdBanner.stale
            ? `${detail} (last confirmed reading; the host Hold state is currently unreadable)`
            : detail;
    };

    function applyFleetLatch(current) {
        // `undefined` is the one value that means "no usable evidence"; every
        // other value — `null` for a released latch included — is a committed
        // fact this document may paint before any confirming read arrives.
        if (current === undefined) return;
        listHandle.applyHostLatch(current);
    }

    // Per-agent rows are EVIDENCE about a fleet, and this document cannot
    // produce them. `applyHostLatch` recomposes only the membership and the
    // agent latches the LAST read happened to carry, so an agent that joined
    // the fleet since, or one somebody held individually from another tab or
    // the CLI, is named wrongly and with total confidence. Ask the host with a
    // read that STARTS after the mutation committed, and hand the answer to the
    // renderer; a read that does not land leaves the fan-out unconfirmed rather
    // than restating a cached snapshot as a receipt.
    //
    // This read is the longest await in the gesture, so it is also where a
    // remount is most likely to land. Ownership is deliberately NOT rechecked
    // here: every caller holds a gate immediately after, and a duplicate check
    // inside would be a guard no test could distinguish from its absence.
    //
    // It is also on a LEASH, and that is load-bearing. `refreshHoldState`
    // awaits `api.getHostHoldState()`, which has no timeout of its own — the
    // client hands the browser's `fetch` no signal — so a host that accepts the
    // connection and never answers holds this promise open for the life of the
    // page. Everything downstream of an unbounded await inherits the hang: the
    // receipt panel, and the `finally` that ends the gesture's fence, which
    // would leave the fleet menu disabled forever with nothing in flight to
    // justify it. Presentation evidence may not fence an operator control.
    //
    // The leash bounds the WAIT, never the read. The request is deliberately
    // left in flight: if it lands late the inner list still publishes it
    // through its own sequence fence and the badge converges — strictly more
    // than aborting it would leave behind. A wait that runs out reports the
    // fan-out `unconfirmed`, which is the same true statement the panel already
    // makes about a read that failed.
    //
    // Running out must ALSO retire that read as the list's coalescing join
    // target. Leaving it there bounds the wait and then wedges the poll: every
    // later tick joins a promise that never settles, so no request is issued
    // and the badge's count and authority never recover. One hung request would
    // otherwise take the whole fleet surface down for the life of the mount.
    async function confirmFleetProjection() {
        let abandonRead = null;
        const read = (async () => {
            await listHandle.refreshHoldState({
                fresh: true,
                onReadStarted: (abandon) => { abandonRead = abandon; },
            });
            const snapshot = typeof listHandle.getHoldState === 'function'
                ? listHandle.getHoldState()
                : null;
            return snapshot && snapshot.confirmed === true ? snapshot : null;
        })();
        const view = doc.defaultView;
        const setTimeoutFn = view && view.setTimeout;
        const clearTimeoutFn = view && view.clearTimeout;
        if (typeof setTimeoutFn !== 'function') {
            // No clock to hang a leash from. An unbounded wait is the defect
            // itself, so decline to wait at all rather than fail open: the read
            // still runs and still converges the badge, and the fan-out says
            // what is true — it could not be confirmed. A document with no
            // window cannot poll either, so this surface is already degraded.
            void read.catch(() => {});
            return null;
        }
        return await new Promise((resolve) => {
            const wait = {
                timer: null,
                // Idempotent by membership, because three things race to end
                // this wait: the read landing, the leash running out, and
                // destroy() retiring the mount underneath it.
                settle: (projection) => {
                    if (!fleetConfirmWaits.delete(wait)) return;
                    if (typeof clearTimeoutFn === 'function') {
                        clearTimeoutFn.call(view, wait.timer);
                    }
                    resolve(projection);
                },
            };
            fleetConfirmWaits.add(wait);
            wait.timer = setTimeoutFn.call(
                view,
                () => {
                    // Only the leash retires the read. A read that LANDED
                    // cleared itself, and destroy() retires the whole list
                    // behind it — neither has a join target to give up on.
                    if (abandonRead) abandonRead();
                    wait.settle(null);
                },
                fleetConfirmTimeoutMs,
            );
            // `refreshHoldState` swallows its own read errors, so the rejection
            // arm is belt-and-braces rather than a live path — but an
            // unhandled rejection here would be a hang by another name.
            read.then((projection) => wait.settle(projection), () => wait.settle(null));
        });
    }

    // Per-agent rows for the fan-out, read back from the host's own composed
    // verdicts. This is where a partially-held fleet becomes legible: an agent
    // held by BOTH latches says so, and after a host resume the ones still held
    // by their own latch are exactly the ones still listed as held.
    //
    // `projection` is the ONLY source of those rows and of the tallies beside
    // them. A null (or unconfirmed) projection prints no rows and no counts:
    // "the fan-out could not be confirmed" is a true statement, and "3 of 3
    // agents held" drawn from a pre-mutation reading is not.
    function renderFleetHoldOutcomes({ action, disposition, detail, projection }) {
        const labels = action === 'release' ? FLEET_RELEASE_LABELS : FLEET_HOLD_LABELS;
        const confirmed = projection && projection.confirmed === true ? projection : null;
        const state = FLEET_REFUSED_DISPOSITIONS.includes(disposition)
            ? 'refused'
            : fleetHoldState(confirmed);
        fleetHoldResults.hidden = false;
        fleetHoldResults.textContent = '';
        fleetHoldResults.dataset.action = action;
        fleetHoldResults.dataset.disposition = disposition;
        fleetHoldResults.dataset.holdState = state;
        fleetHoldResults.dataset.fanout = confirmed ? 'confirmed' : 'unconfirmed';

        const summary = doc.createElement('p');
        const label = labels[disposition] || `${action === 'release' ? 'Resume' : 'Hold'}: ${disposition}`;
        if (!confirmed) {
            summary.textContent = `${label}. The host's per-agent Hold state could not be confirmed.`;
        } else if (confirmed.targetCount === 0) {
            summary.textContent = `${label}. The host named no agents to hold.`;
        } else {
            summary.textContent = `${label}. ${confirmed.heldCount} of ${confirmed.targetCount} agent${confirmed.targetCount === 1 ? '' : 's'} held.`;
        }
        if (detail) summary.title = detail;
        fleetHoldResults.appendChild(summary);

        if (!confirmed || !confirmed.agents.length) return;
        const list = doc.createElement('ul');
        for (const entry of confirmed.agents) {
            const sources = Array.isArray(entry && entry.sources) ? entry.sources : [];
            const row = doc.createElement('li');
            row.dataset.agentId = (entry && entry.agent_id) || '';
            row.dataset.held = entry && entry.held === true ? 'true' : 'false';
            row.dataset.sources = sources.join(' ');
            row.textContent = `${displayIdentity([entry && entry.agent_id])}: ${
                entry && entry.held === true
                    ? `held (${sources.join(', ') || 'unknown source'})`
                    : 'not held'
            }`;
            list.appendChild(row);
        }
        fleetHoldResults.appendChild(list);
    }

    async function holdFleet({ alsoStop }) {
        if (destroyed || !holdBanner.supported || !holdBanner.canHold
            || fleetHoldIsPending()) return;
        // A compound gesture needs BOTH lanes, so it needs both before it asks
        // the operator for anything. The menu already declines to offer the
        // entry while a Stop All runs, but a menu is composed when it OPENS: a
        // Stop All started while the menu sat open would otherwise be discovered
        // only after the reason prompt, leaving a gesture that either degrades
        // into a bare Hold or throws away an answer it asked for. Refuse here,
        // before the prompt, rather than deliver half of what was asked for.
        if (alsoStop && !fleetStopLaneIsFree()) return;
        const total = holdBanner.targetCount;
        const noun = total === 1 ? 'agent' : 'agents';
        const reason = askFleetReason(
            alsoStop
                ? `Stop all in-flight work and hold the host — all ${total} ${noun} will refuse to begin a turn. Reason:`
                : `Hold the host — all ${total} ${noun} will refuse to begin a turn. Reason:`,
            fleetHoldReason,
        );
        if (!reason) return;
        // Test-and-claim, stated together — and for the compound gesture, ONE
        // claim over both lanes with no await between the test and the set, so
        // nothing can take the Stop lane in between. The reason prompt cannot
        // currently yield — an async host asker returns a Promise, which
        // `makeReasonAsker` rejects as a non-string — so this re-read is
        // unreachable today and is deliberately not presented as a tested
        // guard. It is here so the condition the reservation is taken under is
        // written beside the claim rather than inferred from a check upstream.
        if (destroyed || fleetHoldIsPending()) return;
        if (alsoStop && !fleetStopLaneIsFree()) return;
        const operation = { kind: 'hold', reservesStopAll: !!alsoStop };
        claimFleetGesture(operation);
        // "This gesture has already asked the host for a fresh reading." The
        // `finally` below exists for the paths that never asked at all; once a
        // read has been ISSUED a second one is churn, and against the hung host
        // this leash was built for it would be a second hung request.
        let confirmingRead = false;
        try {
            let response = null;
            let unreachable = null;
            try {
                response = await api.setHostHold({
                    scope: 'host',
                    reason,
                    operation_id: newOperationId('ui-host-hold'),
                });
            } catch (error) {
                unreachable = error || new Error('request failed');
            }
            if (unreachable) {
                // Nothing reached the host, so this panel states no fleet fact.
                // The only per-agent table it could restate is the pre-request
                // one already on screen — which is exactly what a receipt may
                // not be drawn from.
                if (mayPublishFleetOutcome(operation)) {
                    renderFleetHoldOutcomes({
                        action: 'hold',
                        disposition: 'unreachable',
                        detail: unreachable.message,
                        projection: null,
                    });
                }
            // A retired mount paints nothing, reports nothing and stops nothing.
            // The Hold it committed is durable and unaffected: the replacement
            // mount reads it back from the host, which is the only place either
            // mount was ever entitled to learn it from.
            } else if (ownsFleetHold(operation)) {
                // BOTH Hold dispositions leave a latch, so a response carrying
                // no current latch contradicts its own receipt. Say
                // indeterminate rather than reading that receipt as a hold —
                // and, below, rather than stopping on the strength of it.
                const current = mutationLatch(response);
                const latched = !!current;
                // Paint the committed latch NOW so a confirming read that never
                // arrives cannot roll it back; the rows below still come only
                // from a read, never from this paint.
                applyFleetLatch(current);
                const receipt = response && response.receipt;
                const disposition = latched && receipt && typeof receipt.disposition === 'string'
                    ? receipt.disposition
                    : 'indeterminate';
                const detail = receipt && receipt.receipt_id
                    ? `Hold receipt ${receipt.receipt_id}`
                    : null;

                // The compound gesture's second half is authorised HERE, from
                // the committed mutation and nothing else: this mount still
                // owns the gesture, and the Hold it is compounding actually
                // latched. A bare Stop All after a failed Hold is a DIFFERENT
                // action from the one the operator asked for — the fleet would
                // stop and then start again on the next heartbeat, with
                // nothing latched and nothing on screen saying so.
                //
                // It is also STARTED here, in the same turn as the mutation
                // that authorised it, with no await in between. Stopping the
                // fleet is the durable half of what the operator asked for; it
                // may not queue behind a read that exists only to draw a
                // receipt, and against a host that answers the POST and then
                // never answers the GET, queueing it there meant the fleet was
                // held but never stopped — the silent half-gesture this
                // ordering exists to make impossible.
                //
                // The half runs INSIDE the fence, because "Stop all and hold"
                // is ONE gesture and a fence that ends halfway through is not a
                // fence. Released at the end of the Hold half, this banner's
                // menu — and a replacement mount's, which inherits the token
                // rather than a local flag — would offer "Resume" while the
                // Stop request was still open, and the gesture could finish
                // with the fleet stopped and UNHELD: free to start again on the
                // next heartbeat, which is the exact state it was asked to
                // prevent.
                //
                // It runs under the reservation this gesture took before its
                // Hold, so the lane was never open for a competing Stop All to
                // claim while that Hold was in flight. Passing the reservation
                // is also what keeps this from looking like a second claim.
                const stopWork = alsoStop && latched
                    ? runStopAll({
                        confirm: false,
                        reservation: operation,
                        // The gesture is over when its Stop is — NOT when the
                        // status read that follows the Stop lands. That read is
                        // convergence bookkeeping, a hung host can stall it
                        // forever, and this fence is local state: nothing about
                        // ending it is the host's to answer. The `finally`
                        // below still ends the fence on every path that never
                        // reached here (including a `runStopAll` that
                        // early-returned without settling), and ending it twice
                        // is a no-op by identity.
                        onSettled: () => endFleetHoldGesture(operation),
                    })
                    : null;
                // Created now, awaited at the bottom. A rejection arriving in
                // that gap would be an unhandled rejection; attaching a handler
                // marks it handled, and the `await` below still reports it.
                if (stopWork) void stopWork.catch(() => { /* reported below */ });

                // Only now the evidence, and only on a leash. Issued AFTER the
                // Stop deliberately: `runStopAll` runs its synchronous
                // browser-work fence before its first await, and a fence that
                // paints a latch would orphan a read issued ahead of it — which
                // would report this fan-out unconfirmed for no better reason
                // than the order two calls were written in.
                confirmingRead = true;
                const projection = await confirmFleetProjection();
                if (mayPublishFleetOutcome(operation)) {
                    renderFleetHoldOutcomes({
                        action: 'hold',
                        disposition,
                        detail,
                        projection,
                    });
                }
                // The Stop half's own verdict, not an assumption that calling it
                // stopped anything. It holds the lane by reservation, so a
                // decline here is an invariant guard rather than a live path —
                // but a compound gesture that silently reported a Stop it never
                // made is precisely what the reservation was built to prevent,
                // so the one place that could still say it says the truth.
                if (stopWork && (await stopWork) === false
                    && mayPublishFleetOutcome(operation) && stopAllResults) {
                    stopAllResults.hidden = false;
                    stopAllResults.textContent = 'The host hold is set, but the Stop half did '
                        + 'not run: in-flight work was NOT stopped.';
                }
            }
        } finally {
            endFleetHoldGesture(operation);
            // Only on the paths that did not already take one: the badge still
            // converges after a request that never reached the host. A mount
            // that lost the gesture still owes the CURRENT owner that
            // convergence — the latch it may have committed is real.
            if (!confirmingRead) refreshFleetHoldThroughOwner();
        }
    }

    async function releaseFleetHold() {
        if (destroyed || !holdBanner.supported || !holdBanner.canHold
            || fleetHoldIsPending()) return;
        const latch = holdBanner.hostHold;
        const observed = latch && typeof latch.hold_receipt_id === 'string'
            ? latch.hold_receipt_id
            : '';
        if (!observed) return;
        const reason = askFleetReason(
            'Release the host hold. Agents held individually stay held. Reason:',
            'Resumed from the agents banner',
        );
        if (!reason) return;
        // Same test-and-claim as holdFleet, and unreachable for the same reason.
        // A release touches one lane: it stops nothing, so it reserves nothing
        // of the Stop lane and leaves a Stop All free to run beside it.
        if (destroyed || fleetHoldIsPending()) return;
        const operation = { kind: 'release', reservesStopAll: false };
        claimFleetGesture(operation);
        let confirmingRead = false;
        try {
            let response;
            try {
                response = await api.releaseHostHold({
                    scope: 'host',
                    reason,
                    operation_id: newOperationId('ui-host-resume'),
                    // The receipt the operator SAW. A host hold replaced since
                    // the last poll is refused as stale rather than released.
                    expected_hold_receipt_id: observed,
                });
            } catch (error) {
                if (mayPublishFleetOutcome(operation)) {
                    renderFleetHoldOutcomes({
                        action: 'release',
                        disposition: 'unreachable',
                        detail: error && error.message,
                        projection: null,
                    });
                }
                return;
            }
            // A retired mount publishes nothing. The release it committed is
            // durable; the replacement mount reads it from the host.
            if (!ownsFleetHold(operation)) return;
            // `current` is authoritative under every disposition, including the
            // superseding latch a stale release was refused against.
            applyFleetLatch(mutationLatch(response));
            const receipt = response && response.receipt;
            const disposition = receipt && typeof receipt.disposition === 'string'
                ? receipt.disposition
                : 'indeterminate';
            // Which agents a host resume LEFT held is the whole point of this
            // panel, and it is the one thing releasing the host latch cannot
            // tell this document: an agent's own latch is on the other axis.
            // On the same leash as the Hold's: this gesture has no second half
            // to unblock, but a hung read here would still hold its fence past
            // the end of the page.
            confirmingRead = true;
            const projection = await confirmFleetProjection();
            if (!mayPublishFleetOutcome(operation)) return;
            renderFleetHoldOutcomes({
                action: 'release',
                disposition,
                detail: receipt && receipt.receipt_id
                    ? `Release receipt ${receipt.receipt_id}`
                    : null,
                projection,
            });
        } finally {
            endFleetHoldGesture(operation);
            if (!confirmingRead) refreshFleetHoldThroughOwner();
        }
    }

    // The inner list may already have published (a synchronous `hold: false`,
    // or a read that resolved before this chrome existed); paint from whatever
    // it last said rather than waiting for the next poll.
    if (typeof listHandle.getHoldState === 'function') {
        holdBanner = listHandle.getHoldState();
    }
    renderFleetHoldState();

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
        // Settle every confirming-read leash this mount still holds, rather
        // than cancel it: a retired gesture must still reach its `finally` to
        // hand the fence back to the current owner, and cancelling the timer
        // alone would strand it on an await nothing can now resolve. `null` is
        // the honest answer anyway — a mount that is gone may not paint the
        // fan-out it was waiting for. It also puts the timer out, so a long
        // leash cannot keep the embedding host's event loop alive past teardown.
        for (const wait of Array.from(fleetConfirmWaits)) wait.settle(null);
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
        if (circuitBanner && circuitBanner.parentNode) circuitBanner.remove();
        // The fleet Hold surface is always built by this mount, never adopted,
        // so it always leaves with it — a leaked kebab keeps a dead menu
        // callback alive over a list handle that has already been destroyed.
        if (builtHoldCountEl && holdCountEl) holdCountEl.remove();
        fleetKebab.remove();
        if (fleetHoldResults.parentNode) fleetHoldResults.remove();
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
        getHoldState: (...a) => listHandle.getHoldState(...a),
        // Repaint the fleet controls against the container-scoped fence. Called
        // by a RETIRED mount whose fleet gesture has just finished, so this
        // mount's menu stops being disabled by an operation that is over. Both
        // lanes, because one reservation fenced both.
        refreshFleetHoldFence: () => {
            if (destroyed) return;
            renderFleetHoldState();
            renderStopAllState();
            void refreshStopAllState();
        },
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
