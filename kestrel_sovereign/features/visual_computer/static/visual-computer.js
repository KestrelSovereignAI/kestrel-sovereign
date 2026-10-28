/** Original-agent visual operator view; pixels are requested only explicitly. */
import API from '/js/api.js';
import { registerPanel } from '/js/ui-ext/panels.js';
import bus from '/js/ui-ext/bus.js';

export function mountVisualComputer(body, api = API) {
    const element = (tag, text) => {
        const node = document.createElement(tag);
        if (text) node.textContent = text;
        return node;
    };
    body.replaceChildren();
    const title = element('h2', 'Visual computer');
    const selector = element('select');
    selector.setAttribute('aria-label', 'Owned visual session');
    const state = element('p', 'Refresh to inspect your visual sessions.');
    state.setAttribute('role', 'status');
    const controls = element('div');
    const image = element('img');
    image.alt = 'Current authorized screen';
    image.hidden = true;
    image.style.maxWidth = '100%';
    const note = element('p', 'Stop fences input. Release requests physical cleanup. Closing this panel keeps cleanup custody.');
    const manual = element('fieldset');
    manual.append(element('legend', 'Private human control'));
    const manualState = element('p', 'Take over, then explicitly open the human view. Agent observation and recording stay suspended. Network navigation is unavailable in this channel.');
    manualState.setAttribute('role', 'status');
    const enable = element('input'); enable.type = 'checkbox';
    enable.setAttribute('aria-label', 'Enable manual input');
    const inputLabel = element('label', ' Enable manual input while the screen has keyboard focus'); inputLabel.prepend(enable);
    const text = element('textarea'); text.maxLength = 16384; text.autocomplete = 'off';
    text.setAttribute('aria-label', 'Private text to type');
    const share = element('input'); share.type = 'checkbox';
    share.setAttribute('aria-label', 'Screen is ready to share with the agent');
    const shareLabel = element('label', ' I confirm this screen is ready to share with the agent. No automatic secret redaction is performed.'); shareLabel.prepend(share);
    manual.append(manualState, inputLabel, text, shareLabel);
    body.append(title, selector, controls, state, image, manual, note);
    let current = null;
    let generation = 0;
    let agent = api.getHostAgent();
    const pendingControls = new Map();
    let human = null;
    let humanFrame = null;
    let humanOwner = null;
    let opening = null;
    let lastInput = null;
    let expiryTimer = null;
    // Retained UI request identities are descriptive; every retry still uses
    // the original authenticated agent/channel and ordinary server checks.
    const retirements = new Map();
    const controlKey = (owner, id, operation) => JSON.stringify([owner, id, operation === 'pause' ? 'pause' : 'controller']);
    const retirementKey = closing => JSON.stringify([closing.owner, closing.lease.session_id, closing.lease.control_id]);
    const hasRetirement = (owner, id) => [...retirements.values()].some(closing => closing.owner === owner && closing.lease.session_id === id);
    const clearHumanPixels = () => {
        humanFrame = null; enable.checked = false; text.value = ''; share.checked = false;
        image.tabIndex = -1;
    };
    const clearPixels = () => { image.hidden = true; image.removeAttribute('src'); clearHumanPixels(); };
    const send = (path, options) => api.requestForAgent('/api/visual-computer' + path, options, agent);
    const humanPath = binding => `/sessions/${encodeURIComponent(binding.session_id)}/human/${encodeURIComponent(binding.control_id)}`;
    const matchesRetirement = (closing, value) => {
        const original = closing.lease;
        const returned = value?.request;
        if (value?.state !== 'revoked' || !/^[a-f0-9]{64}$/.test(closing.requestSha256 || '')
                || value.request_sha256 !== closing.requestSha256 || !returned) return false;
        const fields = ['schema_version', 'control_id', 'session_id', 'lease_id', 'driver_id', 'consumer_id',
            'control_epoch', 'issued_at', 'expires_at', 'max_frame_age_seconds'];
        return fields.every(key => original[key] === returned[key])
            && ['tenant_id', 'principal_id', 'agent_id'].every(key => original.scope?.[key] === returned.scope?.[key])
            && Array.isArray(original.capabilities) && Array.isArray(returned.capabilities)
            && JSON.stringify([...original.capabilities].sort()) === JSON.stringify([...returned.capabilities].sort());
    };
    const retireHuman = async closing => {
        const key = retirementKey(closing);
        const path = '/api/visual-computer' + humanPath(closing.lease);
        const reconcile = async () => {
            const value = await api.requestForAgent(path, undefined, closing.owner);
            if (!matchesRetirement(closing, value)) throw new Error('original channel retirement is unverified');
        };
        try {
            if (closing.refused) await reconcile();
            else {
                try {
                    await api.requestForAgent(path+'/revoke', {method:'POST',body:JSON.stringify(closing.payload)}, closing.owner);
                } catch (error) {
                    if (error?.status !== 409) throw error;
                    // Definitively refused POSTs are not resent. Metadata may
                    // prove a different tab retired this exact accepted lease.
                    closing.refused = true;
                    await reconcile();
                }
            }
            if (retirements.get(key) !== closing) return;
            retirements.delete(key);
            if (!human && !opening && agent === closing.owner && selector.value === closing.lease.session_id)
                manualState.textContent = 'Human channel revoked. A fresh takeover is required before opening another consumer.';
        } catch {
            if (retirements.get(key) !== closing) return;
            if (!human && !opening && agent === closing.owner && selector.value === closing.lease.session_id)
                manualState.textContent = 'Human retirement is uncertain. Use Stop or release; do not reconnect or retry input.';
        }
    };
    const revokeHuman = async () => {
        const lease = human?.request || opening;
        const owner = humanOwner;
        const requestSha256 = human?.request_sha256;
        human = null; opening = null;
        clearTimeout(expiryTimer); clearPixels();
        if (lease) {
            const closing = {lease, owner, requestSha256, payload:{request_id:crypto.randomUUID(), control_epoch:lease.control_epoch}, refused:false};
            const key = retirementKey(closing);
            if (!retirements.has(key)) retirements.set(key, closing);
        }
        await Promise.all([...retirements.values()].map(retireHuman));
    };
    const requireHuman = () => {
        if (!human || human.state !== 'active' || Date.now() >= Date.parse(human.request.expires_at)
                || humanOwner !== agent || current?.session?.controller !== 'human'
                || current.session.control_epoch !== human.request.control_epoch) throw new Error('human view unavailable');
        return human.request;
    };
    const selected = () => encodeURIComponent(selector.value);
    const display = value => {
        current = value;
        const id = selector.value;
        const pending = value.pending_control;
        const key = controlKey(agent, id, pending?.operation);
        // A status read cannot acknowledge a locally retained lost reply.
        if (!pendingControls.has(key) && pending?.session_id === id)
            pendingControls.set(key, {request:pending,owner:agent});
        const pauseKey = controlKey(agent, id, 'pause');
        const pause = pendingControls.get(pauseKey);
        const fresh = pending == null && value.session?.session_id === id
            && Number.isSafeInteger(value.session.control_epoch);
        if (fresh && pause?.refused && value.session.control_epoch >= pause.request.expected_control_epoch) {
            // A refused S is retained as an unverified attempt, not resent.
            // Fresh status permits only a NEXT explicit safety intent.
            pause.history = true;
            pendingControls.set(JSON.stringify([agent, id, 'refused-pause', pause.request.request_id]), pause);
            if (pendingControls.get(pauseKey) === pause) pendingControls.delete(pauseKey);
        } else if (fresh && pause?.acknowledged
                && value.session.control_epoch > pause.request.expected_control_epoch) {
            const controllerKey = controlKey(agent, id, 'takeover');
            if (pause.previous && pendingControls.get(controllerKey) === pause.previous) {
                // S was acknowledged; R's outcome remains unresolved. Retain
                // its exact descriptor as history, never replay it as new work.
                pause.previous.history = true;
                pendingControls.set(JSON.stringify([agent, id, 'unresolved', pause.previous.request.request_id]), pause.previous);
                pendingControls.delete(controllerKey);
            }
            if (pendingControls.get(pauseKey) === pause) pendingControls.delete(pauseKey);
        }
        const session = value.session;
        const lease = value.lease;
        state.textContent = `${value.phase}; controller: ${session?.controller || 'none'}; epoch: ${session?.control_epoch ?? 'unavailable'}; physical resource: ${lease?.state || 'acquisition unknown'}${value.requires_reconciliation ? '; reconciliation required' : ''}`;
        if ([...pendingControls.values()].some(entry => entry.history && entry.owner === agent && entry.request.session_id === id))
            state.textContent += '; earlier control outcome unresolved';
        if (session?.controller !== 'agent' || value.phase !== 'active') clearPixels();
    };
    const refreshState = async () => {
        const version = generation;
        const id = selector.value;
        if (!id) return;
        const value = await send(`/sessions/${selected()}`);
        if (version === generation && id === selector.value) display(value);
    };
    const button = (label, run) => {
        const node = element('button', label);
        node.type = 'button';
        node.addEventListener('click', () => {
            // Each operation owns its busy state. A blocked observation never
            // disables Stop/takeover or queues them behind a shared promise.
            node.disabled = true;
            Promise.resolve().then(run).catch(() => {
                clearPixels();
                state.textContent = 'Operation did not complete. Refresh status and reconcile the original request before retrying input.';
            }).finally(() => { node.disabled = false; });
        });
        controls.append(node);
    };
    const refresh = async () => {
        const version = generation;
        const rows = await send('/sessions');
        if (version !== generation) return;
        const previous = selector.value;
        selector.replaceChildren();
        for (const row of rows) {
            const option = element('option', `${row.session_id} — ${row.phase}`);
            option.value = row.session_id;
            selector.append(option);
        }
        if (rows.some(row => row.session_id === previous)) selector.value = previous;
        await refreshState();
    };
    const control = async (suffix, operation) => {
        if (!current?.session) throw new Error('select session');
        const shareConfirmed = share.checked;
        const version = ++generation;
        const id = selector.value;
        const owner = agent;
        clearPixels();
        if (operation === 'resume' && !shareConfirmed) throw new Error('confirm screen sharing');
        const key = controlKey(owner, id, operation);
        const pauseKey = controlKey(owner, id, 'pause');
        if (operation !== 'pause' && pendingControls.has(pauseKey)) throw new Error('reconcile original safety Stop');
        if (!pendingControls.has(key)) pendingControls.set(key, {
            request:{request_id:crypto.randomUUID(),session_id:id,expected_control_epoch:current.session.control_epoch,operation},owner,
            previous:operation === 'pause' ? pendingControls.get(controlKey(owner, id, 'takeover')) : undefined,
        });
        const entry = pendingControls.get(key);
        const original = entry.request;
        if (original.operation !== operation) throw new Error('reconcile original control');
        if (entry.refused) throw new Error('refresh status before another safety Stop');
        try {
            await api.requestForAgent(`/api/visual-computer/sessions/${encodeURIComponent(id)}/${suffix}`, { method: 'POST', body: JSON.stringify(
            operation === 'resume' ? {...original, share_screen:true} : original) }, owner);
        } catch (error) {
            if (operation === 'pause' && error?.status === 409 && pendingControls.get(key) === entry) entry.refused = true;
            throw error;
        }
        // This acknowledgement retires only the channel governed by this
        // control request, including a close whose network reply is delayed.
        for (const [retiredKey, closing] of retirements) {
            if (closing.owner === owner && closing.lease.session_id === id
                    && closing.lease.control_epoch === original.expected_control_epoch) retirements.delete(retiredKey);
        }
        if (pendingControls.get(key) === entry) {
            if (operation === 'pause') entry.acknowledged = true;
            else pendingControls.delete(key);
        }
        if (version !== generation || id !== selector.value || owner !== agent) return;
        // Preserve the original input ID until its outcome is reconciled.
        human = null; opening = null; clearTimeout(expiryTimer);
        await refreshState();
    };
    const manualButton = (label, run) => {
        const node = element('button', label); node.type = 'button'; manual.append(node);
        node.addEventListener('click', () => {
            node.disabled = true;
            Promise.resolve().then(run).catch(() => {
                clearPixels();
                manualState.textContent = 'Human operation failed or authority expired. Reconcile original IDs; do not retry input.';
            }).finally(() => { node.disabled = false; });
        });
    };
    const showHuman = async () => {
        const binding = requireHuman();
        if (lastInput) throw new Error('reconcile original input');
        const version = generation;
        const value = await send(humanPath(binding)+'/observation');
        if (version !== generation || human?.request !== binding) return;
        requireHuman();
        if (value.control_id !== binding.control_id || value.consumer_id !== binding.consumer_id
                || value.frame.control_epoch !== binding.control_epoch || value.frame.session_id !== binding.session_id
                || value.frame.media_type !== 'image/png') throw new Error('invalid human frame');
        humanFrame = value.frame;
        image.src = `data:image/png;base64,${value.image_base64}`; image.hidden = false; image.tabIndex = 0;
        manualState.textContent = 'Private human view. Enable input explicitly; focus the screen for pointer or keyboard control.';
    };
    const manualInput = async action => {
        const binding = requireHuman();
        if (action.kind === 'type' && !action.text.length) throw new Error('enter private text');
        if (!enable.checked || !humanFrame || lastInput) throw new Error('manual input is not armed');
        const deadline = Math.min(Date.now()+10000, Date.parse(binding.expires_at), Date.parse(humanFrame.observed_at)+15000);
        if (deadline <= Date.now()) throw new Error('fresh human view required');
        const request = {control_id:binding.control_id,consumer_id:binding.consumer_id,action:{
            action_id:crypto.randomUUID(),session_id:binding.session_id,observation_id:humanFrame.observation_id,
            control_epoch:binding.control_epoch,not_after:new Date(deadline).toISOString(),action}};
        lastInput = {action_id:request.action.action_id,control_id:binding.control_id,session_id:binding.session_id,binding,owner:agent};
        clearPixels();
        const result = await send(humanPath(binding)+'/actions', {method:'POST',body:JSON.stringify(request)});
        if (result.control_id !== binding.control_id || result.consumer_id !== binding.consumer_id
                || result.receipt?.action_id !== lastInput?.action_id || result.receipt.outcome === 'unknown') throw new Error('reconcile original input');
        manualState.textContent = `Manual input ${result.receipt.outcome}. Reconcile its original receipt, then request a fresh view.`;
    };
    manualButton('Open human view', async () => {
        if (human || opening || hasRetirement(agent, selector.value) || current?.session?.controller !== 'human') throw new Error('take over first');
        const version = generation;
        opening = {control_id:crypto.randomUUID(),session_id:selector.value,control_epoch:current.session.control_epoch};
        humanOwner = agent;
        const original = opening;
        const value = await send(humanPath(original), {method:'POST'});
        if (version !== generation) return;
        if (value.state !== 'active' || value.request?.control_id !== original.control_id
                || value.request.session_id !== original.session_id || value.request.control_epoch !== original.control_epoch
                || !(Date.parse(value.request.expires_at) > Date.now())) throw new Error('human binding differs');
        human = value;
        expiryTimer = setTimeout(() => { revokeHuman(); }, Math.max(0, Date.parse(value.request.expires_at)-Date.now()));
        await showHuman();
    });
    manualButton('Refresh human screen', showHuman);
    manualButton('Focus screen keyboard', () => {
        requireHuman();
        if (!enable.checked || !humanFrame || lastInput) throw new Error('arm a fresh human screen first');
        image.focus();
    });
    manualButton('Type private text', () => manualInput({kind:'type',text:text.value}));
    manualButton('Reconcile last manual input', async () => {
        const binding = lastInput?.binding;
        if (!lastInput || !binding) throw new Error('no original input');
        const id = lastInput.action_id;
        const result = await api.requestForAgent('/api/visual-computer'+humanPath(binding)+'/actions/'+encodeURIComponent(id), undefined, lastInput.owner);
        if (result.control_id !== binding.control_id || result.consumer_id !== binding.consumer_id
                || result.receipt?.action_id !== id || result.receipt.outcome === 'unknown') throw new Error('input remains unknown');
        lastInput = null;
        manualState.textContent = 'Original manual input reconciled. Refresh human screen before more input.';
    });
    manualButton('Close human view', revokeHuman);
    image.addEventListener('click', event => {
        if (!enable.checked || !humanFrame) return;
        const bounds = image.getBoundingClientRect();
        const x = Math.floor((event.clientX-bounds.left)*humanFrame.width/bounds.width);
        const y = Math.floor((event.clientY-bounds.top)*humanFrame.height/bounds.height);
        image.focus();
        manualInput({kind:'click',x,y,button:'left'}).catch(() => {
            clearPixels(); manualState.textContent = 'Manual input uncertain or refused. Reconcile the original ID.';
        });
    });
    const keys = new Set(['Enter','Tab','Escape','Backspace','Delete','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','PageUp','PageDown']);
    image.addEventListener('keydown', event => {
        if (!enable.checked || !humanFrame || document.activeElement !== image) return;
        const key = event.ctrlKey && event.key.toLowerCase() === 'a' ? 'Control+A'
            : event.shiftKey && event.key === 'Tab' ? 'Shift+Tab' : event.key;
        if (!keys.has(key) && key !== 'Control+A' && key !== 'Shift+Tab' && !(key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey)) return;
        event.preventDefault();
        manualInput(key.length === 1 ? {kind:'type',text:key} : {kind:'key',key}).catch(() => {
            clearPixels(); manualState.textContent = 'Manual input uncertain or refused. Reconcile the original ID.';
        });
    });
    image.addEventListener('blur', () => { enable.checked = false; });
    button('Refresh sessions', refresh);
    button('Show current screen', async () => {
        const version = generation;
        const id = selector.value;
        const value = await send(`/sessions/${selected()}/observation`);
        if (version !== generation || id !== selector.value || current?.session?.controller !== 'agent') return;
        // The server verifies bytes, media type and bounded image dimensions.
        if (!['image/png', 'image/jpeg'].includes(value.frame.media_type)) throw new Error('invalid screen');
        image.src = `data:${value.frame.media_type};base64,${value.image_base64}`;
        image.hidden = false;
    });
    button('Stop input', () => control('stop', 'pause'));
    button('Take over', () => control('takeover', 'takeover'));
    button('Hand back', () => control('handback', 'resume'));
    button('Release environment', async () => {
        const version = ++generation;
        const id = selector.value;
        const owner = agent;
        await revokeHuman();
        const result = await api.requestForAgent(`/api/visual-computer/sessions/${encodeURIComponent(id)}/release`, {method:'POST'}, owner);
        if (version !== generation || id !== selector.value || owner !== agent) return;
        state.textContent = result.released ? 'Physical release verified.' : 'Release remains uncertain; cleanup custody retained.';
        await refreshState();
    });
    selector.addEventListener('change', () => { revokeHuman(); generation++; current = null; clearPixels(); refreshState().catch(() => { state.textContent = 'Session status unavailable.'; }); });
    const reset = () => { revokeHuman(); generation++; agent = api.getHostAgent(); current = null; selector.replaceChildren(); clearPixels(); state.textContent = 'Refresh sessions for the selected agent.'; };
    const unsubscribe = api.onHostAgentChange(reset);
    return { refresh, hide: () => { generation++; revokeHuman(); clearPixels(); }, destroy: () => { generation++; revokeHuman(); clearPixels(); unsubscribe(); } };
}

let view = null;
registerPanel({ panelId: 'visual-computer', label: 'Computer', icon: 'ki ki-monitor', before: 'features',
    gate: () => API.hasCapability('visual-computer'), render: body => { view = mountVisualComputer(body); } });
bus.on('panel:shown', value => { if (value?.panelId === 'visual-computer') view?.refresh().catch(() => {}); });
bus.on('panel:hidden', value => { if (value?.panelId === 'visual-computer') view?.hide(); });
