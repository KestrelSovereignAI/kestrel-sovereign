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
    let pending = null;
    let human = null;
    let humanFrame = null;
    let humanOwner = null;
    let opening = null;
    let lastInput = null;
    let expiryTimer = null;
    let retirement = null;
    const clearHumanPixels = () => {
        humanFrame = null; enable.checked = false; text.value = ''; share.checked = false;
        image.tabIndex = -1;
    };
    const clearPixels = () => { image.hidden = true; image.removeAttribute('src'); clearHumanPixels(); };
    const send = (path, options) => api.requestForAgent('/api/visual-computer' + path, options, agent);
    const humanPath = binding => `/sessions/${encodeURIComponent(binding.session_id)}/human/${encodeURIComponent(binding.control_id)}`;
    const revokeHuman = async () => {
        const lease = retirement?.lease || human?.request || opening;
        const owner = retirement?.owner || humanOwner;
        human = null; opening = null;
        clearTimeout(expiryTimer); clearPixels();
        if (!lease) return;
        const target = humanPath(lease) + '/revoke';
        const payload = retirement?.payload || {request_id:crypto.randomUUID(), control_epoch:lease.control_epoch};
        retirement = {lease, owner, payload};
        manualState.textContent = 'Closing human channel; awaiting input retirement.';
        try {
            await api.requestForAgent('/api/visual-computer'+target, {method:'POST',body:JSON.stringify(payload)}, owner);
            retirement = null;
            manualState.textContent = 'Human channel revoked. A fresh takeover is required before opening another consumer.';
        } catch {
            manualState.textContent = 'Human retirement is uncertain. Use Stop or release; do not reconnect or retry input.';
        }
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
        pending = value.pending_control || null;
        const session = value.session;
        const lease = value.lease;
        state.textContent = `${value.phase}; controller: ${session?.controller || 'none'}; epoch: ${session?.control_epoch ?? 'unavailable'}; physical resource: ${lease?.state || 'acquisition unknown'}${value.requires_reconciliation ? '; reconciliation required' : ''}`;
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
        clearPixels();
        if (operation === 'resume' && !shareConfirmed) throw new Error('confirm screen sharing');
        pending = pending || { request_id: crypto.randomUUID(), session_id: selector.value,
            expected_control_epoch: current.session.control_epoch, operation };
        if (pending.operation !== operation) throw new Error('reconcile original control');
        await send(`/sessions/${selected()}/${suffix}`, { method: 'POST', body: JSON.stringify(
            operation === 'resume' ? {...pending, share_screen:true} : pending) });
        // Preserve the original input ID until its outcome is reconciled.
        human = null; opening = null; clearTimeout(expiryTimer);
        if (version !== generation || id !== selector.value) return;
        pending = null;
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
        if (human || opening || retirement || current?.session?.controller !== 'human') throw new Error('take over first');
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
        generation++;
        await revokeHuman();
        clearPixels();
        const result = await send(`/sessions/${selected()}/release`, {method:'POST'});
        state.textContent = result.released ? 'Physical release verified.' : 'Release remains uncertain; cleanup custody retained.';
        await refreshState();
    });
    selector.addEventListener('change', () => { revokeHuman(); generation++; current = null; pending = null; clearPixels(); refreshState().catch(() => { state.textContent = 'Session status unavailable.'; }); });
    const reset = () => { revokeHuman(); generation++; agent = api.getHostAgent(); current = null; pending = null; selector.replaceChildren(); clearPixels(); state.textContent = 'Refresh sessions for the selected agent.'; };
    const unsubscribe = api.onHostAgentChange(reset);
    return { refresh, hide: () => { generation++; revokeHuman(); clearPixels(); }, destroy: () => { generation++; revokeHuman(); clearPixels(); unsubscribe(); } };
}

let view = null;
registerPanel({ panelId: 'visual-computer', label: 'Computer', icon: 'ki ki-monitor', before: 'features',
    gate: () => API.hasCapability('visual-computer'), render: body => { view = mountVisualComputer(body); } });
bus.on('panel:shown', value => { if (value?.panelId === 'visual-computer') view?.refresh().catch(() => {}); });
bus.on('panel:hidden', value => { if (value?.panelId === 'visual-computer') view?.hide(); });
