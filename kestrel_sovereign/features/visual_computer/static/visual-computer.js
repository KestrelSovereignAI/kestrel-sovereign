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
    body.append(title, selector, controls, state, image, note);
    let current = null;
    let generation = 0;
    let agent = api.getHostAgent();
    let pending = null;
    const clearPixels = () => { image.hidden = true; image.removeAttribute('src'); };
    const send = (path, options) => api.requestForAgent('/api/visual-computer' + path, options, agent);
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
        const version = ++generation;
        const id = selector.value;
        clearPixels();
        pending = pending || { request_id: crypto.randomUUID(), session_id: selector.value,
            expected_control_epoch: current.session.control_epoch, operation };
        if (pending.operation !== operation) throw new Error('reconcile original control');
        await send(`/sessions/${selected()}/${suffix}`, { method: 'POST', body: JSON.stringify(pending) });
        if (version !== generation || id !== selector.value) return;
        pending = null;
        await refreshState();
    };
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
        clearPixels();
        const result = await send(`/sessions/${selected()}/release`, {method:'POST'});
        state.textContent = result.released ? 'Physical release verified.' : 'Release remains uncertain; cleanup custody retained.';
        await refreshState();
    });
    selector.addEventListener('change', () => { generation++; current = null; pending = null; clearPixels(); refreshState().catch(() => { state.textContent = 'Session status unavailable.'; }); });
    const reset = () => { generation++; agent = api.getHostAgent(); current = null; pending = null; selector.replaceChildren(); clearPixels(); state.textContent = 'Refresh sessions for the selected agent.'; };
    const unsubscribe = api.onHostAgentChange(reset);
    return { refresh, hide: () => { generation++; clearPixels(); }, destroy: () => { generation++; clearPixels(); unsubscribe(); } };
}

let view = null;
registerPanel({ panelId: 'visual-computer', label: 'Computer', icon: 'ki ki-monitor', before: 'features',
    gate: () => API.hasCapability('visual-computer'), render: body => { view = mountVisualComputer(body); } });
bus.on('panel:shown', value => { if (value?.panelId === 'visual-computer') view?.refresh().catch(() => {}); });
bus.on('panel:hidden', value => { if (value?.panelId === 'visual-computer') view?.hide(); });
