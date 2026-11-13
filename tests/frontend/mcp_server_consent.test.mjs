import test from 'node:test';
import assert from 'node:assert/strict';
import { MCPConsentFlow, takeConsentRequest, runConsentPage } from '../../kestrel_sovereign/static/js/mcp-server-consent.mjs';

const handle = 'a'.repeat(43);
const prefix = '/api/agent-routes/b3duZWQ';
const storageKey = `kestrel.mcp-server-consent.${prefix}`;
const details = { client_name: 'Owned connector', agent_name: 'Owned agent',
    resource: 'https://kestrel.example/mcp/owned', redirect_uri: 'https://provider.example/callback?keep=&tag=one&tag=two',
    scopes: ['memory:read', 'memory:write'], expires_at: 2_000_000_000 };
function boundary({ query = `?request=${handle}`, remembered } = {}) {
    const events = [];
    const storage = new Map(remembered ? [[storageKey, remembered]] : []);
    const browser = {
        location: { search: query, pathname: '/mcp/owned/consent', assign: url => events.push(['navigate', url]) },
        history: { replaceState: (_state, _title, path) => events.push(['strip', path]) },
        sessionStorage: {
            getItem: key => storage.get(key) ?? null,
            setItem: (key, value) => { events.push(['remember']); storage.set(key, value); },
            removeItem: key => { events.push(['forget']); storage.delete(key); },
        },
    };
    const api = { init: async () => events.push(['authenticate']), requestHost: async (...args) => {
        events.push(['request', ...args]);
        return args[1].method === 'GET' ? details : { redirect_url: details.redirect_uri + '&code=opaque-code&state=original&iss=https%3A%2F%2Fkestrel.example' };
    } };
    const load = async () => { events.push(['import']); return api; };
    return { events, storage, browser, api, load };
}

test('remove the nonce from history before imports, authenticate and read only the selected agent', async () => {
    const b = boundary();
    const flow = new MCPConsentFlow(b.browser, prefix, b.load);
    assert.equal(b.events[0][0], 'strip');
    assert.equal((await flow.load()).client_name, details.client_name);
    await flow.load();
    assert.equal(b.events.filter(e => e[0] === 'request').length, 1);
    assert.equal(b.events.find(e => e[0] === 'request')[1], `${prefix}/api/mcp/server/consent/requests/${handle}`);
    assert.equal(b.storage.get(storageKey), handle);
});

test('resume the same pending hint after a same-origin login reload', () => {
    const b = boundary({ query: '', remembered: handle });
    assert.equal(takeConsentRequest(b.browser, prefix).handle, handle);
    assert.equal(b.events[0][0], 'strip');
});

for (const query of ['?request=bad', `?request=${handle}&request=${handle}`, `?request=${handle}&returnTo=https://evil.example`, '?unexpected=1']) {
    test(`malformed location cannot reuse an older request: ${query.slice(0, 32)}`, () => {
        const b = boundary({ query, remembered: handle });
        assert.throws(() => new MCPConsentFlow(b.browser, prefix, b.load));
        assert.equal(b.events[0][0], 'strip');
        assert.equal(b.storage.size, 0);
        assert.equal(b.events.some(e => e[0] === 'import'), false);
    });
}

for (const bad of ['https://evil.example', '//evil.example', '/api/agents/..', '/api/agent-routes/../../escape']) {
    test(`a local routing hint cannot choose another credential destination: ${bad}`, () => {
        const b = boundary();
        assert.throws(() => new MCPConsentFlow(b.browser, bad, b.load));
        assert.equal(b.events[0][0], 'strip');
        assert.equal(b.events.some(e => e[0] === 'import'), false);
    });
}

for (const decision of ['approve', 'deny']) {
    test(`${decision} consumes the local hint before I/O and navigates only to the original callback`, async () => {
        const b = boundary();
        const flow = new MCPConsentFlow(b.browser, prefix, b.load);
        await flow.load();
        await flow.decide(decision, decision === 'approve' ? ['memory:read'] : []);
        const post = b.events.filter(e => e[0] === 'request').at(-1);
        assert.equal(post[2].method, 'POST');
        assert.deepEqual(JSON.parse(post[2].body), { decision, scopes: decision === 'approve' ? ['memory:read'] : [] });
        assert.equal(post[3], true, 'never automatically replay an unauthenticated mutation');
        assert.equal(b.storage.size, 0);
        assert.ok(b.events.at(-1)[1].startsWith(details.redirect_uri + '&'));
        await assert.rejects(flow.decide(decision, decision === 'approve' ? ['memory:read'] : []));
        assert.equal(b.events.filter(e => e[0] === 'request' && e[2].method === 'POST').length, 1);
    });
}

test('an uncertain decision cannot be retried', async () => {
    const b = boundary();
    const flow = new MCPConsentFlow(b.browser, prefix, b.load);
    await flow.load();
    let attempts = 0;
    b.api.requestHost = async () => { attempts++; throw Error('private-result'); };
    await assert.rejects(flow.decide('approve', ['memory:read']));
    await assert.rejects(flow.decide('approve', ['memory:read']));
    assert.equal(attempts, 1);
    assert.equal(b.storage.size, 0);
    assert.equal(b.events.some(e => e[0] === 'navigate'), false);
});

for (const [decision, scopes] of [['approve', []], ['approve', ['conversation:write']], ['approve', ['memory:read', 'memory:read']], ['deny', ['memory:read']]]) {
    test(`local consent selection cannot expand the original description: ${decision} ${scopes}`, async () => {
        const b = boundary();
        const flow = new MCPConsentFlow(b.browser, prefix, b.load);
        await flow.load();
        await assert.rejects(flow.decide(decision, scopes));
        assert.equal(b.events.filter(e => e[0] === 'request').length, 1);
        assert.equal(b.storage.size, 1);
    });
}

test('a mismatched callback cannot receive the issued code', async () => {
    const b = boundary();
    const flow = new MCPConsentFlow(b.browser, prefix, b.load);
    await flow.load();
    b.api.requestHost = async () => ({ redirect_url: 'https://evil.example/?code=private-code' });
    await assert.rejects(flow.decide('approve', ['memory:read']));
    assert.equal(b.events.some(e => e[0] === 'navigate'), false);
    assert.equal(b.storage.size, 0);
});


function uiBoundary() {
    const b = boundary();
    const element = () => ({
        children: [], events: {}, disabled: false, textContent: '', dataset: {},
        append(...children) { this.children.push(...children); },
        addEventListener(name, callback) { this.events[name] = callback; },
        dispatch(name) { this.events[name]({ preventDefault() {} }); },
        set innerHTML(_value) { throw Error('Untrusted HTML rendering is forbidden'); },
    });
    const ids = ['mcp-consent', 'consent-status', 'consent-allow', 'consent-deny', 'consent-client',
        'consent-agent', 'consent-resource', 'consent-scopes', 'consent-form', 'consent-return'];
    const nodes = Object.fromEntries(ids.map(id => [id, element()]));
    nodes['mcp-consent'].dataset.apiPrefix = prefix;
    b.browser.document = { getElementById: id => nodes[id], createElement: element,
        createTextNode: value => ({ textContent: value }) };
    return { ...b, nodes };
}
const settleEvents = () => new Promise(resolve => setImmediate(resolve));

test('the page renders connector strings as text and submits only checked original scopes', async () => {
    const b = uiBoundary();
    const nativeRequest = b.api.requestHost;
    b.api.requestHost = async (...args) => args[1].method === 'GET'
        ? { ...details, client_name: '<img src=x onerror=privateAction()>' } : nativeRequest(...args);
    await runConsentPage(b.browser, b.load);
    assert.equal(b.nodes['consent-client'].textContent, '<img src=x onerror=privateAction()>');
    assert.equal(b.nodes['consent-allow'].disabled, false);
    assert.equal(b.nodes['consent-scopes'].children.length, 2);
    b.nodes['consent-scopes'].children[1].children[0].checked = false;
    b.nodes['consent-form'].dispatch('submit');
    await settleEvents();
    const post = b.events.find(e => e[0] === 'request' && e[2].method === 'POST');
    assert.deepEqual(JSON.parse(post[2].body), { decision: 'approve', scopes: ['memory:read'] });
    assert.equal(b.nodes['consent-return'].href, '/');
});

test('an empty selection stays local and leaves the decision available', async () => {
    const b = uiBoundary();
    await runConsentPage(b.browser, b.load);
    for (const label of b.nodes['consent-scopes'].children) label.children[0].checked = false;
    b.nodes['consent-form'].dispatch('submit');
    await settleEvents();
    assert.equal(b.events.some(e => e[0] === 'request' && e[2].method === 'POST'), false);
    assert.equal(b.nodes['consent-allow'].disabled, false);
    assert.ok(b.nodes['consent-status'].textContent.startsWith('Choose at least one'));
    assert.equal(b.storage.size, 1);
});

test('a failed decision stays disabled and never exposes or replays the failure', async () => {
    const b = uiBoundary();
    await runConsentPage(b.browser, b.load);
    let attempts = 0;
    b.api.requestHost = async () => { attempts++; throw Error('private-result'); };
    b.nodes['consent-form'].dispatch('submit');
    await settleEvents();
    b.nodes['consent-form'].dispatch('submit');
    await settleEvents();
    assert.equal(attempts, 1);
    assert.equal(b.nodes['consent-allow'].disabled, true);
    assert.equal(b.nodes['consent-deny'].disabled, true);
    assert.equal(b.storage.size, 0);
    assert.equal(b.nodes['consent-status'].textContent.includes('private-result'), false);
});

test('an unavailable description never enables decisions or renders its exception', async () => {
    const b = uiBoundary();
    b.api.requestHost = async () => { throw Error('private-pending-data'); };
    await runConsentPage(b.browser, b.load);
    assert.equal(b.nodes['consent-allow'].disabled, true);
    assert.equal(b.nodes['consent-deny'].disabled, true);
    assert.equal(b.nodes['consent-status'].textContent.includes('private-pending-data'), false);
    assert.equal(b.nodes['consent-return'].href, '/');
});

test('a signed-in user receives a clear owner credential requirement without private diagnostics', async () => {
    const b = uiBoundary();
    b.api.requestHost = async () => { throw Object.assign(Error('private-pending-data'), { status: 403 }); };
    await runConsentPage(b.browser, b.load);
    assert.equal(b.nodes['consent-allow'].disabled, true);
    assert.equal(b.nodes['consent-deny'].disabled, true);
    assert.ok(b.nodes['consent-status'].textContent.includes('owner’s durable API key'));
    assert.equal(b.nodes['consent-status'].textContent.includes('private-pending-data'), false);
});
