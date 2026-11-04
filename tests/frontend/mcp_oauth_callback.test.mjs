import test from 'node:test';
import assert from 'node:assert/strict';
import { completeCallback } from '../../kestrel_sovereign/static/js/mcp-oauth-callback.mjs';

const state = 'owned-state-with-at-least-sixteen-characters';
const operationId = 'a'.repeat(43);
function boundary({ query = `?state=${state}&code=private-code&iss=https://issuer.example`, hint = {
    agentPrefix: '/api/agent-routes/ZGlkOnRlc3Q6b3duZWQ', operationId, returnTo: '/',
} } = {}) {
    const events = [];
    const nodes = { 'oauth-status': {}, 'oauth-return': { hidden: true } };
    const storage = new Map(hint ? [[`kestrel.mcp-oauth.${state}`, JSON.stringify(hint)]] : []);
    const browser = {
        location: { search: query, pathname: '/oauth/mcp/callback' },
        history: { replaceState(_state, _title, url) { events.push(['strip', url]); } },
        sessionStorage: {
            getItem(key) { events.push(['read-hint']); return storage.get(key) ?? null; },
            removeItem(key) { events.push(['consume-hint']); storage.delete(key); },
        },
        document: { getElementById: id => nodes[id] },
    };
    const load = async () => {
        events.push(['import']);
        return { init: async () => events.push(['authenticate']), requestHost: async (...args) => events.push(['post', ...args]) };
    };
    return { browser, events, nodes, storage, load };
}

test('strip query, consume routing hint, then import/authenticate and POST only once', async () => {
    const b = boundary();
    await completeCallback(b.browser, b.load);
    assert.deepEqual(b.events.map(event => event[0]), ['strip', 'read-hint', 'consume-hint', 'import', 'authenticate', 'post']);
    const post = b.events.at(-1);
    assert.equal(post[1], `/api/agent-routes/ZGlkOnRlc3Q6b3duZWQ/api/mcp/oauth/authorizations/${operationId}/callback`);
    assert.deepEqual(JSON.parse(post[2].body), { state, code: 'private-code', iss: 'https://issuer.example' });
    assert.equal(post[3], true, 'disable unauthenticated request replay');
    assert.equal(b.storage.size, 0);
    assert.equal(b.nodes['oauth-return'].href, '/');
    assert.equal(b.nodes['oauth-return'].hidden, false);
    await completeCallback(b.browser, b.load);
    assert.equal(b.events.filter(event => event[0] === 'post').length, 1);
});

for (const [name, options] of [
    ['missing hint', { hint: null }],
    ['external target', { hint: { agentPrefix: 'https://evil.example', operationId, returnTo: '/' } }],
    ['traversal target', { hint: { agentPrefix: '/api/agents/..', operationId, returnTo: '/' } }],
    ['external return', { hint: { agentPrefix: '', operationId, returnTo: '//evil.example' } }],
    ['duplicate state', { query: `?state=${state}&state=other&code=private-code&iss=https://issuer.example` }],
    ['ambiguous result', { query: `?state=${state}&code=private-code&error=private-error&iss=https://issuer.example` }],
    ['duplicate extension parameter', { query: `?state=${state}&code=private-code&iss=https://issuer.example&extension=a&extension=b` }],
]) {
    test(`${name}: remove query and make no authenticated request`, async () => {
        const b = boundary(options);
        await completeCallback(b.browser, b.load);
        assert.equal(b.events[0][0], 'strip');
        assert.equal(b.events.some(event => ['import', 'post'].includes(event[0])), false);
        assert.equal(b.nodes['oauth-status'].textContent.includes('private-'), false);
        assert.equal(b.nodes['oauth-return'].hidden, true);
    });
}

test('never show upstream exception text or replay a failed callback', async () => {
    const b = boundary();
    await completeCallback(b.browser, async () => ({ init: async () => {}, requestHost: async () => { throw Error('private-code'); } }));
    assert.equal(b.storage.size, 0);
    assert.equal(b.nodes['oauth-status'].textContent.includes('private-code'), false);
    assert.equal(b.nodes['oauth-return'].hidden, true);
});

for (const [name, query, expected] of [
    ['authorization extension', `?state=${state}&code=private-code&iss=https://issuer.example&session_state=private-extension`,
        { state, code: 'private-code', iss: 'https://issuer.example' }],
    ['denial details', `?state=${state}&error=access_denied&iss=https://issuer.example&error_description=private-description&error_uri=https://issuer.example/private`,
        { state, error: 'access_denied', iss: 'https://issuer.example' }],
    ['ignored credential-shaped extension', `?state=${state}&code=private-code&iss=https://issuer.example&access_token=private-unrecognized-token`,
        { state, code: 'private-code', iss: 'https://issuer.example' }],
]) {
    test(`OAuth response ${name}: ignore extras and forward only recognized fields`, async () => {
        const b = boundary({ query });
        await completeCallback(b.browser, b.load);
        const post = b.events.find(event => event[0] === 'post');
        assert.ok(post, 'A standard OAuth extension blocked the callback');
        assert.deepEqual(JSON.parse(post[2].body), expected);
        assert.equal(b.nodes['oauth-status'].textContent.includes('private-'), false);
    });
}
