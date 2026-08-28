// #2208: the Identity panel's danger-zone section must be WIRED into the real
// loadIdentity() render path — not just unit-testable in isolation. The original
// implementation shipped a self-contained module (`renderIdentityDangerZone`)
// and an empty `<div id="identity-danger-zone">` slot, but nothing ever called
// the module, so the section rendered empty forever ("passes unit tests, dead in
// production"). This test mounts the REAL identity.js render path and asserts the
// slot gets populated + the type-the-name confirm gate fires the delete handler,
// so that regression can't recur silently.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.CustomEvent = dom.window.CustomEvent;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}
globalThis.location = dom.window.location;
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.window.kicon = globalThis.kicon;
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const API = apiModule.default;
const { Modal, setOverlayRoot } = await import('../../kestrel_sovereign/static/js/ui.js');
const { renderIdentityDangerZone, resolveDeleteAction } = await import(
    '../../kestrel_sovereign/static/js/identity-danger-zone.js'
);

const { loadIdentity } = await import('../../kestrel_sovereign/static/js/identity.js');

function resetPanel() {
    document.body.innerHTML = '';
    for (const id of ['identity-card', 'genesis-audit', 'identity-danger-zone']) {
        const el = document.createElement('div');
        el.id = id;
        document.body.appendChild(el);
    }
}

test('loadIdentity() populates #identity-danger-zone via the native (multi_agent) capability path (#2208)', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'Emma', did: 'did:pkh:emma' });
    API.hasCapability = (cap) => true;
    API.canManageHostAgentLifecycle = (operation) => operation === 'delete';
    let deleted = null;
    API.deleteAgent = async (name) => { deleted = name; return { success: true }; };

    await loadIdentity();

    const zone = document.querySelector('#identity-danger-zone [data-testid="identity-danger-zone"]');
    assert.ok(zone, 'danger zone rendered into the identity panel slot');
    const btn = document.getElementById('danger-zone-delete-btn');
    assert.ok(btn, 'delete action button rendered');

    // Confirm gate: clicking opens the type-the-name modal; the delete handler
    // must not fire until the typed value matches the agent name.
    btn.click();
    const input = document.getElementById('danger-zone-confirm-input');
    assert.ok(input, 'confirm modal with type-the-name input opened');

    const dangerBtn = document.querySelector('#modal-overlay .modal-btn-danger');
    assert.ok(dangerBtn, 'modal danger button present');

    // Wrong value → handler must NOT fire.
    input.value = 'wrong';
    dangerBtn.click();
    assert.equal(deleted, null, 'mismatched confirm name does not fire delete');

    // Correct value → handler fires with the agent name.
    input.value = 'Emma';
    dangerBtn.click();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(deleted, 'Emma', 'matching confirm name fires api.deleteAgent(name)');
});

test('loadIdentity() leaves #identity-danger-zone empty when neither host handler nor native capability applies (#2208)', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'Solo', did: 'did:pkh:solo' });
    // No host handler, and multi_agent capability off → section hidden.
    API.hasCapability = (cap) => cap === 'identity';
    API.canManageHostAgentLifecycle = () => false;
    delete globalThis.KESTREL_UI_CONFIG;

    await loadIdentity();

    assert.equal(
        document.getElementById('identity-danger-zone').innerHTML.trim(),
        '',
        'danger zone stays empty with no delete capability',
    );
});

test('multi-agent feature presence does not expose delete without caller authority', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'OAuthAgent', did: 'did:pkh:oauth' });
    API.hasCapability = () => true;
    API.canManageHostAgentLifecycle = () => false;
    delete globalThis.KESTREL_UI_CONFIG;

    await loadIdentity();

    assert.equal(
        document.getElementById('identity-danger-zone').innerHTML.trim(),
        '',
        'feature availability cannot substitute for sovereign lifecycle authority',
    );
});

test('native delete rechecks caller authority after the panel was rendered', () => {
    let allowed = true;
    let deleted = false;
    const action = resolveDeleteAction({
        identity: { name: 'Emma' },
        api: {
            getHostAgent: () => 'emma',
            canManageHostAgentLifecycle: () => allowed,
            deleteAgent: () => { deleted = true; },
        },
        dz: {},
    });
    assert.equal(action.show, true);

    allowed = false;
    assert.throws(() => action.handler(), /authority is no longer available/i);
    assert.equal(deleted, false);
});

test('stale native delete button disappears instead of opening after revocation', () => {
    resetPanel();
    Modal.hide();
    let allowed = true;
    let deleted = false;
    const container = document.getElementById('identity-danger-zone');
    const api = {
        getHostAgent: () => 'emma',
        canManageHostAgentLifecycle: () => allowed,
        deleteAgent: () => { deleted = true; },
    };
    assert.equal(renderIdentityDangerZone({
        container,
        identity: { name: 'Emma' },
        api,
        Modal,
        Toast: {},
    }), true);

    allowed = false;
    container.querySelector('#danger-zone-delete-btn').click();
    assert.equal(container.innerHTML, '');
    assert.equal(document.getElementById('danger-zone-confirm-input'), null);
    assert.equal(deleted, false);
});

test('loadIdentity() renders a host-injected delete handler even without native capability (#2208)', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'Companion', did: 'did:pkh:companion' });
    API.hasCapability = (cap) => cap === 'identity'; // no native multi_agent
    let hostCalled = null;
    globalThis.KESTREL_UI_CONFIG = {
        dangerZone: {
            delete: {
                label: 'Delete companion',
                handler: (identity) => { hostCalled = identity.name; },
            },
        },
    };

    await loadIdentity();

    const btn = document.getElementById('danger-zone-delete-btn');
    assert.ok(btn, 'host-mapped delete action rendered');
    assert.equal(btn.textContent.trim(), 'Delete companion', 'host label used');

    btn.click();
    const input = document.getElementById('danger-zone-confirm-input');
    input.value = 'Companion';
    document.querySelector('#modal-overlay .modal-btn-danger').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(hostCalled, 'Companion', 'host handler fires with the identity payload');

    delete globalThis.KESTREL_UI_CONFIG;
});

test('IME composition Enter never confirms a danger-zone deletion', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'Emma', did: 'did:pkh:emma' });
    API.hasCapability = () => true;
    API.canManageHostAgentLifecycle = (operation) => operation === 'delete';
    const prevGetHostAgent = API.getHostAgent;
    API.getHostAgent = () => 'Emma';
    let deletes = 0;
    API.deleteAgent = async () => { deletes += 1; return { success: true }; };
    delete globalThis.KESTREL_UI_CONFIG;

    try {
        await loadIdentity();
        document.getElementById('danger-zone-delete-btn').click();
        await new Promise((resolve) => setTimeout(resolve, 60));
        const input = document.getElementById('danger-zone-confirm-input');
        input.value = 'Emma';

        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            isComposing: true,
            bubbles: true,
            cancelable: true,
        }));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(deletes, 0);
        assert.ok(document.getElementById('modal-overlay'));

        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            bubbles: true,
            cancelable: true,
        }));
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(deletes, 1);
    } finally {
        API.getHostAgent = prevGetHostAgent;
    }
});

test('a confirmed deletion still starts when modal focus restoration fails', async () => {
    resetPanel();
    API.getIdentity = async () => ({ name: 'Emma', did: 'did:pkh:emma' });
    API.hasCapability = () => true;
    API.canManageHostAgentLifecycle = (operation) => operation === 'delete';
    const prevGetHostAgent = API.getHostAgent;
    API.getHostAgent = () => 'Emma';
    let deletes = 0;
    API.deleteAgent = async () => { deletes += 1; return { success: true }; };
    delete globalThis.KESTREL_UI_CONFIG;

    try {
        await loadIdentity();
        const opener = document.getElementById('danger-zone-delete-btn');
        opener.focus();
        opener.click();
        const input = document.getElementById('danger-zone-confirm-input');
        input.value = 'Emma';
        opener.focus = () => { throw new Error('opener focus failed'); };

        const reportedError = new Promise((resolve) => {
            window.addEventListener('error', (event) => {
                event.preventDefault();
                resolve(event.error);
            }, { once: true });
        });
        document.querySelector('#modal-overlay .modal-btn-danger').click();

        assert.match((await reportedError).message, /opener focus failed/);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(deletes, 1, 'the confirmed irreversible action is not lost after teardown');
        assert.equal(document.getElementById('modal-overlay'), null);
    } finally {
        Modal.hide();
        API.getHostAgent = prevGetHostAgent;
    }
});

test('closed-shadow danger-zone confirmation queries and arms its own lifecycle', async () => {
    Modal.hide();
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'closed' });
    const container = document.createElement('div');
    const mount = document.createElement('div');
    shadow.append(container, mount);
    document.body.appendChild(host);
    let deletes = 0;
    setOverlayRoot(mount);
    try {
        renderIdentityDangerZone({
            container,
            identity: { name: 'Emma' },
            api: {},
            Modal,
            Toast: {},
            config: {
                dangerZone: {
                    delete: { handler: async () => { deletes += 1; } },
                },
            },
        });
        container.querySelector('#danger-zone-delete-btn').click();
        await new Promise((resolve) => setTimeout(resolve, 60));

        const input = mount.querySelector('#danger-zone-confirm-input');
        const danger = mount.querySelector('.modal-btn-danger');
        assert.ok(input && danger);
        assert.equal(document.getElementById('danger-zone-confirm-input'), null);
        input.value = 'Emma';
        input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
        assert.equal(danger.disabled, false);
        danger.click();
        await new Promise((resolve) => setTimeout(resolve, 0));

        assert.equal(deletes, 1);
        assert.equal(mount.querySelector('.modal-overlay'), null);
    } finally {
        Modal.hide();
        setOverlayRoot(null);
        host.remove();
    }
});

test('native delete targets the manager routing key, not the editable display name (#2208 codex P2)', async () => {
    resetPanel();
    // Renamed agent: identity display name differs from the multi-agent
    // manager's routing key. DELETE must hit the routing key or it 404s.
    API.getIdentity = async () => ({ name: 'Renamed Emma', did: 'did:pkh:emma' });
    API.hasCapability = () => true;
    API.canManageHostAgentLifecycle = (operation) => operation === 'delete';
    const prevGetHostAgent = API.getHostAgent;
    API.getHostAgent = () => 'emma';
    let deleted = null;
    API.deleteAgent = async (name) => { deleted = name; return { success: true }; };
    delete globalThis.KESTREL_UI_CONFIG;

    await loadIdentity();

    const btn = document.getElementById('danger-zone-delete-btn');
    assert.ok(btn, 'delete action rendered');
    btn.click();
    const input = document.getElementById('danger-zone-confirm-input');
    // The confirm gate still types the DISPLAY name the user sees...
    input.value = 'Renamed Emma';
    document.querySelector('#modal-overlay .modal-btn-danger').click();
    await new Promise((r) => setTimeout(r, 0));
    // ...but the DELETE goes to the routing key.
    assert.equal(deleted, 'emma', 'deleteAgent called with the routing key');

    API.getHostAgent = prevGetHostAgent;
});

test('index.css styles the danger-zone section (#2237 shipped markup with no CSS)', async () => {
    const { readFile } = await import('node:fs/promises');
    const css = await readFile(new URL('../../kestrel_sovereign/static/index.css', import.meta.url), 'utf8');
    for (const cls of ['.identity-danger-zone', '.identity-danger-zone-header', '.identity-danger-zone-btn']) {
        assert.ok(css.includes(cls), `${cls} must be styled in index.css`);
    }
});
