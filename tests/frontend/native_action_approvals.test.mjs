import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { JSDOM } from 'jsdom';

test('approval rail loads native target over owned HTTP and offers only once decisions', async () => {
    const requests = [];
    const pending = [
        { id: 'owned-native', agent_name: 'Owned', feature: 'MCPAgent', tool: 'original', args: {},
          native_action: { action: 'cancel', method: 'tasks/cancel', task_handle: '<img src=x onerror=alert(1)>' } },
        { id: 'owned-ordinary', agent_name: 'Owned', feature: 'Files', tool: 'read', args: {} },
    ];
    const server = createServer((request, response) => {
        requests.push({ path: request.url, authorization: request.headers.authorization });
        response.setHeader('content-type', 'application/json');
        if (request.url === '/api/security/pending') response.end(JSON.stringify({ pending }));
        else if (request.url === '/api/security/auto-approve/rules') response.end('{"rules":[]}');
        else if (request.url === '/api/security/auto-approve/audit?limit=25') response.end('{"audit":[]}');
        else { response.statusCode = 404; response.end('{"error":"unowned route"}'); }
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const base = `http://127.0.0.1:${server.address().port}`;
    const dom = new JSDOM('<!doctype html><div id="approvals-list"></div><div id="approvals-rules"></div><div id="approvals-audit"></div><div id="approvals-pending-badge"></div><div id="pending-approvals"></div>', { url: base });
    globalThis.window = dom.window;
    globalThis.document = dom.window.document;
    globalThis.location = dom.window.location;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    globalThis.Node = dom.window.Node;
    globalThis.HTMLElement = dom.window.HTMLElement;
    globalThis.KESTREL_UI_CONFIG = {
        bearerToken: { getToken: async () => 'owned-qa-token' },
        fetchFn: (url, options) => fetch(new URL(url, base), options),
    };
    try {
        await import('../../kestrel_sovereign/static/js/icons.js');
        globalThis.kicon = window.kicon;
        const { loadApprovals } = await import('../../kestrel_sovereign/static/js/approvals.js');
        await loadApprovals();
        const rows = document.querySelectorAll('.approval-row');
        assert.equal(rows.length, 2);
        assert.equal(rows[0].querySelectorAll('button').length, 2);
        assert.ok(!rows[0].textContent.includes('remember'));
        assert.equal(rows[1].querySelectorAll('button').length, 3);
        assert.ok(rows[1].textContent.includes('Approve & remember'));
        const target = rows[0].querySelector('.native-action-target');
        assert.ok(target.textContent.includes('tasks/cancel'));
        assert.ok(target.textContent.includes('<img src=x onerror=alert(1)>'));
        assert.equal(target.querySelector('img'), null);
        assert.equal(requests.length, 3);
        const { Security } = await import('../../kestrel_sovereign/static/js/security.js');
        await Security.loadPendingApprovals();
        const quickTarget = document.querySelector('#pending-approvals .native-action-target');
        assert.ok(quickTarget, 'Settings quick approval must identify the native action');
        assert.ok(quickTarget.textContent.includes('tasks/cancel'));
        assert.ok(quickTarget.textContent.includes('<img src=x onerror=alert(1)>'));
        assert.equal(quickTarget.querySelector('img'), null);
        assert.equal(requests.length, 4);
        assert.ok(requests.every(request => request.authorization === 'Bearer owned-qa-token'));
    } finally {
        const closed = once(server, 'close');
        server.close();
        server.closeAllConnections();
        await closed;
        dom.window.close();
        delete globalThis.KESTREL_UI_CONFIG;
    }
});
