import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { JSDOM } from 'jsdom';

const requests = [];
const server = createServer(async (request, response) => {
    let body = '';
    for await (const chunk of request) body += chunk;
    requests.push({ path: request.url, authorization: request.headers.authorization,
        body: JSON.parse(body) });
    response.setHeader('content-type', 'application/json');
    response.end('{"success":true,"awaiter_gone":false}');
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
const base = `http://127.0.0.1:${server.address().port}`;
const dom = new JSDOM('<!doctype html><body></body>', { url: base });
for (const name of ['window', 'document', 'location', 'localStorage', 'sessionStorage', 'Node', 'HTMLElement']) {
    globalThis[name] = name === 'window' ? dom.window : dom.window[name];
}
globalThis.KESTREL_UI_CONFIG = {
    bearerToken: { getToken: async () => 'owned-approval-qa-token' },
    fetchFn: (url, options) => fetch(new URL(url, base), options),
};
await import('../../kestrel_sovereign/static/js/icons.js');
globalThis.kicon = window.kicon;
const { Security } = await import('../../kestrel_sovereign/static/js/security.js');

// Observe and join the real asynchronous paths; no approval behavior is replaced.
const pending = [];
for (const name of ['_drainApprovalQueue', 'submitApproval']) {
    const actual = Security[name].bind(Security);
    Security[name] = (...args) => {
        const result = actual(...args);
        pending.push(result);
        return result;
    };
}

after(async () => {
    await Promise.all(pending);
    const closed = once(server, 'close');
    server.close();
    server.closeAllConnections();
    await closed;
    dom.window.close();
    delete globalThis.KESTREL_UI_CONFIG;
});

for (const scope of ['off', 'session', 'always']) {
    for (const native of [false, true]) {
        test(`server-queued ${native ? 'native' : 'ordinary'} approval stays visible with Auto ${scope}`, async () => {
            const id = `owned-${scope}-${native}`;
            const start = requests.length;
            Security.globalAutoScope = scope;
            Security.globalAutoMode = scope !== 'off';
            const request = { id, feature: 'Owned', tool: 'explicit-ask', args: {} };
            if (native) request.native_action = {
                action: 'cancel', method: 'tasks/cancel', task_handle: 'original-owned-task',
            };
            try {
                Security.handleApprovalRequest(request);
                Security.handleApprovalRequest(request); // A replay must retain one original decision.
                const modal = document.querySelector('.modal-overlay');
                assert.ok(modal, 'A server-queued approval must not be silently approved by browser Auto');
                assert.equal(document.querySelectorAll('.modal-overlay').length, 1);
                assert.equal(Security.pendingApprovals.get(id), request);
                if (native) {
                    assert.ok(modal.textContent.includes('original-owned-task'));
                    assert.equal(modal.querySelectorAll('.modal-footer .modal-btn').length, 2);
                }
                await new Promise(setImmediate);
                assert.equal(requests.length, start, 'No decision may be sent before an explicit button click');
                const approved = scope !== 'always';
                const label = approved ? 'This Time' : 'Deny';
                [...modal.querySelectorAll('.modal-footer .modal-btn')]
                    .find(button => button.textContent.trim() === label).click();
                await Promise.all(pending);
                assert.equal(requests.length, start + 1);
                assert.deepEqual(requests.at(-1), {
                    path: '/api/security/approve', authorization: 'Bearer owned-approval-qa-token',
                    body: { approval_id: id, approved, scope: 'once' },
                });
                assert.equal(Security.pendingApprovals.has(id), false);
            } finally {
                Security.handleApprovalWithdrawn({ id, reason: 'owned-test-finished' });
                await Promise.all(pending);
            }
        });
    }
}
