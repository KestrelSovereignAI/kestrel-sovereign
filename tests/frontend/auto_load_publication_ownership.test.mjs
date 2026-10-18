import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
Object.assign(globalThis, { window: dom.window, document: dom.window.document, Node: dom.window.Node,
    HTMLElement: dom.window.HTMLElement, location: dom.window.location });
globalThis.CSS = { escape: s => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\\\$&') };
globalThis.sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.localStorage = globalThis.sessionStorage;
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
window.kicon = globalThis.kicon;
globalThis.confirm = () => true;
window.SharedMarkdown = { renderMarkdown: value => String(value), renderStreamingMarkdown: value => String(value),
    highlightCodeBlocks() {}, renderMermaidDiagrams() {}, async finalizeMarkdown() {} };

const { default: API } = await import('../../kestrel_sovereign/static/js/api.js');
const { state, getOrCreateChatPane, Toast } = await import('../../kestrel_sovereign/static/js/ui.js');
const { refreshConversationsPane, initTrashToggle } = await import('../../kestrel_sovereign/static/js/identity.js');
const { wipeAgentChatPane } = await import('../../kestrel_sovereign/static/js/chat.js');
API.hasCapability = () => true;
for (const id of ['conversations-pane', 'conversations-list', 'chat-container']) {
    const el = document.createElement('div'); el.id = id; document.body.appendChild(el);
}
const trash = document.createElement('button');
trash.id = 'trash-toggle-btn'; document.body.appendChild(trash); initTrashToggle();
const errors = [];
Toast.error = message => errors.push(message);
const pending = () => {
    let resolve, reject;
    const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
    return { promise, resolve, reject };
};
async function settle() { for (let i = 0; i < 8; i++) await new Promise(resolve => setTimeout(resolve, 0)); }

for (const phase of ['history', 'restart']) {
    for (const replacement of ['none', 'trash', 'refresh', 'failed-refresh']) {
        test(`actual transcript publication keeps list ownership across ${phase}/${replacement}`, async () => {
            const agent = `handoff-${phase}-${replacement}`;
            const session = `human-${agent}`;
            const barrier = pending(), started = pending(), listReplacement = pending();
            let replacing = false;
            errors.length = 0;
            state.encryptedAtRest = false;
            API.setHostAgent(agent);
            const pane = getOrCreateChatPane(agent);
            state.mountedChatAgent = agent;
            document.getElementById('chat-container').replaceChildren(pane.element);
            API.getConversations = async (_decrypt, _view, cursor = null) => {
                if (replacing && !cursor) return listReplacement.promise;
                return cursor ? { conversations: [{ session_id: session, has_human_turn: true,
                    last_message_at: '2026-10-15T12:00:00Z' }], next_cursor: null }
                    : { conversations: [{ session_id: 'autonomous', has_human_turn: false,
                        last_message_at: '2026-10-15T13:00:00Z' }], next_cursor: 'older' };
            };
            API.listTrash = () => listReplacement.promise;
            const history = { messages: [{ role: 'user', content: 'actual history at publication' }], encrypted_at_rest: true };
            API.getConversation = async () => {
                if (phase === 'history') { started.resolve(); return barrier.promise; }
                return history;
            };
            API.getRestartStatusEvents = async () => {
                if (phase === 'restart') { started.resolve(); return barrier.promise; }
                return { events: [] };
            };
            try {
                refreshConversationsPane();
                await started.promise;
                if (replacement !== 'none') {
                    replacing = true;
                    if (replacement === 'trash') trash.click();
                    else refreshConversationsPane();
                }
                barrier.resolve(phase === 'history' ? history : { events: [] });
                await settle();
                if (replacement === 'none') {
                    assert.equal(pane.sessionId, session, 'valid current auto-load must remain usable');
                    assert.match(pane.element.textContent, /actual history at publication/);
                } else {
                    assert.equal(pane.sessionId, null, 'superseded source published into the actual pane');
                    assert.equal(pane.element.children.length, 0, 'superseded source painted stale history');
                    assert.deepEqual(errors, []);
                    assert.equal(state.encryptedAtRest, false, 'retired source published encryption state');
                }
                if (replacement === 'failed-refresh') listReplacement.reject(new Error('offline replacement failed'));
                else listReplacement.resolve({ messages: [], conversations: [], next_cursor: null });
                await settle();
                if (replacement !== 'none') assert.equal(pane.sessionId, null);
            } finally {
                barrier.resolve(phase === 'history' ? history : { events: [] });
                listReplacement.resolve({ messages: [], conversations: [], next_cursor: null });
                await settle();
                API.getConversations = async () => ({ conversations: [], next_cursor: null });
                API.listTrash = async () => ({ messages: [] });
                if (trash.dataset.mode === 'trash') { trash.click(); await settle(); }
                wipeAgentChatPane(agent);
            }
        });
    }
}
