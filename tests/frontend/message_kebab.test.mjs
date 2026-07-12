// #2410: message-bubble kebab menu + `chat-message-actions` UI-extension slot.
//
// Covers:
//   - the shared builder emits ONE kebab button (no red delete/purge circles)
//   - base menu carries "Move to trash" and a danger "Delete permanently"
//     (separatorBefore), wired to window.deleteMessage / window.purgeMessage
//   - the contextmenu (right-click) accelerator opens the same menu
//   - a registered `chat-message-actions` item appears (above the destructive
//     separator); a `gate: () => false` contribution is hidden; a throwing
//     item provider does NOT break the base items (error isolation)

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.kicon = () => '⋯';
globalThis.window.kicon = globalThis.kicon;

const { buildMessageKebab, messageMenuItems } = await import(
    '../../kestrel_sovereign/static/js/message_kebab.js'
);
const { UI } = await import('../../kestrel_sovereign/static/js/ui-ext/registry.js');
const { closeKebabMenu } = await import('../../kestrel_sovereign/static/js/kebab_menu.js');

function reset() {
    UI._reset();
    closeKebabMenu();
    document.body.innerHTML = '';
}

function bubble(msg) {
    const div = document.createElement('div');
    div.className = 'message agent-message';
    if (msg.id) div.dataset.messageId = msg.id;
    document.body.appendChild(div);
    return div;
}

function openMenu() {
    return document.querySelector('.kebab-menu');
}
function menuLabels() {
    const menu = openMenu();
    return menu ? [...menu.querySelectorAll('.kebab-menu-item')].map((b) => b.textContent) : [];
}

test('builder emits exactly one kebab button (no delete/purge circles)', () => {
    reset();
    const node = bubble({ id: 'm1', role: 'assistant' });
    node.appendChild(buildMessageKebab({ id: 'm1', role: 'assistant' }, node));
    assert.equal(node.querySelectorAll('.kebab-btn').length, 1);
    assert.equal(node.querySelector('.msg-kebab-btn') !== null, true);
    assert.equal(node.querySelector('.msg-delete-btn'), null);
    assert.equal(node.querySelector('.msg-purge-btn'), null);
    const kebab = node.querySelector('.msg-kebab-btn');
    assert.equal(kebab.getAttribute('aria-label'), 'Message actions');
});

test('base menu carries trash + danger permanent-delete wired to window handlers', () => {
    reset();
    const node = bubble({ id: 'm2', role: 'assistant' });
    const calls = [];
    window.deleteMessage = (id, n) => calls.push(['delete', id, n]);
    window.purgeMessage = (id, n) => calls.push(['purge', id, n]);

    const items = messageMenuItems({ id: 'm2', role: 'assistant' }, node);
    assert.deepEqual(items.map((i) => i.label), ['Move to trash', 'Delete permanently']);
    const purge = items[items.length - 1];
    assert.equal(purge.danger, true);
    assert.equal(purge.separatorBefore, true);

    items[0].onSelect();
    purge.onSelect();
    assert.deepEqual(calls, [['delete', 'm2', node], ['purge', 'm2', node]]);
});

test('contextmenu accelerator opens the same menu', () => {
    reset();
    const node = bubble({ id: 'm3', role: 'assistant' });
    node.appendChild(buildMessageKebab({ id: 'm3', role: 'assistant' }, node));
    const ev = new dom.window.Event('contextmenu', { bubbles: true, cancelable: true });
    node.dispatchEvent(ev);
    assert.deepEqual(menuLabels(), ['Move to trash', 'Delete permanently']);
    closeKebabMenu();
});

test('a registered chat-message-actions item appears above the destructive separator', () => {
    reset();
    const node = bubble({ id: 'm4', role: 'assistant' });
    UI.register({
        slot: 'chat-message-actions',
        id: 'copy',
        items: (ctx) => [{ label: `Copy ${ctx.messageId}`, onSelect: () => {} }],
    });
    const labels = messageMenuItems({ id: 'm4', role: 'assistant' }, node).map((i) => i.label);
    assert.deepEqual(labels, ['Move to trash', 'Copy m4', 'Delete permanently']);
});

test('gate: () => false hides a contribution', () => {
    reset();
    const node = bubble({ id: 'm5', role: 'assistant' });
    UI.register({
        slot: 'chat-message-actions',
        id: 'hidden',
        gate: () => false,
        items: () => [{ label: 'Never', onSelect: () => {} }],
    });
    const labels = messageMenuItems({ id: 'm5', role: 'assistant' }, node).map((i) => i.label);
    assert.deepEqual(labels, ['Move to trash', 'Delete permanently']);
});

test('a throwing item provider does not break the base items', () => {
    reset();
    const node = bubble({ id: 'm6', role: 'assistant' });
    UI.register({
        slot: 'chat-message-actions',
        id: 'boom',
        items: () => { throw new Error('boom'); },
    });
    UI.register({
        slot: 'chat-message-actions',
        id: 'ok',
        order: 200,
        items: () => [{ label: 'Still here', onSelect: () => {} }],
    });
    const labels = messageMenuItems({ id: 'm6', role: 'assistant' }, node).map((i) => i.label);
    assert.deepEqual(labels, ['Move to trash', 'Still here', 'Delete permanently']);
});
