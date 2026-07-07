// #2233: body-level UI (Modal/Toast) mounts into a configurable overlay root.
// Embeds that serve kestrel CSS @scope-wrapped to their mount roots (Frinz)
// point the root inside a scope so overlays are styled; default stays body.
import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.location = dom.window.location;
globalThis.window.kicon = (n) => `<span class="ki ki-${n}"></span>`;
globalThis.kicon = globalThis.window.kicon;

const { Modal, Toast, setOverlayRoot, getOverlayRoot } = await import(
    '../../kestrel_sovereign/static/js/ui.js'
);

test('default overlay root is document.body', () => {
    setOverlayRoot(null);
    assert.equal(getOverlayRoot(), document.body);
    Modal.show({ title: 'T', content: '<p>x</p>', buttons: [] });
    assert.ok(document.body.querySelector(':scope > .modal-overlay'), 'modal on body by default');
    Modal.hide();
});

test('setOverlayRoot re-homes modals AND toasts inside the configured root', () => {
    const root = document.createElement('div');
    root.id = 'kestrelOverlayRoot';
    document.body.appendChild(root);
    setOverlayRoot(root);

    Modal.show({ title: 'T', content: '<p>x</p>', buttons: [] });
    assert.ok(root.querySelector('.modal-overlay'), 'modal mounts inside the overlay root');
    assert.equal(document.querySelectorAll('body > .modal-overlay').length, 0,
        'no stray body-level modal');
    Modal.hide();

    Toast.show('hello', 'info', 10);
    assert.ok(root.querySelector('#toast-container'), 'toast container re-homed into the root');

    setOverlayRoot(null);
    root.remove();
});

test('a disconnected root falls back to body (never a dead mount)', () => {
    const root = document.createElement('div');
    setOverlayRoot(root); // never attached to the document
    assert.equal(getOverlayRoot(), document.body, 'disconnected root -> body fallback');
    setOverlayRoot(null);
});
