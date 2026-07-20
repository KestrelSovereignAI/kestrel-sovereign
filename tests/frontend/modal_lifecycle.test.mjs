import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.location = dom.window.location;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}"></span>`;
globalThis.kicon = globalThis.window.kicon;

const activeDocumentKeydownListeners = new Set();
const nativeDocumentAddEventListener = document.addEventListener.bind(document);
const nativeDocumentRemoveEventListener = document.removeEventListener.bind(document);
document.addEventListener = (type, listener, options) => {
    if (type === 'keydown') activeDocumentKeydownListeners.add(listener);
    return nativeDocumentAddEventListener(type, listener, options);
};
document.removeEventListener = (type, listener, options) => {
    if (type === 'keydown') activeDocumentKeydownListeners.delete(listener);
    return nativeDocumentRemoveEventListener(type, listener, options);
};

const { Modal, setOverlayRoot } = await import('../../kestrel_sovereign/static/js/ui.js');

function listenerCount() {
    return activeDocumentKeydownListeners.size;
}

function openModal({ title = 'Lifecycle test', content, buttons = [], onClose } = {}) {
    Modal.show({
        title,
        content: content || '<input id="first-input"><input id="last-input">',
        buttons,
        onClose,
    });
    return document.querySelector('.modal-overlay');
}

function keydown(key, { shiftKey = false } = {}) {
    document.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
        key,
        shiftKey,
        bubbles: true,
        cancelable: true,
    }));
}

afterEach(() => {
    Modal.hide();
    setOverlayRoot(null);
    document.body.replaceChildren();
    assert.equal(listenerCount(), 0, 'each test returns document listeners to baseline');
});

test('renders an accessible dialog, labels its close button, and initially focuses content', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    openModal({ title: 'Accessible lifecycle' });

    const dialog = document.querySelector('.modal-container');
    const heading = document.getElementById(dialog.getAttribute('aria-labelledby'));
    const close = dialog.querySelector('.modal-close-btn');
    assert.equal(dialog.getAttribute('role'), 'dialog');
    assert.equal(dialog.getAttribute('aria-modal'), 'true');
    assert.equal(heading.textContent, 'Accessible lifecycle');
    assert.equal(close.getAttribute('aria-label'), 'Close dialog');
    assert.equal(document.activeElement.id, 'first-input');
    assert.equal(listenerCount(), 1);
});

test('X teardown precedes onClose and restores focus exactly once', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    let closes = 0;

    openModal({
        onClose: () => {
            closes += 1;
            assert.equal(document.querySelector('.modal-overlay'), null);
            assert.equal(listenerCount(), 0);
            assert.equal(document.activeElement, opener);
        },
    });
    document.querySelector('.modal-close-btn').click();

    assert.equal(closes, 1);
    Modal.hide();
    assert.equal(closes, 1, 'a repeated close never invokes onClose again');
});

test('overlay click closes only when the overlay itself is targeted', () => {
    let closes = 0;
    const overlay = openModal({ onClose: () => { closes += 1; } });

    overlay.querySelector('.modal-container').click();
    assert.equal(closes, 0, 'clicks inside the dialog do not dismiss it');
    assert.equal(listenerCount(), 1);

    overlay.click();
    assert.equal(closes, 1);
    assert.equal(listenerCount(), 0);
});

test('Modal.confirm cancel and confirm settle correctly after teardown', async () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();

    const cancelled = Modal.confirm('Cancel?', 'Choose cancel');
    [...document.querySelectorAll('.modal-btn')]
        .find((button) => button.textContent === 'Cancel').click();
    assert.equal(await cancelled, false);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
    assert.equal(document.activeElement, opener);

    const confirmed = Modal.confirm('Confirm?', 'Choose confirm');
    [...document.querySelectorAll('.modal-btn')]
        .find((button) => button.textContent === 'Confirm').click();
    assert.equal(await confirmed, true);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
    assert.equal(document.activeElement, opener);
});

test('Escape closes the active modal and restores its opener', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    let closes = 0;
    openModal({ onClose: () => { closes += 1; } });

    keydown('Escape');

    assert.equal(closes, 1);
    assert.equal(listenerCount(), 0);
    assert.equal(document.activeElement, opener);
});

test('direct and repeated hide share the idempotent close lifecycle', () => {
    let closes = 0;
    openModal({ onClose: () => { closes += 1; } });

    Modal.hide();
    Modal.hide();

    assert.equal(closes, 1);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
});

test('replacement closes the old lifecycle before installing the new one', () => {
    let aCloses = 0;
    let bCloses = 0;
    openModal({ title: 'A', onClose: () => { aCloses += 1; } });

    openModal({ title: 'B', onClose: () => { bCloses += 1; } });

    assert.equal(aCloses, 1);
    assert.equal(bCloses, 0);
    assert.equal(document.querySelectorAll('.modal-overlay').length, 1);
    assert.equal(document.querySelector('.modal-header h3').textContent, 'B');
    assert.equal(listenerCount(), 1, 'replacement owns exactly one document listener');

    Modal.hide();
    assert.equal(aCloses, 1);
    assert.equal(bCloses, 1);
    assert.equal(listenerCount(), 0);
});

test('a reentrant onClose replacement cannot leave two active modal lifecycles', () => {
    openModal({
        title: 'A',
        onClose: () => openModal({ title: 'Newest' }),
    });

    openModal({ title: 'Superseded' });

    assert.equal(document.querySelectorAll('.modal-overlay').length, 1);
    assert.equal(document.querySelector('.modal-header h3').textContent, 'Newest');
    assert.equal(listenerCount(), 1);
});

test('the A-then-B reproduction invokes each callback once without stale Escape listeners', () => {
    let aCloses = 0;
    let bCloses = 0;
    openModal({ title: 'A', onClose: () => { aCloses += 1; } });
    document.querySelector('.modal-close-btn').click();
    assert.equal(listenerCount(), 0);

    openModal({ title: 'B', onClose: () => { bCloses += 1; } });
    keydown('Escape');

    assert.deepEqual({ aCloses, bCloses }, { aCloses: 1, bCloses: 1 });
    assert.equal(listenerCount(), 0);
});

test('onClose exceptions propagate only after lifecycle cleanup', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    let closes = 0;
    openModal({
        onClose: () => {
            closes += 1;
            throw new Error('close failed');
        },
    });

    assert.throws(() => Modal.hide(), /close failed/);
    assert.equal(closes, 1);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
    assert.equal(document.activeElement, opener);
    assert.doesNotThrow(() => Modal.hide());
    assert.equal(closes, 1);
});

test('an action exception after hide cannot strand the modal lifecycle', () => {
    let reportedError = null;
    window.addEventListener('error', (event) => {
        reportedError = event.error;
        event.preventDefault();
    }, { once: true });
    openModal({
        buttons: [{
            label: 'Fail after close',
            onClick: () => {
                Modal.hide();
                throw new Error('action failed');
            },
        }],
    });

    document.querySelector('.modal-btn').click();

    assert.match(reportedError?.message || '', /action failed/);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
});

test('Tab and Shift-Tab wrap focus within the active dialog', () => {
    const outside = document.createElement('button');
    document.body.appendChild(outside);
    outside.focus();
    openModal({
        content: '<input id="first-input"><input id="second-input">',
        buttons: [{ label: 'Last action', onClick: () => {} }],
    });

    const close = document.querySelector('.modal-close-btn');
    const last = document.querySelector('.modal-btn');
    assert.equal(document.activeElement.id, 'first-input');

    last.focus();
    keydown('Tab');
    assert.equal(document.activeElement, close, 'Tab wraps last to first');

    keydown('Tab', { shiftKey: true });
    assert.equal(document.activeElement, last, 'Shift-Tab wraps first to last');

    outside.focus();
    keydown('Tab');
    assert.equal(document.activeElement, close, 'programmatic outside focus is recaptured');

    Modal.hide();
    assert.equal(document.activeElement, outside, 'focus returns to the original opener');
});
