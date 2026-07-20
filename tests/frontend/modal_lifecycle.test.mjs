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

const trackedDocumentListenerTypes = new Set(['focusin', 'keydown']);
const activeDocumentListeners = new Map();
const nativeDocumentAddEventListener = document.addEventListener.bind(document);
const nativeDocumentRemoveEventListener = document.removeEventListener.bind(document);
document.addEventListener = (type, listener, options) => {
    if (trackedDocumentListenerTypes.has(type)) {
        if (!activeDocumentListeners.has(type)) activeDocumentListeners.set(type, new Set());
        activeDocumentListeners.get(type).add(listener);
    }
    return nativeDocumentAddEventListener(type, listener, options);
};
document.removeEventListener = (type, listener, options) => {
    activeDocumentListeners.get(type)?.delete(listener);
    return nativeDocumentRemoveEventListener(type, listener, options);
};

const { Modal, setOverlayRoot } = await import('../../kestrel_sovereign/static/js/ui.js');

function listenerCount(type) {
    if (type) return activeDocumentListeners.get(type)?.size || 0;
    return [...activeDocumentListeners.values()]
        .reduce((count, listeners) => count + listeners.size, 0);
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
    (document.activeElement || document).dispatchEvent(new dom.window.KeyboardEvent('keydown', {
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
    assert.equal(listenerCount(), 2);
    assert.equal(listenerCount('focusin'), 1, 'the lifecycle owns one global focus guard');
    assert.equal(listenerCount('keydown'), 1, 'the lifecycle owns one detached-focus key guard');
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
    assert.equal(listenerCount(), 2);

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
    assert.equal(listenerCount(), 2, 'replacement owns exactly one pair of document guards');

    Modal.hide();
    assert.equal(aCloses, 1);
    assert.equal(bCloses, 1);
    assert.equal(listenerCount(), 0);
});

test('show returns a lifecycle-bound ownership handle', () => {
    let replacementCloses = 0;
    const first = Modal.show({
        title: 'First owner',
        content: '<p>first</p>',
    });

    assert.equal(first.isCurrent(), true);
    const replacement = first.replace({
        title: 'Replacement owner',
        content: '<p>replacement</p>',
        onClose: () => { replacementCloses += 1; },
    });

    assert.ok(replacement, 'a current owner may replace its lifecycle');
    assert.equal(first.isCurrent(), false);
    assert.equal(first.querySelector('.modal-container'), null,
        'a stale owner cannot query the replacement lifecycle');
    assert.equal(replacement.isCurrent(), true);
    assert.equal(replacement.querySelector('.modal-body')?.textContent.trim(), 'replacement');
    assert.equal(first.close(), false, 'a stale owner cannot close the replacement');
    assert.equal(first.replace({ title: 'Stale', content: '' }), null,
        'a stale owner cannot replace the active lifecycle');
    assert.equal(document.querySelector('.modal-header h3').textContent, 'Replacement owner');
    assert.equal(replacementCloses, 0);

    assert.equal(replacement.close(), true);
    assert.equal(replacement.close(), false, 'handle close is idempotent');
    assert.equal(replacement.isCurrent(), false);
    assert.equal(replacementCloses, 1);
});

test('a reentrant onClose replacement cannot leave two active modal lifecycles', () => {
    openModal({
        title: 'A',
        onClose: () => openModal({ title: 'Newest' }),
    });

    openModal({ title: 'Superseded' });

    assert.equal(document.querySelectorAll('.modal-overlay').length, 1);
    assert.equal(document.querySelector('.modal-header h3').textContent, 'Newest');
    assert.equal(listenerCount(), 2);
});

test('a show request superseded by reentrant onClose is cancelled exactly once', async () => {
    let supersededCloses = 0;
    openModal({
        title: 'A',
        onClose: () => openModal({ title: 'Newest' }),
    });

    let supersededHandle;
    const result = new Promise((resolve) => {
        supersededHandle = Modal.show({
            title: 'Superseded',
            content: '<p>never mounted</p>',
            onClose: () => {
                supersededCloses += 1;
                resolve('cancelled');
            },
        });
    });

    assert.equal(await result, 'cancelled');
    assert.equal(supersededCloses, 1);
    assert.equal(supersededHandle.isCurrent(), false);
    assert.equal(supersededHandle.close(), false);
    assert.equal(supersededHandle.replace({ title: 'Stale', content: '' }), null);
    assert.equal(document.querySelectorAll('.modal-overlay').length, 1);
    assert.equal(document.querySelector('.modal-header h3').textContent, 'Newest');
    assert.equal(listenerCount(), 2);
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

test('show setup failure rolls back DOM and listeners without firing onClose', () => {
    const trackedAddEventListener = document.addEventListener;
    let closes = 0;
    document.addEventListener = (type, listener, options) => {
        if (type === 'focusin') throw new Error('focus guard registration failed');
        return trackedAddEventListener(type, listener, options);
    };
    try {
        assert.throws(
            () => openModal({ onClose: () => { closes += 1; } }),
            /focus guard registration failed/,
        );
    } finally {
        document.addEventListener = trackedAddEventListener;
    }

    assert.equal(closes, 0, 'a modal that never finished showing is not reported as closed');
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(Modal._currentModal, null);
    assert.equal(Modal._lifecycle, null);
    assert.equal(listenerCount(), 0);
});

test('focus restoration failure cannot skip onClose or leave lifecycle state behind', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    let closes = 0;
    openModal({ onClose: () => { closes += 1; } });
    opener.focus = () => { throw new Error('focus restore failed'); };

    assert.throws(() => Modal.hide(), /focus restore failed/);
    assert.equal(closes, 1, 'onClose still runs after the focus error');
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
    assert.doesNotThrow(() => Modal.hide());
});

test('an action exception cannot strand the modal lifecycle', () => {
    let reportedError = null;
    window.addEventListener('error', (event) => {
        reportedError = event.error;
        event.preventDefault();
    }, { once: true });
    openModal({
        buttons: [{
            label: 'Fail',
            onClick: () => {
                throw new Error('action failed');
            },
        }],
    });

    document.querySelector('.modal-btn').click();

    assert.match(reportedError?.message || '', /action failed/);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
});

test('an async action rejection cannot strand the active modal lifecycle', async () => {
    let rejectAction;
    let closes = 0;
    const reportedError = new Promise((resolve) => {
        window.addEventListener('error', (event) => {
            event.preventDefault();
            resolve(event.error);
        }, { once: true });
    });
    openModal({
        buttons: [{
            label: 'Start',
            onClick: () => new Promise((resolve, reject) => { rejectAction = reject; }),
        }],
        onClose: () => { closes += 1; },
    });
    document.querySelector('.modal-btn').click();
    rejectAction(new Error('async action failed'));

    assert.match((await reportedError).message, /async action failed/);
    assert.equal(closes, 1);
    assert.equal(document.querySelector('.modal-overlay'), null);
    assert.equal(listenerCount(), 0);
});

test('a stale async action rejection cannot close a replacement', async () => {
    let rejectAction;
    let actionOwnerCloses = 0;
    let replacementCloses = 0;
    const reportedError = new Promise((resolve) => {
        window.addEventListener('error', (event) => {
            event.preventDefault();
            resolve(event.error);
        }, { once: true });
    });
    openModal({
        title: 'Async action owner',
        buttons: [{
            label: 'Start',
            onClick: () => new Promise((resolve, reject) => { rejectAction = reject; }),
        }],
        onClose: () => { actionOwnerCloses += 1; },
    });
    document.querySelector('.modal-btn').click();

    openModal({
        title: 'Replacement',
        onClose: () => { replacementCloses += 1; },
    });
    rejectAction(new Error('async action failed'));

    assert.match((await reportedError).message, /async action failed/);
    assert.equal(actionOwnerCloses, 1);
    assert.equal(replacementCloses, 0);
    assert.equal(document.querySelector('.modal-header h3').textContent, 'Replacement');
    assert.equal(listenerCount(), 2);
});

test('prompt Enter handling is immediate, owned, unique, and replacement-safe', async () => {
    const originalNow = Date.now;
    Date.now = () => 42;
    try {
        const first = Modal.prompt('First', 'first prompt');
        const second = Modal.prompt('Second', 'second prompt', 'default');
        assert.equal(await first, null, 'replacement cancels the first prompt');

        const input = document.querySelector('.modal-body input');
        assert.equal(input.getAttribute('aria-label'), 'Second');
        input.value = 'second value';
        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            bubbles: true,
            cancelable: true,
        }));

        assert.equal(await second, 'second value');
        assert.equal(document.querySelector('.modal-overlay'), null);
        assert.equal(listenerCount(), 0);
    } finally {
        Date.now = originalNow;
    }
});

test('positive tabindex controls determine the real focus-trap order', () => {
    openModal({
        content: `
            <input id="tab-three" tabindex="3">
            <input id="tab-one" tabindex="1">
            <input id="tab-two" tabindex="2">
        `,
    });

    assert.equal(document.activeElement.id, 'tab-one');
    keydown('Tab', { shiftKey: true });
    assert.equal(document.activeElement, document.querySelector('.modal-close-btn'));
});

test('a radio group contributes only its checked control to the focus-trap boundary', () => {
    openModal({
        content: `
            <input id="radio-checked" type="radio" name="choice" checked>
            <input id="radio-unchecked" type="radio" name="choice">
        `,
    });

    const checked = document.getElementById('radio-checked');
    const close = document.querySelector('.modal-close-btn');
    assert.equal(document.activeElement, checked, 'initial focus uses the radio group tab stop');

    keydown('Tab');
    assert.equal(document.activeElement, close,
        'Tab wraps from the checked radio instead of escaping past an unchecked radio');
});

test('an explicit programmatic autofocus target may use tabindex -1', () => {
    openModal({
        content: `
            <h4 id="autofocus-heading" tabindex="-1" autofocus>Dialog context</h4>
            <button id="later-action">Continue</button>
        `,
    });

    assert.equal(document.activeElement, document.getElementById('autofocus-heading'));
});

test('initial focus skips controls in hidden and inert ancestor subtrees', () => {
    openModal({
        content: `
            <div style="display: none"><button id="css-hidden">Hidden</button></div>
            <div inert><button id="inert-control">Inert</button></div>
            <button id="visible-control">Visible</button>
        `,
    });

    assert.equal(document.activeElement.id, 'visible-control');
});

test('shadow-mounted dialogs keep internal focus and restore the deep opener', () => {
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'open' });
    const opener = document.createElement('button');
    const mount = document.createElement('div');
    shadow.append(opener, mount);
    document.body.appendChild(host);
    opener.focus();
    assert.equal(shadow.activeElement, opener);

    setOverlayRoot(mount);
    try {
        const handle = Modal.show({
            title: 'Shadow dialog',
            content: '<input id="shadow-first"><input id="shadow-second">',
        });
        const first = mount.querySelector('#shadow-first');
        const second = mount.querySelector('#shadow-second');
        assert.equal(shadow.activeElement, first);

        second.focus();
        assert.equal(shadow.activeElement, second,
            'document focus guards recognize retargeted focus from inside the dialog');

        handle.close();
        assert.equal(shadow.activeElement, opener,
            'close restores the actual shadow descendant that opened the dialog');
    } finally {
        setOverlayRoot(null);
    }
});

test('shadow-mounted dialogs restore a light-DOM opener outside the overlay root', () => {
    const opener = document.createElement('button');
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'open' });
    const mount = document.createElement('div');
    shadow.appendChild(mount);
    document.body.append(opener, host);
    opener.focus();
    assert.equal(shadow.activeElement, null,
        'the overlay root has no active element before the dialog opens');
    assert.equal(document.activeElement, opener);

    setOverlayRoot(mount);
    try {
        const handle = Modal.show({
            title: 'Shadow dialog from light DOM',
            content: '<input id="shadow-light-opener-input">',
        });
        assert.equal(shadow.activeElement, mount.querySelector('#shadow-light-opener-input'));

        handle.close();
        assert.equal(document.activeElement, opener,
            'close falls back to the document opener when the overlay root had none');
    } finally {
        setOverlayRoot(null);
    }
});

test('closed-shadow dialogs own keyboard focus and restore their inner opener', () => {
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'closed' });
    const opener = document.createElement('button');
    const mount = document.createElement('div');
    shadow.append(opener, mount);
    document.body.appendChild(host);
    opener.focus();
    assert.equal(host.shadowRoot, null, 'the test exercises closed-root retargeting');
    assert.equal(shadow.activeElement, opener);

    setOverlayRoot(mount);
    try {
        const handle = Modal.show({
            title: 'Closed shadow dialog',
            content: '<input id="closed-first"><input id="closed-second">',
        });
        const first = handle.querySelector('#closed-first');
        const second = handle.querySelector('#closed-second');
        const close = handle.querySelector('.modal-close-btn');
        assert.equal(shadow.activeElement, first);

        second.focus();
        assert.equal(shadow.activeElement, second,
            'focusin ownership is derived from the closed root, not document.activeElement');
        second.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'x',
            bubbles: true,
            composed: true,
            cancelable: true,
        }));
        assert.equal(shadow.activeElement, second,
            'a retargeted ordinary key does not reset focus to the first control');

        second.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Tab',
            bubbles: true,
            composed: true,
            cancelable: true,
        }));
        assert.equal(shadow.activeElement, close,
            'Tab wraps using activeElement from the retained closed root');

        close.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Escape',
            bubbles: true,
            composed: true,
            cancelable: true,
        }));
        assert.equal(handle.isCurrent(), false);
        assert.equal(shadow.activeElement, opener,
            'close restores the true inner opener instead of the closed-root host');
    } finally {
        setOverlayRoot(null);
    }
});

test('Modal.prompt reads its value through the handle inside a closed shadow root', async () => {
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'closed' });
    const mount = document.createElement('div');
    shadow.appendChild(mount);
    document.body.appendChild(host);
    setOverlayRoot(mount);
    try {
        const result = Modal.prompt('Shadow prompt');
        const input = mount.querySelector('.modal-body input');
        input.value = 'inside shadow';
        [...mount.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'OK').click();

        assert.equal(await result, 'inside shadow');
        assert.equal(mount.querySelector('.modal-overlay'), null);
    } finally {
        setOverlayRoot(null);
    }
});

test('a rejected async onClose is reported once after synchronous teardown', async () => {
    const originalReportError = window.reportError;
    const reported = [];
    window.reportError = (error) => reported.push(error);
    try {
        openModal({
            onClose: async () => {
                throw new Error('async close failed');
            },
        });

        Modal.hide();
        assert.equal(document.querySelector('.modal-overlay'), null);
        assert.equal(listenerCount(), 0);
        await Promise.resolve();
        await Promise.resolve();

        assert.equal(reported.length, 1);
        assert.match(reported[0].message, /async close failed/);
        Modal.hide();
        assert.equal(reported.length, 1, 'idempotent close never observes onClose twice');
    } finally {
        if (originalReportError === undefined) {
            delete window.reportError;
        } else {
            window.reportError = originalReportError;
        }
    }
});

test('modal keyboard events do not trigger background document shortcuts', () => {
    let backgroundEscapes = 0;
    let backgroundSpaces = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Escape') backgroundEscapes += 1;
        if (event.key === ' ') backgroundSpaces += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        openModal({ buttons: [{ label: 'Approve', onClick: () => {} }] });
        document.querySelector('.modal-btn').focus();
        keydown(' ');
        assert.equal(backgroundSpaces, 0, 'Space from a modal action never reaches app shortcuts');

        keydown('Escape');
        assert.equal(backgroundEscapes, 0);
        assert.equal(document.querySelector('.modal-overlay'), null);
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
});

test('IME composition Escape stays inside the modal without closing it', () => {
    let backgroundEscapes = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Escape') backgroundEscapes += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        openModal({ content: '<input id="composing-input">' });
        const input = document.getElementById('composing-input');
        const compositionEscape = new dom.window.KeyboardEvent('keydown', {
            key: 'Escape',
            isComposing: true,
            bubbles: true,
            cancelable: true,
        });

        assert.equal(input.dispatchEvent(compositionEscape), true,
            'the input retains the Escape default needed to cancel composition');
        assert.ok(document.querySelector('.modal-overlay'),
            'composition cancellation leaves the dialog open');
        assert.equal(backgroundEscapes, 0,
            'composition Escape remains isolated from background shortcuts');

        keydown('Escape');
        assert.equal(document.querySelector('.modal-overlay'), null,
            'a subsequent non-composing Escape still closes the dialog');
        assert.equal(backgroundEscapes, 0);
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
});

test('Escape stays owned after the focused modal control is removed', () => {
    let backgroundEscapes = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Escape') backgroundEscapes += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        openModal({ content: '<button id="removed-control">Remove me</button>' });
        const focused = document.getElementById('removed-control');
        assert.equal(document.activeElement, focused);
        focused.remove();
        assert.equal(document.activeElement, document.body);

        keydown('Escape');

        assert.equal(backgroundEscapes, 0);
        assert.equal(document.querySelector('.modal-overlay'), null);
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
});

test('Tab restores modal focus after the focused control becomes disabled', () => {
    let backgroundTabs = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Tab') backgroundTabs += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        openModal({ content: '<button id="disabled-control">Disable me</button>' });
        const focused = document.getElementById('disabled-control');
        assert.equal(document.activeElement, focused);
        focused.disabled = true;
        // jsdom retains a disabled element as activeElement. Dispatch from
        // body to model the browser's orphaned key target while preserving
        // that extra stale-focus edge in the assertion path.
        document.body.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Tab',
            bubbles: true,
            cancelable: true,
        }));

        assert.equal(backgroundTabs, 0);
        assert.equal(document.activeElement, document.querySelector('.modal-close-btn'));
        assert.ok(document.querySelector('.modal-overlay'));
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
});

test('child-handled Escape and Tab stay inside the modal without modal behavior', () => {
    let backgroundEscapes = 0;
    let backgroundTabs = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Escape') backgroundEscapes += 1;
        if (event.key === 'Tab') backgroundTabs += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        openModal({ content: '<input id="handled-key-input"><button id="other-control">Other</button>' });
        const input = document.getElementById('handled-key-input');
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' || event.key === 'Tab') event.preventDefault();
        });
        input.focus();

        keydown('Escape');
        assert.ok(document.querySelector('.modal-overlay'), 'handled Escape leaves the modal open');
        assert.equal(backgroundEscapes, 0, 'handled Escape never reaches document shortcuts');

        keydown('Tab');
        assert.equal(document.activeElement, input, 'handled Tab skips the modal focus trap');
        assert.equal(backgroundTabs, 0, 'handled Tab never reaches document shortcuts');
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
});

test('prompt Enter settles and closes without reaching document shortcuts', async () => {
    let backgroundEnters = 0;
    const backgroundHandler = (event) => {
        if (event.key === 'Enter') backgroundEnters += 1;
    };
    document.addEventListener('keydown', backgroundHandler);
    try {
        const result = Modal.prompt('Prompt propagation');
        const input = document.querySelector('.modal-body input');
        input.value = 'owned value';
        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            bubbles: true,
            cancelable: true,
        }));

        assert.equal(await result, 'owned value');
        assert.equal(backgroundEnters, 0);
        assert.equal(document.querySelector('.modal-overlay'), null);
    } finally {
        document.removeEventListener('keydown', backgroundHandler);
    }
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
    assert.equal(document.activeElement, last, 'programmatic outside focus is recaptured immediately');

    Modal.hide();
    assert.equal(document.activeElement, outside, 'focus returns to the original opener');
});
