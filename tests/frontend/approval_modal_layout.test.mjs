// #2349: the Permission Required modal layout must not overflow/clip.
// Five action buttons (Deny / Once / Session / Always / Auto:Session /
// Auto:Always) can never fit one non-wrapping row at the 480px modal max-width,
// and the JSON args preview must wrap/scroll instead of truncating mid-string.
// jsdom can't compute real layout, so we assert the CSS contract that makes the
// browser wrap (flex-wrap on the footer, pre-wrap + scroll on the args block)
// and that the full args JSON is present in the DOM (never truncated).
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

const { Modal, setOverlayRoot } = await import('../../kestrel_sovereign/static/js/ui.js');
const { Security } = await import('../../kestrel_sovereign/static/js/security.js');

function normStyle(el) {
    return (el.getAttribute('style') || '').replace(/\s+/g, ' ');
}

test('action row wraps inside the modal instead of overflowing (flex-wrap)', () => {
    const p = Security.showApprovalModal({
        id: 'req-1', feature: 'image', tool: 'generate', args: {},
    });

    const footer = document.querySelector('.modal-overlay .modal-footer');
    assert.ok(footer, 'modal footer rendered');
    const style = normStyle(footer);
    assert.match(style, /flex-wrap:\s*wrap/, 'footer must wrap so buttons never bleed past the modal edge');
    assert.match(style, /display:\s*flex/);

    // All five decision/auto actions render (plus Deny) — six buttons total.
    const buttons = document.querySelectorAll('.modal-overlay .modal-footer .modal-btn');
    assert.equal(buttons.length, 6, 'Deny + This Time + Session + Always + Auto:Session + Auto:Always');
    // Shrinkable: min-width:0 lets flex shrink buttons at narrow embed widths.
    for (const btn of buttons) {
        assert.match(normStyle(btn), /min-width:\s*0/, 'buttons must be shrinkable');
    }

    Modal.hide();
    p.catch(() => {}); // resolver settles on close; nothing to await here
});

test('JSON args preview wraps + scrolls and is never truncated', () => {
    const longPrompt = 'Generate a cozy indoors selfie of a kestrel wearing a tiny wool sweater by the fireplace';
    const p = Security.showApprovalModal({
        id: 'req-2', feature: 'image', tool: 'generate',
        args: { prompt: longPrompt, size: '1024x1024' },
    });

    const pre = document.querySelector('.modal-overlay .args-preview');
    assert.ok(pre, 'args preview rendered');
    const style = normStyle(pre);
    assert.match(style, /white-space:\s*pre-wrap/, 'preview must wrap long lines');
    assert.match(style, /word-break:\s*break-word/, 'preview must break unspaced strings');
    assert.match(style, /overflow-y:\s*auto/, 'preview must scroll vertically');
    assert.match(style, /max-height:/, 'preview must be height-bounded');

    // The full prompt must be present — not clipped mid-string.
    assert.ok(pre.textContent.includes(longPrompt), 'the full args value is visible, not truncated');

    Modal.hide();
    p.catch(() => {});
});

test('the modal is a bounded flex column with a SCROLLABLE body — wrapped footers stay reachable (codex P1)', async () => {
    // Six wrapped action rows + an upgrade banner on a short viewport must
    // never push the footer past max-height:90vh with overflow:hidden — the
    // body scrolls instead, and header/footer are non-shrinking flex items.
    const { Modal } = await import('../../kestrel_sovereign/static/js/ui.js');
    Modal.show({
        title: 'Layout probe',
        content: '<div style="height: 4000px">tall body</div>',
        buttons: [
            { label: 'Deny', type: 'secondary', onClick: () => {} },
            { label: 'Once', type: 'primary', onClick: () => {} },
            { label: 'Always', type: 'primary', onClick: () => {} },
            { label: 'Auto: Session', type: 'danger', onClick: () => {} },
            { label: 'Auto: Always', type: 'danger', onClick: () => {} },
        ],
    });
    const container = document.querySelector('.modal-container');
    assert.ok(container, 'modal rendered');
    assert.equal(container.style.display, 'flex');
    assert.equal(container.style.flexDirection, 'column');
    const body = container.querySelector('.modal-body');
    assert.equal(body.style.overflowY, 'auto', 'body scrolls when content exceeds the bound');
    assert.equal(body.style.minHeight, '0px', 'body may shrink so the footer stays in view');
    const footer = container.querySelector('.modal-footer');
    assert.match(footer.style.flex, /0 0 auto/, 'footer never shrinks/clips');
    Modal.hide();
});
