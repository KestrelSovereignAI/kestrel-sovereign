// #2822: the panel descriptor is the single source of truth for nav tab chrome.
//
// Nav tabs are born two ways: BUILT by the registry (embeds) or declared
// in-place in index.html and ADOPTED by the registry (standalone console).
// Adoption used to stamp only the gating flag — it never looked at the
// descriptor's icon or label — so a descriptor-only change reached embeds and
// silently skipped standalone. That drift is why the `ki-check` typo had to be
// fixed in two files, and why nothing failed when the two disagreed.
//
// Also guarded here: the reason a tab's `data-label-key` must live on the label
// span and never on the button. KestrelTheme._hydrate() assigns
// `el.textContent` for every `[data-label-key]` element; on a button that
// replaces the entire subtree, destroying the icon AND the pending-count badge.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;

/** Fresh nav + panel host, with `navHtml` as the pre-existing in-place strip. */
function mount(navHtml = '') {
    Panels._reset();
    document.body.innerHTML = '';
    const nav = document.createElement('div');
    nav.className = 'nav-tabs';
    nav.innerHTML = navHtml;
    const host = document.createElement('div');
    host.className = 'main-content';
    document.body.append(nav, host);
    return { nav, host };
}

/** The label-hydration pass from theme.js, reproduced exactly (textContent). */
function hydrate(labels) {
    document.querySelectorAll('[data-label-key]').forEach((el) => {
        const key = el.getAttribute('data-label-key');
        if (key && Object.prototype.hasOwnProperty.call(labels, key)) {
            el.textContent = labels[key];
        }
    });
}

// ---- adoption reconciles chrome, not just gating ---------------------------

test('a descriptor icon reaches an ADOPTED in-place tab', () => {
    // The pre-#2821 markup shape: key + text on the button, no icon at all.
    const { nav, host } = mount(
        '<button class="nav-tab" data-panel="identity" data-label-key="tab_identity">Identity</button>',
    );
    const tab = nav.querySelector('.nav-tab');

    Panels.registerPanel({
        panelId: 'identity',
        label: 'Identity',
        labelKey: 'tab_identity',
        icon: 'ki ki-fingerprint',
        gate: () => true,
    });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });

    // Same node — adoption must not replace the tab (listeners, position).
    assert.equal(nav.querySelector('.nav-tab'), tab, 'the in-place tab node must be adopted, not rebuilt');
    const icon = tab.querySelector('.nav-tab-icon');
    assert.ok(icon, 'adoption must give the in-place tab its descriptor icon');
    assert.ok(icon.className.includes('ki-fingerprint'), `got "${icon.className}"`);
    assert.equal(tab.querySelector('.nav-tab-label').textContent, 'Identity');
});

test('a BUILT tab and an ADOPTED tab produce the same chrome', () => {
    const def = {
        panelId: 'memories', label: 'Memories', labelKey: 'tab_memories',
        icon: 'ki ki-brain', gate: () => true,
    };

    const built = mount('');
    Panels.registerPanel(def);
    Panels.renderNav({ navEl: built.nav, hostEl: built.host, ctx: {} });
    const builtTab = built.nav.querySelector('.nav-tab');
    const builtShape = builtTab.innerHTML;

    const adopted = mount(
        '<button class="nav-tab" data-panel="memories" data-label-key="tab_memories">Memories</button>',
    );
    Panels.registerPanel(def);
    Panels.renderNav({ navEl: adopted.nav, hostEl: adopted.host, ctx: {} });
    const adoptedTab = adopted.nav.querySelector('.nav-tab');

    assert.equal(adoptedTab.innerHTML, builtShape, 'both paths must yield identical tab chrome');
});

test('an icon-less contribution falls back to the default glyph, never bare text', () => {
    // Observability registers exactly like this from an out-of-tree package.
    const { nav, host } = mount('');
    Panels.registerPanel({ panelId: 'observability', label: 'Observability', gate: () => true });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });

    const icon = nav.querySelector('.nav-tab .nav-tab-icon');
    assert.ok(icon, 'a contribution with no icon must still get one');
    assert.ok(/\bki-[a-z0-9-]+\b/.test(icon.className), `got "${icon.className}"`);
});

// ---- what reconciliation must not break ------------------------------------

test('badge elements are preserved BY NODE across adoption', () => {
    const { nav, host } = mount(`
        <button class="nav-tab" data-panel="approvals">
            <span data-label-key="tab_approvals">Approvals</span>
            <span id="approvals-pending-badge">7</span>
        </button>`);
    const badgeBefore = document.getElementById('approvals-pending-badge');

    Panels.registerPanel({
        panelId: 'approvals', label: 'Approvals', labelKey: 'tab_approvals',
        icon: 'ki ki-check-square', gate: () => true,
    });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });

    const badgeAfter = document.getElementById('approvals-pending-badge');
    assert.equal(badgeAfter, badgeBefore, 'the badge must be moved, not re-created');
    assert.equal(badgeAfter.textContent, '7', 'a count rendered before nav sync must survive');
    assert.equal(badgeAfter.parentNode, nav.querySelector('.nav-tab'), 'badge stays inside its tab');
});

test('label hydration no longer destroys a tab\'s icon and badge', () => {
    const { nav, host } = mount(
        '<button class="nav-tab" data-panel="security" data-label-key="tab_security">Security</button>',
    );
    Panels.registerPanel({
        panelId: 'security', label: 'Security', labelKey: 'tab_security',
        icon: 'ki ki-lock', gate: () => true,
    });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });

    const tab = nav.querySelector('.nav-tab');
    const badge = document.createElement('span');
    badge.id = 'security-pending-badge';
    badge.textContent = '3';
    tab.appendChild(badge);

    assert.equal(tab.hasAttribute('data-label-key'), false, 'the key must not sit on the button');

    hydrate({ tab_security: 'Sicherheit' });

    assert.ok(tab.querySelector('.nav-tab-icon'), 'the icon must survive hydration');
    assert.equal(tab.querySelector('.nav-tab-label').textContent, 'Sicherheit', 'the label translates');
    assert.equal(document.getElementById('security-pending-badge').textContent, '3',
        'the badge must survive hydration');
});

test('an already-translated label is not reverted to the English descriptor', () => {
    // Standalone hydrates before the registry binds the nav, so adoption sees a
    // tab whose text is already in the active locale.
    const { nav, host } = mount(
        '<button class="nav-tab" data-panel="tasks" data-label-key="tab_tasks">Aufgaben</button>',
    );
    Panels.registerPanel({
        panelId: 'tasks', label: 'Tasks', labelKey: 'tab_tasks',
        icon: 'ki ki-list-checks', gate: () => true,
    });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });

    assert.equal(nav.querySelector('.nav-tab-label').textContent, 'Aufgaben',
        'the descriptor label is the English fallback, not an override');
});

test('reconciliation is idempotent across repeated syncs', () => {
    const { nav, host } = mount(
        '<button class="nav-tab" data-panel="features" data-label-key="tab_features">Features</button>',
    );
    Panels.registerPanel({
        panelId: 'features', label: 'Features', labelKey: 'tab_features',
        icon: 'ki ki-puzzle', gate: () => true,
    });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: {} });
    const once = nav.querySelector('.nav-tab').innerHTML;

    Panels.syncNav();
    Panels.syncNav();

    assert.equal(nav.querySelectorAll('.nav-tab .nav-tab-icon').length, 1, 'no icon duplication');
    assert.equal(nav.querySelector('.nav-tab').innerHTML, once, 'chrome is stable across syncs');
});
