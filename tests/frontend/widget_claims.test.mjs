// Generic shared-singleton claim/release/refresh contract (#2047, ticket 09).
//
// The model selector exposes a `WidgetClaimRegistry` so a feature can
// temporarily seize the shared widget ("while my session is live, this control
// is mine, then I give it back"). Voice's realtime-session takeover is the only
// claimant today; these tests pin the generic contract — acquire/release/
// refresh, single-holder reject on double-acquire, auto-release when the
// claiming capability is disabled — plus the model-selector pin behavior voice
// is reimplemented on top of, and that a SECOND widget could be claimed with the
// same contract (genericity).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/index.js', import.meta.url),
    'utf8',
);

function createOption() {
    const opt = {
        value: '',
        textContent: '',
        dataset: {},
        disabled: false,
        _parent: null,
        remove() {
            if (this._parent) {
                const i = this._parent.options.indexOf(this);
                if (i >= 0) this._parent.options.splice(i, 1);
            }
        },
    };
    return opt;
}

function createSelect() {
    const attrs = {};
    return {
        value: '',
        innerHTML: '',
        style: {},
        disabled: false,
        title: '',
        options: [],
        _attrs: attrs,
        addEventListener() {},
        setAttribute(k, v) { attrs[k] = v; },
        removeAttribute(k) { delete attrs[k]; },
        getAttribute(k) { return attrs[k]; },
        appendChild(opt) {
            opt._parent = this;
            this.options.push(opt);
        },
    };
}

function load() {
    const providerSelect = createSelect();
    const routeSelect = createSelect();
    const modelSelect = createSelect();
    const storage = new Map();

    const context = {
        console: { warn() {}, error() {}, log() {} },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'provider-selector') return providerSelect;
                if (id === 'route-selector') return routeSelect;
                if (id === 'model-selector') return modelSelect;
                return null;
            },
            createElement(tag) {
                if (tag === 'option') return createOption();
                return { tagName: tag };
            },
        },
        localStorage: {
            getItem(key) { return storage.has(key) ? storage.get(key) : null; },
            setItem(key, value) { storage.set(key, String(value)); },
            removeItem(key) { storage.delete(key); },
        },
        fetch: async () => { throw new Error('unexpected fetch'); },
        setTimeout,
        clearTimeout,
    };

    vm.runInNewContext(source, context, { filename: 'model-selector/index.js' });
    return {
        WidgetClaimRegistry: context.window.WidgetClaimRegistry,
        ModelSelector: context.window.SharedModelSelector,
        providerSelect,
        routeSelect,
        modelSelect,
        storage,
    };
}

function newSelector(ModelSelector) {
    return new ModelSelector({
        providerSelectId: 'provider-selector',
        routeSelectId: 'route-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });
}

// ---------------------------------------------------------------------------
// WidgetClaimRegistry — generic contract (no DOM needed; widget is opaque)
// ---------------------------------------------------------------------------

test('acquire runs onAcquire and marks the widget held', () => {
    const { WidgetClaimRegistry } = load();
    const widget = { name: 'fake' };
    const reg = new WidgetClaimRegistry(widget);
    const seen = [];

    const ok = reg.acquire('voice', { onAcquire: (w) => seen.push(w) });

    assert.equal(ok, true);
    assert.equal(reg.isHeld(), true);
    assert.equal(reg.heldBy(), 'voice');
    assert.equal(reg.has('voice'), true);
    assert.deepEqual(seen, [widget]);
});

test('double-acquire by a different holder is rejected (single-holder reject)', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    let secondAcquired = false;

    assert.equal(reg.acquire('voice', {}), true);
    const ok = reg.acquire('other', { onAcquire: () => { secondAcquired = true; } });

    assert.equal(ok, false, 'second holder must be rejected');
    assert.equal(secondAcquired, false, 'rejected acquire must NOT run onAcquire');
    assert.equal(reg.heldBy(), 'voice', 'original holder keeps the claim');
});

test('re-acquire by the SAME holder is an idempotent success without re-running onAcquire', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    let acquires = 0;

    reg.acquire('voice', { onAcquire: () => { acquires += 1; } });
    const ok = reg.acquire('voice', { onAcquire: () => { acquires += 1; } });

    assert.equal(ok, true);
    assert.equal(acquires, 1, 'same-holder re-acquire is a no-op success');
});

test('release runs onRelease and clears the holder', () => {
    const { WidgetClaimRegistry } = load();
    const widget = { name: 'fake' };
    const reg = new WidgetClaimRegistry(widget);
    const released = [];

    reg.acquire('voice', { onRelease: (w) => released.push(w) });
    reg.release('voice');

    assert.equal(reg.isHeld(), false);
    assert.equal(reg.heldBy(), null);
    assert.deepEqual(released, [widget]);
});

test('release is idempotent and ignores a non-matching claimId', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    let releases = 0;

    reg.acquire('voice', { onRelease: () => { releases += 1; } });
    reg.release('someone-else');   // wrong id — must NOT release
    assert.equal(reg.isHeld(), true);
    assert.equal(releases, 0);

    reg.release('voice');
    reg.release('voice');          // already released — no-op
    reg.release();                 // no holder — no-op
    assert.equal(releases, 1);
});

test('refresh runs onRefresh on the current holder, and is a no-op when unheld', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    let refreshes = 0;

    reg.refresh();   // nothing held — no-op
    assert.equal(refreshes, 0);

    reg.acquire('voice', { onRefresh: () => { refreshes += 1; } });
    reg.refresh();
    reg.refresh();
    assert.equal(refreshes, 2);
});

test('auto-releases when the claiming capability is disabled', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    let released = 0;

    reg.acquire('voice', { capability: 'voice', onRelease: () => { released += 1; } });

    // Still present → no release.
    reg.onCapabilitiesChanged({ capabilities: { voice: true, chat: true } });
    assert.equal(reg.isHeld(), true);
    assert.equal(released, 0);

    // Capability flips false → auto-release.
    reg.onCapabilitiesChanged({ capabilities: { voice: false, chat: true } });
    assert.equal(reg.isHeld(), false);
    assert.equal(released, 1);
});

test('capability removed entirely from the map also auto-releases', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    reg.acquire('voice', { capability: 'voice' });

    reg.onCapabilitiesChanged({ capabilities: { chat: true } });  // voice absent
    assert.equal(reg.isHeld(), false);
});

test('a bare/empty capabilities payload does NOT release (cannot prove loss)', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    reg.acquire('voice', { capability: 'voice' });

    reg.onCapabilitiesChanged(null);
    reg.onCapabilitiesChanged({});
    reg.onCapabilitiesChanged({ capabilities: null });
    assert.equal(reg.isHeld(), true);
});

test('a claim without a declared capability is never auto-released', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});
    reg.acquire('voice', {});   // no capability

    reg.onCapabilitiesChanged({ capabilities: { voice: false } });
    assert.equal(reg.isHeld(), true);
});

test('a throwing callback is isolated and does not wedge the registry', () => {
    const { WidgetClaimRegistry } = load();
    const reg = new WidgetClaimRegistry({});

    assert.equal(reg.acquire('voice', { onAcquire: () => { throw new Error('boom'); } }), true);
    // Despite onAcquire throwing, the claim is held and release still works.
    assert.equal(reg.isHeld(), true);
    reg.release('voice');
    assert.equal(reg.isHeld(), false);
});

// Genericity: the same contract guards a DIFFERENT widget. No second feature
// exists yet, but a second widget with its own registry behaves independently —
// proving the contract is not model-selector-specific.
test('a second, distinct widget can be claimed with the same contract', () => {
    const { WidgetClaimRegistry } = load();
    const widgetA = { id: 'model-selector' };
    const widgetB = { id: 'some-other-shared-control' };
    const regA = new WidgetClaimRegistry(widgetA);
    const regB = new WidgetClaimRegistry(widgetB);
    const acquiredOn = [];

    regA.acquire('voice', { onAcquire: (w) => acquiredOn.push(w.id) });
    regB.acquire('captions', { capability: 'captions', onAcquire: (w) => acquiredOn.push(w.id) });

    assert.deepEqual(acquiredOn, ['model-selector', 'some-other-shared-control']);
    // Independent holders, independent auto-release.
    regB.onCapabilitiesChanged({ capabilities: { captions: false } });
    assert.equal(regA.isHeld(), true, 'A is untouched by B losing its capability');
    assert.equal(regB.isHeld(), false);
});

// ---------------------------------------------------------------------------
// Model-selector pin — the mechanism voice's takeover is reimplemented on top of
// ---------------------------------------------------------------------------

test('pinToModel stores prior selection, pins the model, and disables the selects', () => {
    const { ModelSelector, providerSelect, modelSelect } = load();
    const sel = newSelector(ModelSelector);
    sel._populateModels = () => {};   // isolate pin from option-list rebuild
    sel.selectedProvider = 'anthropic';
    sel.selectedModel = 'claude-opus-4-7';

    sel.pinToModel(
        { vendor: 'openai', model: 'gpt-realtime-2', label: '🎙 gpt-realtime-2' },
        'voice owns this',
    );

    assert.equal(sel.isPinned(), true);
    assert.equal(modelSelect.value, 'gpt-realtime-2');
    assert.equal(providerSelect.value, 'openai');
    assert.equal(modelSelect.disabled, true);
    assert.equal(providerSelect.disabled, true);
    assert.equal(modelSelect.getAttribute('aria-disabled'), 'true');
    // Target absent from the (empty) option list → a tagged marker is injected.
    const injected = modelSelect.options.find(o => o.value === 'gpt-realtime-2');
    assert.ok(injected, 'a marker option is injected for the unknown model');
    assert.equal(injected.dataset.claimInjected, 'true');
    assert.equal(injected.textContent, '🎙 gpt-realtime-2');
});

test('unpinSelection restores the prior selection, re-enables, and drops the injected option', () => {
    const { ModelSelector, providerSelect, modelSelect } = load();
    const sel = newSelector(ModelSelector);
    sel._populateModels = () => {};
    sel.selectedProvider = 'anthropic';
    sel.selectedModel = 'claude-opus-4-7';

    sel.pinToModel({ vendor: 'openai', model: 'gpt-realtime-2' }, 'voice owns this');
    sel.unpinSelection();

    assert.equal(sel.isPinned(), false);
    assert.equal(modelSelect.disabled, false);
    assert.equal(providerSelect.disabled, false);
    assert.equal(providerSelect.getAttribute('aria-disabled'), undefined);
    assert.equal(sel.selectedProvider, 'anthropic');
    assert.equal(sel.selectedModel, 'claude-opus-4-7');
    assert.equal(modelSelect.options.some(o => o.dataset.claimInjected === 'true'), false,
        'injected marker option is removed on release');
});

test('a held claim re-pins via onRefresh when the option list rebuilds', () => {
    const { ModelSelector, modelSelect } = load();
    const sel = newSelector(ModelSelector);
    sel.selectedProvider = 'anthropic';
    sel.selectedModel = 'claude-opus-4-7';

    // Voice-style claim: pin on acquire, re-pin on refresh.
    const pin = (w) => w.pinToModel({ vendor: 'openai', model: 'gpt-realtime-2' }, 'owned');
    sel.claims.acquire('voice', { onAcquire: pin, onRefresh: pin });
    assert.equal(modelSelect.value, 'gpt-realtime-2');

    // Simulate the dropdown rebuilding its options out from under the claim
    // (e.g. a model-list reload): the injected marker is wiped.
    modelSelect.options.length = 0;
    modelSelect.value = '';
    // The claim must re-assert. Drive it the way _populateModels does.
    sel.claims.refresh();

    assert.equal(modelSelect.value, 'gpt-realtime-2', 'claim re-pins after rebuild');
    assert.ok(modelSelect.options.find(o => o.value === 'gpt-realtime-2'));
});

test('the model selector wires acquire→pin and disable-driven release end to end', () => {
    const { ModelSelector, modelSelect } = load();
    const sel = newSelector(ModelSelector);
    sel._populateModels = () => {};
    sel.selectedProvider = 'anthropic';
    sel.selectedModel = 'claude-opus-4-7';

    const pin = (w) => w.pinToModel({ vendor: 'openai', model: 'gpt-realtime-2' }, 'owned');
    sel.claims.acquire('voice', {
        capability: 'voice',
        onAcquire: pin,
        onRefresh: pin,
        onRelease: (w) => w.unpinSelection(),
    });
    assert.equal(sel.isPinned(), true);
    assert.equal(modelSelect.disabled, true);

    // Disabling the voice capability auto-releases the claim, which unpins.
    sel.claims.onCapabilitiesChanged({ capabilities: { voice: false } });
    assert.equal(sel.claims.isHeld(), false);
    assert.equal(sel.isPinned(), false);
    assert.equal(modelSelect.disabled, false);
    assert.equal(sel.selectedModel, 'claude-opus-4-7', 'prior selection restored');
});
