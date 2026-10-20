import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import {
    readFileSync,
    readdirSync,
} from 'node:fs';
import {
    dirname,
    extname,
    join,
    relative,
} from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const staticDir = join(repoRoot, 'kestrel_sovereign', 'static');
const css = readFileSync(join(staticDir, 'index.css'), 'utf8');
const approvals = readFileSync(join(staticDir, 'js', 'approvals.js'), 'utf8');

function declarations(source) {
    const values = new Map();
    for (const match of source.matchAll(/(--[A-Za-z0-9_-]+)\s*:\s*([^;}]+);/g)) {
        values.set(match[1], match[2].trim());
    }
    return values;
}

function rootDeclarations(source) {
    const values = new Map();
    for (const match of source.matchAll(/:root\s*\{([^{}]*)\}/g)) {
        for (const [name, value] of declarations(match[1])) values.set(name, value);
    }
    return values;
}

// Third-party documents shipped as pinned reference DATA — the offline
// W3C/IETF specification registry — are not first-party UI. They arrive as
// their generator's output (ReSpec), carrying its own stylesheet and token
// vocabulary (--heading-text, --bg-color, ...) which this theme contract
// deliberately does not own and must not be pressured into declaring.
// Vendored trees are named one by one on purpose: a newly vendored corpus
// lands outside this set and fails the scan until it is classified here.
const vendoredAssetRoots = new Set([
    join(repoRoot, 'kestrel_sovereign', 'data', 'semantic', 'standards'),
]);

function firstPartyAssets(directory) {
    const assets = new Map();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) {
            if (vendoredAssetRoots.has(path)) continue;
            for (const [nestedPath, source] of firstPartyAssets(path)) {
                assets.set(nestedPath, source);
            }
        } else if (['.css', '.html', '.js', '.mjs'].includes(extname(entry.name))) {
            assets.set(relative(repoRoot, path), readFileSync(path, 'utf8'));
        }
    }
    return assets;
}

function undeclaredRequiredTokens(assets, contract) {
    const missing = [];
    for (const [path, source] of assets) {
        // Tokens with a fallback are deliberately optional extension points.
        // A bare var(--token) is a hard dependency on the supported contract.
        for (const match of source.matchAll(/var\(\s*(--[A-Za-z0-9_-]+)\s*\)/g)) {
            if (!contract.has(match[1])) missing.push(`${path}: ${match[1]}`);
        }
    }
    return missing;
}

function assertDeclaredRequiredTokens(assets, contract) {
    const missing = undeclaredRequiredTokens(assets, contract);
    if (missing.length) {
        throw new Error(`Undeclared required design tokens:\n${missing.join('\n')}`);
    }
}

const darkMediaStart = css.indexOf('@media (prefers-color-scheme: dark)');
assert.notEqual(darkMediaStart, -1, 'dark theme override exists');
const defaultTheme = rootDeclarations(css.slice(0, darkMediaStart));
const darkOverrides = rootDeclarations(css.slice(darkMediaStart));

test('default and dark themes use the canonical accent and border token names', () => {
    assert.equal(defaultTheme.get('--accent-color'), '#3b82f6');
    assert.equal(defaultTheme.get('--border-color'), '#e2e8f0');
    assert.equal(darkOverrides.get('--accent-color'), '#60a5fa');
    assert.equal(darkOverrides.get('--border-color'), '#334155');
    assert.equal(defaultTheme.has('--accent'), false, 'no second accent alias layer');
    assert.equal(defaultTheme.has('--border'), false, 'no second border alias layer');

    for (const token of darkOverrides.keys()) {
        assert.ok(defaultTheme.has(token), `${token} override has a default value`);
    }
});

test('scoped theme overrides can replace canonical values without aliases', () => {
    const dom = new JSDOM(`<style>${css}</style>
        <div id="embed" style="--accent-color: #7c3aed; --border-color: #475569;">
            <button id="control">Scoped control</button>
        </div>`, { pretendToBeVisual: true });
    const rootStyle = dom.window.getComputedStyle(dom.window.document.documentElement);
    const controlStyle = dom.window.getComputedStyle(
        dom.window.document.getElementById('control'),
    );

    assert.equal(rootStyle.getPropertyValue('--accent-color'), '#3b82f6');
    assert.equal(rootStyle.getPropertyValue('--border-color'), '#e2e8f0');
    assert.equal(controlStyle.getPropertyValue('--accent-color'), '#7c3aed');
    assert.equal(controlStyle.getPropertyValue('--border-color'), '#475569');
    assert.equal(controlStyle.getPropertyValue('--accent'), '');
    assert.equal(controlStyle.getPropertyValue('--border'), '');
});

test('avatar and approval controls reference visible canonical theme states', () => {
    assert.match(
        css,
        /\.avatar-actions button:hover\s*\{[^}]*background:\s*var\(--accent-color\);[^}]*border-color:\s*var\(--accent-color\);/s,
    );
    assert.match(
        css,
        /\.avatar-generate-panel button\s*\{[^}]*background:\s*var\(--accent-color\);/s,
    );
    assert.match(
        css,
        /\.avatar-options img:hover\s*\{[^}]*border-color:\s*var\(--accent-color\);/s,
    );
    assert.equal(
        [...approvals.matchAll(/border:\s*1px solid var\(--border-color\)/g)].length,
        3,
        'pending approvals, remembered rules, and audit rows all retain a visible border',
    );
});

test('required-token check rejects a synthetic unknown variable', () => {
    assert.throws(
        () => assertDeclaredRequiredTokens(
            new Map([['synthetic.css', '.probe { color: var(--not-in-theme-contract); }']]),
            defaultTheme,
        ),
        /synthetic\.css: --not-in-theme-contract/,
    );
});

function shippedFirstPartyAssets() {
    return new Map([
        ...firstPartyAssets(join(repoRoot, 'kestrel_sovereign')),
        ...firstPartyAssets(join(repoRoot, 'control-panel')),
        ...firstPartyAssets(join(repoRoot, 'examples')),
    ]);
}

test('every shipped first-party var(--token) without a fallback is declared', () => {
    assert.doesNotThrow(
        () => assertDeclaredRequiredTokens(shippedFirstPartyAssets(), defaultTheme),
    );
});

// The vendored-root skip above is an exemption from a contract this repo owns,
// so it has to stay exactly as wide as its justification. This pins both edges:
// first-party UI outside static/ is still scanned, and only the vendored corpus
// is dropped -- so widening the skip to swallow real UI fails here.
test('the vendored skip drops only third-party data, never first-party UI', () => {
    const scanned = [...shippedFirstPartyAssets().keys()];

    for (const firstParty of [
        join('kestrel_sovereign', 'static', 'index.css'),
        join('kestrel_sovereign', 'static', 'js', 'approvals.js'),
        join('kestrel_sovereign', 'features', 'spawn', 'static', 'spawn.js'),
    ]) {
        assert.ok(scanned.includes(firstParty), `${firstParty} is still scanned`);
    }

    const vendored = join('kestrel_sovereign', 'data', 'semantic', 'standards');
    assert.equal(
        scanned.filter((path) => path.startsWith(vendored)).length,
        0,
        'the pinned W3C/IETF specification corpus is not treated as first-party UI',
    );
});
