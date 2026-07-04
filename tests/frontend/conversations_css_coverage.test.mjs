// #2159: the #2149 conversation-list component shipped without any CSS for its
// new class names — the view bar rendered as bare concatenated text, tiles had
// no hierarchy, the kebab was an unstyled default button. This is the
// regression gate: every class name emitted by conversations.js / kebab_menu.js
// must have a matching selector in index.css.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const staticDir = join(here, '..', '..', 'kestrel_sovereign', 'static');
const css = readFileSync(join(staticDir, 'index.css'), 'utf8');

// Every class name the two modules assign to a rendered element. Kept as an
// explicit list (rather than scraped) so a newly-emitted-but-unstyled class is
// caught the moment someone adds it here alongside the JS change.
const EMITTED_CLASSES = [
    // conversations.js
    'conversations-root',
    'conversations-controls',
    'conversations-view-bar',
    'conversations-view-btn',
    'conversations-search',
    'conversations-stats',
    'conversations-list-body',
    'date-group',
    'date-group-label',
    'conversation-item',
    'conversation-meta-row',
    'conversation-time',
    'conversation-msg-count',
    'conversation-preview',
    'conversation-rename-input',
    'conversations-error',
    'empty-state',
    // kebab_menu.js
    'kebab-btn',
    'kebab-menu',
    'kebab-menu-item',
    'kebab-menu-item-danger',
    'kebab-menu-separator',
];

test('every emitted component class has a selector in index.css', () => {
    const missing = EMITTED_CLASSES.filter((cls) => {
        // Match `.cls` as a whole class token (followed by a non-identifier
        // char), so `.kebab-menu` doesn't falsely satisfy `.kebab-menu-item`.
        const re = new RegExp(`\\.${cls.replace(/[-]/g, '\\-')}(?![\\w-])`);
        return !re.test(css);
    });
    assert.deepEqual(missing, [], `index.css is missing styles for: ${missing.join(', ')}`);
});
