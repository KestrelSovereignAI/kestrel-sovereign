import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM(`<!doctype html><html><body>
    <div id="panel-tasks">
        <select id="task-filter"><option value="">All</option></select>
        <button id="btn-refresh-tasks"></button>
        <button id="btn-view-tasks"></button>
        <button id="btn-view-activity"></button>
        <span id="tasks-badge"></span>
        <div id="tasks-container"><div id="task-list"></div></div>
        <div id="activity-container"><div id="activity-list"></div></div>
    </div>
</body></html>`, { url: 'http://localhost/' });

globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.MutationObserver = dom.window.MutationObserver;
globalThis.location = dom.window.location;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.localStorage = dom.window.localStorage;
globalThis.kicon = (name) => `<span class="ki ki-${name}"></span>`;
globalThis.window.kicon = globalThis.kicon;

const API = (await import('../../kestrel_sovereign/static/js/api.js')).default;
const { initTasks, loadTasks } = await import(
    '../../kestrel_sovereign/static/js/tasks.js'
);

const statuses = [
    'queued',
    'running',
    'completed',
    'failed',
    'cancelled',
    'unknown',
    // Live A2A wire names retain their existing API spelling while sharing
    // the canonical visual variants above.
    'submitted',
    'working',
    'canceled',
    'input_required',
];

API.request = async () => ({
    tasks: statuses.map((status, index) => ({
        id: `task-${String(index).padStart(8, '0')}`,
        status,
        agent_id: 'Kite',
        skill: 'test',
        artifacts_count: 0,
    })),
});

initTasks();
await loadTasks();

function badge(status) {
    return [...document.querySelectorAll('.status-badge')]
        .find((element) => element.textContent === status);
}

test('task DOM renders every canonical status as a class-based badge variant', () => {
    const expected = {
        queued: 'status-queued',
        running: 'status-running',
        completed: 'status-completed',
        failed: 'status-failed',
        cancelled: 'status-cancelled',
        unknown: 'status-unknown',
    };

    for (const [status, className] of Object.entries(expected)) {
        const element = badge(status);
        assert.ok(element, `${status} badge rendered`);
        assert.ok(element.classList.contains(className), `${status} uses .${className}`);
        assert.equal(
            element.hasAttribute('style'),
            false,
            `${status} presentation comes from CSS, not an inline style`,
        );
    }
});

test('A2A wire statuses map onto the canonical visual vocabulary', () => {
    const expected = {
        submitted: 'status-queued',
        working: 'status-running',
        canceled: 'status-cancelled',
        input_required: 'status-input-required',
    };

    for (const [status, className] of Object.entries(expected)) {
        assert.ok(badge(status)?.classList.contains(className), `${status} uses .${className}`);
    }
});

test('running and working tasks both contribute to the working-task count', () => {
    const count = document.getElementById('tasks-badge');
    assert.equal(count.textContent, '2');
    assert.equal(count.style.display, 'inline-block');
});
