/**
 * Kestrel Sovereign Console - Database Module
 * Database Explorer
 */

import API from './api.js';
import { state } from './ui.js';
import bus from './ui-ext/bus.js';

// ============================================================================
// Database Explorer
// ============================================================================

state.dbTables = null;
state.dbCurrentTable = null;
state.dbCurrentPage = 0;
state.dbExplorerVisible = false;

// The standalone console and mountPanels embedders can use different API
// clients. Keep the explorer's runtime dependencies explicit so database
// requests never fall back to the process-wide singleton when an embedder
// supplied a mount-scoped client.
let explorerApi = API;
let explorerRoot = document;
let explorerDocument = document;
let detachExplorerRuntime = null;
let awaitingAgentSwitch = false;

function getActiveAgent() {
    return typeof explorerApi.getHostAgent === 'function'
        ? explorerApi.getHostAgent()
        : null;
}

function findExplorerElement(id) {
    if (typeof explorerRoot.getElementById === 'function') {
        return explorerRoot.getElementById(id);
    }
    return explorerRoot.querySelector?.(`#${id}`) || null;
}

function hasSovereigntyCapability() {
    return typeof explorerApi.hasCapability === 'function'
        && explorerApi.hasCapability('sovereignty') === true;
}

state.dbAgent = getActiveAgent();

let dbTablesRequestSeq = 0;
let dbTableRequestSeq = 0;

function resetDbExplorer(agent = getActiveAgent()) {
    // Bump both tokens before clearing the cache so every outstanding response
    // becomes stale, including a request for the same table/page after an
    // A -> B -> A switch.
    dbTablesRequestSeq += 1;
    dbTableRequestSeq += 1;
    state.dbAgent = agent;
    state.dbTables = null;
    state.dbCurrentTable = null;
    state.dbCurrentPage = 0;

    const container = findExplorerElement('db-explorer-container');
    if (container) {
        container.setAttribute(
            'aria-busy',
            String(state.dbExplorerVisible && awaitingAgentSwitch),
        );
        showMessage(
            container,
            state.dbExplorerVisible ? 'Loading database...' : 'Select Browse Database to load tables.',
            'color: var(--text-secondary); padding: 1rem;',
        );
    }
}

function responseBelongsToCurrentAgent(agent) {
    return !awaitingAgentSwitch
        && hasSovereigntyCapability()
        && state.dbAgent === agent
        && getActiveAgent() === agent;
}

function element(tagName, { className = '', text = '', style = '' } = {}) {
    const node = explorerDocument.createElement(tagName);
    if (className) node.className = className;
    if (style) node.style.cssText = style;
    node.textContent = String(text);
    return node;
}

function showMessage(container, message, style = '', role = 'status') {
    const paragraph = element('p', { text: message, style });
    paragraph.setAttribute('role', role);
    container.replaceChildren(paragraph);
}

function formatSize(bytes) {
    const size = Number(bytes);
    if (!Number.isFinite(size) || size < 0) return 'N/A';
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function truncateText(value, maxGraphemes) {
    const text = String(value);
    if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
        const graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })
            .segment(text);
        return Array.from(graphemes, ({ segment }) => segment)
            .slice(0, maxGraphemes)
            .join('');
    }
    // Older hosts still avoid splitting UTF-16 surrogate pairs. Modern hosts
    // take the Segmenter path above, which also preserves combining marks and
    // zero-width-joiner emoji sequences.
    return Array.from(text).slice(0, maxGraphemes).join('');
}

function updateToggleButton(button, visible) {
    const icon = element('span', { className: 'ki ki-cabinet' });
    icon.setAttribute('aria-hidden', 'true');
    const label = element('span', {
        text: visible ? 'Hide Database Explorer' : 'Browse Database',
    });
    if (!visible) label.dataset.labelKey = 'btn_browse_database';
    button.replaceChildren(icon, explorerDocument.createTextNode(' '), label);
    button.type = 'button';
    button.setAttribute('aria-controls', 'db-explorer-section');
    button.setAttribute('aria-expanded', String(visible));
}

function initializeExplorerAccessibility() {
    const toggleButton = findExplorerElement('toggle-db-explorer');
    const section = findExplorerElement('db-explorer-section');
    const container = findExplorerElement('db-explorer-container');

    if (toggleButton) {
        toggleButton.type = 'button';
        toggleButton.setAttribute('aria-controls', 'db-explorer-section');
        toggleButton.setAttribute('aria-expanded', String(state.dbExplorerVisible));
        toggleButton.querySelector?.('.ki')?.setAttribute('aria-hidden', 'true');
    }
    if (section) section.hidden = !state.dbExplorerVisible;
    if (container) container.setAttribute('aria-live', 'polite');
}

function summaryCard(value, label) {
    const card = element('div', {
        style: 'background: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; text-align: center;',
    });
    card.append(
        element('div', {
            text: value,
            style: 'font-size: 1.25rem; font-weight: 600;',
        }),
        element('div', {
            text: label,
            style: 'font-size: 0.7rem; color: var(--text-secondary);',
        }),
    );
    return card;
}

export async function loadDbTables() {
    // setHostAgent() fires before the selected agent's capabilities resolve.
    // Do not cross that boundary until the post-capability agent:switch event
    // confirms the new panel gate.
    if (awaitingAgentSwitch || !hasSovereigntyCapability()) return;

    const requestAgent = getActiveAgent();
    if (state.dbAgent !== requestAgent) resetDbExplorer(requestAgent);
    const requestToken = ++dbTablesRequestSeq;
    const container = findExplorerElement('db-explorer-container');
    container?.setAttribute('aria-busy', 'true');

    try {
        const data = await explorerApi.getDbTables(requestAgent);
        if (
            requestToken !== dbTablesRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
        ) return;
        renderDbExplorer(data);
        state.dbTables = data;
    } catch (error) {
        if (
            requestToken !== dbTablesRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
        ) return;
        if (container) {
            container.setAttribute('aria-busy', 'false');
            showMessage(
                container,
                `Failed to load database: ${error.message}`,
                'color: var(--error); padding: 1rem;',
                'alert',
            );
        }
    }
}

export function renderDbExplorer(data) {
    const container = findExplorerElement('db-explorer-container');
    if (!container) return;
    if (!data || !Array.isArray(data.tables)) {
        throw new TypeError('Invalid database table-list response.');
    }

    const tables = data.tables.filter(
        (table) => table && typeof table.name === 'string',
    );
    container.setAttribute('aria-busy', 'false');

    if (tables.length === 0) {
        showMessage(
            container,
            'No database tables found.',
            'color: var(--text-secondary); text-align: center; padding: 2rem;',
        );
        return;
    }

    const summary = element('div', {
        className: 'db-summary',
        style: 'display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-bottom: 1rem;',
    });
    const totalRows = tables.reduce((sum, table) => {
        const count = Number(table.row_count);
        return Number.isFinite(count) && count >= 0 ? sum + count : sum;
    }, 0);
    summary.append(
        summaryCard(data.table_count ?? tables.length, 'Tables'),
        summaryCard(formatSize(data.db_size), 'DB Size'),
        summaryCard(totalRows, 'Total Rows'),
    );

    const tableList = element('div', {
        className: 'db-table-list',
        style: 'display: flex; flex-direction: column; gap: 0.5rem; max-height: 400px; overflow-y: auto;',
    });
    tableList.setAttribute('aria-label', 'Database tables');

    for (const table of tables) {
        const queryable = table.queryable === true;
        const columns = Array.isArray(table.columns) ? table.columns : [];
        const active = state.dbCurrentTable === table.name;
        const item = element('button', {
            className: 'db-table-item',
            style: `
                display: flex;
                width: 100%;
                justify-content: space-between;
                align-items: center;
                background: ${active ? 'var(--accent-color)' : 'var(--bg-secondary)'};
                color: ${active ? 'white' : 'var(--text-primary)'};
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 0.75rem;
                cursor: ${queryable ? 'pointer' : 'default'};
                opacity: ${queryable ? '1' : '0.6'};
                transition: all 0.2s;
                font: inherit;
                text-align: left;
            `,
        });
        item.type = 'button';
        item.setAttribute('aria-pressed', String(active));

        if (queryable) {
            item.addEventListener('click', () => loadDbTable(table.name));
            item.addEventListener('mouseenter', () => {
                if (state.dbCurrentTable !== table.name) {
                    item.style.borderColor = 'var(--accent-color)';
                }
            });
            item.addEventListener('mouseleave', () => {
                item.style.borderColor = 'var(--border-color)';
            });
        } else {
            item.disabled = true;
        }

        const identity = element('div');
        const tableIdentity = element('div', {
            style: 'font-weight: 500; font-size: 0.85rem;',
        });
        const tableIcon = element('span', {
            text: queryable ? '\u{1F50D}' : '\u{1F512}',
        });
        tableIcon.setAttribute('aria-hidden', 'true');
        tableIdentity.append(
            tableIcon,
            explorerDocument.createTextNode(` ${table.name}`),
        );
        identity.append(
            tableIdentity,
            element('div', {
                text: `${columns.length} columns`,
                style: 'font-size: 0.7rem; opacity: 0.7;',
            }),
        );
        const rowCount = Number(table.row_count);
        const countLabel = Number.isFinite(rowCount) && rowCount >= 0
            ? `${rowCount} rows`
            : 'N/A';
        const count = element('div', {
            text: countLabel,
            style: `
                font-size: 0.75rem;
                background: ${active ? 'rgba(255,255,255,0.2)' : 'var(--bg-tertiary)'};
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
            `,
        });
        item.setAttribute(
            'aria-label',
            `${queryable ? 'View' : 'Unavailable'} table ${table.name}, ${columns.length} columns, ${countLabel}`,
        );
        item.append(identity, count);
        tableList.appendChild(item);
    }

    const viewer = element('div', {
        style: `margin-top: 1rem; display: ${state.dbCurrentTable ? 'block' : 'none'};`,
    });
    viewer.id = 'db-table-viewer';
    viewer.appendChild(element('div', {
        className: 'loading',
        text: 'Select a table to view data...',
    }));

    container.replaceChildren(summary, tableList, viewer);
}

export async function loadDbTable(tableName, page = 0) {
    if (awaitingAgentSwitch || !hasSovereigntyCapability()) return;

    const requestAgent = getActiveAgent();
    if (state.dbAgent !== requestAgent) {
        resetDbExplorer(requestAgent);
        if (state.dbExplorerVisible) void loadDbTables();
        return;
    }

    state.dbCurrentTable = tableName;
    state.dbCurrentPage = page;
    const requestToken = ++dbTableRequestSeq;

    if (state.dbTables) renderDbExplorer(state.dbTables);

    const viewer = findExplorerElement('db-table-viewer');
    if (!viewer) return;

    viewer.style.display = 'block';
    viewer.setAttribute('aria-busy', 'true');
    viewer.replaceChildren(element('div', {
        className: 'loading',
        text: 'Loading table data...',
    }));

    try {
        const limit = 20;
        const offset = page * limit;
        const data = await explorerApi.queryDbTable(tableName, limit, offset, null, requestAgent);

        if (
            requestToken !== dbTableRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
            || state.dbCurrentTable !== tableName
            || state.dbCurrentPage !== page
        ) return;
        if (
            !data
            || data.table !== tableName
            || !Array.isArray(data.columns)
            || !Array.isArray(data.rows)
        ) {
            throw new Error('Database response did not match the requested table.');
        }
        renderDbTableData(data, page, limit, tableName);
    } catch (error) {
        if (
            requestToken !== dbTableRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
            || state.dbCurrentTable !== tableName
            || state.dbCurrentPage !== page
        ) return;
        const currentViewer = findExplorerElement('db-table-viewer');
        if (currentViewer) {
            currentViewer.setAttribute('aria-busy', 'false');
            showMessage(
                currentViewer,
                `Failed to load table: ${error.message}`,
                'color: var(--error);',
                'alert',
            );
        }
    }
}

export function renderDbTableData(data, page, limit, requestedTable = data.table) {
    const viewer = findExplorerElement('db-table-viewer');
    if (!viewer) return;
    if (!data || !Array.isArray(data.columns) || !Array.isArray(data.rows)) {
        throw new TypeError('Invalid database table response.');
    }

    const totalRows = Number(data.total_rows);
    const safeTotalRows = Number.isFinite(totalRows) && totalRows >= 0 ? totalRows : 0;
    const totalPages = Math.ceil(safeTotalRows / limit);
    const header = element('div', {
        style: 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;',
    });
    header.appendChild(element('h4', {
        text: `\u{1F4CA} ${data.table} (${safeTotalRows} rows)`,
        style: 'margin: 0; font-size: 0.9rem;',
    }));

    const pagination = element('div', {
        className: 'db-pagination',
        style: 'display: flex; gap: 0.5rem;',
    });
    pagination.setAttribute('role', 'navigation');
    pagination.setAttribute('aria-label', `Pagination for ${data.table}`);
    const previous = element('button', {
        className: 'btn btn-secondary',
        text: '\u{25C0} Prev',
        style: 'padding: 0.25rem 0.5rem; font-size: 0.75rem;',
    });
    previous.type = 'button';
    previous.disabled = page === 0;
    previous.setAttribute('aria-label', 'Previous database page');
    previous.addEventListener('click', () => loadDbTable(requestedTable, page - 1));

    const next = element('button', {
        className: 'btn btn-secondary',
        text: 'Next \u{25B6}',
        style: 'padding: 0.25rem 0.5rem; font-size: 0.75rem;',
    });
    next.type = 'button';
    next.disabled = !data.has_more;
    next.setAttribute('aria-label', 'Next database page');
    next.addEventListener('click', () => loadDbTable(requestedTable, page + 1));

    const pageStatus = element('span', {
        text: `Page ${page + 1} / ${totalPages || 1}`,
        style: 'font-size: 0.75rem; padding: 0.25rem 0.5rem;',
    });
    pageStatus.setAttribute('aria-live', 'polite');
    pagination.append(
        previous,
        pageStatus,
        next,
    );
    header.appendChild(pagination);

    const tableContainer = element('div', {
        style: 'overflow-x: auto; max-height: 300px; border: 1px solid var(--border-color); border-radius: 8px;',
    });
    tableContainer.setAttribute('role', 'region');
    tableContainer.setAttribute('aria-label', `${data.table} database rows`);
    tableContainer.tabIndex = 0;
    const tableElement = element('table', {
        style: 'width: 100%; border-collapse: collapse; font-size: 0.75rem;',
    });
    const tableHead = element('thead');
    const headingRow = element('tr', {
        style: 'background: var(--bg-tertiary); position: sticky; top: 0;',
    });
    for (const column of data.columns) {
        const heading = element('th', {
            text: column,
            style: 'padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border-color); white-space: nowrap;',
        });
        heading.scope = 'col';
        headingRow.appendChild(heading);
    }
    tableHead.appendChild(headingRow);

    const tableBody = element('tbody');
    if (data.rows.length > 0) {
        for (const row of data.rows) {
            const rowElement = element('tr', {
                style: 'border-bottom: 1px solid var(--border-color);',
            });
            for (const column of data.columns) {
                const cell = element('td', {
                    style: 'padding: 0.5rem; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;',
                });
                const value = row[column];
                if (value === null || value === undefined) {
                    cell.appendChild(element('span', {
                        text: 'null',
                        style: 'color: var(--text-tertiary);',
                    }));
                } else {
                    cell.textContent = truncateText(value, 100);
                }
                rowElement.appendChild(cell);
            }
            tableBody.appendChild(rowElement);
        }
    } else {
        const emptyRow = element('tr');
        const emptyCell = element('td', {
            text: 'No data in this table',
            style: 'padding: 1rem; text-align: center; color: var(--text-secondary);',
        });
        emptyCell.colSpan = Math.max(data.columns.length, 1);
        emptyRow.appendChild(emptyCell);
        tableBody.appendChild(emptyRow);
    }

    tableElement.append(tableHead, tableBody);
    tableContainer.appendChild(tableElement);
    viewer.replaceChildren(header, tableContainer);
    viewer.setAttribute('aria-busy', 'false');
}

export function toggleDbExplorer() {
    const container = findExplorerElement('db-explorer-section');
    const toggleButton = findExplorerElement('toggle-db-explorer');

    if (!container || !toggleButton) return;

    const activeAgent = getActiveAgent();
    if (state.dbAgent !== activeAgent) resetDbExplorer(activeAgent);

    state.dbExplorerVisible = !state.dbExplorerVisible;

    if (state.dbExplorerVisible) {
        container.hidden = false;
        container.style.display = 'block';
        updateToggleButton(toggleButton, true);
        if (!state.dbTables && !awaitingAgentSwitch && hasSovereigntyCapability()) {
            void loadDbTables();
        }
    } else {
        container.style.display = 'none';
        container.hidden = true;
        updateToggleButton(toggleButton, false);
    }
}

function handleExplorerClick(event) {
    const toggleButton = event.target.closest?.('#toggle-db-explorer');
    if (!toggleButton) return;

    // Older hosts embed the explorer button with
    // onclick="toggleDbExplorer()" (or assign the equivalent onclick
    // property). That handler runs at the target before this delegated
    // listener sees the event; invoking the toggle again would immediately
    // undo it. Delegation owns only buttons that have no compatibility
    // handler of their own.
    if (toggleButton.hasAttribute('onclick') || typeof toggleButton.onclick === 'function') {
        return;
    }
    toggleDbExplorer();
}

function explorerCanReload() {
    const section = findExplorerElement('db-explorer-section');
    return state.dbExplorerVisible
        && section !== null
        && section.style.display !== 'none'
        && hasSovereigntyCapability();
}

function handleApiAgentChange(activeAgent) {
    if (state.dbAgent === activeAgent) return;
    // This is the earliest tenant boundary. Evict the previous agent's rows and
    // invalidate in-flight responses synchronously, but do not fetch for the
    // new route yet: its capability map has not resolved.
    awaitingAgentSwitch = true;
    resetDbExplorer(activeAgent);
}

function handleAgentSwitch({ next } = {}) {
    const activeAgent = getActiveAgent();

    // Rapid selections can leave an older or malformed host event queued behind
    // the current route. Only a switch that names the active API agent may
    // release the capability barrier. External routers intentionally keep that
    // API value null, so their payload remains opaque and every event is
    // authoritative.
    if (activeAgent !== null && next !== activeAgent) return;

    const routeChangedWithoutApiHook = state.dbAgent !== activeAgent;

    // A non-null route was already invalidated synchronously through
    // onHostAgentChange. A null route is different: embedders may select the
    // agent in their own router while leaving their API in standalone mode, so
    // each event remains a tenant boundary even though the visible route is
    // unchanged.
    if (activeAgent === null || routeChangedWithoutApiHook) {
        resetDbExplorer(activeAgent);
    }

    const wasAwaitingSwitch = awaitingAgentSwitch;
    awaitingAgentSwitch = false;
    if (
        (wasAwaitingSwitch || activeAgent === null || routeChangedWithoutApiHook)
        && explorerCanReload()
    ) {
        void loadDbTables();
    }
}

function handlePanelHidden({ panelId } = {}) {
    if (panelId !== 'sovereignty') return;

    // Capability gating removes and later recreates the panel body. Keep the
    // module state aligned with the newly mounted explorer's hidden DOM and
    // invalidate any response targeting the detached panel.
    state.dbExplorerVisible = false;
    resetDbExplorer(getActiveAgent());
}

/**
 * Bind the explorer to one console/embed mount.
 *
 * @param {{api?: object, root?: ParentNode}} [options]
 * @returns {() => void} teardown callback
 */
export function initDatabaseExplorer({ api = API, root = document } = {}) {
    if (!root || typeof root.addEventListener !== 'function') {
        throw new TypeError('initDatabaseExplorer requires an event-capable root');
    }
    if (detachExplorerRuntime) detachExplorerRuntime();

    explorerApi = api;
    explorerRoot = root;
    explorerDocument = root.nodeType === 9
        ? root
        : (root.ownerDocument || document);
    awaitingAgentSwitch = false;
    state.dbExplorerVisible = false;
    resetDbExplorer(getActiveAgent());
    initializeExplorerAccessibility();

    root.addEventListener('click', handleExplorerClick);
    const agentChangeSubscription = typeof api.onHostAgentChange === 'function'
        ? api.onHostAgentChange(handleApiAgentChange)
        : null;
    const offAgentChange = typeof agentChangeSubscription === 'function'
        ? agentChangeSubscription
        : () => {};
    const offAgentSwitch = bus.on('agent:switch', handleAgentSwitch);
    const offPanelHidden = bus.on('panel:hidden', handlePanelHidden);
    let attached = true;

    const detach = () => {
        if (!attached) return;
        attached = false;
        root.removeEventListener('click', handleExplorerClick);
        offAgentChange();
        offAgentSwitch();
        offPanelHidden();
        awaitingAgentSwitch = false;
        state.dbExplorerVisible = false;
        resetDbExplorer(getActiveAgent());
        if (detachExplorerRuntime === detach) detachExplorerRuntime = null;
    };
    detachExplorerRuntime = detach;
    return detach;
}

// Retain the public functions for embedders that call them directly. The
// explorer itself uses bound listeners and never constructs inline handlers.
window.loadDbTable = loadDbTable;
window.toggleDbExplorer = toggleDbExplorer;
