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
state.dbAgent = API.getHostAgent();

let dbTablesRequestSeq = 0;
let dbTableRequestSeq = 0;

function resetDbExplorer(agent = API.getHostAgent()) {
    // Bump both tokens before clearing the cache so every outstanding response
    // becomes stale, including a request for the same table/page after an
    // A -> B -> A switch.
    dbTablesRequestSeq += 1;
    dbTableRequestSeq += 1;
    state.dbAgent = agent;
    state.dbTables = null;
    state.dbCurrentTable = null;
    state.dbCurrentPage = 0;

    const container = document.getElementById('db-explorer-container');
    if (container) {
        showMessage(
            container,
            state.dbExplorerVisible ? 'Loading database...' : 'Select Browse Database to load tables.',
            'color: var(--text-secondary); padding: 1rem;',
        );
    }
}

function responseBelongsToCurrentAgent(agent) {
    return state.dbAgent === agent && API.getHostAgent() === agent;
}

function element(tagName, { className = '', text = '', style = '' } = {}) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (style) node.style.cssText = style;
    node.textContent = String(text);
    return node;
}

function showMessage(container, message, style = '') {
    const paragraph = element('p', { text: message, style });
    container.replaceChildren(paragraph);
}

function formatSize(bytes) {
    const size = Number(bytes);
    if (!Number.isFinite(size) || size < 0) return 'N/A';
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(2)} MB`;
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
    const requestAgent = API.getHostAgent();
    if (state.dbAgent !== requestAgent) resetDbExplorer(requestAgent);
    const requestToken = ++dbTablesRequestSeq;

    try {
        const data = await API.getDbTables();
        if (
            requestToken !== dbTablesRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
        ) return;
        state.dbTables = data;
        renderDbExplorer(data);
    } catch (error) {
        if (
            requestToken !== dbTablesRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
        ) return;
        const container = document.getElementById('db-explorer-container');
        if (container) {
            showMessage(
                container,
                `Failed to load database: ${error.message}`,
                'color: var(--error); padding: 1rem;',
            );
        }
    }
}

export function renderDbExplorer(data) {
    const container = document.getElementById('db-explorer-container');
    if (!container) return;

    if (!data.tables || data.tables.length === 0) {
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
    const totalRows = data.tables.reduce((sum, table) => {
        const count = Number(table.row_count);
        return Number.isFinite(count) && count >= 0 ? sum + count : sum;
    }, 0);
    summary.append(
        summaryCard(data.table_count ?? data.tables.length, 'Tables'),
        summaryCard(formatSize(data.db_size), 'DB Size'),
        summaryCard(totalRows, 'Total Rows'),
    );

    const tableList = element('div', {
        className: 'db-table-list',
        style: 'display: flex; flex-direction: column; gap: 0.5rem; max-height: 400px; overflow-y: auto;',
    });

    for (const table of data.tables) {
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
                cursor: ${table.queryable ? 'pointer' : 'default'};
                opacity: ${table.queryable ? '1' : '0.6'};
                transition: all 0.2s;
                font: inherit;
                text-align: left;
            `,
        });
        item.type = 'button';

        if (table.queryable) {
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
        identity.append(
            element('div', {
                text: `${table.queryable ? '\u{1F50D}' : '\u{1F512}'} ${table.name}`,
                style: 'font-weight: 500; font-size: 0.85rem;',
            }),
            element('div', {
                text: `${table.columns.length} columns`,
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
    const requestAgent = API.getHostAgent();
    if (state.dbAgent !== requestAgent) {
        resetDbExplorer(requestAgent);
        if (state.dbExplorerVisible) void loadDbTables();
        return;
    }

    state.dbCurrentTable = tableName;
    state.dbCurrentPage = page;
    const requestToken = ++dbTableRequestSeq;

    if (state.dbTables) renderDbExplorer(state.dbTables);

    const viewer = document.getElementById('db-table-viewer');
    if (!viewer) return;

    viewer.style.display = 'block';
    viewer.replaceChildren(element('div', {
        className: 'loading',
        text: 'Loading table data...',
    }));

    try {
        const limit = 20;
        const offset = page * limit;
        const data = await API.queryDbTable(tableName, limit, offset);

        if (
            requestToken !== dbTableRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
            || state.dbCurrentTable !== tableName
            || state.dbCurrentPage !== page
        ) return;
        renderDbTableData(data, page, limit);
    } catch (error) {
        if (
            requestToken !== dbTableRequestSeq
            || !responseBelongsToCurrentAgent(requestAgent)
            || state.dbCurrentTable !== tableName
            || state.dbCurrentPage !== page
        ) return;
        const currentViewer = document.getElementById('db-table-viewer');
        if (currentViewer) {
            showMessage(
                currentViewer,
                `Failed to load table: ${error.message}`,
                'color: var(--error);',
            );
        }
    }
}

export function renderDbTableData(data, page, limit) {
    const viewer = document.getElementById('db-table-viewer');
    if (!viewer) return;

    const totalPages = Math.ceil(data.total_rows / limit);
    const header = element('div', {
        style: 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;',
    });
    header.appendChild(element('h4', {
        text: `\u{1F4CA} ${data.table} (${data.total_rows} rows)`,
        style: 'margin: 0; font-size: 0.9rem;',
    }));

    const pagination = element('div', {
        className: 'db-pagination',
        style: 'display: flex; gap: 0.5rem;',
    });
    const previous = element('button', {
        className: 'btn btn-secondary',
        text: '\u{25C0} Prev',
        style: 'padding: 0.25rem 0.5rem; font-size: 0.75rem;',
    });
    previous.type = 'button';
    previous.disabled = page === 0;
    previous.addEventListener('click', () => loadDbTable(data.table, page - 1));

    const next = element('button', {
        className: 'btn btn-secondary',
        text: 'Next \u{25B6}',
        style: 'padding: 0.25rem 0.5rem; font-size: 0.75rem;',
    });
    next.type = 'button';
    next.disabled = !data.has_more;
    next.addEventListener('click', () => loadDbTable(data.table, page + 1));

    pagination.append(
        previous,
        element('span', {
            text: `Page ${page + 1} / ${totalPages || 1}`,
            style: 'font-size: 0.75rem; padding: 0.25rem 0.5rem;',
        }),
        next,
    );
    header.appendChild(pagination);

    const tableContainer = element('div', {
        style: 'overflow-x: auto; max-height: 300px; border: 1px solid var(--border-color); border-radius: 8px;',
    });
    const tableElement = element('table', {
        style: 'width: 100%; border-collapse: collapse; font-size: 0.75rem;',
    });
    const tableHead = element('thead');
    const headingRow = element('tr', {
        style: 'background: var(--bg-tertiary); position: sticky; top: 0;',
    });
    for (const column of data.columns) {
        headingRow.appendChild(element('th', {
            text: column,
            style: 'padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border-color); white-space: nowrap;',
        }));
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
                    cell.textContent = String(value).substring(0, 100);
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
}

export function toggleDbExplorer() {
    const container = document.getElementById('db-explorer-section');
    const toggleButton = document.getElementById('toggle-db-explorer');

    if (!container || !toggleButton) return;

    const activeAgent = API.getHostAgent();
    if (state.dbAgent !== activeAgent) resetDbExplorer(activeAgent);

    state.dbExplorerVisible = !state.dbExplorerVisible;

    if (state.dbExplorerVisible) {
        container.style.display = 'block';
        toggleButton.textContent = '\u{1F5C4}\u{FE0F} Hide Database Explorer';
        if (!state.dbTables) loadDbTables();
    } else {
        container.style.display = 'none';
        toggleButton.textContent = '\u{1F5C4}\u{FE0F} Browse Database';
    }
}

document.addEventListener('click', (event) => {
    if (event.target.closest?.('#toggle-db-explorer')) toggleDbExplorer();
});

bus.on('agent:switch', () => {
    resetDbExplorer(API.getHostAgent());
    if (state.dbExplorerVisible) void loadDbTables();
});

// Retain the public functions for embedders that call them directly. The
// explorer itself uses bound listeners and never constructs inline handlers.
window.loadDbTable = loadDbTable;
window.toggleDbExplorer = toggleDbExplorer;
