/**
 * Feature Store — JSON Schema to form renderer and feature management UI.
 *
 * Renders settings forms from a feature's config_schema (JSON Schema),
 * supporting text inputs, number inputs, toggles, and dropdowns.
 */

// ── API helpers ──────────────────────────────────────────────────────

async function fetchFeatures() {
    const resp = await fetch('/api/features/installed');
    if (!resp.ok) throw new Error(`Failed to fetch features: ${resp.status}`);
    return (await resp.json()).features;
}

async function fetchFeatureDetail(name) {
    const resp = await fetch(`/api/features/${encodeURIComponent(name)}`);
    if (!resp.ok) throw new Error(`Failed to fetch feature detail: ${resp.status}`);
    return resp.json();
}

async function fetchFeatureConfig(name) {
    const resp = await fetch(`/api/features/${encodeURIComponent(name)}/config`);
    if (!resp.ok) throw new Error(`Failed to fetch config: ${resp.status}`);
    return resp.json();
}

async function patchFeatureConfig(name, config) {
    const resp = await fetch(`/api/features/${encodeURIComponent(name)}/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Config update failed: ${resp.status}`);
    }
    return resp.json();
}

// ── Schema-to-Form renderer ─────────────────────────────────────────

/**
 * Render a JSON Schema into an HTML form inside the given container.
 *
 * Supported property types:
 *   - string           → <input type="text"> (or <select> if enum)
 *   - integer / number → <input type="number"> (with min/max)
 *   - boolean          → <input type="checkbox"> (toggle)
 *   - string + enum    → <select>
 *
 * @param {HTMLElement} container  DOM element to render into
 * @param {object}      schema     JSON Schema (type: "object")
 * @param {object}      values     Current config values
 * @param {function}    onChange   Called with (key, newValue) on change
 */
function renderSchemaForm(container, schema, values, onChange) {
    container.innerHTML = '';
    const properties = schema.properties || {};
    const required = new Set(schema.required || []);

    for (const [key, prop] of Object.entries(properties)) {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'config-field';

        const label = document.createElement('label');
        label.className = 'config-label';
        label.textContent = _humanize(key);
        if (required.has(key)) {
            const star = document.createElement('span');
            star.className = 'config-required';
            star.textContent = ' *';
            label.appendChild(star);
        }

        if (prop.description) {
            const desc = document.createElement('div');
            desc.className = 'config-description';
            desc.textContent = prop.description;
            label.appendChild(desc);
        }

        fieldDiv.appendChild(label);

        const currentValue = values[key] !== undefined ? values[key] : prop.default;
        const input = _createInput(key, prop, currentValue, onChange);
        fieldDiv.appendChild(input);

        container.appendChild(fieldDiv);
    }
}

function _createInput(key, prop, value, onChange) {
    // Enum → dropdown
    if (prop.enum) {
        const select = document.createElement('select');
        select.className = 'config-input config-select';
        select.dataset.key = key;
        for (const option of prop.enum) {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            if (option === value) opt.selected = true;
            select.appendChild(opt);
        }
        select.addEventListener('change', () => {
            onChange(key, select.value);
        });
        return select;
    }

    // Boolean → toggle
    if (prop.type === 'boolean') {
        const wrapper = document.createElement('label');
        wrapper.className = 'config-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = !!value;
        checkbox.dataset.key = key;
        checkbox.addEventListener('change', () => {
            onChange(key, checkbox.checked);
        });
        const slider = document.createElement('span');
        slider.className = 'config-toggle-slider';
        wrapper.appendChild(checkbox);
        wrapper.appendChild(slider);
        return wrapper;
    }

    // Integer / number → number input
    if (prop.type === 'integer' || prop.type === 'number') {
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'config-input';
        input.dataset.key = key;
        if (value !== undefined && value !== null) input.value = value;
        if (prop.minimum !== undefined) input.min = prop.minimum;
        if (prop.maximum !== undefined) input.max = prop.maximum;
        if (prop.type === 'integer') input.step = '1';
        input.addEventListener('change', () => {
            const parsed = prop.type === 'integer' ? parseInt(input.value, 10) : parseFloat(input.value);
            if (!isNaN(parsed)) onChange(key, parsed);
        });
        return input;
    }

    // Default: string → text input
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'config-input';
    input.dataset.key = key;
    if (value !== undefined && value !== null) input.value = value;
    if (prop.maxLength) input.maxLength = prop.maxLength;
    input.addEventListener('change', () => {
        onChange(key, input.value);
    });
    return input;
}

function _humanize(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ── Feature Store panel rendering ────────────────────────────────────

/**
 * Load and render the Feature Store panel.
 * Called when the user navigates to the Features tab.
 */
async function loadFeatureStore() {
    const panel = document.getElementById('feature-store-content');
    if (!panel) return;

    panel.innerHTML = '<div class="loading"><div class="spinner"></div>Loading features...</div>';

    try {
        const features = await fetchFeatures();
        _renderFeatureList(panel, features);
    } catch (err) {
        panel.innerHTML = `<div class="error">Failed to load features: ${err.message}</div>`;
    }
}

function _renderFeatureList(panel, features) {
    panel.innerHTML = '';

    if (!features.length) {
        panel.innerHTML = '<p style="color:var(--text-secondary);padding:1rem;">No features loaded.</p>';
        return;
    }

    const grid = document.createElement('div');
    grid.className = 'feature-grid';

    for (const feat of features) {
        const card = document.createElement('div');
        card.className = 'feature-card';
        card.innerHTML = `
            <div class="feature-card-header">
                <span class="feature-icon">${feat.icon ? _iconHtml(feat.icon) : '&#x1F9E9;'}</span>
                <span class="feature-name">${_esc(feat.name)}</span>
                ${feat.core ? '<span class="feature-badge core">core</span>' : ''}
            </div>
            <div class="feature-desc">${_esc(feat.description)}</div>
            <div class="feature-tools-count">${feat.tools ? feat.tools.length : 0} skill${feat.tools && feat.tools.length !== 1 ? 's' : ''}</div>
        `;
        card.addEventListener('click', () => _showFeatureDetail(panel, feat.name));
        grid.appendChild(card);
    }

    panel.appendChild(grid);
}

async function _showFeatureDetail(panel, name) {
    panel.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

    try {
        const [detail, configResp] = await Promise.all([
            fetchFeatureDetail(name),
            fetchFeatureConfig(name),
        ]);

        panel.innerHTML = '';

        // Back button
        const backBtn = document.createElement('button');
        backBtn.className = 'btn btn-secondary';
        backBtn.textContent = 'Back';
        backBtn.style.marginBottom = '1rem';
        backBtn.addEventListener('click', () => loadFeatureStore());
        panel.appendChild(backBtn);

        // Header
        const header = document.createElement('div');
        header.className = 'feature-detail-header';
        header.innerHTML = `
            <h3>${_esc(detail.name || name)}</h3>
            <p style="color:var(--text-secondary)">${_esc(detail.description || '')}</p>
        `;
        panel.appendChild(header);

        // Skills list
        if (detail.tools && detail.tools.length) {
            const skillsSection = document.createElement('div');
            skillsSection.className = 'feature-section';
            skillsSection.innerHTML = `<h4>Skills</h4>`;
            const skillList = document.createElement('ul');
            skillList.className = 'feature-skill-list';
            for (const t of detail.tools) {
                const li = document.createElement('li');
                li.innerHTML = `<strong>${_esc(t.name)}</strong> &mdash; ${_esc(t.description)}`;
                skillList.appendChild(li);
            }
            skillsSection.appendChild(skillList);
            panel.appendChild(skillsSection);
        }

        // Config form
        const schema = configResp.config_schema || detail.config_schema;
        const config = configResp.config || {};

        if (schema && schema.properties && Object.keys(schema.properties).length) {
            const configSection = document.createElement('div');
            configSection.className = 'feature-section';
            configSection.innerHTML = `<h4>Configuration</h4>`;

            const formContainer = document.createElement('div');
            formContainer.className = 'config-form';

            const pendingChanges = {};
            renderSchemaForm(formContainer, schema, config, (key, val) => {
                pendingChanges[key] = val;
            });

            configSection.appendChild(formContainer);

            // Save button
            const saveBtn = document.createElement('button');
            saveBtn.className = 'btn btn-primary';
            saveBtn.textContent = 'Save Configuration';
            saveBtn.style.marginTop = '1rem';
            saveBtn.addEventListener('click', async () => {
                saveBtn.disabled = true;
                saveBtn.textContent = 'Saving...';
                try {
                    const merged = { ...config, ...pendingChanges };
                    await patchFeatureConfig(name, merged);
                    saveBtn.textContent = 'Saved!';
                    setTimeout(() => {
                        saveBtn.textContent = 'Save Configuration';
                        saveBtn.disabled = false;
                    }, 1500);
                } catch (err) {
                    saveBtn.textContent = 'Save Configuration';
                    saveBtn.disabled = false;
                    alert('Failed to save: ' + err.message);
                }
            });
            configSection.appendChild(saveBtn);

            panel.appendChild(configSection);
        }

    } catch (err) {
        panel.innerHTML = `
            <button class="btn btn-secondary" onclick="loadFeatureStore()">Back</button>
            <div class="error" style="margin-top:1rem;">Failed to load feature: ${_esc(err.message)}</div>
        `;
    }
}

function _esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function _iconHtml(icon) {
    // If icon is a single emoji or short string, use it directly
    if (icon.length <= 2) return icon;
    // Otherwise treat as a named icon (future: icon library lookup)
    return '&#x1F9E9;';
}

// Expose to global scope for tab switching
window.loadFeatureStore = loadFeatureStore;
window.renderSchemaForm = renderSchemaForm;
