/**
 * Resources panel functionality - Three-tier API keys, wallet, usage
 *
 * Key Resolution Priority:
 * 1. Agent Keys (highest) - This companion's own keys
 * 2. User BYOK - User's passphrase-encrypted keys (shared across companions)
 * 3. Platform Keys - Vending machine model (wallet + margin)
 */

import API from './api.js';

// Session state for BYOK passphrase (cleared on page refresh)
let byokPassphrase = null;
let byokUnlocked = false;

/**
 * Load all resources data
 *
 * #879: each sub-section is gated by its own capability so an embedded host
 * can keep, say, the wallet visible while opting out of agent-scoped keys.
 * Sections whose cap is false are also hidden from the DOM so the user
 * doesn't see misleading empty cards. Headings live in static HTML and don't
 * carry a capability key, so we tag them by sibling-of the *-list container.
 */
async function loadResources() {
    _hideDisabledResourceSections();

    const tasks = [];
    if (API.hasCapability('keys.agent')) tasks.push(refreshAgentKeys());
    if (API.hasCapability('keys.user')) tasks.push(refreshUserKeys());
    if (API.hasCapability('keys.platform')) tasks.push(refreshPlatformAccess());
    if (API.hasCapability('wallet')) tasks.push(refreshWallet());
    if (API.hasCapability('keys')) {
        // Usage and active-key indicator only make sense when at least one
        // key tier is enabled.
        tasks.push(loadUsage());
        tasks.push(updateActiveKeySource());
    }
    await Promise.all(tasks);
}

// Hide the static section wrappers for sub-sections the host opted out of.
// Idempotent — reads the capability map on every call so the resources
// panel is safe to reload.
function _hideDisabledResourceSections() {
    const checks = [
        { cap: 'keys.agent', listId: 'agent-keys-list' },
        { cap: 'keys.user', listId: 'user-keys-list' },
        { cap: 'keys.platform', listId: 'platform-access-list' },
        { cap: 'wallet', listId: 'wallet-details' },
        { cap: 'keys', listId: 'usage-details' },
        { cap: 'keys', listId: 'active-key-source' },
    ];
    for (const { cap, listId } of checks) {
        const el = document.getElementById(listId);
        if (!el) continue;
        // Walk up to the section wrapper (the immediate child of panel-content
        // for the keys/wallet/usage sections; active-key-source is itself the
        // wrapper).  Falls back to the element itself when the parent isn't a
        // distinct wrapper.
        const wrapper = el.id === 'active-key-source'
            ? el
            : el.closest('div[style*="margin-bottom"]') || el.parentElement || el;
        wrapper.style.display = API.hasCapability(cap) ? '' : 'none';
    }
}

/**
 * Update the active key source indicator
 */
async function updateActiveKeySource() {
    const badge = document.getElementById('key-source-badge');
    if (!badge) return;

    try {
        // Check which key source would be used for the default provider
        const response = await API.request('/api/keys/available-sources?provider=openrouter');

        let source = 'None';
        let color = 'var(--text-secondary)';

        // Use the sources object from the response
        const sources = response.sources || {};

        if (sources.agent) {
            source = kicon('robot') + ' Agent Key';
            color = 'var(--success)';
        } else if (sources.user && byokUnlocked) {
            source = kicon('user') + ' Your Key (BYOK)';
            color = 'var(--accent-color)';
        } else if (sources.platform) {
            source = kicon('building') + ' Platform';
            color = 'var(--warning)';
        }

        badge.innerHTML = source;
        badge.style.background = color;
    } catch (error) {
        badge.textContent = 'Unknown';
        badge.style.background = 'var(--text-secondary)';
    }
}

// ============================================================================
// Agent Keys (this companion only)
// ============================================================================

/**
 * Fetch and display agent API keys
 */
async function refreshAgentKeys() {
    const container = document.getElementById('agent-keys-list');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading keys...</div>';

    try {
        const response = await API.request('/api/keys');

        let html = '<div style="display: flex; flex-direction: column; gap: 0.75rem;">';

        if (!response.keys || response.keys.length === 0) {
            html += `
                <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-tertiary); border-radius: 8px;">
                    No agent keys configured. This companion will use User or Platform keys.
                </div>
            `;
        } else {
            for (const key of response.keys) {
                html += renderKeyCard(key, 'agent');
            }
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading agent keys:', error);
        container.innerHTML = `
            <div style="color: var(--error); padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                Error loading keys: ${error.message}
            </div>
        `;
    }
}

/**
 * Show modal to add an agent API key
 */
function showAddAgentKeyModal() {
    showKeyModal('agent', 'Add Agent Key', 'This key is stored securely for this companion only.');
}

// ============================================================================
// User BYOK Keys (shared across companions)
// ============================================================================

/**
 * Fetch and display user BYOK keys
 */
async function refreshUserKeys() {
    const container = document.getElementById('user-keys-list');
    const unlockBtn = document.getElementById('unlock-byok-btn');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading keys...</div>';

    try {
        const response = await API.request('/api/keys/user');

        let html = '<div style="display: flex; flex-direction: column; gap: 0.75rem;">';

        if (!response.keys || response.keys.length === 0) {
            html += `
                <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-tertiary); border-radius: 8px;">
                    No personal keys added. Add your own API keys to avoid platform fees.
                </div>
            `;
            if (unlockBtn) unlockBtn.style.display = 'none';
        } else {
            // Show unlock button if keys exist but not unlocked
            if (unlockBtn) {
                unlockBtn.style.display = byokUnlocked ? 'none' : 'inline-flex';
            }

            for (const key of response.keys) {
                html += renderUserKeyCard(key);
            }
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading user keys:', error);
        container.innerHTML = `
            <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-tertiary); border-radius: 8px;">
                User keys not available. <a href="#" onclick="showAddUserKeyModal(); return false;">Add your first key</a>
            </div>
        `;
    }
}

/**
 * Render a user key card (shows lock status)
 */
function renderUserKeyCard(key) {
    const statusColor = key.is_active ? 'var(--success)' : 'var(--error)';
    const statusText = key.is_active ? (byokUnlocked ? 'Unlocked' : 'Locked') : 'Inactive';
    const lockIcon = byokUnlocked ? kicon('lock-open') : kicon('lock');

    return `
        <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; border: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; text-transform: capitalize;">${key.provider}</span>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: ${statusColor}; font-size: 0.875rem;">${lockIcon} ${statusText}</span>
                    <button class="btn btn-secondary" style="font-size: 0.7rem; padding: 0.2rem 0.4rem; color: var(--error);"
                            onclick="window.deleteUserKey('${key.provider}')" title="Delete key">
                        ${kicon('trash')}
                    </button>
                </div>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">
                <div>Added: ${key.created_at ? new Date(key.created_at).toLocaleDateString() : 'Unknown'}</div>
                ${key.display_name ? `<div>Name: ${key.display_name}</div>` : ''}
            </div>
            ${!byokUnlocked ? `
                <div style="margin-top: 0.75rem;">
                    <button class="btn btn-primary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;" onclick="showUnlockByokModal()">
                        ${kicon('lock-open')} Unlock to Use
                    </button>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Show modal to add a user BYOK key
 */
function showAddUserKeyModal() {
    // Remove existing modal if any
    const existing = document.getElementById('add-key-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'add-key-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    modal.innerHTML = `
        <div style="
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            <h3 style="margin: 0 0 1rem 0;">Add Your Key (BYOK)</h3>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">
                This key will be encrypted with your passphrase and shared across all your companions.
                You pay the provider directly - no wallet charges.
            </p>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">Provider</label>
                    <select id="add-key-provider" style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                    ">
                        <option value="openrouter">OpenRouter</option>
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="google">Google AI</option>
                    </select>
                </div>
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">API Key</label>
                    <input type="password" id="add-key-value" placeholder="sk-..." style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        box-sizing: border-box;
                    ">
                </div>
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">
                        Encryption Passphrase
                    </label>
                    <input type="password" id="add-key-passphrase" placeholder="Enter passphrase" style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        box-sizing: border-box;
                    ">
                </div>
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">
                        Confirm Passphrase
                    </label>
                    <input type="password" id="add-key-passphrase-confirm" placeholder="Confirm passphrase" style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        box-sizing: border-box;
                    ">
                </div>
            </div>
            <div style="background: var(--warning-bg, rgba(255,200,0,0.1)); border: 1px solid var(--warning, #ffaa00); border-radius: 6px; padding: 0.75rem; margin-top: 1rem; font-size: 0.75rem; color: var(--warning, #ffaa00);">
                ${kicon('warning')} If you forget your passphrase, you'll need to re-add your keys. We cannot recover passphrase-encrypted keys.
            </div>
            <div style="display: flex; gap: 0.75rem; margin-top: 1.5rem; justify-content: flex-end;">
                <button class="btn btn-secondary" onclick="window.closeAddKeyModal()">Cancel</button>
                <button class="btn btn-primary" onclick="window.submitAddUserKey()">Add Key</button>
            </div>
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeAddKeyModal();
    });

    document.body.appendChild(modal);
    document.getElementById('add-key-value').focus();
}

/**
 * Submit a new user BYOK key
 */
async function submitAddUserKey() {
    const provider = document.getElementById('add-key-provider').value;
    const apiKey = document.getElementById('add-key-value').value.trim();
    const passphrase = document.getElementById('add-key-passphrase').value;
    const passphraseConfirm = document.getElementById('add-key-passphrase-confirm').value;

    if (!apiKey) {
        alert('Please enter an API key');
        return;
    }

    if (!passphrase) {
        alert('Please enter a passphrase to encrypt your key');
        return;
    }

    if (passphrase !== passphraseConfirm) {
        alert('Passphrases do not match');
        return;
    }

    if (passphrase.length < 8) {
        alert('Passphrase must be at least 8 characters');
        return;
    }

    try {
        const response = await API.request('/api/keys/user', {
            method: 'POST',
            body: JSON.stringify({
                provider,
                api_key: apiKey,
                passphrase,
            }),
        });

        // Store passphrase in session
        byokPassphrase = passphrase;
        byokUnlocked = true;

        closeAddKeyModal();
        alert(response.message || 'Key added successfully');
        await loadResources();
    } catch (error) {
        alert(`Error adding key: ${error.message}`);
    }
}

/**
 * Show modal to unlock BYOK keys
 */
function showUnlockByokModal() {
    // Remove existing modal if any
    const existing = document.getElementById('unlock-byok-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'unlock-byok-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    modal.innerHTML = `
        <div style="
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            max-width: 350px;
            width: 90%;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            <h3 style="margin: 0 0 1rem 0;">Unlock Your Keys</h3>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">
                Enter your passphrase to use your own API keys.
            </p>
            <div>
                <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Passphrase
                </label>
                <input type="password" id="unlock-passphrase" placeholder="Enter passphrase" style="
                    width: 100%;
                    padding: 0.5rem;
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    box-sizing: border-box;
                ">
            </div>
            <div style="display: flex; gap: 0.75rem; margin-top: 1.5rem; justify-content: flex-end;">
                <button class="btn btn-secondary" onclick="window.closeUnlockByokModal()">Cancel</button>
                <button class="btn btn-primary" onclick="window.submitUnlockByok()">Unlock</button>
            </div>
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeUnlockByokModal();
    });

    document.body.appendChild(modal);
    document.getElementById('unlock-passphrase').focus();

    // Allow enter to submit
    document.getElementById('unlock-passphrase').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') submitUnlockByok();
    });
}

/**
 * Close the unlock BYOK modal
 */
function closeUnlockByokModal() {
    const modal = document.getElementById('unlock-byok-modal');
    if (modal) modal.remove();
}

/**
 * Submit passphrase to unlock BYOK
 */
async function submitUnlockByok() {
    const passphrase = document.getElementById('unlock-passphrase').value;

    if (!passphrase) {
        alert('Please enter your passphrase');
        return;
    }

    try {
        // Verify passphrase by trying to list keys
        const response = await API.request('/api/keys/user/verify', {
            method: 'POST',
            body: JSON.stringify({ passphrase }),
        });

        if (response.valid) {
            byokPassphrase = passphrase;
            byokUnlocked = true;
            closeUnlockByokModal();
            await loadResources();
        } else {
            alert('Invalid passphrase');
        }
    } catch (error) {
        alert(`Error verifying passphrase: ${error.message}`);
    }
}

/**
 * Delete a user BYOK key
 */
async function deleteUserKey(provider) {
    if (!confirm(`Are you sure you want to delete your ${provider} key? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await API.request(`/api/keys/user/${provider}`, {
            method: 'DELETE',
        });

        alert(response.message || 'Key deleted successfully');
        await loadResources();
    } catch (error) {
        alert(`Error deleting key: ${error.message}`);
    }
}

// ============================================================================
// Platform Access (vending machine)
// ============================================================================

/**
 * Fetch and display platform access options
 */
async function refreshPlatformAccess() {
    const container = document.getElementById('platform-access-list');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading platform access...</div>';

    try {
        const response = await API.request('/api/keys/platform');

        let html = '<div style="display: flex; flex-direction: column; gap: 0.75rem;">';

        if (!response.providers || response.providers.length === 0) {
            html += `
                <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-tertiary); border-radius: 8px;">
                    No platform providers available. Add your own keys above.
                </div>
            `;
        } else {
            for (const provider of response.providers) {
                const statusColor = provider.is_available ? 'var(--success)' : 'var(--error)';
                const statusText = provider.is_available ? 'Available' : 'Unavailable';
                // margin_pct comes as a formatted string like "15%"
                const marginText = provider.margin_pct ? `${provider.margin_pct} margin` : '';

                html += `
                    <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; border: 1px solid var(--border-color);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600;">${provider.provider_name || provider.provider_id}</span>
                            <span style="color: ${statusColor}; font-size: 0.875rem;">● ${statusText}</span>
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.875rem; margin-top: 0.5rem;">
                            ${provider.pricing_hint ? `<div>${provider.pricing_hint}</div>` : ''}
                            <div style="display: flex; gap: 1rem; margin-top: 0.25rem;">
                                ${marginText ? `<span>${kicon('credit-card')} ${marginText}</span>` : ''}
                                ${provider.rate_limit_per_companion ? `<span>⏱️ ${provider.rate_limit_per_companion} req/min</span>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading platform access:', error);
        container.innerHTML = `
            <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-tertiary); border-radius: 8px;">
                Platform access information not available.
            </div>
        `;
    }
}

// ============================================================================
// Common Key Functions
// ============================================================================

/**
 * Render a key card (for agent keys)
 */
function renderKeyCard(key, type) {
    const statusColor = key.is_active ? 'var(--success)' : 'var(--error)';
    const statusText = key.is_active ? 'Active' : 'Inactive';
    const quotaPercent = key.quota_limit ? Math.round((key.quota_used / key.quota_limit) * 100) : 0;
    const quotaText = key.quota_limit
        ? `${key.quota_used.toLocaleString()} / ${key.quota_limit.toLocaleString()} units (${quotaPercent}%)`
        : `${key.quota_used.toLocaleString()} units (no limit)`;

    const deleteFunc = type === 'agent' ? 'deleteKey' : 'deleteUserKey';

    return `
        <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; border: 1px solid var(--border-color);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; text-transform: capitalize;">${key.provider}</span>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: ${statusColor}; font-size: 0.875rem;">● ${statusText}</span>
                    <button class="btn btn-secondary" style="font-size: 0.7rem; padding: 0.2rem 0.4rem; color: var(--error);"
                            onclick="window.${deleteFunc}('${key.provider}')" title="Delete key">
                        ${kicon('trash')}
                    </button>
                </div>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.875rem;">
                <div style="margin-bottom: 0.25rem;">Quota: ${quotaText}</div>
                ${key.quota_limit ? `
                    <div style="background: var(--bg-primary); border-radius: 4px; height: 6px; margin: 0.5rem 0;">
                        <div style="background: ${quotaPercent > 80 ? 'var(--error)' : 'var(--accent-color)'}; width: ${Math.min(quotaPercent, 100)}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                ` : ''}
                <div>Added: ${key.created_at ? new Date(key.created_at).toLocaleDateString() : 'Unknown'}</div>
            </div>
            <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem;">
                <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;" onclick="window.viewKeyUsage('${key.provider}')">
                    View Usage
                </button>
                <button class="btn btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;" onclick="window.editKeyQuota('${key.provider}', ${key.quota_limit || 0})">
                    Edit Quota
                </button>
            </div>
        </div>
    `;
}

/**
 * Show generic key modal
 */
function showKeyModal(type, title, description) {
    // Remove existing modal if any
    const existing = document.getElementById('add-key-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'add-key-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;

    modal.innerHTML = `
        <div style="
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            <h3 style="margin: 0 0 0.5rem 0;">${title}</h3>
            <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">${description}</p>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">Provider</label>
                    <select id="add-key-provider" style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                    ">
                        <option value="openrouter">OpenRouter</option>
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="google">Google AI</option>
                        <option value="lighthouse">Lighthouse (IPFS)</option>
                        <option value="runpod">RunPod</option>
                        <option value="vastai">Vast.ai</option>
                        <option value="replicate">Replicate</option>
                        <option value="github">GitHub</option>
                    </select>
                </div>
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">API Key</label>
                    <input type="password" id="add-key-value" placeholder="sk-..." style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        box-sizing: border-box;
                    ">
                </div>
                <div>
                    <label style="display: block; margin-bottom: 0.25rem; font-size: 0.875rem; color: var(--text-secondary);">
                        Monthly Quota (units, optional)
                    </label>
                    <input type="number" id="add-key-quota" placeholder="10000" style="
                        width: 100%;
                        padding: 0.5rem;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        background: var(--bg-primary);
                        color: var(--text-primary);
                        box-sizing: border-box;
                    ">
                    <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.25rem;">
                        Leave empty for unlimited
                    </div>
                </div>
            </div>
            <input type="hidden" id="add-key-type" value="${type}">
            <div style="display: flex; gap: 0.75rem; margin-top: 1.5rem; justify-content: flex-end;">
                <button class="btn btn-secondary" onclick="window.closeAddKeyModal()">Cancel</button>
                <button class="btn btn-primary" onclick="window.submitAddKey()">Add Key</button>
            </div>
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeAddKeyModal();
    });

    document.body.appendChild(modal);
    document.getElementById('add-key-value').focus();
}

/**
 * Close the add key modal
 */
function closeAddKeyModal() {
    const modal = document.getElementById('add-key-modal');
    if (modal) modal.remove();
}

/**
 * Submit the new API key (agent keys)
 */
async function submitAddKey() {
    const provider = document.getElementById('add-key-provider').value;
    const apiKey = document.getElementById('add-key-value').value.trim();
    const quotaInput = document.getElementById('add-key-quota').value;
    const quotaLimit = quotaInput ? parseInt(quotaInput, 10) : null;

    if (!apiKey) {
        alert('Please enter an API key');
        return;
    }

    try {
        const response = await API.request('/api/keys', {
            method: 'POST',
            body: JSON.stringify({
                provider,
                api_key: apiKey,
                quota_limit: quotaLimit,
            }),
        });

        closeAddKeyModal();
        alert(response.message || 'Key added successfully');
        await loadResources();
    } catch (error) {
        alert(`Error adding key: ${error.message}`);
    }
}

/**
 * Delete an agent API key
 */
async function deleteKey(provider) {
    if (!confirm(`Are you sure you want to delete the ${provider} API key? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await API.request(`/api/keys/${provider}`, {
            method: 'DELETE',
        });

        alert(response.message || 'Key deleted successfully');
        await loadResources();
    } catch (error) {
        alert(`Error deleting key: ${error.message}`);
    }
}

/**
 * Edit quota for an API key
 */
async function editKeyQuota(provider, currentQuota) {
    const newQuota = prompt(
        `Enter new monthly quota for ${provider} (current: ${currentQuota || 'unlimited'}).\nLeave empty for unlimited:`,
        currentQuota || ''
    );

    if (newQuota === null) return; // Cancelled

    try {
        const quotaLimit = newQuota.trim() ? parseInt(newQuota, 10) : null;

        const response = await API.request(`/api/keys/${provider}`, {
            method: 'PATCH',
            body: JSON.stringify({ quota_limit: quotaLimit }),
        });

        alert(response.message || 'Quota updated successfully');
        await refreshAgentKeys();
    } catch (error) {
        alert(`Error updating quota: ${error.message}`);
    }
}

// ============================================================================
// Wallet & Usage (existing functionality)
// ============================================================================

/**
 * Fetch and display wallet details
 */
async function refreshWallet() {
    const container = document.getElementById('wallet-details');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading wallet...</div>';

    try {
        const wallet = await API.request('/api/wallet');

        container.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.25rem;">Balance</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">${wallet.balance || 0}</div>
                    <div style="color: var(--text-secondary); font-size: 0.875rem;">${wallet.currency || 'FIL'}</div>
                </div>
                <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.25rem;">Audit Reserve</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">${wallet.audit_reserve || 0}</div>
                    <div style="color: var(--text-secondary); font-size: 0.875rem;">${wallet.currency || 'FIL'}</div>
                </div>
                <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; text-align: center;">
                    <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 0.25rem;">Total</div>
                    <div style="font-size: 1.5rem; font-weight: 600;">${wallet.total || 0}</div>
                    <div style="color: var(--text-secondary); font-size: 0.875rem;">${wallet.currency || 'FIL'}</div>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Error loading wallet:', error);
        container.innerHTML = `
            <div style="color: var(--error); padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                Error loading wallet: ${error.message}
            </div>
        `;
    }
}

/**
 * Load usage summary for all providers
 */
async function loadUsage() {
    const container = document.getElementById('usage-details');
    if (!container) return;

    container.innerHTML = '<div class="loading">Loading usage...</div>';

    try {
        // First get list of keys to know which providers to query
        const keysResponse = await API.request('/api/keys');

        if (!keysResponse.keys || keysResponse.keys.length === 0) {
            container.innerHTML = `
                <div style="color: var(--text-secondary); padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                    No usage data available.
                </div>
            `;
            return;
        }

        let html = '<div style="display: flex; flex-direction: column; gap: 1rem;">';

        for (const key of keysResponse.keys) {
            try {
                const usage = await API.request(`/api/keys/${key.provider}/usage?days=30`);

                const totalUnits = usage.usage.reduce((sum, u) => sum + u.units_consumed, 0);
                const totalCost = usage.usage.reduce((sum, u) => sum + (u.cost_estimate_usd || 0), 0);

                html += `
                    <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem;">
                        <div style="font-weight: 600; text-transform: capitalize; margin-bottom: 0.5rem;">${key.provider}</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; color: var(--text-secondary); font-size: 0.875rem;">
                            <div>
                                <div style="color: var(--text-primary); font-size: 1.25rem; font-weight: 600;">${usage.count}</div>
                                <div>Operations</div>
                            </div>
                            <div>
                                <div style="color: var(--text-primary); font-size: 1.25rem; font-weight: 600;">${totalUnits.toLocaleString()}</div>
                                <div>Units</div>
                            </div>
                            <div>
                                <div style="color: var(--text-primary); font-size: 1.25rem; font-weight: 600;">$${totalCost.toFixed(4)}</div>
                                <div>Est. Cost</div>
                            </div>
                        </div>
                    </div>
                `;
            } catch (e) {
                console.warn(`Error loading usage for ${key.provider}:`, e);
            }
        }

        html += '</div>';
        container.innerHTML = html;

    } catch (error) {
        console.error('Error loading usage:', error);
        container.innerHTML = `
            <div style="color: var(--error); padding: 1rem; background: var(--bg-secondary); border-radius: 8px;">
                Error loading usage: ${error.message}
            </div>
        `;
    }
}

/**
 * View detailed usage for a specific provider
 */
async function viewKeyUsage(provider) {
    try {
        const usage = await API.request(`/api/keys/${provider}/usage?days=30`);

        let message = `Usage for ${provider} (last 30 days):\n\n`;

        if (usage.usage.length === 0) {
            message += 'No usage recorded.';
        } else {
            for (const u of usage.usage.slice(0, 20)) {
                const date = new Date(u.recorded_at).toLocaleString();
                const cost = u.cost_estimate_usd ? `$${u.cost_estimate_usd.toFixed(4)}` : 'N/A';
                message += `${date}: ${u.operation} - ${u.units_consumed} units (${cost})\n`;
            }
            if (usage.usage.length > 20) {
                message += `\n... and ${usage.usage.length - 20} more entries`;
            }
        }

        alert(message);
    } catch (error) {
        alert(`Error loading usage: ${error.message}`);
    }
}

// Legacy compatibility
async function refreshKeys() {
    await refreshAgentKeys();
}

// Expose functions globally for onclick handlers
window.refreshKeys = refreshKeys;
window.refreshAgentKeys = refreshAgentKeys;
window.refreshUserKeys = refreshUserKeys;
window.refreshPlatformAccess = refreshPlatformAccess;
window.refreshWallet = refreshWallet;
window.loadResources = loadResources;
window.viewKeyUsage = viewKeyUsage;
window.showAddKeyModal = () => showKeyModal('agent', 'Add Agent Key', 'This key is stored securely for this companion only.');
window.showAddAgentKeyModal = showAddAgentKeyModal;
window.showAddUserKeyModal = showAddUserKeyModal;
window.closeAddKeyModal = closeAddKeyModal;
window.submitAddKey = submitAddKey;
window.submitAddUserKey = submitAddUserKey;
window.deleteKey = deleteKey;
window.deleteUserKey = deleteUserKey;
window.editKeyQuota = editKeyQuota;
window.showUnlockByokModal = showUnlockByokModal;
window.closeUnlockByokModal = closeUnlockByokModal;
window.submitUnlockByok = submitUnlockByok;

// Export for use in app.js
export { loadResources };
