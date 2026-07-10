/**
 * Create Agent dialog (#2351)
 * ============================================================================
 * The standalone console's "+ New" agent affordance opens THIS dialog, which
 * creates a fresh top-level (parentless) agent via `POST /api/agents`
 * (endpoints/models.py::create_agent — name-validated, inception + load, 409 on
 * duplicates). This is a distinct intent from Spawn, which creates a CHILD of an
 * existing parent; the Spawn path stays reachable but SECONDARY, offered as a
 * link inside this dialog when the spawn capability is present.
 *
 * Dependencies are injected so the flow is hermetically testable and so the
 * shared `Modal` (which mounts into the configurable overlay root, #2233)
 * renders correctly in scoped embeds too.
 */

// Client-side mirror of the server's `_AGENT_NAME_RE`
// (endpoints/models.py) so a bad name gives instant feedback before the POST.
const AGENT_NAME_RE = /^[A-Za-z][A-Za-z0-9_-]*$/;

const INPUT_ID = 'create-agent-name-input';
const ERROR_ID = 'create-agent-error';
const SPAWN_LINK_ID = 'create-agent-spawn-link';

/**
 * Open the Create Agent dialog.
 *
 * @param {object}   deps
 * @param {object}   deps.modal          - shared Modal helper (`show`/`hide`).
 * @param {object}   deps.api            - API client exposing `createAgent(name)`.
 * @param {Function} deps.onCreated      - async cb(name) run after a successful
 *                                          create (refresh the list + select).
 * @param {boolean} [deps.spawnAvailable] - when true, render the secondary
 *                                          "Spawn a child agent…" link.
 * @param {Function} [deps.onSpawn]      - cb() invoked when the spawn link is
 *                                          clicked (routes to the Spawn tab).
 */
export function openCreateAgentDialog({ modal, api, onCreated, spawnAvailable = false, onSpawn } = {}) {
    if (!modal || !api) {
        throw new Error('openCreateAgentDialog requires { modal, api }');
    }

    const inputStyle = `
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-size: 1rem;
        background: var(--bg-primary);
        color: var(--text-primary);
        outline: none;
        transition: border-color 0.2s;
    `.replace(/\s+/g, ' ').trim();

    const spawnLink = spawnAvailable
        ? `<div style="margin-top: 1.25rem; font-size: 0.8125rem; color: var(--text-secondary);">
               Looking to create a CHILD of an existing agent?
               <a href="#" id="${SPAWN_LINK_ID}" style="color: var(--accent-color); text-decoration: none;">Spawn a child agent…</a>
           </div>`
        : '';

    const content = `
        <label for="${INPUT_ID}" style="display: block; margin-bottom: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">Agent name</label>
        <input type="text" id="${INPUT_ID}"
            placeholder="e.g. Kestrel"
            autocomplete="off"
            spellcheck="false"
            style="${inputStyle}"
            onfocus="this.style.borderColor='var(--accent-color)'"
            onblur="this.style.borderColor='var(--border-color)'" />
        <div id="${ERROR_ID}" role="alert" style="display: none; margin-top: 0.625rem; color: var(--error); font-size: 0.8125rem; line-height: 1.4;"></div>
        ${spawnLink}
    `;

    let submitting = false;

    const showError = (message) => {
        const el = document.getElementById(ERROR_ID);
        if (el) {
            // textContent (not innerHTML) so a server-supplied detail string can
            // never inject markup into the dialog.
            el.textContent = message;
            el.style.display = 'block';
        }
    };

    const clearError = () => {
        const el = document.getElementById(ERROR_ID);
        if (el) {
            el.textContent = '';
            el.style.display = 'none';
        }
    };

    const submit = async () => {
        if (submitting) return;
        const input = document.getElementById(INPUT_ID);
        const name = (input?.value || '').trim();
        if (!name) {
            showError('Agent name is required.');
            return;
        }
        if (!AGENT_NAME_RE.test(name)) {
            showError('Agent name must start with a letter and contain only letters, numbers, hyphens, or underscores.');
            return;
        }
        submitting = true;
        clearError();
        // Scope the async completion to THIS dialog instance (codex P2): the
        // user can dismiss the dialog while createAgent is in flight and open
        // another modal — the eventual resolution must not hide that unrelated
        // modal or paint errors into a reopened Create Agent. The name input
        // element is unique to this render; if it's gone or detached, the
        // dialog this request belonged to no longer exists.
        const dialogInput = input;
        const stillCurrent = () => !!(dialogInput && dialogInput.isConnected
            && document.getElementById(INPUT_ID) === dialogInput);
        try {
            await api.createAgent(name);
            // Success: close the dialog first, then let the host refresh the list
            // and select the freshly-minted agent. The refresh/select still runs
            // even if the user dismissed the dialog mid-flight — the agent WAS
            // created; only the modal.hide() must be scoped.
            if (stillCurrent()) modal.hide();
            if (onCreated) await onCreated(name);
        } catch (err) {
            // Surface the 409/400 (or any) failure inline — never a toast, so the
            // user can correct the name in place without losing the dialog. But
            // only into the SAME dialog instance that submitted.
            submitting = false;
            if (!stillCurrent()) return;
            const detail = (err && ((err.body && err.body.detail) || err.message)) || 'Failed to create agent.';
            showError(detail);
        }
    };

    modal.show({
        title: 'Create Agent',
        content,
        buttons: [
            { label: 'Cancel', type: 'secondary', onClick: () => modal.hide() },
            { label: 'Create', type: 'primary', onClick: () => { submit(); } },
        ],
    });

    // Wire post-render behavior (Enter-to-submit, live error clear, spawn link).
    // Deferred so it runs after Modal.show has appended the content.
    setTimeout(() => {
        const input = document.getElementById(INPUT_ID);
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    submit();
                }
            });
            input.addEventListener('input', clearError);
        }
        const link = document.getElementById(SPAWN_LINK_ID);
        if (link) {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                modal.hide();
                if (onSpawn) onSpawn();
            });
        }
    }, 0);
}
