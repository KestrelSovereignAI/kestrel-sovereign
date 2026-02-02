/**
 * Shared Two-Dropdown Model Selector Component
 *
 * Features:
 * - Two cascading dropdowns: Provider → Model
 * - Featured model sorting (★ prefix)
 * - localStorage state persistence
 * - Explicit provider passing to backend
 * - Server sync support
 */

const PROVIDER_NAMES = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'ollama': 'Ollama (Local)',
    'openrouter': 'OpenRouter',
    'vertex_ai': 'Google Vertex AI',
    'google': 'Google Gemini',
    'xai': 'xAI',
    'groq': 'Groq',
    'together': 'Together AI',
    'mistral': 'Mistral',
    'deepseek': 'DeepSeek',
    'runpod': 'RunPod',
};

class ModelSelector {
    /**
     * Create a two-dropdown model selector
     * @param {Object} options - Configuration options
     * @param {string} options.providerSelectId - ID of the provider select element
     * @param {string} options.modelSelectId - ID of the model select element
     * @param {string} [options.apiEndpoint='/api/models'] - API endpoint for models
     * @param {string} [options.currentModelEndpoint='/api/model/current'] - API endpoint for current model
     * @param {string} [options.storagePrefix='kestrel'] - localStorage key prefix
     * @param {Function} [options.onModelChange] - Callback when model changes (provider, model) => void
     * @param {Function} [options.getAuthHeader] - Function returning auth header object
     * @param {boolean} [options.sendCommandOnChange=true] - Whether to trigger onModelChange on selection
     */
    constructor(options = {}) {
        this.providerSelect = document.getElementById(options.providerSelectId);
        this.modelSelect = document.getElementById(options.modelSelectId);
        this.apiEndpoint = options.apiEndpoint || '/api/models';
        // Use 'in' check to allow explicit null (disables server sync)
        this.currentModelEndpoint = 'currentModelEndpoint' in options
            ? options.currentModelEndpoint
            : '/api/model/current';
        this.storagePrefix = options.storagePrefix || 'kestrel';
        this.onModelChange = options.onModelChange || (() => {});
        this.getAuthHeader = options.getAuthHeader || (() => ({}));
        this.sendCommandOnChange = options.sendCommandOnChange !== false;

        this.allModelsData = null;
        this.selectedProvider = '';
        this.selectedModel = '';
        this.isInitialLoad = true;

        this._loadState();
    }

    /**
     * Load saved state from localStorage
     */
    _loadState() {
        this.selectedProvider = localStorage.getItem(`${this.storagePrefix}_selected_provider`) || '';
        this.selectedModel = localStorage.getItem(`${this.storagePrefix}_selected_model`) || '';
    }

    /**
     * Save state to localStorage
     */
    _saveState() {
        if (this.selectedProvider) {
            localStorage.setItem(`${this.storagePrefix}_selected_provider`, this.selectedProvider);
        }
        if (this.selectedModel) {
            localStorage.setItem(`${this.storagePrefix}_selected_model`, this.selectedModel);
        }
    }

    /**
     * Initialize the component - load models and bind events
     */
    async init() {
        await this.loadModels();
        this._bindEvents();
        await this.syncWithServer();
        this.isInitialLoad = false;
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', () => this._handleProviderChange());
        }
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => this._handleModelChange());
        }
    }

    /**
     * Load models from API
     */
    async loadModels() {
        if (!this.providerSelect || !this.modelSelect) {
            console.warn('ModelSelector: Provider or model select element not found');
            return null;
        }

        try {
            const headers = {
                'Content-Type': 'application/json',
                ...this.getAuthHeader()
            };

            const response = await fetch(this.apiEndpoint, { headers });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            this.allModelsData = await response.json();
            this._populateProviders();

            return this.allModelsData;
        } catch (e) {
            console.error('ModelSelector: Failed to load models:', e);
            this.providerSelect.innerHTML = '<option value="">Error loading</option>';
            return null;
        }
    }

    /**
     * Populate provider dropdown
     */
    _populateProviders() {
        if (!this.allModelsData?.by_provider) return;

        const providers = Object.keys(this.allModelsData.by_provider).sort((a, b) => {
            // Sort by display name
            const nameA = PROVIDER_NAMES[a] || a;
            const nameB = PROVIDER_NAMES[b] || b;
            return nameA.localeCompare(nameB);
        });

        // Build provider options with model counts
        this.providerSelect.innerHTML = providers.map(p => {
            const displayName = PROVIDER_NAMES[p] || p.charAt(0).toUpperCase() + p.slice(1);
            const count = this.allModelsData.by_provider[p]?.length || 0;
            return `<option value="${p}">${displayName} (${count})</option>`;
        }).join('');

        // Restore saved provider or use first
        if (this.selectedProvider && providers.includes(this.selectedProvider)) {
            this.providerSelect.value = this.selectedProvider;
        } else if (providers.length > 0) {
            this.providerSelect.value = providers[0];
            this.selectedProvider = providers[0];
        }

        // Populate models for selected provider (don't trigger command on initial load)
        this._populateModels();
    }

    /**
     * Populate model dropdown based on selected provider
     */
    _populateModels() {
        const provider = this.providerSelect?.value;
        if (!provider || !this.allModelsData?.by_provider) return;

        const models = [...(this.allModelsData.by_provider[provider] || [])];

        if (models.length === 0) {
            this.modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }

        // Sort: featured first, then alphabetically by display name
        models.sort((a, b) => {
            if (a.is_featured !== b.is_featured) return b.is_featured ? 1 : -1;
            return (a.display_name || a.id).localeCompare(b.display_name || b.id);
        });

        // Build model options
        this.modelSelect.innerHTML = models.map(m => {
            const star = m.is_featured ? '★ ' : '';
            const displayName = m.display_name || m.id;
            return `<option value="${m.id}">${star}${displayName}</option>`;
        }).join('');

        // Restore saved model if it belongs to this provider
        if (this.selectedModel && models.some(m => m.id === this.selectedModel)) {
            this.modelSelect.value = this.selectedModel;
        } else if (models.length > 0) {
            // Select first model if no saved model for this provider
            this.modelSelect.value = models[0].id;
            this.selectedModel = models[0].id;
        }
    }

    /**
     * Handle provider dropdown change
     */
    _handleProviderChange() {
        this.selectedProvider = this.providerSelect.value;
        this._saveState();
        this._populateModels();

        // Update selected model to first in new provider
        this.selectedModel = this.modelSelect.value;
        this._saveState();

        // Always notify callback, pass isInitialLoad so caller can decide
        this.onModelChange(this.selectedProvider, this.selectedModel, this.isInitialLoad);
    }

    /**
     * Handle model dropdown change
     */
    _handleModelChange() {
        this.selectedModel = this.modelSelect.value;
        this._saveState();

        // Always notify callback, pass isInitialLoad so caller can decide
        this.onModelChange(this.selectedProvider, this.selectedModel, this.isInitialLoad);
    }

    /**
     * Sync with server's current model
     */
    async syncWithServer() {
        // Skip if no endpoint configured
        if (!this.currentModelEndpoint) return;

        try {
            const headers = {
                'Content-Type': 'application/json',
                ...this.getAuthHeader()
            };

            const response = await fetch(this.currentModelEndpoint, { headers });
            if (!response.ok) return;

            const data = await response.json();
            if (!data.model) return;

            // Find which provider this model belongs to
            if (this.allModelsData?.by_provider) {
                for (const [provider, models] of Object.entries(this.allModelsData.by_provider)) {
                    if (models.some(m => m.id === data.model)) {
                        // Update provider if different
                        if (this.providerSelect.value !== provider) {
                            this.providerSelect.value = provider;
                            this.selectedProvider = provider;
                            this._populateModels();
                        }
                        // Update model
                        this.modelSelect.value = data.model;
                        this.selectedModel = data.model;
                        this._saveState();
                        break;
                    }
                }
            }
        } catch (e) {
            // Silently fail - endpoint might not exist
        }
    }

    /**
     * Update UI from agent response containing MODEL_CHANGED marker
     * @param {string} content - Response content to check for MODEL_CHANGED
     */
    checkForModelChange(content) {
        if (!content?.includes('MODEL_CHANGED:')) return false;

        try {
            const jsonStr = content.split('MODEL_CHANGED:')[1];
            const syncData = JSON.parse(jsonStr);

            if (syncData.provider && syncData.model) {
                // Extract actual model ID (remove provider prefix if present)
                let modelId = syncData.model;
                if (modelId.startsWith(syncData.provider + '/')) {
                    modelId = modelId.slice(syncData.provider.length + 1);
                }

                // Update provider dropdown
                if (this.providerSelect && this.providerSelect.value !== syncData.provider) {
                    this.providerSelect.value = syncData.provider;
                    this.selectedProvider = syncData.provider;
                    this._populateModels();
                }

                // Update model dropdown
                if (this.modelSelect) {
                    this.modelSelect.value = modelId;
                    this.selectedModel = modelId;
                    this._saveState();
                }

                return true;
            }
        } catch (e) {
            console.warn('ModelSelector: Failed to parse MODEL_CHANGED:', e);
        }

        return false;
    }

    /**
     * Get current selection
     * @returns {{provider: string, model: string}}
     */
    getSelection() {
        return {
            provider: this.selectedProvider,
            model: this.selectedModel
        };
    }

    /**
     * Set selection programmatically
     * @param {string} provider - Provider name
     * @param {string} model - Model ID
     * @param {boolean} [triggerCallback=false] - Whether to trigger onModelChange
     */
    setSelection(provider, model, triggerCallback = false) {
        if (provider && this.providerSelect) {
            this.providerSelect.value = provider;
            this.selectedProvider = provider;
            this._populateModels();
        }

        if (model && this.modelSelect) {
            this.modelSelect.value = model;
            this.selectedModel = model;
        }

        this._saveState();

        if (triggerCallback) {
            this.onModelChange(this.selectedProvider, this.selectedModel);
        }
    }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ModelSelector, PROVIDER_NAMES };
}

// Export globally for script tag usage
window.SharedModelSelector = ModelSelector;
window.PROVIDER_NAMES = PROVIDER_NAMES;
