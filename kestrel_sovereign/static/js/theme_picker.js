/**
 * UI theme picker (epic #986, sub-issue #991).
 *
 * Wires the Display section in the Sovereignty panel to
 * window.KestrelTheme. The dropdowns let the user switch between
 * available themes (loaded from /api/ui/themes) and locales (English
 * only at MVP). Selection persists via localStorage (handled by
 * theme.js — this module only reads/writes through applyTheme()).
 *
 * Loaded after theme.js. Both auto-init on DOMContentLoaded.
 */
(function () {
  'use strict';

  // Display name overrides for known themes — keeps the picker readable
  // without forcing every theme name to be capitalized in its directory.
  // Unknown themes fall back to capitalize(name).
  const THEME_DISPLAY_NAMES = {
    legacy: 'Legacy',
    falconry: 'Falconry',
    plain: 'Plain',
  };

  function capitalize(s) {
    if (!s) return s;
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function displayNameForTheme(name) {
    return THEME_DISPLAY_NAMES[name] || capitalize(name);
  }

  function populateThemes(selectEl, themes, currentTheme) {
    selectEl.innerHTML = '';
    themes.forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = displayNameForTheme(name);
      if (name === currentTheme) opt.selected = true;
      selectEl.appendChild(opt);
    });
  }

  function updateStatus(detail) {
    const status = document.getElementById('theme-picker-status');
    if (!status) return;
    if (!detail || typeof detail.applied !== 'number') {
      status.textContent = '';
      return;
    }
    const fbCount = (detail.fallback_keys || []).length;
    if (fbCount > 0) {
      status.textContent = `${detail.applied} labels applied (${fbCount} fell back to legacy)`;
    } else {
      status.textContent = `${detail.applied} labels applied`;
    }
  }

  async function init() {
    const themeSelect = document.getElementById('theme-picker-theme');
    const localeSelect = document.getElementById('theme-picker-locale');
    if (!themeSelect || !localeSelect) {
      // Picker is only present in index.html; skip silently elsewhere.
      return;
    }
    if (!window.KestrelTheme) {
      console.warn('[theme-picker] KestrelTheme not available; theme.js may have failed to load');
      return;
    }

    // Populate themes from the API. Use the current theme (resolved by
    // theme.js init from localStorage) as the selected value.
    let themes;
    try {
      themes = await window.KestrelTheme.listAvailableThemes();
    } catch (err) {
      console.warn('[theme-picker] failed to list themes:', err);
      themes = ['legacy'];  // fail-safe so the dropdown isn't empty
    }
    const currentTheme = window.KestrelTheme.getCurrentTheme();
    populateThemes(themeSelect, themes, currentTheme);

    const currentLocale = window.KestrelTheme.getCurrentLocale();
    if (localeSelect.querySelector(`option[value="${currentLocale}"]`)) {
      localeSelect.value = currentLocale;
    }

    function onChange() {
      window.KestrelTheme.applyTheme(themeSelect.value, localeSelect.value);
    }
    themeSelect.addEventListener('change', onChange);
    localeSelect.addEventListener('change', onChange);

    // Reflect every successful theme application in the status line.
    window.addEventListener('themechange', (ev) => updateStatus(ev.detail));
    window.addEventListener('themeerror', () => {
      const status = document.getElementById('theme-picker-status');
      if (status) status.textContent = 'Theme load failed — keeping current labels.';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
