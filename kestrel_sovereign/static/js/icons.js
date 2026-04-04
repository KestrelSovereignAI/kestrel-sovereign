/**
 * Kestrel Icon System — SVG icons replacing platform emojis.
 *
 * Loaded as a regular <script> in <head>. Injects CSS mask-image rules and
 * exposes window.kicon(name, size?, cls?) for programmatic use.
 *
 * HTML usage:  <span class="ki ki-lock"></span>
 * JS  usage:  element.innerHTML = kicon('lock');
 *             element.innerHTML = kicon('lock', '1.2em');
 *
 * Icons inherit text colour via `background: currentColor` + CSS mask.
 */
(function () {
    'use strict';

    // Each value is SVG-inner content for a 0 0 24 24 viewBox.
    var PATHS = {
        // --- Security & Auth ---
        'lock': '<rect x="5" y="11" width="14" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 11V7a4 4 0 1 1 8 0v4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'lock-open': '<rect x="5" y="11" width="14" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 11V7a4 4 0 0 1 8 0" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'lock-key': '<rect x="5" y="11" width="14" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 11V7a4 4 0 1 1 8 0v4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="16" r="1.5" fill="currentColor"/>',
        'shield': '<path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7L12 2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'key': '<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',

        // --- Navigation & Actions ---
        'scroll': '<path d="M8 21h12a2 2 0 0 0 2-2v-2H10v2a2 2 0 1 1-4 0V5a2 2 0 1 0-4 0v3h12v11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'sparkles': '<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M19 13l.75 2.25L22 16l-2.25.75L19 19l-.75-2.25L16 16l2.25-.75L19 13z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>',
        'refresh': '<path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 12a9 9 0 0 1-15 6.7L3 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 3v5h-5M3 21v-5h5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'link': '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',

        // --- Data & Content ---
        'clipboard': '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" fill="none" stroke="currentColor" stroke-width="2"/><rect x="8" y="2" width="8" height="4" rx="1" fill="none" stroke="currentColor" stroke-width="2"/>',
        'chart-bar': '<rect x="3" y="12" width="4" height="9" rx="1" fill="none" stroke="currentColor" stroke-width="2"/><rect x="10" y="8" width="4" height="13" rx="1" fill="none" stroke="currentColor" stroke-width="2"/><rect x="17" y="3" width="4" height="18" rx="1" fill="none" stroke="currentColor" stroke-width="2"/>',
        'folder': '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'cabinet': '<rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="3" y1="12" x2="21" y2="12" stroke="currentColor" stroke-width="2"/><line x1="10" y1="8" x2="14" y2="8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="10" y1="16" x2="14" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'globe': '<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><ellipse cx="12" cy="12" rx="4" ry="10" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="2" y1="12" x2="22" y2="12" stroke="currentColor" stroke-width="1.5"/>',
        'inbox': '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" fill="none" stroke="currentColor" stroke-width="2"/>',

        // --- People & Entities ---
        'robot': '<rect x="4" y="8" width="16" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="14" r="1.5" fill="currentColor"/><circle cx="15" cy="14" r="1.5" fill="currentColor"/><path d="M12 2v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="2" r="1" fill="currentColor"/>',
        'user': '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="7" r="4" fill="none" stroke="currentColor" stroke-width="2"/>',
        'building': '<rect x="4" y="3" width="16" height="18" rx="1" fill="none" stroke="currentColor" stroke-width="2"/><rect x="8" y="7" width="3" height="3" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="13" y="7" width="3" height="3" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="8" y="13" width="3" height="3" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="13" y="13" width="3" height="3" fill="none" stroke="currentColor" stroke-width="1.5"/>',

        // --- Finance ---
        'credit-card': '<rect x="2" y="5" width="20" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="2" y1="10" x2="22" y2="10" stroke="currentColor" stroke-width="2"/>',
        'wallet': '<path d="M21 12V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="18" cy="12" r="1.5" fill="currentColor"/>',

        // --- Status ---
        'hourglass': '<path d="M5 3h14M5 21h14M7 3v3a5 5 0 0 0 5 5 5 5 0 0 0 5-5V3M7 21v-3a5 5 0 0 1 5-5 5 5 0 0 1 5 5v3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'gear': '<circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" fill="none" stroke="currentColor" stroke-width="2"/>',
        'check-circle': '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><polyline points="22 4 12 14.01 9 11.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'x-circle': '<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'warning': '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><line x1="12" y1="9" x2="12" y2="13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/>',
        'question': '<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/>',
        'info': '<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><line x1="12" y1="16" x2="12" y2="12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="8" r="0.5" fill="currentColor"/>',

        // --- Tools ---
        'wrench': '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'trash': '<polyline points="3 6 5 6 21 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'puzzle': '<path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.014a2.5 2.5 0 1 0 2.9-2.9c-.486-.196-.937-.558-1.014-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 11.638 2.36c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.968 1.02z" fill="none" stroke="currentColor" stroke-width="2"/>',
        'email': '<rect x="2" y="4" width="20" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><polyline points="22 6 12 13 2 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',

        // --- Kestrel Birds ---
        'kestrel': '<path d="M12 2C9 5 5 7 3 10c2 0 4 1 5 3-1 2-2 5-1 7 1-1 3-2 5-3 2 1 4 2 5 3 1-2 0-5-1-7 1-2 3-3 5-3-2-3-6-5-9-8z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="10" cy="9" r="1" fill="currentColor"/>',
        'claw': '<path d="M6 3c0 3 2 6 4 8M18 3c0 3-2 6-4 8M12 11v4M8 18c-1 2 0 3 1 3s2-1 2-3M16 18c1 2 0 3-1 3s-2-1-2-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="15" r="3" fill="none" stroke="currentColor" stroke-width="2"/>',
        'lightning': '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'eye': '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/>',
        'dove': '<path d="M12 20c-2-2-8-5-8-10 0-3 2-5 5-5 1 0 2 .5 3 1.5C13 5.5 14 5 15 5c3 0 5 2 5 5 0 5-6 8-8 10z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'classical-building': '<path d="M3 21h18M4 21V10M20 21V10M12 3l9 7H3l9-7z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><line x1="8" y1="10" x2="8" y2="21" stroke="currentColor" stroke-width="2"/><line x1="12" y1="10" x2="12" y2="21" stroke="currentColor" stroke-width="2"/><line x1="16" y1="10" x2="16" y2="21" stroke="currentColor" stroke-width="2"/>',

        // --- Permission symbols ---
        'check-box': '<rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><polyline points="9 11 12 14 22 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'empty-box': '<rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>',
        'x-box': '<rect x="3" y="3" width="18" height="18" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'half-circle': '<path d="M12 2a10 10 0 0 1 0 20" fill="currentColor" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/>',

        // --- Misc status glyphs ---
        'checkmark': '<polyline points="20 6 9 17 4 12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
        'x-mark': '<line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/><line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>',

        // --- Marketing page extras ---
        'hospital': '<path d="M3 21h18M5 21V7l7-4 7 4v14" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><line x1="12" y1="10" x2="12" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="9" y1="13" x2="15" y2="13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'laptop': '<rect x="2" y="4" width="20" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="1" y1="20" x2="23" y2="20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'person': '<circle cx="12" cy="7" r="4" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5.5 21c0-4 3-7 6.5-7s6.5 3 6.5 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
        'heart': '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'handshake': '<path d="M7 11l-4 4 4 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M17 11l4 4-4 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M3 15h7l4-4h7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'seedling': '<path d="M12 22V12M12 12C12 7 16 3 21 3c0 5-4 9-9 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 12C12 7 8 3 3 3c0 5 4 9 9 9z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
        'construction': '<path d="M2 20h20M4 20V8l4-4v16M16 20V8l4-4v16M10 20v-8h4v8" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'palette': '<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8" cy="9" r="1.5" fill="currentColor"/><circle cx="13" cy="7" r="1.5" fill="currentColor"/><circle cx="17" cy="10" r="1.5" fill="currentColor"/><circle cx="8" cy="14" r="1.5" fill="currentColor"/><path d="M16 15c0-1.5-1.5-2.5-3-2s-2 2.5-.5 3c2 .5 3.5-.5 3.5-1z" fill="currentColor"/>',
        'pin': '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="10" r="3" fill="none" stroke="currentColor" stroke-width="2"/>',
        'explosion': '<polygon points="12 2 14 9 21 9 15 13 17 21 12 16 7 21 9 13 3 9 10 9" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'briefcase': '<rect x="2" y="7" width="20" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2" fill="none" stroke="currentColor" stroke-width="2"/>',
        'mask': '<path d="M12 4C8 4 4 7 4 12s4 8 8 8 8-5 8-8-4-8-8-8z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="11" r="1.5" fill="currentColor"/><circle cx="15" cy="11" r="1.5" fill="currentColor"/><path d="M9 16c1.5 1 4.5 1 6 0" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
        'document': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><polyline points="14 2 14 8 20 8" fill="none" stroke="currentColor" stroke-width="2"/>',
        'factory': '<path d="M2 20h20M4 20V10l5-4v4l5-4v4l5-4v14" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
        'feather': '<path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><line x1="16" y1="8" x2="2" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><line x1="17.5" y1="15" x2="9" y2="15" stroke="currentColor" stroke-width="1.5"/>',
        'owl': '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="10" r="2" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="15" cy="10" r="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 5l-2-3M12 5l2-3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M10 15c.67.67 1.33 1 2 1s1.33-.33 2-1" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
        'bird': '<path d="M16 7c0-2-1-3-3-3s-3 1-3 3c-3 0-6 2-6 6h18c0-4-3-6-6-6z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M7 13c0 3 2 6 5 8 3-2 5-5 5-8" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><circle cx="11" cy="7" r="1" fill="currentColor"/>',
    };

    // Aliases (e.g. privacy modes mapped in ui.js)
    PATHS['privacy-ephemeral'] = PATHS['lock'];
    PATHS['privacy-isolated'] = PATHS['lock-key'];
    PATHS['privacy-anonymous'] = PATHS['mask'];
    PATHS['privacy-normal'] = PATHS['document'];
    PATHS['privacy-public'] = PATHS['globe'];

    // ---- Generate CSS mask-image rules ---------------------------------
    var css = [
        '.ki{display:inline-block;width:1em;height:1em;vertical-align:-0.125em;',
        'background:currentColor;',
        '-webkit-mask:no-repeat center/contain;mask:no-repeat center/contain;',
        '-webkit-mask-mode:alpha;mask-mode:alpha}'
    ].join('');

    var names = Object.keys(PATHS);
    for (var i = 0; i < names.length; i++) {
        var name = names[i];
        // Build a standalone SVG for use as a mask (alpha channel).
        // Replace currentColor → white so strokes/fills are opaque in the mask.
        var inner = PATHS[name].replace(/currentColor/g, 'white');
        var svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>" + inner + '</svg>';
        // Minimal percent-encoding for data: URI inside url("")
        var encoded = svg
            .replace(/"/g, "'")
            .replace(/#/g, '%23')
            .replace(/</g, '%3C')
            .replace(/>/g, '%3E');
        var uri = 'url("data:image/svg+xml,' + encoded + '")';
        css += '.ki-' + name + '{-webkit-mask-image:' + uri + ';mask-image:' + uri + '}';
    }

    // Inject styles
    var style = document.createElement('style');
    style.id = 'kestrel-icons';
    style.textContent = css;
    (document.head || document.documentElement).appendChild(style);

    // ---- Global API ----------------------------------------------------
    /**
     * Returns an HTML string for the named icon.
     * @param {string} name   Icon key from PATHS
     * @param {string} [size] Optional CSS size override (default: 1em via .ki)
     * @param {string} [cls]  Extra CSS class(es)
     * @returns {string}
     */
    window.kicon = function (name, size, cls) {
        if (!PATHS[name]) {
            console.warn('kicon: unknown icon "' + name + '"');
            return '<span class="ki" title="' + name + '">?</span>';
        }
        var classes = 'ki ki-' + name;
        if (cls) classes += ' ' + cls;
        var sty = size ? ' style="width:' + size + ';height:' + size + '"' : '';
        return '<span class="' + classes + '"' + sty + ' aria-hidden="true"></span>';
    };

    window.KI_NAMES = names;
    window.KI_PATHS = PATHS;
})();
