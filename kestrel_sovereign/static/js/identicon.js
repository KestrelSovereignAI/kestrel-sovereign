/**
 * Kestrel Sovereign — DID-derived Identicon Generator
 * Generates a deterministic 5x5 symmetric grid SVG from any string input.
 */

/**
 * Simple deterministic hash — produces 40 hex chars from any string.
 * Uses a seeded xorshift variant for fast, non-crypto hashing.
 */
function hashString(str) {
    const bytes = new Uint8Array(20);
    let h = 0x811c9dc5; // FNV offset basis
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193); // FNV prime
    }
    for (let i = 0; i < 20; i++) {
        h ^= (h >>> 13);
        h = Math.imul(h, 0x5bd1e995);
        h ^= (h >>> 15);
        h ^= i;
        bytes[i] = (h >>> 0) & 0xff;
    }
    return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Generate an identicon data URL from a seed string.
 * @param {string} seed - Any string (DID, name, etc.)
 * @param {number} size - Pixel size of the output SVG (square)
 * @returns {string} data:image/svg+xml URL
 */
export function generateIdenticon(seed, size = 48) {
    const hex = hashString(seed || 'default');

    // Color: first 3 bytes → HSL
    const hue = parseInt(hex.slice(0, 2), 16) * 360 / 256;
    const sat = 50 + (parseInt(hex.slice(2, 4), 16) * 30 / 256); // 50-80%
    const fg = `hsl(${Math.round(hue)}, ${Math.round(sat)}%, 45%)`;
    const bg = '#1a1a2e';

    // 5x5 grid with horizontal symmetry: compute 3 columns (0,1,2),
    // mirror col 0→4 and col 1→3. Bias toward ~60% fill for denser icons.
    const grid = [];
    for (let row = 0; row < 5; row++) {
        grid[row] = [];
        for (let col = 0; col < 3; col++) {
            const nibbleIndex = row * 3 + col + 6; // offset past color bytes
            const nibble = parseInt(hex[nibbleIndex] || '0', 16);
            grid[row][col] = nibble >= 6; // ~62.5% fill (10 of 16 values)
        }
        // Mirror
        grid[row][3] = grid[row][1];
        grid[row][4] = grid[row][0];
    }

    // Render SVG
    const cellSize = size / 5;
    let rects = '';
    for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 5; col++) {
            if (grid[row][col]) {
                const x = col * cellSize;
                const y = row * cellSize;
                rects += `<rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="${fg}"/>`;
            }
        }
    }

    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">` +
        `<rect width="${size}" height="${size}" fill="${bg}"/>` +
        rects +
        `</svg>`;

    return 'data:image/svg+xml,' + encodeURIComponent(svg);
}
