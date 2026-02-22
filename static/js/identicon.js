/**
 * Kestrel Sovereign — DID-derived Identicon Generator
 * Generates a deterministic 5x5 symmetric grid SVG from an Ethereum address.
 */

/**
 * Generate an identicon data URL from a DID string.
 * @param {string} did - DID in format did:pkh:eip155:1:0xADDRESS
 * @param {number} size - Pixel size of the output SVG (square)
 * @returns {string} data:image/svg+xml URL
 */
export function generateIdenticon(did, size = 48) {
    // Extract hex address from DID, fallback to the raw string
    const parts = (did || '').split(':');
    const raw = (parts.length >= 5 ? parts[parts.length - 1] : did || '').replace(/^0x/i, '').toLowerCase();

    // Pad to at least 20 hex chars (40 nibbles) if short
    const hex = raw.padEnd(40, '0');

    // Color: first 3 bytes → HSL
    const hue = parseInt(hex.slice(0, 2), 16) * 360 / 256;
    const sat = 50 + (parseInt(hex.slice(2, 4), 16) * 30 / 256); // 50-80%
    const fg = `hsl(${Math.round(hue)}, ${Math.round(sat)}%, 45%)`;
    const bg = '#1a1a2e';

    // 5x5 grid with horizontal symmetry: compute 3 columns (0,1,2),
    // mirror col 0→4 and col 1→3
    const grid = [];
    for (let row = 0; row < 5; row++) {
        grid[row] = [];
        for (let col = 0; col < 3; col++) {
            const nibbleIndex = row * 3 + col + 6; // offset past color bytes
            const nibble = parseInt(hex[nibbleIndex] || '0', 16);
            grid[row][col] = nibble % 2 === 1;
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
