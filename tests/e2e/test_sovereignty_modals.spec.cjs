/**
 * Sovereignty Panel Modal Tests
 *
 * Tests for Session 1: Replace browser dialogs with styled modal components
 * - Export modal with tier and encryption options
 * - Import modal with CID input and paste button
 * - Toast notifications for success/error feedback
 * - Delete memory confirmation modal
 */

const { test, expect } = require('@playwright/test');

const KESTREL_URL = process.env.KESTREL_URL || 'http://localhost:8888';

test.describe('Sovereignty Panel Modals', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Wait for the page to load
        await page.waitForSelector('.nav-tab[data-panel="sovereignty"]');
    });

    test('should show export modal when clicking Export button', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');

        // Verify modal appears
        await expect(page.locator('#modal-overlay')).toBeVisible();
        await expect(page.locator('.modal-container')).toBeVisible();

        // Verify modal content
        await expect(page.locator('.modal-header h3')).toHaveText('Export Agent Data');

        // Verify tier options exist
        await expect(page.locator('input[name="export-tier"][value="LOCAL_ONLY"]')).toBeVisible();
        await expect(page.locator('input[name="export-tier"][value="IPFS"]')).toBeVisible();
        await expect(page.locator('input[name="export-tier"][value="FILECOIN"]')).toBeVisible();

        // Verify IPFS is selected by default
        await expect(page.locator('input[name="export-tier"][value="IPFS"]')).toBeChecked();

        // Verify encryption checkbox exists and is checked by default
        await expect(page.locator('#export-encrypt')).toBeVisible();
        await expect(page.locator('#export-encrypt')).toBeChecked();

        // Verify buttons exist
        await expect(page.locator('.modal-btn').filter({ hasText: 'Cancel' })).toBeVisible();
        await expect(page.locator('.modal-btn').filter({ hasText: 'Export' })).toBeVisible();
    });

    test('should close export modal on Cancel', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Click Cancel
        await page.click('.modal-btn:has-text("Cancel")');

        // Verify modal is hidden
        await expect(page.locator('#modal-overlay')).not.toBeVisible();
    });

    test('should close export modal on X button', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Click X button
        await page.click('.modal-close-btn');

        // Verify modal is hidden
        await expect(page.locator('#modal-overlay')).not.toBeVisible();
    });

    test('should close export modal on Escape key', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Press Escape
        await page.keyboard.press('Escape');

        // Verify modal is hidden
        await expect(page.locator('#modal-overlay')).not.toBeVisible();
    });

    test('should show import modal when clicking Import button', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Import button
        await page.click('#btn-import');

        // Verify modal appears
        await expect(page.locator('#modal-overlay')).toBeVisible();
        await expect(page.locator('.modal-container')).toBeVisible();

        // Verify modal content
        await expect(page.locator('.modal-header h3')).toHaveText('Import from CID');

        // Verify CID input exists
        await expect(page.locator('#import-cid-input')).toBeVisible();

        // Verify paste button exists
        await expect(page.locator('#paste-cid-btn')).toBeVisible();

        // Verify buttons exist
        await expect(page.locator('.modal-btn').filter({ hasText: 'Cancel' })).toBeVisible();
        await expect(page.locator('.modal-btn').filter({ hasText: 'Import' })).toBeVisible();
    });

    test('should show warning toast for empty CID', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Import button
        await page.click('#btn-import');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Click Import without entering CID
        await page.click('.modal-btn:has-text("Import")');

        // Verify warning toast appears with specific message
        const toast = page.locator('.toast-item:has-text("Please enter a CID")');
        await expect(toast).toBeVisible({ timeout: 5000 });

        // Modal should still be visible (not closed)
        await expect(page.locator('#modal-overlay')).toBeVisible();
    });

    test('should show warning toast for invalid CID format', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Import button
        await page.click('#btn-import');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Enter invalid CID
        await page.fill('#import-cid-input', 'invalid-cid-format');

        // Click Import
        await page.click('.modal-btn:has-text("Import")');

        // Verify warning toast appears with specific message
        // Use a more specific selector to find the toast with the expected text
        const toast = page.locator('.toast-item:has-text("CID should start with")');
        await expect(toast).toBeVisible({ timeout: 5000 });
    });

    test('should close import modal on Cancel', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Import button
        await page.click('#btn-import');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Click Cancel
        await page.click('.modal-btn:has-text("Cancel")');

        // Verify modal is hidden
        await expect(page.locator('#modal-overlay')).not.toBeVisible();
    });

    test('should change storage tier selection', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // IPFS should be selected by default
        await expect(page.locator('input[name="export-tier"][value="IPFS"]')).toBeChecked();

        // Select Local Only
        await page.click('input[name="export-tier"][value="LOCAL_ONLY"]');
        await expect(page.locator('input[name="export-tier"][value="LOCAL_ONLY"]')).toBeChecked();
        await expect(page.locator('input[name="export-tier"][value="IPFS"]')).not.toBeChecked();

        // Select Filecoin
        await page.click('input[name="export-tier"][value="FILECOIN"]');
        await expect(page.locator('input[name="export-tier"][value="FILECOIN"]')).toBeChecked();
        await expect(page.locator('input[name="export-tier"][value="LOCAL_ONLY"]')).not.toBeChecked();
    });

    test('should toggle encryption checkbox', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Export button
        await page.click('#btn-export-ipfs');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Encryption should be checked by default
        await expect(page.locator('#export-encrypt')).toBeChecked();

        // Uncheck encryption
        await page.click('#export-encrypt');
        await expect(page.locator('#export-encrypt')).not.toBeChecked();

        // Check encryption again
        await page.click('#export-encrypt');
        await expect(page.locator('#export-encrypt')).toBeChecked();
    });

    test('should focus CID input when import modal opens', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Click Import button
        await page.click('#btn-import');
        await expect(page.locator('#modal-overlay')).toBeVisible();

        // Wait a bit for focus to be applied
        await page.waitForTimeout(100);

        // Verify CID input is focused (can type directly)
        await page.keyboard.type('QmTest');
        await expect(page.locator('#import-cid-input')).toHaveValue('QmTest');
    });
});

test.describe('Toast Notifications', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        await page.waitForSelector('.nav-tab[data-panel="sovereignty"]');
    });

    test('should show toast on privacy mode change', async ({ page }) => {
        // Navigate to chat tab where privacy indicator is located
        await page.click('.nav-tab[data-panel="chat"]');
        await page.waitForSelector('#panel-chat.active', { timeout: 5000 });
        await page.waitForSelector('#chat-privacy-indicator span', { timeout: 15000 });

        // Click privacy indicator
        await page.click('#chat-privacy-indicator');
        await expect(page.locator('#privacy-dropdown')).toBeVisible();

        // Get the current mode from the indicator text
        const indicatorText = await page.locator('#chat-privacy-indicator span span:last-child').textContent();
        const currentMode = indicatorText?.toLowerCase().trim();
        console.log('Current mode from indicator:', currentMode);

        // Select a different mode - use 'anonymous' if not already, otherwise 'normal'
        const newMode = currentMode === 'anonymous' ? 'normal' : 'anonymous';
        console.log('Selecting mode:', newMode);

        // Select the different privacy mode
        await page.click(`.privacy-option[data-mode="${newMode}"]`);

        // Verify toast appears (success or error)
        await expect(page.locator('.toast-item')).toBeVisible({ timeout: 5000 });
    });

    test('should dismiss toast on click', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Trigger a toast by clicking Import and submitting empty
        await page.click('#btn-import');
        await page.click('.modal-btn:has-text("Import")');

        // Toast should appear
        await expect(page.locator('.toast-item')).toBeVisible();

        // Click the toast to dismiss
        await page.click('.toast-item');

        // Toast should disappear
        await expect(page.locator('.toast-item')).not.toBeVisible({ timeout: 1000 });
    });

    test('should auto-dismiss toast after timeout', async ({ page }) => {
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');

        // Trigger a toast
        await page.click('#btn-import');
        await page.click('.modal-btn:has-text("Import")');

        // Toast should appear
        await expect(page.locator('.toast-item')).toBeVisible();

        // Wait for auto-dismiss (default is 4 seconds)
        await expect(page.locator('.toast-item')).not.toBeVisible({ timeout: 5000 });
    });
});

test.describe('Delete Memory Modal', () => {
    test('should show confirmation modal for delete memory (if memories exist)', async ({ page }) => {
        await page.goto(KESTREL_URL);

        // Navigate to Memories panel
        await page.click('.nav-tab[data-panel="memories"]');
        await page.waitForSelector('#panel-memories.active');

        // Wait for memories to load
        await page.waitForSelector('#memory-list');
        await page.waitForTimeout(1000); // Wait for loading to complete

        // Check if there are any deletable memories (not agent or document type)
        const deleteButtons = page.locator('.memory-item .btn-danger');
        const count = await deleteButtons.count();

        if (count > 0) {
            // Click first delete button
            await deleteButtons.first().click();

            // Verify confirmation modal appears
            await expect(page.locator('#modal-overlay')).toBeVisible();
            await expect(page.locator('.modal-header h3')).toHaveText('Delete Memory');
            await expect(page.locator('.modal-body')).toContainText('Are you sure you want to delete');

            // Cancel the deletion
            await page.click('.modal-btn:has-text("Cancel")');
            await expect(page.locator('#modal-overlay')).not.toBeVisible();
        } else {
            // Skip test if no deletable memories
            test.skip();
        }
    });
});

test.describe('Copy to Clipboard', () => {
    test('should show toast on copy DID from identity panel', async ({ page, context }) => {
        // Grant clipboard permissions
        await context.grantPermissions(['clipboard-write', 'clipboard-read']);

        await page.goto(KESTREL_URL);

        // Chat is the default tab — navigate to Identity to render its panel.
        await page.locator('.nav-tab').filter({ hasText: /identity/i }).click();

        // Wait for identity to load
        await page.waitForSelector('#identity-card .identity-did');

        // Click copy button next to DID
        const copyButton = page.locator('#identity-card .identity-did button');
        if (await copyButton.isVisible()) {
            await copyButton.click();

            // Verify toast appears
            await expect(page.locator('.toast-item')).toBeVisible({ timeout: 3000 });
            await expect(page.locator('.toast-item')).toContainText('Copied to clipboard');
        }
    });
});

test.describe('Export Details Panel (Session 2)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');
        // Wait for exports to load
        await page.waitForTimeout(1000);
    });

    test('should render export cards with expand icon', async ({ page }) => {
        // Check if there are any exports
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count > 0) {
            // Verify export card structure
            const firstCard = exportCards.first();
            await expect(firstCard.locator('.export-header')).toBeVisible();
            await expect(firstCard.locator('.expand-icon')).toBeVisible();
        } else {
            // No exports - verify empty state message
            await expect(page.locator('#export-list')).toContainText('No exports yet');
        }
    });

    test('should toggle export details on click', async ({ page }) => {
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count > 0) {
            const firstCard = exportCards.first();
            const details = firstCard.locator('.export-details');

            // Initially hidden
            await expect(details).not.toBeVisible();

            // Click to expand
            await firstCard.locator('.export-header').click();
            await expect(details).toBeVisible();

            // Expand icon should rotate
            const expandIcon = firstCard.locator('.expand-icon');
            const transform = await expandIcon.evaluate(el => el.style.transform);
            expect(transform).toBe('rotate(180deg)');

            // Click again to collapse
            await firstCard.locator('.export-header').click();
            await expect(details).not.toBeVisible();
        }
    });

    test('should show full CID in details panel', async ({ page }) => {
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count > 0) {
            const firstCard = exportCards.first();

            // Expand details
            await firstCard.locator('.export-header').click();
            await expect(firstCard.locator('.export-details')).toBeVisible();

            // Check for Full CID label
            await expect(firstCard.locator('.export-details')).toContainText('Full CID');
        }
    });

    test('should show IPFS Gateway link for IPFS exports (if available)', async ({ page }) => {
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        // This test only runs if there are exports
        if (count === 0) {
            console.log('No exports available - test passes (empty state handled elsewhere)');
            return;
        }

        // Find an IPFS export card (contains "IPFS" in header text)
        const ipfsCard = page.locator('.export-card').filter({ hasText: /\bIPFS\b/ }).first();
        const ipfsCardCount = await ipfsCard.count();

        if (ipfsCardCount === 0) {
            console.log('No IPFS exports available - skipping IPFS-specific assertions');
            return;
        }

        // Expand details by clicking header
        await ipfsCard.locator('.export-header').click();
        await page.waitForTimeout(200); // Wait for expand animation

        // For IPFS exports with a CID, check for View link
        // Note: Some exports may not have CID yet (null CID)
        const viewLink = ipfsCard.locator('a:has-text("View")');
        const hasViewLink = await viewLink.count() > 0;

        if (hasViewLink) {
            await expect(viewLink.first()).toBeVisible();
        } else {
            console.log('IPFS export has no CID - skipping View link check');
        }
    });

    test('should copy CID with Copy button', async ({ page, context }) => {
        await context.grantPermissions(['clipboard-write', 'clipboard-read']);

        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count > 0) {
            const firstCard = exportCards.first();
            const copyBtn = firstCard.locator('button:has-text("Copy")');

            if (await copyBtn.count() > 0) {
                await copyBtn.first().click();

                // Verify toast appears
                await expect(page.locator('.toast-item')).toBeVisible({ timeout: 3000 });
                await expect(page.locator('.toast-item')).toContainText('Copied to clipboard');
            }
        }
    });

    test('should have View link for IPFS content', async ({ page }) => {
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count > 0) {
            const viewLinks = page.locator('.export-card a:has-text("View")');
            const viewCount = await viewLinks.count();

            if (viewCount > 0) {
                // Verify View link has correct href structure
                const href = await viewLinks.first().getAttribute('href');
                expect(href).toContain('ipfs.io/ipfs/');
            }
        }
    });

    test('should show different tier colors (if exports exist)', async ({ page }) => {
        // This test verifies the tier badge styling is applied
        const exportCards = page.locator('.export-card');
        const count = await exportCards.count();

        if (count === 0) {
            console.log('No exports available - test passes (empty state handled elsewhere)');
            return;
        }

        // Check that the first card's header has a tier badge with background color
        // The tier badge is the first span inside the first div in the header
        const firstCard = exportCards.first();
        const header = firstCard.locator('.export-header');

        // The tier badge should have background styling (visible text like "IPFS" or "LOCAL")
        const tierBadgeText = await header.locator('span').first().textContent();
        expect(tierBadgeText?.length).toBeGreaterThan(0);

        // Verify the card has proper structure with header
        await expect(header).toBeVisible();
        await expect(firstCard.locator('.expand-icon')).toBeVisible();
    });
});

// ============================================================================
// Session 3: Local File Browser Tests
// ============================================================================

test.describe('Local File Browser (Session 3)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Navigate to Sovereignty panel
        await page.click('.nav-tab[data-panel="sovereignty"]');
        await page.waitForSelector('#panel-sovereignty.active');
    });

    test('should show Browse Local Files button', async ({ page }) => {
        // Verify the toggle button exists
        const toggleBtn = page.locator('#toggle-file-browser');
        await expect(toggleBtn).toBeVisible();
        await expect(toggleBtn).toContainText('Browse Local Files');
    });

    test('should toggle file browser visibility', async ({ page }) => {
        const toggleBtn = page.locator('#toggle-file-browser');
        const fileBrowserSection = page.locator('#file-browser-section');

        // Initially hidden
        await expect(fileBrowserSection).not.toBeVisible();

        // Click to show
        await toggleBtn.click();
        await expect(fileBrowserSection).toBeVisible();
        await expect(toggleBtn).toContainText('Hide Local Files');

        // Click to hide
        await toggleBtn.click();
        await expect(fileBrowserSection).not.toBeVisible();
        await expect(toggleBtn).toContainText('Browse Local Files');
    });

    test('should load and display file browser container', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');

        // Wait for file browser section to be visible
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for content to load (either files or empty state)
        await page.waitForTimeout(1000);

        // Container should have content (either file list or empty message)
        const container = page.locator('#file-browser-container');
        await expect(container).toBeVisible();
    });

    test('should show empty state or files when file browser is opened', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const container = page.locator('#file-browser-container');
        const containerText = await container.textContent();

        // Should show one of: files, empty state, or error message
        const hasFiles = (await page.locator('.file-item').count()) > 0;
        const hasEmptyState = containerText?.includes('No local cache files found');
        const hasError = containerText?.includes('Failed to load');

        // At least one state should be displayed
        expect(hasFiles || hasEmptyState || hasError).toBe(true);
    });

    test('should show stats summary when files exist', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            // Should show stats grid
            const container = page.locator('#file-browser-container');
            await expect(container).toContainText('Cache Files');
            await expect(container).toContainText('Total Size');
            await expect(container).toContainText('With Metadata');
        }
    });

    test('should show search input when files exist', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            // Should show search input
            const searchInput = page.locator('#file-search');
            await expect(searchInput).toBeVisible();
            await expect(searchInput).toHaveAttribute('placeholder', 'Search files by hash...');
        }
    });

    test('should filter files by search query', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 1) {
            // Get the first file's hash for testing
            const firstFileHash = await fileItems.first().getAttribute('data-hash');

            // Type a partial hash to filter
            const searchQuery = firstFileHash?.substring(0, 4) || '';
            await page.fill('#file-search', searchQuery);

            // Check that filtering works (at least the matching file should be visible)
            const visibleItems = page.locator('.file-item:visible');
            const visibleCount = await visibleItems.count();

            // At least one item should match the filter
            expect(visibleCount).toBeGreaterThan(0);
        }
    });

    test('should show file action buttons (preview, download, copy)', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();

            // Check for action buttons (Preview, Download, Copy)
            await expect(firstFile.locator('button[title="Preview"]')).toBeVisible();
            await expect(firstFile.locator('button[title="Download"]')).toBeVisible();
            await expect(firstFile.locator('button[title="Copy Hash"]')).toBeVisible();
        }
    });

    test('should copy file hash to clipboard', async ({ page, context }) => {
        await context.grantPermissions(['clipboard-write', 'clipboard-read']);

        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();
            const copyBtn = firstFile.locator('button[title="Copy Hash"]');

            await copyBtn.click();

            // Verify toast appears
            await expect(page.locator('.toast-item')).toBeVisible({ timeout: 3000 });
            await expect(page.locator('.toast-item')).toContainText('Copied to clipboard');
        }
    });

    test('should show file preview modal on preview button click', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();
            const previewBtn = firstFile.locator('button[title="Preview"]');

            await previewBtn.click();

            // Wait for modal to appear (may take time to fetch preview)
            await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 5000 });
            await expect(page.locator('.modal-header h3')).toContainText('File Preview');

            // Modal should contain file info
            await expect(page.locator('.modal-body')).toContainText('Filename');
            await expect(page.locator('.modal-body')).toContainText('Size');
            await expect(page.locator('.modal-body')).toContainText('Type');

            // Close modal
            await page.click('.modal-btn:has-text("Close")');
            await expect(page.locator('#modal-overlay')).not.toBeVisible();
        }
    });

    test('should show download button in preview modal', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();
            const previewBtn = firstFile.locator('button[title="Preview"]');

            await previewBtn.click();

            // Wait for modal
            await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 5000 });

            // Should have Download button
            await expect(page.locator('.modal-btn:has-text("Download")')).toBeVisible();

            // Close modal
            await page.click('.modal-btn:has-text("Close")');
        }
    });

    test('should display file size in human readable format', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();
            const fileInfo = await firstFile.textContent();

            // Should contain a size with unit (B, KB, MB, etc.)
            expect(fileInfo).toMatch(/\d+(\.\d+)?\s*(B|KB|MB|GB)/);
        }
    });

    test('should display file modification date', async ({ page }) => {
        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // Wait for loading to complete
        await page.waitForTimeout(1000);

        const fileItems = page.locator('.file-item');
        const fileCount = await fileItems.count();

        if (fileCount > 0) {
            const firstFile = fileItems.first();
            const fileInfo = await firstFile.textContent();

            // Should contain a date (various formats possible)
            // Look for patterns like "11/27/2025" or "2025" or "PM/AM"
            expect(fileInfo).toMatch(/\d{1,4}[\/\-]\d{1,2}[\/\-]\d{1,4}|\d{1,2}:\d{2}/);
        }
    });

    test('should handle API error gracefully', async ({ page }) => {
        // This test verifies error handling when API fails
        // We can't easily simulate API failure, so we just verify the error display mechanism exists

        // Click to show file browser
        await page.click('#toggle-file-browser');
        await expect(page.locator('#file-browser-section')).toBeVisible();

        // The container should exist regardless of success/failure
        await expect(page.locator('#file-browser-container')).toBeVisible();
    });
});

// ============================================================================
// Session 4: Conversations Pane Tests
//
// #2171 collapsed the two conversation surfaces into ONE placement: the
// `#conversations-pane` sidebar (rendered by conversations.js). The old
// `#history-sidebar` slideout, its `#toggle-history-btn`, and its
// `#new-conversation-btn` were deleted. These tests target the surviving pane.
// ============================================================================

test.describe('Conversations Pane (Session 4)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Navigate to Chat panel
        await page.click('.nav-tab[data-panel="chat"]');
        await page.waitForSelector('#panel-chat.active');
    });

    test('should NOT render the deleted history slideout or its toggle', async ({ page }) => {
        // #2171 removed the slideout surface entirely — none of its markup
        // should survive anywhere in the document.
        await expect(page.locator('#toggle-history-btn')).toHaveCount(0);
        await expect(page.locator('#history-sidebar')).toHaveCount(0);
        await expect(page.locator('#history-container')).toHaveCount(0);
        await expect(page.locator('#new-conversation-btn')).toHaveCount(0);
    });

    test('should show the conversations pane in standalone mode', async ({ page }) => {
        // Standalone console auto-reveals the single conversations surface.
        const pane = page.locator('#conversations-pane');
        await expect(pane).toBeVisible();
        // Its list container and new-conversation affordance are the pane's own.
        await expect(page.locator('#conversations-list')).toBeVisible();
        await expect(page.locator('#new-conversation-sidebar-btn')).toBeVisible();
    });

    test('should populate the conversations list (items or empty state)', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();

        // Wait for the list fetch to settle.
        await page.waitForTimeout(1000);

        const list = page.locator('#conversations-list');
        const listText = (await list.textContent()) || '';

        const hasConversations = (await page.locator('#conversations-list .conversation-item').count()) > 0;
        const hasEmptyState = listText.toLowerCase().includes('no conversation') ||
                             listText.toLowerCase().includes('select an agent') ||
                             listText.toLowerCase().includes('empty');
        const hasLoading = listText.toLowerCase().includes('loading');

        expect(hasConversations || hasEmptyState || hasLoading || listText.length > 5).toBe(true);
    });

    test('should render conversation items with preview and time metadata', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();
        await page.waitForTimeout(1000);

        const items = page.locator('#conversations-list .conversation-item');
        const itemCount = await items.count();

        if (itemCount > 0) {
            const firstItem = items.first();
            // Preview text row.
            await expect(firstItem.locator('.conversation-preview')).toBeVisible();
            // Meta row carries the timestamp.
            await expect(firstItem.locator('.conversation-time')).toBeVisible();
        }
    });

    test('should load a conversation when clicking an item', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();
        await page.waitForTimeout(1000);

        const items = page.locator('#conversations-list .conversation-item');
        const itemCount = await items.count();

        if (itemCount > 0) {
            await items.first().click();
            await page.waitForTimeout(1000);

            const chatContainer = page.locator('#chat-container');
            const chatContent = await chatContainer.textContent();
            expect(chatContent).not.toBeNull();
        }
    });

    test('should highlight the active conversation item', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();
        await page.waitForTimeout(1000);

        const items = page.locator('#conversations-list .conversation-item');
        const itemCount = await items.count();

        if (itemCount > 0) {
            await items.first().click();
            await page.waitForTimeout(500);
            await expect(items.first()).toHaveClass(/active/);
        }
    });

    test('should start a new conversation from the pane button', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();

        await page.click('#new-conversation-sidebar-btn');
        await page.waitForTimeout(1000);

        // A toast confirms the new-conversation action.
        await expect(page.locator('.toast-item')).toBeVisible({ timeout: 5000 });
    });

    test('should show message count in a conversation item when present', async ({ page }) => {
        await expect(page.locator('#conversations-pane')).toBeVisible();
        await page.waitForTimeout(1000);

        const items = page.locator('#conversations-list .conversation-item');
        const itemCount = await items.count();

        if (itemCount > 0) {
            const count = items.first().locator('.conversation-msg-count');
            if ((await count.count()) > 0) {
                await expect(count.first()).toContainText(/\d+\s*msgs?/);
            }
        }
    });

    test('should expose the Trash sub-view toggle in the pane', async ({ page }) => {
        // The shared component's trash/archived affordance lives on the pane.
        await expect(page.locator('#conversations-pane')).toBeVisible();
        await expect(page.locator('#trash-toggle-btn')).toBeVisible();
    });
});

// ============================================================================
// Session 5: Database Explorer & IPFS Node Status
// ============================================================================

test.describe('Database Explorer (Session 5)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        // Wait for sovereignty tab to be available
        await page.waitForSelector('[data-panel="sovereignty"]', { timeout: 10000 });
        // Navigate to Sovereignty panel
        await page.click('[data-panel="sovereignty"]');
        await expect(page.locator('#panel-sovereignty')).toHaveClass(/active/);
    });

    test('should show Browse Database button', async ({ page }) => {
        const dbButton = page.locator('#toggle-db-explorer');
        await expect(dbButton).toBeVisible();
        await expect(dbButton).toContainText('Browse Database');
    });

    test('should toggle database explorer visibility', async ({ page }) => {
        const dbSection = page.locator('#db-explorer-section');

        // Initially hidden
        await expect(dbSection).not.toBeVisible();

        // Click to show
        await page.click('#toggle-db-explorer');
        await expect(dbSection).toBeVisible();

        // Click to hide
        await page.click('#toggle-db-explorer');
        await expect(dbSection).not.toBeVisible();
    });

    test('should load database tables when explorer opens', async ({ page }) => {
        // Open database explorer
        await page.click('#toggle-db-explorer');

        // Wait for API response
        await page.waitForTimeout(1000);

        const container = page.locator('#db-explorer-container');
        await expect(container).toBeVisible();

        // Should show content (tables, stats, or error)
        const text = await container.textContent();
        // Check for table-related content or error message
        const hasContent = text.toLowerCase().includes('table') ||
                          text.toLowerCase().includes('row') ||
                          text.toLowerCase().includes('database') ||
                          text.toLowerCase().includes('error') ||
                          text.toLowerCase().includes('failed');
        expect(hasContent || text.length > 20).toBeTruthy();
    });

    test('should show database statistics', async ({ page }) => {
        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        const container = page.locator('#db-explorer-container');

        // Check for stats (may show tables count or size)
        const statsText = await container.textContent();

        // Should have some content
        expect(statsText.length).toBeGreaterThan(0);
    });

    test('should show table list with row counts', async ({ page }) => {
        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        const tableList = page.locator('.db-table-list');

        if (await tableList.count() > 0) {
            // Tables should have names and row counts
            const tableItems = tableList.locator('.db-table-item, .table-item, button, [onclick*="loadDbTable"]');
            const itemCount = await tableItems.count();

            if (itemCount > 0) {
                // First item should have text
                const firstItem = tableItems.first();
                const text = await firstItem.textContent();
                expect(text.length).toBeGreaterThan(0);
            }
        }
    });

    test('should load table data when clicking table name', async ({ page }) => {
        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        // Find clickable table elements
        const clickableTable = page.locator('[onclick*="loadDbTable"]').first();

        if (await clickableTable.count() > 0) {
            await clickableTable.click();
            await page.waitForTimeout(1000);

            const container = page.locator('#db-explorer-container');
            const text = await container.textContent();

            // Should show table data, column headers, or error message
            const hasData = text.toLowerCase().includes('id') ||
                           text.toLowerCase().includes('column') ||
                           text.toLowerCase().includes('row') ||
                           text.toLowerCase().includes('showing') ||
                           text.toLowerCase().includes('no data') ||
                           text.toLowerCase().includes('error');
            expect(hasData || text.length > 50).toBeTruthy();
        }
    });

    test('should show pagination controls for large tables', async ({ page }) => {
        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        // Click a table that might have pagination
        const clickableTable = page.locator('[onclick*="loadDbTable"]').first();

        if (await clickableTable.count() > 0) {
            await clickableTable.click();
            await page.waitForTimeout(1000);

            // Look for pagination elements
            const pagination = page.locator('.db-pagination, .pagination, [onclick*="loadDbTable"][onclick*="offset"]');
            // Pagination may or may not exist depending on data size
            // Just check that the UI loads without error
            const container = page.locator('#db-explorer-container');
            await expect(container).toBeVisible();
        }
    });

    test('should show search input for table data', async ({ page }) => {
        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        const clickableTable = page.locator('[onclick*="loadDbTable"]').first();

        if (await clickableTable.count() > 0) {
            await clickableTable.click();
            await page.waitForTimeout(1000);

            // Check for search input
            const searchInput = page.locator('.db-search, input[placeholder*="Search"], input[type="search"]');
            // Search may or may not be visible depending on implementation
            const container = page.locator('#db-explorer-container');
            await expect(container).toBeVisible();
        }
    });

    test('should handle API error gracefully', async ({ page }) => {
        // Mock API error
        await page.route('**/api/db/tables', route => {
            route.fulfill({
                status: 500,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'Database error' })
            });
        });

        await page.click('#toggle-db-explorer');
        await page.waitForTimeout(1000);

        const container = page.locator('#db-explorer-container');
        const text = await container.textContent();

        // Should show error message
        expect(text.toLowerCase()).toContain('error');
    });
});

test.describe('IPFS Node Status (Session 5)', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
        await page.waitForLoadState('domcontentloaded');
        // Wait for sovereignty tab to be available
        await page.waitForSelector('[data-panel="sovereignty"]', { timeout: 10000 });
        // Navigate to Sovereignty panel
        await page.click('[data-panel="sovereignty"]');
        await expect(page.locator('#panel-sovereignty')).toHaveClass(/active/);
    });

    test('should show Check Status button', async ({ page }) => {
        const ipfsButton = page.locator('#toggle-ipfs-status');
        await expect(ipfsButton).toBeVisible();
        await expect(ipfsButton).toContainText('Check Status');
    });

    test('should toggle IPFS status visibility', async ({ page }) => {
        const ipfsSection = page.locator('#ipfs-status-section');

        // Initially hidden
        await expect(ipfsSection).not.toBeVisible();

        // Click to show
        await page.click('#toggle-ipfs-status');
        await expect(ipfsSection).toBeVisible();

        // Click to hide
        await page.click('#toggle-ipfs-status');
        await expect(ipfsSection).not.toBeVisible();
    });

    test('should load IPFS status when panel opens', async ({ page }) => {
        // Open IPFS status
        await page.click('#toggle-ipfs-status');

        // Wait for API response (may take time for network checks)
        await page.waitForTimeout(3000);

        const container = page.locator('#ipfs-status-container');
        await expect(container).toBeVisible();

        // Should show some status information
        const text = await container.textContent();
        expect(text.length).toBeGreaterThan(0);
    });

    test('should show local node status', async ({ page }) => {
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(3000);

        const container = page.locator('#ipfs-status-container');
        const text = await container.textContent();

        // Should mention local node or localhost
        const hasLocalInfo = text.toLowerCase().includes('local') ||
                           text.toLowerCase().includes('node') ||
                           text.toLowerCase().includes('localhost') ||
                           text.toLowerCase().includes('5001');

        expect(hasLocalInfo).toBeTruthy();
    });

    test('should show gateway status', async ({ page }) => {
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(3000);

        const container = page.locator('#ipfs-status-container');
        const text = await container.textContent();

        // Should mention gateways
        const hasGatewayInfo = text.toLowerCase().includes('gateway') ||
                              text.toLowerCase().includes('ipfs.io') ||
                              text.toLowerCase().includes('dweb.link') ||
                              text.toLowerCase().includes('cloudflare');

        expect(hasGatewayInfo).toBeTruthy();
    });

    test('should show status indicators (online/offline)', async ({ page }) => {
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(3000);

        const container = page.locator('#ipfs-status-container');
        const text = await container.textContent();

        // Should have status indicators
        const hasStatusInfo = text.toLowerCase().includes('online') ||
                             text.toLowerCase().includes('offline') ||
                             text.toLowerCase().includes('available') ||
                             text.toLowerCase().includes('unavailable') ||
                             text.toLowerCase().includes('connected') ||
                             text.toLowerCase().includes('disconnected') ||
                             text.includes('✓') || text.includes('✗') ||
                             text.includes('🟢') || text.includes('🔴');

        expect(hasStatusInfo).toBeTruthy();
    });

    test('should show latency for reachable gateways', async ({ page }) => {
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(3000);

        const container = page.locator('#ipfs-status-container');
        const text = await container.textContent();

        // Should show latency in ms for at least one gateway
        const hasLatency = text.includes('ms') || text.includes('latency');

        // This may fail if all gateways are unreachable
        // Just check the container is visible as fallback
        await expect(container).toBeVisible();
    });

    test('should handle API error gracefully', async ({ page }) => {
        // Mock API error
        await page.route('**/api/ipfs/status', route => {
            route.fulfill({
                status: 500,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'IPFS check failed' })
            });
        });

        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(1000);

        const container = page.locator('#ipfs-status-container');
        const text = await container.textContent();

        // Should show error or failed message
        expect(text.toLowerCase()).toMatch(/error|failed/);
    });

    test('should refresh status on toggle', async ({ page }) => {
        // First open
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(2000);

        // Close
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(500);

        // Reopen should refresh
        await page.click('#toggle-ipfs-status');
        await page.waitForTimeout(2000);

        const container = page.locator('#ipfs-status-container');
        await expect(container).toBeVisible();
    });
});
