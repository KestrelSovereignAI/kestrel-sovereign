/**
 * E2E Test for FLUX.1 LoRA Training on Vast.ai
 *
 * Tests the complete "sovereign selfie" pipeline:
 * 1. Verify Vast.ai A100 instance is running
 * 2. Verify Kohya GUI is accessible
 * 3. Verify FLUX.1-dev model is loaded
 * 4. Test LoRA training capabilities
 * 5. Verify inference with trained LoRA
 *
 * Prerequisites:
 * - Running Vast.ai A100 80GB instance with Kohya template
 * - SSH tunnel: ssh -L 7860:localhost:7860 -p <port> root@<host>
 *
 * Run: npx playwright test tests/e2e/test_flux_lora_training.spec.cjs
 */

const { test, expect } = require('@playwright/test');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

// Configuration - update with actual instance details
const KOHYA_GUI_URL = 'http://localhost:7860';
const VASTAI_INSTANCE_ID = '28848338';
const SSH_PORT = '18338';
const SSH_HOST = 'ssh7.vast.ai';

// Skip: Requires running Vast.ai instance with SSH tunnel
// Run manually with: npx playwright test tests/e2e/test_flux_lora_training.spec.cjs
test.describe.skip('FLUX.1 LoRA Training Pipeline', () => {

    test.beforeAll(async () => {
        // Verify SSH tunnel is active
        try {
            const { stdout } = await execAsync(`curl -s -o /dev/null -w "%{http_code}" ${KOHYA_GUI_URL}/ --connect-timeout 5`);
            if (stdout.trim() !== '200') {
                throw new Error(`Kohya GUI not accessible. HTTP status: ${stdout}`);
            }
        } catch (error) {
            console.log('\n=== SSH Tunnel Setup Required ===');
            console.log(`Run: ssh -f -N -L 7860:localhost:7860 -p ${SSH_PORT} root@${SSH_HOST}`);
            throw error;
        }
    });

    test.skip('should verify Vast.ai instance is running', async () => {
        // Check instance status via vastai CLI
        const { stdout } = await execAsync(`
            source .env && VASTAI_API_KEY="$VASTAI_API_KEY" uv run python3 -c "
from vastai_sdk import VastAI
import os
vast = VastAI(api_key=os.environ['VASTAI_API_KEY'])
instances = vast.show_instances()
for inst in instances:
    if inst.get('id') == ${VASTAI_INSTANCE_ID}:
        print('ID:', inst.get('id'))
        print('Status:', inst.get('actual_status'))
        print('GPU:', inst.get('gpu_name'))
        print('VRAM:', str(inst.get('gpu_ram')) + 'MB')
"
        `);

        console.log('\n=== Vast.ai Instance Status ===');
        console.log(stdout);

        expect(stdout).toContain('running');
        expect(stdout).toContain('A100');
        expect(stdout).toContain('81920'); // 80GB VRAM
    });

    test('should access Kohya GUI', async ({ page }) => {
        // Navigate to Kohya GUI
        await page.goto(KOHYA_GUI_URL, { timeout: 30000 });

        // Wait for Gradio interface to load (may take a moment)
        await page.waitForSelector('gradio-app', { timeout: 30000 });

        // Wait for the page to fully load - Gradio apps load async
        await page.waitForLoadState('networkidle', { timeout: 30000 });

        // Verify title or look for Kohya-specific content
        const title = await page.title();
        console.log(`\n=== Kohya GUI Title: ${title} ===`);

        // Take screenshot for evidence
        await page.screenshot({
            path: '/tmp/kohya_gui_screenshot.png',
            fullPage: true
        });
        console.log('Screenshot saved to /tmp/kohya_gui_screenshot.png');

        // Gradio apps may have empty or generic titles - check for gradio-app element instead
        const hasGradioApp = await page.locator('gradio-app').count() > 0;
        expect(hasGradioApp).toBeTruthy();
    });

    test('should verify FLUX.1-dev model is available', async () => {
        // Check model files via SSH
        const { stdout } = await execAsync(`
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${SSH_PORT} root@${SSH_HOST} "
                echo '=== FLUX.1-dev Model Files ==='
                ls -la /workspace/models/flux1-dev/*.safetensors 2>/dev/null | head -5
                echo ''
                echo '=== Model Size ==='
                du -sh /workspace/models/flux1-dev/
                echo ''
                echo '=== Text Encoders ==='
                ls -la /workspace/models/text_encoders/*.safetensors 2>/dev/null
            "
        `);

        console.log('\n=== FLUX.1-dev Model Verification ===');
        console.log(stdout);

        // Verify model files exist
        expect(stdout).toContain('flux1-dev.safetensors');
        expect(stdout).toContain('ae.safetensors');
        expect(stdout).toContain('t5xxl_fp16.safetensors');
    });

    test('should verify trained LoRA exists', async () => {
        // Check for trained LoRA
        const { stdout } = await execAsync(`
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${SSH_PORT} root@${SSH_HOST} "
                echo '=== Trained LoRA Files ==='
                ls -la /workspace/lora_output/*.safetensors 2>/dev/null
                echo ''
                echo '=== LoRA Size ==='
                du -sh /workspace/lora_output/ 2>/dev/null
            "
        `);

        console.log('\n=== Trained LoRA Verification ===');
        console.log(stdout);

        // Verify LoRA was trained
        expect(stdout).toContain('flux_lora_test.safetensors');
    });

    test('should verify GPU memory is available for training', async () => {
        // Check GPU status
        const { stdout } = await execAsync(`
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${SSH_PORT} root@${SSH_HOST} "
                nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,temperature.gpu --format=csv
            "
        `);

        console.log('\n=== GPU Status ===');
        console.log(stdout);

        // Verify A100 80GB
        expect(stdout).toContain('A100');
        expect(stdout).toContain('81920'); // 80GB total
    });

    test('should verify generated test image exists', async () => {
        // Check for generated image
        const { stdout } = await execAsync(`
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${SSH_PORT} root@${SSH_HOST} "
                echo '=== Generated Images ==='
                find /workspace -name '*.png' -newer /workspace/lora_output -type f 2>/dev/null | head -5
                echo ''
                echo '=== Output Directory ==='
                ls -la /workspace/output_test.png/ 2>/dev/null || echo 'No output_test.png directory'
            "
        `);

        console.log('\n=== Generated Image Verification ===');
        console.log(stdout);

        // Verify image was generated
        expect(stdout).toContain('.png');
    });

    test('should run LoRA inference and generate image', async () => {
        // Generate a new image with the trained LoRA
        const { stdout, stderr } = await execAsync(`
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=120 -p ${SSH_PORT} root@${SSH_HOST} "
                cd /opt/workspace-internal/kohya_ss
                source /venv/main/bin/activate

                echo '=== Running Inference with Trained LoRA ==='
                python sd-scripts/flux_minimal_inference.py \\
                    --ckpt /workspace/models/flux1-dev/flux1-dev.safetensors \\
                    --clip_l /workspace/models/flux1-dev/text_encoder/model.safetensors \\
                    --t5xxl /workspace/models/text_encoders/t5xxl_fp16.safetensors \\
                    --ae /workspace/models/flux1-dev/ae.safetensors \\
                    --lora /workspace/lora_output/flux_lora_test.safetensors \\
                    --prompt 'a photo of sks person smiling, portrait, professional headshot' \\
                    --output /workspace/e2e_test_output.png \\
                    --width 512 \\
                    --height 512 \\
                    --steps 10 \\
                    --guidance 3.5 \\
                    --seed 123 \\
                    2>&1 | tail -20

                echo ''
                echo '=== Generated Image ==='
                ls -la /workspace/e2e_test_output.png/ 2>/dev/null | tail -3
            "
        `, { timeout: 180000 }); // 3 minute timeout

        console.log('\n=== Inference Result ===');
        console.log(stdout);
        if (stderr) console.log('stderr:', stderr);

        // Verify inference completed
        expect(stdout).toContain('Saved image to');
        expect(stdout).toContain('.png');
    });

    test('should download generated image locally', async () => {
        // Copy the most recent generated image
        const localPath = '/tmp/e2e_flux_lora_output.png';

        // Get the latest image file and download it
        await execAsync(`
            ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} root@${SSH_HOST} \
                "ls -t /workspace/e2e_test_output.png/*.png | head -1" | \
            xargs -I {} scp -o StrictHostKeyChecking=no -P ${SSH_PORT} \
                "root@${SSH_HOST}:{}" ${localPath}
        `);

        // Verify file was downloaded
        const { stdout } = await execAsync(`ls -la ${localPath}`);
        console.log('\n=== Downloaded Image ===');
        console.log(stdout);

        expect(stdout).toContain('e2e_flux_lora_output.png');

        // Verify file size is reasonable (should be > 100KB for a real image)
        const sizeMatch = stdout.match(/(\d+)\s+\w+\s+\d+/);
        if (sizeMatch) {
            const size = parseInt(sizeMatch[1]);
            expect(size).toBeGreaterThan(100000); // > 100KB
            console.log(`Image size: ${(size/1024).toFixed(1)}KB`);
        }
    });

    test.afterAll(async () => {
        console.log('\n=== Test Summary ===');
        console.log(`Instance ID: ${VASTAI_INSTANCE_ID}`);
        console.log(`SSH: ssh -p ${SSH_PORT} root@${SSH_HOST}`);
        console.log(`Kohya GUI: ${KOHYA_GUI_URL}`);
        console.log('');
        console.log('Generated files saved to /tmp/');
        console.log('');
        console.log('To stop the instance:');
        console.log(`  source .env && VASTAI_API_KEY="$VASTAI_API_KEY" uv run vastai destroy instance ${VASTAI_INSTANCE_ID}`);
    });
});
