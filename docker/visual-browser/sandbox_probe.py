"""Synthetic, offline substrate check; not an agent or a computer-use driver."""

import asyncio
import base64
import hashlib
import json
import signal
import socket

from playwright.async_api import async_playwright
from sandbox_checks import chromium_evidence, process_evidence


async def probe():
    evidence = process_evidence()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True, chromium_sandbox=True, channel="chromium",
        )
        try:
            evidence.update(await chromium_evidence(browser))
            page = await browser.new_page(viewport={"width": 800, "height": 600})
            await page.set_content(
                '<!doctype html><title>Isolated browser fixture</title>'
                '<label>Report <input aria-label="Report"></label>'
                '<button onclick="document.querySelector(\'output\').textContent='
                'document.querySelector(\'input\').value">Save draft</button><output></output>'
            )
            await page.get_by_label("Report").fill("Persistent local browser draft")
            await page.get_by_role("button", name="Save draft").click()
            result = await page.locator("output").inner_text()
            if result != "Persistent local browser draft":
                raise RuntimeError("browser actions did not preserve fixture state")
            png = await page.screenshot()
            refused = []
            for address in ("1.1.1.1", "169.254.169.254"):
                try:
                    connection = socket.create_connection((address, 443), timeout=1)
                except OSError:
                    refused.append(address)
                else:
                    connection.close()
                    raise RuntimeError("offline container unexpectedly reached a network destination")
            return {
                **evidence,
                "fixture_result": result, "refused_destinations": refused,
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "png_base64": base64.b64encode(png).decode(),
            }
        finally:
            await browser.close()


if __name__ == "__main__":
    # Even a lost host controller cannot leave this probe's browser running.
    # This is an OS process deadline, separate from the async operation timeout.
    signal.alarm(90)
    print(json.dumps(asyncio.run(asyncio.wait_for(probe(), timeout=60))))
