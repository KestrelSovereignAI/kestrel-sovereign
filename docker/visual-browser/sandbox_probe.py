"""Synthetic, offline substrate check; not an agent or a computer-use driver."""

import asyncio
import base64
import hashlib
import ipaddress
import json
import re
import signal
import socket
import sys

from sandbox_checks import chromium_evidence, process_evidence


def refuse_owned_peers(fixture):
    if (not isinstance(fixture, dict) or set(fixture) != {"proof_id", "server_id", "network_id", "targets"}
            or not re.fullmatch(r"[a-f0-9]{32}", fixture["proof_id"])
            or any(not re.fullmatch(r"[a-f0-9]{64}", fixture[key]) for key in ("server_id", "network_id"))
            or not isinstance(fixture["targets"], list) or len(fixture["targets"]) != 2):
        raise RuntimeError("exact owned network fixture is required")
    targets = fixture["targets"]
    refused = []
    # Validate every supplied fact before touching any data-plane destination.
    for target, family in zip(targets, (4, 6), strict=True):
        if (not isinstance(target, dict) or set(target) != {"family", "host", "port"}
                or target["family"] != f"ipv{family}" or type(target["port"]) is not int
                or not 1 <= target["port"] <= 65535):
            raise RuntimeError("owned peer endpoint differs")
        address = ipaddress.ip_address(target["host"])
        if (address.version != family or not address.is_private or address.is_loopback
                or address.is_link_local or address.is_multicast or address.is_unspecified):
            raise RuntimeError("owned peer must have an explicit private fixture address")
    for target in targets:
        family = socket.AF_INET if target["family"] == "ipv4" else socket.AF_INET6
        with socket.socket(family) as connection:
            connection.settimeout(.2)
            try:
                connection.connect((target["host"], target["port"]))
            except OSError:
                refused.append({**target, "refused":True})
            else:
                raise RuntimeError("offline container unexpectedly reached an owned peer")
    return {"network_fixture":fixture, "refused_peers":refused}


async def probe(fixture):
    from playwright.async_api import async_playwright
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
            return {
                **evidence,
                "fixture_result": result, **refuse_owned_peers(fixture),
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "png_base64": base64.b64encode(png).decode(),
            }
        finally:
            await browser.close()


if __name__ == "__main__":
    # Even a lost host controller cannot leave this probe's browser running.
    # This is an OS process deadline, separate from the async operation timeout.
    signal.alarm(90)
    if len(sys.argv) == 3 and sys.argv[1] == "--network-only":
        print(json.dumps(refuse_owned_peers(json.loads(sys.argv[2]))))
    elif len(sys.argv) == 2:
        print(json.dumps(asyncio.run(asyncio.wait_for(probe(json.loads(sys.argv[1])), timeout=60))))
    else:
        raise RuntimeError("an exact owned network fixture argument is required")
