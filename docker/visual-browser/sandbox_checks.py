"""Process and Chromium admission checks shared by probe and private worker."""
import os
import resource
from pathlib import Path


def process_evidence():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    capabilities = {name: int(status[name].strip(), 16)
                    for name in ("CapEff", "CapBnd", "CapPrm", "CapInh", "CapAmb")}
    limits, groups = resource.getrlimit(resource.RLIMIT_RTPRIO), os.getgroups()
    if os.getuid() == 0 or any(capabilities.values()) or status["NoNewPrivs"].strip() != "1":
        raise RuntimeError("browser requires non-root, no capabilities and no new privileges")
    if limits != (0, 0) or 0 in groups:
        raise RuntimeError("browser requires no realtime priority and no root group")
    return {"uid": os.getuid(), "capabilities": capabilities, "rtprio_limit": list(limits),
            "groups": groups, "no_new_privileges": True}


async def chromium_evidence(browser):
    page = await browser.new_page(viewport={"width": 800, "height": 600})
    try:
        await page.goto("chrome://sandbox")
        text = await page.locator("body").inner_text()
        for check in ("PID namespaces\tYes", "Network namespaces\tYes", "Seccomp-BPF sandbox\tYes"):
            if check not in text:
                raise RuntimeError("Chromium did not confirm the required sandbox")
        return {"browser_version": browser.version, "sandbox": text}
    finally:
        await page.close()
