# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.64.0"]
# ///
"""Verify the recreation pages and capture previews.

The local Chrome used by the rest of the repo is headless without a GPU,
so it falls back on every WebGL page. This script uses Playwright's
Chromium with software GL, which renders WebGL2 and Three.js correctly.

One-time setup:

    uv run --no-project --with playwright playwright install chromium

Run:

    PLAYWRIGHT_BROWSERS_PATH=.playwright \
      uv run --script docs/visual-essays/recreations/verify.py

WebGPU depends on the host adapter. If no adapter is available, the page
is expected to show its fallback; that is reported separately and does
not fail the run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "previews"
PAGES = [
    ("direct-manipulation", ROOT / "direct-manipulation.html", True),
    ("glsl-field", ROOT / "glsl-field.html", True),
    ("three-scene", ROOT / "three-scene.html", True),
    ("webgpu-field", ROOT / "webgpu-field.html", False),
]
INIT = """
window.__errors = [];
window.addEventListener("error", (event) => {
  window.__errors.push(String(event.message || event.error || "error"));
});
window.addEventListener("unhandledrejection", (event) => {
  window.__errors.push(String(event.reason || "rejection"));
});
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--use-gl=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--no-sandbox",
            ],
        )
        for name, path, required_canvas in PAGES:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.add_init_script(INIT)
            page.goto(path.as_uri(), wait_until="load")
            page.wait_for_timeout(2500)
            result = page.evaluate("""(() => {
              const c = document.querySelector("canvas");
              return {
                status: window.__status || null,
                errors: window.__errors || [],
                canvas: c ? {
                  width: c.width,
                  height: c.height,
                  display: getComputedStyle(c).display,
                } : null,
              };
            })()""")
            page.screenshot(path=str(OUT / f"{name}.jpg"), type="jpeg", quality=84)
            page.close()
            result["name"] = name
            results.append(result)
            print(json.dumps(result, indent=2))

        browser.close()

    bad = []
    for result in results:
        name = result["name"]
        required = next(needs_canvas for n, _, needs_canvas in PAGES if n == name)
        status = result["status"] or {}
        fallback = status.get("error") in {"no-webgpu", "Error: no adapter"}
        if result["errors"]:
            bad.append(f"{name}: runtime errors {result['errors']}")
        if not status.get("ready"):
            bad.append(f"{name}: page did not report ready")
        if required and not (status.get("canvas") and result["canvas"]):
            bad.append(f"{name}: required canvas missing")
        if name == "webgpu-field" and fallback:
            print("webgpu-field: no adapter, fallback is the expected result")
    print(f"\n{len(results)} pages checked, {len(bad)} failures")
    if bad:
        for message in bad:
            print("FAIL", message)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
