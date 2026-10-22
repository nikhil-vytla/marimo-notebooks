# /// script
# requires-python = ">=3.11"
# dependencies = ["websocket-client==1.8.0"]
# ///
r"""Capture screenshots and interaction metadata for visual essay references.

The script talks to an already-running Chrome DevTools endpoint. Start one
with:

    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
      --headless=new --remote-debugging-port=9222 about:blank

Then:

    uv run --script docs/visual-essays/capture.py
    uv run --script docs/visual-essays/capture.py --only distill-momentum
    uv run --script docs/visual-essays/capture.py --all --wait 5

Each entry in `references.json` with `"capture": true` becomes a JPEG in
`assets/` plus a record in `captures.json`. The script is deliberately
small: it opens a tab, waits for the page to settle, asks the page for a
few interaction counts, and screenshots the viewport. It does not try to
drive the interaction. That is what the recreation pages are for.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "references.json"
ASSETS = ROOT / "assets"
CAPTURES = ROOT / "captures.json"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/155.0.0.0 Safari/537.36"
)
LOGGER = logging.getLogger(__name__)

PROBE = """
(() => {
  const count = (selector) => document.querySelectorAll(selector).length;
  const meta = document.querySelector('meta[name="description"]');
  return JSON.stringify({
    title: document.title,
    description: meta ? meta.content : "",
    h1: (document.querySelector("h1") || {}).innerText || "",
    canvases: count("canvas"),
    svgs: count("svg"),
    sliders: count("input[type=range]"),
    buttons: count("button"),
    links: count("a"),
    textLength: document.body ? document.body.innerText.length : 0,
    pageHeight: document.documentElement.scrollHeight,
    pageWidth: document.documentElement.scrollWidth
  });
})()
"""


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(
            ws_url, timeout=30, suppress_origin=True
        )
        self.next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        call_id = self.next_id
        self.ws.send(
            json.dumps({"id": call_id, "method": method, "params": params or {}})
        )
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def eval(self, expression: str):
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        return result.get("result", {}).get("value")

    def close(self) -> None:
        self.ws.close()


def devtools(port: int, path: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def open_tab(port: int, url: str) -> dict:
    return devtools(
        port, "/json/new?" + urllib.parse.quote(url, safe=""), method="PUT"
    )


def close_tab(port: int, tab_id: str) -> None:
    try:
        devtools(port, f"/json/close/{tab_id}")
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.debug("could not close tab %s: %s", tab_id, error)


def wait_ready(cdp: CDP, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if cdp.eval("document.readyState") == "complete":
                return
        except (
            RuntimeError,
            OSError,
            ValueError,
            websocket.WebSocketException,
        ) as error:
            LOGGER.debug("waiting for page: %s", error)
        time.sleep(0.4)
    raise TimeoutError("page did not reach readyState=complete")


def capture(entry: dict, port: int, width: int, height: int, wait: float, force: bool) -> dict:
    slug = entry["slug"]
    image = ASSETS / f"{slug}.jpg"
    if image.exists() and not force:
        print(f"skip {slug}: {image.name} exists")
        return {}
    tab = open_tab(port, entry["url"])
    tab_id = tab["id"]
    try:
        cdp = CDP(tab["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        cdp.call("Emulation.setUserAgentOverride", {"userAgent": UA})
        cdp.call("Page.navigate", {"url": entry["url"]})
        wait_ready(cdp)
        time.sleep(wait)
        selector = entry.get("selector")
        if selector:
            found = cdp.eval(
                "Boolean(document.querySelector(" + json.dumps(selector) + "))"
            )
            if found:
                cdp.eval(
                    "document.querySelector(" + json.dumps(selector) + ")"
                    ".scrollIntoView({block: 'center'})"
                )
                time.sleep(0.8)
        scroll_y = int(entry.get("scroll_y", 0))
        if scroll_y:
            cdp.eval(f"window.scrollTo(0, {scroll_y})")
            time.sleep(0.8)
        else:
            cdp.eval("window.scrollTo(0, 0)")
            time.sleep(0.4)
        probe = json.loads(cdp.eval(PROBE))
        shot = cdp.call("Page.captureScreenshot", {
            "format": "jpeg",
            "quality": 84,
            "fromSurface": True,
        })
        image.write_bytes(base64.b64decode(shot["data"]))
        record = {
            **probe,
            "slug": slug,
            "url": entry["url"],
            "image": f"assets/{image.name}",
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "viewport": [width, height],
        }
        print(
            f"captured {slug}: {probe['title'][:60]!r} "
            f"canvas={probe['canvases']} svg={probe['svgs']} "
            f"sliders={probe['sliders']}"
        )
        return record
    finally:
        close_tab(port, tab_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--wait", type=float, default=3.0)
    parser.add_argument("--only", default="", help="comma-separated slugs")
    parser.add_argument("--all", action="store_true", help="ignore capture flag")
    parser.add_argument("--force", action="store_true", help="recapture existing")
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text())
    only = {slug for slug in args.only.split(",") if slug}
    entries = [
        entry for entry in catalog["entries"]
        if (args.all or entry.get("capture")) and (not only or entry["slug"] in only)
    ]
    if not entries:
        print("no entries selected")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    captures = {}
    if CAPTURES.exists():
        captures = json.loads(CAPTURES.read_text())
    for entry in entries:
        record = capture(
            entry, args.port, args.width, args.height, args.wait, args.force
        )
        if record:
            captures[entry["slug"]] = record
            CAPTURES.write_text(json.dumps(captures, indent=2) + "\n")
    print(f"{len(captures)} captures in {CAPTURES}")


if __name__ == "__main__":
    main()
