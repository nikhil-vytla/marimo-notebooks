# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build the static gallery from references.json and captures.json."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "references.json"
CAPTURES = ROOT / "captures.json"
OUT = ROOT / "index.html"

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual essay reference library</title>
<style>
  :root {
    --paper: #f6f2ea;
    --card: #fffdf8;
    --ink: #292524;
    --muted: #78716c;
    --line: #e7e0d4;
    --accent: #b45309;
    --accent-soft: #fdf0df;
    --blue: #1d4ed8;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.55;
  }
  a { color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 2px; }
  a:hover { color: var(--accent); }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px; }
  header.hero { padding: 64px 0 34px; border-bottom: 1px solid var(--line); }
  .eyebrow {
    font: 650 0.72rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent);
  }
  h1 {
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    font-weight: 600; font-size: clamp(2.4rem, 5vw, 4.4rem);
    line-height: 1.02; letter-spacing: -0.025em; margin: 12px 0 18px;
    max-width: 15ch;
  }
  .lede { max-width: 66ch; font-size: 1.13rem; color: #44403c; margin: 0 0 24px; }
  nav.top { display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.86rem; padding: 18px 0; }
  nav.top a { color: var(--muted); text-decoration: none; }
  nav.top a:hover { color: var(--accent); }
  .stats { display: flex; gap: 28px; flex-wrap: wrap; padding: 26px 0 10px; }
  .stat strong { display: block; font-family: "Iowan Old Style", Georgia, serif; font-size: 1.7rem; line-height: 1; }
  .stat span { font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .controls {
    position: sticky; top: 0; z-index: 5; background: rgba(246,242,234,0.94);
    backdrop-filter: blur(10px); border-bottom: 1px solid var(--line);
    padding: 14px 0; margin-bottom: 24px;
  }
  .controls .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  label.field { display: flex; flex-direction: column; gap: 4px; font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  input[type="search"], select {
    font: inherit; font-size: 0.9rem; color: var(--ink); background: var(--card);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; min-width: 170px;
  }
  input[type="search"]:focus, select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  .count { margin-left: auto; font-size: 0.8rem; color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 18px; padding-bottom: 54px; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } h1 { max-width: 100%; } }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    overflow: hidden; display: flex; flex-direction: column; min-width: 0;
  }
  .shot { aspect-ratio: 16 / 10; background: #e7e0d4; overflow: hidden; display: block; }
  .shot img { width: 100%; height: 100%; object-fit: cover; object-position: top center; display: block; }
  .fallback {
    width: 100%; height: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 6px; padding: 18px;
    text-align: center;
    background: radial-gradient(circle at 20% 20%, #fdf0df 0%, #f3e7d3 52%, #e7dcc9 100%);
  }
  .fallback span {
    font-family: "Iowan Old Style", Georgia, serif; font-size: 1.5rem;
    line-height: 1.05; color: #57534e;
  }
  .fallback small {
    font: 650 0.64rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.1em; text-transform: uppercase; color: #92400e;
  }
  .card-body { padding: 16px 18px 18px; display: flex; flex-direction: column; gap: 10px; }
  .meta { display: flex; gap: 8px; align-items: center; font: 600 0.68rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .meta .site { color: var(--accent); }
  .card h2 { font-family: "Iowan Old Style", Georgia, serif; font-size: 1.36rem; line-height: 1.15; margin: 0; letter-spacing: -0.01em; }
  .card h2 a { color: var(--ink); text-decoration: none; }
  .card h2 a:hover { color: var(--accent); }
  .why { margin: 0; color: #44403c; font-size: 0.91rem; }
  .steal { border-left: 3px solid var(--accent); padding: 2px 0 2px 10px; font-size: 0.87rem; color: #57534e; }
  .steal b { color: var(--ink); font-weight: 650; }
  .tags { display: flex; gap: 6px; flex-wrap: wrap; }
  .tag { background: var(--accent-soft); color: #92400e; border-radius: 999px; padding: 2px 8px; font-size: 0.68rem; }
  .marimo { font-size: 0.78rem; color: var(--muted); border-top: 1px dashed var(--line); padding-top: 10px; }
  .marimo code { font-size: 0.76rem; background: #f4eee4; padding: 1px 4px; border-radius: 4px; }
  .empty { grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--muted); }
  footer { border-top: 1px solid var(--line); padding: 28px 0 60px; color: var(--muted); font-size: 0.82rem; }
  .recreations { padding: 10px 0 42px; border-bottom: 1px solid var(--line); }
  .recreations h2 { font-family: "Iowan Old Style", Georgia, serif; font-size: 1.8rem; margin: 0 0 6px; }
  .recreations p { max-width: 66ch; color: #57534e; }
  .recreations .links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
  .recreations .links a { border: 1px solid var(--line); background: var(--card); border-radius: 999px; padding: 6px 12px; text-decoration: none; color: var(--ink); font-size: 0.82rem; }
  .recreations .links a:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <div class="eyebrow">Reference library · captured October 2026</div>
    <h1>Visual essays that make you operate something</h1>
    <p class="lede">A working shelf of interactive explanations, annotated diagrams, playable simulations, and live shaders. Each card names the mechanism worth stealing and how it maps onto marimo. The catalog is the argument; the screenshots are evidence.</p>
    <nav class="top">
      <a href="#catalog">Catalog</a>
      <a href="#recreations">Recreations</a>
      <a href="README.md">Field notes</a>
      <a href="capture.py">Capture tool</a>
    </nav>
    <div class="stats">
      <div class="stat"><strong>__N_ENTRIES__</strong><span>references</span></div>
      <div class="stat"><strong>__N_CAPTURES__</strong><span>screenshots</span></div>
      <div class="stat"><strong>__N_SITES__</strong><span>sites</span></div>
      <div class="stat"><strong>__N_PATTERNS__</strong><span>patterns</span></div>
    </div>
  </div>
</header>

<section class="recreations" id="recreations">
  <div class="wrap">
    <h2>Recreations</h2>
    <p>Small, dependency-light scaffolds for the patterns that are awkward to express with a chart: a draggable canvas, a GLSL field, a Three.js scene, and a WebGPU shader with a fallback.</p>
    <div class="links">
      <a href="recreations/index.html">Open the lab</a>
      <a href="recreations/direct-manipulation.html">Direct manipulation</a>
      <a href="recreations/glsl-field.html">GLSL field</a>
      <a href="recreations/three-scene.html">Three.js scene</a>
      <a href="recreations/webgpu-field.html">WebGPU field</a>
    </div>
  </div>
</section>

<div class="controls" id="catalog">
  <div class="wrap row">
    <label class="field">Search
      <input id="q" type="search" placeholder="title, site, topic, note" autocomplete="off">
    </label>
    <label class="field">Pattern
      <select id="pattern"><option value="">All patterns</option></select>
    </label>
    <label class="field">Site
      <select id="site"><option value="">All sites</option></select>
    </label>
    <span class="count" id="count"></span>
  </div>
</div>

<main class="wrap">
  <div class="grid" id="grid"></div>
</main>

<footer>
  <div class="wrap">
    Screenshots are for reference and critique. Every page remains the work of its author. Built from
    <code>references.json</code> and <code>captures.json</code>. To recapture a page, run
    <code>uv run --script docs/visual-essays/capture.py --only &lt;slug&gt; --force</code>.
  </div>
</footer>

<script id="data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("data").textContent);
const state = { q: "", pattern: "", site: "" };
const grid = document.getElementById("grid");
const count = document.getElementById("count");
const qInput = document.getElementById("q");
const patternSelect = document.getElementById("pattern");
const siteSelect = document.getElementById("site");

const esc = (value) => String(value).replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[ch]));

const allPatterns = [...new Set(data.flatMap((entry) => entry.patterns))].sort();
const allSites = [...new Set(data.map((entry) => entry.site))].sort();
for (const pattern of allPatterns) {
  const option = document.createElement("option");
  option.value = pattern;
  option.textContent = pattern.replaceAll("-", " ");
  patternSelect.appendChild(option);
}
for (const site of allSites) {
  const option = document.createElement("option");
  option.value = site;
  option.textContent = site;
  siteSelect.appendChild(option);
}

function matches(entry) {
  const haystack = [
    entry.title, entry.site, entry.author, entry.topics.join(" "),
    entry.patterns.join(" "), entry.why, entry.steal
  ].join(" ").toLowerCase();
  return (!state.q || haystack.includes(state.q))
    && (!state.pattern || entry.patterns.includes(state.pattern))
    && (!state.site || entry.site === state.site);
}

function card(entry) {
  const image = entry.image
    ? `<img src="${esc(entry.image)}" alt="Screenshot of ${esc(entry.title)}" loading="lazy">`
    : `<div class="fallback"><span>${esc(entry.site)}</span><small>${esc(entry.patterns[0].replaceAll("-", " "))}</small></div>`;
  const tags = entry.patterns.slice(0, 3).map((pattern) =>
    `<span class="tag">${esc(pattern.replaceAll("-", " "))}</span>`).join("");
  return `<article class="card">
    <a class="shot" href="${esc(entry.url)}" target="_blank" rel="noopener">${image}</a>
    <div class="card-body">
      <div class="meta"><span class="site">${esc(entry.site)}</span><span>${esc(entry.year)}</span></div>
      <h2><a href="${esc(entry.url)}" target="_blank" rel="noopener">${esc(entry.title)}</a></h2>
      <p class="why">${esc(entry.why)}</p>
      <div class="steal"><b>Steal this.</b> ${esc(entry.steal)}</div>
      <div class="tags">${tags}</div>
      <div class="marimo"><b>Marimo.</b> ${esc(entry.marimo)}</div>
    </div>
  </article>`;
}

function render() {
  const visible = data.filter(matches);
  grid.innerHTML = visible.length
    ? visible.map(card).join("")
    : `<div class="empty">No references match that filter.</div>`;
  count.textContent = `${visible.length} of ${data.length}`;
}

qInput.addEventListener("input", () => { state.q = qInput.value.trim().toLowerCase(); render(); });
patternSelect.addEventListener("change", () => { state.pattern = patternSelect.value; render(); });
siteSelect.addEventListener("change", () => { state.site = siteSelect.value; render(); });
render();
</script>
</body>
</html>
"""


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    captures = json.loads(CAPTURES.read_text()) if CAPTURES.exists() else {}
    entries = []
    for entry in catalog["entries"]:
        record = captures.get(entry["slug"], {})
        image = record.get("image") if entry.get("capture") else None
        if image and not (ROOT / image).exists():
            image = None
        entries.append({
            **entry,
            "image": image,
        })
    patterns = Counter(
        pattern for entry in entries for pattern in entry["patterns"]
    )
    sites = {entry["site"] for entry in entries}
    payload = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    document = (
        TEMPLATE.replace("__DATA__", payload)
        .replace("__N_ENTRIES__", str(len(entries)))
        .replace("__N_CAPTURES__", str(sum(1 for e in entries if e["image"])))
        .replace("__N_SITES__", str(len(sites)))
        .replace("__N_PATTERNS__", str(len(patterns)))
    )
    OUT.write_text(document)
    print(
        f"wrote {OUT} ({len(entries)} entries, "
        f"{sum(1 for e in entries if e['image'])} images)"
    )


if __name__ == "__main__":
    main()
