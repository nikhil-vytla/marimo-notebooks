# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
#     "numpy==2.5.3",
#     "pandas==3.0.6",
#     "altair==6.3.0",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, json, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
        <style>
        :root {
          --ink: #171717;
          --muted: #78716c;
          --line: #e7e5e4;
          --paper: #fbfaf7;
          --accent: #b45309;
          --accent-soft: #fffbeb;
          --blue: #1d4ed8;
          --green: #047857;
          --red: #b91c1c;
        }
        .essay-hero {
          padding: 2.2rem 0 1.3rem;
          border-bottom: 1px solid var(--line);
          margin-bottom: 1.2rem;
        }
        .essay-kicker {
          font: 650 0.72rem/1.2 ui-sans-serif, system-ui, sans-serif;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--accent);
        }
        .essay-hero h1 {
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-size: clamp(2.1rem, 4vw, 3.35rem);
          line-height: 1.02;
          letter-spacing: -0.02em;
          margin: 0.5rem 0 0.8rem;
          color: var(--ink);
        }
        .essay-lede {
          font-size: 1.13rem;
          line-height: 1.55;
          color: #44403c;
          max-width: 46rem;
        }
        .essay-meta {
          display: flex;
          gap: 0.75rem;
          flex-wrap: wrap;
          align-items: center;
          font-size: 0.82rem;
          color: var(--muted);
          margin-top: 1rem;
        }
        .status-pill {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          border-radius: 999px;
          padding: 0.18rem 0.55rem;
          font-size: 0.76rem;
          font-weight: 650;
          background: #f5f5f4;
          color: #57534e;
        }
        .status-pill.yes { background: #ecfdf5; color: var(--green); }
        .status-pill.partly { background: var(--accent-soft); color: var(--accent); }
        .status-pill.no { background: #fef2f2; color: var(--red); }
        .essay-toc {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem 1rem;
          padding: 0.7rem 0 0.9rem;
          border-bottom: 1px solid var(--line);
          margin-bottom: 1.3rem;
          font-size: 0.84rem;
        }
        .essay-toc a { color: var(--muted); text-decoration: none; }
        .essay-toc a:hover { color: var(--accent); }
        .essay-note {
          border-left: 3px solid var(--accent);
          background: var(--accent-soft);
          padding: 0.9rem 1.1rem;
          border-radius: 0 0.5rem 0.5rem 0;
          margin: 1.2rem 0;
          font-size: 0.94rem;
          line-height: 1.5;
        }
        .verdict-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
          gap: 0.65rem;
          margin: 1rem 0 1.4rem;
        }
        .verdict-card {
          border: 1px solid var(--line);
          border-radius: 0.7rem;
          padding: 0.85rem 0.9rem;
          background: #fff;
        }
        .verdict-kicker {
          font: 650 0.7rem/1.2 ui-monospace, SFMono-Regular, monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .verdict-number {
          font-family: "Iowan Old Style", Georgia, serif;
          font-size: 1.55rem;
          line-height: 1.1;
          margin: 0.35rem 0 0.2rem;
        }
        .verdict-card.yes .verdict-number { color: var(--green); }
        .verdict-card.no .verdict-number { color: var(--red); }
        .verdict-card.partly .verdict-number { color: var(--accent); }
        .verdict-note { font-size: 0.8rem; line-height: 1.4; color: var(--muted); }
        .matrix {
          border: 1px solid var(--line);
          border-radius: 0.8rem;
          background: #fff;
          padding: 0.9rem;
          margin: 1rem 0 1.2rem;
        }
        .matrix-row {
          display: flex;
          gap: 0.8rem;
          align-items: center;
          padding: 0.45rem 0;
        }
        .matrix-row + .matrix-row { border-top: 1px solid #f5f5f4; }
        .matrix-label {
          width: 5.4rem;
          flex: 0 0 5.4rem;
          font: 650 0.72rem/1.2 ui-monospace, SFMono-Regular, monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .matrix-cells { display: flex; gap: 0.4rem; flex-wrap: wrap; }
        .matrix-cell {
          border-radius: 0.45rem;
          padding: 0.3rem 0.55rem;
          font-size: 0.78rem;
          font-weight: 650;
          min-width: 3.4rem;
          text-align: center;
        }
        .matrix-cell.yes { background: #ecfdf5; color: var(--green); }
        .matrix-cell.partly { background: var(--accent-soft); color: var(--accent); }
        .matrix-cell.no { background: #fef2f2; color: var(--red); }
        .figure-caption {
          font-size: 0.8rem;
          line-height: 1.45;
          color: var(--muted);
          margin: 0.3rem 0 1rem;
        }
        .essay-provenance {
          border-top: 1px solid var(--line);
          margin-top: 2.2rem;
          padding-top: 1rem;
          font-size: 0.82rem;
          line-height: 1.5;
          color: var(--muted);
          display: grid;
          gap: 0.4rem;
        }
        .essay-provenance strong { color: var(--ink); }
        .essay-provenance a { color: var(--blue); text-decoration: none; }
        @media (max-width: 640px) {
          .essay-hero { padding-top: 1.2rem; }
          .essay-hero h1 { font-size: 2rem; }
          .matrix-row { align-items: flex-start; }
          .matrix-label { width: 4.6rem; flex-basis: 4.6rem; }
        }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _():
    META = {
        "kicker": "Visual essay · reproduction and extension",
        "title": "When does a self-predicting network get simpler?",
        "lede": (
            "On MNIST, self-modeling narrows the classifier. On CIFAR, the "
            "ordering inverts. On IMDB, the effect disappears. A toy dial "
            "shows the scale cheat behind one failure. A slowed self-model "
            "head shows the other is a mechanism that dies before it acts."
        ),
        "paper": "Unexpected Benefits of Self-Modeling in Neural Systems",
        "paper_url": "https://arxiv.org/abs/2407.10188",
        "authors": "Premakumar et al., 2024",
        "status": "v1 · guided cut",
        "status_class": "partly",
        "reading_time": "15 min",
        "updated": "2026-10-19",
        "repo": "https://github.com/nikhil-vytla/marimo-notebooks",
    }
    SCORE = [
        {"area": "MNIST", "panel": "2A", "status": "yes"},
        {"area": "MNIST", "panel": "2B", "status": "yes"},
        {"area": "MNIST", "panel": "2C", "status": "yes"},
        {"area": "MNIST", "panel": "2D", "status": "yes"},
        {"area": "CIFAR", "panel": "3A", "status": "no"},
        {"area": "CIFAR", "panel": "3B", "status": "no"},
        {"area": "CIFAR", "panel": "3C", "status": "yes"},
        {"area": "IMDB", "panel": "4A", "status": "no"},
        {"area": "IMDB", "panel": "4B", "status": "no"},
        {"area": "IMDB", "panel": "4C", "status": "partly"},
    ]
    return META, SCORE


@app.cell(hide_code=True)
def _(META, mo):
    mo.Html(
        f"""
        <header class="essay-hero">
          <div class="essay-kicker">{META["kicker"]}</div>
          <h1>{META["title"]}</h1>
          <p class="essay-lede">{META["lede"]}</p>
          <div class="essay-meta">
            <span class="status-pill {META["status_class"]}">
              {META["status"]}
            </span>
            <span>{META["reading_time"]}</span>
            <span>Updated {META["updated"]}</span>
            <span>{META["authors"]}</span>
          </div>
        </header>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
        <nav class="essay-toc">
          <a href="#the-paradox">The paradox</a>
          <a href="#the-mechanism">The mechanism</a>
          <a href="#the-stop-gradient">The stop-gradient</a>
          <a href="#the-pressure">The pressure</a>
          <a href="#the-scoreboard">The scoreboard</a>
          <a href="#what-next">What next</a>
        </nav>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
        <div class="essay-note">
          <strong>How to read this.</strong>
          This is the guided cut: the argument, the mechanism, and the one
          experiment that changes the interpretation. The
          <a href="https://github.com/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_essay.py">lab notebook</a>
          has every panel, probe, and RLCT detail. Read this for the
          argument; open the lab for the evidence.
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The paradox

    The paper's claim is simple. Add a self-prediction term to the loss,
    and the network gets simpler: the classifier weights narrow, a formal
    complexity measure falls, and test accuracy barely moves.

    We reproduced it on MNIST. The same recipe fails, in different
    ways, on CIFAR-10 and IMDB.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
        <div class="verdict-grid">
          <div class="verdict-card yes">
            <div class="verdict-kicker">MNIST · Figure 2A</div>
            <div class="verdict-number">0.0974 → 0.0812</div>
            <div class="verdict-note">
              Narrower weights with more self-modeling. The paper's
              effect, reproduced within 0.001.
            </div>
          </div>
          <div class="verdict-card no">
            <div class="verdict-kicker">CIFAR-10 · Figure 3A</div>
            <div class="verdict-number">0.0252 → 0.0292</div>
            <div class="verdict-note">
              The ordering inverts: the baseline is the narrowest curve.
              Accuracy still matches.
            </div>
          </div>
          <div class="verdict-card no">
            <div class="verdict-kicker">IMDB · Figure 4A</div>
            <div class="verdict-number">0.232 flat</div>
            <div class="verdict-note">
              No separation across AW, and about 9x below the paper's 2.05.
              Accuracy matches at 85.2%.
            </div>
          </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One number does not explain all three. The rest of the essay is the
    search for what does.

    Two candidates survive. First, the self-model loss can be gamed: if
    the target moves, the cheapest way to reduce the error is to shrink
    the thing being predicted. Second, the pressure can die before it
    acts. On IMDB the self-model gradient share falls to about 2% by
    epoch 20, while MNIST stays at 14% to 18%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mechanism

    A hidden layer produces a cloud of activations. A classifier reads
    the cloud. A self-model head tries to predict it from the same
    hidden layer.

    The self-model term is happy when the cloud is easy to predict. The
    cheapest way to make it easy is to shrink the cloud. The classifier
    then grows to keep the logits fixed.
    """)
    return


@app.cell(hide_code=True)
def _(json, mo):
    DIAL = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "compression_dial.json"
        ).read_text()
    )
    return (DIAL,)


@app.cell(hide_code=True)
def _(DIAL, mo):
    import anywidget
    import traitlets

    class CompressionDial(anywidget.AnyWidget):
        _esm = r"""
        function render({ model, el }) {
          el.classList.add("compression-dial");
          el.innerHTML = `
            <style>
              .compression-dial {
                font-family: ui-sans-serif, system-ui, sans-serif;
                color: #171717;
                --cd-muted: #78716c;
                --cd-line: #e7e5e4;
                --cd-accent: #b45309;
                --cd-green: #047857;
              }
              .compression-dial .cd-head {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 4px;
              }
              .compression-dial .cd-title { font-size: 16px; font-weight: 650; }
              .compression-dial .cd-sub {
                font-size: 12.5px;
                line-height: 1.35;
                color: var(--cd-muted);
                max-width: 430px;
                margin-top: 2px;
              }
              .compression-dial .cd-toggle {
                border: 1px solid var(--cd-line);
                background: #fff;
                border-radius: 999px;
                padding: 6px 11px;
                font: 600 12px/1 ui-sans-serif, system-ui, sans-serif;
                cursor: pointer;
                white-space: nowrap;
              }
              .compression-dial .cd-toggle.on {
                background: #ecfdf5;
                border-color: #a7f3d0;
                color: var(--cd-green);
              }
              .compression-dial .cd-toggle.off {
                background: #fffbeb;
                border-color: #fde68a;
                color: var(--cd-accent);
              }
              .compression-dial .cd-svg {
                width: 100%;
                height: auto;
                display: block;
                margin-top: 6px;
              }
              .compression-dial .cd-legend {
                display: flex;
                gap: 14px;
                flex-wrap: wrap;
                font-size: 12px;
                color: var(--cd-muted);
                margin-top: 2px;
              }
              .compression-dial .cd-dot {
                width: 9px;
                height: 9px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
                vertical-align: middle;
              }
              .compression-dial .cd-foot {
                font-size: 12px;
                line-height: 1.45;
                color: var(--cd-muted);
                margin-top: 6px;
              }
            </style>
            <div class="cd-root">
              <div class="cd-head">
                <div>
                  <div class="cd-title">Compression dial</div>
                  <div class="cd-sub">Drag the dial to add self-model pressure. The classifier has to keep the logits fixed.</div>
                </div>
                <button class="cd-toggle on" type="button" aria-pressed="true">detached target</button>
              </div>
              <svg class="cd-svg" viewBox="0 0 640 470" role="img" aria-label="Hidden activations and classifier norms as the self-model weight changes"></svg>
              <div class="cd-legend">
                <span><span class="cd-dot" style="background:#1d4ed8"></span>class 0</span>
                <span><span class="cd-dot" style="background:#b45309"></span>class 1</span>
                <span><span class="cd-dot" style="background:#047857"></span>class 2</span>
              </div>
              <div class="cd-foot">Both modes drive the self-model error to <strong>0.000</strong>. Only one keeps the hidden representation at its natural scale.</div>
            </div>
          `;
          const svg = el.querySelector(".cd-svg");
          const NS = "http://www.w3.org/2000/svg";
          const W = 640, H = 470;
          const panel = { x: 16, y: 48, w: 320, h: 320 };
          const world = Number(model.get("world")) || 5.5;
          const colors = ["#1d4ed8", "#b45309", "#047857"];
          const labels = model.get("labels") || [];
          const maxA = Number(model.get("max_a")) || 1;
          const maxW = Number(model.get("max_w")) || 1;
          const cx = panel.x + panel.w / 2;
          const cy = panel.y + panel.h / 2;

          function mk(name, attrs) {
            const node = document.createElementNS(NS, name);
            for (const [key, value] of Object.entries(attrs)) {
              node.setAttribute(key, String(value));
            }
            return node;
          }
          function sx(x) { return panel.x + ((x + world) / (2 * world)) * panel.w; }
          function sy(y) { return panel.y + ((world - y) / (2 * world)) * panel.h; }
          function polar(centerX, centerY, radius, angle) {
            const rad = (angle * Math.PI) / 180;
            return {
              x: centerX + radius * Math.cos(rad),
              y: centerY + radius * Math.sin(rad),
            };
          }
          function arcPath(centerX, centerY, radius, start, end) {
            const a = polar(centerX, centerY, radius, start);
            const b = polar(centerX, centerY, radius, end);
            const large = end - start <= 180 ? 0 : 1;
            return `M ${a.x} ${a.y} A ${radius} ${radius} 0 ${large} 1 ${b.x} ${b.y}`;
          }
          function lerp(a, b, t) { return a + (b - a) * t; }

          svg.appendChild(mk("rect", {
            x: panel.x, y: panel.y, width: panel.w, height: panel.h,
            rx: 14, fill: "#fff", stroke: "#e7e5e4",
          }));

          const grid = mk("g", {});
          for (const tick of [-4, -2, 0, 2, 4]) {
            grid.appendChild(mk("line", {
              x1: sx(tick), y1: panel.y, x2: sx(tick), y2: panel.y + panel.h,
              stroke: tick === 0 ? "#d6d3d1" : "#f5f5f4",
            }));
            grid.appendChild(mk("line", {
              x1: panel.x, y1: sy(tick), x2: panel.x + panel.w, y2: sy(tick),
              stroke: tick === 0 ? "#d6d3d1" : "#f5f5f4",
            }));
          }
          svg.appendChild(grid);

          const ringRadius = (2.5 / (2 * world)) * panel.w;
          svg.appendChild(mk("circle", {
            cx, cy, r: ringRadius, fill: "none", stroke: "#d6d3d1",
            "stroke-dasharray": "4 4",
          }));
          const ringLabel = mk("text", {
            x: cx, y: cy - ringRadius - 7,
            "text-anchor": "middle", "font-size": 10, fill: "#a8a29e",
          });
          ringLabel.textContent = "|a| = 2.5";
          svg.appendChild(ringLabel);

          const pointsGroup = mk("g", {});
          svg.appendChild(pointsGroup);
          const circles = labels.map((label) => {
            const circle = mk("circle", {
              r: 3.2, fill: colors[label] || "#57534e", opacity: 0.82,
            });
            pointsGroup.appendChild(circle);
            return circle;
          });

          const gauges = [
            { key: "a_norm", label: "hidden state |a|", max: maxA, y: 66 },
            { key: "w_norm", label: "classifier |W|", max: maxW, y: 126 },
          ];
          const gaugeNodes = gauges.map((gauge) => {
            const label = mk("text", {
              x: 372, y: gauge.y, "font-size": 12, fill: "#57534e",
              "font-weight": 600,
            });
            label.textContent = gauge.label;
            svg.appendChild(label);
            svg.appendChild(mk("rect", {
              x: 372, y: gauge.y + 8, width: 200, height: 12, rx: 6,
              fill: "#f5f5f4",
            }));
            const fill = mk("rect", {
              x: 372, y: gauge.y + 8, width: 0, height: 12, rx: 6,
              fill: "#b45309",
            });
            svg.appendChild(fill);
            const value = mk("text", {
              x: 580, y: gauge.y + 18, "font-size": 12, fill: "#57534e",
              "text-anchor": "end", "font-variant-numeric": "tabular-nums",
            });
            svg.appendChild(value);
            return { ...gauge, fill, value };
          });

          const smLabel = mk("text", {
            x: 372, y: 196, "font-size": 12, fill: "#57534e", "font-weight": 600,
          });
          smLabel.textContent = "self-model error";
          svg.appendChild(smLabel);
          const smValue = mk("text", {
            x: 580, y: 196, "font-size": 12, fill: "#57534e",
            "text-anchor": "end", "font-variant-numeric": "tabular-nums",
          });
          smValue.textContent = "0.000";
          svg.appendChild(smValue);
          const smNote = mk("text", {
            x: 372, y: 214, "font-size": 11, fill: "#a8a29e",
          });
          smNote.textContent = "both modes";
          svg.appendChild(smNote);

          const dialX = 176, dialY = 414, dialR = 38;
          svg.appendChild(mk("path", {
            d: arcPath(dialX, dialY, dialR, 135, 405),
            fill: "none", stroke: "#f5f5f4", "stroke-width": 9,
            "stroke-linecap": "round",
          }));
          const active = mk("path", {
            d: arcPath(dialX, dialY, dialR, 135, 140),
            fill: "none", stroke: "#b45309", "stroke-width": 9,
            "stroke-linecap": "round",
          });
          svg.appendChild(active);
          const handle = mk("circle", {
            cx: dialX, cy: dialY, r: 9, fill: "#fff", stroke: "#b45309",
            "stroke-width": 3,
          });
          svg.appendChild(handle);
          for (const tick of [0, 10, 20, 30, 40, 50]) {
            const angle = 135 + (tick / 50) * 270;
            const a = polar(dialX, dialY, dialR - 8, angle);
            const b = polar(dialX, dialY, dialR - 13, angle);
            svg.appendChild(mk("line", {
              x1: a.x, y1: a.y, x2: b.x, y2: b.y,
              stroke: "#d6d3d1", "stroke-width": 1.5,
            }));
          }
          const awText = mk("text", {
            x: dialX, y: dialY + 4, "text-anchor": "middle",
            "font-size": 20, "font-weight": 700, fill: "#171717",
            "font-variant-numeric": "tabular-nums",
          });
          awText.textContent = "5";
          svg.appendChild(awText);
          const awLabel = mk("text", {
            x: dialX, y: dialY + 20, "text-anchor": "middle",
            "font-size": 10, "letter-spacing": "0.12em", fill: "#a8a29e",
          });
          awLabel.textContent = "AW";
          svg.appendChild(awLabel);

          const hit = mk("circle", {
            cx: dialX, cy: dialY, r: 56, fill: "transparent",
            style: "cursor: grab; touch-action: none;",
            tabindex: 0, role: "slider",
            "aria-valuemin": 0, "aria-valuemax": 50, "aria-valuenow": 5,
            "aria-label": "Self-model weight AW",
          });
          svg.appendChild(hit);

          const toggle = el.querySelector(".cd-toggle");
          let dragging = false;

          function frameAt(aw) {
            const mode = model.get("detach") ? "detach" : "no_detach";
            const frames = (model.get("frames") || {})[mode] || [];
            if (!frames.length) return null;
            let i = 0;
            while (i < frames.length - 2 && frames[i + 1].aw < aw) i += 1;
            const a = frames[i];
            const b = frames[Math.min(i + 1, frames.length - 1)];
            const t = b.aw === a.aw ? 0 : Math.max(0, Math.min(1, (aw - a.aw) / (b.aw - a.aw)));
            return { a, b, t };
          }

          function update() {
            const aw = Number(model.get("aw"));
            const pair = frameAt(aw);
            if (!pair) return;
            const { a, b, t } = pair;
            for (let i = 0; i < circles.length; i += 1) {
              const p0 = a.points[i];
              const p1 = b.points[i];
              circles[i].setAttribute("cx", sx(lerp(p0[0], p1[0], t)));
              circles[i].setAttribute("cy", sy(lerp(p0[1], p1[1], t)));
            }
            const values = {
              a_norm: lerp(a.a_norm, b.a_norm, t),
              w_norm: lerp(a.w_norm, b.w_norm, t),
            };
            for (const gauge of gaugeNodes) {
              const width = Math.max(0, Math.min(1, values[gauge.key] / gauge.max)) * 200;
              gauge.fill.setAttribute("width", width);
              gauge.value.textContent = values[gauge.key].toFixed(2);
            }
            const angle = 135 + (Math.max(0, Math.min(50, aw)) / 50) * 270;
            active.setAttribute("d", arcPath(dialX, dialY, dialR, 135, Math.max(135.01, angle)));
            const pos = polar(dialX, dialY, dialR, angle);
            handle.setAttribute("cx", pos.x);
            handle.setAttribute("cy", pos.y);
            awText.textContent = Math.round(aw);
            hit.setAttribute("aria-valuenow", String(Math.round(aw)));
            const detached = !!model.get("detach");
            toggle.textContent = detached ? "detached target" : "no detach";
            toggle.classList.toggle("on", detached);
            toggle.classList.toggle("off", !detached);
            toggle.setAttribute("aria-pressed", String(detached));
          }

          function setAw(value) {
            const aw = Math.max(0, Math.min(50, value));
            model.set("aw", aw);
            model.save_changes();
            update();
          }

          function pointerToAw(event) {
            const rect = svg.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * W;
            const y = ((event.clientY - rect.top) / rect.height) * H;
            let angle = (Math.atan2(y - dialY, x - dialX) * 180) / Math.PI;
            if (angle < 135) angle += 360;
            angle = Math.max(135, Math.min(405, angle));
            return ((angle - 135) / 270) * 50;
          }

          hit.addEventListener("pointerdown", (event) => {
            dragging = true;
            hit.setPointerCapture(event.pointerId);
            setAw(pointerToAw(event));
          });
          hit.addEventListener("pointermove", (event) => {
            if (dragging) setAw(pointerToAw(event));
          });
          hit.addEventListener("pointerup", (event) => {
            dragging = false;
            try { hit.releasePointerCapture(event.pointerId); } catch (error) {}
          });
          hit.addEventListener("pointercancel", () => { dragging = false; });
          hit.addEventListener("keydown", (event) => {
            const step = 5;
            const current = Number(model.get("aw"));
            if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
              event.preventDefault();
              setAw(current - step);
            } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
              event.preventDefault();
              setAw(current + step);
            } else if (event.key === "Home") {
              event.preventDefault();
              setAw(0);
            } else if (event.key === "End") {
              event.preventDefault();
              setAw(50);
            }
          });
          toggle.addEventListener("click", () => {
            model.set("detach", !model.get("detach"));
            model.save_changes();
            update();
          });
          model.on("change:aw", update);
          model.on("change:detach", update);
          update();
          return () => {
            model.off("change:aw", update);
            model.off("change:detach", update);
          };
        }
        export default { render };
        """
        aw = traitlets.Float(5.0).tag(sync=True)
        detach = traitlets.Bool(True).tag(sync=True)
        frames = traitlets.Dict(default_value={}).tag(sync=True)
        labels = traitlets.List(default_value=[]).tag(sync=True)
        max_a = traitlets.Float(1.0).tag(sync=True)
        max_w = traitlets.Float(1.0).tag(sync=True)
        world = traitlets.Float(5.5).tag(sync=True)

    dial = mo.ui.anywidget(
        CompressionDial(
            frames=DIAL["frames"],
            labels=DIAL["labels"],
            max_a=DIAL["max_a"],
            max_w=DIAL["max_w"],
            world=DIAL["world"],
            aw=5.0,
            detach=True,
        )
    )
    dial
    return (dial,)


@app.cell(hide_code=True)
def _(DIAL, dial, mo):
    _out = mo.md("_The dial is available in the interactive notebook._")
    if dial is not None:
        _aw = min(DIAL["aws"], key=lambda _a: abs(_a - dial.aw))
        _detached = next(
            f for f in DIAL["frames"]["detach"] if f["aw"] == _aw
        )
        _plain = next(
            f for f in DIAL["frames"]["no_detach"] if f["aw"] == _aw
        )
        _out = mo.md(f"""
        At AW = **{_aw:.0f}**, the detached run has hidden norm
        **{_detached["a_norm"]:.2f}** and classifier norm
        **{_detached["w_norm"]:.2f}**. Without the stop-gradient:
        **{_plain["a_norm"]:.2f}** and **{_plain["w_norm"]:.2f}**.
        The self-model error is **{_detached["sm_error"]:.3f}** in both
        modes. Both solve the prediction task; only the detached run
        keeps the representation at its natural scale.
        """)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The stop-gradient

    The self-model target is `a.detach()`. It looks like an
    implementation detail. It is the hinge.

    Before you look: which dataset is sensitive to dropping it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    guess = mo.ui.radio(
        options={
            "MNIST": "mnist",
            "CIFAR-10": "cifar",
            "Both": "both",
            "Neither": "neither",
        },
        label="Which dataset is sensitive to the stop-gradient?",
    )
    reveal = mo.ui.checkbox(label="Reveal the ablation")
    mo.vstack([guess, reveal])
    return guess, reveal


@app.cell(hide_code=True)
def _(json, mo, pd):
    _ablation = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "mnist_detach_ablation.json"
        ).read_text()
    )
    _probes = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "cifar_fig3_probes.json"
        ).read_text()
    )
    _rows = []
    for _key, _label in (
        ("detach", "detached target"),
        ("no_detach", "no detach"),
    ):
        for _aw, _value in _ablation[_key].items():
            _rows.append(
                {
                    "mode": _label,
                    "aw": float(_aw),
                    "mean": float(_value["sd_mean"]),
                    "half": float(_value["sd_ci"]),
                    "acc": float(_value["acc_mean"]),
                }
            )
    MNIST_DETACH = pd.DataFrame(_rows)
    MNIST_DETACH["lo"] = MNIST_DETACH["mean"] - MNIST_DETACH["half"]
    MNIST_DETACH["hi"] = MNIST_DETACH["mean"] + MNIST_DETACH["half"]

    _probe_rows = []
    for _variant, _aws in _probes["variants"].items():
        for _aw, _probe in _aws.items():
            _probe_rows.append(
                {
                    "variant": _variant,
                    "aw": float(_aw),
                    "sd_e50": float(_probe["sd_e50"]),
                    "hidden": float(_probe["hidden_absmean"][1]),
                    "wnorm": float(_probe["cls_wnorm"][1]),
                }
            )
    CIFAR_PROBES = pd.DataFrame(_probe_rows)
    return CIFAR_PROBES, MNIST_DETACH


@app.cell(hide_code=True)
def _(
    CIFAR_PROBES,
    MNIST_DETACH,
    alt,
    guess,
    mo,
    reveal,
):
    _df = MNIST_DETACH.copy()
    _order = [f"AW={aw:g}" for aw in (0, 1, 5, 10, 20, 50)]
    _df["aw_label"] = _df["aw"].map(lambda _aw: f"AW={_aw:g}")
    _scale = alt.Scale(
        domain=["detached target", "no detach"],
        range=["#111827", "#b45309"],
    )
    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X(
                "aw_label:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "lo:Q",
                title="Classifier weight SD (epoch 50)",
                scale=alt.Scale(zero=False),
            ),
            y2="hi:Q",
            color=alt.Color("mode:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("aw_label:N", sort=_order),
            y=alt.Y("mean:Q"),
            color=alt.Color(
                "mode:N",
                scale=_scale,
                legend=alt.Legend(title=None, orient="bottom"),
            ),
        )
    )
    _chart = (_band + _line).properties(width="container", height=330)

    def _probe(_variant, _aw):
        _row = CIFAR_PROBES[
            (CIFAR_PROBES["variant"] == _variant)
            & (CIFAR_PROBES["aw"] == _aw)
        ].iloc[0]
        return float(_row["sd_e50"]), float(_row["hidden"]), float(_row["wnorm"])

    _b0, _, _ = _probe("base", 0.0)
    _b2, _b2h, _b2w = _probe("base", 2.0)
    _n0, _, _ = _probe("nodetach", 0.0)
    _n2, _n2h, _n2w = _probe("nodetach", 2.0)
    _cifar = "\n".join(
        [
            (
                "| CIFAR variant | SD (AW 0) | SD (AW 2) "
                "| hidden \\|a\\| (AW 2) | classifier \\|W\\| (AW 2) |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
            f"| base | {_b0:.4f} | {_b2:.4f} | {_b2h:.3f} | {_b2w:.2f} |",
            f"| no detach | {_n0:.4f} | {_n2:.4f} | {_n2h:.3f} | {_n2w:.2f} |",
        ]
    )

    if guess.value == "mnist":
        _lead = "Right. MNIST is the sensitive one."
        _kind = "success"
    elif guess.value is None:
        _lead = "Pick an answer, then reveal."
        _kind = "neutral"
    else:
        _lead = "Not quite. MNIST is the sensitive one."
        _kind = "warn"

    _summary = mo.callout(mo.md(_lead), kind=_kind)
    _prose = mo.md(f"""
    With the stop-gradient, the MNIST SD falls from **0.0974** at AW 0
    to **0.0812** at AW 50. Without it, the SD rises to **0.1156** at
    AW 5 and stays above baseline through AW 20. The paper's main effect
    reverses for most of the range, and accuracy at AW 50 falls to
    **91.5%** instead of **95.6%**.

    The mechanism is the scale cheat from the dial. Without the
    stop-gradient, the target moves. The network can reduce the
    self-model loss by shrinking the activations, and the classifier
    grows to keep the logits fixed.

    CIFAR is the counterexample. At AW 2, dropping the stop-gradient
    changes the SD from **{_b2:.4f}** to **{_n2:.4f}** and the hidden
    norm from **{_b2h:.3f}** to **{_n2h:.3f}**, but the ordering
    survives. The stop-gradient is essential to the MNIST mechanism and
    irrelevant to the CIFAR inversion.

    The CIFAR probe is two seeds and 50 epochs. Treat it as a sign.
    """)
    if reveal.value:
        _out = mo.vstack([_summary, _chart, mo.md(_cifar), _prose])
    else:
        _out = mo.md("_Reveal to see the ablation._")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The pressure has to persist

    On IMDB the MNIST effect was missing. The identity diagnostics show
    why: the self-model head learns a shortcut early, and the self-model
    share of the hidden-layer gradient falls to about **2%** by epoch 20.
    On MNIST the share stays at **14% to 18%** at epoch 50.

    If that is the mechanism, slowing the head down should bring the
    effect back. We scaled the self-model rows' gradient by 0.01 and
    0.001 and re-ran IMDB at AW 0, 100, and 500.
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, pd):
    _runs = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "slowhead_control.json"
        ).read_text()
    )["runs"]
    _labels = {
        "scale1": "×1 (normal)",
        "scale0.01": "×0.01",
        "scale0.001": "×0.001",
    }
    _rows = []
    for _scale, _label in _labels.items():
        for _aw in ("0", "100", "500"):
            _entry = _runs[_scale]["results"][_aw]
            _rows.append(
                {
                    "scale": _label,
                    "aw": f"AW={_aw}",
                    "mean": float(_entry["sd_mean"][100]),
                    "half": float(_entry["sd_ci"][100]),
                    "acc": float(_entry["acc_mean"]),
                }
            )
    SLOWHEAD = pd.DataFrame(_rows)
    SLOWHEAD["lo"] = SLOWHEAD["mean"] - SLOWHEAD["half"]
    SLOWHEAD["hi"] = SLOWHEAD["mean"] + SLOWHEAD["half"]
    return (SLOWHEAD,)


@app.cell(hide_code=True)
def _(SLOWHEAD, alt, mo):
    _order = ["AW=0", "AW=100", "AW=500"]
    _scale = alt.Scale(
        domain=["×1 (normal)", "×0.01", "×0.001"],
        range=["#111827", "#b45309", "#a8a29e"],
    )
    _band = (
        alt.Chart(SLOWHEAD)
        .mark_area(opacity=0.12)
        .encode(
            x=alt.X(
                "aw:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "lo:Q",
                title="Classifier weight SD (epoch 100)",
                scale=alt.Scale(zero=False),
            ),
            y2="hi:Q",
            color=alt.Color("scale:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(SLOWHEAD)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("aw:N", sort=_order),
            y=alt.Y("mean:Q"),
            color=alt.Color(
                "scale:N",
                scale=_scale,
                legend=alt.Legend(title="self-model LR", orient="bottom"),
            ),
        )
    )
    _chart = (_band + _line).properties(width="container", height=300)
    _caption = mo.md(
        "At the normal rate the three AW curves sit on top of each "
        "other. At ×0.01 the pressure survives and AW 100/500 separate "
        "from the baseline. At ×0.001 the self-model term swamps the "
        "classifier: accuracy falls to **54% to 56%**, so those points "
        "are not interpretable."
    )
    mo.vstack([_chart, _caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The slowed-head control is causal evidence for the identity
    mechanism, with caveats: three seeds, 100 epochs, accuracy drops,
    and the effect is not monotone in AW. It explains our missing IMDB
    effect. It does not reproduce the paper's IMDB numbers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The scoreboard

    The guided cut keeps the shape of the result. The full per-panel
    notes, the probes, and the RLCT details are in the lab notebook.
    """)
    return


@app.cell(hide_code=True)
def _(SCORE, mo):
    _areas = ["MNIST", "CIFAR", "IMDB"]
    _rows = []
    for _area in _areas:
        _cells = []
        for _row in SCORE:
            if _row["area"] != _area:
                continue
            _cells.append(
                f'<div class="matrix-cell {_row["status"]}">'
                f'{_row["panel"]} · {_row["status"]}</div>'
            )
        _rows.append(
            f'<div class="matrix-row"><div class="matrix-label">{_area}</div>'
            f'<div class="matrix-cells">{"".join(_cells)}</div></div>'
        )
    mo.Html(f'<div class="matrix">{"".join(_rows)}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What next

    Three experiments would change the conclusion:

    - **The CIFAR architecture gap.** The paper describes skip
      connections from block 3 into the hidden layer. We could not map
      that onto a standard ResNet-18, so our hidden layer sits after
      global pooling. A faithful implementation is the next experiment.
    - **More seeds for the slow-head control.** Three seeds and 100
      epochs is a sign. Ten seeds and 250 epochs is a measurement.
    - **A rank diagnostic for IMDB.** The hidden layer is 128 units from
      a 64-dimensional embedding. A rank test would show whether the
      self-model rows learn a projection, as the weight distance
      suggests.

    The effect is real, fragile, and conditional. That is a more useful
    result than a clean reproduction.
    """)
    return


@app.cell(hide_code=True)
def _(META, mo):
    mo.Html(
        f"""
        <footer class="essay-provenance">
          <div>
            <strong>Paper.</strong>
            <a href="{META["paper_url"]}">{META["paper"]}</a>
            by {META["authors"]}.
          </div>
          <div>
            <strong>Evidence.</strong>
            <a href="https://github.com/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_essay.py">
            The lab notebook</a> has the full reproduction, the probes,
            and the RLCT panels.
          </div>
          <div>
            <strong>Data and code.</strong>
            <a href="{META["repo"]}">Repository</a>.
            Every figure reads a saved result from <code>public/data/</code>.
          </div>
          <div>
            <strong>How this was made.</strong>
            Built with marimo. The guided cut and the lab notebook share
            the same data contract. The compression dial reads
            precomputed toy frames.
          </div>
        </footer>
        """
    )
    return


if __name__ == "__main__":
    app.run()
