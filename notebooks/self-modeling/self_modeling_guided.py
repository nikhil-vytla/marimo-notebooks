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
        .essay-note a { color: var(--blue); }
        .essay-note {
          border-left: 3px solid var(--accent);
          background: var(--accent-soft);
          padding: 0.9rem 1.1rem;
          border-radius: 0 0.5rem 0.5rem 0;
          margin: 1.2rem 0;
          font-size: 0.94rem;
          line-height: 1.5;
        }
        .pullquote {
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-size: 1.45rem;
          line-height: 1.28;
          color: #292524;
          border-left: 4px solid var(--accent);
          padding: 0.2rem 0 0.2rem 1rem;
          margin: 1.7rem 0 1.5rem;
        }
        .markdown h2 {
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-weight: 600;
          letter-spacing: -0.01em;
          margin-top: 2.3rem;
        }
        .markdown h3 {
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-weight: 600;
          letter-spacing: -0.01em;
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
        .mechanism-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
          gap: 0.8rem;
          margin: 1rem 0 0.6rem;
        }
        @media (max-width: 640px) {
          .mechanism-grid { grid-template-columns: 1fr; }
        }
        .mechanism-card {
          border: 1px solid var(--line);
          border-radius: 0.8rem;
          background: #fff;
          padding: 0.9rem 0.95rem;
        }
        .mechanism-kicker {
          font: 650 0.7rem/1.2 ui-monospace, SFMono-Regular, monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .mechanism-svg {
          width: 100%;
          height: auto;
          display: block;
          margin: 0.5rem 0 0.2rem;
        }
        .mechanism-text {
          font-size: 0.86rem;
          line-height: 1.5;
          color: #57534e;
        }
        .mechanism-text strong { color: var(--ink); }
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
        "reading_time": "17 min",
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
          <a href="#the-pressure-has-to-persist">The pressure</a>
          <a href="#the-skip-test">The skip test</a>
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
    <p class="figure-caption">Figure 1. Three panels, three verdicts.
    MNIST and CIFAR numbers are the epoch-50 classifier weight SD; IMDB
    is the weight SD at the end of training. Sources:
    <code>mnist_detach_ablation.json</code>,
    <code>cifar_fig3_probes.json</code>, <code>imdb_fig4.json</code>.</p>
    """)
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
              .compression-dial .cd-actions {
                display: flex;
                align-items: center;
                gap: 8px;
                flex: 0 0 auto;
              }
              .compression-dial .cd-play {
                border: 1px solid var(--cd-line);
                background: #fff;
                border-radius: 999px;
                padding: 6px 11px;
                font: 600 12px/1 ui-sans-serif, system-ui, sans-serif;
                cursor: pointer;
                white-space: nowrap;
              }
              .compression-dial .cd-play:hover {
                border-color: #d6d3d1;
                background: #fafaf9;
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
                  <div class="cd-sub">Drag the dial or the norm strip to add self-model pressure. The classifier has to keep the logits fixed.</div>
                </div>
                <div class="cd-actions">
                  <button class="cd-play" type="button" aria-pressed="false">Play</button>
                  <button class="cd-toggle on" type="button" aria-pressed="true">detached target</button>
                </div>
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

          // Linked view: the same AW state as a pair of norm traces.
          // The dial scrubs it; the strip scrubs the dial.
          const chartX = 372, chartW = 208, chartTop = 264;
          svg.appendChild(mk("rect", {
            x: 364, y: 240, width: 224, height: 174, rx: 10,
            fill: "#fff", stroke: "#e7e5e4",
          }));
          const chartTitle = mk("text", {
            x: chartX, y: 257, "font-size": 10, "font-weight": 600,
            fill: "#57534e",
          });
          chartTitle.textContent = "norms across AW · click to scrub";
          svg.appendChild(chartTitle);

          function sparkline(top, label, color) {
            const labelNode = mk("text", {
              x: chartX, y: top + 9, "font-size": 10, fill: "#57534e",
              "font-weight": 600,
            });
            labelNode.textContent = label;
            svg.appendChild(labelNode);
            const valueNode = mk("text", {
              x: chartX + chartW, y: top + 9, "font-size": 10,
              fill: "#57534e", "text-anchor": "end",
              "font-variant-numeric": "tabular-nums",
            });
            svg.appendChild(valueNode);
            svg.appendChild(mk("line", {
              x1: chartX, y1: top + 46, x2: chartX + chartW, y2: top + 46,
              stroke: "#f5f5f4",
            }));
            const path = mk("path", {
              fill: "none", stroke: color, "stroke-width": 2,
              "stroke-linecap": "round", "stroke-linejoin": "round",
            });
            svg.appendChild(path);
            const dot = mk("circle", { r: 3.2, fill: color });
            svg.appendChild(dot);
            return { path, dot, valueNode, top };
          }

          const hiddenSpark = sparkline(270, "hidden |a|", "#b45309");
          const classSpark = sparkline(330, "classifier |W|", "#111827");
          const guide = mk("line", {
            y1: 266, y2: 390, stroke: "#a8a29e", "stroke-dasharray": "3 3",
          });
          svg.appendChild(guide);
          for (const [tick, anchor] of [[0, "start"], [25, "middle"], [50, "end"]]) {
            const label = mk("text", {
              x: chartX + (tick / 50) * chartW, y: 404,
              "text-anchor": anchor, "font-size": 9, fill: "#a8a29e",
            });
            label.textContent = String(tick);
            svg.appendChild(label);
          }
          const chartHit = mk("rect", {
            x: chartX, y: 264, width: chartW, height: 128,
            fill: "transparent", style: "cursor: ew-resize; touch-action: none;",
            "aria-label": "Scrub AW on the norm chart",
          });
          svg.appendChild(chartHit);

          function normPath(frames, key, maxValue, top) {
            return frames.map((frame, index) => {
              const x = chartX + (frame.aw / 50) * chartW;
              const y = top + 42 - Math.min(1, frame[key] / maxValue) * 28;
              return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
            }).join(" ");
          }

          function updateChart() {
            const aw = Number(model.get("aw"));
            const mode = model.get("detach") ? "detach" : "no_detach";
            const frames = (model.get("frames") || {})[mode] || [];
            const pair = frameAt(aw);
            if (!frames.length || !pair) return;
            const aValue = lerp(pair.a.a_norm, pair.b.a_norm, pair.t);
            const wValue = lerp(pair.a.w_norm, pair.b.w_norm, pair.t);
            hiddenSpark.path.setAttribute("d", normPath(frames, "a_norm", maxA, hiddenSpark.top));
            classSpark.path.setAttribute("d", normPath(frames, "w_norm", maxW, classSpark.top));
            const x = chartX + (aw / 50) * chartW;
            const ya = hiddenSpark.top + 42 - Math.min(1, aValue / maxA) * 28;
            const yw = classSpark.top + 42 - Math.min(1, wValue / maxW) * 28;
            hiddenSpark.dot.setAttribute("cx", x);
            hiddenSpark.dot.setAttribute("cy", ya);
            classSpark.dot.setAttribute("cx", x);
            classSpark.dot.setAttribute("cy", yw);
            hiddenSpark.valueNode.textContent = aValue.toFixed(2);
            classSpark.valueNode.textContent = wValue.toFixed(2);
            guide.setAttribute("x1", x);
            guide.setAttribute("x2", x);
          }

          let chartDragging = false;
          function chartPointer(event) {
            const rect = svg.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * W;
            setAw(((x - chartX) / chartW) * 50);
          }
          chartHit.addEventListener("pointerdown", (event) => {
            stopPlayback();
            chartDragging = true;
            chartHit.setPointerCapture(event.pointerId);
            chartPointer(event);
          });
          chartHit.addEventListener("pointermove", (event) => {
            if (chartDragging) chartPointer(event);
          });
          chartHit.addEventListener("pointerup", (event) => {
            chartDragging = false;
            try { chartHit.releasePointerCapture(event.pointerId); } catch (error) {}
          });
          chartHit.addEventListener("pointercancel", () => { chartDragging = false; });

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
          const playButton = el.querySelector(".cd-play");
          let dragging = false;
          let playing = false;
          let animId = null;

          function stopPlayback() {
            playing = false;
            if (animId !== null) cancelAnimationFrame(animId);
            animId = null;
            playButton.textContent = "Play";
            playButton.setAttribute("aria-pressed", "false");
          }

          function startPlayback() {
            if (playing) {
              stopPlayback();
              return;
            }
            if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
              setAw(50);
              return;
            }
            const from = Number(model.get("aw"));
            const startAw = from >= 49 ? 0 : from;
            const started = performance.now();
            playing = true;
            playButton.textContent = "Pause";
            playButton.setAttribute("aria-pressed", "true");
            function tick(now) {
              const t = Math.min(1, (now - started) / 2200);
              const eased = t < 0.5
                ? 2 * t * t
                : 1 - Math.pow(-2 * t + 2, 2) / 2;
              setAw(startAw + (50 - startAw) * eased);
              if (t < 1 && playing) {
                animId = requestAnimationFrame(tick);
              } else {
                stopPlayback();
              }
            }
            animId = requestAnimationFrame(tick);
          }

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
            updateChart();
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
            stopPlayback();
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
            stopPlayback();
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
            stopPlayback();
            model.set("detach", !model.get("detach"));
            model.save_changes();
            update();
          });
          playButton.addEventListener("click", startPlayback);
          model.on("change:aw", update);
          model.on("change:detach", update);
          update();
          return () => {
            stopPlayback();
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
    <p class="figure-caption">Figure 2. The compression dial. The hidden
    cloud is a 2D toy trained for each AW; the dial interpolates 11
    precomputed frames. Both modes reach self-model error 0.000. Drag
    the dial, click the norm strip, press Play, or toggle the
    stop-gradient. Source:
    <code>compression_dial.json</code>; generator:
    <code>scripts/build_compression_dial.py</code>.</p>
    """)
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
    pd,
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
    _paper = pd.DataFrame(
        {
            "aw_label": _order,
            "paper": [0.097, 0.094, 0.091, 0.088, 0.085, 0.081],
        }
    )
    _paper_points = (
        alt.Chart(_paper)
        .mark_point(
            shape="diamond", size=80, filled=True, stroke="white",
            strokeWidth=1,
        )
        .encode(
            x=alt.X("aw_label:N", sort=_order),
            y=alt.Y(
                "paper:Q",
                scale=alt.Scale(domain=[0.078, 0.120], zero=False),
            ),
            tooltip=[
                alt.Tooltip("aw_label:N", title="AW"),
                alt.Tooltip("paper:Q", format=".3f", title="paper"),
            ],
        )
    )
    _reversal = (
        alt.Chart(pd.DataFrame({"aw_label": ["AW=1", "AW=20"]}))
        .mark_rule(strokeDash=[3, 3], color="#a8a29e", strokeWidth=1)
        .encode(x=alt.X("aw_label:N", sort=_order))
    )
    _labels = (
        alt.Chart(
            pd.DataFrame(
                {
                    "aw_label": ["AW=5", "AW=5"],
                    "y": [0.0882, 0.1180],
                    "mode": ["detached target", "no detach"],
                    "text": ["detached target", "no detach"],
                }
            )
        )
        .mark_text(align="left", dx=7, fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("aw_label:N", sort=_order),
            y=alt.Y("y:Q"),
            text="text:N",
            color=alt.Color("mode:N", scale=_scale, legend=None),
        )
    )
    _line = _line.encode(
        tooltip=[
            alt.Tooltip("mode:N", title="mode"),
            alt.Tooltip("aw_label:N", title="AW"),
            alt.Tooltip("mean:Q", format=".4f", title="SD"),
            alt.Tooltip("half:Q", format=".4f", title="95% CI"),
            alt.Tooltip("acc:Q", format=".3f", title="accuracy"),
        ]
    )
    _chart = (
        _band
        + _reversal
        + _paper_points
        + _line
        + _labels
    ).properties(width="container", height=360)

    def _probe(_variant, _aw):
        _row = CIFAR_PROBES[
            (CIFAR_PROBES["variant"] == _variant)
            & (CIFAR_PROBES["aw"] == _aw)
        ].iloc[0]
        return float(_row["sd_e50"]), float(_row["hidden"]), float(_row["wnorm"])

    _b0, _, _b0w = _probe("base", 0.0)
    _b2, _b2h, _b2w = _probe("base", 2.0)
    _n0, _, _ = _probe("nodetach", 0.0)
    _n2, _n2h, _n2w = _probe("nodetach", 2.0)
    _ratio0 = _b0 / _b0w
    _ratio2 = _b2 / _b2w
    _sd_pct = (_n2 - _b2) / _b2 * 100.0
    _hn_pct = (_n2h - _b2h) / _b2h * 100.0
    _wn_pct = (_n2w - _b2w) / _b2w * 100.0
    _cifar_df = pd.DataFrame(
        [
            {
                "metric": "SD",
                "change": _sd_pct,
                "base": _b2,
                "nodetach": _n2,
            },
            {
                "metric": "hidden |a|",
                "change": _hn_pct,
                "base": _b2h,
                "nodetach": _n2h,
            },
            {
                "metric": "classifier |W|",
                "change": _wn_pct,
                "base": _b2w,
                "nodetach": _n2w,
            },
        ]
    )
    _cifar_bars = (
        alt.Chart(_cifar_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "metric:N",
                sort=["SD", "hidden |a|", "classifier |W|"],
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "change:Q",
                title="change vs base at AW 2 (%)",
                scale=alt.Scale(domain=[-22, 4]),
            ),
            color=alt.condition(
                alt.datum.change < 0,
                alt.value("#b45309"),
                alt.value("#111827"),
            ),
            tooltip=[
                alt.Tooltip("metric:N"),
                alt.Tooltip("change:Q", format="+.1f", title="change (%)"),
                alt.Tooltip("base:Q", format=".4f", title="base"),
                alt.Tooltip("nodetach:Q", format=".4f", title="no detach"),
            ],
        )
        .properties(width="container", height=220)
    )
    _cifar_zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#d6d3d1")
        .encode(y="y:Q")
    )
    _cifar_text = (
        alt.Chart(_cifar_df)
        .mark_text(dy=12, fontSize=11, color="#57534e")
        .encode(
            x=alt.X("metric:N", sort=["SD", "hidden |a|", "classifier |W|"]),
            y=alt.Y("change:Q"),
            text=alt.Text("change:Q", format="+.1f"),
        )
    )
    _cifar_chart = _cifar_zero + _cifar_bars + _cifar_text
    _cifar_caption = mo.md(
        f'<p class="figure-caption">Figure 4. CIFAR-10 at AW 2. '
        f'Dropping the stop-gradient moves the SD by {_sd_pct:+.1f}%, the '
        f'hidden norm by {_hn_pct:+.1f}%, and the classifier norm by '
        f'{_wn_pct:+.1f}%. The ordering survives: {_n2:.4f} vs '
        f'{_b2:.4f}, both above {_b0:.4f} at AW 0. Two seeds, 50 epochs; '
        f'<code>cifar_fig3_probes.json</code>.</p>'
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
    _fig3_caption = mo.md(
        '<p class="figure-caption">Figure 3. MNIST at hidden 512. '
        'Lines are 10 seeds; bands are 95% CIs. Diamonds are the '
        'paper\'s epoch-50 values, read off Figure 2A. The dashed '
        'guides mark AW 1 and 20, where the no-detach ordering reverses. '
        'Source: <code>mnist_detach_ablation.json</code>.</p>'
    )
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

    The SD divided by the classifier norm is flat across AW:
    **{_ratio0:.5f}** at AW 0 and **{_ratio2:.5f}** at AW 2. The
    widening tracks the classifier's scale, not a new kind of
    complexity.

    The CIFAR probe is two seeds and 50 epochs. Treat it as a sign.
    """)
    if reveal.value:
        _out = mo.vstack(
            [
                _summary,
                _chart,
                _fig3_caption,
                _cifar_chart,
                _cifar_caption,
                _prose,
            ]
        )
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
def _(SLOWHEAD, alt, mo, pd):
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
            tooltip=[
                alt.Tooltip("scale:N", title="scale"),
                alt.Tooltip("aw:N", title="AW"),
                alt.Tooltip("mean:Q", format=".4f", title="SD"),
                alt.Tooltip("half:Q", format=".4f", title="95% CI"),
                alt.Tooltip("acc:Q", format=".3f", title="accuracy"),
            ],
        )
    )
    _sd = (_band + _line).properties(width="container", height=220)

    _acc = (
        alt.Chart(SLOWHEAD)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "aw:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "acc:Q",
                title="Test accuracy",
                scale=alt.Scale(domain=[0.5, 0.85]),
            ),
            color=alt.Color("scale:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("scale:N", title="scale"),
                alt.Tooltip("aw:N", title="AW"),
                alt.Tooltip("acc:Q", format=".3f", title="accuracy"),
            ],
        )
    )
    _chance = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(strokeDash=[4, 4], color="#a8a29e")
        .encode(y="y:Q")
    )
    _chance_text = (
        alt.Chart(pd.DataFrame({"aw": ["AW=0"], "y": [0.515], "text": ["chance"]}))
        .mark_text(align="left", dx=4, fontSize=10, color="#a8a29e")
        .encode(
            x=alt.X("aw:N", sort=_order),
            y=alt.Y("y:Q"),
            text="text:N",
        )
    )
    _acc = (_chance + _acc + _chance_text).properties(
        width="container", height=170
    )
    _caption = mo.md(
        '<p class="figure-caption">Figure 5. IMDB at 3 seeds and 100 '
        'epochs. The top panel is the classifier weight SD; the bottom '
        'panel is accuracy, with the 50% chance line. At ×1 the three '
        'AW curves coincide. At ×0.01 the pressure survives and AW '
        '100/500 separate from baseline. At ×0.001 accuracy is near '
        'chance, so the SD is not interpretable. Source: '
        '<code>slowhead_control.json</code>.</p>'
    )
    mo.vstack([_sd, _acc, _caption])
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
    Two failures, two mechanisms. Build the figure one piece at a time,
    or show both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    MECH_STEP = mo.ui.radio(
        options=["1. units", "2. timing", "3. both"],
        value="3. both",
        label="Build the synthesis",
        inline=True,
    )
    return (MECH_STEP,)


@app.cell(hide_code=True)
def _(MECH_STEP, mo):
    _units = """<div class="mechanism-card">
            <div class="mechanism-kicker">CIFAR-10 · units</div>
            <svg class="mechanism-svg" viewBox="0 0 280 150"
                 role="img"
                 aria-label="Hidden activations shrink while the classifier grows">
              <defs>
                <marker id="mech-arrow" viewBox="0 0 10 10" refX="9"
                        refY="5" markerWidth="5" markerHeight="5"
                        orient="auto">
                  <path d="M0,0 L10,5 L0,10 z" fill="#a8a29e"/>
                </marker>
              </defs>
              <text x="44" y="16" text-anchor="middle" font-size="10"
                    fill="#78716c">AW 0</text>
              <text x="200" y="16" text-anchor="middle" font-size="10"
                    fill="#78716c">AW 2</text>
              <g opacity="0.85">
                <circle cx="30" cy="50" r="3.2" fill="#1d4ed8"/>
                <circle cx="42" cy="46" r="3.2" fill="#1d4ed8"/>
                <circle cx="52" cy="56" r="3.2" fill="#1d4ed8"/>
                <circle cx="34" cy="70" r="3.2" fill="#b45309"/>
                <circle cx="46" cy="68" r="3.2" fill="#b45309"/>
                <circle cx="58" cy="74" r="3.2" fill="#b45309"/>
                <circle cx="32" cy="90" r="3.2" fill="#047857"/>
                <circle cx="44" cy="92" r="3.2" fill="#047857"/>
                <circle cx="56" cy="88" r="3.2" fill="#047857"/>
                <circle cx="192" cy="64" r="3.2" fill="#1d4ed8"/>
                <circle cx="199" cy="62" r="3.2" fill="#1d4ed8"/>
                <circle cx="206" cy="68" r="3.2" fill="#1d4ed8"/>
                <circle cx="194" cy="76" r="3.2" fill="#b45309"/>
                <circle cx="201" cy="74" r="3.2" fill="#b45309"/>
                <circle cx="208" cy="80" r="3.2" fill="#b45309"/>
                <circle cx="192" cy="88" r="3.2" fill="#047857"/>
                <circle cx="199" cy="90" r="3.2" fill="#047857"/>
                <circle cx="206" cy="86" r="3.2" fill="#047857"/>
              </g>
              <line x1="95" y1="80" x2="135" y2="80" stroke="#a8a29e"
                    stroke-width="1.5" marker-end="url(#mech-arrow)"/>
              <text x="115" y="70" text-anchor="middle" font-size="9"
                    fill="#78716c">AW ↑</text>
              <rect x="30" y="118" width="40" height="8" rx="3"
                    fill="#b45309" opacity="0.8"/>
              <rect x="176" y="118" width="60" height="8" rx="3"
                    fill="#b45309" opacity="0.8"/>
              <text x="50" y="140" text-anchor="middle" font-size="9"
                    fill="#78716c">|W| small</text>
              <text x="206" y="140" text-anchor="middle" font-size="9"
                    fill="#78716c">|W| grows</text>
            </svg>
            <p class="mechanism-text"><strong>SD / |W| stays flat.</strong>
            The activations shrink and the classifier grows to keep the
            logits fixed. The wider SD is a units change, not a new
            complexity.</p>
          </div>"""
    _timing = """<div class="mechanism-card">
            <div class="mechanism-kicker">IMDB · timing</div>
            <svg class="mechanism-svg" viewBox="0 0 280 150"
                 role="img"
                 aria-label="Self-model gradient share falls to two percent by epoch twenty">
              <line x1="28" y1="120" x2="260" y2="120" stroke="#d6d3d1"/>
              <line x1="28" y1="120" x2="28" y2="30" stroke="#d6d3d1"/>
              <line x1="28" y1="111" x2="260" y2="111" stroke="#d6d3d1"
                    stroke-dasharray="4 4"/>
              <text x="250" y="107" text-anchor="end" font-size="9"
                    fill="#a8a29e">2%</text>
              <polyline points="28,39 49,52 70,84 91,102 112,111 154,111 196,111 238,111"
                        fill="none" stroke="#b45309" stroke-width="3"
                        stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="112" cy="111" r="3.5" fill="#b45309"/>
              <text x="120" y="96" font-size="9" fill="#78716c">pressure dies</text>
              <text x="28" y="24" font-size="9" fill="#78716c">self-model share</text>
              <text x="28" y="136" font-size="9" fill="#a8a29e">0</text>
              <text x="112" y="136" text-anchor="middle" font-size="9"
                    fill="#a8a29e">20</text>
              <text x="196" y="136" text-anchor="middle" font-size="9"
                    fill="#a8a29e">40</text>
              <text x="260" y="136" text-anchor="end" font-size="9"
                    fill="#a8a29e">epoch</text>
            </svg>
            <p class="mechanism-text"><strong>The pressure dies.</strong>
            The self-model share of the hidden-layer gradient falls to
            about 2% by epoch 20. Slow the head down and the effect comes
            back.</p>
          </div>"""
    _cards = []
    if MECH_STEP.value in ("1. units", "3. both"):
        _cards.append(_units)
    if MECH_STEP.value in ("2. timing", "3. both"):
        _cards.append(_timing)
    mo.vstack(
        [
            MECH_STEP,
            mo.Html(
                f'<div class="mechanism-grid">{"".join(_cards)}</div>'
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <p class="figure-caption">Figure 6. Two failures, two mechanisms.
    CIFAR's inversion is a units change: the classifier norm grows with
    the SD, so SD/|W| is flat. IMDB's null is a timing problem: the
    self-model gradient share falls to about 2% by epoch 20. Sources:
    <code>cifar_fig3_probes.json</code>,
    <code>identity_diagnostics.json</code>,
    <code>slowhead_control.json</code>.</p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The skip test

    The synthesis leaves an obvious objection. The paper's hidden input
    includes a skip from block 3, and our first CIFAR run did not. That
    is the largest architectural difference between the two.

    We implemented the simplest faithful version: global-pool block 3
    and block 4, concatenate the 256- and 512-vectors, and feed the
    768-vector into the 2000-wide hidden layer. The self-model target is
    the concatenation of that input and the hidden activations.
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, pd):
    _skip = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "cifar_skip.json"
        ).read_text()
    )
    _labels = {
        "none": "standard ResNet-18",
        "block3": "block-3 skip",
        "spatial": "spatial skip",
    }
    _rows = []
    for _value in _skip["results"].values():
        _rows.append(
            {
                "mode": _labels[_value["skip"]],
                "aw": f"AW={_value['aw']:g}",
                "mean": float(_value["sd_mean"]),
                "half": float(_value["sd_ci"]),
                "acc": float(_value["acc_mean"]),
            }
        )
    CIFAR_SKIP = pd.DataFrame(_rows)
    CIFAR_SKIP["lo"] = CIFAR_SKIP["mean"] - CIFAR_SKIP["half"]
    CIFAR_SKIP["hi"] = CIFAR_SKIP["mean"] + CIFAR_SKIP["half"]

    _b64_rows = []
    for _value in _skip.get("batch64", {}).get("results", {}).values():
        _b64_rows.append(
            {
                "mode": _labels[_value["skip"]],
                "aw": f"AW={_value['aw']:g}",
                "mean": float(_value["sd_mean"]),
                "acc": float(_value["acc_mean"]),
            }
        )
    CIFAR_SKIP_B64 = pd.DataFrame(_b64_rows)
    return (CIFAR_SKIP, CIFAR_SKIP_B64)


@app.cell(hide_code=True)
def _(CIFAR_SKIP, CIFAR_SKIP_B64, alt, mo):
    _order = ["AW=0", "AW=2"]
    _scale = alt.Scale(
        domain=["standard ResNet-18", "block-3 skip"],
        range=["#111827", "#b45309"],
    )
    _band = (
        alt.Chart(CIFAR_SKIP)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X(
                "aw:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "lo:Q",
                title="Classifier weight SD (epoch 150)",
                scale=alt.Scale(zero=False),
            ),
            y2="hi:Q",
            color=alt.Color("mode:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(CIFAR_SKIP)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("aw:N", sort=_order),
            y=alt.Y("mean:Q"),
            color=alt.Color(
                "mode:N",
                scale=_scale,
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("mode:N", title="model"),
                alt.Tooltip("aw:N", title="AW"),
                alt.Tooltip("mean:Q", format=".4f", title="SD"),
                alt.Tooltip("half:Q", format=".4f", title="95% CI"),
                alt.Tooltip("acc:Q", format=".3f", title="accuracy"),
            ],
        )
    )
    _labels = (
        alt.Chart(CIFAR_SKIP[CIFAR_SKIP["aw"] == "AW=2"])
        .mark_text(align="center", dy=-10, fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("aw:N", sort=_order),
            y=alt.Y("mean:Q"),
            text="mode:N",
            color=alt.Color("mode:N", scale=_scale, legend=None),
        )
    )
    _chart = (_band + _line + _labels).properties(
        width="container", height=280
    )
    _caption = mo.md(
        '<p class="figure-caption">Figure 7. The block-3 skip lowers '
        'the level but not the ordering. Lines are means over two seeds; '
        'bands are 95% CIs. At 150 epochs, batch 512. The paper reports '
        'the opposite ordering: the baseline is widest and self-modeling '
        'narrows it. The batch-64 probe is in Table 1. Source: '
        '<code>cifar_skip.json</code>; script: '
        '<code>scripts/cifar_skip_experiment.py</code>.</p>'
    )

    _pivot = CIFAR_SKIP_B64.pivot(
        index="mode", columns="aw", values="mean"
    )
    _acc = CIFAR_SKIP_B64.groupby("mode")["acc"].mean()
    _lines = [
        "| model | AW 0 | AW 2 | accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    _short = {"standard ResNet-18": "standard", "spatial skip": "spatial"}
    for _mode in ["standard ResNet-18", "spatial skip"]:
        _lines.append(
            f"| {_short[_mode]} | {_pivot.loc[_mode, 'AW=0']:.4f} | "
            f"{_pivot.loc[_mode, 'AW=2']:.4f} | "
            f"{_acc.loc[_mode]:.3f} |"
        )
    _table = mo.md(
        "**Table 1. The batch-64 probe.** One seed, 75 epochs. At the "
        "paper's batch size the absolute SD is higher, but the ordering "
        "is still inverted. Source: `cifar_skip.json`.\n\n"
        + "\n".join(_lines)
    )
    mo.vstack([_chart, _caption, _table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The skip changes the level, not the ordering. At 150 epochs the
    standard model goes from **0.0400** at AW 0 to **0.0508** at AW 2.
    The block-3 model goes from **0.0321** to **0.0402**. The skip
    lowers the absolute SD, but the self-modeling run is still wider.

    We also ran the paper's batch size. At batch 64 we used a spatial
    skip: concatenate block 3 with an upsampled block 4, fuse them with
    a 1x1 convolution to 512, then pool. At 75 epochs, the standard
    model goes from **0.0759** to **0.1001**; the spatial skip goes from
    **0.1050** to **0.1101**. The absolute SD is much higher, and the
    ordering is still inverted.

    The block-3 intervals do not overlap. The standard run's two-seed
    intervals do, so read that arm as a sign rather than a measurement.
    The batch-64 probe is one seed. The paper uses 150 epochs and does
    not pin down the skip placement. Both skip placements fail to flip
    the ordering. This is a negative result for the obvious
    architectural fix, not a proof that no implementation would work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        '<blockquote class="pullquote">The stop-gradient is essential '
        'to the MNIST mechanism and irrelevant to the CIFAR inversion. '
        'Both can be true.</blockquote>'
    )
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
    <p class="figure-caption">Table 2. Replication matrix.
    <strong>Yes</strong> means the paper's ordering and scale are within
    our interval; <strong>No</strong> means the ordering is inverted or
    the effect is missing; <strong>Partly</strong> means accuracy matches
    but the effect does not. 3A was also tested with pooled and spatial
    skips, at batch 512 and 64; the ordering stayed inverted. Full notes
    and probes are in the lab notebook.</p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What next

    Three experiments would change the conclusion:

    - **A 10-seed batch-64 protocol.** Our batch-64 probe is one seed
      and 75 epochs. The paper uses 150 epochs. A 10-seed run at the
      paper's exact batch, schedule, and augmentation is the remaining
      faithful CIFAR test.
    - **More seeds for the slow-head control.** Three seeds and 100
      epochs is a sign. Ten seeds and 250 epochs is a measurement.
    - **A rank diagnostic for IMDB.** The hidden layer is 128 units from
      a 64-dimensional embedding. A rank test would show whether the
      self-model rows learn a projection, as the weight distance
      suggests.

    The effect is real, fragile, and conditional. That is a more useful
    result than a clean reproduction.

    *A reproduction that only reports the panels it matches would have
    missed both mechanisms.*
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
