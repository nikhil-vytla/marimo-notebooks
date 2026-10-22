# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


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
          margin-bottom: 1.4rem;
        }
        .essay-kicker {
          font: 650 0.72rem/1.2 ui-sans-serif, system-ui, sans-serif;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--accent);
        }
        .essay-hero h1 {
          font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
          font-size: clamp(2.1rem, 4vw, 3.3rem);
          line-height: 1.02;
          letter-spacing: -0.02em;
          margin: 0.5rem 0 0.8rem;
          color: var(--ink);
        }
        .essay-lede {
          font-size: 1.12rem;
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
        .score-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
          gap: 0.6rem;
          margin: 0.9rem 0 1.2rem;
        }
        .score-card {
          border: 1px solid var(--line);
          border-radius: 0.7rem;
          padding: 0.75rem 0.85rem;
          background: #fff;
        }
        .score-card .panel {
          font: 650 0.7rem/1.2 ui-monospace, SFMono-Regular, monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .score-card .status {
          font-weight: 700;
          margin: 0.3rem 0 0.2rem;
          font-size: 0.95rem;
        }
        .score-card .note {
          font-size: 0.8rem;
          line-height: 1.4;
          color: var(--muted);
        }
        .score-card.yes .status { color: var(--green); }
        .score-card.partly .status { color: var(--accent); }
        .score-card.no .status { color: var(--red); }
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
        }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _():
    META = {
        "kicker": "Visual essay",
        "title": "The paper, as an argument",
        "lede": (
            "One sentence that makes a reader want the next paragraph. "
            "Name the surprise, the tension, or the question."
        ),
        "paper": "Paper title",
        "paper_url": "https://arxiv.org/abs/0000.00000",
        "authors": "Author names, year",
        "status": "in progress",
        "status_class": "partly",
        "reading_time": "18 min",
        "updated": "2026-10-19",
        "repo": "https://github.com/nikhil-vytla/marimo-notebooks",
    }
    SCORE = [
        {
            "panel": "2A",
            "status": "yes",
            "label": "Reproduced",
            "note": "The ordering matches within 0.001.",
        },
        {
            "panel": "3A",
            "status": "no",
            "label": "Not reproduced",
            "note": "The ordering is inverted; the essay explains why.",
        },
        {
            "panel": "4C",
            "status": "partly",
            "label": "Partly",
            "note": "Accuracy matches, the effect does not.",
        },
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
          <a href="#the-question">The question</a>
          <a href="#the-mechanism">The mechanism</a>
          <a href="#what-reproduced">What reproduced</a>
          <a href="#what-did-not">What did not</a>
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
          The first four sections are the argument. The last two are the
          lab notebook: what we ran, what broke, and what we would do
          next. Every number links back to a saved result.
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The question

    Start with the claim in plain language. Then narrow it to the one
    thing this essay tests. A reader should be able to repeat the
    question after two paragraphs.

    Use a short paragraph and one concrete number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The mechanism

    Explain the smallest model that produces the effect. If there is a
    parameter the reader can turn, say what it changes and why.
    """)
    return


@app.cell
def _(mo):
    guess = mo.ui.radio(
        options={
            "The effect gets stronger": "stronger",
            "The effect gets weaker": "weaker",
            "Nothing changes": "nothing",
        },
        label="Predict before you look: what happens when you turn the knob?",
    )
    reveal = mo.ui.checkbox(label="Reveal")
    mo.vstack([guess, reveal])
    return guess, reveal


@app.cell
def _(guess, mo, reveal):
    _correct = guess.value == "stronger"
    if guess.value is None:
        _lead = "Pick an answer, then reveal."
        _kind = "neutral"
    elif _correct:
        _lead = "Right. The effect strengthens."
        _kind = "success"
    else:
        _lead = "Not quite. The effect strengthens."
        _kind = "warn"

    if reveal.value:
        _out = mo.callout(
            mo.md(
                f"{_lead} Replace this with the measured result and the "
                "number that settles it."
            ),
            kind=_kind,
        )
    else:
        _out = mo.md("_Reveal to see the result._")
    _out
    return


@app.cell(hide_code=True)
def _(SCORE, mo):
    _cards = []
    for _row in SCORE:
        _cards.append(
            f"""
            <div class="score-card {_row["status"]}">
              <div class="panel">{_row["panel"]}</div>
              <div class="status">{_row["label"]}</div>
              <div class="note">{_row["note"]}</div>
            </div>
            """
        )
    mo.Html(f'<div class="score-grid">{"".join(_cards)}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What reproduced

    Put the hero result here. One figure, one sentence, and a caption
    that says what the reader is looking at and what would falsify it.

    <p class="figure-caption">Figure 1. Replace with the real caption,
    including n, seeds, and the source file.</p>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What did not

    Negative results are the second act, not an appendix. Show the
    disagreement and the probe that rules out the easy explanation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What next

    End with two or three experiments that would change the
    conclusion. Make them concrete enough that someone could run them.
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
            <strong>Data and code.</strong>
            <a href="{META["repo"]}">Repository</a>.
            Every figure reads a saved result from <code>public/data/</code>.
          </div>
          <div>
            <strong>How this was made.</strong>
            Built with marimo. The essay reads precomputed data; the
            training notebooks write it. See <code>docs/ESSAY-SHELL.md</code>.
          </div>
        </footer>
        """
    )
    return


if __name__ == "__main__":
    app.run()
