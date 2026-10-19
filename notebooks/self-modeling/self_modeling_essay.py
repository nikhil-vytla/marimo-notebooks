# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
#     "numpy==2.5.3",
#     "pandas==3.0.6",
#     "altair==6.3.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, json, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # When a network predicts itself, it gets simpler

    A visual essay on *Unexpected Benefits of Self-Modeling in Neural
    Systems* by Premakumar et al. (2024, AE Studio and Princeton),
    [arXiv:2407.10188](https://arxiv.org/abs/2407.10188).

    Neural networks are usually judged by what they get right. This
    paper asks a stranger question: what happens when a network is
    also rewarded for predicting its own internal state? The surprise
    is that it becomes simpler. Adding a self-prediction term to the
    loss narrows the spread of the final classifier's weights and
    lowers a formal complexity measure, while test accuracy barely
    moves. This notebook walks through the idea, our reproduction
    of Figure 2 on MNIST, and our attempts at the paper's CIFAR-10
    (Figure 3) and IMDB (Figure 4) panels. The MNIST panels
    reproduce. The others only partly do, and the places where they
    diverge turn out to be the interesting part.
    """)
    return


@app.cell
def _(mo):
    _svg = mo.Html(
        """
        <svg viewBox="0 0 860 350" width="100%"
             style="max-width:820px"
             xmlns="http://www.w3.org/2000/svg"
             font-family="system-ui, -apple-system, Segoe UI, sans-serif">
          <defs>
            <marker id="arw" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6"
                    orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#374151"/>
            </marker>
            <marker id="arwd" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="6" markerHeight="6"
                    orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#b45309"/>
            </marker>
          </defs>
          <rect x="443" y="35" width="314" height="237" rx="14"
                fill="none" stroke="#9ca3af" stroke-width="1.2"
                stroke-dasharray="5 4"/>
          <text x="600" y="26" text-anchor="middle" font-size="12"
                fill="#6b7280">
            one output layer, extra rows for self-prediction
          </text>
          <rect x="20" y="112" width="120" height="76" rx="10"
                fill="#e8f0fe" stroke="#374151" stroke-width="1.5"/>
          <text x="80" y="145" text-anchor="middle" font-size="15"
                font-weight="600" fill="#111827">input x</text>
          <text x="80" y="167" text-anchor="middle" font-size="12"
                fill="#4b5563">784 pixels</text>
          <line x1="140" y1="150" x2="203" y2="150" stroke="#374151"
                stroke-width="1.8" marker-end="url(#arw)"/>
          <rect x="205" y="98" width="170" height="104" rx="10"
                fill="#e6f4ea" stroke="#374151" stroke-width="1.5"/>
          <text x="290" y="140" text-anchor="middle" font-size="15"
                font-weight="600" fill="#111827">hidden a</text>
          <text x="290" y="161" text-anchor="middle" font-size="12"
                fill="#4b5563">512 units</text>
          <g fill="#34a853">
            <circle cx="255" cy="182" r="5"/>
            <circle cx="275" cy="182" r="5"/>
            <circle cx="295" cy="182" r="5"/>
            <circle cx="315" cy="182" r="5"/>
            <circle cx="335" cy="182" r="5"/>
          </g>
          <line x1="375" y1="128" x2="452" y2="90" stroke="#374151"
                stroke-width="1.8" marker-end="url(#arw)"/>
          <line x1="375" y1="162" x2="452" y2="210" stroke="#374151"
                stroke-width="1.8" marker-end="url(#arw)"/>
          <rect x="455" y="48" width="290" height="76" rx="10"
                fill="#fce8e6" stroke="#374151" stroke-width="1.5"/>
          <text x="600" y="81" text-anchor="middle" font-size="15"
                font-weight="600" fill="#111827">10 class logits</text>
          <text x="600" y="103" text-anchor="middle" font-size="12"
                fill="#4b5563">softmax + cross-entropy</text>
          <rect x="455" y="180" width="290" height="76" rx="10"
                fill="#f3e8fd" stroke="#374151" stroke-width="1.5"/>
          <text x="600" y="213" text-anchor="middle" font-size="15"
                font-weight="600" fill="#111827">self-prediction a-hat</text>
          <text x="600" y="235" text-anchor="middle" font-size="12"
                fill="#4b5563">extra output rows</text>
          <path d="M 290 202 C 290 292, 420 274, 450 236"
                fill="none" stroke="#b45309" stroke-width="1.8"
                stroke-dasharray="6 5" marker-end="url(#arwd)"/>
          <text x="175" y="300" text-anchor="middle" font-size="12"
                fill="#b45309">detach(a): target,</text>
          <text x="175" y="316" text-anchor="middle" font-size="12"
                fill="#b45309">no gradient back</text>
        </svg>
        """)
    mo.vstack([mo.md("## The idea in one picture"), _svg])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The loss

    The network keeps its ordinary classification loss. On top of it
    we add a second job: predict the hidden activations $a$ from the
    same hidden layer, using extra rows of the output layer.

    $$L = L_{\mathrm{CE}} + \mathrm{AW} \cdot
    \frac{1}{n} \sum_{i=1}^{n}
    \left( \hat{a}_i - \mathrm{detach}(a_i) \right)^2$$

    $\mathrm{AW}$ is the weight on the self-prediction term and
    $\mathrm{detach}(\cdot)$ is a stop-gradient. The target $a$ is
    detached, so the squared error does not pull on the target copy
    of $a$, only on the prediction $\hat{a}$ and the weights that
    produce it.

    To see the trade-off, fix an illustrative step with
    cross-entropy $0.35$ and self-prediction MSE $0.02$. The slider
    sets $\mathrm{AW}$; the total loss is
    $\mathrm{CE} + \mathrm{AW} \cdot \mathrm{MSE}$.
    """)
    return


@app.cell
def _(mo):
    aw_slider = mo.ui.slider(
        start=0,
        stop=50,
        step=1,
        value=0,
        label="AW: weight on the self-prediction loss",
    )
    aw_slider
    return (aw_slider,)


@app.cell
def _(alt, aw_slider, mo, np, pd):
    _ce = 0.35
    _mse = 0.02
    _aw = int(aw_slider.value)
    _self = _aw * _mse
    _total = _ce + _self

    _summary = mo.md(
        f"At AW = **{_aw}**: cross-entropy is **{_ce:.2f}**, the "
        f"self-prediction term is **{_self:.2f}**, so the total loss is "
        f"**{_total:.2f}**. Self-prediction makes up "
        f"{_self / _total:.0%} of it."
    )

    _df = pd.DataFrame(
        {
            "term": ["cross-entropy", "self-prediction"],
            "value": [_ce, _self],
        }
    )
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="loss"),
            color=alt.Color(
                "term:N",
                scale=alt.Scale(
                    domain=["cross-entropy", "self-prediction"],
                    range=["#4c78a8", "#f58518"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["term", "value"],
        )
        .properties(width=520, height=70)
    )

    _sample = np.array([0, 1, 5, 10, 20, 50])
    _lines = [
        "| AW | AW x MSE | total loss |",
        "| ---: | ---: | ---: |",
    ]
    _lines += [
        f"| {int(_a)} | {_a * _mse:.2f} | {_ce + _a * _mse:.2f} |"
        for _a in _sample
    ]
    mo.vstack([_summary, _chart, mo.md("\n".join(_lines))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Puzzle: which curve is the baseline?

    The paper's Figure 2a plots the standard deviation of the
    $10 \times 512$ final-layer classifier weights over 50 epochs,
    one curve per self-prediction weight. Each curve has a real
    label (its value of $\mathrm{AW}$), but here they are drawn in
    neutral gray and shuffled.

    *Baseline* means $\mathrm{AW} = 0$: no self-prediction at all.
    These curves are our variant (a) runs, which follow the paper's
    preprocessing. Pick the curve you think is the baseline, then
    reveal.
    """)
    return


@app.cell
def _(AW_LIST, np):
    _palette = [
        "#1a1a1a",
        "#3d3d3d",
        "#5f5f5f",
        "#828282",
        "#a4a4a4",
        "#c7c7c7",
    ]
    _rng = np.random.default_rng(7)
    _order = [int(_a) for _a in _rng.permutation(AW_LIST)]
    NEUTRAL_LABELS = {
        _aw: f"Line {_i + 1}" for _i, _aw in enumerate(_order)
    }
    NEUTRAL_COLORS = {
        _aw: _palette[_i] for _i, _aw in enumerate(_order)
    }
    return NEUTRAL_COLORS, NEUTRAL_LABELS


@app.cell
def _(AW_LIST, mo):
    _options = [f"Line {_i + 1}" for _i in range(len(AW_LIST))]
    puzzle_guess = mo.ui.dropdown(
        options=_options,
        value=None,
        label="Which line is the baseline (AW = 0)?",
    )
    puzzle_guess_aw50 = mo.ui.dropdown(
        options=_options,
        value=None,
        label="Which line is AW = 50?",
    )
    puzzle_reveal = mo.ui.checkbox(label="Reveal the real labels")
    mo.vstack([puzzle_guess, puzzle_guess_aw50, puzzle_reveal])
    return puzzle_guess, puzzle_guess_aw50, puzzle_reveal


@app.cell
def _(
    AW_LIST,
    FIG2A,
    NEUTRAL_COLORS,
    NEUTRAL_LABELS,
    PAPER_COLORS,
    alt,
    mo,
    puzzle_guess,
    puzzle_guess_aw50,
    puzzle_reveal,
):
    _df = FIG2A[FIG2A["variant"] == "a"].copy()

    if puzzle_reveal.value:
        _df["curve"] = _df["aw"].map(lambda _a: f"AW={_a}")
        _domain = [f"AW={_a}" for _a in AW_LIST]
        _range = [PAPER_COLORS[_a] for _a in AW_LIST]
        _legend = "Self-prediction weight"
    else:
        _df["curve"] = _df["aw"].map(NEUTRAL_LABELS)
        _line_labels = [
            f"Line {_i + 1}" for _i in range(len(AW_LIST))
        ]
        _label_to_aw = {
            _v: _k for _k, _v in NEUTRAL_LABELS.items()
        }
        _domain = _line_labels
        _range = [
            NEUTRAL_COLORS[_label_to_aw[_label]]
            for _label in _line_labels
        ]
        _legend = "Which line?"

    _chart = (
        alt.Chart(_df)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(
                "mean:Q",
                title="SD of classifier weights",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "curve:N",
                scale=alt.Scale(domain=_domain, range=_range),
                legend=alt.Legend(title=_legend),
            ),
            tooltip=[
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("curve:N"),
                alt.Tooltip("mean:Q", format=".4f"),
            ],
        )
        .properties(width=600, height=330)
    )

    _parts = [mo.ui.altair_chart(_chart)]
    if puzzle_reveal.value:
        _baseline = NEUTRAL_LABELS[0]
        _aw50 = NEUTRAL_LABELS[50]

        if puzzle_guess.value is None:
            _verdict = "**You didn't pick a line.** "
        elif puzzle_guess.value == _baseline:
            _verdict = "**Correct.** "
        else:
            _verdict = (
                f"**Not quite** (you picked {puzzle_guess.value}). "
            )
        _parts.append(
            mo.md(
                _verdict
                + f"The baseline is **{_baseline}**, the line for "
                "AW = 0 (no self-prediction)."
            )
        )

        if puzzle_guess_aw50.value is None:
            _verdict50 = "**You didn't pick a line.** "
        elif puzzle_guess_aw50.value == _aw50:
            _verdict50 = "**Correct.** "
        else:
            _verdict50 = (
                f"**Not quite** (you picked "
                f"{puzzle_guess_aw50.value}). "
            )
        _parts.append(
            mo.md(
                _verdict50
                + f"AW = 50 is **{_aw50}**. Hint: look at epoch 1, "
                "where the AW = 50 curve starts lower than the rest."
            )
        )
    mo.vstack(_parts)
    return


@app.cell
def _(json, mo, pd):
    _path = (
        mo.notebook_location()
        / "public"
        / "data"
        / "mnist_fig2a_local.json"
    )
    _raw = json.loads(_path.read_text())

    AW_LIST = [0, 1, 5, 10, 20, 50]
    PAPER_COLORS = {
        0: "#000000",
        1: "#ff1493",
        5: "#00e000",
        10: "#8a2be2",
        20: "#ff4500",
        50: "#00bfff",
    }
    PAPER_EPOCH50 = {
        0: 0.097,
        1: 0.094,
        5: 0.091,
        10: 0.088,
        20: 0.085,
        50: 0.081,
    }

    _rows = []
    for _variant in ("a", "b"):
        for _aw in AW_LIST:
            _entry = _raw[_variant][str(_aw)]
            for _i, (_mean, _half) in enumerate(
                zip(_entry["m"], _entry["h"])
            ):
                _rows.append(
                    {
                        "variant": _variant,
                        "aw": _aw,
                        "epoch": _i + 1,
                        "mean": _mean,
                        "half": _half,
                        "acc": _entry["acc"],
                        "n": _entry["n"],
                    }
                )
    FIG2A = pd.DataFrame(_rows)
    return AW_LIST, FIG2A, PAPER_COLORS, PAPER_EPOCH50


@app.cell
def _(mo):
    mo.md(r"""
    ## Our reproduction of Figure 2a

    We ran the experiment locally: an MLP on MNIST, six values of
    $\mathrm{AW}$, ten seeds each. The shaded bands are the mean
    $\pm$ the reported CI half-width. Diamonds mark the values we
    read off the paper's figure at epoch 50.

    Toggle between the two preprocessing variants we tried.
    """)
    return


@app.cell
def _(mo):
    variant_pick = mo.ui.radio(
        options={
            "a: rescale to [-1, 1] (paper's setup)": "a",
            "b: rescale, then subtract the training mean": "b",
        },
        value="a: rescale to [-1, 1] (paper's setup)",
        label="Preprocessing variant",
    )
    variant_pick
    return (variant_pick,)


@app.cell
def _(
    AW_LIST,
    FIG2A,
    PAPER_COLORS,
    PAPER_EPOCH50,
    alt,
    mo,
    pd,
    variant_pick,
):
    _v = variant_pick.value
    _df = FIG2A[FIG2A["variant"] == _v].copy()
    _df["lo"] = _df["mean"] - _df["half"]
    _df["hi"] = _df["mean"] + _df["half"]
    _df["curve"] = _df["aw"].map(lambda _a: f"AW={_a}")

    _domain = [f"AW={_a}" for _a in AW_LIST]
    _range = [PAPER_COLORS[_a] for _a in AW_LIST]
    _scale = alt.Scale(domain=_domain, range=_range)
    _y_domain = [
        float(_df["lo"].min()) - 0.001,
        float(_df["hi"].max()) + 0.001,
    ]

    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.16)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(
                "lo:Q",
                title="SD of classifier weights",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            y2="hi:Q",
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
        )
    )
    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y(
                "mean:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color("curve:N", scale=_scale, legend=None),
        )
    )

    _paper = pd.DataFrame(
        {
            "curve": [f"AW={_a}" for _a in AW_LIST],
            "epoch": [50] * len(AW_LIST),
            "paper": [PAPER_EPOCH50[_a] for _a in AW_LIST],
        }
    )
    _points = (
        alt.Chart(_paper)
        .mark_point(
            shape="diamond", size=90, filled=True, stroke="white"
        )
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y(
                "paper:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".3f"),
            ],
        )
    )

    _chart = (_band + _line + _points).properties(
        width=620, height=360
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(AW_LIST, FIG2A, PAPER_EPOCH50, mo):
    _rows = []
    for _aw in AW_LIST:
        _a = FIG2A[
            (FIG2A["variant"] == "a")
            & (FIG2A["aw"] == _aw)
            & (FIG2A["epoch"] == 50)
        ].iloc[0]
        _b = FIG2A[
            (FIG2A["variant"] == "b")
            & (FIG2A["aw"] == _aw)
            & (FIG2A["epoch"] == 50)
        ].iloc[0]
        _rows.append(
            {
                "aw": _aw,
                "paper": PAPER_EPOCH50[_aw],
                "a": float(_a["mean"]),
                "ci": float(_a["half"]),
                "b": float(_b["mean"]),
                "acc_a": float(_a["acc"]),
                "acc_b": float(_b["acc"]),
            }
        )

    _lines = [
        (
            "| AW | paper | ours (a) | +/- CI | ours (b) "
            "| acc (a) | acc (b) |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _r in _rows:
        _lines.append(
            f"| {_r['aw']} | {_r['paper']:.3f} | {_r['a']:.4f} | "
            f"{_r['ci']:.4f} | {_r['b']:.4f} | {_r['acc_a']:.3f} | "
            f"{_r['acc_b']:.3f} |"
        )

    _a_range = _rows[0]["a"] - _rows[-1]["a"]
    _b_range = _rows[0]["b"] - _rows[-1]["b"]
    _shrink = 1 - _b_range / _a_range

    _prose = mo.md(f"""
    Our variant (a) tracks the paper to within about 0.001 at epoch
    50 for every value of AW, with the same ordering: more
    self-prediction, narrower weights. Test accuracy slips only from
    {_rows[0]['acc_a']:.1%} at AW = 0 to {_rows[-1]['acc_a']:.1%} at
    AW = 50.

    Variant (b) keeps the ordering but compresses it: the spread
    between the top and bottom curves at epoch 50 is {_a_range:.4f}
    for (a) and {_b_range:.4f} for (b), about {_shrink:.0%} smaller.
    That points to the plain rescale to [-1, 1] as the preprocessing
    behind the paper's effect, not mean-centering.
    """)
    mo.vstack([mo.md("\n".join(_lines)), _prose])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How we reproduced it

    - **Batch size 512.** The paper's Table 1 lists 64, which is
      wrong; we use 512.
    - **Detach the target.** The self-prediction target is
      `a.detach()`, so no gradient flows back through the target
      copy of the activations.
    - **Ten seeds per configuration.** Each curve is the mean over
      10 seeds, and the bands are the 95% CI half-widths.
    - **120 separate CPU runs.** The curves shown come from 120
      runs (6 AWs x 10 seeds x 2 preprocessing variants).
    - **A batched GPU cross-check.** A batched GPU trainer, run
      independently, agreed with the CPU runs within the 95% CI at
      every AW.

    [`self_modeling_train.py`](self_modeling_train.py) is that
    batched trainer, set up for molab's GPU. It reproduces the
    batched run's epoch-50 means to 5 decimals.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Does it hold across network sizes? (Figure 2B and 2D)

    The Figure 2a effect could be a quirk of one architecture. The
    paper repeats the same sweep at hidden sizes 64, 128, 256, and
    512. Figure 2B tracks the spread of the final classifier weights
    at epoch 50; Figure 2D tracks final test accuracy.
    """)
    return


@app.cell
def _(AW_LIST, json, mo, pd):
    _path = (
        mo.notebook_location()
        / "public"
        / "data"
        / "mnist_fig2.json"
    )
    FIG2 = json.loads(_path.read_text())
    FIG2_SIZES = [int(_z) for _z in FIG2["config"]["hidden_sizes"]]
    FIG2_SNAPSHOT_EPOCHS = [int(_e) for _e in FIG2["snapshot_epochs"]]
    FIG2_BINS = int(FIG2["bins"])
    FIG2_BIN_RANGE = [float(_v) for _v in FIG2["bin_range"]]

    _sd_rows = []
    _acc_rows = []
    for _z in FIG2_SIZES:
        _by_aw = FIG2["results"][str(_z)]["aws"]
        for _aw in AW_LIST:
            _entry = _by_aw[str(_aw)]
            _sd_rows.append(
                {
                    "hidden": _z,
                    "aw": _aw,
                    "curve": f"AW={_aw}",
                    "sd": float(_entry["sd_mean"][50]),
                    "half": float(_entry["sd_ci"][50]),
                }
            )
            _acc_rows.append(
                {
                    "hidden": _z,
                    "aw": _aw,
                    "curve": f"AW={_aw}",
                    "acc": float(_entry["acc_mean"]),
                    "half": float(_entry["acc_ci"]),
                }
            )
    FIG2B = pd.DataFrame(_sd_rows)
    FIG2B["lo"] = FIG2B["sd"] - FIG2B["half"]
    FIG2B["hi"] = FIG2B["sd"] + FIG2B["half"]
    FIG2D = pd.DataFrame(_acc_rows)
    FIG2D["lo"] = FIG2D["acc"] - FIG2D["half"]
    FIG2D["hi"] = FIG2D["acc"] + FIG2D["half"]

    _paper_sd = {
        0: [0.262, 0.190, 0.136, 0.098],
        1: [0.235, 0.176, 0.130, 0.093],
        5: [0.222, 0.165, 0.124, 0.091],
        10: [0.231, 0.160, 0.118, 0.089],
        20: [0.120, 0.170, 0.115, 0.087],
        50: [0.096, 0.080, 0.113, 0.083],
    }
    PAPER_FIG2B = pd.DataFrame(
        [
            {
                "hidden": _z,
                "aw": _aw,
                "curve": f"AW={_aw}",
                "paper": _v,
            }
            for _aw, _values in _paper_sd.items()
            for _z, _v in zip(FIG2_SIZES, _values)
        ]
    )

    _paper_acc = {
        20: [0.27, 0.96, 0.96, 0.97],
        50: [0.11, 0.14, 0.93, 0.96],
    }
    PAPER_FIG2D = pd.DataFrame(
        [
            {
                "hidden": _z,
                "aw": _aw,
                "curve": f"AW={_aw}",
                "paper": _v,
            }
            for _aw, _values in _paper_acc.items()
            for _z, _v in zip(FIG2_SIZES, _values)
        ]
    )
    return (
        FIG2,
        FIG2B,
        FIG2D,
        FIG2_BINS,
        FIG2_BIN_RANGE,
        FIG2_SIZES,
        FIG2_SNAPSHOT_EPOCHS,
        PAPER_FIG2B,
        PAPER_FIG2D,
    )


@app.cell
def _(AW_LIST, FIG2B, PAPER_COLORS, PAPER_FIG2B, alt, mo):
    _domain = [f"AW={_a}" for _a in AW_LIST]
    _scale = alt.Scale(
        domain=_domain,
        range=[PAPER_COLORS[_a] for _a in AW_LIST],
    )
    _order = sorted(FIG2B["hidden"].unique().tolist())
    _y_domain = [
        float(min(FIG2B["lo"].min(), PAPER_FIG2B["paper"].min()))
        - 0.005,
        float(max(FIG2B["hi"].max(), PAPER_FIG2B["paper"].max()))
        + 0.005,
    ]
    _x = alt.X(
        "hidden:O",
        title="Hidden size",
        sort=_order,
        axis=alt.Axis(labelAngle=0),
    )
    _band = (
        alt.Chart(FIG2B)
        .mark_area(opacity=0.16)
        .encode(
            x=_x,
            y=alt.Y(
                "lo:Q",
                title="SD of classifier weights (epoch 50)",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            y2="hi:Q",
            color=alt.Color("curve:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(FIG2B)
        .mark_line(strokeWidth=2)
        .encode(
            x=_x,
            y=alt.Y(
                "sd:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
            tooltip=[
                alt.Tooltip("hidden:O"),
                alt.Tooltip("curve:N"),
                alt.Tooltip("sd:Q", format=".4f"),
            ],
        )
    )
    _points = (
        alt.Chart(PAPER_FIG2B)
        .mark_point(filled=False, strokeWidth=1.6, size=70)
        .encode(
            x=_x,
            y=alt.Y(
                "paper:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".3f"),
            ],
        )
    )
    _chart = (_band + _line + _points).properties(
        width=620,
        height=340,
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(AW_LIST, FIG2B, FIG2D, PAPER_COLORS, PAPER_FIG2D, alt, mo):
    _domain = [f"AW={_a}" for _a in AW_LIST]
    _scale = alt.Scale(
        domain=_domain,
        range=[PAPER_COLORS[_a] for _a in AW_LIST],
    )
    _order = sorted(FIG2D["hidden"].unique().tolist())
    _y_domain = [0.0, 1.0]
    _x = alt.X(
        "hidden:O",
        title="Hidden size",
        sort=_order,
        axis=alt.Axis(labelAngle=0),
    )
    _band = (
        alt.Chart(FIG2D)
        .mark_area(opacity=0.16)
        .encode(
            x=_x,
            y=alt.Y(
                "lo:Q",
                title="Final test accuracy",
                scale=alt.Scale(domain=_y_domain),
            ),
            y2="hi:Q",
            color=alt.Color("curve:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(FIG2D)
        .mark_line(strokeWidth=2)
        .encode(
            x=_x,
            y=alt.Y(
                "acc:Q",
                scale=alt.Scale(domain=_y_domain),
            ),
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
            tooltip=[
                alt.Tooltip("hidden:O"),
                alt.Tooltip("curve:N"),
                alt.Tooltip("acc:Q", format=".4f"),
            ],
        )
    )
    _points = (
        alt.Chart(PAPER_FIG2D)
        .mark_point(filled=False, strokeWidth=1.6, size=70)
        .encode(
            x=_x,
            y=alt.Y(
                "paper:Q",
                scale=alt.Scale(domain=_y_domain),
            ),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".2f"),
            ],
        )
    )
    _chart = (_band + _line + _points).properties(
        width=620,
        height=340,
    )

    _sd = {
        (_r["aw"], _r["hidden"]): float(_r["sd"])
        for _r in FIG2B.to_dict("records")
    }
    _acc = {
        (_r["aw"], _r["hidden"]): float(_r["acc"])
        for _r in FIG2D.to_dict("records")
    }
    _collapse = {(20, 64), (50, 64), (50, 128)}
    _healthy = [
        _acc[(_r["aw"], _r["hidden"])]
        for _r in FIG2D.to_dict("records")
        if (_r["aw"], _r["hidden"]) not in _collapse
    ]

    _prose = mo.md(f"""
    Bigger hidden layers have narrower weights overall, and
    self-modeling narrows them further at every size. At AW = 0 the
    epoch-50 SD falls from {_sd[(0, 64)]:.3f} at hidden 64 to
    {_sd[(0, 512)]:.3f} at hidden 512; at AW = 50 it falls from
    {_sd[(50, 64)]:.3f} to {_sd[(50, 512)]:.3f}.

    The exception is small networks at large AW. Accuracy collapses
    to near chance (~11%) for AW 20 at hidden 64
    ({_acc[(20, 64)]:.1%}) and for AW 50 at hidden 64
    ({_acc[(50, 64)]:.1%}) and hidden 128 ({_acc[(50, 128)]:.1%}).
    There the self-model term swamps the classification task, so the
    narrow weights are a failure, not a simplification. Everywhere
    else final accuracy stays in
    [{min(_healthy):.1%}, {max(_healthy):.1%}]. Figure 2D marks the
    paper's AW 20 and AW 50 read-offs; its other four curves sit near
    0.95-0.97 at every size.
    """)
    mo.vstack([mo.ui.altair_chart(_chart), _prose])
    return


@app.cell
def _(FIG2_SIZES, FIG2_SNAPSHOT_EPOCHS, mo):
    hist_epoch = mo.ui.slider(
        steps=FIG2_SNAPSHOT_EPOCHS,
        value=FIG2_SNAPSHOT_EPOCHS[0],
        label="Snapshot epoch",
    )
    hist_size = mo.ui.dropdown(
        options=FIG2_SIZES,
        value=512,
        label="Hidden size",
    )
    mo.hstack([hist_size, hist_epoch])
    return hist_epoch, hist_size


@app.cell
def _(
    FIG2,
    FIG2_BINS,
    FIG2_BIN_RANGE,
    alt,
    hist_epoch,
    hist_size,
    mo,
    np,
    pd,
):
    _size = int(hist_size.value)
    _epoch = int(hist_epoch.value)
    _edges = np.linspace(
        FIG2_BIN_RANGE[0],
        FIG2_BIN_RANGE[1],
        FIG2_BINS + 1,
    )
    _centers = (_edges[:-1] + _edges[1:]) / 2
    _by_aw = FIG2["results"][str(_size)]["aws"]

    # Fixed x-domain for this hidden size, pooled over every snapshot
    # epoch so the slider changes the curves but not the scale. The
    # seed-averaged histograms have near-zero tails in almost every
    # bin, so a bare nonzero test would keep the full [-0.6, 0.6]
    # range and squeeze hidden-512 into a few pixels. Treat bins under
    # 1% of the peak as empty, take the central occupied span across
    # both series, and pad one bin on each side.
    _floor = 0.01 * max(
        max(_counts)
        for _aw in (0, 50)
        for _counts in _by_aw[str(_aw)]["hist_mean"].values()
    )
    _occupied = np.flatnonzero(
        np.any(
            [
                np.asarray(_counts) >= _floor
                for _aw in (0, 50)
                for _counts in _by_aw[str(_aw)]["hist_mean"].values()
            ],
            axis=0,
        )
    )
    _x_domain = [
        float(_edges[max(int(_occupied.min()) - 1, 0)]),
        float(_edges[min(int(_occupied.max()) + 2, FIG2_BINS)]),
    ]

    _rows = []
    for _aw in (0, 50):
        _counts = _by_aw[str(_aw)]["hist_mean"][str(_epoch)]
        for _bin, _count in zip(_centers, _counts):
            _rows.append(
                {
                    "bin": float(_bin),
                    "density": float(_count),
                    "curve": f"AW={_aw}",
                }
            )
    _hist = pd.DataFrame(_rows)

    _chart = (
        alt.Chart(_hist)
        .mark_area(opacity=0.5, interpolate="step")
        .encode(
            x=alt.X(
                "bin:Q",
                title="Classifier weight",
                scale=alt.Scale(domain=_x_domain),
            ),
            y=alt.Y("density:Q", title="Mean count", stack=None),
            color=alt.Color(
                "curve:N",
                scale=alt.Scale(
                    domain=["AW=0", "AW=50"],
                    range=["#000000", "#00bfff"],
                ),
                legend=alt.Legend(title="AW"),
            ),
            tooltip=[
                alt.Tooltip("bin:Q", format=".3f"),
                alt.Tooltip("density:Q", format=".4f"),
            ],
        )
        .properties(width=600, height=300)
    )
    mo.vstack(
        [
            mo.ui.altair_chart(_chart),
            mo.md(
                f"Hidden {_size}, snapshot epoch {_epoch}: watch the "
                "AW = 50 distribution stay narrower as training "
                "goes on."
            ),
        ]
    )
    return


@app.cell
def _(FIG2, FIG2_SIZES, mo):
    _config = FIG2["config"]
    _n = (
        len(FIG2_SIZES)
        * len(_config["aws"])
        * int(_config["seeds"])
    )
    _device = FIG2["device"]["name"]
    mo.md(
        f"Data: {_n} models ({len(FIG2_SIZES)} sizes x "
        f"{len(_config['aws'])} AWs x {int(_config['seeds'])} seeds) "
        f"trained on a molab GPU ({_device}), about 43 s of GPU "
        "time."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A better ruler: RLCT (Figure 2C)

    Weight spread is a crude proxy for complexity. A network can
    hold many small weights and still have many effective degrees of
    freedom, and spread says nothing about how the loss actually
    responds when you move the weights. What we want is a quantity
    that counts the directions the trained solution is genuinely
    free to move in.

    The real log canonical threshold (RLCT), estimated here as the
    local learning coefficient (LLC), does that. It measures the
    effective dimensionality of the loss landscape around the
    trained weights: how fast the loss rises as you wander away
    from them. We estimate it with a noisy random walk (stochastic
    gradient Langevin dynamics, SGLD) tethered to the trained
    weights by a localization strength. Lower RLCT means fewer
    effective parameters, so the same pattern as Figure 2a should
    show up here as a lower curve for larger self-prediction
    weights.
    """)
    return


@app.cell
def _(AW_LIST, PAPER_COLORS, json, mo, pd):
    _path = (
        mo.notebook_location()
        / "public"
        / "data"
        / "mnist_fig2c.json"
    )
    RLCT_RAW = json.loads(_path.read_text())
    RLCT_SIZES = sorted(
        int(_size)
        for _size in RLCT_RAW["by_localization"]["100"]["rlct"]
    )

    _rows = []
    for _loc, _entry in RLCT_RAW["by_localization"].items():
        for _size in RLCT_SIZES:
            for _aw in AW_LIST:
                _e = _entry["rlct"][str(_size)][str(_aw)]
                _rows.append(
                    {
                        "localization": _loc,
                        "hidden": _size,
                        "aw": _aw,
                        "curve": f"AW={_aw}",
                        "mean": float(_e["mean"]),
                        "ci": float(_e["ci"]),
                    }
                )
    FIG2C = pd.DataFrame(_rows)
    FIG2C["lo"] = FIG2C["mean"] - FIG2C["ci"]
    FIG2C["hi"] = FIG2C["mean"] + FIG2C["ci"]

    _paper_values = {
        0: [33, 38, 46, 58],
        1: [32, 37, 45, 58],
        5: [30, 34, 42, 56],
        10: [26, 32, 40, 53],
        20: [-4, 28, 37, 49],
        50: [-5, -16, 28, 37],
    }
    PAPER_FIG2C = pd.DataFrame(
        [
            {
                "aw": _aw,
                "hidden": _size,
                "curve": f"AW={_aw}",
                "paper": _value,
            }
            for _aw, _values in _paper_values.items()
            for _size, _value in zip(RLCT_SIZES, _values)
        ]
    )
    return FIG2C, PAPER_FIG2C, RLCT_RAW, RLCT_SIZES


@app.cell
def _(RLCT_RAW, mo):
    _locs = sorted(
        RLCT_RAW["by_localization"].keys(), key=int
    )
    _options = {f"Localization {_loc}": _loc for _loc in _locs}
    _default = f"Localization {_locs[0]}"
    loc_pick = mo.ui.radio(
        options=_options,
        value=_default,
        label="Localization strength",
    )
    loc_pick
    return (loc_pick,)


@app.cell
def _(
    AW_LIST,
    FIG2C,
    PAPER_COLORS,
    PAPER_FIG2C,
    RLCT_SIZES,
    alt,
    loc_pick,
    mo,
):
    _df = FIG2C[FIG2C["localization"] == loc_pick.value].copy()
    _domain = [f"AW={_a}" for _a in AW_LIST]
    _scale = alt.Scale(
        domain=_domain,
        range=[PAPER_COLORS[_a] for _a in AW_LIST],
    )
    _order = RLCT_SIZES
    _y_domain = [
        float(min(_df["lo"].min(), PAPER_FIG2C["paper"].min()))
        - 3.0,
        float(max(_df["hi"].max(), PAPER_FIG2C["paper"].max()))
        + 3.0,
    ]
    _x = alt.X(
        "hidden:O",
        title="Hidden size",
        sort=_order,
        axis=alt.Axis(labelAngle=0),
    )
    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.16)
        .encode(
            x=_x,
            y=alt.Y(
                "lo:Q",
                title="LLC estimate",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            y2="hi:Q",
            color=alt.Color("curve:N", scale=_scale, legend=None),
        )
    )
    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=_x,
            y=alt.Y(
                "mean:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
            tooltip=[
                alt.Tooltip("hidden:O"),
                alt.Tooltip("curve:N"),
                alt.Tooltip("mean:Q", format=".2f"),
                alt.Tooltip("ci:Q", title="+/- CI", format=".2f"),
            ],
        )
    )
    _points = (
        alt.Chart(PAPER_FIG2C)
        .mark_point(filled=False, strokeWidth=1.6, size=70)
        .encode(
            x=_x,
            y=alt.Y(
                "paper:Q",
                scale=alt.Scale(domain=_y_domain, zero=False),
            ),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".2f"),
            ],
        )
    )
    _chart = (
        _band + _line + _points
    ).properties(
        width=620,
        height=360,
        title=f"LLC at localization {loc_pick.value}",
    )
    mo.vstack(
        [
            mo.ui.altair_chart(_chart),
            mo.md(
                "Hollow points are the values read off the paper's "
                "Figure 2C. Bands are the mean +/- 95% CI half-width "
                "over 10 seeded models."
            ),
        ]
    )
    return


@app.cell
def _(FIG2C, PAPER_FIG2C, RLCT_SIZES, mo):
    _loc = "100"
    _ours = FIG2C[FIG2C["localization"] == _loc].set_index(
        ["aw", "hidden"]
    )
    _paper = PAPER_FIG2C.set_index(["aw", "hidden"])["paper"]

    _rel = []
    for _key, _row in _ours.iterrows():
        _p = float(_paper.loc[_key])
        if _row["mean"] > 1 and _p > 1:
            _rel.append(abs(_row["mean"] - _p) / abs(_p))
    _worst = max(_rel)

    _aw0 = {
        _s: float(_ours.loc[(0, _s), "mean"]) for _s in RLCT_SIZES
    }
    _collapsed_lines = []
    for (_aw, _size), _row in _ours.iterrows():
        if _row["mean"] < 1:
            _collapsed_lines.append(
                f"- AW = {_aw} at hidden {_size}: "
                f"**{_row['mean']:.1f}** "
                f"(95% CI +/- {_row['ci']:.1f})"
            )
    _collapsed = "\n".join(_collapsed_lines)

    _prose = mo.md(f"""
    At localization 100 our estimates land within about
    {_worst:.0%} of the paper at every setting where both are well
    above zero. RLCT rises with hidden size at every AW (at AW = 0
    it climbs from {_aw0[RLCT_SIZES[0]]:.1f} at hidden
    {RLCT_SIZES[0]} to {_aw0[RLCT_SIZES[-1]]:.1f} at hidden
    {RLCT_SIZES[-1]}), and it falls as AW grows at every size. Both
    directions point the same way: more self-prediction and smaller
    networks mean fewer effective parameters.

    The exceptions are the collapsed settings, where the
    self-prediction term swamps classification:

    {_collapsed}

    Negative or near-zero values with wide intervals, matching the
    paper's own plunge (its hollow points there sit at -4, -5 and
    -16).
    """)
    _prose
    return


@app.cell
def _(FIG2C, RLCT_RAW, RLCT_SIZES, mo):
    _h = RLCT_RAW["by_localization"]["100"]["hyperparams"]
    _v = RLCT_RAW["by_localization"]["100"]["validation"]
    _anchor = {
        (_row["localization"], _row["hidden"]): float(_row["mean"])
        for _row in FIG2C[FIG2C["aw"] == 0].to_dict("records")
    }
    _ratio = max(
        _anchor[("100", _s)] / _anchor[("1000", _s)]
        for _s in RLCT_SIZES
    )
    mo.md(f"""
    /// admonition | Reproduction note: which localization?
    The paper's supplement says its MNIST LLC runs used
    localization 1000, but the caption of its own
    localization-calibration figure says 100. The choice changes
    the level. At 1000 our estimates come out up to about
    {_ratio:.0f}x lower than the paper's, with the same ordering;
    at 100 they line up. On pilot models (H=512, AW 0), going
    from 200 to 4000 draws at localization 1000 moved the
    estimate only from 8.5 to 8.7.

    We estimate with a batched SGLD sampler validated against
    devinterp {_h['devinterp_version']} (worst relative difference
    {_v['worst_rel_diff']:.1%} on the pilot models). Settings:
    {_h['num_chains']} chains x {_h['num_draws']} draws, nbeta =
    {_h['nbeta']:.1f} (512/log 512), learning rate {_h['lr']:g}.
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Beyond MNIST: CIFAR-10 (Figure 3)

    The paper repeats the experiment on CIFAR-10 with a ResNet-18.
    The architecture differs from the MNIST MLP: here the classifier
    reads from a hidden layer of 2000 units, and self-prediction adds
    output rows that reconstruct that hidden layer. The panels ask
    the same three questions as before: does weight spread shrink
    (3A), does RLCT fall (3B), and does accuracy hold (3C)?
    """)
    return


@app.cell
def _(json, mo, pd):
    _base = mo.notebook_location() / "public" / "data"

    CIFAR_AWS = [0.0, 0.5, 1.0, 2.0]
    CIFAR_COLORS = {
        0.0: "#000000",
        0.5: "#00e000",
        1.0: "#0000ff",
        2.0: "#a52a2a",
    }
    CIFAR_PAPER_3B = {0.0: 82, 0.5: 72, 1.0: 69, 2.0: 66}
    CIFAR_PAPER_3C = [0.918, 0.920]

    _fig3 = json.loads((_base / "cifar_fig3.json").read_text())
    _fig3b = json.loads(
        (_base / "cifar_fig3b.json").read_text()
    )
    _probes = json.loads(
        (_base / "cifar_fig3_probes.json").read_text()
    )

    CIFAR_CONFIG = _fig3["config"]
    CIFAR_DEVICE = _fig3["device"]["name"]

    _sd_rows = []
    _acc_rows = []
    for _aw in CIFAR_AWS:
        _entry = _fig3["results"][f"{_aw:g}"]
        for _epoch, (_mean, _half) in enumerate(
            zip(_entry["sd_mean"], _entry["sd_ci"])
        ):
            _sd_rows.append(
                {
                    "aw": _aw,
                    "curve": f"AW={_aw:g}",
                    "epoch": _epoch,
                    "mean": float(_mean),
                    "half": float(_half),
                }
            )
        _acc_rows.append(
            {
                "aw": _aw,
                "curve": f"AW={_aw:g}",
                "acc": float(_entry["acc_mean"]),
                "half": float(_entry["acc_ci"]),
            }
        )
    CIFAR_FIG3A = pd.DataFrame(_sd_rows)
    CIFAR_FIG3A["lo"] = CIFAR_FIG3A["mean"] - CIFAR_FIG3A["half"]
    CIFAR_FIG3A["hi"] = CIFAR_FIG3A["mean"] + CIFAR_FIG3A["half"]
    CIFAR_FIG3C = pd.DataFrame(_acc_rows)
    CIFAR_FIG3C["lo"] = CIFAR_FIG3C["acc"] - CIFAR_FIG3C["half"]
    CIFAR_FIG3C["hi"] = CIFAR_FIG3C["acc"] + CIFAR_FIG3C["half"]

    _rlct_rows = []
    for _aw in CIFAR_AWS:
        _entry = _fig3b["rlct"][f"{_aw:g}"]
        _rlct_rows.append(
            {
                "aw": _aw,
                "curve": f"AW={_aw:g}",
                "mean": float(_entry["mean"]),
                "ci": float(_entry["ci"]),
                "n": len(_entry["llc"]),
            }
        )
    CIFAR_FIG3B = pd.DataFrame(_rlct_rows)
    CIFAR_FIG3B["lo"] = CIFAR_FIG3B["mean"] - CIFAR_FIG3B["ci"]
    CIFAR_FIG3B["hi"] = CIFAR_FIG3B["mean"] + CIFAR_FIG3B["ci"]

    _probe_rows = []
    for _variant, _aws in _probes["variants"].items():
        for _aw, _probe in _aws.items():
            _probe_rows.append(
                {
                    "variant": _variant,
                    "aw": float(_aw),
                    "sd_e50": float(_probe["sd_e50"]),
                    "hidden": tuple(_probe["hidden_absmean"]),
                    "cls_wnorm": tuple(_probe["cls_wnorm"]),
                    "acc_e50": float(_probe["acc_e50"]),
                }
            )
    CIFAR_PROBES = pd.DataFrame(_probe_rows)

    return (
        CIFAR_AWS,
        CIFAR_COLORS,
        CIFAR_CONFIG,
        CIFAR_DEVICE,
        CIFAR_FIG3A,
        CIFAR_FIG3B,
        CIFAR_FIG3C,
        CIFAR_PAPER_3B,
        CIFAR_PAPER_3C,
        CIFAR_PROBES,
    )


@app.cell
def _(
    CIFAR_AWS,
    CIFAR_COLORS,
    CIFAR_CONFIG,
    CIFAR_DEVICE,
    CIFAR_FIG3A,
    alt,
    mo,
):
    _domain = [f"AW={_a:g}" for _a in CIFAR_AWS]
    _scale = alt.Scale(
        domain=_domain,
        range=[CIFAR_COLORS[_a] for _a in CIFAR_AWS],
    )
    _band = (
        alt.Chart(CIFAR_FIG3A)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(
                "lo:Q",
                title="SD of classifier weights",
                scale=alt.Scale(zero=False),
            ),
            y2="hi:Q",
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
        )
    )
    _line = (
        alt.Chart(CIFAR_FIG3A)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y("mean:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("mean:Q", format=".4f"),
            ],
        )
    )
    _chart = (_band + _line).properties(width=600, height=330)

    _e250 = {
        _aw: float(
            CIFAR_FIG3A[
                (CIFAR_FIG3A["aw"] == _aw)
                & (CIFAR_FIG3A["epoch"] == 250)
            ]["mean"].iloc[0]
        )
        for _aw in CIFAR_AWS
    }
    _narrow = min(CIFAR_AWS, key=lambda _a: _e250[_a])
    _wide = max(CIFAR_AWS, key=lambda _a: _e250[_a])
    _seeds = int(CIFAR_CONFIG["seeds"])
    _epochs = int(CIFAR_CONFIG["epochs"])

    _prose = mo.md(f"""
    {len(CIFAR_AWS) * _seeds} models ({len(CIFAR_AWS)} AWs x
    {_seeds} seeds, {_epochs} epochs) trained on {CIFAR_DEVICE}.
    Bands are the mean +/- 95% CI half-width over seeds.

    At epoch 250 our baseline (AW = 0) has SD **{_e250[0.0]:.4f}**,
    inside the paper's 0.05-0.06 read-off. The ordering, though, is
    inverted. In the paper the baseline is the widest curve and
    larger AW narrows it. Here AW = 0 is the narrowest
    ({_e250[_narrow]:.4f}) and AW = 2 is the widest
    ({_e250[_wide]:.4f}). Self-modeling widened the classifier
    instead of narrowing it.
    """)
    mo.vstack([mo.ui.altair_chart(_chart), _prose])
    return


@app.cell
def _(
    CIFAR_AWS,
    CIFAR_COLORS,
    CIFAR_FIG3B,
    CIFAR_PAPER_3B,
    alt,
    mo,
    np,
    pd,
):
    _domain = [f"AW={_a:g}" for _a in CIFAR_AWS]
    _scale = alt.Scale(
        domain=_domain,
        range=[CIFAR_COLORS[_a] for _a in CIFAR_AWS],
    )
    _y = alt.Scale(domain=[0, 90])
    _bars = (
        alt.Chart(CIFAR_FIG3B)
        .mark_bar(size=42)
        .encode(
            x=alt.X("curve:N", title=None, sort=_domain),
            y=alt.Y("mean:Q", title="RLCT (LLC estimate)", scale=_y),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("mean:Q", format=".2f"),
                alt.Tooltip("ci:Q", title="+/- CI", format=".2f"),
            ],
        )
    )
    _err = (
        alt.Chart(CIFAR_FIG3B)
        .mark_rule(strokeWidth=1.5, color="#333")
        .encode(
            x=alt.X("curve:N", sort=_domain),
            y=alt.Y("lo:Q", scale=_y),
            y2="hi:Q",
        )
    )
    _paper_df = pd.DataFrame(
        {
            "curve": [f"AW={_a:g}" for _a in CIFAR_AWS],
            "paper": [CIFAR_PAPER_3B[_a] for _a in CIFAR_AWS],
        }
    )
    _paper = (
        alt.Chart(_paper_df)
        .mark_point(
            shape="diamond",
            filled=False,
            size=110,
            strokeWidth=2,
        )
        .encode(
            x=alt.X("curve:N", sort=_domain),
            y=alt.Y("paper:Q", scale=_y),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".0f"),
            ],
        )
    )
    _chart = (_bars + _err + _paper).properties(
        width=560,
        height=330,
    )

    _means = {
        _aw: float(
            CIFAR_FIG3B[CIFAR_FIG3B["aw"] == _aw]["mean"].iloc[0]
        )
        for _aw in CIFAR_AWS
    }
    _spread = max(_means.values()) - min(_means.values())
    _ratio = float(
        np.mean(
            [CIFAR_PAPER_3B[_aw] / _means[_aw] for _aw in CIFAR_AWS]
        )
    )
    _n = int(CIFAR_FIG3B["n"].iloc[0])
    _prose = mo.md(f"""
    Bars are our mean LLC estimate, whiskers are the 95% CI over
    {_n} seeds, and hollow diamonds are the paper's read-offs. Our
    estimates are flat: {_means[0.0]:.1f} at AW = 0,
    {_means[0.5]:.1f} at AW = 0.5, {_means[1.0]:.1f} at AW = 1,
    {_means[2.0]:.1f} at AW = 2. The spread between the largest and
    smallest is {_spread:.2f}, comparable to the individual CIs,
    so the curves overlap and there is no visible self-modeling
    effect.

    The paper reads a monotone fall from 82 to 66, about 20%. Our
    whole scale also sits roughly **{_ratio:.1f}x** below the
    paper's. The same estimator matched the paper on MNIST, so the
    gap more likely comes from the model or training setup than from
    the estimator itself.
    """)
    mo.vstack([mo.ui.altair_chart(_chart), _prose])
    return


@app.cell
def _(
    CIFAR_AWS,
    CIFAR_COLORS,
    CIFAR_FIG3C,
    CIFAR_PAPER_3C,
    alt,
    mo,
    pd,
):
    _domain = [f"AW={_a:g}" for _a in CIFAR_AWS]
    _scale = alt.Scale(
        domain=_domain,
        range=[CIFAR_COLORS[_a] for _a in CIFAR_AWS],
    )
    _y = alt.Scale(domain=[0.90, 0.94])
    _bars = (
        alt.Chart(CIFAR_FIG3C)
        .mark_bar(size=42)
        .encode(
            x=alt.X("curve:N", title=None, sort=_domain),
            y=alt.Y("acc:Q", title="Test accuracy", scale=_y),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("acc:Q", format=".4f"),
                alt.Tooltip("half:Q", title="+/- CI", format=".4f"),
            ],
        )
    )
    _err = (
        alt.Chart(CIFAR_FIG3C)
        .mark_rule(strokeWidth=1.5, color="#333")
        .encode(
            x=alt.X("curve:N", sort=_domain),
            y=alt.Y("lo:Q", scale=_y),
            y2="hi:Q",
        )
    )
    _paper_df = pd.DataFrame(
        {
            "paper": [sum(CIFAR_PAPER_3C) / len(CIFAR_PAPER_3C)],
            "label": ["paper (0.918-0.920)"],
        }
    )
    _paper = (
        alt.Chart(_paper_df)
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.6, color="#888")
        .encode(y=alt.Y("paper:Q", scale=_y))
    )
    _paper_label = (
        alt.Chart(_paper_df)
        .mark_text(align="left", dx=4, dy=-5, color="#888")
        .encode(
            y=alt.Y("paper:Q", scale=_y),
            text="label:N",
        )
    )
    _chart = (_bars + _err + _paper + _paper_label).properties(
        width=560,
        height=300,
    )

    _lo = float(CIFAR_FIG3C["acc"].min())
    _hi = float(CIFAR_FIG3C["acc"].max())
    _prose = mo.md(f"""
    Accuracy is essentially unchanged by self-modeling: {_lo:.1%} to
    {_hi:.1%} across AW, against the paper's 0.918-0.920. The
    dashed line is the middle of the paper's range. On this panel
    the reproduction lands, which is what makes the 3A and 3B
    mismatches worth a second look.
    """)
    mo.vstack([mo.ui.altair_chart(_chart), _prose])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why does CIFAR go the other way?

    Figure 3A and Figure 3B disagree with the paper, but the probes
    in `cifar_fig3_probes.json` tell a consistent story: the network
    is not getting simpler, it is re-parameterizing. The table
    compares four variants at AW = 0 and AW = 2 (means over two
    seeds, 50 epochs). `hidden |a|` is the mean absolute hidden
    activation on test images; `classifier |W|` is the L2 norm of
    the classifier rows. Both are reported first -> last logged
    epoch.
    """)
    return


@app.cell
def _(CIFAR_PROBES, mo):
    _order = ["base", "nodetach", "hidden", "b64"]
    _labels = {
        "base": "base (detached target)",
        "nodetach": "no detach",
        "hidden": "hidden-only target",
        "b64": "batch 64",
    }
    _lines = [
        (
            "| variant | AW | SD @ 50 | hidden \\|a\\| first -> last "
            "| classifier \\|W\\| first -> last |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _variant in _order:
        for _aw in (0.0, 2.0):
            _row = CIFAR_PROBES[
                (CIFAR_PROBES["variant"] == _variant)
                & (CIFAR_PROBES["aw"] == _aw)
            ].iloc[0]
            _hfirst, _hlast = _row["hidden"]
            _wfirst, _wlast = _row["cls_wnorm"]
            _lines.append(
                f"| {_labels[_variant]} | {_aw:g} "
                f"| {_row['sd_e50']:.4f} "
                f"| {_hfirst:.3f} -> {_hlast:.3f} "
                f"| {_wfirst:.2f} -> {_wlast:.2f} |"
            )
    mo.md("\n".join(_lines))
    return


@app.cell
def _(CIFAR_PROBES, mo):
    def _probe(_variant, _aw):
        _row = CIFAR_PROBES[
            (CIFAR_PROBES["variant"] == _variant)
            & (CIFAR_PROBES["aw"] == _aw)
        ].iloc[0]
        return _row

    _b0 = _probe("base", 0.0)
    _b2 = _probe("base", 2.0)
    _n0 = _probe("nodetach", 0.0)
    _n2 = _probe("nodetach", 2.0)
    _h0 = _probe("hidden", 0.0)
    _h2 = _probe("hidden", 2.0)
    _b640 = _probe("b64", 0.0)
    _b642 = _probe("b64", 2.0)

    mo.md(f"""
    **What is happening.** Under self-modeling the network has a
    reason to make its hidden activations predictable, and the
    easiest way is to make them small. In the base run the hidden
    magnitude at AW = 2 starts at {_b2['hidden'][0]:.3f} versus
    {_b0['hidden'][0]:.3f} at AW = 0. Smaller activations force
    larger classifier weights to keep the logits at the same scale,
    so the weight norm grows further at AW = 2
    ({_b2['cls_wnorm'][1]:.2f}) than at AW = 0
    ({_b0['cls_wnorm'][1]:.2f}). Weight spread is not
    scale-invariant: shrink the input and the weights must grow.
    That is exactly why the paper also reports RLCT, which is
    invariant to this rescaling. Our RLCT (3B) is flat, so the
    widening in 3A is a units artifact, not extra complexity.

    **What the probes ruled out.**

    - *Detach.* Removing the stop-gradient (no detach) leaves the
      inversion in place: SD {_n0['sd_e50']:.4f} at AW = 0 and
      {_n2['sd_e50']:.4f} at AW = 2, versus {_b0['sd_e50']:.4f}
      and {_b2['sd_e50']:.4f} for base. Gradient flow through the
      target copy is not the cause.
    - *Batch size.* Batch 64 changes the level a lot
      ({_b640['sd_e50']:.4f} at AW = 0, {_b642['sd_e50']:.4f} at
      AW = 2) but keeps the same ordering, so the inversion is not
      a batch-size artifact.
    - *Hidden-only target.* Making the target the hidden layer
      alone nearly flattens the curves:
      {_h0['sd_e50']:.4f} at AW = 0 and {_h2['sd_e50']:.4f} at
      AW = 2. This is the probe that comes closest to removing the
      inversion.

    **What we could not test.** The paper describes skip
    connections from block 3 into the hidden layer. We could not
    map that onto a standard ResNet-18, so our hidden layer sits
    after global pooling. That architectural difference is a real
    candidate for the gap, but we have not isolated it.
    """)
    return


@app.cell
def _(mo):
    cifar_scale_s = mo.ui.slider(
        start=0.25,
        stop=2.0,
        step=0.05,
        value=1.0,
        label="Activation scale s (toy)",
    )
    cifar_scale_s
    return (cifar_scale_s,)


@app.cell
def _(cifar_scale_s, mo, np):
    _rng = np.random.default_rng(0)
    _a = _rng.normal(size=512)
    _logits = np.random.default_rng(1).normal(size=10)
    _denom = float(_a @ _a)
    _w = np.outer(_logits, _a) / _denom
    _base_sd = float(_w.std())
    _s = float(cifar_scale_s.value)
    _grid = [0.25, 0.5, 1.0, 1.5, 2.0]
    _rows = "\n".join(
        f"| {_g:.2f} | {1.0 / _g:.2f} | {_base_sd / _g:.4f} |"
        for _g in _grid
    )
    mo.md(f"""
    A fixed logit target fixes the product of classifier weights and
    activations. If the activations shrink by a factor $s$, the
    weights must grow by $1/s$ to keep the logits the same. Weight
    SD is proportional to weight scale, so it scales as $1/s$ too.

    With a random toy target ($W a =$ fixed logits, base weight SD
    {_base_sd:.4f}), at $s = {_s:.2f}$ the needed weight scale is
    **{1.0 / _s:.2f}** and the weight SD is
    **{_base_sd / _s:.4f}**.

    | activation scale s | weight scale 1/s | weight SD |
    | ---: | ---: | ---: |
    {_rows}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## IMDB (Figure 4)

    The third experiment is sentiment classification on IMDB with an
    embedding bag and a hidden layer. Figure 4A plots classifier
    weight SD, 4B is RLCT, and 4C is accuracy. We ran the training
    panels and estimated 4B's RLCT. The main run is batch 64 with
    per-model gradient clipping, the setting that matches the
    paper's batch size; it is plotted by default below.
    """)
    return


@app.cell
def _(json, mo):
    _main = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "imdb_fig4.json"
        ).read_text()
    )
    _prefix = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "imdb_fig4_variants.json"
        ).read_text()
    )["variants"]["b64_clip"]["results"]
    _rlct = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "imdb_fig4b.json"
        ).read_text()
    )

    def _sd_acc(_results):
        _sd = [
            float(_results[f"{_aw:g}"]["sd_mean"][-1])
            for _aw in (0.0, 100.0, 500.0)
        ]
        _acc = [
            float(_results[f"{_aw:g}"]["acc_mean"])
            for _aw in (0.0, 100.0, 500.0)
        ]
        return _sd, _acc

    def _fmt(_values):
        return " / ".join(f"{_v:.3f}" for _v in _values)

    _sd_before, _acc_before = _sd_acc(_prefix)
    _sd_after, _acc_after = _sd_acc(_main["results"])
    _rlct_before = [
        float(_rlct["superseded_global_clip_run"]["rlct"][_k]["mean"])
        for _k in ("0", "100", "500")
    ]
    _rlct_after = [
        float(_rlct["rlct"][_k]["mean"])
        for _k in ("0", "100", "500")
    ]
    mo.md(f"""
    /// admonition | Bug we found: coupled gradient clipping
    Our first IMDB runs clipped gradients with one global norm
    across all stacked models, which coupled otherwise-independent
    runs and over-clipped each by about $\\sqrt{{M}}$. A
    frozen-head control caught it: AW = 0 lost accuracy even though
    its self-model term carries zero weight, because the diverging
    frozen models inflated the shared norm. Clipping is now
    per-model, and the conclusions did not change. Between the
    superseded and corrected runs the final SD moved from
    {_fmt(_sd_before)} to {_fmt(_sd_after)}, accuracy from
    {_acc_before[0]:.1%} to {_acc_after[0]:.1%}, and RLCT from
    {_fmt(_rlct_before)} to {_fmt(_rlct_after)}. Every quantity
    stays flat across AW and far below the paper's scale.
    ///
    """)
    return


@app.cell
def _(json, mo, pd):
    _main = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "imdb_fig4.json"
        ).read_text()
    )
    _raw = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "imdb_fig4_variants.json"
        ).read_text()
    )

    IMDB_AWS = [0.0, 100.0, 500.0]
    IMDB_COLORS = {
        0.0: "#000000",
        100.0: "#00e000",
        500.0: "#0000ff",
    }
    IMDB_PAPER_4A = {0.0: 2.05, 100.0: 1.65, 500.0: 0.75}
    IMDB_OPTIONS = {
        "main: batch 64, per-model clip (corrected)": "main",
        "pre-fix: batch 512, grad clip": "b512_clip",
        "pre-fix: batch 64, grad clip": "b64_clip",
        "pre-fix: batch 64, no grad clip (unaffected)": "b64_noclip",
        "pre-fix: batch 512, no detach": "b512_nodetach",
    }
    IMDB_LABELS = {
        _value: _key for _key, _value in IMDB_OPTIONS.items()
    }
    _runs = {"main": _main["results"]}
    _runs.update(
        {
            _key: _variant["results"]
            for _key, _variant in _raw["variants"].items()
        }
    )

    _rows = []
    _summary = []
    for _key, _results in _runs.items():
        for _aw in IMDB_AWS:
            _entry = _results[f"{_aw:g}"]
            for _epoch, (_mean, _half) in enumerate(
                zip(_entry["sd_mean"], _entry["sd_ci"])
            ):
                _rows.append(
                    {
                        "variant": _key,
                        "aw": _aw,
                        "curve": f"AW={_aw:g}",
                        "epoch": _epoch,
                        "mean": float(_mean),
                        "half": float(_half),
                    }
                )
            _summary.append(
                {
                    "variant": _key,
                    "aw": _aw,
                    "final_sd": float(_entry["sd_mean"][-1]),
                    "acc": float(_entry["acc_mean"]),
                    "n": int(_entry["n"]),
                    "n_diverged": int(_entry["n_diverged"]),
                }
            )
    IMDB_FIG4A = pd.DataFrame(_rows)
    IMDB_FIG4A["lo"] = IMDB_FIG4A["mean"] - IMDB_FIG4A["half"]
    IMDB_FIG4A["hi"] = IMDB_FIG4A["mean"] + IMDB_FIG4A["half"]
    IMDB_SUMMARY = pd.DataFrame(_summary)
    return (
        IMDB_AWS,
        IMDB_COLORS,
        IMDB_FIG4A,
        IMDB_LABELS,
        IMDB_OPTIONS,
        IMDB_PAPER_4A,
        IMDB_SUMMARY,
    )


@app.cell
def _(IMDB_OPTIONS, mo):
    imdb_variant = mo.ui.dropdown(
        options=IMDB_OPTIONS,
        value=next(iter(IMDB_OPTIONS)),
        label=(
            "IMDB run variant (non-main entries were trained "
            "before the clipping fix; b64_noclip unaffected)"
        ),
    )
    imdb_variant
    return (imdb_variant,)


@app.cell
def _(
    IMDB_AWS,
    IMDB_COLORS,
    IMDB_FIG4A,
    IMDB_LABELS,
    IMDB_PAPER_4A,
    alt,
    imdb_variant,
    mo,
    pd,
):
    _df = IMDB_FIG4A[
        IMDB_FIG4A["variant"] == imdb_variant.value
    ].copy()
    _domain = [f"AW={_a:g}" for _a in IMDB_AWS]
    _scale = alt.Scale(
        domain=_domain,
        range=[IMDB_COLORS[_a] for _a in IMDB_AWS],
    )
    _paper_df = pd.DataFrame(
        {
            "curve": [f"AW={_a:g}" for _a in IMDB_AWS],
            "epoch": [500] * len(IMDB_AWS),
            "paper": [IMDB_PAPER_4A[_a] for _a in IMDB_AWS],
        }
    )
    _y_hi = max(
        float(_df["hi"].max()),
        max(IMDB_PAPER_4A.values()),
    ) * 1.1
    _y = alt.Scale(domain=[0.0, _y_hi], zero=True)

    _band = (
        alt.Chart(_df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y(
                "lo:Q",
                title="SD of classifier weights",
                scale=_y,
            ),
            y2="hi:Q",
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
        )
    )
    _line = (
        alt.Chart(_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y("mean:Q", scale=_y),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("mean:Q", format=".4f"),
            ],
        )
    )
    _points = (
        alt.Chart(_paper_df)
        .mark_point(
            shape="diamond",
            filled=False,
            size=110,
            strokeWidth=2,
        )
        .encode(
            x=alt.X("epoch:Q"),
            y=alt.Y("paper:Q", scale=_y),
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("curve:N"),
                alt.Tooltip("paper:Q", title="paper", format=".2f"),
            ],
        )
    )
    _chart = (_band + _line + _points).properties(
        width=600,
        height=340,
        title=(
            "IMDB SD vs epoch, "
            f"{IMDB_LABELS[imdb_variant.value]}"
        ),
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(IMDB_SUMMARY, mo):
    _lines = [
        "| variant | AW | final SD | accuracy | diverged |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _row in IMDB_SUMMARY.to_dict("records"):
        _lines.append(
            f"| {_row['variant']} | {_row['aw']:g} "
            f"| {_row['final_sd']:.4f} | {_row['acc']:.4f} "
            f"| {_row['n_diverged']}/{_row['n']} |"
        )
    mo.md("\n".join(_lines))
    return


@app.cell
def _(IMDB_PAPER_4A, IMDB_SUMMARY, mo):
    _base = IMDB_SUMMARY[
        (IMDB_SUMMARY["variant"] == "main")
        & (IMDB_SUMMARY["aw"] == 0.0)
    ].iloc[0]
    _gap = IMDB_PAPER_4A[0.0] / float(_base["final_sd"])
    _healthy = IMDB_SUMMARY[IMDB_SUMMARY["n_diverged"] == 0]
    _spreads = [
        float(
            _healthy[_healthy["variant"] == _v]["final_sd"].max()
            - _healthy[_healthy["variant"] == _v]["final_sd"].min()
        )
        for _v in _healthy["variant"].unique()
    ]
    _worst = max(_spreads)
    _div = int(IMDB_SUMMARY["n_diverged"].sum())
    _intended = int(IMDB_SUMMARY["n"].max()) * len(IMDB_SUMMARY)
    mo.md(f"""
    The main run reproduces the paper's accuracy: **{_base['acc']:.1%}**
    at batch 64, a little above the paper's 0.819-0.836. But no
    variant separates the AWs. Across the healthy cells the final SD
    spans at most **{_worst:.3f}**, and every curve overlaps every
    other one. {_div} of {_intended} runs diverged, all of them the
    no-clip batch-64 variant at AW = 500.

    The weight SD also sits about **{_gap:.0f}x** below the paper's
    epoch-500 read-offs: baseline {IMDB_PAPER_4A[0.0]:.2f} in the
    paper versus {_base['final_sd']:.3f} here. Whatever the paper's
    self-modeling term does to IMDB, our runs show almost none of
    it.

    **Hypothesis.** Our output layer predicts the hidden
    activations from those same activations. If the self-model rows
    can learn to map their input to itself, they can drive the
    self-model loss to near zero, and then the self-model term
    stops pushing on the rest of the network. With SGD at learning
    rate 0.1 and AW up to 500, that shortcut is found quickly. This
    would explain both failures at once: no AW separation and no
    movement in the classifier weights. The diagnostics in the
    next section support this, though on their own they do not
    prove it.
    """)
    return


@app.cell
def _(json, mo, pd):
    _path = (
        mo.notebook_location()
        / "public"
        / "data"
        / "imdb_fig4b.json"
    )
    _raw = json.loads(_path.read_text())

    IMDB_PAPER_4B = {0.0: 13.7, 100.0: 11.2, 500.0: 5.2}
    _rows = []
    for _entry in _raw["rlct"].values():
        _aw = float(_entry["aw"])
        _rows.append(
            {
                "aw": _aw,
                "mean": float(_entry["mean"]),
                "half": float(_entry["ci"]),
                "paper": IMDB_PAPER_4B[_aw],
            }
        )
    IMDB_FIG4B = pd.DataFrame(_rows).sort_values("aw")
    IMDB_FIG4B["lo"] = IMDB_FIG4B["mean"] - IMDB_FIG4B["half"]
    IMDB_FIG4B["hi"] = IMDB_FIG4B["mean"] + IMDB_FIG4B["half"]
    IMDB_FIG4B["curve"] = IMDB_FIG4B["aw"].map(
        lambda _a: f"AW={_a:g}"
    )
    return IMDB_FIG4B, IMDB_PAPER_4B


@app.cell
def _(IMDB_AWS, IMDB_COLORS, IMDB_FIG4B, alt, mo):
    _domain = [f"AW={_a:g}" for _a in IMDB_AWS]
    _scale = alt.Scale(
        domain=_domain,
        range=[IMDB_COLORS[_a] for _a in IMDB_AWS],
    )
    _y = alt.Scale(zero=True)
    _bars = (
        alt.Chart(IMDB_FIG4B)
        .mark_bar(size=64, opacity=0.85)
        .encode(
            x=alt.X("curve:N", title="AW", sort=_domain),
            y=alt.Y("mean:Q", title="RLCT (LLC)", scale=_y),
            color=alt.Color("curve:N", scale=_scale, legend=None),
            tooltip=[
                alt.Tooltip("curve:N", title="AW"),
                alt.Tooltip("mean:Q", title="ours", format=".3f"),
                alt.Tooltip("half:Q", title="95% CI", format=".3f"),
            ],
        )
    )
    _ci = (
        alt.Chart(IMDB_FIG4B)
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("curve:N", sort=_domain),
            y=alt.Y("lo:Q", scale=_y),
            y2="hi:Q",
            color=alt.Color("curve:N", scale=_scale, legend=None),
        )
    )
    _paper = (
        alt.Chart(IMDB_FIG4B)
        .mark_point(
            shape="diamond",
            filled=False,
            size=130,
            strokeWidth=2,
        )
        .encode(
            x=alt.X("curve:N", sort=_domain),
            y=alt.Y("paper:Q", scale=_y),
            color=alt.Color(
                "curve:N",
                scale=_scale,
                legend=alt.Legend(title="AW"),
            ),
            tooltip=[
                alt.Tooltip("curve:N", title="AW"),
                alt.Tooltip("paper:Q", title="paper", format=".1f"),
            ],
        )
    )
    _chart = (_bars + _ci + _paper).properties(
        width=460,
        height=320,
        title="IMDB RLCT by AW: ours (bars) vs paper (diamonds)",
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(IMDB_FIG4B, mo):
    _means = {
        float(_r["aw"]): float(_r["mean"])
        for _r in IMDB_FIG4B.to_dict("records")
    }
    _spread = max(_means.values()) - min(_means.values())
    _ratio = 13.7 / _means[0.0]
    mo.md(f"""
    ### Figure 4B: RLCT

    The RLCT is flat across AW: **{_means[0.0]:.2f}**,
    **{_means[100.0]:.2f}**, and **{_means[500.0]:.2f}** (spread
    {_spread:.2f}), about **{_ratio:.0f}x** below the paper's
    baseline of 13.7 and nowhere near its 13.7 -> 5.2 slope. This
    matches the flat weight SD above and is consistent with the
    identity-map picture supported by the diagnostics below. The
    estimator
    follows the paper's supplement settings (devinterp 2.0.1,
    lr 1e-3, localization 10, nbeta = batch/log(batch) at batch 64,
    4 chains x 2000 draws), recorded in `imdb_fig4b.json`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why doesn't IMDB feel the pressure?

    The identity-map picture says the self-model term should keep
    pushing on the network until the self-model rows can predict
    the hidden activations, at which point the pressure disappears.
    We logged the self-model loss, the distance of the self-model
    rows from the identity, and the self-model share of the
    hidden-layer gradient on shorter diagnostic runs (3 seeds; IMDB
    100 epochs, MNIST 50) that mirror the main setups.
    """)
    return


@app.cell
def _(json, mo, pd):
    IDENTITY = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "identity_diagnostics.json"
        ).read_text()
    )
    _rows = []
    for _aw in (100.0, 500.0):
        _share = IDENTITY["imdb_normal"]["results"][
            f"{_aw:g}"
        ]["identity"]["sm_grad_share"]["mean"]
        for _epoch, _value in enumerate(_share, start=1):
            _rows.append(
                {
                    "series": f"IMDB AW={_aw:g}",
                    "source": "IMDB (100 epochs)",
                    "epoch": _epoch,
                    "share": float(_value),
                }
            )
    for _aw in (10.0, 50.0):
        _share = IDENTITY["mnist_h512"]["results"]["512"][
            "aws"
        ][f"{_aw:g}"]["identity"]["sm_grad_share"]["mean"]
        for _epoch, _value in enumerate(_share, start=1):
            _rows.append(
                {
                    "series": f"MNIST AW={_aw:g}",
                    "source": "MNIST (50 epochs)",
                    "epoch": _epoch,
                    "share": float(_value),
                }
            )
    IDENTITY_SHARE = pd.DataFrame(_rows)
    return IDENTITY, IDENTITY_SHARE


@app.cell
def _(IDENTITY_SHARE, alt, mo):
    _order = [
        "IMDB AW=100",
        "IMDB AW=500",
        "MNIST AW=10",
        "MNIST AW=50",
    ]
    _scale = alt.Scale(
        domain=_order,
        range=["#00e000", "#0000ff", "#f58518", "#d62728"],
    )
    _line = (
        alt.Chart(IDENTITY_SHARE)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "epoch:Q",
                title="Epoch",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "share:Q",
                title=(
                    "Self-model share of the "
                    "hidden-layer gradient"
                ),
                scale=alt.Scale(domain=[0.0, 1.0]),
            ),
            color=alt.Color(
                "series:N",
                scale=_scale,
                legend=alt.Legend(
                    title=None, orient="bottom", columns=2
                ),
            ),
            tooltip=[
                alt.Tooltip("series:N"),
                alt.Tooltip("epoch:Q"),
                alt.Tooltip("share:Q", format=".3f"),
            ],
        )
        .properties(
            width=600,
            height=340,
            title="Self-model pressure over training",
        )
    )
    mo.ui.altair_chart(_line)
    return


@app.cell
def _(IDENTITY, mo):
    _imdb = IDENTITY["imdb_normal"]["results"]
    _mnist = IDENTITY["mnist_h512"]["results"]["512"]["aws"]
    _frozen = IDENTITY["imdb_frozen"]["results"]

    _sm100_5 = float(_imdb["100"]["identity"]["sm_loss"]["mean"][4])
    _sm500_5 = float(_imdb["500"]["identity"]["sm_loss"]["mean"][4])
    _share100_20 = float(
        _imdb["100"]["identity"]["sm_grad_share"]["mean"][19]
    )
    _share500_20 = float(
        _imdb["500"]["identity"]["sm_grad_share"]["mean"][19]
    )
    _id100 = float(_imdb["100"]["identity"]["id_dist"]["mean"][-1])
    _id500 = float(_imdb["500"]["identity"]["id_dist"]["mean"][-1])
    _mn10_50 = float(
        _mnist["10"]["identity"]["sm_grad_share"]["mean"][-1]
    )
    _mn50_50 = float(
        _mnist["50"]["identity"]["sm_grad_share"]["mean"][-1]
    )
    _fz_loss = max(
        float(_frozen["100"]["identity"]["sm_loss"]["mean"][-1]),
        float(_frozen["500"]["identity"]["sm_loss"]["mean"][-1]),
    )
    _fz_acc = float(_frozen["100"]["acc_mean"])

    mo.md(f"""
    On IMDB the self-model loss collapses almost immediately,
    **{_sm100_5:.1e}** at AW 100 and **{_sm500_5:.1e}** at AW 500
    by epoch 5, and the self-model share of the hidden-layer
    gradient falls to **{_share100_20:.0%}** and
    **{_share500_20:.0%}** by epoch 20. On MNIST the share is still
    **{_mn10_50:.0%}** at AW 10 and **{_mn50_50:.0%}** at AW 50 at
    the end of its 50-epoch run. The pressure
    persists where the effect appears and evaporates where it does
    not.

    Why is IMDB so easy? The hidden layer is a linear map of a
    64-dimensional embedding, so its 128 activations live in a
    64-dimensional subspace; many linear maps reproduce them
    exactly, and the self-model rows can drive the loss to zero
    without becoming the identity. They do not: the relative
    distance $\\lVert W_s - I\\rVert_F / \\lVert I\\rVert_F$ is still
    **{_id100:.2f}** at AW 100 and **{_id500:.2f}** at AW 500. The
    shortcut is a projection, not a copy.

    We tried to make the test causal by freezing the self-model
    rows at their random init. Both AW 100 and AW 500 ran away
    (self-model loss to about **{_fz_loss:.0e}**, accuracy
    **{_fz_acc:.0%}**), because a fixed random head with a
    detached, moving target has no stable solution. That control
    is therefore inconclusive: it breaks the run rather than
    holding the shortcut fixed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Slowing the shortcut down

    If the effect is missing on IMDB because the self-model rows
    learn their shortcut and the pressure dies before it can act,
    then slowing those rows down should bring it back. We scaled
    the self-model rows' gradient by 0.01 and 0.001 after
    per-model clipping and re-ran AW 0/100/500 (3 seeds, 100
    epochs); nothing else changed.
    """)
    return


@app.cell
def _(json, mo):
    _runs = json.loads(
        (
            mo.notebook_location()
            / "public"
            / "data"
            / "slowhead_control.json"
        ).read_text()
    )["runs"]
    _scales = (
        ("scale1", "1 (normal)"),
        ("scale0.01", "0.01"),
        ("scale0.001", "0.001"),
    )
    _aws = (0.0, 100.0, 500.0)
    _stat = {}
    _lines = [
        (
            "| self-model LR scale | AW | epoch-100 SD (± CI) | "
            "accuracy | epoch-100 self-model share |"
        ),
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for _scale, _label in _scales:
        for _aw in _aws:
            _entry = _runs[_scale]["results"][f"{_aw:g}"]
            _sd = float(_entry["sd_mean"][100])
            _half = float(_entry["sd_ci"][100])
            _acc = float(_entry["acc_mean"])
            _raw = _entry["identity"]["sm_grad_share"]["mean"][99]
            _share = None if _raw is None else float(_raw)
            _stat[(_scale, _aw)] = {
                "sd": _sd,
                "half": _half,
                "acc": _acc,
                "share": _share,
            }
            _share_text = "—" if _share is None else f"{_share:.1%}"
            _lines.append(
                f"| {_label} | {_aw:g} | {_sd:.3f} ± {_half:.3f} "
                f"| {_acc:.1%} | {_share_text} |"
            )
    _table = "\n".join(_lines)

    def _effect(_scale, _aw):
        _base = _stat[(_scale, 0.0)]
        _v = _stat[(_scale, _aw)]
        _pct = (_v["sd"] - _base["sd"]) / _base["sd"] * 100.0
        _b_lo = _base["sd"] - _base["half"]
        _b_hi = _base["sd"] + _base["half"]
        _lo = _v["sd"] - _v["half"]
        _hi = _v["sd"] + _v["half"]
        _overlap = not (_hi < _b_lo or _lo > _b_hi)
        return _pct, _overlap

    _pct100, _ov100 = _effect("scale0.01", 100.0)
    _pct500, _ov500 = _effect("scale0.01", 500.0)
    _pct100n, _ov100n = _effect("scale1", 100.0)
    _pct500n, _ov500n = _effect("scale1", 500.0)
    _ci100 = "overlaps" if _ov100 else "does not overlap"
    _ci500 = "overlaps" if _ov500 else "does not overlap"
    _normal_ci = (
        "both intervals overlap the baseline"
        if _ov100n and _ov500n
        else "the intervals behave as shown in the table"
    )
    _share100 = _stat[("scale0.01", 100.0)]["share"]
    _share500 = _stat[("scale0.01", 500.0)]["share"]
    _acc_small100 = _stat[("scale0.001", 100.0)]["acc"]
    _acc_small500 = _stat[("scale0.001", 500.0)]["acc"]

    mo.md(f"""
    {_table}

    At self-model scale **0.01** the pressure survives. The
    epoch-100 self-model share is **{_share100:.0%}** at AW 100
    and **{_share500:.0%}** at AW 500, the same range as MNIST.
    With that pressure still present, AW 100 narrows the epoch-100
    SD by **{_pct100:+.1f}%** and AW 500 by **{_pct500:+.1f}%**
    against the AW 0 baseline; the AW 100 interval {_ci100} the
    baseline and the AW 500 interval {_ci500} it. In the normal
    run (scale 1) the same changes are only **{_pct100n:+.1f}%**
    and **{_pct500n:+.1f}%**, and {_normal_ci}. Changing only how
    fast the shortcut can be learned switches the effect on, which
    is causal support for the mechanism: the narrowing tracks the
    pressure, not the run configuration.

    At self-model scale **0.001** the pressure instead swamps
    classification. Accuracy falls to **{_acc_small100:.0%}** and
    **{_acc_small500:.0%}**, near the 50% chance line, like the
    small-network collapse on MNIST, so the classifier never forms
    and those SDs are not interpretable. The control also has
    caveats: 3 seeds and 100 epochs; accuracy drops as the
    self-model rows slow, where the paper's AW 500 is slightly
    *better* than baseline; the effect is not monotone in AW (AW
    100 narrows more than AW 500); and it is smaller than the
    paper's. This is a modified setup that explains our null
    result, not a reproduction of the paper's numbers.

    Taken with the gradient-share evidence above, the control adds
    causal evidence for the mechanism, from a small modified
    experiment: on IMDB the self-model pressure dies before it can
    act, and when it is held open the classifier narrows as
    predicted. More seeds and longer runs would firm this up.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scoreboard

    | paper panel | reproduced? | note |
    | --- | --- | --- |
    | 2A | yes | MNIST SD ordering matches within ~0.001 at epoch 50 (variant a). |
    | 2B | yes | Hidden-size sweep and small-net collapse match. |
    | 2C | yes | RLCT ordering matches at localization 100, within a few percent. |
    | 2D | yes | Accuracy collapse of small nets at large AW reproduced. |
    | 3A | no | SD ordering inverted: baseline is the narrowest curve. |
    | 3B | no | RLCT flat across AW and about 3x below the paper's scale. |
    | 3C | yes | Accuracy 0.921-0.923, flat across AW, matching 0.918-0.920. |
    | 4A | no | Corrected run: SD 0.232 at every AW, ~9x below the paper's 2.05; no AW separation. |
    | 4B | no | Corrected run: RLCT flat across AW (2.10/2.13/2.13 vs paper 13.7 -> 5.2). |
    | 4C | partly | Corrected accuracy 85.2%, near the paper's range but flat across AW. |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    /// admonition | Coming soon: Exercise
    What if you forget `.detach()`? A short interactive exercise:
    predict what changes, then check. Placeholder for now.
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
