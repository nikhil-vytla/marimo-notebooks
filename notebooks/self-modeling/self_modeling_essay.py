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
    moves. This notebook walks through the idea and our reproduction
    of the paper's Figure 2a on MNIST.
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
    mo.md("""
    /// admonition | Coming soon: RLCT
    A better ruler for complexity. The paper measures the model with
    the real log canonical threshold, not just weight spread. This
    placeholder will explain and plot it.
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    /// admonition | Coming soon: CIFAR-10 and IMDB
    The same auxiliary objective on a ResNet-18 for CIFAR-10 and an
    embedding bag for IMDB. Placeholder until those runs are in.
    ///
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
