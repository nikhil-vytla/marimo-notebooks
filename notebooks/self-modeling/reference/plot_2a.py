#!/usr/bin/env python
"""Reproduce Figure 2A of the self-modeling paper (MNIST, hidden=512).

Reads per-run JSONs written by train.py, aggregates seeds into mean +/- 95% CI
curves of the standard deviation of the weight distribution, and saves a figure.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AWS = [0, 1, 5, 10, 20, 50]
EPOCHS = 50

COLORS = {
    0: "#000000",
    1: "#ff1493",
    5: "#00e000",
    10: "#8a2be2",
    20: "#ff4500",
    50: "#00bfff",
}
LABELS = {
    0: "Baseline",
    1: "AW=1.0",
    5: "AW=5.0",
    10: "AW=10.0",
    20: "AW=20.0",
    50: "AW=50.0",
}
TITLES = {
    "a": "a: rescale to [-1,1]",
    "b": "b: rescale + center",
}

# Two-sided 95% (0.975) Student-t critical values, df 1..30.
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_crit(n):
    """Two-sided 95% critical t value for df = n - 1."""
    return T975.get(n - 1, 1.96)


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def load_setting(root, variant, aw, epochs):
    """Load all seed runs for one (variant, aw).

    Returns (matrix, accs) where matrix has shape (n, epochs) and accs is the
    list of final test accuracies. Returns None when no complete run exists.
    """
    pattern = str(Path(root) / variant / f"aw{aw:g}_h512_s*.json")
    files = sorted(glob.glob(pattern))
    sds = []
    accs = []
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warn(f"skipping unreadable {path}: {exc}")
            continue
        sd = data.get("sd")
        if not isinstance(sd, list) or len(sd) < epochs + 1:
            have = len(sd) if isinstance(sd, list) else "none"
            warn(f"skipping {path}: need {epochs + 1} sd values, got {have}")
            continue
        sds.append([float(v) for v in sd[1 : epochs + 1]])
        acc = data.get("test_acc")
        if isinstance(acc, list) and acc:
            accs.append(float(acc[-1]))
    if not sds:
        return None
    return np.asarray(sds, dtype=float), accs


def compute_curve(sd_mat):
    """Mean curve and 95% CI half-width for a (n, epochs) seed matrix."""
    n = sd_mat.shape[0]
    mean = sd_mat.mean(axis=0)
    if n > 1:
        std = sd_mat.std(axis=0, ddof=1)
        half = t_crit(n) * std / np.sqrt(n)
    else:
        half = None
    return mean, half


def draw_panel(ax, root, variant, epochs, show_legend):
    rows = []
    x = np.arange(1, epochs + 1)
    for aw in AWS:
        loaded = load_setting(root, variant, aw, epochs)
        if loaded is None:
            rows.append({"variant": variant, "aw": aw, "n": 0})
            continue
        sd_mat, accs = loaded
        n = sd_mat.shape[0]
        mean, half = compute_curve(sd_mat)
        color = COLORS[aw]
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.0,
            label=LABELS[aw],
            zorder=3,
        )
        if half is not None:
            ax.fill_between(
                x,
                mean - half,
                mean + half,
                color=color,
                alpha=0.3,
                linewidth=0,
                zorder=2,
            )
        rows.append(
            {
                "variant": variant,
                "aw": aw,
                "n": n,
                "mean50": float(mean[-1]),
                "ci50": float(half[-1]) if half is not None else 0.0,
                "acc": float(np.mean(accs)) if accs else float("nan"),
            }
        )
    if show_legend and any(r["n"] > 0 for r in rows):
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    return rows


def print_table(rows):
    header = (
        f"{'variant':<7} {'aw':>5} {'n':>3} {'mean_sd@50':>11} "
        f"{'ci_half':>9} {'mean_test_acc':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["n"] == 0:
            print(
                f"{r['variant']:<7} {r['aw']:>5g} {0:>3} "
                f"{'-':>11} {'-':>9} {'-':>14}"
            )
        else:
            print(
                f"{r['variant']:<7} {r['aw']:>5g} {r['n']:>3} "
                f"{r['mean50']:>11.6f} {r['ci50']:>9.6f} {r['acc']:>14.4f}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2A (MNIST, hidden=512)."
    )
    parser.add_argument("--variant", default="a")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--root", default="results", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    variants = [v.strip() for v in args.variant.split(",") if v.strip()]
    bad = [v for v in variants if v not in TITLES]
    if bad:
        raise SystemExit(f"unknown variant(s): {', '.join(bad)} (use a and/or b)")

    out = args.out
    if out is None:
        out = f"figs/fig2a_{args.variant.replace(',', '_')}.png"
    out_path = Path(out)

    n_panels = len(variants)
    figsize = (4.6 * n_panels, 3.6)
    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize, sharey=True, squeeze=False
    )
    axes = axes[0]
    fig.patch.set_facecolor("white")

    all_rows = []
    for i, variant in enumerate(variants):
        ax = axes[i]
        ax.set_facecolor("white")
        ax.grid(False)
        rows = draw_panel(ax, args.root, variant, EPOCHS, show_legend=(i == 0))
        all_rows.extend(rows)

        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xlim(1, EPOCHS)
        if i == 0:
            ax.set_ylabel("SD of Weight Distribution")
        if n_panels > 1:
            ax.set_title(TITLES[variant], fontsize=10)

        # Panel letter, bold, top-left just outside the axes.
        ax.text(
            -0.16,
            1.04,
            chr(ord("A") + i),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print_table(all_rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
