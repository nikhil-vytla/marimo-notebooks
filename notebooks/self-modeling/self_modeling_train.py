# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "numpy==2.5.3",
#     "datasets==5.1.0",
#     "pillow==12.3.0",
#     "altair==6.3.0",
#     "pandas==3.0.6",
# ]
# ///
# smoke: skip

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch import nn

    return F, alt, load_dataset, mo, nn, np, os, pd, time, torch


@app.cell
def _(mo):
    mo.md("""
    # Self-Modeling in Neural Systems — Figure 2 training

    Reproduces Figure 2 (A–D, RLCT excluded) of
    [*Unexpected Benefits of Self-Modeling in Neural Systems*](https://arxiv.org/abs/2407.10188).

    Amortized MNIST MLP `784 -> H -> Linear(10 + H)`: the extra output rows
    predict the post-ReLU hidden activations. **AW = 0** is the baseline.
    Everything below is self-contained; heavy work only runs when you click
    **Train**.
    """)
    return


@app.cell
def _(torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device_name = "CPU"
    return device, device_name


@app.cell
def _(device, device_name, mo):
    mo.md(f"**Device:** `{device}` — {device_name}")
    return


@app.cell
def _(mo):
    hidden_sizes = mo.ui.multiselect(
        options=[64, 128, 256, 512],
        value=[64, 128, 256, 512],
        label="Hidden sizes",
    )
    aws = mo.ui.multiselect(
        options=[0, 1, 5, 10, 20, 50],
        value=[0, 1, 5, 10, 20, 50],
        label="Self-modeling weight (AW)",
    )
    seeds = mo.ui.number(value=10, start=1, stop=50, label="Seeds")
    epochs = mo.ui.number(value=50, start=1, stop=200, label="Epochs")
    smoke = mo.ui.checkbox(
        value=False, label="Smoke test (2 epochs, 2 seeds)"
    )
    train_button = mo.ui.run_button(label="Train")
    return aws, epochs, hidden_sizes, seeds, smoke, train_button


@app.cell
def _(aws, epochs, hidden_sizes, mo, seeds, smoke, train_button):
    mo.md(
        """
    ### Configuration

    Each run trains `len(AW) x seeds` independent models in one batched
    forward/backward. SGD `lr=0.005`, momentum `0.9`, nesterov, batch 512.
    """
    )
    mo.vstack(
        [
            mo.hstack([hidden_sizes, aws], justify="start", gap=2),
            mo.hstack([seeds, epochs], justify="start", gap=2),
            mo.hstack([smoke, train_button], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(aws, epochs, hidden_sizes, mo, os, seeds, smoke, train_button):
    _mode = mo.app_meta().mode
    _script = _mode == "script"
    _smoke_env = os.environ.get("SMOKE") == "1"
    smoke_test = bool(smoke.value) or (_script and _smoke_env)

    if _script:
        _env_hidden = os.environ.get("SM_HIDDEN", "")
        _parsed = [int(h) for h in _env_hidden.split(",") if h.strip()]
        eff_hidden = _parsed or [int(h) for h in hidden_sizes.value]
    else:
        eff_hidden = [int(h) for h in hidden_sizes.value]

    eff_aws = [float(a) for a in aws.value]
    eff_seeds = 2 if smoke_test else int(seeds.value)
    eff_epochs = 2 if smoke_test else int(epochs.value)
    train_clicked = bool(train_button.value) or _script
    return (
        eff_aws,
        eff_epochs,
        eff_hidden,
        eff_seeds,
        smoke_test,
        train_clicked,
    )


@app.cell
def _(F, load_dataset, nn, np, torch):
    BATCH = 512
    LR = 0.005
    MOMENTUM = 0.9

    # Two-sided 95% t critical values by degrees of freedom (df = n - 1).
    _T95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }

    def t95(df):
        if df <= 0:
            return 0.0
        return _T95.get(int(df), 1.96)

    class SelfModelMLP(nn.Module):
        def __init__(self, hidden, self_model):
            super().__init__()
            self.self_model = self_model
            self.n_classes = 10
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10 + (hidden if self_model else 0))

    def build_params(models, hidden, device):
        """Stacked init, equivalent to training a SelfModelMLP per (aw, seed)."""
        M = len(models)
        W1 = torch.empty(M, hidden, 784, device=device)
        b1 = torch.empty(M, hidden, device=device)
        W2 = torch.empty(M, 10 + hidden, hidden, device=device)
        b2 = torch.empty(M, 10 + hidden, device=device)
        for m, (aw, seed) in enumerate(models):
            self_model = aw > 0
            torch.manual_seed(seed)
            ref = SelfModelMLP(hidden, self_model)
            with torch.no_grad():
                W1[m] = ref.hidden.weight
                b1[m] = ref.hidden.bias
                W2[m, :10] = ref.out.weight[:10]
                b2[m, :10] = ref.out.bias[:10]
                if self_model:
                    W2[m, 10:] = ref.out.weight[10:]
                    b2[m, 10:] = ref.out.bias[10:]
                else:
                    torch.manual_seed(seed)
                    ref_self = SelfModelMLP(hidden, True)
                    W2[m, 10:] = ref_self.out.weight[10:]
                    b2[m, 10:] = ref_self.out.bias[10:]
        return [
            nn.Parameter(W1),
            nn.Parameter(b1),
            nn.Parameter(W2),
            nn.Parameter(b2),
        ]

    def forward_batched(params, xb):
        """xb: (M, B, 784) -> logits (M, B, 10), a_hat (M, B, H), a (M, B, H)."""
        W1, b1, W2, b2 = params
        a = torch.relu(
            torch.baddbmm(b1.unsqueeze(1), xb, W1.transpose(1, 2))
        )
        out = torch.baddbmm(b2.unsqueeze(1), a, W2.transpose(1, 2))
        return out[..., :10], out[..., 10:], a

    def train_step(params, opt, xb, yb, aws):
        logits, a_hat, a = forward_batched(params, xb)
        M, B = yb.shape
        ce = (
            F.cross_entropy(
                logits.reshape(-1, 10), yb.reshape(-1), reduction="none"
            )
            .reshape(M, B)
            .mean(dim=1)
        )
        sm = (a_hat - a.detach()).pow(2).mean(dim=(1, 2))
        (ce + aws * sm).sum().backward()
        opt.step()
        opt.zero_grad()
        return ce.detach(), sm.detach()

    def evaluate_batched(params, X, y, device):
        M = params[0].shape[0]
        correct = torch.zeros(M, device=device)
        n = y.numel()
        with torch.no_grad():
            for lo in range(0, n, 512):
                hi = min(lo + 512, n)
                xb = X[lo:hi].unsqueeze(0).expand(M, hi - lo, 784)
                logits, _, _ = forward_batched(params, xb)
                correct += (logits.argmax(dim=-1) == y[lo:hi]).sum(dim=1)
        return correct / n

    def classifier_sd(params):
        W2 = params[2]
        flat = W2[:, :10].reshape(W2.shape[0], -1).detach()
        return flat.std(dim=1)

    def classifier_hist(params, bins=60, lo=-0.6, hi=0.6):
        W2 = params[2][:, :10].detach().float().cpu()
        return torch.stack(
            [torch.histc(W2[m], bins=bins, min=lo, max=hi) for m in range(len(W2))]
        )

    def load_mnist(device):
        raw = load_dataset("ylecun/mnist")

        def _split(split):
            X = np.stack(
                [
                    np.asarray(im, dtype=np.uint8).reshape(-1)
                    for im in raw[split]["image"]
                ]
            )
            Xt = torch.from_numpy(X).to(torch.float32) / 127.5 - 1.0
            yt = torch.tensor(raw[split]["label"], dtype=torch.int64)
            return Xt.to(device), yt.to(device)

        Xtr, ytr = _split("train")
        Xte, yte = _split("test")
        return Xtr, ytr, Xte, yte

    return (
        BATCH,
        LR,
        MOMENTUM,
        build_params,
        classifier_hist,
        classifier_sd,
        evaluate_batched,
        load_mnist,
        t95,
        train_step,
    )


@app.cell
def _(
    BATCH,
    LR,
    MOMENTUM,
    build_params,
    classifier_hist,
    classifier_sd,
    evaluate_batched,
    load_mnist,
    mo,
    time,
    torch,
    train_step,
):
    @mo.persistent_cache
    def train_hidden(hidden, aws_list, seeds_n, epochs, device_type):
        """Train every (aw, seed) for one hidden size; return raw per-model data."""
        device = torch.device(device_type)
        X, y, Xte, yte = load_mnist(device)

        models = [
            (float(aw), seed) for aw in aws_list for seed in range(seeds_n)
        ]
        M = len(models)
        aws = torch.tensor(
            [aw for aw, _ in models], dtype=torch.float32, device=device
        )
        params = build_params(models, hidden, device)
        opt = torch.optim.SGD(
            params, lr=LR, momentum=MOMENTUM, nesterov=True
        )
        N = X.shape[0]
        seeds = [seed for _, seed in models]
        gens = [torch.Generator() for _ in range(M)]

        snapshot_epochs = [e for e in (0, 1, 2, 5, 10, 20, 30, 40, 50) if e <= epochs]
        sd_hist = [[] for _ in range(M)]
        acc_hist = [[] for _ in range(M)]
        hist_snap = {e: [] for e in snapshot_epochs}
        epoch_seconds = []

        def _record_hist(epoch):
            counts = classifier_hist(params).to(torch.int64).tolist()
            for m in range(M):
                hist_snap[epoch].append(counts[m])

        init_sd = classifier_sd(params).tolist()
        for m in range(M):
            sd_hist[m].append(init_sd[m])
        if 0 in hist_snap:
            _record_hist(0)

        for epoch in mo.status.progress_bar(
            range(1, epochs + 1),
            title=f"hidden={hidden}",
            show_rate=True,
            show_eta=True,
        ):
            for m in range(M):
                gens[m].manual_seed(seeds[m] * 1000 + epoch)
            perms = torch.stack(
                [torch.randperm(N, generator=gens[m]) for m in range(M)]
            ).to(device)

            if device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()

            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                idx = perms[:, lo:hi]
                train_step(params, opt, X[idx], y[idx], aws)

            sd_vec = classifier_sd(params)
            acc_vec = evaluate_batched(params, Xte, yte, device)

            if device.type == "mps":
                torch.mps.synchronize()
            epoch_seconds.append(time.perf_counter() - start)

            for m in range(M):
                sd_hist[m].append(sd_vec[m].item())
                acc_hist[m].append(acc_vec[m].item())
            if epoch in hist_snap:
                _record_hist(epoch)

        runs = [
            {
                "aw": aw,
                "seed": seed,
                "sd": sd_hist[m],
                "test_acc": acc_hist[m],
                "final_acc": acc_hist[m][-1],
                "hist": {str(e): hist_snap[e][m] for e in snapshot_epochs},
            }
            for m, (aw, seed) in enumerate(models)
        ]
        return {
            "hidden": hidden,
            "epochs": epochs,
            "snapshot_epochs": snapshot_epochs,
            "bins": 60,
            "bin_range": [-0.6, 0.6],
            "models": runs,
            "epoch_seconds": epoch_seconds,
        }

    return (train_hidden,)


@app.cell
def _(
    device,
    device_name,
    eff_aws,
    eff_epochs,
    eff_hidden,
    eff_seeds,
    mo,
    train_clicked,
    train_hidden,
):
    if train_clicked and eff_hidden:
        runs = []
        with mo.status.progress_bar(
            total=len(eff_hidden), title="Hidden sizes"
        ) as bar:
            for _h in eff_hidden:
                runs.append(
                    train_hidden(_h, eff_aws, eff_seeds, eff_epochs, device.type)
                )
                bar.update(subtitle=f"hidden={_h} done")

        print(f"device: {device} ({device_name})")
        for _run in runs:
            _secs = _run["epoch_seconds"]
            _mean = sum(_secs) / len(_secs) if _secs else 0.0
            print(
                f"hidden={_run['hidden']}: epochs={_run['epochs']} "
                f"{_mean:.2f}s/epoch"
            )
        _h512 = [r for r in runs if r["hidden"] == 512]
        if _h512:
            print("hidden 512 final mean SD per AW:")
            for _aw in eff_aws:
                _vals = [
                    m["sd"][-1]
                    for m in _h512[0]["models"]
                    if abs(m["aw"] - _aw) < 1e-9
                ]
                if _vals:
                    print(
                        f"  AW={_aw:g}: {sum(_vals) / len(_vals):.5f} "
                        f"(n={len(_vals)})"
                    )
    else:
        runs = None
    return (runs,)


@app.cell
def _(
    device,
    device_name,
    eff_aws,
    eff_epochs,
    eff_hidden,
    eff_seeds,
    np,
    runs,
    smoke_test,
    t95,
):
    if runs:
        def _mean_ci(values):
            arr = np.asarray(values, dtype=float)
            n = arr.size
            if n == 0:
                return 0.0, 0.0
            if n == 1:
                return float(arr[0]), 0.0
            half = t95(n - 1) * float(arr.std(ddof=1)) / np.sqrt(n)
            return float(arr.mean()), float(half)

        _results = {}
        for _run in runs:
            _aw_map = {}
            for _aw in eff_aws:
                _key = f"{_aw:g}"
                _runs_aw = [
                    m
                    for m in _run["models"]
                    if abs(m["aw"] - _aw) < 1e-9
                ]
                _sd_mean, _sd_ci = [], []
                for _e in range(_run["epochs"] + 1):
                    _m, _c = _mean_ci([r["sd"][_e] for r in _runs_aw])
                    _sd_mean.append(_m)
                    _sd_ci.append(_c)
                _acc_mean, _acc_ci = _mean_ci(
                    [r["final_acc"] for r in _runs_aw]
                )
                _acc_ep_mean, _acc_ep_ci = [], []
                for _e in range(_run["epochs"]):
                    _m2, _c2 = _mean_ci(
                        [r["test_acc"][_e] for r in _runs_aw]
                    )
                    _acc_ep_mean.append(_m2)
                    _acc_ep_ci.append(_c2)
                _hist_mean = {}
                for _e in _run["snapshot_epochs"]:
                    _stack = np.asarray(
                        [r["hist"][str(_e)] for r in _runs_aw], dtype=float
                    )
                    _hist_mean[str(_e)] = _stack.mean(axis=0).tolist()
                _aw_map[_key] = {
                    "aw": _aw,
                    "n": len(_runs_aw),
                    "sd_mean": _sd_mean,
                    "sd_ci": _sd_ci,
                    "acc_mean": _acc_mean,
                    "acc_ci": _acc_ci,
                    "acc_mean_per_epoch": _acc_ep_mean,
                    "acc_ci_per_epoch": _acc_ep_ci,
                    "hist_mean": _hist_mean,
                }
            _results[str(_run["hidden"])] = {
                "hidden": _run["hidden"],
                "epochs": _run["epochs"],
                "snapshot_epochs": _run["snapshot_epochs"],
                "bins": _run["bins"],
                "bin_range": _run["bin_range"],
                "aws": _aw_map,
            }

        fig2 = {
            "config": {
                "hidden_sizes": eff_hidden,
                "aws": eff_aws,
                "seeds": eff_seeds,
                "epochs": eff_epochs,
                "batch_size": 512,
                "lr": 0.005,
                "momentum": 0.9,
                "nesterov": True,
                "smoke": smoke_test,
            },
            "device": {"type": device.type, "name": device_name},
            "snapshot_epochs": sorted(
                {e for r in runs for e in r["snapshot_epochs"]}
            ),
            "bins": 60,
            "bin_range": [-0.6, 0.6],
            "results": _results,
        }
    else:
        fig2 = None
    return (fig2,)


@app.cell
def _(fig2, mo):
    import json
    from pathlib import Path

    if fig2 is not None:
        _payload = json.dumps(fig2).encode("utf-8")
        _loc = mo.notebook_location()
        _msg = "notebook location unavailable"
        if _loc is not None:
            _path = Path(str(_loc)) / "public" / "data" / "mnist_fig2.json"
            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
                _path.write_bytes(_payload)
                _msg = f"wrote `{_path}`"
            except OSError as _err:
                _msg = f"could not write `{_path}`: {_err}"
        _out = mo.vstack(
            [
                mo.md(f"{_msg} ({len(_payload) / 1024:.1f} KiB)"),
                mo.download(
                    _payload,
                    filename="fig2.json",
                    mimetype="application/json",
                    label="Download fig2.json",
                ),
            ]
        )
    else:
        _out = mo.md("_Run training to produce results._")
    _out
    return


@app.cell
def _(alt, fig2, pd):
    if fig2 is None:
        charts = None
    else:
        _colors = {
            "0": "#000000",
            "1": "#ff1493",
            "5": "#00e000",
            "10": "#8a2be2",
            "20": "#ff4500",
            "50": "#00bfff",
        }
        _keys = [f"{a:g}" for a in fig2["config"]["aws"]]
        _range = [_colors[k] for k in _keys if k in _colors]
        _scale = alt.Scale(
            domain=[k for k in _keys if k in _colors], range=_range
        )
        _color = alt.Color(
            "aw:N", scale=_scale, legend=alt.Legend(title="AW")
        )

        _hkey = "512" if "512" in fig2["results"] else max(
            fig2["results"], key=lambda k: int(k)
        )
        _a_rows = []
        for _aw_key, _res in fig2["results"][_hkey]["aws"].items():
            for _e, (_m, _c) in enumerate(zip(_res["sd_mean"], _res["sd_ci"])):
                _a_rows.append(
                    {"aw": _aw_key, "epoch": _e, "mean": _m, "lo": _m - _c, "hi": _m + _c}
                )
        _a = pd.DataFrame(_a_rows)
        _a_base = alt.Chart(_a).encode(x=alt.X("epoch:Q", title="epoch"))
        _a_band = _a_base.mark_area(opacity=0.15).encode(
            y=alt.Y("lo:Q", title="classifier weight SD"), y2="hi:Q", color=_color
        )
        _a_line = _a_base.mark_line().encode(y="mean:Q", color=_color)
        fig2a = (_a_band + _a_line).properties(
            width=380, height=260, title=f"Fig 2A — SD vs epoch (hidden {_hkey})"
        )

        _final_sd, _final_acc = [], []
        for _h_key, _res in fig2["results"].items():
            _last = _res["epochs"]
            for _aw_key, _aw_res in _res["aws"].items():
                _m, _c = _aw_res["sd_mean"][_last], _aw_res["sd_ci"][_last]
                _final_sd.append(
                    {
                        "hidden": int(_h_key),
                        "aw": _aw_key,
                        "mean": _m,
                        "lo": max(0.0, _m - _c),
                        "hi": _m + _c,
                    }
                )
                _final_acc.append(
                    {
                        "hidden": int(_h_key),
                        "aw": _aw_key,
                        "mean": _aw_res["acc_mean"],
                        "lo": max(0.0, _aw_res["acc_mean"] - _aw_res["acc_ci"]),
                        "hi": _aw_res["acc_mean"] + _aw_res["acc_ci"],
                    }
                )
        _hidden_sort = sorted({r["hidden"] for r in _final_sd})
        _x = alt.X("hidden:O", title="hidden size", sort=_hidden_sort)

        def _band_line(rows, title):
            _df = pd.DataFrame(rows)
            _b = alt.Chart(_df).encode(x=_x)
            _band = _b.mark_area(opacity=0.15).encode(
                y=alt.Y("lo:Q", title=None), y2="hi:Q", color=_color
            )
            _line = _b.mark_line().encode(y="mean:Q", color=_color)
            return (_band + _line).properties(width=380, height=260, title=title)

        fig2b = _band_line(_final_sd, "Fig 2B — final SD vs hidden size")
        fig2d = _band_line(_final_acc, "Fig 2D — final accuracy vs hidden size")
        charts = [fig2a, fig2b, fig2d]
    return (charts,)


@app.cell
def _(charts, mo):
    if charts is None:
        _out = mo.md("_Run training to produce charts._")
    else:
        _out = mo.vstack(
            [
                mo.hstack([charts[0], charts[1]], gap=1),
                charts[2],
            ]
        )
    _out
    return


if __name__ == "__main__":
    app.run()
