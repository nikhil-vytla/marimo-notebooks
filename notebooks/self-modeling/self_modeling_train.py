# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "numpy==2.5.3",
#     "datasets==5.1.0",
#     "pillow==12.3.0",
#     "altair==6.3.0",
#     "pandas==3.0.6",
#     "devinterp==2.0.1",
#     "zarr==3.1.3",
# ]
# ///
# smoke: skip

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


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

    return F, alt, mo, nn, np, os, pd, time, torch


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
    mo.md(f"""
    **Device:** `{device}` — {device_name}
    """)
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
def _(F, nn, np, torch):
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
        # Import here rather than at module top level so the loader always uses
        # whichever ``datasets`` version the kernel environment currently provides.
        from datasets import load_dataset

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
        forward_batched,
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

        with torch.no_grad():
            final_params = {
                "W1": params[0].detach().cpu().clone(),
                "b1": params[1].detach().cpu().clone(),
                "W2": params[2][:, :10].detach().cpu().clone(),
                "b2": params[3][:, :10].detach().cpu().clone(),
            }

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
            "final_params": final_params,
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
    return Path, json


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


@app.cell
def _(mo):
    rlct_button = mo.ui.run_button(label="Estimate RLCT")
    rlct_plateau_button = mo.ui.run_button(label="Plateau check (loc=100)")

    mo.vstack(
        [
            mo.md(
                """
    ### Figure 2C — RLCT vs hidden size

    Estimates the local learning coefficient (LLC) of the pruned classification
    head ``784 -> H -> 10`` (``W1, b1, W2[:10], b2[:10]``) for every model in
    ``runs``. Click **Estimate RLCT** below.

    The estimator is a **batched SGLD** sampler: every ``(aw, seed)`` model of one
    hidden size is stacked into one parameter tensor and all chains advance
    together, which is mathematically the same per-model estimator as
    ``devinterp`` 2.0.1's SGLD/LLC. It first validates against ``devinterp`` on the
    four pilot models (H=512, AW in {0, 50}, seeds {0, 1}) at localization 100 with
    4 chains x 4000 draws (pass: every model within 3%), and runs a 4 x 6000-draw
    convergence check. Localization 1000 keeps the restored ``devinterp`` results
    and adds a cheap 4 x 200-draw batched cross-check. ``devinterp`` is retained
    for the validation cell. **Plateau check** re-estimates the same pilots with
    ``devinterp`` and reports the shortest draw count whose last two chain-averaged
    loss quarters agree within 2%.
    """
            ),
            mo.hstack([rlct_button, rlct_plateau_button], justify="start", gap=2),
        ]
    )

    return rlct_button, rlct_plateau_button


@app.cell
def _(F, load_mnist, mo, nn, np, os, runs, t95, time, torch):
    import shutil
    import tempfile

    RLCT_BATCH = 512
    RLCT_LR = 1e-4
    RLCT_NUM_BURNIN_STEPS = 0
    RLCT_STEPS_BW_DRAWS = 1
    RLCT_NUM_INIT_LOSS_BATCHES = 117
    RLCT_INIT_SEED = 0
    RLCT_NBETA = 512.0 / float(np.log(512.0))
    RLCT_NBETA_CONVENTION = "512/log(512) (devinterp batch-based optimal temperature)"

    # Published Figure 2C payload; also the source for previously computed results.
    RLCT_DATA_PATH = os.path.join(
        str(mo.notebook_location() or ""), "public", "data", "mnist_fig2c.json"
    )

    # Explicit sampling configuration. ``localizations`` are estimated in order and
    # cached under their own keys, so the published localization=1000 results and a
    # new localization=100 sweep coexist.
    rlct_config = {
        "localizations": (1000.0, 100.0),
        "num_chains": 4,
        "num_draws": 200,
    }
    # Results at this localization are already published; restore, never recompute.
    RLCT_PUBLISHED_LOCALIZATION = 1000.0
    # Plateau check: four pilot models (H=512, AW in {0, 50}, seeds {0, 1}).
    rlct_plateau_config = {
        "hidden": 512,
        "localization": 100.0,
        "num_chains": 4,
        "draws": (1000, 2000, 4000),
        "pairs": ((0.0, 0), (0.0, 1), (50.0, 0), (50.0, 1)),
        "tolerance": 0.02,
    }

    # Small indirection so the model does not store (and deepcopy) the train set.
    _RLCT_LOOKUP = {}


    class _RlctMLP(nn.Module):
        """Plain 784 -> H -> 10 MLP whose forward reads pixel rows by index."""

        def __init__(self, hidden, key):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)
            self._key = key

        def forward(self, input_ids):
            _xlook = _RLCT_LOOKUP[self._key][0]
            return self.out(torch.relu(self.hidden(_xlook[input_ids[:, 0].long()])))


    class _RlctDataset(torch.utils.data.Dataset):
        """Rows are (train index, label); details are looked up on device."""

        def __init__(self, labels):
            self.pairs = torch.stack(
                [
                    torch.arange(labels.numel(), dtype=torch.int64),
                    labels.detach().long().cpu(),
                ],
                dim=1,
            )

        def __len__(self):
            return self.pairs.shape[0]

        def __getitem__(self, i):
            return {"input_ids": self.pairs[i]}


    def _rlct_loss(model, input_ids):
        labels = input_ids[:, 1].long()
        return F.cross_entropy(model(input_ids), labels, reduction="none").unsqueeze(1)


    def _rlct_mean_ci(values):
        arr = np.asarray(values, dtype=float)
        n = arr.size
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return float(arr[0]), 0.0
        half = t95(n - 1) * float(arr.std(ddof=1)) / np.sqrt(n)
        return float(arr.mean()), float(half)


    def _loss_quarters(loss_trace):
        """Chain-averaged loss over four equal draw quarters (q1..q4)."""
        arr = np.asarray(loss_trace, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        edges = np.linspace(0, arr.shape[1], 5).astype(int)
        return [
            float(arr[:, edges[i] : edges[i + 1]].mean())
            if edges[i + 1] > edges[i]
            else float("nan")
            for i in range(4)
        ]


    def rlct_weight_fingerprint(runs, hidden):
        """Short hash of the final weights, so caches track the trained model."""
        import hashlib as _hashlib

        run = next(r for r in runs if r["hidden"] == hidden)
        return _hashlib.sha256(
            b"".join(
                run["final_params"][k].numpy().tobytes()
                for k in ("W1", "b1", "W2", "b2")
            )
        ).hexdigest()[:16]


    @mo.persistent_cache
    def estimate_rlct(
        hidden,
        pairs,
        epochs_n,
        device_type,
        weights_fingerprint,
        localization,
        num_chains,
        num_draws,
    ):
        """LLC for the ``(aw, seed)`` pairs of one hidden size at one localization."""
        from devinterp.slt.llc import llc

        device = torch.device(device_type)
        X, y, _Xte, _yte = load_mnist(device)
        dataset = _RlctDataset(y)
        run = next(
            r for r in runs if r["hidden"] == hidden and r["epochs"] == epochs_n
        )
        params = run["final_params"]
        out = {}
        with mo.status.progress_bar(
            total=len(pairs),
            title=f"RLCT H={hidden} loc={localization:g} draws={num_draws}",
        ) as bar:
            for aw, seed in pairs:
                m = next(
                    i
                    for i, e in enumerate(run["models"])
                    if abs(e["aw"] - aw) < 1e-9 and e["seed"] == seed
                )
                key = (hidden, float(aw), seed)
                model = _RlctMLP(hidden, key)
                with torch.no_grad():
                    model.hidden.weight.copy_(params["W1"][m])
                    model.hidden.bias.copy_(params["b1"][m])
                    model.out.weight.copy_(params["W2"][m])
                    model.out.bias.copy_(params["b2"][m])
                _RLCT_LOOKUP[key] = (X, y, key)
                _tmp = tempfile.mkdtemp()
                _started = time.perf_counter()
                try:
                    res = llc(
                        model,
                        dataset,
                        {},
                        lr=RLCT_LR,
                        n_beta=RLCT_NBETA,
                        num_chains=num_chains,
                        num_draws=num_draws,
                        batch_size=RLCT_BATCH,
                        num_burnin_steps=RLCT_NUM_BURNIN_STEPS,
                        num_steps_bw_draws=RLCT_STEPS_BW_DRAWS,
                        localization=float(localization),
                        num_init_loss_batches=RLCT_NUM_INIT_LOSS_BATCHES,
                        init_seed=RLCT_INIT_SEED,
                        device=device_type,
                        loss_fn=_rlct_loss,
                        noise_level=1.0,
                        llc_weight_decay=0.0,
                        bounding_box_size=None,
                        sampling_method="sgmcmc_sgld",
                        gradient_accumulation_steps=1,
                        shuffle=True,
                        match_sampling_input_ids_across_chains=True,
                        init_noise=None,
                        save_metrics=False,
                        param_masks=None,
                        output_path=os.path.join(_tmp, "samples.zarr"),
                    )
                    entry = out.setdefault(
                        f"{float(aw):g}",
                        {
                            "aw": float(aw),
                            "seeds": [],
                            "llc": [],
                            "loss_quarters": [],
                            "seconds": [],
                        },
                    )
                    entry["seeds"].append(int(seed))
                    entry["llc"].append(
                        float(res["llc_per_chain"].values.mean())
                    )
                    entry["loss_quarters"].append(
                        _loss_quarters(res["loss_trace"].values)
                    )
                    entry["seconds"].append(time.perf_counter() - _started)
                finally:
                    _RLCT_LOOKUP.pop(key, None)
                    shutil.rmtree(_tmp, ignore_errors=True)
                bar.update(subtitle=f"AW={aw:g} seed={seed}")
        for entry in out.values():
            entry["mean"], entry["ci"] = _rlct_mean_ci(entry["llc"])
            entry["mala"] = [None] * len(entry["llc"])
        return out


    return (
        RLCT_BATCH,
        RLCT_DATA_PATH,
        RLCT_INIT_SEED,
        RLCT_LR,
        RLCT_NBETA,
        RLCT_NBETA_CONVENTION,
        RLCT_NUM_BURNIN_STEPS,
        RLCT_NUM_INIT_LOSS_BATCHES,
        RLCT_PUBLISHED_LOCALIZATION,
        RLCT_STEPS_BW_DRAWS,
        estimate_rlct,
        rlct_plateau_config,
        rlct_weight_fingerprint,
    )


@app.cell
def rlct_batched_estimator(
    F,
    RLCT_BATCH,
    RLCT_INIT_SEED,
    RLCT_LR,
    RLCT_NBETA,
    forward_batched,
    load_mnist,
    mo,
    np,
    runs,
    t95,
    torch,
):
    # --- Batched SGLD LLC estimator -------------------------------------------
    # Samples every (aw, seed) model of one hidden size as a single stacked batch.
    # Chains are an extra leading dimension, so one (M * num_chains) x ... tensor
    # samples M models with num_chains chains each. This is mathematically the same
    # per-model estimator as devinterp 2.0.1's sgld/llc:
    #   * init loss L0 = mean CE over the same 117-batch train cycle devinterp uses
    #     (num_init_loss_batches=117, batch 512, the seeded shuffle from init_seed);
    #   * each step: w <- w - (lr/2)(nbeta*g + localization*(w - w0))
    #                      + sqrt(lr) * N(0, 1), with g the minibatch mean CE grad;
    #   * the minibatch mean CE is recorded every step (num_burnin_steps=0,
    #     num_steps_bw_draws=1);
    #   * per chain LLC = nbeta * (mean recorded loss - L0); model LLC = mean over
    #     chains.
    # With match_sampling_input_ids_across_chains=True devinterp walks one fixed,
    # seeded 117-batch permutation for every chain (and, because init_seed is the
    # same, for every model), so we reproduce that exact order.
    RLCT_BATCHED_CONFIG = {
        "num_chains": 4,
        "localizations": {1000.0: 200, 100.0: 4000},
        "seed": 0,
    }


    class _RlctIndexDataset(torch.utils.data.Dataset):
        """(train index, label) rows for a DataLoader over the MNIST train set."""

        def __init__(self, labels):
            self.pairs = torch.stack(
                [
                    torch.arange(labels.numel(), dtype=torch.int64),
                    labels.detach().long().cpu(),
                ],
                dim=1,
            )

        def __len__(self):
            return self.pairs.shape[0]

        def __getitem__(self, i):
            return {"input_ids": self.pairs[i]}


    def _rlct_batch_order(y):
        """The fixed minibatch order devinterp 2.0.1 samples with this config."""
        generator = torch.Generator().manual_seed(RLCT_INIT_SEED)
        loader = torch.utils.data.DataLoader(
            _RlctIndexDataset(y),
            batch_size=RLCT_BATCH,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )
        return torch.stack([batch["input_ids"][:, 0] for batch in loader])


    @mo.persistent_cache
    def estimate_rlct_batched(
        hidden,
        pairs,
        epochs_n,
        device_type,
        weights_fingerprint,
        localization,
        num_chains,
        num_steps,
        seed,
    ):
        """Batched SGLD LLC for the (aw, seed) pairs of one hidden size."""
        device = torch.device(device_type)
        X, y, _Xte, _yte = load_mnist(device)
        batch_idx = _rlct_batch_order(y)
        n_batches = batch_idx.shape[0]

        run = next(
            r for r in runs if r["hidden"] == hidden and r["epochs"] == epochs_n
        )
        picks = [
            next(
                i
                for i, e in enumerate(run["models"])
                if abs(e["aw"] - aw) < 1e-9 and e["seed"] == seed_value
            )
            for aw, seed_value in pairs
        ]
        M = len(picks)
        # Per-model initial weights, then expanded along a new chain dimension.
        w0_single = [
            run["final_params"][k][picks].contiguous()
            for k in ("W1", "b1", "W2", "b2")
        ]
        with torch.no_grad():
            L0 = torch.zeros(M, device=device)
            for b in range(n_batches):
                idx = batch_idx[b]
                xb = X[idx].unsqueeze(0).expand(M, -1, -1)
                logits, _a_hat, _a = forward_batched(w0_single, xb)
                ce = F.cross_entropy(
                    logits.reshape(-1, 10),
                    y[idx].unsqueeze(0).expand(M, -1).reshape(-1),
                    reduction="none",
                ).reshape(M, RLCT_BATCH).mean(dim=1)
                L0 += ce / n_batches

        w0 = [
            t.repeat_interleave(num_chains, dim=0).contiguous() for t in w0_single
        ]
        params = [t.clone().requires_grad_(True) for t in w0]
        W1, b1, W2, b2 = params
        L0 = L0.repeat_interleave(num_chains)

        torch.manual_seed(seed)
        losses = torch.empty(len(pairs) * num_chains, num_steps, device=device)
        for step in range(num_steps):
            idx = batch_idx[step % n_batches]
            xb = X[idx].unsqueeze(0).expand(losses.shape[0], -1, -1)
            yb = y[idx].unsqueeze(0).expand(losses.shape[0], -1)
            logits, _a_hat, _a = forward_batched(params, xb)
            ce = F.cross_entropy(
                logits.reshape(-1, 10), yb.reshape(-1), reduction="none"
            ).reshape(losses.shape[0], RLCT_BATCH).mean(dim=1)
            losses[:, step] = ce.detach()
            ce.sum().backward()
            with torch.no_grad():
                for i, p in enumerate(params):
                    p.add_(p - w0[i], alpha=-0.5 * RLCT_LR * localization)
                    p.add_(p.grad, alpha=-0.5 * RLCT_LR * RLCT_NBETA)
                    p.add_(torch.randn_like(p), alpha=RLCT_LR**0.5)
                    p.grad = None

        llc_per_chain = (
            RLCT_NBETA * (losses.mean(dim=1) - L0)
        ).reshape(len(pairs), num_chains)
        llc_per_model = llc_per_chain.mean(dim=1)

        out = {}
        for m, (aw, seed_value) in enumerate(pairs):
            entry = out.setdefault(
                f"{float(aw):g}", {"aw": float(aw), "seeds": [], "llc": []}
            )
            entry["seeds"].append(int(seed_value))
            entry["llc"].append(float(llc_per_model[m]))
        for entry in out.values():
            arr = np.asarray(entry["llc"], dtype=float)
            n = arr.size
            entry["mean"] = float(arr.mean())
            entry["ci"] = (
                0.0
                if n <= 1
                else float(t95(n - 1) * arr.std(ddof=1) / np.sqrt(n))
            )
        return out

    return RLCT_BATCHED_CONFIG, estimate_rlct_batched


@app.cell
def _(
    RLCT_BATCH,
    RLCT_BATCHED_CONFIG,
    RLCT_DATA_PATH,
    RLCT_INIT_SEED,
    RLCT_LR,
    RLCT_NBETA,
    RLCT_NBETA_CONVENTION,
    RLCT_NUM_BURNIN_STEPS,
    RLCT_NUM_INIT_LOSS_BATCHES,
    RLCT_PUBLISHED_LOCALIZATION,
    RLCT_STEPS_BW_DRAWS,
    device,
    eff_aws,
    eff_epochs,
    eff_hidden,
    eff_seeds,
    estimate_rlct,
    estimate_rlct_batched,
    np,
    os,
    rlct_button,
    rlct_plateau,
    rlct_weight_fingerprint,
    runs,
):

    import importlib.metadata


    def _load_prior_rlct(path):
        """Prior RLCT results keyed by localization, from an earlier JSON payload."""
        import json as _json

        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path) as _f:
                blob = _json.load(_f)
        except (OSError, ValueError):
            return {}
        if "by_localization" in blob:
            return blob["by_localization"]
        if "rlct" in blob:
            # Pre-``by_localization`` payload: these are the published results.
            return {
                f"{RLCT_PUBLISHED_LOCALIZATION:g}": {
                    "rlct": blob["rlct"],
                    "hyperparams": blob.get("hyperparams", {}),
                }
            }
        return {}


    def rlct_hyperparams(localization, num_chains, num_draws, device_type):
        return {
            "devinterp_version": importlib.metadata.version("devinterp"),
            "llc_entry_point": "devinterp.slt.llc.llc",
            "sampler": "devinterp.optim.SGMCMC.sgld via devinterp.slt.sampling.sample",
            "loss": (
                "cross-entropy on pruned classification head "
                "(W1, b1, W2[:10], b2[:10]), MNIST train set"
            ),
            "lr": RLCT_LR,
            "localization": float(localization),
            "nbeta": RLCT_NBETA,
            "nbeta_convention": RLCT_NBETA_CONVENTION,
            "nbeta_library_default": 512.0 / float(np.log(512.0)),
            "num_chains": int(num_chains),
            "num_draws": int(num_draws),
            "num_burnin_steps": RLCT_NUM_BURNIN_STEPS,
            "num_steps_bw_draws": RLCT_STEPS_BW_DRAWS,
            "batch_size": RLCT_BATCH,
            "num_init_loss_batches": RLCT_NUM_INIT_LOSS_BATCHES,
            "init_seed": RLCT_INIT_SEED,
            "device": str(device_type),
            "mala_acceptance": "not implemented in devinterp 2.0.1",
            "pilot": {
                "nbeta_n_over_log_n": float(60000 / np.log(60000)),
                "nbeta_batch_default": 512.0 / float(np.log(512.0)),
                "chosen": "batch_default",
            },
        }


    def _rlct_batched_hyperparams(localization, num_chains, num_draws, device_type):
        out = rlct_hyperparams(localization, num_chains, num_draws, device_type)
        out["estimator"] = "batched SGLD, validated against devinterp 2.0.1"
        out["seed"] = RLCT_BATCHED_CONFIG["seed"]
        out["data_order"] = (
            "one fixed seeded 117-batch permutation shared across chains and models "
            "(devinterp match_sampling_input_ids_across_chains=True, "
            "epoch_mode='cycle')"
        )
        out["chains"] = (
            "chains are an extra leading dimension of the stacked parameter tensor, "
            "so one step advances every (model, chain) at once"
        )
        return out


    def _rlct_validation(batched, devinterp_rows, tolerance):
        """Per-model batched vs devinterp comparison at one localization."""
        models = []
        worst = 0.0
        for aw_key, entry in batched.items():
            for seed, value in zip(entry["seeds"], entry["llc"]):
                reference = devinterp_rows.get(f"{aw_key}:{seed}")
                if reference is None:
                    continue
                ref_value = float(reference["llc"])
                rel = abs(value - ref_value) / abs(ref_value) if ref_value else float("inf")
                worst = max(worst, rel)
                models.append(
                    {
                        "aw": float(entry["aw"]),
                        "seed": int(seed),
                        "batched": float(value),
                        "devinterp": ref_value,
                        "rel_diff": float(rel),
                    }
                )
        return {
            "models": models,
            "worst_rel_diff": float(worst),
            "tolerance": float(tolerance),
            "passed": bool(worst <= tolerance),
        }


    def run_rlct_batched_sweep(
        hidden_sizes,
        aws_list,
        seeds_n,
        epochs_n,
        device_type,
        validation_draws=4000,
        convergence_draws=(4000, 6000),
        convergence_tolerance=0.02,
        validation_tolerance=0.03,
    ):
        """Validate the batched estimator, then run the full loc=100 sweep.

        Returns a ``by_localization`` payload. The localized-1000 entry keeps the
        restored devinterp results and adds a cheap batched cross-check.
        """
        n_chains = RLCT_BATCHED_CONFIG["num_chains"]
        seed = RLCT_BATCHED_CONFIG["seed"]
        pairs_all = tuple((float(a), s) for a in aws_list for s in range(seeds_n))
        fingerprints = {
            int(h): rlct_weight_fingerprint(runs, int(h)) for h in hidden_sizes
        }
        pilot_pairs = ((0.0, 0), (0.0, 1), (50.0, 0), (50.0, 1))

        # 1. validation against devinterp on the four pilot models.
        batched_validation = estimate_rlct_batched(
            512,
            pilot_pairs,
            epochs_n,
            device_type,
            fingerprints[512],
            100.0,
            n_chains,
            int(validation_draws),
            seed,
        )
        if rlct_plateau is not None:
            devinterp_rows = rlct_plateau["models_by_draws"].get(
                str(int(validation_draws)), {}
            )
        else:
            devinterp = estimate_rlct(
                512,
                pilot_pairs,
                epochs_n,
                device_type,
                fingerprints[512],
                100.0,
                n_chains,
                int(validation_draws),
            )
            devinterp_rows = {
                f"{aw_key}:{s}": {"llc": v}
                for aw_key, entry in devinterp.items()
                for s, v in zip(entry["seeds"], entry["llc"])
            }
        validation = _rlct_validation(
            batched_validation, devinterp_rows, validation_tolerance
        )

        # 2. convergence: did the chosen draw count stop moving?
        convergence = {}
        for draws in convergence_draws:
            convergence[str(int(draws))] = estimate_rlct_batched(
                512,
                pilot_pairs,
                epochs_n,
                device_type,
                fingerprints[512],
                100.0,
                n_chains,
                int(draws),
                seed,
            )
        short_key, long_key = (str(int(d)) for d in sorted(convergence_draws))
        convergence_models = []
        convergence_worst = 0.0
        for aw_key, entry in convergence[long_key].items():
            for s, long_value in zip(entry["seeds"], entry["llc"]):
                idx = convergence[short_key][aw_key]["seeds"].index(s)
                short_value = convergence[short_key][aw_key]["llc"][idx]
                rel = (
                    abs(long_value - short_value) / abs(long_value)
                    if long_value
                    else float("inf")
                )
                convergence_worst = max(convergence_worst, rel)
                convergence_models.append(
                    {
                        "aw": float(entry["aw"]),
                        "seed": int(s),
                        short_key: float(short_value),
                        long_key: float(long_value),
                        "rel_diff": float(rel),
                    }
                )
        convergence_check = {
            "draws": [int(d) for d in convergence_draws],
            "models": convergence_models,
            "worst_rel_diff": float(convergence_worst),
            "tolerance": float(convergence_tolerance),
            "passed": bool(convergence_worst <= convergence_tolerance),
        }

        # 3. full loc=100 batched sweep over every hidden size.
        per_hidden = {}
        for h in hidden_sizes:
            per_hidden[str(int(h))] = estimate_rlct_batched(
                int(h),
                pairs_all,
                epochs_n,
                device_type,
                fingerprints[int(h)],
                100.0,
                n_chains,
                4000,
                seed,
            )

        # 4. loc=1000 batched cross-check against the restored devinterp results.
        prior = _load_prior_rlct(RLCT_DATA_PATH)
        restored = prior.get(f"{RLCT_PUBLISHED_LOCALIZATION:g}") or prior.get(
            str(int(RLCT_PUBLISHED_LOCALIZATION))
        )
        cross_check = {}
        cross_worst = 0.0
        for h in hidden_sizes:
            batched_1000 = estimate_rlct_batched(
                int(h),
                pairs_all,
                epochs_n,
                device_type,
                fingerprints[int(h)],
                RLCT_PUBLISHED_LOCALIZATION,
                n_chains,
                200,
                seed,
            )
            cross_check[str(int(h))] = {}
            for aw_key, entry in batched_1000.items():
                reference = (restored or {}).get("rlct", {}).get(
                    str(int(h)), {}
                ).get(aw_key)
                row = {
                    "batched_mean": entry["mean"],
                    "batched_ci": entry["ci"],
                    "devinterp_mean": None,
                    "devinterp_ci": None,
                    "rel_diff": None,
                    "ci_overlap": None,
                }
                if reference is not None:
                    ref_mean, ref_ci = float(reference["mean"]), float(reference["ci"])
                    row["devinterp_mean"] = ref_mean
                    row["devinterp_ci"] = ref_ci
                    row["rel_diff"] = (
                        abs(entry["mean"] - ref_mean) / abs(ref_mean)
                        if ref_mean
                        else float("inf")
                    )
                    row["ci_overlap"] = bool(
                        abs(entry["mean"] - ref_mean) <= entry["ci"] + ref_ci
                    )
                    cross_worst = max(cross_worst, row["rel_diff"])
                cross_check[str(int(h))][aw_key] = row

        return {
            "1000": {
                "rlct": (restored or {}).get("rlct"),
                "hyperparams": (restored or {}).get("hyperparams")
                or rlct_hyperparams(
                    RLCT_PUBLISHED_LOCALIZATION, n_chains, 200, device_type
                ),
                "batched_cross_check": cross_check,
                "cross_check_worst_rel_diff": float(cross_worst),
            },
            "100": {
                "rlct": per_hidden,
                "hyperparams": _rlct_batched_hyperparams(
                    100.0, n_chains, 4000, device_type
                ),
                "validation": validation,
                "convergence_check": convergence_check,
            },
        }


    if rlct_button.value and runs:
        rlct_by_localization = run_rlct_batched_sweep(
            eff_hidden, eff_aws, eff_seeds, eff_epochs, device.type
        )
        print("batched RLCT sweep complete")
    else:
        # Restore whatever was last written so reloading keeps Figure 2C.
        rlct_by_localization = _load_prior_rlct(RLCT_DATA_PATH) or None

    return rlct_by_localization, rlct_hyperparams


@app.cell
def _(
    device,
    eff_epochs,
    estimate_rlct,
    rlct_plateau_button,
    rlct_plateau_config,
    rlct_weight_fingerprint,
    runs,
    time,
):
    def _rlct_plateau_decision(models_by_draws, draws, tolerance):
        """Shortest draw count whose q4 is within ``tolerance`` of q3 for all models."""
        for n in sorted(draws):
            rows = models_by_draws.get(str(n), {})
            if not rows:
                continue
            if all(
                abs(row["loss_quarters"][2]) > 0
                and abs(row["loss_quarters"][3] - row["loss_quarters"][2])
                / abs(row["loss_quarters"][2])
                <= tolerance
                for row in rows.values()
            ):
                return int(n)
        return None


    if rlct_plateau_button.value and runs:
        _pcfg = rlct_plateau_config
        _hidden = _pcfg["hidden"]
        _fingerprint = rlct_weight_fingerprint(runs, _hidden)
        _models_by_draws = {}
        _total_seconds = {}
        for _n in _pcfg["draws"]:
            _started = time.perf_counter()
            _res = estimate_rlct(
                _hidden,
                _pcfg["pairs"],
                eff_epochs,
                device.type,
                _fingerprint,
                float(_pcfg["localization"]),
                int(_pcfg["num_chains"]),
                int(_n),
            )
            _total_seconds[str(_n)] = time.perf_counter() - _started
            _rows = {}
            for _aw_key, _entry in _res.items():
                for _i, _seed in enumerate(_entry["seeds"]):
                    _rows[f"{_aw_key}:{_seed}"] = {
                        "aw": float(_entry["aw"]),
                        "seed": int(_seed),
                        "llc": float(_entry["llc"][_i]),
                        "loss_quarters": [
                            float(_q) for _q in _entry["loss_quarters"][_i]
                        ],
                        "seconds": float(_entry["seconds"][_i]),
                    }
            _models_by_draws[str(_n)] = _rows
            print(f"plateau draws={_n}: {_total_seconds[str(_n)]:.1f}s")
        rlct_plateau = {
            "hidden": _hidden,
            "localization": float(_pcfg["localization"]),
            "num_chains": int(_pcfg["num_chains"]),
            "draws": [int(_n) for _n in _pcfg["draws"]],
            "q3_q4_tolerance": float(_pcfg["tolerance"]),
            "models_by_draws": _models_by_draws,
            "total_seconds": _total_seconds,
            "chosen_draws": _rlct_plateau_decision(
                _models_by_draws, _pcfg["draws"], _pcfg["tolerance"]
            ),
        }
        print(f"plateau chosen draws: {rlct_plateau['chosen_draws']}")
    else:
        rlct_plateau = None

    return (rlct_plateau,)


@app.cell
def _(alt, np, pd, rlct_by_localization, rlct_plateau, t95):
    if rlct_by_localization is None and rlct_plateau is None:
        fig2c = None
    else:
        _colors = {
            "0": "#000000",
            "1": "#ff1493",
            "5": "#00e000",
            "10": "#8a2be2",
            "20": "#ff4500",
            "50": "#00bfff",
        }
        _rows = []
        for _loc_key, _entry in (rlct_by_localization or {}).items():
            for _h, _aw_map in _entry["rlct"].items():
                for _aw, _d in _aw_map.items():
                    _rows.append(
                        {
                            "localization": _loc_key,
                            "hidden": int(_h),
                            "aw": _aw,
                            "mean": _d["mean"],
                            "lo": _d["mean"] - _d["ci"],
                            "hi": _d["mean"] + _d["ci"],
                            "source": "sweep",
                        }
                    )
        if rlct_plateau is not None and (
            f"{rlct_plateau['localization']:g}" not in (rlct_by_localization or {})
        ):
            # No full sweep at this localization yet: plot the pilot models so the
            # localization shows up alongside the longer-restored results.
            _loc_key = f"{rlct_plateau['localization']:g}"
            _chosen = rlct_plateau.get("chosen_draws") or rlct_plateau["draws"][-1]
            _by_aw = {}
            for _row in rlct_plateau["models_by_draws"][str(_chosen)].values():
                _by_aw.setdefault(f"{_row['aw']:g}", []).append(_row["llc"])
            for _aw, _vals in _by_aw.items():
                _arr = np.asarray(_vals, dtype=float)
                _m = float(_arr.mean())
                _c = (
                    0.0
                    if _arr.size == 1
                    else t95(_arr.size - 1)
                    * float(_arr.std(ddof=1))
                    / np.sqrt(_arr.size)
                )
                _rows.append(
                    {
                        "localization": _loc_key,
                        "hidden": int(rlct_plateau["hidden"]),
                        "aw": _aw,
                        "mean": _m,
                        "lo": _m - _c,
                        "hi": _m + _c,
                        "source": "plateau pilot",
                    }
                )
        _df = pd.DataFrame(_rows)
        _keys = [k for k in _colors if k in set(_df["aw"])]
        _scale = alt.Scale(domain=_keys, range=[_colors[k] for k in _keys])
        _color = alt.Color("aw:N", scale=_scale, legend=alt.Legend(title="AW"))
        _x = alt.X(
            "hidden:O",
            title="hidden size",
            sort=sorted(_df["hidden"].unique()),
        )
        _base = alt.Chart(_df).encode(
            x=_x,
            tooltip=["localization:N", "hidden:O", "aw:N", "mean:Q", "source:N"],
        )
        _band = _base.mark_area(opacity=0.15).encode(
            y=alt.Y("lo:Q", title="LLC"), y2="hi:Q", color=_color
        )
        _line = _base.mark_line(point=True).encode(
            y=alt.Y("mean:Q", title="LLC"), color=_color
        )
        _ordered = sorted(_df["localization"].unique(), key=float)
        fig2c = (
            (_band + _line)
            .properties(width=280, height=240, title="Fig 2C — RLCT vs hidden size")
            .facet(
                column=alt.Column(
                    "localization:N", title="localization", sort=_ordered
                )
            )
        )

    fig2c

    return


@app.cell
def _(
    Path,
    RLCT_DATA_PATH,
    device,
    json,
    mo,
    os,
    rlct_by_localization,
    rlct_hyperparams,
    rlct_plateau,
):
    if rlct_by_localization is None and rlct_plateau is None:
        _out = mo.md("_Click **Estimate RLCT** to produce Figure 2C data._")
    else:
        import copy as _copy

        _by_loc = (
            _copy.deepcopy(rlct_by_localization)
            if rlct_by_localization is not None
            else {}
        )
        # Keep localizations written by earlier runs (e.g. the restored
        # localization=1000 results) when this run did not produce them.
        if os.path.exists(RLCT_DATA_PATH):
            try:
                with open(RLCT_DATA_PATH) as _prior_file:
                    _prior_blob = json.load(_prior_file)
                for _key, _value in _prior_blob.get("by_localization", {}).items():
                    _by_loc.setdefault(_key, _value)
            except (OSError, ValueError):
                pass
        if rlct_plateau is not None:
            _plateau_key = f"{rlct_plateau['localization']:g}"
            _slot = _by_loc.setdefault(_plateau_key, {})
            _slot["plateau_check"] = rlct_plateau
            if "rlct" not in _slot:
                # The plateau check ran, but the full sweep at this localization has
                # not (yet). Record that explicitly instead of implying a sweep.
                _slot["rlct"] = None
                _slot["hyperparams"] = rlct_hyperparams(
                    rlct_plateau["localization"],
                    rlct_plateau["num_chains"],
                    rlct_plateau.get("chosen_draws")
                    or rlct_plateau["draws"][-1],
                    device,
                )
                _slot["sweep_status"] = "not run"
        _payload_obj = {"by_localization": _by_loc}
        _payload = json.dumps(_payload_obj).encode("utf-8")
        _loc = mo.notebook_location()
        _msg = "notebook location unavailable"
        if _loc is not None:
            _path = Path(str(_loc)) / "public" / "data" / "mnist_fig2c.json"
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
                    filename="mnist_fig2c.json",
                    mimetype="application/json",
                    label="Download mnist_fig2c.json",
                ),
            ]
        )
    _out

    return


if __name__ == "__main__":
    app.run()
