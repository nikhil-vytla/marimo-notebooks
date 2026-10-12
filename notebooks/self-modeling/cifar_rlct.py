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
#     "devinterp==2.0.1",
#     "zarr==3.1.3",
# ]
# ///
# smoke: skip

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import importlib.metadata
    import json
    import os
    import shutil
    import tempfile
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch import nn

    return (
        F,
        Path,
        alt,
        hashlib,
        importlib,
        json,
        load_dataset,
        mo,
        nn,
        np,
        os,
        pd,
        shutil,
        tempfile,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Self-Modeling — Figure 3B RLCT (CIFAR-10)

    Estimates the local learning coefficient (LLC) of the pruned
    classifier at epoch 250 for each self-modeling weight AW. The paper
    reports baseline ~82, AW 0.5 ~72, AW 1 ~69, AW 2 ~66.

    Input is the **pruned** checkpoint written by `cifar_train.py`: the full
    ResNet-18 network with the self-model output rows removed. Read them from
    `CIFAR_CKPT_DIR` (default: this notebook's `checkpoints/` folder).
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
    ### Conventions

    **Device:** `{device}` — {device_name}

    - **Network.** The pruned checkpoint has no self-model rows, so we rebuild
      the classifier-only ResNet-18 (`10` outputs) and load the weights
      directly. `epoch 250` is the final checkpoint.
    - **Loss.** Mean cross-entropy on the CIFAR-10 **train** split with the
      same per-channel normalization as training and **no augmentation**.
    - **BatchNorm.** `RLCT_BN_MODE=eval` (default) keeps the network in
      `eval()` mode, so BatchNorm uses the frozen running statistics from
      training and they never drift during sampling. `devinterp` calls
      `model.train()` internally; the model class overrides `train()` to force
      `eval()`. `RLCT_BN_MODE=train` instead lets `train()` take effect, so
      BatchNorm uses per-minibatch statistics during both SGLD sampling and
      the init-loss computation (as when `devinterp` is handed a model left
      in train mode); its running statistics may update.
    - **Precision.** bf16 autocast on CUDA (devinterp's custom `loss_fn`
      lets us wrap the forward pass); fp32 on MPS/CPU.
    - **Localization.** Default `100`: the paper's supplement text says `1000`
      but its ResNet calibration figure uses `100`.
    """)
    return


@app.cell
def _(
    F,
    Path,
    hashlib,
    importlib,
    load_dataset,
    mo,
    nn,
    np,
    os,
    shutil,
    tempfile,
    time,
    torch,
):
    N_CLASSES = 10
    HIDDEN_WIDTH = 2000
    POOLED_WIDTH = 512

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

    def mean_ci(values):
        arr = np.asarray(values, dtype=float)
        n = arr.size
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return float(arr[0]), 0.0
        half = t95(n - 1) * float(arr.std(ddof=1)) / np.sqrt(n)
        return float(arr.mean()), float(half)

    class BasicBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_ch, out_ch, 3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(
                out_ch, out_ch, 3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch, 1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            identity = (
                x if self.downsample is None else self.downsample(x)
            )
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

    class CifarResNet18Pruned(nn.Module):
        """ResNet-18 + hidden 2000, classifier rows only (10 logits)."""

        def __init__(self, bn_mode="eval"):
            super().__init__()
            self.bn_mode = bn_mode
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer1 = self._make_layer(64, 64, 2, 1)
            self.layer2 = self._make_layer(64, 128, 2, 2)
            self.layer3 = self._make_layer(128, 256, 2, 2)
            self.layer4 = self._make_layer(256, 512, 2, 2)
            self.hidden = nn.Linear(POOLED_WIDTH, HIDDEN_WIDTH)
            self.out = nn.Linear(HIDDEN_WIDTH, N_CLASSES)

        @staticmethod
        def _make_layer(in_ch, out_ch, blocks, stride):
            layers = [BasicBlock(in_ch, out_ch, stride)]
            layers.extend(
                BasicBlock(out_ch, out_ch, 1) for _ in range(blocks - 1)
            )
            return nn.Sequential(*layers)

        def train(self, mode=True):
            # devinterp calls model.train(); force eval unless bn_mode="train".
            if self.bn_mode == "eval":
                return super().train(False)
            return super().train(mode)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            pooled = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
            hidden = F.relu(self.hidden(pooled))
            return self.out(hidden)

    class IndexDataset(torch.utils.data.Dataset):
        """Rows are (train index, label); the image is looked up on device."""

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

    def load_cifar_train(device, subset_n):
        """uint8 train images plus the training per-channel mean/std."""
        raw = load_dataset("uoft-cs/cifar10", split="train")
        n = len(raw) if subset_n is None else min(int(subset_n), len(raw))
        images = np.empty((n, 3, 32, 32), dtype=np.uint8)
        labels = np.empty((n,), dtype=np.int64)
        for i, example in enumerate(raw):
            if i >= n:
                break
            image = np.asarray(example["img"], dtype=np.uint8)
            images[i] = image.reshape(32, 32, 3).transpose(2, 0, 1)
            labels[i] = example["label"]
        X = torch.from_numpy(images).to(device)
        y = torch.from_numpy(labels).to(device)

        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_sq = torch.zeros(3, dtype=torch.float64)
        total = 0
        for lo in range(0, n, 5000):
            chunk = X[lo:lo + 5000].float().div(255.0).cpu().double()
            channel_sum += chunk.sum(dim=(0, 2, 3))
            channel_sq += chunk.pow(2).sum(dim=(0, 2, 3))
            total += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        mean = (channel_sum / total).float().view(1, 3, 1, 1).to(device)
        var = (
            channel_sq / total - (channel_sum / total).pow(2)
        ).clamp(min=1e-12)
        std = var.sqrt().float().view(1, 3, 1, 1).to(device)
        return X, y, mean, std

    def load_pruned_model(path, device, bn_mode="eval"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = CifarResNet18Pruned(bn_mode=bn_mode)
        model.load_state_dict(state)
        model.eval()
        return model.to(device)

    def parse_pruned_models(pruned_dir):
        """Recover (aw, seed) from `run_aw<aw>_seed<seed>.pt` files."""
        models = []
        for path in sorted(pruned_dir.glob("run_aw*_seed*.pt")):
            body = path.stem.removeprefix("run_")
            aw_text, seed_text = body.split("_seed")
            models.append(
                {
                    "aw": float(aw_text.removeprefix("aw")),
                    "seed": int(seed_text),
                    "path": path,
                }
            )
        return models

    def file_fingerprint(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def loss_quarters(loss_trace):
        """Chain-averaged loss over four equal draw quarters (q1..q4)."""
        arr = np.asarray(loss_trace, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        chain_mean = arr.mean(axis=0)
        edges = np.linspace(0, chain_mean.shape[0], 5).astype(int)
        out = []
        for index in range(4):
            lo, hi = edges[index], edges[index + 1]
            out.append(
                float(chain_mean[lo:hi].mean())
                if hi > lo
                else float("nan")
            )
        return out

    # Single source of truth for every RLCT hyperparameter.
    RLCT_CONFIG = {
        "lr": 1e-4,
        "localization": 100.0,
        "localization_note": (
            "paper supplement text says 1000; its ResNet calibration "
            "figure says 100"
        ),
        "batch_size": 512,
        "nbeta": 512.0 / float(np.log(512.0)),
        "nbeta_convention": "batch/log(batch) with batch=512",
        "num_chains": 4,
        "num_draws": 2000,
        "num_burnin_steps": 0,
        "num_steps_bw_draws": 1,
        "seed": 0,
        "num_init_loss_batches": 97,
        "noise_level": 1.0,
        "llc_weight_decay": 0.0,
        "bounding_box_size": None,
        "sampling_method": "sgmcmc_sgld",
        "bn_mode": "eval",
        "gradient_accumulation_steps": 1,
        "shuffle": True,
        "match_sampling_input_ids_across_chains": True,
        "param_masks": None,
        "save_metrics": False,
        "init_noise": None,
        "calibration_localizations": (100.0, 1000.0),
        "calibration_num_draws": (500, 1000, 2000),
        "calibration_num_chains": 2,
        "calibration_pairs": (
            ("aw0_seed0", 0.0, 0),
            ("aw2_seed0", 2.0, 0),
        ),
        "plateau_tolerance": 0.02,
    }

    def init_batches(subset_n):
        n_train = 50000 if subset_n is None else int(subset_n)
        return max(
            1,
            min(
                RLCT_CONFIG["num_init_loss_batches"],
                n_train // RLCT_CONFIG["batch_size"],
            ),
        )

    def estimate_model(
        path_str,
        aw,
        seed,
        fingerprint,
        device_type,
        num_chains,
        num_draws,
        localization,
        subset_n,
        lr,
        nbeta,
        batch_size,
        num_burnin_steps,
        num_steps_bw_draws,
        init_seed,
        noise_level,
        llc_weight_decay,
        sampling_method,
        bn_mode,
        init_loss_batches,
    ):
        """Run devinterp's llc() once for one pruned checkpoint."""
        from devinterp.slt.llc import llc

        device = torch.device(device_type)
        X, y, mean, std = load_cifar_train(device, subset_n)
        model = load_pruned_model(Path(path_str), device, bn_mode)
        dataset = IndexDataset(y)
        channels_last = device.type == "cuda"

        def _loss_fn(net, input_ids):
            index = input_ids[:, 0].long()
            labels = input_ids[:, 1].long()
            images = X[index].float().div(255.0)
            images = (images - mean) / std
            if channels_last:
                images = images.contiguous(
                    memory_format=torch.channels_last
                )
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    logits = net(images)
            else:
                logits = net(images)
            return F.cross_entropy(
                logits, labels, reduction="none"
            ).unsqueeze(1)

        tmp = tempfile.mkdtemp(prefix="cifar_rlct_")
        started = time.perf_counter()
        try:
            result = llc(
                model,
                dataset,
                {},
                lr=lr,
                n_beta=nbeta,
                param_masks=None,
                loss_fn=_loss_fn,
                num_chains=num_chains,
                num_draws=num_draws,
                batch_size=batch_size,
                num_burnin_steps=num_burnin_steps,
                num_steps_bw_draws=num_steps_bw_draws,
                localization=float(localization),
                num_init_loss_batches=init_loss_batches,
                init_seed=init_seed,
                device=device_type,
                noise_level=noise_level,
                llc_weight_decay=llc_weight_decay,
                bounding_box_size=None,
                sampling_method=sampling_method,
                gradient_accumulation_steps=1,
                shuffle=True,
                match_sampling_input_ids_across_chains=True,
                init_noise=None,
                save_metrics=False,
                output_path=os.path.join(tmp, "samples.zarr"),
            )
            per_chain = np.asarray(
                result["llc_per_chain"].values, dtype=float
            )
            value = float(per_chain.mean())
            quarters = loss_quarters(result["loss_trace"].values)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {
            "aw": float(aw),
            "seed": int(seed),
            "fingerprint": fingerprint,
            "llc": value,
            "loss_quarters": quarters,
            "seconds": time.perf_counter() - started,
        }

    @mo.persistent_cache
    def estimate_model_cached(
        path_str,
        aw,
        seed,
        fingerprint,
        device_type,
        num_chains,
        num_draws,
        localization,
        subset_n,
        lr,
        nbeta,
        batch_size,
        num_burnin_steps,
        num_steps_bw_draws,
        init_seed,
        noise_level,
        llc_weight_decay,
        sampling_method,
        bn_mode,
        init_loss_batches,
    ):
        return estimate_model(
            path_str,
            aw,
            seed,
            fingerprint,
            device_type,
            num_chains,
            num_draws,
            localization,
            subset_n,
            lr,
            nbeta,
            batch_size,
            num_burnin_steps,
            num_steps_bw_draws,
            init_seed,
            noise_level,
            llc_weight_decay,
            sampling_method,
            bn_mode,
            init_loss_batches,
        )

    def plateau_choice(results, localization, tolerance):
        """Shortest draws whose q4 is within tolerance of q3 everywhere."""
        loc_key = f"{localization:g}"
        draw_keys = None
        for per_loc in results.values():
            if loc_key in per_loc:
                draw_keys = sorted(per_loc[loc_key], key=int)
                break
        if not draw_keys:
            return None
        for draws in draw_keys:
            ok = True
            for per_loc in results.values():
                row = per_loc.get(loc_key, {}).get(draws)
                if row is None:
                    ok = False
                    break
                q3, q4 = row["loss_quarters"][2], row["loss_quarters"][3]
                if abs(q3) == 0:
                    ok = False
                    break
                if abs(q4 - q3) / abs(q3) > tolerance:
                    ok = False
                    break
            if ok:
                return int(draws)
        return None

    def rlct_hyperparams(device_type, subset_n, bn_mode):
        return {
            "devinterp_version": importlib.metadata.version("devinterp"),
            "estimator": "devinterp.slt.llc.llc",
            "sampler": "SGLD (sgmcmc_sgld)",
            "loss": (
                "mean cross-entropy on the CIFAR-10 train split, "
                "training normalization, no augmentation"
            ),
            "model": (
                "CIFAR ResNet-18 + hidden 2000, classifier rows only"
            ),
            "epoch": 250,
            "bn_mode": bn_mode,
            "batchnorm": (
                "eval mode, frozen running stats"
                if bn_mode == "eval"
                else "train mode, per-minibatch statistics "
                "(running stats may update)"
            ),
            "lr": RLCT_CONFIG["lr"],
            "localization": RLCT_CONFIG["localization"],
            "localization_note": RLCT_CONFIG["localization_note"],
            "nbeta": RLCT_CONFIG["nbeta"],
            "nbeta_convention": RLCT_CONFIG["nbeta_convention"],
            "batch_size": RLCT_CONFIG["batch_size"],
            "num_chains": RLCT_CONFIG["num_chains"],
            "num_draws": RLCT_CONFIG["num_draws"],
            "num_burnin_steps": RLCT_CONFIG["num_burnin_steps"],
            "num_steps_bw_draws": RLCT_CONFIG["num_steps_bw_draws"],
            "num_init_loss_batches": RLCT_CONFIG["num_init_loss_batches"],
            "seed": RLCT_CONFIG["seed"],
            "noise_level": RLCT_CONFIG["noise_level"],
            "llc_weight_decay": RLCT_CONFIG["llc_weight_decay"],
            "sampling_method": RLCT_CONFIG["sampling_method"],
            "precision": (
                "bf16 autocast on CUDA"
                if device_type == "cuda"
                else "fp32"
            ),
            "device": str(device_type),
            "subset": None if subset_n is None else int(subset_n),
        }

    return (
        HIDDEN_WIDTH,
        N_CLASSES,
        POOLED_WIDTH,
        BasicBlock,
        CifarResNet18Pruned,
        IndexDataset,
        RLCT_CONFIG,
        estimate_model,
        estimate_model_cached,
        file_fingerprint,
        init_batches,
        load_cifar_train,
        load_pruned_model,
        loss_quarters,
        mean_ci,
        parse_pruned_models,
        plateau_choice,
        rlct_hyperparams,
    )


@app.cell
def _(Path, mo, os):
    checkpoint_dir = Path(
        os.environ.get(
            "CIFAR_CKPT_DIR",
            str(Path(str(mo.notebook_location() or ".")) / "checkpoints"),
        )
    )
    pruned_dir = checkpoint_dir / "pruned"
    mo.md(
        f"**Checkpoints:** `{checkpoint_dir}` "
        f"(pruned weights in `{pruned_dir}`)"
    )
    return checkpoint_dir, pruned_dir


@app.cell
def _(RLCT_CONFIG, mo, os):
    _script = mo.app_meta().mode == "script"
    _env_mode = os.environ.get("RLCT_MODE", "").strip().lower()
    mode = _env_mode if _env_mode in ("calibrate", "sweep") else "calibrate"

    _env_subset = os.environ.get("RLCT_SUBSET", "").strip()
    subset_n = int(_env_subset) if _env_subset else None

    _env_bn = os.environ.get("RLCT_BN_MODE", "").strip().lower()
    bn_mode = (
        _env_bn if _env_bn in ("eval", "train") else RLCT_CONFIG["bn_mode"]
    )
    RLCT_CONFIG["bn_mode"] = bn_mode

    _env_cal_locs = os.environ.get("RLCT_CAL_LOCS", "").strip()
    calibration_localizations = (
        tuple(
            float(part)
            for part in _env_cal_locs.split(",")
            if part.strip()
        )
        if _env_cal_locs
        else RLCT_CONFIG["calibration_localizations"]
    )

    _env_cal_models = os.environ.get("RLCT_CAL_MODELS", "").strip()
    calibration_models = (
        tuple(
            part.strip()
            for part in _env_cal_models.split(",")
            if part.strip()
        )
        if _env_cal_models
        else None
    )

    _env_draws = os.environ.get("RLCT_CAL_DRAWS", "").strip()
    if not _env_draws:
        _env_draws = os.environ.get("RLCT_DRAWS", "").strip()
    if _env_draws:
        calibration_draws = tuple(
            int(part) for part in _env_draws.split(",") if part.strip()
        )
    else:
        calibration_draws = RLCT_CONFIG["calibration_num_draws"]

    _env_chains = os.environ.get("RLCT_CHAINS", "").strip()
    calibration_chains = (
        int(_env_chains)
        if _env_chains
        else RLCT_CONFIG["calibration_num_chains"]
    )

    _env_sweep_draws = os.environ.get("RLCT_SWEEP_DRAWS", "").strip()
    sweep_draws = (
        int(_env_sweep_draws)
        if _env_sweep_draws
        else RLCT_CONFIG["num_draws"]
    )
    _env_sweep_chains = os.environ.get("RLCT_SWEEP_CHAINS", "").strip()
    sweep_chains = (
        int(_env_sweep_chains)
        if _env_sweep_chains
        else RLCT_CONFIG["num_chains"]
    )

    run_button = mo.ui.run_button(label="Run RLCT estimation")
    mode_picker = mo.ui.dropdown(
        options=["calibrate", "sweep"], value=mode, label="Mode"
    )
    run_requested = _script or bool(run_button.value)
    selected_mode = mode if _script else str(mode_picker.value)

    _out = mo.vstack(
        [
            mode_picker,
            run_button,
            mo.md(
                f"**Mode:** `{selected_mode}` · **subset:** `{subset_n}` · "
                f"**BatchNorm:** `{bn_mode}` · "
                f"**calibration localizations:** "
                f"`{calibration_localizations}` · "
                f"**calibration draws:** `{calibration_draws}` · "
                f"**calibration chains:** `{calibration_chains}` · "
                f"**calibration models:** `{calibration_models}` · "
                f"**sweep draws:** `{sweep_draws}` · "
                f"**sweep chains:** `{sweep_chains}`\n\n"
                "Set `RLCT_MODE`, `RLCT_SUBSET`, `RLCT_BN_MODE`, "
                "`RLCT_DRAWS`, `RLCT_CAL_LOCS`, `RLCT_CAL_DRAWS`, "
                "`RLCT_CAL_MODELS`, `RLCT_CHAINS`, `RLCT_SWEEP_DRAWS`, "
                "`RLCT_SWEEP_CHAINS`, and `RLCT_OUT` to drive this "
                "headlessly."
            ),
        ]
    )
    _out
    return (
        bn_mode,
        calibration_chains,
        calibration_draws,
        calibration_localizations,
        calibration_models,
        run_requested,
        selected_mode,
        subset_n,
        sweep_chains,
        sweep_draws,
    )


@app.cell
def _(
    RLCT_CONFIG,
    bn_mode,
    calibration_chains,
    calibration_draws,
    calibration_localizations,
    calibration_models,
    device,
    estimate_model,
    file_fingerprint,
    init_batches,
    mo,
    parse_pruned_models,
    plateau_choice,
    pruned_dir,
    run_requested,
    selected_mode,
    subset_n,
):
    if run_requested and selected_mode == "calibrate":
        _models = parse_pruned_models(pruned_dir)
        _by_key = {
            f"aw{_m['aw']:g}_seed{_m['seed']}": _m for _m in _models
        }
        _results = {}
        _lines = []
        _pairs = RLCT_CONFIG["calibration_pairs"]
        if calibration_models is not None:
            _pairs = tuple(
                _pair
                for _pair in _pairs
                if _pair[0] in calibration_models
            )
        for _key, _aw, _seed in _pairs:
            _m = _by_key.get(_key)
            if _m is None:
                raise FileNotFoundError(
                    f"calibration model {_key} not found in {pruned_dir}; "
                    f"expected run_aw{_aw:g}_seed{_seed}.pt"
                )
            _fp = file_fingerprint(_m["path"])
            _per_loc = {}
            for _loc in calibration_localizations:
                _per_draws = {}
                for _draws in calibration_draws:
                    _entry = estimate_model(
                        str(_m["path"]),
                        _m["aw"],
                        _m["seed"],
                        _fp,
                        device.type,
                        calibration_chains,
                        _draws,
                        _loc,
                        subset_n,
                        RLCT_CONFIG["lr"],
                        RLCT_CONFIG["nbeta"],
                        RLCT_CONFIG["batch_size"],
                        RLCT_CONFIG["num_burnin_steps"],
                        RLCT_CONFIG["num_steps_bw_draws"],
                        RLCT_CONFIG["seed"],
                        RLCT_CONFIG["noise_level"],
                        RLCT_CONFIG["llc_weight_decay"],
                        RLCT_CONFIG["sampling_method"],
                        bn_mode,
                        init_batches(subset_n),
                    )
                    _per_draws[str(_draws)] = {
                        "llc": _entry["llc"],
                        "loss_quarters": _entry["loss_quarters"],
                        "seconds": _entry["seconds"],
                    }
                    _lines.append(
                        f"{_key} loc={_loc:g} draws={_draws} "
                        f"llc={_entry['llc']:.4f} "
                        f"q1..q4="
                        f"{[round(_q, 4) for _q in _entry['loss_quarters']]} "
                        f"{_entry['seconds']:.1f}s"
                    )
                _per_loc[str(_loc)] = _per_draws
            _results[_key] = _per_loc
        calibration = {
            "device": device.type,
            "subset": subset_n,
            "num_chains": calibration_chains,
            "bn_mode": bn_mode,
            "localizations": [
                float(_loc) for _loc in calibration_localizations
            ],
            "num_draws": [int(_d) for _d in calibration_draws],
            "plateau_tolerance": RLCT_CONFIG["plateau_tolerance"],
            "plateau_chosen_draws": plateau_choice(
                _results,
                RLCT_CONFIG["localization"],
                RLCT_CONFIG["plateau_tolerance"],
            ),
            "results": _results,
        }
        print("\n".join(_lines))
        _out = mo.md(
            f"Calibrated {len(_results)} models at "
            f"`{calibration['localizations']}` x "
            f"`{calibration['num_draws']}` draws; plateau-chosen draws: "
            f"`{calibration['plateau_chosen_draws']}`."
        )
    else:
        calibration = None
        _out = mo.md("_Set `RLCT_MODE=calibrate` to run calibration._")
    _out
    return (calibration,)


@app.cell
def _(
    RLCT_CONFIG,
    bn_mode,
    device,
    estimate_model_cached,
    file_fingerprint,
    init_batches,
    mean_ci,
    mo,
    parse_pruned_models,
    pruned_dir,
    run_requested,
    selected_mode,
    subset_n,
    sweep_chains,
    sweep_draws,
):
    if run_requested and selected_mode == "sweep":
        _models = parse_pruned_models(pruned_dir)
        if not _models:
            rlct = None
            _out = mo.md(f"No pruned checkpoints found in `{pruned_dir}`.")
        else:
            _entries = []
            with mo.status.progress_bar(
                total=len(_models), title="Fig 3B RLCT sweep"
            ) as _bar:
                for _m in _models:
                    _fp = file_fingerprint(_m["path"])
                    _entry = estimate_model_cached(
                        str(_m["path"]),
                        _m["aw"],
                        _m["seed"],
                        _fp,
                        device.type,
                        sweep_chains,
                        sweep_draws,
                        RLCT_CONFIG["localization"],
                        subset_n,
                        RLCT_CONFIG["lr"],
                        RLCT_CONFIG["nbeta"],
                        RLCT_CONFIG["batch_size"],
                        RLCT_CONFIG["num_burnin_steps"],
                        RLCT_CONFIG["num_steps_bw_draws"],
                        RLCT_CONFIG["seed"],
                        RLCT_CONFIG["noise_level"],
                        RLCT_CONFIG["llc_weight_decay"],
                        RLCT_CONFIG["sampling_method"],
                        bn_mode,
                        init_batches(subset_n),
                    )
                    _entries.append(_entry)
                    _bar.update(
                        subtitle=(
                            f"aw={_m['aw']:g} seed={_m['seed']} "
                            f"llc={_entry['llc']:.3f} "
                            f"{_entry['seconds']:.1f}s"
                        )
                    )
            rlct = {}
            for _entry in _entries:
                _aw_key = f"{_entry['aw']:g}"
                _slot = rlct.setdefault(
                    _aw_key,
                    {
                        "aw": _entry["aw"],
                        "llc": [],
                        "seeds": [],
                        "seconds": [],
                    },
                )
                _slot["llc"].append(_entry["llc"])
                _slot["seeds"].append(_entry["seed"])
                _slot["seconds"].append(_entry["seconds"])
            for _slot in rlct.values():
                _slot["mean"], _slot["ci"] = mean_ci(_slot["llc"])
            print(
                "RLCT: "
                + " | ".join(
                    f"aw={_k}: {_v['mean']:.2f}+/-{_v['ci']:.2f}"
                    for _k, _v in sorted(
                        rlct.items(), key=lambda kv: float(kv[0])
                    )
                )
            )
            _out = mo.md(
                f"Estimated {len(_entries)} models from `{pruned_dir}`."
            )
    else:
        rlct = None
        _out = mo.md("_Set `RLCT_MODE=sweep` to run the sweep._")
    _out
    return (rlct,)


@app.cell
def _(
    Path,
    bn_mode,
    calibration,
    device,
    json,
    mo,
    os,
    rlct,
    rlct_hyperparams,
    run_requested,
    subset_n,
):
    if run_requested and (rlct is not None or calibration is not None):
        _loc = mo.notebook_location()
        _default = (
            str(
                Path(str(_loc))
                / "public"
                / "data"
                / "cifar_fig3b.json"
            )
            if _loc is not None
            else None
        )
        _path_str = os.environ.get("RLCT_OUT", "").strip() or _default
        if _path_str is None:
            json_path = None
            _msg = "no output path (notebook location unavailable)"
        else:
            json_path = Path(_path_str)
            _prior_cal = None
            if calibration is None and json_path.exists():
                try:
                    _prior_cal = json.loads(
                        json_path.read_text()
                    ).get("calibration")
                except (OSError, ValueError):
                    _prior_cal = None
            _payload = {
                "rlct": rlct,
                "hyperparams": rlct_hyperparams(
                    device.type, subset_n, bn_mode
                ),
                "calibration": (
                    calibration if calibration is not None else _prior_cal
                ),
            }
            _bytes = json.dumps(_payload, indent=2).encode("utf-8")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_bytes(_bytes)
            _msg = f"wrote `{json_path}` ({len(_bytes) / 1024:.1f} KiB)"
        _out = mo.md(_msg)
    else:
        json_path = None
        _out = mo.md("_No results to write._")
    _out
    return (json_path,)


@app.cell
def _(alt, mo, pd, rlct):
    PAPER_VALUES = {"0": 82.0, "0.5": 72.0, "1": 69.0, "2": 66.0}
    COLORS = {
        "0": "#000000",
        "0.5": "#00e000",
        "1": "#0000ff",
        "2": "#a52a2a",
    }
    if rlct is None:
        chart = None
        _out = mo.md("_Run the sweep to chart Fig 3B._")
    else:
        _domain = [k for k in ("0", "0.5", "1", "2") if k in rlct]
        _rows = [
            {
                "aw": _k,
                "mean": _v["mean"],
                "lo": _v["mean"] - _v["ci"],
                "hi": _v["mean"] + _v["ci"],
            }
            for _k, _v in rlct.items()
        ]
        _df = pd.DataFrame(_rows)
        _color = alt.Color(
            "aw:N",
            scale=alt.Scale(
                domain=_domain, range=[COLORS[_k] for _k in _domain]
            ),
            legend=alt.Legend(title="AW"),
        )
        _x = alt.X("aw:N", title="AW", sort=_domain)
        _base = alt.Chart(_df).encode(x=_x, color=_color)
        _bars = _base.mark_bar().encode(
            y=alt.Y("mean:Q", title="RLCT (LLC)")
        )
        _err = _base.mark_errorbar().encode(
            y=alt.Y("lo:Q"), y2="hi:Q"
        )
        _paper_df = pd.DataFrame(
            [
                {"aw": _k, "paper": _v}
                for _k, _v in PAPER_VALUES.items()
                if _k in _domain
            ]
        )
        _paper = (
            alt.Chart(_paper_df)
            .mark_point(
                shape="diamond", filled=True, size=90, color="#ff8c00"
            )
            .encode(
                x=alt.X("aw:N", sort=_domain),
                y=alt.Y("paper:Q", title="RLCT (LLC)"),
            )
        )
        chart = (_bars + _err + _paper).properties(
            width=360, height=300, title="Fig 3B — RLCT by AW"
        )
        _out = chart
    _out
    return (chart,)


@app.cell
def _(json_path, mo, rlct):
    _paper = {"0": 82.0, "0.5": 72.0, "1": 69.0, "2": 66.0}
    _where = "—" if json_path is None else f"`{json_path}`"
    if rlct is None:
        _out = mo.md(
            f"_Paper values vs. estimates appear after a sweep. Output: "
            f"{_where}._"
        )
    else:
        _lines = [
            "## Paper comparison",
            "",
            f"Output: {_where}",
            "",
            "| AW | paper | estimate (mean ± CI) | n |",
            "| --- | --- | --- | --- |",
        ]
        for _k in ("0", "0.5", "1", "2"):
            if _k not in rlct:
                continue
            _v = rlct[_k]
            _lines.append(
                f"| {_k} | {_paper[_k]:g} | "
                f"{_v['mean']:.1f} ± {_v['ci']:.1f} | "
                f"{len(_v['llc'])} |"
            )
        _out = mo.md("\n".join(_lines))
    _out
    return


if __name__ == "__main__":
    app.run()
