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
    import contextlib
    import json
    import math
    import os
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
        contextlib,
        json,
        load_dataset,
        math,
        mo,
        nn,
        np,
        os,
        pd,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Self-Modeling in Neural Systems — Figure 3 training (CIFAR-10)

    Reproduces Figure 3 (A and C) of
    [*Unexpected Benefits of Self-Modeling in Neural Systems*](https://arxiv.org/abs/2407.10188).

    A CIFAR-style ResNet-18 produces class logits plus extra output rows that
    predict a detached target of its own internal activations. **AW** is the
    weight on that self-modeling loss; **AW = 0** is the baseline. Heavy work
    only runs when you click **Train**, and every run checkpoints to disk so a
    restarted session resumes instead of starting over.
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
    mo.md("""
    ## Assumptions where the paper is underspecified

    - The paper says the self-model target is the concatenation of the pooled
      block-4 output and the hidden activations, and separately mentions "skip
      connections from the third block to the hidden layer". A standard
      ResNet-18 has no such skip, so this notebook uses only the stated target:
      `[pooled block-4 (512), hidden activations (2000)]` = 2512 values,
      detached. `CIFAR_TARGET=hidden` drops the pooled part and
      `CIFAR_DETACH=0` lets gradients flow through the target too.
    - Every run (including AW = 0) has the same output layer,
      `Linear(2000, 10 + target_dim)` (`target_dim` = 2512 for `concat`, 2000
      for `hidden`), so seeds share identical classification rows.
      AW = 0 simply drops the self-modeling term; the extra rows are ignored.
    - Augmentation follows the usual RandomAffine recipe: rotation in
      `[-15, 15]` degrees, translation up to 10% of the image, horizontal flip.
      It is done in torch on the GPU (no torchvision dependency).
    """)
    return


@app.cell
def _(device, mo):
    _compile_default = device.type == "cuda"
    aws = mo.ui.multiselect(
        options=[0, 0.5, 1, 2],
        value=[0, 0.5, 1, 2],
        label="Self-modeling weight (AW)",
    )
    seeds = mo.ui.number(value=10, start=1, stop=50, label="Seeds")
    epochs = mo.ui.number(value=250, start=1, stop=1000, label="Epochs")
    concurrent = mo.ui.number(
        value=4, start=1, stop=16, label="Concurrent runs"
    )
    checkpoint_every = mo.ui.number(
        value=10, start=1, stop=50, label="Checkpoint every (epochs)"
    )
    smoke = mo.ui.checkbox(
        value=False, label="Smoke test (2 epochs, 1 seed, 2000 images)"
    )
    train_button = mo.ui.run_button(label="Train")
    compile_flag = mo.ui.checkbox(
        value=_compile_default, label="torch.compile"
    )
    compile_mode = mo.ui.dropdown(
        options=["reduce-overhead", "max-autotune"],
        value="reduce-overhead",
        label="Compile mode",
    )
    detach_target = mo.ui.checkbox(
        value=True, label="Detach self-model target (CIFAR_DETACH)"
    )
    target_mode = mo.ui.dropdown(
        options=["concat", "hidden"],
        value="concat",
        label="Self-model target (CIFAR_TARGET)",
    )
    batch_size = mo.ui.number(
        value=512, start=1, stop=4096, label="Batch size (CIFAR_BATCH)"
    )
    return (
        aws,
        batch_size,
        checkpoint_every,
        compile_flag,
        compile_mode,
        concurrent,
        detach_target,
        epochs,
        seeds,
        smoke,
        target_mode,
        train_button,
    )


@app.cell
def _(
    aws,
    batch_size,
    checkpoint_every,
    compile_flag,
    compile_mode,
    concurrent,
    detach_target,
    epochs,
    mo,
    seeds,
    smoke,
    target_mode,
    train_button,
):
    mo.md(
        """
    ### Configuration

    Adam `lr=1e-3`, no schedule, no weight decay, batch 512 by default
    (`CIFAR_BATCH`). Each run trains one ResNet-18 for the chosen number of
    epochs. `len(AW) * seeds` runs are interleaved in groups of *Concurrent
    runs*; finished runs are skipped on restart.

    `torch.compile` defaults on for CUDA and off elsewhere; `reduce-overhead`
    captures CUDA graphs (fastest for this fixed-shape loop) and `max-autotune`
    spends longer tuning kernels for a possibly better steady state.
    """
    )
    mo.vstack(
        [
            mo.hstack([aws, seeds], justify="start", gap=2),
            mo.hstack([epochs, concurrent, checkpoint_every], justify="start", gap=2),
            mo.hstack([compile_flag, compile_mode], justify="start", gap=2),
            mo.hstack(
                [detach_target, target_mode, batch_size],
                justify="start",
                gap=2,
            ),
            mo.hstack([smoke, train_button], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(
    aws,
    batch_size,
    checkpoint_every,
    compile_flag,
    compile_mode,
    concurrent,
    detach_target,
    epochs,
    mo,
    os,
    seeds,
    smoke,
    target_mode,
    train_button,
):
    _mode = mo.app_meta().mode
    _script = _mode == "script"
    _smoke_env = os.environ.get("SMOKE") == "1"
    smoke_test = bool(smoke.value) or (_script and _smoke_env)

    eff_aws = [float(a) for a in aws.value]
    eff_seeds = 1 if smoke_test else int(seeds.value)
    eff_epochs = 2 if smoke_test else int(epochs.value)
    eff_concurrent = 1 if smoke_test else int(concurrent.value)
    eff_ckpt_every = 1 if smoke_test else int(checkpoint_every.value)
    eff_compile = bool(compile_flag.value)
    eff_compile_mode = str(compile_mode.value)
    eff_detach = bool(detach_target.value)
    eff_target = str(target_mode.value)
    eff_batch = max(1, int(batch_size.value))

    if _script:
        _env_aws = os.environ.get("CIFAR_AWS", "")
        _parsed = [float(a) for a in _env_aws.split(",") if a.strip()]
        if _parsed:
            eff_aws = _parsed
        for _name, _target in (
            ("CIFAR_SEEDS", "seeds"),
            ("CIFAR_EPOCHS", "epochs"),
            ("CIFAR_CONCURRENT", "concurrent"),
            ("CIFAR_CKPT_EVERY", "ckpt_every"),
        ):
            _value = os.environ.get(_name)
            if not _value:
                continue
            if _target == "seeds":
                eff_seeds = int(_value)
            elif _target == "epochs":
                eff_epochs = int(_value)
            elif _target == "concurrent":
                eff_concurrent = int(_value)
            else:
                eff_ckpt_every = int(_value)
        _compile_env = os.environ.get("CIFAR_COMPILE")
        if _compile_env is not None:
            eff_compile = _compile_env.strip().lower() not in (
                "", "0", "false", "off", "no"
            )
        _compile_mode_env = os.environ.get("CIFAR_COMPILE_MODE")
        if _compile_mode_env:
            eff_compile_mode = _compile_mode_env.strip()
        _detach_env = os.environ.get("CIFAR_DETACH")
        if _detach_env is not None:
            eff_detach = _detach_env.strip().lower() not in (
                "0", "false", "off", "no"
            )
        _target_env = os.environ.get("CIFAR_TARGET")
        if _target_env:
            eff_target = _target_env.strip()
        _batch_env = os.environ.get("CIFAR_BATCH")
        if _batch_env:
            eff_batch = max(1, int(_batch_env))

    if eff_target not in ("concat", "hidden"):
        raise ValueError(
            f"CIFAR_TARGET must be 'concat' or 'hidden', got {eff_target!r}"
        )

    train_clicked = bool(train_button.value) or _script
    return (
        eff_aws,
        eff_batch,
        eff_ckpt_every,
        eff_compile,
        eff_compile_mode,
        eff_concurrent,
        eff_detach,
        eff_epochs,
        eff_seeds,
        eff_target,
        smoke_test,
        train_clicked,
    )


@app.cell
def _(Path, mo, os):
    _default = Path(str(mo.notebook_location() or ".")) / "checkpoints"
    checkpoint_dir = Path(os.environ.get("CIFAR_CKPT_DIR", str(_default)))
    return (checkpoint_dir,)


@app.cell
def _(
    F,
    Path,
    contextlib,
    load_dataset,
    math,
    mo,
    nn,
    np,
    os,
    time,
    torch,
):
    SMOKE_TRAIN_N = 2000
    SMOKE_TEST_N = 2000
    BATCH_SIZE = 512
    LR = 1e-3
    HIDDEN_WIDTH = 2000
    N_CLASSES = 10
    POOLED_WIDTH = 512
    SELF_MODEL_SIZE = POOLED_WIDTH + HIDDEN_WIDTH
    TARGET_DIMS = {"concat": SELF_MODEL_SIZE, "hidden": HIDDEN_WIDTH}
    PROBE_N = 2048
    ROTATION_DEG = 15.0
    TRANSLATION = 0.10
    CHANNELS_LAST = "cuda"

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
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            identity = x if self.downsample is None else self.downsample(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

    class CifarResNet18SelfModel(nn.Module):
        """CIFAR ResNet-18 -> hidden 2000 -> logits + self-model rows."""

        def __init__(self, target="concat"):
            super().__init__()
            if target not in TARGET_DIMS:
                raise ValueError(
                    f"unknown self-model target {target!r}; "
                    f"expected one of {sorted(TARGET_DIMS)}"
                )
            self.target = target
            self.self_model_rows = TARGET_DIMS[target]
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
            self.out = nn.Linear(HIDDEN_WIDTH, N_CLASSES + self.self_model_rows)

        @staticmethod
        def _make_layer(in_ch, out_ch, blocks, stride):
            layers = [BasicBlock(in_ch, out_ch, stride)]
            layers.extend(BasicBlock(out_ch, out_ch, 1) for _ in range(blocks - 1))
            return nn.Sequential(*layers)

        def extract(self, x):
            """Return (pooled block-4 features, hidden activations)."""
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            pooled = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
            hidden = F.relu(self.hidden(pooled))
            return pooled, hidden

        def forward(self, x):
            pooled, hidden = self.extract(x)
            out = self.out(hidden)
            logits = out[:, :N_CLASSES]
            a_hat = out[:, N_CLASSES:]
            if self.target == "hidden":
                a = hidden
            else:
                a = torch.cat([pooled, hidden], dim=1)
            return logits, a_hat, a

    def build_model(seed, device, target="concat"):
        torch.manual_seed(seed)
        model = CifarResNet18SelfModel(target=target).to(device)
        if device.type == CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        return model

    def classifier_sd(model):
        weights = model.out.weight[:N_CLASSES].detach().float()
        return float(weights.std())

    def classifier_wnorm(model):
        weights = model.out.weight[:N_CLASSES].detach().float()
        return float(torch.linalg.vector_norm(weights))

    def prune_state_dict(model):
        """Whole-network weights with the self-model rows removed."""
        pruned = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("out.")
        }
        pruned["out.weight"] = model.out.weight[:N_CLASSES].detach().cpu().clone()
        pruned["out.bias"] = model.out.bias[:N_CLASSES].detach().cpu().clone()
        return pruned

    def load_cifar(device, train_limit=None, test_limit=None):
        raw = load_dataset("uoft-cs/cifar10")

        def _to_tensors(split, limit):
            data = raw[split]
            n = len(data) if limit is None else min(limit, len(data))
            images = np.empty((n, 3, 32, 32), dtype=np.uint8)
            labels = np.empty((n,), dtype=np.int64)
            for i, example in enumerate(data):
                if i >= n:
                    break
                image = np.asarray(example["img"], dtype=np.uint8)
                images[i] = image.reshape(32, 32, 3).transpose(2, 0, 1)
                labels[i] = example["label"]
            return torch.from_numpy(images), torch.from_numpy(labels)

        X_train, y_train = _to_tensors("train", train_limit)
        X_test, y_test = _to_tensors("test", test_limit)

        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_sq = torch.zeros(3, dtype=torch.float64)
        total = 0
        for lo in range(0, X_train.shape[0], 5000):
            chunk = X_train[lo:lo + 5000].float().div(255.0)
            channel_sum += chunk.sum(dim=(0, 2, 3)).double()
            channel_sq += chunk.pow(2).sum(dim=(0, 2, 3)).double()
            total += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
        mean = channel_sum / total
        var = (channel_sq / total - mean.pow(2)).clamp(min=1e-12)
        mean = mean.float().view(1, 3, 1, 1).to(device)
        std = var.sqrt().float().view(1, 3, 1, 1).to(device)
        return (
            X_train.to(device),
            y_train.to(device),
            X_test.to(device),
            y_test.to(device),
            mean,
            std,
        )

    def augment_batch(images, generator):
        """Random affine (rotation, translation) + hflip on uint8 images."""
        x = images.float().div(255.0)
        b = x.shape[0]
        angle = (torch.rand(b, generator=generator) * 2.0 - 1.0) * math.radians(
            ROTATION_DEG
        )
        offset = 2.0 * TRANSLATION
        tx = (torch.rand(b, generator=generator) * 2.0 - 1.0) * offset
        ty = (torch.rand(b, generator=generator) * 2.0 - 1.0) * offset
        flip = torch.rand(b, generator=generator) < 0.5
        cos, sin = angle.cos(), angle.sin()
        theta = torch.zeros(b, 2, 3)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 1, 2] = ty
        theta[flip, 0, 0] *= -1.0
        theta[flip, 0, 1] *= -1.0
        theta = theta.to(x.device)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    def normalize(x, mean, std):
        return (x - mean) / std

    @torch.no_grad()
    def activation_stats(model, X, mean, std, batch_size=BATCH_SIZE):
        """Mean |activation| of hidden and pooled features on a probe batch."""
        model.eval()
        hidden_abs = 0.0
        pooled_abs = 0.0
        n = X.shape[0]
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            xb = normalize(X[lo:hi].float().div(255.0), mean, std)
            if xb.device.type == CHANNELS_LAST:
                xb = xb.contiguous(memory_format=torch.channels_last)
            pooled, hidden = model.extract(xb)
            hidden_abs += float(hidden.abs().sum())
            pooled_abs += float(pooled.abs().sum())
        model.train()
        return hidden_abs / (n * HIDDEN_WIDTH), pooled_abs / (n * POOLED_WIDTH)

    def autocast_context(device):
        if device.type == CHANNELS_LAST:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def strip_orig_mod(state_dict):
        """Drop the '_orig_mod.' prefix added by torch.compile wrappers.

        Let an uncompiled run resume a checkpoint saved from a compiled model
        (or the reverse) without a key mismatch.
        """
        prefix = "_orig_mod."
        return {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
        }

    def build_train_step(model, aw, compile_enabled, compile_mode, detach=True):
        """Forward + classification + self-model loss for one batch.

        `dynamic=False` keeps each batch shape on its own static graph; the
        last partial batch (336 images for batch 512) therefore costs one extra
        graph but stays in training instead of being dropped.
        """

        def step(xb, yb):
            logits, a_hat, a = model(xb)
            ce = F.cross_entropy(logits, yb)
            target = a.detach() if detach else a
            sm = (a_hat - target).pow(2).mean()
            return ce + aw * sm

        if not compile_enabled:
            return step
        return torch.compile(step, mode=compile_mode, dynamic=False)

    @torch.no_grad()
    def test_accuracy(model, X, y, mean, std, batch_size=BATCH_SIZE):
        model.eval()
        correct = 0
        for lo in range(0, X.shape[0], batch_size):
            hi = min(lo + batch_size, X.shape[0])
            xb = normalize(X[lo:hi].float().div(255.0), mean, std)
            if xb.device.type == CHANNELS_LAST:
                xb = xb.contiguous(memory_format=torch.channels_last)
            logits, _, _ = model(xb)
            correct += int((logits.argmax(dim=1) == y[lo:hi]).sum())
        model.train()
        return correct / X.shape[0]

    def _atomic_save(obj, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_checkpoint(
        path, aw, seed, epoch, target_epochs, model, optimizer, generator,
        history, finished, detach, target, batch_size,
    ):
        _atomic_save(
            {
                "aw": aw,
                "seed": seed,
                "epoch": epoch,
                "target_epochs": target_epochs,
                "detach": detach,
                "target": target,
                "batch_size": batch_size,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(),
                "history": history,
                "finished": finished,
            },
            path,
        )

    def run_experiment(cfg):
        device = cfg["device"]
        aws = cfg["aws"]
        seeds_n = cfg["seeds"]
        epochs = cfg["epochs"]
        concurrent = max(1, int(cfg["concurrent"]))
        ckpt_every = max(1, int(cfg["ckpt_every"]))
        smoke = cfg["smoke"]
        ckpt_dir = Path(cfg["ckpt_dir"])
        compile_enabled = bool(cfg.get("compile", False))
        compile_mode = str(cfg.get("compile_mode", "reduce-overhead"))
        detach = bool(cfg.get("detach", True))
        target = str(cfg.get("target", "concat"))
        batch_size = max(1, int(cfg.get("batch_size", BATCH_SIZE)))
        use_cudagraphs = (
            compile_enabled
            and compile_mode == "reduce-overhead"
            and device.type == "cuda"
        )

        X_train, y_train, X_test, y_test, mean, std = load_cifar(
            device,
            SMOKE_TRAIN_N if smoke else None,
            SMOKE_TEST_N if smoke else None,
        )
        n_train = X_train.shape[0]
        n_steps = (n_train + batch_size - 1) // batch_size
        probe_n = min(PROBE_N, X_test.shape[0])
        X_probe = X_test[:probe_n]

        states = []
        for aw in aws:
            for seed in range(seeds_n):
                aw = float(aw)
                key = f"aw{aw:g}_seed{seed}"
                path = ckpt_dir / f"run_{key}.pt"
                checkpoint = None
                if path.exists():
                    checkpoint = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                if checkpoint is not None:
                    _stored = (
                        bool(checkpoint.get("detach", True)),
                        str(checkpoint.get("target", "concat")),
                        int(checkpoint.get("batch_size", BATCH_SIZE)),
                    )
                    if _stored != (detach, target, batch_size):
                        raise ValueError(
                            f"Checkpoint {path} was trained with "
                            f"detach={_stored[0]}, target={_stored[1]!r}, "
                            f"batch={_stored[2]}; requested detach={detach}, "
                            f"target={target!r}, batch={batch_size}. "
                            "Refusing to resume; point CIFAR_CKPT_DIR at a "
                            "separate directory for probe runs."
                        )
                if checkpoint is not None and checkpoint["epoch"] >= epochs:
                    _history = checkpoint["history"]
                    for _field in ("hidden_absmean", "pooled_absmean", "cls_wnorm"):
                        _history.setdefault(_field, [])
                    states.append(
                        {
                            "key": key,
                            "aw": aw,
                            "seed": seed,
                            "model": None,
                            "train_step": None,
                            "optimizer": None,
                            "generator": None,
                            "history": _history,
                            "epoch": int(checkpoint["epoch"]),
                            "path": path,
                            "resumed": True,
                            "finished": True,
                            "epoch_seconds": [],
                        }
                    )
                    continue

                model = build_model(seed, device, target)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LR, fused=device.type == "cuda"
                )
                train_step = build_train_step(
                    model, aw, compile_enabled, compile_mode, detach
                )
                generator = torch.Generator()
                resumed = checkpoint is not None
                if resumed:
                    model.load_state_dict(
                        strip_orig_mod(checkpoint["model"])
                    )
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    generator.set_state(checkpoint["generator"])
                    history = checkpoint["history"]
                    start_epoch = int(checkpoint["epoch"])
                else:
                    generator.manual_seed(seed)
                    history = {
                        "sd": [classifier_sd(model)],
                        "test_acc": [],
                        "hidden_absmean": [],
                        "pooled_absmean": [],
                        "cls_wnorm": [],
                    }
                    start_epoch = 0
                for _field in ("hidden_absmean", "pooled_absmean", "cls_wnorm"):
                    history.setdefault(_field, [])
                states.append(
                    {
                        "key": key,
                        "aw": aw,
                        "seed": seed,
                        "model": model,
                        "train_step": train_step,
                        "optimizer": optimizer,
                        "generator": generator,
                        "history": history,
                        "epoch": start_epoch,
                        "path": path,
                        "resumed": resumed,
                        "finished": False,
                        "epoch_seconds": [],
                    }
                )

        active = [s for s in states if not s["finished"]]
        total_steps = sum(max(0, epochs - s["epoch"]) for s in active)
        done = 0

        if total_steps == 0:
            print("All runs already finished; nothing to train.")
        else:
            with mo.status.progress_bar(
                total=total_steps, title="CIFAR-10 self-modeling"
            ) as bar:
                for group_start in range(0, len(active), concurrent):
                    group = active[group_start:group_start + concurrent]
                    for epoch in range(1, epochs + 1):
                        step_states = [s for s in group if s["epoch"] < epoch]
                        if not step_states:
                            continue
                        permutations = {
                            s["key"]: torch.randperm(
                                n_train, generator=s["generator"]
                            )
                            for s in step_states
                        }
                        if device.type == "mps":
                            torch.mps.synchronize()
                        start = time.perf_counter()
                        for step in range(n_steps):
                            lo = step * batch_size
                            hi = min(lo + batch_size, n_train)
                            for s in step_states:
                                index = permutations[s["key"]][lo:hi]
                                xb = augment_batch(
                                    X_train[index], s["generator"]
                                )
                                xb = normalize(xb, mean, std)
                                if device.type == CHANNELS_LAST:
                                    xb = xb.contiguous(
                                        memory_format=torch.channels_last
                                    )
                                yb = y_train[index]
                                s["optimizer"].zero_grad(set_to_none=True)
                                if use_cudagraphs:
                                    torch.compiler.cudagraph_mark_step_begin()
                                with autocast_context(device):
                                    loss = s["train_step"](xb, yb)
                                loss.backward()
                                s["optimizer"].step()
                        if device.type == "mps":
                            torch.mps.synchronize()
                        elapsed = time.perf_counter() - start
                        per_model_seconds = elapsed / len(step_states)
                        for s in step_states:
                            accuracy = test_accuracy(
                                s["model"], X_test, y_test, mean, std,
                                batch_size=batch_size,
                            )
                            _hidden_absmean, _pooled_absmean = activation_stats(
                                s["model"], X_probe, mean, std, batch_size
                            )
                            s["history"]["sd"].append(classifier_sd(s["model"]))
                            s["history"]["test_acc"].append(accuracy)
                            s["history"]["hidden_absmean"].append(
                                _hidden_absmean
                            )
                            s["history"]["pooled_absmean"].append(
                                _pooled_absmean
                            )
                            s["history"]["cls_wnorm"].append(
                                classifier_wnorm(s["model"])
                            )
                            s["epoch"] = epoch
                            s["epoch_seconds"].append(per_model_seconds)
                            finished = epoch >= epochs
                            if epoch % ckpt_every == 0 or finished:
                                print(
                                    f"{s['key']} epoch {epoch}/{epochs} | "
                                    f"{per_model_seconds:.2f} s/epoch | "
                                    f"acc={accuracy:.4f} | "
                                    f"hidden_absmean={_hidden_absmean:.4f} | "
                                    f"pooled_absmean={_pooled_absmean:.4f} | "
                                    f"cls_wnorm={s['history']['cls_wnorm'][-1]:.4f}"
                                )
                                save_checkpoint(
                                    s["path"], s["aw"], s["seed"], epoch,
                                    epochs, s["model"], s["optimizer"],
                                    s["generator"], s["history"], finished,
                                    detach, target, batch_size,
                                )
                                if finished:
                                    _atomic_save(
                                        prune_state_dict(s["model"]),
                                        ckpt_dir / "pruned" / f"run_{s['key']}.pt",
                                    )
                            done += 1
                            bar.update(
                                subtitle=(
                                    f"{s['key']} epoch {epoch}/{epochs} | "
                                    f"{per_model_seconds:.2f} s/epoch | "
                                    f"{done}/{total_steps} run-epochs"
                                )
                            )

        results = {}
        runs = {}

        def _epoch_means(aw_states, field):
            means = []
            for index in range(epochs):
                values = [
                    s["history"][field][index]
                    for s in aw_states
                    if index < len(s["history"][field])
                ]
                m, _ = mean_ci(values)
                means.append(m)
            return means

        for aw in aws:
            aw = float(aw)
            aw_states = [s for s in states if abs(s["aw"] - aw) < 1e-12]
            sd_mean, sd_ci = [], []
            for index in range(epochs + 1):
                values = [
                    s["history"]["sd"][index]
                    for s in aw_states
                    if index < len(s["history"]["sd"])
                ]
                m, c = mean_ci(values)
                sd_mean.append(m)
                sd_ci.append(c)
            accs = [
                s["history"]["test_acc"][-1]
                for s in aw_states
                if s["history"]["test_acc"]
            ]
            acc_mean, acc_ci = mean_ci(accs)
            results[f"{aw:g}"] = {
                "aw": aw,
                "n": len(aw_states),
                "sd_mean": sd_mean,
                "sd_ci": sd_ci,
                "hidden_absmean_mean": _epoch_means(
                    aw_states, "hidden_absmean"
                ),
                "pooled_absmean_mean": _epoch_means(
                    aw_states, "pooled_absmean"
                ),
                "cls_wnorm_mean": _epoch_means(aw_states, "cls_wnorm"),
                "acc_mean": acc_mean,
                "acc_ci": acc_ci,
            }
            for s in aw_states:
                seconds = s["epoch_seconds"]
                _final_acc = (
                    s["history"]["test_acc"][-1]
                    if s["history"]["test_acc"]
                    else None
                )
                runs[s["key"]] = {
                    "aw": aw,
                    "seed": s["seed"],
                    "final_acc": _final_acc,
                    "final_sd": s["history"]["sd"][-1],
                    "final_hidden_absmean": (
                        s["history"]["hidden_absmean"][-1]
                        if s["history"]["hidden_absmean"]
                        else None
                    ),
                    "final_pooled_absmean": (
                        s["history"]["pooled_absmean"][-1]
                        if s["history"]["pooled_absmean"]
                        else None
                    ),
                    "final_cls_wnorm": (
                        s["history"]["cls_wnorm"][-1]
                        if s["history"]["cls_wnorm"]
                        else None
                    ),
                    "mean_epoch_seconds": (
                        sum(seconds) / len(seconds) if seconds else None
                    ),
                    "resumed": s["resumed"],
                    "finished": s["finished"],
                }

        return {
            "config": {
                "aws": [float(a) for a in aws],
                "seeds": seeds_n,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": LR,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "torch_compile": compile_enabled,
                "compile_mode": compile_mode if compile_enabled else None,
                "concurrent_runs": concurrent,
                "checkpoint_interval": ckpt_every,
                "checkpoint_dir": str(ckpt_dir),
                "hidden_width": HIDDEN_WIDTH,
                "detach": detach,
                "target": target,
                "target_dim": TARGET_DIMS[target],
                "self_model_size": TARGET_DIMS[target],
                "augmentation": {
                    "rotation_deg": ROTATION_DEG,
                    "translation": TRANSLATION,
                    "horizontal_flip": True,
                },
                "normalization": "per-channel train mean/std",
                "smoke": smoke,
            },
            "device": {"type": device.type, "name": cfg["device_name"]},
            "epochs": epochs,
            "results": results,
            "runs": runs,
        }

    return (run_experiment,)


@app.cell
def _(
    checkpoint_dir,
    device,
    device_name,
    eff_aws,
    eff_batch,
    eff_ckpt_every,
    eff_compile,
    eff_compile_mode,
    eff_concurrent,
    eff_detach,
    eff_epochs,
    eff_seeds,
    eff_target,
    mo,
    run_experiment,
    smoke_test,
    time,
    train_clicked,
):
    if train_clicked and eff_aws:
        _start = time.perf_counter()
        fig3 = run_experiment(
            {
                "aws": eff_aws,
                "seeds": eff_seeds,
                "epochs": eff_epochs,
                "batch_size": eff_batch,
                "concurrent": eff_concurrent,
                "ckpt_every": eff_ckpt_every,
                "smoke": smoke_test,
                "detach": eff_detach,
                "target": eff_target,
                "compile": eff_compile,
                "compile_mode": eff_compile_mode,
                "device": device,
                "device_name": device_name,
                "ckpt_dir": checkpoint_dir,
            }
        )
        _elapsed = time.perf_counter() - _start
        _compile_note = (
            f"compile={eff_compile_mode}" if eff_compile else "compile=off"
        )
        print(f"device: {device} ({device_name}) | {_compile_note}")
        print(f"total wall time: {_elapsed:.1f}s")
        print(f"checkpoints: {checkpoint_dir}")
        for _key, _record in fig3["runs"].items():
            _secs = _record["mean_epoch_seconds"]
            _secs = "n/a" if _secs is None else f"{_secs:.2f}s/epoch"
            _hidden = _record["final_hidden_absmean"]
            _pooled = _record["final_pooled_absmean"]
            _wnorm = _record["final_cls_wnorm"]
            _hidden = "n/a" if _hidden is None else f"{_hidden:.4f}"
            _pooled = "n/a" if _pooled is None else f"{_pooled:.4f}"
            _wnorm = "n/a" if _wnorm is None else f"{_wnorm:.4f}"
            print(
                f"{_key}: final acc={_record['final_acc']:.4f} "
                f"sd={_record['final_sd']:.5f} "
                f"hidden_absmean={_hidden} pooled_absmean={_pooled} "
                f"cls_wnorm={_wnorm} {_secs} "
                f"resumed={_record['resumed']}"
            )
        _out = mo.md(
            f"Trained {len(fig3['runs'])} runs in {_elapsed:.1f}s "
            f"on `{device}`."
        )
    else:
        fig3 = None
        _out = mo.md("_Run training to produce results._")
    _out
    return (fig3,)


@app.cell
def _(Path, fig3, json, mo):
    if fig3 is not None:
        _payload = json.dumps(fig3).encode("utf-8")
        _loc = mo.notebook_location()
        _msg = "notebook location unavailable"
        if _loc is not None:
            _path = Path(str(_loc)) / "public" / "data" / "cifar_fig3.json"
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
                    filename="cifar_fig3.json",
                    mimetype="application/json",
                    label="Download cifar_fig3.json",
                ),
            ]
        )
    else:
        _out = mo.md("_Run training to produce results._")
    _out
    return


@app.cell
def _(alt, fig3, pd):
    if fig3 is None:
        charts = None
    else:
        _colors = {
            "0": "#000000",
            "0.5": "#00e000",
            "1": "#0000ff",
            "2": "#a52a2a",
        }
        _keys = [f"{a:g}" for a in fig3["config"]["aws"]]
        _domain = [k for k in _keys if k in _colors]
        _color = alt.Color(
            "aw:N",
            scale=alt.Scale(
                domain=_domain, range=[_colors[k] for k in _domain]
            ),
            legend=alt.Legend(title="AW"),
        )

        _rows = []
        for _aw_key, _res in fig3["results"].items():
            for _e, (_m, _c) in enumerate(
                zip(_res["sd_mean"], _res["sd_ci"])
            ):
                _rows.append(
                    {
                        "aw": _aw_key,
                        "epoch": _e,
                        "mean": _m,
                        "lo": _m - _c,
                        "hi": _m + _c,
                    }
                )
        _df = pd.DataFrame(_rows)
        _base = alt.Chart(_df).encode(x=alt.X("epoch:Q", title="epoch"))
        _band = _base.mark_area(opacity=0.15).encode(
            y=alt.Y("lo:Q", title="classifier weight SD"),
            y2="hi:Q",
            color=_color,
        )
        _line = _base.mark_line().encode(y="mean:Q", color=_color)
        fig3a = (_band + _line).properties(
            width=420, height=280, title="Fig 3A — SD vs epoch"
        )

        _acc_rows = []
        for _aw_key, _res in fig3["results"].items():
            _acc_rows.append(
                {
                    "aw": _aw_key,
                    "mean": _res["acc_mean"],
                    "lo": max(0.0, _res["acc_mean"] - _res["acc_ci"]),
                    "hi": _res["acc_mean"] + _res["acc_ci"],
                }
            )
        _acc_df = pd.DataFrame(_acc_rows)
        _acc_base = alt.Chart(_acc_df).encode(
            x=alt.X("aw:N", title="AW", sort=_domain),
            color=_color,
        )
        _err = _acc_base.mark_errorbar().encode(y=alt.Y("lo:Q"), y2="hi:Q")
        _pts = _acc_base.mark_point(filled=True, size=70).encode(
            y=alt.Y("mean:Q", title="final test accuracy")
        )
        fig3c = (_err + _pts).properties(
            width=320, height=280, title="Fig 3C — final accuracy"
        )
        charts = [fig3a, fig3c]
    return (charts,)


@app.cell
def _(charts, mo):
    if charts is None:
        _out = mo.md("_Run training to produce charts._")
    else:
        _out = mo.hstack(charts, gap=1)
    _out
    return


@app.cell
def _(fig3, mo):
    _lines = [
        "## Comparison with the paper",
        "",
        "| Quantity | Paper (Fig 3) | Notes |",
        "| --- | --- | --- |",
        "| 3A SD rise | ~0.015 at init to ~0.05–0.06 at epoch 250 | SD grows for every AW; baseline ends widest |",
        "| 3B RLCT at epoch 250 | baseline ~82, AW 0.5 ~72, AW 1 ~69, AW 2 ~66 | computed in a later step from the pruned weights saved here |",
        "| 3C final accuracy | ~0.918–0.920 for all AW | self-modeling does not hurt accuracy |",
    ]
    if fig3 is not None:
        _lines.append("")
        _lines.append("This run:")
        for _aw_key, _res in fig3["results"].items():
            _lines.append(
                f"- AW={_aw_key}: final SD "
                f"{_res['sd_mean'][-1]:.4f} ± {_res['sd_ci'][-1]:.4f}, "
                f"accuracy {_res['acc_mean']:.4f} ± {_res['acc_ci']:.4f} "
                f"(n={_res['n']})"
            )
    mo.md("\n".join(_lines))
    return


@app.cell
def _(fig3, mo):
    if fig3 is None:
        _out = mo.md(
            "_Per-epoch timing appears here after training._"
        )
    else:
        _secs = [
            r["mean_epoch_seconds"]
            for r in fig3["runs"].values()
            if r["mean_epoch_seconds"] is not None
        ]
        _mean = sum(_secs) / len(_secs) if _secs else float("nan")
        _out = mo.md(
            f"""
    ### Runtime

    Measured here: **{_mean:.2f} s/epoch/model** on
    `{fig3['device']['type']}` ({fig3['device']['name']}), batch 512,
    full CIFAR-10. On a data-center GPU with bf16 autocast and
    `channels_last`, expect roughly 1–3 s/epoch/model; the full 40-run
    sweep (10 seeds × {len(fig3['config']['aws'])} AW values ×
    {fig3['config']['epochs']} epochs) therefore lands in the ballpark of
    a few GPU-hours when runs are interleaved a few at a time.
    """
        )
    _out
    return


if __name__ == "__main__":
    app.run()
