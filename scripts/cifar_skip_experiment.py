"""CIFAR-10 block-3 skip experiment for the self-modeling essay.

The paper says the hidden layer receives "the output of the fourth
convolutional block and the skip-connections from the third block."
A standard ResNet-18 has no such skip. This script tests the simplest
faithful reading: global-average-pool block 3 and block 4, concatenate
them, and feed that 768-vector into the 2000-wide hidden layer. The
self-model target is the concatenation of that input and the hidden
activations.

Run on a CUDA machine:

    CIFAR_SKIP_SMOKE=1 python scripts/cifar_skip_experiment.py
    python scripts/cifar_skip_experiment.py

The script appends one JSON line per completed run, so a restarted
session skips finished runs.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 10
HIDDEN_WIDTH = 2000
POOLED4_WIDTH = 512
POOLED3_WIDTH = 256
LR = 1e-3
ROTATION_DEG = 15.0
TRANSLATION = 0.10
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def t95(df: int) -> float:
    return T95.get(int(df), 1.96)


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    half = t95(arr.size - 1) * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return float(arr.mean()), float(half)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class CifarResNet18Skip(nn.Module):
    """ResNet-18 with an optional pooled block-3 skip into the hidden layer."""

    def __init__(self, skip: str = "none"):
        super().__init__()
        if skip not in ("none", "block3"):
            raise ValueError(f"unknown skip mode {skip!r}")
        self.skip = skip
        self.input_width = POOLED4_WIDTH + (
            POOLED3_WIDTH if skip == "block3" else 0
        )
        self.target_width = self.input_width + HIDDEN_WIDTH
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.hidden = nn.Linear(self.input_width, HIDDEN_WIDTH)
        self.out = nn.Linear(HIDDEN_WIDTH, N_CLASSES + self.target_width)

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride)]
        layers.extend(BasicBlock(out_ch, out_ch, 1) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        block3 = self.layer3(x)
        block4 = self.layer4(block3)
        pooled4 = torch.flatten(F.adaptive_avg_pool2d(block4, 1), 1)
        if self.skip == "block3":
            pooled3 = torch.flatten(F.adaptive_avg_pool2d(block3, 1), 1)
            hidden_input = torch.cat([pooled4, pooled3], dim=1)
        else:
            hidden_input = pooled4
        hidden = F.relu(self.hidden(hidden_input))
        return hidden_input, hidden

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_input, hidden = self.extract(x)
        out = self.out(hidden)
        target = torch.cat([hidden_input, hidden], dim=1)
        return out[:, :N_CLASSES], out[:, N_CLASSES:], target


def load_cifar(limit: int | None = None):
    raw = load_dataset("uoft-cs/cifar10")

    def to_tensors(split: str, limit_n: int | None):
        data = raw[split]
        n = len(data) if limit_n is None else min(limit_n, len(data))
        images = np.empty((n, 3, 32, 32), dtype=np.uint8)
        labels = np.empty((n,), dtype=np.int64)
        for i, example in enumerate(data):
            if i >= n:
                break
            images[i] = np.asarray(example["img"], dtype=np.uint8).reshape(
                32, 32, 3
            ).transpose(2, 0, 1)
            labels[i] = example["label"]
        return torch.from_numpy(images).to(DEVICE), torch.from_numpy(
            labels
        ).to(DEVICE)

    x_train, y_train = to_tensors("train", limit)
    x_test, y_test = to_tensors("test", limit)
    channel_sum = torch.zeros(3, dtype=torch.float64, device=DEVICE)
    channel_sq = torch.zeros(3, dtype=torch.float64, device=DEVICE)
    total = 0
    for lo in range(0, x_train.shape[0], 5000):
        chunk = x_train[lo:lo + 5000].float().div(255.0)
        channel_sum += chunk.sum(dim=(0, 2, 3)).double()
        channel_sq += chunk.pow(2).sum(dim=(0, 2, 3)).double()
        total += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    mean = (channel_sum / total).float().view(1, 3, 1, 1).to(DEVICE)
    var = (channel_sq / total - (channel_sum / total).pow(2)).clamp(
        min=1e-12
    )
    std = var.sqrt().float().view(1, 3, 1, 1).to(DEVICE)
    return x_train, y_train, x_test, y_test, mean, std


def augment_batch(images: torch.Tensor, generator: torch.Generator):
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


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (x - mean) / std


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch: int = 512,
) -> float:
    model.eval()
    correct = 0
    for lo in range(0, x_test.shape[0], batch):
        hi = min(lo + batch, x_test.shape[0])
        xb = normalize(x_test[lo:hi].float().div(255.0), mean, std)
        logits, _, _ = model(xb)
        correct += int((logits.argmax(dim=1) == y_test[lo:hi]).sum())
    model.train()
    return correct / x_test.shape[0]


def base_module(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def classifier_sd(model: nn.Module) -> float:
    base = base_module(model)
    return float(base.out.weight[:N_CLASSES].detach().float().std())


def train_run(
    skip: str,
    aw: float,
    seed: int,
    epochs: int,
    batch: int,
    x_train,
    y_train,
    x_test,
    y_test,
    mean,
    std,
):
    torch.manual_seed(seed)
    model = CifarResNet18Skip(skip=skip).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if (
        os.environ.get("CIFAR_SKIP_COMPILE", "1") == "1"
        and DEVICE.type == "cuda"
    ):
        try:
            model = torch.compile(
                model, mode="reduce-overhead", dynamic=False
            )
        except RuntimeError as _err:
            print(f"  compile failed, continuing eager: {_err}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, fused=DEVICE.type == "cuda")
    use_amp = (
        os.environ.get("CIFAR_SKIP_AMP", "1") == "1"
        and DEVICE.type == "cuda"
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_train.shape[0]
    n_steps = n // batch
    sd_curve = [classifier_sd(model)]
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=generator).to(DEVICE)
        for step in range(n_steps):
            idx = perm[step * batch:(step + 1) * batch]
            xb = augment_batch(x_train[idx], generator)
            xb = normalize(xb, mean, std)
            if DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logits, a_hat, target = model(xb)
                ce = F.cross_entropy(logits, yb)
                sm = (a_hat - target.detach()).pow(2).mean()
                loss = ce + aw * sm
            loss.backward()
            optimizer.step()
        sd_curve.append(classifier_sd(model))
        if epoch % 25 == 0 or epoch == epochs:
            print(
                f"  {skip} AW={aw:g} seed={seed} epoch {epoch}/{epochs} "
                f"sd={sd_curve[-1]:.5f} {time.perf_counter() - start:.0f}s",
                flush=True,
            )
    accuracy = evaluate(model, x_test, y_test, mean, std, batch)
    elapsed = time.perf_counter() - start
    return {
        "skip": skip,
        "aw": float(aw),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch": int(batch),
        "sd": sd_curve,
        "sd_final": sd_curve[-1],
        "accuracy": accuracy,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    smoke = os.environ.get("CIFAR_SKIP_SMOKE") == "1"
    epochs = int(os.environ.get("CIFAR_SKIP_EPOCHS", "2" if smoke else "150"))
    batch = int(os.environ.get("CIFAR_SKIP_BATCH", "512"))
    seeds = int(os.environ.get("CIFAR_SKIP_SEEDS", "1" if smoke else "3"))
    limit = 2000 if smoke else None
    aws = [0.0, 2.0] if smoke else [0.0, 0.5, 1.0, 2.0]
    _aws_env = os.environ.get("CIFAR_SKIP_AWS", "")
    if _aws_env:
        aws = [float(a) for a in _aws_env.split(",") if a.strip()]
    _seeds_env = os.environ.get("CIFAR_SKIP_SEEDS")
    if _seeds_env:
        seeds = int(_seeds_env)
    _modes_env = os.environ.get("CIFAR_SKIP_MODES", "")
    modes = (
        [m.strip() for m in _modes_env.split(",") if m.strip()]
        if _modes_env
        else ["none", "block3"]
    )
    out_dir = Path(os.environ.get("CIFAR_SKIP_OUT", "/marimo/public/data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "cifar_skip_runs.jsonl"
    summary_path = out_dir / "cifar_skip.json"

    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["skip"], row["aw"], row["seed"]))
    print(
        f"device={DEVICE} epochs={epochs} batch={batch} seeds={seeds} "
        f"smoke={smoke} done={len(done)}",
        flush=True,
    )
    x_train, y_train, x_test, y_test, mean, std = load_cifar(limit)
    print(
        f"train={tuple(x_train.shape)} test={tuple(x_test.shape)} "
        f"mean={mean.flatten().tolist()} std={std.flatten().tolist()}",
        flush=True,
    )

    for skip in modes:
        for aw in aws:
            for seed in range(seeds):
                key = (skip, float(aw), seed)
                if key in done:
                    print(f"skip completed {key}", flush=True)
                    continue
                print(f"train {key}", flush=True)
                result = train_run(
                    skip, aw, seed, epochs, batch,
                    x_train, y_train, x_test, y_test, mean, std,
                )
                with jsonl.open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                done.add(key)

    rows = [
        json.loads(line)
        for line in jsonl.read_text().splitlines()
        if line.strip()
    ]
    results = {}
    for skip in modes:
        for aw in aws:
            subset = [
                r for r in rows
                if r["skip"] == skip
                and abs(r["aw"] - aw) < 1e-9
            ]
            if not subset:
                continue
            sd_mean, sd_ci = mean_ci([r["sd_final"] for r in subset])
            acc_mean, acc_ci = mean_ci([r["accuracy"] for r in subset])
            results[f"{skip}:{aw:g}"] = {
                "skip": skip,
                "aw": float(aw),
                "n": len(subset),
                "sd_mean": sd_mean,
                "sd_ci": sd_ci,
                "acc_mean": acc_mean,
                "acc_ci": acc_ci,
                "sd_curves": [r["sd"] for r in subset],
            }
    payload = {
        "config": {
            "device": str(DEVICE),
            "device_name": (
                torch.cuda.get_device_name(0)
                if DEVICE.type == "cuda"
                else "cpu"
            ),
            "epochs": epochs,
            "batch": batch,
            "seeds": seeds,
            "smoke": smoke,
            "limit": limit,
            "compile": os.environ.get("CIFAR_SKIP_COMPILE", "1") == "1",
            "amp": os.environ.get("CIFAR_SKIP_AMP", "1") == "1",
            "skip": "none = standard ResNet-18 hidden input; block3 = concat pooled block3 and block4",
            "target": "concat(hidden input, hidden activations)",
        },
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {summary_path}", flush=True)
    for key, value in results.items():
        print(
            f"{key}: sd={value['sd_mean']:.5f} +/- {value['sd_ci']:.5f} "
            f"acc={value['acc_mean']:.4f} n={value['n']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
