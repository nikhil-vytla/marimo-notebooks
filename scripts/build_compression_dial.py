"""Build the precomputed frames for the compression dial.

The essay widget reads this JSON. The toy is a 2D three-class problem:
a trainable hidden layer, a linear classifier, and a self-model head
that predicts the hidden layer. The dial sets AW. With the
stop-gradient, the self-model head learns to copy the hidden state and
the representation keeps its scale. Without it, the target moves: the
hidden state shrinks, and the classifier grows to keep the logits
fixed. That is the scale cheat, in one picture.

Run with:

    uv run --no-project --with numpy scripts/build_compression_dial.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STEPS = 600
LR = 0.05
SEED = 0
N_PER_CLASS = 30
AW_VALUES = [float(aw) for aw in range(0, 51, 5)]
OUT = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "self-modeling"
    / "public"
    / "data"
    / "compression_dial.json"
)


def make_data(seed: int = SEED):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[1.2, 0.0], [-0.6, 1.0], [-0.6, -1.0]], dtype=float
    )
    xs = []
    for center in centers:
        xs.append(center + 0.25 * rng.normal(size=(N_PER_CLASS, 2)))
    x = np.vstack(xs)
    y = np.repeat([0, 1, 2], N_PER_CLASS)
    return x, y


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def train(aw: float, detach: bool):
    x, y = make_data()
    rng = np.random.default_rng(SEED + 1)
    wh = np.eye(2) + 0.01 * rng.normal(size=(2, 2))
    bh = np.zeros(2)
    wc = 0.3 * rng.normal(size=(3, 2))
    bc = np.zeros(3)
    ws = np.zeros((2, 2))
    bs = np.zeros(2)
    params = [wh, bh, wc, bc, ws, bs]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n = len(y)
    eye = np.eye(3)

    for step in range(1, STEPS + 1):
        a = x @ wh.T + bh
        z = a @ wc.T + bc
        p = softmax(z)
        dz = (p - eye[y]) / n

        g_wc = dz.T @ a
        g_bc = dz.sum(axis=0)
        da = dz @ wc

        residual = a @ ws.T + bs - a
        g_ws = residual.T @ a / n
        g_bs = residual.mean(axis=0)
        da += aw * (residual @ ws) / n
        if not detach:
            da += -aw * residual / n

        g_wh = da.T @ x
        g_bh = da.sum(axis=0)
        grads = [g_wh, g_bh, g_wc, g_bc, g_ws, g_bs]

        for i, (param, grad) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * grad
            v[i] = 0.999 * v[i] + 0.001 * grad * grad
            m_hat = m[i] / (1 - 0.9**step)
            v_hat = v[i] / (1 - 0.999**step)
            param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    a = x @ wh.T + bh
    z = a @ wc.T + bc
    sm = ((a @ ws.T + bs - a) ** 2).mean()
    ce = -np.log(softmax(z)[np.arange(n), y] + 1e-12).mean()
    acc = float((z.argmax(axis=1) == y).mean())
    return {
        "aw": float(aw),
        "points": np.round(a, 4).tolist(),
        "a_norm": round(float(np.linalg.norm(a, axis=1).mean()), 4),
        "w_norm": round(float(np.linalg.norm(wc)), 4),
        "sm_error": round(float(sm), 6),
        "ce": round(float(ce), 6),
        "acc": round(acc, 4),
    }


def main() -> None:
    frames = {"detach": [], "no_detach": []}
    for aw in AW_VALUES:
        frames["detach"].append(train(aw, detach=True))
        frames["no_detach"].append(train(aw, detach=False))

    all_frames = frames["detach"] + frames["no_detach"]
    payload = {
        "meta": {
            "toy": "3-class 2D linear hidden layer",
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "n_per_class": N_PER_CLASS,
            "note": (
                "Each frame is the final state of a full-batch Adam run. "
                "The dial interpolates between frames."
            ),
        },
        "class_names": ["class 0", "class 1", "class 2"],
        "labels": np.repeat([0, 1, 2], N_PER_CLASS).tolist(),
        "world": 5.5,
        "max_a": max(f["a_norm"] for f in all_frames),
        "max_w": max(f["w_norm"] for f in all_frames),
        "max_sm": max(f["sm_error"] for f in all_frames),
        "aws": AW_VALUES,
        "frames": frames,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
