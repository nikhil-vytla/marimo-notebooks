"""CPU equivalence test: batched training == per-model ``train.py`` training.

Two models are trained for 3 SGD steps on the *same* fixed batch of 32 inputs:
one with the stacked batched implementation, one with the reference
``SelfModelMLP`` + ``self_model_loss`` per model.  Classifier weights and the
hidden layer must match to within 1e-6.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loss import self_model_loss
from model import SelfModelMLP
from train_batched import build_params, train_step

HIDDEN = 16
BATCH = 32
STEPS = 3
LR = 0.005
MOMENTUM = 0.9


def _reference_step(model, opt, x, y, aw):
    logits, a_hat, a = model(x)
    total, _, _ = self_model_loss(logits, y, a_hat, a, aw)
    opt.zero_grad()
    total.backward()
    opt.step()


def test_batched_matches_separate_models():
    models = [(0.0, 0), (10.0, 1)]
    device = torch.device("cpu")

    torch.manual_seed(123)
    x_fixed = torch.randn(BATCH, 784)
    y_fixed = torch.randint(0, 10, (BATCH,))

    params = build_params(models, HIDDEN, device)
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, nesterov=True)
    aws = torch.tensor([aw for aw, _ in models], dtype=torch.float32)

    xb = x_fixed.unsqueeze(0).expand(len(models), BATCH, 784)
    yb = y_fixed.unsqueeze(0).expand(len(models), BATCH)

    refs = []
    for aw, seed in models:
        torch.manual_seed(seed)
        m = SelfModelMLP(hidden=HIDDEN, self_model=aw > 0)
        o = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True)
        refs.append((m, o, aw))

    for _ in range(STEPS):
        train_step(params, opt, xb, yb, aws)
        for m, o, aw in refs:
            _reference_step(m, o, x_fixed, y_fixed, aw)

    W1, b1, W2, b2 = params
    for i, (ref, _, _) in enumerate(refs):
        assert torch.allclose(W1[i], ref.hidden.weight, atol=1e-6), "W1 mismatch"
        assert torch.allclose(b1[i], ref.hidden.bias, atol=1e-6), "b1 mismatch"
        assert torch.allclose(
            W2[i, :10], ref.out.weight[:10], atol=1e-6
        ), "W2 classifier mismatch"
        assert torch.allclose(
            b2[i, :10], ref.out.bias[:10], atol=1e-6
        ), "b2 classifier mismatch"


def test_baseline_slice_untouched_by_self_rows():
    """With aw=0, the self rows of W2 receive zero gradient and stay at init."""
    models = [(0.0, 7)]
    device = torch.device("cpu")

    torch.manual_seed(1)
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))

    params = build_params(models, HIDDEN, device)
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, nesterov=True)
    aws = torch.tensor([0.0])
    before = params[2][:, 10:].clone()

    train_step(params, opt, x.unsqueeze(0), y.unsqueeze(0), aws)

    assert torch.equal(params[2][:, 10:], before)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("all tests passed")
