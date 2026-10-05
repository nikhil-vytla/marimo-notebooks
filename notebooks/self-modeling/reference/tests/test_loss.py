import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loss import self_model_loss
from model import SelfModelMLP


def _make(self_model=True):
    torch.manual_seed(0)
    m = SelfModelMLP(hidden=32, self_model=self_model)
    x = torch.randn(8, 784)
    y = torch.randint(0, 10, (8,))
    return m, x, y


def test_baseline_is_ce():
    m, x, y = _make(self_model=False)
    logits, a_hat, a = m(x)
    total, ce, sm = self_model_loss(logits, y, a_hat, a, aw=5)
    assert a_hat is None
    assert torch.allclose(total, F.cross_entropy(logits, y))
    assert sm.item() == 0.0


def test_total_formula():
    m, x, y = _make()
    logits, a_hat, a = m(x)
    total, ce, sm = self_model_loss(logits, y, a_hat, a, aw=5)
    assert torch.allclose(total, ce + 5 * sm)


def test_target_detached():
    m, x, y = _make()

    logits, a_hat, a = m(x)
    m.zero_grad()
    _, _, sm = self_model_loss(logits, y, a_hat, a, aw=5)
    sm.backward()
    g_detached = m.hidden.weight.grad.clone()

    m.zero_grad()
    logits, a_hat, a = m(x)
    F.mse_loss(a_hat, a.detach()).backward()
    g_ref = m.hidden.weight.grad.clone()
    assert torch.allclose(g_detached, g_ref)

    m.zero_grad()
    logits, a_hat, a = m(x)
    F.mse_loss(a_hat, a).backward()
    g_nondet = m.hidden.weight.grad.clone()
    assert not torch.allclose(g_detached, g_nondet)


def test_no_grad_into_target():
    m, x, y = _make()
    logits, a_hat, a = m(x)
    _, _, sm = self_model_loss(logits, y, a_hat, a, aw=5)
    g_sm = torch.autograd.grad(sm, a, retain_graph=True)[0]

    ref = F.mse_loss(m.out(a)[:, 10:], a.detach())
    g_ref = torch.autograd.grad(ref, a, retain_graph=True)[0]
    assert torch.allclose(g_sm, g_ref)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("all tests passed")
