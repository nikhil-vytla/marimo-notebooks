import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import SelfModelMLP


def _batch(n=4, in_dim=784):
    return torch.randn(n, in_dim)


def test_baseline():
    m = SelfModelMLP(hidden=512, self_model=False)
    logits, a_hat, a = m(_batch())
    assert logits.shape == (4, 10)
    assert a_hat is None
    assert a.shape == (4, 512)
    assert m.out.weight.shape == (10, 512)


def test_self_model():
    m = SelfModelMLP(hidden=512, self_model=True)
    logits, a_hat, a = m(_batch())
    assert logits.shape == (4, 10)
    assert a_hat.shape == (4, 512)
    assert a.shape == (4, 512)
    assert m.out.weight.shape == (522, 512)
    cw = m.classifier_weight()
    assert cw.shape == (10, 512)
    assert torch.equal(cw, m.out.weight[:10])


def test_a_nonnegative():
    m = SelfModelMLP(hidden=512, self_model=True)
    _, _, a = m(_batch())
    assert (a >= 0).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("all tests passed")
