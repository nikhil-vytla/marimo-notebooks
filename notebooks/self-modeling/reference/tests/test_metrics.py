import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import weight_sd
from model import SelfModelMLP


def test_excludes_self_model_rows():
    torch.manual_seed(0)
    m = SelfModelMLP(hidden=4, self_model=True)
    handmade = torch.randn(10, 4)
    with torch.no_grad():
        m.out.weight[:10] = handmade
        m.out.weight[10:] = 1000.0
    assert abs(weight_sd(m) - torch.std(handmade).item()) < 1e-6


def test_baseline_matches_out_weight():
    torch.manual_seed(0)
    m = SelfModelMLP(hidden=4, self_model=False)
    assert weight_sd(m) == m.out.weight.std().item()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("all tests passed")
