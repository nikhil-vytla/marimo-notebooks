"""Train many independent MNIST self-model MLPs simultaneously as one stacked model.

Mathematically equivalent to running ``train.py`` once per (aw, seed), but all
models share a single forward/backward on one device.  Parameters are stacked
along a leading model dimension; because every model owns a disjoint slice and
the loss is summed over models, each slice receives exactly the gradient it
would have received in isolation.  SGD is elementwise, so the optimizer is
equivalent too.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders
from model import SelfModelMLP


def build_params(models, hidden, device):
    """Build stacked parameters for ``models`` = list of (aw, seed).

    Initialization matches ``train.py``: for each model we seed and build the
    same ``SelfModelMLP`` that ``train.py`` would build for that seed.  For
    ``aw == 0`` the classifier rows come from the baseline model; the extra
    self rows are arbitrary (they never receive gradient) and are taken from a
    ``self_model=True`` model built under the same seed.
    """
    M = len(models)
    W1 = torch.empty(M, hidden, 784, device=device)
    b1 = torch.empty(M, hidden, device=device)
    W2 = torch.empty(M, 10 + hidden, hidden, device=device)
    b2 = torch.empty(M, 10 + hidden, device=device)

    for m, (aw, seed) in enumerate(models):
        self_model = aw > 0
        torch.manual_seed(seed)
        ref = SelfModelMLP(hidden=hidden, self_model=self_model)
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
                ref_self = SelfModelMLP(hidden=hidden, self_model=True)
                W2[m, 10:] = ref_self.out.weight[10:]
                b2[m, 10:] = ref_self.out.bias[10:]

    params = [
        nn.Parameter(W1),
        nn.Parameter(b1),
        nn.Parameter(W2),
        nn.Parameter(b2),
    ]
    return params


def forward_batched(params, xb):
    """xb: (M, B, 784) -> logits (M,B,10), a_hat (M,B,H), a (M,B,H)."""
    W1, b1, W2, b2 = params
    a = torch.relu(torch.baddbmm(b1.unsqueeze(1), xb, W1.transpose(1, 2)))
    out = torch.baddbmm(b2.unsqueeze(1), a, W2.transpose(1, 2))
    return out[..., :10], out[..., 10:], a


def train_step(params, opt, xb, yb, aws):
    """One SGD step.  Returns per-model (ce, sm) means over the batch."""
    logits, a_hat, a = forward_batched(params, xb)
    M, B = yb.shape
    ce = F.cross_entropy(
        logits.reshape(-1, 10), yb.reshape(-1), reduction="none"
    ).reshape(M, B).mean(dim=1)
    sm = (a_hat - a.detach()).pow(2).mean(dim=(1, 2))
    total = (ce + aws * sm).sum()
    opt.zero_grad()
    total.backward()
    opt.step()
    return ce.detach(), sm.detach()


def evaluate_batched(params, X, y, device):
    """Test accuracy per model on the full test set, batched over models."""
    M = params[0].shape[0]
    correct = torch.zeros(M, device=device)
    n = y.numel()
    with torch.no_grad():
        for lo in range(0, n, 512):
            hi = min(lo + 512, n)
            xb = X[lo:hi].unsqueeze(0).expand(M, hi - lo, 784)
            yb = y[lo:hi]
            logits, _, _ = forward_batched(params, xb)
            correct += (logits.argmax(dim=-1) == yb.unsqueeze(0)).sum(dim=1)
    return correct / n


def classifier_sd(params):
    """Std of each model's 10 x hidden classifier weight -> (M,)."""
    W2 = params[2]
    return W2[:, :10].reshape(W2.shape[0], -1).detach().std(dim=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws", default="0,1,5,10,20,50")
    parser.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out-root", default="results_batched")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    aws_list = [float(a) for a in args.aws.split(",") if a.strip()]

    # Model order: aw outer, seed inner (any order is fine; only disjointness matters).
    models = [(aw, seed) for aw in aws_list for seed in range(args.seeds)]
    M = len(models)

    params = build_params(models, args.hidden, device)
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, nesterov=True
    )
    aws = torch.tensor([aw for aw, _ in models], dtype=torch.float32, device=device)

    train_loader, test_loader = get_loaders(batch_size=512, center=args.center)
    X = train_loader.dataset.tensors[0].to(device)
    y = train_loader.dataset.tensors[1].to(device)
    Xte = test_loader.dataset.tensors[0].to(device)
    yte = test_loader.dataset.tensors[1].to(device)

    N = X.shape[0]
    seeds = [seed for _, seed in models]
    gens = [torch.Generator() for _ in range(M)]

    sd = [[v] for v in classifier_sd(params).tolist()]
    test_acc = [[] for _ in range(M)]
    train_ce = [[] for _ in range(M)]
    train_sm = [[] for _ in range(M)]
    epoch_time = []

    if device.type == "mps":
        torch.mps.synchronize()

    for epoch in range(1, args.epochs + 1):
        # Independent permutation per model per epoch (need not match train.py).
        for m in range(M):
            gens[m].manual_seed(seeds[m] * 1000 + epoch)
        perms = torch.stack(
            [torch.randperm(N, generator=gens[m]) for m in range(M)]
        ).to(device)

        params_train = params
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()

        ce_acc = torch.zeros(M, device=device)
        sm_acc = torch.zeros(M, device=device)
        n_batches = 0
        for lo in range(0, N, 512):
            hi = min(lo + 512, N)
            idx = perms[:, lo:hi]
            xb = X[idx]
            yb = y[idx]
            ce, sm = train_step(params_train, optimizer, xb, yb, aws)
            ce_acc += ce
            sm_acc += sm
            n_batches += 1

        sd_vec = classifier_sd(params)
        acc_vec = evaluate_batched(params, Xte, yte, device)

        if device.type == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - start

        mean_ce = ce_acc / n_batches
        mean_sm = sm_acc / n_batches
        for m in range(M):
            sd[m].append(sd_vec[m].item())
            test_acc[m].append(acc_vec[m].item())
            train_ce[m].append(mean_ce[m].item())
            train_sm[m].append(mean_sm[m].item())
        epoch_time.append(seconds)

        print(
            f"epoch {epoch:3d} {seconds:6.2f}s "
            f"sd [{sd_vec.min().item():.6f}, {sd_vec.max().item():.6f}] "
            f"acc {acc_vec.mean().item():.4f}",
            flush=True,
        )

    group = "b" if args.center else "a"
    out_dir = Path(args.out_root) / group
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, (aw, seed) in enumerate(models):
        out_path = out_dir / f"aw{aw:g}_h{args.hidden}_s{seed}.json"
        config = {
            "aw": aw,
            "hidden": args.hidden,
            "seed": seed,
            "epochs": args.epochs,
            "center": args.center,
            "device": args.device,
            "out": str(out_path),
        }
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": config,
                    "sd": sd[m],
                    "test_acc": test_acc[m],
                    "train_ce": train_ce[m],
                    "train_sm": train_sm[m],
                    "epoch_time": epoch_time,
                },
                f,
                indent=2,
            )
    print(f"wrote {M} runs to {out_dir}")


if __name__ == "__main__":
    main()
