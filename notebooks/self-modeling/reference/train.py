import argparse
import json
import time
from pathlib import Path

import torch

from data import get_loaders
from loss import self_model_loss
from metrics import weight_sd
from model import SelfModelMLP


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aw", type=float, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        group = "b" if args.center else "a"
        args.out = f"results/{group}/aw{args.aw:g}_h{args.hidden}_s{args.seed}.json"

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    model = SelfModelMLP(hidden=args.hidden, self_model=args.aw > 0).to(device)
    train_loader, test_loader = get_loaders(batch_size=512, center=args.center)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, nesterov=True
    )

    sd = [weight_sd(model)]
    test_acc = []
    train_ce = []
    train_sm = []
    epoch_time = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        ce_sum = 0.0
        sm_sum = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, a_hat, a = model(x)
            total, ce, sm = self_model_loss(logits, y, a_hat, a, args.aw)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            ce_sum += ce.item()
            sm_sum += sm.item()
            n_batches += 1
        seconds = time.perf_counter() - start

        sd.append(weight_sd(model))
        acc = evaluate(model, test_loader, device)
        mean_ce = ce_sum / n_batches
        mean_sm = sm_sum / n_batches

        test_acc.append(acc)
        train_ce.append(mean_ce)
        train_sm.append(mean_sm)
        epoch_time.append(seconds)

        print(
            f"epoch {epoch:3d} sd {sd[-1]:.6f} test_acc {acc:.4f} "
            f"ce {mean_ce:.4f} sm {mean_sm:.4f} sec {seconds:.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "sd": sd,
                "test_acc": test_acc,
                "train_ce": train_ce,
                "train_sm": train_sm,
                "epoch_time": epoch_time,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
