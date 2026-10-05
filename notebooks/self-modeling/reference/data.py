import struct
import zlib
from pathlib import Path

import numpy as np
import torch
from datasets import Image, load_dataset
from torch.utils.data import DataLoader, TensorDataset

CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "mnist_uint8.pt"


def _decode_png(blob):
    """Decode an 8-bit grayscale, non-interlaced 28x28 PNG into a uint8 array."""
    pos, idat = 8, bytearray()
    while pos < len(blob):
        size = struct.unpack(">I", blob[pos : pos + 4])[0]
        kind = blob[pos + 4 : pos + 8]
        if kind == b"IDAT":
            idat += blob[pos + 8 : pos + 8 + size]
        elif kind == b"IEND":
            break
        pos += 12 + size

    raw = zlib.decompress(bytes(idat))
    out = np.empty((28, 28), dtype=np.uint8)
    prev = np.zeros(28, dtype=np.int32)
    p = 0
    for row in range(28):
        f = raw[p]
        line = np.frombuffer(raw, dtype=np.uint8, count=28, offset=p + 1).astype(np.int32)
        p += 29
        cur = line.copy()
        if f == 2:  # Up
            cur = (line + prev) & 0xFF
        elif f == 1:  # Sub
            for x in range(1, 28):
                cur[x] = (cur[x] + cur[x - 1]) & 0xFF
        elif f == 3:  # Average
            for x in range(28):
                cur[x] = (cur[x] + ((cur[x - 1] if x else 0) + prev[x]) // 2) & 0xFF
        elif f == 4:  # Paeth
            for x in range(28):
                a = cur[x - 1] if x else 0
                c = prev[x - 1] if x else 0
                pp = a + prev[x] - c
                pa, pb, pc = abs(pp - a), abs(pp - prev[x]), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (prev[x] if pb <= pc else c)
                cur[x] = (cur[x] + pr) & 0xFF
        prev = cur
        out[row] = prev
    return out


def _raw_tensors():
    """Return (X_train, y_train, X_test, y_test) as raw uint8/int64 tensors, cached."""
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        return cache["X_train"], cache["y_train"], cache["X_test"], cache["y_test"]

    ds = load_dataset("ylecun/mnist")
    cache = {}
    for split in ("train", "test"):
        data = ds[split].cast_column("image", Image(decode=False))
        X = np.stack([_decode_png(row["bytes"]) for row in data["image"]])
        cache[f"X_{split}"] = torch.from_numpy(X.reshape(len(X), -1).astype(np.uint8))
        cache[f"y_{split}"] = torch.tensor(data["label"], dtype=torch.int64)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)
    return cache["X_train"], cache["y_train"], cache["X_test"], cache["y_test"]


def get_loaders(batch_size=512, num_workers=0, center=False):
    X_train_u8, y_train, X_test_u8, y_test = _raw_tensors()
    X_train = X_train_u8.to(torch.float32) / 127.5 - 1.0
    X_test = X_test_u8.to(torch.float32) / 127.5 - 1.0

    if center:
        mean = X_train.mean()
        X_train = X_train - mean
        X_test = X_test - mean

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    X, y = next(iter(train_loader))
    all_X = train_loader.dataset.tensors[0]
    print(f"train size: {len(train_loader.dataset)}")
    print(f"test size: {len(test_loader.dataset)}")
    print(f"batch X shape: {tuple(X.shape)}, batch y shape: {tuple(y.shape)}")
    print(f"train X min: {all_X.min().item()}")
    print(f"train X max: {all_X.max().item()}")
    print(f"train X mean: {all_X.mean().item()}")

    train_loader_c, test_loader_c = get_loaders(center=True)
    all_X_c = train_loader_c.dataset.tensors[0]
    print(f"centered train X mean: {all_X_c.mean().item()}")
    print(f"centered train X min: {all_X_c.min().item()}")
    print(f"centered train X max: {all_X_c.max().item()}")
