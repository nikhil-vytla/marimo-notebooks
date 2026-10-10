# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "numpy==2.5.3",
#     "datasets==5.1.0",
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
    import json
    import math
    import os
    import re
    import time
    from collections import Counter
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
        Counter,
        F,
        Path,
        alt,
        json,
        load_dataset,
        math,
        mo,
        nn,
        np,
        os,
        pd,
        re,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Self-Modeling in Neural Systems — Figure 4 training (IMDB)

    Reproduces Figure 4 (A and C, RLCT excluded) of
    [*Unexpected Benefits of Self-Modeling in Neural Systems*](https://arxiv.org/abs/2407.10188).

    Text classifier: `EmbeddingBag(dim=64, mode="mean") -> Linear(128) -> Linear(2 + 128)`.
    The extra output rows predict the hidden activations. **AW = 0** is the baseline.
    Everything below is self-contained; heavy work only runs when you click **Train**.
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
    ### Figure 4 reference values (paper)

    - **4A** — the SD of the final-layer weights rises from ~0 at epoch 1 to ~2.0 for
      the baseline by epoch 500; AW=100 is narrower (~1.55) and AW=500 narrower still
      (~1.15).
    - **4C** — test accuracy ~0.82 (baseline), ~0.82 (AW=100), ~0.83 (AW=500): a small
      gain for AW=500.

    The corpus is tokenized with torchtext's legacy `basic_english` and a `min_freq=1`
    vocabulary from the train split. The paper reports **100,683** tokens. With
    `.split()` (which drops the empty-string token) plus `<unk>` at index 0 we get
    exactly 100,683; the other variants give 100,682 / 100,684 / 147,157 / 147,158.
    The training cell prints which variant was selected.
    """)
    return


@app.cell
def _(BATCH, DETACH_TARGET, GRAD_CLIP, mo):
    aws = mo.ui.multiselect(
        options=[0, 100, 500],
        value=[0, 100, 500],
        label="Self-modeling weight (AW)",
    )
    seeds = mo.ui.number(value=10, start=1, stop=50, label="Seeds")
    epochs = mo.ui.number(value=500, start=1, stop=2000, label="Epochs")
    batch = mo.ui.number(
        value=BATCH, start=1, stop=4096, step=1, label="Batch size"
    )
    clip = mo.ui.number(
        value=GRAD_CLIP,
        start=0.0,
        stop=10.0,
        step=0.1,
        label="Grad-norm clip (0 = off)",
    )
    detach = mo.ui.checkbox(
        value=DETACH_TARGET, label="Detach self-model target"
    )
    out_path = mo.ui.text(
        value="",
        label="Output JSON path (blank = public/data/imdb_fig4.json)",
        full_width=True,
    )
    smoke = mo.ui.checkbox(
        value=False, label="Smoke test (2 epochs, 2 seeds)"
    )
    train_button = mo.ui.run_button(label="Train")
    return (
        aws,
        batch,
        clip,
        detach,
        epochs,
        out_path,
        seeds,
        smoke,
        train_button,
    )


@app.cell
def _(
    aws,
    batch,
    clip,
    detach,
    epochs,
    mo,
    out_path,
    seeds,
    smoke,
    train_button,
):
    mo.md("""
    ### Configuration

    Each run trains `len(AW) x seeds` independent models in one batched
    forward/backward: SGD `lr=0.1`, momentum 0, no nesterov. Batch size, gradient
    clipping, and target detachment are all configurable below; the defaults are
    batch 512, clip 1.0, detached target.

    On CUDA all selected AWs train together (one stacked embedding table per model,
    ~770 MB at 30 models). On MPS/CPU the groups are trained one AW at a time.

    In script mode (`SMOKE=1` or any `SM_*` variable set) the controls below are
    overridden: `SM_BATCH`, `SM_CLIP` (`0` disables clipping), `SM_DETACH` (`0`
    trains against the live activations), `SM_AWS` (comma-separated, e.g. `0,500`),
    `SM_OUT` (JSON output path), `SM_SEEDS`, `SM_EPOCHS`.
    """)
    mo.vstack(
        [
            mo.hstack([aws, seeds, epochs], justify="start", gap=2),
            mo.hstack([batch, clip, detach], justify="start", gap=2),
            out_path,
            mo.hstack([smoke, train_button], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(
    aws,
    batch,
    clip,
    detach,
    epochs,
    mo,
    os,
    out_path,
    seeds,
    smoke,
    train_button,
):
    _mode = mo.app_meta().mode
    _script = _mode == "script"
    _smoke_env = os.environ.get("SMOKE") == "1"
    smoke_test = bool(smoke.value) or (_script and _smoke_env)

    eff_batch = int(batch.value)
    eff_clip = float(clip.value)
    eff_detach = bool(detach.value)
    eff_aws = [float(a) for a in aws.value]
    eff_seeds = 2 if smoke_test else int(seeds.value)
    eff_epochs = 2 if smoke_test else int(epochs.value)
    eff_out = str(out_path.value).strip()

    # Script-mode overrides for short real runs (e.g. SM_SEEDS=10 SM_EPOCHS=3).
    if _script:
        if os.environ.get("SM_BATCH"):
            eff_batch = int(os.environ["SM_BATCH"])
        if os.environ.get("SM_CLIP") is not None:
            eff_clip = float(os.environ["SM_CLIP"])
        if os.environ.get("SM_DETACH") is not None:
            eff_detach = os.environ["SM_DETACH"] != "0"
        if os.environ.get("SM_AWS"):
            eff_aws = [
                float(a)
                for a in os.environ["SM_AWS"].split(",")
                if a.strip()
            ]
        if os.environ.get("SM_OUT"):
            eff_out = os.environ["SM_OUT"]
        if os.environ.get("SM_SEEDS"):
            eff_seeds = int(os.environ["SM_SEEDS"])
        if os.environ.get("SM_EPOCHS"):
            eff_epochs = int(os.environ["SM_EPOCHS"])

    train_clicked = bool(train_button.value) or _script
    return (
        eff_aws,
        eff_batch,
        eff_clip,
        eff_detach,
        eff_epochs,
        eff_out,
        eff_seeds,
        smoke_test,
        train_clicked,
    )


@app.cell
def _(re, torch):
    BATCH = 512
    LR = 0.1
    MOMENTUM = 0.0
    GRAD_CLIP = 1.0
    DETACH_TARGET = True
    D = 64
    H = 128
    C = 2
    PAPER_VOCAB = 100683

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

    def basic_english_normalize(line):
        """torchtext's legacy basic_english tokenizer, in plain Python."""
        line = line.lower()
        line = re.sub(r"\'", " '  ", line)
        line = re.sub(r"\"", "", line)
        line = re.sub(r"\.", " . ", line)
        line = re.sub(r"<br \/>", " ", line)
        line = re.sub(r",", " , ", line)
        line = re.sub(r"\(", " ( ", line)
        line = re.sub(r"\)", " ) ", line)
        line = re.sub(r"\!", " ! ", line)
        line = re.sub(r"\?", " ? ", line)
        line = re.sub(r"\;", " ", line)
        line = re.sub(r"\:", " ", line)
        line = re.sub(r"\s+", " ", line)
        return line.split()

    def encode_reviews(token_lists, stoi):
        """Review -> flat token ids plus bag offsets (no padding)."""
        _ids = []
        _offsets = [0]
        for _tokens in token_lists:
            if not _tokens:
                _tokens = ["<unk>"]
            for _word in _tokens:
                _ids.append(stoi.get(_word, 0))
            _offsets.append(len(_ids))
        return (
            torch.tensor(_ids, dtype=torch.int64),
            torch.tensor(_offsets, dtype=torch.int64),
        )

    def pack_batch(flat_ids, flat_offsets, idx, vocab_size):
        """Pack a (M, B) batch of review indices for a single embedding_bag call.

        On MPS this runs on the CPU (all arguments are integer tensors); the caller
        moves the returned packed ids and offsets to the training device. Doing the
        int64 gather/repeat_interleave there keeps them off the MPS command buffer,
        where a faulted kernel used to return garbage and make the lengths negative.

        On every other device (CUDA in particular) the tensors stay wherever they
        already are, so this is a single fused device-side gather.

        Token ids are shifted by ``model * vocab_size`` so one flat table holds one
        embedding table per model. Returns ``(packed_token_ids, bag_offsets)``.
        """
        M, B = idx.shape
        _bag = idx.reshape(-1)
        _starts = flat_offsets[_bag]
        _lengths = flat_offsets[_bag + 1] - _starts
        _bad = _lengths < 0
        assert not bool(_bad.any()), (
            "pack_batch: negative review length "
            f"({int(_bad.sum())} of {_lengths.numel()} bags, "
            f"min={int(_lengths.min())}); flat_offsets must be "
            "monotonically non-decreasing and idx must index existing bags"
        )
        _total = int(_lengths.sum())
        _excl = _lengths.cumsum(0) - _lengths
        assert bool((_excl[1:] >= _excl[:-1]).all()), (
            "pack_batch: bag offsets are not monotonically non-decreasing"
        )
        _pos = torch.arange(_total, device=flat_ids.device)
        _src = (
            _starts.repeat_interleave(_lengths)
            + _pos
            - _excl.repeat_interleave(_lengths)
        )
        _model_of_bag = torch.arange(M, device=flat_ids.device).repeat_interleave(
            B
        )
        _model_of_token = _model_of_bag.repeat_interleave(_lengths)
        return flat_ids[_src] + _model_of_token * vocab_size, _excl

    return (
        BATCH,
        C,
        D,
        DETACH_TARGET,
        GRAD_CLIP,
        H,
        LR,
        MOMENTUM,
        PAPER_VOCAB,
        basic_english_normalize,
        encode_reviews,
        pack_batch,
        t95,
    )


@app.cell
def _(C, D, F, H, nn, pack_batch, torch):
    def build_params(models, vocab_size, device):
        """Stacked init: one EmbeddingBag plus two Linears per (aw, seed) model.

        The output layer always has ``C + H`` rows so the classifier rows are
        initialized identically across AW for a given seed.
        """
        M = len(models)
        emb = torch.empty(M * vocab_size, D, device=device)
        Wh = torch.empty(M, H, D, device=device)
        bh = torch.empty(M, H, device=device)
        Wo = torch.empty(M, C + H, H, device=device)
        bo = torch.empty(M, C + H, device=device)
        for m, (_aw, _seed) in enumerate(models):
            torch.manual_seed(_seed)
            _ref_emb = nn.EmbeddingBag(vocab_size, D, mode="mean")
            _ref_h = nn.Linear(D, H)
            _ref_o = nn.Linear(H, C + H)
            with torch.no_grad():
                emb[m * vocab_size : (m + 1) * vocab_size] = _ref_emb.weight
                Wh[m] = _ref_h.weight
                bh[m] = _ref_h.bias
                Wo[m] = _ref_o.weight
                bo[m] = _ref_o.bias
        return {
            "emb": nn.Parameter(emb),
            "Wh": nn.Parameter(Wh),
            "bh": nn.Parameter(bh),
            "Wo": nn.Parameter(Wo),
            "bo": nn.Parameter(bo),
        }

    def forward_batched(params, packed, offsets, M, B):
        """(M, B) packed reviews -> classifier logits, a_hat, hidden a."""
        _emb = F.embedding_bag(
            packed, params["emb"], offsets, mode="mean"
        ).reshape(M, B, D)
        _a = torch.baddbmm(
            params["bh"].unsqueeze(1), _emb, params["Wh"].transpose(1, 2)
        )
        _out = torch.baddbmm(
            params["bo"].unsqueeze(1), _a, params["Wo"].transpose(1, 2)
        )
        return _out[..., :C], _out[..., C:], _a

    def evaluate_batched(
        params, flat_ids, flat_offsets, labels, vocab_size, device, M
    ):
        n = labels.numel()
        correct = torch.zeros(M, device=device)
        with torch.no_grad():
            for lo in range(0, n, 512):
                hi = min(lo + 512, n)
                idx = torch.arange(lo, hi, device=flat_ids.device)
                idx = idx.unsqueeze(0).expand(M, hi - lo)
                packed, offsets = pack_batch(
                    flat_ids, flat_offsets, idx, vocab_size
                )
                packed = packed.to(device)
                offsets = offsets.to(device)
                logits, _, _ = forward_batched(
                    params, packed, offsets, M, hi - lo
                )
                correct += (logits.argmax(dim=-1) == labels[lo:hi]).sum(dim=1)
        return correct / n

    def classifier_sd(params, M):
        flat = params["Wo"][:, :C].reshape(M, -1).detach()
        return flat.std(dim=1)

    return build_params, classifier_sd, evaluate_batched, forward_batched


@app.cell
def _(
    Counter,
    PAPER_VOCAB,
    basic_english_normalize,
    encode_reviews,
    load_dataset,
    mo,
    torch,
):
    @mo.persistent_cache
    def prepare_imdb():
        """Download IMDB, build the vocab, and flatten train/test reviews.

        Returns CPU tensors: flat token ids, bag offsets, and labels per split.
        """
        raw = load_dataset("stanfordnlp/imdb")
        train_tokens = [
            basic_english_normalize(t) for t in raw["train"]["text"]
        ]
        test_tokens = [
            basic_english_normalize(t) for t in raw["test"]["text"]
        ]

        train_counter = Counter()
        for _tokens in train_tokens:
            train_counter.update(_tokens)
        all_counter = Counter(train_counter)
        for _tokens in test_tokens:
            all_counter.update(_tokens)

        variants = {
            "train_only_with_unk": (sorted(train_counter), True),
            "train_only_without_unk": (sorted(train_counter), False),
            "train_plus_test_with_unk": (sorted(all_counter), True),
            "train_plus_test_without_unk": (sorted(all_counter), False),
        }
        sizes = {
            key: len(words) + (1 if has_unk else 0)
            for key, (words, has_unk) in variants.items()
        }
        # `split(" ")` would keep the empty-string token as a vocab entry.
        sizes["train_split_space_with_unk"] = len(train_counter) + 2

        chosen = next(
            (key for key, size in sizes.items() if size == PAPER_VOCAB),
            "train_only_with_unk",
        )
        words, has_unk = variants[chosen]
        itos = (["<unk>"] if has_unk else []) + words
        stoi = {word: i for i, word in enumerate(itos)}

        train_ids, train_offsets = encode_reviews(train_tokens, stoi)
        test_ids, test_offsets = encode_reviews(test_tokens, stoi)

        vocab_info = {
            "size": len(itos),
            "paper": PAPER_VOCAB,
            "chosen": chosen,
            "matches_paper": len(itos) == PAPER_VOCAB,
            "variants": sizes,
        }
        return {
            "vocab_info": vocab_info,
            "train_ids": train_ids,
            "train_offsets": train_offsets,
            "train_labels": torch.tensor(
                raw["train"]["label"], dtype=torch.int64
            ),
            "test_ids": test_ids,
            "test_offsets": test_offsets,
            "test_labels": torch.tensor(
                raw["test"]["label"], dtype=torch.int64
            ),
        }

    return (prepare_imdb,)


@app.cell
def _(
    F,
    LR,
    MOMENTUM,
    build_params,
    classifier_sd,
    evaluate_batched,
    forward_batched,
    math,
    mo,
    pack_batch,
    prepare_imdb,
    time,
    torch,
):
    @mo.persistent_cache
    def train_group(
        aws_list,
        seeds_n,
        epochs,
        device_type,
        batch_size,
        grad_clip,
        detach_target,
    ):
        """Train every (aw, seed) in one AW group together; return raw per-model data."""
        data = prepare_imdb()
        vocab_size = data["vocab_info"]["size"]
        device = torch.device(device_type)

        # On CUDA the integer data and the packing gather stay on the device. On
        # MPS (and CPU) they stay on the host: the CPU pack_batch workaround keeps
        # the int64 gather/repeat_interleave off the MPS command buffer, where a
        # faulted kernel used to return garbage and make the lengths negative.
        if device.type == "cuda":
            train_ids = data["train_ids"].to(device)
            train_offsets = data["train_offsets"].to(device)
            test_ids = data["test_ids"].to(device)
            test_offsets = data["test_offsets"].to(device)
        else:
            train_ids = data["train_ids"]
            train_offsets = data["train_offsets"]
            test_ids = data["test_ids"]
            test_offsets = data["test_offsets"]
        train_labels = data["train_labels"].to(device)
        test_labels = data["test_labels"].to(device)
        N = train_labels.numel()

        models = [
            (float(aw), int(seed))
            for aw in aws_list
            for seed in range(seeds_n)
        ]
        M = len(models)
        aws_t = torch.tensor(
            [aw for aw, _ in models], dtype=torch.float32, device=device
        )
        seeds = [seed for _, seed in models]
        params = build_params(models, vocab_size, device)
        opt = torch.optim.SGD(
            list(params.values()), lr=LR, momentum=MOMENTUM, nesterov=False
        )
        gens = [torch.Generator() for _ in range(M)]

        sd_hist = [[] for _ in range(M)]
        acc_hist = [[] for _ in range(M)]
        diverged_epoch = [None] * M
        epoch_seconds = []

        init_sd = classifier_sd(params, M).tolist()
        for m in range(M):
            sd_hist[m].append(init_sd[m])

        def mark_diverged(state, alive_mask, epoch):
            for m in range(M):
                if bool(state[m]) and diverged_epoch[m] is None:
                    diverged_epoch[m] = epoch
            return alive_mask & ~state

        def sanitize_grads(alive_mask):
            # A dead model can still emit NaN/inf gradients (0 upstream times a
            # NaN local Jacobian). Replace non-finite grads with zero and drop
            # the dead rows entirely before the global-norm clip, otherwise one
            # bad model would poison every other model's gradient.
            _emb_g = params["emb"].grad
            if _emb_g is not None:
                _emb_g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                _emb_g.view(M, vocab_size, -1)[~alive_mask] = 0.0
            for _key in ("Wh", "bh", "Wo", "bo"):
                _g = params[_key].grad
                if _g is not None:
                    _g.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    _g[~alive_mask] = 0.0

        def finite_weights():
            with torch.no_grad():
                _ok = torch.ones(M, dtype=torch.bool, device=device)
                for m in range(M):
                    _ok[m] = (
                        torch.isfinite(
                            params["emb"][
                                m * vocab_size : (m + 1) * vocab_size
                            ]
                        ).all()
                        & torch.isfinite(params["Wh"][m]).all()
                        & torch.isfinite(params["bh"][m]).all()
                        & torch.isfinite(params["Wo"][m]).all()
                        & torch.isfinite(params["bo"][m]).all()
                    )
                return _ok

        alive = torch.ones(M, dtype=torch.bool, device=device)
        for epoch in mo.status.progress_bar(
            range(1, epochs + 1),
            title=f"AW={aws_list}",
            show_rate=True,
            show_eta=True,
        ):
            for m in range(M):
                gens[m].manual_seed(seeds[m] * 1000 + epoch)
            perms = torch.stack(
                [torch.randperm(N, generator=gens[m]) for m in range(M)]
            )
            if device.type == "cuda":
                perms = perms.to(device)

            if device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()

            for lo in range(0, N, batch_size):
                hi = min(lo + batch_size, N)
                idx = perms[:, lo:hi]
                packed, offsets = pack_batch(
                    train_ids, train_offsets, idx, vocab_size
                )
                if device.type != "cuda":
                    packed = packed.to(device)
                    offsets = offsets.to(device)
                yb = train_labels[idx.to(device)]
                logits, a_hat, a = forward_batched(
                    params, packed, offsets, M, hi - lo
                )
                ce = (
                    F.cross_entropy(
                        logits.reshape(-1, 2),
                        yb.reshape(-1),
                        reduction="none",
                    )
                    .reshape(M, -1)
                    .mean(dim=1)
                )
                target = a.detach() if detach_target else a
                sm = (a_hat - target).pow(2).mean(dim=(1, 2))
                loss = ce + aws_t * sm
                finite = torch.isfinite(loss)
                if bool((alive & ~finite).any()):
                    alive = mark_diverged(alive & ~finite, alive, epoch)
                safe = torch.where(alive, loss, torch.zeros_like(loss))
                safe.sum().backward()
                if bool((~finite).any()) or not bool(alive.all()):
                    sanitize_grads(alive)
                if grad_clip > 0:
                    _norm = torch.nn.utils.clip_grad_norm_(
                        list(params.values()), grad_clip
                    )
                    if not bool(torch.isfinite(_norm)):
                        # A finite loss can still hide an inf gradient; the first
                        # clip then turns every grad into NaN. Wipe and re-clip.
                        sanitize_grads(alive)
                        torch.nn.utils.clip_grad_norm_(
                            list(params.values()), grad_clip
                        )
                opt.step()
                opt.zero_grad()

            weights_ok = finite_weights()
            if bool((alive & ~weights_ok).any()):
                alive = mark_diverged(alive & ~weights_ok, alive, epoch)

            sd_vec = classifier_sd(params, M)
            acc_vec = evaluate_batched(
                params,
                test_ids,
                test_offsets,
                test_labels,
                vocab_size,
                device,
                M,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            epoch_seconds.append(time.perf_counter() - start)

            for m in range(M):
                _sd = sd_vec[m].item()
                _acc = acc_vec[m].item()
                sd_hist[m].append(_sd if math.isfinite(_sd) else 0.0)
                acc_hist[m].append(_acc if math.isfinite(_acc) else 0.0)

        runs = [
            {
                "aw": aw,
                "seed": seed,
                "sd": sd_hist[m],
                "test_acc": acc_hist[m],
                "final_acc": acc_hist[m][-1],
                "diverged": diverged_epoch[m] is not None,
                "diverged_epoch": diverged_epoch[m],
            }
            for m, (aw, seed) in enumerate(models)
        ]
        weights = [
            {
                "aw": aw,
                "seed": seed,
                "emb": params["emb"][
                    m * vocab_size : (m + 1) * vocab_size
                ]
                .detach()
                .cpu()
                .clone(),
                "Wh": params["Wh"][m].detach().cpu().clone(),
                "bh": params["bh"][m].detach().cpu().clone(),
                "Wo": params["Wo"][m, :2].detach().cpu().clone(),
                "bo": params["bo"][m, :2].detach().cpu().clone(),
            }
            for m, (aw, seed) in enumerate(models)
        ]
        return {
            "aws": [float(aw) for aw in aws_list],
            "epochs": epochs,
            "seeds": seeds_n,
            "models": runs,
            "weights": weights,
            "epoch_seconds": epoch_seconds,
            "vocab_info": data["vocab_info"],
        }

    return (train_group,)


@app.cell
def _(
    device,
    device_name,
    eff_aws,
    eff_batch,
    eff_clip,
    eff_detach,
    eff_epochs,
    eff_seeds,
    mo,
    train_clicked,
    train_group,
):
    if train_clicked and eff_aws:
        _groups = (
            [eff_aws] if device.type == "cuda" else [[a] for a in eff_aws]
        )
        runs = []
        with mo.status.progress_bar(
            total=len(_groups), title="AW groups"
        ) as _bar:
            for _group in _groups:
                runs.append(
                    train_group(
                        _group,
                        eff_seeds,
                        eff_epochs,
                        device.type,
                        eff_batch,
                        eff_clip,
                        eff_detach,
                    )
                )
                _bar.update(subtitle=f"AW={_group} done")

        print(f"device: {device} ({device_name})")
        _vocab = runs[0]["vocab_info"]
        print(
            f"vocab size: {_vocab['size']} "
            f"(paper {_vocab['paper']}, match={_vocab['matches_paper']}, "
            f"chosen={_vocab['chosen']})"
        )
        print(f"vocab variants: {_vocab['variants']}")
        for _r in runs:
            _secs = _r["epoch_seconds"]
            _mean = sum(_secs) / len(_secs) if _secs else 0.0
            _aws = ",".join(f"{a:g}" for a in _r["aws"])
            print(
                f"AW=[{_aws}]: epochs={_r['epochs']} {_mean:.2f}s/epoch"
            )
            for _aw in _r["aws"]:
                _ms = [
                    m for m in _r["models"] if abs(m["aw"] - _aw) < 1e-9
                ]
                _ok = [m for m in _ms if not m["diverged"]]
                _n_div = len(_ms) - len(_ok)
                if _ok:
                    _sd = sum(m["sd"][-1] for m in _ok) / len(_ok)
                    _acc = sum(m["final_acc"] for m in _ok) / len(_ok)
                else:
                    _sd = 0.0
                    _acc = 0.0
                print(
                    f"  AW={_aw:g}: final SD={_sd:.4f} "
                    f"acc={_acc:.4f} (n={len(_ok)}, "
                    f"diverged={_n_div})"
                )
    else:
        runs = None
    return (runs,)


@app.cell
def _(
    LR,
    MOMENTUM,
    device,
    device_name,
    eff_aws,
    eff_batch,
    eff_clip,
    eff_detach,
    eff_epochs,
    eff_seeds,
    np,
    runs,
    smoke_test,
    t95,
):
    if runs:
        by_aw = {}
        for _group in runs:
            for _model in _group["models"]:
                by_aw.setdefault(_model["aw"], []).append(_model)

        results = {}
        for _aw in eff_aws:
            _ms = by_aw.get(_aw, [])
            _ok = [m for m in _ms if not m["diverged"]]
            _n = len(_ok)
            _n_diverged = len(_ms) - _n
            sd_mean, sd_ci = [], []
            if _n:
                for _e in range(eff_epochs + 1):
                    _vals = np.asarray(
                        [m["sd"][_e] for m in _ok], dtype=float
                    )
                    sd_mean.append(float(_vals.mean()))
                    if _n > 1:
                        _half = (
                            t95(_n - 1)
                            * float(_vals.std(ddof=1))
                            / np.sqrt(_n)
                        )
                    else:
                        _half = 0.0
                    sd_ci.append(_half)
                _acc = np.asarray(
                    [m["final_acc"] for m in _ok], dtype=float
                )
                _acc_mean = float(_acc.mean())
                if _n > 1:
                    _acc_ci = float(
                        t95(_n - 1) * _acc.std(ddof=1) / np.sqrt(_n)
                    )
                else:
                    _acc_ci = 0.0
            else:
                _acc_mean = 0.0
                _acc_ci = 0.0
            results[f"{_aw:g}"] = {
                "aw": float(_aw),
                "n": _n,
                "n_diverged": _n_diverged,
                "diverged_epochs": [
                    m["diverged_epoch"]
                    for m in _ms
                    if m["diverged"]
                ],
                "sd_mean": sd_mean,
                "sd_ci": sd_ci,
                "acc_mean": _acc_mean,
                "acc_ci": _acc_ci,
            }

        fig4 = {
            "config": {
                "aws": eff_aws,
                "seeds": eff_seeds,
                "epochs": eff_epochs,
                "batch_size": eff_batch,
                "lr": LR,
                "momentum": MOMENTUM,
                "nesterov": False,
                "grad_clip": eff_clip,
                "detach_target": eff_detach,
                "smoke": smoke_test,
            },
            "device": {"type": device.type, "name": device_name},
            "vocab": runs[0]["vocab_info"],
            "results": results,
        }
        fig4_weights = {}
        for _group in runs:
            for _model in _group["weights"]:
                fig4_weights.setdefault(
                    f"{_model['aw']:g}", []
                ).append(_model)
    else:
        fig4 = None
        fig4_weights = None
    return fig4, fig4_weights


@app.cell
def _(Path, eff_out, fig4, json, math, mo):
    def _finite(obj):
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_finite(v) for v in obj]
        return obj

    if fig4 is not None:
        _payload = json.dumps(_finite(fig4)).encode("utf-8")
        _loc = mo.notebook_location()
        if eff_out:
            _path = Path(eff_out)
            if not _path.is_absolute() and _loc is not None:
                _path = Path(str(_loc)) / _path
        elif _loc is not None:
            _path = Path(str(_loc)) / "public" / "data" / "imdb_fig4.json"
        else:
            _path = None

        if _path is None:
            _msg = "notebook location unavailable (set SM_OUT to a path)"
        else:
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
                    filename="imdb_fig4.json",
                    mimetype="application/json",
                    label="Download imdb_fig4.json",
                ),
            ]
        )
    else:
        _out = mo.md("_Run training to produce results._")
    _out
    return


@app.cell
def _(alt, fig4, pd):
    if fig4 is None:
        charts = None
    else:
        _colors = {
            "0": "#000000",
            "100": "#00e000",
            "500": "#8a2be2",
        }
        _keys = [f"{a:g}" for a in fig4["config"]["aws"]]
        _dom = [k for k in _keys if k in _colors]
        _rng = [_colors[k] for k in _dom]
        _color = alt.Color(
            "aw:N",
            scale=alt.Scale(domain=_dom, range=_rng),
            legend=alt.Legend(title="AW"),
        )

        _rows = []
        for _key, _res in fig4["results"].items():
            for _e, (_m, _c) in enumerate(
                zip(_res["sd_mean"], _res["sd_ci"])
            ):
                _rows.append(
                    {
                        "aw": _key,
                        "epoch": _e,
                        "mean": _m,
                        "lo": max(0.0, _m - _c),
                        "hi": _m + _c,
                    }
                )
        _df = pd.DataFrame(_rows)
        _base = alt.Chart(_df).encode(x=alt.X("epoch:Q", title="epoch"))
        _band = _base.mark_area(opacity=0.15).encode(
            y=alt.Y("lo:Q", title="SD of classifier weights"),
            y2="hi:Q",
            color=_color,
        )
        _line = _base.mark_line().encode(y="mean:Q", color=_color)
        fig4a = (_band + _line).properties(
            width=420, height=280, title="Fig 4A — SD vs epoch"
        )

        _rows_c = []
        for _key, _res in fig4["results"].items():
            _rows_c.append(
                {
                    "aw": _key,
                    "mean": _res["acc_mean"],
                    "lo": max(0.0, _res["acc_mean"] - _res["acc_ci"]),
                    "hi": _res["acc_mean"] + _res["acc_ci"],
                }
            )
        _df_c = pd.DataFrame(_rows_c)
        _base_c = alt.Chart(_df_c).encode(
            x=alt.X("aw:N", title="AW", sort=_dom)
        )
        _bar = _base_c.mark_bar().encode(
            y=alt.Y(
                "mean:Q",
                title="test accuracy",
                scale=alt.Scale(zero=False),
            ),
            color=_color,
        )
        _err = _base_c.mark_errorbar().encode(
            y=alt.Y("lo:Q"), y2="hi:Q", color=_color
        )
        fig4c = (_bar + _err).properties(
            width=420, height=280, title="Fig 4C — final accuracy"
        )
        charts = [fig4a, fig4c]
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
def _(mo):
    mo.md("""
    ### Implementation notes and deviations

    - **Batch size 512.** The paper's Table 1 lists 64; the repo's MNIST correction
      establishes 512. Override with `SM_BATCH=64` (or the UI).
    - **Detached target.** By default `a_hat` predicts `a.detach()`. The paper's
      methods say both `a_hat` and `a` are functions of the weights (i.e. not
      detached), which is the stronger regularizer. `SM_DETACH=0` (or the UI) trains
      against the live activations.
    - **Gradient clipping.** The detached self-modeling gradient at `lr=0.1` blows up
      for AW=500 within the first epoch, so the global gradient norm is clipped at 1.0.
      Baseline gradient norms stay below 0.5, so the baseline is untouched. `SM_CLIP=0`
      (or the UI) skips the `clip_grad_norm_` call entirely.
    - **Divergence guard.** If a model's loss or any of its weights becomes
      non-finite, that model stops updating (its gradient rows are masked before the
      global clip, so it cannot corrupt the others), keeps training the rest, and is
      recorded with the epoch it happened. Diverged seeds are excluded from the
      means/CIs and reported per AW as `n_diverged`.
    - **Tokenizer / vocab.** `.split()` drops the empty-string token, giving 100,682
      unique train tokens plus `<unk>` = the paper's 100,683.
    - **Pruned weights.** `fig4_weights` holds the final embedding, hidden `W`/`b`, and
      the two classifier rows per model for the later RLCT step. It is kept in memory
      only and is not part of `imdb_fig4.json`.
    """)
    return


if __name__ == "__main__":
    app.run()
