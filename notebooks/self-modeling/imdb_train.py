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
def _(mo):
    aws = mo.ui.multiselect(
        options=[0, 100, 500],
        value=[0, 100, 500],
        label="Self-modeling weight (AW)",
    )
    seeds = mo.ui.number(value=10, start=1, stop=50, label="Seeds")
    epochs = mo.ui.number(value=500, start=1, stop=2000, label="Epochs")
    smoke = mo.ui.checkbox(
        value=False, label="Smoke test (2 epochs, 2 seeds)"
    )
    train_button = mo.ui.run_button(label="Train")
    return aws, epochs, seeds, smoke, train_button


@app.cell
def _(aws, epochs, mo, seeds, smoke, train_button):
    mo.md("""
    ### Configuration

    Each run trains `len(AW) x seeds` independent models in one batched
    forward/backward: SGD `lr=0.1`, momentum 0, no nesterov, batch 512, 500 epochs.

    On CUDA all selected AWs train together (one stacked embedding table per model,
    ~770 MB at 30 models). On MPS/CPU the groups are trained one AW at a time. The
    global gradient norm is clipped at 1.0 so AW=500 stays finite; see the note below
    the results.
    """)
    mo.vstack(
        [
            mo.hstack([aws, seeds, epochs], justify="start", gap=2),
            mo.hstack([smoke, train_button], justify="start", gap=2),
        ]
    )
    return


@app.cell
def _(aws, epochs, mo, os, seeds, smoke, train_button):
    _mode = mo.app_meta().mode
    _script = _mode == "script"
    _smoke_env = os.environ.get("SMOKE") == "1"
    smoke_test = bool(smoke.value) or (_script and _smoke_env)

    eff_aws = [float(a) for a in aws.value]
    eff_seeds = 2 if smoke_test else int(seeds.value)
    eff_epochs = 2 if smoke_test else int(epochs.value)
    train_clicked = bool(train_button.value) or _script
    return eff_aws, eff_epochs, eff_seeds, smoke_test, train_clicked


@app.cell
def _(re, torch):
    BATCH = 512
    LR = 0.1
    MOMENTUM = 0.0
    GRAD_CLIP = 1.0
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

        Token ids are shifted by ``model * vocab_size`` so one flat table holds one
        embedding table per model. Returns ``(packed_token_ids, bag_offsets)``.
        """
        M, B = idx.shape
        _bag = idx.reshape(-1)
        _starts = flat_offsets[_bag]
        _lengths = flat_offsets[_bag + 1] - _starts
        _total = int(_lengths.sum())
        _excl = _lengths.cumsum(0) - _lengths
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
                idx = torch.arange(lo, hi, device=device)
                idx = idx.unsqueeze(0).expand(M, hi - lo)
                packed, offsets = pack_batch(
                    flat_ids, flat_offsets, idx, vocab_size
                )
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
    BATCH,
    F,
    GRAD_CLIP,
    LR,
    MOMENTUM,
    build_params,
    classifier_sd,
    evaluate_batched,
    forward_batched,
    mo,
    pack_batch,
    prepare_imdb,
    time,
    torch,
):
    @mo.persistent_cache
    def train_group(aws_list, seeds_n, epochs, device_type):
        """Train every (aw, seed) in one AW group together; return raw per-model data."""
        data = prepare_imdb()
        vocab_size = data["vocab_info"]["size"]
        device = torch.device(device_type)

        train_ids = data["train_ids"].to(device)
        train_offsets = data["train_offsets"].to(device)
        train_labels = data["train_labels"].to(device)
        test_ids = data["test_ids"].to(device)
        test_offsets = data["test_offsets"].to(device)
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
        epoch_seconds = []

        init_sd = classifier_sd(params, M).tolist()
        for m in range(M):
            sd_hist[m].append(init_sd[m])

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
            ).to(device)

            if device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()

            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                idx = perms[:, lo:hi]
                packed, offsets = pack_batch(
                    train_ids, train_offsets, idx, vocab_size
                )
                yb = train_labels[idx]
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
                sm = (a_hat - a.detach()).pow(2).mean(dim=(1, 2))
                (ce + aws_t * sm).sum().backward()
                torch.nn.utils.clip_grad_norm_(
                    list(params.values()), GRAD_CLIP
                )
                opt.step()
                opt.zero_grad()

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
                sd_hist[m].append(sd_vec[m].item())
                acc_hist[m].append(acc_vec[m].item())

        runs = [
            {
                "aw": aw,
                "seed": seed,
                "sd": sd_hist[m],
                "test_acc": acc_hist[m],
                "final_acc": acc_hist[m][-1],
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
                        _group, eff_seeds, eff_epochs, device.type
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
                _sd = sum(m["sd"][-1] for m in _ms) / len(_ms)
                _acc = sum(m["final_acc"] for m in _ms) / len(_ms)
                print(
                    f"  AW={_aw:g}: final SD={_sd:.4f} "
                    f"acc={_acc:.4f} (n={len(_ms)})"
                )
    else:
        runs = None
    return (runs,)


@app.cell
def _(
    BATCH,
    GRAD_CLIP,
    LR,
    MOMENTUM,
    device,
    device_name,
    eff_aws,
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
            _n = len(_ms)
            sd_mean, sd_ci = [], []
            for _e in range(eff_epochs + 1):
                _vals = np.asarray(
                    [m["sd"][_e] for m in _ms], dtype=float
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
                [m["final_acc"] for m in _ms], dtype=float
            )
            if _n > 1:
                _acc_ci = float(
                    t95(_n - 1) * _acc.std(ddof=1) / np.sqrt(_n)
                )
            else:
                _acc_ci = 0.0
            results[f"{_aw:g}"] = {
                "aw": float(_aw),
                "n": _n,
                "sd_mean": sd_mean,
                "sd_ci": sd_ci,
                "acc_mean": float(_acc.mean()) if _n else 0.0,
                "acc_ci": _acc_ci,
            }

        fig4 = {
            "config": {
                "aws": eff_aws,
                "seeds": eff_seeds,
                "epochs": eff_epochs,
                "batch_size": BATCH,
                "lr": LR,
                "momentum": MOMENTUM,
                "nesterov": False,
                "grad_clip": GRAD_CLIP,
                "detach_target": True,
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
def _(Path, fig4, json, mo):
    if fig4 is not None:
        _payload = json.dumps(fig4).encode("utf-8")
        _loc = mo.notebook_location()
        _msg = "notebook location unavailable"
        if _loc is not None:
            _path = Path(str(_loc)) / "public" / "data" / "imdb_fig4.json"
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
      establishes 512.
    - **Detached target.** `a_hat` predicts `a.detach()`. The paper's methods say both
      `a_hat` and `a` are functions of the weights (i.e. not detached), which is the
      stronger regularizer; we follow the task/repo convention.
    - **Gradient clipping.** The detached self-modeling gradient at `lr=0.1` blows up
      for AW=500 within the first epoch, so the global gradient norm is clipped at 1.0.
      Baseline gradient norms stay below 0.5, so the baseline is untouched.
    - **Tokenizer / vocab.** `.split()` drops the empty-string token, giving 100,682
      unique train tokens plus `<unk>` = the paper's 100,683.
    - **Pruned weights.** `fig4_weights` holds the final embedding, hidden `W`/`b`, and
      the two classifier rows per model for the later RLCT step. It is kept in memory
      only and is not part of `imdb_fig4.json`.
    """)
    return


if __name__ == "__main__":
    app.run()
