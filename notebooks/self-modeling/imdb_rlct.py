# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "numpy==2.5.3",
#     "datasets==5.1.0",
#     "altair==6.3.0",
#     "pandas==3.0.6",
#     "devinterp==2.0.1",
#     "zarr==3.1.3",
# ]
# ///
# smoke: skip

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import importlib.metadata
    import json
    import os
    import re
    import shutil
    import tempfile
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
        hashlib,
        importlib,
        json,
        load_dataset,
        mo,
        nn,
        np,
        os,
        pd,
        re,
        shutil,
        tempfile,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Self-Modeling — Figure 4B RLCT (IMDB)

    Estimates the local learning coefficient (LLC) of the pruned text
    classifier at epoch 250 for each self-modeling weight AW. The paper
    reports baseline ~13.7, AW 100 ~11.2, AW 500 ~5.2.

    Input is the pruned classifier written by `imdb_train.py` with
    `SM_SAVE_EPOCHS`: the full `EmbeddingBag` table, the hidden `W`/`b`, and
    the two classifier rows. Read them from `IMDB_CKPT_DIR` (default: this
    notebook's `checkpoints/imdb/`), e.g.
    `checkpoints/imdb/epoch250/aw0_seed0.pt`.
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
    mo.md(f"""
    ### Conventions

    **Device:** `{device}` — {device_name}

    - **Network.** The pruned checkpoint has no self-model rows, so we rebuild
      the classifier only: `EmbeddingBag(dim=64, mode="mean")` →
      `Linear(64,128)` → `Linear(128,2)`. The hidden layer is linear in the
      trainer too, so the rebuild is exact.
    - **Vocab.** The same `basic_english` tokenizer and `train_only_with_unk`
      vocabulary as `imdb_train.py` (100,683 entries, `<unk>` at index 0).
      The tokenizer/packing code is copied here; nothing is imported from
      sibling notebooks.
    - **Loss.** Mean cross-entropy per review on the IMDB **train** split
      (optionally the first `RLCT_SUBSET` reviews). Each batch is packed into
      flat `(ids, offsets)` for `EmbeddingBag`; devinterp's SGLD does the
      shuffling, so init loss needs no separate order.
    - **SGLD.** `lr=1e-3`, localization `10`, `nbeta = batch/log(batch)` with
      batch 64. These are the paper's IMDB supplement values.
    """)
    return


@app.cell
def _(
    Counter,
    F,
    Path,
    hashlib,
    importlib,
    load_dataset,
    mo,
    nn,
    np,
    os,
    re,
    shutil,
    tempfile,
    time,
    torch,
):
    N_CLASSES = 2
    D_MODEL = 64
    HIDDEN = 128
    PAPER_VOCAB = 100683
    TRAIN_SIZE = 25000

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

    def mean_ci(values):
        arr = np.asarray(values, dtype=float)
        n = arr.size
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return float(arr[0]), 0.0
        half = t95(n - 1) * float(arr.std(ddof=1)) / np.sqrt(n)
        return float(arr.mean()), float(half)

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
        """Pack a (M, B) batch of review indices into one embedding_bag call.

        Copied from `imdb_train.py`. Token ids are shifted by
        ``model * vocab_size`` so one flat table can hold one embedding table
        per model; here M is always 1. Runs on the CPU so the int64 gather
        stays off the MPS command buffer.
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
        _pos = torch.arange(_total, device=flat_ids.device)
        _src = (
            _starts.repeat_interleave(_lengths)
            + _pos
            - _excl.repeat_interleave(_lengths)
        )
        _model_of_bag = torch.arange(
            M, device=flat_ids.device
        ).repeat_interleave(B)
        _model_of_token = _model_of_bag.repeat_interleave(_lengths)
        return flat_ids[_src] + _model_of_token * vocab_size, _excl

    class ImdbClassifier(nn.Module):
        """Classifier-only network exactly as built by `imdb_train.py`."""

        def __init__(self, vocab_size):
            super().__init__()
            self.emb = nn.EmbeddingBag(vocab_size, D_MODEL, mode="mean")
            self.hidden = nn.Linear(D_MODEL, HIDDEN)
            self.out = nn.Linear(HIDDEN, N_CLASSES)

        def forward(self, packed, offsets):
            x = self.emb(packed, offsets)
            x = self.hidden(x)
            return self.out(x)

    class IndexDataset(torch.utils.data.Dataset):
        """Rows are (train index, label); the review span is looked up on device."""

        def __init__(self, labels, n):
            m = labels.numel() if n is None else min(int(n), labels.numel())
            self.pairs = torch.stack(
                [
                    torch.arange(m, dtype=torch.int64),
                    labels[:m].detach().long().cpu(),
                ],
                dim=1,
            )

        def __len__(self):
            return self.pairs.shape[0]

        def __getitem__(self, i):
            return {"input_ids": self.pairs[i]}

    @mo.persistent_cache
    def prepare_imdb_train():
        """Tokenize IMDB train, build the trainer's vocab, and pack reviews."""
        raw = load_dataset("stanfordnlp/imdb", split="train")
        token_lists = [
            basic_english_normalize(text) for text in raw["text"]
        ]
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # `imdb_train.py` selects `train_only_with_unk` because
        # len(train_counter) + 1 == the paper's 100,683.
        words = sorted(counter)
        itos = ["<unk>"] + words
        stoi = {word: i for i, word in enumerate(itos)}
        flat_ids, flat_offsets = encode_reviews(token_lists, stoi)
        vocab_info = {
            "size": len(itos),
            "paper": PAPER_VOCAB,
            "chosen": "train_only_with_unk",
            "matches_paper": len(itos) == PAPER_VOCAB,
        }
        return {
            "vocab_info": vocab_info,
            "flat_ids": flat_ids,
            "flat_offsets": flat_offsets,
            "labels": torch.tensor(raw["label"], dtype=torch.int64),
        }

    def subset_data(data, subset_n):
        """First `subset_n` reviews, sharing the full-train vocabulary."""
        if subset_n is None:
            return data["flat_ids"], data["flat_offsets"], data["labels"]
        m = min(int(subset_n), data["labels"].numel())
        end = int(data["flat_offsets"][m])
        return (
            data["flat_ids"][:end],
            data["flat_offsets"][: m + 1],
            data["labels"][:m],
        )

    def make_loss_fn(flat_ids, flat_offsets, vocab_size, device):
        """Per-review CE, packed into (ids, offsets) for `EmbeddingBag`."""

        def loss_fn(model, input_ids):
            idx = input_ids[:, 0].long().cpu()
            packed, offsets = pack_batch(
                flat_ids, flat_offsets, idx.unsqueeze(0), vocab_size
            )
            packed = packed.to(device)
            offsets = offsets.to(device)
            logits = model(packed, offsets)
            labels = input_ids[:, 1].long().to(device)
            return F.cross_entropy(
                logits, labels, reduction="none"
            ).unsqueeze(1)

        return loss_fn

    def load_pruned_model(path, device, vocab_size):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = ImdbClassifier(vocab_size)
        model.load_state_dict(
            {
                "emb.weight": blob["emb"],
                "hidden.weight": blob["Wh"],
                "hidden.bias": blob["bh"],
                "out.weight": blob["Wo"],
                "out.bias": blob["bo"],
            }
        )
        model.eval()
        return model.to(device)

    def parse_pruned_models(epoch_dir):
        """Recover (aw, seed) from `aw<aw>_seed<seed>.pt` files."""
        models = []
        for path in sorted(epoch_dir.glob("aw*_seed*.pt")):
            aw_text, seed_text = path.stem.split("_seed")
            models.append(
                {
                    "aw": float(aw_text.removeprefix("aw")),
                    "seed": int(seed_text),
                    "path": path,
                }
            )
        return models

    def file_fingerprint(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def loss_quarters(loss_trace):
        """Chain-averaged loss over four equal draw quarters (q1..q4)."""
        arr = np.asarray(loss_trace, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        chain_mean = arr.mean(axis=0)
        edges = np.linspace(0, chain_mean.shape[0], 5).astype(int)
        out = []
        for index in range(4):
            lo, hi = edges[index], edges[index + 1]
            out.append(
                float(chain_mean[lo:hi].mean())
                if hi > lo
                else float("nan")
            )
        return out

    # Single source of truth for every RLCT hyperparameter.
    RLCT_CONFIG = {
        "lr": 1e-3,
        "localization": 10.0,
        "localization_note": "paper IMDB supplement value",
        "batch_size": 64,
        "nbeta": 64.0 / float(np.log(64.0)),
        "nbeta_convention": "batch/log(batch) with batch=64",
        "num_chains": 4,
        "num_draws": 2000,
        "num_burnin_steps": 0,
        "num_steps_bw_draws": 1,
        "seed": 0,
        "num_init_loss_batches": 97,
        "noise_level": 1.0,
        "llc_weight_decay": 0.0,
        "bounding_box_size": None,
        "sampling_method": "sgmcmc_sgld",
        "gradient_accumulation_steps": 1,
        "shuffle": True,
        "match_sampling_input_ids_across_chains": True,
        "param_masks": None,
        "save_metrics": False,
        "init_noise": None,
        "epoch": 250,
        "calibration_localizations": (10.0,),
        "calibration_num_draws": (500, 1000, 2000),
        "calibration_num_chains": 2,
        "calibration_pairs": (
            ("aw0_seed0", 0.0, 0),
            ("aw500_seed0", 500.0, 0),
        ),
        "plateau_tolerance": 0.02,
    }

    def init_batches(subset_n):
        n_train = TRAIN_SIZE if subset_n is None else int(subset_n)
        return max(
            1,
            min(
                RLCT_CONFIG["num_init_loss_batches"],
                n_train // RLCT_CONFIG["batch_size"],
            ),
        )

    def estimate_model(
        path_str,
        aw,
        seed,
        fingerprint,
        device_type,
        num_chains,
        num_draws,
        localization,
        subset_n,
        lr,
        nbeta,
        batch_size,
        num_burnin_steps,
        num_steps_bw_draws,
        init_seed,
        noise_level,
        llc_weight_decay,
        sampling_method,
        init_loss_batches,
    ):
        """Run devinterp's llc() once for one pruned checkpoint."""
        from devinterp.slt.llc import llc

        device = torch.device(device_type)
        data = prepare_imdb_train()
        vocab_size = data["vocab_info"]["size"]
        flat_ids, flat_offsets, labels = subset_data(data, subset_n)
        model = load_pruned_model(Path(path_str), device, vocab_size)
        dataset = IndexDataset(labels, None)
        loss_fn = make_loss_fn(flat_ids, flat_offsets, vocab_size, device)

        tmp = tempfile.mkdtemp(prefix="imdb_rlct_")
        started = time.perf_counter()
        try:
            result = llc(
                model,
                dataset,
                {},
                lr=lr,
                n_beta=nbeta,
                param_masks=None,
                loss_fn=loss_fn,
                num_chains=num_chains,
                num_draws=num_draws,
                batch_size=batch_size,
                num_burnin_steps=num_burnin_steps,
                num_steps_bw_draws=num_steps_bw_draws,
                localization=float(localization),
                num_init_loss_batches=init_loss_batches,
                init_seed=init_seed,
                device=device_type,
                noise_level=noise_level,
                llc_weight_decay=llc_weight_decay,
                bounding_box_size=None,
                sampling_method=sampling_method,
                gradient_accumulation_steps=1,
                shuffle=True,
                match_sampling_input_ids_across_chains=True,
                init_noise=None,
                save_metrics=False,
                output_path=os.path.join(tmp, "samples.zarr"),
            )
            per_chain = np.asarray(
                result["llc_per_chain"].values, dtype=float
            )
            value = float(per_chain.mean())
            quarters = loss_quarters(result["loss_trace"].values)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {
            "aw": float(aw),
            "seed": int(seed),
            "fingerprint": fingerprint,
            "llc": value,
            "loss_quarters": quarters,
            "seconds": time.perf_counter() - started,
        }

    @mo.persistent_cache
    def estimate_model_cached(
        path_str,
        aw,
        seed,
        fingerprint,
        device_type,
        num_chains,
        num_draws,
        localization,
        subset_n,
        lr,
        nbeta,
        batch_size,
        num_burnin_steps,
        num_steps_bw_draws,
        init_seed,
        noise_level,
        llc_weight_decay,
        sampling_method,
        init_loss_batches,
    ):
        return estimate_model(
            path_str,
            aw,
            seed,
            fingerprint,
            device_type,
            num_chains,
            num_draws,
            localization,
            subset_n,
            lr,
            nbeta,
            batch_size,
            num_burnin_steps,
            num_steps_bw_draws,
            init_seed,
            noise_level,
            llc_weight_decay,
            sampling_method,
            init_loss_batches,
        )

    def plateau_choice(results, localization, tolerance):
        """Shortest draws whose q4 is within tolerance of q3 everywhere."""
        loc_key = f"{localization:g}"
        draw_keys = None
        for per_loc in results.values():
            if loc_key in per_loc:
                draw_keys = sorted(per_loc[loc_key], key=int)
                break
        if not draw_keys:
            return None
        for draws in draw_keys:
            ok = True
            for per_loc in results.values():
                row = per_loc.get(loc_key, {}).get(draws)
                if row is None:
                    ok = False
                    break
                q3, q4 = row["loss_quarters"][2], row["loss_quarters"][3]
                if abs(q3) == 0:
                    ok = False
                    break
                if abs(q4 - q3) / abs(q3) > tolerance:
                    ok = False
                    break
            if ok:
                return int(draws)
        return None

    def rlct_hyperparams(device_type, subset_n, epoch):
        return {
            "devinterp_version": importlib.metadata.version("devinterp"),
            "estimator": "devinterp.slt.llc.llc",
            "sampler": "SGLD (sgmcmc_sgld)",
            "loss": (
                "mean cross-entropy per review on the IMDB train split"
            ),
            "model": (
                "EmbeddingBag(dim=64, mean) -> Linear(64,128) -> "
                "Linear(128,2)"
            ),
            "vocab": "basic_english, train_only_with_unk, 100683",
            "epoch": int(epoch),
            "lr": RLCT_CONFIG["lr"],
            "localization": RLCT_CONFIG["localization"],
            "localization_note": RLCT_CONFIG["localization_note"],
            "nbeta": RLCT_CONFIG["nbeta"],
            "nbeta_convention": RLCT_CONFIG["nbeta_convention"],
            "batch_size": RLCT_CONFIG["batch_size"],
            "num_chains": RLCT_CONFIG["num_chains"],
            "num_draws": RLCT_CONFIG["num_draws"],
            "num_burnin_steps": RLCT_CONFIG["num_burnin_steps"],
            "num_steps_bw_draws": RLCT_CONFIG["num_steps_bw_draws"],
            "num_init_loss_batches": RLCT_CONFIG["num_init_loss_batches"],
            "seed": RLCT_CONFIG["seed"],
            "noise_level": RLCT_CONFIG["noise_level"],
            "llc_weight_decay": RLCT_CONFIG["llc_weight_decay"],
            "sampling_method": RLCT_CONFIG["sampling_method"],
            "precision": "fp32",
            "device": str(device_type),
            "subset": None if subset_n is None else int(subset_n),
        }

    return (
        D_MODEL,
        HIDDEN,
        ImdbClassifier,
        IndexDataset,
        N_CLASSES,
        PAPER_VOCAB,
        RLCT_CONFIG,
        TRAIN_SIZE,
        basic_english_normalize,
        encode_reviews,
        estimate_model,
        estimate_model_cached,
        file_fingerprint,
        init_batches,
        load_pruned_model,
        loss_quarters,
        make_loss_fn,
        mean_ci,
        pack_batch,
        parse_pruned_models,
        plateau_choice,
        prepare_imdb_train,
        rlct_hyperparams,
        subset_data,
        t95,
    )


@app.cell
def _(Path, mo, os):
    checkpoint_dir = Path(
        os.environ.get("IMDB_CKPT_DIR", "")
        or os.environ.get("SM_SAVE_DIR", "")
        or str(
            Path(str(mo.notebook_location() or "."))
            / "checkpoints"
            / "imdb"
        )
    )
    return (checkpoint_dir,)


@app.cell
def _(RLCT_CONFIG, mo, os, checkpoint_dir, epoch):
    epoch_dir = checkpoint_dir / f"epoch{epoch}"
    mo.md(
        f"**Checkpoints:** `{checkpoint_dir}` · "
        f"**epoch directory:** `{epoch_dir}`"
    )
    return (epoch_dir,)


@app.cell
def _(RLCT_CONFIG, mo, os):
    _script = mo.app_meta().mode == "script"
    _env_mode = os.environ.get("RLCT_MODE", "").strip().lower()
    mode = _env_mode if _env_mode in ("calibrate", "sweep") else "calibrate"

    _env_subset = os.environ.get("RLCT_SUBSET", "").strip()
    subset_n = int(_env_subset) if _env_subset else None

    _env_epoch = os.environ.get("RLCT_EPOCH", "").strip()
    epoch = int(_env_epoch) if _env_epoch else RLCT_CONFIG["epoch"]

    _env_cal_locs = os.environ.get("RLCT_CAL_LOCS", "").strip()
    calibration_localizations = (
        tuple(
            float(part)
            for part in _env_cal_locs.split(",")
            if part.strip()
        )
        if _env_cal_locs
        else RLCT_CONFIG["calibration_localizations"]
    )

    _env_cal_models = os.environ.get("RLCT_CAL_MODELS", "").strip()
    calibration_models = (
        tuple(
            part.strip()
            for part in _env_cal_models.split(",")
            if part.strip()
        )
        if _env_cal_models
        else None
    )

    _env_draws = os.environ.get("RLCT_CAL_DRAWS", "").strip()
    if not _env_draws:
        _env_draws = os.environ.get("RLCT_DRAWS", "").strip()
    if _env_draws:
        calibration_draws = tuple(
            int(part) for part in _env_draws.split(",") if part.strip()
        )
    else:
        calibration_draws = RLCT_CONFIG["calibration_num_draws"]

    _env_chains = os.environ.get("RLCT_CHAINS", "").strip()
    calibration_chains = (
        int(_env_chains)
        if _env_chains
        else RLCT_CONFIG["calibration_num_chains"]
    )

    _env_sweep_draws = os.environ.get("RLCT_SWEEP_DRAWS", "").strip()
    sweep_draws = (
        int(_env_sweep_draws)
        if _env_sweep_draws
        else RLCT_CONFIG["num_draws"]
    )
    _env_sweep_chains = os.environ.get("RLCT_SWEEP_CHAINS", "").strip()
    sweep_chains = (
        int(_env_sweep_chains)
        if _env_sweep_chains
        else RLCT_CONFIG["num_chains"]
    )

    run_button = mo.ui.run_button(label="Run RLCT estimation")
    mode_picker = mo.ui.dropdown(
        options=["calibrate", "sweep"], value=mode, label="Mode"
    )
    run_requested = _script or bool(run_button.value)
    selected_mode = mode if _script else str(mode_picker.value)

    _out = mo.vstack(
        [
            mode_picker,
            run_button,
            mo.md(
                f"**Mode:** `{selected_mode}` · **subset:** `{subset_n}` · "
                f"**epoch:** `{epoch}` · "
                f"**calibration localizations:** "
                f"`{calibration_localizations}` · "
                f"**calibration draws:** `{calibration_draws}` · "
                f"**calibration chains:** `{calibration_chains}` · "
                f"**calibration models:** `{calibration_models}` · "
                f"**sweep draws:** `{sweep_draws}` · "
                f"**sweep chains:** `{sweep_chains}`\n\n"
                "Set `IMDB_CKPT_DIR`, `RLCT_MODE`, `RLCT_SUBSET`, "
                "`RLCT_EPOCH`, `RLCT_DRAWS`, `RLCT_CAL_LOCS`, "
                "`RLCT_CAL_DRAWS`, `RLCT_CAL_MODELS`, `RLCT_CHAINS`, "
                "`RLCT_SWEEP_DRAWS`, `RLCT_SWEEP_CHAINS`, and `RLCT_OUT` "
                "to drive this headlessly."
            ),
        ]
    )
    _out
    return (
        calibration_chains,
        calibration_draws,
        calibration_localizations,
        calibration_models,
        epoch,
        run_requested,
        selected_mode,
        subset_n,
        sweep_chains,
        sweep_draws,
    )


@app.cell
def _(
    RLCT_CONFIG,
    calibration_chains,
    calibration_draws,
    calibration_localizations,
    calibration_models,
    device,
    epoch_dir,
    estimate_model,
    file_fingerprint,
    init_batches,
    mo,
    parse_pruned_models,
    plateau_choice,
    run_requested,
    selected_mode,
    subset_n,
):
    if run_requested and selected_mode == "calibrate":
        _models = parse_pruned_models(epoch_dir)
        _by_key = {
            f"aw{_m['aw']:g}_seed{_m['seed']}": _m for _m in _models
        }
        _results = {}
        _lines = []
        _pairs = RLCT_CONFIG["calibration_pairs"]
        if calibration_models is not None:
            _pairs = tuple(
                _pair
                for _pair in _pairs
                if _pair[0] in calibration_models
            )
        for _key, _aw, _seed in _pairs:
            _m = _by_key.get(_key)
            if _m is None:
                raise FileNotFoundError(
                    f"calibration model {_key} not found in {epoch_dir}; "
                    f"expected aw{_aw:g}_seed{_seed}.pt"
                )
            _fp = file_fingerprint(_m["path"])
            _per_loc = {}
            for _loc in calibration_localizations:
                _per_draws = {}
                for _draws in calibration_draws:
                    _entry = estimate_model(
                        str(_m["path"]),
                        _m["aw"],
                        _m["seed"],
                        _fp,
                        device.type,
                        calibration_chains,
                        _draws,
                        _loc,
                        subset_n,
                        RLCT_CONFIG["lr"],
                        RLCT_CONFIG["nbeta"],
                        RLCT_CONFIG["batch_size"],
                        RLCT_CONFIG["num_burnin_steps"],
                        RLCT_CONFIG["num_steps_bw_draws"],
                        RLCT_CONFIG["seed"],
                        RLCT_CONFIG["noise_level"],
                        RLCT_CONFIG["llc_weight_decay"],
                        RLCT_CONFIG["sampling_method"],
                        init_batches(subset_n),
                    )
                    _per_draws[str(_draws)] = {
                        "llc": _entry["llc"],
                        "loss_quarters": _entry["loss_quarters"],
                        "seconds": _entry["seconds"],
                    }
                    _lines.append(
                        f"{_key} loc={_loc:g} draws={_draws} "
                        f"llc={_entry['llc']:.4f} "
                        f"q1..q4="
                        f"{[round(_q, 4) for _q in _entry['loss_quarters']]} "
                        f"{_entry['seconds']:.1f}s"
                    )
                _per_loc[str(_loc)] = _per_draws
            _results[_key] = _per_loc
        calibration = {
            "device": device.type,
            "subset": subset_n,
            "epoch": epoch_dir.name,
            "num_chains": calibration_chains,
            "localizations": [
                float(_loc) for _loc in calibration_localizations
            ],
            "num_draws": [int(_d) for _d in calibration_draws],
            "plateau_tolerance": RLCT_CONFIG["plateau_tolerance"],
            "plateau_chosen_draws": plateau_choice(
                _results,
                RLCT_CONFIG["localization"],
                RLCT_CONFIG["plateau_tolerance"],
            ),
            "results": _results,
        }
        print("\n".join(_lines))
        _out = mo.md(
            f"Calibrated {len(_results)} models at "
            f"`{calibration['localizations']}` x "
            f"`{calibration['num_draws']}` draws; plateau-chosen draws: "
            f"`{calibration['plateau_chosen_draws']}`."
        )
    else:
        calibration = None
        _out = mo.md("_Set `RLCT_MODE=calibrate` to run calibration._")
    _out
    return (calibration,)


@app.cell
def _(
    RLCT_CONFIG,
    device,
    epoch_dir,
    estimate_model_cached,
    file_fingerprint,
    init_batches,
    mean_ci,
    mo,
    parse_pruned_models,
    run_requested,
    selected_mode,
    subset_n,
    sweep_chains,
    sweep_draws,
):
    if run_requested and selected_mode == "sweep":
        _models = parse_pruned_models(epoch_dir)
        if not _models:
            rlct = None
            _out = mo.md(f"No pruned checkpoints found in `{epoch_dir}`.")
        else:
            _entries = []
            with mo.status.progress_bar(
                total=len(_models), title="Fig 4B RLCT sweep"
            ) as _bar:
                for _m in _models:
                    _fp = file_fingerprint(_m["path"])
                    _entry = estimate_model_cached(
                        str(_m["path"]),
                        _m["aw"],
                        _m["seed"],
                        _fp,
                        device.type,
                        sweep_chains,
                        sweep_draws,
                        RLCT_CONFIG["localization"],
                        subset_n,
                        RLCT_CONFIG["lr"],
                        RLCT_CONFIG["nbeta"],
                        RLCT_CONFIG["batch_size"],
                        RLCT_CONFIG["num_burnin_steps"],
                        RLCT_CONFIG["num_steps_bw_draws"],
                        RLCT_CONFIG["seed"],
                        RLCT_CONFIG["noise_level"],
                        RLCT_CONFIG["llc_weight_decay"],
                        RLCT_CONFIG["sampling_method"],
                        init_batches(subset_n),
                    )
                    _entries.append(_entry)
                    _bar.update(
                        subtitle=(
                            f"aw={_m['aw']:g} seed={_m['seed']} "
                            f"llc={_entry['llc']:.3f} "
                            f"{_entry['seconds']:.1f}s"
                        )
                    )
            rlct = {}
            for _entry in _entries:
                _aw_key = f"{_entry['aw']:g}"
                _slot = rlct.setdefault(
                    _aw_key,
                    {
                        "aw": _entry["aw"],
                        "llc": [],
                        "seeds": [],
                        "seconds": [],
                    },
                )
                _slot["llc"].append(_entry["llc"])
                _slot["seeds"].append(_entry["seed"])
                _slot["seconds"].append(_entry["seconds"])
            for _slot in rlct.values():
                _slot["mean"], _slot["ci"] = mean_ci(_slot["llc"])
            print(
                "RLCT: "
                + " | ".join(
                    f"aw={_k}: {_v['mean']:.2f}+/-{_v['ci']:.2f}"
                    for _k, _v in sorted(
                        rlct.items(), key=lambda kv: float(kv[0])
                    )
                )
            )
            _out = mo.md(
                f"Estimated {len(_entries)} models from `{epoch_dir}`."
            )
    else:
        rlct = None
        _out = mo.md("_Set `RLCT_MODE=sweep` to run the sweep._")
    _out
    return (rlct,)


@app.cell
def _(
    Path,
    calibration,
    device,
    epoch,
    json,
    mo,
    os,
    rlct,
    rlct_hyperparams,
    run_requested,
    subset_n,
):
    if run_requested and (rlct is not None or calibration is not None):
        _loc = mo.notebook_location()
        _default = (
            str(Path(str(_loc)) / "public" / "data" / "imdb_fig4b.json")
            if _loc is not None
            else None
        )
        _path_str = os.environ.get("RLCT_OUT", "").strip() or _default
        if _path_str is None:
            json_path = None
            _msg = "no output path (notebook location unavailable)"
        else:
            json_path = Path(_path_str)
            _prior_cal = None
            if calibration is None and json_path.exists():
                try:
                    _prior_cal = json.loads(
                        json_path.read_text()
                    ).get("calibration")
                except (OSError, ValueError):
                    _prior_cal = None
            _payload = {
                "rlct": rlct,
                "hyperparams": rlct_hyperparams(
                    device.type, subset_n, epoch
                ),
                "calibration": (
                    calibration if calibration is not None else _prior_cal
                ),
            }
            _bytes = json.dumps(_payload, indent=2).encode("utf-8")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_bytes(_bytes)
            _msg = f"wrote `{json_path}` ({len(_bytes) / 1024:.1f} KiB)"
        _out = mo.md(_msg)
    else:
        json_path = None
        _out = mo.md("_No results to write._")
    _out
    return (json_path,)


@app.cell
def _(alt, mo, pd, rlct):
    PAPER_VALUES = {"0": 13.7, "100": 11.2, "500": 5.2}
    COLORS = {"0": "#000000", "100": "#00e000", "500": "#0000ff"}
    if rlct is None:
        chart = None
        _out = mo.md("_Run the sweep to chart Fig 4B._")
    else:
        _domain = [k for k in ("0", "100", "500") if k in rlct]
        _rows = [
            {
                "aw": _k,
                "mean": _v["mean"],
                "lo": _v["mean"] - _v["ci"],
                "hi": _v["mean"] + _v["ci"],
            }
            for _k, _v in rlct.items()
        ]
        _df = pd.DataFrame(_rows)
        _color = alt.Color(
            "aw:N",
            scale=alt.Scale(
                domain=_domain, range=[COLORS[_k] for _k in _domain]
            ),
            legend=alt.Legend(title="AW"),
        )
        _x = alt.X("aw:N", title="AW", sort=_domain)
        _base = alt.Chart(_df).encode(x=_x, color=_color)
        _bars = _base.mark_bar().encode(
            y=alt.Y("mean:Q", title="RLCT (LLC)")
        )
        _err = _base.mark_errorbar().encode(
            y=alt.Y("lo:Q"), y2="hi:Q"
        )
        _paper_df = pd.DataFrame(
            [
                {"aw": _k, "paper": _v}
                for _k, _v in PAPER_VALUES.items()
                if _k in _domain
            ]
        )
        _paper = (
            alt.Chart(_paper_df)
            .mark_point(
                shape="diamond", filled=True, size=90, color="#ff8c00"
            )
            .encode(
                x=alt.X("aw:N", sort=_domain),
                y=alt.Y("paper:Q", title="RLCT (LLC)"),
            )
        )
        chart = (_bars + _err + _paper).properties(
            width=360, height=300, title="Fig 4B — RLCT by AW"
        )
        _out = chart
    _out
    return (chart,)


@app.cell
def _(json_path, mo, rlct):
    _paper = {"0": 13.7, "100": 11.2, "500": 5.2}
    _where = "—" if json_path is None else f"`{json_path}`"
    if rlct is None:
        _out = mo.md(
            f"_Paper values vs. estimates appear after a sweep. Output: "
            f"{_where}._"
        )
    else:
        _lines = [
            "## Paper comparison",
            "",
            f"Output: {_where}",
            "",
            "| AW | paper | estimate (mean ± CI) | n |",
            "| --- | --- | --- | --- |",
        ]
        for _k in ("0", "100", "500"):
            if _k not in rlct:
                continue
            _v = rlct[_k]
            _lines.append(
                f"| {_k} | {_paper[_k]:g} | "
                f"{_v['mean']:.1f} ± {_v['ci']:.1f} | "
                f"{len(_v['llc'])} |"
            )
        _out = mo.md("\n".join(_lines))
    _out
    return


if __name__ == "__main__":
    app.run()
