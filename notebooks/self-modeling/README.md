# Self-Modeling in Neural Systems

Status: in-progress

Reproduction of [Unexpected Benefits of Self-Modeling in Neural Systems](https://arxiv.org/abs/2407.10188) (Premakumar et al., 2024).

- Essay: https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_essay.py
- Train (GPU): https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_train.py

## Files

- `self_modeling_essay.py` — visual essay; reads `public/data/`; saved outputs in `__marimo__/session/`.
- `self_modeling_train.py` — GPU notebook for molab; marked `# smoke: skip`; writes `public/data/mnist_fig2.json`. Its hidden-512 run reproduces the original batched trainer's epoch-50 means to 5 decimals. `SM_LOG_IDENTITY=1` records the self-model loss, `||W_s - I||_F/||I||_F`, `||b_s||` and the self-model/CE gradient share per hidden size and AW. Script-mode overrides: `SM_HIDDEN`, `SM_AWS`, `SM_SEEDS`, `SM_EPOCHS`, `SM_OUT`.
- `imdb_train.py` — GPU notebook for molab; marked `# smoke: skip`; IMDB Fig 4A/4C trainer (EmbeddingBag -> linear hidden -> 2 + 128 output rows), writes `public/data/imdb_fig4.json`. Gradient clipping is applied per model; an earlier build used one global norm across the stacked models, which coupled otherwise-independent runs (its results are the pre-fix `imdb_fig4_variants.json`). `SM_SAVE_EPOCHS=250,500` (plus optional `SM_SAVE_DIR`, default `checkpoints/imdb`) writes the pruned classifier at those epochs for the RLCT notebook. `SM_LOG_IDENTITY=1` adds the same per-AW identity diagnostics (self-model loss, `||W_s - I||_F/||I||_F`, `||b_s||`, self-model gradient share), `SM_FREEZE_SELFHEAD=1` freezes the self-model output rows at their random init, and `SM_SELFHEAD_LR_SCALE=<s>` scales the self-model output-row gradient after per-model clipping (default `1.0`).
- `imdb_rlct.py` — GPU notebook for molab (Figure 4B RLCT, IMDB); marked `# smoke: skip`; reads `checkpoints/imdb/epoch250/aw<aw>_seed<seed>.pt` and writes `public/data/imdb_fig4b.json`.
- `cifar_train.py` — GPU notebook for molab (Figure 3, CIFAR-10); marked `# smoke: skip`; writes `public/data/cifar_fig3.json` and checkpoints runs under `checkpoints/` so a restarted session resumes.
- `cifar_rlct.py` — GPU notebook for molab (Figure 3B RLCT, CIFAR-10); marked `# smoke: skip`; reads the pruned weights in `checkpoints/pruned/run_aw<aw>_seed<seed>.pt` and writes `public/data/cifar_fig3b.json`.
- `public/data/` — `mnist_fig2a_local.json` (CPU sweep summary for preprocessing variants `a` and `b`), `mnist_fig2.json` (training-notebook output from the molab GPU: all four hidden sizes, 10 seeds, weight histograms), `mnist_fig2c.json` (RLCT), `imdb_fig4.json` (corrected, per-model clipping), `imdb_fig4_variants.json` (pre-fix variants), `imdb_fig4b.json`, `identity_diagnostics.json` (per-epoch self-model loss, identity distance, bias norm and self-model gradient share for IMDB normal/frozen and MNIST hidden-512), `slowhead_control.json` (slowed self-model-head control on IMDB: per-epoch SD, accuracy and gradient share at self-model LR scales 1, 0.01, 0.001), `cifar_fig3.json` and `cifar_fig3b.json`.

## Corrections

These override the paper where it is wrong or unclear:

1. Batch size is 512. The paper's Table 1 lists 64, which is wrong.
2. MNIST pixels are linearly rescaled to `[-1, 1]` (`x / 127.5 - 1`). No mean subtraction: a variant that also subtracts the mean does not match the paper.
3. The target activations are detached before computing the self-modeling loss.

The original script implementation (data, model, loss, sweep, batched trainer, tests and 180 per-run results) lives outside this repo. `public/data/mnist_fig2a_local.json` is its summary.

## Done / next

- Done: Fig 2A reproduced within ~0.001; Fig 2B and 2D reproduced on the molab GPU (240 models, ~43 s of GPU time), including the accuracy collapse of small networks at large AW.
- Done: Fig 2C (RLCT) at localization 100 matches the paper within a few percent; localization 1000 (the supplement's text value) is ~7x lower with the same ordering. Estimated with a batched SGLD sampler validated against devinterp 2.0.1; results in `public/data/mnist_fig2c.json`.
- Done: IMDB Fig 4 on the molab GPU (`public/data/imdb_fig4.json`; pre-fix variants in `imdb_fig4_variants.json`). Accuracy reproduces with batch 64 (85.2%); the self-modeling effect on weight SD does not (AW 0/100/500 within ~1%, and SD 0.232 against the paper's 2.05, ~9x below).
- Done: CIFAR-10 Fig 3 on the molab GPU (40 runs x 250 epochs, `public/data/cifar_fig3.json`; RLCT in `cifar_fig3b.json`). Accuracy reproduces (92.1-92.3%). The weight-SD ordering is inverted (baseline narrowest), and RLCT shows no self-modeling effect (~24-25 at every AW vs the paper's 82 -> 66). Probes (no detach, hidden-only target, batch 64, BatchNorm-train sampling) don't recover the paper's direction; see the essay.
- Done: IMDB Fig 4B (RLCT) on the molab GPU (`public/data/imdb_fig4b.json`). RLCT is flat across AW (2.10/2.13/2.13) and ~6.5x below the paper's 13.7 -> 5.2 baseline.
- Found and fixed a coupled-gradient-clipping bug: one global norm across the stacked models over-clipped each run and coupled the variants. A frozen-head control caught it (AW 0 lost accuracy despite a zero-weight self-model term). The conclusions held after the per-model fix (see the essay's reproduction note). Identity diagnostics are in `public/data/identity_diagnostics.json`; the self-model gradient share on IMDB falls to ~2% by epoch 20 while MNIST stays at ~14-18% at epoch 50.
- Done: slowed self-model-head control on IMDB (`public/data/slowhead_control.json`). Scaling the self-model rows' gradient to 0.01 keeps the self-model gradient share at ~18-43% (MNIST-like) and AW 100/500 narrow the epoch-100 weight SD by ~20-25% with CIs disjoint from baseline, where the normal run is flat; scale 0.001 collapses accuracy to near chance. This supports the identity mechanism causally, with caveats in the essay.
- Done: essay covers every panel, with a scoreboard of what reproduced.
- Next: the `.detach()` exercise.
