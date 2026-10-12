# Self-Modeling in Neural Systems

Status: in-progress

Reproduction of [Unexpected Benefits of Self-Modeling in Neural Systems](https://arxiv.org/abs/2407.10188) (Premakumar et al., 2024).

- Essay: https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_essay.py
- Train (GPU): https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_train.py

## Files

- `self_modeling_essay.py` — visual essay; reads `public/data/`; saved outputs in `__marimo__/session/`.
- `self_modeling_train.py` — GPU notebook for molab; marked `# smoke: skip`; writes `public/data/mnist_fig2.json`. Its hidden-512 run reproduces the original batched trainer's epoch-50 means to 5 decimals.
- `imdb_train.py` — GPU notebook for molab; marked `# smoke: skip`; IMDB Fig 4A/4C trainer (EmbeddingBag -> linear hidden -> 2 + 128 output rows), writes `public/data/imdb_fig4.json`.
- `cifar_train.py` — GPU notebook for molab (Figure 3, CIFAR-10); marked `# smoke: skip`; writes `public/data/cifar_fig3.json` and checkpoints runs under `checkpoints/` so a restarted session resumes.
- `cifar_rlct.py` — GPU notebook for molab (Figure 3B RLCT, CIFAR-10); marked `# smoke: skip`; reads the pruned weights in `checkpoints/pruned/run_aw<aw>_seed<seed>.pt` and writes `public/data/cifar_fig3b.json`.
- `public/data/` — `mnist_fig2a_local.json` (CPU sweep summary for preprocessing variants `a` and `b`), `mnist_fig2.json` (training-notebook output from the molab GPU: all four hidden sizes, 10 seeds, weight histograms), `mnist_fig2c.json` (RLCT), `imdb_fig4.json` (+ `imdb_fig4_variants.json`), `cifar_fig3.json` and `cifar_fig3b.json`.

## Corrections

These override the paper where it is wrong or unclear:

1. Batch size is 512. The paper's Table 1 lists 64, which is wrong.
2. MNIST pixels are linearly rescaled to `[-1, 1]` (`x / 127.5 - 1`). No mean subtraction: a variant that also subtracts the mean does not match the paper.
3. The target activations are detached before computing the self-modeling loss.

The original script implementation (data, model, loss, sweep, batched trainer, tests and 180 per-run results) lives outside this repo. `public/data/mnist_fig2a_local.json` is its summary.

## Done / next

- Done: Fig 2A reproduced within ~0.001; Fig 2B and 2D reproduced on the molab GPU (240 models, ~43 s of GPU time), including the accuracy collapse of small networks at large AW.
- Done: Fig 2C (RLCT) at localization 100 matches the paper within a few percent; localization 1000 (the supplement's text value) is ~7x lower with the same ordering. Estimated with a batched SGLD sampler validated against devinterp 2.0.1; results in `public/data/mnist_fig2c.json`.
- Done: IMDB Fig 4 on the molab GPU (`public/data/imdb_fig4.json`, variants in `imdb_fig4_variants.json`). Accuracy reproduces with batch 64 (84.9%); the self-modeling effect on weight SD does not (AW 0/100/500 within ~1%, and SD ~10x below the paper's).
- Done: CIFAR-10 Fig 3 on the molab GPU (40 runs x 250 epochs, `public/data/cifar_fig3.json`; RLCT in `cifar_fig3b.json`). Accuracy reproduces (92.1-92.3%). The weight-SD ordering is inverted (baseline narrowest), and RLCT shows no self-modeling effect (~24-25 at every AW vs the paper's 82 -> 66). Probes (no detach, hidden-only target, batch 64, BatchNorm-train sampling) don't recover the paper's direction; see the essay.
- Next: essay sections for Figures 3 and 4; IMDB Fig 4B (RLCT).
