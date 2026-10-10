# Self-Modeling in Neural Systems

Status: in-progress

Reproduction of [Unexpected Benefits of Self-Modeling in Neural Systems](https://arxiv.org/abs/2407.10188) (Premakumar et al., 2024).

- Essay: https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_essay.py
- Train (GPU): https://molab.marimo.io/github/nikhil-vytla/marimo-notebooks/blob/main/notebooks/self-modeling/self_modeling_train.py

## Files

- `self_modeling_essay.py` — visual essay; reads `public/data/`; saved outputs in `__marimo__/session/`.
- `self_modeling_train.py` — GPU notebook for molab; marked `# smoke: skip`; writes `public/data/mnist_fig2.json`. Its hidden-512 run reproduces the original batched trainer's epoch-50 means to 5 decimals.
- `imdb_train.py` — GPU notebook for molab; marked `# smoke: skip`; IMDB Fig 4A/4C trainer (EmbeddingBag -> linear hidden -> 2 + 128 output rows), writes `public/data/imdb_fig4.json`.
- `public/data/` — `mnist_fig2a_local.json` (CPU sweep summary for preprocessing variants `a` and `b`) and `mnist_fig2.json` (training-notebook output from the molab GPU: all four hidden sizes, 10 seeds, weight histograms). `imdb_fig4.json` is written by `imdb_train.py`.

## Corrections

These override the paper where it is wrong or unclear:

1. Batch size is 512. The paper's Table 1 lists 64, which is wrong.
2. MNIST pixels are linearly rescaled to `[-1, 1]` (`x / 127.5 - 1`). No mean subtraction: a variant that also subtracts the mean does not match the paper.
3. The target activations are detached before computing the self-modeling loss.

The original script implementation (data, model, loss, sweep, batched trainer, tests and 180 per-run results) lives outside this repo. `public/data/mnist_fig2a_local.json` is its summary.

## Done / next

- Done: Fig 2A reproduced within ~0.001; Fig 2B and 2D reproduced on the molab GPU (240 models, ~43 s of GPU time), including the accuracy collapse of small networks at large AW.
- Next: RLCT (Fig 2C, in progress), IMDB Fig 4B (RLCT) on top of `imdb_train.py`'s saved weights, CIFAR-10 (Fig 3, ~4–6 h on molab GPU).
