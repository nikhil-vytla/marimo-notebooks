# Reference implementation (known-good)

Plain-script reproduction of Figure 2A that the notebooks are checked against. Treat it as a regression oracle: if a notebook disagrees with these results, the notebook is wrong until proven otherwise. Don't refactor it.

How it was verified:

- 11 tests pass (`python tests/test_*.py`). `test_loss.py` fails if the `.detach()` on the self-model target is removed. `test_batched.py` checks the batched trainer against separate single-model runs (atol 1e-6).
- `results/{a,b}/` hold 10 seeds × 6 AWs × 50 epochs per preprocessing variant from `sweep.py` (CPU). Variant `a` (rescale to [-1, 1]) matches the paper's Fig 2A within ~0.001 at epoch 50. Variant `b` (rescale + subtract the mean) does not.
- `results/batched/a/` is the same sweep from `train_batched.py` on Apple MPS. Each AW's epoch-50 mean agrees with the CPU sweep within the 95% CI.

Settings: `CORRECTIONS.md` (batch size 512, [-1, 1] rescale, detached target) override the paper's Table 1.

Note: `results/a/aw0_h512_s0.json` came from a standalone `train.py` run with the same settings, so its `config.out` path differs from the others.

Run (Python 3.12, from this folder): `pip install -r requirements.txt`, then `python train.py --aw 10 --seed 0`. Pillow isn't needed: `data.py` decodes the MNIST PNGs itself and caches them in `.cache/`. Plotting needs `matplotlib`.
