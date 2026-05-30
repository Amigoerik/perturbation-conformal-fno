# Perturbation-Based Conformal Prediction for Neural Operators

Source code for the paper:

> **Operator learning for the 2D incompressible Navier–Stokes equations: a conformal prediction approach in the data-scarce regime**
> Bowen Gang, Weinan Wang, Hao Deng.

This repository contains the full experimental pipeline that produces the
main comparison (Table 1), the relative-radius reductions (Table 2), the
smoothing/floor ablation (Table 3), the perturbation-noise sensitivity sweep
(Table 4), and the qualitative diagnostic figures (Figures 1–2).

## Overview

We wrap a trained Fourier Neural Operator (FNO) with split conformal prediction
and build the local uncertainty scale from the pointwise disagreement between
two FNOs trained on near-identical datasets — one on the original labels and one
on labels perturbed by small Gaussian noise. The method is studied in the
**data-scarce regime**, where the total label budget is fixed and methods that
require a separate uncertainty network must split that budget across models.

The notebook compares five uncertainty-quantification (UQ) methods, all wrapped
in the same split-conformal backbone so that the only thing that differs is the
nonconformity scale:

1. **Perturbation (ours)** — two FNOs (base + perturbed-label); pointwise
   disagreement is the uncertainty scale. No extra data needed.
2. **MC Dropout** — dropout kept active at test time; predictive std from 20
   stochastic forward passes.
3. **Laplace** — diagonal last-layer Laplace approximation; one scalar
   predictive variance per sample.
4. **UQNO** (Ma et al., arXiv:2402.01960) — a separate quantile FNO trained
   with pinball loss; the training budget must be split between the base FNO
   and this E-network.
5. **Unscaled (sigma == 1)** — constant-width conformal bands (ablation baseline).

## Repository contents

```
perturbation.ipynb     # Main notebook: data loading, models, all experiments and figures
README.md              # This file
requirements.txt       # Python dependencies
.gitignore             # Excludes data, checkpoints, caches from the repo
LICENSE                # MIT license
```

The notebook is self-contained and organized into sequential cells:

| Cell | Content |
|------|---------|
| 0 | Imports, global config, data loading, data-scarce split |
| 1 | `SpectralConv2d` and the FNO architecture |
| 2 | `train_fno` training loop (Adam + StepLR + relative L2 loss) |
| 3 | Perturbed-label construction (`EPSILON = 0.05 * std(train labels)`) |
| 4 | Train base / perturbed / UQNO operators |
| 5 | Laplace last-layer feature extraction |
| 6 | MC Dropout routine |
| 7 | Conformal calibration over `alpha = [0.02, 0.04, 0.06, 0.08, 0.10]` |
| 8-9 | Main comparison with +/- standard errors over 1000 reshuffles (Tables 1-2) |
| 10 | Perturbation-noise sensitivity sweep (Table 4) |
| 11 | Random-seed sensitivity |
| 12 | Qualitative figures (Figures 1-2) |

## Data

The experiments use the standard **2D incompressible Navier–Stokes vorticity**
benchmark: 1200 trajectories at viscosity nu = 1e-5 on a 64 x 64 periodic grid,
with `T_in = 10` input frames and `T_out = 10` output frames per sample.

The dataset is hosted on Kaggle:

> https://www.kaggle.com/datasets/erikdeng/navierstokes-v1e-5-n1200-t20

**To run locally:**

1. Download the dataset from the Kaggle link above (e.g. via the "Download"
   button, or with the Kaggle CLI:
   `kaggle datasets download -d erikdeng/navierstokes-v1e-5-n1200-t20`).
2. Unzip it and place the `.mat` file(s) into a folder named `data/` in the
   repository root.
3. The notebook reads from `DATA_PATH = "./data"` by default (Cell 0).

**To run on Kaggle directly:** add the dataset to your Kaggle notebook; the code
auto-detects the Kaggle input path (`/kaggle/input/...`) and no change is needed.

The loader (`load_ns_data`) handles both old-format `.mat` (via
`scipy.io.loadmat`) and HDF5-style v7.3 `.mat` (via `h5py`), and reads the
vorticity field stored under the key `u`.

> Note: the raw dataset and any trained model checkpoints (e.g. `fno_*.pt`) are
> **not** included in this repository because of their size — they are excluded
> via `.gitignore`. Download the data from the Kaggle link above before running.

## Data-scarce split

Total budget = 1200 samples, identical across methods:

```
Ours:   800 base-FNO train | 200 calibration | 200 test
UQNO:   400 base-FNO train | 400 quantile-FNO (E) train | 200 calibration | 200 test
```

Calibration and test sets are shared across all methods so the comparison is
fair under a matched total label budget.

## Key hyperparameters (as in the paper)

- FNO: width 32, 4 spectral-conv blocks, 12 retained Fourier modes per spatial
  dimension, GELU activations, projection head 32 -> 128 -> `T_out`, dropout 0.1
  (base model).
- Training: Adam, initial lr 1e-3, StepLR (step 200, factor 0.5), batch size 10,
  relative L2 loss, 500 epochs.
- Perturbation noise: `sigma_eps = 0.05 * std(u_train)`.
- Spatial smoothing: 15 x 15 averaging window (`SMOOTH_KS_PERT = 15`).
- Floor: `tau_0 = 0.1 * median` of the training disagreement.
- Reporting: average +/- standard error over `N_REPEATS = 1000` calibration/test
  reshuffles of the merged 400-sample pool.

## How to run

1. Install dependencies (a GPU is strongly recommended; the code falls back to
   CPU automatically but training 500 epochs on CPU is slow):

   ```bash
   pip install -r requirements.txt
   ```

2. Download the data and place it in `./data` (see the **Data** section above).

3. Open and run the notebook top to bottom:

   ```bash
   jupyter notebook perturbation.ipynb
   ```

   Cells must be executed in order, since later cells reuse the operators and
   splits created earlier.

## Reproducibility

Random seeds are fixed in Cell 0 (`torch.manual_seed(42)`, `np.random.seed(42)`).
The seed-sensitivity experiment (Cell 11) retrains the base and perturbed FNOs
over `SEEDS = [42, 123, 256, 789, 1024]` to confirm the conclusions are stable.
Exact numbers may vary slightly across hardware/CUDA/cuDNN versions.

## Citation

If you use this code, please cite the paper. (Add the final BibTeX entry here
once the paper has a venue / arXiv number.)

## License

Released under the MIT License (see `LICENSE`).
