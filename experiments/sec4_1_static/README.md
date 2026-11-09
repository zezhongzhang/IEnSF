# Paper §4.1 — Static Bayesian inference

Recreates the low-dimensional posterior sampling examples (Figures 1–4).

| Folder | Paper scenario | Notebook |
|---|---|---|
| `linear_gaussian/` | Linear observation, analytic Gaussian posterior | `run.ipynb` |
| `non_gaussian/` | Nonlinear observation, non-Gaussian posterior | `run.ipynb` |
| `bias_correction/` | Prior far from posterior; iterative bias correction | `run.ipynb` |

Install from repo root: `pip install -e .`

Each notebook imports score tools from `iensf` and KF/EnKF from `da_utils`.
