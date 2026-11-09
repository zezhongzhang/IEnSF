# Paper §4.2 — Harmonic oscillator DA

Recreates the 2D linear-Gaussian filtering comparison (Figures 5–7).

Run order (from this folder, after `pip install -e .` at repo root):

1. `run_kf.ipynb` — analytic Kalman filter (reference posterior)
2. `run_pf.ipynb` — particle filter
3. `run_enkf.ipynb` — EnKF
4. `run_ensf_old.ipynb` — original EnSF
5. `run_iensf_multiple.ipynb` — IEnSF (multiple repetitions)
6. `plot.ipynb` — KL / mean / spread / iteration plots

Optional: `run_iensf_single.ipynb`, `run_iensf_with_true_post.ipynb`

Outputs go to `result/`; figures are written next to `plot.ipynb`.
