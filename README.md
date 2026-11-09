# IEnSF

Code for **IEnSF: Iterative Ensemble Score Filter for Reducing Error in Posterior Score Estimation in Nonlinear Data Assimilation** ([arXiv:2510.20159](https://arxiv.org/abs/2510.20159)).

## Install

```bash
pip install -e .
```

## Packages

- `iensf` — score-based filtering
  - `diffusion` — EnSF primitives for §4.1–§4.2
  - `update` — localized `iensf_update` for §4.3 L96
- `da_utils` — classical filters and helpers (KF, EnKF, PF, LETKF, recorders, models)

## Experiments

- [`experiments/sec4_1_static`](experiments/sec4_1_static) — §4.1 static Bayesian examples
- [`experiments/sec4_2_harmonic`](experiments/sec4_2_harmonic) — §4.2 harmonic oscillator DA
- [`experiments/sec4_3_l96`](experiments/sec4_3_l96) — §4.3 Lorenz-96 (reviewer protocol, d=1000)
