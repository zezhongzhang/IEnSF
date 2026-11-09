# Paper §4.3 — Lorenz-96 (reviewer protocol)

1000-D L96 DA comparison under the **JCP review** setup (not the older d=100 paper notebooks).

Protocol source of truth: `param_problem.csv` + `single_*/param_*.csv` (and the notes below). The earlier review README’s `N_step=7000` claim is **not** what was run; CSVs use **`N_step=1500`**.

## Reviewer points

| Review comment | Setting here |
|----------------|--------------|
| Poor ensemble init | `init_mode=attractor` + `data/ensemble_pool_d_1000.npy` |
| Longer window + spin-up RMSE | `N_step=1500`; metric only for `t > spinup_steps` |
| Retune atan | denser obs (`obs_every=2`), RTPS / var_split / blend in IEnSF CSV |
| EnSF vs IEnSF | original EnSF (damped analytic lik) vs localized `iensf_update` |

## Shared settings

| | linear | atan |
|--|--------|------|
| dim / F / dt | 1000 / 8 / 0.02 | same |
| N_step | 1500 | 1500 |
| obs_gap | 10 | 10 |
| obs | `y=x`, every 4, σ=0.1 | `y=atan(x)`, every 2, σ=0.05 |
| init | attractor pool | same |
| spinup_steps | 1000 | 500 |

## Tuned method parameters (`single_*/param_*.csv`)

| | linear | atan |
|--|--------|------|
| LETKF | N=20, infl 1.15, r_loc 2 | N=20, infl 1.3, r_loc 2 |
| EnSF | N=100, Nt=200, eps 0.1/0.15 | N=100, Nt=200, eps 0.2/0.15 |
| IEnSF | N=100, var_split 1.0, infl 0.9, r_loc 2, RTPS 0.3, blend 0 | var_split 0.25, infl 0.8, r_loc 3, RTPS 0.3, blend 0.3 |

## Run

From repo root:

```bash
pip install -e .
cd experiments/sec4_3_l96
```

Smoke test:

```bash
python run_production.py --case linear --method letkf --run-up2 1 --n-step 40 --device cpu
```

Full production (CUDA recommended):

```bash
python run_production.py --case linear --method letkf --run-up2 6
python run_production.py --case linear --method ensf  --run-up2 6
python run_production.py --case linear --method iensf --run-up2 6
python run_production.py --case atan   --method letkf --run-up2 6
python run_production.py --case atan   --method ensf  --run-up2 6
python run_production.py --case atan   --method iensf --run-up2 6
```

Outputs land in `{case}/single_*/results_*/` (`result_production.npy`, `summary_spinup.csv`, optional plot).

Reference spin-up summaries from the original review rerun are under `{case}/plot/compare_rmse_spinup_summary.csv`.

## Code map

| Piece | Location |
|-------|----------|
| `iensf_update` + localization | `src/iensf/update.py` |
| LETKF | `src/da_utils/letkf.py` |
| Original EnSF sampler | `src/iensf/diffusion.py` |
| DA loops | `production_run.py` |
| Attractor init / spin-up RMSE | `review_utils.py` |
