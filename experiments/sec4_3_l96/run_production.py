#!/usr/bin/env python3
"""Run one §4.3 L96 method with the reviewer-tuned CSV parameters.

Examples (from this directory):

  python run_production.py --case linear --method letkf --run-up2 1 --n-step 50
  python run_production.py --case atan --method iensf --device cuda
  python run_production.py --case linear --method ensf --run-up2 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from production_run import (
    load_filter_params,
    load_problem_context,
    make_obs_setup,
    run_ensf_production,
    run_iensf_production,
    run_letkf_production,
    save_production_results,
)

ROOT = Path(__file__).resolve().parent
METHOD_DIRS = {
    "letkf": "single_LETKF",
    "ensf": "single_EnSF",
    "iensf": "single_IEnSF",
}
METHOD_PARAM = {
    "letkf": "param_LETKF.csv",
    "ensf": "param_EnSF.csv",
    "iensf": "param_IEnSF.csv",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["linear", "atan"], required=True)
    p.add_argument("--method", choices=["letkf", "ensf", "iensf"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--run-up2", type=int, default=None, help="Override number of reps")
    p.add_argument("--n-step", type=int, default=None, help="Override N_step (smoke tests)")
    p.add_argument("--n-iter", type=int, default=None, help="Override IEnSF N_iter")
    p.add_argument("--print-step-info", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    case_dir = ROOT / args.case
    os.chdir(case_dir)  # so relative state_init_dir=../data resolves

    problem = load_problem_context(
        str(case_dir / "param_problem.csv"),
        obs_kind=args.case,
    )
    if args.n_step is not None:
        problem.N_step = int(args.n_step)

    param_csv = case_dir / METHOD_DIRS[args.method] / METHOD_PARAM[args.method]
    param = load_filter_params(str(param_csv))
    obs_fun, obs_index = make_obs_setup(problem)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print(
        f"case={args.case} method={args.method} dim={problem.dim_x} "
        f"N_step={problem.N_step} obs_every={problem.obs_every} "
        f"spinup={problem.spinup_steps} device={args.device}"
    )
    print(pd.Series(param))

    common = dict(
        param=param,
        problem=problem,
        obs_fun=obs_fun,
        obs_index=obs_index,
        device=args.device,
        dtype=dtype,
        run_up2=args.run_up2,
        print_step_info=args.print_step_info,
    )

    if args.method == "letkf":
        results = run_letkf_production(**common)
        label = "LETKF"
    elif args.method == "ensf":
        results = run_ensf_production(**common)
        label = "EnSF"
    else:
        results = run_iensf_production(**common, n_iter=args.n_iter)
        n_iter = args.n_iter if args.n_iter is not None else int(param["N_iter"])
        label = f"IEnSF_iter{n_iter}"

    save_dir = case_dir / METHOD_DIRS[args.method] / f"results_{label}"
    summary = save_production_results(
        results,
        str(save_dir),
        param=param,
        method_label=label,
        spinup_steps=problem.spinup_steps,
        save_plot=not args.no_plot,
    )
    print("saved:", save_dir)
    print(summary)


if __name__ == "__main__":
    main()
