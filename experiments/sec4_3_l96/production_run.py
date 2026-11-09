"""Production DA loops for §4.3 L96 (reviewer protocol).

Uses package APIs:
  da_utils — rk4, lorenz96_drift, ResultRecorder, letkf_update
  iensf    — diffusion (original EnSF) and update.iensf_update (IEnSF)
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from da_utils import ResultRecorder, lorenz96_drift, rk4
from da_utils.letkf import create_obs_index, letkf_update
from iensf import (
    NoiseSchedule,
    ReverseSampler,
    ScoreRep,
    build_localization_structure,
    create_iensf_sampler,
    get_C,
    iensf_update,
    post_score,
)

from review_utils import (
    diverged,
    init_ensemble,
    load_problem_extras,
    plot_rmse_results,
    posterior_rmse_after_spinup,
)

SOLVER_NAMES = ["SDE_euler", "ODE_euler", "DPM_solver", "EDM_euler", "EDM_heun"]

_RESULT_SKIP_COLS = frozenset(
    {
        "index",
        "rmse_mean_spinup",
        "rmse_mean",
        "rmse_mean_spinup_obs",
        "rmse_mean_spinup_unobs",
    }
)


@dataclass
class ProblemContext:
    dim_x: int
    dt: float
    N_step: int
    run_up2: int
    obs_gap: int
    obs_sigma: float
    spinup_steps: int
    rmse_max: float
    init_mode: str
    init_perturb: float
    ensemble_pool: Optional[np.ndarray]
    state_init_all: np.ndarray
    # linear: every 4; atan: every 2 (reviewer retune)
    obs_every: int
    obs_kind: str  # 'linear' | 'atan'


def load_problem_context(
    param_problem_csv: str,
    *,
    obs_kind: str,
    obs_every: Optional[int] = None,
) -> ProblemContext:
    param_problem = pd.read_csv(param_problem_csv)
    dim_x = int(param_problem["dim_x"].iloc[0])
    state_init_dir = str(param_problem["state_init_dir"].iloc[0])
    init_mode, init_perturb, spinup_steps, ensemble_pool, rmse_max = load_problem_extras(
        param_problem, state_init_dir, dim_x
    )
    state_init_all = np.load(f"{state_init_dir}/state_init_d_{dim_x}_rep_10.npy")
    if obs_every is None:
        obs_every = 4 if obs_kind == "linear" else 2
    return ProblemContext(
        dim_x=dim_x,
        dt=float(param_problem["dt"].iloc[0]),
        N_step=int(param_problem["N_step"].iloc[0]),
        run_up2=int(param_problem["run_up2"].iloc[0]),
        obs_gap=int(param_problem["obs_gap"].iloc[0]),
        obs_sigma=float(param_problem["obs_sigma"].iloc[0]),
        spinup_steps=spinup_steps,
        rmse_max=rmse_max,
        init_mode=init_mode,
        init_perturb=init_perturb,
        ensemble_pool=ensemble_pool,
        state_init_all=state_init_all,
        obs_every=int(obs_every),
        obs_kind=obs_kind,
    )


def make_obs_setup(problem: ProblemContext):
    """Sparse obs on a regular lattice. Shapes: obs_index (dim_x,); y (dim_obs,)."""
    obs_index = create_obs_index(problem.dim_x, problem.obs_every)
    if problem.obs_kind == "linear":
        obs_fun = lambda x: x[:, obs_index] if x.ndim == 2 else x[obs_index]
    elif problem.obs_kind == "atan":
        obs_fun = (
            lambda x: torch.atan(x[:, obs_index])
            if x.ndim == 2
            else torch.atan(x[obs_index])
        )
    else:
        raise ValueError(f"Unknown obs_kind={problem.obs_kind!r}")
    return obs_fun, obs_index


def _analytic_likelihood_score(problem: ProblemContext, obs_index, obs_value, xt):
    """damped externally; returns score shaped like xt: (K, dim_x) or (dim_x,)."""
    score_x = torch.zeros_like(xt)
    sig2 = problem.obs_sigma**2
    if problem.obs_kind == "linear":
        if xt.ndim == 2:
            score_x[:, obs_index] = -(xt[:, obs_index] - obs_value) / sig2
        else:
            score_x[obs_index] = -(xt[obs_index] - obs_value) / sig2
    else:
        if xt.ndim == 2:
            xo = xt[:, obs_index]
            score_x[:, obs_index] = (
                -(torch.atan(xo) - obs_value) / sig2 * (1.0 / (1.0 + xo**2))
            )
        else:
            xo = xt[obs_index]
            score_x[obs_index] = (
                -(torch.atan(xo) - obs_value) / sig2 * (1.0 / (1.0 + xo**2))
            )
    return score_x


def load_filter_params(
    param_csv: str,
    *,
    row_index: Optional[int] = None,
) -> Dict:
    df = pd.read_csv(param_csv)
    if row_index is not None and "index" in df.columns and row_index in df["index"].values:
        row = df.loc[df["index"] == row_index].iloc[0]
    elif row_index is not None:
        row = df.iloc[row_index]
    else:
        row = df.iloc[0]
    return {k: row[k] for k in row.index if k not in _RESULT_SKIP_COLS}


def _make_recorder(obs_index) -> ResultRecorder:
    obs_index_rec = (
        obs_index if isinstance(obs_index, torch.Tensor) else torch.from_numpy(obs_index)
    )
    return ResultRecorder(rmse=True, obs_index=obs_index_rec)


def _resolve_run_up2(problem: ProblemContext, run_up2: Optional[int]) -> int:
    n_rep = int(run_up2 if run_up2 is not None else problem.run_up2)
    return min(n_rep, problem.state_init_all.shape[0])


def _pick_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def run_letkf_production(
    param: Dict,
    problem: ProblemContext,
    obs_fun: Callable,
    obs_index,
    *,
    device: str = "cuda",
    dtype=torch.float32,
    clip_tol: float = 50.0,
    seed_offset: int = 0,
    run_up2: Optional[int] = None,
    print_step_info: bool = False,
) -> List[dict]:
    device = _pick_device(device)
    ensemble_size = int(param["ensemble_size"])
    inflation = float(param["inflation"])
    r_loc = float(param["r_loc"])
    n_rep = _resolve_run_up2(problem, run_up2)

    forward_fn = functools.partial(rk4, fn=lorenz96_drift, t=0, dt=problem.dt)
    recorder = _make_recorder(obs_index)
    seed_vec = np.arange(n_rep) + seed_offset
    results = []

    for rep_id in range(n_rep):
        torch.manual_seed(int(seed_vec[rep_id]))
        state_true = torch.from_numpy(problem.state_init_all[rep_id]).to(
            device=device, dtype=dtype
        )
        x_state = init_ensemble(
            state_true,
            ensemble_size,
            problem.init_mode,
            problem.init_perturb,
            problem.ensemble_pool,
            device,
            dtype,
            int(seed_vec[rep_id]),
        )

        recorder.clear()
        rmse_temp_2 = float("nan")
        for i in range(problem.N_step):
            state_true = forward_fn(state_true)
            x_state = forward_fn(x_state)
            x_state = torch.clip(x_state, min=-clip_tol, max=clip_tol)

            recorder.record(state_true, x_state, t=i, posterior=False)
            rmse_temp_1 = recorder.results["rmse"][-1]
            if diverged(rmse_temp_1, problem.rmse_max):
                print(f"rep {rep_id}: break at prior step {i}, rmse={rmse_temp_1}")
                break

            if i % problem.obs_gap == 0 and i != 0:
                obs_value = obs_fun(state_true)
                obs_value = obs_value + torch.randn_like(obs_value) * problem.obs_sigma
                # x_state: (K, dim_x); obs_value: (dim_obs,)
                x_state = letkf_update(
                    ensemble_prior=x_state,
                    obs_value=obs_value,
                    obs_fun=obs_fun,
                    obs_sigma=problem.obs_sigma,
                    obs_index=obs_index,
                    localization_radius=r_loc,
                    inflation=inflation,
                )
                recorder.record(state_true, x_state, t=i, posterior=True)
                rmse_temp_2 = recorder.results["rmse"][-1]
                if diverged(rmse_temp_2, problem.rmse_max):
                    print(f"rep {rep_id}: break at posterior step {i}, rmse={rmse_temp_2}")
                    break
                if print_step_info:
                    print(f"step {i} DA: before {rmse_temp_1:.4f}, after {rmse_temp_2:.4f}")

        print(
            f"rep {rep_id}: final rmse {rmse_temp_2:.4f} "
            f"(obs {recorder.results['rmse_obs'][-1]:.4f}, "
            f"unobs {recorder.results['rmse_non_obs'][-1]:.4f})"
        )
        results.append(recorder.to_numpy())
    return results


def run_ensf_production(
    param: Dict,
    problem: ProblemContext,
    obs_fun: Callable,
    obs_index,
    *,
    device: str = "cuda",
    dtype=torch.float32,
    clip_tol: float = 50.0,
    seed_offset: int = 0,
    run_up2: Optional[int] = None,
    print_step_info: bool = False,
) -> List[dict]:
    """Original EnSF: one reverse-diffusion step, diagonal GM prior, damped lik score."""
    device = _pick_device(device)
    ensemble_size = int(param["ensemble_size"])
    Nt = int(param["Nt_SDE"])
    gm_var_ratio = float(param.get("gm_var_ratio", 0.0))
    inflation = float(param["inflation"])
    eps_a = float(param["eps_a"])
    eps_b = float(param["eps_b"])
    solver = SOLVER_NAMES[int(param["solver_type"])]
    n_rep = _resolve_run_up2(problem, run_up2)

    forward_fn = functools.partial(rk4, fn=lorenz96_drift, t=0, dt=problem.dt)
    recorder = _make_recorder(obs_index)
    seed_vec = np.arange(n_rep) + seed_offset
    damp_fn = lambda t: (1 - t) * 1.0
    results = []

    for rep_id in range(n_rep):
        ns = NoiseSchedule(ns_type="linear", eps_a=eps_a, eps_b=eps_b)
        dm = ReverseSampler(noise_schedule=ns)
        score_rep = ScoreRep(dm=dm, dim_x=problem.dim_x, obs_model=obs_fun, obs_sigma=problem.obs_sigma)

        torch.manual_seed(int(seed_vec[rep_id]))
        state_true = torch.from_numpy(problem.state_init_all[rep_id]).to(
            device=device, dtype=dtype
        )
        x_state = init_ensemble(
            state_true,
            ensemble_size,
            problem.init_mode,
            problem.init_perturb,
            problem.ensemble_pool,
            device,
            dtype,
            int(seed_vec[rep_id]),
        )

        recorder.clear()
        rmse_temp_2 = float("nan")
        for i in range(problem.N_step):
            state_true = forward_fn(state_true)
            x_state = forward_fn(x_state)
            x_state = torch.clip(x_state, min=-clip_tol, max=clip_tol)

            recorder.record(state_true, x_state, t=i, posterior=False)
            rmse_temp_1 = recorder.results["rmse"][-1]
            if diverged(rmse_temp_1, problem.rmse_max):
                print(f"rep {rep_id}: break at prior step {i}, rmse={rmse_temp_1}")
                break

            if i % problem.obs_gap == 0 and i != 0:
                obs_value = obs_fun(state_true)
                obs_value = obs_value + problem.obs_sigma * torch.randn_like(obs_value)

                # prior_mean_gm: (K, dim_x); prior_var: scalar broadcast
                prior_sample_mean = torch.mean(x_state, dim=0)
                prior_mean_gm = (x_state - prior_sample_mean) * inflation + prior_sample_mean
                score_prior_gm_fn = functools.partial(
                    score_rep.score_gaussian_diffusion_diag,
                    prior_mean=prior_mean_gm,
                    prior_var=gm_var_ratio,
                )
                score_likelihood_fn = lambda xt, t, ov=obs_value: (
                    _analytic_likelihood_score(problem, obs_index, ov, xt) * damp_fn(t)
                )
                post_score_fn = functools.partial(
                    post_score,
                    score_prior=score_prior_gm_fn,
                    score_likelihood=score_likelihood_fn,
                    score_max=1000,
                )

                x_T = torch.randn(ensemble_size, problem.dim_x, device=device, dtype=dtype)
                x_T = (x_T - torch.mean(x_T, dim=0)) / torch.std(x_T, dim=0)
                x_state = dm.sample_gen(
                    x_T=x_T, score_fn=post_score_fn, Nt=Nt, solver_type=solver
                )

                recorder.record(state_true, x_state, t=i, posterior=True)
                rmse_temp_2 = recorder.results["rmse"][-1]
                if diverged(rmse_temp_2, problem.rmse_max):
                    print(f"rep {rep_id}: break at posterior step {i}, rmse={rmse_temp_2}")
                    break
                if print_step_info:
                    print(f"step {i} DA: before {rmse_temp_1:.4f}, after {rmse_temp_2:.4f}")

        print(f"rep {rep_id}: final rmse {rmse_temp_2:.4f}")
        results.append(recorder.to_numpy())
    return results


def run_iensf_production(
    param: Dict,
    problem: ProblemContext,
    obs_fun: Callable,
    obs_index,
    *,
    n_iter: Optional[int] = None,
    device: str = "cuda",
    dtype=torch.float32,
    clip_tol: float = 50.0,
    seed_offset: int = 0,
    run_up2: Optional[int] = None,
    print_step_info: bool = False,
) -> List[dict]:
    """Localized IEnSF / EnSF-GMM via ``iensf_update`` (passes RTPS + diag blend)."""
    device = _pick_device(device)
    ensemble_size = int(param["ensemble_size"])
    Nt = int(param["Nt_SDE"])
    var_split = float(param["var_split"])
    inflation = float(param["inflation"])
    eps_a = float(param["eps_a"])
    eps_b = float(param["eps_b"])
    r_loc = float(param["r_loc"])
    n_iter_val = int(n_iter if n_iter is not None else param["N_iter"])
    ns_type = str(param.get("ns_type", "linear"))
    mode = str(param.get("mode", "legacy"))
    eta1 = float(param.get("eta1", 0.5))
    eta2 = float(param.get("eta2", 0.5))
    rtps_relax = float(param.get("rtps_relax", 0.0))
    likelihood_diag_blend = float(param.get("likelihood_diag_blend", 0.0))
    solver = SOLVER_NAMES[int(param["solver_type"])]

    n_rep = _resolve_run_up2(problem, run_up2)
    forward_fn = functools.partial(rk4, fn=lorenz96_drift, t=0, dt=problem.dt)
    # C: (dim_x, dim_x) Gaspari–Cohn localization
    C = get_C(dim_x=problem.dim_x, radius=r_loc).to(device=device, dtype=dtype)
    loc_struct_cache: Dict[float, object] = {}
    recorder = _make_recorder(obs_index)
    seed_vec = np.arange(n_rep) + seed_offset
    results = []

    for rep_id in range(n_rep):
        dm, score_rep = create_iensf_sampler(
            problem.dim_x,
            obs_fun,
            problem.obs_sigma,
            ns_type=ns_type,
            eps_a=eps_a,
            eps_b=eps_b,
            device=device,
            dtype=dtype,
        )
        torch.manual_seed(int(seed_vec[rep_id]))
        state_true = torch.from_numpy(problem.state_init_all[rep_id]).to(
            device=device, dtype=dtype
        )
        x_state = init_ensemble(
            state_true,
            ensemble_size,
            problem.init_mode,
            problem.init_perturb,
            problem.ensemble_pool,
            device,
            dtype,
            int(seed_vec[rep_id]),
        )

        loc_struct = None
        if mode == "cg":
            if r_loc not in loc_struct_cache:
                loc_struct_cache[r_loc] = build_localization_structure(
                    problem.dim_x, r_loc, device=device, dtype=dtype
                )
            loc_struct = loc_struct_cache[r_loc]

        recorder.clear()
        rmse_temp_2 = float("nan")
        for i in range(problem.N_step):
            state_true = forward_fn(state_true)
            x_state = forward_fn(x_state)
            x_state = torch.clip(x_state, min=-clip_tol, max=clip_tol)

            recorder.record(state_true, x_state, t=i, posterior=False)
            rmse_temp_1 = recorder.results["rmse"][-1]
            if diverged(rmse_temp_1, problem.rmse_max):
                print(f"rep {rep_id}: break at prior step {i}, rmse={rmse_temp_1}")
                break

            if i % problem.obs_gap == 0 and i != 0:
                obs_value = obs_fun(state_true)
                obs_value = obs_value + torch.randn_like(obs_value) * problem.obs_sigma
                # x_state: (K, dim_x) -> analysis (K, dim_x)
                x_state = iensf_update(
                    x_state,
                    obs_value,
                    C,
                    score_rep,
                    dm,
                    n_iter=n_iter_val,
                    var_split=var_split,
                    inflation=inflation,
                    Nt=Nt,
                    solver_type=solver,
                    eta1=eta1,
                    eta2=eta2,
                    mode=mode,
                    loc_struct=loc_struct,
                    rtps_relax=rtps_relax,
                    likelihood_diag_blend=likelihood_diag_blend,
                )
                recorder.record(state_true, x_state, t=i, posterior=True)
                rmse_temp_2 = recorder.results["rmse"][-1]
                if diverged(rmse_temp_2, problem.rmse_max):
                    print(f"rep {rep_id}: break at posterior step {i}, rmse={rmse_temp_2}")
                    break
                if print_step_info:
                    print(f"step {i} DA: before {rmse_temp_1:.4f}, after {rmse_temp_2:.4f}")

        print(
            f"rep {rep_id}: final rmse {rmse_temp_2:.4f} "
            f"(obs {recorder.results['rmse_obs'][-1]:.4f}, "
            f"unobs {recorder.results['rmse_non_obs'][-1]:.4f})"
        )
        results.append(recorder.to_numpy())
    return results


def save_production_results(
    result_rep_all: List[dict],
    save_dir: str,
    *,
    param: Dict,
    method_label: str,
    spinup_steps: int,
    save_plot: bool = True,
    colors: Optional[List[str]] = None,
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "result_production.npy"), result_rep_all, allow_pickle=True)
    pd.Series(param, name="value").to_csv(os.path.join(save_dir, "tuned_params.csv"))

    spinup_rmse = posterior_rmse_after_spinup(result_rep_all, spinup_steps)
    spinup_obs = posterior_rmse_after_spinup(result_rep_all, spinup_steps, "rmse_obs")
    spinup_unobs = posterior_rmse_after_spinup(result_rep_all, spinup_steps, "rmse_non_obs")
    summary = {
        "method": method_label,
        "rmse_mean_spinup": spinup_rmse,
        "rmse_mean_spinup_obs": spinup_obs,
        "rmse_mean_spinup_unobs": spinup_unobs,
        "n_reps": len(result_rep_all),
        **{k: param[k] for k in param},
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary_spinup.csv"), index=False)
    print(
        f"spin-up posterior RMSE (t > {spinup_steps}): "
        f"total {spinup_rmse:.4f}, obs {spinup_obs:.4f}, unobs {spinup_unobs:.4f}"
    )

    if save_plot:
        if colors is None:
            colors = ["#%06X" % np.random.randint(0, 0xFFFFFF) for _ in result_rep_all]
        plot_rmse_results(
            result_rep_all,
            save_path=os.path.join(save_dir, "plot_production.png"),
            title=f"{method_label} production",
            colors=colors,
        )
    return summary
