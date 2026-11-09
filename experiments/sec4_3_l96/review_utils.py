"""Helpers for JCP review reruns (§4.3): poor-attractor ensemble init, spin-up RMSE."""

import numpy as np
import torch


def load_problem_extras(param_problem, state_init_dir, dim_x):
    """Read init_mode / spinup_steps and load attractor pool if needed."""
    init_mode = str(param_problem['init_mode'].iloc[0])
    init_perturb = float(param_problem['init_perturb'].iloc[0])
    spinup_steps = int(param_problem['spinup_steps'].iloc[0])
    rmse_max = float(param_problem['rmse_max'].iloc[0])

    ensemble_pool = None
    if init_mode == 'attractor':
        pool_path = f'{state_init_dir}/ensemble_pool_d_{dim_x}.npy'
        ensemble_pool = np.load(pool_path)
    return init_mode, init_perturb, spinup_steps, ensemble_pool, rmse_max


def diverged(rmse, rmse_max):
    """True if assimilation should stop (RMSE blow-up or NaN)."""
    return rmse > rmse_max or np.isnan(rmse)


def init_ensemble(state_true, ensemble_size, init_mode, init_perturb,
                  ensemble_pool, device, dtype, seed):
    """
    Initialize ensemble members.

    attractor: independent random states on the L96 attractor (not centered at truth).
    near_truth: legacy paper init (truth + small Gaussian noise).
    """
    if init_mode == 'attractor':
        rng = np.random.default_rng(seed)
        pool_idx = rng.choice(len(ensemble_pool), size=ensemble_size, replace=False)
        x_state = torch.from_numpy(ensemble_pool[pool_idx]).to(device=device, dtype=dtype)
    elif init_mode == 'near_truth':
        x_state = state_true + torch.randn(ensemble_size, state_true.shape[-1],
                                           device=device, dtype=dtype) * init_perturb
    else:
        raise ValueError(f'Unknown init_mode: {init_mode}')
    return x_state


def posterior_rmse_after_spinup(result_rep, spinup_steps, key='rmse'):
    """Mean posterior RMSE over DA steps with time index > spinup_steps.

    Args:
        result_rep: List of recorder dicts from one tuning case (all reps).
        spinup_steps: Ignore posterior steps at or before this time index.
        key: ``'rmse'`` (all dims), ``'rmse_obs'``, or ``'rmse_non_obs'``.
    """
    rmse_vals = []
    for temp_result in result_rep:
        post_index = np.asarray(temp_result['posterior'], dtype=bool)
        t_all = np.asarray(temp_result['time'])
        rmse_all = np.asarray(temp_result[key])
        mask = post_index & (t_all > spinup_steps)
        if mask.any():
            rmse_vals.append(rmse_all[mask].mean())
    if not rmse_vals:
        return np.nan
    return float(np.mean(rmse_vals))


def plot_rmse_results(result_rep_all, save_path, title, colors=None, ylim=(0, 5)):
    """Plot full / observed / unobserved RMSE time series (no shock overlay)."""
    import matplotlib.pyplot as plt

    series = [
        ('rmse', 'RMSE (all dims)'),
        ('rmse_obs', 'RMSE (observed dims)'),
        ('rmse_non_obs', 'RMSE (unobserved dims)'),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 10), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, series):
        for i, results_np in enumerate(result_rep_all):
            if key not in results_np or len(results_np[key]) == 0:
                continue
            post_index = np.asarray(results_np['posterior'], dtype=bool)
            c = colors[i] if colors is not None and i < len(colors) else None
            ax.plot(results_np['time'], results_np[key], '-', color=c, alpha=0.2)
            ax.plot(
                np.asarray(results_np['time'])[post_index],
                np.asarray(results_np[key])[post_index],
                '.-',
                color=c,
                label=f'rep:{i}',
            )
        ax.set_ylabel(ylabel)
        ax.grid(which='both')
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel('step')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
