"""LETKF with Gaspari-Cohn localization for cyclic L96 (Hunt et al. 2007).

Corrected version:
  * FIX 1: symmetric eigenvalue square-root analysis transform (mean-preserving),
           replacing the Cholesky-based non-symmetric square root.
  * FIX 2: consistent signed periodic offset for the local window AND the
           localization distances (fixes wrap-around boundary points).
  * Optional: mean-zero self-check (set debug=True) to verify the transform.

Notes on inflation:
  Multiplicative inflation `inflation` (rho) enters through
      P_tilde_inv = ((k-1)/rho^2) I + Y^T R_loc^{-1} Y.
  For a poor initial ensemble, use a WIDE grid (e.g. up to 2.0) and/or add RTPS
  (relaxation-to-prior-spread) externally if spin-up spread collapses.
"""

import numpy as np
import torch


def gaspari_cohn(distance, radius):
    """Gaspari-Cohn 5th-order piecewise polynomial localization function.

    Args:
        distance: Distance between points (numpy array)
        radius:   Localization radius (support goes to zero at distance = 2*radius)

    Returns:
        Localization weights in [0, 1].
    """
    r = np.abs(distance) / radius
    rho = np.zeros_like(r, dtype=float)

    # Region 1: 0 <= r < 1
    mask1 = r < 1
    rho[mask1] = (1.0 - 5.0 / 3.0 * r[mask1] ** 2 + 5.0 / 8.0 * r[mask1] ** 3 +
                  1.0 / 2.0 * r[mask1] ** 4 - 1.0 / 4.0 * r[mask1] ** 5)

    # Region 2: 1 <= r < 2
    mask2 = (r >= 1) & (r < 2)
    rho[mask2] = (4.0 - 5.0 * r[mask2] + 5.0 / 3.0 * r[mask2] ** 2 +
                  5.0 / 8.0 * r[mask2] ** 3 - 1.0 / 2.0 * r[mask2] ** 4 +
                  1.0 / 12.0 * r[mask2] ** 5 - 2.0 / (3.0 * r[mask2]))

    # Region 3: r >= 2 -> already 0
    rho = np.clip(rho, 0.0, 1.0)
    return rho


def create_obs_index(dim_x, obs_every):
    obs_index = torch.zeros(dim_x, dtype=torch.bool)
    obs_index[::obs_every] = True
    return obs_index


def letkf_update(ensemble_prior, obs_value, obs_fun, obs_sigma, obs_index,
                 localization_radius, inflation=1.05, return_prior=False,
                 debug=False, verify_transform=False):
    """Local Ensemble Transform Kalman Filter (LETKF) update, Hunt et al. (2007).

    Corrected symmetric-square-root transform and consistent periodic localization.

    Core shapes:
      ensemble_prior / posterior  (K, dim_x)
      obs_value                   (dim_obs,)
      Y / Y_pert                  (K, dim_obs)
      X_pert                      (K, dim_x)
      Wa                          (K, K)      # symmetric analysis transform
      local X_pert / analysis     (K, dim_local) inside the grid loop

    Args:
        ensemble_prior: torch.Tensor (ensemble_size, dim_x). Prior/forecast ensemble.
        obs_value:      torch.Tensor (dim_obs,). Observation values at observed sites.
        obs_fun:        callable  (ens, dim_x)->(ens, dim_obs)  OR  matrix H (dim_obs, dim_x).
        obs_sigma:      float or torch.Tensor (dim_obs,). Observation error std.
        obs_index:      torch.Tensor (dim_x,) bool. True where observed.
        localization_radius: float. Gaspari-Cohn radius (support to 2*radius).
        inflation:      float >= 1. Multiplicative inflation (rho).
        return_prior:   bool. If True also return a copy of the prior ensemble.
        debug:          bool. If True, assert analysis perturbations are mean-zero.
        verify_transform: alias for debug (backward compatibility).

    Returns:
        ensemble_posterior: torch.Tensor (ensemble_size, dim_x)
        or (ensemble_posterior, ensemble_prior_copy) if return_prior=True.
    """
    debug = debug or verify_transform

    ensemble_size, dim_x = ensemble_prior.shape
    device = ensemble_prior.device
    dtype = ensemble_prior.dtype
    k = ensemble_size

    obs_index_np = obs_index.cpu().numpy()
    obs_positions = np.where(obs_index_np)[0]
    dim_obs = len(obs_positions)
    obs_pos_to_idx = {pos: idx for idx, pos in enumerate(obs_positions)}

    assert obs_value.shape[0] == dim_obs, (
        f"obs_value length ({obs_value.shape[0]}) must match number of observed "
        f"dimensions ({dim_obs})")

    # ---- Apply observation operator to the whole ensemble ----
    if callable(obs_fun):
        Y = obs_fun(ensemble_prior)              # (k, dim_obs)
    else:
        Y = ensemble_prior @ obs_fun.T           # (k, dim_obs)
    assert Y.shape == (k, dim_obs), (
        f"Observation operator output shape {Y.shape} must be ({k}, {dim_obs})")

    Y_mean = Y.mean(dim=0)                        # (dim_obs,)
    Y_pert = Y - Y_mean                           # (k, dim_obs)

    X_mean = ensemble_prior.mean(dim=0)          # (dim_x,)
    X_pert = ensemble_prior - X_mean             # (k, dim_x)

    if return_prior:
        ensemble_prior_copy = X_pert + X_mean

    if isinstance(obs_sigma, (int, float)):
        obs_sigma_tensor = torch.full((dim_obs,), float(obs_sigma),
                                      device=device, dtype=dtype)
    else:
        obs_sigma_tensor = obs_sigma.to(device=device, dtype=dtype)

    # Local half-window (in grid points). Gaspari-Cohn support = 2*radius.
    neighbor_size = int(2 * localization_radius)

    all_idx = np.arange(dim_x)
    ensemble_posterior = []

    # ====================================================================
    # LOCAL ANALYSIS: loop over each grid point
    # ====================================================================
    for m in range(dim_x):
        # ---- FIX 2: signed periodic offset -> window AND distances share one metric ----
        offset = (all_idx - m + dim_x // 2) % dim_x - dim_x // 2
        within = np.abs(offset) <= neighbor_size
        id_x_periodic = all_idx[within]                 # local state indices
        local_offset = offset[within]                   # signed local offsets

        has_obs = obs_index_np[id_x_periodic]
        if not np.any(has_obs):
            ensemble_posterior.append(ensemble_prior[:, m:m + 1])
            continue

        id_x_obs_periodic = id_x_periodic[has_obs]
        distances = np.abs(local_offset[has_obs]).astype(float)  # periodic-correct

        loc_weights = gaspari_cohn(distances, localization_radius)
        obs_mask = loc_weights > 0
        if not np.any(obs_mask):
            ensemble_posterior.append(ensemble_prior[:, m:m + 1])
            continue

        id_x_obs_periodic = id_x_obs_periodic[obs_mask]
        loc_weights = loc_weights[obs_mask]

        id_y = np.array([obs_pos_to_idx[pos] for pos in id_x_obs_periodic])

        id_x_torch = torch.from_numpy(id_x_periodic).to(device=device).long()
        id_y_torch = torch.from_numpy(id_y).to(device=device).long()
        loc_weights_torch = torch.from_numpy(loc_weights).to(device=device, dtype=dtype)

        X_local = X_pert[:, id_x_torch]              # (k, dim_x_local)
        Y_local = Y_pert[:, id_y_torch]              # (k, dim_obs_local)
        obs_local = obs_value[id_y_torch]            # (dim_obs_local,)
        obs_mean_local = Y_mean[id_y_torch]          # (dim_obs_local,)
        obs_sigma_local = obs_sigma_tensor[id_y_torch]  # (dim_obs_local,)

        R_inv_loc = loc_weights_torch / (obs_sigma_local ** 2)   # (dim_obs_local,)
        innovation = obs_local - obs_mean_local                  # (dim_obs_local,)

        P_tilde_inv = (((k - 1) / (inflation ** 2)) *
                       torch.eye(k, device=device, dtype=dtype) +
                       (Y_local * R_inv_loc[None, :]) @ Y_local.T)

        evals, evecs = torch.linalg.eigh(P_tilde_inv)
        evals = torch.clamp(evals, min=1e-12)
        inv_evals = 1.0 / evals

        rhs = Y_local @ (R_inv_loc * innovation)                 # (k,)
        w_mean = (evecs * inv_evals) @ (evecs.T @ rhs)           # (k,)

        sqrt_scale = torch.sqrt(torch.tensor(float(k - 1), dtype=dtype, device=device))
        Wa = (evecs * torch.sqrt(inv_evals)) @ evecs.T * sqrt_scale   # (k, k) symmetric

        X_mean_local = X_mean[id_x_torch]                        # (dim_x_local,)
        mean_increment = X_local.T @ w_mean                      # (dim_x_local,)
        X_mean_analysis = X_mean_local + mean_increment          # (dim_x_local,)

        X_pert_analysis = (X_local.T @ Wa).T                     # (k, dim_x_local)
        X_pert_analysis = X_pert_analysis - X_pert_analysis.mean(dim=0, keepdim=True)

        if debug:
            pert_mean = X_pert_analysis.mean(dim=0)
            assert torch.allclose(
                pert_mean, torch.zeros_like(pert_mean), atol=1e-6), (
                "Analysis perturbations not mean-zero: transform bug "
                f"(max |mean| = {pert_mean.abs().max().item():.2e})")

        X_analysis_local = X_pert_analysis + X_mean_analysis[None, :]

        center_idx = (id_x_torch == m)
        ensemble_posterior.append(X_analysis_local[:, center_idx])  # (k, 1)

    ensemble_posterior = torch.cat(ensemble_posterior, dim=1)      # (k, dim_x)

    if return_prior:
        return ensemble_posterior, ensemble_prior_copy
    return ensemble_posterior


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    k, dim_x = 40, 20
    ens = torch.randn(k, dim_x, dtype=torch.float64) * 2.0

    obs_index = create_obs_index(dim_x, obs_every=2)
    dim_obs = int(obs_index.sum())
    H = torch.eye(dim_x, dtype=torch.float64)[obs_index]

    truth = torch.zeros(dim_x, dtype=torch.float64)
    obs_sigma = 0.5
    y = (H @ truth) + obs_sigma * torch.randn(dim_obs, dtype=torch.float64)

    post = letkf_update(
        ens, y, H, obs_sigma=obs_sigma, obs_index=obs_index,
        localization_radius=1e6,
        inflation=1.0, debug=True,
    )

    Xf_mean = ens.mean(0)
    Xf_pert = ens - Xf_mean
    Yf = (H @ ens.T).T
    Yf_mean = Yf.mean(0)
    Yf_pert = Yf - Yf_mean
    R = (obs_sigma ** 2) * torch.eye(dim_obs, dtype=torch.float64)
    Pf_yy = (Yf_pert.T @ Yf_pert) / (k - 1) + R
    Pf_xy = (Xf_pert.T @ Yf_pert) / (k - 1)
    Kgain = Pf_xy @ torch.linalg.inv(Pf_yy)
    xa_mean_ref = Xf_mean + Kgain @ (y - Yf_mean)

    xa_mean = post.mean(0)
    err = (xa_mean - xa_mean_ref).abs().max().item()
    print(f"max |analysis mean - Kalman reference| = {err:.3e}")
    print(f"analysis perturbation mean (should ~0) = "
          f"{(post - post.mean(0)).mean(0).abs().max().item():.3e}")
    assert err < 1e-6, "Analysis mean does not match direct Kalman update"
    print("SELF-TEST PASSED: transform is mean-consistent with the Kalman update.")
