"""Kalman / ensemble Kalman updates.

Shape convention used everywhere below:
  N      = ensemble size
  dim_x  = state dimension
  dim_obs = observation dimension
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def KF_update(prior_mean, prior_cov, H, obs_value, obs_cov):
    """Analytical Kalman filter update.

    prior_mean: (dim_x,)
    prior_cov:  (dim_x, dim_x)
    H:          (dim_x, dim_obs)   # used as mean @ H
    obs_value:  (dim_obs,)
    obs_cov:    (dim_obs, dim_obs)
    returns:    post_mean (dim_x,), post_cov (dim_x, dim_x)
    """
    # innovation: (dim_obs,)
    y_tilde = obs_value - torch.matmul(prior_mean, H)

    # innovation cov S: (dim_obs, dim_obs)
    S = torch.linalg.multi_dot([H.T, prior_cov, H]) + obs_cov

    # Kalman gain K: (dim_x, dim_obs)
    K = torch.linalg.solve(S, prior_cov @ H, left=False)

    # posterior mean: (dim_x,)
    post_mean = prior_mean + torch.matmul(y_tilde, K.T)

    # posterior cov: (dim_x, dim_x)
    post_cov = prior_cov - torch.linalg.multi_dot([K, H.T, prior_cov])
    return post_mean, post_cov


def EnKF_update(prior_ensemble, obs_fun, obs_value, obs_sigma):
    """Stochastic EnKF update.

    prior_ensemble: (N, dim_x)
    obs_fun:        (N, dim_x) -> (N, dim_obs)
    obs_value:      (dim_obs,)
    obs_sigma:      scalar observation noise std
    returns:        posterior ensemble (N, dim_x)
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    # A: (N, dim_x) anomalies
    A = prior_ensemble - torch.mean(prior_ensemble, dim=0)

    # D: (N, dim_obs) perturbed observations
    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma

    # HX: (N, dim_obs), HA: (N, dim_obs)
    HX = obs_fun(prior_ensemble)
    HA = HX - torch.mean(HX, dim=0)

    # P: (dim_obs, dim_obs) predicted observation covariance
    P = torch.matmul(HA.T, HA) / (N - 1) + torch.eye(dim_obs, device=obs_value.device) * obs_sigma ** 2

    # cross_cov: (dim_obs, dim_x), K: (dim_obs, dim_x)
    cross_cov = torch.matmul(HA.T, A) / (N - 1)
    K = torch.linalg.solve(P, cross_cov)

    # ensemble_post: (N, dim_x)
    ensemble_post = prior_ensemble + torch.matmul(D - HX, K)
    return ensemble_post


def EnKF_update_diag_P(prior_ensemble, obs_fun, obs_value, obs_sigma):
    """EnKF with diagonalized predicted observation covariance.

    prior_ensemble: (N, dim_x)
    obs_fun:        (N, dim_x) -> (N, dim_obs)
    obs_value:      (dim_obs,)
    obs_sigma:      scalar
    returns:        (N, dim_x)
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    A = prior_ensemble - torch.mean(prior_ensemble, dim=0)  # (N, dim_x)
    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma  # (N, dim_obs)

    HX = obs_fun(prior_ensemble)  # (N, dim_obs)
    HA = HX - torch.mean(HX, dim=0)  # (N, dim_obs)

    P = torch.matmul(HA.T, HA) / (N - 1) + torch.eye(dim_obs, device=obs_value.device) * obs_sigma ** 2
    P_diag = torch.diag(P)  # (dim_obs,)

    cross_cov = torch.matmul(HA.T, A) / (N - 1)  # (dim_obs, dim_x)
    K = cross_cov / P_diag[:, None]  # (dim_obs, dim_x)

    ensemble_post = prior_ensemble + torch.matmul(D - HX, K)  # (N, dim_x)
    return ensemble_post


def EnKF_update_direct_sparse_diag(prior_ensemble, obs_dims, obs_value, obs_sigma):
    """Independent (diagonal) EnKF on selected state coordinates.

    prior_ensemble: (N, dim_x)
    obs_dims:       (dim_obs,) integer indices into state
    obs_value:      (dim_obs,)
    obs_sigma:      scalar
    returns:        (N, dim_x)
    """
    dim_obs = obs_value.shape[0]
    N = prior_ensemble.shape[0]

    D = obs_value + torch.randn(N, dim_obs, device=obs_value.device) * obs_sigma  # (N, dim_obs)
    HX = prior_ensemble[:, obs_dims]  # (N, dim_obs)
    prior_var = torch.var(HX, dim=0)  # (dim_obs,)

    K = prior_var / (prior_var + obs_sigma ** 2)  # (dim_obs,)

    ensemble_post = prior_ensemble.clone()
    ensemble_post[:, obs_dims] = prior_ensemble[:, obs_dims] + (D - HX) * K
    return ensemble_post


def KF_with_sample(prior_mean, prior_cov, obs_fun, obs_value, obs_sigma, N):
    """Sample-based KF update for a (possibly nonlinear) observation map.

    prior_mean: (dim_x,)
    prior_cov:  (dim_x, dim_x)
    obs_fun:    (N, dim_x) -> (N, dim_obs)
    obs_value:  (dim_obs,)
    obs_sigma:  scalar
    N:          number of Monte Carlo samples
    returns:    post_mean (dim_x,), post_cov (dim_x, dim_x)
    """
    dim_obs = obs_value.shape[0]

    sample_prior = MultivariateNormal(prior_mean, prior_cov).rsample(torch.Size([N]))  # (N, dim_x)
    hx = obs_fun(sample_prior)  # (N, dim_obs)

    hx_mean = torch.mean(hx, dim=0)  # (dim_obs,)
    hx_centered = hx - hx_mean  # (N, dim_obs)

    # S: (dim_obs, dim_obs)
    S = torch.matmul(hx_centered.T, hx_centered) / (N - 1) + torch.eye(
        dim_obs, device=obs_value.device
    ) * obs_sigma ** 2

    # cross_cov: (dim_obs, dim_x)
    cross_cov = torch.matmul(hx_centered.T, sample_prior - prior_mean) / (N - 1)

    # K: (dim_obs, dim_x)
    K = torch.linalg.solve(S, cross_cov)

    post_mean = prior_mean + torch.matmul(obs_value - hx_mean, K)  # (dim_x,)
    post_cov = prior_cov - torch.matmul(cross_cov.T, K)  # (dim_x, dim_x)
    return post_mean, post_cov
