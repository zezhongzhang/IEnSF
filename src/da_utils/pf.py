"""Particle filter helpers.

Shape convention: N = particle count, dim_x = state dimension.
"""

import numpy as np
import torch


def PF_resample(prior_sample_pf, log_likelihood_fn, boot_size):
    """Bootstrap particle filter resample step.

    prior_sample_pf:   (N, dim_x) prior particles
    log_likelihood_fn: maps (N, dim_x) -> (N,) unnormalized log weights
    boot_size:         number of posterior particles to draw
    returns:           (boot_size, dim_x) resampled posterior particles
    """
    N_pf = prior_sample_pf.shape[0]
    log_l = log_likelihood_fn(prior_sample_pf)  # (N,)
    log_l = log_l - torch.max(log_l)
    l_pf = torch.exp(log_l)  # (N,)
    p = l_pf / torch.sum(l_pf)  # (N,) normalized weights
    idx = np.random.choice(N_pf, boot_size, p=p.cpu().numpy(), replace=True)
    return prior_sample_pf[idx, :]  # (boot_size, dim_x)
