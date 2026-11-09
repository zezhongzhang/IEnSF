# ula_refine.py  (new, standalone — does not modify iensf.py)

import math
import torch


def posterior_score_GM_direct(
    x: torch.Tensor,
    mu_components: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    score_rep,
    temper_tau: float = 1.0,
) -> torch.Tensor:
    """Exact (no-diffusion) posterior score of a GM(mu_k, Sigma) prior reweighted
    by the likelihood p(y|x), evaluated directly at t=0. temper_tau >= 1 softens
    the mixture weights (1.0 = untempered)."""
    x_rot = x @ eigvecs
    mu_rot = mu_components @ eigvecs
    diff = x_rot[:, None, :] - mu_rot[None, :, :]
    inv_eigvals = 1.0 / eigvals.clamp_min(1e-12)
    score_rot = -diff * inv_eigvals[None, None, :]
    log_prior_k = -0.5 * torch.sum(diff ** 2 * inv_eigvals[None, None, :], dim=2)

    log_lik = score_rep.log_likelihood(x)

    log_post_k = (log_prior_k + log_lik[:, None]) / temper_tau
    log_post_k = log_post_k - torch.max(log_post_k, dim=1, keepdim=True)[0]
    w = torch.exp(log_post_k)
    w = w / torch.sum(w, dim=1, keepdim=True)

    prior_score_rot = torch.sum(score_rot * w[:, :, None], dim=1)
    prior_score = prior_score_rot @ eigvecs.T

    lik_score = score_rep.score_likelihood_auto(x.clone())
    return prior_score + lik_score


def ula_refine(
    x_init: torch.Tensor,
    mu_components: torch.Tensor,
    Sigma: torch.Tensor,
    score_rep,
    n_steps: int = 20,
    step_size: float = 1e-4,
    temper_tau: float = 2.0,
    eigval_floor: float = 0.0,
) -> torch.Tensor:
    """Short ULA refinement applied to samples already close to the posterior
    (e.g. IEnSF's output). Not annealed -- keep n_steps small."""
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = eigvals.clamp_min(eigval_floor if eigval_floor > 0.0 else 1e-10)

    x = x_init.clone()
    for _ in range(n_steps):
        score = posterior_score_GM_direct(
            x, mu_components, eigvals, eigvecs, score_rep, temper_tau=temper_tau,
        )
        noise = torch.randn_like(x)
        x = x + step_size * score + math.sqrt(2.0 * step_size) * noise
    return x