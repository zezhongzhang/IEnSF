"""IEnSF / EnSF-GMM assimilation update for cyclic L96 (shared by linear and atan branches).

EnSF is ``n_iter=1``; IEnSF uses ``n_iter>1`` with relaxed inner iterations on the
posterior Gaussian (mu_star, cov_star).

Self-contained: includes the diffusion/score primitives used by the update step
(no dependency on ScoreFilter).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    'NoiseSchedule',
    'EDMNoiseSchedule',
    'ReverseSampler',
    'ScoreRep',
    'post_score',
    'make_edm_native_score_fn',
    'gaspari_cohn',
    'get_C',
    'LocalizationStructure',
    'build_localization_structure',
    'create_iensf_sampler',
    'iensf_update',
]


# ---------------------------------------------------------------------------
# Diffusion primitives (subset of ScoreFilter.diffusion used by IEnSF)
# ---------------------------------------------------------------------------


class NoiseSchedule:
    """Forward noise schedule for the diffusion model."""

    def __init__(self, ns_type: str = 'linear', eps_a: float = 1e-5, eps_b: float = 1e-5):
        self.ns_type = ns_type
        self.eps_a = eps_a
        self.eps_b = eps_b

    def get_values(self, t):
        """Return alpha, beta_sq, d_log_alpha_dt, d_beta_sq_dt at pseudo-time t."""
        if self.ns_type == 'linear':
            eps_a = self.eps_a
            eps_b = self.eps_b
            alpha_t = 1 - (1 - eps_a) * t
            beta_sq_t = eps_b + (1 - eps_b) * t
            d_alpha_dt = -(1 - eps_a)
            d_log_alpha_dt = d_alpha_dt / alpha_t
            d_beta_sq_dt = 1 - eps_b
            return alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt
        raise ValueError('Unknown noise schedule type!')


class EDMNoiseSchedule:
    """EDM power-law sigma schedule mapped to VP (alpha, beta) coefficients.

    EDM uses x_noisy = x + sigma * eps.  For compatibility with the IEnSF score
    formulas (which are written in VP form), we map

        alpha = 1 / sqrt(1 + sigma^2),   beta^2 = sigma^2 / (1 + sigma^2),

    so that z_t = alpha * z_0 + beta * eps and sigma = beta / alpha.

    IMPORTANT: samples fed into the EDM steppers live in native x_noisy space, not
    the z_t space these (alpha, beta) values parameterize. See
    ``make_edm_native_score_fn`` for the required conversion.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

    @classmethod
    def from_linear_eps(cls, eps_a: float = 1e-5, eps_b: float = 1e-5, rho: float = 7.0):
        """Match the linear schedule endpoints via sigma = beta / alpha."""
        sigma_max = math.sqrt(eps_b + (1.0 - eps_b)) / max(eps_a, 1e-12)
        sigma_min = math.sqrt(eps_b) / math.sqrt(eps_b + (1.0 - eps_b))
        return cls(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)

    def _as_tensor(self, sigma):
        if isinstance(sigma, torch.Tensor):
            return sigma
        return torch.as_tensor(sigma, dtype=torch.float64)

    def sigma_at_u(self, u):
        """Noise level at normalized progress u in [0, 1] (0 = noisiest)."""
        u = self._as_tensor(u)
        inv_rho = 1.0 / self.rho
        smax = self.sigma_max ** inv_rho
        smin = self.sigma_min ** inv_rho
        return (smax + u * (smin - smax)) ** self.rho

    def make_sigma_schedule(self, Nt: int, device=None, dtype=torch.float64):
        """Nt+1 sigmas from sigma_max (u=0) down to sigma_min (u=1)."""
        u = torch.linspace(0.0, 1.0, Nt + 1, device=device, dtype=dtype)
        return self.sigma_at_u(u)

    def get_values(self, sigma):
        """VP-equivalent (alpha, beta_sq, d_log_alpha, d_beta_sq) at noise level sigma."""
        sigma = self._as_tensor(sigma)
        denom = 1.0 + sigma ** 2
        alpha = 1.0 / torch.sqrt(denom)
        beta_sq = sigma ** 2 / denom
        d_alpha_dsigma = -sigma / denom ** 1.5
        d_beta_sq_dsigma = 2.0 * sigma / denom ** 2
        d_log_alpha_dsigma = d_alpha_dsigma / alpha
        return alpha, beta_sq, d_log_alpha_dsigma, d_beta_sq_dsigma


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------


def gaspari_cohn(distance: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaspari-Cohn localization weights (torch, for the full C matrix)."""
    r = distance.abs() / radius
    rho = torch.zeros_like(r)

    mask1 = r < 1
    if mask1.any():
        r1 = r[mask1]
        rho[mask1] = (
            1.0
            - (5.0 / 3.0) * r1 ** 2
            + (5.0 / 8.0) * r1 ** 3
            + 0.5 * r1 ** 4
            - 0.25 * r1 ** 5
        )

    mask2 = (r >= 1) & (r < 2)
    if mask2.any():
        r2 = r[mask2]
        rho[mask2] = (
            4.0
            - 5.0 * r2
            + (5.0 / 3.0) * r2 ** 2
            + (5.0 / 8.0) * r2 ** 3
            - 0.5 * r2 ** 4
            + (1.0 / 12.0) * r2 ** 5
            - 2.0 / (3.0 * r2)
        )

    return rho


def get_C(dim_x: int, radius: float, device=None, dtype=torch.float64) -> torch.Tensor:
    """Periodic Gaspari-Cohn localization matrix (dim_x, dim_x)."""
    n = dim_x
    L = float(n)
    x = torch.linspace(0.0, L, n, dtype=dtype, device=device)
    dx = x[:, None] - x[None, :]
    dist = torch.minimum(dx.abs(), L - dx.abs())
    return gaspari_cohn(dist, radius)


@dataclass
class LocalizationStructure:
    """Sparse representation of compactly-supported localization matrix C."""
    dim_x: int
    row_idx: torch.Tensor
    col_idx: torch.Tensor
    C_vals: torch.Tensor
    diag_mask: torch.Tensor


def build_localization_structure(
    dim_x: int, radius: float, device=None, dtype=torch.float64
) -> LocalizationStructure:
    """Build sparse (row, col, value) structure of C once per experiment."""
    C_dense = get_C(dim_x, radius, device=device, dtype=dtype)
    idx = torch.nonzero(C_dense, as_tuple=False)
    row_idx = idx[:, 0].contiguous()
    col_idx = idx[:, 1].contiguous()
    C_vals = C_dense[row_idx, col_idx].contiguous()
    diag_mask = (row_idx == col_idx)
    return LocalizationStructure(
        dim_x=dim_x, row_idx=row_idx, col_idx=col_idx, C_vals=C_vals, diag_mask=diag_mask
    )


def precompute_edge_weights(
    A: torch.Tensor, loc: LocalizationStructure, scale: float
) -> torch.Tensor:
    """weight_ij = scale * C_ij * (A_i . A_j) for every nonzero (i, j)."""
    row_idx, col_idx, C_vals = loc.row_idx, loc.col_idx, loc.C_vals
    dot_ij = (A[row_idx] * A[col_idx]).sum(dim=1)
    return scale * C_vals * dot_ij


def sparse_localized_matvec(
    V: torch.Tensor, loc: LocalizationStructure, weight_ij: torch.Tensor
) -> torch.Tensor:
    """(localized_cov) @ V without forming the (d, d) matrix."""
    contrib = weight_ij[:, None] * V[loc.col_idx, :]
    out = torch.zeros_like(V)
    out.index_add_(0, loc.row_idx, contrib)
    return out


def pcg_solve(
    matvec_fn,
    rhs: torch.Tensor,
    diag_precond: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Jacobi-preconditioned CG, batched over columns of rhs."""
    x = torch.zeros_like(rhs)
    r = rhs - matvec_fn(x)
    z = r / diag_precond[:, None]
    p = z.clone()
    rz_old = (r * z).sum(dim=0, keepdim=True)
    rhs_norm = rhs.norm(dim=0, keepdim=True).clamp_min(1e-30)

    for _ in range(max_iter):
        Ap = matvec_fn(p)
        pAp = (p * Ap).sum(dim=0, keepdim=True).clamp_min(1e-30)
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel_resid = r.norm(dim=0, keepdim=True) / rhs_norm
        if torch.all(rel_resid < tol):
            break
        z = r / diag_precond[:, None]
        rz_new = (r * z).sum(dim=0, keepdim=True)
        beta = rz_new / rz_old.clamp_min(1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x


# ---------------------------------------------------------------------------
# Score representations
# ---------------------------------------------------------------------------


class ScoreRep:
    """Score representations for prior diffusion and likelihood guidance."""

    def __init__(self, dm, dim_x: int, obs_model=None, obs_sigma=None):
        self.obs_model = obs_model
        self.obs_sigma = obs_sigma
        self.dm = dm
        self.ns = dm.noise_schedule
        self.dim_x = dim_x
        self.obs_value = None

    def set_obs_value(self, obs_value):
        self.obs_value = obs_value

    def log_likelihood(self, zt):
        y_hat = self.obs_model(zt)
        return torch.sum(-0.5 * (y_hat - self.obs_value) ** 2, dim=-1) / self.obs_sigma ** 2

    def score_likelihood_auto(self, zt):
        zt.requires_grad_(True)
        zt.grad = None
        y_hat = self.obs_model(zt)
        loss = torch.sum(-0.5 * (y_hat - self.obs_value) ** 2) / self.obs_sigma ** 2
        loss.backward()
        grad = zt.grad
        zt.requires_grad_(False)
        return grad

    def score_diffusion_GM_full(
        self,
        zt: torch.Tensor,
        t,
        mu_components: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
        cond_y: bool = False,
        target_ess_frac: float = 1.0,
        temper_bisection_iters: int = 30,
        temper_max: float = 1e8,
    ) -> torch.Tensor:
        """Exact GM mixture prior score with optional w^obs reweighting.

        High-d mixture collapse: the exact softmax over Kc components collapses
        to a single component as d grows (effective sample size -> 1), which can
        crash multiple particles onto one component and destroy ensemble spread.
        ``target_ess_frac`` in (0, 1] enforces a minimum average effective sample
        size via adaptive tempering (log_pt / tau, tau >= 1). The gradient of the
        tempered log-mixture requires the component scores to be scaled by 1/tau
        as well (handled below).

        target_ess_frac >= 1.0 -> keep the exact formula unless it already needs
        flattening to hit full ESS (in practice tau=1); use a small value
        (e.g. 0.3-0.6) to actively counter collapse in high dimensions.
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        denom = alpha_t ** 2 * eigvals + beta_sq_t

        zt_rot = zt @ eigvecs
        mu_t_rot = (mu_components * alpha_t) @ eigvecs

        diff_rot = zt_rot[:, None, :] - mu_t_rot[None, :, :]
        score_rot = -diff_rot / denom[None, None, :]
        log_pt = 0.5 * torch.sum(score_rot * diff_rot, dim=2)

        if cond_y:
            J0t = alpha_t * eigvals / denom
            mu0t_rot = (
                mu_t_rot[None, :, :] * (1.0 - alpha_t * J0t)[None, None, :]
                + zt_rot[:, None, :] * J0t[None, None, :]
            )
            Kp, Kc, d = mu0t_rot.shape
            mu0t = mu0t_rot.reshape(Kp * Kc, d) @ eigvecs.T
            loglik = self.log_likelihood(mu0t).reshape(Kp, Kc)
            log_pt = log_pt + loglik

        log_pt = log_pt - torch.max(log_pt, dim=1, keepdim=True)[0]

        tau = self._temper_tau(
            log_pt, target_ess_frac, temper_bisection_iters, temper_max)

        lp = log_pt / tau
        lp = lp - torch.max(lp, dim=1, keepdim=True)[0]
        pt = torch.exp(lp)
        wt = pt / torch.sum(pt, dim=1, keepdim=True)

        # Gradient of log sum_k exp(log_pt_k / tau) = sum_k w_k * (1/tau) * grad log_pt_k
        score_rot_final = torch.sum((score_rot / tau) * wt[:, :, None], dim=1)
        return score_rot_final @ eigvecs.T

    @staticmethod
    def _temper_tau(log_pt, target_ess_frac, bisection_iters, temper_max):
        """Smallest tau >= 1 whose tempered weights meet the ESS target."""
        Kc = log_pt.shape[1]
        target_ess = max(1.0, float(target_ess_frac) * Kc)

        def ess_at(tau):
            lp = log_pt / tau
            lp = lp - torch.max(lp, dim=1, keepdim=True)[0]
            w = torch.exp(lp)
            w = w / torch.sum(w, dim=1, keepdim=True)
            return (1.0 / torch.sum(w ** 2, dim=1)).mean().item()

        if ess_at(1.0) >= target_ess:
            return 1.0
        if ess_at(temper_max) < target_ess:
            return temper_max
        lo, hi = 1.0, temper_max
        for _ in range(bisection_iters):
            mid = math.sqrt(lo * hi)
            if ess_at(mid) < target_ess:
                lo = mid
            else:
                hi = mid
        return hi

    def score_likelihood_full(
        self,
        zt: torch.Tensor,
        t,
        mu_star: torch.Tensor,
        eigvals_star: torch.Tensor,
        eigvecs_star: torch.Tensor,
        eigval_floor: float = 0.0,
    ) -> torch.Tensor:
        """Likelihood score with full (non-diagonal) J(t) via cov_star eigenbasis."""
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        if eigval_floor > 0.0:
            eigvals_star = eigvals_star.clamp_min(eigval_floor)
        denom = alpha_t ** 2 * eigvals_star + beta_sq_t
        J0t = alpha_t * eigvals_star / denom

        zt_rot = zt @ eigvecs_star
        mu_star_rot = mu_star @ eigvecs_star

        mu0_rot = (
            mu_star_rot[None, :] * (1.0 - alpha_t * J0t)[None, :]
            + zt_rot * J0t[None, :]
        )
        mu0 = mu0_rot @ eigvecs_star.T

        sc_eval = self.score_likelihood_auto(mu0)

        sc_eval_rot = sc_eval @ eigvecs_star
        scaled_rot = sc_eval_rot * J0t[None, :]
        return scaled_rot @ eigvecs_star.T

    def score_gaussian_diffusion_eigh(
        self,
        zt: torch.Tensor,
        t,
        prior_mean: torch.Tensor,
        eigvals: torch.Tensor,
        eigvecs: torch.Tensor,
    ) -> torch.Tensor:
        """Single-Gaussian (own-component) score via cached eigendecomposition."""
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        diff = zt - mu_t
        diff_rot = diff @ eigvecs
        denom = alpha_t ** 2 * eigvals + beta_sq_t
        score_rot = -diff_rot / denom[None, :]
        return score_rot @ eigvecs.T

    def score_gaussian_diffusion_cg(
        self,
        zt: torch.Tensor,
        t,
        prior_mean: torch.Tensor,
        loc: LocalizationStructure,
        weight_ij: torch.Tensor,
        cg_max_iter: int = 50,
        cg_tol: float = 1e-6,
    ) -> torch.Tensor:
        """Single-Gaussian score via sparse CG (large-d fallback)."""
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        rhs = (zt - mu_t).T

        def matvec(V):
            Lv = sparse_localized_matvec(V, loc, weight_ij)
            return beta_sq_t * V + alpha_t ** 2 * Lv

        diag_loc = torch.zeros(loc.dim_x, device=rhs.device, dtype=rhs.dtype)
        diag_loc.index_add_(0, loc.row_idx[loc.diag_mask], weight_ij[loc.diag_mask])
        diag_precond = (beta_sq_t + alpha_t ** 2 * diag_loc).clamp_min(1e-12)

        sol = pcg_solve(matvec, rhs, diag_precond, max_iter=cg_max_iter, tol=cg_tol)
        return -sol.T

    def get_p_z0_zt_diag(self, zt, t, mu_0, var_0):
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        J_0t = var_0 * alpha_t / (alpha_t ** 2 * var_0 + beta_sq_t)
        mu_0t = mu_0 * (1 - alpha_t * J_0t) + zt * J_0t
        var_0t = var_0 * beta_sq_t / (alpha_t ** 2 * var_0 + beta_sq_t)
        return mu_0t, var_0t, J_0t

    def score_likelihood_3_diag_generic(
        self, zt, t, prior_mean, prior_var, post_mean, post_var
    ):
        _, _, J_0t = self.get_p_z0_zt_diag(zt[[0]], t, mu_0=prior_mean, var_0=prior_var)
        mu_0ty, _, _ = self.get_p_z0_zt_diag(zt, t, mu_0=post_mean, var_0=post_var)
        sc_eval = self.score_likelihood_auto(mu_0ty)
        return sc_eval * J_0t

    def score_gaussian_diffusion(self, zt, t, prior_mean, prior_cov):
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        cov_t = alpha_t ** 2 * prior_cov + beta_sq_t * torch.eye(
            self.dim_x, device=zt.device, dtype=zt.dtype
        )
        return -torch.linalg.solve(cov_t, zt - mu_t, left=False)

    def score_diffusion_GM_diag(self, zt, t, mu_0, var_0, cond_y: bool = False):
        """GM prior score with diagonal cov (optional observation reweighting)."""
        assert var_0.shape == (self.dim_x,), 'Variance shape does not match!'
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)

        mu_t = mu_0 * alpha_t
        var_t = alpha_t ** 2 * var_0 + beta_sq_t

        x_diff = zt[:, None, :] - mu_t[None, :, :]
        score = -x_diff / var_t[None, None, :]
        log_pt = 0.5 * torch.sum(score * x_diff, dim=2)

        if cond_y:
            J_0t = var_0 * alpha_t / (alpha_t ** 2 * var_0 + beta_sq_t)
            mu_0t = mu_0 * (1 - alpha_t * J_0t) + zt[:, None, :] * J_0t[None, None, :]
            log_pt = log_pt + self.log_likelihood(mu_0t)

        log_pt = log_pt - torch.max(log_pt, dim=1, keepdim=True)[0]
        pt = torch.exp(log_pt)
        wt = pt / torch.sum(pt, dim=1, keepdim=True)
        return torch.sum(score * wt[:, :, None], dim=1)


class ReverseSampler:
    """Reverse-time SDE/ODE sampler driven by a score function."""

    def __init__(self, noise_schedule):
        self.noise_schedule = noise_schedule

    def sample_SDE_euler(self, x_T, score_fn, Nt):
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)
        for i in range(Nt):
            xt = self.SDE_euler_stepping(xt, score_fn, t_mesh[i], t_mesh[i + 1])
        return xt

    def sample_ODE_euler(self, x_T, score_fn, Nt):
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)
        for i in range(Nt):
            xt = self.ODE_euler_stepping(xt, score_fn, t_mesh[i], t_mesh[i + 1])
        return xt

    def sample_ODE_DPM_solver(self, x_T, score_fn, Nt):
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)
        for i in range(Nt):
            xt = self.ODE_DPM_solver_stepping(xt, score_fn, t_mesh[i], t_mesh[i + 1])
            if torch.isnan(xt).any():
                print('nan generated! abort sampling!')
                break
        return xt

    def sample_EDM_euler(self, x_T, score_fn, Nt):
        ns = self.noise_schedule
        if not isinstance(ns, EDMNoiseSchedule):
            raise TypeError('EDM solvers require an EDMNoiseSchedule.')
        xt = x_T.detach().clone()
        sigmas = ns.make_sigma_schedule(Nt, device=xt.device, dtype=xt.dtype)
        for i in range(Nt):
            xt = self.EDM_euler_stepping(xt, score_fn, sigmas[i], sigmas[i + 1])
        return xt

    def sample_EDM_heun(self, x_T, score_fn, Nt):
        ns = self.noise_schedule
        if not isinstance(ns, EDMNoiseSchedule):
            raise TypeError('EDM solvers require an EDMNoiseSchedule.')
        xt = x_T.detach().clone()
        sigmas = ns.make_sigma_schedule(Nt, device=xt.device, dtype=xt.dtype)
        for i in range(Nt):
            xt = self.EDM_heun_stepping(xt, score_fn, sigmas[i], sigmas[i + 1])
        return xt

    def sample_gen(self, x_T, score_fn, Nt, solver_type: str = 'DPM_solver'):
        if solver_type == 'DPM_solver':
            return self.sample_ODE_DPM_solver(x_T, score_fn, Nt)
        if solver_type == 'ODE_euler':
            return self.sample_ODE_euler(x_T, score_fn, Nt)
        if solver_type == 'SDE_euler':
            return self.sample_SDE_euler(x_T, score_fn, Nt)
        if solver_type == 'EDM_euler':
            return self.sample_EDM_euler(x_T, score_fn, Nt)
        if solver_type == 'EDM_heun':
            return self.sample_EDM_heun(x_T, score_fn, Nt)
        raise ValueError('Unknown solver!')

    def get_sde_coef(self, t):
        alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt = self.noise_schedule.get_values(t)
        sde_a = d_log_alpha_dt
        sde_b_sq = d_beta_sq_dt - 2 * d_log_alpha_dt * beta_sq_t
        return sde_a, sde_b_sq

    def SDE_euler_stepping(self, xt, score_fn, t_now, t_next):
        dt = t_now - t_next
        sde_a, sde_b_sq = self.get_sde_coef(t_now)
        noise_scale = math.sqrt(float(dt) * float(sde_b_sq))
        return (
            xt
            - dt * (sde_a * xt - sde_b_sq * score_fn(xt, t_now))
            + noise_scale * torch.randn_like(xt)
        )

    def ODE_euler_stepping(self, xt, score_fn, t_now, t_next):
        dt = t_now - t_next
        sde_a, sde_b_sq = self.get_sde_coef(t_now)
        return xt - dt * (sde_a * xt - sde_b_sq * score_fn(xt, t_now) / 2)

    def ODE_DPM_solver_stepping(self, xt, score_fn, t_now, t_next):
        ns = self.noise_schedule
        alpha_now, beta_sq_now, _, _ = ns.get_values(t_now)
        lambda_now = torch.log(alpha_now) - torch.log(beta_sq_now) / 2
        beta_now = torch.sqrt(beta_sq_now)

        alpha_next, beta_sq_next, _, _ = ns.get_values(t_next)
        lambda_next = torch.log(alpha_next) - torch.log(beta_sq_next) / 2
        beta_next = torch.sqrt(beta_sq_next)

        hi = lambda_next - lambda_now
        c1 = alpha_next / alpha_now
        c2 = beta_next * beta_now * (torch.exp(hi) - 1)
        return c1 * xt + c2 * score_fn(xt, t_now)

    @staticmethod
    def _edm_denoise(xt, score_fn, sigma):
        score = score_fn(xt, sigma)
        return xt + sigma ** 2 * score

    def EDM_euler_stepping(self, xt, score_fn, sigma_now, sigma_next):
        denoised = self._edm_denoise(xt, score_fn, sigma_now)
        d = (xt - denoised) / sigma_now
        return xt + (sigma_next - sigma_now) * d

    def EDM_heun_stepping(self, xt, score_fn, sigma_now, sigma_next):
        denoised1 = self._edm_denoise(xt, score_fn, sigma_now)
        d1 = (xt - denoised1) / sigma_now
        x_euler = xt + (sigma_next - sigma_now) * d1
        denoised2 = self._edm_denoise(x_euler, score_fn, sigma_next)
        d2 = (x_euler - denoised2) / sigma_next
        return xt + (sigma_next - sigma_now) * 0.5 * (d1 + d2)


def post_score(
    zt,
    t,
    score_prior,
    score_likelihood,
    score_max: float = 1000.0,
    *,
    score_max_prior: Optional[float] = None,
    score_max_likelihood: Optional[float] = None,
    likelihood_scale: float = 1.0,
):
    """Combine prior and likelihood scores with optional per-term clipping.

    Clipping each term before summing avoids the full J(t) likelihood from
    dominating the mixture prior when only a combined bound is used.
    """
    sm_p = score_max if score_max_prior is None else score_max_prior
    sm_l = score_max if score_max_likelihood is None else score_max_likelihood
    sp = torch.clip(score_prior(zt, t), min=-sm_p, max=sm_p)
    sl = torch.clip(score_likelihood(zt, t), min=-sm_l, max=sm_l) * likelihood_scale
    return torch.clip(sp + sl, min=-score_max, max=score_max)


def make_edm_native_score_fn(score_fn_vp, ns: EDMNoiseSchedule):
    """Wrap a VP-space score for native EDM sample space (x_noisy = x0 + sigma*eps)."""

    def score_fn_native(xt, sigma):
        alpha, _beta_sq, _, _ = ns.get_values(sigma)
        zt = alpha * xt
        score_z = score_fn_vp(zt, sigma)
        return alpha * score_z

    return score_fn_native


def _apply_post_analysis_inflation(
    analysis: torch.Tensor,
    inflation: float,
    var_split: float,
) -> torch.Tensor:
    """LETKF-style multiplicative inflation when variance is entirely in cov.

    For var_split < 1, ``inflation`` already enters the GM mean perturbation
    (mu_gmm). For var_split = 1 that path is zero, so inflate analysis anomalies.
    """
    if inflation == 1.0:
        return analysis
    mean_pert_scale = math.sqrt(max(0.0, 1.0 - var_split ** 2))
    if mean_pert_scale > 1e-6:
        return analysis
    a_mean = analysis.mean(dim=0)
    return a_mean + inflation * (analysis - a_mean)


# ---------------------------------------------------------------------------
# IEnSF update
# ---------------------------------------------------------------------------
#
# ``mode`` selects which score approximations are used inside ``iensf_update``:
#
#   'full'   -- Paper-faithful IEnSF score formulas (Prop. 3.4 + Eq. 38).
#               Prior: full GM mixture score over all ensemble components with
#               optional w^obs (Eq. 48-49). Likelihood: full symmetric J(t) from
#               cov_star, not diagonal. At d ~ 10^3 the exact formulas need
#               ``full_stabilize=True`` (default): tempered mixture, separate score
#               clipping, optional diagonal likelihood blend, cond_y=False.
#               Cost: O(K^2 * d) per score eval + one eigh(sigma_gmm) per DA step.
#
#   'legacy' -- Recommended production mode. Own-component GM prior + diagonal J(t)
#               likelihood (score_likelihood_3_diag_generic), with cov_star inner
#               refinement. When var_split=1, ``inflation`` is applied as LETKF-style
#               post-analysis spread inflation (mean perturbation path is off).
#               Optional ``likelihood_diag_blend`` in (0,1) adds a fraction of the
#               full J(t) likelihood for cross-variable coupling without ``mode='full'``.
#
#   'cg'     -- Large-d fallback (d >> few thousand). Prior: single-Gaussian
#               score via sparse CG on the localized covariance (never forms d×d
#               matrix). Likelihood: same diagonal approximation as 'legacy'.
#               Requires ``loc_struct`` from build_localization_structure().
#               Does NOT implement the full GM mixture or w^obs terms.
#
# All modes share: cov_star inner-iteration update, EDM native score wrapping,
# and eig/cg caching to avoid O(Nt * d^3) work per reverse-diffusion trajectory.


def create_iensf_sampler(
    dim_x: int,
    obs_fun,
    obs_sigma: float,
    *,
    ns_type: str = 'linear',
    eps_a: float = 1e-5,
    eps_b: float = 1e-5,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    rho: float = 7.0,
    device=None,
    dtype=torch.float32,
) -> Tuple[ReverseSampler, ScoreRep]:
    """Build diffusion objects once per experiment rep (reuse across DA steps)."""
    del device, dtype
    if ns_type == 'linear':
        ns = NoiseSchedule(ns_type='linear', eps_a=eps_a, eps_b=eps_b)
    elif ns_type == 'edm':
        if sigma_min is None and sigma_max is None:
            ns = EDMNoiseSchedule.from_linear_eps(eps_a=eps_a, eps_b=eps_b, rho=rho)
        else:
            ns = EDMNoiseSchedule(
                sigma_min=sigma_min if sigma_min is not None else 0.002,
                sigma_max=sigma_max if sigma_max is not None else 80.0,
                rho=rho,
            )
    else:
        raise ValueError(f"Unknown ns_type {ns_type!r}; use 'linear' or 'edm'.")
    dm = ReverseSampler(noise_schedule=ns)
    score_rep = ScoreRep(dm=dm, dim_x=dim_x, obs_model=obs_fun, obs_sigma=obs_sigma)
    return dm, score_rep


def _init_terminal_noise(
    ensemble_size: int,
    dim_x: int,
    device,
    dtype,
    reverse_sampler: ReverseSampler,
) -> torch.Tensor:
    """Terminal state for reverse sampling (linear: unit variance; EDM: sigma_max)."""
    ns = reverse_sampler.noise_schedule
    if isinstance(ns, EDMNoiseSchedule):
        sigma_max = torch.tensor(ns.sigma_max, device=device, dtype=dtype)
        return torch.randn(ensemble_size, dim_x, device=device, dtype=dtype) * sigma_max
    x_T = torch.randn(ensemble_size, dim_x, device=device, dtype=dtype)
    return (x_T - x_T.mean(dim=0)) / x_T.std(dim=0)


def iensf_update(
    ensemble_prior: torch.Tensor,
    obs_value: torch.Tensor,
    localization_matrix: torch.Tensor,
    score_rep: ScoreRep,
    reverse_sampler: ReverseSampler,
    *,
    n_iter: int = 1,
    var_split: float,
    inflation: float,
    Nt: int,
    solver_type: str = 'SDE_euler',
    eta1: float = 0.5,
    eta2: float = 0.5,
    mode: str = 'legacy',
    score_max: float = 1000.0,
    score_max_prior: Optional[float] = None,
    score_max_likelihood: Optional[float] = None,
    likelihood_scale: float = 1.0,
    likelihood_diag_blend: Optional[float] = None,
    target_ess_frac: float = 1.0,
    cond_y: bool = False,
    full_stabilize: bool = True,
    eigval_floor: Optional[float] = None,
    rtps_relax: float = 0.0,
    loc_struct: Optional[LocalizationStructure] = None,
    cg_max_iter: int = 50,
    cg_tol: float = 1e-6,
) -> torch.Tensor:
    """One analysis step: EnSF-style (n_iter=1) or IEnSF (n_iter>1).

    Core shapes:
      ensemble_prior / analysis  (K, dim_x)
      obs_value                  (dim_obs,)
      localization_matrix C      (dim_x, dim_x)
      mu_gmm                     (K, dim_x)   # GM component means
      sigma_gmm / cov_star       (dim_x, dim_x)
      mu_star                    (dim_x,)
      x_T / current_sample       (K, dim_x)

    Args:
        mode: Score approximation level. See module comment block above
            ``create_iensf_sampler`` for details. Summary:
            ``'legacy'`` = recommended stable mode; ``'full'`` = paper formulas;
            ``'cg'`` = sparse large-d fallback (needs ``loc_struct``).
        score_max: Final clip on the combined score in ``post_score``.
        score_max_prior: Per-term prior clip. None -> ``score_max``.
        score_max_likelihood: Per-term likelihood clip. None -> ``score_max``.
        likelihood_scale: Multiplier on the clipped likelihood score.
        likelihood_diag_blend: Blend toward diagonal J(t) in [0, 1] (0 = full J(t)
            only for ``mode='full'`` without stabilize; legacy default 0 = diagonal
            only). Use 0.2-0.5 on legacy to add mild cross-variable coupling.
            None with ``full_stabilize`` uses 0.5.
        inflation: GMM mean inflation when var_split < 1; when var_split = 1,
            applied as LETKF-style post-analysis spread inflation on anomalies.
        target_ess_frac: (mode='full') minimum ESS fraction for the GM mixture;
            ``full_stabilize`` forces 0.5 when left at 1.0.
        cond_y: (mode='full') enable w^obs in the mixture prior. Default False;
            True sharpens weights and often destabilizes at d ~ 10^3.
        full_stabilize: (mode='full') When True (default), use own-component prior
            instead of the GM mixture, temper the mixture ESS target, clip prior and
            likelihood scores separately, blend full/diagonal J(t), and ridge
            cov_star eigenvalues. Set False for the paper-exact formulas.
        eigval_floor: Ridge on cov_star eigenvalues in full J(t); auto-set when
            None and ``full_stabilize``.
        rtps_relax: Relaxation-to-prior-spread coefficient in [0, 1]. After the
            analysis, ensemble anomalies are rescaled toward the prior spread by
            ``(1-a)*std_post + a*std_prior`` per dimension. 0 disables (default).
            Helps prevent spread collapse / blow-up (LETKF-style stabilizer).
    """
    if mode not in ('full', 'legacy', 'cg'):
        raise ValueError("mode must be one of 'full', 'legacy', 'cg'")
    if mode == 'cg' and loc_struct is None:
        raise ValueError(
            "mode='cg' requires `loc_struct` from build_localization_structure."
        )

    x_state = ensemble_prior
    ensemble_size, dim_x = x_state.shape
    device = x_state.device
    dtype = x_state.dtype
    C = localization_matrix.to(device=device, dtype=dtype)

    X_cov = torch.cov(x_state.T) * C
    X_mean = x_state.mean(dim=0)
    prior_std = x_state.std(dim=0)  # for optional RTPS spread relaxation
    mu_gmm = inflation * torch.sqrt(torch.tensor(1.0 - var_split ** 2, device=device, dtype=dtype)) * (
        x_state - X_mean
    ) + X_mean
    sigma_gmm = X_cov * var_split

    score_rep.set_obs_value(obs_value=obs_value)

    x_T = _init_terminal_noise(ensemble_size, dim_x, device, dtype, reverse_sampler)

    stabilize_full = mode == 'full' and full_stabilize
    ess_frac = target_ess_frac
    cond_y_full = cond_y
    blend_diag = 0.0 if likelihood_diag_blend is None else float(likelihood_diag_blend)
    sm_prior = score_max_prior
    sm_lik = score_max_likelihood
    lscale = likelihood_scale
    eig_floor = eigval_floor
    use_mixture_prior = mode == 'full' and not stabilize_full
    if stabilize_full:
        if ess_frac >= 1.0:
            ess_frac = 0.5
        if likelihood_diag_blend is None:
            blend_diag = 0.5
        # Keep clip bounds near score_max: overly tight per-term clips (e.g. 15/30)
        # distort the score field across inner iterations and can blow up the ensemble.
        if sm_prior is None:
            sm_prior = score_max
        if sm_lik is None:
            sm_lik = score_max

    eigvals_sigma, eigvecs_sigma = None, None
    if mode in ('full', 'legacy'):
        eigvals_sigma, eigvecs_sigma = torch.linalg.eigh(sigma_gmm)
        eigvals_sigma = eigvals_sigma.to(dtype=dtype)
        eigvecs_sigma = eigvecs_sigma.to(dtype=dtype)

    if mode == 'full' and use_mixture_prior:
        # Paper prior: mixture over all K GM components; optional w^obs
        score_prior_fn = functools.partial(
            score_rep.score_diffusion_GM_full,
            mu_components=mu_gmm,
            eigvals=eigvals_sigma,
            eigvecs=eigvecs_sigma,
            cond_y=cond_y_full,
            target_ess_frac=ess_frac,
        )
    elif mode == 'full':
        # Stabilized prior: own-component Gaussian (avoids high-d mixture collapse
        # during inner iterations while keeping full J(t) likelihood).
        score_prior_fn = functools.partial(
            score_rep.score_gaussian_diffusion_eigh,
            prior_mean=mu_gmm,
            eigvals=eigvals_sigma,
            eigvecs=eigvecs_sigma,
        )
    elif mode == 'legacy':
        # Prior: each particle scored against its own GM component mean only
        score_prior_fn = functools.partial(
            score_rep.score_gaussian_diffusion_eigh,
            prior_mean=mu_gmm,
            eigvals=eigvals_sigma,
            eigvecs=eigvecs_sigma,
        )
    else:
        # mode == 'cg': sparse matvec prior score; diagonal likelihood (below)
        A = (x_state - X_mean).T
        weight_ij = precompute_edge_weights(A, loc_struct, scale=var_split / (ensemble_size - 1))
        score_prior_fn = functools.partial(
            score_rep.score_gaussian_diffusion_cg,
            prior_mean=mu_gmm,
            loc=loc_struct,
            weight_ij=weight_ij,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )

    current_sample = x_state.clone()
    mu_star = current_sample.mean(dim=0)
    cov_star = torch.cov(current_sample.T) * C

    for _ in range(n_iter):
        if mode == 'full':
            eigvals_star, eigvecs_star = torch.linalg.eigh(cov_star)
            eigvals_star = eigvals_star.to(dtype=dtype)
            eigvecs_star = eigvecs_star.to(dtype=dtype)
            if eig_floor is None and stabilize_full:
                eig_floor = (torch.trace(cov_star) / max(dim_x, 1)).item() * 1e-4
            eig_floor_val = 0.0 if eig_floor is None else float(eig_floor)
            score_likelihood_full_fn = functools.partial(
                score_rep.score_likelihood_full,
                mu_star=mu_star,
                eigvals_star=eigvals_star,
                eigvecs_star=eigvecs_star,
                eigval_floor=eig_floor_val,
            )
            if blend_diag > 0.0:
                score_likelihood_diag_fn = functools.partial(
                    score_rep.score_likelihood_3_diag_generic,
                    prior_mean=mu_gmm,
                    prior_var=torch.diag(sigma_gmm),
                    post_mean=mu_star,
                    post_var=torch.diag(cov_star),
                )
                blend = blend_diag

                def score_likelihood_fn(zt, t, _full=score_likelihood_full_fn,
                                        _diag=score_likelihood_diag_fn, _b=blend):
                    return (1.0 - _b) * _full(zt, t) + _b * _diag(zt, t)
            else:
                score_likelihood_fn = score_likelihood_full_fn
        else:
            score_likelihood_diag_fn = functools.partial(
                score_rep.score_likelihood_3_diag_generic,
                prior_mean=mu_gmm,
                prior_var=torch.diag(sigma_gmm),
                post_mean=mu_star,
                post_var=torch.diag(cov_star),
            )
            if blend_diag > 0.0:
                eigvals_star, eigvecs_star = torch.linalg.eigh(cov_star)
                eigvals_star = eigvals_star.to(dtype=dtype)
                eigvecs_star = eigvecs_star.to(dtype=dtype)
                if eig_floor is None:
                    eig_floor = (torch.trace(cov_star) / max(dim_x, 1)).item() * 1e-4
                eig_floor_val = float(eig_floor)
                score_likelihood_full_fn = functools.partial(
                    score_rep.score_likelihood_full,
                    mu_star=mu_star,
                    eigvals_star=eigvals_star,
                    eigvecs_star=eigvecs_star,
                    eigval_floor=eig_floor_val,
                )
                blend = blend_diag

                def score_likelihood_fn(zt, t, _full=score_likelihood_full_fn,
                                        _diag=score_likelihood_diag_fn, _b=blend):
                    return (1.0 - _b) * _full(zt, t) + _b * _diag(zt, t)
            else:
                score_likelihood_fn = score_likelihood_diag_fn

        post_score_fn = functools.partial(
            post_score,
            score_prior=score_prior_fn,
            score_likelihood=score_likelihood_fn,
            score_max=score_max,
            score_max_prior=sm_prior,
            score_max_likelihood=sm_lik,
            likelihood_scale=lscale,
        )

        if solver_type in ('EDM_euler', 'EDM_heun'):
            ns = reverse_sampler.noise_schedule
            if not isinstance(ns, EDMNoiseSchedule):
                raise TypeError(
                    f"solver_type={solver_type!r} requires an EDMNoiseSchedule."
                )
            step_score_fn = make_edm_native_score_fn(post_score_fn, ns)
        else:
            step_score_fn = post_score_fn

        current_sample = reverse_sampler.sample_gen(
            x_T=x_T,
            score_fn=step_score_fn,
            Nt=Nt,
            solver_type=solver_type,
        )

        current_mean = current_sample.mean(dim=0)
        current_cov = torch.cov(current_sample.T) * C

        mu_star = (1.0 - eta1) * mu_star + eta1 * current_mean
        cov_star = (1.0 - eta2) * cov_star + eta2 * current_cov

    analysis = current_sample.clone()

    if rtps_relax > 0.0:
        # Relaxation to prior spread (Whitaker & Hamill 2012): rescale posterior
        # anomalies so per-dim std -> (1-a)*std_post + a*std_prior. Preserves mean.
        a_mean = analysis.mean(dim=0)
        anom = analysis - a_mean
        post_std = anom.std(dim=0)
        target_std = (1.0 - rtps_relax) * post_std + rtps_relax * prior_std
        scale = torch.where(post_std > 1e-12, target_std / post_std, torch.ones_like(post_std))
        analysis = a_mean + anom * scale

    analysis = _apply_post_analysis_inflation(analysis, inflation, var_split)

    return analysis
