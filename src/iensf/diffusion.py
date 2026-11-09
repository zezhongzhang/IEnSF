import functools

import torch
import numpy as np


class NoiseSchedule:
    """
    Forward noise schedule for diffusion model
    """

    def __init__(self, ns_type='linear', eps_a=0.00001, eps_b=0.00001):
        self.ns_type = ns_type
        self.eps_a = eps_a
        self.eps_b = eps_b

    def get_values(self, t):
        # TODO: cosine schedule
        """
        Forward noising kernel: p(zt|z0) = N(x0 * alpha(t), beta_sq(t) * I)
            - alpha_t(0) = 1
            - alpha_t(1) = 0
            - sigma2_t(0) = 0
            - sigma2_t(1) = 1
        :param t:
        :return: alpha, beta_sq, d_log_alpha_dt, d_beta_sq_dt
        """
        if self.ns_type == 'linear':
            """
            alpha_t(1) = esp_a
            alpha_t(0) = esp_b
            """
            eps_a = self.eps_a
            eps_b = self.eps_b
            alpha_t = 1 - (1 - eps_a) * t
            beta_sq_t = eps_b + (1 - eps_b) * t
            # derivative
            d_alpha_dt = - (1 - eps_a)
            d_log_alpha_dt = d_alpha_dt / alpha_t
            d_beta_sq_dt = (1 - eps_b)
            return alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt
        else:
            raise ValueError('Unknown noise schedule type!')


class ScoreRep:
    """Score helpers for diffusion-based filtering.

    Shape convention:
      N, N_eval, N_ensemble = batch / ensemble sizes
      dim_x, dim_obs        = state / observation sizes
      zt                    = (N, dim_x) noisy state at diffusion time t
      scores                = (N, dim_x)
    """

    def __init__(self, dm, dim_x, obs_model=None, obs_sigma=None):
        self.obs_model = obs_model
        self.obs_sigma = obs_sigma
        self.dm = dm
        self.ns = dm.noise_schedule
        self.dim_x = dim_x
        self.obs_value = None  # (dim_obs,) set per assimilation step

    def set_obs_model(self, obs_model, obs_sigma):
        """obs_model: (..., dim_x) -> (..., dim_obs); obs_sigma: scalar."""
        self.obs_model = obs_model
        self.obs_sigma = obs_sigma

    def set_obs_value(self, obs_value):
        """obs_value: (dim_obs,)."""
        self.obs_value = obs_value

    ############################################################
    # likelihood score
    ############################################################
    def score_likelihood_1_generic(self, zt, t, damp_fn):
        # zt: (N, dim_x) -> score (N, dim_x)
        return self.score_likelihood_auto(zt) * damp_fn(t)

    def score_likelihood_2_full_generic(self, zt, t, prior_mean, prior_cov):
        # zt: (N, dim_x); prior_mean: (dim_x,); prior_cov: (dim_x, dim_x)
        mu_0t, cov_0t, J_0t = self.get_p_z0_zt_full(zt, t, mu_0=prior_mean, cov_0=prior_cov)
        sc_eval = self.score_likelihood_auto(mu_0t)  # (N, dim_x)
        sc_final = torch.matmul(sc_eval, J_0t)  # (N, dim_x)
        return sc_final

    def score_likelihood_2_diag_generic(self, zt, t, prior_mean, prior_var):
        # zt: (N, dim_x); prior_mean: (dim_x,); prior_var: (dim_x,)
        mu_0t, cov_0t, J_0t = self.get_p_z0_zt_diag(zt, t, mu_0=prior_mean, var_0=prior_var)
        sc_eval = self.score_likelihood_auto(mu_0t)  # (N, dim_x)
        sc_final = sc_eval * J_0t  # (N, dim_x); J_0t: (dim_x,)
        return sc_final

    def score_likelihood_2_eig_generic(self, zt, t, prior_mean, prior_Q, prior_L):
        # zt: (N, dim_x); prior_Q: (dim_x, dim_x); prior_L: (dim_x,)
        mu_0t, cov_0t, J_0t = self.get_p_z0_zt_eig(zt, t, mu_0=prior_mean, cov_Q=prior_Q, cov_L=prior_L)
        sc_eval = self.score_likelihood_auto(mu_0t)
        sc_final = torch.matmul(sc_eval, J_0t)
        return sc_final

    def score_likelihood_3_full_generic(self, zt, t, prior_mean, prior_cov, post_mean, post_cov):
        # zt: (N, dim_x)
        # J_0t from prior Gaussian; evaluation point from posterior Gaussian (mu_star, cov_star)
        mu_0t, cov_0t, J_0t = self.get_p_z0_zt_full(zt[[0]], t, mu_0=prior_mean, cov_0=prior_cov)
        # mu_0ty: (N, dim_x); J_0t: (dim_x, dim_x)
        mu_0ty, cov_0ty, J_0ty = self.get_p_z0_zt_full(zt, t, mu_0=post_mean, cov_0=post_cov)
        sc_eval = self.score_likelihood_auto(mu_0ty)  # (N, dim_x)
        sc_final = torch.matmul(sc_eval, J_0t)  # (N, dim_x)
        return sc_final

    def score_likelihood_3_diag_generic(self, zt, t, prior_mean, prior_var, post_mean, post_var):
        # zt: (N, dim_x); prior/post var: (dim_x,)
        mu_0t, cov_0t, J_0t = self.get_p_z0_zt_diag(zt[[0]], t, mu_0=prior_mean, var_0=prior_var)
        mu_0ty, cov_0ty, J_0ty = self.get_p_z0_zt_diag(zt, t, mu_0=post_mean, var_0=post_var)
        sc_eval = self.score_likelihood_auto(mu_0ty)  # (N, dim_x)
        sc_final = sc_eval * J_0t  # (N, dim_x)
        return sc_final


    
    ############################################################
    # likelihood calculation
    ############################################################
    def log_likelihood(self, zt):
        """
        log data likelihood
        :param zt: (..., dim_x)
        :param obs_value: (dim_obs)
        :return: (...)
        """
        y_hat = self.obs_model(zt)
        # log likelihood
        log_l = torch.sum(-0.5 * (y_hat - self.obs_value) ** 2, dim=-1) / self.obs_sigma ** 2
        return log_l

    def likelihood(self, zt):
        """
        data likelihood
        :param zt: (N, dim_x)
        :param obs_value: (dim_obs)
        :return: (N)
        """
        return torch.exp(self.log_likelihood(zt, self.obs_value))


    def score_likelihood_auto(self, zt):
        """
        Calculate the gradient of log likelihood using auto grad
        :param zt: (N, dim_x)
        :param obs_value: (dim_obs)
        :return: (N, dim_x)
        """
        zt.requires_grad_(True)
        zt.grad = None

        # obs
        y_hat = self.obs_model(zt)

        # compute likelihood score
        loss = torch.sum(-0.5 * (y_hat - self.obs_value) ** 2) / self.obs_sigma ** 2
        loss.backward()

        grad = zt.grad
        zt.requires_grad_(False)
        return grad

    ############################################################
    # diffusion gaussian score
    ############################################################
    def score_gaussian_diffusion(self, zt, t, prior_mean, prior_cov):
        """
        Score function of gaussian diffusion model with full covariance matrix
        :param zt: (N, dim_x)
        :param t: (int)
        :param prior_mean: (dim_x) or (N, dim_x) for ensemble case
        :param prior_cov: (dim_x, dim_x)
        :return:
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        cov_t = alpha_t ** 2 * prior_cov + beta_sq_t * torch.eye(self.dim_x, device=zt.device)
        score_t = self.score_gaussian_cov(zt, mu_t, cov_t)
        return score_t


    def score_gaussian_diffusion_eig(self, zt, t, prior_mean, prior_Q, prior_L):
        """
        Score function of gaussian diffusion model with eigen decomposition of the covariance matrix
        :param zt: (N, dim_x)
        :param t: (int)
        :param prior_mean: (dim_x) or (N, dim_x) for ensemble case
        :param prior_Q:  (dim_x, dim_x)
        :param prior_L: (dim_x)
        :return:
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        omega_t = (prior_Q / (alpha_t ** 2 * prior_L + beta_sq_t)) @ prior_Q.T
        score_t = self.score_gaussian_omega(zt, mu_t, omega_t)
        return score_t


    def score_gaussian_diffusion_diag(self, zt, t, prior_mean, prior_var):
        """
        Score function of gaussian diffusion model with diagonal covariance matrix
        :param zt: (N, dim_x)
        :param t: (int)
        :param prior_mean: (dim_x) or (N, dim_x) for ensemble case
        :param prior_var: (dim_x) variance
        :return:
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = prior_mean * alpha_t
        var_t = alpha_t ** 2 * prior_var + beta_sq_t
        score_t = self.score_gaussian_diag(zt, mu_t, var=var_t)
        return score_t


    def score_diffusion_GM_diag_old(self, zt, t, mu_0, var_0):
        """
        Score function of gaussian mixture diffusion model with
        diagonal covariance matrix for each ensemble.
        :param zt: (N, dim_x)
        :param t: (int)
        :param mu_0: (N_ensemble, dim_x)
        :param var_0: (dim_x)
        :return:
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        mu_t = mu_0 * alpha_t
        var_t = alpha_t ** 2 * var_0 + beta_sq_t
        score_gm = self.score_GM_diag(zt, mu_t, var_t)
        return score_gm


    def score_diffusion_GM_diag(self, zt, t, mu_0, var_0, cond_y=False):
        """
        prior GM diffusion score with conditioning and with diag cov
        :param zt: (N_eval, dim_x)
        :param t: (int)
        :param mu_0: (N_ensemble, dim_x)
        :param var_0: (dim_x)
        :return: score_gm: ()
        """
        assert var_0.shape == (self.dim_x, ), 'Variance shape dose not match!'
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)

        # forward kernel
        mu_t = mu_0 * alpha_t
        var_t = alpha_t ** 2 * var_0 + beta_sq_t


        # score eval
        x_diff = (zt[:, None, :] - mu_t[None, :, :]) # (N_eval, N_ensemble, dim_x)

        score = - x_diff / var_t[None, None, :] # (N_eval, N_ensemble, dim_x)

        log_pt = 0.5 * torch.sum(score * x_diff, dim=2) # (N_eval, N_ensemble)
        if cond_y:
            # p_xi_zt_y
            # reverse kernel
            J_0t = var_0 * alpha_t / (alpha_t ** 2 * var_0 + beta_sq_t)  # (dim_x)
            mu_0t = mu_0 * (1 - alpha_t * J_0t) + zt[:, None, :] * J_0t[None, None, :]  # (n_eval, n_ensemble, dim_x)
            # evaluate log likelihood
            log_l = self.log_likelihood(mu_0t)  # (N_eval, N_ensemble)
            log_pt += log_l

        # numerical normalization
        log_pt = log_pt - torch.max(log_pt, dim=1, keepdim=True)[0] # (N_eval, N_ensemble)

        pt = torch.exp(log_pt) # (N_eval, N_ensemble)

        # prior score weight
        wt = pt / torch.sum(pt, dim=1, keepdim=True) # (N_eval, N_ensemble)

        score_gm = torch.sum(score * wt[:, :, None], dim=1) # (N_eval, dim_x)
        return score_gm

    ############################################################
    # reverse Gaussian transition kernel p(z0|zt)
    ############################################################



    def get_p_z0_zt_full(self, zt, t, mu_0, cov_0):
        """Reverse kernel p(z0|zt) for a full Gaussian prior.

        zt: (N, dim_x); mu_0: (dim_x,) or (N, dim_x); cov_0: (dim_x, dim_x)
        returns: mu_0t (N, dim_x), cov_0t (dim_x, dim_x), J_0t (dim_x, dim_x)
        """
        dim_x = self.dim_x
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        cov_t = alpha_t ** 2 * cov_0 + beta_sq_t * torch.eye(dim_x, device=zt.device)

        J_0t = alpha_t * torch.linalg.solve(A=cov_t, B=cov_0, left=False)  # (dim_x, dim_x)
        mu_0t = mu_0 + torch.matmul(zt - alpha_t * mu_0, J_0t)  # (N, dim_x)
        cov_0t = cov_0 - alpha_t * torch.matmul(J_0t, cov_0)  # (dim_x, dim_x)
        return mu_0t, cov_0t, J_0t

    def get_p_z0_zt_diag(self, zt, t, mu_0, var_0):
        """Reverse kernel p(z0|zt) for a diagonal Gaussian prior.

        zt: (N, dim_x); mu_0: (dim_x,) or (N, dim_x); var_0: (dim_x,)
        returns: mu_0t (N, dim_x), var_0t (dim_x,), J_0t (dim_x,)
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)
        J_0t = var_0 * alpha_t / (alpha_t ** 2 * var_0 + beta_sq_t)  # (dim_x,)
        mu_0t = mu_0 * (1 - alpha_t * J_0t) + zt * J_0t  # (N, dim_x)
        var_0t = var_0 * beta_sq_t / (alpha_t ** 2 * var_0 + beta_sq_t)  # (dim_x,)
        return mu_0t, var_0t, J_0t


    def get_p_z0_zt_eig(self, zt, t, mu_0, cov_Q, cov_L):
        """
        compute the Gaussian mean and cov of the p(z0|zt) given Gaussian p(z0)
        and eigen decomposition of covariance
        :param zt: (N, dim_x)
        :param t: (int)
        :param mu_0: (dim_x) or (N, dim_x)
        :param cov_Q: (dim_x, dim_x)- rotation matrix Q
        :param cov_L: (dim_x) - eigen values
        :return: mean-(N, dim_x) , cov-(dim_x, dim_x), Jacobian-(dim_x, dim_x)
        """
        alpha_t, beta_sq_t, _, _ = self.ns.get_values(t)

        J_0t = (cov_Q * (cov_L * alpha_t / (alpha_t ** 2 * cov_L + beta_sq_t))) @ cov_Q.T

        mu_0t = mu_0 + torch.matmul(zt - alpha_t * mu_0, J_0t)

        # eig_value = cov_L - alpha_t**2 * cov_L**2 / (alpha_t ** 2 * cov_L + beta_sq_t)
        eig_value = cov_L * beta_sq_t / (alpha_t ** 2 * cov_L + beta_sq_t)

        cov_0t = (cov_Q * eig_value) @ cov_Q.T

        return mu_0t, cov_0t, J_0t

    @staticmethod
    def post_score(zt, t, score_prior, score_likelihood):
        return score_prior(zt, t) + score_likelihood(zt, t)

    @staticmethod
    def get_post_socre_fn(score_prior, score_likelihood):
        return functools.partial(ScoreRep.post_score, score_prior=score_prior, score_likelihood=score_likelihood)

    @staticmethod
    def score_gaussian_diag(x, mu, var):
        return - (x - mu) / var

    @staticmethod
    def score_gaussian_cov(x, mu, cov):
        # score -(x - mu) * cov^-1
        return - torch.linalg.solve(cov, x - mu, left=False)

    @staticmethod
    def score_gaussian_omega(x, mu, omega):
        # score -(x - mu) * cov^-1
        return - torch.matmul(x - mu, omega)

    @staticmethod
    def line_search_mu(mu_old, mu_new, N_search, post_score_static_fn):
        """
        Linear search for optimal post mu update based on score norm
        :param mu_old:
        :param mu_new:
        :param N_int: number of linear search points
        :param post_score_fn: postereior score function (zt, t) -> (sc)
        :return:
        """
        w_vec = torch.linspace(0, 1, N_search, device=mu_old.device)
        mu_int = w_vec[:, None] * mu_old[None, :] + (1 - w_vec[:, None]) * mu_new[None, :]
        sc_check = post_score_static_fn(mu_int)
        score_norm = torch.sum(sc_check ** 2, dim=1)
        mu_opt_loc = torch.argmin(score_norm)
        mu_opt = mu_int[mu_opt_loc]
        return mu_opt

    @staticmethod
    def score_GM_diag(x, mu_vec, var):
        """
        static score for GM prior
        :param x: (n_eval, dim_x)
        :param mu_vec: (n_sample, dim_x)
        :param var: (dim_x)
        :return: (N_batch, dim_x)
        """
        # x_diff: (n_eval, n_sample, dim_x)
        x_diff = (x[:, None, :] - mu_vec[None, :, :])

        # score: (n_eval, n_sample, dim_x)
        score = - x_diff / var[None, None, :]

        # log_p: (n_eval, n_sample)
        # log_pt = -0.5*torch.sum(x_diff**2 / var, dim=2)
        log_pt = 0.5 * torch.sum(score * x_diff, dim=2)

        # numerical normalization
        log_pt = log_pt - torch.max(log_pt, dim=1, keepdim=True)[0]

        # pt: (n_eval, n_sample)
        pt = torch.exp(log_pt)

        # wt: (n_eval, n_sample)
        wt = pt / torch.sum(pt, dim=1, keepdim=True)

        score_gm = torch.sum(score * wt[:, :, None], dim=1)
        return score_gm



class ReverseSampler:
    """Reverse-time sampler.

    x_T / xt: (N, dim_x) particles in diffusion space
    score_fn: (xt, t) -> (N, dim_x)
    """

    def __init__(self, noise_schedule):
        self.noise_schedule = noise_schedule

    def sample_SDE_euler(self, x_T, score_fn, Nt):
        # x_T: (N, dim_x) -> returns xt at t=0 with same shape
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)  # from 1 -> 0
        for i in range(Nt):
            t_now = t_mesh[i]
            t_next = t_mesh[i + 1]
            xt = self.SDE_euler_stepping(xt, score_fn, t_now, t_next)
        return xt

    def sample_ODE_euler(self, x_T, score_fn, Nt):
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)  # from 1 -> 0
        for i in range(Nt):
            t_now = t_mesh[i]
            t_next = t_mesh[i + 1]
            xt = self.ODE_euler_stepping(xt, score_fn, t_now, t_next)
        return xt

    def sample_ODE_DPM_solver(self, x_T, score_fn, Nt):
        xt = x_T.detach().clone()
        t_mesh = torch.linspace(1, 0, Nt + 1)  # from 1 -> 0
        for i in range(Nt):
            t_now = t_mesh[i]
            t_next = t_mesh[i + 1]
            xt = self.ODE_DPM_solver_stepping(xt, score_fn, t_now, t_next)
            nan_count = torch.isnan(xt).sum().item()
            if nan_count > 0:
                print('nan generated! abort sampling!')
                break
        return xt

    def sample_ODE_SDE_mix(self,  x_T, score_fn, Nt):
        pass


    def sample_gen(self, x_T, score_fn, Nt, solver_type='DPM_solver'):
        if solver_type == 'DPM_solver':
            xt = self.sample_ODE_DPM_solver(x_T, score_fn, Nt)
        elif solver_type == 'ODE_euler':
            xt = self.sample_ODE_euler(x_T, score_fn, Nt)
        elif solver_type == 'SDE_euler':
            xt = self.sample_SDE_euler(x_T, score_fn, Nt)
        else:
            raise ValueError('Unknown solver!')
        return xt


    def get_sde_coef(self, t):
        """
        Get coefficients for reverse SDE.
        :param t:
        :return:
        """
        alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt = self.noise_schedule.get_values(t)
        sde_a = d_log_alpha_dt
        sde_b_sq = d_beta_sq_dt - 2 * d_log_alpha_dt * beta_sq_t
        return sde_a, sde_b_sq


    def SDE_euler_stepping(self, xt, score_fn, t_now, t_next):
        dt = t_now - t_next
        sde_a, sde_b_sq = self.get_sde_coef(t_now)
        xt_next = xt - dt * (sde_a * xt - sde_b_sq * score_fn(xt, t_now)) + \
                  np.sqrt(dt) * np.sqrt(sde_b_sq) * torch.randn_like(xt)
        return xt_next


    def ODE_euler_stepping(self, xt, score_fn, t_now, t_next):
        dt = t_now - t_next
        sde_a, sde_b_sq = self.get_sde_coef(t_now)
        xt_next = xt - dt * (sde_a * xt - sde_b_sq * score_fn(xt, t_now) / 2)
        return xt_next


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

        # clip large value for stability
        # tol = 20
        # c1 = torch.clip(c1, max=tol)
        # c2 = torch.clip(c2, max=tol)

        # update
        xt_next = c1 * xt + c2 * score_fn(xt, t_now)
        # xt_next = alpha_next / alpha_now * xt - beta_next * (torch.exp(hi) - 1) * (-score_fn(xt, t_now) * beta_now)

        # xt_next = alpha_next / alpha_now * xt - beta_next * (torch.exp(hi) - 1) * (-score_fn(xt, t_now) * beta_next)
        return xt_next


def Langevin(sample, eps, drift_fn, Nt):
    for i in range(Nt):
        drift_eval = drift_fn(sample)
        sample = sample + eps*drift_eval + np.sqrt(2 * eps) * torch.randn_like(sample)
    return sample


def post_score(zt, t, score_prior, score_likelihood, score_max=1000):
    # zt: (N, dim_x); each score_*: (N, dim_x) -> combined score (N, dim_x)
    score_eval = score_prior(zt, t) + score_likelihood(zt, t)
    score_eval = torch.clip(score_eval, min=-score_max, max=score_max)
    return score_eval


def post_score_static(x, score_prior_static, score_likelihood_static, score_max=1000):
    # x: (N, dim_x) at t=0 (no diffusion time)
    s_prior = score_likelihood_static(x)
    s_likelihood = score_prior_static(x)
    score_eval = s_prior + s_likelihood
    score_eval = torch.clip(score_eval, min=-score_max, max=score_max)
    return score_eval
